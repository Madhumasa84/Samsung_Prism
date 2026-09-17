"""Bounded asynchronous retrieval scheduling for the Phase 2 controller.

The repository's retriever contract is intentionally synchronous. This module
keeps that public interface unchanged and runs ``Retriever.search`` in a
bounded executor so transcript events continue through the asyncio loop while
retrieval is in progress. Cancellation is an optimisation only: revision and
session checks are the correctness boundary.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from dataclasses import dataclass
from collections import deque
from typing import Any

from .config import Settings
from .contracts import (
    StreamingDecision,
    StreamingRetrievalRequest,
    StreamingRetrievalResult,
    StreamingSchedulerConfig,
    TraceError,
    Usage,
)
from .retrieval import Retriever, tokenize
from .multi_intent import (
    MultiIntentRetriever,
    filter_single_query_evidence,
    reuse_validation,
    single_query_reuse_validation,
)
from .streaming import StreamingControllerError, StreamingDecisionController, _canonical_query
from .trace import TraceCollector


class RetrievalSchedulerError(RuntimeError):
    """Raised when bounded scheduler execution cannot accept a request."""


def scheduler_config_from_settings(settings: Settings) -> StreamingSchedulerConfig:
    """Build scheduler settings from the validated environment-backed config."""

    return StreamingSchedulerConfig(
        max_concurrency=settings.streaming_max_concurrency,
        max_pending_requests=settings.streaming_max_pending_requests,
        request_timeout_s=settings.streaming_request_timeout_s,
        max_retries=settings.streaming_max_retries,
        retry_backoff_s=settings.streaming_retry_backoff_s,
        max_total_requests=settings.streaming_max_total_requests,
        final_wait_timeout_s=settings.streaming_final_wait_timeout_s,
        cancel_on_supersession=settings.streaming_cancel_on_supersession,
    )


@dataclass
class _RequestState:
    request: StreamingRetrievalRequest
    decision: StreamingDecision
    task: asyncio.Task[None] | None = None
    executor_future: concurrent.futures.Future[Any] | None = None
    poll_task: asyncio.Task[Any] | None = None
    late_observer: asyncio.Task[None] | None = None
    result: StreamingRetrievalResult | None = None
    attempts: int = 0
    actual_started_s: float | None = None
    actual_started_time_s: float | None = None
    actual_completed_time_s: float | None = None
    scheduled_time_s: float | None = None
    started: bool = False
    superseded: bool = False
    closed: bool = False
    cancellation_requested: bool = False
    cancellation_confirmed: bool = False
    decomposition: dict[str, Any] | None = None
    retrieval_details: dict[str, Any] | None = None


class AsyncRetrievalScheduler:
    """Session/utterance-scoped retrieval scheduler.

    At most ``max_pending_requests`` request states are active, plus one latest
    coalesced candidate waiting for capacity. The executor itself is bounded
    by ``max_concurrency``. A request is allocated when the controller has
    made a retrieval decision, which advances the controller's
    session-scoped retrieval revision immediately; a queued request therefore
    supersedes old evidence before it starts running.
    """

    def __init__(
        self,
        *,
        controller: StreamingDecisionController,
        retriever: Retriever,
        traces: TraceCollector,
        retrieval_model_identity: str,
        config: StreamingSchedulerConfig | None = None,
    ) -> None:
        self.controller = controller
        self.retriever = retriever
        self.traces = traces
        self.retrieval_model_identity = retrieval_model_identity
        self.config = config or StreamingSchedulerConfig()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_concurrency,
            thread_name_prefix="flowcontext-retrieval",
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._active: dict[str, _RequestState] = {}
        self._pending_latest: _RequestState | None = None
        # Phase 2 keeps one latest pending query because a material correction
        # supersedes the previous stream.  Phase 4 can have several
        # independent unresolved intents in one candidate revision, so its
        # caller may opt into this bounded FIFO alongside the legacy latest
        # slot.
        self._parallel_pending: deque[_RequestState] = deque()
        self._results: list[StreamingRetrievalResult] = []
        self._allocated_requests = 0
        self._closed = False
        self._changed = asyncio.Event()
        self.last_error: TraceError | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def results(self) -> list[StreamingRetrievalResult]:
        # Completion traces retain real completion order, while the persisted
        # result list is stable for inspection and replay comparison.
        return sorted(self._results, key=lambda result: result.retrieval_revision)

    @property
    def scheduled_request_count(self) -> int:
        return self._allocated_requests

    @property
    def retrieval_attempt_count(self) -> int:
        return sum(result.attempts for result in self._results)

    @property
    def retrieval_call_count(self) -> int:
        """Count actual ``Retriever.search`` calls, including retries."""

        return sum(result.attempts for result in self._results)

    @property
    def retrieval_error_count(self) -> int:
        return sum(result.error is not None for result in self._results)

    @property
    def retrieval_usage(self) -> Usage:
        calls = [
            _retrieval_usage(result.query)
            for result in self._results
            for _ in range(result.attempts)
        ]
        if not calls:
            return Usage()
        return Usage(
            input_tokens=sum(item.input_tokens for item in calls),
            output_tokens=0,
            total_tokens=sum(item.total_tokens for item in calls),
            estimated=True,
        )

    @property
    def cancelled_request_count(self) -> int:
        return sum(result.status == "cancelled" for result in self._results)

    @property
    def superseded_request_count(self) -> int:
        return sum(result.superseded or result.status == "superseded" for result in self._results)

    @property
    def timed_out_request_count(self) -> int:
        return sum(result.status == "timed_out" for result in self._results)

    @property
    def stale_result_count(self) -> int:
        return sum(result.stale for result in self._results)

    def _trace(
        self,
        event_type: str,
        *,
        state: _RequestState | None = None,
        source_timestamp_s: float | None = None,
        duration_ms: float = 0.0,
        usage: Usage | None = None,
        error: TraceError | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        values = dict(attributes or {})
        if state is not None:
            request = state.request
            values.setdefault("request_id", request.request_id)
            values.setdefault("decision_id", request.decision_id)
            values.setdefault("transcript_revision", request.transcript_revision)
            values.setdefault("retrieval_revision", request.retrieval_revision)
            values.setdefault("query", request.query)
            values.setdefault("is_final", state.decision.is_final_event)
        self.traces.add(
            event_type,
            source_timestamp_s=source_timestamp_s,
            duration_ms=duration_ms,
            usage=usage or Usage(),
            model_identity=self.retrieval_model_identity,
            error=error,
            attributes=values,
        )

    def _new_state(
        self,
        decision: StreamingDecision,
        *,
        query_override: str | None = None,
    ) -> _RequestState:
        if self._closed:
            raise RetrievalSchedulerError("retrieval scheduler is closed")
        if self._allocated_requests >= self.config.max_total_requests:
            error = RetrievalSchedulerError(
                "retrieval request limit reached; revised query was not scheduled"
            )
            self.last_error = TraceError(error_type=type(error).__name__, message=str(error))
            self.traces.add_error(
                "streaming_retrieval_schedule_failed",
                error,
                source_timestamp_s=decision.source_timestamp_s,
                model_identity=self.retrieval_model_identity,
                attributes={
                    "decision_id": decision.decision_id,
                    "transcript_revision": decision.transcript_revision,
                    "retrieval_request_limit": self.config.max_total_requests,
                },
            )
            raise error
        try:
            request = self.controller.begin_retrieval(decision, query_override=query_override)
        except StreamingControllerError as exc:
            self.last_error = TraceError(error_type=type(exc).__name__, message=str(exc))
            raise RetrievalSchedulerError(str(exc)) from exc
        self._allocated_requests += 1
        scheduled_time_s = self.traces.elapsed_s()
        request = request.model_copy(update={"scheduled_time_s": scheduled_time_s})
        return _RequestState(
            request=request,
            decision=decision,
            scheduled_time_s=scheduled_time_s,
        )

    def _trace_scheduled(self, state: _RequestState, queue_state: str) -> None:
        self._trace(
            "streaming_retrieval_scheduled",
            state=state,
            source_timestamp_s=state.request.source_timestamp_s,
            usage=_retrieval_usage(state.request.query),
            attributes={
                "queue_state": queue_state,
                "selected_retrieval_backend": getattr(self.retriever, "backend", "unknown"),
                "retrieval_method": getattr(self.retriever, "method", getattr(self.retriever, "backend", "unknown")),
                "max_concurrency": self.config.max_concurrency,
                "max_pending_requests": self.config.max_pending_requests,
                "scheduled_time_s": state.scheduled_time_s,
            },
        )

    def schedule(
        self,
        decision: StreamingDecision,
        *,
        query_override: str | None = None,
        allow_parallel_queries: bool = False,
    ) -> StreamingRetrievalRequest:
        """Schedule a decision without blocking the event loop.

        A RETRIEVE decision is allowed to be represented by a SKIP decision
        only for final recovery: ``query_override`` makes that recovery
        explicit in the request while leaving the per-event decision record
        unchanged.
        """

        if decision.decision != "RETRIEVE" and query_override is None:
            raise RetrievalSchedulerError("only RETRIEVE decisions can be scheduled without a query override")
        if (
            allow_parallel_queries
            and len(self._active) >= self.config.max_pending_requests
            and len(self._parallel_pending) >= self.config.max_pending_requests
        ):
            raise RetrievalSchedulerError(
                "independent retrieval pending limit reached; targeted work was not scheduled"
            )
        state = self._new_state(decision, query_override=query_override)
        query_key = _canonical_query(state.request.query)

        if not allow_parallel_queries:
            for active_state in list(self._active.values()):
                if active_state.result is None and _canonical_query(active_state.request.query) != query_key:
                    self._mark_superseded(active_state, reason="material_query_change")
            if self._pending_latest is not None:
                if _canonical_query(self._pending_latest.request.query) != query_key:
                    self._cancel_pending(self._pending_latest, reason="coalesced_by_newer_query")
                    self._pending_latest = None
                else:
                    # The final path should normally reuse a matching state. This
                    # guard avoids allocating duplicate work if a caller forces a
                    # same-query schedule concurrently.
                    self._cancel_pending(self._pending_latest, reason="duplicate_pending_query")
                    self._pending_latest = None

        if len(self._active) < self.config.max_pending_requests:
            self._activate(state)
            self._trace_scheduled(state, "active")
        elif allow_parallel_queries:
            self._parallel_pending.append(state)
            self._trace_scheduled(state, "parallel_pending")
        else:
            if self._pending_latest is not None:
                self._cancel_pending(self._pending_latest, reason="coalesced_by_newer_query")
            self._pending_latest = state
            self._trace_scheduled(state, "coalesced_pending")
            self._trace(
                "streaming_retrieval_coalesced",
                state=state,
                source_timestamp_s=state.request.source_timestamp_s,
                attributes={"coalesced_pending": True},
            )
        return state.request

    def supersede_all(self, *, reason: str = "candidate_revision_superseded") -> None:
        """Mark every outstanding request obsolete without trusting cancellation.

        This is intentionally separate from ``close``.  A Phase 4 correction
        keeps the session alive while the old executor work may continue; the
        revision-aware controller and the coordinator's expected-revision
        check reject any late result that does finish.
        """

        for state in list(self._active.values()):
            self._mark_superseded(state, reason=reason)
        pending = self._pending_latest
        self._pending_latest = None
        if pending is not None:
            self._cancel_pending(pending, reason=reason)
        parallel = list(self._parallel_pending)
        self._parallel_pending.clear()
        for state in parallel:
            self._cancel_pending(state, reason=reason)

    def result_for_request(self, request_id: str) -> StreamingRetrievalResult | None:
        """Return a completed result by request ID, if the scheduler retained it."""

        for result in reversed(self._results):
            if result.request_id == request_id:
                return result
        return None

    async def wait_for_request(
        self,
        request_id: str,
        *,
        timeout_s: float | None = None,
    ) -> StreamingRetrievalResult | None:
        """Await one scheduled request without running final-query recovery."""

        result = self.result_for_request(request_id)
        if result is not None:
            return result
        state = self._active.get(request_id)
        if state is None and self._pending_latest is not None:
            if self._pending_latest.request.request_id == request_id:
                state = self._pending_latest
        if state is None:
            state = next(
                (
                    item
                    for item in self._parallel_pending
                    if item.request.request_id == request_id
                ),
                None,
            )
        if state is None:
            return None
        deadline_s = time.monotonic() + (
            self.config.final_wait_timeout_s if timeout_s is None else max(0.0, timeout_s)
        )
        result = await self._await_state(state, deadline_s=deadline_s)
        return result

    def _activate(self, state: _RequestState) -> None:
        self._active[state.request.request_id] = state
        state.task = asyncio.create_task(
            self._run_state(state),
            name=f"flowcontext-retrieval-{state.request.request_id}",
        )

    def _mark_superseded(self, state: _RequestState, *, reason: str) -> None:
        if state.result is not None or state.superseded:
            return
        state.superseded = True
        self._trace(
            "streaming_retrieval_superseded",
            state=state,
            source_timestamp_s=state.request.source_timestamp_s,
            attributes={"reason": reason, "superseded_immediately": True},
        )
        if self.config.cancel_on_supersession:
            self._request_cancellation(state, reason=reason)

    def _request_cancellation(self, state: _RequestState, *, reason: str) -> None:
        if state.cancellation_requested:
            return
        state.cancellation_requested = True
        confirmed = False
        if not state.started:
            confirmed = True
        elif state.executor_future is not None:
            # concurrent.futures.Future.cancel() is false once a worker has
            # started. A true value is the only cancellation confirmation we
            # expose for an executor-backed request.
            confirmed = state.executor_future.cancel()
        state.cancellation_confirmed = confirmed
        self._trace(
            "streaming_retrieval_cancellation_requested",
            state=state,
            source_timestamp_s=state.request.source_timestamp_s,
            attributes={
                "reason": reason,
                "cancellation_requested": True,
                "cancellation_confirmed": confirmed,
            },
        )
        if confirmed:
            self._trace(
                "streaming_retrieval_cancellation_confirmed",
                state=state,
                source_timestamp_s=state.request.source_timestamp_s,
                attributes={"reason": reason, "cancellation_confirmed": True},
            )
            if not state.started:
                self._record_terminal(
                    state,
                    status="cancelled",
                    error=TraceError(error_type="RetrievalCancelled", message="request cancelled before worker start"),
                )
                if state.task is not None and not state.task.done():
                    state.task.cancel()

    def _cancel_pending(self, state: _RequestState, *, reason: str) -> None:
        self._mark_superseded(state, reason=reason)
        if state.result is None:
            # _mark_superseded confirms a not-yet-started request and records
            # the terminal cancellation. This fallback is defensive for a
            # caller that disabled cancellation tracing in its config.
            self._record_terminal(
                state,
                status="superseded",
                error=TraceError(error_type="RetrievalSuperseded", message=reason),
            )

    def _base_result(
        self,
        state: _RequestState,
        *,
        hits: list,
        accepted: bool,
        stale: bool,
        status: str,
        completed_s: float,
        error: TraceError | None = None,
    ) -> StreamingRetrievalResult:
        actual_completed_time_s = state.actual_completed_time_s or self.traces.elapsed_s()
        state.decomposition = state.decomposition or self._decomposition_for_state(state)
        state.retrieval_details = state.retrieval_details or self._retrieval_details_for_state(state)
        return StreamingRetrievalResult(
            request_id=state.request.request_id,
            decision_id=state.request.decision_id,
            session_id=state.request.session_id,
            utterance_id=state.request.utterance_id,
            transcript_revision=state.request.transcript_revision,
            retrieval_revision=state.request.retrieval_revision,
            query=state.request.query,
            hits=list(hits),
            decomposition=state.decomposition,
            retrieval_details=state.retrieval_details,
            accepted=accepted,
            stale=stale,
            source_timestamp_s=state.request.source_timestamp_s,
            monotonic_started_s=state.request.monotonic_started_s,
            monotonic_completed_s=max(completed_s, state.request.monotonic_started_s),
            duration_ms=max(0.0, (completed_s - state.request.monotonic_started_s) * 1000),
            status=status,
            attempts=state.attempts,
            superseded=state.superseded,
            cancellation_requested=state.cancellation_requested,
            cancellation_confirmed=state.cancellation_confirmed,
            error=error,
            is_final_event=state.decision.is_final_event,
            scheduled_time_s=state.scheduled_time_s,
            actual_started_time_s=state.actual_started_time_s,
            actual_completed_time_s=actual_completed_time_s,
            evidence_ready_time_s=None,
            actual_duration_ms=(
                0.0
                if state.actual_started_time_s is None
                else max(
                    0.0,
                    (actual_completed_time_s - state.actual_started_time_s)
                    * 1000,
                )
            ),
        )

    def _search_request(self, state: _RequestState) -> Any:
        """Pass revision context to the opt-in Phase 3 adapter when present."""

        search_with_revision = getattr(self.retriever, "search_with_revision", None)
        if callable(search_with_revision):
            return search_with_revision(
                state.request.query,
                parent_revision=state.request.transcript_revision,
                retrieval_revision=state.request.retrieval_revision,
            )
        candidates = self.retriever.search(state.request.query)
        # Phase 2 remains a single parent-query path, but it must not pass a
        # weak lexical distractor to answer generation merely because the
        # chunk has a valid ID or a non-zero retrieval score.
        evidence, _decisions = filter_single_query_evidence(
            state.request.query,
            candidates,
        )
        return evidence

    def _decomposition_for_state(self, state: _RequestState) -> dict[str, Any] | None:
        get_decomposition = getattr(self.retriever, "decomposition_for", None)
        if not callable(get_decomposition):
            return None
        decomposition = get_decomposition(state.request.retrieval_revision)
        if decomposition is None:
            return None
        return decomposition.model_dump(mode="json")

    def _retrieval_details_for_state(self, state: _RequestState) -> dict[str, Any] | None:
        get_result = getattr(self.retriever, "result_for", None)
        if not callable(get_result):
            return None
        result = get_result(state.request.retrieval_revision)
        if result is None:
            return None
        return result.model_dump(mode="json")

    def _trace_decomposition(self, state: _RequestState, *, obsolete: bool = False) -> None:
        decomposition = state.decomposition
        if decomposition is None:
            return
        status = decomposition.get("status", "success")
        event_type = (
            "streaming_decomposition_fallback"
            if status == "fallback"
            else "streaming_decomposition_completed"
        )
        attributes = {
            "decomposition_status": status,
            "decomposition_method": decomposition.get("decomposition_method"),
            "decomposition_provider": decomposition.get("provider"),
            "decomposition_model": decomposition.get("provider_model"),
            "decomposition_revision": decomposition.get("transcript_revision"),
            "intent_count": len(decomposition.get("intents", [])),
            "decomposition_attempts": decomposition.get("attempts", 0),
            "decomposition_repair_attempts": decomposition.get("repair_attempts", 0),
            "preserved_intent_ids": decomposition.get("preserved_intent_ids", []),
            "superseded_intent_ids": decomposition.get("superseded_intent_ids", []),
            "provider_errors": decomposition.get("failure_reason"),
        }
        if obsolete or decomposition.get("transcript_revision") != state.request.transcript_revision:
            attributes["ignored_as_obsolete"] = True
            event_type = "streaming_decomposition_obsolete"
        usage = Usage.model_validate(decomposition.get("usage", {}))
        self._trace(
            event_type,
            state=state,
            source_timestamp_s=state.request.source_timestamp_s,
            duration_ms=float(decomposition.get("latency_ms", 0.0) or 0.0),
            usage=usage,
            attributes=attributes,
        )

    def _record_terminal(
        self,
        state: _RequestState,
        *,
        status: str,
        error: TraceError | None = None,
    ) -> StreamingRetrievalResult:
        if state.result is not None:
            return state.result
        result = self._base_result(
            state,
            hits=[],
            accepted=False,
            stale=True,
            status=status,
            completed_s=time.monotonic(),
            error=error,
        )
        state.result = result
        if not self._closed:
            self._results.append(result)
            if status == "cancelled":
                self._trace(
                    "streaming_retrieval_cancelled",
                    state=state,
                    source_timestamp_s=state.request.source_timestamp_s,
                    duration_ms=result.duration_ms,
                    usage=_retrieval_usage(state.request.query),
                    error=error,
                    attributes={"cancellation_confirmed": True},
                )
            else:
                self._trace(
                    "streaming_retrieval_completed",
                    state=state,
                    source_timestamp_s=state.request.source_timestamp_s,
                    duration_ms=result.duration_ms,
                    usage=_retrieval_usage(state.request.query),
                    error=error,
                    attributes={"status": status, "accepted": False, "stale": True},
                )
            self._trace(
                "streaming_retrieval_stale_rejected",
                state=state,
                source_timestamp_s=state.request.source_timestamp_s,
                duration_ms=result.duration_ms,
                usage=_retrieval_usage(state.request.query),
                attributes={"status": status, "late_result_published": False},
            )
        self._finish_state(state)
        return result

    def _record_result(self, state: _RequestState, result: StreamingRetrievalResult) -> None:
        if state.result is not None:
            return
        state.decomposition = state.decomposition or self._decomposition_for_state(state)
        state.retrieval_details = state.retrieval_details or self._retrieval_details_for_state(state)
        state.actual_completed_time_s = state.actual_completed_time_s or self.traces.elapsed_s()
        result = result.model_copy(
            update={
                "is_final_event": state.decision.is_final_event,
                "scheduled_time_s": state.scheduled_time_s,
                "actual_started_time_s": state.actual_started_time_s,
                "actual_completed_time_s": state.actual_completed_time_s,
                "decomposition": state.decomposition,
                "retrieval_details": state.retrieval_details,
                "evidence_ready_time_s": (
                    state.actual_completed_time_s
                    if result.accepted and not result.stale and bool(result.hits)
                    else None
                ),
                "actual_duration_ms": (
                    0.0
                    if state.actual_started_time_s is None
                    else max(
                        0.0,
                        (state.actual_completed_time_s - state.actual_started_time_s) * 1000,
                    )
                ),
            }
        )
        state.result = result
        if self._closed or state.closed:
            self._finish_state(state)
            return
        self._results.append(result)
        if state.retrieval_details is not None:
            details = state.retrieval_details
            self._trace(
                "streaming_retrieval_assembly_completed",
                state=state,
                source_timestamp_s=state.request.source_timestamp_s,
                duration_ms=float(details.get("total_duration_ms", 0.0) or 0.0),
                usage=Usage.model_validate(details.get("retrieval_usage", {})),
                attributes={
                    "retrieval_mode": details.get("retrieval_mode"),
                    "retrieval_configuration": details.get("retrieval_config"),
                    "intent_results": details.get("intent_results", []),
                    "fusion_decisions": [
                        decision
                        for item in details.get("intent_results", [])
                        for decision in item.get("fusion_decisions", [])
                    ],
                    "assembly_decisions": details.get("assembly_decisions", []),
                    "missing_intent_ids": details.get("missing_intent_ids", []),
                    "context_tokens_used": details.get("context_tokens_used"),
                    "context_budget_tokens": details.get("context_budget_tokens"),
                },
            )
        self._trace_decomposition(state, obsolete=result.stale or result.status in {"superseded", "cancelled"})
        if result.status == "timed_out":
            self._trace(
                "streaming_retrieval_timed_out",
                state=state,
                source_timestamp_s=state.request.source_timestamp_s,
                duration_ms=result.duration_ms,
                usage=_retrieval_usage(state.request.query),
                error=result.error,
                attributes={"attempts": result.attempts},
            )
        elif result.status == "failed":
            self._trace(
                "streaming_retrieval_failed",
                state=state,
                source_timestamp_s=state.request.source_timestamp_s,
                duration_ms=result.duration_ms,
                usage=_retrieval_usage(state.request.query),
                error=result.error,
                attributes={"attempts": result.attempts},
            )
        else:
            self._trace(
                "streaming_retrieval_completed",
                state=state,
                source_timestamp_s=state.request.source_timestamp_s,
                duration_ms=result.duration_ms,
                usage=_retrieval_usage(state.request.query),
                error=result.error,
                attributes={
                    "status": result.status,
                    "hit_count": len(result.hits),
                    "hit_chunk_ids": [hit.chunk_id for hit in result.hits],
                    "accepted": result.accepted,
                    "stale": result.stale,
                    "attempts": result.attempts,
                    "actual_monotonic_start_s": state.actual_started_s,
                    "actual_duration_ms": (
                        0.0
                        if state.actual_started_s is None
                        else max(0.0, (time.monotonic() - state.actual_started_s) * 1000)
                    ),
                    "scheduled_time_s": state.scheduled_time_s,
                    "actual_started_time_s": state.actual_started_time_s,
                    "actual_completed_time_s": state.actual_completed_time_s,
                    "retrieval_mode": (
                        state.retrieval_details or {}
                    ).get("retrieval_mode"),
                    "missing_intent_ids": (
                        state.retrieval_details or {}
                    ).get("missing_intent_ids", []),
                    "context_tokens_used": (
                        state.retrieval_details or {}
                    ).get("context_tokens_used"),
                    "context_budget_tokens": (
                        state.retrieval_details or {}
                    ).get("context_budget_tokens"),
                },
            )
            if result.accepted and not result.stale and result.hits:
                self._trace(
                    "streaming_evidence_ready",
                    state=state,
                    source_timestamp_s=state.request.source_timestamp_s,
                    duration_ms=result.actual_duration_ms,
                    usage=_retrieval_usage(state.request.query),
                    attributes={
                        "valid_evidence": True,
                        "evidence_ready_time_s": result.evidence_ready_time_s,
                        "hit_chunk_ids": [hit.chunk_id for hit in result.hits],
                    },
                )
        if result.stale:
            self._trace(
                "streaming_retrieval_stale_rejected",
                state=state,
                source_timestamp_s=state.request.source_timestamp_s,
                duration_ms=result.duration_ms,
                usage=_retrieval_usage(state.request.query),
                attributes={
                    "status": result.status,
                    "late_result_published": False,
                    "hit_chunk_ids": [hit.chunk_id for hit in result.hits],
                },
            )
        self._finish_state(state)

    def _finish_state(self, state: _RequestState) -> None:
        if state.late_observer is not None and not state.late_observer.done():
            return
        self._active.pop(state.request.request_id, None)
        self._changed.set()
        self._promote_pending()

    def _promote_pending(self) -> None:
        if self._closed:
            return
        while len(self._active) < self.config.max_pending_requests:
            if self._parallel_pending:
                state = self._parallel_pending.popleft()
                if state.result is not None or state.superseded:
                    continue
                self._activate(state)
                self._trace_scheduled(state, "promoted_from_parallel_pending")
                continue
            if self._pending_latest is None:
                return
            state = self._pending_latest
            self._pending_latest = None
            if state.result is not None or state.superseded:
                continue
            self._activate(state)
            self._trace_scheduled(state, "promoted_from_pending")

    def _completed_result_for_query(self, query: str) -> StreamingRetrievalResult | None:
        query_key = _canonical_query(query)
        for result in reversed(self._results):
            if (
                result.status == "completed"
                and result.accepted
                and not result.stale
                and result.retrieval_revision == self.controller.latest_retrieval_revision
                and _canonical_query(result.query) == query_key
                and self._evidence_is_appropriate(query, result.hits)
            ):
                return result
        return None

    def _evidence_validation(self, query: str, hits: list) -> dict[str, Any]:
        validator = getattr(self.retriever, "validate_evidence_for_query", None)
        if isinstance(self.retriever, MultiIntentRetriever) and callable(validator):
            return validator(
                query,
                hits,
                parent_revision=self.controller.transcript_revision,
            )
        if not isinstance(self.retriever, MultiIntentRetriever):
            return single_query_reuse_validation(query, hits)
        return reuse_validation(query, hits)

    def _evidence_is_appropriate(self, query: str, hits: list) -> bool:
        return bool(self._evidence_validation(query, hits).get("lexical_relevance_proxy"))

    def _matching_state(self, query: str) -> _RequestState | None:
        query_key = _canonical_query(query)
        candidates = list(self._active.values())
        if self._pending_latest is not None:
            candidates.append(self._pending_latest)
        for state in reversed(candidates):
            if (
                state.result is None
                and not state.superseded
                and _canonical_query(state.request.query) == query_key
            ):
                return state
        return None

    async def _await_state(
        self,
        state: _RequestState,
        *,
        deadline_s: float,
    ) -> StreamingRetrievalResult | None:
        while state.result is None and not self._closed:
            if state.task is None:
                self._promote_pending()
                if state.task is None:
                    remaining = deadline_s - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._changed.clear()
                    try:
                        await asyncio.wait_for(self._changed.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        return None
                    continue
            remaining = deadline_s - time.monotonic()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(asyncio.shield(state.task), timeout=remaining)
            except asyncio.TimeoutError:
                self._trace(
                    "streaming_retrieval_final_wait_timeout",
                    state=state,
                    source_timestamp_s=state.request.source_timestamp_s,
                    attributes={"final_wait_timeout_s": self.config.final_wait_timeout_s},
                )
                return None
            except asyncio.CancelledError:
                return None
        return state.result

    async def finish(
        self,
        *,
        final_query: str,
        final_decision: StreamingDecision,
    ) -> StreamingRetrievalResult | None:
        """Return conservative final evidence, scheduling a final query if needed."""

        if self._closed:
            return None
        deadline_s = time.monotonic() + self.config.final_wait_timeout_s
        scheduled_recovery = False
        while time.monotonic() < deadline_s:
            reusable = self._completed_result_for_query(final_query)
            if reusable is not None:
                self._trace(
                    "streaming_evidence_reused",
                    source_timestamp_s=reusable.source_timestamp_s,
                    duration_ms=reusable.actual_duration_ms,
                    usage=_retrieval_usage(reusable.query),
                    attributes={
                        "request_id": reusable.request_id,
                        "decision_id": reusable.decision_id,
                        "query": reusable.query,
                        "retrieval_revision": reusable.retrieval_revision,
                        "is_final_event": reusable.is_final_event,
                        "evidence_ready_time_s": reusable.evidence_ready_time_s,
                        "early_evidence_reused": not reusable.is_final_event,
                        "reuse_validation": self._evidence_validation(final_query, reusable.hits),
                    },
                )
                return reusable
            state = self._matching_state(final_query)
            if state is not None:
                if state.task is None:
                    self._promote_pending()
                result = await self._await_state(state, deadline_s=deadline_s)
                if (
                    result is not None
                    and result.status == "completed"
                    and result.accepted
                    and not result.stale
                    and self._evidence_is_appropriate(final_query, result.hits)
                ):
                    return result
                # A failed/timed-out matching request must not be treated as
                # evidence. One bounded final recovery may try the complete
                # query again, subject to max_total_requests.
                if scheduled_recovery:
                    return None
                scheduled_recovery = True
            else:
                if scheduled_recovery:
                    return None
                try:
                    self.schedule(final_decision, query_override=final_query)
                except RetrievalSchedulerError:
                    return None
                scheduled_recovery = True
            # A just-scheduled request is now discoverable on the next loop.
        return None

    async def _observe_late(
        self,
        state: _RequestState,
        poll_task: asyncio.Future[Any],
    ) -> None:
        hits: list = []
        late_error: TraceError | None = None
        observer_cancelled = False
        try:
            hits = await poll_task
        except asyncio.CancelledError:
            observer_cancelled = True
        except Exception as exc:
            late_error = TraceError(error_type=type(exc).__name__, message=str(exc))
        finally:
            state.late_observer = None
            state.poll_task = None
        if not self._closed and not observer_cancelled and late_error is None:
            self._trace(
                "streaming_retrieval_late_result_rejected",
                state=state,
                source_timestamp_s=state.request.source_timestamp_s,
                usage=_retrieval_usage(state.request.query),
                attributes={
                    "late_result_published": False,
                    "hit_chunk_ids": [hit.chunk_id for hit in hits],
                    "reason": "request_timed_out_or_was_superseded",
                },
            )
            self._trace(
                "streaming_retrieval_stale_rejected",
                state=state,
                source_timestamp_s=state.request.source_timestamp_s,
                usage=_retrieval_usage(state.request.query),
                attributes={"late_result_published": False, "late": True},
            )
        elif not self._closed and not observer_cancelled and late_error is not None:
            self._trace(
                "streaming_retrieval_late_result_rejected",
                state=state,
                source_timestamp_s=state.request.source_timestamp_s,
                error=late_error,
                attributes={"late_result_published": False},
            )
        self._finish_state(state)

    async def _run_state(self, state: _RequestState) -> None:
        try:
            async with self._semaphore:
                if state.result is not None or state.superseded or state.closed or self._closed:
                    if state.result is None:
                        self._record_terminal(
                            state,
                            status="cancelled" if state.cancellation_confirmed else "superseded",
                            error=TraceError(
                                error_type="RetrievalCancelled",
                                message="request was closed before worker start",
                            ),
                        )
                    return
                state.started = True
                state.actual_started_s = time.monotonic()
                state.actual_started_time_s = self.traces.elapsed_s()
                self._trace(
                    "streaming_retrieval_started",
                    state=state,
                    source_timestamp_s=state.request.source_timestamp_s,
                    usage=_retrieval_usage(state.request.query),
                    attributes={
                        "actual_start_observed": True,
                        "actual_monotonic_start_s": state.actual_started_s,
                        "actual_started_time_s": state.actual_started_time_s,
                        "scheduled_monotonic_s": state.request.monotonic_started_s,
                        "scheduled_time_s": state.scheduled_time_s,
                        "selected_retrieval_backend": getattr(self.retriever, "backend", "unknown"),
                        "retrieval_method": getattr(self.retriever, "method", getattr(self.retriever, "backend", "unknown")),
                    },
                )
                hits = None
                last_error: Exception | None = None
                for retry_number in range(self.config.max_retries + 1):
                    if state.superseded or state.closed or self._closed:
                        self._record_terminal(
                            state,
                            status="cancelled" if state.cancellation_confirmed else "superseded",
                            error=TraceError(
                                error_type="RetrievalSuperseded",
                                message="request superseded before retrieval attempt",
                            ),
                        )
                        return
                    state.attempts += 1
                    # Submit only the callable to the bounded executor. The
                    # concurrent future is polled asynchronously below so
                    # event-loop wake-up behavior cannot delay a completed
                    # worker result. Its own cancel() return value remains
                    # the only cancellation confirmation we expose.
                    executor_future = self._executor.submit(
                        self._search_request,
                        state,
                    )
                    state.executor_future = executor_future
                    poll_task = asyncio.create_task(
                        _poll_executor_future(executor_future),
                        name=f"flowcontext-poll-retrieval-{state.request.request_id}-{state.attempts}",
                    )
                    state.poll_task = poll_task
                    try:
                        hits = await asyncio.wait_for(
                            asyncio.shield(poll_task),
                            timeout=self.config.request_timeout_s,
                        )
                    except asyncio.TimeoutError:
                        error = TraceError(
                            error_type="RetrievalTimeout",
                            message=f"retrieval timed out after {self.config.request_timeout_s:.3f}s",
                        )
                        if not executor_future.done():
                            # Install the drain before recording the terminal
                            # timeout so the state remains visible to close()
                            # and to the pending-request bound.
                            state.late_observer = asyncio.create_task(
                                self._observe_late(state, poll_task),
                                name=f"flowcontext-late-retrieval-{state.request.request_id}",
                            )
                        result = self._base_result(
                            state,
                            hits=[],
                            accepted=False,
                            stale=True,
                            status="timed_out",
                            completed_s=time.monotonic(),
                            error=error,
                        )
                        self._record_result(state, result)
                        return
                    except asyncio.CancelledError:
                        if state.poll_task is not None and not state.poll_task.done():
                            state.poll_task.cancel()
                        state.poll_task = None
                        if state.result is None:
                            self._record_terminal(
                                state,
                                status="cancelled" if state.cancellation_confirmed else "superseded",
                                error=TraceError(
                                    error_type="RetrievalCancelled",
                                    message="retrieval task cancellation was observed",
                                ),
                            )
                        return
                    except Exception as exc:
                        last_error = exc
                        state.decomposition = self._decomposition_for_state(state)
                        state.executor_future = None
                        state.poll_task = None
                        if retry_number < self.config.max_retries and not state.superseded and not self._closed:
                            self._trace(
                                "streaming_retrieval_retry_scheduled",
                                state=state,
                                source_timestamp_s=state.request.source_timestamp_s,
                                error=TraceError(error_type=type(exc).__name__, message=str(exc)),
                                attributes={
                                    "retry_number": retry_number + 1,
                                    "max_retries": self.config.max_retries,
                                },
                            )
                            if self.config.retry_backoff_s:
                                await asyncio.sleep(self.config.retry_backoff_s)
                            continue
                        error = TraceError(error_type=type(exc).__name__, message=str(exc))
                        result = self._base_result(
                            state,
                            hits=[],
                            accepted=False,
                            stale=False,
                            status="failed",
                            completed_s=time.monotonic(),
                            error=error,
                        )
                        self.last_error = error
                        self._record_result(state, result)
                        return
                    else:
                        state.executor_future = None
                        state.poll_task = None
                        state.decomposition = self._decomposition_for_state(state)
                        if state.closed or self._closed:
                            # Session closure prevents publication even when a
                            # worker happens to finish at the same time.
                            self._record_terminal(
                                state,
                                status="cancelled" if state.cancellation_confirmed else "superseded",
                                error=TraceError(
                                    error_type="RetrievalSessionClosed",
                                    message="late retrieval result was not published after session closure",
                                ),
                            )
                            return
                        try:
                            state.actual_completed_time_s = self.traces.elapsed_s()
                            controller_result = self.controller.complete_retrieval(
                                state.request,
                                hits,
                                monotonic_completed_s=time.monotonic(),
                            )
                        except Exception as exc:
                            error = TraceError(error_type=type(exc).__name__, message=str(exc))
                            self.last_error = error
                            result = self._base_result(
                                state,
                                hits=[],
                                accepted=False,
                                stale=False,
                                status="failed",
                                completed_s=time.monotonic(),
                                error=error,
                            )
                            self._record_result(state, result)
                            return
                        status = "superseded" if state.superseded or controller_result.stale else "completed"
                        result = controller_result.model_copy(
                            update={
                                "status": status,
                                "attempts": state.attempts,
                                "superseded": state.superseded or controller_result.stale,
                                "cancellation_requested": state.cancellation_requested,
                                "cancellation_confirmed": state.cancellation_confirmed,
                            }
                        )
                        self._record_result(state, result)
                        return
                if hits is None and last_error is not None:
                    error = TraceError(error_type=type(last_error).__name__, message=str(last_error))
                    self.last_error = error
                    self._record_result(
                        state,
                        self._base_result(
                            state,
                            hits=[],
                            accepted=False,
                            stale=False,
                            status="failed",
                            completed_s=time.monotonic(),
                            error=error,
                        ),
                    )
        except asyncio.CancelledError:
            if state.result is None:
                self._record_terminal(
                    state,
                    status="cancelled" if state.cancellation_confirmed else "superseded",
                    error=TraceError(error_type="RetrievalCancelled", message="retrieval task was cancelled"),
                )
        except Exception as exc:
            error = TraceError(error_type=type(exc).__name__, message=str(exc))
            self.last_error = error
            self._record_result(
                state,
                self._base_result(
                    state,
                    hits=[],
                    accepted=False,
                    stale=False,
                    status="failed",
                    completed_s=time.monotonic(),
                    error=error,
                ),
            )
        finally:
            self._finish_state(state)

    async def close(self) -> None:
        """Close the session boundary without waiting on unkillable threads."""

        if self._closed:
            return
        self._closed = True
        if self._pending_latest is not None:
            pending = self._pending_latest
            self._pending_latest = None
            pending.closed = True
            self._request_cancellation(pending, reason="session_closed")
            if pending.result is None:
                self._record_terminal(
                    pending,
                    status="cancelled",
                    error=TraceError(error_type="RetrievalCancelled", message="session closed before worker start"),
                )
        parallel = list(self._parallel_pending)
        self._parallel_pending.clear()
        for pending in parallel:
            pending.closed = True
            self._request_cancellation(pending, reason="session_closed")
            if pending.result is None:
                self._record_terminal(
                    pending,
                    status="cancelled",
                    error=TraceError(error_type="RetrievalCancelled", message="session closed before worker start"),
                )
        for state in list(self._active.values()):
            state.closed = True
            if state.result is None:
                self._request_cancellation(state, reason="session_closed")
        for state in list(self._active.values()):
            if state.late_observer is not None and not state.late_observer.done():
                state.late_observer.cancel()
            if state.poll_task is not None and not state.poll_task.done():
                state.poll_task.cancel()
        # Do not await started retrieval tasks: their executor thread may be
        # cancellation-resistant. Their completion path sees _closed and is
        # forbidden from publishing evidence or traces as current output.
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._changed.set()

    async def aclose(self) -> None:
        """Alias for callers that prefer an explicit async-close name."""

        await self.close()


def _retrieval_usage(query: str) -> Usage:
    token_count = len(tokenize(query))
    return Usage(input_tokens=token_count, output_tokens=0, total_tokens=token_count, estimated=True)


async def _poll_executor_future(future: concurrent.futures.Future[Any]) -> Any:
    """Observe a worker future without blocking the asyncio event loop.

    Polling at a small fixed interval is intentional here. It keeps the
    scheduler independent of thread-to-event-loop callback wake-up behavior in
    constrained runtimes while still allowing transcript events and timers to
    run. The retriever itself remains bounded by the executor.
    """

    while not future.done():
        await asyncio.sleep(0.001)
    if future.cancelled():
        raise asyncio.CancelledError()
    return future.result()
