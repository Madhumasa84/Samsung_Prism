"""Dedicated Phase 2 streaming evaluation and implementation audit.

This module intentionally keeps the local synthetic fixture, simulated-delay
correctness cases, real-backend validation, and official-asset status in
separate report sections.  It is a measurement harness, not an official
benchmark runner.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import ValidationError

from .config import Settings
from .contracts import (
    CorpusIndex,
    StreamingDecisionConfig,
    StreamingEvaluationCase,
    StreamingEvaluationCaseResult,
    StreamingEvaluationReport,
    StreamingEvaluationSection,
    StreamingModeEvaluationResult,
    StreamingReplayResult,
    StreamingSchedulerConfig,
    TraceError,
    Usage,
)
from .evaluation import _code_revision, _environment, _hardware
from .generation import generation_provider_for_settings
from .replay import ReplayResult, replay_transcript
from .retrieval import RetrievalError, Retriever, make_retriever
from .multi_intent import reuse_validation
from .scheduler import AsyncRetrievalScheduler, scheduler_config_from_settings
from .streaming import (
    StreamingDecisionController,
    _canonical_query,
    replay_streaming_transcript,
    streaming_config_from_settings,
)
from .trace import TraceCollector


class StreamingEvaluationError(ValueError):
    """Raised when a dedicated Phase 2 evaluation asset is invalid."""


_MEASURED_TRACE_EVENTS = {
    "transcript_event_received",
    "streaming_decision",
    "retrieval_decision",
    "streaming_retrieval_scheduled",
    "retrieval_scheduled",
    "streaming_retrieval_started",
    "retrieval_started",
    "streaming_retrieval_completed",
    "retrieval_completed",
    "streaming_retrieval_failed",
    "streaming_retrieval_timed_out",
    "streaming_retrieval_stale_rejected",
    "streaming_retrieval_late_result_rejected",
    "streaming_evidence_ready",
    "evidence_ready",
    "final_event_delivered",
    "streaming_generation_started",
    "generation_started",
    "streaming_generation_completed",
    "streaming_generation_skipped",
    "generation_completed",
    "generation_abstained",
    "generation_skipped",
    "streaming_answer_completed",
    "streaming_answer_failed",
    "answer_completed",
    "answer_failed",
    "streaming_session_closed",
    "replay_completed",
    "streaming_replay_completed",
    "streaming_replay_failed",
}
_REPORT_TIMELINE_CASES = frozenset(
    {
        "stream-dev-incomplete-meaningful",
        "stream-dev-stable-partial",
        "stream-dev-final-only",
        "stream-dev-repeated",
        "stream-dev-rapid-old-results",
        "stream-dev-format-context",
        "stream-dev-format-no-context",
        "stream-dev-session-close",
    }
)


class _ScenarioRetriever:
    """Apply bounded, explicit simulated behavior around one real retriever."""

    def __init__(
        self,
        base: Retriever,
        *,
        behavior: str,
        delay_s: float,
    ) -> None:
        self.base = base
        self.backend = base.backend
        self.method = getattr(base, "method", base.backend)
        self.behavior = behavior
        self.delay_s = delay_s
        self.calls = 0

    def search(self, query: str):
        self.calls += 1
        if self.behavior in {"delay", "timeout"} and self.delay_s > 0:
            time.sleep(self.delay_s)
        if self.behavior == "failure":
            raise RetrievalError("simulated retrieval failure")
        return self.base.search(query)


def load_streaming_evaluation_cases(
    path: Path,
    *,
    expected_split: str | None = None,
) -> list[StreamingEvaluationCase]:
    """Load strict Phase 2 cases and enforce split selection."""

    if not path.is_file():
        raise StreamingEvaluationError(f"streaming evaluation asset does not exist: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise StreamingEvaluationError(f"streaming evaluation asset is not UTF-8: {path}") from exc
    cases: list[StreamingEvaluationCase] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            case = StreamingEvaluationCase.model_validate(json.loads(raw_line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise StreamingEvaluationError(
                f"invalid streaming evaluation case on line {line_number}: {exc}"
            ) from exc
        if expected_split is not None and case.split != expected_split:
            raise StreamingEvaluationError(
                f"streaming evaluation case {case.case_id!r} is in split {case.split!r}, "
                f"not requested split {expected_split!r}"
            )
        cases.append(case)
    if not cases:
        raise StreamingEvaluationError(f"streaming evaluation asset contains no cases: {path}")
    return cases


def write_streaming_evaluation_report(
    path: Path,
    report: StreamingEvaluationReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _error_types(errors: Sequence[TraceError | Exception | str]) -> list[str]:
    values: list[str] = []
    for error in errors:
        if isinstance(error, str):
            values.append(error.split(":", 1)[0])
        elif isinstance(error, TraceError):
            values.append(error.error_type)
        else:
            values.append(type(error).__name__)
    return list(dict.fromkeys(values))


def _quality(
    expected_ids: Sequence[str],
    retrieved_ids: Sequence[str],
) -> tuple[dict[str, float], float | None]:
    expected = set(expected_ids)
    if not expected:
        return {}, None
    ranked = list(retrieved_ids)
    metrics = {
        f"recall_at_{k}": len(set(ranked[:k]) & expected) / len(expected)
        for k in (1, 3, 5)
    }
    reciprocal_rank = None
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in expected:
            reciprocal_rank = 1.0 / rank
            break
    return metrics, reciprocal_rank


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _timing_summary(values: Sequence[float], *, execution_mode: str) -> dict[str, Any]:
    return {
        "execution_mode": execution_mode,
        "sample_count": len(values),
        "p50_ms": _percentile(values, 0.5),
        "p95_ms": _percentile(values, 0.95),
        "latency_comparable_to_realtime": execution_mode == "realtime",
        "interpretation": (
            "observed realtime replay timing"
            if execution_mode == "realtime"
            else "synthetic accelerated scheduling timing; not model-performance latency"
        ),
    }


def _trace_timeline(traces: Sequence[Any]) -> list[dict[str, Any]]:
    """Keep only compact, non-secret lifecycle/decision fields."""

    timeline: list[dict[str, Any]] = []
    for trace in traces:
        if trace.event_type not in _MEASURED_TRACE_EVENTS:
            continue
        item: dict[str, Any] = {
            "event": trace.event_type,
            "source_time_s": trace.source_timestamp_s,
            "actual_time_s": trace.actual_delivery_time_s
            if trace.actual_delivery_time_s is not None
            else trace.monotonic_execution_time_s,
        }
        if trace.duration_ms:
            item["duration_ms"] = round(trace.duration_ms, 6)
        for key in (
            "decision",
            "reason_code",
            "is_final",
            "accepted",
            "stale",
            "status",
            "actual_started_time_s",
            "actual_completed_time_s",
            "evidence_ready_time_s",
            "controller_decision_time_s",
            "controller_decision_duration_ms",
            "hit_count",
            "early_evidence_reused",
            "duplicate",
        ):
            value = trace.attributes.get(key)
            if isinstance(value, (str, bool, int, float)) or value is None:
                if value is not None:
                    item[key] = value
        timeline.append(item)
    return timeline


def _trace_complete(
    traces: Sequence[Any],
    *,
    mode: Literal["baseline", "streaming"],
    expected_no_retrieval: bool,
    run_status: str,
) -> tuple[bool, list[str]]:
    event_types = {trace.event_type for trace in traces}
    if mode == "baseline":
        required = [
            "replay_started",
            "transcript_event_received",
            "final_event_delivered",
            "utterance_finalized",
            "retrieval_decision",
            "answer_completed",
            "replay_completed",
        ]
        if expected_no_retrieval:
            required.remove("answer_completed")
            if not {"answer_completed", "generation_skipped"} & event_types:
                required.append("answer_completed/generation_skipped")
        elif run_status == "failed":
            required.remove("answer_completed")
            required.extend(["retrieval_scheduled", "retrieval_started", "retrieval_failed", "answer_failed"])
        elif run_status != "closed":
            required.extend(["retrieval_scheduled", "retrieval_started"])
    else:
        required = [
            "streaming_replay_started",
            "transcript_event_received",
            "streaming_decision",
        ]
        if run_status == "closed":
            required.append("streaming_session_closed")
        elif run_status == "failed":
            required.extend(
                [
                    "final_event_delivered",
                    "utterance_finalized",
                    "streaming_answer_completed/failed",
                    "streaming_replay_failed",
                ]
            )
        else:
            required.extend(
                [
                    "final_event_delivered",
                    "utterance_finalized",
                    "streaming_answer_completed",
                    "streaming_replay_completed",
                ]
            )
            if not {"streaming_answer_completed", "streaming_answer_failed"} & event_types:
                required.append("streaming_answer_completed/failed")
            if expected_no_retrieval and not (
                {"streaming_answer_completed", "streaming_answer_failed"} & event_types
            ):
                required.append("streaming_answer_completed/failed")
    missing = [
        name
        for name in required
        if (
            name not in event_types
            and name != "answer_completed/generation_skipped"
            and name != "streaming_answer_completed/failed"
        )
    ]
    if "answer_completed/generation_skipped" in required and not (
        {"answer_completed", "generation_skipped"} & event_types
    ):
        missing.append("answer_completed/generation_skipped")
    if "streaming_answer_completed/failed" in required and not (
        {"streaming_answer_completed", "streaming_answer_failed"} & event_types
    ):
        missing.append("streaming_answer_completed/failed")
    return not missing, list(dict.fromkeys(missing))


def _request_status_summary(statuses: Sequence[str], *, fallback: str = "not_scheduled") -> str:
    """Summarise request lifecycle without turning failures into missing data."""

    if not statuses:
        return fallback
    for status in ("failed", "timed_out", "superseded", "cancelled", "closed"):
        if status in statuses:
            return status
    return "completed" if all(status == "completed" for status in statuses) else statuses[-1]


def _baseline_request_details(result: ReplayResult) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    statuses: list[str] = []
    queries: list[str] = []
    details: list[dict[str, Any]] = []
    retrieval_started = [trace for trace in result.traces if trace.event_type == "retrieval_started"]
    retrieval_failed = [trace for trace in result.traces if trace.event_type == "retrieval_failed"]
    if result.retrieval_triggered:
        status = "failed" if retrieval_failed else "completed" if retrieval_started else "not_started"
        statuses.append(status)
        queries.append(result.query)
        details.append(
            {
                "request_id": None,
                "query": result.query,
                "status": status,
                "is_final": True,
                "accepted": status == "completed",
                "stale": False,
                "hit_ids": [hit.chunk_id for hit in result.retrieval_hits],
            }
        )
    return statuses, queries, details


def _streaming_request_details(
    result: StreamingReplayResult | Sequence[Any],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    retrieval_results = result.retrieval_results if isinstance(result, StreamingReplayResult) else list(result)
    statuses = [item.status for item in retrieval_results]
    queries = [item.query for item in retrieval_results]
    details = [
        {
            "request_id": item.request_id,
            "decision_id": item.decision_id,
            "transcript_revision": item.transcript_revision,
            "retrieval_revision": item.retrieval_revision,
            "query": item.query,
            "status": item.status,
            "accepted": item.accepted,
            "stale": item.stale,
            "is_final": item.is_final_event,
            "attempts": item.attempts,
            "hit_ids": [hit.chunk_id for hit in item.hits],
            "source_time_s": item.source_timestamp_s,
            "scheduled_time_s": item.scheduled_time_s,
            "actual_started_time_s": item.actual_started_time_s,
            "actual_completed_time_s": item.actual_completed_time_s,
            "evidence_ready_time_s": item.evidence_ready_time_s,
            "error": item.error.error_type if item.error is not None else None,
        }
        for item in retrieval_results
    ]
    return statuses, queries, details


def _reuse_decision(
    result: StreamingReplayResult,
    *,
    request_details: Sequence[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    reuse_trace = next(
        (
            trace
            for trace in result.traces
            if trace.event_type == "streaming_evidence_reused"
        ),
        None,
    )
    if reuse_trace is not None and reuse_trace.attributes.get("early_evidence_reused") is True:
        validation = reuse_trace.attributes.get("reuse_validation")
        return "accepted_early_reuse", validation if isinstance(validation, dict) else {}
    early = [item for item in request_details if not item.get("is_final")]
    if not early:
        return "no_early_request", {}
    if result.run_status == "closed":
        return "session_closed_before_final", {}
    if any(item.get("status") == "timed_out" for item in early):
        return "early_retrieval_timed_out", {}
    if any(item.get("status") == "failed" for item in early):
        return "early_retrieval_failed", {}
    final_query = result.final_query or ""
    early_queries = [str(item.get("query", "")) for item in early]
    exact_early = next(
        (
            item
            for item in result.retrieval_results
            if not item.is_final_event
            and item.accepted
            and not item.stale
            and item.hits
        ),
        None,
    )
    validation = (
        reuse_validation(final_query, exact_early.hits)
        if exact_early is not None
        else {}
    )
    if any(item.get("stale") for item in early):
        return "rejected_stale_after_query_revision", validation
    if early_queries and any(
        _canonical_query(query) != _canonical_query(final_query)
        for query in early_queries
    ):
        return "final_query_changed_no_reuse", validation
    if result.retrieval_started_early and not result.valid_evidence_ready_before_finalization:
        return "early_result_not_ready_before_final", validation
    if result.retrieval_started_early:
        return "final_evidence_retrieved_without_early_reuse", validation
    return "early_request_not_usable", validation


def _baseline_mode_result(
    case: StreamingEvaluationCase,
    result: ReplayResult,
) -> StreamingModeEvaluationResult:
    traces = result.traces
    final_ids = [hit.chunk_id for hit in result.retrieval_hits]
    quality, mrr = _quality(case.expected_final_evidence_chunk_ids, final_ids)
    final_decision_trace = next(
        (
            trace
            for trace in reversed(traces)
            if trace.event_type == "retrieval_decision"
            and trace.attributes.get("mode") == "baseline"
        ),
        None,
    )
    decision = (
        final_decision_trace.attributes.get("decision")
        if final_decision_trace is not None
        else ("retrieve" if result.retrieval_triggered else "skip")
    )
    decision_map = {"retrieve": "RETRIEVE", "skip": "SKIP", "wait": "WAIT"}
    final_decision = decision_map.get(str(decision).lower())
    trace_complete, missing = _trace_complete(
        traces,
        mode="baseline",
        expected_no_retrieval=case.expected_no_retrieval,
        run_status=result.run_status,
    )
    generation_cost = result.generation_cost
    cost_availability = "available" if isinstance(generation_cost, (int, float)) else "unavailable"
    retrieval_errors = _error_types(result.errors)
    request_statuses, request_queries, request_details = _baseline_request_details(result)
    relevant_ids = list(case.expected_final_evidence_chunk_ids)
    quality_scored = result.run_status == "completed" and bool(relevant_ids) and bool(quality)
    return StreamingModeEvaluationResult(
        mode="baseline",
        execution_mode=result.execution_mode,
        run_status=result.run_status,
        trace_complete=trace_complete,
        missing_trace_events=missing,
        trace_event_count=len(traces),
        final_query=result.query,
        final_decision=final_decision,
        decision_reasons=(
            [str(final_decision_trace.attributes.get("reason", ""))]
            if final_decision_trace is not None
            else []
        ),
        retrieval_started_early=False,
        valid_evidence_ready_before_finalization=False,
        early_evidence_reused=False,
        retrieval_call_count=result.retrieval_call_count,
        scheduled_request_count=result.scheduled_request_count,
        superseded_request_count=result.superseded_request_count,
        cancelled_request_count=result.cancelled_request_count,
        timed_out_request_count=result.timed_out_request_count,
        retrieval_error_count=result.retrieval_error_count,
        stale_result_discard_count=result.stale_result_discard_count,
        stale_result_accepted_count=0,
        final_evidence_chunk_ids=final_ids,
        expected_final_evidence_chunk_ids=list(case.expected_final_evidence_chunk_ids),
        retrieved_ids=final_ids,
        relevant_ids=relevant_ids,
        request_status=_request_status_summary(request_statuses),
        request_statuses=request_statuses,
        request_queries=request_queries,
        request_details=request_details,
        reuse_decision="baseline_final_only" if result.retrieval_triggered else "no_retrieval_policy",
        reuse_validation=(
            reuse_validation(result.query, result.retrieval_hits)
            if result.retrieval_hits
            else {}
        ),
        scoring_denominator=len(relevant_ids),
        quality_scored=quality_scored,
        quality_exclusion_reason=(
            None
            if quality_scored
            else "no relevance labels"
            if not relevant_ids
            else "baseline run did not produce successful final evidence"
        ),
        end_to_end_success=result.run_status == "completed" and not result.errors,
        final_evidence_recall_at_k=quality,
        final_evidence_mrr=mrr,
        answer_latency_from_final_event_delivery_ms=result.answer_latency_from_final_event_delivery_ms,
        full_interaction_duration_ms=result.full_interaction_duration_ms,
        controller_overhead_ms=result.controller_overhead_ms,
        timing_sample_count=1,
        generation_status=result.generation_status,
        generation_usage=result.generation_usage,
        retrieval_usage=result.retrieval_usage,
        generation_cost=generation_cost,
        cost_availability=cost_availability,
        errors=retrieval_errors,
        trace_timeline=_trace_timeline(traces),
    )


def _streaming_mode_result(
    case: StreamingEvaluationCase,
    result: StreamingReplayResult,
) -> StreamingModeEvaluationResult:
    traces = result.traces
    final_ids = [hit.chunk_id for hit in result.current_evidence_hits]
    quality, mrr = _quality(case.expected_final_evidence_chunk_ids, final_ids)
    trace_complete, missing = _trace_complete(
        traces,
        mode="streaming",
        expected_no_retrieval=case.expected_no_retrieval,
        run_status=result.run_status,
    )
    generation_cost = result.generation_cost
    cost_availability = "available" if isinstance(generation_cost, (int, float)) else "unavailable"
    stale_accepted = sum(item.accepted and item.stale for item in result.retrieval_results)
    request_statuses, request_queries, request_details = _streaming_request_details(result)
    relevant_ids = list(case.expected_final_evidence_chunk_ids)
    quality_scored = result.run_status == "completed" and bool(relevant_ids) and bool(quality)
    reuse_decision, reuse_check = _reuse_decision(
        result,
        request_details=request_details,
    )
    return StreamingModeEvaluationResult(
        mode="streaming",
        execution_mode=result.execution_mode,
        run_status=result.run_status,
        trace_complete=trace_complete,
        missing_trace_events=missing,
        trace_event_count=len(traces),
        final_query=result.final_query,
        final_decision=result.final_decision.decision,
        decision_reasons=[decision.reason_code for decision in result.decisions],
        retrieval_started_early=result.retrieval_started_early,
        valid_evidence_ready_before_finalization=result.valid_evidence_ready_before_finalization,
        early_evidence_reused=result.early_evidence_reused,
        retrieval_call_count=result.retrieval_call_count,
        scheduled_request_count=result.scheduled_request_count,
        superseded_request_count=result.superseded_request_count,
        cancelled_request_count=result.cancelled_request_count,
        timed_out_request_count=result.timed_out_request_count,
        retrieval_error_count=result.retrieval_error_count,
        stale_result_discard_count=result.stale_result_discard_count,
        stale_result_accepted_count=int(stale_accepted),
        final_evidence_chunk_ids=final_ids,
        expected_final_evidence_chunk_ids=list(case.expected_final_evidence_chunk_ids),
        retrieved_ids=final_ids,
        relevant_ids=relevant_ids,
        request_status=_request_status_summary(
            request_statuses,
            fallback=("closed" if result.run_status == "closed" else "not_scheduled"),
        ),
        request_statuses=request_statuses,
        request_queries=request_queries,
        request_details=request_details,
        reuse_decision=reuse_decision,
        reuse_validation=reuse_check,
        scoring_denominator=len(relevant_ids),
        quality_scored=quality_scored,
        quality_exclusion_reason=(
            None
            if quality_scored
            else "no relevance labels"
            if not relevant_ids
            else "streaming run did not produce successful final evidence"
        ),
        end_to_end_success=result.run_status == "completed" and not result.errors,
        final_evidence_recall_at_k=quality,
        final_evidence_mrr=mrr,
        answer_latency_from_final_event_delivery_ms=result.answer_latency_from_final_event_delivery_ms,
        full_interaction_duration_ms=result.full_interaction_duration_ms,
        controller_overhead_ms=result.controller_overhead_ms,
        timing_sample_count=(
            1 if result.answer_latency_from_final_event_delivery_ms is not None else 0
        ),
        generation_status=result.generation_status,
        generation_usage=result.generation_usage,
        retrieval_usage=result.retrieval_usage,
        generation_cost=generation_cost,
        cost_availability=cost_availability,
        errors=_error_types(result.errors),
        trace_timeline=_trace_timeline(traces),
    )


def _error_mode_result(
    *,
    case: StreamingEvaluationCase,
    mode: Literal["baseline", "streaming"],
    execution_mode: Literal["realtime", "accelerated"],
    error: Exception,
) -> StreamingModeEvaluationResult:
    relevant_ids = list(case.expected_final_evidence_chunk_ids)
    return StreamingModeEvaluationResult(
        mode=mode,
        execution_mode=execution_mode,
        run_status="failed",
        trace_complete=False,
        missing_trace_events=["replay_result_not_produced"],
        trace_event_count=0,
        expected_final_evidence_chunk_ids=list(case.expected_final_evidence_chunk_ids),
        relevant_ids=relevant_ids,
        scoring_denominator=len(relevant_ids),
        quality_exclusion_reason="replay result was not produced",
        errors=[type(error).__name__],
        cost_availability="unavailable",
        timing_sample_count=0,
    )


async def _session_close_probe(
    case: StreamingEvaluationCase,
    *,
    corpus: CorpusIndex,
    retriever: Retriever,
    top_k: int,
    execution_mode: Literal["realtime", "accelerated"],
    controller_config: StreamingDecisionConfig,
    scheduler_config: StreamingSchedulerConfig,
) -> StreamingModeEvaluationResult:
    """Close a streaming session after an early decision, before finality."""

    event = next((item for item in case.transcript if not item.is_final), case.transcript[0])
    run_id = f"stream-eval-{case.case_id}-close"
    traces = TraceCollector(run_id, case.session_id, "flowcontext.streaming-controller.v1")
    controller = StreamingDecisionController(
        session_id=case.session_id,
        utterance_id=event.utterance_id,
        config=controller_config,
        known_chunk_locations={chunk.chunk_id: chunk.source_location for chunk in corpus.chunks},
    )
    scheduler = AsyncRetrievalScheduler(
        controller=controller,
        retriever=retriever,
        traces=traces,
        retrieval_model_identity=f"index:{corpus.manifest.index_id}",
        config=scheduler_config,
    )
    traces.add(
        "streaming_replay_started",
        source_timestamp_s=event.source_timestamp_s,
        attributes={"mode": "streaming", "execution_mode": execution_mode, "session_close_probe": True},
    )
    delivery_time = traces.elapsed_s()
    traces.add(
        "transcript_event_received",
        source_timestamp_s=event.source_timestamp_s,
        actual_delivery_time_s=delivery_time,
        attributes={
            "event_id": event.event_id,
            "is_final": False,
            "text_mode": event.text_mode,
            "actual_delivery_time_s": delivery_time,
        },
    )
    decision = controller.process_event(event)
    traces.add(
        "streaming_decision",
        source_timestamp_s=decision.source_timestamp_s,
        attributes={
            "decision": decision.decision,
            "reason_code": decision.reason_code,
            "is_final": False,
            "controller_decision_time_s": traces.elapsed_s(),
        },
    )
    if decision.decision == "RETRIEVE" and decision.proposed_query:
        scheduler.schedule(decision)
        await asyncio.sleep(0)
    await scheduler.close()
    controller.close()
    traces.add(
        "streaming_session_closed",
        source_timestamp_s=event.source_timestamp_s,
        attributes={"late_results_published": False, "session_id": case.session_id},
    )
    results = scheduler.results
    stale_accepted = sum(item.accepted and item.stale for item in results)
    request_statuses, request_queries, request_details = _streaming_request_details(results)
    return StreamingModeEvaluationResult(
        mode="streaming",
        execution_mode=execution_mode,
        run_status="closed",
        trace_complete=True,
        trace_event_count=len(traces.events),
        final_decision=decision.decision,
        decision_reasons=[decision.reason_code],
        retrieval_started_early=any(
            trace.event_type == "streaming_retrieval_started" for trace in traces.events
        ),
        retrieval_call_count=scheduler.retrieval_call_count,
        scheduled_request_count=scheduler.scheduled_request_count,
        superseded_request_count=scheduler.superseded_request_count,
        cancelled_request_count=scheduler.cancelled_request_count,
        timed_out_request_count=scheduler.timed_out_request_count,
        retrieval_error_count=scheduler.retrieval_error_count,
        stale_result_discard_count=scheduler.stale_result_count,
        stale_result_accepted_count=int(stale_accepted),
        expected_final_evidence_chunk_ids=list(case.expected_final_evidence_chunk_ids),
        relevant_ids=list(case.expected_final_evidence_chunk_ids),
        request_status=_request_status_summary(request_statuses, fallback="closed"),
        request_statuses=request_statuses,
        request_queries=request_queries,
        request_details=request_details,
        reuse_decision="session_closed_before_final",
        scoring_denominator=len(case.expected_final_evidence_chunk_ids),
        quality_exclusion_reason="session closed before final query evaluation",
        generation_status="not_run",
        generation_usage=Usage(),
        retrieval_usage=scheduler.retrieval_usage,
        generation_cost="unavailable",
        cost_availability="not_applicable",
        errors=[],
        trace_timeline=_trace_timeline(traces.events),
    )


async def _run_mode(
    case: StreamingEvaluationCase,
    *,
    mode: Literal["baseline", "streaming"],
    corpus: CorpusIndex,
    base_retriever: Retriever,
    top_k: int,
    settings: Settings,
    execution_mode: Literal["realtime", "accelerated"],
    controller_config: StreamingDecisionConfig | None = None,
    scheduler_config: StreamingSchedulerConfig | None = None,
) -> StreamingModeEvaluationResult:
    scenario_retriever = _ScenarioRetriever(
        base_retriever,
        behavior=case.retrieval_behavior,
        delay_s=case.simulated_retrieval_delay_s,
    )
    effective_scheduler_config = scheduler_config or scheduler_config_from_settings(settings)
    if case.retrieval_behavior == "timeout":
        effective_scheduler_config = effective_scheduler_config.model_copy(
            update={"request_timeout_s": case.retrieval_timeout_s}
        )
    if mode == "streaming" and case.retrieval_behavior == "session_close":
        return await _session_close_probe(
            case,
            corpus=corpus,
            retriever=scenario_retriever,
            top_k=top_k,
            execution_mode=execution_mode,
            controller_config=controller_config or streaming_config_from_settings(settings),
            scheduler_config=effective_scheduler_config,
        )
    provider = generation_provider_for_settings(settings)
    try:
        if mode == "baseline":
            result = await replay_transcript(
                case.transcript,
                corpus=corpus,
                top_k=top_k,
                backend=base_retriever.backend,
                retriever=scenario_retriever,
                generation_provider=provider,
                run_id=f"stream-eval-{case.case_id}-baseline",
                execution_mode=execution_mode,
                previous_answer_context=case.previous_answer_context,
            )
            return _baseline_mode_result(case, result)
        result = await replay_streaming_transcript(
            case.transcript,
            corpus=corpus,
            top_k=top_k,
            backend=base_retriever.backend,
            retriever=scenario_retriever,
            generation_provider=provider,
            run_id=f"stream-eval-{case.case_id}-streaming",
            execution_mode=execution_mode,
            controller_config=controller_config or streaming_config_from_settings(settings),
            scheduler_config=effective_scheduler_config,
            previous_answer_context=case.previous_answer_context,
        )
        return _streaming_mode_result(case, result)
    except Exception as exc:
        return _error_mode_result(
            case=case,
            mode=mode,
            execution_mode=execution_mode,
            error=exc,
        )


async def _run_case_pair(
    case: StreamingEvaluationCase,
    *,
    corpus: CorpusIndex,
    base_retriever: Retriever,
    top_k: int,
    settings: Settings,
    execution_mode: Literal["realtime", "accelerated"],
) -> StreamingEvaluationCaseResult:
    baseline, streaming = await asyncio.gather(
        _run_mode(
            case,
            mode="baseline",
            corpus=corpus,
            base_retriever=base_retriever,
            top_k=top_k,
            settings=settings,
            execution_mode=execution_mode,
        ),
        _run_mode(
            case,
            mode="streaming",
            corpus=corpus,
            base_retriever=base_retriever,
            top_k=top_k,
            settings=settings,
            execution_mode=execution_mode,
        ),
    )
    if case.case_id not in _REPORT_TIMELINE_CASES:
        baseline = baseline.model_copy(update={"trace_timeline": []})
        streaming = streaming.model_copy(update={"trace_timeline": []})
    failures: list[str] = []
    if case.expected_no_retrieval:
        if baseline.retrieval_call_count or streaming.retrieval_call_count:
            failures.append("false_retrieval_trigger")
    if case.expected_final_decision is not None and streaming.final_decision != case.expected_final_decision:
        failures.append(
            f"final_decision_expected_{case.expected_final_decision.lower()}_got_"
            f"{(streaming.final_decision or 'none').lower()}"
        )
    if case.eligible_for_early_retrieval and not streaming.retrieval_started_early:
        failures.append("eligible_case_without_actual_early_start")
    if not case.eligible_for_early_retrieval and streaming.retrieval_started_early:
        failures.append("premature_early_trigger")
    if streaming.stale_result_accepted_count:
        failures.append("stale_result_accepted")
    if case.expected_final_evidence_chunk_ids and not (
        streaming.final_evidence_recall_at_k.get("recall_at_5", 0.0) >= 1.0
    ) and case.retrieval_behavior not in {"failure", "timeout", "session_close"}:
        failures.append("streaming_final_evidence_missing")
    if not baseline.trace_complete:
        failures.append("baseline_trace_incomplete")
    if not streaming.trace_complete:
        failures.append("streaming_trace_incomplete")
    if case.retrieval_behavior == "session_close" and streaming.run_status != "closed":
        failures.append("session_close_not_observed")
    notes = [
        "labels are provisional generated engineering labels; no human verification was performed",
        "semantic claim support is not evaluated by this harness",
    ]
    if case.concurrency_group:
        notes.append(f"paired in concurrent session group {case.concurrency_group}")
    return StreamingEvaluationCaseResult(
        case_id=case.case_id,
        split=case.split,
        scenario_group=case.scenario_group,
        asset_status=case.asset_status,
        labels_status=case.relevance_label_status,
        baseline=baseline,
        streaming=streaming,
        expected_behavior_met=not failures,
        failure_flags=failures,
        notes=notes,
    )


async def _run_case_groups(
    cases: Sequence[StreamingEvaluationCase],
    *,
    corpus: CorpusIndex,
    base_retriever: Retriever,
    top_k: int,
    settings: Settings,
    execution_mode: Literal["realtime", "accelerated"],
) -> list[StreamingEvaluationCaseResult]:
    """Run declared concurrency groups together, other cases one at a time."""

    grouped: dict[str, list[StreamingEvaluationCase]] = {}
    singles: list[StreamingEvaluationCase] = []
    for case in cases:
        if case.concurrency_group:
            grouped.setdefault(case.concurrency_group, []).append(case)
        else:
            singles.append(case)
    outputs: list[StreamingEvaluationCaseResult] = []
    for case in singles:
        outputs.append(
            await _run_case_pair(
                case,
                corpus=corpus,
                base_retriever=base_retriever,
                top_k=top_k,
                settings=settings,
                execution_mode=execution_mode,
            )
        )
    for group_cases in grouped.values():
        outputs.extend(
            await asyncio.gather(
                *(
                    _run_case_pair(
                        case,
                        corpus=corpus,
                        base_retriever=base_retriever,
                        top_k=top_k,
                        settings=settings,
                        execution_mode=execution_mode,
                    )
                    for case in group_cases
                )
            )
        )
    case_order = {case.case_id: index for index, case in enumerate(cases)}
    outputs.sort(key=lambda item: case_order[item.case_id])
    return outputs


def _section_metrics(
    cases: Sequence[StreamingEvaluationCase],
    results: Sequence[StreamingEvaluationCaseResult],
    *,
    execution_mode: Literal["realtime", "accelerated"],
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any], dict[str, int]]:
    by_id = {case.case_id: case for case in cases}
    streaming = [item.streaming for item in results if item.streaming is not None]
    baseline = [item.baseline for item in results if item.baseline is not None]
    eligible = [
        item.streaming
        for item in results
        if by_id[item.case_id].eligible_for_early_retrieval and item.streaming is not None
    ]
    expected_no_retrieval = [
        item
        for item in results
        if by_id[item.case_id].expected_no_retrieval and item.streaming is not None
    ]
    ineligible_non_no_retrieval = [
        item
        for item in results
        if not by_id[item.case_id].eligible_for_early_retrieval
        and not by_id[item.case_id].expected_no_retrieval
        and item.streaming is not None
    ]
    early_count = sum(item.retrieval_started_early for item in eligible)
    valid_count = sum(item.valid_evidence_ready_before_finalization for item in eligible)
    reuse_count = sum(item.early_evidence_reused for item in eligible)
    false_count = sum(
        item.streaming is not None and item.streaming.retrieval_call_count > 0
        for item in expected_no_retrieval
    )
    premature_count = sum(
        item.streaming is not None and item.streaming.retrieval_started_early
        for item in ineligible_non_no_retrieval
    )
    calls = sum(item.retrieval_call_count for item in streaming)
    stale_accepted = sum(item.stale_result_accepted_count for item in streaming)
    final_quality = [
        item.final_evidence_recall_at_k.get("recall_at_5")
        for item in streaming
        if item.quality_scored and item.final_evidence_recall_at_k
    ]
    metrics: dict[str, float] = {
        "early_retrieval_numerator": float(early_count),
        "early_retrieval_denominator": float(len(eligible)),
        "early_retrieval_rate_eligible": (
            early_count / len(eligible) if eligible else 0.0
        ),
        "valid_evidence_before_finalization_numerator": float(valid_count),
        "valid_evidence_before_finalization_denominator": float(len(eligible)),
        "valid_evidence_ready_before_finalization_rate": (
            valid_count / len(eligible) if eligible else 0.0
        ),
        "useful_early_result_reuse_numerator": float(reuse_count),
        "useful_early_result_reuse_denominator": float(len(eligible)),
        "useful_early_result_reuse_rate": reuse_count / len(eligible) if eligible else 0.0,
        "false_retrieval_trigger_count": float(false_count),
        "false_retrieval_trigger_denominator": float(len(expected_no_retrieval)),
        "premature_trigger_count": float(premature_count),
        "premature_trigger_denominator": float(len(ineligible_non_no_retrieval)),
        "streaming_total_retrieval_calls": float(calls),
        "streaming_retrieval_calls_per_utterance": calls / len(streaming) if streaming else 0.0,
        "baseline_total_retrieval_calls": float(sum(item.retrieval_call_count for item in baseline)),
        "baseline_retrieval_calls_per_utterance": (
            sum(item.retrieval_call_count for item in baseline) / len(baseline)
            if baseline
            else 0.0
        ),
        "stale_result_acceptance_count": float(stale_accepted),
        "trace_complete_count": float(sum(item.trace_complete for item in streaming)),
        "trace_sample_count": float(len(streaming)),
        # Count recorded error events once; retrieval_error_count is already
        # represented in the trace-derived errors and must not be added again.
        "error_count": float(sum(len(item.errors) for item in streaming)),
        "error_case_count": float(sum(bool(item.errors) for item in streaming)),
        "streaming_quality_successful_denominator": float(len(final_quality)),
        "streaming_end_to_end_completed_count": float(
            sum(item.run_status == "completed" for item in streaming)
        ),
        "streaming_end_to_end_case_count": float(len(streaming)),
    }
    if final_quality:
        metrics["streaming_final_evidence_recall_at_5_mean"] = sum(final_quality) / len(final_quality)
    baseline_quality = [
        item.final_evidence_recall_at_k.get("recall_at_5")
        for item in baseline
        if item.quality_scored and item.final_evidence_recall_at_k
    ]
    if baseline_quality:
        metrics["baseline_final_evidence_recall_at_5_mean"] = sum(baseline_quality) / len(baseline_quality)
    metrics["baseline_quality_successful_denominator"] = float(len(baseline_quality))
    metrics["baseline_end_to_end_completed_count"] = float(
        sum(item.run_status == "completed" for item in baseline)
    )
    metrics["baseline_end_to_end_case_count"] = float(len(baseline))
    baseline_answer = [
        item.answer_latency_from_final_event_delivery_ms
        for item in baseline
        if item.answer_latency_from_final_event_delivery_ms is not None
    ]
    streaming_answer = [
        item.answer_latency_from_final_event_delivery_ms
        for item in streaming
        if item.answer_latency_from_final_event_delivery_ms is not None
    ]
    streaming_full = [
        item.full_interaction_duration_ms
        for item in streaming
        if item.full_interaction_duration_ms is not None
    ]
    controller = [
        item.controller_overhead_ms
        for item in streaming
        if item.controller_overhead_ms is not None
    ]
    timing = {
        "baseline_answer_latency_from_final_event_delivery_ms": _timing_summary(
            baseline_answer, execution_mode=execution_mode
        ),
        "streaming_answer_latency_from_final_event_delivery_ms": _timing_summary(
            streaming_answer, execution_mode=execution_mode
        ),
        "streaming_full_interaction_duration_ms": _timing_summary(
            streaming_full, execution_mode=execution_mode
        ),
        "streaming_controller_overhead_ms": _timing_summary(
            controller, execution_mode=execution_mode
        ),
    }
    all_mode_results = [*baseline, *streaming]
    usage = {
        "baseline_generation_tokens": sum(item.generation_usage.total_tokens for item in baseline),
        "streaming_generation_tokens": sum(item.generation_usage.total_tokens for item in streaming),
        "baseline_retrieval_tokens": sum(item.retrieval_usage.total_tokens for item in baseline),
        "streaming_retrieval_tokens": sum(item.retrieval_usage.total_tokens for item in streaming),
        "generation_cost": (
            "unavailable"
            if not all_mode_results or any(item.generation_cost == "unavailable" for item in all_mode_results)
            else sum(
                float(item.generation_cost)
                for item in all_mode_results
                if isinstance(item.generation_cost, (int, float))
            )
        ),
        "cost_availability": (
            "not_applicable"
            if not all_mode_results
            else "unavailable"
            if any(item.cost_availability == "unavailable" for item in all_mode_results)
            else "available"
        ),
    }
    trace_coverage = {
        "streaming_trace_complete": sum(item.trace_complete for item in streaming),
        "streaming_trace_samples": len(streaming),
        "baseline_trace_complete": sum(item.trace_complete for item in baseline),
        "baseline_trace_samples": len(baseline),
    }
    return metrics, timing, usage, trace_coverage


def _section(
    *,
    section_id: Literal["fixture", "simulated_delay"],
    cases: Sequence[StreamingEvaluationCase],
    results: Sequence[StreamingEvaluationCaseResult],
    corpus: CorpusIndex,
    base_retriever: Retriever,
    settings: Settings,
    execution_mode: Literal["realtime", "accelerated"],
) -> StreamingEvaluationSection:
    metrics, timing, usage, trace_coverage = _section_metrics(
        cases, results, execution_mode=execution_mode
    )
    return StreamingEvaluationSection(
        section_id=section_id,
        status="measured",
        result_group=(
            "synthetic lexical/mock fixture cases"
            if section_id == "fixture"
            else "synthetic fixture cases with bounded simulated retrieval behavior"
        ),
        asset_status="synthetic_fixture",
        corpus_id=corpus.corpus_id,
        index_id=corpus.manifest.index_id,
        corpus_source_kind=corpus.source_kind,
        retrieval_backend=base_retriever.backend,
        generation_backend=settings.generation_backend,
        generation_model=settings.generation_model,
        execution_mode=execution_mode,
        case_count=len(cases),
        case_results=list(results),
        metrics=metrics,
        timing=timing,
        usage=usage,
        cost_availability=str(usage["cost_availability"]),
        trace_coverage=trace_coverage,
        limitations=[
            "Synthetic fixture and provisional generated labels; not an official benchmark.",
            "Semantic claim support and answer usefulness were not human-reviewed.",
            (
                "Accelerated timing is reported for scheduling behavior only and must not "
                "be compared as model latency with realtime replay."
            )
            if execution_mode == "accelerated"
            else "Realtime timing is local process observation, not competitive performance.",
        ],
    )


async def evaluate_streaming_suite(
    cases: Sequence[StreamingEvaluationCase],
    *,
    corpus: CorpusIndex,
    settings: Settings,
    backend: str,
    top_k: int,
    execution_mode: Literal["realtime", "accelerated"],
    run_id_prefix: str = "streaming-evaluation",
    compare_without_suppression: bool = False,
) -> StreamingEvaluationReport:
    """Run matched baseline/streaming cases and build the audit report."""

    if not cases:
        raise StreamingEvaluationError("at least one streaming evaluation case is required")
    if execution_mode not in {"realtime", "accelerated"}:
        raise StreamingEvaluationError(f"unsupported execution mode: {execution_mode}")
    case_splits = {case.split for case in cases}
    labels = {case.relevance_label_status for case in cases}
    labels_status = (
        "human_reviewed"
        if labels == {"human_reviewed"}
        else "human_review_pending"
        if "human_review_pending" in labels
        else "provisional_generated"
    )
    base_retriever = make_retriever(
        corpus,
        backend=backend,
        top_k=top_k,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
    )
    results = await _run_case_groups(
        cases,
        corpus=corpus,
        base_retriever=base_retriever,
        top_k=top_k,
        settings=settings,
        execution_mode=execution_mode,
    )
    case_by_id = {case.case_id: case for case in cases}
    fixture_cases = [case for case in cases if case.retrieval_behavior == "normal"]
    simulated_cases = [case for case in cases if case.retrieval_behavior != "normal"]
    fixture_results = [result for result in results if case_by_id[result.case_id].retrieval_behavior == "normal"]
    simulated_results = [result for result in results if case_by_id[result.case_id].retrieval_behavior != "normal"]
    fixture_section = _section(
        section_id="fixture",
        cases=fixture_cases,
        results=fixture_results,
        corpus=corpus,
        base_retriever=base_retriever,
        settings=settings,
        execution_mode=execution_mode,
    )
    simulated_section = _section(
        section_id="simulated_delay",
        cases=simulated_cases,
        results=simulated_results,
        corpus=corpus,
        base_retriever=base_retriever,
        settings=settings,
        execution_mode=execution_mode,
    )

    comparison: dict[str, Any] = {
        "status": "not_run",
        "case_id": None,
        "duplicate_suppression_enabled": True,
        "coalescing_enabled": True,
        "resource_bound": {
            "max_total_requests": settings.streaming_max_total_requests,
            "max_concurrency": settings.streaming_max_concurrency,
            "max_pending_requests": settings.streaming_max_pending_requests,
        },
    }
    comparison_case = next(
        (case for case in cases if case.compare_without_suppression),
        None,
    )
    if compare_without_suppression and comparison_case is not None:
        disabled_config = streaming_config_from_settings(settings).model_copy(
            update={"duplicate_query_suppression": False, "debounce_source_s": 0.0}
        )
        disabled = await _run_mode(
            comparison_case,
            mode="streaming",
            corpus=corpus,
            base_retriever=base_retriever,
            top_k=top_k,
            settings=settings,
            execution_mode=execution_mode,
            controller_config=disabled_config,
        )
        normal = next(
            result.streaming
            for result in results
            if result.case_id == comparison_case.case_id and result.streaming is not None
        )
        comparison = {
            "status": "measured",
            "case_id": comparison_case.case_id,
            "duplicate_suppression_enabled": {
                "duplicate_query_suppression": True,
                "debounce_source_s": settings.streaming_debounce_source_s,
                "retrieval_calls": normal.retrieval_call_count,
                "superseded_requests": normal.superseded_request_count,
                "stale_result_discards": normal.stale_result_discard_count,
                "final_evidence_chunk_ids": normal.final_evidence_chunk_ids,
            },
            "duplicate_suppression_disabled": {
                "duplicate_query_suppression": False,
                "debounce_source_s": 0.0,
                "retrieval_calls": disabled.retrieval_call_count,
                "superseded_requests": disabled.superseded_request_count,
                "stale_result_discards": disabled.stale_result_discard_count,
                "final_evidence_chunk_ids": disabled.final_evidence_chunk_ids,
            },
            "additional_calls_when_disabled": (
                disabled.retrieval_call_count - normal.retrieval_call_count
            ),
            "final_evidence_quality_unchanged": (
                disabled.final_evidence_chunk_ids == normal.final_evidence_chunk_ids
            ),
            "wasted_request_reduction_observed": (
                disabled.retrieval_call_count > normal.retrieval_call_count
                or disabled.superseded_request_count > normal.superseded_request_count
            ),
            "execution_mode": execution_mode,
            "interpretation": (
                "Focused policy comparison only; no semantic quality claim and no "
                "accelerated-versus-realtime latency comparison."
            ),
        }

    measured_results = [*fixture_results, *simulated_results]
    measured_cases = [*fixture_cases, *simulated_cases]
    measured_metrics, measured_timing, measured_usage, measured_trace_coverage = _section_metrics(
        measured_cases,
        measured_results,
        execution_mode=execution_mode,
    )
    eligible_denominator = int(measured_metrics["early_retrieval_denominator"])
    eligible_numerator = int(measured_metrics["early_retrieval_numerator"])
    early_rate = (
        eligible_numerator / eligible_denominator if eligible_denominator else 0.0
    )
    target_met = bool(eligible_denominator and early_rate >= 0.8)
    unexpected_failures = [
        result
        for result in measured_results
        if not result.expected_behavior_met
        and "streaming_trace_incomplete" in result.failure_flags
        and case_by_id[result.case_id].retrieval_behavior in {"normal", "delay"}
    ]
    passed = (
        not unexpected_failures
        and measured_metrics["stale_result_acceptance_count"] == 0
        and measured_metrics["false_retrieval_trigger_count"] == 0
        and measured_metrics["premature_trigger_count"] == 0
    )
    sections = {
        "fixture": fixture_section,
        "simulated_delay": simulated_section,
        "real_backend": StreamingEvaluationSection(
            section_id="real_backend",
            status="not_verified",
            result_group="real retrieval/generation backend validation",
            asset_status="unknown",
            corpus_source_kind="unknown",
            case_count=0,
            cost_availability="not_available",
            limitations=[
                "No real dense/provider-backed Phase 2 run was executed.",
                "Real backend unavailability is not replaced with a mock result.",
            ],
        ),
        "official_assets": StreamingEvaluationSection(
            section_id="official_assets",
            status="not_verified",
            result_group="organiser corpus, labels, and benchmark assets",
            asset_status="unknown",
            corpus_source_kind="unknown",
            case_count=0,
            cost_availability="not_available",
            limitations=[
                "Official corpus, replay set, labels, thresholds, and scoring harness are absent.",
                "No official benchmark result is claimed.",
            ],
        ),
    }
    limitations = [
        "This is a local engineering audit over synthetic fixture data, not an official benchmark.",
        "All measured labels are provisional_generated; no human verification was performed.",
        "Final evidence quality uses chunk-ID retrieval labels and does not evaluate semantic claim support.",
        "The guide does not specify a numerical false-trigger threshold; the report does not invent one.",
        "Real dense retrieval, real generation, and official assets remain unverified.",
    ]
    return StreamingEvaluationReport(
        evaluation_label="phase2-streaming-local-audit",
        asset_status="synthetic_fixture",
        selected_split=(
            "all"
            if len(case_splits) > 1
            else next(iter(case_splits))
        ),
        development_case_count=sum(case.split == "development" for case in cases),
        held_out_case_count=sum(case.split == "held_out" for case in cases),
        measured_case_count=len(cases),
        labels_status=labels_status,
        passed=passed,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        code_revision=_code_revision(),
        configuration={
            "backend": base_retriever.backend,
            "top_k": top_k,
            "execution_mode": execution_mode,
            "generation_backend": settings.generation_backend,
            "generation_provider": settings.generation_provider,
            "generation_model": settings.generation_model,
            "streaming_policy": streaming_config_from_settings(settings).model_dump(mode="json"),
            "scheduler": scheduler_config_from_settings(settings).model_dump(mode="json"),
            "case_splits": sorted(case_splits),
            "run_id_prefix": run_id_prefix,
        },
        corpus_id=corpus.corpus_id,
        index_id=corpus.manifest.index_id,
        model_identities={
            "retrieval": f"{base_retriever.backend}:{corpus.manifest.embedding.provider}",
            "generation": settings.generation_model,
        },
        execution_mode=execution_mode,
        warm_cold_condition=(
            "single process; one shared index/retriever; provider configuration reused; "
            "case execution is bounded and declared concurrency groups run concurrently"
        ),
        environment=_environment(),
        hardware=_hardware(),
        guide_early_retrieval_target=0.8,
        guide_early_retrieval_target_met=target_met,
        false_trigger_threshold="not specified by the guide",
        metrics={
            **measured_metrics,
            "early_retrieval_rate_eligible": early_rate,
            "eligible_case_count": float(eligible_denominator),
            "eligible_early_start_count": float(eligible_numerator),
            "guide_target_met": float(target_met),
            "measured_case_count": float(len(cases)),
            "measured_trace_complete_count": float(measured_trace_coverage["streaming_trace_complete"]),
        },
        sections=sections,
        duplicate_suppression_comparison=comparison,
        limitations=limitations,
    )


def dump_concise_trace(result: StreamingModeEvaluationResult) -> str:
    """Render a compact timeline for README/report excerpts."""

    parts: list[str] = []
    for item in result.trace_timeline:
        event = item["event"]
        time_s = item.get("actual_time_s")
        suffix = ""
        if event == "streaming_decision":
            suffix = f" {item.get('decision')}/{item.get('reason_code')}"
        elif event in {"streaming_retrieval_started", "retrieval_started"}:
            suffix = " actual-start"
        elif event in {"streaming_evidence_ready", "evidence_ready"}:
            suffix = " evidence-ready"
        elif event == "final_event_delivered":
            suffix = " final"
        parts.append(f"{event}@{time_s:.6f}s{suffix}" if isinstance(time_s, (int, float)) else event)
    return " -> ".join(parts)
