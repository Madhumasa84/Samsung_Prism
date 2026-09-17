"""Explainable Phase 2 streaming decisions and optional Phase 3 answer replay.

The controller remains a small synchronous policy boundary. The replay runner
connects it to :mod:`flowcontext.scheduler` for bounded asynchronous retrieval,
while deliberately leaving early answer generation and selective answer
refinement to later phases. Phase 3 multi-intent behavior is opt-in and keeps
the final-only generation boundary.
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from pydantic import ValidationError

from .config import Settings
from .contracts import (
    Answer,
    CorpusIndex,
    PreviousAnswerContext,
    RetrievalHit,
    RetrievalMode,
    StreamingDecision,
    StreamingDecisionConfig,
    StreamingReplayResult,
    StreamingRetrievalRequest,
    StreamingRetrievalResult,
    StreamingSchedulerConfig,
    StreamingRunManifest,
    TraceError,
    TranscriptEvent,
    Usage,
)
from .generation import (
    GenerationOutcome,
    GenerationProvider,
    MockGenerationProvider,
    generate_grounded_answer,
    generation_model_identity,
)
from .multi_intent import (
    DecompositionProvider,
    MultiIntentRetriever,
    StructuredMultiIntentDecomposer,
    intent_metadata_from_hits,
    make_multi_intent_retriever,
    reuse_validation,
    single_query_evidence_is_appropriate,
    unsupported_intent_queries_from_hits,
)
from .replay import (
    ReplayError,
    _detect_code_revision,
    _retrieval_model_identity,
    _wait_for_source_timing,
    normalize_transcript,
)
from .retrieval import Retriever, make_retriever, tokenize
from .trace import TraceCollector


class StreamingControllerError(ValueError):
    """Raised for invalid controller state or an unusable streaming request."""


class StreamingReplayError(ReplayError):
    """Raised when a streaming replay cannot produce a valid result."""


_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")

# Policy vocabulary, not corpus-specific answers. These sets make the default
# controller explainable and can be replaced by a later policy layer.
_FUNCTION_WORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "am",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "keep",
        "make",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "put",
        "should",
        "that",
        "the",
        "their",
        "this",
        "to",
        "turn",
        "us",
        "we",
        "what",
        "when",
        "where",
        "which",
        "with",
        "write",
        "would",
        "you",
        "your",
    }
)
_QUESTION_SIGNALS = frozenset(
    {
        "can",
        "compare",
        "could",
        "do",
        "does",
        "explain",
        "find",
        "give",
        "how",
        "is",
        "list",
        "show",
        "tell",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "would",
    }
)
_REQUEST_SIGNALS = frozenset(
    {
        "book",
        "compare",
        "find",
        "get",
        "need",
        "recommend",
        "reserve",
        "show",
        "tell",
        "want",
        "looking",
        "keep",
        "make",
        "put",
        "search",
        "turn",
        "write",
    }
)
_CORRECTION_MARKERS = frozenset(
    {"actually", "change", "changed", "correction", "instead", "rather", "sorry", "wait"}
)
_FORMATTING_MARKERS = frozenset(
    {
        "bullet",
        "bullets",
        "format",
        "formatted",
        "headings",
        "list",
        "numbered",
        "paragraph",
        "short",
        "shorter",
        "table",
        "summarize",
        "summary",
        "concise",
    }
)
_CONSTRAINT_MARKERS = frozenset(
    {
        "attendee",
        "attendees",
        "available",
        "budget",
        "capacity",
        "catering",
        "date",
        "day",
        "distance",
        "parking",
        "people",
        "price",
        "time",
        "under",
        "over",
        "least",
        "most",
        "minimum",
        "maximum",
        "near",
        "nearby",
        "within",
        "without",
        "vegetarian",
        "vegan",
    }
)
_INCOMPLETE_ENDINGS = frozenset(
    {
        "about",
        "after",
        "and",
        "at",
        "before",
        "can",
        "compare",
        "for",
        "from",
        "how",
        "if",
        "in",
        "is",
        "list",
        "of",
        "on",
        "or",
        "please",
        "tell",
        "that",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
)
_GREETING_MARKERS = frozenset(
    {
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank",
        "good",
        "morning",
        "afternoon",
        "evening",
    }
)


@dataclass(frozen=True)
class _QueryProfile:
    tokens: frozenset[str]
    topic_terms: frozenset[str]
    question_signal: bool
    request_signal: bool
    correction: bool
    constraint: bool
    formatting_only: bool
    greeting_only: bool
    incomplete: bool


def _tokens(text: str) -> frozenset[str]:
    return frozenset(token.lower() for token in _TOKEN_PATTERN.findall(text))


def _canonical_query(text: str) -> str:
    return " ".join(token.lower() for token in _TOKEN_PATTERN.findall(text))


def _constraint_signature(text: str) -> frozenset[str]:
    tokens = set(_tokens(text)) & _CONSTRAINT_MARKERS
    tokens.update(f"number:{value}" for value in _NUMBER_PATTERN.findall(text))
    return frozenset(tokens)


def _profile(text: str) -> _QueryProfile:
    tokens = _tokens(text)
    topic_terms = frozenset(tokens - _FUNCTION_WORDS - _FORMATTING_MARKERS)
    question_signal = bool(tokens & _QUESTION_SIGNALS) or "?" in text
    request_signal = bool(tokens & _REQUEST_SIGNALS)
    correction = bool(tokens & _CORRECTION_MARKERS) or bool(
        re.search(r"\b(?:no|not)\s+(?:i\s+mean|rather)\b", text.lower())
    )
    constraint = bool(_constraint_signature(text))
    formatting_only = bool(tokens & _FORMATTING_MARKERS) and not topic_terms and not constraint
    greeting_only = bool(tokens) and tokens.issubset(_GREETING_MARKERS)
    final_tokens = _TOKEN_PATTERN.findall(text.lower())
    last_token = final_tokens[-1] if final_tokens else ""
    incomplete = bool(text.strip()) and not text.rstrip().endswith((".", "?", "!")) and (
        last_token in _INCOMPLETE_ENDINGS or (len(tokens) <= 1 and not constraint)
    )
    return _QueryProfile(
        tokens=tokens,
        topic_terms=topic_terms,
        question_signal=question_signal,
        request_signal=request_signal,
        correction=correction,
        constraint=constraint,
        formatting_only=formatting_only,
        greeting_only=greeting_only,
        incomplete=incomplete,
    )


def streaming_config_from_settings(settings: Settings) -> StreamingDecisionConfig:
    """Convert environment-backed settings to the validated controller contract."""

    return StreamingDecisionConfig(
        debounce_source_s=settings.streaming_debounce_source_s,
        min_query_chars=settings.streaming_min_query_chars,
        min_topic_terms=settings.streaming_min_topic_terms,
        min_new_content_terms=settings.streaming_min_new_content_terms,
        retrieve_on_correction=settings.streaming_retrieve_on_correction,
        retrieve_on_constraint_change=settings.streaming_retrieve_on_constraint_change,
        duplicate_query_suppression=settings.streaming_duplicate_query_suppression,
        final_decision_bypasses_debounce=settings.streaming_final_bypasses_debounce,
    )


class StreamingDecisionController:
    """Stateful, session-isolated policy for partial transcript decisions.

    The controller is synchronous. It emits a revision-bound request; the
    caller decides how to execute it. Retrieval revisions advance only when a
    new query is scheduled, so an in-flight request remains usable when a
    transcript event does not materially change its canonical query.
    """

    def __init__(
        self,
        *,
        session_id: str,
        utterance_id: str,
        config: StreamingDecisionConfig | None = None,
        previous_answer_context: PreviousAnswerContext | None = None,
        known_chunk_locations: Mapping[str, str] | None = None,
    ) -> None:
        if not session_id.strip() or not utterance_id.strip():
            raise StreamingControllerError("session_id and utterance_id must not be blank")
        if previous_answer_context is not None and previous_answer_context.session_id != session_id:
            raise StreamingControllerError(
                "previous answer context belongs to another session and cannot be used"
            )
        self.session_id = session_id
        self.utterance_id = utterance_id
        self.config = config or StreamingDecisionConfig()
        self.previous_answer_context = previous_answer_context
        self._known_chunk_locations = dict(known_chunk_locations or {})
        self._assembled_text = ""
        self._revision = 0
        self._decision_counter = 0
        self._request_counter = 0
        self._retrieval_revision_counter = 0
        self._latest_retrieval_revision = 0
        self._seen_events: dict[str, TranscriptEvent] = {}
        self._last_sequence: int | None = None
        self._last_source_timestamp: float | None = None
        self._closed = False
        self._final_text: str | None = None
        self._final_text_mode: str | None = None
        self._attempted_queries: set[str] = set()
        self._last_retrieval_query: str | None = None
        self._last_retrieval_source_timestamp: float | None = None
        self._completed_request_ids: set[str] = set()
        self._current_evidence_hits: list[RetrievalHit] = []
        self._current_evidence_query: str | None = None
        self._pending_event_ids: list[str] = []

    @property
    def current_text(self) -> str:
        return self._assembled_text.strip()

    @property
    def transcript_revision(self) -> int:
        return self._revision

    @property
    def latest_retrieval_revision(self) -> int:
        """Return the newest retrieval revision allocated in this session."""

        return self._latest_retrieval_revision

    @property
    def current_evidence_hits(self) -> list[RetrievalHit]:
        if self._current_evidence_query != _canonical_query(self.current_text):
            return []
        return list(self._current_evidence_hits)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Close the utterance/session boundary.

        The asynchronous scheduler owns tasks and calls this only after it has
        stopped publication. Closing prevents new controller state mutation;
        scheduler cleanup is deliberately separate from this synchronous
        policy object.
        """

        self._closed = True

    def _validate_event_order(self, event: TranscriptEvent) -> tuple[bool, str | None]:
        previous = self._seen_events.get(event.event_id)
        if previous is not None:
            if previous.model_dump(mode="json") != event.model_dump(mode="json"):
                raise StreamingControllerError(
                    f"event_id {event.event_id!r} was repeated with conflicting event content"
                )
            return True, "identical_event_replay"
        if self._last_source_timestamp is not None and event.source_timestamp_s < self._last_source_timestamp:
            raise StreamingControllerError(
                f"source_timestamp_s moved backwards at event {event.event_id!r}"
            )
        if self._last_sequence is not None and event.sequence_number <= self._last_sequence:
            raise StreamingControllerError(
                f"sequence_number must increase for event {event.event_id!r}"
            )
        if self._closed:
            if (
                event.is_final
                and event.text == self._final_text
                and event.text_mode == self._final_text_mode
            ):
                return True, "repeated_final"
            raise StreamingControllerError(
                f"event {event.event_id!r} arrived after utterance finalisation"
            )
        return False, None

    def process_event(
        self,
        event: TranscriptEvent,
        *,
        is_duplicate: bool | None = None,
        duplicate_reason: str | None = None,
    ) -> StreamingDecision:
        """Consume one validated event and return exactly one decision.

        is_duplicate may be supplied by the shared Phase 1 normalizer. When
        omitted, the controller applies the same local duplicate policy for
        direct callers and unit tests.
        """

        if event.session_id != self.session_id or event.utterance_id != self.utterance_id:
            raise StreamingControllerError("event session/utterance does not match controller state")
        detected_duplicate, detected_reason = self._validate_event_order(event)
        duplicate = detected_duplicate if is_duplicate is None else is_duplicate
        reason = duplicate_reason or detected_reason
        if duplicate and not detected_duplicate:
            raise StreamingControllerError("duplicate classification disagrees with event state")
        if duplicate:
            if reason is None:
                reason = "duplicate_event"
            self._seen_events.setdefault(event.event_id, event)
            if self._last_sequence is None or event.sequence_number > self._last_sequence:
                self._last_sequence = event.sequence_number
                self._last_source_timestamp = event.source_timestamp_s
            return self._make_decision(
                event,
                decision="SKIP",
                reason_code=(
                    "repeated_final_ignored"
                    if reason == "repeated_final"
                    else "duplicate_event_ignored"
                ),
                proposed_query=None,
                needs_previous_answer_context=False,
                previous_answer_context_used=False,
            )
        self._seen_events[event.event_id] = event
        self._last_sequence = event.sequence_number
        self._last_source_timestamp = event.source_timestamp_s
        self._revision += 1
        self._pending_event_ids.append(event.event_id)
        if event.text_mode == "cumulative":
            self._assembled_text = event.text
        else:
            self._assembled_text += event.text
        if event.is_final:
            query = self.current_text
            if not query:
                raise StreamingControllerError("final transcript produced an empty query")
            self._closed = True
            self._final_text = self._assembled_text
            self._final_text_mode = event.text_mode
        return self._decide(event, is_final=event.is_final)

    def _make_decision(
        self,
        event: TranscriptEvent,
        *,
        decision: Literal["WAIT", "RETRIEVE", "SKIP"],
        reason_code: str,
        proposed_query: str | None,
        needs_previous_answer_context: bool,
        previous_answer_context_used: bool,
    ) -> StreamingDecision:
        self._decision_counter += 1
        coalesced_event_ids = list(dict.fromkeys(self._pending_event_ids or [event.event_id]))
        result = StreamingDecision(
            decision_id=f"{self.session_id}-{self.utterance_id}-decision-{self._decision_counter:04d}",
            session_id=self.session_id,
            utterance_id=self.utterance_id,
            transcript_revision=self._revision,
            decision=decision,
            reason_code=reason_code,
            proposed_query=proposed_query,
            source_event_id=event.event_id,
            source_timestamp_s=event.source_timestamp_s,
            monotonic_decision_time_s=time.monotonic(),
            is_final_event=event.is_final,
            needs_previous_answer_context=needs_previous_answer_context,
            previous_answer_context_used=previous_answer_context_used,
            coalesced_event_ids=coalesced_event_ids,
        )
        if result.decision == "RETRIEVE" and result.proposed_query:
            canonical = _canonical_query(result.proposed_query)
            self._attempted_queries.add(canonical)
            self._last_retrieval_query = canonical
            self._last_retrieval_source_timestamp = event.source_timestamp_s
            self._pending_event_ids.clear()
        return result

    def _decide(self, event: TranscriptEvent, *, is_final: bool) -> StreamingDecision:
        query = self.current_text
        profile = _profile(query)
        if profile.formatting_only:
            if self.previous_answer_context is None:
                return self._make_decision(
                    event,
                    decision="SKIP",
                    reason_code="formatting_needs_previous_answer_context",
                    proposed_query=None,
                    needs_previous_answer_context=True,
                    previous_answer_context_used=False,
                )
            return self._make_decision(
                event,
                decision="SKIP",
                reason_code="formatting_with_previous_answer_context",
                proposed_query=None,
                needs_previous_answer_context=False,
                previous_answer_context_used=True,
            )
        if profile.greeting_only:
            return self._make_decision(
                event,
                decision="SKIP",
                reason_code="greeting_no_retrieval",
                proposed_query=None,
                needs_previous_answer_context=False,
                previous_answer_context_used=False,
            )
        if not is_final and profile.incomplete:
            return self._make_decision(
                event,
                decision="WAIT",
                reason_code="incomplete_phrase",
                proposed_query=None,
                needs_previous_answer_context=False,
                previous_answer_context_used=False,
            )
        if len(query) < self.config.min_query_chars:
            return self._make_decision(
                event,
                decision="SKIP" if is_final else "WAIT",
                reason_code="final_query_too_short" if is_final else "partial_query_too_short",
                proposed_query=None,
                needs_previous_answer_context=False,
                previous_answer_context_used=False,
            )
        if len(profile.topic_terms) < self.config.min_topic_terms and not profile.constraint:
            return self._make_decision(
                event,
                decision="SKIP" if is_final else "WAIT",
                reason_code="final_no_corpus_signal" if is_final else "no_corpus_signal",
                proposed_query=None,
                needs_previous_answer_context=False,
                previous_answer_context_used=False,
            )
        meaningful_request = (
            profile.question_signal
            or profile.request_signal
            or profile.correction
            or profile.constraint
            or len(profile.topic_terms) >= max(2, self.config.min_topic_terms + 1)
        )
        if not meaningful_request:
            return self._make_decision(
                event,
                decision="SKIP" if is_final else "WAIT",
                reason_code=(
                    "final_no_action_or_question_signal"
                    if is_final
                    else "unstable_fragment"
                ),
                proposed_query=None,
                needs_previous_answer_context=False,
                previous_answer_context_used=False,
            )
        canonical = _canonical_query(query)
        if self.config.duplicate_query_suppression and canonical in self._attempted_queries:
            return self._make_decision(
                event,
                decision="SKIP",
                reason_code="duplicate_query_suppressed",
                proposed_query=None,
                needs_previous_answer_context=False,
                previous_answer_context_used=False,
            )
        if (
            not self.config.duplicate_query_suppression
            and self._last_retrieval_query is not None
            and canonical == self._last_retrieval_query
        ):
            # The disabled-policy comparison must actually exercise the
            # duplicate path.  Keeping this branch explicit makes the
            # experiment meaningful: an exact repeated query is allowed to
            # allocate another search, while the default policy suppresses it.
            return self._make_decision(
                event,
                decision="RETRIEVE",
                reason_code="duplicate_query_allowed",
                proposed_query=query,
                needs_previous_answer_context=False,
                previous_answer_context_used=False,
            )
        if self._last_retrieval_query is None:
            return self._make_decision(
                event,
                decision="RETRIEVE",
                reason_code="final_complete_query" if is_final else "meaningful_partial_query",
                proposed_query=query,
                needs_previous_answer_context=False,
                previous_answer_context_used=False,
            )
        prior_tokens = set(tokenize(self._last_retrieval_query))
        current_tokens = set(tokenize(query))
        new_terms = current_tokens - prior_tokens
        constraint_changed = _constraint_signature(query) != _constraint_signature(
            self._last_retrieval_query
        )
        if profile.correction and self.config.retrieve_on_correction:
            reason_code = "explicit_correction"
        elif constraint_changed and self.config.retrieve_on_constraint_change:
            reason_code = "constraint_change"
        elif len(new_terms) >= self.config.min_new_content_terms:
            reason_code = "new_topic_information"
        else:
            reason_code = "no_meaningful_new_information"
        if reason_code in {"explicit_correction", "constraint_change"}:
            return self._make_decision(
                event,
                decision="RETRIEVE",
                reason_code=reason_code,
                proposed_query=query,
                needs_previous_answer_context=False,
                previous_answer_context_used=False,
            )
        if reason_code == "new_topic_information":
            source_gap = (
                None
                if self._last_retrieval_source_timestamp is None
                else event.source_timestamp_s - self._last_retrieval_source_timestamp
            )
            if not is_final and (source_gap is None or source_gap < self.config.debounce_source_s):
                return self._make_decision(
                    event,
                    decision="WAIT",
                    reason_code="debounce_coalescing",
                    proposed_query=None,
                    needs_previous_answer_context=False,
                    previous_answer_context_used=False,
                )
            return self._make_decision(
                event,
                decision="RETRIEVE",
                reason_code="final_new_topic_information" if is_final else reason_code,
                proposed_query=query,
                needs_previous_answer_context=False,
                previous_answer_context_used=False,
            )
        if is_final and self.config.final_decision_bypasses_debounce:
            return self._make_decision(
                event,
                decision="SKIP",
                reason_code="final_no_new_information",
                proposed_query=None,
                needs_previous_answer_context=False,
                previous_answer_context_used=False,
            )
        return self._make_decision(
            event,
            decision="WAIT",
            reason_code=reason_code,
            proposed_query=None,
            needs_previous_answer_context=False,
            previous_answer_context_used=False,
        )

    def begin_retrieval(
        self,
        decision: StreamingDecision,
        *,
        query_override: str | None = None,
    ) -> StreamingRetrievalRequest:
        """Turn a RETRIEVE decision into a revision-bound request."""

        if decision.session_id != self.session_id or decision.utterance_id != self.utterance_id:
            raise StreamingControllerError("retrieval decision belongs to another session/utterance")
        if decision.decision != "RETRIEVE" and query_override is None:
            raise StreamingControllerError("only RETRIEVE decisions can start retrieval")
        if decision.transcript_revision > self._revision:
            raise StreamingControllerError("retrieval decision refers to a future transcript revision")
        query = query_override or decision.proposed_query
        if query is None or not query.strip():
            raise StreamingControllerError("retrieval requests require a non-empty query")
        self._request_counter += 1
        self._retrieval_revision_counter += 1
        self._latest_retrieval_revision = self._retrieval_revision_counter
        # A new query revision invalidates previously current evidence
        # immediately. The scheduler may later accept the new result; stale
        # old passages must never remain available to final generation.
        self._current_evidence_hits = []
        self._current_evidence_query = None
        return StreamingRetrievalRequest(
            request_id=(
                f"{self.session_id}-{self.utterance_id}-retrieval-"
                f"{self._request_counter:04d}-{uuid.uuid4().hex[:12]}"
            ),
            decision_id=decision.decision_id,
            session_id=self.session_id,
            utterance_id=self.utterance_id,
            transcript_revision=decision.transcript_revision,
            retrieval_revision=self._retrieval_revision_counter,
            query=query,
            source_timestamp_s=decision.source_timestamp_s,
            monotonic_started_s=time.monotonic(),
        )

    def complete_retrieval(
        self,
        request: StreamingRetrievalRequest,
        hits: Sequence[RetrievalHit],
        *,
        monotonic_completed_s: float | None = None,
    ) -> StreamingRetrievalResult:
        """Accept current evidence or mark it stale after a revised query.

        The conservative stale policy is exact canonical-query equality plus
        the latest session-scoped retrieval revision. Raw transcript revision
        differences alone do not invalidate a request: punctuation, spacing,
        or another non-material event may arrive while its search is running.
        """

        if request.request_id in self._completed_request_ids:
            raise StreamingControllerError(f"retrieval request {request.request_id!r} completed twice")
        if request.session_id != self.session_id or request.utterance_id != self.utterance_id:
            raise StreamingControllerError("retrieval request belongs to another session/utterance")
        for hit in hits:
            if self._known_chunk_locations:
                expected_location = self._known_chunk_locations.get(hit.chunk_id)
                if expected_location is None:
                    raise StreamingControllerError(
                        f"retrieval hit {hit.chunk_id!r} is not present in the configured corpus"
                    )
                if expected_location != hit.source_location:
                    raise StreamingControllerError(
                        f"retrieval hit {hit.chunk_id!r} has an inconsistent source location"
                    )
        completed = monotonic_completed_s if monotonic_completed_s is not None else time.monotonic()
        stale = (
            request.retrieval_revision != self._latest_retrieval_revision
            or _canonical_query(self.current_text) != _canonical_query(request.query)
        )
        accepted = not stale
        self._completed_request_ids.add(request.request_id)
        if accepted:
            self._current_evidence_hits = list(hits)
            self._current_evidence_query = _canonical_query(request.query)
        return StreamingRetrievalResult(
            request_id=request.request_id,
            decision_id=request.decision_id,
            session_id=request.session_id,
            utterance_id=request.utterance_id,
            transcript_revision=request.transcript_revision,
            retrieval_revision=request.retrieval_revision,
            query=request.query,
            hits=list(hits),
            accepted=accepted,
            stale=stale,
            source_timestamp_s=request.source_timestamp_s,
            monotonic_started_s=request.monotonic_started_s,
            monotonic_completed_s=completed,
            duration_ms=max(0.0, (completed - request.monotonic_started_s) * 1000),
        )


def _retrieval_usage(query: str) -> Usage:
    token_count = len(tokenize(query))
    return Usage(input_tokens=token_count, output_tokens=0, total_tokens=token_count, estimated=True)


async def replay_streaming_transcript(
    events: Sequence[TranscriptEvent],
    *,
    corpus: CorpusIndex,
    top_k: int = 5,
    backend: str = "dense",
    retriever: Retriever | None = None,
    run_id: str | None = None,
    model_identity: str = "flowcontext.streaming-controller.v1",
    embedding_cache_dir: Path | None = None,
    embedding_local_files_only: bool = False,
    execution_mode: Literal["realtime", "accelerated"] = "accelerated",
    controller_config: StreamingDecisionConfig | None = None,
    scheduler_config: StreamingSchedulerConfig | None = None,
    previous_answer_context: PreviousAnswerContext | None = None,
    generation_provider: GenerationProvider | None = None,
    decomposition_provider: DecompositionProvider | None = None,
    multi_intent: bool = False,
    retrieval_mode: RetrievalMode | None = None,
    context_budget_tokens: int = 1200,
    multi_intent_max_workers: int | None = None,
    multi_intent_rrf_k: int = 60,
    multi_intent_reranking_enabled: bool = False,
    is_generation_superseded: Callable[[], bool] | None = None,
    semantic_verifier: Any = None,
) -> StreamingReplayResult:
    """Replay events, schedule early retrieval, and answer only after finality.

    Transcript processing remains in the asyncio event loop while the existing
    synchronous retriever runs in the bounded scheduler executor. No answer is
    generated for a partial event. At finalisation, evidence is reused only for
    an exact canonical final-query match and the current retrieval revision.
    """

    # Local import keeps the controller module usable by direct callers while
    # the scheduler reuses its private canonical-query policy helper.
    from .scheduler import AsyncRetrievalScheduler, RetrievalSchedulerError

    if execution_mode not in {"realtime", "accelerated"}:
        raise StreamingReplayError(f"unsupported replay execution mode: {execution_mode!r}")
    records = normalize_transcript(events)
    event_list = [record.event for record in records]
    first_event = event_list[0]
    effective_run_id = run_id or f"stream-{first_event.session_id}-{first_event.utterance_id}"
    config = controller_config or StreamingDecisionConfig()
    construction_backend = backend
    if multi_intent and retrieval_mode in {"dense", "lexical", "mock"}:
        construction_backend = retrieval_mode
    elif multi_intent and retrieval_mode == "hybrid":
        construction_backend = corpus.manifest.embedding.backend
    selected_retriever = retriever or make_retriever(
        corpus,
        backend=construction_backend,
        top_k=top_k,
        cache_dir=embedding_cache_dir,
        local_files_only=embedding_local_files_only,
    )
    if multi_intent and not isinstance(selected_retriever, MultiIntentRetriever):
        decomposer = (
            StructuredMultiIntentDecomposer(decomposition_provider)
            if decomposition_provider is not None
            else None
        )
        if retrieval_mode is None:
            selected_retriever = MultiIntentRetriever(
                selected_retriever,
                top_k=top_k,
                decomposer=decomposer,
                context_budget_tokens=context_budget_tokens,
                max_workers=multi_intent_max_workers,
                rrf_k=multi_intent_rrf_k,
                reranking_enabled=multi_intent_reranking_enabled,
            )
        else:
            selected_retriever = make_multi_intent_retriever(
                corpus,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                base_retriever=selected_retriever,
                cache_dir=embedding_cache_dir,
                local_files_only=embedding_local_files_only,
                max_workers=multi_intent_max_workers,
                rrf_k=multi_intent_rrf_k,
                context_budget_tokens=context_budget_tokens,
                decomposer=decomposer,
                reranking_enabled=multi_intent_reranking_enabled,
            )
    selected_generation_provider = generation_provider or MockGenerationProvider()
    controller = StreamingDecisionController(
        session_id=first_event.session_id,
        utterance_id=first_event.utterance_id,
        config=config,
        previous_answer_context=previous_answer_context,
        known_chunk_locations={chunk.chunk_id: chunk.source_location for chunk in corpus.chunks},
    )
    selected_retrieval_identity = _retrieval_model_identity(corpus)
    selected_generation_identity = generation_model_identity(selected_generation_provider.config)
    traces = TraceCollector(effective_run_id, first_event.session_id, model_identity)
    scheduler = AsyncRetrievalScheduler(
        controller=controller,
        retriever=selected_retriever,
        traces=traces,
        retrieval_model_identity=selected_retrieval_identity,
        config=scheduler_config,
    )
    replay_started = time.perf_counter()
    traces.add(
        "streaming_replay_started",
        source_timestamp_s=first_event.source_timestamp_s,
        attributes={
            "controller_mode": "explainable_streaming_controller",
            "mode": "streaming",
            "execution_mode": execution_mode,
            "selected_retrieval_backend": selected_retriever.backend,
            "selected_retrieval_model": selected_retrieval_identity,
            "retrieval_top_k": top_k,
            "multi_intent": multi_intent,
            "retrieval_scheduler": "async_bounded_executor.v1",
            "retrieval_scheduler_config": scheduler.config.model_dump(mode="json"),
            "generation": {
                "backend": selected_generation_provider.config.backend,
                "model": selected_generation_identity,
                "early_generation": False,
            },
            "controller_config": config.model_dump(mode="json"),
        },
    )
    decisions: list[StreamingDecision] = []
    retrieval_results: list[StreamingRetrievalResult] = []
    final_event: TranscriptEvent | None = None
    final_decision: StreamingDecision | None = None
    previous_source_timestamp: float | None = None
    scheduling_error: TraceError | None = None
    finalization_started: float | None = None
    final_event_delivery_time_s = 0.0
    controller_overhead_ms = 0.0

    try:
        for record in records:
            event = record.event
            await _wait_for_source_timing(previous_source_timestamp, event.source_timestamp_s, execution_mode)
            previous_source_timestamp = event.source_timestamp_s
            delivery_time_s = traces.elapsed_s()
            traces.add(
                "transcript_event_received",
                source_timestamp_s=event.source_timestamp_s,
                actual_delivery_time_s=delivery_time_s,
                attributes={
                    "controller_mode": "explainable_streaming_controller",
                    "event_id": event.event_id,
                    "utterance_id": event.utterance_id,
                    "sequence_number": event.sequence_number,
                    "is_final": event.is_final,
                    "text_mode": event.text_mode,
                    "duplicate": record.is_duplicate,
                    "duplicate_reason": record.duplicate_reason,
                    "actual_delivery_time_s": delivery_time_s,
                },
            )
            if event.is_final and final_event is None:
                final_event_delivery_time_s = delivery_time_s
                finalization_started = time.perf_counter()
                traces.add(
                    "final_event_delivered",
                    source_timestamp_s=event.source_timestamp_s,
                    actual_delivery_time_s=delivery_time_s,
                    model_identity=model_identity,
                    attributes={
                        "event_id": event.event_id,
                        "actual_delivery_time_s": delivery_time_s,
                        "execution_mode": execution_mode,
                    },
                )
            decision_started = time.perf_counter()
            decision = controller.process_event(
                event,
                is_duplicate=record.is_duplicate,
                duplicate_reason=record.duplicate_reason,
            )
            decision_duration_ms = (time.perf_counter() - decision_started) * 1000
            controller_overhead_ms += decision_duration_ms
            decisions.append(decision)
            traces.add(
                "streaming_decision",
                source_timestamp_s=decision.source_timestamp_s,
                model_identity=model_identity,
                attributes={
                    "decision_id": decision.decision_id,
                    "transcript_revision": decision.transcript_revision,
                    "decision": decision.decision,
                    "reason_code": decision.reason_code,
                    "proposed_query": decision.proposed_query,
                    "is_final_event": decision.is_final_event,
                    "needs_previous_answer_context": decision.needs_previous_answer_context,
                    "previous_answer_context_used": decision.previous_answer_context_used,
                    "coalesced_event_ids": decision.coalesced_event_ids,
                    "controller_decision_time_s": traces.elapsed_s(),
                    "controller_decision_duration_ms": decision_duration_ms,
                    "actual_delivery_time_s": delivery_time_s,
                },
                )
            if event.is_final and final_event is None:
                final_event = event
                final_decision = decision
                traces.add(
                    "utterance_finalized",
                    source_timestamp_s=event.source_timestamp_s,
                    model_identity=model_identity,
                    attributes={
                        "final_event_id": event.event_id,
                        "transcript_revision": decision.transcript_revision,
                        "final_query": controller.current_text,
                        "decision": decision.decision,
                        "reason_code": decision.reason_code,
                    },
                )
            if decision.decision != "RETRIEVE" or decision.proposed_query is None:
                continue
            try:
                scheduler.schedule(decision)
                # Give the scheduler task one turn so the early-retrieval
                # metric is based on an observed worker start, never merely a
                # decision or a queued request.
                await asyncio.sleep(0)
            except RetrievalSchedulerError as exc:
                scheduling_error = TraceError(error_type=type(exc).__name__, message=str(exc))
                # The scheduler has already emitted its structured scheduling
                # failure trace; keep processing until finalisation so the run
                # can produce an explicit abstention result.
    except Exception as exc:
        # Normalisation happens before scheduler creation, but controller or
        # scheduling errors can still occur at runtime. Do not leak executor
        # workers when an event-processing failure aborts the session.
        await scheduler.close()
        controller.close()
        traces.add_error(
            "streaming_replay_failed",
            exc,
            source_timestamp_s=previous_source_timestamp,
        )
        raise

    if final_event is None or final_decision is None:
        await scheduler.close()
        controller.close()
        error = StreamingReplayError("streaming replay did not reach a final transcript event")
        traces.add_error("streaming_replay_failed", error)
        raise error

    final_query = controller.current_text
    formatting_only_final = final_decision.reason_code in {
        "formatting_needs_previous_answer_context",
        "formatting_with_previous_answer_context",
    }
    no_retrieval_final = final_decision.reason_code == "greeting_no_retrieval"
    final_intent_plan = (
        selected_retriever.decompose_for_query(
            final_query,
            parent_revision=controller.transcript_revision,
        )
        if (
            multi_intent
            and isinstance(selected_retriever, MultiIntentRetriever)
            and not formatting_only_final
            and not no_retrieval_final
        )
        else None
    )
    final_intent_queries = (
        [intent.query for intent in final_intent_plan.intents]
        if final_intent_plan is not None
        else None
    )
    if final_intent_plan is not None:
        traces.add(
            "streaming_final_decomposition_fallback"
            if final_intent_plan.status == "fallback"
            else "streaming_final_decomposition_selected",
            source_timestamp_s=final_event.source_timestamp_s,
            duration_ms=final_intent_plan.latency_ms,
            usage=final_intent_plan.usage,
            attributes={
                "decomposition_status": final_intent_plan.status,
                "decomposition_method": final_intent_plan.decomposition_method,
                "decomposition_provider": final_intent_plan.provider,
                "decomposition_model": final_intent_plan.provider_model,
                "transcript_revision": final_intent_plan.transcript_revision,
                "intent_count": len(final_intent_plan.intents),
                "attempts": final_intent_plan.attempts,
                "repair_attempts": final_intent_plan.repair_attempts,
                "preserved_intent_ids": final_intent_plan.preserved_intent_ids,
                "superseded_intent_ids": final_intent_plan.superseded_intent_ids,
                "failure_reason": final_intent_plan.failure_reason,
                "finalisation_boundary": True,
            },
        )
    final_retrieval_result = None
    if not formatting_only_final and not no_retrieval_final:
        final_retrieval_result = await scheduler.finish(
            final_query=final_query,
            final_decision=final_decision,
        )
    retrieval_results = scheduler.results
    final_evidence_available = (
        final_retrieval_result is not None
        and final_retrieval_result.status == "completed"
        and final_retrieval_result.accepted
        and not final_retrieval_result.stale
        and (
            selected_retriever.validate_evidence_for_query(
                final_query,
                final_retrieval_result.hits,
                parent_revision=controller.transcript_revision,
            ).get("lexical_relevance_proxy", False)
            if isinstance(selected_retriever, MultiIntentRetriever)
            else single_query_evidence_is_appropriate(
                final_query,
                final_retrieval_result.hits,
            )
        )
    )
    # Do not use controller state as a proxy for final validity: it may still
    # hold evidence from an earlier query when a final recovery was rejected
    # by the scheduler's resource limit. Only the final scheduler result can
    # authorize passages for generation.
    final_evidence_hits = (
        list(controller.current_evidence_hits) if final_evidence_available else []
    )
    unsupported_intent_queries = (
        unsupported_intent_queries_from_hits(
            final_intent_plan,
            final_retrieval_result.hits if final_retrieval_result is not None else [],
        )
        if final_intent_plan is not None
        else []
    )
    retrieval_error: TraceError | None = scheduling_error
    if not final_evidence_available and not final_decision.needs_previous_answer_context:
        failed_results = [
            result
            for result in retrieval_results
            if result.status in {"failed", "timed_out"} and result.error is not None
        ]
        if failed_results:
            retrieval_error = failed_results[-1].error
        elif retrieval_error is None and final_decision.decision == "RETRIEVE":
            completed_current_result = any(
                result.status == "completed"
                and result.accepted
                and not result.stale
                and _canonical_query(result.query) == _canonical_query(final_query)
                for result in retrieval_results
            )
            # A completed search with zero hits, or a completed partial
            # multi-intent result that fails the coverage gate, is an honest
            # no-evidence/uncertain answer—not a transport failure. Only add
            # an operational retrieval error when no current completed result
            # exists at all.
            if not completed_current_result:
                retrieval_error = scheduler.last_error or TraceError(
                    error_type="RetrievalUnavailable",
                    message="no current final-query retrieval result was available",
                )

    # Answer generation is deliberately entered only after the final event has
    # been processed and final evidence selection is complete. Greeting and
    # formatting-only turns are handled from policy/session context and never
    # forced through corpus retrieval or a model call.
    generation_started = time.perf_counter()
    generation_outcome = None
    generation_error: TraceError | None = None
    try:
        if final_decision.reason_code == "greeting_no_retrieval":
            generation_outcome = GenerationOutcome(
                answer=Answer(
                    answer_text="Hello. I can help answer questions from the configured corpus.",
                    factual_claims=[],
                    uncertainty="Greeting turn; no corpus retrieval was requested.",
                    answer_version=1,
                ),
                status="skipped",
                usage=Usage(),
                cost="unavailable",
                attempts=0,
                repair_attempts=0,
            )
        elif final_decision.reason_code == "formatting_with_previous_answer_context" and previous_answer_context:
            generation_outcome = GenerationOutcome(
                answer=Answer(
                    answer_text=previous_answer_context.answer_text,
                    factual_claims=[],
                    uncertainty=(
                        "Formatting-only turn used the existing session answer; "
                        "no corpus retrieval or selective rewrite was performed."
                    ),
                    answer_version=previous_answer_context.answer_version,
                ),
                status="skipped",
                usage=Usage(),
                cost="unavailable",
                attempts=0,
                repair_attempts=0,
            )
        elif final_decision.needs_previous_answer_context:
            generation_outcome = GenerationOutcome(
                answer=Answer(
                    answer_text="Please provide the original question or a completed answer to format.",
                    factual_claims=[],
                    uncertainty=(
                        "Previous answer context is required; no corpus search or invented content was used."
                    ),
                    answer_version=1,
                ),
                status="abstained",
                usage=Usage(),
                cost="unavailable",
                attempts=0,
                repair_attempts=0,
            )
        elif final_evidence_hits:
            traces.add(
                "streaming_generation_started",
                source_timestamp_s=final_event.source_timestamp_s,
                model_identity=selected_generation_identity,
                attributes={
                    "final_only": True,
                    "early_generation": False,
                    "evidence_chunk_ids": [hit.chunk_id for hit in final_evidence_hits],
                    "provider_backend": selected_generation_provider.config.backend,
                },
            )
            generation_outcome = await generate_grounded_answer(
                final_query,
                final_evidence_hits,
                corpus,
                selected_generation_provider,
                intent_queries=final_intent_queries,
                unsupported_intent_queries=unsupported_intent_queries,
                decomposition=final_intent_plan,
                verifier=semantic_verifier,
                is_superseded=is_generation_superseded,
                transcript_revision=final_event.sequence_number if final_event else None,
            )
        else:
            traces.add(
                "streaming_generation_skipped",
                source_timestamp_s=final_event.source_timestamp_s,
                model_identity=selected_generation_identity,
                attributes={
                    "reason": (
                        "needs_previous_answer_context"
                        if final_decision.needs_previous_answer_context
                        else "no_current_evidence"
                    ),
                    "final_only": True,
                    "early_generation": False,
                },
            )
            generation_outcome = await generate_grounded_answer(
                final_query,
                final_evidence_hits,
                corpus,
                selected_generation_provider,
                intent_queries=final_intent_queries,
                unsupported_intent_queries=unsupported_intent_queries,
                decomposition=final_intent_plan,
                verifier=semantic_verifier,
                is_superseded=is_generation_superseded,
                transcript_revision=final_event.sequence_number if final_event else None,
            )
    except Exception as exc:
        generation_error = TraceError(error_type=type(exc).__name__, message=str(exc))
        generation_outcome = None

    generation_latency_ms = (time.perf_counter() - generation_started) * 1000
    complete_answer_latency_ms = (
        (time.perf_counter() - finalization_started) * 1000
        if finalization_started is not None
        else generation_latency_ms
    )
    if generation_outcome is None:
        answer = Answer(
            answer_text="I cannot safely answer from the configured corpus.",
            factual_claims=[],
            uncertainty=(
                "Answer generation failed before a validated corpus-grounded answer was produced; "
                "no outside knowledge was used."
            ),
            answer_version=1,
        )
        generation_status = "failed"
        generation_usage = Usage()
        generation_cost: float | Literal["unavailable"] = "unavailable"
        generation_attempts = 0
        generation_repair_attempts = 0
    else:
        answer = generation_outcome.answer
        generation_status = generation_outcome.status
        generation_usage = generation_outcome.usage
        generation_cost = generation_outcome.cost
        generation_attempts = generation_outcome.attempts
        generation_repair_attempts = generation_outcome.repair_attempts
        if generation_outcome.error_type and generation_outcome.status != "stale_rejected":
            generation_error = TraceError(
                error_type=generation_outcome.error_type,
                message=generation_outcome.error_message or generation_outcome.error_type,
            )
    if generation_error is not None:
        traces.add(
            "streaming_generation_failed",
            source_timestamp_s=final_event.source_timestamp_s,
            duration_ms=generation_latency_ms,
            model_identity=selected_generation_identity,
            usage=generation_usage,
            cost=generation_cost,
            error=generation_error,
            attributes={
                "final_only": True,
                "generation_status": generation_status,
                "attempts": generation_attempts,
                "repair_attempts": generation_repair_attempts,
            },
        )
    else:
        generation_terminal_event = (
            "streaming_generation_stale_rejected"
            if generation_status == "stale_rejected"
            else "streaming_generation_skipped"
            if generation_status == "skipped" or (generation_attempts == 0 and not final_evidence_hits)
            else "streaming_generation_completed"
        )
        traces.add(
            generation_terminal_event,
            source_timestamp_s=final_event.source_timestamp_s,
            duration_ms=generation_latency_ms,
            model_identity=selected_generation_identity,
            usage=generation_usage,
            cost=generation_cost,
            attributes={
                "final_only": True,
                "early_generation": False,
                "generation_status": generation_status,
                "attempts": generation_attempts,
                "repair_attempts": generation_repair_attempts,
                "generation_usage": (
                    generation_outcome.generation_usage.model_dump(mode="json")
                    if generation_outcome and generation_outcome.generation_usage
                    else None
                ),
                "repair_usage": (
                    generation_outcome.repair_usage.model_dump(mode="json")
                    if generation_outcome and generation_outcome.repair_usage
                    else None
                ),
                "verification_usage": (
                    generation_outcome.verification_usage.model_dump(mode="json")
                    if generation_outcome and generation_outcome.verification_usage
                    else None
                ),
                "citation_ids": [
                    chunk_id
                    for claim in answer.factual_claims
                    for chunk_id in claim.supporting_chunk_ids
                ],
                "answer_version": answer.answer_version,
                "streamed": False,
                "first_content_observed": False,
                "latency_metric": "complete_answer_latency_ms",
                "answer_latency_metric": "answer_latency_from_final_event_delivery_ms",
            },
        )
    traces.add(
        "streaming_answer_completed" if generation_error is None else "streaming_answer_failed",
        source_timestamp_s=final_event.source_timestamp_s,
        duration_ms=complete_answer_latency_ms,
        model_identity=selected_generation_identity,
        usage=generation_usage,
        cost=generation_cost,
        error=generation_error,
        attributes={
            "answer_version": answer.answer_version,
            "citation_ids": [
                chunk_id
                for claim in answer.factual_claims
                for chunk_id in claim.supporting_chunk_ids
            ],
            "citation_id_validation": "performed_by_generation_pipeline",
            "semantic_support_evaluated": bool(generation_outcome and generation_outcome.verification_report),
            "streamed": False,
            "latency_metric": "complete_answer_latency_ms",
            "answer_latency_metric": "answer_latency_from_final_event_delivery_ms",
            "generation_status": generation_status,
        },
    )

    await scheduler.close()
    controller.close()
    retrieval_results = scheduler.results
    execution_duration_ms = (time.perf_counter() - replay_started) * 1000
    stale_count = scheduler.stale_result_count
    early_results = [
        result
        for result in retrieval_results
        if (
            not result.is_final_event
            and result.actual_started_time_s is not None
            and result.actual_started_time_s < final_event_delivery_time_s
        )
    ]
    early_count = len(early_results)
    final_count = sum(
        result.is_final_event and result.actual_started_time_s is not None
        for result in retrieval_results
    )
    suppressed_count = sum(
        decision.reason_code == "duplicate_query_suppressed" for decision in decisions
    )
    final_query_key = _canonical_query(final_query)
    valid_evidence_results = [
        result
        for result in retrieval_results
        if (
            result.status == "completed"
            and result.accepted
            and not result.stale
            and bool(result.hits)
            and _canonical_query(result.query) == final_query_key
            and result.evidence_ready_time_s is not None
            and result.evidence_ready_time_s <= final_event_delivery_time_s
        )
    ]
    valid_evidence_ready_before_finalization = bool(valid_evidence_results)
    early_evidence_reused = any(
        trace.event_type == "streaming_evidence_reused"
        and trace.attributes.get("early_evidence_reused") is True
        for trace in traces.events
    )
    evidence_ready_time_s = (
        min(result.evidence_ready_time_s for result in valid_evidence_results)
        if valid_evidence_results
        else (
            final_retrieval_result.evidence_ready_time_s
            if final_retrieval_result is not None
            and final_retrieval_result.evidence_ready_time_s is not None
            else None
        )
    )
    generation_started_time_s = next(
        (
            trace.monotonic_execution_time_s
            for trace in traces.events
            if trace.event_type == "streaming_generation_started"
        ),
        None,
    )
    generation_completed_time_s = next(
        (
            trace.monotonic_execution_time_s
            for trace in traces.events
            if trace.event_type == "streaming_generation_completed"
        ),
        None,
    )
    trace_errors = [trace.error for trace in traces.events if trace.error is not None]
    retrieval_error_count = scheduler.retrieval_error_count + int(scheduling_error is not None)
    retrieval_call_count = scheduler.retrieval_call_count
    run_status: Literal["completed", "needs_context", "abstained", "failed"]
    if final_decision.needs_previous_answer_context:
        run_status = "needs_context"
    elif final_decision.reason_code == "formatting_with_previous_answer_context":
        # The previous answer is used only for controller classification in
        # Phase 2; no selective reformatting is attempted. The turn is a
        # successful no-retrieval classification, with an explicit skipped
        # generation stage.
        run_status = "completed"
    elif final_decision.reason_code == "greeting_no_retrieval":
        run_status = "completed"
    elif retrieval_error is not None and not final_evidence_available:
        run_status = "failed"
    elif generation_error is not None:
        run_status = "failed"
    elif generation_status in {"abstained", "skipped", "stale_rejected"}:
        run_status = "abstained"
    else:
        run_status = "completed"
    terminal_error = generation_error or retrieval_error
    traces.add(
        "streaming_replay_completed" if run_status != "failed" else "streaming_replay_failed",
        source_timestamp_s=final_event.source_timestamp_s,
        duration_ms=execution_duration_ms,
        model_identity=model_identity,
        error=terminal_error,
        attributes={
            "mode": "streaming",
            "run_status": run_status,
            "final_decision": final_decision.decision,
            "final_reason_code": final_decision.reason_code,
            "retrieval_attempt_count": scheduler.retrieval_attempt_count,
            "retrieval_call_count": retrieval_call_count,
            "scheduled_request_count": scheduler.scheduled_request_count,
            "early_retrieval_count": early_count,
            "final_retrieval_count": final_count,
            "suppressed_duplicate_query_count": suppressed_count,
            "stale_retrieval_count": stale_count,
            "stale_result_discard_count": stale_count,
            "unnecessary_retrieval_count": scheduler.superseded_request_count,
            "cancelled_request_count": scheduler.cancelled_request_count,
            "timed_out_request_count": scheduler.timed_out_request_count,
            "retrieval_started_early": early_count > 0,
            "valid_evidence_ready_before_finalization": valid_evidence_ready_before_finalization,
            "early_evidence_reused": early_evidence_reused,
            "multi_intent": multi_intent,
            "intent_metadata": intent_metadata_from_hits(final_evidence_hits),
            "answer_latency_from_final_event_delivery_ms": complete_answer_latency_ms,
            "full_interaction_duration_ms": execution_duration_ms,
            "controller_overhead_ms": controller_overhead_ms,
            "generation_status": generation_status,
            "generation_model": selected_generation_identity,
            "execution_mode": execution_mode,
            "timing_note": "monotonic execution duration; accelerated replay is not real-time evidence",
        },
    )
    traces.add(
        "streaming_timing_summary",
        source_timestamp_s=final_event.source_timestamp_s,
        model_identity=model_identity,
        attributes={
            "mode": "streaming",
            "execution_mode": execution_mode,
            "final_event_delivery_time_s": final_event_delivery_time_s,
            "evidence_ready_time_s": evidence_ready_time_s,
            "retrieval_started_early": early_count > 0,
            "valid_evidence_ready_before_finalization": valid_evidence_ready_before_finalization,
            "early_evidence_reused": early_evidence_reused,
            "multi_intent": multi_intent,
            "reuse_validation": reuse_validation(
                final_query,
                final_evidence_hits,
                plan=final_intent_plan,
            ),
            "unsupported_intent_count": len(unsupported_intent_queries),
            "answer_latency_from_final_event_delivery_ms": complete_answer_latency_ms,
            "full_interaction_duration_ms": execution_duration_ms,
            "controller_overhead_ms": controller_overhead_ms,
            "retrieval_call_count": retrieval_call_count,
            "retrieval_error_count": retrieval_error_count,
            "stale_result_discard_count": stale_count,
            "accelerated_latency_comparable": execution_mode == "realtime",
            "retrieval_details": (
                final_retrieval_result.retrieval_details
                if final_retrieval_result is not None
                else None
            ),
        },
    )
    final_retrieval_details = (
        final_retrieval_result.retrieval_details
        if final_retrieval_result is not None
        else next(
            (
                result.retrieval_details
                for result in reversed(retrieval_results)
                if result.retrieval_details is not None
            ),
            None,
        )
    )
    return StreamingReplayResult(
        run_id=effective_run_id,
        session_id=first_event.session_id,
        utterance_id=first_event.utterance_id,
        controller_mode="explainable_streaming_controller",
        corpus_source_kind=corpus.source_kind,
        retrieval_backend=selected_retriever.backend,
        execution_mode=execution_mode,
        run_status=run_status,
        final_event_id=final_event.event_id,
        final_source_timestamp_s=final_event.source_timestamp_s,
        transcript_event_count=len(event_list),
        duplicate_event_count=sum(record.is_duplicate for record in records),
        source_duration_s=max(
            0.0,
            event_list[-1].source_timestamp_s - event_list[0].source_timestamp_s,
        ),
        execution_duration_ms=execution_duration_ms,
        final_query=controller.current_text,
        final_decision=final_decision,
        decisions=decisions,
        retrieval_results=retrieval_results,
        current_evidence_hits=controller.current_evidence_hits,
        decomposition=(
            final_intent_plan.model_dump(mode="json")
            if final_intent_plan is not None
            else None
        ),
        retrieval_details=final_retrieval_details,
        decomposition_history=(
            [
                execution.result.model_dump(mode="json")
                for execution in selected_retriever.decomposer.history
            ]
            if isinstance(selected_retriever, MultiIntentRetriever)
            and selected_retriever.decomposer is not None
            else [
                result.decomposition
                for result in retrieval_results
                if result.decomposition is not None
            ]
        ),
        retrieval_attempt_count=scheduler.retrieval_attempt_count,
        early_retrieval_count=early_count,
        final_retrieval_count=final_count,
        suppressed_duplicate_query_count=suppressed_count,
        stale_retrieval_count=stale_count,
        # This is deliberately operational: only scheduler-superseded work is
        # counted. Semantic answer usefulness needs a separate judged review.
        unnecessary_retrieval_count=scheduler.superseded_request_count,
        scheduled_request_count=scheduler.scheduled_request_count,
        cancelled_request_count=scheduler.cancelled_request_count,
        superseded_request_count=scheduler.superseded_request_count,
        timed_out_request_count=scheduler.timed_out_request_count,
        generation_backend=selected_generation_provider.config.backend,
        generation_model=selected_generation_identity,
        generation_status=(
            "abstained" if generation_status == "stale_rejected" else generation_status
        ),
        generation_usage=generation_usage,
        generation_cost=generation_cost,
        generation_attempts=generation_attempts,
        generation_repair_attempts=generation_repair_attempts,
        complete_answer_latency_ms=complete_answer_latency_ms,
        answer=answer,
        traces=traces.events,
        mode="streaming",
        final_event_delivery_time_s=final_event_delivery_time_s,
        evidence_ready_time_s=evidence_ready_time_s,
        generation_started_time_s=generation_started_time_s,
        generation_completed_time_s=generation_completed_time_s,
        generation_first_content_time_s=None,
        retrieval_started_early=early_count > 0,
        valid_evidence_ready_before_finalization=valid_evidence_ready_before_finalization,
        early_evidence_reused=early_evidence_reused,
        answer_latency_from_final_event_delivery_ms=complete_answer_latency_ms,
        full_interaction_duration_ms=execution_duration_ms,
        controller_overhead_ms=controller_overhead_ms,
        retrieval_call_count=retrieval_call_count,
        retrieval_error_count=retrieval_error_count,
        stale_result_discard_count=stale_count,
        retrieval_usage=scheduler.retrieval_usage,
        errors=trace_errors,
        corpus_id=corpus.corpus_id,
        index_id=corpus.manifest.index_id,
        retrieval_top_k=top_k,
    )


def write_streaming_replay(path: Path, result: StreamingReplayResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_streaming_replay(path: Path) -> StreamingReplayResult:
    if not path.is_file():
        raise StreamingReplayError(f"streaming replay output does not exist: {path}")
    try:
        return StreamingReplayResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, ValidationError) as exc:
        raise StreamingReplayError(f"invalid streaming replay output {path}: {exc}") from exc


def _streaming_environment() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "os_name": os.name,
        "executable": sys.executable,
    }


def _streaming_manifest_configuration(
    result: StreamingReplayResult,
    settings: Settings,
    *,
    corpus: CorpusIndex,
    transcript_path: Path,
    index_path: Path,
) -> dict[str, Any]:
    return {
        "transcript_path": str(transcript_path),
        "index_path": str(index_path),
        "execution_mode": result.execution_mode,
        "mode": "streaming",
        "comparison_metrics": {
            "retrieval_started_early": result.retrieval_started_early,
            "valid_evidence_ready_before_finalization": result.valid_evidence_ready_before_finalization,
            "early_evidence_reused": result.early_evidence_reused,
            "answer_latency_from_final_event_delivery_ms": result.answer_latency_from_final_event_delivery_ms,
            "full_interaction_duration_ms": result.full_interaction_duration_ms,
            "controller_overhead_ms": result.controller_overhead_ms,
            "retrieval_call_count": result.retrieval_call_count,
            "retrieval_error_count": result.retrieval_error_count,
            "stale_result_discard_count": result.stale_result_discard_count,
        },
        "index": {
            "manifest_version": corpus.manifest.manifest_version,
            "corpus_id": corpus.corpus_id,
            "index_id": corpus.manifest.index_id,
            "source_fingerprint": corpus.manifest.source_fingerprint,
            "build_fingerprint": corpus.manifest.build_fingerprint,
        },
        "controller": result.final_decision.schema_version,
        "controller_config": {
            "debounce_source_s": settings.streaming_debounce_source_s,
            "min_query_chars": settings.streaming_min_query_chars,
            "min_topic_terms": settings.streaming_min_topic_terms,
            "min_new_content_terms": settings.streaming_min_new_content_terms,
            "retrieve_on_correction": settings.streaming_retrieve_on_correction,
            "retrieve_on_constraint_change": settings.streaming_retrieve_on_constraint_change,
            "duplicate_query_suppression": settings.streaming_duplicate_query_suppression,
            "final_decision_bypasses_debounce": settings.streaming_final_bypasses_debounce,
        },
        "retrieval": {
            "backend": result.retrieval_backend,
            "top_k": result.retrieval_top_k,
            "phase3_mode": settings.multi_intent_retrieval_mode,
            "phase3_context_budget_tokens": settings.multi_intent_context_budget_tokens,
            "phase3_max_workers": settings.multi_intent_max_workers,
            "phase3_rrf_k": settings.multi_intent_rrf_k,
            "phase3_details": result.retrieval_details,
            "scheduler": {
                "implementation": "async_bounded_executor.v1",
                "max_concurrency": settings.streaming_max_concurrency,
                "max_pending_requests": settings.streaming_max_pending_requests,
                "request_timeout_s": settings.streaming_request_timeout_s,
                "max_retries": settings.streaming_max_retries,
                "retry_backoff_s": settings.streaming_retry_backoff_s,
                "max_total_requests": settings.streaming_max_total_requests,
                "final_wait_timeout_s": settings.streaming_final_wait_timeout_s,
                "cancel_on_supersession": settings.streaming_cancel_on_supersession,
            },
        },
        "generation": {
            "backend": result.generation_backend,
            "provider": settings.generation_provider,
            "model": result.generation_model,
            "timeout_s": settings.generation_timeout_s,
            "max_retries": settings.generation_max_retries,
            "retry_backoff_s": settings.generation_retry_backoff_s,
            "max_repair_attempts": settings.generation_max_repair_attempts,
            "max_output_tokens": settings.generation_max_output_tokens,
            "input_price_per_million": settings.generation_input_price_per_million,
            "output_price_per_million": settings.generation_output_price_per_million,
            "api_key_env": settings.generation_api_key_env,
            "api_key_configured": bool(os.environ.get(settings.generation_api_key_env)),
            "early_generation": False,
        },
    }


def build_streaming_run_manifest(
    result: StreamingReplayResult,
    *,
    corpus: CorpusIndex,
    settings: Settings,
    transcript_path: Path,
    index_path: Path,
    result_path: Path,
    trace_path: Path,
) -> StreamingRunManifest:
    return StreamingRunManifest(
        run_id=result.run_id,
        session_id=result.session_id,
        utterance_id=result.utterance_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        code_revision=_detect_code_revision(),
        configuration=_streaming_manifest_configuration(
            result,
            settings,
            corpus=corpus,
            transcript_path=transcript_path,
            index_path=index_path,
        ),
        corpus_id=corpus.corpus_id,
        index_id=corpus.manifest.index_id,
        corpus_source_kind=corpus.source_kind,
        model_identities={
            "retrieval": _retrieval_model_identity(corpus),
            "controller": "flowcontext.streaming-controller.v1",
            "scheduler": "flowcontext.async-retrieval-scheduler.v1",
            "generation": result.generation_model,
        },
        execution_mode=result.execution_mode,
        run_status=result.run_status,
        source_duration_s=result.source_duration_s,
        actual_execution_duration_ms=result.execution_duration_ms,
        result_path=str(result_path),
        trace_path=str(trace_path),
        environment=_streaming_environment(),
        secrets_excluded=True,
        mode="streaming",
    )


def write_streaming_run_manifest(path: Path, manifest: StreamingRunManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
