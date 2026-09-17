"""Timestamped replay for the complete-utterance Phase 1 baseline.

The runner consumes every transcript event, but it invokes retrieval and
generation exactly once, after the first valid final event. Repeated final
events are recorded as idempotent duplicates and never retrigger the baseline.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import ValidationError

from .answering import ExtractiveBaseline
from .config import Settings
from .contracts import (
    Answer,
    CorpusIndex,
    PreviousAnswerContext,
    ReplayResult,
    ReplayRunManifest,
    RetrievalMode,
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
from .retrieval import Retriever, make_retriever, tokenize
from .multi_intent import (
    DecompositionProvider,
    MultiIntentRetriever,
    StructuredMultiIntentDecomposer,
    filter_single_query_evidence,
    intent_metadata_from_hits,
    make_multi_intent_retriever,
)
from .trace import TraceCollector


ReplayExecutionMode = Literal["realtime", "accelerated"]

_GREETING_WORDS = frozenset(
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
_FORMATTING_WORDS = frozenset(
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
_FORMATTING_FILLER_WORDS = frozenset(
    {
        "a",
        "as",
        "be",
        "can",
        "could",
        "for",
        "i",
        "in",
        "it",
        "make",
        "me",
        "more",
        "of",
        "please",
        "put",
        "that",
        "the",
        "this",
        "to",
    }
)


def _no_retrieval_reason(query: str) -> str | None:
    """Classify turns that must not be sent to the corpus retriever."""

    raw_tokens = {token.lower() for token in re.findall(r"[\w]+", query)}
    if raw_tokens and raw_tokens.issubset(_GREETING_WORDS):
        return "greeting_no_retrieval"
    # This baseline has no prior-answer argument in the public CLI path. Keep
    # formatting-only input explicit rather than turning an instruction to
    # reformat missing context into a corpus search.
    if raw_tokens and not (raw_tokens - (_FORMATTING_WORDS | _FORMATTING_FILLER_WORDS)) and any(
        token in _FORMATTING_WORDS for token in raw_tokens
    ):
        return "formatting_needs_previous_answer_context"
    return None


def _no_retrieval_outcome(reason: str) -> GenerationOutcome:
    if reason == "greeting_no_retrieval":
        answer_text = "Hello. I can help answer questions from the configured corpus."
        uncertainty = "Greeting turn; no corpus retrieval was requested."
        status = "skipped"
    else:
        answer_text = "Please provide the original question or a completed answer to format."
        uncertainty = "Formatting context was not supplied; no corpus search or invented content was used."
        status = "abstained"
    return GenerationOutcome(
        answer=Answer(
            answer_text=answer_text,
            factual_claims=[],
            uncertainty=uncertainty,
            answer_version=1,
        ),
        status=status,
        usage=Usage(),
        cost="unavailable",
        attempts=0,
        repair_attempts=0,
    )


def _formatting_with_context_outcome(context: PreviousAnswerContext) -> GenerationOutcome:
    """Reuse same-session answer context without corpus search or rewriting."""

    return GenerationOutcome(
        answer=Answer(
            answer_text=context.answer_text,
            factual_claims=[],
            uncertainty=(
                "Formatting-only turn used the existing session answer; "
                "no corpus retrieval or selective rewrite was performed."
            ),
            answer_version=context.answer_version,
        ),
        status="skipped",
        usage=Usage(),
        cost="unavailable",
        attempts=0,
        repair_attempts=0,
    )


class ReplayError(ValueError):
    """Raised when a transcript cannot be replayed deterministically."""


@dataclass(frozen=True)
class ReplayEventRecord:
    """Validated input event plus its idempotent duplicate classification."""

    event: TranscriptEvent
    is_duplicate: bool = False
    duplicate_reason: str | None = None


def load_transcript(path: Path) -> list[TranscriptEvent]:
    """Load the provisional internal JSONL transcript format."""

    if not path.is_file():
        raise ReplayError(f"transcript does not exist: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ReplayError(f"transcript is not valid UTF-8: {path}") from exc
    events: list[TranscriptEvent] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            events.append(TranscriptEvent.model_validate(json.loads(raw_line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ReplayError(f"invalid transcript event on line {line_number} of {path}: {exc}") from exc
    if not events:
        raise ReplayError(f"transcript contains no events: {path}")
    return events


def _same_event(left: TranscriptEvent, right: TranscriptEvent) -> bool:
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def normalize_transcript(events: Sequence[TranscriptEvent]) -> list[ReplayEventRecord]:
    """Validate ordering and mark safe duplicates without mutating transcript state.

    Duplicate policy:

    * an identical repeated event ID is accepted and ignored for assembly;
    * a repeated final with a new event ID is accepted only when its final text
      and text mode exactly match the first final event;
    * conflicting event IDs, non-final events after finalisation, divergent
      final text, decreasing timestamps, and non-increasing non-duplicate
      sequence numbers fail clearly.
    """

    if not events:
        raise ReplayError("transcript contains no events")
    event_list = list(events)
    session_ids = {event.session_id for event in event_list}
    utterance_ids = {event.utterance_id for event in event_list}
    if len(session_ids) != 1 or len(utterance_ids) != 1:
        raise ReplayError("Phase 1 replay accepts one session and one utterance per invocation")

    records: list[ReplayEventRecord] = []
    seen_events: dict[str, TranscriptEvent] = {}
    final_event: TranscriptEvent | None = None
    previous_sequence: int | None = None
    previous_source_timestamp: float | None = None

    for event in event_list:
        if (
            previous_source_timestamp is not None
            and event.source_timestamp_s < previous_source_timestamp
        ):
            raise ReplayError(
                "source_timestamp_s values must be non-decreasing; "
                f"event {event.event_id!r} is out of order"
            )
        if previous_sequence is not None and event.sequence_number < previous_sequence:
            raise ReplayError(
                "sequence_number values must be non-decreasing for duplicate-safe replay; "
                f"event {event.event_id!r} is out of order"
            )

        if final_event is not None:
            if (
                not event.is_final
                or event.text != final_event.text
                or event.text_mode != final_event.text_mode
            ):
                raise ReplayError(
                    "only an identical repeated final event may follow utterance finalisation; "
                    f"event {event.event_id!r} is invalid after final event {final_event.event_id!r}"
                )
            previous = seen_events.get(event.event_id)
            if previous is not None and not _same_event(previous, event):
                raise ReplayError(
                    f"event_id {event.event_id!r} was repeated with conflicting event content"
                )
            if previous is None and previous_sequence is not None and event.sequence_number <= previous_sequence:
                raise ReplayError(
                    "sequence_number values must strictly increase for a repeated final with a new event_id; "
                    f"event {event.event_id!r} is invalid"
                )
            seen_events[event.event_id] = event
            records.append(
                ReplayEventRecord(
                    event=event,
                    is_duplicate=True,
                    duplicate_reason="repeated_final",
                )
            )
            previous_sequence = event.sequence_number
            previous_source_timestamp = event.source_timestamp_s
            continue

        previous = seen_events.get(event.event_id)
        if previous is not None:
            if not _same_event(previous, event):
                raise ReplayError(
                    f"event_id {event.event_id!r} was repeated with conflicting event content"
                )
            records.append(
                ReplayEventRecord(
                    event=event,
                    is_duplicate=True,
                    duplicate_reason="identical_event_replay",
                )
            )
            previous_sequence = event.sequence_number
            previous_source_timestamp = event.source_timestamp_s
            continue

        if previous_sequence is not None and event.sequence_number <= previous_sequence:
            raise ReplayError(
                "sequence_number values must strictly increase for non-duplicate events; "
                f"event {event.event_id!r} is invalid"
            )
        seen_events[event.event_id] = event
        records.append(ReplayEventRecord(event=event))
        previous_sequence = event.sequence_number
        previous_source_timestamp = event.source_timestamp_s
        if event.is_final:
            final_event = event

    if final_event is None:
        raise ReplayError("a replay must contain at least one final transcript event")
    return records


def validate_transcript(events: Sequence[TranscriptEvent]) -> None:
    """Compatibility wrapper that validates and discards duplicate markings."""

    normalize_transcript(events)


def reconstruct_transcript(records: Sequence[ReplayEventRecord]) -> str:
    """Assemble incremental fragments and replace state with cumulative text."""

    assembled_text = ""
    for record in records:
        if record.is_duplicate:
            continue
        if record.event.text_mode == "cumulative":
            assembled_text = record.event.text
        else:
            assembled_text += record.event.text
    return assembled_text


async def _wait_for_source_timing(
    previous_source_timestamp_s: float | None,
    source_timestamp_s: float,
    execution_mode: ReplayExecutionMode,
) -> None:
    """Apply source-time gaps only in realtime mode; always yield to asyncio."""

    if previous_source_timestamp_s is None or execution_mode == "accelerated":
        await asyncio.sleep(0)
        return
    await asyncio.sleep(max(0.0, source_timestamp_s - previous_source_timestamp_s))


def _retrieval_model_identity(corpus: CorpusIndex) -> str:
    embedding = corpus.manifest.embedding
    return ":".join(
        value
        for value in (embedding.provider, embedding.model_name or embedding.backend)
        if value
    )


def _answer_citation_ids(answer) -> list[str]:
    return sorted(
        {
            chunk_id
            for claim in answer.factual_claims
            for chunk_id in claim.supporting_chunk_ids
        }
    )


async def _execute_after_final(
    *,
    query: str,
    final_event: TranscriptEvent,
    finalization_started: float,
    corpus: CorpusIndex,
    retriever: Retriever,
    generation_provider: GenerationProvider,
    traces: TraceCollector,
    retrieval_model_identity: str,
    generation_model_identity_value: str,
    intent_queries: Sequence[str] | None = None,
) -> tuple[list, Any, float]:
    """Run the one retrieval/generation path after finalisation."""

    traces.add(
        "retrieval_decision",
        source_timestamp_s=final_event.source_timestamp_s,
        attributes={
            "decision": "retrieve",
            "reason": "utterance_final",
            "retrieval_called": True,
        },
    )
    scheduled_time_s = traces.elapsed_s()
    scheduled_trace = traces.add(
        "retrieval_scheduled",
        source_timestamp_s=final_event.source_timestamp_s,
        model_identity=retrieval_model_identity,
        usage=Usage(
            input_tokens=len(tokenize(query)),
            output_tokens=0,
            total_tokens=len(tokenize(query)),
            estimated=True,
        ),
        attributes={
            "selected_retrieval_backend": retriever.backend,
            "retrieval_method": getattr(retriever, "method", retriever.backend),
            "queue_state": "immediate_after_final_event",
            "scheduled_time_s": scheduled_time_s,
        },
    )
    actual_started_time_s = traces.elapsed_s()
    retrieval_start_trace = traces.add(
        "retrieval_started",
        source_timestamp_s=final_event.source_timestamp_s,
        model_identity=retrieval_model_identity,
        attributes={
            "actual_start_observed": True,
            "scheduled_time_s": scheduled_trace.monotonic_execution_time_s,
            "actual_started_time_s": actual_started_time_s,
            "selected_retrieval_backend": retriever.backend,
            "selected_retrieval_model": retrieval_model_identity,
            "retrieval_method": getattr(retriever, "method", retriever.backend),
        },
    )
    retrieval_started = time.perf_counter()
    try:
        candidates = retriever.search(query)
        if isinstance(retriever, MultiIntentRetriever):
            hits = candidates
            evidence_filter_decisions: list[dict[str, Any]] = []
        else:
            hits, evidence_filter_decisions = filter_single_query_evidence(query, candidates)
    except Exception as exc:
        traces.add_error(
            "retrieval_failed",
            exc,
            source_timestamp_s=final_event.source_timestamp_s,
            model_identity=retrieval_model_identity,
        )
        raise
    retrieval_details_model = getattr(retriever, "last_result", None)
    retrieval_details = (
        retrieval_details_model.model_dump(mode="json")
        if retrieval_details_model is not None
        else None
    )
    if retrieval_details is not None:
        traces.add(
            "retrieval_assembly_completed",
            source_timestamp_s=final_event.source_timestamp_s,
            duration_ms=float(retrieval_details.get("total_duration_ms", 0.0) or 0.0),
            usage=Usage.model_validate(retrieval_details.get("retrieval_usage", {})),
            attributes={
                "retrieval_mode": retrieval_details.get("retrieval_mode"),
                "retrieval_configuration": retrieval_details.get("retrieval_config"),
                "intent_result_count": len(retrieval_details.get("intent_results", [])),
                "missing_intent_ids": retrieval_details.get("missing_intent_ids", []),
                "context_budget_tokens": retrieval_details.get("context_budget_tokens"),
                "context_tokens_used": retrieval_details.get("context_tokens_used"),
                "assembly_decisions": retrieval_details.get("assembly_decisions", []),
                "evidence_filter_decisions": evidence_filter_decisions,
            },
        )
    decomposition = getattr(retriever, "last_decomposition", None)
    if decomposition is not None:
        traces.add(
            "decomposition_fallback" if decomposition.status == "fallback" else "decomposition_completed",
            source_timestamp_s=final_event.source_timestamp_s,
            duration_ms=decomposition.latency_ms,
            usage=decomposition.usage,
            attributes={
                "decomposition_status": decomposition.status,
                "decomposition_method": decomposition.decomposition_method,
                "decomposition_provider": decomposition.provider,
                "decomposition_model": decomposition.provider_model,
                "transcript_revision": decomposition.transcript_revision,
                "intent_count": len(decomposition.intents),
                "attempts": decomposition.attempts,
                "repair_attempts": decomposition.repair_attempts,
                "preserved_intent_ids": decomposition.preserved_intent_ids,
                "superseded_intent_ids": decomposition.superseded_intent_ids,
                "failure_reason": decomposition.failure_reason,
            },
        )
    unsupported_intent_queries = list(getattr(retriever, "unsupported_intent_queries", []))
    effective_intent_queries = (
        list(decomposition.sub_questions)
        if decomposition is not None
        else list(intent_queries or [])
    )
    retrieval_duration_ms = (time.perf_counter() - retrieval_started) * 1000
    query_tokens = tokenize(query)
    actual_completed_time_s = traces.elapsed_s()
    retrieval_completed_trace = traces.add(
        "retrieval_completed",
        source_timestamp_s=final_event.source_timestamp_s,
        duration_ms=retrieval_duration_ms,
        model_identity=retrieval_model_identity,
        usage=Usage(
            input_tokens=len(query_tokens),
            output_tokens=0,
            total_tokens=len(query_tokens),
            estimated=True,
        ),
        attributes={
            "hit_count": len(hits),
            "candidate_hit_count": len(candidates),
            "candidate_chunk_ids": [hit.chunk_id for hit in candidates],
            "multi_intent": decomposition is not None,
            "intent_metadata": intent_metadata_from_hits(hits),
            "unsupported_intent_count": len(unsupported_intent_queries),
            "selected_retrieval_backend": retriever.backend,
            "selected_retrieval_model": retrieval_model_identity,
            "retrieval_method": getattr(retriever, "method", retriever.backend),
            "scheduled_time_s": scheduled_trace.monotonic_execution_time_s,
            "actual_started_time_s": retrieval_start_trace.monotonic_execution_time_s,
            "actual_completed_time_s": actual_completed_time_s,
            "actual_duration_ms": retrieval_duration_ms,
        },
    )
    if hits:
        traces.add(
            "evidence_ready",
            source_timestamp_s=final_event.source_timestamp_s,
            duration_ms=retrieval_duration_ms,
            model_identity=retrieval_model_identity,
            usage=Usage(
                input_tokens=len(query_tokens),
                output_tokens=0,
                total_tokens=len(query_tokens),
                estimated=True,
            ),
            attributes={
                "valid_evidence": True,
                "evidence_ready_time_s": retrieval_completed_trace.monotonic_execution_time_s,
                "hit_chunk_ids": [hit.chunk_id for hit in hits],
            },
        )

    generation_config = generation_provider.config
    traces.add(
        "generation_started",
        source_timestamp_s=final_event.source_timestamp_s,
        model_identity=generation_model_identity_value,
        attributes={
            "generation_backend": generation_config.backend,
            "generation_model": generation_config.model,
            "streaming_observed": False,
            "latency_metric": "complete_answer_latency_ms",
        },
    )
    generation_started = time.perf_counter()
    try:
        generation_outcome = await generate_grounded_answer(
            query,
            hits,
            corpus,
            generation_provider,
            intent_queries=effective_intent_queries,
            unsupported_intent_queries=unsupported_intent_queries,
            decomposition=decomposition,
        )
    except Exception as exc:
        traces.add_error(
            "generation_failed",
            exc,
            source_timestamp_s=final_event.source_timestamp_s,
            model_identity=generation_model_identity_value,
            attributes={"generation_backend": generation_config.backend},
        )
        raise
    generation_duration_ms = (time.perf_counter() - generation_started) * 1000
    generation_error = None
    if generation_outcome.error_type and generation_outcome.error_message:
        generation_error = TraceError(
            error_type=generation_outcome.error_type,
            message=generation_outcome.error_message,
        )
    citation_ids = _answer_citation_ids(generation_outcome.answer)
    outcome_event_type = (
        "generation_skipped"
        if generation_outcome.status == "skipped"
        else "generation_abstained"
        if generation_outcome.status == "abstained"
        else "generation_completed"
    )
    outcome_attributes = {
        "answer_version": generation_outcome.answer.answer_version,
        "generation_backend": generation_config.backend,
        "generation_model": generation_config.model,
        "generation_attempts": generation_outcome.attempts,
        "generation_repair_attempts": generation_outcome.repair_attempts,
        "generation_usage": generation_outcome.generation_usage.model_dump(mode="json") if generation_outcome.generation_usage else None,
        "repair_usage": generation_outcome.repair_usage.model_dump(mode="json") if generation_outcome.repair_usage else None,
        "verification_usage": generation_outcome.verification_usage.model_dump(mode="json") if generation_outcome.verification_usage else None,
        "citation_chunk_ids": citation_ids,
        "citation_ids_validated": generation_outcome.status == "success",
        "semantic_support_evaluated": bool(generation_outcome.verification_report),
        "model_call": generation_outcome.attempts > 0,
        "streaming_observed": False,
        "latency_metric": "complete_answer_latency_ms",
    }
    traces.add(
        outcome_event_type,
        source_timestamp_s=final_event.source_timestamp_s,
        duration_ms=generation_duration_ms,
        usage=generation_outcome.usage,
        cost=generation_outcome.cost,
        model_identity=generation_model_identity_value,
        error=generation_error,
        attributes=outcome_attributes,
    )
    complete_answer_latency_ms = (time.perf_counter() - finalization_started) * 1000
    answer_event_type = "answer_failed" if generation_error is not None else "answer_completed"
    traces.add(
        answer_event_type,
        source_timestamp_s=final_event.source_timestamp_s,
        duration_ms=complete_answer_latency_ms,
        usage=generation_outcome.usage,
        cost=generation_outcome.cost,
        model_identity=generation_model_identity_value,
        error=generation_error,
        attributes={
            "answer_status": generation_outcome.status,
            "answer_version": generation_outcome.answer.answer_version,
            "citation_chunk_ids": citation_ids,
            "citation_ids_validated": generation_outcome.status == "success",
            "semantic_support_evaluated": False,
            "streaming_observed": False,
            "latency_metric": "complete_answer_latency_ms",
        },
    )
    return hits, generation_outcome, complete_answer_latency_ms


async def replay_transcript(
    events: Sequence[TranscriptEvent],
    *,
    corpus: CorpusIndex,
    top_k: int = 5,
    backend: str = "dense",
    retriever: Retriever | None = None,
    generation_provider: GenerationProvider | None = None,
    decomposition_provider: DecompositionProvider | None = None,
    run_id: str | None = None,
    model_identity: str = ExtractiveBaseline.model_identity,
    embedding_cache_dir: Path | None = None,
    embedding_local_files_only: bool = False,
    execution_mode: ReplayExecutionMode = "accelerated",
    previous_answer_context: PreviousAnswerContext | None = None,
    multi_intent: bool = False,
    retrieval_mode: RetrievalMode | None = None,
    context_budget_tokens: int = 1200,
    multi_intent_max_workers: int | None = None,
    multi_intent_rrf_k: int = 60,
    multi_intent_reranking_enabled: bool = False,
) -> ReplayResult:
    """Replay source timing and run the baseline once at first finalisation."""

    if execution_mode not in {"realtime", "accelerated"}:
        raise ReplayError(f"unsupported replay execution mode: {execution_mode!r}")
    event_records = normalize_transcript(events)
    event_list = [record.event for record in event_records]
    session_id = event_list[0].session_id
    utterance_id = event_list[0].utterance_id
    if previous_answer_context is not None and previous_answer_context.session_id != session_id:
        raise ReplayError("previous answer context belongs to another session")
    effective_run_id = run_id or f"run-{session_id}"
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
    selected_backend = selected_retriever.backend
    selected_generation_provider = generation_provider or MockGenerationProvider()
    generation_config = selected_generation_provider.config
    selected_generation_identity = generation_model_identity(generation_config)
    selected_retrieval_identity = _retrieval_model_identity(corpus)
    traces = TraceCollector(effective_run_id, session_id, model_identity)
    replay_started = time.perf_counter()
    traces.add(
        "replay_started",
        source_timestamp_s=event_list[0].source_timestamp_s,
        attributes={
            "baseline_mode": "wait_for_complete_utterance",
            "mode": "baseline",
            "execution_mode": execution_mode,
            "selected_retrieval_backend": selected_backend,
            "selected_retrieval_model": selected_retrieval_identity,
            "retrieval_top_k": top_k,
            "multi_intent": multi_intent,
            "selected_generation_backend": generation_config.backend,
            "selected_generation_model": generation_config.model,
            "streaming_generation": False,
        },
    )

    assembled_text = ""
    final_event: TranscriptEvent | None = None
    retrieval_hits: list = []
    generation_outcome = None
    retrieval_details: dict[str, Any] | None = None
    complete_answer_latency_ms = 0.0
    finalization_started: float | None = None
    final_event_delivery_time_s = 0.0
    retrieval_triggered = False
    previous_source_timestamp: float | None = None
    duplicate_event_count = sum(record.is_duplicate for record in event_records)

    for record in event_records:
        event = record.event
        await _wait_for_source_timing(
            previous_source_timestamp,
            event.source_timestamp_s,
            execution_mode,
        )
        previous_source_timestamp = event.source_timestamp_s
        delivery_time_s = traces.elapsed_s()
        traces.add(
            "transcript_event_received",
            source_timestamp_s=event.source_timestamp_s,
            actual_delivery_time_s=delivery_time_s,
            attributes={
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
        if record.is_duplicate:
            continue

        if event.text_mode == "cumulative":
            assembled_text = event.text
        else:
            assembled_text += event.text

        if not event.is_final:
            traces.add(
                "retrieval_decision",
                source_timestamp_s=event.source_timestamp_s,
                attributes={
                    "decision": "wait",
                    "reason": "utterance_not_final",
                    "retrieval_called": False,
                },
            )
            continue

        final_event = event
        final_event_delivery_time_s = delivery_time_s
        finalization_started = time.perf_counter()
        traces.add(
            "final_event_delivered",
            source_timestamp_s=event.source_timestamp_s,
            actual_delivery_time_s=delivery_time_s,
            attributes={
                "event_id": event.event_id,
                "actual_delivery_time_s": delivery_time_s,
                "delivery_mode": execution_mode,
            },
        )
        query = assembled_text.strip()
        if not query:
            error = ReplayError("final transcript produced an empty query")
            traces.add_error("replay_failed", error, source_timestamp_s=event.source_timestamp_s)
            raise error
        traces.add(
            "utterance_finalized",
            source_timestamp_s=event.source_timestamp_s,
            attributes={
                "final_event_id": event.event_id,
                "text_mode": event.text_mode,
                "query_character_count": len(query),
                "duplicate_events_ignored": duplicate_event_count,
            },
        )
        no_retrieval_reason = _no_retrieval_reason(query)
        if (
            no_retrieval_reason == "formatting_needs_previous_answer_context"
            and previous_answer_context is not None
        ):
            no_retrieval_reason = "formatting_with_previous_answer_context"
        if no_retrieval_reason is not None:
            retrieval_hits = []
            retrieval_triggered = False
            traces.add(
                "retrieval_decision",
                source_timestamp_s=event.source_timestamp_s,
                model_identity=model_identity,
                attributes={
                    "decision": "skip",
                    "reason": no_retrieval_reason,
                    "retrieval_called": False,
                    "mode": "baseline",
                },
            )
            if no_retrieval_reason == "formatting_with_previous_answer_context":
                generation_outcome = _formatting_with_context_outcome(previous_answer_context)
            else:
                generation_outcome = _no_retrieval_outcome(no_retrieval_reason)
            traces.add(
                "generation_skipped",
                source_timestamp_s=event.source_timestamp_s,
                model_identity=selected_generation_identity,
                usage=generation_outcome.usage,
                cost=generation_outcome.cost,
                attributes={
                    "reason": no_retrieval_reason,
                    "generation_backend": generation_config.backend,
                    "generation_model": generation_config.model,
                    "model_call": False,
                    "first_answer_content_observed": False,
                },
            )
            complete_answer_latency_ms = (time.perf_counter() - finalization_started) * 1000
            traces.add(
                "answer_completed",
                source_timestamp_s=event.source_timestamp_s,
                duration_ms=complete_answer_latency_ms,
                model_identity=selected_generation_identity,
                usage=generation_outcome.usage,
                cost=generation_outcome.cost,
                attributes={
                    "answer_status": generation_outcome.status,
                    "no_retrieval": True,
                    "latency_metric": "answer_latency_from_final_event_delivery_ms",
                },
            )
        else:
            retrieval_triggered = True
            try:
                retrieval_hits, generation_outcome, complete_answer_latency_ms = await _execute_after_final(
                    query=query,
                    final_event=event,
                    finalization_started=finalization_started,
                    corpus=corpus,
                    retriever=selected_retriever,
                    generation_provider=selected_generation_provider,
                    traces=traces,
                    retrieval_model_identity=selected_retrieval_identity,
                    generation_model_identity_value=selected_generation_identity,
                    intent_queries=None,
                )
                result_model = getattr(selected_retriever, "last_result", None)
                retrieval_details = (
                    result_model.model_dump(mode="json")
                    if result_model is not None
                    else None
                )
            except Exception as exc:
                # Retrieval failures still produce a complete, inspectable
                # baseline artifact. Generation is not attempted and the
                # explicit failed answer prevents a caller from mistaking a
                # missing backend for a successful mock fallback.
                retrieval_hits = []
                result_model = getattr(selected_retriever, "last_result", None)
                retrieval_details = (
                    result_model.model_dump(mode="json")
                    if result_model is not None
                    else None
                )
                retrieval_error = TraceError(error_type=type(exc).__name__, message=str(exc))
                generation_outcome = GenerationOutcome(
                    answer=Answer(
                        answer_text="I cannot safely answer because retrieval failed.",
                        factual_claims=[],
                        uncertainty=(
                            "The configured retrieval backend failed; no outside knowledge "
                            "or fallback backend was used."
                        ),
                        answer_version=1,
                    ),
                    status="abstained",
                    usage=Usage(),
                    cost="unavailable",
                    attempts=0,
                    repair_attempts=0,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                complete_answer_latency_ms = (time.perf_counter() - finalization_started) * 1000
                traces.add(
                    "generation_skipped",
                    source_timestamp_s=event.source_timestamp_s,
                    model_identity=selected_generation_identity,
                    usage=generation_outcome.usage,
                    cost=generation_outcome.cost,
                    error=retrieval_error,
                    attributes={
                        "reason": "retrieval_failed",
                        "generation_backend": generation_config.backend,
                        "generation_model": generation_config.model,
                        "model_call": False,
                    },
                )
                traces.add(
                    "answer_failed",
                    source_timestamp_s=event.source_timestamp_s,
                    duration_ms=complete_answer_latency_ms,
                    model_identity=selected_generation_identity,
                    usage=generation_outcome.usage,
                    cost=generation_outcome.cost,
                    error=retrieval_error,
                    attributes={
                        "answer_status": generation_outcome.status,
                        "retrieval_failed": True,
                        "latency_metric": "answer_latency_from_final_event_delivery_ms",
                    },
                )

    if final_event is None or generation_outcome is None or finalization_started is None:
        error = ReplayError("no final transcript event")
        traces.add_error("replay_failed", error)
        raise error

    query = assembled_text.strip()
    execution_duration_ms = (time.perf_counter() - replay_started) * 1000
    run_status: Literal["completed", "abstained", "failed"]
    if generation_outcome.error_type:
        run_status = "failed"
    elif not retrieval_triggered and generation_outcome.status == "skipped":
        # A greeting is handled successfully by policy even though it does
        # not invoke retrieval or a generation provider.
        run_status = "completed"
    elif generation_outcome.status == "success":
        run_status = "completed"
    else:
        run_status = "abstained"
    traces.add(
        "replay_completed",
        source_timestamp_s=event_list[-1].source_timestamp_s,
        duration_ms=execution_duration_ms,
        attributes={
            "retrieval_triggered_after_final": retrieval_triggered,
            "retrieval_started_early": False,
            "generation_status": generation_outcome.status,
            "run_status": run_status,
            "execution_mode": execution_mode,
            "duplicate_events_ignored": duplicate_event_count,
        },
    )
    return ReplayResult(
        run_id=effective_run_id,
        session_id=session_id,
        utterance_id=utterance_id,
        baseline_mode="wait_for_complete_utterance",
        corpus_source_kind=corpus.source_kind,
        retrieval_backend=selected_backend,
        generation_backend=generation_config.backend,
        generation_model=generation_config.model,
        generation_status=(
            "abstained" if generation_outcome.status == "stale_rejected" else generation_outcome.status
        ),
        generation_usage=generation_outcome.usage,
        generation_cost=generation_outcome.cost,
        generation_attempts=generation_outcome.attempts,
        generation_repair_attempts=generation_outcome.repair_attempts,
        execution_mode=execution_mode,
        run_status=run_status,
        final_event_id=final_event.event_id,
        final_source_timestamp_s=final_event.source_timestamp_s,
        transcript_event_count=len(event_list),
        duplicate_event_count=duplicate_event_count,
        source_duration_s=max(
            0.0,
            event_list[-1].source_timestamp_s - event_list[0].source_timestamp_s,
        ),
        execution_duration_ms=execution_duration_ms,
        complete_answer_latency_ms=complete_answer_latency_ms,
        query=query,
        retrieval_triggered=retrieval_triggered,
        retrieval_hits=retrieval_hits,
        decomposition=(
            getattr(selected_retriever, "last_decomposition", None).model_dump(mode="json")
            if multi_intent and getattr(selected_retriever, "last_decomposition", None) is not None
            else None
        ),
        retrieval_details=retrieval_details,
        answer=generation_outcome.answer,
        traces=traces.events,
        mode="baseline",
        final_event_delivery_time_s=final_event_delivery_time_s,
        evidence_ready_time_s=next(
            (
                trace.monotonic_execution_time_s
                for trace in traces.events
                if trace.event_type == "evidence_ready"
            ),
            None,
        ),
        generation_started_time_s=next(
            (
                trace.monotonic_execution_time_s
                for trace in traces.events
                if trace.event_type == "generation_started"
            ),
            None,
        ),
        generation_completed_time_s=next(
            (
                trace.monotonic_execution_time_s
                for trace in traces.events
                if trace.event_type in {"generation_completed", "generation_abstained"}
            ),
            None,
        ),
        generation_first_content_time_s=None,
        retrieval_started_early=False,
        valid_evidence_ready_before_finalization=False,
        early_evidence_reused=False,
        answer_latency_from_final_event_delivery_ms=complete_answer_latency_ms,
        full_interaction_duration_ms=execution_duration_ms,
        controller_overhead_ms=0.0,
        retrieval_call_count=sum(
            1 for trace in traces.events if trace.event_type == "retrieval_started"
        ),
        scheduled_request_count=int(retrieval_triggered),
        superseded_request_count=0,
        cancelled_request_count=0,
        timed_out_request_count=0,
        retrieval_error_count=sum(
            1
            for trace in traces.events
            if trace.error is not None
            and (
                trace.event_type.startswith("retrieval")
                or trace.event_type == "retrieval_scheduled"
            )
        ),
        stale_result_discard_count=0,
        retrieval_usage=next(
            (
                trace.usage
                for trace in reversed(traces.events)
                if trace.event_type in {"retrieval_completed", "retrieval_started", "retrieval_scheduled"}
                and trace.usage.total_tokens > 0
            ),
            Usage(),
        ),
        errors=[trace.error for trace in traces.events if trace.error is not None],
        corpus_id=corpus.corpus_id,
        index_id=corpus.manifest.index_id,
        retrieval_top_k=top_k,
    )


def write_replay(path: Path, result: ReplayResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_replay(path: Path) -> ReplayResult:
    if not path.is_file():
        raise ReplayError(f"replay output does not exist: {path}")
    try:
        return ReplayResult.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ReplayError(f"invalid replay output {path}: {exc}") from exc


def _detect_code_revision() -> str | None:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _safe_manifest_configuration(
    settings: Settings,
    result: ReplayResult,
    *,
    transcript_path: Path,
    index_path: Path,
) -> dict[str, Any]:
    """Return configuration without endpoint URLs, key values, or other secrets."""

    return {
        "transcript_path": str(transcript_path),
        "index_path": str(index_path),
        "execution_mode": result.execution_mode,
        "mode": "baseline",
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
        "retrieval": {
            "backend": result.retrieval_backend,
            "top_k": result.retrieval_top_k,
            "phase3_mode": settings.multi_intent_retrieval_mode,
            "phase3_context_budget_tokens": settings.multi_intent_context_budget_tokens,
            "phase3_max_workers": settings.multi_intent_max_workers,
            "phase3_rrf_k": settings.multi_intent_rrf_k,
            "phase3_details": result.retrieval_details,
        },
        "chunking": {
            "max_chars": settings.chunk_max_chars,
            "overlap_chars": settings.chunk_overlap_chars,
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
            "api_key_env": settings.generation_api_key_env,
            "api_key_configured": bool(os.environ.get(settings.generation_api_key_env)),
            "input_price_per_million": settings.generation_input_price_per_million,
            "output_price_per_million": settings.generation_output_price_per_million,
        },
    }


def build_run_manifest(
    result: ReplayResult,
    *,
    corpus: CorpusIndex,
    settings: Settings,
    transcript_path: Path,
    index_path: Path,
    result_path: Path,
    trace_path: Path,
) -> ReplayRunManifest:
    """Build a secret-free manifest for a completed or abstained replay."""

    generation_trace = next(
        (
            trace
            for trace in result.traces
            if trace.event_type in {
                "generation_started",
                "generation_completed",
                "generation_abstained",
                "generation_skipped",
            }
        ),
        None,
    )
    generation_identity = (
        generation_trace.model_identity
        if generation_trace is not None
        else f"{settings.generation_provider}:{result.generation_model}"
    )
    return ReplayRunManifest(
        run_id=result.run_id,
        session_id=result.session_id,
        utterance_id=result.utterance_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        code_revision=_detect_code_revision(),
        configuration=_safe_manifest_configuration(
            settings,
            result,
            transcript_path=transcript_path,
            index_path=index_path,
        ),
        corpus_id=corpus.corpus_id,
        index_id=corpus.manifest.index_id,
        corpus_source_kind=corpus.source_kind,
        model_identities={
            "retrieval": _retrieval_model_identity(corpus),
            "generation": generation_identity,
        },
        execution_mode=result.execution_mode,
        run_status=result.run_status,
        source_duration_s=result.source_duration_s,
        actual_execution_duration_ms=result.execution_duration_ms,
        complete_answer_latency_ms=result.complete_answer_latency_ms,
        result_path=str(result_path),
        trace_path=str(trace_path),
        environment={
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "os_name": os.name,
            "executable": sys.executable,
        },
        secrets_excluded=True,
        mode="baseline",
    )


def write_run_manifest(path: Path, manifest: ReplayRunManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
