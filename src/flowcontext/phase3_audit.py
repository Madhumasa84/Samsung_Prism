"""Dedicated Phase 3 evaluation suite and audit.

The application does not contain expected intents, expected answers, or
relevance labels.  Those records are loaded from the external Phase 3 JSONL
assets and are kept in the report only as evaluation data.  This module runs
three matched conditions over the same corpus and replay cases:

* A: original final-event baseline;
* B: Phase 2 streaming with one parent query; and
* C: Phase 3 streaming with decomposition and evidence fusion.

The local fixture is synthetic and its labels are provisional.  Citation-ID
validity is deterministic provenance checking; it is intentionally not used
as semantic claim support.  The report therefore keeps the guide's citation
support target separate and marks it ``NOT VERIFIED`` until claims are
reviewed by a governed human or model evaluator.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, Sequence

from pydantic import ValidationError

from .config import Settings
from .contracts import (
    CorpusIndex,
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    Phase3EvaluationCase,
    RetrievalHit,
    StreamingDecisionConfig,
    StreamingSchedulerConfig,
    Usage,
)
from .evaluation import _code_revision, _environment, _hardware
from .generation import (
    GenerationProviderError,
    generation_provider_for_settings,
)
from .multi_intent import (
    DecompositionProviderError,
    MultiIntentPlan,
    MultiIntentRetriever,
    decomposition_provider_for_settings,
)
from .retrieval import RetrievalError, Retriever, make_retriever
from .replay import replay_transcript
from .scheduler import scheduler_config_from_settings
from .streaming import (
    replay_streaming_transcript,
    streaming_config_from_settings,
)


class Phase3AuditError(ValueError):
    """Raised when dedicated Phase 3 evaluation assets are unusable."""


_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_REVIEW_STATUSES = (
    "provisional_generated",
    "model_reviewed",
    "human_review_pending",
    "human_reviewed",
)
_CONDITIONS = ("A_original_baseline", "B_phase2_streaming_single_query", "C_phase3_streaming_fused")


@dataclass
class _RunOutcome:
    """Internal result wrapper that keeps one failed arm from hiding a case."""

    condition: str
    result: Any | None
    error: str | None
    retriever_calls: int
    backend: str


class _ScenarioRetriever:
    """Apply explicit fixture behavior around a real local retriever."""

    def __init__(self, base: Retriever, *, behavior: str, delay_s: float) -> None:
        self.base = base
        self.backend = base.backend
        self.method = getattr(base, "method", base.backend)
        self.behavior = behavior
        self.delay_s = delay_s
        self.calls = 0

    def search(self, query: str) -> list[RetrievalHit]:
        self.calls += 1
        if self.behavior in {"delay", "timeout"} and self.delay_s > 0:
            time.sleep(self.delay_s)
        if self.behavior == "failure":
            raise RetrievalError("simulated Phase 3 retrieval failure")
        return self.base.search(query)


class _FailingGenerationProvider:
    """Explicit provider fault used only by labelled fixture cases."""

    def __init__(self, config: GenerationConfig) -> None:
        self._config = config
        self.calls = 0

    @property
    def config(self) -> GenerationConfig:
        return self._config

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        del request
        self.calls += 1
        raise GenerationProviderError(
            "simulated Phase 3 generation provider failure",
            retryable=False,
        )


class _FailingDecompositionProvider:
    """Explicit decomposition-provider fault used by one held-out case."""

    def __init__(self, config: GenerationConfig) -> None:
        self._config = config
        self.calls = 0

    @property
    def config(self) -> GenerationConfig:
        return self._config

    async def decompose(self, request: Any) -> GenerationResult:
        del request
        self.calls += 1
        raise DecompositionProviderError(
            "simulated Phase 3 decomposition provider failure",
            retryable=False,
        )


def load_phase3_evaluation_cases(
    path: Path,
    *,
    expected_split: str | None = None,
) -> list[Phase3EvaluationCase]:
    """Load the strict external Phase 3 JSONL label asset."""

    if not path.is_file():
        raise Phase3AuditError(f"Phase 3 evaluation asset does not exist: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise Phase3AuditError(f"Phase 3 evaluation asset is not UTF-8: {path}") from exc
    cases: list[Phase3EvaluationCase] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            case = Phase3EvaluationCase.model_validate(json.loads(raw_line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise Phase3AuditError(
                f"invalid Phase 3 evaluation case on line {line_number}: {exc}"
            ) from exc
        if expected_split is not None and case.split != expected_split:
            raise Phase3AuditError(
                f"Phase 3 case {case.case_id!r} is in split {case.split!r}, "
                f"not requested split {expected_split!r}"
            )
        cases.append(case)
    if not cases:
        raise Phase3AuditError(f"Phase 3 evaluation asset contains no cases: {path}")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise Phase3AuditError("Phase 3 case IDs must be unique across the selected assets")
    return cases


def load_phase3_implementation_changes(path: Path) -> dict[str, Any]:
    """Load the external implementation-change log without embedding labels."""

    if not path.is_file():
        return {
            "schema_version": "flowcontext.phase3-implementation-change-log.v1",
            "status": "not_available",
            "changes": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase3AuditError(f"invalid implementation-change log: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("changes"), list):
        raise Phase3AuditError("implementation-change log must contain a changes list")
    return payload


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_PATTERN.findall(value)}


def _normalized_tokens(value: str) -> set[str]:
    tokens = _tokens(value)
    if "do" in tokens and "not" in tokens:
        tokens.discard("do")
    if "does" in tokens and "not" in tokens:
        tokens.discard("does")
    return tokens


def _overlap(reference: str, candidate: str) -> float:
    expected = _normalized_tokens(reference)
    actual = _normalized_tokens(candidate)
    return len(expected & actual) / len(expected) if expected else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _usage(value: Any) -> Usage:
    if isinstance(value, Usage):
        return value
    if isinstance(value, dict):
        try:
            return Usage.model_validate(value)
        except ValidationError:
            return Usage()
    return Usage()


def _usage_sum(usages: Sequence[Usage]) -> Usage:
    if not usages:
        return Usage()
    return Usage(
        input_tokens=sum(item.input_tokens for item in usages),
        output_tokens=sum(item.output_tokens for item in usages),
        total_tokens=sum(item.total_tokens for item in usages),
        estimated=any(item.estimated for item in usages),
    )


def _trace_attributes(result: Any, event_names: set[str]) -> dict[str, Any]:
    for trace in reversed(getattr(result, "traces", [])):
        if trace.event_type in event_names:
            return dict(trace.attributes)
    return {}


def _decomposition_usage_and_calls(result: Any) -> tuple[Usage, int, int]:
    history = getattr(result, "decomposition_history", None) or []
    if history:
        plans = [item for item in history if isinstance(item, dict)]
        return (
            _usage_sum([_usage(item.get("usage")) for item in plans]),
            sum(int(item.get("attempts", 0) or 0) for item in plans),
            sum(int(item.get("repair_attempts", 0) or 0) for item in plans),
        )
    plan = getattr(result, "decomposition", None)
    if isinstance(plan, dict):
        return (
            _usage(plan.get("usage")),
            int(plan.get("attempts", 0) or 0),
            int(plan.get("repair_attempts", 0) or 0),
        )
    return Usage(), 0, 0


def _terminal_generation_usage(result: Any) -> dict[str, Usage]:
    attrs = _trace_attributes(
        result,
        {
            "generation_completed",
            "generation_abstained",
            "generation_skipped",
            "streaming_generation_completed",
            "streaming_generation_failed",
            "streaming_generation_skipped",
            "streaming_generation_stale_rejected",
        },
    )
    return {
        "initial_generation": _usage(attrs.get("generation_usage")),
        "repair": _usage(attrs.get("repair_usage")),
        "verification": _usage(attrs.get("verification_usage")),
    }


def _trace_completeness(condition: str, result: Any | None) -> tuple[bool, list[str]]:
    if result is None:
        return False, ["replay_result"]
    event_types = {trace.event_type for trace in result.traces}
    missing: list[str] = []

    def require(name: str) -> None:
        if name not in event_types:
            missing.append(name)

    def require_any(name: str, options: set[str]) -> None:
        if not event_types.intersection(options):
            missing.append(name)

    if condition == "A_original_baseline":
        for name in (
            "replay_started",
            "transcript_event_received",
            "final_event_delivered",
            "utterance_finalized",
            "retrieval_decision",
        ):
            require(name)
        if getattr(result, "retrieval_triggered", False):
            require("retrieval_scheduled")
            require("retrieval_started")
            require_any("retrieval_completed/retrieval_failed", {"retrieval_completed", "retrieval_failed"})
        else:
            require_any("generation_skipped/answer_completed", {"generation_skipped", "answer_completed"})
        require_any(
            "generation_terminal",
            {"generation_completed", "generation_abstained", "generation_skipped"},
        )
        require_any("answer_terminal", {"answer_completed", "answer_failed"})
        require("replay_completed")
    else:
        for name in (
            "streaming_replay_started",
            "transcript_event_received",
            "streaming_decision",
            "final_event_delivered",
            "utterance_finalized",
        ):
            require(name)
        require_any("generation_terminal", {"streaming_generation_completed", "streaming_generation_failed", "streaming_generation_skipped", "streaming_generation_stale_rejected"})
        require_any("answer_terminal", {"streaming_answer_completed", "streaming_answer_failed"})
        require_any("streaming_replay_terminal", {"streaming_replay_completed", "streaming_replay_failed"})
    return not missing, list(dict.fromkeys(missing))


def _controller_config(settings: Settings, case: Phase3EvaluationCase) -> StreamingDecisionConfig:
    values = streaming_config_from_settings(settings).model_dump(mode="python")
    values.update(case.controller_overrides)
    try:
        return StreamingDecisionConfig.model_validate(values)
    except ValidationError as exc:
        raise Phase3AuditError(
            f"invalid controller_overrides for case {case.case_id}: {exc}"
        ) from exc


def _scheduler_config(settings: Settings, case: Phase3EvaluationCase) -> StreamingSchedulerConfig:
    values = scheduler_config_from_settings(settings).model_dump(mode="python")
    if case.retrieval_behavior == "timeout":
        values["request_timeout_s"] = case.retrieval_timeout_s
    try:
        return StreamingSchedulerConfig.model_validate(values)
    except ValidationError as exc:
        raise Phase3AuditError(
            f"invalid scheduler settings for case {case.case_id}: {exc}"
        ) from exc


def _generation_provider(settings: Settings, case: Phase3EvaluationCase) -> Any:
    provider = generation_provider_for_settings(settings)
    if case.provider_failure_stage == "generation":
        return _FailingGenerationProvider(provider.config)
    return provider


def _decomposition_provider(settings: Settings, case: Phase3EvaluationCase) -> Any:
    provider = decomposition_provider_for_settings(settings)
    if case.provider_failure_stage == "decomposition":
        return _FailingDecompositionProvider(provider.config)
    return provider


async def _run_condition(
    case: Phase3EvaluationCase,
    *,
    corpus: CorpusIndex,
    settings: Settings,
    backend: str,
    retrieval_mode: str,
    top_k: int,
    execution_mode: Literal["realtime", "accelerated"],
    condition: str,
) -> _RunOutcome:
    try:
        base = make_retriever(
            corpus,
            backend=backend,
            top_k=top_k,
            cache_dir=settings.embedding_cache_dir,
            local_files_only=settings.embedding_local_files_only,
        )
        scenario_retriever = _ScenarioRetriever(
            base,
            behavior=case.retrieval_behavior,
            delay_s=case.retrieval_delay_s,
        )
        provider = _generation_provider(settings, case)
        if condition == "A_original_baseline":
            result = await replay_transcript(
                case.transcript,
                corpus=corpus,
                top_k=top_k,
                backend=backend,
                retriever=scenario_retriever,
                generation_provider=provider,
                run_id=f"phase3-audit-{case.case_id}-A",
                execution_mode=execution_mode,
            )
        elif condition == "B_phase2_streaming_single_query":
            result = await replay_streaming_transcript(
                case.transcript,
                corpus=corpus,
                top_k=top_k,
                backend=backend,
                retriever=scenario_retriever,
                generation_provider=provider,
                run_id=f"phase3-audit-{case.case_id}-B",
                execution_mode=execution_mode,
                controller_config=_controller_config(settings, case),
                scheduler_config=_scheduler_config(settings, case),
                multi_intent=False,
            )
        elif condition == "C_phase3_streaming_fused":
            result = await replay_streaming_transcript(
                case.transcript,
                corpus=corpus,
                top_k=top_k,
                backend=backend,
                retriever=scenario_retriever,
                generation_provider=provider,
                decomposition_provider=_decomposition_provider(settings, case),
                run_id=f"phase3-audit-{case.case_id}-C",
                execution_mode=execution_mode,
                controller_config=_controller_config(settings, case),
                scheduler_config=_scheduler_config(settings, case),
                multi_intent=True,
                retrieval_mode=retrieval_mode,
                context_budget_tokens=settings.multi_intent_context_budget_tokens,
                multi_intent_max_workers=settings.multi_intent_max_workers,
                multi_intent_rrf_k=settings.multi_intent_rrf_k,
                multi_intent_reranking_enabled=settings.multi_intent_reranking_enabled,
            )
        else:
            raise Phase3AuditError(f"unknown Phase 3 audit condition: {condition}")
        return _RunOutcome(condition, result, None, scenario_retriever.calls, backend)
    except Exception as exc:  # retain the failed arm in the machine-readable audit
        return _RunOutcome(
            condition,
            None,
            f"{type(exc).__name__}: {exc}",
            0,
            backend,
        )


def _expected_union(case: Phase3EvaluationCase) -> list[str]:
    return list(
        dict.fromkeys(
            chunk_id
            for intent in case.expected_intents
            if intent.answerable
            for chunk_id in intent.relevant_chunk_ids
        )
    )


def _retrieval_metrics(expected_ids: Sequence[str], retrieved_ids: Sequence[str]) -> dict[str, float]:
    expected = set(expected_ids)
    if not expected:
        return {}
    ranked = list(retrieved_ids)
    return {
        f"recall_at_{k}": len(set(ranked[:k]) & expected) / len(expected)
        for k in (1, 3, 5)
    }


def _hit_record(hit: RetrievalHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "source_location": hit.source_location,
        "snippet_text": hit.snippet_text,
        "rank": hit.rank,
        "score": hit.score,
        "retrieval_method": hit.retrieval_method,
        "intent_id": hit.intent_id,
        "intent_ids": list(hit.intent_ids),
        "intent_ranks": dict(hit.intent_ranks),
        "intent_scores": dict(hit.intent_scores),
        "rrf_score": hit.rrf_score,
    }


def _claim_records(answer: Any) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": claim.claim_id,
            "claim_text": claim.claim_text,
            "intent_id": claim.intent_id,
            "intent_ids": list(claim.intent_ids),
            "supporting_chunk_ids": list(claim.supporting_chunk_ids),
            "supporting_excerpts": list(claim.supporting_excerpts),
            "semantic_support": claim.semantic_support,
        }
        for claim in getattr(answer, "factual_claims", [])
    ]


def _mode_record(
    case: Phase3EvaluationCase,
    outcome: _RunOutcome,
    *,
    corpus: CorpusIndex,
    top_k: int,
) -> dict[str, Any]:
    result = outcome.result
    known_ids = {chunk.chunk_id for chunk in corpus.chunks}
    if result is None:
        trace_complete, missing_trace = _trace_completeness(outcome.condition, None)
        return {
            "condition": outcome.condition,
            "run_status": "failed",
            "generation_status": "failed",
            "final_query": None,
            "retrieval_backend": outcome.backend,
            "retrieved_ids": [],
            "retrieved_hits": [],
            "relevant_ids": _expected_union(case),
            "retrieval_metrics": _retrieval_metrics(_expected_union(case), []),
            "retrieval_quality_scored": False,
            "retrieval_quality_exclusion_reason": "replay result was not produced",
            "retrieval_details": None,
            "decomposition": None,
            "decomposition_status": None,
            "answer_text": "",
            "answer_uncertainty": None,
            "intent_statuses": [],
            "verification_audit": None,
            "factual_claims": [],
            "citation_ids": [],
            "citation_id_count": 0,
            "citation_ids_valid": None,
            "unknown_citation_ids": [],
            "semantic_claim_support": "not_evaluated",
            "supported_answer": {
                "supported_intent_count": 0,
                "answerable_intent_denominator": sum(intent.answerable for intent in case.expected_intents),
                "coverage": 0.0 if any(intent.answerable for intent in case.expected_intents) else None,
                "expected_answer_substrings_passed": False,
            },
            "complete_request_evidence_coverage": {
                "covered_intent_count": 0,
                "answerable_intent_denominator": sum(intent.answerable for intent in case.expected_intents),
                "coverage": 0.0 if any(intent.answerable for intent in case.expected_intents) else None,
                "all_answerable_intents_covered": False,
            },
            "uncertainty_behavior": {
                "uncertainty_present": False,
                "unsupported_intents_marked": False,
                "no_factual_claims": True,
                "status": "not_assessed_due_to_replay_failure",
            },
            "per_intent_retrieval": [],
            "early_retrieval_started": False,
            "valid_evidence_ready_before_final": False,
            "useful_early_reuse": False,
            "stale_result_discard_count": 0,
            "stale_result_accepted_count": 0,
            "final_event_to_answer_latency_ms": None,
            "full_interaction_duration_ms": None,
            "retrieval_call_count": outcome.retriever_calls,
            "model_calls": {"generation": 0, "decomposition": 0, "total": 0},
            "token_usage": {
                "retrieval": Usage().model_dump(mode="json"),
                "decomposition": Usage().model_dump(mode="json"),
                "generation": Usage().model_dump(mode="json"),
                "repair": Usage().model_dump(mode="json"),
                "verification": Usage().model_dump(mode="json"),
                "total": Usage().model_dump(mode="json"),
            },
            "errors": [outcome.error] if outcome.error else [],
            "trace": {
                "complete": trace_complete,
                "missing_events": missing_trace,
                "event_count": 0,
                "event_types": [],
            },
        }

    if outcome.condition == "A_original_baseline":
        hits = list(result.retrieval_hits)
        final_query = result.query
        retrieval_details = result.retrieval_details
        generation_status = result.generation_status
        retrieval_started_early = False
        valid_evidence_ready = False
        useful_reuse = False
        stale_discard = 0
        stale_accepted = 0
        final_latency = result.answer_latency_from_final_event_delivery_ms
        full_duration = result.full_interaction_duration_ms
        retrieval_calls = result.retrieval_call_count
        retrieval_usage = result.retrieval_usage
        result_errors = list(result.errors)
    else:
        hits = list(result.current_evidence_hits)
        final_query = result.final_query
        retrieval_details = result.retrieval_details
        generation_status = result.generation_status
        retrieval_started_early = result.retrieval_started_early
        valid_evidence_ready = result.valid_evidence_ready_before_finalization
        useful_reuse = result.early_evidence_reused
        stale_discard = result.stale_result_discard_count
        stale_accepted = sum(
            item.accepted and item.stale for item in result.retrieval_results
        )
        final_latency = result.answer_latency_from_final_event_delivery_ms
        full_duration = result.full_interaction_duration_ms
        retrieval_calls = result.retrieval_call_count
        retrieval_usage = result.retrieval_usage
        result_errors = list(result.errors)

    answer = result.answer
    retrieved_ids = [hit.chunk_id for hit in hits]
    expected_ids = _expected_union(case)
    retrieval_detail_errors = (
        list(retrieval_details.get("errors", []))
        if isinstance(retrieval_details, dict)
        else []
    )
    errors = list(dict.fromkeys(
        [
            *(
                error.error_type if hasattr(error, "error_type") else str(error)
                for error in result_errors
            ),
            *retrieval_detail_errors,
        ]
    ))
    cited_ids = sorted(
        {
            chunk_id
            for claim in answer.factual_claims
            for chunk_id in claim.supporting_chunk_ids
        }
    )
    unknown_citations = sorted(set(cited_ids) - set(retrieved_ids))
    citation_valid = (
        set(cited_ids).issubset(known_ids) and not unknown_citations
        if cited_ids
        else None
    )
    trace_complete, missing_trace = _trace_completeness(outcome.condition, result)
    decomp_usage, decomp_calls, decomp_repairs = _decomposition_usage_and_calls(result)
    terminal_usage = _terminal_generation_usage(result)
    generation_usage = result.generation_usage
    repair_usage = terminal_usage["repair"]
    verification_usage = terminal_usage["verification"]
    initial_generation_usage = terminal_usage["initial_generation"]
    if generation_usage.total_tokens and initial_generation_usage.total_tokens == 0 and repair_usage.total_tokens == 0:
        initial_generation_usage = generation_usage
    total_usage = _usage_sum(
        [retrieval_usage, decomp_usage, generation_usage, verification_usage]
    )
    # ``generation_usage`` is already initial+repair in the runtime result.
    # Keep the reported total additive by using the stage-specific split here.
    generation_stage_total = _usage_sum([initial_generation_usage, repair_usage])
    total_usage = _usage_sum([retrieval_usage, decomp_usage, generation_stage_total, verification_usage])
    generation_calls = int(getattr(result, "generation_attempts", 0) or 0)
    if generation_calls == 0:
        generation_calls = sum(
            trace.attributes.get("generation_attempts", 0) or 0
            for trace in result.traces
            if trace.event_type in {
                "generation_completed",
                "generation_abstained",
                "generation_skipped",
                "streaming_generation_completed",
                "streaming_generation_failed",
                "streaming_generation_skipped",
            }
        )
    decomposition = getattr(result, "decomposition", None)
    decomp_status = decomposition.get("status") if isinstance(decomposition, dict) else None
    if decomp_calls == 0 and isinstance(decomposition, dict):
        decomp_calls = int(decomposition.get("attempts", 0) or 0)
        decomp_repairs = int(decomposition.get("repair_attempts", 0) or 0)
    model_calls = {
        "generation": generation_calls,
        "decomposition": decomp_calls,
        "total": generation_calls + decomp_calls,
        "generation_repair_attempts": int(getattr(result, "generation_repair_attempts", 0) or 0),
        "decomposition_repair_attempts": decomp_repairs,
    }
    supported_answer = _supported_answer(case, answer)
    trace_types = [trace.event_type for trace in result.traces]
    return {
        "condition": outcome.condition,
        "run_status": result.run_status,
        "generation_status": generation_status,
        "final_query": final_query,
        "retrieval_backend": getattr(result, "retrieval_backend", outcome.backend),
        "retrieved_ids": retrieved_ids,
        "retrieved_hits": [_hit_record(hit) for hit in hits],
        "relevant_ids": expected_ids,
        "retrieval_metrics": _retrieval_metrics(expected_ids, retrieved_ids),
        "retrieval_quality_scored": bool(expected_ids)
        and result.run_status == "completed"
        and not retrieval_detail_errors
        and case.retrieval_behavior not in {"failure", "timeout"},
        "retrieval_quality_exclusion_reason": (
            None
            if bool(expected_ids)
            and result.run_status == "completed"
            and not retrieval_detail_errors
            and case.retrieval_behavior not in {"failure", "timeout"}
            else "no relevant labels"
            if not expected_ids
            else "declared fault or unsuccessful retrieval"
        ),
        "retrieval_details": retrieval_details,
        "decomposition": decomposition,
        "decomposition_status": decomp_status,
        "answer_text": answer.answer_text,
        "answer_uncertainty": answer.uncertainty,
        "intent_statuses": [
            status.model_dump(mode="json")
            for status in getattr(answer, "intent_statuses", [])
        ],
        "verification_audit": answer.verification_audit,
        "factual_claims": _claim_records(answer),
        "citation_ids": cited_ids,
        "citation_id_count": len(cited_ids),
        "citation_ids_valid": citation_valid,
        "unknown_citation_ids": unknown_citations,
        "semantic_claim_support": "not_evaluated",
        "supported_answer": supported_answer,
        "complete_request_evidence_coverage": {
            "pending": True,
        },
        "uncertainty_behavior": {
            "uncertainty_present": bool(answer.uncertainty.strip()),
            "unsupported_intents_marked": False,
            "no_factual_claims": not bool(answer.factual_claims),
            "status": "pending_intent_mapping",
        },
        "per_intent_retrieval": [],
        "early_retrieval_started": retrieval_started_early,
        "valid_evidence_ready_before_final": valid_evidence_ready,
        "useful_early_reuse": useful_reuse,
        "stale_result_discard_count": stale_discard,
        "stale_result_accepted_count": int(stale_accepted),
        "final_event_to_answer_latency_ms": final_latency,
        "full_interaction_duration_ms": full_duration,
        "retrieval_call_count": retrieval_calls,
        "model_calls": model_calls,
        "token_usage": {
            "retrieval": retrieval_usage.model_dump(mode="json"),
            "decomposition": decomp_usage.model_dump(mode="json"),
            "generation": initial_generation_usage.model_dump(mode="json"),
            "repair": repair_usage.model_dump(mode="json"),
            "verification": verification_usage.model_dump(mode="json"),
            "total": total_usage.model_dump(mode="json"),
        },
        "cost": getattr(result, "generation_cost", "unavailable"),
        "errors": errors,
        "trace": {
            "complete": trace_complete,
            "missing_events": missing_trace,
            "event_count": len(result.traces),
            "event_types": trace_types,
        },
    }


def _supported_answer(case: Phase3EvaluationCase, answer: Any) -> dict[str, Any]:
    answer_text = answer.answer_text.casefold()
    per_intent: list[dict[str, Any]] = []
    for intent in case.expected_intents:
        if not intent.answerable:
            continue
        missing = [
            expected
            for expected in intent.expected_answer_substrings
            if expected.casefold() not in answer_text
        ]
        per_intent.append(
            {
                "intent_key": intent.intent_key,
                "supported": not missing,
                "missing_answer_substrings": missing,
                "expected_answer_substrings": list(intent.expected_answer_substrings),
            }
        )
    denominator = len(per_intent)
    numerator = sum(item["supported"] for item in per_intent)
    return {
        "supported_intent_count": numerator,
        "answerable_intent_denominator": denominator,
        "coverage": numerator / denominator if denominator else None,
        "expected_answer_substrings_passed": all(item["supported"] for item in per_intent)
        if per_intent
        else all(
            expected.casefold() in answer_text
            for expected in case.expected_answer_substrings
        ),
        "per_intent": per_intent,
    }


def _constraint_match(expected_value: str, actual_values: Sequence[str]) -> bool:
    expected = _normalized_tokens(expected_value)
    if not expected:
        return True
    return any(
        expected <= _normalized_tokens(actual)
        or len(expected & _normalized_tokens(actual)) / len(expected) >= 0.5
        for actual in actual_values
    )


def _relationship_matches(expected: str, actual: str) -> bool:
    if expected == "single":
        return actual == "independent"
    return expected == actual


def _intent_match(case: Phase3EvaluationCase, plan_payload: Any) -> dict[str, Any]:
    if plan_payload is None:
        return {
            "status": "not_applicable",
            "gold_intent_count": len(case.expected_intents),
            "predicted_intent_count": None,
            "matched_count": None,
            "missed_intent_keys": [],
            "unnecessary_predicted_intent_ids": [],
            "exact_match": None,
            "pairs": [],
            "rubric": _matching_rubric(),
        }
    plan = MultiIntentPlan.model_validate(plan_payload)
    predicted = list(plan.intents)
    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for gold_index, gold in enumerate(case.expected_intents):
        for pred_index, pred in enumerate(predicted):
            source_overlap = max(
                _overlap(gold.source_hint, pred.source_text),
                _overlap(gold.source_hint, pred.query),
            )
            relation_match = _relationship_matches(gold.expected_relationship, pred.relationship)
            expected_values = [constraint.value for constraint in gold.expected_constraints]
            actual_values = [constraint.value for constraint in pred.constraints]
            constraints_matched = [
                value for value in expected_values if _constraint_match(value, actual_values)
            ]
            constraint_rate = (
                len(constraints_matched) / len(expected_values)
                if expected_values
                else 1.0
            )
            viable = source_overlap > 0.0 and relation_match
            score = source_overlap + (1.0 if relation_match else 0.0) + constraint_rate
            candidates.append(
                (
                    score if viable else -1.0,
                    gold_index,
                    pred_index,
                    {
                        "gold_intent_key": gold.intent_key,
                        "predicted_intent_id": pred.intent_id,
                        "predicted_query": pred.query,
                        "predicted_source_text": pred.source_text,
                        "expected_relationship": gold.expected_relationship,
                        "predicted_relationship": pred.relationship,
                        "relationship_match": relation_match,
                        "source_overlap": round(source_overlap, 6),
                        "expected_constraint_values": expected_values,
                        "matched_constraint_values": constraints_matched,
                        "constraint_coverage": constraint_rate,
                        "predicted_constraint_values": actual_values,
                        "matched": viable,
                    },
                )
            )
    pairs: list[dict[str, Any]] = []
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    for score, gold_index, pred_index, pair in sorted(candidates, reverse=True):
        if score < 0 or gold_index in used_gold or pred_index in used_pred:
            continue
        used_gold.add(gold_index)
        used_pred.add(pred_index)
        pairs.append(pair)
    pairs.sort(key=lambda pair: next(i for i, item in enumerate(case.expected_intents) if item.intent_key == pair["gold_intent_key"]))
    matched_gold_keys = {pair["gold_intent_key"] for pair in pairs}
    missed = [intent.intent_key for intent in case.expected_intents if intent.intent_key not in matched_gold_keys]
    extra = [pred.intent_id for index, pred in enumerate(predicted) if index not in used_pred]
    shared_expected = [constraint.value for constraint in case.shared_constraints]
    shared_actual = [constraint.value for constraint in plan.shared_constraints]
    shared_values_matched = [value for value in shared_expected if _constraint_match(value, shared_actual)]
    dependencies_ok = True
    for pair in pairs:
        gold = next(item for item in case.expected_intents if item.intent_key == pair["gold_intent_key"])
        if gold.expected_relationship == "dependent":
            dependencies_ok = dependencies_ok and any(
                dependency.dependent_intent_id == pair["predicted_intent_id"]
                and dependency.relation in {"depends_on", "refines"}
                for dependency in plan.dependencies
            )
    exact = (
        plan.status == "success"
        and len(predicted) == len(case.expected_intents)
        and not missed
        and not extra
        and all(pair["constraint_coverage"] == 1.0 for pair in pairs)
        and len(shared_values_matched) == len(shared_expected)
        and dependencies_ok
        and not (
            any(intent.expected_relationship == "single" for intent in case.expected_intents)
            and plan.boundary_count != 0
        )
    )
    return {
        "status": "assessed",
        "decomposition_status": plan.status,
        "gold_intent_count": len(case.expected_intents),
        "predicted_intent_count": len(predicted),
        "matched_count": len(pairs),
        "intent_recall": len(pairs) / len(case.expected_intents) if case.expected_intents else None,
        "intent_precision": len(pairs) / len(predicted) if predicted else 0.0,
        "missed_intent_keys": missed,
        "unnecessary_predicted_intent_ids": extra,
        "shared_constraints": {
            "expected": shared_expected,
            "predicted": shared_actual,
            "matched": shared_values_matched,
            "coverage": len(shared_values_matched) / len(shared_expected) if shared_expected else 1.0,
        },
        "dependency_relationships_ok": dependencies_ok,
        "exact_match": exact,
        "pairs": pairs,
        "predicted_intent_ids": [intent.intent_id for intent in predicted],
        "rubric": _matching_rubric(),
    }


def _matching_rubric() -> dict[str, Any]:
    return {
        "unit": "one external expected_intent record",
        "one_to_one_matching": True,
        "candidate_requirements": [
            "source_hint shares at least one normalized content token with predicted source_text or query",
            "relationship matches; expected 'single' maps to exactly one independent intent",
        ],
        "constraint_score": "all expected intent-local values must match a predicted constraint; all shared values must match plan.shared_constraints",
        "dependent_score": "dependent expected intents also require a depends_on/refines edge to the matched predicted intent",
        "exact_case_match": "predicted count equals gold count, every gold intent is matched, no predicted intent is extra, constraints/dependencies pass, plan status is success, and a single case has boundary_count=0",
        "multi_intent_target_denominator": "cases with two or more external expected intents",
        "extra_intent_definition": "a predicted intent left unmatched after one-to-one matching",
    }


def _per_intent_retrieval(
    case: Phase3EvaluationCase,
    record: dict[str, Any],
    match: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    details = record.get("retrieval_details")
    if not isinstance(details, dict) or match.get("status") != "assessed":
        denominator = sum(intent.answerable for intent in case.expected_intents)
        return (
            [],
            {
                "covered_intent_count": 0,
                "answerable_intent_denominator": denominator,
                "coverage": 0.0 if denominator else None,
                "all_answerable_intents_covered": False if denominator else True,
                "all_labeled_relevant_ids_in_final_evidence": False if denominator else True,
            },
        )
    result_by_id = {
        item.get("intent", {}).get("intent_id"): item
        for item in details.get("intent_results", [])
        if isinstance(item, dict)
    }
    pair_by_gold = {pair["gold_intent_key"]: pair for pair in match.get("pairs", [])}
    final_ids = set(record.get("retrieved_ids", []))
    per_intent: list[dict[str, Any]] = []
    covered_count = 0
    all_ids_present = True
    for gold in case.expected_intents:
        if not gold.answerable:
            continue
        pair = pair_by_gold.get(gold.intent_key)
        predicted_id = pair.get("predicted_intent_id") if pair else None
        item = result_by_id.get(predicted_id) if predicted_id else None
        intent_hits = [
            hit.get("chunk_id")
            for hit in (item or {}).get("hits", [])
            if isinstance(hit, dict) and hit.get("chunk_id")
        ]
        final_relevant = set(gold.relevant_chunk_ids) & final_ids
        intent_recall = _retrieval_metrics(gold.relevant_chunk_ids, intent_hits)
        covered = bool(final_relevant)
        all_gold_ids_present = set(gold.relevant_chunk_ids) <= final_ids
        covered_count += int(covered)
        all_ids_present = all_ids_present and all_gold_ids_present
        per_intent.append(
            {
                "intent_key": gold.intent_key,
                "predicted_intent_id": predicted_id,
                "retrieval_status": (item or {}).get("status", "missing_predicted_intent"),
                "intent_retrieved_ids": intent_hits,
                "final_evidence_ids": sorted(final_relevant),
                "relevant_ids": list(gold.relevant_chunk_ids),
                "recall_at_k": intent_recall,
                "represented_in_final_evidence": covered,
                "all_labeled_relevant_ids_in_final_evidence": all_gold_ids_present,
                "backend_rankings": (item or {}).get("backend_hits", {}),
                "fusion_decisions": (item or {}).get("fusion_decisions", []),
                "evidence_dependencies": {
                    "intent_ids": (item or {}).get("dependency_intent_ids", []),
                    "chunk_ids": (item or {}).get("dependency_context_chunk_ids", []),
                },
            }
        )
    denominator = len(per_intent)
    return (
        per_intent,
        {
            "covered_intent_count": covered_count,
            "answerable_intent_denominator": denominator,
            "coverage": covered_count / denominator if denominator else None,
            "all_answerable_intents_covered": covered_count == denominator if denominator else True,
            "all_labeled_relevant_ids_in_final_evidence": all_ids_present if denominator else True,
        },
    )


def _uncertainty_behavior(case: Phase3EvaluationCase, record: dict[str, Any]) -> dict[str, Any]:
    answer_text = str(record.get("answer_text", "")).casefold()
    uncertainty_markers = (
        "uncertain",
        "insufficient",
        "unsupported",
        "unavailable",
        "not available",
        "cannot",
        "no verified evidence",
        "evidence was unavailable",
    )
    uncertainty = any(marker in answer_text for marker in uncertainty_markers)
    statuses = [
        status
        for status in record.get("intent_statuses", [])
        if isinstance(status, dict)
    ]
    unsupported_statuses = [
        status
        for status in statuses
        if status.get("status") in {"insufficient_evidence", "needs_clarification"}
    ]
    unsupported_intent_ids = {
        status.get("intent_id")
        for status in unsupported_statuses
        if status.get("intent_id")
    }
    no_claim_for_unsupported = not any(
        unsupported_intent_ids.intersection(
            set(claim.get("intent_ids", []))
            or ({claim.get("intent_id")} if claim.get("intent_id") else set())
        )
        for claim in record.get("factual_claims", [])
        if isinstance(claim, dict)
    )
    if case.answerability == "partially_answerable":
        supported = record.get("supported_answer", {}).get("supported_intent_count", 0) > 0
        return {
            "uncertainty_present": uncertainty,
            "unsupported_intents_marked": bool(unsupported_statuses),
            "no_factual_claim_for_unsupported_intent": no_claim_for_unsupported,
            "supported_portion_present": supported,
            "status": "pass" if supported and uncertainty and unsupported_statuses and no_claim_for_unsupported else "fail",
            "semantic_fabrication_check": "not_evaluated",
        }
    if case.answerability == "unanswerable":
        no_claims = not record.get("factual_claims")
        return {
            "uncertainty_present": uncertainty,
            "unsupported_intents_marked": bool(unsupported_statuses) or uncertainty,
            "no_factual_claim_for_unsupported_intent": no_claims,
            "supported_portion_present": False,
            "status": "pass" if no_claims and uncertainty and record.get("run_status") in {"abstained", "completed"} else "fail",
            "semantic_fabrication_check": "not_evaluated",
        }
    return {
        "uncertainty_present": uncertainty,
        "unsupported_intents_marked": True,
        "no_factual_claim_for_unsupported_intent": True,
        "supported_portion_present": record.get("supported_answer", {}).get("supported_intent_count", 0) > 0,
        "status": "not_applicable",
        "semantic_fabrication_check": "not_evaluated",
    }


def _enrich_case_result(
    case: Phase3EvaluationCase,
    condition_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    c_record = condition_records["C_phase3_streaming_fused"]
    intent_match = _intent_match(case, c_record.get("decomposition"))
    per_intent, coverage = _per_intent_retrieval(case, c_record, intent_match)
    c_record["per_intent_retrieval"] = per_intent
    c_record["complete_request_evidence_coverage"] = coverage
    for record in condition_records.values():
        # A and B have no Phase 3 plan; retain their parent-query evidence and
        # make the non-applicability explicit rather than manufacturing intent IDs.
        if record is not c_record:
            record["per_intent_retrieval"] = [
                {
                    "status": "not_applicable_single_query_condition",
                    "relevant_ids_union": _expected_union(case),
                    "retrieved_ids": list(record.get("retrieved_ids", [])),
                }
            ]
            answerable = sum(intent.answerable for intent in case.expected_intents)
            present = len(set(_expected_union(case)) & set(record.get("retrieved_ids", []))) > 0
            record["complete_request_evidence_coverage"] = {
                "covered_intent_count": int(present) if answerable else 0,
                "answerable_intent_denominator": answerable,
                "coverage": (1.0 if present else 0.0) if answerable else None,
                "all_answerable_intents_covered": present if answerable else True,
                "all_labeled_relevant_ids_in_final_evidence": set(_expected_union(case)) <= set(record.get("retrieved_ids", [])) if answerable else True,
            }
        record["uncertainty_behavior"] = _uncertainty_behavior(case, record)
    return {
        "case_id": case.case_id,
        "split": case.split,
        "scenario_group": case.scenario_group,
        "variant_family": case.variant_family,
        "evaluation_role": case.evaluation_role,
        "asset_status": case.asset_status,
        "answerability": case.answerability,
        "label_review_status": case.label_review_status.model_dump(mode="json"),
        "implementation_change_ids": list(case.implementation_change_ids),
        "gold": {
            "expected_intents": [intent.model_dump(mode="json") for intent in case.expected_intents],
            "shared_constraints": [constraint.model_dump(mode="json") for constraint in case.shared_constraints],
            "expected_answer_substrings": list(case.expected_answer_substrings),
            "retrieval_behavior": case.retrieval_behavior,
            "provider_failure_stage": case.provider_failure_stage,
        },
        "intent_matching": intent_match,
        "conditions": condition_records,
        "correction_audit": {
            "requires_early_retrieval": case.require_early_retrieval,
            "has_non_final_event": any(not event.is_final for event in case.transcript),
            "final_text": case.transcript[-1].text,
            "streaming_stale_result_accepted_count": sum(
                condition_records[name].get("stale_result_accepted_count", 0)
                for name in ("B_phase2_streaming_single_query", "C_phase3_streaming_fused")
            ),
        },
    }


def _cost_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    costs = [record.get("cost", "unavailable") for record in records]
    numeric = [value for value in costs if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {
        "value": float(sum(numeric)) if len(numeric) == len(costs) and costs else "unavailable",
        "available": bool(costs) and len(numeric) == len(costs),
        "pricing_note": "cost is unavailable unless every run reports numeric usage and configured prices",
    }


def _resource_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def stage_total(stage: str) -> Usage:
        return _usage_sum([_usage(record.get("token_usage", {}).get(stage)) for record in records])

    errors = [error for record in records for error in record.get("errors", [])]
    known_error_types = (
        "MultiIntentRetrievalError",
        "RetrievalTimeout",
        "RetrievalError",
        "GenerationCallFailed",
        "DecompositionProviderError",
        "GenerationProviderError",
    )

    def error_type(error: str) -> str:
        return next(
            (candidate for candidate in known_error_types if candidate in error),
            error.split(":", 1)[0],
        )

    return {
        "retrieval_calls": sum(record.get("retrieval_call_count", 0) for record in records),
        "model_calls": {
            "generation": sum(record.get("model_calls", {}).get("generation", 0) for record in records),
            "decomposition": sum(record.get("model_calls", {}).get("decomposition", 0) for record in records),
            "total": sum(record.get("model_calls", {}).get("total", 0) for record in records),
        },
        "tokens": {
            stage: stage_total(stage).model_dump(mode="json")
            for stage in ("retrieval", "decomposition", "generation", "repair", "verification", "total")
        },
        "error_count": len(errors),
        "error_case_count": sum(bool(record.get("errors")) for record in records),
        "errors_by_type": dict(Counter(error_type(error) for error in errors)),
        "cost": _cost_summary(records),
    }


def _quality_summary(
    rows: Sequence[dict[str, Any]],
    *,
    condition: str,
    matched_case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected = [
        row["conditions"][condition]
        for row in rows
        if matched_case_ids is None or row["case_id"] in matched_case_ids
    ]
    selected = [record for record in selected if record.get("retrieval_quality_scored")]
    values_by_k = {
        f"recall_at_{k}": [record.get("retrieval_metrics", {}).get(f"recall_at_{k}", 0.0) for record in selected]
        for k in (1, 3, 5)
    }
    denominators = [len(record.get("relevant_ids", [])) for record in selected]
    numerators = {
        key: sum(
            int(round(record.get("retrieval_metrics", {}).get(key, 0.0) * len(record.get("relevant_ids", []))))
            for record in selected
        )
        for key in values_by_k
    }
    return {
        "case_denominator": len(selected),
        "relevant_id_denominator": sum(denominators),
        "included_case_ids": [
            row["case_id"]
            for row in rows
            if (matched_case_ids is None or row["case_id"] in matched_case_ids)
            and row["conditions"][condition].get("retrieval_quality_scored")
        ],
        "recall_at_1_macro_mean": _mean(values_by_k["recall_at_1"]),
        "recall_at_3_macro_mean": _mean(values_by_k["recall_at_3"]),
        "recall_at_5_macro_mean": _mean(values_by_k["recall_at_5"]),
        "recall_at_1_micro": numerators["recall_at_1"] / sum(denominators) if denominators else None,
        "recall_at_3_micro": numerators["recall_at_3"] / sum(denominators) if denominators else None,
        "recall_at_5_micro": numerators["recall_at_5"] / sum(denominators) if denominators else None,
    }


def _per_intent_retrieval_summary(
    rows: Sequence[dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    """Aggregate Phase 3 intent-local rankings without scoring semantics."""

    if condition != "C_phase3_streaming_fused":
        return {
            "status": "NOT_APPLICABLE",
            "reason": "A and B intentionally retrieve one parent query and have no intent-local rankings",
        }
    entries = [
        entry
        for row in rows
        for entry in row["conditions"][condition].get("per_intent_retrieval", [])
        if isinstance(entry.get("recall_at_k"), dict)
    ]
    represented = sum(bool(entry.get("represented_in_final_evidence")) for entry in entries)
    all_ids = sum(bool(entry.get("all_labeled_relevant_ids_in_final_evidence")) for entry in entries)
    answerable_expected = sum(
        intent.get("answerable", False)
        for row in rows
        for intent in row["gold"]["expected_intents"]
    )
    return {
        "status": "measured" if entries else "NOT VERIFIED",
        "intent_denominator": len(entries),
        "answerable_intent_label_count": answerable_expected,
        "excluded_intent_record_count": max(answerable_expected - len(entries), 0),
        "intent_recall_at_1_macro_mean": _mean(
            [entry["recall_at_k"].get("recall_at_1", 0.0) for entry in entries]
        ),
        "intent_recall_at_3_macro_mean": _mean(
            [entry["recall_at_k"].get("recall_at_3", 0.0) for entry in entries]
        ),
        "intent_recall_at_5_macro_mean": _mean(
            [entry["recall_at_k"].get("recall_at_5", 0.0) for entry in entries]
        ),
        "represented_in_final_evidence_count": represented,
        "represented_in_final_evidence_rate": represented / len(entries) if entries else None,
        "all_labeled_relevant_ids_in_final_evidence_count": all_ids,
        "all_labeled_relevant_ids_in_final_evidence_rate": all_ids / len(entries) if entries else None,
    }


def _condition_summary(rows: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    records = [row["conditions"][condition] for row in rows]
    answerable_records = [
        (row, row["conditions"][condition])
        for row in rows
        if row["answerability"] in {"answerable", "partially_answerable"}
    ]
    supported_numerator = sum(
        record.get("supported_answer", {}).get("supported_intent_count", 0)
        for _, record in answerable_records
    )
    supported_denominator = sum(
        record.get("supported_answer", {}).get("answerable_intent_denominator", 0)
        for _, record in answerable_records
    )
    latencies = [
        record["final_event_to_answer_latency_ms"]
        for record in records
        if isinstance(record.get("final_event_to_answer_latency_ms"), (int, float))
    ]
    # Eligibility is read from the case metadata in the dedicated result.  It
    # is not inferred from whether the controller chose to retrieve.
    eligible_ids = [
        row["case_id"]
        for row in rows
        if row.get("correction_audit", {}).get("requires_early_retrieval")
    ]
    early_records = [row["conditions"][condition] for row in rows if row["case_id"] in eligible_ids]
    stale_accepted = sum(record.get("stale_result_accepted_count", 0) for record in records)
    complete_coverage = [
        record.get("complete_request_evidence_coverage", {}).get("coverage")
        for record in records
        if isinstance(record.get("complete_request_evidence_coverage", {}).get("coverage"), (int, float))
    ]
    trace_count = sum(record.get("trace", {}).get("complete", False) for record in records)
    execution_modes = {
        record.get("execution_mode")
        for record in records
        if record.get("execution_mode")
    }
    return {
        "case_denominator": len(records),
        "run_status_counts": dict(Counter(record.get("run_status") for record in records)),
        "completed_case_count": sum(record.get("run_status") == "completed" for record in records),
        "failed_case_ids": [row["case_id"] for row in rows if row["conditions"][condition].get("run_status") == "failed"],
        "abstained_case_ids": [row["case_id"] for row in rows if row["conditions"][condition].get("run_status") == "abstained"],
        "retrieval_quality": _quality_summary(rows, condition=condition),
        "per_intent_retrieval": _per_intent_retrieval_summary(rows, condition),
        "supported_answer_coverage": {
            "supported_intent_count": supported_numerator,
            "answerable_intent_denominator": supported_denominator,
            "coverage": supported_numerator / supported_denominator if supported_denominator else None,
            "interpretation": "Only answerable intent units enter this denominator; unanswerable cases cannot score by abstaining.",
        },
        "complete_request_evidence_coverage": {
            "case_count_with_applicable_coverage": len(complete_coverage),
            "macro_mean": _mean(complete_coverage),
            "fully_covered_case_count": sum(value == 1.0 for value in complete_coverage),
            "fully_covered_case_denominator": len(complete_coverage),
        },
        "citation_id_validity": {
            "case_denominator": len(records),
            "citation_run_denominator": sum(
                record.get("citation_id_count", 0) > 0 for record in records
            ),
            "valid_case_count": sum(record.get("citation_ids_valid") is True for record in records),
            "valid_case_rate": (
                sum(record.get("citation_ids_valid") is True for record in records)
                / sum(record.get("citation_id_count", 0) > 0 for record in records)
                if any(record.get("citation_id_count", 0) > 0 for record in records)
                else None
            ),
            "no_citation_case_count": sum(
                record.get("citation_id_count", 0) == 0 for record in records
            ),
            "citation_id_sample_count": sum(record.get("citation_id_count", 0) for record in records),
            "invalid_citation_id_count": sum(len(record.get("unknown_citation_ids", [])) for record in records),
            "semantic_support_status": "NOT VERIFIED",
            "interpretation": "IDs were checked against known/supplied chunks only; this is not citation support or entailment.",
        },
        "latency_final_event_to_answer_ms": {
            "sample_count": len(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "execution_mode": next(iter(execution_modes), "not_embedded"),
        },
        "early_retrieval": {
            "eligible_case_ids": eligible_ids,
            "eligible_case_count": len(eligible_ids),
            "actual_early_start_count": sum(record.get("early_retrieval_started", False) for record in early_records),
            "actual_early_start_rate": sum(record.get("early_retrieval_started", False) for record in early_records) / len(early_records) if early_records else None,
            "useful_early_reuse_count": sum(record.get("useful_early_reuse", False) for record in early_records),
            "useful_early_reuse_rate": sum(record.get("useful_early_reuse", False) for record in early_records) / len(early_records) if early_records else None,
            "stale_result_discard_count": sum(record.get("stale_result_discard_count", 0) for record in early_records),
            "stale_result_accepted_count": stale_accepted,
        },
        "resources": _resource_summary(records),
        "trace_completeness": {
            "complete_case_count": trace_count,
            "case_denominator": len(records),
            "rate": trace_count / len(records) if records else None,
            "incomplete_case_ids": [row["case_id"] for row in rows if not row["conditions"][condition].get("trace", {}).get("complete", False)],
        },
        "quality_scoring_note": "Retrieval quality excludes declared failure/timeout runs and uses only external provisional relevant-ID labels.",
    }


def _phase3_split_summary(rows: Sequence[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    condition_summary = {
        condition: _condition_summary(selected, condition)
        for condition in _CONDITIONS
    }
    compounds = [row for row in selected if len(row["gold"]["expected_intents"]) > 1]
    exact_compounds = [row for row in compounds if row["intent_matching"].get("exact_match") is True]
    missed = [
        item
        for row in selected
        for item in row["intent_matching"].get("missed_intent_keys", [])
    ]
    extra = [
        item
        for row in selected
        for item in row["intent_matching"].get("unnecessary_predicted_intent_ids", [])
    ]
    single_extra_cases = [
        row["case_id"]
        for row in selected
        if len(row["gold"]["expected_intents"]) == 1
        and row["intent_matching"].get("predicted_intent_count") is not None
        and row["intent_matching"].get("predicted_intent_count") != 1
    ]
    # A/B/C retrieval quality is a matched triple, not three independently
    # selected denominators.
    triple_ids = [
        row["case_id"]
        for row in selected
        if all(
            row["conditions"][condition].get("retrieval_quality_scored")
            for condition in _CONDITIONS
        )
    ]
    matched_quality = {
        condition: _quality_summary(selected, condition=condition, matched_case_ids=triple_ids)
        for condition in _CONDITIONS
    }
    return {
        "split": split,
        "case_count": len(selected),
        "compound_case_count": len(compounds),
        "single_case_count": len(selected) - len(compounds),
        "multi_intent_identification": {
            "rubric": _matching_rubric(),
            "compound_case_denominator": len(compounds),
            "correct_multi_intent_case_count": len(exact_compounds),
            "identification_rate": len(exact_compounds) / len(compounds) if compounds else None,
            "correct_case_ids": [row["case_id"] for row in exact_compounds],
            "missed_intent_count": len(missed),
            "missed_intent_keys": missed,
            "unnecessary_extra_intent_count": len(extra),
            "unnecessary_extra_intent_ids": extra,
            "single_case_extra_intent_case_ids": single_extra_cases,
            "target_70_percent": {
                "target": 0.70,
                "status": "PASS" if compounds and len(exact_compounds) / len(compounds) >= 0.70 else "FAIL" if compounds else "NOT VERIFIED",
                "denominator": len(compounds),
            },
        },
        "matched_comparison_retrieval_quality": {
            "matched_case_ids": triple_ids,
            "case_denominator": len(triple_ids),
            "A_original_baseline": matched_quality["A_original_baseline"],
            "B_phase2_streaming_single_query": matched_quality["B_phase2_streaming_single_query"],
            "C_phase3_streaming_fused": matched_quality["C_phase3_streaming_fused"],
            "interpretation": "All three conditions share this successful retrieval denominator; operational failures remain in every-case summaries.",
        },
        "conditions": condition_summary,
        "end_to_end_outcomes": {
            condition: {
                "case_denominator": len(selected),
                "expected_fault_case_ids": [
                    row["case_id"] for row in selected
                    if row["gold"].get("retrieval_behavior") != "normal"
                    or row["gold"].get("provider_failure_stage") != "none"
                ],
                "non_completed_case_ids": [
                    row["case_id"] for row in selected
                    if row["conditions"][condition].get("run_status") != "completed"
                ],
                "run_status_counts": condition_summary[condition]["run_status_counts"],
            }
            for condition in _CONDITIONS
        },
        "case_results": selected,
    }


async def _retrieval_ablation(
    cases: Sequence[Phase3EvaluationCase],
    *,
    corpus: CorpusIndex,
    settings: Settings,
    backend: str,
    retrieval_mode: str | None = None,
    top_k: int,
) -> dict[str, Any]:
    """Measure single-query vs decomposed lexical retrieval directly.

    This is deliberately an engineering ablation over the local fixture.  It
    is not a model-quality comparison: both sides use the same lexical index,
    and the decomposed side uses the explicitly labelled offline rule-based
    planner to isolate retrieval/evidence behavior.
    """

    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.retrieval_behavior != "normal" or case.provider_failure_stage != "none":
            continue
        query = case.transcript[-1].text
        expected_ids = _expected_union(case)
        if not expected_ids:
            continue
        base = make_retriever(
            corpus,
            backend=backend,
            top_k=top_k,
            cache_dir=settings.embedding_cache_dir,
            local_files_only=settings.embedding_local_files_only,
        )
        single = base.search(query)
        decomposed = MultiIntentRetriever(
            base,
            top_k=top_k,
            max_workers=settings.multi_intent_max_workers,
            rrf_k=settings.multi_intent_rrf_k,
            context_budget_tokens=settings.multi_intent_context_budget_tokens,
        )
        fused = decomposed.search(query)
        details = decomposed.last_result.model_dump(mode="json") if decomposed.last_result else None
        rows.append(
            {
                "case_id": case.case_id,
                "expected_ids": expected_ids,
                "single_query_ids": [hit.chunk_id for hit in single],
                "decomposed_ids": [hit.chunk_id for hit in fused],
                "single_query_recall_at_5": _retrieval_metrics(expected_ids, [hit.chunk_id for hit in single]).get("recall_at_5"),
                "decomposed_recall_at_5": _retrieval_metrics(expected_ids, [hit.chunk_id for hit in fused]).get("recall_at_5"),
                "decomposed_intent_count": len(decomposed.last_decomposition.intents) if decomposed.last_decomposition else None,
                "decomposed_missing_intent_ids": details.get("missing_intent_ids", []) if details else [],
                "decomposed_complete_evidence": not bool(details and details.get("missing_intent_ids")),
            }
        )
    single_values = [row["single_query_recall_at_5"] for row in rows]
    decomposed_values = [row["decomposed_recall_at_5"] for row in rows]
    return {
        "status": "measured" if rows else "NOT VERIFIED",
        "asset_status": "synthetic_fixture",
        "label_status": "provisional_generated",
        "backend": backend,
        "top_k": top_k,
        "case_denominator": len(rows),
        "single_query_recall_at_5_macro_mean": _mean(single_values),
        "decomposed_recall_at_5_macro_mean": _mean(decomposed_values),
        "single_query_vs_decomposed_delta": (
            _mean(decomposed_values) - _mean(single_values)
            if single_values and decomposed_values
            else None
        ),
        "complete_decomposed_evidence_case_count": sum(row["decomposed_complete_evidence"] for row in rows),
        "case_results": rows,
        "interpretation": "Measured lexical retrieval behavior only; no mock/model-quality finding is claimed.",
    }


def _dense_hybrid_ablation(corpus: CorpusIndex) -> dict[str, Any]:
    if importlib.util.find_spec("sentence_transformers") is None:
        return {
            "status": "NOT VERIFIED",
            "dense_only": "NOT VERIFIED",
            "hybrid": "NOT VERIFIED",
            "reason": "sentence-transformers is not installed; no dense index/model run is available",
            "mock_comparison_run": False,
        }
    if corpus.manifest.embedding.backend != "dense":
        return {
            "status": "NOT VERIFIED",
            "dense_only": "NOT VERIFIED",
            "hybrid": "NOT VERIFIED",
            "reason": "the supplied index is lexical-only and contains no dense embeddings; a dense build/model run was not performed",
            "mock_comparison_run": False,
        }
    return {
        "status": "NOT VERIFIED",
        "dense_only": "NOT VERIFIED",
        "hybrid": "NOT VERIFIED",
        "reason": "no completed real dense-only and hybrid paired run was recorded",
        "mock_comparison_run": False,
    }


def _label_status_summary(cases: Sequence[Phase3EvaluationCase]) -> dict[str, Any]:
    dimensions = ("intent_labels", "relevance_labels", "answer_expectations")
    summary: dict[str, Any] = {}
    for dimension in dimensions:
        counts = Counter(getattr(case.label_review_status, dimension) for case in cases)
        summary[dimension] = {
            status: counts.get(status, 0) for status in _REVIEW_STATUSES
        }
    overall = Counter(
        status
        for case in cases
        for status in (
            case.label_review_status.intent_labels,
            case.label_review_status.relevance_labels,
            case.label_review_status.answer_expectations,
        )
    )
    summary["overall"] = {status: overall.get(status, 0) for status in _REVIEW_STATUSES}
    summary["semantic_claim_support_review"] = {
        "status": "not_evaluated",
        "model_reviewed_count": 0,
        "human_reviewed_count": 0,
    }
    return summary


def _data_integrity(cases: Sequence[Phase3EvaluationCase]) -> dict[str, Any]:
    family_splits: dict[str, set[str]] = {}
    for case in cases:
        family_splits.setdefault(case.variant_family, set()).add(case.split)
    cross_split = {
        family: sorted(splits)
        for family, splits in family_splits.items()
        if len(splits) > 1
    }
    return {
        "case_count": len(cases),
        "development_case_count": sum(case.split == "development" for case in cases),
        "held_out_case_count": sum(case.split == "held_out" for case in cases),
        "unique_case_ids": len({case.case_id for case in cases}) == len(cases),
        "variant_families_crossing_splits": cross_split,
        "related_variants_kept_in_same_split": not cross_split,
        "scenario_group_counts": dict(Counter(case.scenario_group for case in cases)),
        "evaluation_role_counts": dict(Counter(case.evaluation_role for case in cases)),
        "case_ids_by_evaluation_role": {
            role: [case.case_id for case in cases if case.evaluation_role == role]
            for role in (
                "historical_baseline",
                "diagnostic_regression",
                "untouched_generalization",
            )
        },
        "answerability_counts": dict(Counter(case.answerability for case in cases)),
        "retrieval_behavior_counts": dict(Counter(case.retrieval_behavior for case in cases)),
        "provider_failure_stage_counts": dict(Counter(case.provider_failure_stage for case in cases)),
        "case_ids_by_split": {
            split: [case.case_id for case in cases if case.split == split]
            for split in ("development", "held_out")
        },
    }


def _implementation_change_audit(
    cases: Sequence[Phase3EvaluationCase],
    change_log: dict[str, Any],
) -> dict[str, Any]:
    case_ids = {case.case_id for case in cases}
    referenced_by_case = {
        case.case_id: list(case.implementation_change_ids)
        for case in cases
        if case.implementation_change_ids
    }
    changes = []
    for change in change_log.get("changes", []):
        if not isinstance(change, dict):
            continue
        informed = [case_id for case_id in change.get("informed_by_case_ids", []) if case_id in case_ids]
        changes.append(
            {
                **change,
                "case_ids_present_in_selected_suite": informed,
                "referenced_by_case_labels": [
                    case_id
                    for case_id, ids in referenced_by_case.items()
                    if change.get("change_id") in ids
                ],
            }
        )
    return {
        "source_status": change_log.get("status", "unknown"),
        "changes": changes,
        "case_labels_informing_changes": referenced_by_case,
    }


def _provider_failure_audit(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for row in rows:
        fault = row["gold"].get("provider_failure_stage")
        retrieval_fault = row["gold"].get("retrieval_behavior")
        if fault == "none" and retrieval_fault == "normal":
            continue
        arms: dict[str, Any] = {}
        for condition in _CONDITIONS:
            record = row["conditions"][condition]
            arms[condition] = {
                "run_status": record.get("run_status"),
                "generation_status": record.get("generation_status"),
                "decomposition_status": record.get("decomposition_status"),
                "errors": record.get("errors", []),
                "retrieval_call_count": record.get("retrieval_call_count", 0),
                "citation_ids_valid": record.get("citation_ids_valid"),
                "explicit_failure_observed": bool(record.get("errors")) or record.get("decomposition_status") == "fallback",
            }
        audits.append(
            {
                "case_id": row["case_id"],
                "provider_failure_stage": fault,
                "retrieval_behavior": retrieval_fault,
                "arms": arms,
                "silent_mock_or_backend_substitution": False,
                "interpretation": "Faults are fixture probes; failures are not quality successes and decomposition fallback is explicitly labelled.",
            }
        )
    return audits


def _capability_status(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    all_rows = [
        row
        for split in ("development", "held_out")
        for row in report["splits"][split]["case_results"]
    ]
    def cases_exact(groups: set[str] | None = None) -> tuple[int, int]:
        chosen = [row for row in all_rows if groups is None or row["scenario_group"] in groups]
        return sum(row["intent_matching"].get("exact_match") is True for row in chosen), len(chosen)

    def status_from_rate(numerator: int, denominator: int) -> str:
        return "PASS" if denominator and numerator == denominator else "FAIL" if denominator else "NOT VERIFIED"

    capability: dict[str, dict[str, Any]] = {}
    for label, groups in {
        "single_questions_not_decomposed": {"single_question_constraints", "single_catering_paraphrase", "unsupported_request", "unsupported_insurance_request"},
        "compound_independent_questions": {"independent_compound", "independent_capacity_catering"},
        "dependent_questions": {"dependent_question", "dependent_dietary"},
        "shared_constraints": {"shared_scope"},
        "negation": {"negation"},
        "comparisons": {"comparison", "conflicting_evidence_precedence"},
        "partial_answerability": {"partial_answer"},
        "entity_confusion": {"entity_confusion"},
    }.items():
        numerator, denominator = cases_exact(groups)
        capability[label] = {
            "status": status_from_rate(numerator, denominator),
            "local_case_numerator": numerator,
            "local_case_denominator": denominator,
            "evidence_basis": "synthetic_fixture_provisional_labels",
        }
    partial_rows = [
        row for row in all_rows if row["answerability"] == "partially_answerable"
    ]
    partial_c_records = [
        row["conditions"]["C_phase3_streaming_fused"] for row in partial_rows
    ]
    capability["uncertainty_on_partial_requests"] = {
        "status": (
            "PASS"
            if partial_c_records
            and all(record["uncertainty_behavior"]["status"] == "pass" for record in partial_c_records)
            else "FAIL"
            if partial_c_records
            else "NOT VERIFIED"
        ),
        "case_denominator": len(partial_c_records),
        "passing_case_count": sum(
            record["uncertainty_behavior"]["status"] == "pass"
            for record in partial_c_records
        ),
        "scope": "Phase 3 C answer uncertainty behavior; semantic fabrication remains unreviewed",
    }
    unanswerable_rows = [
        row for row in all_rows if row["answerability"] == "unanswerable"
    ]
    unanswerable_c_records = [
        row["conditions"]["C_phase3_streaming_fused"] for row in unanswerable_rows
    ]
    capability["uncertainty_on_unanswerable_requests"] = {
        "status": (
            "PASS"
            if unanswerable_c_records
            and all(record["uncertainty_behavior"]["status"] == "pass" for record in unanswerable_c_records)
            else "FAIL"
            if unanswerable_c_records
            else "NOT VERIFIED"
        ),
        "case_denominator": len(unanswerable_c_records),
        "passing_case_count": sum(
            record["uncertainty_behavior"]["status"] == "pass"
            for record in unanswerable_c_records
        ),
        "scope": "Phase 3 C answer uncertainty behavior; semantic fabrication remains unreviewed",
    }
    conflict_rows = [
        row
        for row in all_rows
        if row["scenario_group"] == "conflicting_evidence_precedence"
    ]
    conflict_c_records = [
        row["conditions"]["C_phase3_streaming_fused"] for row in conflict_rows
    ]
    capability["conflicting_evidence_handling"] = {
        "status": (
            "PASS"
            if conflict_c_records
            and all(
                record.get("intent_statuses")
                and all(item.get("reason") for item in record["intent_statuses"])
                for record in conflict_c_records
            )
            else "FAIL"
            if conflict_c_records
            else "NOT VERIFIED"
        ),
        "case_denominator": len(conflict_c_records),
        "scope": "structured conflict/precedence status and provenance; semantic claim support is NOT VERIFIED",
    }
    correction_rows = [row for row in all_rows if row["gold"].get("retrieval_behavior") == "delay"]
    stale_acceptance = sum(
        row["conditions"][condition].get("stale_result_accepted_count", 0)
        for row in correction_rows
        for condition in ("B_phase2_streaming_single_query", "C_phase3_streaming_fused")
    )
    capability["corrections_and_stale_protection"] = {
        "status": "PASS" if correction_rows and stale_acceptance == 0 else "FAIL" if correction_rows else "NOT VERIFIED",
        "correction_case_count": len(correction_rows),
        "stale_result_accepted_count": stale_acceptance,
        "engineering_regression_status": "covered separately by Phase 2 tests",
    }
    fault_rows = _provider_failure_audit(all_rows)
    capability["provider_and_retrieval_failures"] = {
        "status": "PASS" if fault_rows and all(item["arms"] for item in fault_rows) else "NOT VERIFIED",
        "fault_case_count": len(fault_rows),
        "silent_substitution_observed": any(item["silent_mock_or_backend_substitution"] for item in fault_rows),
    }
    citation = report["citation_id_validity"]
    capability["citation_id_validity"] = {
        "status": "PASS" if citation["invalid_id_count"] == 0 else "FAIL",
        "invalid_id_count": citation["invalid_id_count"],
        "case_validity_rate": citation["case_validity_rate"],
        "scope": "deterministic provenance only",
    }
    capability["semantic_claim_support"] = {
        "status": "NOT VERIFIED",
        "reason": "no human or governed semantic judge review was run; existing citation IDs do not establish support",
    }
    capability["supported_answer_coverage"] = {
        "status": "PASS" if report["supported_answer_coverage"]["supported_intent_count"] > 0 else "FAIL",
        **report["supported_answer_coverage"],
        "scope": "local substring coverage, not semantic entailment",
    }
    target = report["targets"]["multi_intent_identification_70_percent"]
    capability["guide_multi_intent_70_percent_target"] = {
        "status": target["status"],
        "target": target["target"],
        "development": target["development"],
        "held_out": target["held_out"],
        "scope": "local provisional-label measurement; not official benchmark validation",
    }
    capability["guide_citation_support_85_percent_target"] = {
        "status": "NOT VERIFIED",
        "target": 0.85,
        "reason": "semantic claim support remains unreviewed; citation-ID validity is a separate metric",
    }
    capability["real_backend_validation"] = {
        "status": "NOT VERIFIED",
        "dense_hybrid": report["ablations"]["dense_only_vs_hybrid"]["status"],
        "provider_generation": "NOT VERIFIED",
    }
    capability["official_benchmark_validation"] = {
        "status": "NOT VERIFIED",
        "reason": "official corpus, replay cases, labels, thresholds, and organiser harness are absent",
    }
    capability["phase4_selective_claim_updates"] = {
        "status": "NOT VERIFIED",
        "scope": "intentionally not implemented or evaluated in Phase 3",
    }
    scenario_labels = {
        "single_questions_not_decomposed",
        "compound_independent_questions",
        "dependent_questions",
        "shared_constraints",
        "negation",
        "comparisons",
        "partial_answerability",
        "entity_confusion",
    }
    scenario_failures = [
        label for label in scenario_labels if capability[label]["status"] == "FAIL"
    ]
    capability["engineering_implementation"] = {
        "status": "PASS" if all(
            row["conditions"][condition]["trace"]["complete"]
            for row in all_rows
            for condition in _CONDITIONS
        ) else "FAIL",
        "scope": "local runtime implementation and audit trace execution",
        "note": "Unit/regression test status is reported by the verification run, not inferred from quality labels.",
    }
    quality_failures = [
        *scenario_failures,
        *[
            label
            for label in (
                "uncertainty_on_partial_requests",
                "uncertainty_on_unanswerable_requests",
                "conflicting_evidence_handling",
            )
            if capability[label]["status"] == "FAIL"
        ],
    ]
    capability["local_measured_quality"] = {
        "status": "FAIL" if quality_failures else "PASS",
        "scope": "synthetic lexical/mock fixture with provisional external labels",
        "failed_capabilities": quality_failures,
        "note": "This is local measured quality, not real-backend or official benchmark validation.",
    }
    return capability


def _matching_target(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = {
        split: splits[split]["multi_intent_identification"]["identification_rate"]
        for split in ("development", "held_out")
    }
    usable = [value for value in values.values() if isinstance(value, (int, float))]
    return {
        "target": 0.70,
        "status": "PASS" if usable and all(value >= 0.70 for value in usable) else "FAIL" if usable else "NOT VERIFIED",
        "development": values["development"],
        "held_out": values["held_out"],
        "interpretation": "Measured separately from citation support; denominators and provisional-label status are visible above.",
    }


async def evaluate_phase3_audit(
    cases: Sequence[Phase3EvaluationCase],
    *,
    corpus: CorpusIndex,
    settings: Settings,
    backend: str,
    top_k: int,
    execution_mode: Literal["realtime", "accelerated"],
    implementation_changes: dict[str, Any] | None = None,
    retrieval_mode: str | None = None,
) -> dict[str, Any]:
    """Run and assemble the complete matched Phase 3 audit."""

    if not cases:
        raise Phase3AuditError("at least one Phase 3 evaluation case is required")
    if execution_mode not in {"realtime", "accelerated"}:
        raise Phase3AuditError(f"unsupported execution mode: {execution_mode}")
    if top_k >= len(corpus.chunks):
        raise Phase3AuditError(
            f"top_k={top_k} must be smaller than corpus chunk_count={len(corpus.chunks)} "
            "for a non-trivial audit"
        )
    known_chunk_ids = {chunk.chunk_id for chunk in corpus.chunks}
    for case in cases:
        for intent in case.expected_intents:
            labelled_ids = set(intent.relevance_labels)
            unknown_ids = labelled_ids - known_chunk_ids
            if unknown_ids:
                raise Phase3AuditError(
                    f"case {case.case_id} intent {intent.intent_key} labels unknown chunks: "
                    f"{sorted(unknown_ids)}"
                )
            if not set(intent.relevant_chunk_ids) <= known_chunk_ids:
                raise Phase3AuditError(
                    f"case {case.case_id} intent {intent.intent_key} has unknown relevant chunks"
                )
    if any(case.asset_status != "synthetic_fixture" for case in cases):
        asset_status = "mixed_or_non_fixture"
    else:
        asset_status = "synthetic_fixture"
    all_case_rows: list[dict[str, Any]] = []
    selected_retrieval_mode = retrieval_mode or backend
    if selected_retrieval_mode not in {"dense", "lexical", "hybrid", "mock"}:
        raise Phase3AuditError(
            f"unsupported Phase 3 retrieval mode: {selected_retrieval_mode}"
        )
    for case in cases:
        outcomes = await asyncio.gather(
            *(
                _run_condition(
                    case,
                    corpus=corpus,
                    settings=settings,
                    backend=backend,
                    retrieval_mode=selected_retrieval_mode,
                    top_k=top_k,
                    execution_mode=execution_mode,
                    condition=condition,
                )
                for condition in _CONDITIONS
            )
        )
        raw_records = {
            outcome.condition: _mode_record(case, outcome, corpus=corpus, top_k=top_k)
            for outcome in outcomes
        }
        for record in raw_records.values():
            record["execution_mode"] = execution_mode
        all_case_rows.append(_enrich_case_result(case, raw_records))

    splits = {
        split: _phase3_split_summary(all_case_rows, split)
        for split in ("development", "held_out")
    }
    all_rows = all_case_rows
    all_condition_summaries = {
        condition: _condition_summary(all_rows, condition)
        for condition in _CONDITIONS
    }
    citation_case_count = sum(
        row["conditions"][condition].get("citation_ids_valid") is True
        for row in all_rows
        for condition in _CONDITIONS
    )
    citation_run_denominator = sum(
        row["conditions"][condition].get("citation_id_count", 0) > 0
        for row in all_rows
        for condition in _CONDITIONS
    )
    citation_case_denominator = len(all_rows) * len(_CONDITIONS)
    citation_invalid_count = sum(
        len(row["conditions"][condition].get("unknown_citation_ids", []))
        for row in all_rows
        for condition in _CONDITIONS
    )
    supported_numerator = sum(
        row["conditions"][condition].get("supported_answer", {}).get("supported_intent_count", 0)
        for row in all_rows
        for condition in _CONDITIONS
    )
    supported_denominator = sum(
        row["conditions"][condition].get("supported_answer", {}).get("answerable_intent_denominator", 0)
        for row in all_rows
        for condition in _CONDITIONS
    )
    label_summary = _label_status_summary(cases)
    data_integrity = _data_integrity(cases)
    change_audit = _implementation_change_audit(cases, implementation_changes or {})
    ablation_single_vs_decomposed = await _retrieval_ablation(
        cases,
        corpus=corpus,
        settings=settings,
        backend=backend,
        top_k=top_k,
    )
    report: dict[str, Any] = {
        "schema_version": "flowcontext.phase3-audit.v1",
        "report_version": "flowcontext.phase3-evaluation.v1",
        "evaluation_label": "phase3-dedicated-matched-audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset_status": asset_status,
        "synthetic_data_provisional": asset_status == "synthetic_fixture",
        "label_review_status_asset": "data/evaluation/phase3_label_review_status.json",
        "official_benchmark_claim": False,
        "code_revision": _code_revision(),
        "code_revision_note": "repository HEAD identifier; this audit was run against the current working tree",
        "configuration": {
            "backend": backend,
            "phase3_retrieval_mode": selected_retrieval_mode,
            "top_k": top_k,
            "execution_mode": execution_mode,
            "generation_backend": settings.generation_backend,
            "generation_provider": settings.generation_provider,
            "generation_model": settings.generation_model,
            "decomposition_provider": settings.generation_provider,
            "decomposition_model": settings.generation_model,
            "multi_intent_context_budget_tokens": settings.multi_intent_context_budget_tokens,
            "multi_intent_max_workers": settings.multi_intent_max_workers,
            "multi_intent_rrf_k": settings.multi_intent_rrf_k,
            "multi_intent_reranking_enabled": settings.multi_intent_reranking_enabled,
            "conditions": {
                "A_original_baseline": "original final-event baseline; one parent-query retrieval after finalisation",
                "B_phase2_streaming_single_query": "Phase 2 streaming controller/scheduler; one parent query; final-only generation",
                "C_phase3_streaming_fused": "Phase 3 streaming controller/scheduler; structured decomposition; per-intent retrieval; RRF evidence fusion; final-only generation",
            },
            "fairness": "same corpus/index, top-k, settings, replay events, execution mode, process, hardware, and generation configuration; C additionally requires its decomposition stage by design",
        },
        "corpus": {
            "corpus_id": corpus.corpus_id,
            "index_id": corpus.manifest.index_id,
            "source_kind": corpus.source_kind,
            "document_count": len(corpus.documents),
            "chunk_count": len(corpus.chunks),
            "top_k": top_k,
            "top_k_less_than_chunk_count": top_k < len(corpus.chunks),
            "source_path": corpus.manifest.source_path,
            "chunking": corpus.manifest.chunking.model_dump(mode="json"),
            "embedding": corpus.manifest.embedding.model_dump(mode="json"),
            "interpretation": "top-k is reported with corpus size because top-k must not trivially include every chunk",
        },
        "data_integrity": data_integrity,
        "label_review_status": label_summary,
        "implementation_change_audit": change_audit,
        "matched_comparison": {
            "conditions": list(_CONDITIONS),
            "same_corpus": True,
            "same_top_k": True,
            "same_model_configuration": True,
            "same_hardware": True,
            "same_replay_conditions": True,
            "development": splits["development"]["matched_comparison_retrieval_quality"],
            "held_out": splits["held_out"]["matched_comparison_retrieval_quality"],
            "all": {
                condition: _quality_summary(
                    all_rows,
                    condition=condition,
                    matched_case_ids=[
                        row["case_id"]
                        for row in all_rows
                        if all(row["conditions"][other].get("retrieval_quality_scored") for other in _CONDITIONS)
                    ],
                )
                for condition in _CONDITIONS
            },
        },
        "splits": splits,
        "resources_by_condition": all_condition_summaries,
        "citation_id_validity": {
            "valid_case_count": citation_case_count,
            "case_denominator": citation_case_denominator,
            "citation_run_denominator": citation_run_denominator,
            "case_validity_rate": citation_case_count / citation_run_denominator if citation_run_denominator else None,
            "no_citation_case_count": citation_case_denominator - citation_run_denominator,
            "invalid_id_count": citation_invalid_count,
            "semantic_support": "NOT VERIFIED",
            "interpretation": "Citation IDs only establish that an ID is known and supplied to the answer; they do not establish claim support.",
        },
        "semantic_claim_support": {
            "status": "NOT VERIFIED",
            "assessed_claim_count": 0,
            "claim_count_observed": sum(
                len(row["conditions"][condition].get("factual_claims", []))
                for row in all_rows
                for condition in _CONDITIONS
            ),
            "review_status": "not_evaluated",
            "reason": "No human or governed semantic evaluator was available; no semantic quality claim is inferred from citation IDs or substrings.",
        },
        "supported_answer_coverage": {
            "supported_intent_count": supported_numerator,
            "answerable_intent_denominator": supported_denominator,
            "coverage": supported_numerator / supported_denominator if supported_denominator else None,
            "nonzero_supported_portion_observed": supported_numerator > 0,
            "interpretation": "This denominator includes only answerable intent units and is reported separately so all-abstention cannot look successful.",
        },
        "provider_and_retrieval_failures": _provider_failure_audit(all_rows),
        "corrections_and_reuse": [
            {
                "case_id": row["case_id"],
                "requires_early_retrieval": row["correction_audit"]["requires_early_retrieval"],
                "conditions": {
                    condition: {
                        "early_retrieval_started": row["conditions"][condition].get("early_retrieval_started", False),
                        "valid_evidence_ready_before_final": row["conditions"][condition].get("valid_evidence_ready_before_final", False),
                        "useful_early_reuse": row["conditions"][condition].get("useful_early_reuse", False),
                        "stale_result_discard_count": row["conditions"][condition].get("stale_result_discard_count", 0),
                        "stale_result_accepted_count": row["conditions"][condition].get("stale_result_accepted_count", 0),
                    }
                    for condition in ("B_phase2_streaming_single_query", "C_phase3_streaming_fused")
                },
            }
            for row in all_rows
            if row["correction_audit"]["requires_early_retrieval"]
            or len(row["gold"]["expected_intents"]) > 1
        ],
        "ablations": {
            "single_query_vs_decomposed": ablation_single_vs_decomposed,
            "dense_only_vs_hybrid": _dense_hybrid_ablation(corpus),
        },
        "targets": {
            "multi_intent_identification_70_percent": _matching_target(splits),
            "citation_support_85_percent": {
                "target": 0.85,
                "status": "NOT VERIFIED",
                "reason": "semantic claim support remains unreviewed; citation-ID validity is not a substitute",
            },
        },
        "environment": _environment(),
        "hardware": _hardware(),
        "capability_status": {},
        "scope_boundary": {
            "phase3_complete": True,
            "phase4_started": False,
            "phase4_selective_claim_updates": "out_of_scope",
            "phase4_handoff_document": "docs/phase4-handoff.md",
        },
        "limitations": [
            "All cases and corpus documents are synthetic fixture data grounded in the available local corpus; they are not organiser assets.",
            "All external intent, relevance, and answer labels are provisional_generated; no model-reviewed or human-reviewed labels are present.",
            "The mock generation/decomposition providers are contract-test providers, not model-quality evidence.",
            "Semantic claim support was not assessed. The guide's 85% citation-support target is NOT VERIFIED.",
            "Dense-only and hybrid retrieval were not run because no usable dense dependency/model/index was available.",
            "Real provider-backed generation and official benchmark validation were not run.",
            "The Phase 2 final-event stale-result and session-isolation regressions are verified by the existing test suite; this report does not claim Phase 4 state updates.",
        ],
    }
    report["capability_status"] = _capability_status(report)
    report["validation_domains"] = {
        "engineering_implementation": {
            "status": report["capability_status"]["engineering_implementation"]["status"],
            "evidence": "dedicated A/B/C runner completed and trace completeness was checked",
        },
        "local_measured_quality": {
            "status": report["capability_status"]["local_measured_quality"]["status"],
            "evidence": "provisional synthetic corpus/label measurements; see per-capability results",
        },
        "real_backend_validation": {
            "status": "NOT VERIFIED",
            "evidence": "no usable dense model/index or live provider run was available",
        },
        "official_benchmark_validation": {
            "status": "NOT VERIFIED",
            "evidence": "official corpus, labels, harness, and organiser API are unavailable",
        },
    }
    return report


def _status_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "| Capability | Status | Evidence boundary |",
        "|---|---|---|",
    ]
    for name, item in report["capability_status"].items():
        evidence = item.get("scope") or item.get("evidence_basis") or "local audit record"
        lines.append(f"| `{name}` | **{item.get('status', 'NOT VERIFIED')}** | {evidence} |")
    return lines


def _resource_markdown(summary: dict[str, Any]) -> list[str]:
    resources = summary["resources"]
    tokens = resources["tokens"]
    return [
        f"- Retrieval calls: {resources['retrieval_calls']}; model calls: {resources['model_calls']}",
        f"- Tokens: retrieval {tokens['retrieval']['total_tokens']}, decomposition {tokens['decomposition']['total_tokens']}, generation {tokens['generation']['total_tokens']}, repair {tokens['repair']['total_tokens']}, verification {tokens['verification']['total_tokens']}, total {tokens['total']['total_tokens']}",
        f"- Errors: {resources['error_count']} events across {resources['error_case_count']} cases; cost: `{resources['cost']['value']}` ({'available' if resources['cost']['available'] else 'unavailable'})",
    ]


def write_phase3_audit_report(path: Path, report: dict[str, Any]) -> Path:
    """Write machine-readable JSON and its Markdown companion."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path = path.with_suffix(".md")
    lines = [
        "# Phase 3 dedicated evaluation and audit",
        "",
        "Status: local synthetic fixture / lexical retrieval / mock providers; not an official benchmark result.",
        "",
        "Phase 4 was not started. Historical Phase 1/2/3 reports are preserved; this is a new report.",
        "",
        "## Matched conditions",
        "",
        "| Condition | Definition |",
        "|---|---|",
        "| A | Original final-event baseline; one parent-query retrieval after finalisation. |",
        "| B | Phase 2 streaming controller with a single parent query and final-only generation. |",
        "| C | Phase 3 streaming with structured decomposition, per-intent retrieval, RRF fusion, and grounded synthesis. |",
        "",
        f"Corpus: **{report['corpus']['document_count']} documents / {report['corpus']['chunk_count']} chunks**; top-k: **{report['corpus']['top_k']}**; top-k less than corpus chunks: **{report['corpus']['top_k_less_than_chunk_count']}**.",
        "",
        "All three conditions use the same corpus/index, top-k, configuration, hardware, and replay cases. Condition C has the additional decomposition stage required by Phase 3.",
        "",
        "## Label and split integrity",
        "",
        f"Development cases: {report['data_integrity']['development_case_count']}; held-out cases: {report['data_integrity']['held_out_case_count']}; related variant families cross splits: `{report['data_integrity']['variant_families_crossing_splits'] or 'none'}`.",
        "",
        "Scenario coverage: "
        + "; ".join(
            f"{name}={count}"
            for name, count in report["data_integrity"]["scenario_group_counts"].items()
        )
        + ".",
        "",
        "Label status counts are explicit: all current intent, relevance, and answer-expectation labels are `provisional_generated`; model-reviewed and human-reviewed counts are zero. Semantic claim support review is `not_evaluated`.",
        "",
        "## Guide targets",
        "",
        f"- Multi-intent identification target (70%): **{report['targets']['multi_intent_identification_70_percent']['status']}** locally against provisional labels; development `{report['targets']['multi_intent_identification_70_percent']['development']}`, held-out `{report['targets']['multi_intent_identification_70_percent']['held_out']}`.",
        "- Citation-support target (85%): **NOT VERIFIED**. Citation-ID validity is reported separately and does not establish semantic support.",
        "",
        "## Split results",
    ]
    for split in ("development", "held_out"):
        summary = report["splits"][split]
        multi = summary["multi_intent_identification"]
        lines.extend(
            [
                "",
                f"### {split.replace('_', ' ').title()}",
                "",
                f"Cases: {summary['case_count']} (single {summary['single_case_count']}, compound {summary['compound_case_count']}). Multi-intent exact identification: {multi['correct_multi_intent_case_count']}/{multi['compound_case_denominator']} = `{multi['identification_rate']}`; missed intents: {multi['missed_intent_count']}; unnecessary extra intents: {multi['unnecessary_extra_intent_count']}.",
                "",
                "| Condition | Matched Recall@5 | Matched cases | Completed / all | Final-event→answer p50/p95 ms |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        matched = summary["matched_comparison_retrieval_quality"]
        for condition in _CONDITIONS:
            q = matched[condition]
            cs = summary["conditions"][condition]
            latency = cs["latency_final_event_to_answer_ms"]
            lines.append(
                f"| {condition} | {q['recall_at_5_macro_mean']} | {q['case_denominator']} | {cs['completed_case_count']}/{cs['case_denominator']} | {latency['p50']} / {latency['p95']} |"
            )
        lines.extend(
            [
                "",
                "Per-condition resources:",
            ]
        )
        for condition in _CONDITIONS:
            lines.append(f"**{condition}**")
            lines.extend(_resource_markdown(summary["conditions"][condition]))
        c_condition = summary["conditions"]["C_phase3_streaming_fused"]
        citation = c_condition["citation_id_validity"]
        intent_summary = c_condition["per_intent_retrieval"]
        if intent_summary.get("status") == "measured":
            intent_line = (
                f"Phase 3 per-intent retrieval Recall@1/@3/@5 (macro): "
                f"`{intent_summary['intent_recall_at_1_macro_mean']}` / "
                f"`{intent_summary['intent_recall_at_3_macro_mean']}` / "
                f"`{intent_summary['intent_recall_at_5_macro_mean']}` over "
                f"{intent_summary['intent_denominator']} applicable labelled intent records."
            )
        else:
            intent_line = f"Phase 3 per-intent retrieval: `{intent_summary.get('status', 'NOT VERIFIED')}` ({intent_summary.get('reason', 'no applicable cases')})."
        b_early = summary["conditions"]["B_phase2_streaming_single_query"]["early_retrieval"]
        c_early = summary["conditions"]["C_phase3_streaming_fused"]["early_retrieval"]
        early_line = (
            "Phase 2/3 early retrieval and reuse: "
            f"B started {b_early['actual_early_start_count']}/{b_early['eligible_case_count']}, "
            f"useful reuse {b_early['useful_early_reuse_count']}, "
            f"stale accepted {b_early['stale_result_accepted_count']}; "
            f"C started {c_early['actual_early_start_count']}/{c_early['eligible_case_count']}, "
            f"useful reuse {c_early['useful_early_reuse_count']}, "
            f"stale accepted {c_early['stale_result_accepted_count']}."
        )
        lines.extend(
            [
                "",
                f"Phase 3 complete-request evidence coverage (macro): `{c_condition['complete_request_evidence_coverage']['macro_mean']}` over {c_condition['complete_request_evidence_coverage']['case_count_with_applicable_coverage']} cases.",
                intent_line,
                f"Phase 3 supported-answer coverage (provisional substring check): `{c_condition['supported_answer_coverage']['supported_intent_count']}/{c_condition['supported_answer_coverage']['answerable_intent_denominator']}` = `{c_condition['supported_answer_coverage']['coverage']}`; this is not semantic support.",
                f"Phase 3 citation-ID validity: `{citation['valid_case_count']}/{citation['citation_run_denominator']}` emitted-citation runs = `{citation['valid_case_rate']}`; {citation['no_citation_case_count']} runs emitted no citations; semantic support: **NOT VERIFIED**.",
                early_line,
            ]
        )
    lines.extend(
        [
            "",
            "## Focused ablations",
            "",
            f"Single-query versus decomposed retrieval: `{report['ablations']['single_query_vs_decomposed']['status']}` over {report['ablations']['single_query_vs_decomposed']['case_denominator']} normal, answerable labelled cases; lexical-only engineering measurement, not a model-quality finding.",
            f"Dense-only versus hybrid: **{report['ablations']['dense_only_vs_hybrid']['status']}** — {report['ablations']['dense_only_vs_hybrid']['reason']}.",
            "",
            "## Failures and limitations",
            "",
        ]
    )
    for item in report["provider_and_retrieval_failures"]:
        lines.append(
            f"- `{item['case_id']}`: provider `{item['provider_failure_stage']}`, retrieval `{item['retrieval_behavior']}`; each arm is retained with status/error/fallback details in JSON."
        )
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(["", "## PASS / FAIL / NOT VERIFIED", "", *_status_table(report), ""])
    lines.extend(
        [
            "### Validation domains",
            "",
            "| Domain | Status | Evidence boundary |",
            "|---|---|---|",
            *(
                f"| `{name}` | **{item['status']}** | {item['evidence']} |"
                for name, item in report["validation_domains"].items()
            ),
            "",
        ]
    )
    lines.extend(
        [
            "## Phase 4 handoff",
            "",
            "The interface-only handoff is [docs/phase4-handoff.md](../docs/phase4-handoff.md). It covers intent IDs, typed constraints, claim records, evidence dependencies, revision/supersession records, and session-state boundaries. No Phase 4 selective claim-update implementation was added.",
            "",
            "The machine-readable report contains complete per-case condition records, intent matching, per-intent rankings/fusion decisions, evidence dependencies, claim records, latency, calls/tokens/errors/cost, trace completeness, and label-review provenance.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path
