"""Transparent evaluation and reproducibility workflow for Phase 1.

The evaluator keeps three distinctions explicit:

* the supplied corpus/evaluation assets may be synthetic fixtures rather than
  organiser assets;
* citation-ID validity is an identifier check, not semantic claim support; and
* accelerated replay timings are functional-test measurements, not real-time
  latency evidence.

No model judge is enabled here. Claim support requires the documented human
review procedure in :func:`_claim_support_review` and is reported as
``not_evaluated`` until a reviewer records labels.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Sequence

from pydantic import ValidationError

from .config import Settings
from .contracts import (
    ClaimSupportReview,
    CorpusIndex,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationReport,
    EvaluationSection,
    EvaluationSuiteReport,
    LatencySummary,
    ReplayResult,
    Usage,
)
from .generation import GenerationProvider, generation_provider_for_settings
from .replay import replay_transcript
from .retrieval import make_retriever


class EvaluationError(ValueError):
    """Raised when evaluation assets are missing or invalid."""


REQUIRED_TRACE_EVENT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("replay_started",),
    ("transcript_event_received",),
    ("final_event_delivered",),
    ("utterance_finalized",),
    ("retrieval_scheduled",),
    ("retrieval_started",),
    ("retrieval_completed",),
    ("generation_started",),
    ("generation_completed", "generation_abstained", "generation_skipped"),
    ("answer_completed", "answer_failed"),
    ("replay_completed",),
)


def load_evaluation_cases(path: Path, *, expected_split: str | None = None) -> list[EvaluationCase]:
    """Load one JSONL evaluation asset and optionally enforce its split."""

    if not path.is_file():
        raise EvaluationError(f"evaluation asset does not exist: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EvaluationError(f"evaluation asset is not valid UTF-8: {path}") from exc
    cases: list[EvaluationCase] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            case = EvaluationCase.model_validate(json.loads(raw_line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise EvaluationError(f"invalid evaluation case on line {line_number}: {exc}") from exc
        if expected_split is not None and case.split != expected_split:
            raise EvaluationError(
                f"evaluation case {case.case_id!r} is in split {case.split!r}, "
                f"not requested split {expected_split!r}"
            )
        cases.append(case)
    if not cases:
        raise EvaluationError(f"evaluation asset contains no cases: {path}")
    return cases


def _expected_evidence_ids(case: EvaluationCase) -> list[str]:
    """Return canonical expected evidence IDs with legacy compatibility."""

    return list(case.expected_evidence_chunk_ids or case.expected_citation_chunk_ids)


def _retrieval_started_after_final(result: ReplayResult) -> bool:
    starts = [
        trace
        for trace in result.traces
        if trace.event_type in {"retrieval_started", "retrieval_completed"}
    ]
    if not starts:
        return False
    return all(
        trace.source_timestamp_s is not None
        and trace.source_timestamp_s >= result.final_source_timestamp_s
        for trace in starts
    )


def _trace_completeness(result: ReplayResult) -> tuple[bool, list[str]]:
    event_types = {trace.event_type for trace in result.traces}
    required_groups = REQUIRED_TRACE_EVENT_GROUPS
    if not result.retrieval_triggered:
        required_groups = tuple(
            group
            for group in required_groups
            if group not in {
                ("retrieval_scheduled",),
                ("retrieval_started",),
                ("retrieval_completed",),
            }
        )
    missing = [
        "/".join(group)
        for group in required_groups
        if not any(event_type in event_types for event_type in group)
    ]
    return not missing, missing


def _trace_errors(result: ReplayResult) -> list[str]:
    return [
        f"{trace.error.error_type}: {trace.error.message}"
        for trace in result.traces
        if trace.error is not None
    ]


def _answer_citation_ids(result: ReplayResult) -> set[str]:
    return {
        chunk_id
        for claim in result.answer.factual_claims
        for chunk_id in claim.supporting_chunk_ids
    }


def _retrieval_metrics(
    expected_ids: set[str],
    retrieved_ids: Sequence[str],
    *,
    labels_available: bool,
    ks: Sequence[int] = (1, 3, 5),
) -> tuple[dict[str, float], float | None]:
    if not labels_available or not expected_ids:
        return {}, None
    retrieved = list(retrieved_ids)
    metrics: dict[str, float] = {}
    for k in ks:
        top_k = set(retrieved[:k])
        metrics[f"recall_at_{k}"] = len(top_k & expected_ids) / len(expected_ids)
    reciprocal_rank: float | None = None
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected_ids:
            reciprocal_rank = 1.0 / rank
            break
    return metrics, reciprocal_rank


def _latency_summary(values: Sequence[float]) -> LatencySummary:
    """Use inclusive linear interpolation and preserve the sample count."""

    if not values:
        return LatencySummary(sample_count=0)
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * weight

    return LatencySummary(
        sample_count=len(ordered),
        p50_ms=percentile(0.50),
        p95_ms=percentile(0.95),
    )


def _combined_usage(usages: Sequence[Usage]) -> Usage:
    if not usages:
        return Usage()
    return Usage(
        input_tokens=sum(usage.input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
        estimated=any(usage.estimated for usage in usages),
    )


def _combined_cost(values: Sequence[float | str]) -> float | str:
    numeric_values = [
        value
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not values or len(numeric_values) != len(values):
        return "unavailable"
    return float(sum(numeric_values))


def _claim_support_review() -> ClaimSupportReview:
    return ClaimSupportReview(
        status="not_evaluated",
        procedure=(
            "For each factual claim, a human reviewer reads the cited source chunk(s), "
            "checks whether the text entails the claim, and records supported, "
            "unsupported, or uncertain. Reviewers must not treat a valid chunk ID "
            "as evidence of support."
        ),
        limitations=[
            "No human review labels were supplied or performed for this local run.",
            "No model judge was run; there is no judge model or judge score.",
        ],
    )


def _code_revision() -> str | None:
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


def _environment() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "os_name": os.name,
        "executable": sys.executable,
    }


def _hardware() -> dict[str, str]:
    return {
        "machine": platform.machine() or "unknown",
        "cpu_count": str(os.cpu_count() or "unknown"),
        "gpu": "not_detected_by_phase1",
    }


def _empty_case_result(
    case: EvaluationCase,
    *,
    waited_until_final: bool = False,
    error: str | None = None,
) -> EvaluationCaseResult:
    expected_ids = _expected_evidence_ids(case)
    errors = [error] if error else []
    return EvaluationCaseResult(
        case_id=case.case_id,
        split=case.split,
        answerability=case.answerability,
        answer_substrings_passed=not case.expected_answer_substrings,
        missing_answer_substrings=list(case.expected_answer_substrings),
        expected_evidence_chunk_ids=expected_ids,
        citation_ids_valid=False,
        waited_until_final=waited_until_final,
        generation_status="abstained",
        run_status="failed" if error else "abstained",
        trace_complete=False,
        errors=errors,
        passed=False,
        notes=["case did not produce a replay result"] + errors,
    )


def evaluate_case(case: EvaluationCase, result: ReplayResult, corpus: CorpusIndex) -> EvaluationCaseResult:
    """Evaluate traceability and explicit expectations for one replay result."""

    expected_ids = _expected_evidence_ids(case)
    if case.session_id != result.session_id:
        return _empty_case_result(
            case,
            error="case session_id does not match replay session_id",
        )

    answer_text = result.answer.answer_text.casefold()
    missing_substrings = [
        expected for expected in case.expected_answer_substrings if expected.casefold() not in answer_text
    ]
    known_chunk_ids = {chunk.chunk_id for chunk in corpus.chunks}
    cited_ids = _answer_citation_ids(result)
    actual_hit_ids = {hit.chunk_id for hit in result.retrieval_hits}
    unknown_ids = sorted(cited_ids - actual_hit_ids)
    missing_citations = [expected for expected in expected_ids if expected not in cited_ids]
    citation_ids_valid = cited_ids.issubset(known_chunk_ids) and cited_ids.issubset(actual_hit_ids)
    waited_until_final = _retrieval_started_after_final(result)
    answer_substrings_passed = not missing_substrings
    expected_citations_passed = not missing_citations
    abstained = result.generation_status in {"abstained", "skipped"}
    if case.answerability == "unanswerable":
        answerability_passed = abstained and not result.answer.factual_claims
    elif case.answerability == "partially_answerable":
        answerability_passed = result.generation_status == "success" and answer_substrings_passed
    else:
        answerability_passed = result.generation_status == "success"
    passed = (
        answerability_passed
        and answer_substrings_passed
        and expected_citations_passed
        and citation_ids_valid
        and (waited_until_final if case.require_wait_until_final else True)
        and result.run_status != "failed"
    )
    retrieval_label_ids = {
        chunk_id for chunk_id, label in case.relevance_labels.items() if label == "relevant"
    }
    recall_metrics, reciprocal_rank = _retrieval_metrics(
        retrieval_label_ids,
        [hit.chunk_id for hit in result.retrieval_hits],
        labels_available=bool(case.relevance_labels),
    )
    trace_complete, missing_trace_events = _trace_completeness(result)
    notes: list[str] = [
        "citation IDs validated against known and supplied chunks; semantic claim support not evaluated",
        "accelerated/realtime latency interpretation is recorded by the run execution mode",
    ]
    if result.corpus_source_kind == "synthetic_fixture" or case.asset_status == "synthetic_fixture":
        notes.append("fixture-only result; not competition performance")
    if not result.retrieval_hits:
        notes.append("no retrieval hits")
    if case.capabilities_deferred:
        notes.append("deferred capabilities: " + ", ".join(case.capabilities_deferred))
    if not case.relevance_labels and expected_ids:
        notes.append("expected evidence exists but no explicit relevance labels were supplied")
    errors = _trace_errors(result)
    retrieval_trace = next(
        (trace for trace in result.traces if trace.event_type == "retrieval_completed"),
        None,
    )
    return EvaluationCaseResult(
        case_id=case.case_id,
        split=case.split,
        answerability=case.answerability,
        answer_substrings_passed=answer_substrings_passed,
        missing_answer_substrings=missing_substrings,
        expected_evidence_chunk_ids=expected_ids,
        retrieved_chunk_ids=[hit.chunk_id for hit in result.retrieval_hits],
        retrieval_labels_available=bool(case.relevance_labels),
        retrieval_recall_at_k=recall_metrics,
        retrieval_reciprocal_rank=reciprocal_rank,
        citation_ids_valid=citation_ids_valid,
        citation_ids_checked=len(cited_ids),
        unknown_citation_ids=unknown_ids,
        semantic_support_evaluated=False,
        claim_support_status="not_evaluated",
        missing_expected_citations=missing_citations,
        waited_until_final=waited_until_final,
        abstained=abstained,
        generation_status=result.generation_status,
        run_status=result.run_status,
        retrieval_latency_ms=retrieval_trace.duration_ms if retrieval_trace else None,
        answer_latency_ms=result.complete_answer_latency_ms,
        generation_usage=result.generation_usage,
        generation_cost=result.generation_cost,
        trace_complete=trace_complete,
        missing_trace_events=missing_trace_events,
        errors=errors,
        passed=passed,
        notes=notes,
    )


def evaluate(
    cases: list[EvaluationCase],
    result: ReplayResult,
    corpus: CorpusIndex,
) -> EvaluationReport:
    """Compatibility evaluator for one legacy replay artifact."""

    if not cases:
        raise EvaluationError("at least one evaluation case is required")
    case_results = [evaluate_case(case, result, corpus) for case in cases]
    passed_count = sum(case.passed for case in case_results)
    citation_validity_count = sum(case.citation_ids_valid for case in case_results)
    wait_count = sum(case.waited_until_final for case in case_results)
    total = len(case_results)
    asset_statuses = {case.asset_status for case in cases}
    if asset_statuses == {"official"}:
        asset_status = "official"
    elif asset_statuses == {"synthetic_fixture"}:
        asset_status = "synthetic_fixture"
    else:
        asset_status = "unknown"
    return EvaluationReport(
        evaluation_label="phase1-local-evaluation",
        asset_status=asset_status,
        retrieval_backend=result.retrieval_backend,
        generation_backend=result.generation_backend,
        generation_model=result.generation_model,
        generation_status=result.generation_status,
        execution_mode=result.execution_mode,
        semantic_support_evaluated=False,
        competition_performance_claim=False,
        passed=passed_count == total,
        cases=case_results,
        metrics={
            "case_pass_rate": passed_count / total,
            "citation_id_validity_rate": citation_validity_count / total,
            "semantic_support_evaluation_rate": 0.0,
            "wait_until_final_rate": wait_count / total,
        },
    )


def _suite_metrics(case_results: Sequence[EvaluationCaseResult]) -> dict[str, float]:
    total = len(case_results)
    if not total:
        return {}
    metrics: dict[str, float] = {
        "case_pass_rate": sum(result.passed for result in case_results) / total,
        "citation_id_validation_case_rate": sum(result.citation_ids_valid for result in case_results) / total,
        "semantic_support_evaluation_rate": 0.0,
        "wait_until_final_rate": sum(result.waited_until_final for result in case_results) / total,
        "trace_completeness_rate": sum(result.trace_complete for result in case_results) / total,
        "error_case_rate": sum(bool(result.errors) for result in case_results) / total,
        "early_retrieval_event_count": 0.0,
    }
    labelled = [result for result in case_results if result.retrieval_labels_available]
    metrics["retrieval_labelled_case_count"] = float(len(labelled))
    for key in ("recall_at_1", "recall_at_3", "recall_at_5"):
        values = [
            result.retrieval_recall_at_k[key]
            for result in labelled
            if key in result.retrieval_recall_at_k
        ]
        if values:
            metrics[f"retrieval_{key}"] = fmean(values)
    reciprocal_ranks = [
        result.retrieval_reciprocal_rank
        for result in labelled
        if result.retrieval_reciprocal_rank is not None
    ]
    if reciprocal_ranks:
        metrics["retrieval_mrr"] = fmean(reciprocal_ranks)

    answerable = [result for result in case_results if result.answerability == "answerable"]
    unanswerable = [result for result in case_results if result.answerability == "unanswerable"]
    metrics["answerable_case_count"] = float(len(answerable))
    metrics["unanswerable_case_count"] = float(len(unanswerable))
    if answerable:
        metrics["answerable_abstention_rate"] = sum(result.abstained for result in answerable) / len(answerable)
    if unanswerable:
        metrics["unanswerable_abstention_rate"] = sum(result.abstained for result in unanswerable) / len(unanswerable)

    citation_count = sum(result.citation_ids_checked for result in case_results)
    invalid_citation_count = sum(len(result.unknown_citation_ids) for result in case_results)
    metrics["citation_id_sample_count"] = float(citation_count)
    if citation_count:
        metrics["citation_id_validity_rate"] = max(0.0, citation_count - invalid_citation_count) / citation_count
    return metrics


def _section(
    *,
    section_id: str,
    status: str,
    result_group: str,
    corpus: CorpusIndex,
    settings: Settings,
    cases: Sequence[EvaluationCase],
    case_results: Sequence[EvaluationCaseResult],
    retrieval_backend: str | None,
    generation_backend: str | None,
    generation_model: str | None,
    execution_mode: str | None,
    limitations: Sequence[str],
) -> EvaluationSection:
    retrieval_latencies = [
        result.retrieval_latency_ms
        for result in case_results
        if result.retrieval_latency_ms is not None
    ]
    answer_latencies = [
        result.answer_latency_ms
        for result in case_results
        if result.answer_latency_ms is not None
    ]
    usages = [result.generation_usage for result in case_results]
    costs = [result.generation_cost for result in case_results]
    error_count = sum(len(result.errors) for result in case_results)
    return EvaluationSection(
        section_id=section_id,
        status=status,
        result_group=result_group,
        asset_status=corpus.source_kind,
        corpus_id=corpus.corpus_id,
        index_id=corpus.manifest.index_id,
        corpus_source_kind=corpus.source_kind,
        retrieval_backend=retrieval_backend,  # type: ignore[arg-type]
        generation_backend=generation_backend,
        generation_model=generation_model,
        execution_mode=execution_mode,
        warm_cold_condition=(
            "cold CLI process; index/retriever/provider initialized once and reused across cases; "
            "per-case timings exclude process startup"
        ),
        case_count=len(case_results),
        answerable_case_count=sum(case.answerability == "answerable" for case in cases),
        unanswerable_case_count=sum(case.answerability == "unanswerable" for case in cases),
        case_results=list(case_results),
        metrics=_suite_metrics(case_results),
        retrieval_latency=_latency_summary(retrieval_latencies),
        answer_latency=_latency_summary(answer_latencies),
        generation_usage=_combined_usage(usages),
        generation_cost=_combined_cost(costs),
        claim_support=_claim_support_review(),
        error_count=error_count,
        trace_complete_count=sum(result.trace_complete for result in case_results),
        trace_required_event_set=[" or ".join(group) for group in REQUIRED_TRACE_EVENT_GROUPS],
        limitations=list(limitations),
    )


async def evaluate_suite(
    cases: Sequence[EvaluationCase],
    *,
    corpus: CorpusIndex,
    settings: Settings,
    backend: str | None = None,
    top_k: int | None = None,
    execution_mode: str = "accelerated",
    generation_provider: GenerationProvider | None = None,
    run_id_prefix: str = "evaluation",
    evaluation_asset: str | None = None,
) -> EvaluationSuiteReport:
    """Run a fixed evaluation split once through the Phase 1 baseline.

    The index and provider are constructed once and reused. This makes the
    report's warm/cold condition explicit and avoids tuning or changing
    configuration between development and held-out cases.
    """

    if not cases:
        raise EvaluationError("at least one evaluation case is required")
    selected_backend = backend or settings.retrieval_backend
    selected_top_k = top_k or settings.retrieval_top_k
    selected_retriever = make_retriever(
        corpus,
        backend=selected_backend,
        top_k=selected_top_k,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
    )
    selected_provider = generation_provider or generation_provider_for_settings(settings)
    case_results: list[EvaluationCaseResult] = []
    for case in cases:
        if not case.transcript:
            raise EvaluationError(
                f"evaluation case {case.case_id!r} has no embedded transcript; "
                "suite cases must include their transcript events"
            )
        try:
            result = await replay_transcript(
                case.transcript,
                corpus=corpus,
                top_k=selected_top_k,
                backend=selected_backend,
                retriever=selected_retriever,
                generation_provider=selected_provider,
                run_id=f"{run_id_prefix}-{case.case_id}",
                model_identity=settings.model_identity,
                embedding_cache_dir=settings.embedding_cache_dir,
                embedding_local_files_only=settings.embedding_local_files_only,
                execution_mode=execution_mode,  # type: ignore[arg-type]
            )
        except Exception as exc:
            case_results.append(
                _empty_case_result(
                    case,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            case_results.append(evaluate_case(case, result, corpus))

    all_cases = list(cases)
    asset_statuses = {case.asset_status for case in all_cases}
    asset_status = (
        "official"
        if asset_statuses == {"official"}
        else "synthetic_fixture"
        if asset_statuses == {"synthetic_fixture"}
        else "unknown"
    )
    generation_config = selected_provider.config
    run_errors = sum(bool(result.errors) for result in case_results)
    common_limitations = [
        "Official corpus/evaluation assets were not supplied; this is a local synthetic fixture run.",
        "Relevance labels are provisional generated labels and are not human-verified ground truth.",
        "Claim support is not evaluated; citation-ID validity only checks identifier traceability.",
        "Phase 1 early retrieval is intentionally absent; all retrieval waits for the final event.",
        "Accelerated replay latency is not evidence of real-time latency.",
    ]
    if generation_config.backend == "mock":
        common_limitations.append("Mock generation is an engineering provider, not model-quality evidence.")
    else:
        common_limitations.append("Real-provider execution is represented only if the configured call completes.")

    measured_status = "failed" if run_errors else "measured"
    mock_section = _section(
        section_id="mock",
        status=measured_status if generation_config.backend == "mock" else "not_verified",
        result_group="mock_fixture" if generation_config.backend == "mock" else "not_run",
        corpus=corpus,
        settings=settings,
        cases=all_cases if generation_config.backend == "mock" else [],
        case_results=case_results if generation_config.backend == "mock" else [],
        retrieval_backend=selected_backend,
        generation_backend=generation_config.backend if generation_config.backend == "mock" else None,
        generation_model=generation_config.model if generation_config.backend == "mock" else None,
        execution_mode=execution_mode if generation_config.backend == "mock" else None,
        limitations=common_limitations,
    )
    fixture_section = _section(
        section_id="fixture",
        status=measured_status,
        result_group="mock_fixture" if generation_config.backend == "mock" else "real_provider_fixture",
        corpus=corpus,
        settings=settings,
        cases=all_cases,
        case_results=case_results,
        retrieval_backend=selected_backend,
        generation_backend=generation_config.backend,
        generation_model=generation_config.model,
        execution_mode=execution_mode,
        limitations=common_limitations,
    )
    real_section = _section(
        section_id="real_model",
        status=(
            measured_status
            if generation_config.backend == "openai_compatible"
            else "not_verified"
        ),
        result_group="real_provider_fixture" if generation_config.backend == "openai_compatible" else "not_run",
        corpus=corpus,
        settings=settings,
        cases=all_cases if generation_config.backend == "openai_compatible" else [],
        case_results=case_results if generation_config.backend == "openai_compatible" else [],
        retrieval_backend=selected_backend if generation_config.backend == "openai_compatible" else None,
        generation_backend=generation_config.backend if generation_config.backend == "openai_compatible" else None,
        generation_model=generation_config.model if generation_config.backend == "openai_compatible" else None,
        execution_mode=execution_mode if generation_config.backend == "openai_compatible" else None,
        limitations=(
            common_limitations
            if generation_config.backend == "openai_compatible"
            else common_limitations + ["No real generation credentials/model access were available for this run."]
        ),
    )
    splits = {case.split for case in all_cases}
    selected_split = next(iter(splits)) if len(splits) == 1 else "all"
    return EvaluationSuiteReport(
        evaluation_label="phase1-baseline-suite",
        asset_status=asset_status,
        selected_split=selected_split,
        development_case_count=sum(case.split == "development" for case in all_cases),
        held_out_case_count=sum(case.split == "held_out" for case in all_cases),
        labels_status=(
            "human_reviewed"
            if all(case.relevance_label_status == "human_reviewed" for case in all_cases)
            else "human_review_pending"
            if any(case.relevance_label_status == "human_review_pending" for case in all_cases)
            else "provisional_generated"
        ),
        passed=all(result.passed for result in case_results),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        code_revision=_code_revision(),
        configuration={
            "baseline_mode": "wait_for_complete_utterance",
            "early_retrieval": "absent",
            "evaluation_asset": evaluation_asset or "embedded_case_sequence",
            "index": {
                "manifest_version": corpus.manifest.manifest_version,
                "index_id": corpus.manifest.index_id,
                "build_fingerprint": corpus.manifest.build_fingerprint,
                "source_fingerprint": corpus.manifest.source_fingerprint,
            },
            "embedding": corpus.manifest.embedding.model_dump(mode="json"),
            "retrieval": {
                "backend": selected_backend,
                "top_k": selected_top_k,
            },
            "chunking": {
                "max_chars": settings.chunk_max_chars,
                "overlap_chars": settings.chunk_overlap_chars,
            },
            "generation": {
                "backend": generation_config.backend,
                "provider": generation_config.provider,
                "model": generation_config.model,
                "timeout_s": generation_config.timeout_s,
                "max_retries": generation_config.max_retries,
                "max_repair_attempts": generation_config.max_repair_attempts,
                "max_output_tokens": generation_config.max_output_tokens,
                "api_key_env": generation_config.api_key_env,
                "api_key_configured": bool(
                    os.environ.get(generation_config.api_key_env or "")
                ),
                "input_price_per_million": generation_config.input_price_per_million,
                "output_price_per_million": generation_config.output_price_per_million,
            },
            "labels_status": "provisional_generated",
            "claim_support_status": "not_evaluated",
        },
        corpus_id=corpus.corpus_id,
        index_id=corpus.manifest.index_id,
        model_identities={
            "retrieval": ":".join(
                value
                for value in (
                    corpus.manifest.embedding.provider,
                    corpus.manifest.embedding.model_name or corpus.manifest.embedding.backend,
                    corpus.manifest.embedding.revision,
                )
                if value
            ),
            "generation": f"{generation_config.provider}:{generation_config.model}",
        },
        execution_mode=execution_mode,  # type: ignore[arg-type]
        warm_cold_condition=(
            "cold CLI process; index/retriever/provider initialized once and reused across cases; "
            "per-case timings exclude process startup"
        ),
        environment=_environment(),
        hardware=_hardware(),
        sections={
            "mock": mock_section,
            "fixture": fixture_section,
            "real_model": real_section,
        },
        official_assets_available=asset_status == "official",
        limitations=common_limitations,
    )


def write_report(path: Path, report: EvaluationReport | EvaluationSuiteReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
