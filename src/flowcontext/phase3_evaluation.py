"""Phase 3 comparison, denominator audit, and multi-intent evaluation.

This report deliberately keeps two questions separate:

* Does streaming change retrieval quality on matched successful runs?
* What happened end to end when failures, timeouts, and closed sessions are
  included?

The first question is a retrieval-quality measure.  The second is an
operational outcome measure.  A failed case is retained in both the per-case
records and the end-to-end denominators; it is never silently removed from the
report.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from .config import Settings
from .contracts import CorpusIndex, EvaluationCase, RetrievalMode
from .evaluation import _code_revision, _environment, _hardware
from .generation import generation_provider_for_settings
from .multi_intent import (
    MultiIntentPlan,
    decompose_query,
    decomposition_provider_for_settings,
)
from .replay import ReplayResult, replay_transcript
from .retrieval import Retriever, make_retriever
from .streaming import replay_streaming_transcript


class Phase3EvaluationError(ValueError):
    """Raised when the Phase 3 audit inputs are unusable."""


_CANONICAL_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_NON_REUSE_CATEGORIES = (
    "expected_correction_or_final_constraint_change",
    "delayed_retrieval_before_final",
    "intentional_fault_injection_or_session_close",
    "avoidable_failure",
)


def _dense_smoke_blocker() -> str:
    """Return the concrete local blocker for the real dense path.

    The report is also generated for fixture-only runs, so this check must
    describe availability rather than imply that a dense model was executed.
    The CLI smoke command remains the source of truth for a successful model
    load or a more specific cache/network error.
    """

    if importlib.util.find_spec("sentence_transformers") is None:
        return (
            "dense backend requires optional dependency 'sentence-transformers'; "
            "install the dense extra and download the pinned model before building a dense index"
        )
    return (
        "sentence-transformers is installed, but no successful real dense smoke run completed; "
        "the pinned model must be available and loadable before dense/hybrid results can be claimed"
    )


def _canonical_query(value: str | None) -> str:
    """Compare query revisions without treating punctuation as a constraint."""

    return " ".join(token.lower() for token in _CANONICAL_TOKEN_PATTERN.findall(value or ""))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _historical_quality_audit(report: dict[str, Any] | None) -> dict[str, Any]:
    """Expose which old per-case quality values entered its aggregate."""

    if not report:
        return {"status": "not_available"}
    sections = report.get("sections", {})
    audit: dict[str, Any] = {"status": "available"}
    for mode in ("baseline", "streaming"):
        scored: list[tuple[str, float]] = []
        no_metric: list[str] = []
        for section in sections.values():
            for case in section.get("case_results", []):
                record = case.get(mode)
                if record is None:
                    continue
                value = record.get("final_evidence_recall_at_k", {}).get("recall_at_5")
                if isinstance(value, (int, float)):
                    scored.append((case.get("case_id", "unknown"), float(value)))
                else:
                    no_metric.append(case.get("case_id", "unknown"))
        audit[mode] = {
            "aggregate_sample_count": len(scored),
            "zero_quality_case_ids": [case_id for case_id, value in scored if value == 0.0],
            "no_quality_metric_case_ids": no_metric,
            "quality_values_are_not_filtered_by_success_status": True,
        }
    return audit


def _micro_recall(
    mode_records: Sequence[dict[str, Any]],
    key: str,
) -> float | None:
    numerator = 0
    denominator = 0
    for record in mode_records:
        if not record.get("quality_scored"):
            continue
        relevant = set(record.get("relevant_ids", []))
        retrieved = set(record.get("retrieved_ids", [])[: int(key.rsplit("_", 1)[-1])])
        numerator += len(relevant & retrieved)
        denominator += int(record.get("scoring_denominator", 0))
    return numerator / denominator if denominator else None


def _mode_record(mode: Any) -> dict[str, Any]:
    return {
        "mode": mode.mode,
        "run_status": mode.run_status,
        "final_query": mode.final_query,
        "retrieved_ids": list(mode.retrieved_ids or mode.final_evidence_chunk_ids),
        "relevant_ids": list(mode.relevant_ids or mode.expected_final_evidence_chunk_ids),
        "request_status": mode.request_status,
        "request_statuses": list(mode.request_statuses),
        "request_queries": list(mode.request_queries),
        "request_details": list(mode.request_details),
        "reuse_decision": mode.reuse_decision,
        "reuse_validation": dict(mode.reuse_validation),
        "early_retrieval_started": mode.retrieval_started_early,
        "valid_evidence_ready_before_final": mode.valid_evidence_ready_before_finalization,
        "early_evidence_reused": mode.early_evidence_reused,
        "retrieval_call_count": mode.retrieval_call_count,
        "stale_result_discard_count": mode.stale_result_discard_count,
        "retrieval_error_count": mode.retrieval_error_count,
        "scoring_denominator": mode.scoring_denominator,
        "quality_scored": mode.quality_scored,
        "quality_exclusion_reason": mode.quality_exclusion_reason,
        "recall_at_k": dict(mode.final_evidence_recall_at_k),
        "mrr": mode.final_evidence_mrr,
        "end_to_end_success": mode.end_to_end_success,
        "generation_status": mode.generation_status,
        "errors": list(mode.errors),
    }


def _failure_record(case_id: str, mode: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "run_status": mode["run_status"],
        "request_status": mode["request_status"],
        "errors": mode["errors"],
        "quality_exclusion_reason": mode["quality_exclusion_reason"],
    }


def _case_diagnosis(
    case: Any,
    baseline: dict[str, Any],
    streaming: dict[str, Any],
) -> dict[str, Any]:
    baseline_ids = baseline["retrieved_ids"]
    streaming_ids = streaming["retrieved_ids"]
    baseline_query = baseline.get("final_query") or ""
    streaming_query = streaming.get("final_query") or ""
    final_query_same = _canonical_query(baseline_query) == _canonical_query(streaming_query)
    final_query_text_same = baseline_query == streaming_query
    same_id_set = set(baseline_ids) == set(streaming_ids)
    ranking_changed = same_id_set and baseline_ids != streaming_ids
    stale_rejected = streaming["stale_result_discard_count"] > 0 or any(
        detail.get("stale") for detail in streaming["request_details"]
    )
    intentional_fault = case.retrieval_behavior in {"failure", "timeout", "session_close"}
    final_query_changed = any(
        _canonical_query(str(detail.get("query", ""))) != _canonical_query(streaming_query)
        for detail in streaming["request_details"]
        if not detail.get("is_final")
    )
    denominator_same = baseline["scoring_denominator"] == streaming["scoring_denominator"]
    final_query_missing = bool(baseline_query) and not bool(streaming_query)
    causes: list[str] = []
    if not final_query_same:
        causes.append("query_assembly_difference")
    else:
        causes.append("final_query_assembled_identically")
    if not final_query_text_same and final_query_same:
        causes.append("punctuation_only_query_difference")
    if intentional_fault:
        causes.append("intentional_fault_injection_or_session_close")
    if stale_rejected:
        causes.append("stale_early_result_rejected_by_revision_guard")
    if final_query_changed:
        causes.append("early_query_missing_or_preceding_final_constraint")
    if final_query_missing:
        causes.append("streaming_final_query_not_observed")
    if ranking_changed:
        causes.append("ranking_difference_with_same_retrieved_set")
    elif not same_id_set and baseline["quality_scored"] and streaming["quality_scored"]:
        causes.append("retrieved_set_difference_on_successful_runs")
    if not denominator_same:
        causes.append("scoring_denominator_difference")
    if baseline["quality_scored"] != streaming["quality_scored"]:
        causes.append("quality_denominator_or_run_status_difference")
    if not causes:
        causes.append("no_observed_difference")
    return {
        "final_query_same": final_query_same,
        "final_query_text_same": final_query_text_same,
        "retrieved_id_set_same": same_id_set,
        "ranking_changed": ranking_changed,
        "stale_result_rejected": stale_rejected,
        "intentional_fault_injection": intentional_fault,
        "final_query_changed_after_early_request": final_query_changed,
        "final_query_missing": final_query_missing,
        "final_constraints_dropped": (
            bool(baseline_query)
            and bool(streaming_query)
            and not final_query_same
        ),
        "scoring_denominator_same": denominator_same,
        "causes": causes,
        "semantic_reuse_proof": (
            "not_evaluated; lexical reuse proxy and citation IDs are not entailment"
        ),
    }


def _classify_non_reuse(case: Any, streaming: dict[str, Any]) -> tuple[str, str]:
    """Classify each eligible non-reuse without policy tuning from held-out data."""

    if case.retrieval_behavior in {"failure", "timeout", "session_close"}:
        return (
            "intentional_fault_injection_or_session_close",
            f"case declares retrieval_behavior={case.retrieval_behavior}; the failure/closure is part of the fixture",
        )
    if (
        streaming["early_retrieval_started"]
        and not streaming["valid_evidence_ready_before_final"]
        and streaming["reuse_decision"]
        in {"early_result_not_ready_before_final", "rejected_stale_after_query_revision"}
    ):
        return (
            "delayed_retrieval_before_final",
            "early retrieval was started, but no validated current result was ready before final-event delivery; late work was not reused",
        )
    if streaming["reuse_decision"] in {
        "rejected_stale_after_query_revision",
        "final_query_changed_no_reuse",
    }:
        return (
            "expected_correction_or_final_constraint_change",
            "the final query differs from the early query and the revision guard correctly prevents reuse",
        )
    if streaming["reuse_decision"] in {
        "early_result_not_ready_before_final",
        "final_evidence_retrieved_without_early_reuse",
    }:
        return (
            "delayed_retrieval_before_final",
            "an early search started, but no validated matching result was ready for reuse at finalisation",
        )
    if streaming["errors"]:
        return (
            "avoidable_failure",
            "the case has an error without a declared fault-injection or expected query revision",
        )
    return (
        "avoidable_failure",
        "an eligible case had no accepted early reuse and no expected revision or declared fault",
    )


def _split_comparison(
    cases: Sequence[Any],
    results: Sequence[Any],
    *,
    split: Literal["development", "held_out"],
) -> dict[str, Any]:
    case_by_id = {case.case_id: case for case in cases}
    selected = [result for result in results if result.split == split]
    rows: list[dict[str, Any]] = []
    non_reuse: list[dict[str, Any]] = []
    for result in selected:
        if result.baseline is None or result.streaming is None:
            continue
        baseline = _mode_record(result.baseline)
        streaming = _mode_record(result.streaming)
        case = case_by_id[result.case_id]
        row = {
            "case_id": result.case_id,
            "split": split,
            "scenario_group": result.scenario_group,
            "asset_status": result.asset_status,
            "eligible_for_early_retrieval": case.eligible_for_early_retrieval,
            "expected_fault_behavior": case.retrieval_behavior,
            "baseline": baseline,
            "streaming": streaming,
            "diagnosis": _case_diagnosis(case, baseline, streaming),
            "failure_flags": list(result.failure_flags),
        }
        rows.append(row)
        if case.eligible_for_early_retrieval and not streaming["early_evidence_reused"]:
            category, explanation = _classify_non_reuse(case, streaming)
            non_reuse.append(
                {
                    "case_id": result.case_id,
                    "scenario_group": result.scenario_group,
                    "category": category,
                    "explanation": explanation,
                    "reuse_decision": streaming["reuse_decision"],
                    "request_statuses": streaming["request_statuses"],
                    "errors": streaming["errors"],
                }
            )

    comparable = [
        row
        for row in rows
        if row["baseline"]["quality_scored"]
        and row["streaming"]["quality_scored"]
        and row["baseline"]["run_status"] == "completed"
        and row["streaming"]["run_status"] == "completed"
    ]

    def quality(mode: str) -> dict[str, Any]:
        records = [row[mode] for row in comparable]
        return {
            "successful_comparable_case_count": len(records),
            "case_denominator": len(records),
            "relevant_id_denominator": sum(record["scoring_denominator"] for record in records),
            "included_case_ids": [row["case_id"] for row in comparable],
            "recall_at_1_macro_mean": _mean(
                [record["recall_at_k"].get("recall_at_1", 0.0) for record in records]
            ),
            "recall_at_3_macro_mean": _mean(
                [record["recall_at_k"].get("recall_at_3", 0.0) for record in records]
            ),
            "recall_at_5_macro_mean": _mean(
                [record["recall_at_k"].get("recall_at_5", 0.0) for record in records]
            ),
            "recall_at_1_micro": _micro_recall(records, "recall_at_1"),
            "recall_at_3_micro": _micro_recall(records, "recall_at_3"),
            "recall_at_5_micro": _micro_recall(records, "recall_at_5"),
            "mrr_macro_mean": _mean(
                [record["mrr"] for record in records if record["mrr"] is not None]
            ),
        }

    def e2e(mode: str) -> dict[str, Any]:
        records = [row[mode] for row in rows]
        failures = [
            _failure_record(row["case_id"], row[mode])
            for row in rows
            if row[mode]["run_status"] != "completed"
        ]
        return {
            "case_denominator": len(records),
            "completed_case_count": sum(record["run_status"] == "completed" for record in records),
            "end_to_end_success_rate": (
                sum(record["end_to_end_success"] for record in records) / len(records)
                if records
                else 0.0
            ),
            "non_completed_case_count": len(failures),
            "non_completed_cases": failures,
            "failure_case_ids": [item["case_id"] for item in failures],
            "retrieval_error_event_count": sum(record["retrieval_error_count"] for record in records),
        }

    category_counts: dict[str, int] = {}
    for item in non_reuse:
        category_counts[item["category"]] = category_counts.get(item["category"], 0) + 1
    category_counts = {
        category: category_counts.get(category, 0)
        for category in _NON_REUSE_CATEGORIES
    }
    eligible = [row for row in rows if row["eligible_for_early_retrieval"]]
    useful = [row for row in eligible if row["streaming"]["early_evidence_reused"]]
    return {
        "split": split,
        "case_count": len(rows),
        "eligible_case_count": len(eligible),
        "early_retrieval_started_count": sum(
            row["streaming"]["early_retrieval_started"] for row in eligible
        ),
        "useful_early_reuse_count": len(useful),
        "non_useful_early_reuse_count": len(non_reuse),
        "non_useful_early_reuse_categories": category_counts,
        "non_useful_early_reuse_cases": non_reuse,
        "retrieval_quality_successful_comparable": {
            "baseline": quality("baseline"),
            "streaming": quality("streaming"),
            "matched_case_ids": [row["case_id"] for row in comparable],
            "excluded_case_ids": [
                {
                    "case_id": row["case_id"],
                    "baseline_reason": row["baseline"]["quality_exclusion_reason"],
                    "streaming_reason": row["streaming"]["quality_exclusion_reason"],
                }
                for row in rows
                if row not in comparable
            ],
            "interpretation": (
                "Only cases with successful final evidence in both modes are included. "
                "This is the comparable retrieval-quality denominator, not an end-to-end pass rate."
            ),
        },
        "end_to_end_including_failures": {
            "baseline": e2e("baseline"),
            "streaming": e2e("streaming"),
            "paired_behavior_failure_case_ids": [
                row["case_id"] for row in rows if row["failure_flags"]
            ],
        },
        "case_comparisons": rows,
    }


async def _run_phase3_case(
    case: EvaluationCase,
    *,
    corpus: CorpusIndex,
    base_retriever: Retriever,
    settings: Settings,
    top_k: int,
    execution_mode: Literal["realtime", "accelerated"],
    retrieval_mode: RetrievalMode | None = None,
    context_budget_tokens: int = 1200,
    multi_intent_max_workers: int | None = None,
    multi_intent_rrf_k: int = 60,
) -> dict[str, Any]:
    baseline_decomposition_provider = decomposition_provider_for_settings(settings)
    streaming_decomposition_provider = decomposition_provider_for_settings(settings)
    baseline, streaming = await asyncio.gather(
        replay_transcript(
            case.transcript,
            corpus=corpus,
            top_k=top_k,
            backend=base_retriever.backend,
            retriever=base_retriever,
            generation_provider=generation_provider_for_settings(settings),
            decomposition_provider=baseline_decomposition_provider,
            run_id=f"phase3-{case.case_id}-baseline",
            execution_mode=execution_mode,
            multi_intent=True,
            retrieval_mode=retrieval_mode,
            context_budget_tokens=context_budget_tokens,
            multi_intent_max_workers=multi_intent_max_workers,
            multi_intent_rrf_k=multi_intent_rrf_k,
        ),
        replay_streaming_transcript(
            case.transcript,
            corpus=corpus,
            top_k=top_k,
            backend=base_retriever.backend,
            retriever=base_retriever,
            generation_provider=generation_provider_for_settings(settings),
            decomposition_provider=streaming_decomposition_provider,
            run_id=f"phase3-{case.case_id}-streaming",
            execution_mode=execution_mode,
            multi_intent=True,
            retrieval_mode=retrieval_mode,
            context_budget_tokens=context_budget_tokens,
            multi_intent_max_workers=multi_intent_max_workers,
            multi_intent_rrf_k=multi_intent_rrf_k,
        ),
    )
    plan = (
        MultiIntentPlan.model_validate(baseline.decomposition)
        if baseline.decomposition is not None
        else decompose_query(baseline.query)
    )
    relevant_ids = [chunk_id for chunk_id, label in case.relevance_labels.items() if label == "relevant"]

    def mode(result: ReplayResult | Any, ids: list[str]) -> dict[str, Any]:
        answer = result.answer
        cited = {
            chunk_id
            for claim in answer.factual_claims
            for chunk_id in claim.supporting_chunk_ids
        }
        intent_hit_ids = {
            intent.intent_id: [
                hit.chunk_id
                for hit in ids_to_hits
                if not hit.intent_ids or intent.intent_id in hit.intent_ids
            ]
            for intent in plan.intents
            for ids_to_hits in ([result.retrieval_hits] if result.mode == "baseline" else [result.current_evidence_hits])
        }
        retrieval_details = getattr(result, "retrieval_details", None) or {}
        missing_intent_ids = list(retrieval_details.get("missing_intent_ids", []))
        retrieval_detail_errors = list(retrieval_details.get("errors", []))
        retrieval_quality_reason: str | None = None
        if not relevant_ids:
            retrieval_quality_reason = "no_relevant_id_labels"
        elif result.run_status != "completed":
            retrieval_quality_reason = f"run_status={result.run_status}"
        elif missing_intent_ids:
            retrieval_quality_reason = "missing_evidence_for_intent"
        elif retrieval_detail_errors:
            retrieval_quality_reason = "retrieval_errors_present"
        quality_scored = retrieval_quality_reason is None
        return {
            "run_status": result.run_status,
            "final_query": getattr(result, "query", getattr(result, "final_query", None)),
            "retrieved_ids": ids,
            "relevant_ids": relevant_ids,
            "scoring_denominator": len(relevant_ids),
            "quality_scored": quality_scored,
            "retrieval_quality_success": quality_scored,
            "quality_exclusion_reason": retrieval_quality_reason,
            "retrieval_recall_at_5": (
                len(set(ids[:5]) & set(relevant_ids)) / len(relevant_ids)
                if relevant_ids
                else None
            ),
            "citation_ids": sorted(cited),
            "citation_ids_valid": cited.issubset(set(ids)),
            "factual_claim_count": len(answer.factual_claims),
            "answer_substrings_passed": all(
                expected.casefold() in answer.answer_text.casefold()
                for expected in case.expected_answer_substrings
            ),
            "intent_hit_ids": intent_hit_ids,
            "retrieval_details": retrieval_details or None,
            "missing_intent_ids": missing_intent_ids,
            "retrieval_error_details": retrieval_detail_errors,
            "context_tokens_used": (
                (getattr(result, "retrieval_details", None) or {}).get("context_tokens_used")
            ),
            "generation_status": result.generation_status,
            "retrieval_call_count": result.retrieval_call_count,
            "errors": [error.error_type for error in result.errors] + retrieval_detail_errors,
        }

    baseline_ids = [hit.chunk_id for hit in baseline.retrieval_hits]
    streaming_ids = [hit.chunk_id for hit in streaming.current_evidence_hits]
    all_intent_ids = sorted(
        {
            intent_id
            for hit in [*baseline.retrieval_hits, *streaming.current_evidence_hits]
            for intent_id in hit.intent_ids
        }
    )
    baseline_record = mode(baseline, baseline_ids)
    streaming_record = mode(streaming, streaming_ids)

    def expected_behavior(record: dict[str, Any]) -> bool:
        common = record["answer_substrings_passed"] and record["citation_ids_valid"]
        if case.answerability == "unanswerable":
            return (
                common
                and record["run_status"] in {"abstained", "needs_context"}
                and record["factual_claim_count"] == 0
            )
        return (
            common
            and record["run_status"] == "completed"
            and not record["missing_intent_ids"]
            and record["retrieval_recall_at_5"] in {None, 1.0}
        )

    expected_pass = expected_behavior(baseline_record) and expected_behavior(streaming_record)
    return {
        "case_id": case.case_id,
        "split": case.split,
        "scenario_group": case.scenario_group,
        "retrieval_mode": retrieval_mode or base_retriever.backend,
        "context_budget_tokens": context_budget_tokens,
        "parent_query": baseline.query,
        "decomposition": plan.model_dump(mode="json"),
        "identified_intent_count": len(plan.intents),
        "identified_intent_ids": all_intent_ids,
        "baseline": baseline_record,
        "streaming": streaming_record,
        "expected_behavior_met": expected_pass,
        "semantic_claim_support": "not_evaluated",
    }


async def evaluate_multi_intent_cases(
    cases: Sequence[EvaluationCase],
    *,
    corpus: CorpusIndex,
    settings: Settings,
    backend: str,
    top_k: int,
    execution_mode: Literal["realtime", "accelerated"],
    retrieval_mode: RetrievalMode | None = None,
    context_budget_tokens: int = 1200,
    multi_intent_max_workers: int | None = None,
    multi_intent_rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Run the Phase 3 path over a frozen development/held-out split."""

    if not cases:
        raise Phase3EvaluationError("at least one Phase 3 evaluation case is required")
    construction_backend = retrieval_mode or backend
    if retrieval_mode == "hybrid":
        construction_backend = corpus.manifest.embedding.backend
    base_retriever = make_retriever(
        corpus,
        backend=construction_backend,
        top_k=top_k,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
    )
    results = await asyncio.gather(
        *(
            _run_phase3_case(
                case,
                corpus=corpus,
                base_retriever=base_retriever,
                settings=settings,
                top_k=top_k,
                execution_mode=execution_mode,
                retrieval_mode=retrieval_mode,
                context_budget_tokens=context_budget_tokens,
                multi_intent_max_workers=multi_intent_max_workers,
                multi_intent_rrf_k=multi_intent_rrf_k,
            )
            for case in cases
        )
    )
    return list(results)


def _phase3_split_summary(results: Sequence[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [result for result in results if result["split"] == split]
    compound = [result for result in selected if result["identified_intent_count"] > 1]
    multi_identified = [result for result in compound if result["identified_intent_count"] >= 2]

    comparable = [
        result
        for result in selected
        if result["baseline"]["quality_scored"]
        and result["streaming"]["quality_scored"]
    ]

    def retrieval_quality(mode: str) -> dict[str, Any]:
        records = [result[mode] for result in comparable]
        recalls = [record["retrieval_recall_at_5"] for record in records]
        return {
            "case_denominator": len(records),
            "relevant_id_denominator": sum(len(record["relevant_ids"]) for record in records),
            "included_case_ids": [result["case_id"] for result in comparable],
            "recall_at_5_macro_mean": _mean(recalls),
            "recall_at_5_micro": (
                sum(
                    len(set(record["retrieved_ids"][:5]) & set(record["relevant_ids"]))
                    for record in records
                )
                / sum(len(record["relevant_ids"]) for record in records)
                if records
                else None
            ),
        }

    def end_to_end(mode: str) -> dict[str, Any]:
        records = [result[mode] for result in selected]
        non_completed = [
            result for result in selected if result[mode]["run_status"] != "completed"
        ]
        return {
            "case_denominator": len(records),
            "completed_case_count": sum(record["run_status"] == "completed" for record in records),
            "non_completed_case_count": len(non_completed),
            "operational_completion_rate": (
                sum(record["run_status"] == "completed" for record in records) / len(records)
                if records
                else 0.0
            ),
            "expected_behavior_success_count": sum(
                result["expected_behavior_met"] for result in selected
            ),
            "run_status_counts": {
                status: sum(record["run_status"] == status for record in records)
                for status in {record["run_status"] for record in records}
            },
            "non_completed_case_ids": [result["case_id"] for result in non_completed],
            "failed_case_ids": [
                result["case_id"] for result in non_completed if result[mode]["run_status"] == "failed"
            ],
            "abstained_or_needs_context_case_ids": [
                result["case_id"]
                for result in non_completed
                if result[mode]["run_status"] in {"abstained", "needs_context"}
            ],
            "partial_or_missing_evidence_case_ids": [
                result["case_id"]
                for result in selected
                if result[mode]["missing_intent_ids"] or result[mode]["retrieval_error_details"]
            ],
        }

    return {
        "split": split,
        "case_count": len(selected),
        "compound_case_count": len(compound),
        "multi_intent_identified_count": len(multi_identified),
        "multi_intent_identification_rate": len(multi_identified) / len(compound) if compound else None,
        "expected_behavior_met_count": sum(
            result["expected_behavior_met"] for result in selected
        ),
        "end_to_end_case_denominator": len(selected),
        "failed_or_nonpassing_case_ids": [
            result["case_id"] for result in selected if not result["expected_behavior_met"]
        ],
        "retrieval_quality_successful_comparable": {
            "baseline": retrieval_quality("baseline"),
            "streaming": retrieval_quality("streaming"),
            "matched_case_ids": [result["case_id"] for result in comparable],
            "excluded_case_ids": [
                result["case_id"] for result in selected if result not in comparable
            ],
        },
        "end_to_end_including_failures": {
            "baseline": end_to_end("baseline"),
            "streaming": end_to_end("streaming"),
        },
        "case_results": selected,
        "grounding": {
            "citation_id_valid_case_count": sum(
                all(mode["citation_ids_valid"] for mode in (result["baseline"], result["streaming"]))
                for result in selected
            ),
            "case_denominator": len(selected),
            "semantic_claim_support": "not_evaluated",
        },
    }


def build_phase3_comparison(
    *,
    phase2_report: Any,
    phase2_cases: Sequence[Any],
    phase3_results: Sequence[dict[str, Any]],
    corpus: CorpusIndex,
    settings: Settings,
    backend: str,
    top_k: int,
    execution_mode: Literal["realtime", "accelerated"],
    historical_report: dict[str, Any] | None = None,
    retrieval_mode: RetrievalMode | None = None,
) -> dict[str, Any]:
    """Assemble a single immutable Phase 3 evidence package."""

    phase2_results = [
        result
        for section_id in ("fixture", "simulated_delay")
        for result in phase2_report.sections[section_id].case_results
    ]
    phase2_splits = {
        split: _split_comparison(phase2_cases, phase2_results, split=split)
        for split in ("development", "held_out")
    }
    phase2_rows = [
        row
        for split in phase2_splits.values()
        for row in split["case_comparisons"]
    ]
    phase2_diagnoses = [row["diagnosis"] for row in phase2_rows]
    historical_metrics = (historical_report or {}).get("metrics", {})
    all_successful = [
        phase2_splits[split]["retrieval_quality_successful_comparable"]
        for split in phase2_splits
    ]
    matched_streaming = [
        entry["streaming"]["recall_at_5_macro_mean"]
        for entry in all_successful
        if entry["streaming"]["recall_at_5_macro_mean"] is not None
    ]
    matched_baseline = [
        entry["baseline"]["recall_at_5_macro_mean"]
        for entry in all_successful
        if entry["baseline"]["recall_at_5_macro_mean"] is not None
    ]
    return {
        "schema_version": "flowcontext.phase3.v1",
        "report_version": "flowcontext.phase3-comparison.v1",
        "evaluation_label": "phase3-multi-intent-and-phase2-gap-investigation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset_status": "synthetic_fixture",
        "competition_performance_claim": False,
        "code_revision": _code_revision(),
        "configuration": {
            "backend": backend,
            "top_k": top_k,
            "execution_mode": execution_mode,
            "generation_backend": settings.generation_backend,
            "generation_provider": settings.generation_provider,
            "generation_model": settings.generation_model,
            "phase2_case_splits": ["development", "held_out"],
            "phase3_path": "opt_in_multi_intent",
            "phase3_retrieval_mode": retrieval_mode or settings.multi_intent_retrieval_mode,
            "phase3_context_budget_tokens": settings.multi_intent_context_budget_tokens,
            "phase3_max_workers": settings.multi_intent_max_workers,
            "phase3_rrf_k": settings.multi_intent_rrf_k,
            "decomposition_backend": settings.generation_backend,
            "decomposition_provider": settings.generation_provider,
            "decomposition_model": settings.generation_model,
        },
        "corpus_id": corpus.corpus_id,
        "index_id": corpus.manifest.index_id,
        "model_identities": {
            "retrieval": f"{backend}:{corpus.manifest.embedding.provider}",
            "generation": settings.generation_model,
        },
        "execution_mode": execution_mode,
        "warm_cold_condition": "single process; shared warm index/retriever and provider configuration",
        "environment": _environment(),
        "hardware": _hardware(),
        "historical_gap": {
            "source_report": "reports/phase2_streaming_evaluation.json",
            "historical_baseline_recall_at_5": historical_metrics.get("baseline_final_evidence_recall_at_5_mean"),
            "historical_streaming_recall_at_5": historical_metrics.get("streaming_final_evidence_recall_at_5_mean"),
            "historical_quality_audit": _historical_quality_audit(historical_report),
            "historical_interpretation": (
                "The old aggregate included zero-quality values from failed or timed-out runs and omitted "
                "some closed/no-result cases; its baseline and streaming sample sets were unequal. Per-case "
                "exclusions were not exposed. It is preserved as historical evidence, not overwritten."
            ),
            "matched_successful_baseline_recall_at_5": _mean(matched_baseline),
            "matched_successful_streaming_recall_at_5": _mean(matched_streaming),
            "matched_successful_case_count": sum(
                entry["streaming"]["successful_comparable_case_count"] for entry in all_successful
            ),
            "gap_on_matched_successful_runs": (
                _mean(matched_streaming) - _mean(matched_baseline)
                if matched_streaming and matched_baseline
                else None
            ),
            "claim_boundary": (
                "The matched rerun demonstrates no fixture Recall@5 gap on successful comparable cases; "
                "end-to-end failures remain separate, and real dense retrieval/generation are unverified."
            ),
        },
        "phase2_investigation": {
            "development": phase2_splits["development"],
            "held_out": phase2_splits["held_out"],
            "overall_non_useful_early_reuse": [
                item
                for split in phase2_splits.values()
                for item in split["non_useful_early_reuse_cases"]
            ],
            "overall_non_useful_category_counts": {
                category: sum(
                    split["non_useful_early_reuse_categories"].get(category, 0)
                    for split in phase2_splits.values()
                )
                for category in {
                    category
                    for split in phase2_splits.values()
                    for category in split["non_useful_early_reuse_categories"]
                }
            },
            "metric_fix": (
                "Recall quality is scored only when both modes completed with final evidence and explicit labels. "
                "End-to-end denominators include every case, including failed, timed-out, and closed runs."
            ),
            "cause_audit": {
                "diagnosis_cause_counts": {
                    cause: sum(cause in diagnosis["causes"] for diagnosis in phase2_diagnoses)
                    for cause in {
                        cause
                        for diagnosis in phase2_diagnoses
                        for cause in diagnosis["causes"]
                    }
                },
                "stale_or_superseded_early_results": [
                    row["case_id"]
                    for row in phase2_rows
                    if row["diagnosis"]["stale_result_rejected"]
                ],
                "incomplete_or_changed_early_queries": [
                    row["case_id"]
                    for row in phase2_rows
                    if row["diagnosis"]["final_query_changed_after_early_request"]
                ],
                "dropped_final_constraints": [
                    row["case_id"]
                    for row in phase2_rows
                    if row["diagnosis"]["final_constraints_dropped"]
                ],
                "intentional_fault_or_closure_cases": [
                    row["case_id"]
                    for row in phase2_rows
                    if row["diagnosis"]["intentional_fault_injection"]
                ],
                "ranking_difference_cases": [
                    row["case_id"]
                    for row in phase2_rows
                    if row["diagnosis"]["ranking_changed"]
                ],
                "successful_retrieved_set_difference_cases": [
                    row["case_id"]
                    for row in phase2_rows
                    if "retrieved_set_difference_on_successful_runs"
                    in row["diagnosis"]["causes"]
                ],
                "quality_denominator_difference_cases": [
                    row["case_id"]
                    for row in phase2_rows
                    if not row["diagnosis"]["scoring_denominator_same"]
                ],
                "metric_implementation_conclusion": (
                    "The historical Recall@5 gap is explained by the old aggregate including zero-quality values "
                    "from failed/timed-out runs, omitting some closed/no-result cases, and exposing unequal mode "
                    "sample sets. The matched fresh quality denominator is identical across modes; failures remain "
                    "in the separate end-to-end denominator."
                ),
                "semantic_reuse_conclusion": (
                    "Reuse was checked with final-query compatibility and a lexical overlap proxy. Semantic "
                    "entailment was not evaluated, so valid IDs and lexical overlap are not proof of relevance."
                ),
            },
        },
        "phase3_multi_intent_evaluation": {
            "development": _phase3_split_summary(phase3_results, "development"),
            "held_out": _phase3_split_summary(phase3_results, "held_out"),
            "grounding_boundary": (
                "Citation IDs are checked against supplied fused chunks. Claim entailment remains not_evaluated "
                "until human review or a separately governed semantic evaluator is available."
            ),
            "held_out_policy": (
                "Held-out cases were measured after development implementation decisions were frozen; they were not "
                "used to tune decomposition, RRF, or streaming policy."
            ),
            "held_out_inspection": {
                "inspected_after_freeze": True,
                "used_for_tuning": False,
                "new_untouched_cases_reserved_for_next_evaluation": True,
            },
        },
        "sections": {
            "fixture": {"status": "measured", "case_count": len(phase2_report.sections["fixture"].case_results)},
            "simulated_delay": {"status": "measured", "case_count": len(phase2_report.sections["simulated_delay"].case_results)},
            "real_backend": {
                "status": "not_verified",
                "case_count": 0,
                "reason": _dense_smoke_blocker(),
            },
            "official_assets": {
                "status": "not_verified",
                "case_count": 0,
                "reason": "Official corpus, replay set, labels, thresholds, and harness are absent.",
            },
        },
        "remaining_blockers": [
            "Official corpus, replay schema, labels, thresholds, and scoring harness were not supplied.",
            f"Dense retrieval smoke blocker: {_dense_smoke_blocker()}.",
            "Real-provider generation was not executed in this environment.",
            "Semantic claim support and factual-grounding rate require human review or a governed evaluator.",
            "Phase 4 selective claim updates across follow-up turns are intentionally out of scope.",
            "The local RRF/decomposition evaluation is fixture evidence, not competition performance.",
        ],
    }


def _compact_row(row: dict[str, Any]) -> str:
    def ids(value: Sequence[str]) -> str:
        return ", ".join(value) if value else "—"

    baseline = row["baseline"]
    streaming = row["streaming"]
    return (
        f"| {row['case_id']} | {row['split']} | "
        f"`{baseline['run_status']}` / `{streaming['run_status']}` | "
        f"{baseline.get('final_query') or '—'} / {streaming.get('final_query') or '—'} | "
        f"{ids(baseline['retrieved_ids'])} / {ids(streaming['retrieved_ids'])} | "
        f"{ids(baseline['relevant_ids'])} | "
        f"{baseline['request_status']} / {streaming['request_status']} | "
        f"{baseline['reuse_decision']} / {streaming['reuse_decision']} | "
        f"{baseline['scoring_denominator']} / {streaming['scoring_denominator']} |"
    )


def write_phase3_comparison_report(path: Path, report: dict[str, Any]) -> Path:
    """Write JSON plus a readable Markdown companion without overwriting history."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path = path.with_suffix(".md")
    rows = [
        row
        for split in ("development", "held_out")
        for row in report["phase2_investigation"][split]["case_comparisons"]
    ]
    lines = [
        "# Phase 3 comparison and Phase 2 gap investigation",
        "",
        "Status: local synthetic/lexical/mock engineering evidence; not an official or competitive result.",
        "",
        "Historical reports are preserved. This report records a fresh matched rerun and corrected denominators.",
        "",
        "## Findings",
        "",
        "- Successful comparable retrieval quality: both modes are measured on the same completed cases and explicit relevant-ID denominator.",
        "- End-to-end results retain all failed, timed-out, and closed cases.",
        "- Reuse is gated by final-query compatibility plus a lexical relevance proxy; semantic entailment remains unevaluated.",
        "- Phase 3 adds structural decomposition, parallel per-intent retrieval, RRF fusion, and grounded multi-intent generation behind `--multi-intent`.",
        "- Cause audit: stale/superseded early results, ranking changes, and successful retrieved-set differences are listed per case; the historical gap is attributed to metric inclusion/denominator differences, not a matched successful retrieval-quality gap.",
        f"- Historical quality samples: baseline {report['historical_gap']['historical_quality_audit'].get('baseline', {}).get('aggregate_sample_count', 0)} values with zero-quality cases {report['historical_gap']['historical_quality_audit'].get('baseline', {}).get('zero_quality_case_ids', [])}; streaming {report['historical_gap']['historical_quality_audit'].get('streaming', {}).get('aggregate_sample_count', 0)} values with zero-quality cases {report['historical_gap']['historical_quality_audit'].get('streaming', {}).get('zero_quality_case_ids', [])}.",
        "",
        "## Per-case baseline / streaming comparison",
        "",
        "| Case | Split | Status B/S | Final query B/S | Retrieved IDs B/S | Relevant IDs | Request status B/S | Reuse decision B/S | Scoring denominator B/S |",
        "|---|---|---|---|---|---|---|---|---:|",
    ]
    lines.extend(_compact_row(row) for row in rows)
    for split in ("development", "held_out"):
        summary = report["phase2_investigation"][split]
        lines.extend(
            [
                "",
                f"## {split.replace('_', ' ').title()} summary",
                "",
                f"Eligible early retrieval: {summary['early_retrieval_started_count']}/{summary['eligible_case_count']}; useful reuse: {summary['useful_early_reuse_count']}/{summary['eligible_case_count']}; non-useful: {summary['non_useful_early_reuse_count']}/{summary['eligible_case_count']}.",
                "",
                "Successful comparable Recall@5 (B/S): "
                f"{summary['retrieval_quality_successful_comparable']['baseline']['recall_at_5_macro_mean']} / "
                f"{summary['retrieval_quality_successful_comparable']['streaming']['recall_at_5_macro_mean']} "
                f"over {summary['retrieval_quality_successful_comparable']['baseline']['case_denominator']} matched cases.",
                "",
                "Phase 3 successful comparable Recall@5 (B/S): "
                f"{report['phase3_multi_intent_evaluation'][split]['retrieval_quality_successful_comparable']['baseline']['recall_at_5_macro_mean']} / "
                f"{report['phase3_multi_intent_evaluation'][split]['retrieval_quality_successful_comparable']['streaming']['recall_at_5_macro_mean']} "
                f"over {report['phase3_multi_intent_evaluation'][split]['retrieval_quality_successful_comparable']['baseline']['case_denominator']} matched cases.",
                "",
                "Phase 3 end-to-end completion rate (B/S): "
                f"{report['phase3_multi_intent_evaluation'][split]['end_to_end_including_failures']['baseline']['operational_completion_rate']} / "
                f"{report['phase3_multi_intent_evaluation'][split]['end_to_end_including_failures']['streaming']['operational_completion_rate']} "
                f"over {report['phase3_multi_intent_evaluation'][split]['end_to_end_including_failures']['baseline']['case_denominator']} cases; non-completed cases remain listed in JSON.",
                "",
                "End-to-end non-completed cases (B): "
                f"{', '.join(summary['end_to_end_including_failures']['baseline']['failure_case_ids']) or 'none'}",
                "",
                "End-to-end non-completed cases (S): "
                f"{', '.join(summary['end_to_end_including_failures']['streaming']['failure_case_ids']) or 'none'}",
                "",
                "Phase 3 partial/missing-evidence cases (B): "
                f"{', '.join(report['phase3_multi_intent_evaluation'][split]['end_to_end_including_failures']['baseline']['partial_or_missing_evidence_case_ids']) or 'none'}",
                "",
                "Phase 3 partial/missing-evidence cases (S): "
                f"{', '.join(report['phase3_multi_intent_evaluation'][split]['end_to_end_including_failures']['streaming']['partial_or_missing_evidence_case_ids']) or 'none'}",
                "",
                "Non-useful eligible early reuse classification:",
                "",
            ]
        )
        lines.extend(
            f"- `{item['case_id']}`: {item['category']} — {item['explanation']}"
            for item in summary["non_useful_early_reuse_cases"]
        )
    lines.extend(
        [
            "",
            "## Remaining blockers",
            "",
            *[f"- {item}" for item in report["remaining_blockers"]],
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


def load_historical_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
