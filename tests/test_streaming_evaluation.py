from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from flowcontext.config import load_settings
from flowcontext.contracts import PreviousAnswerContext
from flowcontext.ingestion import CorpusIngestor, load_document_inputs
from flowcontext.replay import replay_transcript
from flowcontext.retrieval import make_retriever
from flowcontext.streaming_evaluation import (
    evaluate_streaming_suite,
    load_streaming_evaluation_cases,
)


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "data" / "synthetic"
EVALUATION = ROOT / "data" / "evaluation"


class StreamingEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CorpusIngestor().ingest(
            load_document_inputs(SYNTHETIC / "documents.jsonl"),
            "synthetic_fixture",
        )

    def test_dedicated_assets_have_frozen_splits_and_pre_execution_eligibility(self) -> None:
        development = load_streaming_evaluation_cases(
            EVALUATION / "streaming_development.jsonl",
            expected_split="development",
        )
        held_out = load_streaming_evaluation_cases(
            EVALUATION / "streaming_held_out.jsonl",
            expected_split="held_out",
        )
        self.assertGreaterEqual(len(development), 20)
        self.assertGreaterEqual(len(held_out), 8)
        self.assertTrue(all(case.relevance_label_status == "provisional_generated" for case in development + held_out))
        self.assertTrue(all(case.split == "development" for case in development))
        self.assertTrue(all(case.split == "held_out" for case in held_out))
        eligible = [case for case in development + held_out if case.eligible_for_early_retrieval]
        self.assertTrue(eligible)
        self.assertTrue(all(case.earliest_reasonable_retrieval_event_id for case in eligible))
        self.assertTrue(any(case.concurrency_group for case in development + held_out))

    def test_matched_audit_reports_early_start_and_disabled_policy_comparison(self) -> None:
        cases = load_streaming_evaluation_cases(EVALUATION / "streaming_development.jsonl")
        selected = [
            case
            for case in cases
            if case.case_id in {
                "stream-dev-stable-partial",
                "stream-dev-repeated",
                "stream-dev-greeting",
                "stream-dev-format-no-context",
            }
        ]
        report = asyncio.run(
            evaluate_streaming_suite(
                selected,
                corpus=self.index,
                settings=load_settings(),
                backend="lexical",
                top_k=5,
                execution_mode="accelerated",
                compare_without_suppression=True,
            )
        )
        self.assertEqual(report.selected_split, "development")
        self.assertTrue(report.competition_performance_claim is False)
        self.assertEqual(set(report.sections), {"fixture", "simulated_delay", "real_backend", "official_assets"})
        self.assertEqual(report.metrics["stale_result_acceptance_count"], 0.0)
        self.assertEqual(report.metrics["false_retrieval_trigger_count"], 0.0)
        self.assertEqual(report.metrics["early_retrieval_denominator"], 2.0)
        self.assertEqual(report.metrics["early_retrieval_numerator"], 2.0)
        self.assertEqual(report.duplicate_suppression_comparison["status"], "measured")
        self.assertTrue(report.duplicate_suppression_comparison["wasted_request_reduction_observed"])

    def test_baseline_formatting_context_skips_corpus_retrieval(self) -> None:
        events = load_streaming_evaluation_cases(
            EVALUATION / "streaming_development.jsonl",
            expected_split="development",
        )
        case = next(item for item in events if item.case_id == "stream-dev-format-context")
        retriever = make_retriever(self.index, backend="lexical", top_k=5)
        result = asyncio.run(
            replay_transcript(
                case.transcript,
                corpus=self.index,
                backend="lexical",
                retriever=retriever,
                previous_answer_context=PreviousAnswerContext(
                    session_id=case.session_id,
                    answer_text="A prior answer",
                    answer_version=1,
                ),
            )
        )
        self.assertEqual(result.retrieval_call_count, 0)
        self.assertEqual(result.run_status, "completed")
        self.assertEqual(result.answer.answer_text, "A prior answer")
        self.assertTrue(any(
            trace.attributes.get("reason") == "formatting_with_previous_answer_context"
            for trace in result.traces
            if trace.event_type == "retrieval_decision"
        ))

    def test_timeout_case_uses_declared_timeout_and_stays_explicit(self) -> None:
        cases = load_streaming_evaluation_cases(
            EVALUATION / "streaming_development.jsonl",
            expected_split="development",
        )
        case = next(item for item in cases if item.case_id == "stream-dev-retrieval-timeout")
        report = asyncio.run(
            evaluate_streaming_suite(
                [case],
                corpus=self.index,
                settings=load_settings(),
                backend="lexical",
                top_k=5,
                execution_mode="accelerated",
            )
        )
        result = report.sections["simulated_delay"].case_results[0].streaming
        self.assertGreaterEqual(result.timed_out_request_count, 1)
        self.assertIn("RetrievalTimeout", result.errors)
        self.assertEqual(result.stale_result_accepted_count, 0)
