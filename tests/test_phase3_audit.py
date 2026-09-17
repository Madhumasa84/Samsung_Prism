from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path

from flowcontext.config import load_settings
from flowcontext.ingestion import CorpusIngestor, load_document_inputs
from flowcontext.phase3_audit import (
    Phase3AuditError,
    load_phase3_evaluation_cases,
    _matching_rubric,
    evaluate_phase3_audit,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "data" / "evaluation"
SYNTHETIC = ROOT / "data" / "synthetic"


class Phase3AuditTests(unittest.TestCase):
    def test_dedicated_assets_keep_variants_split_and_labels_provisional(self) -> None:
        development = load_phase3_evaluation_cases(
            EVALUATION / "phase3_development.jsonl",
            expected_split="development",
        )
        held_out = load_phase3_evaluation_cases(
            EVALUATION / "phase3_held_out.jsonl",
            expected_split="held_out",
        )
        cases = [*development, *held_out]
        self.assertEqual(len(development), 13)
        self.assertEqual(len(held_out), 7)
        self.assertEqual(len({case.case_id for case in cases}), 20)
        review_status = json.loads(
            (EVALUATION / "phase3_label_review_status.json").read_text(encoding="utf-8")
        )
        self.assertEqual(review_status["counts"]["cases"], len(cases))
        self.assertEqual(review_status["counts"]["intent_records"], sum(len(case.expected_intents) for case in cases))
        self.assertEqual(
            review_status["review_policy"]["semantic_claim_support"],
            "not_evaluated",
        )
        families: dict[str, set[str]] = {}
        for case in cases:
            families.setdefault(case.variant_family, set()).add(case.split)
            self.assertEqual(case.asset_status, "synthetic_fixture")
            self.assertEqual(case.label_review_status.intent_labels, "provisional_generated")
            self.assertEqual(case.label_review_status.relevance_labels, "provisional_generated")
            self.assertEqual(case.label_review_status.answer_expectations, "provisional_generated")
        self.assertFalse([family for family, splits in families.items() if len(splits) > 1])

    def test_matching_rubric_is_explicit(self) -> None:
        rubric = _matching_rubric()
        self.assertTrue(rubric["one_to_one_matching"])
        self.assertIn("all expected intent-local values", rubric["constraint_score"])
        self.assertIn("unmatched", rubric["extra_intent_definition"])

    def test_audit_rejects_trivial_top_k(self) -> None:
        cases = load_phase3_evaluation_cases(
            EVALUATION / "phase3_development.jsonl",
            expected_split="development",
        )[:1]
        corpus = CorpusIngestor().ingest(
            load_document_inputs(SYNTHETIC / "phase3_documents.jsonl"),
            "synthetic_fixture",
        )
        with self.assertRaises(Phase3AuditError):
            asyncio.run(
                evaluate_phase3_audit(
                    cases,
                    corpus=corpus,
                    settings=load_settings(None),
                    backend="lexical",
                    top_k=len(corpus.chunks),
                    execution_mode="accelerated",
                )
            )


if __name__ == "__main__":
    unittest.main()
