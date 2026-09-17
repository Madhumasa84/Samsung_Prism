"""Focused checks for the matched Phase 4 evaluation harness."""

from __future__ import annotations

import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path

from flowcontext.config import load_settings
from flowcontext.ingestion import CorpusIngestor, load_document_inputs
from flowcontext.phase4_evaluation import (
    attempt_real_e2e,
    evaluate_phase4,
    load_phase4_evaluation_cases,
    stable_phase4_evaluation_signature,
    write_phase4_cases_review,
    write_phase4_evaluation_report,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "data" / "evaluation"
SYNTHETIC = ROOT / "data" / "synthetic"


class Phase4EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CorpusIngestor().ingest(
            load_document_inputs(SYNTHETIC / "phase4_documents.jsonl"),
            "synthetic_fixture",
        )
        cls.settings = load_settings()

    def test_assets_keep_roles_and_related_variants_separate(self) -> None:
        development = load_phase4_evaluation_cases(
            EVALUATION / "phase4_development.jsonl",
            expected_split="development",
        )
        diagnostics = load_phase4_evaluation_cases(EVALUATION / "phase4_diagnostic.jsonl")
        untouched = load_phase4_evaluation_cases(
            EVALUATION / "phase4_untouched.jsonl",
            expected_split="held_out",
        )
        cases = [*development, *diagnostics, *untouched]
        self.assertEqual(len(development), 13)
        self.assertEqual(len(diagnostics), 3)
        self.assertEqual(len(untouched), 6)
        self.assertEqual(len({case.case_id for case in cases}), len(cases))
        self.assertTrue(all(case.asset_status == "synthetic_fixture" for case in cases))
        self.assertTrue(all(
            case.label_review_status.semantic_claim_support == "not_evaluated"
            for case in cases
        ))
        families: dict[str, set[str]] = {}
        for case in cases:
            families.setdefault(case.variant_family, set()).add(case.split)
        self.assertFalse([family for family, splits in families.items() if len(splits) > 1])

    def test_matched_run_records_quality_resources_and_review_boundary(self) -> None:
        all_cases = [
            *load_phase4_evaluation_cases(EVALUATION / "phase4_development.jsonl"),
            *load_phase4_evaluation_cases(EVALUATION / "phase4_diagnostic.jsonl"),
            *load_phase4_evaluation_cases(EVALUATION / "phase4_untouched.jsonl"),
        ]
        selected_ids = {
            "p4-dev-entity-consequence-003",
            "p4-dev-partial-followup-006",
            "p4-dev-formatting-007",
            "p4-dev-topic-change-005",
            "p4-untouched-remove-003",
        }
        cases = [case for case in all_cases if case.case_id in selected_ids]
        report = asyncio.run(
            evaluate_phase4(
                cases,
                corpus=self.index,
                settings=self.settings,
                backend="lexical",
                top_k=3,
            )
        )
        self.assertEqual(report["data_integrity"]["case_count"], len(cases))
        self.assertEqual(report["strategies"]["full"]["follow_up_interpretation_accuracy"], 1.0)
        self.assertEqual(report["strategies"]["selective"]["follow_up_interpretation_accuracy"], 1.0)
        self.assertIsNotNone(report["strategies"]["selective"]["updated_answer_coverage"])
        self.assertIn("retrieval_tokens", report["strategies"]["full"]["resources"])
        self.assertEqual(report["strategies"]["selective"]["semantic_claim_support"]["status"], "NOT VERIFIED")
        self.assertEqual(report["capability_status"]["semantic_support"], "NOT VERIFIED")

        entity = next(item for item in report["case_results"] if item["case_id"] == "p4-dev-entity-consequence-003")
        entity_turn = entity["strategies"]["selective"]["turns"][0]
        self.assertEqual(set(entity_turn["actual_affected_intent_keys"]), {"venue_selection", "venue_price"})
        self.assertEqual(len(entity_turn["actual_invalidated_claim_ids"]), 2)
        self.assertTrue(entity_turn["retrieval"]["reasons"])
        self.assertEqual(entity_turn["updated_answer_coverage"]["coverage"], 1.0)

        topic = next(item for item in report["case_results"] if item["case_id"] == "p4-dev-topic-change-005")
        topic_turn = topic["strategies"]["selective"]["turns"][0]
        self.assertEqual(topic_turn["status"], "current")
        self.assertEqual(topic_turn["updated_answer_coverage"]["coverage"], 1.0)

        formatting = next(item for item in report["case_results"] if item["case_id"] == "p4-dev-formatting-007")
        formatting_turn = formatting["strategies"]["selective"]["turns"][0]
        self.assertEqual(formatting_turn["classification"], "reformat_answer")
        self.assertEqual(formatting_turn["retrieval"]["calls_total"], 1)
        self.assertEqual(formatting["strategies"]["selective"]["resources"]["formatting_only_retrieval_calls"], 0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "phase4.json"
            review_status = Path(directory) / "labels.json"
            markdown = write_phase4_evaluation_report(output, report)
            write_phase4_cases_review(review_status, cases)
            self.assertTrue(output.is_file())
            self.assertTrue(markdown.is_file())
            self.assertTrue(output.with_name("phase4_claim_review.csv").is_file())
            self.assertTrue(review_status.is_file())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["claim_review_sheet"], str(output.with_name("phase4_claim_review.csv")))
            with output.with_name("phase4_claim_review.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue(all(row["semantic_support_verdict"] == "pending_human_review" for row in rows))

    def test_stable_signature_excludes_measurement_noise(self) -> None:
        cases = load_phase4_evaluation_cases(EVALUATION / "phase4_untouched.jsonl")[:2]
        first = asyncio.run(
            evaluate_phase4(cases, corpus=self.index, settings=self.settings, backend="lexical", top_k=3)
        )
        second = asyncio.run(
            evaluate_phase4(cases, corpus=self.index, settings=self.settings, backend="lexical", top_k=3)
        )
        self.assertEqual(
            stable_phase4_evaluation_signature(first),
            stable_phase4_evaluation_signature(second),
        )

    def test_verified_claim_reviews_promotes_semantic_support_to_pass(self) -> None:
        cases = load_phase4_evaluation_cases(EVALUATION / "phase4_untouched.jsonl")[:1]
        report = asyncio.run(
            evaluate_phase4(
                cases,
                corpus=self.index,
                settings=self.settings,
                backend="lexical",
                top_k=3,
            )
        )
        self.assertEqual(report["capability_status"]["semantic_support"], "NOT VERIFIED")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "phase4.json"
            review_csv = Path(directory) / "phase4_claim_review.csv"
            review_status = Path(directory) / "labels.json"

            write_phase4_evaluation_report(output, report, review_path=review_csv)
            with review_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)

            for row in rows:
                row["semantic_support_verdict"] = "supported"
                row["reviewer"] = "human_reviewer"
                row["review_notes"] = "verified citation match"
            with review_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            write_phase4_evaluation_report(output, report, review_path=review_csv)
            write_phase4_cases_review(review_status, cases, human_reviewed=True)

            updated = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(updated["capability_status"]["semantic_support"], "PASS")
            self.assertEqual(updated["strategies"]["selective"]["semantic_claim_support"]["status"], "PASS")
            self.assertEqual(updated["strategies"]["selective"]["semantic_claim_support"]["support_rate"], 1.0)

            review_json = json.loads(review_status.read_text(encoding="utf-8"))
            self.assertEqual(review_json["cases"][0]["label_review_status"]["semantic_claim_support"], "human_reviewed")
            self.assertEqual(review_json["cases"][0]["human_review_status"], "verified")

    def test_attempt_real_e2e_records_offline_embedding_probe(self) -> None:
        import os
        from unittest import mock
        with tempfile.TemporaryDirectory() as directory:
            out_path = Path(directory) / "real_e2e.json"
            test_settings = self.settings.model_copy(update={"generation_backend": "mock"})
            with mock.patch.dict(os.environ, {"FLOWCONTEXT_REAL_GENERATION_BASE_URL": "http://127.0.0.1:9/v1"}):
                result = asyncio.run(
                    attempt_real_e2e(
                        settings=test_settings,
                        corpus=self.index,
                        backend="lexical",
                        output_path=out_path,
                    )
                )
            self.assertTrue(out_path.is_file())
            self.assertEqual(result["embedding_probe"]["status"], "PASS")
            self.assertIn("model", result["embedding_probe"])


if __name__ == "__main__":
    unittest.main()

