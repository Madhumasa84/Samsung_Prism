"""Phase 4 answer publication, presentation, and revision-race checks."""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from flowcontext.contracts import Answer, StreamingSchedulerConfig, Usage
from flowcontext.generation import MockGenerationProvider
from flowcontext.ingestion import CorpusIngestor, load_document_inputs
from flowcontext.multi_intent import decompose_query
from flowcontext.phase4 import (
    PatchConflictError,
    Phase4AnswerPublisher,
    Phase4SelectiveUpdateCoordinator,
    Phase4SessionStore,
)
from flowcontext.phase4_replay import _passage_for_hit
from flowcontext.retrieval import make_retriever


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "data" / "synthetic"


class RecordingRetriever:
    backend = "lexical"

    def __init__(self, index) -> None:
        self.index = index
        self.base = make_retriever(index, backend="lexical", top_k=5)
        self.calls: list[str] = []

    def search(self, query: str):
        self.calls.append(query)
        return self.base.search(query)


class TranslationProvider:
    """Synthetic presentation provider used only to exercise provider labels."""

    def __init__(self, *, add_unknown_citation: bool = False) -> None:
        self.config = MockGenerationProvider().config
        self.add_unknown_citation = add_unknown_citation
        self.calls = 0

    def translate(self, *, answer: Answer, language: str) -> str:
        self.calls += 1
        rendered = answer.answer_text
        return rendered + (" [unknown-source]" if self.add_unknown_citation else "")


class Phase4PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CorpusIngestor().ingest(
            load_document_inputs(SYNTHETIC / "documents.jsonl"),
            "synthetic_fixture",
        )

    def _store(self) -> Phase4SessionStore:
        return Phase4SessionStore(max_sessions=8, max_records_per_session=128)

    def _initial(
        self,
        query: str,
        *,
        session_id: str = "publication-session",
        provider: MockGenerationProvider | None = None,
        retriever: RecordingRetriever | None = None,
    ) -> tuple[Phase4SessionStore, object, Phase4AnswerPublisher, RecordingRetriever]:
        store = self._store()
        retriever = retriever or RecordingRetriever(self.index)
        plan = decompose_query(query)
        state = store.create_session(
            session_id,
            initial_plan=plan,
            utterance_id="u-0",
            initial_turn=0,
            corpus_id=self.index.corpus_id,
            index_id=self.index.manifest.index_id,
        )
        hits = retriever.search(query)
        retrieval_revision = state.retrieval_revision + 1
        for hit in hits:
            passage = _passage_for_hit(hit, state, self.index)
            if passage is None:
                continue
            state = store.record_evidence(
                session_id,
                passage,
                intent_ids=passage.intent_ids,
                retrieval_revision=retrieval_revision,
                corpus_id=self.index.corpus_id,
                index_id=self.index.manifest.index_id,
                expected_revision=state.state_revision,
            )
        publisher = Phase4AnswerPublisher(
            store,
            self.index,
            generation_provider=provider or MockGenerationProvider(),
        )
        result = asyncio.run(
            publisher.generate_and_publish(
                session_id,
                expected_revision=state.state_revision,
                triggering_turn=0,
                triggering_utterance_id="u-0",
                retrieval_usage=Usage(input_tokens=1, total_tokens=1, estimated=True),
                retrieval_call_count=1,
                retrieval_attempt_count=1,
                retrieval_model_identity="test.retriever",
            )
        )
        self.assertTrue(result.published)
        return store, result.state, publisher, retriever

    def test_initial_publication_and_history_are_inspectable(self) -> None:
        store, state, _publisher, _retriever = self._initial(
            "Which Venue B in Pune can host 40 attendees? What are the catering options?"
        )
        self.assertEqual(state.answer_status, "current")
        self.assertEqual(state.current_answer_version, 1)
        version = state.current_answer
        assert version is not None
        self.assertIsNone(version.parent_version)
        self.assertEqual(version.change_kind, "initial")
        self.assertTrue(any(item.change_type == "added" for item in version.claim_changes))
        self.assertEqual(version.generation_execution_mode, "mock_provider")
        self.assertEqual(version.retrieval_call_count, 1)
        self.assertEqual(state.revision_history[-1].answer_version, 1)
        self.assertEqual(state.revision_history[-1].publication_status, "completed")
        self.assertEqual(store.get(state.session_id).current_answer_version, 1)

    def test_factual_update_preserves_unrelated_claim_ids_and_records_changes(self) -> None:
        store, state, publisher, retriever = self._initial(
            "Which Venue B in Pune can host 40 attendees? What are the catering options?",
            session_id="local-publication",
        )
        old_catering_claim_id = next(
            record.claim_id
            for record in state.current_claims
            if "catering" in record.claim.claim_text.casefold()
        )
        parent_claim_record_ids = set(state.answer_versions[0].claim_record_ids)
        parent_evidence_ids = {
            evidence_id
            for record in state.current_claims
            for evidence_id in record.supporting_evidence_ids
        }
        patch = store.propose_follow_up(
            "local-publication",
            "Actually, make it 30 attendees.",
            utterance_id="u-1",
            turn_index=1,
        )

        async def run() -> object:
            coordinator = Phase4SelectiveUpdateCoordinator(
                store,
                retriever,
                scheduler_config=StreamingSchedulerConfig(final_wait_timeout_s=1.0),
            )
            plan = coordinator.apply_patch(patch)
            assert plan is not None
            candidate = await coordinator.execute_plan(plan)
            stored_plan = next(
                item for item in candidate.selective_update_plans if item.plan_id == plan.plan_id
            )
            published = await publisher.generate_and_publish(
                "local-publication",
                expected_revision=candidate.state_revision,
                triggering_patch_id=patch.patch_id,
                triggering_turn=1,
                triggering_utterance_id="u-1",
                retrieval_usage=stored_plan.retrieval_usage,
                retrieval_call_count=stored_plan.retrieval_call_count,
                retrieval_attempt_count=stored_plan.retrieval_attempt_count,
                retrieval_model_identity="test.retriever",
            )
            await coordinator.close()
            return published.state

        state = asyncio.run(run())
        assert state is not None
        new_by_intent = {record.intent_ids[0]: record.claim_id for record in state.current_claims}
        self.assertIn(old_catering_claim_id, new_by_intent.values())
        self.assertEqual(state.current_answer_version, 2)
        version = state.current_answer
        assert version is not None
        self.assertEqual(version.parent_version, 1)
        self.assertEqual(version.change_kind, "factual_update")
        self.assertTrue(any(item.change_type == "preserved" for item in version.claim_changes))
        self.assertTrue(any(item.change_type in {"added", "removed", "modified"} for item in version.claim_changes))
        self.assertGreaterEqual(len(retriever.calls), 2)
        self.assertTrue(any(item.status != "current" for item in state.claim_records if item.answer_version == 1))
        self.assertTrue(parent_claim_record_ids <= {item.record_id for item in state.claim_records})
        self.assertTrue(parent_evidence_ids <= {item.evidence_id for item in state.evidence_records})

    def test_failed_reassessment_keeps_siblings_and_marks_target_unresolved(self) -> None:
        failing_provider = MockGenerationProvider(mode="invalid_json")
        store, state, publisher, _retriever = self._initial(
            "Which Venue B in Pune can host 40 attendees? What are the catering options?",
            session_id="failed-publication",
            provider=MockGenerationProvider(),
        )
        patch = store.propose_follow_up(
            "failed-publication",
            "Actually, make it 30 attendees.",
            utterance_id="u-1",
            turn_index=1,
        )
        store.apply_patch(patch)
        candidate = store.get("failed-publication")
        failing = Phase4AnswerPublisher(store, self.index, generation_provider=failing_provider)
        result = asyncio.run(
            failing.generate_and_publish(
                "failed-publication",
                expected_revision=candidate.state_revision,
                triggering_patch_id=patch.patch_id,
                triggering_turn=1,
                triggering_utterance_id="u-1",
            )
        )
        self.assertTrue(result.published)
        self.assertEqual(result.state.answer_status, "partial")
        self.assertTrue(result.state.current_claims)
        self.assertTrue(result.state.current_answer is not None)
        assert result.state.current_answer is not None
        self.assertTrue(result.state.current_answer.unresolved_intent_ids)
        self.assertTrue(any(item.change_type == "preserved" for item in result.state.current_answer.claim_changes))
        self.assertNotEqual(result.state.current_answer.answer.answer_text, state.current_answer.answer.answer_text)

    def test_presentation_revision_uses_zero_retrieval_and_stable_claim_ids(self) -> None:
        store, state, publisher, retriever = self._initial(
            "Which Venue B in Pune can host 40 attendees? What are the catering options?",
            session_id="format-publication",
        )
        old_claim_ids = [record.claim_id for record in state.current_claims]
        patch = store.propose_follow_up(
            "format-publication",
            "Make it two bullets.",
            utterance_id="u-1",
            turn_index=1,
        )
        store.apply_patch(patch)
        before_calls = len(retriever.calls)
        result = asyncio.run(
            publisher.present_and_publish(
                "format-publication",
                expected_revision=store.get("format-publication").state_revision,
            )
        )
        self.assertTrue(result.published)
        self.assertEqual(len(retriever.calls), before_calls)
        self.assertEqual(result.state.current_answer_version, 2)
        self.assertEqual([record.claim_id for record in result.state.current_claims], old_claim_ids)
        assert result.state.current_answer is not None
        self.assertEqual(result.state.current_answer.change_kind, "presentation")
        self.assertEqual(result.state.current_answer.retrieval_call_count, 0)
        self.assertEqual(result.state.current_answer.generation_attempts, 0)
        self.assertTrue(result.state.current_answer.answer.answer_text.startswith("- "))

    def test_translation_is_presentation_only_and_keeps_provider_label(self) -> None:
        store, state, publisher, retriever = self._initial(
            "Which Venue B in Pune can host 40 attendees?",
            session_id="translation-publication",
        )
        old_claim_ids = [record.claim_id for record in state.current_claims]
        patch = store.propose_follow_up(
            "translation-publication",
            "Translate the current answer into Hindi.",
            utterance_id="u-1",
            turn_index=1,
        )
        store.apply_patch(patch)
        before_retrieval = len(retriever.calls)
        translation = TranslationProvider()
        result = asyncio.run(
            publisher.present_and_publish(
                "translation-publication",
                expected_revision=store.get("translation-publication").state_revision,
                presentation_provider=translation,
            )
        )
        self.assertTrue(result.published)
        self.assertEqual(len(retriever.calls), before_retrieval)
        self.assertEqual(translation.calls, 1)
        self.assertEqual([record.claim_id for record in result.state.current_claims], old_claim_ids)
        assert result.state.current_answer is not None
        self.assertEqual(result.state.current_answer.change_kind, "presentation")
        self.assertEqual(result.state.current_answer.presentation_execution_mode, "mock_provider")
        self.assertEqual(result.state.current_answer.retrieval_call_count, 0)
        self.assertEqual(result.state.pending_requests[-1].kind, "presentation")

    def test_presentation_rejects_unknown_citation_tags(self) -> None:
        store, _state, publisher, retriever = self._initial(
            "Which Venue B in Pune can host 40 attendees?",
            session_id="bad-translation-publication",
        )
        patch = store.propose_follow_up(
            "bad-translation-publication",
            "Translate the current answer into Hindi.",
            utterance_id="u-1",
            turn_index=1,
        )
        store.apply_patch(patch)
        result = asyncio.run(
            publisher.present_and_publish(
                "bad-translation-publication",
                expected_revision=store.get("bad-translation-publication").state_revision,
                presentation_provider=TranslationProvider(add_unknown_citation=True),
            )
        )
        self.assertFalse(result.published)
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(result.state.current_answer_version, 1)
        self.assertEqual(len(result.state.answer_versions), 1)
        self.assertTrue(any(item.kind == "presentation" for item in result.state.superseded_requests))

    def test_stale_publication_cannot_commit_a_new_version(self) -> None:
        store, state, _publisher, _retriever = self._initial(
            "Which Venue B in Pune can host 40 attendees?",
            session_id="atomic-publication",
        )
        old_revision = state.state_revision
        patch = store.propose_follow_up(
            "atomic-publication",
            "Make it two bullets.",
            utterance_id="u-1",
            turn_index=1,
        )
        store.apply_patch(patch)
        with self.assertRaises(PatchConflictError):
            store.publish_answer(
                "atomic-publication",
                Answer(
                    answer_text="late",
                    factual_claims=[],
                    uncertainty="late result",
                    answer_version=1,
                ),
                expected_revision=old_revision,
            )
        current = store.get("atomic-publication")
        self.assertEqual(current.current_answer_version, 1)
        self.assertEqual(len(current.answer_versions), 1)
        self.assertEqual(current.answer_status, "presentation_pending")

    def test_correction_supersedes_generation_that_is_still_running(self) -> None:
        store, state, _publisher, retriever = self._initial(
            "Which Venue B in Pune can host 40 attendees? What are the catering options?",
            session_id="generation-race",
        )
        delayed = Phase4AnswerPublisher(
            store,
            self.index,
            generation_provider=MockGenerationProvider(timeout_delay_s=0.05),
        )
        patch = store.propose_follow_up(
            "generation-race",
            "Actually, make it 30 attendees.",
            utterance_id="u-1",
            turn_index=1,
        )

        async def run() -> tuple[object, object]:
            old_task = asyncio.create_task(
                delayed.generate_and_publish(
                    "generation-race",
                    expected_revision=state.state_revision,
                    triggering_turn=0,
                    triggering_utterance_id="u-0",
                )
            )
            await asyncio.sleep(0)
            coordinator = Phase4SelectiveUpdateCoordinator(
                store,
                retriever,
                scheduler_config=StreamingSchedulerConfig(final_wait_timeout_s=1.0),
            )
            plan = coordinator.apply_patch(patch)
            assert plan is not None
            candidate = await coordinator.execute_plan(plan)
            stored_plan = next(
                item for item in candidate.selective_update_plans if item.plan_id == plan.plan_id
            )
            fresh = await delayed.generate_and_publish(
                "generation-race",
                expected_revision=candidate.state_revision,
                triggering_patch_id=patch.patch_id,
                triggering_turn=1,
                triggering_utterance_id="u-1",
                retrieval_usage=stored_plan.retrieval_usage,
                retrieval_call_count=stored_plan.retrieval_call_count,
                retrieval_attempt_count=stored_plan.retrieval_attempt_count,
            )
            old = await old_task
            await coordinator.close()
            return old, fresh

        old, fresh = asyncio.run(run())
        self.assertFalse(old.published)
        self.assertEqual(old.status, "superseded")
        self.assertTrue(fresh.published)
        self.assertEqual(store.get("generation-race").current_answer_version, 2)


if __name__ == "__main__":
    unittest.main()
