"""Focused Phase 4 session-state and follow-up interpretation checks."""

from __future__ import annotations

import unittest

from flowcontext.contracts import Answer, EvidencePassage, FactualClaim, Phase4ClaimDependency
from flowcontext.multi_intent import MockDecompositionProvider, StructuredMultiIntentDecomposer, decompose_query
from flowcontext.phase4 import (
    PatchConflictError,
    PatchValidationError,
    Phase4FollowUpInterpreter,
    Phase4SessionStore,
)


class Phase4SessionTests(unittest.TestCase):
    def make_store(self, *, max_sessions: int = 8) -> Phase4SessionStore:
        return Phase4SessionStore(max_sessions=max_sessions, max_records_per_session=32)

    def make_one_intent_session(self, session_id: str = "session-a") -> tuple[Phase4SessionStore, object]:
        store = self.make_store()
        state = store.create_session(
            session_id,
            initial_plan=decompose_query("Which Hall Alpha in Pune can host an event?"),
            utterance_id="utterance-1",
            initial_turn=1,
            corpus_id="corpus-v1",
            index_id="index-v1",
        )
        return store, state

    def test_add_replace_remove_preserves_typed_values_and_user_origin(self) -> None:
        store, state = self.make_one_intent_session()
        initial_intent_id = state.active_intents[0].intent_id

        add = store.propose_follow_up(
            "session-a",
            "Also require a projector for at least 40 attendees on 2026-10-02.",
            utterance_id="utterance-2",
            turn_index=2,
        )
        self.assertEqual(add.classification, "add_constraint")
        self.assertEqual(add.base_revision, 0)
        self.assertEqual(add.proposed_revision, 1)
        self.assertTrue(any(item.constraint.kind == "quantity" for item in add.added_constraints))
        self.assertTrue(any(item.constraint.kind == "date" for item in add.added_constraints))
        self.assertTrue(any(item.constraint.kind == "other" for item in add.added_constraints))
        self.assertTrue(all(item.corpus_fact is False for item in add.added_constraints))
        state = store.apply_patch(add)
        self.assertNotEqual(state.active_intents[0].intent_id, initial_intent_id)
        self.assertTrue(any(item.origin_turn == 2 for item in state.current_constraints))

        replace = store.propose_follow_up(
            "session-a",
            "Actually, make it 50 attendees in Mumbai.",
            utterance_id="utterance-3",
            turn_index=3,
        )
        self.assertEqual(replace.classification, "replace_constraint")
        self.assertTrue(replace.replacement_constraints)
        self.assertTrue(any(item.constraint.value == "50 attendees" for item in replace.replacement_constraints.values()))
        state = store.apply_patch(replace)
        current_values = {(item.constraint.kind, item.constraint.value) for item in state.current_constraints}
        self.assertIn(("quantity", "50 attendees"), current_values)
        self.assertIn(("location", "in Mumbai"), current_values)
        self.assertNotIn(("quantity", "at least 40 attendees"), current_values)
        self.assertTrue(any(item.status == "superseded" for item in state.intent_history))

        remove = store.propose_follow_up(
            "session-a",
            "Remove the projector requirement.",
            utterance_id="utterance-4",
            turn_index=4,
        )
        self.assertEqual(remove.classification, "remove_constraint")
        state = store.apply_patch(remove)
        self.assertFalse(any("projector" in item.constraint.value.casefold() for item in state.current_constraints))
        self.assertTrue(any(item.status == "removed" and "projector" in item.constraint.value.casefold() for item in state.constraint_records))

        add_negation = store.propose_follow_up(
            "session-a",
            "Also exclude Hall Beta.",
            utterance_id="utterance-5",
            turn_index=5,
        )
        self.assertEqual(add_negation.classification, "add_constraint")
        self.assertTrue(any(item.constraint.kind == "negation" for item in add_negation.added_constraints))
        self.assertTrue(any("Hall Beta" in item.constraint.value for item in add_negation.added_constraints))

    def test_add_question_resolves_one_reference_and_keeps_it_contextual(self) -> None:
        store, _state = self.make_one_intent_session()
        patch = store.propose_follow_up(
            "session-a",
            "What is its cancellation policy?",
            utterance_id="utterance-2",
            turn_index=2,
        )
        self.assertEqual(patch.classification, "add_question")
        self.assertEqual(len(patch.reference_resolutions), 1)
        self.assertIn("Hall Alpha", patch.new_intents[0].intent.query)
        inherited = [item for item in patch.added_constraints if item.origin == "inherited_context"]
        self.assertEqual(len(inherited), 1)
        self.assertFalse(inherited[0].corpus_fact)
        state = store.apply_patch(patch)
        self.assertEqual(len(state.active_intents), 2)
        self.assertEqual(state.active_intents[0].origin_turn, 1)
        self.assertEqual(state.active_intents[1].origin_turn, 2)
        self.assertEqual(state.active_topic, "hall alpha pune host event")

    def test_add_question_keeps_separate_multi_intent_questions_and_shared_constraints(self) -> None:
        store = self.make_store()
        state = store.create_session(
            "multi",
            initial_plan=decompose_query("What is the cancellation policy? What are the catering options?"),
        )
        original_ids = {item.intent_id for item in state.active_intents}
        patch = store.propose_follow_up(
            "multi",
            "What is the parking policy? What are the access hours?",
            utterance_id="utterance-2",
            turn_index=2,
        )
        self.assertEqual(patch.classification, "add_question")
        self.assertEqual(len(patch.new_intents), 2)
        state = store.apply_patch(patch)
        self.assertEqual(len(state.active_intents), 4)
        self.assertTrue(original_ids <= {item.intent_id for item in state.active_intents})

    def test_shared_constraint_stays_shared_when_a_question_is_added(self) -> None:
        original = decompose_query("Which Hall Alpha in Pune can host an event?")
        location = next(item for item in original.intents[0].constraints if item.kind == "location")
        plan = original.model_copy(
            update={
                "intents": [original.intents[0].model_copy(update={"constraints": []})],
                "shared_constraints": [location],
            }
        )
        store = self.make_store()
        store.create_session("shared", initial_plan=plan)
        patch = store.propose_follow_up(
            "shared",
            "What are the access hours?",
            utterance_id="utterance-2",
            turn_index=2,
        )
        state = store.apply_patch(patch)
        shared = [item for item in state.current_constraints if item.scope == "shared"]
        self.assertEqual(len(shared), 1)
        self.assertIsNone(shared[0].intent_id)
        self.assertEqual(len(state.current_constraints), 1)

        replace = store.propose_follow_up(
            "shared",
            "Actually, change the location to Mumbai for all.",
            utterance_id="utterance-3",
            turn_index=3,
        )
        state = store.apply_patch(replace)
        shared = [item for item in state.current_constraints if item.scope == "shared"]
        self.assertEqual([item.constraint.value for item in shared], ["Mumbai"])

        remove = store.propose_follow_up(
            "shared",
            "Remove the location requirement for all.",
            utterance_id="utterance-4",
            turn_index=4,
        )
        state = store.apply_patch(remove)
        self.assertFalse(state.current_constraints)

    def test_shared_constraint_addition_revises_each_intent_without_leaking_scope(self) -> None:
        store = self.make_store()
        store.create_session(
            "shared-add",
            initial_plan=decompose_query("What is the cancellation policy? What are the catering options?"),
        )
        patch = store.propose_follow_up(
            "shared-add",
            "Also require all questions to include an agenda.",
            utterance_id="utterance-2",
            turn_index=2,
        )
        self.assertEqual(patch.classification, "add_constraint")
        self.assertEqual(len(patch.intent_replacements), 2)
        state = store.apply_patch(patch)
        self.assertEqual(len(state.active_intents), 2)
        shared = [item for item in state.current_constraints if item.scope == "shared"]
        self.assertEqual(len(shared), 1)
        self.assertIsNone(shared[0].intent_id)

    def test_ambiguous_reference_requests_targeted_clarification_without_semantic_mutation(self) -> None:
        store = self.make_store()
        state = store.create_session(
            "comparison",
            initial_plan=decompose_query("Compare Hall Alpha versus Hall Beta for price and capacity."),
        )
        before = state.model_dump()
        patch = store.propose_follow_up(
            "comparison",
            "What is that venue's cancellation policy?",
            utterance_id="utterance-2",
            turn_index=2,
        )
        self.assertEqual(patch.classification, "clarification_required")
        self.assertEqual(patch.clarification.reason_code, "ambiguous_reference")
        self.assertIn("Hall Alpha", patch.clarification.question)
        self.assertIn("Hall Beta", patch.clarification.question)
        state = store.apply_patch(patch)
        self.assertEqual(
            [item.intent_id for item in state.active_intents],
            [item["intent"]["intent_id"] for item in before["active_intents"]],
        )
        self.assertEqual(state.current_constraint_record_ids, before["current_constraint_record_ids"])
        self.assertIsNotNone(state.pending_clarification)

    def test_change_topic_drops_old_constraints_and_formatting_does_not_retrieve(self) -> None:
        store, _state = self.make_one_intent_session()
        topic = store.propose_follow_up(
            "session-a",
            "Now tell me about catering options.",
            utterance_id="utterance-2",
            turn_index=2,
        )
        self.assertEqual(topic.classification, "change_topic")
        state = store.apply_patch(topic)
        self.assertEqual(state.active_topic, "now about catering options")
        self.assertFalse(state.current_constraints)
        self.assertTrue(all(item.status != "active" for item in state.constraint_records))

        format_patch = store.propose_follow_up(
            "session-a",
            "Put the current answer in bullet points.",
            utterance_id="utterance-3",
            turn_index=3,
        )
        self.assertEqual(format_patch.classification, "reformat_answer")
        state = store.apply_patch(format_patch)
        self.assertEqual(state.pending_update.format_instruction, "bullets")
        self.assertFalse(state.pending_update.requires_retrieval)
        self.assertEqual(state.retrieval_revision, 1)

    def test_formatting_with_a_new_factual_question_requires_clarification(self) -> None:
        store, _state = self.make_one_intent_session()
        patch = store.propose_follow_up(
            "session-a",
            "Make it two bullets and what is the cancellation policy?",
            utterance_id="utterance-2",
            turn_index=2,
        )
        self.assertEqual(patch.classification, "clarification_required")
        self.assertIsNotNone(patch.clarification)
        assert patch.clarification is not None
        self.assertIn("new factual question", patch.clarification.question)
        self.assertFalse(patch.added_constraints)
        self.assertFalse(patch.new_intents)

    def test_proposal_is_revision_bound_and_stale_application_is_rejected(self) -> None:
        store, _state = self.make_one_intent_session()
        first = store.propose_follow_up(
            "session-a",
            "Also require a projector.",
            utterance_id="utterance-2",
            turn_index=2,
        )
        store.apply_patch(first)
        with self.assertRaises(PatchConflictError):
            store.apply_patch(first)
        with self.assertRaises(PatchConflictError):
            store.propose_follow_up(
                "session-a",
                "Also require a microphone.",
                utterance_id="utterance-3",
                turn_index=3,
                base_revision=0,
            )

    def test_evidence_claim_dependencies_and_answer_versions_are_session_scoped(self) -> None:
        store, state = self.make_one_intent_session()
        intent_id = state.active_intents[0].intent_id
        passage = EvidencePassage(
            chunk_id="hall-alpha#chunk-0000",
            source_location="fixture://hall-alpha",
            text="Hall Alpha can host events in Pune.",
            rank=1,
            score=0.99,
            retrieval_method="lexical",
            intent_id=intent_id,
            intent_ids=[intent_id],
        )
        state = store.record_evidence("session-a", passage, intent_ids=[intent_id])
        claim_one = FactualClaim(
            claim_id="claim-hall",
            claim_text="Hall Alpha can host events in Pune.",
            supporting_chunk_ids=[passage.chunk_id],
            intent_ids=[intent_id],
            supporting_excerpts=[passage.text],
        )
        claim_two = FactualClaim(
            claim_id="claim-follow-up",
            claim_text="The venue is the active Hall Alpha request.",
            supporting_chunk_ids=[passage.chunk_id],
            intent_ids=[intent_id],
        )
        answer = Answer(
            answer_text="Hall Alpha can host events in Pune.",
            factual_claims=[claim_one, claim_two],
            uncertainty="The corpus does not establish any other venue facts.",
            answer_version=1,
        )
        state = store.publish_answer(
            "session-a",
            answer,
            extra_dependencies={
                "claim-follow-up": [
                    Phase4ClaimDependency(
                        dependency_id="claim-edge",
                        dependency_type="claim",
                        target_id="claim-hall",
                        relation="depends_on",
                    )
                ]
            },
        )
        self.assertEqual(state.current_answer_version, 1)
        self.assertEqual(len(state.answer_versions), 1)
        self.assertTrue(all(item.semantic_support_status == "unreviewed" for item in state.current_claims))
        self.assertTrue(any(item.dependency_type == "evidence" for item in state.current_claims[0].dependencies))
        self.assertTrue(any(item.dependency_type == "claim" for item in state.current_claims[1].dependencies))
        self.assertEqual(state.answer_versions[0].immutable, True)

        format_patch = store.propose_follow_up(
            "session-a",
            "Put the current answer in bullet points.",
            utterance_id="utterance-format",
            turn_index=3,
        )
        state = store.apply_patch(format_patch)
        self.assertTrue(state.current_evidence)
        self.assertEqual(state.current_utterance_id, "utterance-1")

    def test_provider_execution_mode_is_labelled_without_fake_real_quality(self) -> None:
        provider = MockDecompositionProvider()
        decomposer = StructuredMultiIntentDecomposer(provider)
        interpreter = Phase4FollowUpInterpreter(decomposer)
        store, _state = self.make_one_intent_session()
        patch = store.propose_follow_up(
            "session-a",
            "What is its cancellation policy?",
            utterance_id="utterance-2",
            turn_index=2,
            interpreter=interpreter,
        )
        self.assertEqual(patch.execution_mode, "mock_provider")
        self.assertEqual(patch.provider_identity, provider.config.provider)
        self.assertNotEqual(patch.execution_mode, "real_provider")

    def test_simultaneous_sessions_are_isolated_and_clearable(self) -> None:
        store = self.make_store(max_sessions=2)
        store.create_session("one", initial_plan=decompose_query("Which Hall Alpha is available?"))
        store.create_session("two", initial_plan=decompose_query("Which Hall Beta is available?"))
        patch = store.propose_follow_up(
            "one",
            "Also require a projector.",
            utterance_id="one-u2",
            turn_index=2,
        )
        store.apply_patch(patch)
        self.assertFalse(any("projector" in item.constraint.value.casefold() for item in store.get("two").current_constraints))
        self.assertTrue(store.clear("one"))
        self.assertNotIn("one", store.session_ids())
        self.assertIn("two", store.session_ids())
        store.create_session("three", initial_plan=decompose_query("Which Hall Gamma is available?"))
        store.create_session("four", initial_plan=decompose_query("Which Hall Delta is available?"))
        self.assertNotIn("two", store.session_ids())
        store.clear_all()
        self.assertEqual(store.session_ids(), [])

    def test_per_session_bound_preserves_live_state_and_rejects_unsafe_growth(self) -> None:
        store = Phase4SessionStore(max_sessions=2, max_records_per_session=2)
        store.create_session("bounded", initial_plan=decompose_query("Which Hall Alpha is available?"))
        first = store.propose_follow_up(
            "bounded",
            "Also require a projector.",
            utterance_id="bounded-u2",
            turn_index=2,
        )
        state = store.apply_patch(first)
        self.assertEqual(len(state.current_constraints), 2)
        second = store.propose_follow_up(
            "bounded",
            "Also require a microphone.",
            utterance_id="bounded-u3",
            turn_index=3,
        )
        with self.assertRaises(PatchValidationError):
            store.apply_patch(second)
        state = store.get("bounded")
        self.assertEqual(state.state_revision, 1)
        self.assertEqual(len(state.current_constraints), 2)


if __name__ == "__main__":
    unittest.main()
