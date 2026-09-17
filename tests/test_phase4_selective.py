"""Dependency invalidation and selective retrieval checks for Phase 4."""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from pathlib import Path

from flowcontext.contracts import (
    Answer,
    EvidencePassage,
    FactualClaim,
    Phase4ClaimDependency,
    RetrievalHit,
    StreamingSchedulerConfig,
)
from flowcontext.ingestion import CorpusIngestor, load_document_inputs
from flowcontext.multi_intent import decompose_query
from flowcontext.phase4 import Phase4SelectiveUpdateCoordinator, Phase4SessionStore
from flowcontext.retrieval import make_retriever


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "data" / "synthetic"


class EmptyRetriever:
    backend = "lexical"
    method = "empty_test_retriever"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str) -> list[RetrievalHit]:
        self.calls.append(query)
        return []


class DelayedRetriever:
    backend = "lexical"
    method = "delayed_test_retriever"

    def __init__(self, base, *, delay_s: float = 0.0) -> None:
        self.base = base
        self.delay_s = delay_s
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def search(self, query: str) -> list[RetrievalHit]:
        with self._lock:
            self.calls.append(query)
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.base.search(query)


class Phase4SelectiveUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CorpusIngestor().ingest(
            load_document_inputs(SYNTHETIC / "documents.jsonl"),
            "synthetic_fixture",
        )
        cls.base_retriever = make_retriever(cls.index, backend="lexical", top_k=3)

    def _store(self, *, max_records: int = 64) -> Phase4SessionStore:
        return Phase4SessionStore(max_sessions=8, max_records_per_session=max_records)

    def _published(
        self,
        session_id: str,
        query: str,
        *,
        texts: list[str] | None = None,
        extra_dependencies: dict[str, list[Phase4ClaimDependency]] | None = None,
    ):
        store = self._store()
        plan = decompose_query(query)
        state = store.create_session(
            session_id,
            initial_plan=plan,
            corpus_id="synthetic_fixture",
            index_id="synthetic-index",
        )
        passages: list[EvidencePassage] = []
        texts = texts or ["The corpus records the requested information."] * len(state.active_intents)
        for index, intent in enumerate(state.active_intents):
            passage = EvidencePassage(
                chunk_id=f"phase4-{session_id}-{index}",
                source_location=f"fixture://{session_id}/{index}",
                text=texts[index],
                rank=1,
                score=0.9,
                retrieval_method="test_fixture",
                intent_id=intent.intent_id,
                intent_ids=[intent.intent_id],
            )
            passages.append(passage)
            state = store.record_evidence(
                session_id,
                passage,
                intent_ids=[intent.intent_id],
            )
        claims = [
            FactualClaim(
                claim_id=f"claim-{index}",
                claim_text=passage.text,
                supporting_chunk_ids=[passage.chunk_id],
                intent_ids=[state.active_intents[index].intent_id],
            )
            for index, passage in enumerate(passages)
        ]
        state = store.publish_answer(
            session_id,
            Answer(
                answer_text="\n".join(item.claim_text for item in claims),
                factual_claims=claims,
                uncertainty="No other corpus facts were established.",
                answer_version=1,
            ),
            extra_dependencies=extra_dependencies,
        )
        return store, state, passages

    def test_local_correction_preserves_unrelated_claim_and_evidence(self) -> None:
        store, state, passages = self._published(
            "local",
            "Which Hall Alpha in Pune can host an event for 40 attendees? What are the catering options?",
            texts=[
                "Hall Alpha in Pune can host events for 40 attendees.",
                "Catering options include a buffet and plated service.",
            ],
        )
        old_claims = {item.claim_id: item.record_id for item in state.current_claims}
        correction = store.propose_follow_up(
            "local",
            "Actually, make it 50 attendees.",
            utterance_id="local-u2",
            turn_index=2,
        )
        state = store.apply_patch(correction)
        plan = state.selective_update_plans[-1]

        self.assertEqual(len(plan.retrieval_tasks), 1)
        self.assertEqual(
            plan.retrieval_tasks[0].query.split(";")[0],
            "Which Hall Alpha in Pune can host an event for 50 attendees",
        )
        self.assertIn(old_claims["claim-1"], plan.preserved_claim_ids)
        self.assertIn(old_claims["claim-0"], plan.invalidated_claim_ids)
        self.assertEqual(len(state.current_claims), 1)
        self.assertEqual(state.current_claims[0].claim_id, "claim-1")
        self.assertEqual(len(state.current_evidence), 1)
        self.assertEqual(state.current_evidence[0].passage.chunk_id, passages[1].chunk_id)
        self.assertEqual(len(plan.retrieval_intent_ids), 1)
        self.assertIn(plan.retrieval_intent_ids[0], {item.intent_id for item in state.active_intents})
        self.assertNotIn(
            state.current_claims[0].intent_ids[0],
            plan.retrieval_intent_ids,
        )

    def test_changed_entity_invalidates_entity_bound_dependent_claims(self) -> None:
        store, state, _passages = self._published(
            "entity",
            "Which Hall Alpha in Pune can host an event?",
            texts=["Hall Alpha in Pune can host events and has a cancellation policy."],
        )
        old_claim_id = state.current_claims[0].record_id
        correction = store.propose_follow_up(
            "entity",
            "Actually, use Hall Beta.",
            utterance_id="entity-u2",
            turn_index=2,
        )
        state = store.apply_patch(correction)
        plan = state.selective_update_plans[-1]

        self.assertTrue(plan.affected_entity_ids)
        self.assertIn(old_claim_id, plan.invalidated_claim_ids)
        self.assertEqual(len(plan.retrieval_tasks), 1)
        self.assertIn("selected entity", next(iter(plan.retrieval_reasons.values())))
        self.assertFalse(state.current_claims)
        self.assertFalse(state.current_evidence)
        self.assertTrue(all(item.status != "current" for item in state.claim_records))
        self.assertTrue(all(item.status != "current" for item in state.evidence_records))

    def test_claim_dependency_invalidation_propagates_to_dependent_claim(self) -> None:
        store, state, _passages = self._published(
            "claim-chain",
            "Which Hall Alpha in Pune can host an event? What are the catering options?",
            texts=[
                "Hall Alpha in Pune can host events.",
                "Catering options include a buffet.",
            ],
            extra_dependencies={
                "claim-1": [
                    Phase4ClaimDependency(
                        dependency_id="claim-edge",
                        dependency_type="claim",
                        target_id="claim-0",
                        relation="depends_on",
                    )
                ]
            },
        )
        first = {item.claim_id: item.record_id for item in state.current_claims}
        patch = store.propose_follow_up(
            "claim-chain",
            "Actually, make it Hall Beta for that venue.",
            utterance_id="claim-chain-u2",
            turn_index=2,
        )
        state = store.apply_patch(patch)
        plan = state.selective_update_plans[-1]

        self.assertEqual(set(plan.invalidated_claim_ids), {first["claim-0"], first["claim-1"]})
        self.assertEqual(plan.preserved_claim_ids, [])
        self.assertEqual(plan.invalidation_reasons[first["claim-1"]], "dependent claim was invalidated")
        self.assertNotIn(first["claim-0"], state.current_claim_record_ids)
        self.assertNotIn(first["claim-1"], state.current_claim_record_ids)
        self.assertTrue(any(item.intent_id != state.active_intents[0].intent_id for item in plan.retrieval_tasks))

    def test_constraint_removal_requires_broader_reconsideration(self) -> None:
        store, _state, _passages = self._published(
            "remove",
            "Which Hall Alpha in Pune can host an event for 40 attendees?",
            texts=["Hall Alpha in Pune can host events for 40 attendees."],
        )
        removal = store.propose_follow_up(
            "remove",
            "Remove the 40 attendees requirement.",
            utterance_id="remove-u2",
            turn_index=2,
        )
        state = store.apply_patch(removal)
        plan = state.selective_update_plans[-1]
        self.assertEqual(len(plan.retrieval_tasks), 1)
        self.assertIn("broadens", next(iter(plan.retrieval_reasons.values())))
        self.assertTrue(plan.invalidated_evidence_ids)

    def test_existing_evidence_can_be_rebound_without_retrieval(self) -> None:
        store, _state, _passages = self._published(
            "reuse",
            "Which Hall Alpha in Pune can host an event?",
            texts=["Hall Alpha in Pune can host events. A projector is available."],
        )
        patch = store.propose_follow_up(
            "reuse",
            "Also require a projector.",
            utterance_id="reuse-u2",
            turn_index=2,
        )
        state = store.apply_patch(patch)
        plan = state.selective_update_plans[-1]
        self.assertFalse(plan.retrieval_tasks)
        self.assertTrue(plan.reused_evidence_ids)
        self.assertTrue(plan.preserved_claim_ids)
        self.assertTrue(state.current_evidence[0].reused_from_evidence_id)
        self.assertTrue(state.current_claims)

    def test_missing_new_evidence_remains_unanswered(self) -> None:
        store = self._store()
        store.create_session(
            "missing",
            initial_plan=decompose_query("Which Hall Alpha in Pune can host an event?"),
            corpus_id="synthetic_fixture",
            index_id="synthetic-index",
        )
        retriever = EmptyRetriever()
        coordinator = Phase4SelectiveUpdateCoordinator(
            store,
            retriever,
            scheduler_config=StreamingSchedulerConfig(final_wait_timeout_s=1.0),
        )

        async def run() -> None:
            patch = store.propose_follow_up(
                "missing",
                "Also require a projector.",
                utterance_id="missing-u2",
                turn_index=2,
            )
            plan = coordinator.apply_patch(patch)
            assert plan is not None
            state = await coordinator.execute_plan(plan)
            self.assertEqual(retriever.calls, [plan.retrieval_tasks[0].query])
            self.assertFalse(state.current_evidence)
            self.assertEqual(state.selective_update_plans[-1].retrieval_tasks[0].evidence_ids, [])
            self.assertEqual(state.selective_update_plans[-1].retrieval_tasks[0].status, "completed")
            await coordinator.close()

        asyncio.run(run())

    def test_second_correction_supersedes_in_flight_work_and_discards_late_result(self) -> None:
        store = self._store()
        store.create_session(
            "racing",
            initial_plan=decompose_query(
                "Which Hall Alpha in Pune can host an event for 40 attendees?"
            ),
            corpus_id="synthetic_fixture",
            index_id="synthetic_fixture",
        )
        retriever = DelayedRetriever(self.base_retriever, delay_s=0.06)
        coordinator = Phase4SelectiveUpdateCoordinator(
            store,
            retriever,
            scheduler_config=StreamingSchedulerConfig(
                max_concurrency=1,
                max_pending_requests=2,
                max_total_requests=8,
                final_wait_timeout_s=1.0,
            ),
        )

        async def run() -> None:
            first = store.propose_follow_up(
                "racing",
                "Actually, make it 50 attendees.",
                utterance_id="racing-u2",
                turn_index=2,
            )
            first_plan = coordinator.apply_patch(first)
            assert first_plan is not None
            await asyncio.sleep(0.01)
            second = store.propose_follow_up(
                "racing",
                "Actually, make it 60 attendees.",
                utterance_id="racing-u3",
                turn_index=3,
            )
            second_plan = coordinator.apply_patch(second)
            assert second_plan is not None
            await coordinator.execute_plan(second_plan)
            await asyncio.sleep(0.09)
            state = store.get("racing")
            old_plan = next(item for item in state.selective_update_plans if item.plan_id == first_plan.plan_id)
            self.assertEqual(old_plan.status, "superseded")
            self.assertTrue(old_plan.discarded_result_ids)
            self.assertTrue(any(item.stale for item in coordinator._runtimes["racing"].scheduler.results))
            self.assertTrue(all(item.candidate_state_revision == state.state_revision for item in second_plan.retrieval_tasks))
            self.assertTrue(any("60 attendees" in query for query in retriever.calls))
            await coordinator.close()

        asyncio.run(run())

    def test_targeted_work_is_isolated_across_simultaneous_sessions(self) -> None:
        store = self._store()
        for session_id, venue in (("session-a", "Hall Alpha"), ("session-b", "Hall Beta")):
            store.create_session(
                session_id,
                initial_plan=decompose_query(f"Which {venue} is available?"),
                corpus_id="synthetic_fixture",
                index_id="synthetic_fixture",
            )
        coordinator = Phase4SelectiveUpdateCoordinator(
            store,
            self.base_retriever,
            scheduler_config=StreamingSchedulerConfig(final_wait_timeout_s=1.0),
        )

        async def run() -> None:
            patches = [
                store.propose_follow_up(
                    "session-a",
                    "Also require a projector.",
                    utterance_id="a-u2",
                    turn_index=2,
                ),
                store.propose_follow_up(
                    "session-b",
                    "Also require a microphone.",
                    utterance_id="b-u2",
                    turn_index=2,
                ),
            ]
            plans = [coordinator.apply_patch(patch) for patch in patches]
            await asyncio.gather(*(coordinator.execute_plan(plan) for plan in plans if plan is not None))
            first, second = store.get("session-a"), store.get("session-b")
            self.assertTrue(all(item.session_id == "session-a" for item in first.current_evidence))
            self.assertTrue(all(item.session_id == "session-b" for item in second.current_evidence))
            self.assertNotEqual(first.selective_update_plans[-1].patch_id, second.selective_update_plans[-1].patch_id)
            await coordinator.close()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
