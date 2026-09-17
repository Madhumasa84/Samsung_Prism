from __future__ import annotations

import asyncio
import os
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from flowcontext.config import load_settings
from flowcontext.contracts import GenerationConfig, GenerationResult, RetrievalHit, TranscriptEvent
from flowcontext.generation import MockGenerationProvider
from flowcontext.ingestion import CorpusIngestor, load_document_inputs
from flowcontext.multi_intent import (
    DecompositionProviderError,
    DecompositionTimeout,
    MockDecompositionProvider,
    MultiIntentRetriever,
    MultiIntentRetrievalError,
    OpenAICompatibleDecompositionProvider,
    StructuredMultiIntentDecomposer,
    decompose_query,
    evidence_is_appropriate,
    retrieve_multi_intent,
    reuse_validation,
)
from flowcontext.replay import replay_transcript
from flowcontext.retrieval import make_retriever
from flowcontext.streaming import replay_streaming_transcript
from flowcontext.streaming_evaluation import (
    evaluate_streaming_suite,
    load_streaming_evaluation_cases,
)


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "data" / "synthetic"
EVALUATION = ROOT / "data" / "evaluation"


def _event(sequence: int, text: str, *, final: bool = False) -> TranscriptEvent:
    return TranscriptEvent(
        session_id="phase3-test-session",
        utterance_id="phase3-test-utterance",
        event_id=f"phase3-test-event-{sequence}",
        sequence_number=sequence,
        source_timestamp_s=float(sequence - 1) * 0.2,
        text=text,
        is_final=final,
        text_mode="cumulative",
    )


class PartialFailureRetriever:
    backend = "lexical"
    method = "partial_failure_test_retriever"

    def __init__(self, base) -> None:
        self.base = base

    def search(self, query: str):
        if "cancellation" in query.casefold():
            raise RuntimeError("intent-specific test failure")
        return self.base.search(query)


class ScriptedRetriever:
    """Small deterministic backend fixture for retrieval assembly tests."""

    def __init__(self, backend: str, responses, *, delay_s: float = 0.0) -> None:
        self.backend = backend
        self.method = f"{backend}_scripted"
        self.responses = responses
        self.delay_s = delay_s
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def search(self, query: str):
        if self.delay_s:
            time.sleep(self.delay_s)
        with self._lock:
            self.calls.append(query)
        response = self.responses(query) if callable(self.responses) else self.responses
        if isinstance(response, BaseException):
            raise response
        return [item.model_copy() for item in response]


def _scripted_hit(chunk_id: str, text: str, *, rank: int = 1, score: float = 1.0) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source_location=f"fixture://{chunk_id}",
        snippet_text=text,
        rank=rank,
        score=score,
        retrieval_method="scripted",
    )


class InvalidThenValidDecompositionProvider:
    """Provider fixture proving repair is bounded and locally validated."""

    def __init__(self) -> None:
        self._config = GenerationConfig(
            backend="mock",
            provider="test.invalid-then-valid",
            model="test-decomposer",
            timeout_s=0.2,
            max_retries=0,
            retry_backoff_s=0.0,
            max_repair_attempts=1,
            max_output_tokens=500,
        )
        self.calls = 0

    @property
    def config(self) -> GenerationConfig:
        return self._config

    async def decompose(self, request):
        self.calls += 1
        if self.calls == 1:
            return GenerationResult(raw_text="{not valid json")
        plan = decompose_query(
            request.original_transcript,
            transcript_revision=request.transcript_revision,
            max_intents=request.max_intents,
        )
        return GenerationResult(raw_text=plan.model_dump_json())


class ConstraintDroppingDecompositionProvider:
    """Provider fixture proving required transcript constraints are enforced."""

    def __init__(self) -> None:
        self._config = GenerationConfig(
            backend="mock",
            provider="test.constraint-dropping",
            model="test-decomposer",
            timeout_s=0.2,
            max_retries=0,
            retry_backoff_s=0.0,
            max_repair_attempts=0,
            max_output_tokens=500,
        )

    @property
    def config(self) -> GenerationConfig:
        return self._config

    async def decompose(self, request):
        plan = decompose_query(
            request.original_transcript,
            transcript_revision=request.transcript_revision,
            max_intents=request.max_intents,
        )
        return GenerationResult(
            raw_text=plan.model_copy(
                update={
                    "shared_constraints": [],
                    "intents": [
                        intent.model_copy(update={"constraints": []})
                        for intent in plan.intents
                    ],
                }
            ).model_dump_json()
        )


class InventingValueDecompositionProvider:
    """Provider fixture proving new retrieval values are rejected."""

    def __init__(self) -> None:
        self._config = GenerationConfig(
            backend="mock",
            provider="test.inventing-value",
            model="test-decomposer",
            timeout_s=0.2,
            max_retries=0,
            retry_backoff_s=0.0,
            max_repair_attempts=0,
            max_output_tokens=500,
        )

    @property
    def config(self) -> GenerationConfig:
        return self._config

    async def decompose(self, request):
        plan = decompose_query(
            request.original_transcript,
            transcript_revision=request.transcript_revision,
            max_intents=request.max_intents,
        )
        intent = plan.intents[0].model_copy(
            update={"query": f"{plan.intents[0].query} in Paris"}
        )
        return GenerationResult(
            raw_text=plan.model_copy(update={"intents": [intent]}).model_dump_json()
        )


class Phase3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CorpusIngestor().ingest(
            load_document_inputs(SYNTHETIC / "documents.jsonl"),
            "synthetic_fixture",
        )
        cls.retriever = make_retriever(cls.index, backend="lexical", top_k=5)

    def test_decomposer_keeps_noun_phrase_and_expands_shared_heads(self) -> None:
        compound = decompose_query(
            "Which Pune venues can host 30 or more attendees, and what are "
            "the cancellation and catering policies?"
        )
        self.assertEqual(
            [intent.query for intent in compound.intents],
            [
                "Which Pune venues can host 30 or more attendees",
                "what are the cancellation policies",
                "what are the catering policies?",
            ],
        )
        self.assertEqual(
            {constraint.kind for constraint in compound.intents[0].constraints},
            {"entity", "quantity"},
        )
        self.assertFalse(compound.shared_constraints)
        noun_phrase = decompose_query(
            "Which venue includes a projector and a breakout room?"
        )
        self.assertEqual(len(noun_phrase.intents), 1)

    def test_package_conjunction_is_one_information_need(self) -> None:
        query = "What comes with the standard lunch and drinks package?"
        plan = decompose_query(query)
        self.assertEqual(len(plan.intents), 1)
        self.assertEqual(plan.intents[0].source_span.text, query)
        self.assertIn("lunch", plan.intents[0].query.casefold())
        self.assertIn("drinks", plan.intents[0].query.casefold())

    def test_weak_lexical_candidate_stays_out_of_answer_evidence(self) -> None:
        query = "What is the organiser's registration tax number?"
        plan = decompose_query(query)
        candidate = _scripted_hit(
            "parking",
            "Synthetic workshop parking information is not specified in the available venue options.",
        )
        result = retrieve_multi_intent(
            plan,
            ScriptedRetriever("lexical", [candidate]),
            top_k=5,
        )
        self.assertTrue(result.fused_hits)
        self.assertFalse(result.answer_evidence_hits)
        self.assertEqual(result.missing_intent_ids, plan.intent_ids)

    def test_one_question_preserves_multiple_constraints_and_spans(self) -> None:
        query = (
            "Which Pune venue can host at least 30 attendees and does not allow "
            "outside catering on 2026-10-02?"
        )
        plan = decompose_query(query, transcript_revision=7)
        self.assertEqual(len(plan.intents), 1)
        intent = plan.intents[0]
        self.assertEqual(plan.transcript_revision, 7)
        self.assertEqual(plan.original_transcript, query)
        self.assertEqual(plan.intent_ids, [intent.intent_id for intent in plan.intents])
        self.assertEqual(intent.source_span.text, query)
        kinds = {constraint.kind for constraint in intent.constraints}
        self.assertTrue({"entity", "quantity", "negation", "date"} <= kinds)
        for constraint in intent.constraints:
            self.assertEqual(query[constraint.source_span.start:constraint.source_span.end], constraint.source_span.text)
            self.assertIn(constraint.value.casefold(), constraint.source_span.text.casefold())

        varied = decompose_query(
            "Find laptops under $1000 and with 16GB RAM in Delhi."
        )
        self.assertEqual(len(varied.intents), 1)
        self.assertEqual(
            {constraint.kind for constraint in varied.intents[0].constraints},
            {"quantity", "location"},
        )
        self.assertIn("under $1000", varied.sub_questions[0])
        self.assertIn("16GB RAM", varied.sub_questions[0])

    def test_independent_questions_are_distinct(self) -> None:
        plan = decompose_query(
            "What is the cancellation policy? What are the catering options?"
        )
        self.assertEqual(len(plan.intents), 2)
        self.assertEqual(
            {intent.relationship for intent in plan.intents},
            {"independent"},
        )
        self.assertFalse(plan.dependencies)

    def test_dependent_question_is_marked_and_keeps_reference_ambiguity(self) -> None:
        plan = decompose_query(
            "Which venues meet the budget, and for each selected venue what is its cancellation policy?"
        )
        self.assertEqual(len(plan.intents), 2)
        self.assertEqual(plan.intents[1].relationship, "dependent")
        self.assertEqual(plan.dependencies[0].relation, "depends_on")
        self.assertEqual(
            plan.dependencies[0].prerequisite_intent_id,
            plan.intents[0].intent_id,
        )
        self.assertTrue(plan.ambiguities)
        self.assertTrue(any("selected venue" in item.description for item in plan.ambiguities))

    def test_comparison_remains_one_related_information_need(self) -> None:
        plan = decompose_query(
            "Compare Plan Alpha versus Plan Beta for price and capacity."
        )
        self.assertEqual(len(plan.intents), 1)
        self.assertEqual(plan.intents[0].relationship, "comparison")
        self.assertTrue(any(item.kind == "comparison" for item in plan.intents[0].constraints))
        self.assertIn("Plan Alpha", plan.intents[0].query)
        self.assertIn("Plan Beta", plan.intents[0].query)

    def test_negation_is_not_dropped(self) -> None:
        plan = decompose_query("Which venues do not allow outside catering in Pune?")
        self.assertEqual(len(plan.intents), 1)
        negations = [item for item in plan.intents[0].constraints if item.kind == "negation"]
        self.assertEqual(len(negations), 1)
        self.assertIn("not", negations[0].value.casefold())
        self.assertIn("not", plan.sub_questions[0].casefold())

    def test_repeated_wording_is_deduplicated_without_merging_different_needs(self) -> None:
        plan = decompose_query(
            "What is the cancellation policy? What are the cancellation policies?"
        )
        self.assertEqual(len(plan.intents), 1)
        separate = decompose_query(
            "What is the cancellation policy? What are the catering options?"
        )
        self.assertEqual(len(separate.intents), 2)

    def test_revision_reuses_only_the_unchanged_intent(self) -> None:
        adapter = MultiIntentRetriever(self.retriever, top_k=5)
        first_query = "What is the cancellation policy and what are the catering options?"
        second_query = "What is the cancellation policy and what are the parking options?"
        adapter.search_with_revision(first_query, parent_revision=1, retrieval_revision=1)
        adapter.search_with_revision(second_query, parent_revision=2, retrieval_revision=2)
        self.assertEqual(len(adapter.last_result.reused_intent_ids), 1)
        self.assertEqual(len(adapter.last_result.superseded_intent_ids), 1)
        self.assertTrue(next(item for item in adapter.last_result.intent_results if item.reused).reused)
        self.assertIn(
            "parking",
            next(item.intent.query for item in adapter.last_result.intent_results if not item.reused).casefold(),
        )

    def test_invalid_provider_output_is_repaired_and_locally_validated(self) -> None:
        provider = InvalidThenValidDecompositionProvider()
        decomposer = StructuredMultiIntentDecomposer(provider)
        result = asyncio.run(
            decomposer.decompose(
                "What is the cancellation policy? What are the catering options?",
                transcript_revision=4,
            )
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.repair_attempts, 1)
        self.assertEqual(len(result.intents), 2)

    def test_provider_failure_requires_explicit_fallback_and_timeout_is_traced(self) -> None:
        invalid = MockDecompositionProvider(mode="invalid_schema")
        decomposer = StructuredMultiIntentDecomposer(invalid)
        with self.assertRaises(DecompositionProviderError):
            asyncio.run(
                decomposer.decompose(
                    "What is the cancellation policy?",
                    transcript_revision=2,
                )
            )
        fallback = asyncio.run(
            decomposer.decompose(
                "What is the cancellation policy?",
                transcript_revision=2,
                allow_original_query_fallback=True,
            )
        )
        self.assertEqual(fallback.status, "fallback")
        self.assertNotEqual(fallback.decomposition_method, "model_backed_structured_validated_v1")
        self.assertTrue(fallback.failure_reason)

        timeout_provider = MockDecompositionProvider(
            mode="timeout",
            config=GenerationConfig(
                backend="mock",
                provider="test.timeout-decomposer",
                model="test-timeout",
                timeout_s=0.01,
                max_retries=0,
                retry_backoff_s=0.0,
                max_repair_attempts=0,
                max_output_tokens=100,
            ),
            timeout_delay_s=0.05,
        )
        timeout_result = asyncio.run(
            StructuredMultiIntentDecomposer(timeout_provider).decompose(
                "What is the catering policy?",
                transcript_revision=3,
                allow_original_query_fallback=True,
            )
        )
        self.assertEqual(timeout_result.status, "fallback")
        self.assertIn("timed out", timeout_result.failure_reason)

        with self.assertRaises(DecompositionTimeout):
            asyncio.run(
                StructuredMultiIntentDecomposer(timeout_provider).decompose(
                    "What is the catering policy?",
                    transcript_revision=3,
                )
            )

    def test_provider_cannot_drop_transcript_constraints(self) -> None:
        with self.assertRaises(DecompositionProviderError):
            asyncio.run(
                StructuredMultiIntentDecomposer(
                    ConstraintDroppingDecompositionProvider()
                ).decompose(
                    "Which venue in Pune can host at least 30 attendees and does not allow outside catering on 2026-10-02?",
                    transcript_revision=8,
                )
            )

    def test_provider_cannot_invent_missing_query_values(self) -> None:
        with self.assertRaises(DecompositionProviderError):
            asyncio.run(
                StructuredMultiIntentDecomposer(
                    InventingValueDecompositionProvider()
                ).decompose(
                    "Which venue is in Pune?",
                    transcript_revision=8,
                )
            )

    def test_configured_real_provider_failure_is_explicit_and_terminates(self) -> None:
        config = GenerationConfig(
            backend="openai_compatible",
            provider="test.real.decomposer",
            model="test-real-decomposer",
            base_url="https://example.invalid/v1",
            api_key_env="FLOWCONTEXT_PHASE3_TEST_MISSING_KEY",
            timeout_s=0.2,
            max_retries=0,
            retry_backoff_s=0.0,
            max_repair_attempts=0,
            max_output_tokens=500,
        )
        provider = OpenAICompatibleDecompositionProvider(config=config)
        with patch.dict(os.environ, {config.api_key_env: ""}, clear=False):
            result = asyncio.run(
                StructuredMultiIntentDecomposer(provider).decompose(
                    "What is the catering policy?",
                    transcript_revision=9,
                    allow_original_query_fallback=True,
                )
            )
        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.provider, "test.real.decomposer")
        self.assertIn("credentials", result.failure_reason)

    def test_parallel_retrieval_fuses_deduplicates_and_keeps_provenance(self) -> None:
        plan = decompose_query(
            "Which venue is in Pune and what is the cancellation policy?"
        )
        result = retrieve_multi_intent(plan, self.retriever, top_k=5)
        self.assertEqual([item.status for item in result.intent_results], ["completed", "completed"])
        self.assertTrue(result.fused_hits)
        self.assertEqual(
            len({hit.chunk_id for hit in result.fused_hits}),
            len(result.fused_hits),
        )
        self.assertTrue(all(hit.retrieval_method == "rrf_fusion" for hit in result.fused_hits))
        self.assertTrue(all(hit.intent_ids for hit in result.fused_hits))
        self.assertTrue(any(len(hit.intent_ids) == 1 for hit in result.fused_hits))
        self.assertTrue(any(hit.rrf_score is not None for hit in result.fused_hits))

    def test_hybrid_fuses_complementary_rankings_without_adding_raw_scores(self) -> None:
        plan = decompose_query("What is the cancellation policy?")
        lexical = ScriptedRetriever(
            "lexical",
            [
                _scripted_hit("lexical-best", "cancellation policy details", score=0.95),
                _scripted_hit("semantic-best", "terms for ending a booking", rank=2, score=0.20),
            ],
        )
        dense = ScriptedRetriever(
            "dense",
            [
                _scripted_hit("semantic-best", "terms for ending a booking", score=0.87),
                _scripted_hit("lexical-best", "cancellation policy details", rank=2, score=0.11),
            ],
        )
        result = retrieve_multi_intent(
            plan,
            lexical,
            retrieval_mode="hybrid",
            dense_retriever=dense,
            top_k=2,
        )
        intent_result = result.intent_results[0]
        self.assertEqual(set(intent_result.backend_hits), {"lexical", "dense"})
        self.assertEqual(result.retrieval_mode, "hybrid")
        self.assertEqual([hit.chunk_id for hit in intent_result.hits], [
            "lexical-best",
            "semantic-best",
        ])
        self.assertTrue(all(hit.retrieval_method == "rrf_fusion" for hit in intent_result.hits))
        self.assertEqual(intent_result.hits[0].score, intent_result.hits[0].rrf_score)
        self.assertNotEqual(intent_result.hits[0].score, 0.95 + 0.11)
        self.assertTrue(any(
            decision["stage"] == "per_intent_backend_fusion"
            and decision["input_ranks"] == {"lexical": 1, "dense": 2}
            for decision in intent_result.fusion_decisions
        ))

    def test_independent_intents_run_concurrently_with_a_bound(self) -> None:
        plan = decompose_query(
            "What is the cancellation policy? What are the catering options? Where is the venue?"
        )
        active = 0
        maximum = 0
        lock = threading.Lock()

        class BoundedRetriever(ScriptedRetriever):
            def search(self, query: str):
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                try:
                    time.sleep(0.03)
                    return [_scripted_hit("shared", "shared evidence")]
                finally:
                    with lock:
                        active -= 1

        result = retrieve_multi_intent(
            plan,
            BoundedRetriever("lexical", []),
            top_k=5,
            max_workers=2,
        )
        self.assertEqual(len(result.intent_results), 3)
        self.assertGreaterEqual(maximum, 2)
        self.assertLessEqual(maximum, 2)
        self.assertEqual(set(result.fused_hits[0].intent_ids), set(plan.intent_ids))

    def test_duplicate_chunk_across_intents_keeps_all_support_relationships(self) -> None:
        plan = decompose_query("What is the cancellation policy? What are the catering options?")
        shared = _scripted_hit("shared-policy", "cancellation policy and catering options")

        def response(query: str):
            if "catering" in query.casefold():
                return [shared, _scripted_hit("catering-only", "catering menu options", rank=2)]
            return [shared, _scripted_hit("cancellation-only", "cancellation policy", rank=2)]

        result = retrieve_multi_intent(plan, ScriptedRetriever("lexical", response), top_k=5)
        shared_hits = [hit for hit in result.fused_hits if hit.chunk_id == "shared-policy"]
        self.assertEqual(len(shared_hits), 1)
        self.assertEqual(set(shared_hits[0].intent_ids), set(plan.intent_ids))
        self.assertEqual(set(shared_hits[0].intent_ranks), set(plan.intent_ids))
        self.assertEqual(len({hit.chunk_id for hit in result.fused_hits}), len(result.fused_hits))
        self.assertEqual(shared_hits[0].source_location, "fixture://shared-policy")

    def test_assembly_budget_is_fair_to_an_uneven_intent_distribution(self) -> None:
        plan = decompose_query("What are the cancellation options? What are the catering options?")

        def response(query: str):
            if "catering" in query.casefold():
                return [_scripted_hit("catering", "catering options menu", score=0.1)]
            return [
                _scripted_hit("cancel-1", "cancellation options policy", score=1.0),
                _scripted_hit("cancel-2", "cancellation refund timing", rank=2, score=0.9),
                _scripted_hit("cancel-3", "cancellation contact details", rank=3, score=0.8),
            ]

        result = retrieve_multi_intent(
            plan,
            ScriptedRetriever("lexical", response),
            top_k=4,
            context_budget_tokens=6,
        )
        selected_ids = {hit.chunk_id for hit in result.fused_hits}
        self.assertIn("cancel-1", selected_ids)
        self.assertIn("catering", selected_ids)
        self.assertEqual(result.missing_intent_ids, [])
        self.assertLessEqual(result.context_tokens_used, 6)
        self.assertTrue(any(
            decision["reason"] == "context_budget" and not decision["selected"]
            for decision in result.assembly_decisions
        ))

    def test_missing_intent_is_explicit_even_when_another_intent_has_hits(self) -> None:
        plan = decompose_query("What are the cancellation options? What are the catering options?")

        def response(query: str):
            return [] if "catering" in query.casefold() else [
                _scripted_hit("cancel", "cancellation options policy")
            ]

        result = retrieve_multi_intent(plan, ScriptedRetriever("lexical", response), top_k=5)
        missing_query = next(
            intent.query for intent in plan.intents if intent.intent_id in result.missing_intent_ids
        )
        self.assertIn("catering", missing_query.casefold())
        self.assertNotIn("catering", {hit.chunk_id for hit in result.fused_hits})

    def test_hybrid_partial_backend_failure_is_traced_without_mislabelling_success(self) -> None:
        plan = decompose_query("What is the cancellation policy?")
        lexical = ScriptedRetriever("lexical", [_scripted_hit("cancel", "cancellation policy")])
        dense = ScriptedRetriever("dense", RuntimeError("dense smoke unavailable"))
        result = retrieve_multi_intent(
            plan,
            lexical,
            retrieval_mode="hybrid",
            dense_retriever=dense,
            top_k=5,
        )
        item = result.intent_results[0]
        self.assertEqual(item.status, "completed")
        self.assertTrue(item.backend_errors)
        self.assertEqual(result.retrieval_mode, "hybrid")
        self.assertTrue(result.fused_hits)
        self.assertIn("dense", item.backend_errors[0])

        failing_lexical = ScriptedRetriever("lexical", RuntimeError("lexical failed"))
        failing_dense = ScriptedRetriever("dense", RuntimeError("dense failed"))
        with self.assertRaises(MultiIntentRetrievalError) as caught:
            retrieve_multi_intent(
                plan,
                failing_lexical,
                retrieval_mode="hybrid",
                dense_retriever=failing_dense,
                top_k=5,
            )
        self.assertEqual(caught.exception.result.intent_results[0].status, "failed")
        self.assertFalse(caught.exception.result.fused_hits)

    def test_dependent_retrieval_waits_for_bounded_prerequisite_context(self) -> None:
        plan = decompose_query(
            "Which venues meet the budget, and for each selected venue what is its cancellation policy?"
        )
        calls: list[str] = []

        def response(query: str):
            calls.append(query)
            if "selected venue" in query.casefold():
                self.assertIn("Venue Alpha", query)
                return [_scripted_hit("alpha-policy", "Venue Alpha cancellation policy")]
            return [_scripted_hit("alpha", "Venue Alpha meets the budget")]

        result = retrieve_multi_intent(plan, ScriptedRetriever("lexical", response), top_k=5)
        self.assertEqual(result.intent_results[1].status, "completed")
        self.assertEqual(result.intent_results[1].dependency_intent_ids, [plan.intents[0].intent_id])
        self.assertEqual(result.intent_results[1].dependency_context_chunk_ids, ["alpha"])
        self.assertEqual(result.dependency_order, plan.intent_ids)
        self.assertEqual(len(calls), 2)

    def test_dependent_retrieval_does_not_run_after_empty_prerequisite(self) -> None:
        plan = decompose_query(
            "Which venues meet the budget, and for each selected venue what is its cancellation policy?"
        )
        dependent_queries: list[str] = []

        def response(query: str):
            if "selected venue" in query.casefold():
                dependent_queries.append(query)
            return []

        result = retrieve_multi_intent(plan, ScriptedRetriever("lexical", response), top_k=5)
        self.assertEqual(result.intent_results[0].status, "completed")
        self.assertEqual(result.intent_results[1].status, "failed")
        self.assertEqual(dependent_queries, [])
        self.assertIn(plan.intents[1].intent_id, result.missing_intent_ids)

    def test_late_old_revision_cannot_replace_current_multi_intent_result(self) -> None:
        def response(query: str):
            if "old" in query.casefold():
                time.sleep(0.08)
            return [_scripted_hit("current" if "new" in query.casefold() else "old", query)]

        adapter = MultiIntentRetriever(ScriptedRetriever("lexical", response), top_k=5)
        old_thread = threading.Thread(
            target=adapter.search_with_revision,
            kwargs={"query": "What is the old cancellation policy?", "parent_revision": 1, "retrieval_revision": 1},
        )
        old_thread.start()
        time.sleep(0.01)
        adapter.search_with_revision(
            "What is the new cancellation policy?",
            parent_revision=2,
            retrieval_revision=2,
        )
        old_thread.join()
        self.assertEqual(adapter.last_result.parent_revision, 2)
        self.assertEqual(adapter.last_result.retrieval_revision, 2)
        self.assertIsNotNone(adapter.result_for(1))

    def test_decomposition_artifact_uses_retrieval_revision_key(self) -> None:
        adapter = MultiIntentRetriever(self.retriever, top_k=5)
        adapter.search_with_revision(
            "What is the cancellation policy?",
            parent_revision=17,
            retrieval_revision=3,
        )
        decomposition = adapter.decomposition_for(3)
        self.assertIsNotNone(decomposition)
        self.assertEqual(decomposition.transcript_revision, 17)

    def test_reranking_is_disabled_by_default_and_context_is_session_local(self) -> None:
        reranker_calls: list[str] = []

        def reranker(query: str, hits):
            reranker_calls.append(query)
            return list(reversed(hits))

        base = ScriptedRetriever("lexical", [_scripted_hit("policy", "cancellation policy")])
        adapter_a = MultiIntentRetriever(base, top_k=5, reranker=reranker)
        adapter_b = MultiIntentRetriever(base, top_k=5, reranker=reranker)
        adapter_a.search_with_revision("What is the cancellation policy?", parent_revision=1, retrieval_revision=1)
        adapter_b.search_with_revision("What is the cancellation policy?", parent_revision=1, retrieval_revision=1)
        self.assertEqual(reranker_calls, [])
        self.assertIsNot(adapter_a.last_result, adapter_b.last_result)
        self.assertEqual(adapter_a.last_result.retrieval_revision, adapter_b.last_result.retrieval_revision)

    def test_valid_chunk_id_without_final_query_support_cannot_be_reused(self) -> None:
        venue_hit = self.retriever.search("venue Pune")[0]
        validation = reuse_validation("What is the cancellation policy?", [venue_hit])
        self.assertEqual(validation["decision"], "rejected")
        self.assertFalse(validation["lexical_relevance_proxy"])
        self.assertEqual(validation["semantic_support"], "not_evaluated")
        self.assertFalse(evidence_is_appropriate("What is the cancellation policy?", [venue_hit]))

    def test_unannotated_hits_cannot_claim_coverage_for_multiple_intents(self) -> None:
        plan = decompose_query("What is the cancellation policy? What are the catering options?")
        hit = _scripted_hit("shared", "cancellation policy only")
        validation = reuse_validation(plan.original_transcript, [hit], plan=plan)
        self.assertEqual(validation["decision"], "rejected")
        self.assertFalse(validation["lexical_relevance_proxy"])

    def test_partial_intent_failure_is_explicitly_uncertain(self) -> None:
        result = asyncio.run(
            replay_transcript(
                [_event(1, "Which venue in Pune and what is the cancellation policy?", final=True)],
                corpus=self.index,
                retriever=PartialFailureRetriever(self.retriever),
                generation_provider=MockGenerationProvider(),
                backend="lexical",
                multi_intent=True,
            )
        )
        self.assertEqual(result.run_status, "completed")
        self.assertIn("Evidence was unavailable for one or more requested intents", result.answer.uncertainty)
        self.assertTrue(result.answer.factual_claims)
        self.assertTrue(all(
            claim.supporting_chunk_ids[0] in {hit.chunk_id for hit in result.retrieval_hits}
            for claim in result.answer.factual_claims
        ))
        self.assertIsNotNone(result.retrieval_details)
        self.assertTrue(result.retrieval_details["missing_intent_ids"])

    def test_completed_no_hit_streaming_search_is_abstention_not_failure(self) -> None:
        result = asyncio.run(
            replay_streaming_transcript(
                [_event(1, "What is the organiser's registration tax number?", final=True)],
                corpus=self.index,
                retriever=self.retriever,
                backend="lexical",
                execution_mode="accelerated",
                multi_intent=True,
            )
        )
        self.assertEqual(result.run_status, "abstained")
        self.assertEqual(result.generation_status, "skipped")
        self.assertFalse(result.errors)

    def test_streaming_persists_final_multi_intent_retrieval_details(self) -> None:
        result = asyncio.run(
            replay_streaming_transcript(
                [_event(1, "What is the cancellation policy and what are the catering options?", final=True)],
                corpus=self.index,
                retriever=self.retriever,
                backend="lexical",
                execution_mode="accelerated",
                multi_intent=True,
            )
        )
        self.assertIsNotNone(result.retrieval_details)
        self.assertEqual(result.retrieval_details["retrieval_mode"], "lexical")
        self.assertEqual(len(result.retrieval_details["intent_results"]), 2)
        self.assertTrue(any(
            trace.event_type == "streaming_retrieval_assembly_completed"
            and trace.attributes.get("retrieval_mode") == "lexical"
            for trace in result.traces
        ))

    def test_streaming_revision_does_not_reuse_early_single_intent_for_compound_final(self) -> None:
        result = asyncio.run(
            replay_streaming_transcript(
                [
                    _event(1, "Which venue in Pune"),
                    _event(2, "Which venue in Pune and what is the cancellation policy?", final=True),
                ],
                corpus=self.index,
                retriever=self.retriever,
                backend="lexical",
                execution_mode="accelerated",
                multi_intent=True,
            )
        )
        self.assertFalse(result.early_evidence_reused)
        self.assertTrue(any(
            hit.chunk_id == "synthetic.cancellation-policy#chunk-0000"
            for hit in result.current_evidence_hits
        ))
        self.assertTrue(any(
            trace.event_type == "streaming_timing_summary"
            and trace.attributes.get("multi_intent") is True
            for trace in result.traces
        ))

    def test_model_decomposer_runs_at_stable_revisions_not_every_token(self) -> None:
        provider = MockDecompositionProvider(
            config=GenerationConfig(
                backend="mock",
                provider="test.counted-decomposer",
                model="test-decomposer",
                timeout_s=0.2,
                max_retries=0,
                retry_backoff_s=0.0,
                max_repair_attempts=0,
                max_output_tokens=500,
            )
        )
        result = asyncio.run(
            replay_streaming_transcript(
                [
                    _event(1, "Which venue in Pune"),
                    _event(2, "Which venue in Pune and"),
                    _event(3, "Which venue in Pune and what is the cancellation policy?", final=True),
                ],
                corpus=self.index,
                retriever=self.retriever,
                backend="lexical",
                execution_mode="accelerated",
                decomposition_provider=provider,
                multi_intent=True,
            )
        )
        self.assertLess(provider.calls, 3)
        self.assertEqual(provider.calls, len(result.decomposition_history))
        self.assertTrue(any(
            trace.event_type in {"streaming_decomposition_completed", "streaming_decomposition_obsolete"}
            for trace in result.traces
        ))

    def test_quality_denominator_keeps_failed_case_out_of_quality_but_in_report(self) -> None:
        cases = load_streaming_evaluation_cases(
            EVALUATION / "streaming_development.jsonl",
            expected_split="development",
        )
        selected = [
            case
            for case in cases
            if case.case_id in {"stream-dev-stable-partial", "stream-dev-retrieval-failure"}
        ]
        report = asyncio.run(
            evaluate_streaming_suite(
                selected,
                corpus=self.index,
                settings=load_settings(),
                backend="lexical",
                top_k=5,
                execution_mode="accelerated",
            )
        )
        all_results = [
            result
            for section in report.sections.values()
            for result in section.case_results
        ]
        failed = next(result for result in all_results if result.case_id == "stream-dev-retrieval-failure")
        self.assertFalse(failed.baseline.quality_scored)
        self.assertFalse(failed.streaming.quality_scored)
        self.assertEqual(failed.baseline.scoring_denominator, 1)
        self.assertEqual(failed.streaming.scoring_denominator, 1)
        self.assertEqual(failed.baseline.run_status, "failed")
        self.assertEqual(failed.streaming.run_status, "failed")


if __name__ == "__main__":
    unittest.main()
