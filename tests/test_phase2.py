from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from flowcontext.config import load_settings
from flowcontext.contracts import (
    PreviousAnswerContext,
    StreamingDecisionConfig,
    TranscriptEvent,
)
from flowcontext.ingestion import CorpusIngestor, load_document_inputs
from flowcontext.replay import ReplayError, load_transcript
from flowcontext.streaming import (
    StreamingControllerError,
    StreamingDecisionController,
    replay_streaming_transcript,
)
from flowcontext.retrieval import make_retriever


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "data" / "synthetic"


def _event(
    sequence: int,
    text: str,
    *,
    final: bool = False,
    mode: str = "incremental",
    session: str = "phase2-session",
    utterance: str = "phase2-utterance",
    timestamp: float | None = None,
) -> TranscriptEvent:
    return TranscriptEvent(
        session_id=session,
        utterance_id=utterance,
        event_id=f"{session}-{sequence}-{final}",
        sequence_number=sequence,
        source_timestamp_s=float(sequence - 1 if timestamp is None else timestamp),
        text=text,
        is_final=final,
        text_mode=mode,
    )


class Phase2StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CorpusIngestor().ingest(
            load_document_inputs(SYNTHETIC / "documents.jsonl"),
            "synthetic_fixture",
        )

    def test_incomplete_partial_waits_and_final_decision_retrieves(self) -> None:
        events = [
            _event(1, "Which ", timestamp=0.0),
            _event(2, "venue in Pune?", final=True, timestamp=0.2),
        ]
        result = asyncio.run(
            replay_streaming_transcript(events, corpus=self.index, backend="lexical")
        )
        self.assertEqual(result.decisions[0].decision, "WAIT")
        self.assertEqual(result.decisions[0].reason_code, "incomplete_phrase")
        self.assertEqual(result.early_retrieval_count, 0)
        self.assertEqual(result.final_retrieval_count, 1)
        self.assertEqual(result.final_decision.decision, "RETRIEVE")

    def test_meaningful_partial_starts_retrieval_before_final(self) -> None:
        events = [
            _event(1, "Which venue", timestamp=0.0),
            _event(2, " in Pune?", final=True, timestamp=0.2),
        ]
        result = asyncio.run(
            replay_streaming_transcript(events, corpus=self.index, backend="lexical")
        )
        self.assertEqual(result.early_retrieval_count, 1)
        self.assertEqual(result.final_retrieval_count, 1)
        self.assertLess(
            result.retrieval_results[0].source_timestamp_s,
            result.final_source_timestamp_s,
        )
        self.assertEqual(
            result.retrieval_results[0].query,
            "Which venue",
        )

    def test_incremental_and_cumulative_updates_make_same_final_query(self) -> None:
        incremental = [
            _event(1, "Which venue", timestamp=0.0),
            _event(2, " in Pune?", final=True, timestamp=0.2),
        ]
        cumulative = [
            _event(
                1,
                "Which venue",
                mode="cumulative",
                session="cumulative-phase2",
                utterance="cumulative-utterance",
                timestamp=0.0,
            ),
            _event(
                2,
                "Which venue in Pune?",
                final=True,
                mode="cumulative",
                session="cumulative-phase2",
                utterance="cumulative-utterance",
                timestamp=0.2,
            ),
        ]
        async def run_pair():
            return await asyncio.gather(
                replay_streaming_transcript(
                    incremental,
                    corpus=self.index,
                    backend="lexical",
                    run_id="phase2-incremental",
                ),
                replay_streaming_transcript(
                    cumulative,
                    corpus=self.index,
                    backend="lexical",
                    run_id="phase2-cumulative",
                ),
            )
        first, second = asyncio.run(run_pair())
        self.assertEqual(first.final_query, second.final_query)
        self.assertEqual(first.early_retrieval_count, second.early_retrieval_count)
        self.assertEqual(
            [decision.decision for decision in first.decisions],
            [decision.decision for decision in second.decisions],
        )

    def test_repeated_fragment_is_suppressed_and_duplicate_final_is_idempotent(self) -> None:
        first = _event(1, "Which venue", timestamp=0.0)
        repeated_fragment = _event(2, "Which venue", mode="cumulative", timestamp=0.1)
        repeated_fragment = repeated_fragment.model_copy(
            update={"event_id": "phase2-session-repeated-fragment"}
        )
        final = _event(3, "Which venue", final=True, mode="cumulative", timestamp=0.2)
        repeated_final = final.model_copy(
            update={
                "event_id": "phase2-session-repeated-final",
                "sequence_number": 4,
                "source_timestamp_s": 0.3,
            }
        )
        result = asyncio.run(
            replay_streaming_transcript(
                [first, repeated_fragment, final, repeated_final],
                corpus=self.index,
                backend="lexical",
            )
        )
        self.assertEqual(result.suppressed_duplicate_query_count, 2)
        self.assertEqual(result.duplicate_event_count, 1)
        self.assertEqual(result.retrieval_attempt_count, 1)
        self.assertEqual(result.decisions[-1].reason_code, "repeated_final_ignored")

    def test_correction_triggers_new_revision_bound_retrieval(self) -> None:
        events = [
            _event(1, "I need a venue", timestamp=0.0),
            _event(
                2,
                "Actually, I need cancellation policy?",
                final=True,
                mode="cumulative",
                timestamp=0.1,
            ),
        ]
        result = asyncio.run(
            replay_streaming_transcript(events, corpus=self.index, backend="lexical")
        )
        self.assertEqual(result.early_retrieval_count, 1)
        self.assertEqual(result.final_retrieval_count, 1)
        self.assertEqual(result.final_decision.reason_code, "explicit_correction")
        self.assertFalse(result.retrieval_results[0].accepted)
        self.assertTrue(result.retrieval_results[0].stale)
        self.assertTrue(result.retrieval_results[-1].accepted)
        self.assertEqual(result.unnecessary_retrieval_count, 1)

    def test_debounce_coalesces_new_partial_information_but_final_bypasses_it(self) -> None:
        config = StreamingDecisionConfig(debounce_source_s=1.0)
        controller = StreamingDecisionController(
            session_id="debounce-session",
            utterance_id="debounce-utterance",
            config=config,
        )
        first = _event(
            1,
            "Which venue",
            session="debounce-session",
            utterance="debounce-utterance",
            timestamp=0.0,
        )
        second = _event(
            2,
            "Which venue in Pune",
            mode="cumulative",
            session="debounce-session",
            utterance="debounce-utterance",
            timestamp=0.1,
        )
        final = _event(
            3,
            "Which venue in Pune?",
            final=True,
            mode="cumulative",
            session="debounce-session",
            utterance="debounce-utterance",
            timestamp=0.2,
        )
        self.assertEqual(controller.process_event(first).decision, "RETRIEVE")
        self.assertEqual(controller.process_event(second).reason_code, "debounce_coalescing")
        self.assertEqual(controller.process_event(final).decision, "RETRIEVE")
        self.assertEqual(controller.current_text, "Which venue in Pune?")

    def test_constraint_change_bypasses_debounce(self) -> None:
        controller = StreamingDecisionController(
            session_id="constraint-session",
            utterance_id="constraint-utterance",
            config=StreamingDecisionConfig(debounce_source_s=10.0),
        )
        first = _event(
            1,
            "I need a venue",
            session="constraint-session",
            utterance="constraint-utterance",
            timestamp=0.0,
        )
        second = _event(
            2,
            "I need a venue for 30 attendees",
            mode="cumulative",
            session="constraint-session",
            utterance="constraint-utterance",
            timestamp=0.1,
        )
        self.assertEqual(controller.process_event(first).decision, "RETRIEVE")
        changed = controller.process_event(second)
        self.assertEqual(changed.decision, "RETRIEVE")
        self.assertEqual(changed.reason_code, "constraint_change")

    def test_formatting_requires_current_session_context(self) -> None:
        without_context = asyncio.run(
            replay_streaming_transcript(
                [_event(1, "Please make it shorter", final=True)],
                corpus=self.index,
                backend="lexical",
            )
        )
        self.assertEqual(without_context.run_status, "needs_context")
        self.assertEqual(without_context.final_decision.decision, "SKIP")
        self.assertTrue(without_context.final_decision.needs_previous_answer_context)
        self.assertEqual(without_context.retrieval_attempt_count, 0)

        with_context = asyncio.run(
            replay_streaming_transcript(
                [
                    _event(
                        1,
                        "Please make it shorter",
                        final=True,
                        session="context-session",
                        utterance="context-utterance",
                    )
                ],
                corpus=self.index,
                backend="lexical",
                previous_answer_context=PreviousAnswerContext(
                    session_id="context-session",
                    answer_text="A prior answer from this session.",
                    answer_version=1,
                ),
            )
        )
        self.assertEqual(with_context.run_status, "completed")
        self.assertEqual(with_context.final_decision.decision, "SKIP")
        self.assertTrue(with_context.final_decision.previous_answer_context_used)

        with self.assertRaises(StreamingControllerError):
            StreamingDecisionController(
                session_id="new-session",
                utterance_id="utterance",
                previous_answer_context=PreviousAnswerContext(
                    session_id="old-session",
                    answer_text="must not leak",
                    answer_version=1,
                ),
            )

    def test_greeting_is_handled_without_retrieval(self) -> None:
        result = asyncio.run(
            replay_streaming_transcript(
                [_event(1, "Hello", final=True, mode="cumulative")],
                corpus=self.index,
                backend="lexical",
            )
        )
        self.assertEqual(result.run_status, "completed")
        self.assertEqual(result.final_decision.reason_code, "greeting_no_retrieval")
        self.assertEqual(result.retrieval_call_count, 0)
        self.assertIsNone(result.generation_started_time_s)
        self.assertIsNone(result.generation_completed_time_s)

    def test_early_metrics_require_actual_start_and_record_reuse(self) -> None:
        events = [
            _event(
                1,
                "Which venue in Pune",
                mode="cumulative",
                session="timing-session",
                utterance="timing-utterance",
                timestamp=0.0,
            ),
            _event(
                2,
                "Which venue in Pune?",
                final=True,
                mode="cumulative",
                session="timing-session",
                utterance="timing-utterance",
                timestamp=0.1,
            ),
        ]
        result = asyncio.run(
            replay_streaming_transcript(
                events,
                corpus=self.index,
                backend="lexical",
                execution_mode="realtime",
            )
        )
        self.assertTrue(result.retrieval_started_early)
        self.assertTrue(result.valid_evidence_ready_before_finalization)
        self.assertTrue(result.early_evidence_reused)
        self.assertEqual(result.retrieval_call_count, 1)
        early_result = result.retrieval_results[0]
        self.assertIsNotNone(early_result.actual_started_time_s)
        self.assertLess(early_result.actual_started_time_s, result.final_event_delivery_time_s)

    def test_stale_result_is_not_accepted_after_revision(self) -> None:
        retriever = make_retriever(self.index, backend="lexical", top_k=2)
        controller = StreamingDecisionController(
            session_id="stale-session",
            utterance_id="stale-utterance",
            config=StreamingDecisionConfig(debounce_source_s=0.0),
            known_chunk_locations={chunk.chunk_id: chunk.source_location for chunk in self.index.chunks},
        )
        first_decision = controller.process_event(
            _event(
                1,
                "Which venue",
                session="stale-session",
                utterance="stale-utterance",
                timestamp=0.0,
            )
        )
        request = controller.begin_retrieval(first_decision)
        controller.process_event(
            _event(
                2,
                "Which venue in Pune",
                mode="cumulative",
                session="stale-session",
                utterance="stale-utterance",
                timestamp=0.1,
            )
        )
        stale = controller.complete_retrieval(request, retriever.search(request.query))
        self.assertTrue(stale.stale)
        self.assertFalse(stale.accepted)
        self.assertEqual(controller.current_evidence_hits, [])

    def test_concurrent_streaming_sessions_remain_isolated(self) -> None:
        first_events = [
            _event(
                1,
                "Which venue",
                session="session-a",
                utterance="utterance-a",
                timestamp=0.0,
            ),
            _event(
                2,
                " in Pune?",
                final=True,
                session="session-a",
                utterance="utterance-a",
                timestamp=0.1,
            ),
        ]
        second_events = [
            _event(
                1,
                "What is the cancellation",
                session="session-b",
                utterance="utterance-b",
                timestamp=0.0,
            ),
            _event(
                2,
                " policy?",
                final=True,
                session="session-b",
                utterance="utterance-b",
                timestamp=0.1,
            ),
        ]
        async def run_pair():
            return await asyncio.gather(
                replay_streaming_transcript(
                    first_events,
                    corpus=self.index,
                    backend="lexical",
                    run_id="stream-a",
                ),
                replay_streaming_transcript(
                    second_events,
                    corpus=self.index,
                    backend="lexical",
                    run_id="stream-b",
                ),
            )
        first, second = asyncio.run(run_pair())
        self.assertEqual(first.session_id, "session-a")
        self.assertEqual(second.session_id, "session-b")
        self.assertTrue(all(trace.run_id == "stream-a" for trace in first.traces))
        self.assertTrue(all(trace.run_id == "stream-b" for trace in second.traces))
        self.assertNotEqual(first.final_query, second.final_query)

    def test_streaming_trace_has_decisions_retrievals_and_completion(self) -> None:
        result = asyncio.run(
            replay_streaming_transcript(
                load_transcript(ROOT / "examples" / "replay" / "incremental.jsonl"),
                corpus=self.index,
                backend="lexical",
                run_id="trace-check",
            )
        )
        event_types = {trace.event_type for trace in result.traces}
        self.assertTrue(
            {
                "streaming_replay_started",
                "transcript_event_received",
                "streaming_decision",
                "utterance_finalized",
                "streaming_retrieval_started",
                "streaming_retrieval_completed",
                "streaming_replay_completed",
            }.issubset(event_types)
        )
        self.assertTrue(all(decision.session_id == result.session_id for decision in result.decisions))
        for retrieval in result.retrieval_results:
            for hit in retrieval.hits:
                chunk = next(item for item in self.index.chunks if item.chunk_id == hit.chunk_id)
                self.assertEqual(hit.source_location, chunk.source_location)

    def test_invalid_transcript_order_is_rejected_before_controller_runs(self) -> None:
        events = [
            _event(1, "Which venue", timestamp=0.5),
            _event(2, " in Pune?", final=True, timestamp=0.1),
        ]
        with self.assertRaises(ReplayError):
            asyncio.run(
                replay_streaming_transcript(
                    events,
                    corpus=self.index,
                    backend="lexical",
                )
            )

    def test_streaming_config_is_environment_configurable(self) -> None:
        settings = load_settings(
            environ={
                "FLOWCONTEXT_STREAMING_DEBOUNCE_SOURCE_S": "1.25",
                "FLOWCONTEXT_STREAMING_MIN_QUERY_CHARS": "12",
                "FLOWCONTEXT_STREAMING_DUPLICATE_QUERY_SUPPRESSION": "false",
                "FLOWCONTEXT_STREAMING_MAX_CONCURRENCY": "3",
                "FLOWCONTEXT_STREAMING_REQUEST_TIMEOUT_S": "1.5",
            }
        )
        self.assertEqual(settings.streaming_debounce_source_s, 1.25)
        self.assertEqual(settings.streaming_min_query_chars, 12)
        self.assertFalse(settings.streaming_duplicate_query_suppression)
        self.assertEqual(settings.streaming_max_concurrency, 3)
        self.assertEqual(settings.streaming_request_timeout_s, 1.5)
