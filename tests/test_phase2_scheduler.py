from __future__ import annotations

import asyncio
import threading
import time
import unittest
from pathlib import Path

from flowcontext.contracts import (
    GenerationRequest,
    GenerationResult,
    StreamingSchedulerConfig,
    TranscriptEvent,
)
from flowcontext.generation import MockGenerationProvider
from flowcontext.ingestion import CorpusIngestor, load_document_inputs
from flowcontext.retrieval import make_retriever
from flowcontext.scheduler import AsyncRetrievalScheduler
from flowcontext.streaming import StreamingDecisionController, replay_streaming_transcript


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "data" / "synthetic"


def _event(
    sequence: int,
    text: str,
    *,
    final: bool = False,
    mode: str = "incremental",
    session: str = "scheduler-session",
    utterance: str = "scheduler-utterance",
    timestamp: float | None = None,
) -> TranscriptEvent:
    return TranscriptEvent(
        session_id=session,
        utterance_id=utterance,
        event_id=f"{session}-event-{sequence}",
        sequence_number=sequence,
        source_timestamp_s=float(sequence - 1 if timestamp is None else timestamp),
        text=text,
        is_final=final,
        text_mode=mode,
    )


class DelayedRetriever:
    """Controllable synchronous retriever used only by scheduler tests."""

    backend = "lexical"
    method = "delayed_test_retriever"

    def __init__(self, base, *, delay_s: float = 0.0, delays: dict[str, float] | None = None, fail: bool = False):
        self.base = base
        self.delay_s = delay_s
        self.delays = delays or {}
        self.fail = fail
        self.calls: list[tuple[str, float, str]] = []
        self._lock = threading.Lock()

    def search(self, query: str):
        delay = self.delays.get(query, self.delay_s)
        with self._lock:
            self.calls.append((query, time.monotonic(), threading.current_thread().name))
        if delay:
            time.sleep(delay)
        if self.fail:
            raise RuntimeError("delayed test retriever failure")
        return self.base.search(query)


class FlakyRetriever(DelayedRetriever):
    def __init__(self, base, *, failures: int) -> None:
        super().__init__(base)
        self.failures = failures

    def search(self, query: str):
        with self._lock:
            self.calls.append((query, time.monotonic(), threading.current_thread().name))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("transient delayed test retriever failure")
        return self.base.search(query)


class RecordingMockProvider(MockGenerationProvider):
    def __init__(self, *, delay_s: float = 0.0) -> None:
        super().__init__()
        self.delay_s = delay_s
        self.supplied_chunk_ids: list[list[str]] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.supplied_chunk_ids.append([passage.chunk_id for passage in request.passages])
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return await super().generate(request)


class Phase2SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CorpusIngestor().ingest(
            load_document_inputs(SYNTHETIC / "documents.jsonl"),
            "synthetic_fixture",
        )
        cls.base_retriever = make_retriever(cls.index, backend="lexical", top_k=3)

    def test_no_retrieval_is_started_before_final_event(self) -> None:
        retriever = DelayedRetriever(self.base_retriever, delay_s=0.01)
        result = asyncio.run(
            replay_streaming_transcript(
                [
                    _event(1, "Which ", timestamp=0.0),
                    _event(2, "venue in Pune?", final=True, timestamp=0.01),
                ],
                corpus=self.index,
                retriever=retriever,
                backend="lexical",
            )
        )
        started = [trace for trace in result.traces if trace.event_type == "streaming_retrieval_started"]
        finalised_index = next(i for i, trace in enumerate(result.traces) if trace.event_type == "utterance_finalized")
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(len(started), 1)
        self.assertGreater(
            next(i for i, trace in enumerate(result.traces) if trace.event_type == "streaming_retrieval_started"),
            finalised_index,
        )

    def test_events_continue_while_early_retrieval_runs(self) -> None:
        retriever = DelayedRetriever(
            self.base_retriever,
            delays={"Which venue": 0.06, "Which venue in Pune?": 0.005},
        )
        result = asyncio.run(
            replay_streaming_transcript(
                [
                    _event(1, "Which venue", timestamp=0.0),
                    _event(2, " in Pune?", final=True, timestamp=0.01),
                ],
                corpus=self.index,
                retriever=retriever,
                backend="lexical",
                scheduler_config=StreamingSchedulerConfig(final_wait_timeout_s=1.0),
            )
        )
        early_start = next(
            i
            for i, trace in enumerate(result.traces)
            if trace.event_type == "streaming_retrieval_started"
            and trace.attributes.get("query") == "Which venue"
        )
        final_receipt = next(
            i
            for i, trace in enumerate(result.traces)
            if trace.event_type == "transcript_event_received"
            and trace.attributes.get("event_id") == "scheduler-session-event-2"
        )
        self.assertLess(early_start, final_receipt)
        self.assertEqual(result.final_query, "Which venue in Pune?")
        self.assertTrue(any(hit.chunk_id == "synthetic.venue-options#chunk-0000" for hit in result.current_evidence_hits))

    def test_non_material_event_keeps_matching_in_flight_request(self) -> None:
        retriever = DelayedRetriever(self.base_retriever, delay_s=0.03)
        provider = RecordingMockProvider()
        result = asyncio.run(
            replay_streaming_transcript(
                [
                    _event(1, "Which venue", timestamp=0.0),
                    _event(2, "Which venue?", final=True, mode="cumulative", timestamp=0.01),
                ],
                corpus=self.index,
                retriever=retriever,
                backend="lexical",
                generation_provider=provider,
                scheduler_config=StreamingSchedulerConfig(final_wait_timeout_s=1.0),
            )
        )
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(result.decisions[-1].decision, "SKIP")
        self.assertEqual(result.decisions[-1].reason_code, "duplicate_query_suppressed")
        self.assertEqual(result.retrieval_results[0].transcript_revision, 1)
        self.assertTrue(result.retrieval_results[0].accepted)
        self.assertEqual(len(provider.supplied_chunk_ids), 1)

    def test_material_correction_rejects_old_result_and_only_final_evidence_is_generated(self) -> None:
        retriever = DelayedRetriever(
            self.base_retriever,
            delays={"Which venue": 0.04, "Actually, cancellation policy?": 0.005},
        )
        provider = RecordingMockProvider(delay_s=0.08)
        result = asyncio.run(
            replay_streaming_transcript(
                [
                    _event(1, "Which venue", timestamp=0.0),
                    _event(
                        2,
                        "Actually, cancellation policy?",
                        final=True,
                        mode="cumulative",
                        timestamp=0.01,
                    ),
                ],
                corpus=self.index,
                retriever=retriever,
                backend="lexical",
                generation_provider=provider,
                scheduler_config=StreamingSchedulerConfig(final_wait_timeout_s=1.0),
            )
        )
        self.assertEqual(result.run_status, "completed")
        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(len(result.current_evidence_hits), 1)
        self.assertIn("cancellation-policy", result.current_evidence_hits[0].chunk_id)
        old = next(item for item in result.retrieval_results if item.query == "Which venue")
        final = next(item for item in result.retrieval_results if item.query == "Actually, cancellation policy?")
        self.assertTrue(old.stale)
        self.assertFalse(old.accepted)
        self.assertTrue(old.superseded)
        self.assertTrue(old.cancellation_requested)
        self.assertFalse(old.cancellation_confirmed)
        self.assertTrue(final.accepted)
        self.assertEqual(provider.supplied_chunk_ids, [["synthetic.cancellation-policy#chunk-0000"]])
        self.assertNotIn("synthetic.venue-options#chunk-0000", provider.supplied_chunk_ids[0])
        self.assertIn("streaming_retrieval_stale_rejected", {trace.event_type for trace in result.traces})

    def test_rapid_corrections_coalesce_pending_requests(self) -> None:
        retriever = DelayedRetriever(self.base_retriever, delay_s=0.04)
        events = [
            _event(1, "Which venue", timestamp=0.0),
            _event(2, "Actually, venue in Pune", mode="cumulative", timestamp=0.01),
            _event(3, "Actually, cancellation policy", mode="cumulative", timestamp=0.02),
            _event(4, "Actually, cancellation policy for attendees", mode="cumulative", timestamp=0.03),
            _event(5, "Actually, cancellation policy?", final=True, mode="cumulative", timestamp=0.04),
        ]
        result = asyncio.run(
            replay_streaming_transcript(
                events,
                corpus=self.index,
                retriever=retriever,
                backend="lexical",
                scheduler_config=StreamingSchedulerConfig(
                    max_concurrency=1,
                    max_pending_requests=1,
                    max_total_requests=8,
                    final_wait_timeout_s=1.0,
                ),
            )
        )
        # Four revised queries are scheduled after the first, but only the
        # first and latest pending query need worker execution under the
        # one-active/one-pending bound.
        self.assertLessEqual(len(retriever.calls), 2)
        self.assertLessEqual(result.scheduled_request_count, 8)
        self.assertGreaterEqual(result.cancelled_request_count, 1)
        self.assertTrue(any(trace.event_type == "streaming_retrieval_coalesced" for trace in result.traces))

    def test_retrieval_failure_and_timeout_are_explicit(self) -> None:
        failed = asyncio.run(
            replay_streaming_transcript(
                [_event(1, "Which venue?", final=True)],
                corpus=self.index,
                retriever=DelayedRetriever(self.base_retriever, fail=True),
                backend="lexical",
                scheduler_config=StreamingSchedulerConfig(final_wait_timeout_s=0.5),
            )
        )
        self.assertEqual(failed.run_status, "failed")
        self.assertEqual(failed.retrieval_results[-1].status, "failed")
        self.assertIn("streaming_retrieval_failed", {trace.event_type for trace in failed.traces})

        timed_out = asyncio.run(
            replay_streaming_transcript(
                [_event(1, "Which venue?", final=True, session="timeout-session", utterance="timeout-utterance")],
                corpus=self.index,
                retriever=DelayedRetriever(self.base_retriever, delay_s=0.04),
                backend="lexical",
                scheduler_config=StreamingSchedulerConfig(
                    request_timeout_s=0.01,
                    final_wait_timeout_s=0.1,
                    max_total_requests=2,
                ),
            )
        )
        self.assertEqual(timed_out.run_status, "failed")
        self.assertGreaterEqual(timed_out.timed_out_request_count, 1)
        self.assertIn("streaming_retrieval_timed_out", {trace.event_type for trace in timed_out.traces})

    def test_retrieval_retries_are_bounded_and_traced(self) -> None:
        retriever = FlakyRetriever(self.base_retriever, failures=1)
        result = asyncio.run(
            replay_streaming_transcript(
                [_event(1, "Which venue?", final=True)],
                corpus=self.index,
                retriever=retriever,
                backend="lexical",
                scheduler_config=StreamingSchedulerConfig(
                    max_retries=1,
                    retry_backoff_s=0.001,
                    final_wait_timeout_s=0.5,
                ),
            )
        )
        self.assertEqual(result.run_status, "completed")
        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(result.retrieval_results[-1].attempts, 2)
        self.assertIn("streaming_retrieval_retry_scheduled", {trace.event_type for trace in result.traces})

    def test_completed_result_from_an_older_revision_is_not_reused(self) -> None:
        retriever = DelayedRetriever(self.base_retriever)
        events = [
            _event(1, "Which venue?", timestamp=0.0),
            _event(
                2,
                "Actually, cancellation policy?",
                mode="cumulative",
                timestamp=0.05,
            ),
            _event(
                3,
                "Which venue?",
                final=True,
                mode="cumulative",
                timestamp=0.1,
            ),
        ]
        result = asyncio.run(
            replay_streaming_transcript(
                events,
                corpus=self.index,
                retriever=retriever,
                backend="lexical",
                execution_mode="realtime",
                scheduler_config=StreamingSchedulerConfig(final_wait_timeout_s=0.5),
            )
        )
        self.assertEqual(result.run_status, "completed")
        self.assertEqual([query for query, _, _ in retriever.calls], [
            "Which venue?",
            "Actually, cancellation policy?",
            "Which venue?",
        ])
        self.assertEqual(result.retrieval_results[-1].retrieval_revision, 3)
        self.assertTrue(result.retrieval_results[-1].accepted)

    def test_request_limit_cannot_send_prior_query_evidence_to_generation(self) -> None:
        provider = RecordingMockProvider()
        result = asyncio.run(
            replay_streaming_transcript(
                [
                    _event(1, "Which venue", timestamp=0.0),
                    _event(
                        2,
                        "Actually, cancellation policy?",
                        final=True,
                        mode="cumulative",
                        timestamp=0.01,
                    ),
                ],
                corpus=self.index,
                retriever=DelayedRetriever(self.base_retriever),
                backend="lexical",
                generation_provider=provider,
                scheduler_config=StreamingSchedulerConfig(
                    max_total_requests=1,
                    final_wait_timeout_s=0.2,
                ),
            )
        )
        self.assertEqual(result.run_status, "failed")
        self.assertEqual(result.generation_status, "skipped")
        self.assertEqual(provider.supplied_chunk_ids, [])
        self.assertEqual(result.current_evidence_hits, [])

    def test_session_close_prevents_late_publication(self) -> None:
        retriever = DelayedRetriever(self.base_retriever, delay_s=0.05)
        traces = __import__("flowcontext.trace", fromlist=["TraceCollector"]).TraceCollector(
            "close-run", "close-session", "scheduler-test"
        )
        controller = StreamingDecisionController(
            session_id="close-session",
            utterance_id="close-utterance",
            known_chunk_locations={chunk.chunk_id: chunk.source_location for chunk in self.index.chunks},
        )
        decision = controller.process_event(
            _event(
                1,
                "Which venue?",
                final=True,
                session="close-session",
                utterance="close-utterance",
            )
        )

        async def run() -> None:
            scheduler = AsyncRetrievalScheduler(
                controller=controller,
                retriever=retriever,
                traces=traces,
                retrieval_model_identity="test-retriever",
                config=StreamingSchedulerConfig(final_wait_timeout_s=0.5),
            )
            scheduler.schedule(decision)
            await asyncio.sleep(0.005)
            await scheduler.close()
            await asyncio.sleep(0.07)
            self.assertTrue(scheduler.closed)
            self.assertEqual(controller.current_evidence_hits, [])
            self.assertFalse(any(item.accepted for item in scheduler.results))

        asyncio.run(run())

    def test_identical_text_in_concurrent_sessions_stays_isolated(self) -> None:
        async def run_pair():
            return await asyncio.gather(
                replay_streaming_transcript(
                    [_event(1, "Which venue?", final=True, session="same-a", utterance="u-a")],
                    corpus=self.index,
                    retriever=DelayedRetriever(self.base_retriever, delay_s=0.01),
                    backend="lexical",
                ),
                replay_streaming_transcript(
                    [_event(1, "Which venue?", final=True, session="same-b", utterance="u-b")],
                    corpus=self.index,
                    retriever=DelayedRetriever(self.base_retriever, delay_s=0.01),
                    backend="lexical",
                ),
            )

        first, second = asyncio.run(run_pair())
        self.assertEqual(first.final_query, second.final_query)
        self.assertEqual(first.session_id, "same-a")
        self.assertEqual(second.session_id, "same-b")
        self.assertNotEqual(first.retrieval_results[0].request_id, second.retrieval_results[0].request_id)
        self.assertTrue(all(trace.session_id == "same-a" for trace in first.traces))
        self.assertTrue(all(trace.session_id == "same-b" for trace in second.traces))

    def test_scheduler_requests_have_session_revision_and_lifecycle_traces(self) -> None:
        retriever = DelayedRetriever(self.base_retriever, delay_s=0.01)
        result = asyncio.run(
            replay_streaming_transcript(
                [
                    _event(1, "Which venue", timestamp=0.0),
                    _event(2, " in Pune?", final=True, timestamp=0.01),
                ],
                corpus=self.index,
                retriever=retriever,
                backend="lexical",
            )
        )
        request_revisions = [item.retrieval_revision for item in result.retrieval_results]
        self.assertEqual(request_revisions, sorted(set(request_revisions)))
        self.assertTrue(all(item.request_id for item in result.retrieval_results))
        trace_types = {trace.event_type for trace in result.traces}
        self.assertTrue(
            {
                "streaming_retrieval_scheduled",
                "streaming_retrieval_started",
                "streaming_retrieval_completed",
                "streaming_generation_completed",
                "streaming_answer_completed",
            }.issubset(trace_types)
        )
        self.assertFalse(any(trace.event_type == "streaming_answer_first_token" for trace in result.traces))
        answer_completed = next(trace for trace in result.traces if trace.event_type == "streaming_answer_completed")
        self.assertEqual(answer_completed.attributes["latency_metric"], "complete_answer_latency_ms")
