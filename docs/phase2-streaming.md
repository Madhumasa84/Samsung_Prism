# Phase 2 streaming controller

This document records the Phase 2 boundary and measurements. The optional
Phase 3 multi-intent path built on these interfaces is documented separately
in [`phase3.md`](phase3.md); the Phase 2 behavior described below remains the
baseline for comparison.

Phase 2 adds an explainable decision controller and bounded asynchronous
retrieval scheduler around the existing transcript normalizer and retriever.
It is an engineering implementation on the local synthetic fixture; the
supplied Theme 4 guide does not include an organiser event schema, corpus, or
benchmark. No competition or quality result is claimed.

`replay --mode baseline` is the complete-utterance comparison: retrieval is
started after final-event delivery. `replay --mode streaming` can start
retrieval from a meaningful partial. Both modes use the same explicit index,
retrieval backend, `--top-k`, and generation configuration. Results, manifests,
and CLI summaries label both `mode` and `execution_mode` (`realtime` or
`accelerated`).

## Boundary

The controller consumes the existing provisional TranscriptEvent contract.
text_mode="incremental" appends a fragment, while
text_mode="cumulative" replaces the assembled utterance. The shared
normalize_transcript function validates one session/utterance, ordering,
conflicting event IDs, and repeated finals before controller execution.

For every event, the controller emits one StreamingDecision:

* WAIT means the text is incomplete, lacks corpus-bearing topic content, is
  unstable, or is inside the configured source-time coalescing window.
* RETRIEVE means a question/request has meaningful topic content, a new topic
  or constraint is visible, or an explicit correction changed the request.
* SKIP means no new corpus evidence is needed. Greeting turns are classified
  as `greeting_no_retrieval`. Formatting-only text gets SKIP only with a
  non-empty PreviousAnswerContext from the same session; otherwise the result
  is `needs_context`, not a fabricated answer.

The policy uses question/request signals, content terms, punctuation,
correction markers, and constraint signatures. A minimum query length is only
one guard; word count alone does not trigger retrieval. The vocabulary is
generic policy vocabulary and does not contain guide example answers.

## Coalescing, scheduling, and stale work

debounce_source_s is a source-timestamp gap. A revised partial with new topic
terms waits when it arrives inside that window; its event ID is included in
the next retrieval decision's coalesced_event_ids. Corrections and constraint
changes bypass this coalescing guard. Final events always receive a decision
immediately, so a required final retrieval cannot remain pending.

The controller does not start a wall-clock debounce timer: source-time
coalescing is decided when an event arrives. Retrieval itself is scheduled
asynchronously. The existing synchronous `Retriever.search` method runs in a
per-replay bounded `ThreadPoolExecutor`, with an asyncio semaphore, pending
request bound, per-request timeout, bounded exception retries, and a total
request limit. This keeps transcript event handling responsive without
requiring a new retriever API.

When capacity is full, only the latest revised query is retained as a pending
candidate. A material query change advances the session-scoped retrieval
revision and marks older requests superseded immediately. The scheduler calls
`Future.cancel()` where possible, but reports cancellation as confirmed only
when the underlying executor future confirms it. A running/cancellation-
resistant thread may finish later; its result is checked against retrieval
revision, exact canonical query, and session closure, then traced as stale and
never becomes current evidence. Session closure cancels pending work and
prevents any late evidence publication without waiting indefinitely on worker
threads.

Each RETRIEVE decision becomes a revision-bound
`StreamingRetrievalRequest` with a unique request ID, the transcript revision
that caused it, and a session-scoped retrieval revision. A later transcript
event does not invalidate an in-flight request merely because its raw revision
changed; exact canonical-query equality permits reuse for punctuation/spacing
or otherwise non-material updates. Duplicate canonical queries are suppressed
before a new request. “Unnecessary retrieval” in the replay result is only the
operational count of superseded work; semantic answer usefulness cannot be
inferred without a judged answer evaluation.

At finalisation, the scheduler first reuses accepted evidence only when the
final canonical query exactly matches the completed request and that result
belongs to the controller's latest retrieval revision. This prevents a query
from being reused after the conversation changed to another query and then
returned to the original wording. It then awaits a matching in-flight/pending
request when possible. Otherwise it schedules the complete final query,
subject to the same resource limits. Answer generation starts only after this
selection. Current accepted hits, never stale hits, are passed to the existing
structured generation pipeline. With same-session formatting context, the
prior answer is returned without a corpus search or selective rewrite. Missing
context returns an explicit clarification. Phase 2 does not perform selective
answer refinement.

## CLI

Build a local fixture index and run the Phase 2 controller:

    uv run flowcontext build-index \
      --input data/synthetic/documents.jsonl \
      --output artifacts/fixture-lexical-index.json \
      --backend lexical \
      --source-kind synthetic_fixture

    uv run flowcontext replay \
      --mode streaming \
      --transcript examples/streaming/early-retrieval.jsonl \
      --index artifacts/fixture-lexical-index.json \
      --backend lexical \
      --top-k 5 \
      --execution-mode realtime \
      --output artifacts/streaming-replay.json

The default flowcontext replay mode remains the Phase 1
wait_for_complete_utterance baseline. The streaming command writes a
Phase 2 result, JSONL trace, and secret-free streaming manifest beside the
requested output. accelerated mode consumes events without waiting for source
gaps; realtime mode waits for those gaps. Do not compare accelerated and
realtime wall-clock latency: accelerated delays validate scheduling behavior,
not measured model performance.

Controller settings can be supplied in .env or with the replay options:

    FLOWCONTEXT_STREAMING_DEBOUNCE_SOURCE_S=0.35
    FLOWCONTEXT_STREAMING_MIN_QUERY_CHARS=8
    FLOWCONTEXT_STREAMING_MIN_TOPIC_TERMS=1
    FLOWCONTEXT_STREAMING_MIN_NEW_CONTENT_TERMS=1
    FLOWCONTEXT_STREAMING_RETRIEVE_ON_CORRECTION=true
    FLOWCONTEXT_STREAMING_RETRIEVE_ON_CONSTRAINT_CHANGE=true
    FLOWCONTEXT_STREAMING_DUPLICATE_QUERY_SUPPRESSION=true
    FLOWCONTEXT_STREAMING_FINAL_BYPASSES_DEBOUNCE=true
    FLOWCONTEXT_STREAMING_MAX_CONCURRENCY=2
    FLOWCONTEXT_STREAMING_MAX_PENDING_REQUESTS=4
    FLOWCONTEXT_STREAMING_REQUEST_TIMEOUT_S=5
    FLOWCONTEXT_STREAMING_MAX_RETRIES=0
    FLOWCONTEXT_STREAMING_RETRY_BACKOFF_S=0.05
    FLOWCONTEXT_STREAMING_MAX_TOTAL_REQUESTS=16
    FLOWCONTEXT_STREAMING_FINAL_WAIT_TIMEOUT_S=5
    FLOWCONTEXT_STREAMING_CANCEL_ON_SUPERSESSION=true

The optional --previous-answer argument accepts a completed successful Phase 1
replay result only when its session_id matches the streaming transcript. It
passes only the answer text/version to the formatting classifier; it is not a
selective refinement or answer-update mechanism.

## Timing, trace, and reporting

Streaming traces include replay start, every event's source timestamp and
observed delivery time, controller decision time/overhead, final-event
delivery, retrieval scheduling, actual retrieval start/end, evidence-ready,
cancellation request/confirmation, supersession, timeout, stale-result
rejection, final-only generation, answer completion/failure, a timing summary,
and replay completion. Retrieval result records also carry relative scheduled,
actual-start, actual-end, and evidence-ready times. Generation first-content is
omitted unless a provider actually exposes it; the current provider is
non-streaming.

The comparison fields are separate:

* `retrieval_started_early` is true only when an actual retrieval-start event
  precedes final-event delivery. A decision or queued request does not count.
* `valid_evidence_ready_before_finalization` requires accepted, non-stale,
  non-empty evidence for the final canonical query before final delivery.
* `early_evidence_reused` records exact-query reuse at finalisation.
* `answer_latency_from_final_event_delivery_ms` starts at observed final-event
  delivery; `full_interaction_duration_ms` covers first delivery through the
  terminal answer/replay event.
* `retrieval_call_count` counts actual backend calls, including retries;
  scheduled requests, superseded requests, stale-result discards, errors,
  controller overhead, and estimated/real usage remain separate.

Trace attributes contain request/query identifiers and corpus chunk IDs but
never API keys or secret values. Real dense/provider unavailability is an
explicit `BLOCKED` CLI result or an explicit failed trace; no mock fallback is
selected silently.

Representative useful-retrieval timeline (`realtime`, synthetic fixture):

```text
transcript_event_received(source=0.0, delivery=t0)
streaming_decision(RETRIEVE, final=false)
streaming_retrieval_scheduled -> streaming_retrieval_started -> streaming_retrieval_completed
streaming_evidence_ready(time < final delivery)
transcript_event_received(source=0.1, delivery=t1)
final_event_delivered(t1)
streaming_decision(SKIP, duplicate_query_suppressed)
streaming_evidence_reused(early=true)
streaming_generation_started -> streaming_generation_completed -> streaming_answer_completed
```

This phase does not generate early answers, decompose multiple intents,
fuse/rerank results, or selectively update claims. The scheduler correctness
tests use controllable simulated delays and are not real-backend performance
measurements. Dense retrieval and real provider execution remain unverified in
this workspace.

## Dedicated evaluation and known limitations

The checked-in Phase 2 suite is in
[`data/evaluation/streaming_development.jsonl`](../data/evaluation/streaming_development.jsonl)
and [`data/evaluation/streaming_held_out.jsonl`](../data/evaluation/streaming_held_out.jsonl).
It covers incomplete/final-only speech, stable partials, incremental and
cumulative input, repeated fragments, corrections to entities/quantities/
locations/dates/negation, rapid late results, greetings, formatting context,
failure/timeout/closure, and concurrent sessions. Eligibility and the earliest
reasonable retrieval event are labelled before execution. Labels are
`provisional_generated`; no human review is claimed. Variants remain in their
own split and the held-out file is frozen for measurement.

Use `flowcontext evaluate-streaming` to run matched baseline and streaming
cases. The JSON report keeps fixture, simulated-delay, real-backend, and
official-asset sections separate and records actual early-start numerator /
eligible denominator, valid pre-final evidence, exact reuse, no-retrieval false
triggers, premature triggers, calls per utterance, stale acceptance (an
invariant of zero), final evidence Recall@k/MRR, final-event-to-answer timing,
full duration, controller overhead, trace coverage, usage, and cost
availability. Each timing summary carries its sample count and replay mode.
The local report says whether the guide's 80% early-retrieval target is met;
it is not an official benchmark result. No numerical false-trigger threshold is
invented because the guide does not supply one.

The focused `--compare-disabled-scheduling` run disables duplicate suppression
and source-time coalescing for the marked repeated/coalescing case. It keeps
`max_total_requests`, concurrency, pending-request, timeout, and final-wait
bounds in force, and reports extra calls, superseded/stale work, and whether
the final evidence IDs changed. This exposes policy waste without treating
semantic usefulness as proved.

Known limitations exposed by the suite include final-only generation, no
first-content observation for the non-streaming provider, exact canonical-query
reuse only, provisional chunk-ID labels rather than semantic support review,
synthetic lexical retrieval, and failure/timeout paths that can have no final
evidence. Rapid corrections intentionally show that early evidence can be
useful for scheduling while still being discarded when the final query changes.
Formatting with context reuses the prior answer but does not selectively
rewrite it; without context it returns a clarification. Multi-intent
decomposition and evidence merging are explicitly deferred to the Phase 3
boundary and are not started here.
