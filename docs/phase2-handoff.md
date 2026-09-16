# Phase 2 handoff (historical boundary)

This handoff records what Phase 2 delivered before Phase 3 began. The current
Phase 3 implementation and its measured comparison are in
[`phase3.md`](phase3.md).

Phase 2 implements an explainable streaming decision controller and bounded
asynchronous early retrieval on top of the Phase 1 contracts. The controller
consumes partial events, decides WAIT/RETRIEVE/SKIP, and the scheduler binds
each retrieval request to both the transcript revision that produced it and a
session-scoped retrieval revision. Retrieval results for superseded revisions
are marked stale and cannot replace current evidence. Answer generation remains
final-only.

## Implemented interfaces

1. TranscriptEvent and normalize_transcript remain the input boundary.
   Incremental fragments append and cumulative transcripts replace the current
   text. Ordering, duplicate, repeated-final, and session/utterance checks are
   shared with the Phase 1 baseline.
2. StreamingDecisionController consumes normalized events and emits
   StreamingDecision records with revision, reason code, optional query,
   source-event timestamp, and monotonic decision time. Its policy uses
   explainable topic, request/question, incomplete-phrase, correction, and
   constraint signals.
3. StreamingRetrievalRequest and StreamingRetrievalResult form the
   revision-bound retrieval boundary. AsyncRetrievalScheduler runs the existing
   synchronous `Retriever.search` in a bounded executor, coalesces one latest
   pending query, and applies timeout/retry/request limits. Results include
   exact existing RetrievalHit IDs and source locations.
4. TraceCollector records source time separately from actual event delivery,
   controller decision time/overhead, request scheduling/actual start/end,
   final-event delivery, evidence-ready time, generation start/completion and
   observed first content, supersession, cancellation requests and
   confirmations, timeouts, stale-result rejection, selected backend,
   accepted/stale state, hit IDs, usage, and monotonic duration.
   StreamingRunManifest records controller, scheduler, generation,
   corpus/index, model, execution mode, and environment metadata without
   secrets.
5. `replay --mode baseline` retrieves after final-event delivery;
   `replay --mode streaming` is the Phase 2 controller path. Both use the
   same explicit corpus/index, backend, top-k, and generation configuration.
   Every result labels `mode` and `execution_mode`; accelerated timing is
   scheduling evidence only and is not compared with realtime latency.

## Policy record

The default source-time debounce window is 0.35 seconds. It is configurable
through Settings, .env, or streaming replay flags. A meaningful first partial
can retrieve before finalisation. Later new topic terms are coalesced inside
the source-time window; explicit corrections and changed constraints trigger
fresh retrieval. Canonical duplicate queries are suppressed. The final event
always receives an immediate decision.

Formatting-only requests use SKIP only with a non-empty
PreviousAnswerContext from the current session. Without it, the result is
needs_context. Context is not shared across controllers and is not a Phase 4
selective refinement mechanism. Greeting turns are handled as explicit
no-retrieval policy turns.

The source-time debounce remains event-driven rather than a wall-clock timer.
The retrieval scheduler does own asyncio tasks and bounded executor work; it
does not expose early answers. After finalisation it conservatively reuses or
awaits exact-query evidence only from the latest retrieval revision, then
invokes the existing structured generation provider once. An older completed
result is not reused merely because a later correction returns to its wording.
No multi-intent decomposition, hybrid fusion, reranking, or selective answer
update is included. The operational
unnecessary_retrieval_count is only revision-superseded work; semantic
usefulness needs separate answer-support review.

## Phase 3 boundary at handoff (historical)

Phase 2 hands future multi-intent work a deliberately narrow interface:

* `StreamingDecisionController` emits one revision-bound decision and
  `StreamingRetrievalRequest` per selected query. A future decomposer may emit
  a parent request plus stable subrequest IDs without changing transcript
  delivery or session identity.
* `AsyncRetrievalScheduler` owns bounded request execution and stale-result
  rejection. A future multi-intent scheduler should accept a parent revision,
  sub-intent ID, and cancellation/supersession token, then publish isolated
  `StreamingRetrievalResult` items that can be merged only for the current
  parent revision.
* `RetrievalHit` IDs and source locations remain the evidence merge boundary.
  Future fusion/reranking should preserve provenance, rank, query/sub-intent
  identity, and a deterministic final evidence set rather than passing raw
  provider text between branches.
* Final-only generation currently receives one final query and one validated
  evidence set. A future answer lifecycle may receive an ordered evidence
  merge, partial intent completion state, and an explicit answer update
  version; it must retain the existing citation validator and never accept a
  stale sub-intent result.
* Trace and report schemas should add parent/sub-intent IDs, merge decisions,
  per-intent readiness, and answer-update timing while preserving the current
  source-time/actual-time fields and `mode`/`execution_mode` labels.

These are interface notes only. Query decomposition, parallel multi-intent
retrieval, evidence merging, selective answer updates, and provider streaming
were not implemented or evaluated in Phase 2.

## Verification and limitations

The Phase 2 unit tests cover incomplete utterances, meaningful partials,
incremental/cumulative equivalence, repeated fragments, duplicate finals,
corrections, constraint/debounce behavior, formatting context, stale results,
concurrent sessions, invalid ordering, and trace completeness. Separate
simulated-delay scheduler tests cover event-loop responsiveness, same-query
reuse, late cancellation-resistant results, rapid correction coalescing,
finalisation races, failure/timeout, session closure, and final-only citation
inputs.

The supplied workspace still contains no official corpus, event schema,
evaluation benchmark, or organiser API. The local corpus and retrieval backend
used for engineering checks are synthetic/lexical fixtures. Real dense
retrieval and real-provider generation remain unverified. The next boundary is
multi-intent decomposition and later selective answer refinement; do not add
those to this scheduler.
