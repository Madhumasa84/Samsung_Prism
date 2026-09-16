# Schema notes

The only supplied specification was `Theme 4 Guide_RAG.pdf`. It contains a
descriptive architecture, event examples, and one illustrative JSON record. It
does not publish a machine-readable organiser schema or an API contract. The
repository therefore defines the explicitly versioned local contracts in
`flowcontext.phase1.v1` and does not claim organiser compatibility.

The local contracts cover transcript events, documents, chunks, retrieval hits,
quoted generation passages, structured answers, provider usage, errors, and
execution traces, and replay run manifests. If the organiser later supplies a schema, an adapter should
be added at the CLI/input boundary rather than changing the internal provenance
and citation-ID validation contracts in place. No adapter is currently required
because no external schema is available.

The guide's example fields such as `retrieval_events`, `sub_queries`, and
`citations` describe future/illustrative behavior. Phase 1 intentionally does
not implement early retrieval, multi-intent decomposition, or selective answer
refinement. Phase 2 adds local flowcontext.phase2.v1 controller/retrieval
request/result contracts, but this is still not an organiser schema.

## Provisional transcript replay format

No official organiser event schema was supplied. The application therefore
accepts a provisional JSONL format, one `flowcontext.phase1.v1` object per line:

```json
{
  "session_id": "session-1",
  "utterance_id": "utterance-1",
  "event_id": "event-1",
  "sequence_number": 1,
  "source_timestamp_s": 0.4,
  "text": "partial text",
  "is_final": false,
  "text_mode": "incremental"
}
```

`incremental` text is appended to the current utterance. `cumulative` text
replaces it, so cumulative snapshots are not concatenated. Phase 1 accepts one
session and one utterance per replay invocation; separate invocations may run
concurrently without sharing state.

Sequence numbers must increase for distinct events and source timestamps must
not decrease. An identical repeated event ID is accepted and ignored. A final
event may be repeated with a new event ID only when its text and mode exactly
match the first final and its sequence number is later; the repeat is traced
but cannot trigger another retrieval/generation execution. Conflicting IDs,
non-final events after final, and divergent repeated finals fail validation.

`--execution-mode realtime` waits between source timestamps. The default
`accelerated` mode does not wait. Trace `source_timestamp_s` is source data;
`monotonic_execution_time_s` and duration fields are measured locally. The
non-streaming Phase 1 provider reports complete-answer latency, never
time-to-first-token.

## Phase 2 decision records

The streaming mode emits one controller decision per received event. Each
decision carries session_id, utterance_id, transcript_revision, decision,
reason_code, optional proposed_query, source_event_id,
source_timestamp_s, monotonic_decision_time_s, and the event IDs coalesced
into that decision. RETRIEVE decisions produce a revision-bound
StreamingRetrievalRequest. Its corresponding StreamingRetrievalResult is
accepted only when the assembled query still matches that revision; otherwise
it is stale and cannot replace current evidence.

The replay --mode streaming command uses these local records and runs the
existing synchronous retriever through the bounded
`AsyncRetrievalScheduler`. `StreamingRetrievalRequest.retrieval_revision` is
session/utterance scoped and advances when a material query is scheduled;
`transcript_revision` records the event that caused that query. Exact
canonical-query equality permits an in-flight result to survive a
non-material transcript update. Superseded, timed-out, cancelled, and late
results are explicit lifecycle records and cannot become current evidence.

Scheduler settings are local implementation configuration, not an organiser
API: concurrency, pending/coalescing, timeout, retry, final-wait, cancellation,
and total-request bounds are environment-backed through `Settings`. The
existing generation provider is invoked once, only after finalisation and
current evidence selection. It is non-streaming in this phase, so no first
token event is claimed. The implementation does not claim a streaming
provider protocol, organiser compatibility, semantic grounding, or answer
updates.

## Phase 3 multi-intent records

Phase 3 keeps the Phase 1 `RetrievalHit`, `EvidencePassage`, and `Answer`
contracts backward-compatible by adding optional intent provenance. The local
`flowcontext.phase3.v1` records are:

* `TextSpan`, `DecompositionConstraint`, `DecompositionIntent`,
  `DecompositionDependency`, `DecompositionAmbiguity`, and
  `DecompositionResult` for a revision-bound, source-traceable parent-query
  decomposition;
* `IntentRetrievalResult` and `MultiIntentRetrievalResult` for isolated
  per-intent status, backend rankings, dependency context, timings, errors,
  fusion/assembly decisions, context usage, missing intents, and fused
  evidence; and
* optional `intent_id`, `intent_ids`, per-intent ranks/scores, and `rrf_score`
  fields on retrieval hits/passages, plus ordered intent queries on generation
  requests.

`DecompositionResult` preserves the exact transcript and revision, bounds the
result to eight intents (the configured decomposer defaults to six), and
requires exact character spans for every intent and constraint. Constraints
carry typed values for entities, quantities, dates, locations, negation,
comparisons, and relationships. Dependencies distinguish independent,
dependent, and comparison work; unresolved references remain explicit
ambiguities rather than being resolved by invention. A provider result must be
`status="success"`. A failed configured provider can only produce an explicit
`status="fallback"` original-query result with a failure reason; it is never
silently replaced by the offline rule parser.

`StructuredMultiIntentDecomposer` uses the same validated `GenerationConfig`
provider boundary as answer generation. The OpenAI-compatible implementation
is a real JSON provider with bounded timeout/repair handling. The deterministic
`MockDecompositionProvider` and `decompose_query` helper are labelled offline
execution modes. Decomposition is invoked at stable controller retrieval
revisions and at finalisation when needed, not for every token; provider usage,
latency, attempts, repairs, and obsolete revisions are traceable.

Fused evidence is deduplicated by chunk ID only after source location and text
agree. The original provenance remains attached to every fused hit. A final
query is allowed into generation only when it has current revision-bound
evidence and the streaming multi-intent path has coverage for each identified
intent. Partial retrieval is represented as explicit uncertainty or
abstention; it is not converted into an unsupported complete answer.

The Phase 3 retrieval mode is recorded as `dense`, `lexical`, `mock`, or
`hybrid`. Hybrid mode retains both input rankings and applies reciprocal-rank
fusion (`1 / (k + rank)`) within each intent before a second intent-level
fusion. It never adds raw lexical and dense scores. Evidence assembly performs
a coverage pass for every intent before global-rank filling and enforces one
total context budget; `missing_intent_ids` is not a factual-support judgment.

These are internal contracts, not an organiser schema. The guide's illustrative
`sub_queries`, `retrieval_events`, and `citations` fields are not treated as a
compatibility guarantee. Selective claim updates across follow-up turns are
outside Phase 3.
