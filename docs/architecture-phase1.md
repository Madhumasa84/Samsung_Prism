# FlowContext Phase 1 architecture

Phase 1 is a small offline-first Python 3.11 application. It has one
provenance boundary around the configured corpus and one complete-utterance
baseline path:

```text
source JSONL
    -> validated ingestion + hashes + paragraph/size chunks
    -> deterministic index + manifest
    -> selected retriever (dense | explicit lexical | local mock)
    -> retrieved corpus passages with real chunk IDs
    -> generation provider (mock | OpenAI-compatible)
    -> validated Answer + citations/uncertainty
    -> timestamped replay trace + run manifest + evaluation report
```

## Interfaces

* `DocumentInput`, `Document`, `Chunk`, and `IndexManifest` define ingestion,
  provenance, deterministic IDs, chunking settings, source versions, and
  content hashes.
* `EmbeddingProvider` is the replaceable dense embedding boundary. The CPU
  Sentence Transformers provider is pinned in configuration; lexical retrieval
  is a separately selected diagnostic backend and is never an implicit dense
  fallback.
* `Retriever.search(query)` returns ranked `RetrievalHit` objects. The caller
  resolves every hit against the loaded `CorpusIndex` before generation.
* `GenerationProvider.generate(request)` receives only quoted retrieved
  passages. `generate_grounded_answer` validates the structured `Answer`,
  rejects unknown citations, bounds retries/repair, and abstains on malformed
  or insufficient evidence.
* `TranscriptEvent` and `replay_transcript` accept the provisional event
  adapter. Incremental text appends; cumulative text replaces. Retrieval and
  generation are invoked once, after the final event, and duplicate finals are
  idempotent.
* `ExecutionTrace` and `ReplayRunManifest` separate source timestamps from
  monotonic execution measurements and carry usage, cost availability, model
  identity, errors, configuration, index identity, and environment metadata.
* `evaluate_suite` runs a fixed split through the same replay baseline and
  emits separate mock, fixture, and real-model report sections.

The only currently supported corpus loader is strict document JSONL because no
official corpus format was supplied. The competition guide PDF is an asset
inventory input, not a knowledge document. No web search, external evidence,
query decomposition, early retrieval, hybrid/reranking, or selective answer
update path exists in this phase.

## Runtime boundary

The CLI is the application boundary: `inspect-corpus`, `build-index`/`ingest`,
`retrieve`, `replay`, `answer`, `evaluate`, and `evaluate-suite`. The default
configuration selects dense retrieval, but the offline fixture commands select
lexical explicitly and the mock provider explicitly. A dense dependency/model
failure is surfaced rather than converted to lexical retrieval.

The container image intentionally copies only application code, documentation,
examples, and the synthetic fixture. Restricted corpus files, model weights,
credentials, and generated artifacts are mounted or configured separately.
