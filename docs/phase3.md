# Phase 3: multi-intent retrieval and grounded answers

Phase 3 is an opt-in extension of the Phase 2 replay path. The default
complete-utterance baseline and the default streaming controller remain
available unchanged; callers select the new path with `--multi-intent` (or
`multi_intent=True` in the replay APIs).

The local implementation is engineering evidence on the synthetic corpus. The
Theme 4 guide, organiser corpus, replay schema, labels, thresholds, and scoring
harness are not supplied, so no official or competition result is claimed.

## Boundary and data flow

1. `StructuredMultiIntentDecomposer` is the application component. It sends a
   stable controller revision to the configured structured-output provider,
   validates the response against `DecompositionResult`, and records provider
   attempts, bounded repairs, token usage, latency, source spans, constraints,
   ambiguities, and dependencies. The CLI's default provider is explicitly a
   deterministic mock model. `decompose_query` is an explicitly labelled
   offline rule-based compatibility path; it is never substituted after a real
   provider failure.
2. A decomposition preserves the exact `original_transcript` and
   `transcript_revision`. Each `DecompositionIntent` contains a search-ready
   sub-question, exact `source_span`, and intent-specific constraints. The
   result also carries `shared_constraints`, `DecompositionDependency` edges
   (`depends_on`, `compares_with`, or `refines`), and unresolved
   `DecompositionAmbiguity` records. Numeric/date values in provider output
   must be present in the transcript, and every span must point back to the
   exact source text. The contract bounds the result to eight intents; the
   configured decomposer defaults to six.
3. The controller invokes decomposition only for a meaningful stable
   retrieval decision and at finalisation when final evidence validation needs
   the complete request. It is not called from every transcript-token event.
   Unchanged intent IDs/fingerprints may reuse their isolated retrieval work;
   changed constraints and affected dependents are superseded. Results whose
   transcript or retrieval revision is obsolete are rejected by the existing
   controller guard and are traced as obsolete.
4. On provider failure, the caller may request an original-query plan with
   `status="fallback"`, `decomposition_method="original_query_fallback_after_provider_failure"`,
   and `failure_reason`. This is a retrieval-preserving safety path, not a
   successful decomposition and not a mock substitution.
5. `retrieve_multi_intent` supports explicit `dense`, `lexical`, `mock`, and
   `hybrid` modes. Independent intents run in a bounded executor. `depends_on`
   and `refines` edges form a finite prerequisite graph; dependent work waits
   for prerequisite evidence and receives at most two prerequisite chunk IDs
   and 32 context tokens. One intent/backend failure is retained as structured
   state; an all-failed request raises without fabricating evidence.
6. Backend lists are fused within each intent first with reciprocal-rank fusion
   (`1 / (rrf_k + rank)`), never by adding incomparable raw scores. The intent
   lists are then fused across the evidence set. Fused hits preserve exact
   chunk IDs, source locations/text, per-intent ranks/score provenance, and the
   RRF score. Lexical and dense lists remain available in the audit record.
7. Evidence assembly deduplicates only after provenance agreement, keeps every
   supporting intent relationship, and performs a mandatory coverage pass before
   global-rank filling. A total token budget and `top_k` bound apply to the
   final set; `missing_intent_ids` records intents not represented after those
   bounds. Retrieval score is not treated as factual support.
8. Grounded synthesis receives the parent query, ordered intent queries, and
   only corpus-resolved passages. Provider citations are still validated
   against the supplied chunk IDs. Missing intent evidence is surfaced as
   uncertainty; a completed search with no hits is an abstention, not a
   transport failure.
9. Streaming uses the same revision/session guards as Phase 2. Final evidence
   must be current and pass the final-intent coverage check before it is sent
   to generation. An early result with valid IDs but insufficient final-query
   support is not reusable. Citation-ID validity is traceability, not semantic
   entailment.

Selective claim updates across later follow-up turns, early answer generation,
and provider token streaming are outside this phase. They belong to a later
answer lifecycle and are deliberately not added to the scheduler.

The CLI selects the retrieval mode with `--retrieval-mode dense|lexical|hybrid|mock`.
`--backend` still selects the persisted Phase 1/2
index backend; an explicit hybrid request constructs both lexical and dense
backends and fails clearly if the dense index, dependency, or model is
unavailable. Reranking is recorded in the configuration but disabled by default
until a measured latency/quality benefit justifies enabling it.

## Evaluation

The comparison command preserves the checked-in Phase 2 reports and writes a
new JSON/Markdown pair:

```bash
UV_CACHE_DIR=/tmp/flowcontext-uv-cache uv run --locked flowcontext evaluate-phase3 \
  --corpus artifacts/fixture-lexical-index.json \
  --backend lexical --top-k 5 --split all \
  --execution-mode realtime \
  --output reports/phase3_retrieval_comparison_realtime.json
```

An end-to-end decomposition payload with independent and dependent questions,
typed constraints, spans, ambiguity records, and a dependency edge is shown in
[`examples/multi_intent/README.md`](../examples/multi_intent/README.md). The
default CLI settings use the explicitly labelled mock structured provider for
offline contract checks. Set `FLOWCONTEXT_GENERATION_BACKEND=openai_compatible`
and the existing generation endpoint/model/key settings to exercise the real
provider path; credentials are never written to output.

The report contains:

- a per-case baseline/streaming table with final queries, retrieved IDs,
  relevant IDs, request statuses and queries, reuse decisions, validation
  details, errors, and scoring denominators;
- separate development and held-out Phase 2 comparisons;
- successful comparable retrieval quality with explicit case and relevant-ID
  denominators;
- end-to-end counts that retain failed, timed-out, closed, and abstained cases;
- partial or missing-intent evidence is visible and excluded from successful
  full-request retrieval quality;
- diagnoses for query assembly, dropped/missing final constraints, stale
  rejection, ranking/set differences, fault injection, and denominator
  differences;
- per-intent IDs and fused evidence for the Phase 3 development/held-out
  cases, including backend rankings, fusion decisions, assembly decisions,
  context usage, and missing-intent markers; and
- explicit `not_verified` sections for real backends and official assets.

The held-out files were inspected only after implementation decisions were
frozen and were not used to tune decomposition, fusion, or streaming policy.
New untouched cases should be reserved for the next evaluation. Local labels
remain provisional and semantic claim support remains `not_evaluated`.

## Current local findings

The fresh matched Phase 2 rerun found:

- 23/23 eligible cases started retrieval early;
- 10/23 eligible cases achieved useful early reuse;
- the historical aggregate remains 96.0% baseline versus 91.7% streaming
  Recall@5 in `reports/phase2_streaming_evaluation.json`;
- after restricting retrieval quality to 22 successful comparable cases with
  explicit relevant-ID denominators, both modes measured 100% Recall@5; and
- end-to-end outcomes remain separate and visible: development includes the
  baseline no-context abstention and injected retrieval failure, while
  streaming additionally exposes the injected timeout and session closure;
  held-out has no operational fault cases.

The 13 eligible cases without useful early reuse classify as 9 expected
correction/final-constraint changes, 1 delayed rapid-correction retrieval, and
3 declared failure/timeout/session-close probes. No avoidable non-reuse failure
was observed. The rapid case is not counted as a useful reuse because its
current result was not ready before final-event delivery and late work was
correctly rejected.

On the small Phase 3 set, the configured explicitly labelled mock structured
decomposer identified the one compound case in each split (1/1 in development
and 1/1 in held-out). Successful answerable retrieval quality was 100%
Recall@5 in both modes over four cases per split; unsupported cases abstained
with no factual claims. These are small fixture measurements, not evidence
that the real dense/generation gap is resolved.

## Blockers

The attempted real dense smoke command was blocked because the optional
`sentence-transformers` dependency is not installed; install the dense extra
and download the pinned model before building a dense index. Provider-backed
generation was not run. The official
corpus, replay format, labels, thresholds, semantic grounding evaluator, and
organiser API remain unavailable. The lexical fixture, mock generator, and
provisional labels therefore cannot support a competition-performance claim.
