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
8. Grounded synthesis receives the parent query, ordered intent queries,
   decomposed intents, shared constraints, and evidence passages grouped by intent.
   Provider citations are deterministically validated against supplied chunk IDs and
   source text excerpts. Missing intent evidence is surfaced as uncertainty; a
   completed search with no hits is an abstention, not a transport failure.
9. Streaming uses the same revision/session guards as Phase 2. Final evidence
   must be current and pass the final-intent coverage check before it is sent
   to generation. An early result with valid IDs but insufficient final-query
   support is not reusable. Citation-ID validity is traceability, not semantic
   entailment.

### Unified Grounded Answer Synthesis Architecture

Unified synthesis (`flowcontext.synthesis`) combines retrieved evidence across
decomposed sub-questions into an auditable, structured answer:

- **Structured Input**: `GenerationRequest` carries `decomposed_intents`,
  `shared_constraints`, and `intent_evidence` mapping each intent ID to its
  retrieved passages. Untrusted passage text is treated strictly as data, never
  as instructions (`_INSTRUCTION_PATTERN` guards prevent prompt injection).
- **Structured Output**: Synthesis produces an `Answer` contract containing:
  - `answer_text`: Unified response covering all sub-questions.
  - `factual_claims`: Sequence of atomic `FactualClaim` records, each mapping to
    its addressed `intent_ids`, supporting `chunk_ids`, exact supporting
    excerpts, character spans, and semantic support verdict.
  - `intent_statuses`: Per-intent `IntentStatusRecord` reporting `answered`,
    `insufficient_evidence`, `conflicting_evidence`, or `needs_clarification`.
  - `verification_audit`: `SemanticVerificationReport` detailing the verifier
    model, per-claim verdicts, latency, and explicit limitations.
- **Answer Consistency**: To prevent the user-visible answer from asserting facts
  missing from structured claims, the answer is rendered directly from validated
  claim records (`render_unified_answer`) or verified consistent via
  `check_answer_consistency`.
- **Citation Provenance vs Semantic Support**:
  - Deterministic provenance validation (`validate_citations_and_excerpts`) checks
    that every cited chunk ID was actually supplied for this generation and that
    every cited excerpt matches source chunk text verbatim.
  - *Provenance is not truth*: exact excerpt matching proves only that the text
    appears in the document, not that the passage supports the proposition.
  - Pluggable semantic verification (`SemanticVerifier`, `RuleBasedSemanticVerifier`,
    `MockSemanticVerifier`) evaluates proposition support, lexical overlap, negation
    signals, and entity isolation. Claims failing semantic support are pruned or
    marked uncertain. Automated verification explicitly records that it does not
    guarantee real-world ground truth.
- **Conflicting Evidence Resolution**: When passages make contradictory claims,
  `resolve_conflicting_evidence` applies deterministic precedence rules:
  1. *Date precedence*: Later publication dates (`source_date`) supersede earlier ones.
  2. *Authority precedence*: Official policy (`official_policy`) supersedes drafts (`draft`)
     or preliminary guidelines.
  3. *Unresolvable conflict*: When metadata is insufficient to break ties, the conflict
     is reported explicitly with `conflicting_evidence` intent status.
- **Cross-Entity Evidence Isolation**: `check_cross_entity_match` prevents attributes,
  rules, or policies of one entity from being attributed to another entity.
- **Finalisation-Only Generation & Stale Guard**: Generation executes only on
  utterance finalisation. If transcript revisions advance while generation is in
  flight, `is_superseded` rejects the late result with `StaleGenerationError`
  and records `streaming_generation_stale_rejected` in trace logs.
- **Usage Accounting**: Token usage is recorded separately for initial generation
  (`generation_usage`), bounded validation repair (`repair_usage`), and semantic
  verification (`verification_usage`). Unknown/unpriced costs are reported as
  `"unavailable"`.
- **Provider Status**: Real provider API keys (OpenAI/Anthropic) are absent from
  the execution environment. Deterministic mock providers are used for verification
  and contract testing; mock outputs are clearly distinguished from live provider calls.
- **Scope Boundary**: Selective cross-turn answer updates across later turns belong
  to Phase 4 and are explicitly excluded from Phase 3.

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

The dedicated audit command preserves historical comparison reports and writes
a new JSON/Markdown pair from labels kept outside application code:

```bash
UV_CACHE_DIR=/tmp/flowcontext-uv-cache uv run --locked flowcontext build-index \
  --input data/synthetic/phase3_documents.jsonl \
  --output artifacts/phase3-lexical-index.json \
  --backend lexical --source-kind synthetic_fixture

UV_CACHE_DIR=/tmp/flowcontext-uv-cache uv run --locked flowcontext evaluate-phase3 \
  --corpus artifacts/phase3-lexical-index.json \
  --backend lexical --top-k 5 --split all \
  --execution-mode realtime \
  --phase3-development-cases data/evaluation/phase3_development.jsonl \
  --phase3-held-out-cases data/evaluation/phase3_held_out.jsonl \
  --implementation-changes data/evaluation/phase3_implementation_changes.json \
  --output reports/phase3_evaluation_realtime.json
```

The suite has 13 development and 7 held-out cases, with related
`variant_family` values kept in one split. It covers single questions that
must not be decomposed; independent and dependent requests; shared and
intent-local constraints; negation; comparisons; partial/unanswerable
requests; conflicts; entity confusion; corrections; and provider/retrieval
failures. The synthetic Phase 3 corpus contains 9 documents / 10 chunks and
the audit uses `k=5`, so k does not include every chunk.

An end-to-end decomposition payload with independent and dependent questions,
typed constraints, spans, ambiguity records, and a dependency edge is shown in
[`examples/multi_intent/README.md`](../examples/multi_intent/README.md). The
default CLI settings use the explicitly labelled mock structured provider for
offline contract checks. Set `FLOWCONTEXT_GENERATION_BACKEND=openai_compatible`
and the existing generation endpoint/model/key settings to exercise the real
provider path; credentials are never written to output.

The report contains:

- matched A original final-event baseline, B Phase 2 streaming single query,
  and C Phase 3 streaming decomposition/fusion arms;
- a one-to-one intent-matching rubric with missed intents and unnecessary
  extra intents, reported separately for development and held-out;
- successful retrieval Recall@1/@3/@5 with explicit denominators, per-intent
  recall, complete-request evidence coverage, and end-to-end outcomes that
  retain failures, timeouts, and abstentions;
- citation-ID validity as deterministic provenance only; semantic claim
  support remains `NOT VERIFIED` unless claims are genuinely reviewed;
- supported-answer coverage with an answerable-intent denominator so
  abstaining on every request cannot score as success, plus uncertainty on
  partial/unanswerable cases;
- early retrieval, useful early reuse, stale-result acceptance, final-event to
  answer latency, retrieval/model calls, tokens, errors, cost availability,
  and trace completeness;
- the implementation-change log showing which cases informed changes; and
- a lexical single-query versus decomposed engineering ablation plus an
  explicit `NOT VERIFIED` dense-only versus hybrid section. Mock comparisons
  are not presented as model-quality findings.

The held-out files were inspected only after implementation decisions were
frozen and were not used to tune decomposition, fusion, or streaming policy.
All intent, relevance, and answer-expectation labels are currently
`provisional_generated`; the standalone review-status asset records zero
model-reviewed and human-reviewed labels. New untouched cases should be
reserved for the next evaluation. Local labels remain provisional and semantic
claim support remains `NOT VERIFIED`.

## Current local findings

The fresh Phase 3 audit is reported in
[`reports/phase3_evaluation_realtime.md`](../reports/phase3_evaluation_realtime.md)
and its machine-readable companion. The local lexical/mock run found:

- Multi-intent exact identification was 100% (4/4) on development and 75%
  (3/4) on held-out, meeting the guide's 70% target separately in both splits
  under the explicit rubric. One held-out single-question paraphrase caused an
  unnecessary extra intent, which is retained as a failure rather than tuned
  away.
- C's complete-request evidence coverage was 0.8333 macro on development and
  0.8333 on held-out; supported-answer substring coverage was measured with
  an answerable-intent denominator, but is not semantic support.
- The local C uncertainty checks failed on the partial-answer and unanswerable
  probes, and the provisional supported-answer substring coverage was 2/15
  answerable intents in development and 2/10 in held-out. These are retained
  quality failures, not converted into success by citation presence.
- Citation-ID validity was deterministic provenance and does not establish
  the guide's 85% citation-support target, which remains `NOT VERIFIED`.
- The correction probe recorded zero stale-result acceptance in the streaming
  arms. Retrieval/generation/decomposition failures and timeout outcomes are
  retained as operational failures, not quality successes.

These values are local synthetic/provisional observations. Inspect the JSON
for full per-case A/B/C records, misses/extras, ranked evidence, claims,
latencies, resources, and trace events.

## Blockers

The attempted real dense smoke command was blocked because the optional
`sentence-transformers` dependency is not installed; install the dense extra
and download the pinned model before building a dense index. Provider-backed
generation was not run. The official corpus, replay format, labels,
thresholds, semantic grounding evaluator, and organiser API remain unavailable.
The lexical fixture, mock generator, and provisional labels therefore cannot
support a competition-performance claim. Phase 4 implementation and local
evaluation are recorded separately in [`phase4-handoff.md`](phase4-handoff.md)
and [`phase5-handoff.md`](phase5-handoff.md); those reports do not rewrite
these historical Phase 3 scores.
