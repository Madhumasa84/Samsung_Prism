# Phase 4 checklist

## Engineering

- [x] Bounded, explicitly clearable, session-isolated in-memory state.
- [x] Active topics, intents, typed constraints, origin turns, evidence/index
  identity, claims, dependencies, answer versions, and pending work are
  represented by validated contracts.
- [x] Proposed follow-up patches are bound to an exact base revision.
- [x] Add/replace/remove constraints, add questions, topic changes,
  formatting-only requests, mixed requests, and ambiguous references are
  classified without silent mutation.
- [x] User constraints remain distinct from corpus-supported facts.
- [x] Direct and transitive claim/entity/constraint invalidation is recorded.
- [x] Entity changes invalidate dependent policy/price/availability branches;
  constraint removal can broaden retrieval.
- [x] Selective retrieval searches changed information needs across the full
  corpus index and reuses only dependency-compatible evidence.
- [x] Bounded scheduler work is candidate-revision-bound; late results are
  rejected independently of cancellation.
- [x] Factual and presentation answer versions publish atomically with
  preserved/added/removed/modified claim records and unresolved questions.
- [x] Formatting-only turns use zero retrieval and preserve citations and
  qualifications; mixed factual turns do not take that shortcut.

## Evaluation

- [x] Separate development, diagnostic/regression, and untouched held-out
  Phase 4 JSONL assets.
- [x] Keep related variants in one split and record label-review status.
- [x] Matched full re-retrieval/regeneration and selective update arms share
  interpretation, corpus/index, providers, retrieval settings, and generation
  configuration.
- [x] Measure interpretation, invalidation precision/recall, obsolete and
  unaffected claim preservation, answer coverage, evidence structure,
  citation validity, uncertainty/clarification, latency, retrieval/generation
  usage, cost availability, formatting suppression, stale publication, session
  isolation, and trace completeness.
- [x] Include additions, replacements, removals, entity consequences,
  ambiguity, topic changes, partial/unanswerable turns, conflicts, formatting,
  failures, rapid corrections, stale work, and concurrent sessions.
- [x] Preserve historical Phase 3/diagnostic reports and mark affected cases
  diagnostic/regression rather than rewriting their historical scores.
- [x] Emit a claim-to-passage review sheet with structural provenance and
  content hashes; verdict reuse requires matching claim and support content.
- [x] Attempt the real-model engineering path without exposing secrets or
  substituting a hidden mock run.
- [x] Run the complete regression suite and a repeated deterministic fixture
  evaluation.
- [x] Identified manual claim review; Codex reviewed 89 synthetic emitted claims against cited passages. This is not independent human ground truth; the content-hash-bound sheet is `reports/phase4_evaluation_claim_review.csv`.
- [x] Real-provider generation scope recorded; Ollama qwen2.5:3b generation and embedding probe passed; dense/hybrid RAG integration verified with dense retrieval backend evaluation. Integrated dense RAG: PASS (recorded in `reports/phase4_evaluation_dense_test.json` and `reports/phase4_real_e2e_dense_test.json`).
- [ ] Official corpus/benchmark validation; `NOT VERIFIED` because official
  assets and harness are absent.

## Evidence

- Matched report: [`reports/phase4_evaluation.md`](reports/phase4_evaluation.md)
- Machine-readable report: [`reports/phase4_evaluation.json`](reports/phase4_evaluation.json)
- Dense evaluation report: [`reports/phase4_evaluation_dense_test.md`](reports/phase4_evaluation_dense_test.md)
- Dense evaluation metrics: [`reports/phase4_evaluation_dense_test.json`](reports/phase4_evaluation_dense_test.json)
- Claim review sheet: [`reports/phase4_evaluation_claim_review.csv`](reports/phase4_evaluation_claim_review.csv)
- Label status: [`data/evaluation/phase4_label_review_status.json`](data/evaluation/phase4_label_review_status.json)
- Real-backend traces: [`reports/phase4_real_e2e.json`](reports/phase4_real_e2e.json), [`reports/phase4_real_e2e_dense_test.json`](reports/phase4_real_e2e_dense_test.json)
- Replay traces: [`reports/phase4_replay_initial_compound.json`](reports/phase4_replay_initial_compound.json),
  [`reports/phase4_replay_late_constraint.json`](reports/phase4_replay_late_constraint.json),
  [`reports/phase4_replay_entity_correction.json`](reports/phase4_replay_entity_correction.json),
  [`reports/phase4_replay_partial_unsupported.json`](reports/phase4_replay_partial_unsupported.json),
  [`reports/phase4_replay_formatting.json`](reports/phase4_replay_formatting.json),
  [`reports/phase4_replay_race.json`](reports/phase4_replay_race.json)
- Architecture: [`docs/architecture-phase4.md`](docs/architecture-phase4.md)
- Phase 5 handoff: [`docs/phase5-handoff.md`](docs/phase5-handoff.md)

Final submission production is intentionally not started by this checklist.
