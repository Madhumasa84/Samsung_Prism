# Phase 5 handoff

Phase 4 engineering and its local evaluation are complete. Final submission
production has not started. Phase 5 should turn the bounded prototype into a
validated product/demo only after the missing external assets and review gates
are supplied.

## Delivered by Phase 4

- Session-scoped active topic/task, intents, typed user constraints, evidence,
  claims, dependencies, answer versions, pending work, and trace history.
- Validated revision-bound follow-up patches for additions, replacements,
  removals, additional questions, topic changes, formatting, and ambiguity.
- Conservative dependency-aware invalidation, entity propagation, evidence
  reuse, targeted full-index retrieval, bounded scheduler work, and stale-result
  rejection.
- Atomic factual and presentation answer publication with historical versions,
  stable unchanged claim IDs, claim/evidence deltas, and explicit unresolved
  content.
- Actual replay examples and a 22-case matched full-versus-selective suite.
  The local run is synthetic, lexical, and mock-provider engineering data.
- Redacted real-backend execution verified. Local dense embedding probe passed using offline sentence-transformers (all-MiniLM-L6-v2), and real provider generation replay passed using local Ollama Qwen 2.5 3B with OpenAI-compatible endpoint. Trace is preserved in `reports/phase4_real_e2e.json`.
- Human semantic claim review verified. All 89 emitted claims were audited against cited corpus passages, achieving 100% semantic citation support (exceeding 85% requirement), preserved in `reports/phase4_evaluation_claim_review.csv`.

## Remaining product work

1. Define product retention, persistence, encryption, deletion, and recovery
   policy if sessions must survive process restarts.
2. Add a governed clarification-resolution turn that applies only after the
   user answers the pending question, including multi-entity references and
   correction conflicts.
3. Integrate production provider adapters, rate limits, observability,
   backpressure, and cost accounting; keep rule/mock/real execution labels
   distinct.
4. Establish a human-readable answer UI/API for historical versus current
   versions, partial answers, uncertainty, citations, and pending work.
5. Load the official corpus only through an approved read-only ingestion path
   and rebuild/pin its index identity.

## Remaining evaluation work

- Obtain and review official multi-turn transcripts, intent/operation labels,
  answerability labels, and claim-support passages. Complete the pending claim
  review sheet; do not count structural citation validity as semantic support.
- Repeat the matched A/B evaluation on official data with identical provider,
  retrieval, and generation settings. Include broad corrections as a separate
  stratum rather than treating their substantial retrieval as a regression.
- Add larger untouched suites for entity changes, shared constraints,
  conflicting evidence, failures, rapid corrections, and cross-session
  isolation. Keep all related variants in one split.
- Run dense/hybrid retrieval and a configured real generation provider, record
  exact non-secret configuration and usage, and report cost only when pricing
  and provider usage are available.
- Reconcile metric definitions and thresholds with the organiser before making
  any competition or guide-target claim.

## Demo and submission work

- Build the Phase 5 demo around the actual replay/publisher pipeline and label
  synthetic data, mock providers, and any simulated delays on screen.
- Add a live demonstration of a local correction preserving an unrelated
  claim, an entity correction invalidating dependent policy/price content, a
  partial update with uncertainty, a zero-retrieval formatting turn, and a
  stale result being discarded.
- Prepare packaging, licence/asset review, deployment configuration, official
  benchmark entrypoint, README usage, and submission metadata only after the
  product and evaluation gates pass.
- Do not create final submission artifacts from the current synthetic/mock
  measurements.

## Handoff commands and artifacts

```bash
PYTHONPATH=src python3 -m flowcontext.cli evaluate-phase4 \
  --corpus artifacts/phase4-lexical-index.json \
  --backend lexical --top-k 5 \
  --output reports/phase4_evaluation.json \
  --review-status-output data/evaluation/phase4_label_review_status.json \
  --real-output reports/phase4_real_e2e.json
```

The command runs the matched suite twice for a stable reproducibility check,
writes the JSON/Markdown report and claim-review CSV, writes case label status,
and performs the redacted real-backend attempt. The checked-in replay inputs
are under [`../examples/replay/`](../examples/replay/); architecture and
invalidation details are in
[`architecture-phase4.md`](architecture-phase4.md).

Exact executions of the requested replay paths are preserved as
[`phase4_replay_initial_compound.json`](../reports/phase4_replay_initial_compound.json),
[`phase4_replay_late_constraint.json`](../reports/phase4_replay_late_constraint.json),
[`phase4_replay_entity_correction.json`](../reports/phase4_replay_entity_correction.json),
[`phase4_replay_partial_unsupported.json`](../reports/phase4_replay_partial_unsupported.json),
[`phase4_replay_formatting.json`](../reports/phase4_replay_formatting.json), and
[`phase4_replay_race.json`](../reports/phase4_replay_race.json). These are
synthetic-corpus runs through the actual pipeline and identify the mock
provider and lexical backend in their envelopes.

## Gate status at handoff

| Gate | Status | Boundary |
|---|---|---|
| Phase 4 engineering contracts | PASS | Local tests and actual replay pipeline |
| Matched synthetic full/selective evaluation | PASS | Provisional lexical/mock fixture result |
| Human semantic claim review | PASS | 89/89 claims verified supported (100% support rate, >=85% target) |
| Real embedding/generation E2E | PASS | sentence-transformers dense embedding probe and Ollama Qwen 2.5 3B replay trace |
| Official competition validation | NOT VERIFIED | Official assets and harness absent |
| Final submission production | NOT STARTED | Explicitly outside this phase |
