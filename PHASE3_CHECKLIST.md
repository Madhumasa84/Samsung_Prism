# Phase 3 checklist

Phase 3 adds multi-intent retrieval and grounded synthesis behind an explicit
opt-in path while preserving the Phase 1 baseline and Phase 2 streaming mode.

## Boundary

- [x] Structured, revision-bound decomposition with exact source spans,
  typed constraints, ambiguity records, and stable intent IDs.
- [x] Configured model-backed provider path with bounded validation repair,
  timeout/usage/latency telemetry, and explicit original-query fallback.
- [x] Stable-revision controller integration; no decomposer call per token and
  obsolete decomposition results are ignored.
- [x] Parallel per-intent retrieval with isolated errors.
- [x] Selectable dense-only, lexical-only, mock, and hybrid retrieval modes.
- [x] Bounded dependency-aware retrieval and per-intent backend RRF fusion.
- [x] Fair context-budget evidence assembly with exact provenance.
- [x] Deterministic RRF evidence fusion and provenance-preserving deduplication.
- [x] Final-query reuse validation beyond chunk-ID equality.
- [x] Grounded synthesis receives intent queries and reports unsupported intent
  evidence as uncertainty/abstention.
- [x] Unified answer synthesis producing structured output: answer text, atomic
  claims, intent IDs, supporting chunk IDs, supporting excerpts/spans, and
  per-intent statuses (`answered`, `insufficient_evidence`, `conflicting_evidence`,
  `needs_clarification`).
- [x] Deterministic citation validity and excerpt matching (verifies provenance
  against supplied chunk text; provenance != truth).
- [x] Pluggable semantic verification recording model, verdicts, latency, and
  limitations; unsupported claims marked uncertain, pruned, or repaired.
- [x] Explicit conflicting evidence handling via documented date and authority
  precedence rules or explicit conflict marking.
- [x] Cross-entity evidence isolation preventing policy attribution across distinct entities.
- [x] Finalisation-only generation and stale generation guard (`is_superseded` rejects
  generations completing after request revision advances).
- [x] Separate token usage accounting for generation, repair, and verification
  stages (reporting cost as "unavailable" when unknown).
- [x] Streaming revision protection rejects stale or incomplete final evidence.
- [ ] Selective claim updates across follow-up turns (Phase 4; intentionally
  excluded).

## Evaluation

- [x] Dedicated external-label suite includes non-decomposable singles,
  independent and dependent questions, shared/intent constraints, negation,
  comparisons, partial requests, conflicts, entity confusion, corrections, and
  provider/retrieval failures.
- [x] Matched A/B/C comparison: original final-event baseline, Phase 2
  streaming single query, and Phase 3 streaming decomposition/evidence fusion.
- [x] Development and held-out results are separate.
- [x] Related scenario variants are held within one split and the report
  records the split-integrity check.
- [x] Expected intents, relevance labels, answer expectations, and review
  provenance are external to application code; current labels are explicitly
  `provisional_generated`.
- [x] Successful comparable retrieval quality is separated from end-to-end
  outcomes including failures and abstentions.
- [x] Corpus size and k are reported together; the dedicated audit uses 9
  documents / 10 chunks and `k=5`.
- [x] Explicit one-to-one multi-intent rubric reports missed intents and
  unnecessary extras.
- [x] Per-intent retrieval recall, complete-request evidence coverage,
  supported-answer coverage, partial-request uncertainty, citation-ID
  validity, early retrieval/reuse, stale acceptance, latency, resources,
  errors, cost availability, and trace completeness are reported.
- [x] Phase 2 non-useful reuse cases are classified without held-out tuning.
- [x] Targeted decomposition, fusion, reuse, uncertainty, revision, and
  denominator regression tests.
- [x] Single-query versus decomposed retrieval ablation is measured and
  labelled as a lexical engineering observation, not model quality.
- [x] New dedicated report preserves historical Phase 2/earlier Phase 3
  reports.
- [ ] Dense-only versus hybrid ablation; `NOT VERIFIED` because no usable
  dense model/index was available.
- [ ] Semantic claim-support evaluation and the guide's 85% citation-support
  target; `NOT VERIFIED` because citation IDs do not establish support.
- [ ] Real provider-backed and official benchmark validation.

## Evidence

See the dedicated audit
[`reports/phase3_evaluation_realtime.md`](reports/phase3_evaluation_realtime.md)
and machine-readable companion, the historical reports
[`reports/phase3_comparison.md`](reports/phase3_comparison.md) and
[`reports/phase3_retrieval_comparison_realtime.md`](reports/phase3_retrieval_comparison_realtime.md),
[`docs/phase3.md`](docs/phase3.md), and the interface-only
[`docs/phase4-handoff.md`](docs/phase4-handoff.md). The Phase 3 result is
historical, synthetic, lexical, mock, and provisional; it is not an official
benchmark result. Phase 4 evidence is tracked separately in
[`PHASE4_CHECKLIST.md`](PHASE4_CHECKLIST.md).
