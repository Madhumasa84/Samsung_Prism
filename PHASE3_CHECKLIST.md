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
- [x] Streaming revision protection rejects stale or incomplete final evidence.
- [ ] Selective claim updates across follow-up turns (Phase 4; intentionally
  excluded).

## Evaluation

- [x] Per-case baseline/streaming comparison with status, IDs, queries, reuse,
  and scoring denominators.
- [x] Development and held-out results are separate.
- [x] Successful comparable retrieval quality is separated from end-to-end
  outcomes including failures and abstentions.
- [x] Phase 2 non-useful reuse cases are classified without held-out tuning.
- [x] Targeted decomposition, fusion, reuse, uncertainty, revision, and
  denominator regression tests.
- [x] New comparison report preserves historical Phase 2 reports.
- [ ] Real dense/provider-backed and semantic claim-support evaluation.

## Evidence

See [`reports/phase3_comparison.md`](reports/phase3_comparison.md), the new
retrieval audit [`reports/phase3_retrieval_comparison_realtime.md`](reports/phase3_retrieval_comparison_realtime.md), and
[`docs/phase3.md`](docs/phase3.md). The checked-in result is synthetic,
lexical, mock, and provisional; it is not an official benchmark result.
