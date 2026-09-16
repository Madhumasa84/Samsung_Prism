# Phase 2 checklist

This is the completed Phase 2 audit record. Phase 3 is now tracked separately
in [`PHASE3_CHECKLIST.md`](PHASE3_CHECKLIST.md); the historical Phase 2
measurements below are preserved.

Phase 2 is the streaming-controller, scheduler, replay, CLI, tracing, and
evaluation audit. The Phase 3 status in this historical checklist reflects
the boundary at the time of the Phase 2 handoff; current Phase 3 work is in
[`PHASE3_CHECKLIST.md`](PHASE3_CHECKLIST.md).

## Implementation

- [x] Preserve baseline retrieval after final-event delivery.
- [x] Preserve streaming retrieval during partial input.
- [x] Share corpus, retrieval backend, top-k, and generation configuration in
  matched baseline/streaming evaluation.
- [x] Support realtime and accelerated replay modes and label them in reports.
- [x] Record source and actual delivery times, controller decisions, retrieval
  scheduling/start/end, final delivery, evidence readiness, generation timing,
  request lifecycle, errors, usage, and cost availability.
- [x] Define early retrieval by actual retrieval start before final delivery.
- [x] Handle greetings, no-retrieval turns, formatting with context, and
  formatting without context (explicit clarification).
- [x] Keep real backend unavailability explicit; no silent mock fallback.
- [x] Keep structured traces concise and free of secrets.

## Evaluation

- [x] Add frozen development and held-out streaming JSONL suites.
- [x] Cover incomplete/final-only speech, stable partials, incremental and
  cumulative transcripts, repeats, entity/quantity/location/date/negation
  corrections, rapid corrections, late old results, greetings, formatting,
  failures, timeouts, closure, and concurrent sessions.
- [x] Declare eligibility, earliest retrieval point, expected final evidence,
  and provisional labels before controller execution.
- [x] Keep fixture, simulated-delay, real-backend, and official-asset results
  separate.
- [x] Compare baseline and streaming under matched conditions.
- [x] Report early retrieval, valid pre-final evidence, reuse, false and
  premature triggers, calls per utterance, stale acceptance, final evidence,
  timing samples, overhead, errors, usage, cost, and trace coverage.
- [x] Run a bounded duplicate-suppression/coalescing-disabled comparison.
- [x] State the 80% guide target result without inventing a false-trigger
  threshold or claiming an official benchmark.

## Audit results

| Area | Result |
|---|---|
| Early retrieval on eligible cases | 23/23 in the measured all-split fixture audit |
| Valid evidence before finalisation | 10/23 realtime |
| Useful early reuse | 10/23 realtime |
| False retrieval triggers | 0/6 |
| Premature triggers | 0/2 |
| Stale-result acceptance | 0 |
| Streaming retrieval calls | 37 total; 1.194 per utterance |
| Trace coverage | 31/31 paired cases |
| Duplicate policy comparison | realtime 1 vs 3; accelerated 1 vs 2; final IDs unchanged |
| Official/real backend validation | NOT VERIFIED |

## Documentation and handoff

- [x] Update [README](README.md), evaluation methodology, architecture/policy
  docs, known limitations, and examples.
- [x] Add measured JSON reports and the concise
  [Phase 2 evaluation report](reports/phase2_streaming_evaluation.md).
- [x] Update the AI-assistance log with commands and observed outcomes.
- [x] Add a Phase 3 boundary/handoff describing future multi-intent retrieval
  interfaces without implementing them.
- [x] Re-run baseline regression, clean setup, container, compile, config, and
  smoke checks affected by the changes.

## Explicit blockers

- [ ] Real dense/provider-backed validation: unavailable in this environment.
- [ ] Official corpus, replay set, human labels, thresholds, and scoring
  harness: not supplied.
- [ ] Semantic answer-support evaluation and provider cost measurement: not
  available with the fixture/mock configuration.

These blockers prevent validated competitive-performance claims, but do not
block the local engineering completion of Phase 2.
