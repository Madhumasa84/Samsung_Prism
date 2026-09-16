# Phase 2 streaming evaluation and implementation audit

Status: engineering audit complete. This is not an official benchmark or a
competitive-performance claim. Phase 3 was not started.

Machine-readable results:

- [realtime report](phase2_streaming_evaluation.json)
- [accelerated report](phase2_streaming_evaluation_accelerated.json)

## Method

The dedicated suite contains 22 development cases and 9 held-out cases (31
measured cases; 23 eligible for early retrieval). Related variants remain in
the same split. Eligibility, the earliest reasonable retrieval point, and
expected final evidence are declared in the JSONL case files before the
controller runs. Labels are `provisional_generated`; no human verification is
claimed.

Baseline and streaming use the same synthetic corpus, lexical backend, top-k
5, mock generation configuration, and replay events. Fixture, bounded
simulated-delay, real-backend, and official-asset results are separate. The
primary timing report is realtime replay. Accelerated replay is a scheduling
check only and must not be compared with realtime wall-clock latency.

## Realtime measured results

| Metric | Result |
|---|---:|
| Early retrieval on eligible cases | 23/23 = 100.0%; guide target (>=80%) met on this measured set |
| Valid evidence ready before finalisation | 10/23 = 43.5% |
| Useful early-result reuse | 10/23 = 43.5% |
| False retrieval triggers | 0/6 |
| Premature triggers | 0/2 |
| Streaming retrieval calls | 37 total; 1.194 per utterance |
| Baseline retrieval calls | 25 total; 0.806 per utterance |
| Stale-result acceptance | 0 |
| Final evidence Recall@5 | streaming 91.7%; baseline 96.0% |
| Complete traces | 31/31 |
| Errors | 6, all surfaced in the bounded failure/timeout cases |
| Cost | unavailable; fixture usage is estimated and no provider prices are configured |

Only an actual retrieval start before final-event delivery counts as early.
Scheduling or a controller decision alone does not count. Answer latency is
measured from final-event delivery. Realtime timing samples are reported per
section, with fixture `n=27` and simulated-delay `n=4`; one closed session has
no answer-latency sample.

| Realtime section | Baseline answer p50/p95 ms | Streaming answer p50/p95 ms | Streaming full interaction p50/p95 ms | Controller overhead p50/p95 ms |
|---|---:|---:|---:|---:|
| Fixture, n=27 | 1.60 / 3.37 | 2.39 / 4.52 | 207.29 / 715.55 | 0.32 / 0.63 |
| Simulated delay, n=4/3 | 17.10 / 33.05 | 12.98 / 32.88 | 206.27 / 242.34 | 0.29 / 0.89 |

The accelerated report repeats functional counts, with valid pre-final
evidence at 0/23 because synthetic accelerated execution does not create the
same delivery interval. Its timing interpretation is explicitly
`synthetic accelerated scheduling timing; not model-performance latency`.
Its functional run surfaced 8 bounded retrieval/cancellation errors, also
without accepting stale evidence.

## Duplicate-suppression comparison

The focused `stream-dev-repeated` comparison was resource-bounded. In the
realtime run, enabled policy made 1 retrieval call; with duplicate suppression
and coalescing disabled it made 3 calls (+2) and returned the same final
evidence IDs. In the separate accelerated run, disabling the policy made 2
calls (+1), with 1 superseded request and 1 stale-result discard; final
evidence IDs were unchanged. This supports reduced wasted requests in this
focused case; it is not a semantic quality claim.

## Representative trace timelines

The JSON report keeps compact full timelines for representative development
cases and trace coverage for every case. Times below are source seconds; every
event also records actual monotonic delivery time.

- Incomplete speech: `WAIT/incomplete_phrase` → meaningful partial
  `RETRIEVE` → actual start → evidence ready → final delivery →
  `SKIP/duplicate_query_suppressed` → generation/answer.
- Final-only request: `WAIT/incomplete_phrase` → final delivery →
  `RETRIEVE/final_complete_query` → evidence → answer.
- Repeated fragments: one `RETRIEVE`/actual start, then two
  `SKIP/duplicate_query_suppressed` decisions; no duplicate searches.
- Rapid correction: three retrieval starts were observed; the first two late
  results were explicitly stale-rejected, and only the latest result was
  accepted. Stale-result acceptance remained zero.
- Formatting with context: final delivery → `SKIP/formatting_with_previous_answer_context`;
  no corpus search. Without context the run is an explicit clarification
  (`needs_context`), not invented content.
- Session close: a request was actually started, then the session closed;
  no answer was fabricated.
- Failure and timeout: backend failure and the declared per-case timeout are
  surfaced as failed/abstained traces; timeout work is stale-rejected and does
  not become evidence.

## Asset and backend separation

The `fixture` section is synthetic lexical/mock data; `simulated_delay` is the
same fixture with bounded retrieval delays/failure/timeout/session-close
behaviour. `real_backend` is `NOT VERIFIED`: no real dense/provider-backed run
was silently replaced by a mock. `official_assets` is `NOT VERIFIED`: the
official corpus, replay set, labels, thresholds, and scoring harness were not
provided. Therefore these numbers must not be described as official results.

## PASS / FAIL / NOT VERIFIED

| Area | Status | Evidence / limitation |
|---|---|---|
| Streaming decisions | PASS | 31 paired cases and trace-level decision reasons |
| Actual early retrieval | PASS | 23/23 eligible cases had an actual pre-final start |
| Duplicate suppression | PASS | realtime 1 vs 3 calls; accelerated 1 vs 2; final IDs unchanged |
| Correction handling | PASS | entity/quantity/location/date/negation cases plus rapid corrections |
| Stale-result rejection | PASS | accepted stale results: 0; late old results rejected |
| Final-query evidence validity | PASS | chunk-ID validity/Recall@5 measured; semantic support unverified |
| Session isolation | PASS | concurrent session cases and isolation tests |
| No-retrieval behaviour | PASS | 0/6 false triggers; greetings/formatting bypass corpus search |
| Trace completeness | PASS | 31/31 paired traces complete |
| Reproducibility | PASS | frozen JSONL splits, commands, corpus fingerprint, and two reports |
| Real-backend validation | NOT VERIFIED | backend/package/credentials/assets unavailable |

## Known limitations and blockers

- Labels are provisional generated labels; they need review before any stronger
  quality claim.
- Final evidence quality is identifier-level retrieval-label quality, not
  human semantic answer evaluation. The guide supplies no numerical
  false-trigger threshold, so none is invented here.
- Mock generation cost is unavailable; real provider usage/pricing remains
  unmeasured.
- Real dense/provider-backed and official-asset validation remain explicit
  blockers.
- A useful early start is not guaranteed to finish before finalisation: the
  measured useful-reuse rate is 10/23, and rapid corrections can create
  superseded work. Multi-intent retrieval interfaces are deferred to Phase 3.

Engineering completion is therefore PASS for the local Phase 2 interfaces,
trace/reporting, and reproducible fixture audit; validated competitive
performance is NOT VERIFIED.
