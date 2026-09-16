# Phase 3 comparison and Phase 2 gap investigation

Status: local synthetic/lexical/mock engineering evidence; not an official or competitive result.

Historical reports are preserved. This report records a fresh matched rerun and corrected denominators.

## Findings

- Successful comparable retrieval quality: both modes are measured on the same completed cases and explicit relevant-ID denominator.
- End-to-end results retain all failed, timed-out, and closed cases.
- Reuse is gated by final-query compatibility plus a lexical relevance proxy; semantic entailment remains unevaluated.
- Phase 3 adds structural decomposition, parallel per-intent retrieval, RRF fusion, and grounded multi-intent generation behind `--multi-intent`.
- Cause audit: stale/superseded early results, ranking changes, and successful retrieved-set differences are listed per case; the historical gap is attributed to metric inclusion/denominator differences, not a matched successful retrieval-quality gap.
- Historical quality samples: baseline 25 values with zero-quality cases ['stream-dev-retrieval-failure']; streaming 24 values with zero-quality cases ['stream-dev-retrieval-failure', 'stream-dev-retrieval-timeout'].

## Per-case baseline / streaming comparison

| Case | Split | Status B/S | Final query B/S | Retrieved IDs B/S | Relevant IDs | Request status B/S | Reuse decision B/S | Scoring denominator B/S |
|---|---|---|---|---|---|---|---|---:|
| stream-dev-incomplete-meaningful | development | `completed` / `completed` | Which venue in Pune? / Which venue in Pune? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / accepted_early_reuse | 1 / 1 |
| stream-dev-final-only | development | `completed` / `completed` | What is the cancellation policy? / What is the cancellation policy? | synthetic.cancellation-policy#chunk-0000 / synthetic.cancellation-policy#chunk-0000 | synthetic.cancellation-policy#chunk-0000 | completed / completed | baseline_final_only / no_early_request | 1 / 1 |
| stream-dev-stable-partial | development | `completed` / `completed` | Which venue in Pune? / Which venue in Pune? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / accepted_early_reuse | 1 / 1 |
| stream-dev-incremental | development | `completed` / `completed` | Which venue in Pune? / Which venue in Pune? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / accepted_early_reuse | 1 / 1 |
| stream-dev-cumulative | development | `completed` / `completed` | Which venue in Pune? / Which venue in Pune? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / accepted_early_reuse | 1 / 1 |
| stream-dev-repeated | development | `completed` / `completed` | Which venue in Pune? / Which venue in Pune? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / accepted_early_reuse | 1 / 1 |
| stream-dev-correction-entity | development | `completed` / `completed` | Actually, which venue in Pune accommodates 30 attendees? / Actually, which venue in Pune accommodates 30 attendees? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / final_query_changed_no_reuse | 1 / 1 |
| stream-dev-correction-quantity | development | `completed` / `completed` | Actually, which venue in Pune for 30 attendees? / Actually, which venue in Pune for 30 attendees? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / final_query_changed_no_reuse | 1 / 1 |
| stream-dev-correction-location | development | `completed` / `completed` | Actually, which venue accommodates 30 attendees in Pune? / Actually, which venue accommodates 30 attendees in Pune? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / final_query_changed_no_reuse | 1 / 1 |
| stream-dev-correction-date | development | `completed` / `completed` | Actually, which venue in Pune available Monday? / Actually, which venue in Pune available Monday? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / final_query_changed_no_reuse | 1 / 1 |
| stream-dev-correction-negation | development | `completed` / `completed` | Actually, which venue in Pune without catering? / Actually, which venue in Pune without catering? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001, synthetic.catering-options#chunk-0000 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001, synthetic.catering-options#chunk-0000 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / final_query_changed_no_reuse | 1 / 1 |
| stream-dev-coalescing | development | `completed` / `completed` | Which venue in Pune? / Which venue in Pune? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / final_query_changed_no_reuse | 1 / 1 |
| stream-dev-greeting | development | `completed` / `completed` | Hello! / Hello! | — / — | — | not_scheduled / not_scheduled | no_retrieval_policy / no_early_request | 0 / 0 |
| stream-dev-thanks | development | `completed` / `completed` | Thanks! / Thanks! | — / — | — | not_scheduled / not_scheduled | no_retrieval_policy / no_early_request | 0 / 0 |
| stream-dev-format-context | development | `completed` / `completed` | Please make it shorter. / Please make it shorter. | — / — | — | not_scheduled / not_scheduled | no_retrieval_policy / no_early_request | 0 / 0 |
| stream-dev-format-no-context | development | `abstained` / `needs_context` | Please make it shorter. / Please make it shorter. | — / — | — | not_scheduled / not_scheduled | no_retrieval_policy / no_early_request | 0 / 0 |
| stream-dev-concurrent-venue | development | `completed` / `completed` | Which venue in Pune? / Which venue in Pune? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / accepted_early_reuse | 1 / 1 |
| stream-dev-concurrent-cancellation | development | `completed` / `completed` | What is the cancellation policy? / What is the cancellation policy? | synthetic.cancellation-policy#chunk-0000 / synthetic.cancellation-policy#chunk-0000 | synthetic.cancellation-policy#chunk-0000 | completed / completed | baseline_final_only / accepted_early_reuse | 1 / 1 |
| stream-dev-rapid-old-results | development | `completed` / `completed` | Actually, which venue in Pune? / Actually, which venue in Pune? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / superseded | baseline_final_only / rejected_stale_after_query_revision | 1 / 1 |
| stream-dev-retrieval-failure | development | `failed` / `failed` | Which venue in Pune? / Which venue in Pune? | — / — | synthetic.venue-options#chunk-0000 | failed / failed | baseline_final_only / early_retrieval_failed | 1 / 1 |
| stream-dev-retrieval-timeout | development | `completed` / `failed` | Which venue in Pune? / Which venue in Pune? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / — | synthetic.venue-options#chunk-0000 | completed / timed_out | baseline_final_only / early_retrieval_timed_out | 1 / 1 |
| stream-dev-session-close | development | `completed` / `closed` | Which venue in Pune? / — | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / — | synthetic.venue-options#chunk-0000 | completed / closed | baseline_final_only / session_closed_before_final | 1 / 1 |
| stream-heldout-paraphrase-venue | held_out | `completed` / `completed` | Which Pune venue can fit 30 people? / Which Pune venue can fit 30 people? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / accepted_early_reuse | 1 / 1 |
| stream-heldout-final-cancellation | held_out | `completed` / `completed` | Can you explain cancellation refunds? / Can you explain cancellation refunds? | synthetic.cancellation-policy#chunk-0000 / synthetic.cancellation-policy#chunk-0000 | synthetic.cancellation-policy#chunk-0000 | completed / completed | baseline_final_only / no_early_request | 1 / 1 |
| stream-heldout-incremental-catering | held_out | `completed` / `completed` | Tell me about catering options. / Tell me about catering options. | synthetic.catering-options#chunk-0000, synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.catering-options#chunk-0000, synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.catering-options#chunk-0000 | completed / completed | baseline_final_only / final_query_changed_no_reuse | 1 / 1 |
| stream-heldout-cumulative-catering | held_out | `completed` / `completed` | Tell me about catering options. / Tell me about catering options. | synthetic.catering-options#chunk-0000, synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.catering-options#chunk-0000, synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.catering-options#chunk-0000 | completed / completed | baseline_final_only / accepted_early_reuse | 1 / 1 |
| stream-heldout-correction-entity | held_out | `completed` / `completed` | Actually, tell me about Venue B in Pune. / Actually, tell me about Venue B in Pune. | synthetic.venue-options#chunk-0001, synthetic.venue-options#chunk-0000 / synthetic.venue-options#chunk-0001, synthetic.venue-options#chunk-0000 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / final_query_changed_no_reuse | 1 / 1 |
| stream-heldout-correction-quantity | held_out | `completed` / `completed` | Sorry, find a venue for 40 people in Pune. / Sorry, find a venue for 40 people in Pune. | synthetic.venue-options#chunk-0001, synthetic.venue-options#chunk-0000 / synthetic.venue-options#chunk-0001, synthetic.venue-options#chunk-0000 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / final_query_changed_no_reuse | 1 / 1 |
| stream-heldout-greeting | held_out | `completed` / `completed` | Good morning! / Good morning! | — / — | — | not_scheduled / not_scheduled | no_retrieval_policy / no_early_request | 0 / 0 |
| stream-heldout-format-context | held_out | `completed` / `completed` | Put that in bullets. / Put that in bullets. | — / — | — | not_scheduled / not_scheduled | no_retrieval_policy / no_early_request | 0 / 0 |
| stream-heldout-concurrent | held_out | `completed` / `completed` | Which venue in Pune? / Which venue in Pune? | synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 / synthetic.venue-options#chunk-0000, synthetic.venue-options#chunk-0001 | synthetic.venue-options#chunk-0000 | completed / completed | baseline_final_only / accepted_early_reuse | 1 / 1 |

## Development summary

Eligible early retrieval: 17/17; useful reuse: 7/17; non-useful: 10/17.

Successful comparable Recall@5 (B/S): 1.0 / 1.0 over 15 matched cases.

Phase 3 successful comparable Recall@5 (B/S): 1.0 / 1.0 over 4 matched cases.

Phase 3 end-to-end completion rate (B/S): 0.8 / 0.8 over 5 cases; non-completed cases remain listed in JSON.

End-to-end non-completed cases (B): stream-dev-format-no-context, stream-dev-retrieval-failure

End-to-end non-completed cases (S): stream-dev-format-no-context, stream-dev-retrieval-failure, stream-dev-retrieval-timeout, stream-dev-session-close

Phase 3 partial/missing-evidence cases (B): dev-unsupported-001

Phase 3 partial/missing-evidence cases (S): dev-unsupported-001

Non-useful eligible early reuse classification:

- `stream-dev-correction-entity`: expected_correction_or_final_constraint_change — the final query differs from the early query and the revision guard correctly prevents reuse
- `stream-dev-correction-quantity`: expected_correction_or_final_constraint_change — the final query differs from the early query and the revision guard correctly prevents reuse
- `stream-dev-correction-location`: expected_correction_or_final_constraint_change — the final query differs from the early query and the revision guard correctly prevents reuse
- `stream-dev-correction-date`: expected_correction_or_final_constraint_change — the final query differs from the early query and the revision guard correctly prevents reuse
- `stream-dev-correction-negation`: expected_correction_or_final_constraint_change — the final query differs from the early query and the revision guard correctly prevents reuse
- `stream-dev-coalescing`: expected_correction_or_final_constraint_change — the final query differs from the early query and the revision guard correctly prevents reuse
- `stream-dev-rapid-old-results`: delayed_retrieval_before_final — early retrieval was started, but no validated current result was ready before final-event delivery; late work was not reused
- `stream-dev-retrieval-failure`: intentional_fault_injection_or_session_close — case declares retrieval_behavior=failure; the failure/closure is part of the fixture
- `stream-dev-retrieval-timeout`: intentional_fault_injection_or_session_close — case declares retrieval_behavior=timeout; the failure/closure is part of the fixture
- `stream-dev-session-close`: intentional_fault_injection_or_session_close — case declares retrieval_behavior=session_close; the failure/closure is part of the fixture

## Held Out summary

Eligible early retrieval: 6/6; useful reuse: 3/6; non-useful: 3/6.

Successful comparable Recall@5 (B/S): 1.0 / 1.0 over 7 matched cases.

Phase 3 successful comparable Recall@5 (B/S): 1.0 / 1.0 over 4 matched cases.

Phase 3 end-to-end completion rate (B/S): 0.8 / 0.8 over 5 cases; non-completed cases remain listed in JSON.

End-to-end non-completed cases (B): none

End-to-end non-completed cases (S): none

Phase 3 partial/missing-evidence cases (B): heldout-unsupported-001

Phase 3 partial/missing-evidence cases (S): heldout-unsupported-001

Non-useful eligible early reuse classification:

- `stream-heldout-incremental-catering`: expected_correction_or_final_constraint_change — the final query differs from the early query and the revision guard correctly prevents reuse
- `stream-heldout-correction-entity`: expected_correction_or_final_constraint_change — the final query differs from the early query and the revision guard correctly prevents reuse
- `stream-heldout-correction-quantity`: expected_correction_or_final_constraint_change — the final query differs from the early query and the revision guard correctly prevents reuse

## Remaining blockers

- Official corpus, replay schema, labels, thresholds, and scoring harness were not supplied.
- Dense retrieval smoke blocker: dense backend requires optional dependency 'sentence-transformers'; install the dense extra and download the pinned model before building a dense index.
- Real-provider generation was not executed in this environment.
- Semantic claim support and factual-grounding rate require human review or a governed evaluator.
- Phase 4 selective claim updates across follow-up turns are intentionally out of scope.
- The local RRF/decomposition evaluation is fixture evidence, not competition performance.
