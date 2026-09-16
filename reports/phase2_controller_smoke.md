# Historical Phase 2 controller engineering smoke

This is the pre-scheduler controller snapshot retained for comparison. The
current asynchronous scheduler smoke is in
[`phase2_scheduler_smoke.md`](phase2_scheduler_smoke.md).

This report is a measured local smoke artifact, not an organiser benchmark.
The corpus is the three-document/four-chunk synthetic fixture and retrieval is
the explicitly selected lexical backend. No generation was run in Phase 2.

The final CLI matrix built the same index twice (built, then up_to_date) and
ran accelerated streaming replay:

| Input | Status | Retrieval attempts | Early | Final | Stale | Operationally unnecessary | Traces |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| early-retrieval.jsonl | completed | 2 | 1 | 1 | 0 | 0 | 11 |
| cumulative-correction.jsonl | completed | 2 | 1 | 1 | 0 | 0 | 11 |
| formatting-needs-context.jsonl | needs_context | 0 | 0 | 0 | 0 | 0 | 5 |

The two early-retrieval rows performed one search before finalisation and one
for the revised final query. The correction row ended with
explicit_correction. Formatting without same-session prior-answer context
returned SKIP plus an explicit needs_context outcome. The operational
unnecessary count is only revision-superseded work; it is not a semantic
usefulness judgment.

The Phase 1 default baseline was also run in the same matrix and remained
wait_for_complete_utterance with no early retrieval. All 37 unit tests,
offline smoke, trace validation, and manifest validation passed. Real dense
retrieval and real-provider generation remain unverified because the optional
model/package and credentials were unavailable.
