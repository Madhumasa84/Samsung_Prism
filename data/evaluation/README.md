# Local Phase 1 evaluation set

These files are a **local synthetic evaluation set**, not Samsung PRISM
official assets. They were authored from the three-document engineering
fixture in [`../synthetic/documents.jsonl`](../synthetic/documents.jsonl) because
the workspace contains no official corpus, replay prompts, labels, benchmark
runner, or scoring rules.

`development.jsonl` and `held_out.jsonl` are separate files. Related venue,
cancellation, catering, correction, and formatting scenarios are kept within a
single split. The held-out file is not used to tune the implementation; it is
only run as a fixed check. Every record embeds its provisional transcript,
expected evidence chunk IDs, answerability, and an answer-support rubric.

The `relevance_labels` and expected evidence IDs are generated engineering
labels and are `provisional_generated`; they are not human-verified ground
truth. A valid citation ID proves only that the answer cited a supplied chunk.
Claim support requires the documented human review procedure in
[`../../docs/evaluation.md`](../../docs/evaluation.md), which has not been
performed for these files.

The cases intentionally cover simple questions, compound requests, an
unsupported question, a cumulative correction, and formatting requests.
Capabilities such as early retrieval, query decomposition, selective answer
updates, external actions, and phase-specific streaming formatting are marked
as deferred and are not scored as Phase 1 features.

## Dedicated Phase 2 streaming suite

`streaming_development.jsonl` and `streaming_held_out.jsonl` are the separate
Phase 2 suite. They keep paraphrases and related scenario variants in one
split, and the held-out file is frozen before measurement rather than used for
policy tuning. The 31 cases cover incomplete and final-only speech, stable
partials, incremental/cumulative transcripts, repeated fragments, entity/
quantity/location/date/negation corrections, rapid corrections and late
results, greetings, formatting with and without same-session context, failure,
timeout, session closure, and concurrent sessions.

Each case records early-retrieval eligibility and the earliest reasonable
non-final event before the controller runs. The measured numerator is actual
`streaming_retrieval_started` before final-event delivery; a decision or queued
request is not an early retrieval. Expected evidence and relevance labels are
`provisional_generated`, not human-verified. The suite compares baseline and
streaming under the same corpus, index, backend, top-k, generation settings,
and replay mode. Fixture, simulated-delay, real-backend, and official-asset
sections remain separate in the machine-readable report.

Run the development and held-out checks explicitly:

```bash
uv run flowcontext evaluate-streaming \
  --cases data/evaluation/streaming_development.jsonl \
  --split development \
  --corpus artifacts/fixture-lexical-index.json \
  --backend lexical --top-k 5 \
  --execution-mode realtime \
  --compare-disabled-scheduling \
  --output artifacts/phase2-streaming-development.json

uv run flowcontext evaluate-streaming \
  --cases data/evaluation/streaming_held_out.jsonl \
  --split held_out \
  --corpus artifacts/fixture-lexical-index.json \
  --backend lexical --top-k 5 \
  --execution-mode realtime \
  --output artifacts/phase2-streaming-held-out.json
```

Use `--execution-mode accelerated` for bounded scheduling/correction checks.
Its wall-clock values are synthetic behavior observations and must not be
compared with realtime latency. The report does not invent a numerical
false-trigger threshold; it reports false triggers and premature triggers
directly. The 80% early-retrieval value is a guide target for this local
engineering set, not an official benchmark threshold.

## Dedicated Phase 3 audit data

`phase3_development.jsonl` (13 cases) and `phase3_held_out.jsonl` (7 cases)
are a separate, corpus-grounded Phase 3 suite. The corpus is
[`../synthetic/phase3_documents.jsonl`](../synthetic/phase3_documents.jsonl):
9 synthetic documents become 10 indexed chunks. The audit uses `k=5`, so
retrieval quality cannot pass merely by returning every chunk. All three
matched arms use this same index, configuration, hardware, and replay events:

* A — original final-event baseline;
* B — Phase 2 streaming with one parent query; and
* C — Phase 3 streaming with decomposition, intent-local retrieval, RRF
  fusion, and grounded synthesis.

The suite covers single questions that must stay whole, independent and
dependent questions, shared and intent-local constraints, negation,
comparisons, partial and unsupported requests, conflicting evidence,
cross-entity distractors, correction-time retrieval, and provider/retrieval
faults. `variant_family` is checked so related variants do not cross splits.

Expected intents, relevance IDs/labels, and answer expectations live only in
these external JSONL files; they are not imported by runtime retrieval or
generation code. Every label dimension is currently
`provisional_generated`. The standalone
[`phase3_label_review_status.json`](phase3_label_review_status.json) records
that model-reviewed and human-reviewed counts are zero. Semantic claim
support is not reviewed; citation-ID validity is reported separately and does
not establish the 85% citation-support target.

Run the dedicated audit against its dedicated index:

```bash
uv run flowcontext build-index \
  --input data/synthetic/phase3_documents.jsonl \
  --output artifacts/phase3-lexical-index.json \
  --backend lexical --source-kind synthetic_fixture

uv run flowcontext evaluate-phase3 \
  --corpus artifacts/phase3-lexical-index.json \
  --backend lexical --top-k 5 --split all --execution-mode realtime \
  --phase3-development-cases data/evaluation/phase3_development.jsonl \
  --phase3-held-out-cases data/evaluation/phase3_held_out.jsonl \
  --implementation-changes data/evaluation/phase3_implementation_changes.json \
  --output reports/phase3_evaluation_realtime.json
```

The matched report records per-intent retrieval recall, complete-request
evidence coverage, multi-intent misses/extras under an explicit rubric,
supported-answer coverage, uncertainty on partial cases, early retrieval and
reuse, stale-result acceptance, final-event-to-answer latency, calls/tokens/
errors/cost availability, and trace completeness. Focused single-query versus
decomposed retrieval is labelled as a lexical engineering ablation. Dense-only
versus hybrid is `NOT VERIFIED` because no real dense index/model was
available; no mock comparison is presented as model quality. Historical
reports are preserved.

### Diagnostic and Phase 4 evaluation cases

The four previously failing cases are marked
`evaluation_role: "diagnostic_regression"` in the Phase 3 JSONL files:

* `p3-heldout-single-catering-001` — one package question must remain one intent;
* `p3-dev-partial-008` — answer the venue portion and mark the tax-number portion unsupported;
* `p3-dev-unsupported-013` — abstain on the unsupported tax-number request; and
* `p3-heldout-unsupported-007` — abstain on the unsupported insurance-code request.

Their labels and the historical report are unchanged. The role only prevents a
fixed diagnostic from being mistaken for a newly untouched generalisation
case.

## Dedicated Phase 4 matched evaluation

Phase 4 uses three JSONL assets over the dedicated synthetic corpus in
[`../synthetic/phase4_documents.jsonl`](../synthetic/phase4_documents.jsonl):

* `phase4_development.jsonl` — 13 development cases;
* `phase4_diagnostic.jsonl` — three previously inspected Phase 3 failure
  regressions; and
* `phase4_untouched.jsonl` — six untouched held-out generalisation cases.

The evaluator combines these into 22 cases while retaining the split and role
fields. Related `variant_family` values are checked for split crossing. Cases
cover constraint addition/replacement/removal, entity consequences, local
claim preservation, ambiguity, topic changes, partial/unanswerable follow-ups,
conflicts, formatting and mixed turns, retrieval/generation failures, rapid
corrections, stale results, and concurrent sessions. All labels are currently
`provisional_generated`; no person completed the review. The claim sheet's
structural citation field is machine-computed, while semantic support remains
`pending_human_review`.

Run the matched arms with:

```bash
uv run flowcontext evaluate-phase4 \
  --corpus artifacts/phase4-lexical-index.json \
  --backend lexical --top-k 5 \
  --output reports/phase4_evaluation.json \
  --review-status-output data/evaluation/phase4_label_review_status.json \
  --real-output reports/phase4_real_e2e.json
```

Full re-retrieval/regeneration and selective updates share the same
interpretation patch, corpus/index, provider configuration, retrieval settings,
generation settings, and transcript. A broad correction can legitimately
search the full index; selective means changed information needs, not a
restriction to previously seen chunks. The measured run records quality as
well as resource use, and does not call zero savings an improvement.

The machine-readable report, Markdown report, claim review sheet, and label
status are [`../../reports/phase4_evaluation.json`](../../reports/phase4_evaluation.json),
[`../../reports/phase4_evaluation.md`](../../reports/phase4_evaluation.md),
[`../../reports/phase4_evaluation_claim_review.csv`](../../reports/phase4_evaluation_claim_review.csv),
and `phase4_label_review_status.json`. The prior diagnostic output remains
separate in [`../../reports/phase4_diagnostic_regression.json`](../../reports/phase4_diagnostic_regression.json)
and its human-readable companion. `phase4_generalization.jsonl` is retained as
an earlier reserved asset and is not part of the final 22-case suite.
