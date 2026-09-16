# Phase 1 evaluation and reproducibility

## Asset status

No official Theme 4 corpus, transcript replay set, relevance labels, scoring
harness, thresholds, or evaluation API was present in the supplied workspace.
The files in [`../data/evaluation/`](../data/evaluation/) are therefore a
clearly labelled local synthetic set grounded in the three-document fixture.
They are not organiser data and their results are not competition performance.

Development and held-out cases are separate JSONL files. Related paraphrases
and scenarios stay within one split. Every case embeds its provisional
transcript, expected evidence chunk IDs, answerability, relevance labels, and
an answer-support rubric. Labels are `provisional_generated` and are not
human-verified ground truth. The held-out file is run once as a fixed check; it
is not used for tuning.

The cases cover simple questions, a compound request, unsupported requests, a
cumulative correction, and formatting turns. `capabilities_deferred` records
features intentionally outside Phase 1, including early retrieval, query
decomposition, selective answer updates, external search/actions, and
streaming-format updates.

## Commands

Build the fixture index once, then run each split explicitly:

```bash
uv sync --locked --python 3.11
uv run flowcontext build-index \
  --input data/synthetic/documents.jsonl \
  --output artifacts/fixture-lexical-index.json \
  --backend lexical \
  --source-kind synthetic_fixture

uv run flowcontext evaluate-suite \
  --cases data/evaluation/development.jsonl \
  --split development \
  --corpus artifacts/fixture-lexical-index.json \
  --backend lexical \
  --execution-mode accelerated \
  --output artifacts/evaluation-development.json

uv run flowcontext evaluate-suite \
  --cases data/evaluation/held_out.jsonl \
  --split held_out \
  --corpus artifacts/fixture-lexical-index.json \
  --backend lexical \
  --execution-mode accelerated \
  --output artifacts/evaluation-held_out.json
```

`--split all` can be used without `--cases` to load the configured development
and held-out paths. The suite constructs one retriever and one provider in the
CLI process and reuses them across cases. Reports call this condition a cold
process with a warm/reused index and provider; process startup is excluded from
per-case latency.

## Metrics and interpretation

The report records separate `mock`, `fixture`, and `real_model` sections. The
mock and fixture sections can refer to the same measured local execution, but
the duplication is labelled so provider and corpus provenance are not confused.
The real-model section is `not_verified` until a real provider is actually
configured and completes the suite.

The measured metrics are limited to what the local data and traces support:

* Retrieval Recall@1, @3, @5 and MRR use only cases with explicit
  `relevance_labels`. Recall is the fraction of labelled relevant chunk IDs in
  the first *k* ranked hits. MRR is the reciprocal rank of the first relevant
  hit. These local labels are provisional.
* Citation-ID validity checks that every answer citation is a known chunk and
  was supplied in that answer's retrieved hit set. It does not establish that
  a claim is entailed by the cited text.
* Claim support is reported as `not_evaluated`. The review procedure is for a
  human reviewer to read each claim with its cited chunk(s) and label it
  supported, unsupported, or uncertain. No model judge is used, so there is no
  judge model, rubric score, or model-judge limitation to report.
* Abstention rates are split by answerable and unanswerable cases. Provider
  errors remain errors and are not silently counted as successful abstention.
* Retrieval and complete-answer latency include sample counts, p50, and p95.
  Percentiles use inclusive linear interpolation. Non-streaming generation is
  complete-answer latency, never time-to-first-token. Accelerated timings are
  not real-time evidence.
* Usage is summed from trace records. Mock usage is estimated. Cost is
  `unavailable` unless real usage and both configured per-million-token prices
  are present; it is never represented as zero when unavailable.
* Trace completeness checks for replay start, event receipt, finalisation,
  retrieval start/end, generation start/outcome, answer outcome, and replay
  completion. Early retrieval is recorded as absent by design.

The JSON report also records code revision when available, configuration,
corpus/index identity, model identities and revisions from the index manifest,
execution mode, warm/cold condition, Python/environment metadata, hardware
metadata, errors, and cost availability. Secrets and endpoint values are not
written.

## Real-provider verification

The real provider is OpenAI-compatible and reads the API key only from the
environment variable named by `FLOWCONTEXT_GENERATION_API_KEY_ENV`. For a
controlled integration run, use a private environment and a real index:

```bash
export FLOWCONTEXT_GENERATION_BACKEND=openai_compatible
export FLOWCONTEXT_GENERATION_PROVIDER=<provider-name>
export FLOWCONTEXT_GENERATION_MODEL=<model-name>
export FLOWCONTEXT_GENERATION_BASE_URL=https://<provider-endpoint>/v1
export FLOWCONTEXT_GENERATION_API_KEY_ENV=FLOWCONTEXT_GENERATION_API_KEY
export FLOWCONTEXT_GENERATION_API_KEY=<secret-in-process-environment-only>
uv run flowcontext evaluate-suite \
  --cases data/evaluation/development.jsonl \
  --split development \
  --corpus artifacts/fixture-lexical-index.json \
  --backend lexical \
  --execution-mode accelerated
```

Do not put the key in `.env.example`, reports, manifests, traces, or source
control. This workspace did not run that command because credentials and model
access were unavailable.

## Phase 2 controller and scheduler measurements

Phase 2 is evaluated separately from the Phase 1 answer baseline. A streaming
replay result records scheduled requests, actual retrieval attempts, attempts
before the final event, final-event attempts, duplicate-query suppressions,
stale results, superseded/cancelled/timed-out requests, and operationally
unnecessary retrievals. The last value counts only retrieval work whose query
revision was superseded; it is not a semantic answer-usefulness judgment. A
suppressed duplicate or coalesced pending request is not an executed search and
is reported separately.

The controller also records every WAIT/RETRIEVE/SKIP decision, decision reason,
reconstructed query revision, source timestamp, monotonic decision time, and
coalesced event IDs. Scheduler traces distinguish request scheduling, actual
worker start, completion, cancellation request/confirmation, supersession,
timeout, and stale-result rejection. A Phase 2 smoke/evaluation run must
separately report whether any retrieval occurred before finalisation. It must
not report an accelerated source-time gap or executor duration as real-time
latency or as an answer-quality gain. The final-only answer path reports
identifier validity separately from semantic claim support and passes only
current final-query evidence to generation.

Correctness tests use controllable simulated delays and are reported separately
from real-backend performance. For real retrieval measurements, report the
selected backend/model, index identity, hardware, warm/cold condition, sample
count, and p50/p95 only when enough actual samples exist. Cancellation-resistant
threads and remote requests may continue after a timeout; their late results
must remain rejected and must not be counted as current evidence. Accelerated
replay timing is never real-time latency evidence.

The controller policy is tuned only against development examples. Held-out
cases, when used, are a fixed measurement set and are not used to select
markers, thresholds, or debounce settings. With no official event schema,
corpus, or benchmark in this workspace, all Phase 2 measurements remain local
engineering evidence on the synthetic fixture. No real dense or real
generation run was available for this scheduler phase.

## Dedicated Phase 2 streaming evaluation suite

The Phase 2 suite is separate from the legacy Phase 1 answer suite. Its
machine-readable output is `flowcontext.streaming-evaluation.v1` and always
contains distinct `fixture`, `simulated_delay`, `real_backend`, and
`official_assets` sections. The local fixture section uses the synthetic
corpus, an explicitly selected lexical backend, the configured top-k, and the
configured generation provider. Simulated-delay cases wrap that same backend
with bounded delay, failure, timeout, or session-closure behavior. Real
backend and official-asset sections remain `not_verified` when their assets or
dependencies are absent; they never silently use the mock provider.

The case files are `data/evaluation/streaming_development.jsonl` and
`streaming_held_out.jsonl`. Related variants and paraphrases are kept within
one split. Development cases exercise policy behavior; held-out cases are
frozen before measurement and are not used to tune thresholds or markers.
Each record carries provisional generated labels, an eligibility decision made
before controller execution, the earliest reasonable non-final retrieval event
and source time, expected final evidence, and expected no-retrieval behavior.
No human verification is claimed.

The suite runs baseline and streaming for every case under matched corpus,
backend, top-k, generation, and replay-mode settings. It records actual early
retrieval rate on eligible cases; accepted evidence ready before final-event
delivery; exact early-result reuse; false triggers on no-retrieval turns;
premature triggers on ineligible turns; request/call/supersession/timeout/
error/usage/cost counts; controller overhead; final-query evidence
Recall@1/@3/@5 and MRR from provisional chunk-ID labels; and final-event-
delivery-to-answer latency/full interaction duration with sample count and
replay mode.

Early retrieval is true only when an observed
`streaming_retrieval_started` event occurs before `final_event_delivered`.
Decision time and queueing do not qualify. Valid pre-final evidence must be
accepted, non-stale, non-empty evidence for the final canonical query. Stale
result acceptance is an invariant and must remain zero.

The guide's 80% early-retrieval value is recorded as a local guide target. The
report says whether this measured set reaches it, but makes no official
benchmark claim. The guide does not define a numerical false-trigger
threshold, so the report intentionally reports counts without inventing one.
The focused `--compare-disabled-scheduling` run disables duplicate suppression
and source-time coalescing on a marked case, keeps scheduler bounds, and
compares calls, supersession/stale discards, and final evidence IDs. It is a
waste-policy experiment, not a semantic quality claim.

Run the suite with:

```bash
uv run flowcontext evaluate-streaming --split development \
  --corpus artifacts/fixture-lexical-index.json --backend lexical \
  --execution-mode realtime --compare-disabled-scheduling \
  --output artifacts/phase2-streaming-development.json
uv run flowcontext evaluate-streaming --split held_out \
  --corpus artifacts/fixture-lexical-index.json --backend lexical \
  --execution-mode realtime \
  --output artifacts/phase2-streaming-held-out.json
```

The same commands with `--execution-mode accelerated` validate scheduling and
reordering behavior. Accelerated wall-clock values are not model-performance
latencies and are not compared with realtime values. A real dense or provider
run remains an external validation blocker until its package, model,
credentials, and access are available.

## Phase 3 comparison and multi-intent evaluation

Phase 3 uses `evaluate-phase3` to write a new comparison report while leaving
the historical Phase 2 JSON/Markdown reports untouched:

```bash
uv run flowcontext evaluate-phase3 --split all \
  --corpus artifacts/fixture-lexical-index.json \
  --backend lexical --top-k 5 --execution-mode realtime \
  --output reports/phase3_comparison.json
```

The report separates development from held-out cases and exposes a row for
each matched Phase 2 case. Each mode records its final query, request
statuses/queries, retrieved IDs, relevant IDs, reuse decision and validation,
errors, and scoring denominator. Retrieval quality is calculated only on
successful comparable cases with relevance labels. End-to-end counts retain
every failed, timed-out, closed, or abstained case; an expected abstention is
not silently treated as a successful factual answer.

The Phase 3 section measures structural decomposition, per-intent evidence
coverage, fused IDs/provenance, citation-ID validity, expected answer behavior,
and the same successful/every-case separation. The reuse check is a
conservative lexical proxy and reports semantic support as `not_evaluated`;
matching chunk IDs do not prove entailment. Held-out cases are frozen
measurements after development implementation decisions, not a tuning set.
Real dense/provider-backed retrieval and generation, official assets, and
human semantic grounding review remain `not_verified`.
