# AI assistance log

## Available prompt records

- Two user requests were available in this session on 2026-09-15. The first
  requested the Phase 1 foundation for Samsung PRISM Theme 4, including guide
  and asset inspection, synthetic-fixture labelling, contracts/configuration,
  replay/evaluation/smoke checks, and evidence-based hand-off.
- The continuation request asked for corpus ingestion, deterministic indexing,
  dense and lexical baseline retrieval, safe rebuilds, CLI commands, and the
  corresponding verification/documentation, while explicitly deferring
  streaming decisions, query decomposition, and selective answer updates.
- A third continuation request was available in this session on 2026-09-15. It
  asked to use the existing index/contracts/retriever for a complete-utterance
  grounded answer baseline, add replaceable generation with bounded retries,
  validate citations, handle safety/failure modes, and report real-provider
  execution only when actually available.
- A fourth continuation request was available in this session on 2026-09-15.
  It asked to connect that baseline to timestamped transcript replay, support
  realtime and accelerated modes, define duplicate/final-event behavior,
  isolate concurrent sessions, emit JSONL traces, and save a secret-free run
  manifest without implementing the Phase 2 streaming controller.
- No prior prompt transcript, organiser messages, repository history, or hidden
  benchmark instructions were available in the workspace. None are inferred.

## Work performed

- Inventoried the workspace with `rg --files`, `find`, `ls`, `file`, `sha256sum`,
  and a read-only Git check. The directory contained only the guide PDF, its
  zone metadata, and empty metadata directories; it was not a Git worktree.
- Inspected the six-page scanned guide by rendering it with PyMuPDF and visually
  reviewing the rendered pages. Confirmed that the guide contains no official
  corpus, replay assets, scoring harness, or machine-readable schema.
- Checked Python runtimes and installed modules. Selected the available Python
  3.11.15 executable and pinned Pydantic 2.13.4 plus its runtime dependencies.
- Added the Phase 1 package, Pydantic contracts, config loader, deterministic
  corpus ingestion/chunking, lexical retriever, extractive no-model answerer,
  async replay, structured tracing, local evaluation, CLI, synthetic fixture,
  documentation, and tests.

## Continuation work performed

- Read the current implementation and checklist before changing files.
- Added strict JSONL corpus loading, source inspection, paragraph-first
  deterministic chunking with configurable overlap, provenance propagation, and
  empty/unsupported-input errors.
- Added deterministic index manifests and fingerprints, content-addressed source
  versions, atomic index writes, safe `up_to_date` reuse, stale source/config
  rejection, and explicit `--force` rebuilds.
- Added a replaceable embedding provider protocol, CPU Sentence Transformers
  provider pinned to `all-MiniLM-L6-v2` revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, explicit lexical retrieval, and a
  fixture-only hash-embedding mock. The model metadata was checked against the
  official Hugging Face model card/revision pages; weights were not downloaded.
- Added `inspect-corpus`, `build-index`, and `retrieve` CLI paths while retaining
  the existing replay/evaluation commands. The selected retrieval backend is
  emitted in manifests, responses, and replay traces.
- Tools used: local shell inspection and execution, `apply_patch` for edits,
  the Python 3.11 environment, `uv` for dependency locking/sync/checking, and
  official web documentation lookup for dense-model metadata. No agent framework,
  frontend, microphone input, or training was added; real model-backed answer
  synthesis was not part of that initial pass.

## Generation continuation work performed

- Re-read the current contracts, replay, retriever, checklist, and README before
  changing the existing baseline.
- Added `GenerationConfig`, quoted `EvidencePassage`/`GenerationRequest`
  contracts, provider results, generation status/usage/cost fields, and explicit
  distinction between citation-ID validity and unevaluated semantic support.
- Added an async generation-provider protocol with an offline deterministic mock
  and a minimal OpenAI-compatible JSON provider. The real provider reads its API
  key only from the named environment variable; no secret is persisted or
  included in trace/error messages. Request prompts contain no web-search/tool
  configuration and explicitly mark retrieved passages as untrusted data.
- Added bounded per-attempt timeouts, exponential retry limits, one bounded
  structured-output repair path, unknown-citation rejection, and explicit
  abstention for invalid output, provider failure, and missing evidence.
- Wired generation into the existing final-event-only replay path. Retrieval
  remains one complete-utterance query; no early retrieval, decomposition, or
  selective follow-up was added. Replay traces now include generation duration,
  provider/model identity, attempts, usage, cost, and errors.
- Updated evaluation output to report citation-ID validity separately from
  semantic support, and updated the README, checklist, synthetic-fixture note,
  and schema notes.

## Timestamped replay continuation work performed

- Re-read the current replay implementation, contracts, CLI, tests, README,
  and checklist before changing the existing architecture.
- Added explicit incremental append versus cumulative replacement assembly;
  cumulative snapshots are not concatenated.
- Added source-time `realtime` replay and no-wait `accelerated` replay. Source
  timestamps remain in trace fields while monotonic execution times and
  measured durations are recorded separately.
- Added duplicate-safe validation: identical repeated event IDs are ignored,
  identical repeated final text/mode is accepted and ignored, and conflicting,
  out-of-order, post-final, or divergent events fail clearly.
- Triggered the existing retrieval/generation path immediately after the first
  valid final event and never more than once. Added finalisation, generation
  start, answer completion/failure, citation/version, and non-streaming
  `complete_answer_latency_ms` trace events. No first-answer-content event is
  emitted because the current provider interface is non-streaming.
- Added companion JSONL trace writing and `flowcontext.replay.manifest.v1`
  manifests containing code revision when available, safe configuration,
  corpus/index IDs, retrieval/generation identities, execution mode, timing,
  and environment details. API key values and endpoint URLs are excluded.
- Added separate provisional replay examples for incremental, cumulative, and
  repeated-final input under `examples/replay/`.
- Added tests for no pre-final retrieval, equivalent transcript reconstruction,
  realtime versus accelerated timing, concurrent session isolation, duplicate
  final idempotence, malformed sequence handling, provider failure status,
  required traces, and manifest/JSONL output.

## Verification commands actually run

- `PYTHONPATH=src python3 -m compileall -q src tests`: pass under the available
  Python 3.10.16 environment.
- `PYTHONPATH=src python3 -m unittest discover -s tests -v`: 4 tests passed under
  the available Python 3.10.16/Pydantic 2.13.4 environment.
- `UV_CACHE_DIR=/tmp/flowcontext-uv-cache uv lock`: resolved the six-package
  Python 3.11-compatible lock; the first sandbox attempt was blocked by the
  managed cache/network policy, then the approved elevated retry succeeded.
- `UV_CACHE_DIR=/tmp/flowcontext-uv-cache uv sync --locked --python /home/masa84/.local/bin/python3.11`
  and its offline recheck: installed/checked Python 3.11.15 with the pinned
  runtime dependencies.
- `.venv/bin/python -m unittest discover -s tests -v`,
  `.venv/bin/python -m compileall -q src tests`, and
  `.venv/bin/python -m flowcontext.cli smoke`: all passed under Python 3.11.15.
- After the continuation changes, `.venv/bin/python -m unittest discover -s
  tests -v` passed 9 tests, compileall passed, and the offline `smoke` command
  passed all fixture/contract checks.
- `UV_CACHE_DIR=/tmp/flowcontext-uv-cache uv lock` resolved 68 packages,
  including the declared optional dense extra; `uv lock --check --offline`
  passed afterward. The attempted newer `uv lock --all-extras` spelling was
  rejected by this installed uv version and made no change.
- `UV_CACHE_DIR=/tmp/flowcontext-uv-cache uv sync --locked --python 3.11` and
  `uv pip check --python .venv/bin/python --offline` passed.
- The CLI sequence `inspect-corpus`, two `build-index` calls, `retrieve`,
  `replay`, `evaluate`, and separate mock build/retrieve ran on
  `data/synthetic/`: inspection/build counts were 3 documents and 4 chunks;
  the second build reported `up_to_date`; replay produced one final-only
  retrieval and 13 trace events; the fixture evaluation passed. Build times
  reported by the CLI were approximately 0.00049–0.00069 seconds for this
  three-document fixture.
- `.venv/bin/python -m flowcontext.cli dense-smoke` returned exit code 3 and
  `BLOCKED`: `sentence-transformers` is not installed and the pinned model was
  not available locally. No real dense retrieval or dense quality score was
  claimed.
- CLI checks against the guide PDF and an incompatible dense request returned
  useful exit-code-2 errors for unsupported format and wrong backend; unit tests
  separately covered empty, blank, malformed, duplicate, and stale inputs.
- The real CLI sequence `ingest`, `replay`, and `evaluate` was run against the
  synthetic JSONL assets, writing temporary `/tmp/flowcontext-phase1-*` outputs.
  It passed with 3 documents, 4 chunks, one final-only retrieval, 13 trace
  events, and a fixture-only evaluation pass.
- `UV_CACHE_DIR=/tmp/flowcontext-uv-cache uv run --offline flowcontext smoke` and
  `.venv/bin/python -m flowcontext --help`: both passed.
- `.venv/bin/python -m pip check` was attempted but the uv-managed environment
  intentionally has no pip module; the equivalent
  `UV_CACHE_DIR=/tmp/flowcontext-uv-cache uv pip check --python .venv/bin/python --offline`
  passed.
- Final hardening added bounded overlap handling for small chunks, blank-text
  chunk errors, UTF-8 input errors, metadata/source-kind coverage in the source
  fingerprint, exact embedding-config matching at retrieval, and explicit force
  rebuild coverage. The final unit run passed 9 tests; the final CLI run again
  reported 3 documents/4 chunks, lexical build times of 0.000589 and 0.000831
  seconds, mock build time of 0.000837 seconds, `up_to_date` reuse, resolved
  snippets, and a passing fixture evaluation.
- Final `UV_CACHE_DIR=/tmp/flowcontext-uv-cache uv run --offline flowcontext smoke`
  passed. Final `dense-smoke` returned exit code 3 with the same explicit
  missing-optional-dependency blocker. `python -m flowcontext --help` passed.
- Generation failure-mode tests covered no evidence, unknown citation IDs,
  malformed JSON, timeout/retry exhaustion, instruction-like evidence, and a
  valid traceable mock answer. The real generation integration was not run:
  the configured provider credentials/model access were unavailable.

## Final generation continuation verification

- Hardened bounded repair feedback so provider-controlled citation text is not
  echoed into a subsequent provider request, and added the selected retrieval
  provider/model identity to retrieval trace events.
- Changed a provider response with zero factual claims into the explicit
  uncertainty/abstention answer, rather than exposing an unverified answer text.
- Added a bounded oversized-token chunking check so a configured maximum is
  enforced without emitting empty chunks.
- `.venv/bin/python -m compileall -q src tests`: passed.
- `.venv/bin/python -m unittest discover -s tests -v`: 16 tests passed.
- `.venv/bin/python -m flowcontext.cli smoke`: passed. The report identified
  lexical retrieval and mock generation, validated fixture citations, and made
  no competition-performance claim.
- In a fresh `/tmp/flowcontext-phase1-final.*` directory, ran
  `config-check`, `inspect-corpus`, two lexical `build-index` calls,
  `retrieve --source`, `replay --source`, and `evaluate`. Inspection and both
  index outputs reported 3 documents and 4 chunks; indexing took 0.000493 s
  and 0.000453 s, with the second build reporting `up_to_date`. Retrieval
  returned provenance-resolving chunk IDs. Replay used one final-event query,
  produced 13 trace events, and reported mock generation success with 410
  estimated tokens and cost `unavailable`. The synthetic evaluation passed
  with citation-ID validity 1.0 and semantic-support evaluation 0.0; this is a
  fixture/contract result, not retrieval-quality or competition performance.
- `.venv/bin/python -m flowcontext.cli dense-smoke`: exit 3, explicitly
  `BLOCKED` because `sentence-transformers` and the pinned model are not
  available locally. `config-check` also reported no configured generation API
  key. No real dense retrieval or real provider-backed answer was executed.
- A replay with `FLOWCONTEXT_GENERATION_BACKEND=openai_compatible` and a
  deliberately absent `FLOWCONTEXT_MISSING_KEY` completed without a network
  call, returned `generation_status=abstained`, recorded one bounded attempt,
  and logged only the environment-variable name plus `cost=unavailable`.
- `.venv/bin/python -m flowcontext --help`, `uv lock --check --offline`, and
  `uv pip check --python .venv/bin/python --offline`: passed.
- Repeated the complete CLI sequence after the final chunker change in a second
  fresh `/tmp/flowcontext-phase1-final2.*` directory: 3 documents/4 chunks,
  lexical build times 0.001227 s and 0.000735 s, `up_to_date` reuse, resolved
  retrieval snippets, mock replay success with 13 traces, and a passing
  synthetic evaluation.
- `UV_CACHE_DIR=/tmp/flowcontext-uv-cache uv run --offline flowcontext smoke`:
  passed after the final code change.
- After the timestamped replay change, `.venv/bin/python -m compileall -q src
  tests` passed and `.venv/bin/python -m unittest discover -s tests -v` passed
  23 tests. `.venv/bin/python -m flowcontext.cli smoke` and
  `.venv/bin/python -m flowcontext.cli replay --help` passed.
- In a fresh `/tmp/flowcontext-phase1-replay-final.*` directory, the CLI built
  the fixture index with 3 documents/4 chunks in 0.001527 s, replayed the
  synthetic transcript in accelerated mode in 1.441 ms, and replayed it in
  realtime mode in 2106.820 ms for 2.1 s of source time. Both produced 16
  trace events and mock generation success; the accelerated run wrote a
  JSONL trace and manifest and the fixture evaluation passed. These timing
  values are engineering observations, not real-time model latency claims.
- The three separate example transcripts were replayed successfully. The
  incremental and cumulative examples both reconstructed `Which venue in
  Pune?`; the duplicate-final example reported one ignored duplicate and one
  baseline execution.
- Added explicit `run_status` to replay results/manifests and made the CLI
  report `completed`, `abstained`, or `failed`. The final accelerated replay
  reported `completed`, 16 trace events, 1.018 ms complete-answer latency, and
  1.306 ms total execution. A bounded missing-key provider run exited 1 with
  `run_status=failed`, one generation attempt, and an `answer_failed` trace.
- Removed two stale unused imports found by `ruff check src tests`; the final
  lint run passed, followed by passing compile and 23-test runs.

This log records only commands executed during this session; no unobserved
model call, organiser benchmark run, or missing prompt history is claimed.

## Evaluation and reproducibility continuation

- Re-read the current contracts, evaluator, CLI, configuration, replay runner,
  checklist, asset inventory, and tests before editing. Confirmed the official
  corpus, official evaluation set, organiser schema/API, and Git revision are
  still unavailable; preserved the existing synthetic fixture boundary.
- Extended the validated evaluation contracts with split, embedded transcript,
  answerability, expected evidence IDs, provisional relevance labels, support
  rubrics, deferred capabilities, per-case retrieval/citation/abstention/
  latency/usage/error/trace fields, latency summaries, human-review status,
  and provenance-separated suite reports.
- Added `data/evaluation/development.jsonl` and
  `data/evaluation/held_out.jsonl`, plus their README. The ten local cases
  cover simple, compound, unsupported, cumulative-correction, and formatting
  turns. They are synthetic and provisional, not official or human-verified.
- Added `evaluate-suite` and `answer` CLI commands. The suite runs each fixed
  case through the existing final-event-only replay, records early retrieval as
  absent, computes labelled Recall@1/@3/@5 and MRR, citation-ID validity,
  answerable/unanswerable abstention rates, inclusive-linear p50/p95 latency,
  usage/cost availability, errors, and trace completeness. The answer command
  can run a single complete query through the same final-event baseline or
  display an already validated replay answer without a second execution.
- Added `docs/evaluation.md`, `docs/architecture-phase1.md`, and
  `docs/phase2-handoff.md`, updated the README/checklist, and added a minimal
  `Dockerfile`/`.dockerignore` that excludes credentials, restricted data,
  model weights, and generated artifacts.
- Ran a temporary lexical fixture index plus development and held-out suite
  commands. Both suites passed with 5 cases each, 4 labelled answerable cases,
  1 expected-abstention case, 100% trace completeness, 100% citation-ID
  validity for emitted IDs, and cost `unavailable`. Development retrieval
  Recall@1/@3/@5 was 0.8333/0.9167/1.0 with MRR 1.0; held-out was 1.0/1.0/1.0
  with MRR 1.0. These are provisional fixture-label results only.
- `compileall` and the existing 23 standard-library tests passed after the
  implementation. Real dense retrieval, real-provider generation, official
  asset evaluation, and a container build remain unverified in this
  environment.

## Final reproducibility verification

- `UV_CACHE_DIR=/tmp/flowcontext-uv-cache uv lock --check --offline` and
  `uv pip check --python .venv/bin/python --offline`: passed; the lock resolved
  68 packages and the installed core environment was compatible.
- A fresh temporary environment at `/tmp/flowcontext-clean.fW7z9H/.venv` was
  created with `UV_PROJECT_ENVIRONMENT` and `uv sync --locked --offline
  --python 3.11`; it installed the package and passed `python -m
  flowcontext.cli smoke`.
- `docker build --tag flowcontext-phase1 .` passed through the Docker daemon,
  and `docker run --rm flowcontext-phase1 smoke` passed. The image contains no
  credentials, restricted corpus, model weights, or generated reports.
- The reproducible `answer --query` command ran against a temporary lexical
  index and wrote a replay result, JSONL trace, and manifest. A subsequent
  `answer --run` read the same validated answer without another retrieval.
- The checked-in `reports/phase1_baseline.json` was generated by one
  `evaluate-suite --split all` command over 5 development and 5 held-out cases.
  It measured the synthetic fixture/mock run only: 3 documents, 4 chunks,
  explicit lexical retrieval, Recall@1/@3/@5 = 0.9167/0.9583/1.0, MRR = 1.0,
  19 emitted citation IDs with validity 1.0, answerable abstention 0.0,
  unanswerable abstention 1.0, 10/10 complete traces, retrieval latency
  p50/p95 0.0221/0.0292 ms, complete-answer latency p50/p95 0.3047/0.4946 ms,
  estimated usage 1,921 tokens, and cost `unavailable`.
- The report keeps mock, fixture, and real-model sections distinct. The
  real-model section is `not_verified`; no official benchmark, human
  claim-support review, real dense model, or real provider call was executed.
- Final command-matrix verification in `/tmp/flowcontext-phase1-final-check.w0WEGb`
  passed `inspect-corpus`, first lexical build (0.000546 s), repeated build
  (`up_to_date`, 0.000473 s), source-fresh retrieval, replay (16 traces), the
  compatibility evaluation, and the all-split suite. The suite again passed
  all 10 cases with the checked-in report metrics.
- `dense-smoke --local-files-only` returned the documented exit code 3 blocker
  for the missing optional package/model. A missing-key OpenAI-compatible
  replay returned exit code 1, `run_status=failed`, one bounded attempt,
  `generation_abstained`/`answer_failed` errors, and cost `unavailable` without
  making a provider request.
- Final `compileall`, offline `ruff check`, 24-test unittest discovery,
  offline smoke, lock/check validation, report/schema validation, and the
  no-obvious-credential-value scan passed.
- After pinning the base-image digest and recording the index embedding
  configuration in the report, the report was regenerated and the final
  compile/lint/24-test/smoke/report-integrity pass succeeded again. The final
  digest-pinned Docker build and container smoke also passed.

## Phase 2 streaming-controller continuation

- Re-read the current Phase 2 handoff, contracts, replay runner, retriever,
  settings, README, checklist, evaluation methodology, and tests before
  editing. Confirmed the organiser corpus/event schema/benchmark remain
  unavailable and retained the synthetic/lexical/mock labels.
- Added Phase 2 contracts for controller configuration, session-scoped prior
  answer context, revisioned WAIT/RETRIEVE/SKIP decisions, revision-bound
  retrieval requests/results, streaming replay results, and secret-free
  manifests.
- Added an explainable controller using topic/request signals, incomplete
  phrase detection, correction markers, constraint signatures, source-time
  debounce/coalescing, duplicate-query suppression, final-decision bypass, and
  same-session formatting context. No model call is made per transcript chunk.
- Added direct early retrieval replay on top of the existing normalizer and
  retriever. It records decision/retrieval traces, exact hit IDs and source
  locations, stale-result rejection, backend selection, and operational early,
  suppressed, stale, and unnecessary-retrieval counts. It deliberately does
  not implement the asynchronous scheduler, generation, decomposition, or
  selective answer refinement.
- Added replay --mode streaming while preserving the default Phase 1 baseline,
  provisional streaming examples, and docs/phase2-streaming.md. The controller
  has no background timer; source-time coalescing is event-driven and the next
  scheduler owns timer fire/cancel behavior.
- Added tests for incomplete and meaningful partials, incremental/cumulative
  equivalence, repeated fragments/finals, corrections, debounce, formatting
  context, stale results, session isolation, invalid ordering, configurable
  settings, source provenance, and trace completeness. Initial CLI smoke on the
  synthetic lexical index showed one early and one final retrieval for the
  early-retrieval example; real dense/provider execution remains unverified.
- Recorded the final local controller smoke in
  `reports/phase2_controller_smoke.md` and its JSON companion. The measured
  CLI matrix built the lexical index (`built`, then `up_to_date`), ran the
  early-retrieval and correction examples with one early and one final attempt
  each, returned `needs_context` without retrieval for formatting-only input,
  and separately confirmed the Phase 1 baseline remains final-only. The final
  focused suite contains 37 tests after adding the explicit constraint-change
  debounce case.

## Asynchronous retrieval scheduler continuation

- Re-read the current Phase 2 controller, handoff, contracts, replay runner,
  retriever, settings, README, checklist, evaluation notes, and tests before
  editing. Retained the synthetic/lexical/mock labels and the Phase 1 default
  baseline; official assets, real dense retrieval, and real generation remain
  unverified.
- Added `StreamingSchedulerConfig`, session/utterance retrieval revisions,
  request lifecycle fields, scheduler counters, and final-only streaming
  answer metadata to the local Pydantic contracts. Extended environment and
  CLI configuration for concurrency, pending/coalescing, timeout, retry,
  final-wait, cancellation, and request limits.
- Added `AsyncRetrievalScheduler` using a bounded `ThreadPoolExecutor` around
  the existing synchronous `Retriever.search` interface. It keeps event
  processing on the asyncio loop, coalesces a latest pending query, preserves
  same-query in-flight work, marks material revisions superseded immediately,
  attempts cancellation without overstating confirmation, and rejects stale or
  post-closure results by revision/session checks.
- Wired `replay --mode streaming` to schedule early retrieval, conservatively
  reuse/await final-query evidence, and call the existing structured generation
  provider only after finalisation. No early answer generation, decomposition,
  fusion/reranking, or selective answer update was added. Streaming traces now
  include scheduling, actual starts, completion/failure, cancellation,
  supersession, timeout, stale rejection, final-only generation, usage, cost
  availability, and answer version/citations. Non-streaming generation is
  reported with complete-answer latency, never first-token latency.
- Added 12 controllable-delay scheduler tests for event-loop responsiveness,
  same-query reuse, old/new races, cancellation-resistant late results, rapid
  corrections, finalisation during retrieval, failure/timeout, concurrent
  identical sessions, session closure, final-only evidence, query-cycle stale
  reuse, and trace/revision lifecycle. Updated the prior correction test to expect superseded early
  evidence under the new correctness policy.
- Updated README, schema notes, Phase 2 streaming/handoff/evaluation docs,
  checklist, examples guidance, and added the measured
  `reports/phase2_scheduler_smoke.{md,json}`. The old controller smoke report
  is explicitly marked historical.
- Tools used: local shell inspection and execution, `apply_patch`, Python 3.11,
  `uv run --offline` for lint/tests/smoke, and the existing CLI against the
  synthetic lexical index. Verification passed: compileall, Ruff, 49 unit
  tests, offline smoke, and accelerated command-level build/replay checks.
  No real dense or real-provider performance command was executed because the
  optional model/package and credentials remain unavailable.

## Phase 2 evaluation and implementation audit

- Re-read the existing evaluation methodology and kept fixture,
  simulated-delay, real-backend, and official-asset results in separate report
  sections. Added frozen `streaming_development.jsonl` and
  `streaming_held_out.jsonl` suites with provisional generated labels,
  pre-controller eligibility, earliest reasonable retrieval points, expected
  final evidence, and the requested correction, repetition, formatting,
  failure, timeout, closure, and concurrency coverage.
- Added the machine-readable `flowcontext.streaming-evaluation.v1` evaluator
  and `evaluate-streaming` CLI command. It runs matched baseline/streaming
  conditions, labels realtime versus accelerated execution, records trace
  coverage and timing samples, and reports early starts only when retrieval
  actually starts before final-event delivery. It does not compare accelerated
  wall-clock latency with realtime latency.
- Hardened baseline formatting-only handling and retrieval-failure reporting:
  previous-answer formatting uses context without corpus search, missing
  context returns an explicit clarification, and backend failure remains a
  surfaced abstention/error rather than a mock fallback. Added the explicit
  duplicate-suppression-disabled comparison.
- During the final audit, found that the timeout case's declared timeout was
  not being applied to the evaluator scheduler. Applied the per-case timeout,
  added a regression test, and regenerated both reports; the timeout now
  surfaces as `RetrievalTimeout` with stale acceptance still zero.
- Added `reports/phase2_streaming_evaluation.{md,json}` and the accelerated
  JSON report, `PHASE2_CHECKLIST.md`, and the Phase 3 boundary/interface note.
  No Phase 3 implementation was started.
- Realtime all-split fixture audit observed 23/23 eligible early starts,
  10/23 valid pre-final evidence results, 10/23 useful reuse, 0/6 false
  triggers, 0/2 premature triggers, 0 stale-result acceptances, 37 streaming
  calls, 25 baseline calls, and complete traces for 31/31 cases. These are
  local synthetic engineering measurements, not official benchmark results.
- Final verification ran `python -m unittest discover -s tests -v` (55 tests),
  plus
  `python -m compileall -q src tests`, `ruff check src tests`,
  `flowcontext config-check`, `flowcontext smoke`, clean offline `uv sync`
  plus config/smoke, approved network retry for the missing locked wheel,
  Docker build/run smoke, synthetic index build, and both realtime and
  accelerated `evaluate-streaming --split all` runs with
  `--compare-disabled-scheduling`. The explicit dense/local-files-only check
  remained `BLOCKED` because the optional model/package was unavailable.
- Re-ran the existing baseline build/replay/evaluation regression after the
  final source changes; the 5-case baseline suite passed with fixture Recall@5
  1.0, citation-ID validity 1.0, complete traces, and no errors. Clean locked
  setup/config/smoke and the rebuilt Docker smoke also passed.
- No secrets were added to traces or logs. Real dense/provider validation,
  official assets/labels, semantic answer-support review, and provider cost
  measurement remain unresolved; no human verification is claimed.

## Phase 3 multi-intent retrieval and Phase 2 comparison audit

- Re-read the repository README, Theme 4 guide, Phase 2 handoff, Phase 2
  controller/scheduler/retriever/generation implementation, tests, and all
  checked-in Phase 2 reports before editing. The guide PDF was image-only in
  this environment, so its rendered architecture/gates were inspected; no
  organiser schema or official assets were inferred.
- Reproduced the Phase 2 all-split realtime fixture run with the rebuilt
  lexical index. The historical 96.0% baseline versus 91.7% streaming
  Recall@5 remains preserved. The new per-case audit showed 23/23 eligible
  early starts and 10/23 useful early reuse. On the matched successful
  denominator, both modes were 100% Recall@5 over 22 cases; the historical
  gap mixed unequal successful-evidence denominators with failures/closures.
- Classified all 13 eligible non-useful early-reuse cases without using
  held-out examples for tuning: 9 expected correction/final-constraint
  changes, 1 delayed rapid-correction result not ready before final delivery,
  3 declared failure/timeout/session-close probes, and 0 avoidable failures.
  Punctuation-only query differences are compared canonically and are not
  misreported as dropped constraints.
- Added `multi_intent.py` with structural decomposition, stable intent IDs,
  bounded parallel retrieval, isolated intent errors, deterministic RRF,
  provenance-preserving chunk deduplication, and a final-query reuse
  relevance proxy. Reuse requires more than a valid chunk ID; semantic
  entailment is explicitly reported as unevaluated.
- Added opt-in `--multi-intent` integration to baseline and streaming replay.
  The streaming path requires current final-intent coverage before generation,
  rejects stale/incomplete final evidence, and turns completed no-hit or
  partial-intent retrieval into explicit uncertainty/abstention rather than a
  transport failure. Default baseline/streaming modes remain selectable.
- Added `evaluate-phase3`, `reports/phase3_comparison.json`, and its Markdown
  companion. The report separates development/held-out cases, successful
  comparable retrieval quality, and end-to-end outcomes including failures,
  timeouts, closure, and abstentions. It also records Phase 3 per-intent
  evidence and citation-ID validity without claiming semantic grounding.
  Held-out cases were measured after implementation decisions were frozen;
  new untouched cases remain reserved for later evaluation.
- Added seven focused Phase 3 regression tests for decomposition boundaries,
  RRF/provenance, valid-ID-but-unsupported reuse, partial-intent uncertainty,
  streaming revision protection, and failed-case scoring denominators.
- Commands and outcomes: the Phase 3 CLI audit completed with
  `... evaluate-phase3 --corpus artifacts/fixture-lexical-index.json
  --backend lexical --top-k 5 --split all --execution-mode realtime`;
  focused Phase 3 tests passed (6); the full suite was re-run after the final
  source changes and passed (62 tests). Compile/lint and report-integrity
  checks remain part of the final verification matrix below.
- Remaining blockers are unchanged in substance: official corpus/replay
  assets, human labels and semantic claim review, real dense retrieval, and
  provider-backed generation/cost measurement. Phase 4 selective claim
  updates across follow-up turns were intentionally not implemented.

## Phase 3 structured decomposition continuation

- Reworked the initial clause list into the revision-bound
  `DecompositionResult` contract. It preserves the exact transcript,
  character-offset spans, typed shared/intent constraints, ambiguities, and
  dependency/comparison edges. Revision matching uses query/constraint
  fingerprints so unchanged work can survive a correction; affected intents
  and dependents are superseded.
- Added `StructuredMultiIntentDecomposer` behind the configured generation
  provider boundary. The OpenAI-compatible implementation is a real JSON-only
  provider with timeout handling, schema/source-provenance validation, and at
  most the configured repair budget. `MockDecompositionProvider` and
  `decompose_query` remain explicitly labelled offline modes. Real-provider
  errors never silently select the mock; the optional original-query fallback
  is recorded with `status="fallback"` and its failure reason.
- Integrated decomposition with the existing controller/scheduler at stable
  retrieval revisions and finalisation, including cache protection against a
  duplicate provider call when finalisation races an in-flight retrieval.
  Obsolete revision decompositions are traced and cannot become current
  evidence. Final reuse still requires lexical support for each final intent;
  valid chunk IDs alone are not semantic support.
- Added varied regressions for coordinated constraints, independent and
  dependent questions, comparisons, negation, ambiguity, repeated wording,
  one-intent corrections, invalid JSON/schema, timeout/fallback, provider
  preservation validation, and the no-per-token provider-call boundary.
- Added the [multi-intent example](examples/multi_intent/README.md) and
  expanded the schema/Phase 3 notes. No selective claim updates across later
  turns were implemented; that remains Phase 4.
