# Phase 1 checklist

## Assets and boundaries

- [x] Inventory supplied files, dependencies, guide, and available official assets.
- [x] Distinguish the competition guide, actual corpus, evaluation assets, and synthetic fixture.
- [x] Report the missing official corpus, replay set, benchmark, scoring harness, and organiser schema/API.
- [x] Keep the guide PDF outside the knowledge index; only inspected corpus formats are loaded.

## Contracts and configuration

- [x] Define validated `flowcontext.phase1.v1` Pydantic contracts for transcript events, documents/chunks, retrieval hits, answers, and execution traces.
- [x] Record source versions, content hashes, page/section provenance, chunking, embedding metadata, and index fingerprints.
- [x] Add validated configuration, a credential-free `.env.example`, and a Python 3.11 lockfile.

## Ingestion and indexing

- [x] Implement the required strict JSONL document loader only.
- [x] Reject missing, malformed, duplicate, unsupported, and empty/blank sources with useful errors.
- [x] Preserve titles, section hierarchy, page ranges, source locations, and source versions.
- [x] Produce deterministic document-derived corpus IDs and ordinal chunk IDs.
- [x] Implement readable paragraph-first word-window chunks with configurable size and overlap, without empty chunks.
- [x] Write a manifest containing source versions/fingerprint, chunking settings, embedding configuration, counts, and build fingerprint.
- [x] Support reproducible loads and atomic writes; reject stale source/config reuse unless `--force` is explicit.
- [x] Keep transcripts, evaluation cases, and development queries out of the strict corpus index.

## Retrieval

- [x] Add a replaceable CPU dense embedding interface using the pinned Sentence Transformers configuration.
- [x] Keep Phase 1 application defaults dense-only top-k.
- [x] Add lexical retrieval as a separately selected diagnostic backend; no dense-to-lexical fallback.
- [x] Add a local hash-embedding mock only for plumbing tests, labelled separately from real dense retrieval.
- [x] Return ranked snippets with exact chunk IDs and source locations.
- [ ] Run a real dense model smoke test: blocked in the current environment because the optional package and model weights are absent.

## End-to-end grounded answer baseline

- [x] Wait for the final transcript event and use the complete utterance as one retrieval query.
- [x] Add a replaceable, environment-configured generation provider interface.
- [x] Add bounded generation timeout, retry, and structured-output repair handling.
- [x] Pass only retrieved corpus passages with their real chunk IDs to generation.
- [x] Treat passages as untrusted quoted evidence; no passage text is executed and no web/tool search is configured.
- [x] Validate structured answers, required uncertainty, and citations against the supplied hit set.
- [x] Abstain clearly for missing evidence, provider failures, malformed output, and unknown citations.
- [x] Distinguish citation-ID validity from semantic claim support; semantic support remains unevaluated.
- [x] Trace retrieval/generation durations, model identities, usage, errors, attempts, and unavailable cost.
- [x] Add an offline mock generation provider and failure-mode tests.
- [ ] Run a real end-to-end generation answer: unverified because credentials/provider access are unavailable.

## Existing Phase 1 baseline and verification

- [x] Preserve the complete-utterance replay baseline and final-marker wait behavior.
- [x] Preserve asynchronous transcript replay and structured execution traces.
- [x] Add provisional timestamped JSONL replay with explicit incremental/cumulative reconstruction.
- [x] Support realtime source-time replay and accelerated functional replay with separate monotonic timing.
- [x] Define duplicate event handling, including idempotent repeated finals and clear ordering failures.
- [x] Isolate each replay run to one session/utterance and verify concurrent invocation isolation.
- [x] Emit companion JSONL traces and a secret-free run manifest with code/config/index/model/environment metadata.
- [x] Report non-streaming complete-answer latency without time-to-first-token labelling.
- [x] Provide fixture-labelled evaluation without competition-performance claims.
- [x] Verify repeated builds preserve index/chunk IDs and report `up_to_date`.
- [x] Verify changed source and changed chunk configuration reject stale index reuse.
- [x] Verify fixture evidence, hit-to-source resolution, and metadata provenance.
- [x] Verify unsupported and empty inputs fail loudly.
- [x] Run offline compile, dependency, unit-test, smoke, and CLI checks.

## Evaluation and reproducibility workflow

- [x] Add separate local development and held-out evaluation JSONL files with embedded transcripts.
- [x] Cover simple, compound, unsupported, cumulative-correction, and formatting turns.
- [x] Keep related scenarios in one split and document that held-out cases are not tuning data.
- [x] Record expected evidence IDs, answerability, support rubrics, and provisional-label status per case.
- [x] Measure labelled Recall@k/MRR, citation-ID validity, abstention behavior, latency samples/p50/p95, usage, errors, and trace completeness only where supported.
- [x] Separate mock, fixture, and real-model report sections; keep semantic claim support unevaluated pending review.
- [x] Record settings, model identities, index identity, execution mode, warm/cold condition, hardware, environment, and code revision when available.
- [x] Add reproducible `evaluate-suite`, report writing, documented human claim-support review procedure, and Phase 2 handoff.
- [x] Produce the checked-in measured baseline report and machine-readable companion.
- [x] Add the short Phase 1 architecture document and explicit Phase 2 interface handoff.

## Packaging and clean setup

- [x] Document install, inspect, ingest/index, retrieve, answer, replay, evaluate, model download, credential, and data-mount commands.
- [x] Add a container build/run path that excludes secrets, restricted data, model weights, and generated artifacts.
- [x] Verify lockfile consistency, offline installation/smoke behavior, and the available clean-environment path.
- [x] Verify the container build/run: Docker image build and container smoke passed; dense extra remains intentionally absent.
- [ ] Verify real dense retrieval and real-provider generation: blocked/unverified because optional weights, credentials, and model access are unavailable.

## Phase 2 controller continuation

- [x] Consume shared incremental/cumulative transcript events and preserve one-session/utterance isolation.
- [x] Emit revisioned WAIT/RETRIEVE/SKIP decisions with reason codes, proposed queries, source timestamps, and monotonic decision times.
- [x] Trigger direct early retrieval for meaningful partial requests; keep Phase 1 final-only baseline as the default selectable replay mode.
- [x] Handle incomplete phrases, meaningful topic content, corrections, constraint changes, repeated fragments, cumulative revisions, and formatting turns with/without same-session context.
- [x] Apply configurable source-time debounce/coalescing and duplicate-query suppression; final decisions bypass debounce.
- [x] Expose revision-bound retrieval requests and stale-result rejection; count early, suppressed, stale, and operationally unnecessary retrieval work.
- [x] Connect retrieval to a bounded async executor scheduler without blocking transcript event handling.
- [x] Bound concurrency, pending/coalesced requests, timeouts, retries, and total request allocation; keep cancellation confirmation explicit.
- [x] Preserve useful in-flight work for non-material query updates and reject superseded/cancellation-resistant late results by retrieval revision.
- [x] Trace scheduling, actual start, completion/failure, supersession, cancellation request/confirmation, timeout, stale rejection, and final-only answer execution.
- [x] Generate only after finalisation using current final-query evidence; preserve the Phase 1 provider/citation validator and non-streaming latency label.
- [x] Document the async scheduler boundary, conservative final-evidence reuse, simulated-delay correctness tests, and real-backend limitations.

## Deliberately deferred

- [ ] Official corpus ingestion and official benchmark execution, pending organiser assets.
- [ ] Query decomposition and multi-intent parallel retrieval.
- [ ] Hybrid fusion and reranking comparisons.
- [ ] Session-only selective answer refinement.
- [x] Asynchronous retrieval scheduler, bounded task ownership/cancellation, and simulated overlap/cancellation correctness tests.
- [ ] Answer generation from early evidence, selective answer refinement, and first-token/partial-answer measurements.
- [x] Implement the provider boundary for model-backed synthesis and usage/cost reporting.
- [ ] Execute a real provider-backed answer and validate documented pricing before reporting cost.
- [ ] Production deployment packaging beyond the verified minimal container path.
