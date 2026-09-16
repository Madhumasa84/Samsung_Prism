# FlowContext — Theme 4 Phase 1/2/3

Phase 1 provides corpus inspection/ingestion, deterministic indexing, a
replaceable embedding interface, dense top-k retrieval, an explicit lexical
diagnostic baseline, timestamped complete-utterance replay, corpus-grounded
structured answer generation, and local evaluation. Phase 2 adds an
explainable streaming decision controller and a bounded asynchronous early
retrieval scheduler. Phase 3 adds an opt-in multi-intent decomposer,
per-intent retrieval, deterministic evidence fusion, grounded synthesis, and
uncertainty handling. Selective answer updates across follow-up turns and early
answer generation remain out of scope.

The supplied workspace contains only the [Theme 4 guide](<Theme 4 Guide_RAG.pdf>).
The official knowledge corpus, replay prompts, labels, benchmark runner, scoring
harness, and organiser schema/API are missing. The files under
[`data/synthetic/`](data/synthetic/) are invented engineering fixtures only;
their retrieval/evaluation results are not competition performance.

## Quick start

The project targets Python 3.11. Using `uv`:

```bash
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
uv sync --locked --python 3.11
uv run flowcontext config-check
uv run flowcontext inspect-corpus
uv run flowcontext smoke
uv run python -m unittest discover -s tests -v
```

The offline `smoke` command intentionally uses the lexical fixture path and a
separate local hash-embedding mock. It does not claim to be a real dense-model
run. To try the real dense path, install the optional dependency and allow the
model snapshot to be downloaded:

```bash
uv sync --locked --python 3.11 --extra dense
uv run flowcontext dense-smoke
```

`dense-smoke` reports `BLOCKED` with exit code 3 when the optional package or
pinned model is unavailable. Set `FLOWCONTEXT_EMBEDDING_LOCAL_FILES_ONLY=true`
to require a pre-populated local model cache.

## Reproducible complete-utterance baseline

This is the Phase 1 end-to-end baseline. It builds the explicit lexical
diagnostic index, waits for the final transcript event, retrieves once using
the complete utterance, and generates with the offline mock provider:

```bash
uv run flowcontext build-index \
  --input data/synthetic/documents.jsonl \
  --output artifacts/fixture-lexical-index.json \
  --backend lexical \
  --source-kind synthetic_fixture
uv run flowcontext replay \
  --transcript data/synthetic/transcript.jsonl \
  --index artifacts/fixture-lexical-index.json \
  --source data/synthetic/documents.jsonl \
  --backend lexical \
  --execution-mode accelerated \
  --output artifacts/replay.json
```

The default generation backend is `mock` so this command is offline and
reproducible. Its answer, JSONL trace, and run manifest identify mock
execution. The mock is an engineering provider, not a model-quality result.

## Available corpus format

The only corpus-like asset supplied for this implementation is the synthetic
JSONL fixture. The loader intentionally supports `.jsonl` only: one document
object per line with `document_id`, `source_location`, `text`, optional `title`,
`source_version`, `page_start`/`page_end`, `section`, `section_hierarchy`, and
`metadata`. Empty extraction, malformed JSON, duplicate IDs, and unsupported
extensions are errors. No PDF, HTML, or arbitrary text loader is implied by the
guide or included without an actual corpus format to inspect.

Titles, section hierarchy, page ranges, source versions, and source locations
are copied into documents and chunks. Document IDs are the supplied stable
source IDs; chunk IDs are deterministic `{document_id}#chunk-{ordinal}` values.
Identical text in different source locations remains distinct because provenance
and source IDs are preserved rather than content-deduplicated.

## Inspect, build, and retrieve

Inspect source health without embeddings:

```bash
uv run flowcontext inspect-corpus --input data/synthetic/documents.jsonl
```

The default `build-index` backend is dense-only. For the available fixture, use
the explicit lexical diagnostic backend or the separate local mock backend:

```bash
uv run flowcontext build-index \
  --input data/synthetic/documents.jsonl \
  --output artifacts/fixture-lexical-index.json \
  --backend lexical \
  --source-kind synthetic_fixture

uv run flowcontext retrieve \
  --index artifacts/fixture-lexical-index.json \
  --source data/synthetic/documents.jsonl \
  --backend lexical \
  --query "Which venue in Pune accommodates 30 attendees?" \
  --top-k 2
```

For a dense index after installing the optional dependency:

```bash
uv run flowcontext build-index \
  --input data/synthetic/documents.jsonl \
  --output artifacts/fixture-dense-index.json \
  --backend dense \
  --source-kind synthetic_fixture
uv run flowcontext retrieve \
  --index artifacts/fixture-dense-index.json \
  --backend dense \
  --query "Which venue in Pune accommodates 30 attendees?"
```

Every hit includes the exact chunk ID, rank, score, retrieval method, snippet,
and source location. The selected backend is also recorded in the response and
index manifest. A dense load/model failure raises an error; it never silently
switches to lexical retrieval.

## Corpus-grounded generation

Replay supplies retrieved passages as structured, quoted data containing their
real chunk IDs. The generation prompt treats passages as untrusted evidence,
never as instructions, and does not configure web search or tools. The
application itself does not execute passage text or use facts outside the
configured corpus.

Generation is selected through environment-backed settings in
[`.env.example`](.env.example):

```text
FLOWCONTEXT_GENERATION_BACKEND=mock
FLOWCONTEXT_GENERATION_PROVIDER=flowcontext.mock
FLOWCONTEXT_GENERATION_MODEL=mock-grounded-v1
FLOWCONTEXT_GENERATION_TIMEOUT_S=30
FLOWCONTEXT_GENERATION_MAX_RETRIES=2
FLOWCONTEXT_GENERATION_MAX_REPAIR_ATTEMPTS=1
```

The optional real backend is `openai_compatible`. Set its provider, model,
base URL, and the name of an environment variable containing the API key. Put
the secret only in the process environment; it is never written to config
files, manifests, traces, or error messages. Calls have bounded per-attempt
timeouts, retries, and structured-output repair attempts.

Every generated answer must contain `answer_text`, factual claims,
`supporting_chunk_ids`, explicit uncertainty, and `answer_version`. The local
validator rejects malformed output and citations not present in the supplied
retrieval hits; after the bounded repair limit it returns an explicit
abstention. A structurally valid response with no factual claims is also
converted to a clear uncertainty response. Valid citation IDs prove identifier
traceability only. Evaluation reports expose `citation_id_validity_rate` and
mark semantic support evaluation as false; no semantic grounding guarantee is
advertised.

Generation traces include duration, provider/model identity, attempts, token
usage, errors, and `cost`. Cost is `"unavailable"` unless real provider usage
and both documented per-million-token prices are configured. Estimated mock
usage never becomes a price.

## Index reproducibility and provenance

Each index stores a `flowcontext.index.v1` manifest containing source
fingerprints and per-document source versions, chunking strategy/max size/
overlap, embedding provider/model/revision/license/dimensions, counts, and a
build fingerprint. Source content is normalized only for line endings and outer
whitespace before hashing. Repeated builds with the same inputs/configuration
produce the same corpus/index IDs and chunk IDs.

If an output index exists with the same build fingerprint, the build is reported
`up_to_date`. If source content or relevant chunking/embedding configuration has
changed, the command refuses to reuse the old index unless `--force` is given;
replacement is written atomically. Retrieval/replay can additionally receive
`--source` to reject a stale source fingerprint before querying. Development
transcripts and evaluation answers are not accepted as corpus documents because
the only loader is a strict document JSONL contract.

## Dense model choice and download requirements

The configured CPU provider is `sentence-transformers/all-MiniLM-L6-v2` at Hub
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0, 384 dimensions.
The model is loaded on CPU and normalized embeddings are indexed. Install the
optional `dense` extra, then allow Sentence Transformers to download the pinned
snapshot into its cache (or set `FLOWCONTEXT_EMBEDDING_CACHE_DIR`). The exact
revision and licence are written to the index manifest; no weights are bundled
in this repository. Revisit this generic English short-paragraph model after the
official corpus format, language mix, and licensing constraints are supplied.

## Transcript replay and evaluation

Baseline replay consumes partial events but performs no retrieval or generation
until the first valid final event. The provisional internal JSONL event format has
explicit `text_mode` values: `incremental` appends a fragment, while
`cumulative` replaces the assembled transcript. Cumulative events are never
concatenated with one another. The guide supplied no official event schema;
see [`docs/schema-notes.md`](docs/schema-notes.md) and the separate
[`examples/replay/`](examples/replay/) inputs.

Use `--execution-mode realtime` to wait for source-timestamp gaps. The default
`accelerated` mode consumes events without waiting for those gaps for
functional testing. Source timestamps and actual monotonic execution times
remain separate. Accelerated timing must not be used as real-time latency
evidence. Repeated identical event IDs and identical repeated final events are
traced and ignored for assembly; conflicting or divergent post-final events
fail clearly.

Replay writes the result JSON plus companion `<stem>.traces.jsonl` and
`<stem>.manifest.json` files unless explicit paths are supplied. The trace
contains event receipt, finalisation, retrieval, generation, answer status,
citations, versions, usage, costs, and errors. The non-streaming provider
reports `complete_answer_latency_ms`; no time-to-first-token event is emitted.
Provider failures are saved with `run_status=failed` and return CLI exit code 1
after the result, trace, and manifest are written.

Replay remains the existing complete-utterance baseline. Use an explicitly
lexical or dense index/backend; the generation backend comes from environment
configuration:

```bash
uv run flowcontext replay \
  --transcript data/synthetic/transcript.jsonl \
  --index artifacts/fixture-lexical-index.json \
  --source data/synthetic/documents.jsonl \
  --backend lexical \
  --output artifacts/replay.json
uv run flowcontext evaluate \
  --run artifacts/replay.json \
  --gold data/synthetic/evaluation.jsonl \
  --corpus artifacts/fixture-lexical-index.json
```

Artifacts under `artifacts/` are ignored by Git. Model-backed answer generation
is implemented behind the provider interface but has not been executed against
a real model in the current environment.

The `answer` command can print an existing validated answer, or run a single
complete query as one final transcript event through the same baseline. It
does not perform a second retrieval for an existing artifact:

```bash
uv run flowcontext answer --run artifacts/replay.json
uv run flowcontext answer \
  --query "Which venue in Pune accommodates 30 attendees?" \
  --index artifacts/fixture-lexical-index.json \
  --backend lexical \
  --output artifacts/answer.json
```

## Phase 2 streaming decisions and early retrieval

Two explicitly selectable retrieval modes use the same corpus/index,
retrieval backend, top-k, and generation settings:

* `--mode baseline` (the default) waits for final-event delivery, then
  retrieves and generates from the complete utterance.
* `--mode streaming` runs the Phase 2 controller and scheduler; a meaningful
  partial can start retrieval before final-event delivery, while generation
  remains final-only.

Always label replay timing with both `mode` and `--execution-mode`. The
execution modes are:

* `realtime`: waits for source-timestamp gaps and is the only mode suitable for
  wall-clock latency observations.
* `accelerated`: consumes events without source gaps for deterministic
  scheduling checks. Do not compare its wall-clock latency with realtime; its
  synthetic timing validates behavior, not model performance.

Build one shared fixture index and run the baseline comparison:

```bash
uv run flowcontext build-index \
  --input data/synthetic/documents.jsonl \
  --output artifacts/fixture-lexical-index.json \
  --backend lexical --source-kind synthetic_fixture

uv run flowcontext replay --mode baseline \
  --transcript data/synthetic/transcript.jsonl \
  --index artifacts/fixture-lexical-index.json \
  --source data/synthetic/documents.jsonl \
  --backend lexical --top-k 5 \
  --execution-mode realtime --output artifacts/baseline.json
```

Run streaming against that same index/backend/top-k:

```bash
uv run flowcontext replay --mode streaming \
  --transcript examples/streaming/early-retrieval.jsonl \
  --index artifacts/fixture-lexical-index.json \
  --backend lexical --top-k 5 \
  --execution-mode realtime \
  --output artifacts/streaming-replay.json
```

Both commands write a result, JSONL trace, and secret-free manifest. Reports
include source time and actual delivery time for every transcript event;
controller decision time and overhead; retrieval scheduling, actual start/end,
and evidence-ready times; final-event delivery; generation start/completion;
request/call counts, retries, supersession, stale-result discards, errors, and
usage. A non-streaming provider emits no first-content event.

The separate flags below provide reproducible scenarios:

```bash
# Meaningful partial: actual early start, and (in realtime) reusable evidence.
uv run flowcontext replay --mode streaming \
  --transcript examples/streaming/early-retrieval.jsonl \
  --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
  --execution-mode realtime --output artifacts/early-useful.json

# Incomplete speech: WAIT until the final event.
uv run flowcontext replay --mode streaming \
  --transcript examples/streaming/incomplete-speech.jsonl \
  --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
  --execution-mode accelerated --output artifacts/wait-incomplete.json

# Correction while an earlier search can still be in flight.
uv run flowcontext replay --mode streaming \
  --transcript examples/streaming/cumulative-correction.jsonl \
  --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
  --execution-mode accelerated --output artifacts/correction.json

# Repeated cumulative fragments: duplicate canonical queries are suppressed.
uv run flowcontext replay --mode streaming \
  --transcript examples/streaming/repeated-fragments.jsonl \
  --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
  --execution-mode accelerated --output artifacts/repeated.json

# Formatting with same-session context: no retrieval.
uv run flowcontext replay --mode streaming \
  --transcript examples/streaming/formatting-with-context.jsonl \
  --previous-answer artifacts/baseline.json \
  --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
  --execution-mode accelerated --output artifacts/formatting.json

# Formatting without context: explicit clarification, still no retrieval.
uv run flowcontext replay --mode streaming \
  --transcript examples/streaming/formatting-needs-context.jsonl \
  --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
  --execution-mode accelerated --output artifacts/formatting-needs-context.json

# Greeting turn: policy response, no retrieval.
uv run flowcontext replay --mode streaming \
  --transcript examples/streaming/greeting.jsonl \
  --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
  --execution-mode accelerated --output artifacts/greeting.json

# No useful partial: final-event retrieval fallback.
uv run flowcontext replay --mode streaming \
  --transcript examples/streaming/final-fallback.jsonl \
  --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
  --execution-mode accelerated --output artifacts/final-fallback.json
```

Early retrieval is counted only when `streaming_retrieval_started` occurs before
`final_event_delivered`; a decision or queued request alone does not qualify.
`valid_evidence_ready_before_finalization` requires non-empty accepted evidence
for the final canonical query, and `early_evidence_reused` records exact-query
reuse. `unnecessary_retrieval_count` is operationally superseded work, not a
semantic usefulness judgment. Greeting turns and formatting-only turns without
context skip corpus search; missing formatting context returns an explicit
clarification. Dense/real-provider unavailability is an explicit `BLOCKED`
result or failed trace—there is no silent mock fallback.

See [`docs/phase2-streaming.md`](docs/phase2-streaming.md) and
[`examples/streaming/`](examples/streaming/) for the policy and trace
timeline.

Run the dedicated matched Phase 2 suite against the same fixture index. The
development and held-out files are separate; the latter is frozen for
measurement and labels remain provisional:

```bash
uv run flowcontext evaluate-streaming \
  --cases data/evaluation/streaming_development.jsonl \
  --split development --corpus artifacts/fixture-lexical-index.json \
  --backend lexical --top-k 5 --execution-mode realtime \
  --compare-disabled-scheduling \
  --output artifacts/phase2-streaming-development.json

uv run flowcontext evaluate-streaming \
  --cases data/evaluation/streaming_held_out.jsonl \
  --split held_out --corpus artifacts/fixture-lexical-index.json \
  --backend lexical --top-k 5 --execution-mode realtime \
  --output artifacts/phase2-streaming-held-out.json
```

Use `--execution-mode accelerated` for bounded scheduling checks. Its timing
is synthetic and must not be compared with realtime replay latency. The
machine-readable audit and its checked-in local measurement are
[`reports/phase2_streaming_evaluation.json`](reports/phase2_streaming_evaluation.json)
and [`reports/phase2_streaming_evaluation.md`](reports/phase2_streaming_evaluation.md).
Fixture, simulated-delay, real-backend, and official-asset results are kept
separate; real backend and official assets remain `NOT VERIFIED` here.

## Phase 3 multi-intent retrieval and grounded answers

Phase 3 is opt-in so the earlier modes remain directly comparable:

* the default baseline still waits for the complete utterance and retrieves
  once;
* the existing streaming mode still uses revision-bound early retrieval and
  final-only generation; and
* `--multi-intent` adds structural decomposition, parallel per-intent
  retrieval, reciprocal-rank fusion with provenance-preserving deduplication,
  and grounded synthesis with explicit uncertainty for missing intent
  evidence. Select `--retrieval-mode dense`, `lexical`, `mock`, or `hybrid`;
  hybrid fuses lexical and dense ranked lists with RRF within each intent
  before assembling a fair, total-budget evidence set.

The decomposition contract is `flowcontext.phase3.v1`: it preserves the exact
transcript revision, source spans, typed constraints, ambiguities, and
independent/dependent/comparison relationships. The configured structured
provider is invoked only at stable retrieval revisions and finalisation, with
bounded timeout/repair accounting. The default local provider is explicitly a
mock model; a real OpenAI-compatible provider is selected only by
`FLOWCONTEXT_GENERATION_BACKEND=openai_compatible`. Provider failure can be
recorded only as an explicit original-query fallback, never as successful
decomposition. See the [varied decomposition example](examples/multi_intent/README.md).

Run the new comparison and audit against the same fixture index:

```bash
uv run flowcontext evaluate-phase3 \
  --corpus artifacts/fixture-lexical-index.json \
  --backend lexical --top-k 5 --split all \
  --execution-mode realtime \
  --output reports/phase3_retrieval_comparison_realtime.json
```

The Markdown/JSON report records every Phase 2 case with baseline and
streaming final queries, retrieved/relevant IDs, request status, reuse
decisions, validation details, and scoring denominators. It keeps successful
comparable retrieval quality separate from end-to-end results that include
failures, timeouts, closure, and abstentions. The retrieval follow-up is
recorded in
[`reports/phase3_retrieval_comparison_realtime.md`](reports/phase3_retrieval_comparison_realtime.md)
with its machine-readable companion
[`reports/phase3_retrieval_comparison_realtime.json`](reports/phase3_retrieval_comparison_realtime.json);
the earlier Phase 3 and Phase 2 reports are preserved.

Phase 3 validates final-query compatibility with a conservative lexical
support proxy; valid chunk IDs alone are not treated as semantic grounding.
Per-intent backend rankings, RRF inputs, assembly decisions, context usage, and
missing-intent IDs are retained in replay/trace artifacts. Reranking is
disabled by default. An unavailable dense dependency/model is reported as a
failure or blocked smoke test, never silently converted to lexical retrieval.
The real dense/provider path, official assets, human labels, and semantic claim
support remain unverified. The attempted dense smoke was blocked because the
optional `sentence-transformers` dependency is not installed; the pinned model
must also be downloaded before a real dense or hybrid run can be claimed.
Selective claim updates across follow-up turns are
intentionally deferred to Phase 4.

See [`docs/phase3.md`](docs/phase3.md) and
[`PHASE3_CHECKLIST.md`](PHASE3_CHECKLIST.md) for the boundary, held-out policy,
findings, and blockers.

## Evaluation and reproducibility

The official corpus, official evaluation cases, labels, scoring harness, and
organiser API remain missing. The separate
[`data/evaluation/`](data/evaluation/) files are synthetic, corpus-grounded
engineering cases only. They contain development and held-out splits, inline
transcripts, provisional generated relevance labels, answerability, expected
evidence IDs, support rubrics, and explicitly deferred capabilities. Related
scenarios remain in the same split, and the held-out file is not used for
tuning. Claim support is not evaluated; the report distinguishes that from
identifier validity.

Run the fixed suite with the same explicitly selected lexical fixture backend:

```bash
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

Reports keep `mock`, `fixture`, and `real_model` sections separate. They record
case counts, provisional-label status, Recall@1/@3/@5 and MRR where labels
exist, citation-ID validity, answerable/unanswerable abstention behavior,
sample-counted retrieval and complete-answer p50/p95 latency, usage, cost
availability, errors, trace completeness, code revision when available,
settings, index identity, model identities, hardware, environment, execution
mode, and warm/cold conditions. `cost` is `unavailable` without real provider
usage and documented prices. Accelerated latency is not real-time evidence.
The report's human claim-support procedure is documented in
[`docs/evaluation.md`](docs/evaluation.md); no human labels or model judge were
used for the local report.

The reproducible command sequence is install -> inspect -> ingest/build ->
retrieve -> replay -> answer -> evaluate:

```bash
uv sync --locked --python 3.11
uv run flowcontext inspect-corpus --input data/synthetic/documents.jsonl
uv run flowcontext ingest \
  --input data/synthetic/documents.jsonl \
  --output artifacts/fixture-lexical-index.json \
  --backend lexical --source-kind synthetic_fixture
uv run flowcontext retrieve \
  --index artifacts/fixture-lexical-index.json \
  --backend lexical \
  --query "Which venue in Pune accommodates 30 attendees?"
uv run flowcontext replay \
  --transcript data/synthetic/transcript.jsonl \
  --index artifacts/fixture-lexical-index.json \
  --backend lexical --execution-mode accelerated \
  --output artifacts/replay.json
uv run flowcontext answer --run artifacts/replay.json
uv run flowcontext evaluate-suite \
  --cases data/evaluation/development.jsonl --split development \
  --corpus artifacts/fixture-lexical-index.json --backend lexical
```

The container path is:

```bash
docker build -t flowcontext-phase1 .
docker run --rm flowcontext-phase1 smoke
docker run --rm \
  -v "$PWD/data:/app/mounted-data:ro" \
  -v "$PWD/artifacts:/app/artifacts" \
  flowcontext-phase1 build-index \
  --input /app/mounted-data/synthetic/documents.jsonl \
  --output /app/artifacts/container-index.json \
  --backend lexical --source-kind synthetic_fixture
```

Mount an official corpus read-only at runtime if one is supplied later; do not
copy it into the image. The image includes only application code, the local
synthetic fixture, examples, and documentation. It does not include model
weights, restricted data, credentials, or generated reports. The dense model is
downloaded separately by installing the optional `dense` extra and allowing the
pinned Sentence Transformers revision to populate its cache. Real generation
credentials are supplied only as environment variables named by the provider
configuration. In this workspace the real dense and real-provider paths remain
unverified.

See [`docs/evaluation.md`](docs/evaluation.md),
[`docs/architecture-phase1.md`](docs/architecture-phase1.md),
[`docs/phase2-handoff.md`](docs/phase2-handoff.md), and the checked-in
[`reports/phase1_baseline.md`](reports/phase1_baseline.md) for historical
measured local results and limitations. Phase 3 evidence is in
[`docs/phase3.md`](docs/phase3.md).

## Package layout

```text
src/flowcontext/
  contracts.py   # Pydantic contracts, manifests, snippets, traces
  config.py      # validated .env-style configuration
  ingestion.py   # strict JSONL loader, hashes, chunks, atomic index writes
  indexing.py    # backend/provider selection and safe builds
  embeddings.py  # provider protocol, CPU Sentence Transformers, test mock
  retrieval.py   # dense cosine + explicit lexical retrievers
  generation.py  # structured corpus-grounded providers, retries, validation
  replay.py      # async complete-utterance transcript replay
  streaming.py   # Phase 2 decisions, final-only answer replay
  scheduler.py   # bounded async retrieval, coalescing, cancellation, stale guards
  multi_intent.py # Phase 3 decomposition, per-intent retrieval, RRF fusion
  phase3_evaluation.py # Phase 3 comparison, grounding, and denominator audit
  streaming_evaluation.py # matched Phase 2 suite and provenance-separated audit
  answering.py   # retained extractive helper and factual-claim utility
  trace.py       # structured execution telemetry
  evaluation.py  # split evaluation, metrics, provenance-separated reports
  cli.py         # inspect/build/retrieve/replay/answer/evaluate/smoke commands
tests/           # standard-library unit tests
data/synthetic/  # invented fixture, never an official corpus
examples/replay/ # provisional input examples, separate from application code
examples/streaming/ # provisional Phase 2 controller examples
docs/            # asset, schema, indexing, and phase handoff notes
```

See [`PHASE1_CHECKLIST.md`](PHASE1_CHECKLIST.md) and
[`PHASE2_CHECKLIST.md`](PHASE2_CHECKLIST.md), the [asset inventory](docs/assets.md),
[`PHASE3_CHECKLIST.md`](PHASE3_CHECKLIST.md), the [asset inventory](docs/assets.md),
[schema notes](docs/schema-notes.md), [Phase 3 notes](docs/phase3.md), and
[indexing notes](docs/indexing.md).
