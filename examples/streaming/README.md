# Phase 2 streaming examples

These provisional JSONL inputs exercise the local controller and bounded
retrieval scheduler. They are not official organiser event files. Use the
same lexical index, backend, top-k setting, and generation configuration for
baseline/streaming comparisons.

Build the shared fixture index first:

    uv run flowcontext build-index \
      --input data/synthetic/documents.jsonl \
      --output artifacts/fixture-lexical-index.json \
      --backend lexical --source-kind synthetic_fixture

Examples:

    # Useful early retrieval; realtime lets evidence finish before final delivery.
    uv run flowcontext replay --mode streaming \
      --transcript examples/streaming/early-retrieval.jsonl \
      --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
      --execution-mode realtime --output artifacts/streaming-useful.json

    # Incomplete speech waits, then retrieves at final delivery.
    uv run flowcontext replay --mode streaming \
      --transcript examples/streaming/incomplete-speech.jsonl \
      --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
      --execution-mode accelerated --output artifacts/streaming-wait.json

    # A correction can supersede an in-flight partial search.
    uv run flowcontext replay --mode streaming \
      --transcript examples/streaming/cumulative-correction.jsonl \
      --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
      --execution-mode accelerated --output artifacts/streaming-correction.json

    # Repeated cumulative fragments do not issue duplicate canonical searches.
    uv run flowcontext replay --mode streaming \
      --transcript examples/streaming/repeated-fragments.jsonl \
      --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
      --execution-mode accelerated --output artifacts/streaming-repeated.json

    # First produce a successful baseline answer for the same session.
    uv run flowcontext replay --mode baseline \
      --transcript data/synthetic/transcript.jsonl \
      --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
      --execution-mode accelerated --output artifacts/baseline.json

    # Formatting uses same-session context and performs no corpus search.
    uv run flowcontext replay --mode streaming \
      --transcript examples/streaming/formatting-with-context.jsonl \
      --previous-answer artifacts/baseline.json \
      --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
      --execution-mode accelerated --output artifacts/streaming-formatting.json

    # Formatting without context returns a clarification and performs no search.
    uv run flowcontext replay --mode streaming \
      --transcript examples/streaming/formatting-needs-context.jsonl \
      --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
      --execution-mode accelerated --output artifacts/streaming-formatting-needs-context.json

    # Greeting turns are handled by policy without corpus retrieval.
    uv run flowcontext replay --mode streaming \
      --transcript examples/streaming/greeting.jsonl \
      --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
      --execution-mode accelerated --output artifacts/streaming-greeting.json

    # No useful partial search: final-event retrieval is the fallback.
    uv run flowcontext replay --mode streaming \
      --transcript examples/streaming/final-fallback.jsonl \
      --index artifacts/fixture-lexical-index.json --backend lexical --top-k 5 \
      --execution-mode accelerated --output artifacts/streaming-fallback.json

Each command writes a result, `<stem>.traces.jsonl`, and secret-free manifest.
The trace records source time plus observed delivery time for each event,
decision time/overhead, retrieval scheduled/start/end, evidence-ready time,
final delivery, generation start/completion, and no first-content event unless
a provider actually exposes streamed content. The runner never generates an
early answer.
