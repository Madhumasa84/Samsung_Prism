# Phase 2 controller/scheduler integration report

This is a reproducible local engineering report, not a competition or model
performance benchmark. It uses the three-document/four-chunk synthetic corpus,
one lexical index, top_k=5, and the configured mock generation provider for
both explicitly selected replay modes. The shared index was
index-03e98680ae845e95 for corpus corpus-3762c90f2912589e.

## Executed commands

The index and replay matrix were executed with the repository source on the
Python path:

    PYTHONPATH=src python -m flowcontext build-index --input data/synthetic/documents.jsonl \
      --output /tmp/flowcontext-phase2-run/fixture-lexical-index.json \
      --backend lexical --source-kind synthetic_fixture --force

    PYTHONPATH=src python -m flowcontext replay --mode baseline \
      --transcript data/synthetic/transcript.jsonl \
      --index /tmp/flowcontext-phase2-run/fixture-lexical-index.json \
      --source data/synthetic/documents.jsonl --backend lexical --top-k 5 \
      --execution-mode realtime --output /tmp/flowcontext-phase2-run/baseline.json

    PYTHONPATH=src python -m flowcontext replay --mode streaming \
      --transcript examples/streaming/early-retrieval.jsonl \
      --index /tmp/flowcontext-phase2-run/fixture-lexical-index.json \
      --source data/synthetic/documents.jsonl --backend lexical --top-k 5 \
      --execution-mode realtime --output /tmp/flowcontext-phase2-run/early-useful.json

The same streaming command was run for incomplete-speech.jsonl,
cumulative-correction.jsonl, repeated-fragments.jsonl,
formatting-with-context.jsonl with the same-session baseline answer,
formatting-needs-context.jsonl, greeting.jsonl, and final-fallback.jsonl;
those examples and commands are documented in the repository README files.
The CLI wrapper was also checked with:

    UV_CACHE_DIR=/tmp/flowcontext-uv-cache uv run --offline --python 3.11 flowcontext --help

## Replay matrix

calls/scheduled counts actual backend calls versus allocated requests;
superseded/stale counts operational supersession and stale-result discards.
early/ready/reused are the three separate early-usefulness fields.

| scenario | mode | execution | status | calls/scheduled | early/ready/reused | superseded/stale | controller ms | answer ms | full ms |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| baseline transcript | baseline | realtime | completed | 1/1 | no/no/no | 0/0 | 0.00 | 3.04 | 2106.64 |
| useful partial | streaming | realtime | completed | 1/1 | yes/yes/yes | 0/0 | 0.92 | 1.68 | 106.57 |
| incomplete speech | streaming | accelerated | completed | 1/1 | no/no/no | 0/0 | 1.02 | 3.62 | 5.51 |
| correction in flight | streaming | accelerated | completed | 2/2 | yes/no/no | 1/1 | 0.85 | 4.97 | 8.36 |
| repeated fragments | streaming | accelerated | completed | 1/1 | yes/no/no | 0/0 | 1.11 | 2.11 | 6.19 |
| formatting with context | streaming | accelerated | completed | 0/0 | no/no/no | 0/0 | 0.36 | 0.44 | 0.67 |
| formatting missing context | streaming | accelerated | needs_context | 0/0 | no/no/no | 0/0 | 0.31 | 0.38 | 0.58 |
| greeting | streaming | accelerated | completed | 0/0 | no/no/no | 0/0 | 0.35 | 0.44 | 0.75 |
| final-event fallback | streaming | accelerated | completed | 1/1 | no/no/no | 0/0 | 0.86 | 3.88 | 5.31 |

The correction run also reported unnecessary_retrieval_count=1, making the
cost of the superseded partial search visible. Repeated fragments suppressed
two duplicate canonical-query decisions and made one retrieval call. All runs
reported zero retrieval errors and zero timeouts. Retrieval usage is estimated
lexical query-token usage; mock generation usage is estimated by the provider.
The realtime useful run used 3 estimated retrieval input tokens and 179
estimated generation tokens. The baseline used 11 retrieval input tokens and
410 estimated generation tokens.

Accelerated rows validate scheduling and trace behavior only. Their wall-clock
durations must not be compared with the realtime rows or interpreted as model
latency. Every CLI result and manifest includes mode and execution_mode.

## Representative trace timeline

Current realtime useful-partial trace, with replay-relative execution times:

    transcript_event_received source=0.0 delivery=0.00322
    streaming_decision RETRIEVE decision_time=0.00401
    streaming_retrieval_scheduled t=0.00447
    streaming_retrieval_started t=0.00464
    streaming_retrieval_completed t=0.00647
    streaming_evidence_ready t=0.00660
    transcript_event_received source=0.1 delivery=0.10728
    final_event_delivered t=0.10728
    streaming_decision SKIP reason=duplicate_query_suppressed t=0.10771
    streaming_evidence_reused early=true t=0.10788
    streaming_generation_started t=0.10794
    streaming_generation_completed t=0.10913
    streaming_answer_completed t=0.10928

Early retrieval is true only because the actual start precedes final-event
delivery. The queued request and the decision alone do not qualify. Valid
evidence requires accepted, non-stale, non-empty hits for the final canonical
query and an evidence-ready time before final delivery.

## Verification and blockers

    PYTHONPATH=src pytest -q                 -> 51 passed
    python -m compileall -q src tests        -> passed
    PYTHONPATH=src python -m flowcontext config-check -> exit 0
    PYTHONPATH=src python -m flowcontext smoke        -> PASS
    PYTHONPATH=src python -m flowcontext evaluate ... -> PASS, case pass rate 1.0

The explicit dense smoke check returned exit code 3 with status BLOCKED because
sentence-transformers is not installed. No mock fallback was selected by that
command. Real dense retrieval, a real generation provider, provider
usage/cost, retrieval quality, and semantic claim support remain unverified.
Traces contain timings, decisions, request/query identifiers, corpus chunk IDs,
and structured errors, but no API keys or secret values.
