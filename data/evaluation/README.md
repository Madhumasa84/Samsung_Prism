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
