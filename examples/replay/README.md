# Replay examples

These are provisional internal JSONL examples, not organiser-supplied event
formats. The Theme 4 guide did not include a machine-readable transcript
schema. Each file reconstructs the same question, `Which venue in Pune?`, but
uses a different transcript encoding.

Run them with the existing CLI and a compatible corpus index, for example:

```bash
uv run flowcontext replay \
  --transcript examples/replay/incremental.jsonl \
  --index artifacts/fixture-lexical-index.json \
  --backend lexical \
  --execution-mode accelerated
```

`duplicate-final.jsonl` demonstrates the idempotent repeated-final policy. The
second final event is received and traced, but does not trigger another
retrieval or generation execution.

## Phase 4 replay examples

The `phase4-*.jsonl` files use the same executable pipeline for an initial
compound answer, a late constraint update, an entity-changing correction, a
partially unsupported update, a presentation-only request, and a correction
that races older generation work. They are synthetic fixture examples and use
the offline mock generation provider unless the caller injects a configured
real provider. Build the compatible lexical index first, then run for example:

```bash
uv run flowcontext build-index \
  --input data/synthetic/documents.jsonl \
  --output artifacts/fixture-lexical-index.json \
  --backend lexical --source-kind synthetic_fixture

uv run flowcontext phase4-replay \
  --turns examples/replay/phase4-formatting.jsonl \
  --index artifacts/fixture-lexical-index.json \
  --backend lexical --output artifacts/phase4-formatting.json
```

The formatting turn is published with `change_kind="presentation"` and zero
retrieval calls. The race example starts a second generation of the current
answer and applies its correction while that work is in flight:

```bash
uv run flowcontext phase4-replay \
  --turns examples/replay/phase4-race.jsonl \
  --index artifacts/fixture-lexical-index.json \
  --backend lexical --race --generation-delay-s 0.05 \
  --output artifacts/phase4-race.json
```

The output contains the proposed patch, selective plan, immutable answer
versions, claim/evidence change records, provider labels, and any discarded
late result. It is an engineering replay, not a real-model quality claim.

The complete matched evaluation uses the same actual replay/publisher path for
both update arms and records per-turn traces in
[`../../reports/phase4_evaluation.json`](../../reports/phase4_evaluation.json).
The six measured replay executions are also preserved as
`phase4_replay_initial_compound.json`, `phase4_replay_late_constraint.json`,
`phase4_replay_entity_correction.json`, `phase4_replay_partial_unsupported.json`,
`phase4_replay_formatting.json`, and `phase4_replay_race.json` under
[`../../reports/`](../../reports/).
