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
