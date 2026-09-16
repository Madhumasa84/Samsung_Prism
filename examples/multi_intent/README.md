# Phase 3 decomposition example

This is a provisional local example, not an organiser transcript schema. It
deliberately mixes an independent question with a dependent follow-up and
keeps the numeric, currency, location, relationship, and negation-sensitive
text traceable to the parent request.

Input:

```text
Which laptops in Bengaluru cost under ₹80,000 and have 16GB RAM? What is the return window? For each selected model, does it support Linux?
```

The explicitly labelled offline compatibility helper produces the following
contract-shaped excerpt. The complete object, including ordinals, constraint
IDs, exact spans, usage, and provider metadata, is available from
`result.model_dump(mode="json")`; this shortened view omits some repetitive
fields throughout:

```json
{
  "schema_version": "flowcontext.phase3.v1",
  "transcript_revision": 12,
  "intents": [
    {
      "intent_id": "intent-9fc000ab66",
      "query": "Which laptops in Bengaluru cost under ₹80,000 and have 16GB RAM",
      "relationship": "independent",
      "source_span": {"start": 0, "end": 63, "text": "Which laptops in Bengaluru cost under ₹80,000 and have 16GB RAM"},
      "constraints": [
        {"kind": "location", "value": "in Bengaluru", "source_span": {"start": 14, "end": 26, "text": "in Bengaluru"}},
        {"kind": "quantity", "value": "under ₹80,000", "source_span": {"start": 32, "end": 45, "text": "under ₹80,000"}},
        {"kind": "quantity", "value": "16GB RAM", "source_span": {"start": 55, "end": 63, "text": "16GB RAM"}}
      ]
    },
    {"intent_id": "intent-eccb5b0617", "query": "What is the return window", "relationship": "independent"},
    {"intent_id": "intent-3ec079ab86", "query": "For each selected model, does it support Linux?", "relationship": "dependent"}
  ],
  "dependencies": [
    {
      "prerequisite_intent_id": "intent-9fc000ab66",
      "dependent_intent_id": "intent-3ec079ab86",
      "relation": "depends_on"
    }
  ],
  "ambiguities": [
    {"description": "Reference 'selected model' needs an antecedent or selection context."},
    {"description": "Reference 'it support' needs an antecedent or selection context."}
  ],
  "decomposition_method": "offline_rule_based_v2",
  "status": "success"
}
```

For an opt-in retrieval run, `--multi-intent` uses the configured
decomposition provider. With the default local settings the provider is the
explicitly labelled mock model (`flowcontext.mock:mock-grounded-v1`); a real run requires:

```bash
# Set FLOWCONTEXT_PHASE3_API_KEY in the environment or a secret manager first.
FLOWCONTEXT_GENERATION_BACKEND=openai_compatible \
FLOWCONTEXT_GENERATION_BASE_URL=https://provider.example/v1 \
FLOWCONTEXT_GENERATION_API_KEY_ENV=FLOWCONTEXT_PHASE3_API_KEY \
FLOWCONTEXT_GENERATION_MODEL=your-decomposer-model \
uv run flowcontext retrieve --multi-intent \
  --index artifacts/fixture-lexical-index.json --backend lexical \
  --query "Which laptops in Bengaluru cost under ₹80,000 and have 16GB RAM?"
```

That path reports `model_backed_structured_validated_v1`, provider attempts,
bounded repairs, token usage, latency, and any explicit original-query
fallback. A provider failure is not reported as successful decomposition.
