# Phase 3 architecture and evaluation boundary

Phase 3 adds a revision-bound multi-intent retrieval path while preserving the
Phase 1 final-event baseline and Phase 2 single-query streaming path. The
implementation is opt-in; no selective claim update across follow-up turns is
implemented here.

## Runtime flow

```text
transcript events
      |
      v
Phase 2 controller/scheduler -----> revisioned retrieval requests
      |                                      |
      | final query                          v
      |                             MultiIntentRetriever
      |                                      |
      |                     decompose -> per-intent retrieval
      |                                      |
      |                         backend RRF -> intent RRF
      |                                      |
      |                         budgeted evidence assembly
      v                                      v
final-only generation <----------- unified grounded synthesis
      |
      v
Answer + claims + intent statuses + citation/provenance audit + trace
```

The controller owns session/utterance and transcript revisions. A
`DecompositionResult` preserves the original transcript, exact source spans,
stable intent IDs, shared constraints, intent-local constraints, dependencies,
and ambiguities. Retrieval results carry both the parent/retrieval revision and
the evidence provenance. A stale or incomplete result cannot become current
evidence; a decomposer failure is recorded as an explicit original-query
fallback rather than silently substituting a mock plan.

Evidence fusion combines ranked lists by reciprocal rank, not incomparable
raw scores. Evidence assembly performs mandatory intent coverage before global
rank fill, respects the top-k and context budgets, and preserves all intent
relationships on a deduplicated chunk. `missing_intent_ids` and per-intent
backend rankings remain in the audit record.

Generation receives the parent request, the ordered intent records, shared
constraints, grouped evidence, and exact chunk provenance. The structured
answer contains atomic claim records, supporting chunk IDs/excerpts/spans,
intent statuses, uncertainty, and answer version. Deterministic citation-ID
and excerpt validation is separate from semantic claim support. Semantic
support is `NOT VERIFIED` in the current local audit.

## Evaluation architecture

The dedicated external label assets are
[`data/evaluation/phase3_development.jsonl`](../data/evaluation/phase3_development.jsonl)
and
[`data/evaluation/phase3_held_out.jsonl`](../data/evaluation/phase3_held_out.jsonl).
Runtime modules do not import expected intents, expected answer strings, or
relevance labels. The audit loads those records at the evaluation boundary and
checks split and variant-family integrity.

The matched arms are run over the same synthetic index, replay events, model
configuration, process hardware, and `k`:

| Arm | Retrieval/answer path |
|---|---|
| A | Original final-event baseline; one complete parent-query retrieval. |
| B | Phase 2 streaming controller/scheduler; one parent-query retrieval. |
| C | Phase 3 streaming; decomposition, intent-local retrieval, RRF fusion, and unified synthesis. |

The current audit index is 9 documents / 10 chunks with `k=5`; the window is
deliberately smaller than the corpus. The report separates successful labelled
retrieval quality from failures, timeouts, abstentions, and provider faults.
It records one-to-one intent matching, per-intent Recall@1/@3/@5, complete
request evidence coverage, citation-ID validity, provisional supported-answer
coverage, uncertainty behavior, early retrieval/reuse, stale acceptance,
latency, calls, tokens, errors, cost availability, and trace completeness.

The single-query versus decomposed comparison is a measured lexical
engineering ablation only. Dense-only versus hybrid is not run without a
dense model/index; no mock comparison is presented as model quality. Real
provider and official benchmark validation remain outside the local evidence
boundary.

## Phase boundary

Phase 4 handoff fields are documented in
[`phase4-handoff.md`](phase4-handoff.md): intent IDs and constraints, claim
records, evidence dependencies, revision/supersession records, and
session-state interfaces. This document does not define or implement Phase 4
selective answer updates.
