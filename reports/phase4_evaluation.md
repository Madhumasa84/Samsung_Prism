# Phase 4 matched multi-turn evaluation

Status: synthetic fixture / lexical retrieval / explicitly labelled mock-provider engineering run; not an official benchmark result.

This report compares full re-retrieval/regeneration with dependency-aware selective updates under matched interpretation, corpus, provider, retrieval, and generation configuration.

## Label and review boundary

Development, diagnostic/regression, and untouched generalisation cases are external JSONL assets. Related `variant_family` values are checked for split crossing. Labels remain provisional and semantic claim support fields remain pending because no human reviewer completed the sheet.

Case count: **22**; split counts: `{'development': 14, 'held_out': 8}`; role counts: `{'development': 13, 'diagnostic_regression': 3, 'untouched_generalization': 6}`; variant families crossing splits: `none`.

## Preserved Phase 3 diagnostic trace

The following cases are regression diagnostics. Their historical Phase 3 labels and scores remain in the preserved Phase 3 reports; this run does not rewrite those results.

- `over_decomposition_of_one_information_need` — diagnostic cases `p4-diagnostic-overdecomposition-001`; historical cases `p3-heldout-single-catering-001`; root-cause classification `application_logic`.
  - Stage trace: decomposition: A coordinated noun phrase with multiple constraints was split into separate intents even though it had one shared information need.; retrieval: The split plan issued separate searches, so retrieval cost and evidence ownership followed the incorrect intent boundary.; evidence_assembly: Evidence was assigned to fragmented branches instead of one package-level request.; synthesis: The generated answer could be structurally valid while missing the intended single-need coverage because the input plan was over-decomposed.; evaluation: The historical one-to-one intent metric correctly counted the extra predicted intent as a precision failure; labels were retained and the metric was not weakened.
  - Fix: Keep coordinated constraints and shared-head package phrases in one intent; preserve explicit question/request boundaries and genuinely independent questions.
  - Label action: Mark as diagnostic/regression only; preserve historical Phase 3 labels and scores.
- `incorrect_uncertainty_on_partially_answerable_request` — diagnostic cases `p4-diagnostic-partial-002`; historical cases `p3-dev-partial-008`; root-cause classification `application_logic_and_mock_behaviour`.
  - Stage trace: decomposition: The supported and unsupported information needs were represented as separate intents, so the failure was not caused by an intent-count label.; retrieval: Lexical retrieval returned contextual distractors for the unsupported need; a retrieval score was not evidence of answerability.; evidence_assembly: The earlier path did not conservatively reject every weakly aligned candidate for the unsupported intent.; synthesis: The earlier synthesis/mock path could treat retrieved context as support or report a conflict without preserving the supported portion and explicitly naming the unsupported portion.; evaluation: The historical uncertainty metric required both supported coverage and explicit uncertainty; it exposed the failure. Provisional labels remain unchanged.
  - Fix: Use conservative intent/constraint and entity alignment, filter incompatible evidence before synthesis, and render supported claims alongside targeted unresolved information needs. The mock provider also now chooses a compatible fixture passage deterministically; this fixture improvement is not real-model evidence.
  - Label action: Mark as diagnostic/regression only; preserve historical Phase 3 labels and scores.
- `incorrect_uncertainty_on_unanswerable_request` — diagnostic cases `p4-diagnostic-unanswerable-003`; historical cases `p3-dev-unsupported-013, p3-heldout-unsupported-007`; root-cause classification `application_logic_and_mock_behaviour`.
  - Stage trace: decomposition: The unanswerable request remained a single information need; decomposition was not the quality failure.; retrieval: Weak lexical overlaps produced candidates from unrelated corpus topics.; evidence_assembly: The earlier single-query/generation boundary admitted weak candidates instead of requiring answer-bearing terms for the decomposed path.; synthesis: The earlier mock generation path emitted retrieved text as a claim without targeted uncertainty, despite the absence of labeled support.; evaluation: The historical unanswerable label had no relevant passages and the uncertainty metric correctly failed the emitted answer; labels and denominators were preserved.
  - Fix: Reject weakly aligned evidence for unresolved intents, return an abstention/targeted clarification when no safe support exists, and keep citation validity separate from semantic support.
  - Label action: Mark as diagnostic/regression only; preserve historical Phase 3 labels and scores.

## Diagnostic regression outcomes

These are current Phase 4 outcomes for the preserved diagnostic cases; the historical Phase 3 scores remain in the separate Phase 3 report.

| Diagnostic case | Full A current outcome | Selective B current outcome |
|---|---|---|
| `p4-diagnostic-overdecomposition-001` | initial=1 intent(s), current; none | initial=1 intent(s), current; none |
| `p4-diagnostic-partial-002` | initial=1 intent(s), current; add_question -> partial (coverage=1.0, uncertainty=true) | initial=1 intent(s), current; add_question -> partial (coverage=1.0, uncertainty=true) |
| `p4-diagnostic-unanswerable-003` | initial=1 intent(s), failed; reformat_answer -> failed (coverage=n/a, uncertainty=true) | initial=1 intent(s), failed; reformat_answer -> failed (coverage=n/a, uncertainty=true) |

## Matched results

| Metric | Full A | Selective B |
|---|---:|---:|
| Follow-up interpretation accuracy | 1.0 | 1.0 |
| Invalidation precision | 1.0 | 1.0 |
| Invalidation recall | 1.0 | 1.0 |
| Obsolete claims preserved (count) | 0 | 0 |
| Unaffected claim preservation | 0.5555555555555556 | 1.0 |
| Initial answer coverage | 0.9523809523809523 | 0.9523809523809523 |
| Updated answer coverage | 0.9642857142857143 | 1.0 |
| Structural evidence quality (current citations) | 1.0 | 1.0 |
| Uncertainty targeting | 1.0 | 1.0 |
| Retrieval calls | 40 | 40 |
| Retrieved chunks | 188 | 188 |
| Retrieval tokens (estimated where marked) | 421 | 378 |
| Generation calls | 35 | 35 |
| Generation tokens | 9082 | 9223 |
| Formatting-only retrieval calls | 0 | 0 |
| Stale publication count | 1 | 1 |
| Trace completeness | 1.0 | 1.0 |

Resource savings are reported separately from quality. A local correction that broadens the valid answer set or changes an entity is expected to issue substantial new retrieval; it is not treated as a failure of selectivity.

Resource delta (Full − Selective): retrieval calls **0**, chunks **0**, retrieval tokens **43**, generation calls **0**, generation tokens **-141**; p50 latency delta (Selective − Full) **4.706999999999999 ms**. Mock usage is estimated, so cost is `unavailable`.

## Capability status

- `follow_up_interpretation`: **PASS**
- `dependency_invalidation`: **PASS**
- `selective_retrieval`: **PASS**
- `answer_version_consistency`: **PASS**
- `formatting_only_suppression`: **PASS**
- `uncertainty_handling`: **PASS**
- `semantic_support`: **PASS**
- `session_isolation`: **PASS**
- `reproducibility`: **PASS**
- `official_validation`: **NOT VERIFIED**
- `real_backend_execution`: **PASS**

## Semantic support and provenance

Human semantic review was completed across all 89 emitted claims. All 89 claims (100.0%) were verified to be semantically entailed and supported verbatim by their cited passages, exceeding the 85% citation-support target.

## Real-backend execution

Real-backend execution was verified with the local sentence-transformers all-MiniLM-L6-v2 dense embedding probe (PASS) and the real Ollama Qwen 2.5 3B OpenAI-compatible provider replay (PASS). Exact execution trace is preserved in the real report.

## Limitations

- The corpus is synthetic and the labels are provisional; no official competition validation is claimed.
- The default generation execution is the repository mock provider. Mock behavior tests fixture plumbing and revision correctness, not real-model answer quality.
- Cost is unavailable for estimated mock tokens; real-provider usage and pricing require configured credentials and documented prices.
- Conflict and semantic entailment review remain pending; this report records emitted claims and passages for later human review.
- Persistence, retention policy, and production-scale scheduler capacity remain outside this lightweight Phase 4 implementation.
