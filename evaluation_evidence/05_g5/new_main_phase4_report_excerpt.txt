# Phase 4 matched multi-turn evaluation

Status: synthetic fixture / lexical retrieval / explicitly labelled mock-provider engineering run; not an official benchmark result.

This report compares full re-retrieval/regeneration with dependency-aware selective updates under matched interpretation, corpus, provider, retrieval, and generation configuration.

## Label and review boundary

Development, diagnostic/regression, and untouched generalisation cases are external JSONL assets. Related `variant_family` values are checked for split crossing. Case labels remain provisional; claim-support status below is computed only from the current content-hash-bound review sheet and explicit reviewer identities.

Case count: **8**; split counts: `{'held_out': 8}`; role counts: `{'diagnostic_regression': 2, 'untouched_generalization': 6}`; variant families crossing splits: `none`.

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
| `p4-diagnostic-unanswerable-003` | initial=1 intent(s), failed; reformat_answer -> failed (coverage=n/a, uncertainty=true) | initial=1 intent(s), failed; reformat_answer -> failed (coverage=n/a, uncertainty=true) |

## Matched results

| Metric | Full A | Selective B |
|---|---:|---:|
| Follow-up interpretation accuracy | 1.0 | 1.0 |
| Invalidation precision | 1.0 | 1.0 |
| Invalidation recall | 1.0 | 1.0 |
| Obsolete claims preserved (count) | 0 | 0 |
| Unaffected claim preservation | 0.5 | 1.0 |
| Initial answer coverage | 1.0 | 1.0 |
| Updated answer coverage | 0.75 | 0.625 |
| Structural evidence quality (current citations) | 1.0 | 1.0 |
| Uncertainty targeting | 1.0 | 1.0 |
| Retrieval calls | 12 | 12 |
| Retrieved chunks | 36 | 36 |
| Retrieval tokens (estimated where marked) | 112 | 103 |
| Generation calls | 9 | 9 |
| Generation tokens | 1978 | 1837 |
| Formatting-only retrieval calls | 0 | 0 |
| Superseded request count | 0 | 0 |
| Rejected stale result count | 0 | 0 |
| Accepted stale publication count | 0 | 0 |
| Trace completeness | 1.0 | 1.0 |

Resource measurements are reported separately from correctness. A local correction that broadens the valid answer set or changes an entity is expected to issue substantial new retrieval; this run does not establish a selective efficiency gain.

Resource delta (Full − Selective): retrieval calls **0**, chunks **0**, retrieval tokens **9**, generation calls **0**, generation tokens **141**; p50 latency delta (Selective − Full) **1.6095000000000006 ms**. Mock usage is estimated, so cost is `unavailable`.

## Capability status

- `follow_up_interpretation`: **PASS**
- `dependency_invalidation`: **PASS**
- `selective_retrieval`: **PASS**
- `selective_efficiency`: **NOT VERIFIED**
- `answer_version_consistency`: **PASS**
- `formatting_only_suppression`: **PASS**
- `uncertainty_handling`: **PASS**
- `semantic_support`: **NOT VERIFIED**
- `session_isolation`: **PASS**
- `reproducibility`: **PASS**
- `official_validation`: **NOT VERIFIED**
- `real_embedding_probe`: **PASS**
- `real_generation_execution`: **NOT VERIFIED**
- `integrated_dense_rag`: **NOT VERIFIED**
- `real_backend_execution`: **NOT VERIFIED**

## Semantic support and provenance

Citation-ID validity and exact source/excerpt provenance are structural checks. They do not prove that a claim is semantically entailed. The review sheet therefore keeps `semantic_support_verdict=pending_human_review` (or lacks an explicit reviewer identity) and does not turn valid IDs or retrieval scores into a quality pass.

## Real-provider execution scope

No real-provider generation replay completed in this run. The embedding probe status is **PASS**, while integrated dense/hybrid RAG status is **NOT VERIFIED**. No integrated dense/hybrid retrieval replay has completed.

## Limitations

- The corpus is synthetic and the labels are provisional; no official competition validation is claimed.
- The default generation execution is the repository mock provider. Mock behavior tests fixture plumbing and revision correctness, not real-model answer quality.
- Cost is unavailable for estimated mock tokens; real-provider usage and pricing require configured credentials and documented prices.
- Case labels remain provisional, and the manual Codex claim review is not independent human ground truth.
- Persistence, retention policy, and production-scale scheduler capacity remain outside this lightweight Phase 4 implementation.
