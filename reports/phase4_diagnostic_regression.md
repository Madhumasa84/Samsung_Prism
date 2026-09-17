# Phase 3 dedicated evaluation and audit

Status: local synthetic fixture / lexical retrieval / mock providers; not an official benchmark result.

Phase 4 was not started. Historical Phase 1/2/3 reports are preserved; this is a new report.

## Matched conditions

| Condition | Definition |
|---|---|
| A | Original final-event baseline; one parent-query retrieval after finalisation. |
| B | Phase 2 streaming controller with a single parent query and final-only generation. |
| C | Phase 3 streaming with structured decomposition, per-intent retrieval, RRF fusion, and grounded synthesis. |

Corpus: **9 documents / 10 chunks**; top-k: **5**; top-k less than corpus chunks: **True**.

All three conditions use the same corpus/index, top-k, configuration, hardware, and replay cases. Condition C has the additional decomposition stage required by Phase 3.

## Label and split integrity

Development cases: 13; held-out cases: 7; related variant families cross splits: `none`.

Scenario coverage: single_question_constraints=2; independent_compound=1; shared_scope=1; dependent_question=1; negation=1; comparison=1; partial_answer=1; entity_confusion=1; correction_retrieval=1; retrieval_failure=1; generation_provider_failure=1; unsupported_request=1; single_catering_paraphrase=1; independent_capacity_catering=1; dependent_dietary=1; conflicting_evidence_precedence=1; decomposition_provider_failure=1; retrieval_timeout=1; unsupported_insurance_request=1.

Label status counts are explicit: all current intent, relevance, and answer-expectation labels are `provisional_generated`; model-reviewed and human-reviewed counts are zero. Semantic claim support review is `not_evaluated`.

## Diagnostic regression outcomes

The historical before-state remains in `reports/phase3_evaluation_realtime.*`;
these cases were not relabelled. The current C-condition outcomes are:

| Case | Historical failure | Current result |
|---|---|---|
| `p3-heldout-single-catering-001` | 2 intents and false conflict | 1 intent; supported package answer |
| `p3-dev-partial-008` | weak parking claim for unsupported tax intent | venue claim plus explicit tax uncertainty |
| `p3-dev-unsupported-013` | parking claim for unsupported tax request | abstained with no claims |
| `p3-heldout-unsupported-007` | false conflict/claim for unsupported insurance request | abstained with no claims |

The audit metric now reports partial uncertainty `1/1` and unanswerable
uncertainty `2/2`. This is still fixture/mock behavior validation; semantic
claim support and real-provider quality remain unverified.

## Guide targets

- Multi-intent identification target (70%): **PASS** locally against provisional labels; development `1.0`, held-out `0.75`.
- Citation-support target (85%): **NOT VERIFIED**. Citation-ID validity is reported separately and does not establish semantic support.

## Split results

### Development

Cases: 13 (single 9, compound 4). Multi-intent exact identification: 4/4 = `1.0`; missed intents: 0; unnecessary extra intents: 0.

| Condition | Matched Recall@5 | Matched cases | Completed / all | Final-event→answer p50/p95 ms |
|---|---:|---:|---:|---:|
| A_original_baseline | 1.0 | 8 | 10/13 | 9.801554000205215 / 101.17760640005115 |
| B_phase2_streaming_single_query | 1.0 | 8 | 10/13 | 11.029388000224571 / 106.69230460007358 |
| C_phase3_streaming_fused | 0.9375 | 8 | 8/13 | 18.33719399974143 / 113.26830860007222 |

Per-condition resources:
**A_original_baseline**
- Retrieval calls: 13; model calls: {'generation': 11, 'decomposition': 0, 'total': 11}
- Tokens: retrieval 96, decomposition 0, generation 4245, repair 0, verification 0, total 4341
- Errors: 2 events across 2 cases; cost: `unavailable` (unavailable)
**B_phase2_streaming_single_query**
- Retrieval calls: 14; model calls: {'generation': 11, 'decomposition': 0, 'total': 11}
- Tokens: retrieval 99, decomposition 0, generation 4245, repair 0, verification 0, total 4344
- Errors: 2 events across 2 cases; cost: `unavailable` (unavailable)
**C_phase3_streaming_fused**
- Retrieval calls: 14; model calls: {'generation': 9, 'decomposition': 14, 'total': 23}
- Tokens: retrieval 99, decomposition 2319, generation 1655, repair 0, verification 482, total 4555
- Errors: 4 events across 3 cases; cost: `unavailable` (unavailable)

Phase 3 complete-request evidence coverage (macro): `0.7083333333333334` over 12 cases.
Phase 3 per-intent retrieval Recall@1/@3/@5 (macro): `0.5333333333333333` / `0.8666666666666667` / `0.8666666666666667` over 15 applicable labelled intent records.
Phase 3 supported-answer coverage (provisional substring check): `7/15` = `0.4666666666666667`; this is not semantic support.
Phase 3 citation-ID validity: `8/8` emitted-citation runs = `1.0`; 5 runs emitted no citations; semantic support: **NOT VERIFIED**.
Phase 2/3 early retrieval and reuse: B started 1/1, useful reuse 0, stale accepted 0; C started 1/1, useful reuse 0, stale accepted 0.

### Held Out

Cases: 7 (single 3, compound 4). Multi-intent exact identification: 3/4 = `0.75`; missed intents: 1; unnecessary extra intents: 0.

| Condition | Matched Recall@5 | Matched cases | Completed / all | Final-event→answer p50/p95 ms |
|---|---:|---:|---:|---:|
| A_original_baseline | 1.0 | 5 | 6/7 | 9.37786899976345 / 95.87147929987614 |
| B_phase2_streaming_single_query | 1.0 | 5 | 5/7 | 10.915954000211059 / 30.20477799977924 |
| C_phase3_streaming_fused | 1.0 | 5 | 5/7 | 19.160434999776044 / 34.402036700021185 |

Per-condition resources:
**A_original_baseline**
- Retrieval calls: 7; model calls: {'generation': 6, 'decomposition': 0, 'total': 6}
- Tokens: retrieval 54, decomposition 0, generation 2159, repair 0, verification 0, total 2213
- Errors: 0 events across 0 cases; cost: `unavailable` (unavailable)
**B_phase2_streaming_single_query**
- Retrieval calls: 7; model calls: {'generation': 5, 'decomposition': 0, 'total': 5}
- Tokens: retrieval 54, decomposition 0, generation 1886, repair 0, verification 0, total 1940
- Errors: 1 events across 1 cases; cost: `unavailable` (unavailable)
**C_phase3_streaming_fused**
- Retrieval calls: 7; model calls: {'generation': 5, 'decomposition': 7, 'total': 12}
- Tokens: retrieval 54, decomposition 990, generation 1286, repair 0, verification 416, total 2746
- Errors: 1 events across 1 cases; cost: `unavailable` (unavailable)

Phase 3 complete-request evidence coverage (macro): `0.8333333333333334` over 6 cases.
Phase 3 per-intent retrieval Recall@1/@3/@5 (macro): `0.4444444444444444` / `0.7777777777777778` / `0.8888888888888888` over 9 applicable labelled intent records.
Phase 3 supported-answer coverage (provisional substring check): `6/10` = `0.6`; this is not semantic support.
Phase 3 citation-ID validity: `5/5` emitted-citation runs = `1.0`; 2 runs emitted no citations; semantic support: **NOT VERIFIED**.
Phase 2/3 early retrieval and reuse: B started 0/0, useful reuse 0, stale accepted 0; C started 0/0, useful reuse 0, stale accepted 0.

## Focused ablations

Single-query versus decomposed retrieval: `measured` over 13 normal, answerable labelled cases; lexical-only engineering measurement, not a model-quality finding.
Dense-only versus hybrid: **NOT VERIFIED** — sentence-transformers is not installed; no dense index/model run is available.

## Failures and limitations

- `p3-dev-correction-010`: provider `none`, retrieval `delay`; each arm is retained with status/error/fallback details in JSON.
- `p3-dev-retrieval-failure-011`: provider `none`, retrieval `failure`; each arm is retained with status/error/fallback details in JSON.
- `p3-dev-generation-failure-012`: provider `generation`, retrieval `normal`; each arm is retained with status/error/fallback details in JSON.
- `p3-heldout-decomposer-failure-005`: provider `decomposition`, retrieval `normal`; each arm is retained with status/error/fallback details in JSON.
- `p3-heldout-retrieval-timeout-006`: provider `none`, retrieval `timeout`; each arm is retained with status/error/fallback details in JSON.
- All cases and corpus documents are synthetic fixture data grounded in the available local corpus; they are not organiser assets.
- All external intent, relevance, and answer labels are provisional_generated; no model-reviewed or human-reviewed labels are present.
- The mock generation/decomposition providers are contract-test providers, not model-quality evidence.
- Semantic claim support was not assessed. The guide's 85% citation-support target is NOT VERIFIED.
- Dense-only and hybrid retrieval were not run because no usable dense dependency/model/index was available.
- Real provider-backed generation and official benchmark validation were not run.
- The Phase 2 final-event stale-result and session-isolation regressions are verified by the existing test suite; this report does not claim Phase 4 state updates.

## PASS / FAIL / NOT VERIFIED

| Capability | Status | Evidence boundary |
|---|---|---|
| `single_questions_not_decomposed` | **PASS** | synthetic_fixture_provisional_labels |
| `compound_independent_questions` | **PASS** | synthetic_fixture_provisional_labels |
| `dependent_questions` | **PASS** | synthetic_fixture_provisional_labels |
| `shared_constraints` | **PASS** | synthetic_fixture_provisional_labels |
| `negation` | **PASS** | synthetic_fixture_provisional_labels |
| `comparisons` | **PASS** | synthetic_fixture_provisional_labels |
| `partial_answerability` | **PASS** | synthetic_fixture_provisional_labels |
| `entity_confusion` | **PASS** | synthetic_fixture_provisional_labels |
| `uncertainty_on_partial_requests` | **PASS** | Phase 3 C answer uncertainty behavior; semantic fabrication remains unreviewed |
| `uncertainty_on_unanswerable_requests` | **PASS** | Phase 3 C answer uncertainty behavior; semantic fabrication remains unreviewed |
| `conflicting_evidence_handling` | **PASS** | structured conflict/precedence status and provenance; semantic claim support is NOT VERIFIED |
| `corrections_and_stale_protection` | **PASS** | local audit record |
| `provider_and_retrieval_failures` | **PASS** | local audit record |
| `citation_id_validity` | **PASS** | deterministic provenance only |
| `semantic_claim_support` | **NOT VERIFIED** | local audit record |
| `supported_answer_coverage` | **PASS** | local substring coverage, not semantic entailment |
| `guide_multi_intent_70_percent_target` | **PASS** | local provisional-label measurement; not official benchmark validation |
| `guide_citation_support_85_percent_target` | **NOT VERIFIED** | local audit record |
| `real_backend_validation` | **NOT VERIFIED** | local audit record |
| `official_benchmark_validation` | **NOT VERIFIED** | local audit record |
| `phase4_selective_claim_updates` | **NOT VERIFIED** | intentionally not implemented or evaluated in Phase 3 |
| `engineering_implementation` | **PASS** | local runtime implementation and audit trace execution |
| `local_measured_quality` | **PASS** | synthetic lexical/mock fixture with provisional external labels |

### Validation domains

| Domain | Status | Evidence boundary |
|---|---|---|
| `engineering_implementation` | **PASS** | dedicated A/B/C runner completed and trace completeness was checked |
| `local_measured_quality` | **PASS** | provisional synthetic corpus/label measurements; see per-capability results |
| `real_backend_validation` | **NOT VERIFIED** | no usable dense model/index or live provider run was available |
| `official_benchmark_validation` | **NOT VERIFIED** | official corpus, labels, harness, and organiser API are unavailable |

## Phase 4 handoff

The interface-only handoff is [docs/phase4-handoff.md](../docs/phase4-handoff.md). It covers intent IDs, typed constraints, claim records, evidence dependencies, revision/supersession records, and session-state boundaries. No Phase 4 selective claim-update implementation was added.

The machine-readable report contains complete per-case condition records, intent matching, per-intent rankings/fusion decisions, evidence dependencies, claim records, latency, calls/tokens/errors/cost, trace completeness, and label-review provenance.
