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

## Guide targets

- Multi-intent identification target (70%): **PASS** locally against provisional labels; development `1.0`, held-out `0.75`.
- Citation-support target (85%): **NOT VERIFIED**. Citation-ID validity is reported separately and does not establish semantic support.

## Split results

### Development

Cases: 13 (single 9, compound 4). Multi-intent exact identification: 4/4 = `1.0`; missed intents: 0; unnecessary extra intents: 0.

| Condition | Matched Recall@5 | Matched cases | Completed / all | Final-event→answer p50/p95 ms |
|---|---:|---:|---:|---:|
| A_original_baseline | 1.0 | 9 | 11/13 | 3.8753039998482564 / 86.08901460001988 |
| B_phase2_streaming_single_query | 1.0 | 9 | 11/13 | 4.494742000133556 / 88.24985979999809 |
| C_phase3_streaming_fused | 0.9074074074074073 | 9 | 10/13 | 7.524385000124312 / 92.27940760001701 |

Per-condition resources:
**A_original_baseline**
- Retrieval calls: 13; model calls: {'generation': 12, 'decomposition': 0, 'total': 12}
- Tokens: retrieval 96, decomposition 0, generation 5117, repair 0, verification 0, total 5213
- Errors: 2 events across 2 cases; cost: `unavailable` (unavailable)
**B_phase2_streaming_single_query**
- Retrieval calls: 14; model calls: {'generation': 12, 'decomposition': 0, 'total': 12}
- Tokens: retrieval 99, decomposition 0, generation 5117, repair 0, verification 0, total 5216
- Errors: 2 events across 2 cases; cost: `unavailable` (unavailable)
**C_phase3_streaming_fused**
- Retrieval calls: 14; model calls: {'generation': 12, 'decomposition': 14, 'total': 26}
- Tokens: retrieval 99, decomposition 2319, generation 2908, repair 0, verification 682, total 6008
- Errors: 3 events across 2 cases; cost: `unavailable` (unavailable)

Phase 3 complete-request evidence coverage (macro): `0.8333333333333334` over 12 cases.
Phase 3 per-intent retrieval Recall@1/@3/@5 (macro): `0.5333333333333333` / `0.8666666666666667` / `0.9333333333333333` over 15 applicable labelled intent records.
Phase 3 supported-answer coverage (provisional substring check): `2/15` = `0.13333333333333333`; this is not semantic support.
Phase 3 citation-ID validity: `10/10` emitted-citation runs = `1.0`; 3 runs emitted no citations; semantic support: **NOT VERIFIED**.
Phase 2/3 early retrieval and reuse: B started 1/1, useful reuse 0, stale accepted 0; C started 1/1, useful reuse 0, stale accepted 0.

### Held Out

Cases: 7 (single 3, compound 4). Multi-intent exact identification: 3/4 = `0.75`; missed intents: 1; unnecessary extra intents: 1.

| Condition | Matched Recall@5 | Matched cases | Completed / all | Final-event→answer p50/p95 ms |
|---|---:|---:|---:|---:|
| A_original_baseline | 1.0 | 5 | 7/7 | 5.158859999937704 / 95.18003099997252 |
| B_phase2_streaming_single_query | 1.0 | 5 | 6/7 | 5.751918999976624 / 29.6049851000589 |
| C_phase3_streaming_fused | 1.0 | 5 | 6/7 | 8.517489000041678 / 31.151270799955448 |

Per-condition resources:
**A_original_baseline**
- Retrieval calls: 7; model calls: {'generation': 7, 'decomposition': 0, 'total': 7}
- Tokens: retrieval 54, decomposition 0, generation 3050, repair 0, verification 0, total 3104
- Errors: 0 events across 0 cases; cost: `unavailable` (unavailable)
**B_phase2_streaming_single_query**
- Retrieval calls: 7; model calls: {'generation': 6, 'decomposition': 0, 'total': 6}
- Tokens: retrieval 54, decomposition 0, generation 2777, repair 0, verification 0, total 2831
- Errors: 1 events across 1 cases; cost: `unavailable` (unavailable)
**C_phase3_streaming_fused**
- Retrieval calls: 7; model calls: {'generation': 6, 'decomposition': 7, 'total': 13}
- Tokens: retrieval 54, decomposition 1029, generation 1819, repair 0, verification 510, total 3412
- Errors: 1 events across 1 cases; cost: `unavailable` (unavailable)

Phase 3 complete-request evidence coverage (macro): `0.8333333333333334` over 6 cases.
Phase 3 per-intent retrieval Recall@1/@3/@5 (macro): `0.4444444444444444` / `0.7777777777777778` / `0.8888888888888888` over 9 applicable labelled intent records.
Phase 3 supported-answer coverage (provisional substring check): `2/10` = `0.2`; this is not semantic support.
Phase 3 citation-ID validity: `6/6` emitted-citation runs = `1.0`; 1 runs emitted no citations; semantic support: **NOT VERIFIED**.
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
| `single_questions_not_decomposed` | **FAIL** | synthetic_fixture_provisional_labels |
| `compound_independent_questions` | **PASS** | synthetic_fixture_provisional_labels |
| `dependent_questions` | **PASS** | synthetic_fixture_provisional_labels |
| `shared_constraints` | **PASS** | synthetic_fixture_provisional_labels |
| `negation` | **PASS** | synthetic_fixture_provisional_labels |
| `comparisons` | **PASS** | synthetic_fixture_provisional_labels |
| `partial_answerability` | **PASS** | synthetic_fixture_provisional_labels |
| `entity_confusion` | **PASS** | synthetic_fixture_provisional_labels |
| `uncertainty_on_partial_requests` | **FAIL** | Phase 3 C answer uncertainty behavior; semantic fabrication remains unreviewed |
| `uncertainty_on_unanswerable_requests` | **FAIL** | Phase 3 C answer uncertainty behavior; semantic fabrication remains unreviewed |
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
| `local_measured_quality` | **FAIL** | synthetic lexical/mock fixture with provisional external labels |

### Validation domains

| Domain | Status | Evidence boundary |
|---|---|---|
| `engineering_implementation` | **PASS** | dedicated A/B/C runner completed and trace completeness was checked |
| `local_measured_quality` | **FAIL** | provisional synthetic corpus/label measurements; see per-capability results |
| `real_backend_validation` | **NOT VERIFIED** | no usable dense model/index or live provider run was available |
| `official_benchmark_validation` | **NOT VERIFIED** | official corpus, labels, harness, and organiser API are unavailable |

## Phase 4 handoff

The interface-only handoff is [docs/phase4-handoff.md](../docs/phase4-handoff.md). It covers intent IDs, typed constraints, claim records, evidence dependencies, revision/supersession records, and session-state boundaries. No Phase 4 selective claim-update implementation was added.

The machine-readable report contains complete per-case condition records, intent matching, per-intent rankings/fusion decisions, evidence dependencies, claim records, latency, calls/tokens/errors/cost, trace completeness, and label-review provenance.
