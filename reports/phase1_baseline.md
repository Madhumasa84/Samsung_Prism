# FlowContext Phase 1 measured baseline

Measured on 2026-09-15 in the supplied workspace with Python 3.11.15. This is
an engineering report on the local synthetic fixture, not a Samsung PRISM
competition result.

## Asset and execution provenance

| Field | Value |
| --- | --- |
| Official corpus/evaluation assets | Missing from supplied workspace |
| Corpus | 3 synthetic documents, 4 chunks |
| Index | `index-03e98680ae845e95` |
| Corpus ID | `corpus-3762c90f2912589e` |
| Source fingerprint | `3762c90f2912589e2bed44f3311ad369f406264c89ccefbb90b8561e76c2a234` |
| Build fingerprint | `03e98680ae845e95e42aefb18868af26debc6e1245e1cbb4c023b2e763b7ab9e` |
| Index backend | Explicit lexical diagnostic backend |
| Generation | `flowcontext.mock:mock-grounded-v1` |
| Execution mode | Accelerated replay; source timestamps retained separately |
| Warm/cold condition | Cold CLI process; index/retriever/provider initialized once and reused across cases; process startup excluded from per-case latency |
| Hardware | Linux WSL2, x86_64, 12 CPUs; GPU not detected by Phase 1 |
| Code revision | Unavailable: supplied directory is not a Git worktree |
| Label status | `provisional_generated`; not human-verified |

The reproducible index command measured `0.000532 s` for this three-document
fixture. The index was then used by one fixed `--split all` suite run with five
development and five held-out cases. No held-out case was used for tuning.

## Measured local result

The full machine-readable report is
[`phase1_baseline.json`](phase1_baseline.json). The `mock` and `fixture`
sections intentionally describe the same run from provider and corpus
provenance perspectives; they are not two independent benchmark samples.

| Section | Status | Cases | Case pass rate | Trace complete | Cost |
| --- | --- | ---: | ---: | ---: | --- |
| Mock provider | measured | 10 | 1.0 | 10/10 | unavailable |
| Synthetic fixture | measured | 10 | 1.0 | 10/10 | unavailable |
| Real model | not verified | 0 | not measured | 0 | unavailable |

For the measured mock/fixture run:

| Metric | Value |
| --- | ---: |
| Labelled retrieval cases | 8 |
| Recall@1 | 0.9167 |
| Recall@3 | 0.9583 |
| Recall@5 | 1.0 |
| MRR | 1.0 |
| Citation-ID validity rate | 1.0 over 19 emitted citation IDs |
| Answerable cases | 8 |
| Answerable abstention rate | 0.0 |
| Unanswerable cases | 2 |
| Unanswerable abstention rate | 1.0 |
| Retrieval latency | n=10, p50 0.0221 ms, p95 0.0292 ms |
| Complete-answer latency | n=10, p50 0.3047 ms, p95 0.4946 ms |
| Estimated token usage | 1,921 total (506 input, 1,415 output) |
| Cost | unavailable; mock usage is estimated and no real price config was present |
| Early retrieval events | 0; intentionally absent in Phase 1 |
| Semantic claim support | not evaluated |
| Error cases | 0 |

The retrieval metrics use only the eight cases with explicit provisional
relevance labels. The two unsupported cases are excluded from Recall/MRR and
are measured in the abstention metrics. Percentiles use inclusive linear
interpolation. Complete-answer latency is not time-to-first-token because the
provider is non-streaming.

## Evaluation and review status

The local cases cover simple questions, a compound request, unsupported
requests, a cumulative correction, and formatting turns. Every case contains
its transcript, answerability, expected evidence IDs, support rubric, and
deferred-capability annotations. Development and held-out files are separate;
related scenarios stay in one split.

Claim support is `not_evaluated`. The required review procedure is to read each
claim with its cited chunk(s) and label it supported, unsupported, or uncertain.
No human review and no model judge were run. Citation-ID validity therefore
does not imply factual grounding.

## Commands actually used for the report

```bash
./.venv/bin/python -m flowcontext.cli build-index \
  --input data/synthetic/documents.jsonl \
  --output /tmp/flowcontext-phase1-baseline.GOeNEh/index.json \
  --backend lexical --source-kind synthetic_fixture
./.venv/bin/python -m flowcontext.cli evaluate-suite \
  --split all --corpus /tmp/flowcontext-phase1-baseline.GOeNEh/index.json \
  --backend lexical --execution-mode accelerated \
  --output reports/phase1_baseline.json
```

The clean-install, unit, smoke, container, and dense-blocker checks are
recorded in the completion handoff and AI-assistance log.

## Blockers and interpretation

* Official corpus, official evaluation data/labels, organiser schema/API,
  scoring harness, and performance thresholds are missing.
* Real dense retrieval was not executed because the optional package and pinned
  model weights were unavailable. The lexical backend was selected explicitly;
  it is not a dense substitute.
* Real-provider generation was not executed because credentials/model access
  were unavailable. Cost remains `unavailable`.
* Human claim-support review was not performed.

This report establishes engineering readiness for the Phase 1 interfaces and
offline workflow only. It does not establish performance readiness on official
assets. Phase 2 streaming control, early retrieval, decomposition, and
selective answer updates were not started.
