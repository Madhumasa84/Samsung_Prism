# Synthetic engineering fixture

This directory is intentionally **not** the Samsung PRISM Theme 4 knowledge
corpus or an organiser benchmark. The workspace supplied for Phase 1 contained
only `Theme 4 Guide_RAG.pdf`; no official corpus, replay set, labels, or scoring
harness were present.

These JSONL files are small, invented fixtures used only to smoke-test contracts,
deterministic ingestion, complete-utterance replay, citation-ID validation,
mock generation, and the CLI offline. Fixture results must not be reported as
competition performance or semantic grounding quality.

`phase3_documents.jsonl` is a larger synthetic fixture for the dedicated Phase
3 audit. It contains 9 documents and is indexed as 10 chunks so the audit's
 `k=5` retrieval window is non-trivial. It intentionally includes current and
 draft policy evidence, positive/negative catering evidence, a same-named
 different-city entity, and unrelated distractors. The documents and all
 Phase 3 labels remain provisional engineering data.

`phase4_documents.jsonl` is a separate 12-document/13-chunk fixture for the
session-state, dependency-invalidation, answer-version, and matched
full-versus-selective evaluation. It includes venue capacity/entity/price
branches, policy conflicts, unsupported information needs, and distractors.
Its documents are synthetic and fixture-only; the lexical index and mock
provider exercise pipeline behavior, not real-model semantic quality.
