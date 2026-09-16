# Asset inventory and provenance

Inventory performed before implementation on 2026-09-15 (Asia/Calcutta):

| Category | Available asset | Status and interpretation |
| --- | --- | --- |
| Competition guide | [`Theme 4 Guide_RAG.pdf`](../Theme%204%20Guide_RAG.pdf) | Present. Six-page scanned PDF; SHA-256 `319adef6b15661e62b133b4805b4399cee58ef3e11012c054051b6974fd829db`. |
| Official knowledge corpus | None | **Missing.** No documents, source export, chunk manifest, or organiser corpus identifier was supplied. |
| Official evaluation assets | None | **Missing.** No held-out prompts, transcript replay set, labels, benchmark runner, thresholds file, or scoring harness was supplied. |
| Official schemas/API | None | **Missing.** The guide's page-4 JSON is an illustrative “Structured Output Event Record”, not identified as a required API/schema. |
| Example IDs in guide | `Doc_2`, `Doc_31`, `Doc_09` | Examples only; they were not treated as real documents or fabricated into the corpus. |
| Engineering fixture | [`data/synthetic/`](../data/synthetic/) | Clearly labelled synthetic corpus/transcript data for offline tests only; its results are not competition performance. |
| Local evaluation set | [`data/evaluation/`](../data/evaluation/) | Clearly labelled development/held-out cases grounded in the synthetic fixture; labels are provisional and not human-verified. |

The guide was inspected as a rendered scanned document. It specifies corpus
isolation, grounding, session-bound state, telemetry, and the Phase 1 roadmap,
but it does not provide the data needed to claim official benchmark coverage.

## Runtime and model assets

The application targets Python 3.11 and has a core Pydantic dependency plus a
locked optional `dense` extra for Sentence Transformers. The core environment
was installed and checked locally; the optional dense package, its transitive
CPU/runtime stack, and model weights are not installed or bundled in this
workspace. No local model cache was found during inspection.

The selected dense configuration is `sentence-transformers/all-MiniLM-L6-v2`,
pinned to revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with Apache-2.0 metadata and 384
dimensions. The model card and revision are configuration/documentation
references, not downloaded assets. A dense build requires installing the
optional extra and allowing the pinned snapshot to be downloaded, or providing
an equivalent local cache with `FLOWCONTEXT_EMBEDDING_LOCAL_FILES_ONLY=true`.
