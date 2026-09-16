# Phase 1 indexing notes

## Source boundary

The only supplied corpus-like source is the synthetic `.jsonl` fixture. The
loader accepts one validated document object per line and rejects non-`.jsonl`
extensions, malformed JSON, duplicate IDs, blank text, and missing required
document provenance. It does not attempt to parse the guide PDF as knowledge
content.

## Determinism

Documents are sorted by their supplied stable `document_id`. Document hashes are
SHA-256 of canonical text (line endings normalized and outer whitespace removed).
Chunks prefer blank-line paragraph boundaries, then split oversized paragraphs at
word boundaries using configurable `max_chars` and `overlap_chars`. Chunk IDs are
`{document_id}#chunk-{ordinal:04d}`; source location, source version, section
hierarchy, page range, and text remain on each chunk.

The source fingerprint covers the document ID, location, version, text hash,
title, page range, section hierarchy, and JSON metadata. Thus changes to indexed
provenance metadata are stale-index changes too, not just body-text edits.

The manifest fingerprint covers source versions/content hashes, chunking settings,
and the full embedding configuration. It intentionally does not include build
time or the local source path, so identical content/configuration has identical
IDs and fingerprints across machines.

## Embedding backends

The default index/retrieval backend is dense cosine retrieval. The provider
interface is deliberately small: `EmbeddingProvider.config` plus
`EmbeddingProvider.embed(texts)`. The production provider is lazy-loaded
Sentence Transformers on CPU, using:

- model: `sentence-transformers/all-MiniLM-L6-v2`
- revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- licence: `apache-2.0`
- output: 384 normalized dimensions

The `sentence-transformers` optional dependency and Hub model weights are not
bundled. A dense build therefore requires `uv sync --extra dense` (or an
equivalent install) and a model download/cache. `--local-files-only` can enforce
offline model loading. A load or encode error is surfaced as a dense error and
does not invoke the lexical retriever.

Lexical retrieval is a separately selectable diagnostic backend. The hash
embedding provider is a local mock for plumbing tests only; neither it nor the
synthetic fixture supports a competition-quality claim.

## Safe reuse

`build-index` computes the source/build fingerprint before writing. An identical
existing index is reported `up_to_date`; a changed source, chunking setting, or
embedding configuration fails unless `--force` is explicitly supplied. Writes
use a temporary file and atomic replacement. Query commands accept `--source` to
recompute the source fingerprint and reject stale indexes before retrieving.

## CLI surface

The implemented Phase 1 commands are:

- `flowcontext inspect-corpus --input <source.jsonl>` for extraction health and
  counts without embeddings.
- `flowcontext build-index --input <source.jsonl> --backend dense|lexical|mock`
  for deterministic index construction.
- `flowcontext retrieve --index <index.json> --backend <backend> --query <text>`
  for ranked, provenance-bearing snippets.

`--backend lexical` is explicit and diagnostic. It is not selected implicitly
when dense loading fails. `mock` is only a local plumbing check and is not a
competition model.
