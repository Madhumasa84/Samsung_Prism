"""JSONL corpus ingestion and deterministic paragraph/size chunking."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

from pydantic import ValidationError

from .contracts import (
    Chunk,
    ChunkingConfig,
    CorpusIndex,
    Document,
    DocumentInput,
    EmbeddingConfig,
    IndexManifest,
    SourceVersion,
)

if TYPE_CHECKING:
    from .embeddings import EmbeddingProvider


class IngestionError(ValueError):
    """Raised when a corpus cannot be validated or chunked safely."""


class UnsupportedCorpusFormatError(IngestionError):
    """Raised when no loader is registered for a corpus input format."""


class StaleIndexError(IngestionError):
    """Raised when an existing index does not match requested source/config."""


SUPPORTED_CORPUS_SUFFIXES = {".jsonl"}


def normalize_text(text: str) -> str:
    """Normalize line endings while retaining meaningful content for hashing."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise IngestionError("document text must not be blank")
    return normalized


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_document_inputs(path: Path) -> list[DocumentInput]:
    """Load the only supplied corpus format: one JSON document per JSONL line."""

    if not path.is_file():
        raise IngestionError(f"corpus input does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_CORPUS_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_CORPUS_SUFFIXES))
        raise UnsupportedCorpusFormatError(
            f"unsupported corpus format {path.suffix or '<no extension>'!r} for {path}; "
            f"supported formats: {supported}"
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise IngestionError(f"corpus input is not valid UTF-8: {path}") from exc
    documents: list[DocumentInput] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
            document_input = DocumentInput.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise IngestionError(f"invalid document on line {line_number} of {path}: {exc}") from exc
        if document_input.document_id in seen_ids:
            raise IngestionError(f"duplicate document_id on line {line_number}: {document_input.document_id}")
        seen_ids.add(document_input.document_id)
        documents.append(document_input)
    if not documents:
        raise IngestionError(f"corpus input contains no documents: {path}")
    return documents


def _overlap_tail(
    words: list[str],
    overlap_chars: int,
    next_word: str,
    max_chars: int,
) -> list[str]:
    """Return the largest readable suffix that still leaves room for next_word."""

    if overlap_chars <= 0 or len(next_word) >= max_chars:
        return []
    available = min(overlap_chars, max_chars - len(next_word) - 1)
    if available <= 0:
        return []
    overlap: list[str] = []
    overlap_length = 0
    for prior_word in reversed(words):
        next_length = overlap_length + len(prior_word) + (1 if overlap else 0)
        if next_length > available:
            break
        overlap.insert(0, prior_word)
        overlap_length = next_length
    return overlap


def _split_by_size(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        if len(word) > max_chars:
            if current:
                pieces.append(" ".join(current))
                current = []
                current_length = 0
            pieces.extend(word[start : start + max_chars] for start in range(0, len(word), max_chars))
            continue
        proposed_length = current_length + len(word) + (1 if current else 0)
        if current and proposed_length > max_chars:
            pieces.append(" ".join(current))
            current = _overlap_tail(current, overlap_chars, word, max_chars)
            current_length = len(" ".join(current))
            if current and current_length + 1 + len(word) > max_chars:
                current = []
                current_length = 0
        current.append(word)
        current_length += len(word) + (1 if current_length else 0)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(text: str, max_chars: int, overlap_chars: int = 0) -> list[str]:
    """Prefer paragraph boundaries and split oversized paragraphs on word boundaries."""

    config = ChunkingConfig(max_chars=max_chars, overlap_chars=overlap_chars)
    if not text.strip():
        raise IngestionError("cannot chunk blank text")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        chunks.extend(_split_by_size(paragraph, config.max_chars, config.overlap_chars))
    if any(not chunk.strip() for chunk in chunks):
        raise IngestionError("chunker produced an empty chunk")
    return chunks


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _section_hierarchy(document_input: DocumentInput) -> list[str]:
    if document_input.section_hierarchy:
        return list(document_input.section_hierarchy)
    return [document_input.section] if document_input.section else []


def _materialize_documents(inputs: Iterable[DocumentInput]) -> list[Document]:
    materialized_inputs = sorted(list(inputs), key=lambda item: item.document_id)
    if not materialized_inputs:
        raise IngestionError("cannot build an empty corpus index")
    documents: list[Document] = []
    seen_ids: set[str] = set()
    for document_input in materialized_inputs:
        if document_input.document_id in seen_ids:
            raise IngestionError(f"duplicate document_id: {document_input.document_id}")
        seen_ids.add(document_input.document_id)
        normalized_text = normalize_text(document_input.text)
        content_hash = sha256_text(normalized_text)
        documents.append(
            Document(
                document_id=document_input.document_id,
                source_location=document_input.source_location,
                source_version=document_input.source_version or f"sha256:{content_hash}",
                title=document_input.title,
                content_hash=content_hash,
                text=normalized_text,
                page_start=document_input.page_start,
                page_end=document_input.page_end,
                section=document_input.section,
                section_hierarchy=_section_hierarchy(document_input),
                metadata=document_input.metadata,
            )
        )
    return documents


def source_versions_for_documents(documents: Sequence[Document]) -> list[SourceVersion]:
    return [
        SourceVersion(
            document_id=document.document_id,
            source_location=document.source_location,
            source_version=document.source_version,
            content_hash=document.content_hash,
        )
        for document in documents
    ]


def source_fingerprint_for_documents(documents: Sequence[Document]) -> str:
    # Metadata is part of the indexed source: a title, section, page range, or
    # source annotation change must not silently reuse an old index even when
    # the extracted body text is unchanged.
    source_records = [
        {
            "document_id": document.document_id,
            "source_location": document.source_location,
            "source_version": document.source_version,
            "content_hash": document.content_hash,
            "title": document.title,
            "page_start": document.page_start,
            "page_end": document.page_end,
            "section": document.section,
            "section_hierarchy": document.section_hierarchy,
            "metadata": document.metadata,
        }
        for document in documents
    ]
    return sha256_text(_canonical_json(source_records))


def source_fingerprint_for_inputs(inputs: Iterable[DocumentInput]) -> str:
    return source_fingerprint_for_documents(_materialize_documents(inputs))


class CorpusIngestor:
    """Build a versioned corpus index with stable IDs and provenance."""

    def __init__(self, max_chars: int = 1200, overlap_chars: int = 0) -> None:
        self.chunking = ChunkingConfig(max_chars=max_chars, overlap_chars=overlap_chars)
        self.max_chars = self.chunking.max_chars
        self.overlap_chars = self.chunking.overlap_chars

    def ingest(
        self,
        inputs: Iterable[DocumentInput],
        source_kind: str = "unknown",
        *,
        source_path: Path | None = None,
        embedding_config: EmbeddingConfig | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> CorpusIndex:
        if source_kind not in {"official", "synthetic_fixture", "unknown"}:
            raise ValueError("source_kind must be official, synthetic_fixture, or unknown")
        documents = _materialize_documents(inputs)
        if embedding_provider is not None and embeddings is not None:
            raise IngestionError("provide either embedding_provider or precomputed embeddings, not both")
        if embedding_provider is not None:
            provider_config = embedding_provider.config
            if embedding_config is not None and provider_config != embedding_config:
                raise IngestionError("embedding provider config does not match requested embedding config")
            embedding_config = provider_config
        effective_embedding = embedding_config or EmbeddingConfig(
            backend="lexical",
            provider="flowcontext.lexical",
            normalize=False,
            download_required=False,
        )
        if effective_embedding.backend in {"dense", "mock"} and embeddings is None and embedding_provider is None:
            raise IngestionError("dense index build requires embeddings from an embedding provider")
        chunk_specs: list[tuple[Document, int, str]] = []
        for document in documents:
            chunk_specs.extend(
                (document, ordinal, chunk_value)
                for ordinal, chunk_value in enumerate(
                    chunk_text(document.text, self.max_chars, self.overlap_chars)
                )
            )
        if embedding_provider is not None:
            embeddings = embedding_provider.embed([chunk_value for _, _, chunk_value in chunk_specs])
        if embeddings is not None and len(embeddings) != len(chunk_specs):
            raise IngestionError("embedding count does not match generated chunk count")
        chunks: list[Chunk] = []
        embedding_index = 0
        for document, ordinal, chunk_value in chunk_specs:
            vector = None
            if embeddings is not None:
                vector = list(embeddings[embedding_index])
                embedding_index += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}#chunk-{ordinal:04d}",
                    document_id=document.document_id,
                    source_location=document.source_location,
                    source_version=document.source_version,
                    content_hash=sha256_text(chunk_value),
                    text=chunk_value,
                    ordinal=ordinal,
                    title=document.title,
                    page_start=document.page_start,
                    page_end=document.page_end,
                    section=document.section,
                    section_hierarchy=document.section_hierarchy,
                    metadata=document.metadata,
                    embedding=vector,
                )
            )
        if not documents or not chunks:
            raise IngestionError("cannot build an empty corpus index")
        source_versions = source_versions_for_documents(documents)
        source_fingerprint = source_fingerprint_for_documents(documents)
        build_fingerprint = sha256_text(
            _canonical_json(
                {
                    "schema_version": "flowcontext.index.v1",
                    "source_kind": source_kind,
                    "source_format": "jsonl",
                    "source_fingerprint": source_fingerprint,
                    "chunking": self.chunking.model_dump(mode="json"),
                    "embedding": effective_embedding.model_dump(mode="json"),
                }
            )
        )
        corpus_fingerprint = source_fingerprint[:16]
        corpus_id = f"corpus-{corpus_fingerprint}"
        manifest = IndexManifest(
            index_id=f"index-{build_fingerprint[:16]}",
            corpus_id=corpus_id,
            source_kind=source_kind,
            source_format="jsonl",
            source_path=str(source_path) if source_path else None,
            source_fingerprint=source_fingerprint,
            source_versions=source_versions,
            document_count=len(documents),
            chunk_count=len(chunks),
            chunking=self.chunking,
            embedding=effective_embedding,
            build_fingerprint=build_fingerprint,
        )
        return CorpusIndex(
            corpus_id=corpus_id,
            source_kind=source_kind,
            manifest=manifest,
            documents=documents,
            chunks=chunks,
        )


def write_index(path: Path, index: CorpusIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(index.model_dump_json(indent=2) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_index(path: Path) -> CorpusIndex:
    if not path.is_file():
        raise IngestionError(f"corpus index does not exist: {path}")
    try:
        return CorpusIndex.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise IngestionError(f"invalid corpus index {path}: {exc}") from exc


def assert_index_matches_source(
    index: CorpusIndex,
    source_path: Path,
    *,
    max_chars: int,
    overlap_chars: int,
) -> None:
    """Reject a stale index before a caller can query it."""

    inputs = load_document_inputs(source_path)
    source_fingerprint = source_fingerprint_for_inputs(inputs)
    expected_chunking = ChunkingConfig(max_chars=max_chars, overlap_chars=overlap_chars)
    if index.manifest.source_fingerprint != source_fingerprint:
        raise StaleIndexError(
            "index source_fingerprint does not match the configured corpus; rebuild with --force"
        )
    if index.manifest.chunking != expected_chunking:
        raise StaleIndexError(
            "index chunking configuration does not match the requested configuration; rebuild with --force"
        )
