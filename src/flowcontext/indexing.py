"""Index build orchestration on top of the existing ingestion package."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from .config import Settings
from .contracts import EmbeddingConfig, CorpusIndex
from .embeddings import (
    DEFAULT_MODEL_CARD_URL,
    DEFAULT_MODEL_LICENSE,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    EmbeddingProvider,
    HashEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from .ingestion import (
    CorpusIngestor,
    IngestionError,
    StaleIndexError,
    load_document_inputs,
    load_index,
    write_index,
)


RetrievalBackend = Literal["dense", "lexical", "mock"]


def embedding_config_for_settings(settings: Settings, backend: RetrievalBackend) -> EmbeddingConfig:
    if backend == "dense":
        return EmbeddingConfig(
            backend="dense",
            provider="sentence-transformers",
            model_name=settings.embedding_model,
            revision=settings.embedding_revision,
            license=settings.embedding_license,
            dimensions=settings.embedding_dimensions,
            normalize=True,
            model_card_url=settings.embedding_model_card_url or DEFAULT_MODEL_CARD_URL,
            download_required=True,
        )
    if backend == "mock":
        return EmbeddingConfig(
            backend="mock",
            provider="flowcontext.hash_embedding",
            model_name="hash-bow-embedding",
            revision="local",
            license="internal-test-only",
            dimensions=settings.embedding_dimensions,
            normalize=True,
            download_required=False,
        )
    return EmbeddingConfig(
        backend="lexical",
        provider="flowcontext.lexical",
        normalize=False,
        download_required=False,
    )


def provider_for_settings(
    settings: Settings,
    backend: RetrievalBackend,
) -> EmbeddingProvider | None:
    if backend == "lexical":
        return None
    if backend == "mock":
        return HashEmbeddingProvider(dimensions=settings.embedding_dimensions)
    return SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model or DEFAULT_MODEL_NAME,
        revision=settings.embedding_revision or DEFAULT_MODEL_REVISION,
        license_name=settings.embedding_license or DEFAULT_MODEL_LICENSE,
        model_card_url=settings.embedding_model_card_url or DEFAULT_MODEL_CARD_URL,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
    )


def build_index_from_source(
    source_path: Path,
    *,
    settings: Settings,
    backend: RetrievalBackend | None = None,
    force: bool = False,
    output_path: Path | None = None,
) -> tuple[CorpusIndex, float, str]:
    """Build or safely reuse an index, returning index, elapsed seconds, and status."""

    selected_backend: RetrievalBackend = backend or settings.retrieval_backend
    started = time.perf_counter()
    inputs = load_document_inputs(source_path)
    provider = provider_for_settings(settings, selected_backend)
    embedding_config = provider.config if provider is not None else embedding_config_for_settings(settings, selected_backend)
    if (
        selected_backend == "dense"
        and embedding_config.dimensions != settings.embedding_dimensions
    ):
        raise IngestionError(
            "loaded dense model dimensions do not match FLOWCONTEXT_EMBEDDING_DIMENSIONS="
            f"{settings.embedding_dimensions}"
        )
    index = CorpusIngestor(
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
    ).ingest(
        inputs,
        settings.corpus_status,
        source_path=source_path,
        embedding_config=embedding_config,
        embedding_provider=provider,
    )
    elapsed = time.perf_counter() - started
    if output_path is None:
        return index, elapsed, "built"
    had_existing_output = output_path.exists()
    if had_existing_output:
        existing = load_index(output_path)
        if existing.manifest.build_fingerprint == index.manifest.build_fingerprint:
            return existing, elapsed, "up_to_date"
        if not force:
            raise StaleIndexError(
                f"existing index {output_path} is stale for the requested source/config; "
                "rerun with --force to rebuild safely"
            )
    write_index(output_path, index)
    return index, elapsed, "rebuilt" if had_existing_output else "built"


def inspect_corpus(source_path: Path, *, settings: Settings) -> dict[str, object]:
    """Inspect supported source content without embeddings or retrieval."""

    inputs = load_document_inputs(source_path)
    index = CorpusIngestor(
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
    ).ingest(inputs, settings.corpus_status, source_path=source_path)
    return {
        "status": "ok",
        "source_path": str(source_path),
        "source_format": "jsonl",
        "source_kind": settings.corpus_status,
        "documents": len(index.documents),
        "chunks": len(index.chunks),
        "characters": sum(len(document.text) for document in index.documents),
        "documents_with_titles": sum(document.title is not None for document in index.documents),
        "documents_with_sections": sum(bool(document.section_hierarchy) for document in index.documents),
        "documents_with_page_locations": sum(
            document.page_start is not None or document.page_end is not None
            for document in index.documents
        ),
        "source_fingerprint": index.manifest.source_fingerprint,
        "chunking": index.manifest.chunking.model_dump(mode="json"),
        "embedding": "not indexed (inspection only)",
    }
