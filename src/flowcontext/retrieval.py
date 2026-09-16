"""Replaceable dense and lexical retrievers for the Phase 1 index."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Protocol

from .contracts import CorpusIndex, RetrievalHit
from .embeddings import EmbeddingProvider, provider_from_config


_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "for",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "with",
    }
)


class RetrievalError(RuntimeError):
    """Base error for retrieval/index compatibility failures."""


class IndexConfigurationError(RetrievalError):
    """Raised when a requested backend does not match the persisted index."""


class Retriever(Protocol):
    """Minimal retriever interface shared by dense and lexical implementations."""

    backend: str

    def search(self, query: str) -> list[RetrievalHit]:
        ...


def tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
        if token.lower() not in _STOP_WORDS
    }


def _hit_for_chunk(chunk, *, rank: int, score: float, method: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk.chunk_id,
        source_location=chunk.source_location,
        snippet_text=chunk.text,
        rank=rank,
        score=score,
        retrieval_method=method,
    )


class LexicalRetriever:
    """Rank chunks by unique query-term coverage as a diagnostic fallback."""

    backend = "lexical"
    method = "lexical_overlap"

    def __init__(self, index: CorpusIndex, top_k: int = 5) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.index = index
        self.top_k = top_k
        self._chunks = {chunk.chunk_id: chunk for chunk in index.chunks}
        self._chunk_tokens = {
            chunk.chunk_id: tokenize(
                " ".join(
                    [
                        chunk.title or "",
                        *chunk.section_hierarchy,
                        chunk.section or "",
                        chunk.text,
                    ]
                )
            )
            for chunk in index.chunks
        }

    def search(self, query: str) -> list[RetrievalHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scored: list[tuple[float, int, str]] = []
        for chunk in self.index.chunks:
            overlap = query_tokens & self._chunk_tokens[chunk.chunk_id]
            if not overlap:
                continue
            score = len(overlap) / len(query_tokens)
            scored.append((score, len(overlap), chunk.chunk_id))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [
            _hit_for_chunk(
                self._chunks[chunk_id],
                rank=rank,
                score=score,
                method=self.method,
            )
            for rank, (score, _, chunk_id) in enumerate(scored[: self.top_k], start=1)
        ]


class DenseRetriever:
    """Cosine retrieval over persisted chunk embeddings."""

    backend = "dense"
    method = "dense_cosine"
    expected_embedding_backend = "dense"

    def __init__(self, index: CorpusIndex, provider: EmbeddingProvider, top_k: int = 5) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if index.manifest.embedding.backend != self.expected_embedding_backend:
            raise IndexConfigurationError(
                f"{self.backend} retrieval requires an index built with "
                f"backend={self.expected_embedding_backend}; "
                f"this index uses {index.manifest.embedding.backend!r}"
            )
        if provider.config != index.manifest.embedding:
            raise IndexConfigurationError(
                "dense provider configuration does not exactly match the index embedding configuration"
            )
        if any(chunk.embedding is None for chunk in index.chunks):
            raise IndexConfigurationError("dense index contains a chunk without an embedding")
        self.index = index
        self.provider = provider
        self.top_k = top_k
        self._chunks = {chunk.chunk_id: chunk for chunk in index.chunks}

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left or not right:
            raise RetrievalError("embedding vectors must have the same non-zero dimension")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            raise RetrievalError("cannot calculate cosine similarity for a zero vector")
        return sum(left_value * right_value for left_value, right_value in zip(left, right)) / (
            left_norm * right_norm
        )

    def search(self, query: str) -> list[RetrievalHit]:
        if not query.strip():
            return []
        query_vector = self.provider.embed([query])[0]
        scored = [
            (self._cosine(query_vector, list(chunk.embedding or [])), chunk.chunk_id)
            for chunk in self.index.chunks
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            _hit_for_chunk(self._chunks[chunk_id], rank=rank, score=score, method=self.method)
            for rank, (score, chunk_id) in enumerate(scored[: self.top_k], start=1)
        ]


# Compatibility name retained for callers of the Phase 1 lexical baseline.
BaselineRetriever = LexicalRetriever


def make_retriever(
    index: CorpusIndex,
    *,
    backend: str = "dense",
    top_k: int = 5,
    provider: EmbeddingProvider | None = None,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> Retriever:
    """Select exactly one backend; dense failures are not converted to lexical."""

    if backend == "lexical":
        return LexicalRetriever(index, top_k=top_k)
    if backend == "dense":
        if index.manifest.embedding.backend != "dense":
            raise IndexConfigurationError(
                "dense retrieval requested but index embedding backend is "
                f"{index.manifest.embedding.backend!r}; rebuild a dense index"
            )
        effective_provider = provider or provider_from_config(
            index.manifest.embedding,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return DenseRetriever(index, effective_provider, top_k=top_k)
    if backend == "mock":
        if index.manifest.embedding.backend != "mock":
            raise IndexConfigurationError(
                "mock retrieval requested but index was not built with the mock embedding backend"
            )
        effective_provider = provider or provider_from_config(
            index.manifest.embedding,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return _DenseMockRetriever(index, effective_provider, top_k=top_k)
    raise IndexConfigurationError(f"unsupported retrieval backend {backend!r}")


class _DenseMockRetriever(DenseRetriever):
    """Dense cosine retrieval over the local hash provider; fixture/testing only."""

    backend = "mock"
    method = "mock_dense_cosine"
    expected_embedding_backend = "mock"
