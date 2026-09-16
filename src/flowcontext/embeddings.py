"""Replaceable embedding providers for corpus indexing and dense retrieval.

The optional Sentence Transformers provider is lazy-imported so lexical and
mock fixture checks remain usable when model dependencies or model weights are
not installed. Dense failures are raised to the caller; they are never silently
converted into lexical retrieval.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Protocol, Sequence

from .contracts import EmbeddingConfig


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_MODEL_LICENSE = "apache-2.0"
DEFAULT_MODEL_DIMENSIONS = 384
DEFAULT_MODEL_CARD_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(RuntimeError):
    """Base error for provider loading and embedding failures."""


class DenseRetrievalUnavailable(EmbeddingError):
    """Raised when the selected dense provider cannot be loaded or used."""


class EmbeddingProvider(Protocol):
    """Minimal interface allowing providers to be replaced without retriever changes."""

    @property
    def config(self) -> EmbeddingConfig:
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class HashEmbeddingProvider:
    """Deterministic local mock used only for fixture/index plumbing tests."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 1:
            raise ValueError("mock embedding dimensions must be positive")
        self._dimensions = dimensions
        self._config = EmbeddingConfig(
            backend="mock",
            provider="flowcontext.hash_embedding",
            model_name="hash-bow-embedding",
            revision="local",
            license="internal-test-only",
            dimensions=dimensions,
            normalize=True,
            download_required=False,
        )
        self._token_pattern = re.compile(r"[\w]+", re.UNICODE)

    @property
    def config(self) -> EmbeddingConfig:
        return self._config

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if not text.strip():
                raise EmbeddingError("cannot embed blank text")
            vector = [0.0] * self._dimensions
            tokens = {token.casefold() for token in self._token_pattern.findall(text)}
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                first = int.from_bytes(digest[:8], "big") % self._dimensions
                second = int.from_bytes(digest[8:16], "big") % self._dimensions
                vector[first] += 1.0
                vector[second] += 0.5
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0:
                raise EmbeddingError("mock embedding produced a zero vector")
            vectors.append([value / norm for value in vector])
        return vectors


class SentenceTransformerEmbeddingProvider:
    """CPU Sentence Transformers provider pinned to an explicit Hub revision."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        revision: str = DEFAULT_MODEL_REVISION,
        license_name: str = DEFAULT_MODEL_LICENSE,
        model_card_url: str = DEFAULT_MODEL_CARD_URL,
        cache_dir: Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DenseRetrievalUnavailable(
                "dense backend requires optional dependency 'sentence-transformers'; "
                "install the dense extra and download the pinned model before building a dense index"
            ) from exc

        load_kwargs: dict[str, Any] = {
            "device": "cpu",
            "revision": revision,
        }
        if cache_dir is not None:
            load_kwargs["cache_folder"] = str(cache_dir)
        if local_files_only:
            load_kwargs["local_files_only"] = True
        try:
            self._model = SentenceTransformer(model_name, **load_kwargs)
        except Exception as exc:
            mode = "local cache" if local_files_only else "model download/cache"
            raise DenseRetrievalUnavailable(
                f"could not load dense model {model_name!r} at revision {revision!r} from {mode}: {exc}"
            ) from exc

        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise DenseRetrievalUnavailable("dense model did not report an embedding dimension")
        self._config = EmbeddingConfig(
            backend="dense",
            provider="sentence-transformers",
            model_name=model_name,
            revision=revision,
            license=license_name,
            dimensions=int(dimension),
            normalize=True,
            model_card_url=model_card_url,
            download_required=True,
        )

    @property
    def config(self) -> EmbeddingConfig:
        return self._config

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise EmbeddingError("cannot embed an empty batch or blank text")
        try:
            encoded = self._model.encode(
                list(texts),
                normalize_embeddings=self._config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vectors = encoded.tolist()
        except Exception as exc:
            raise DenseRetrievalUnavailable(f"dense embedding failed: {exc}") from exc
        if len(vectors) != len(texts):
            raise EmbeddingError("embedding provider returned the wrong batch size")
        if any(
            len(vector) != self._config.dimensions
            or any(not math.isfinite(float(value)) for value in vector)
            for vector in vectors
        ):
            raise EmbeddingError("embedding provider returned an invalid vector")
        return [[float(value) for value in vector] for vector in vectors]


def provider_from_config(
    config: EmbeddingConfig,
    *,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> EmbeddingProvider:
    """Construct exactly the requested provider; no fallback is performed."""

    if config.backend == "dense":
        if config.provider != "sentence-transformers":
            raise EmbeddingError(f"unsupported dense embedding provider {config.provider!r}")
        return SentenceTransformerEmbeddingProvider(
            model_name=config.model_name or DEFAULT_MODEL_NAME,
            revision=config.revision or DEFAULT_MODEL_REVISION,
            license_name=config.license or DEFAULT_MODEL_LICENSE,
            model_card_url=config.model_card_url or DEFAULT_MODEL_CARD_URL,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    if config.backend == "mock":
        return HashEmbeddingProvider(dimensions=config.dimensions or 64)
    raise EmbeddingError(f"backend {config.backend!r} does not use an embedding provider")
