"""Configuration loading without credential-bearing dependencies."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    """Validated runtime configuration for the Phase 1/2 CLI."""

    model_config = ConfigDict(extra="forbid")

    app_name: str = "flowcontext-phase1"
    corpus_path: Path = Path("data/synthetic/documents.jsonl")
    transcript_path: Path = Path("data/synthetic/transcript.jsonl")
    evaluation_path: Path = Path("data/synthetic/evaluation.jsonl")
    evaluation_development_path: Path = Path("data/evaluation/development.jsonl")
    evaluation_held_out_path: Path = Path("data/evaluation/held_out.jsonl")
    streaming_evaluation_development_path: Path = Path("data/evaluation/streaming_development.jsonl")
    streaming_evaluation_held_out_path: Path = Path("data/evaluation/streaming_held_out.jsonl")
    artifacts_dir: Path = Path("artifacts")
    index_path: Path = Path("artifacts/corpus-index.json")
    corpus_status: Literal["official", "synthetic_fixture", "unknown"] = "synthetic_fixture"
    retrieval_backend: Literal["dense", "lexical", "mock"] = "dense"
    retrieval_top_k: int = Field(default=5, ge=1, le=100)
    chunk_max_chars: int = Field(default=1200, ge=1, le=100_000)
    chunk_overlap_chars: int = Field(default=0, ge=0, le=99_999)
    model_identity: str = "baseline.extractive.v1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    embedding_license: str = "apache-2.0"
    embedding_dimensions: int = Field(default=384, ge=1)
    embedding_model_card_url: str = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_dir: Path | None = None
    embedding_local_files_only: bool = False
    generation_backend: Literal["mock", "openai_compatible"] = "mock"
    generation_provider: str = Field(default="flowcontext.mock", min_length=1)
    generation_model: str = Field(default="mock-grounded-v1", min_length=1)
    generation_base_url: str = Field(default="https://api.openai.com/v1", min_length=1)
    generation_api_key_env: str = Field(default="FLOWCONTEXT_GENERATION_API_KEY", min_length=1)
    generation_timeout_s: float = Field(default=30.0, ge=0.01, le=120.0, allow_inf_nan=False)
    generation_max_retries: int = Field(default=2, ge=0, le=3)
    generation_retry_backoff_s: float = Field(default=0.5, ge=0.0, le=10.0, allow_inf_nan=False)
    generation_max_repair_attempts: int = Field(default=1, ge=0, le=2)
    generation_max_output_tokens: int = Field(default=600, ge=1, le=4096)
    generation_input_price_per_million: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    generation_output_price_per_million: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    # Phase 3 retrieval settings are opt-in at the caller boundary.  The
    # existing retrieval_backend remains the Phase 1/2 index/backend choice;
    # this mode controls how a multi-intent request consumes that index.
    multi_intent_retrieval_mode: Literal["dense", "lexical", "hybrid", "mock"] = "dense"
    multi_intent_context_budget_tokens: int = Field(default=1200, ge=1, le=100_000)
    multi_intent_max_workers: int = Field(default=4, ge=1, le=32)
    multi_intent_rrf_k: int = Field(default=60, ge=1, le=10_000)
    multi_intent_reranking_enabled: bool = False
    streaming_debounce_source_s: float = Field(default=0.35, ge=0.0, le=30.0, allow_inf_nan=False)
    streaming_min_query_chars: int = Field(default=8, ge=1, le=1000)
    streaming_min_topic_terms: int = Field(default=1, ge=1, le=20)
    streaming_min_new_content_terms: int = Field(default=1, ge=1, le=20)
    streaming_retrieve_on_correction: bool = True
    streaming_retrieve_on_constraint_change: bool = True
    streaming_duplicate_query_suppression: bool = True
    streaming_final_bypasses_debounce: bool = True
    streaming_max_concurrency: int = Field(default=2, ge=1, le=16)
    streaming_max_pending_requests: int = Field(default=4, ge=1, le=64)
    streaming_request_timeout_s: float = Field(default=5.0, ge=0.01, le=120.0, allow_inf_nan=False)
    streaming_max_retries: int = Field(default=0, ge=0, le=3)
    streaming_retry_backoff_s: float = Field(default=0.05, ge=0.0, le=10.0, allow_inf_nan=False)
    streaming_max_total_requests: int = Field(default=16, ge=1, le=256)
    streaming_final_wait_timeout_s: float = Field(default=5.0, ge=0.01, le=120.0, allow_inf_nan=False)
    streaming_cancel_on_supersession: bool = True

    @model_validator(mode="after")
    def chunk_overlap_is_valid(self) -> Settings:
        if self.chunk_overlap_chars >= self.chunk_max_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_max_chars")
        if self.generation_backend == "openai_compatible":
            if not self.generation_base_url.startswith(("http://", "https://")):
                raise ValueError("generation_base_url must use http:// or https://")
        return self


_ENV_TO_FIELD = {
    "FLOWCONTEXT_APP_NAME": "app_name",
    "FLOWCONTEXT_CORPUS_PATH": "corpus_path",
    "FLOWCONTEXT_TRANSCRIPT_PATH": "transcript_path",
    "FLOWCONTEXT_EVALUATION_PATH": "evaluation_path",
    "FLOWCONTEXT_EVALUATION_DEVELOPMENT_PATH": "evaluation_development_path",
    "FLOWCONTEXT_EVALUATION_HELD_OUT_PATH": "evaluation_held_out_path",
    "FLOWCONTEXT_STREAMING_EVALUATION_DEVELOPMENT_PATH": "streaming_evaluation_development_path",
    "FLOWCONTEXT_STREAMING_EVALUATION_HELD_OUT_PATH": "streaming_evaluation_held_out_path",
    "FLOWCONTEXT_ARTIFACTS_DIR": "artifacts_dir",
    "FLOWCONTEXT_INDEX_PATH": "index_path",
    "FLOWCONTEXT_CORPUS_STATUS": "corpus_status",
    "FLOWCONTEXT_RETRIEVAL_BACKEND": "retrieval_backend",
    "FLOWCONTEXT_RETRIEVAL_TOP_K": "retrieval_top_k",
    "FLOWCONTEXT_CHUNK_MAX_CHARS": "chunk_max_chars",
    "FLOWCONTEXT_CHUNK_OVERLAP_CHARS": "chunk_overlap_chars",
    "FLOWCONTEXT_MODEL_IDENTITY": "model_identity",
    "FLOWCONTEXT_EMBEDDING_MODEL": "embedding_model",
    "FLOWCONTEXT_EMBEDDING_REVISION": "embedding_revision",
    "FLOWCONTEXT_EMBEDDING_LICENSE": "embedding_license",
    "FLOWCONTEXT_EMBEDDING_DIMENSIONS": "embedding_dimensions",
    "FLOWCONTEXT_EMBEDDING_MODEL_CARD_URL": "embedding_model_card_url",
    "FLOWCONTEXT_EMBEDDING_CACHE_DIR": "embedding_cache_dir",
    "FLOWCONTEXT_EMBEDDING_LOCAL_FILES_ONLY": "embedding_local_files_only",
    "FLOWCONTEXT_GENERATION_BACKEND": "generation_backend",
    "FLOWCONTEXT_GENERATION_PROVIDER": "generation_provider",
    "FLOWCONTEXT_GENERATION_MODEL": "generation_model",
    "FLOWCONTEXT_GENERATION_BASE_URL": "generation_base_url",
    "FLOWCONTEXT_GENERATION_API_KEY_ENV": "generation_api_key_env",
    "FLOWCONTEXT_GENERATION_TIMEOUT_S": "generation_timeout_s",
    "FLOWCONTEXT_GENERATION_MAX_RETRIES": "generation_max_retries",
    "FLOWCONTEXT_GENERATION_RETRY_BACKOFF_S": "generation_retry_backoff_s",
    "FLOWCONTEXT_GENERATION_MAX_REPAIR_ATTEMPTS": "generation_max_repair_attempts",
    "FLOWCONTEXT_GENERATION_MAX_OUTPUT_TOKENS": "generation_max_output_tokens",
    "FLOWCONTEXT_GENERATION_INPUT_PRICE_PER_MILLION": "generation_input_price_per_million",
    "FLOWCONTEXT_GENERATION_OUTPUT_PRICE_PER_MILLION": "generation_output_price_per_million",
    "FLOWCONTEXT_MULTI_INTENT_RETRIEVAL_MODE": "multi_intent_retrieval_mode",
    "FLOWCONTEXT_MULTI_INTENT_CONTEXT_BUDGET_TOKENS": "multi_intent_context_budget_tokens",
    "FLOWCONTEXT_MULTI_INTENT_MAX_WORKERS": "multi_intent_max_workers",
    "FLOWCONTEXT_MULTI_INTENT_RRF_K": "multi_intent_rrf_k",
    "FLOWCONTEXT_MULTI_INTENT_RERANKING_ENABLED": "multi_intent_reranking_enabled",
    "FLOWCONTEXT_STREAMING_DEBOUNCE_SOURCE_S": "streaming_debounce_source_s",
    "FLOWCONTEXT_STREAMING_MIN_QUERY_CHARS": "streaming_min_query_chars",
    "FLOWCONTEXT_STREAMING_MIN_TOPIC_TERMS": "streaming_min_topic_terms",
    "FLOWCONTEXT_STREAMING_MIN_NEW_CONTENT_TERMS": "streaming_min_new_content_terms",
    "FLOWCONTEXT_STREAMING_RETRIEVE_ON_CORRECTION": "streaming_retrieve_on_correction",
    "FLOWCONTEXT_STREAMING_RETRIEVE_ON_CONSTRAINT_CHANGE": "streaming_retrieve_on_constraint_change",
    "FLOWCONTEXT_STREAMING_DUPLICATE_QUERY_SUPPRESSION": "streaming_duplicate_query_suppression",
    "FLOWCONTEXT_STREAMING_FINAL_BYPASSES_DEBOUNCE": "streaming_final_bypasses_debounce",
    "FLOWCONTEXT_STREAMING_MAX_CONCURRENCY": "streaming_max_concurrency",
    "FLOWCONTEXT_STREAMING_MAX_PENDING_REQUESTS": "streaming_max_pending_requests",
    "FLOWCONTEXT_STREAMING_REQUEST_TIMEOUT_S": "streaming_request_timeout_s",
    "FLOWCONTEXT_STREAMING_MAX_RETRIES": "streaming_max_retries",
    "FLOWCONTEXT_STREAMING_RETRY_BACKOFF_S": "streaming_retry_backoff_s",
    "FLOWCONTEXT_STREAMING_MAX_TOTAL_REQUESTS": "streaming_max_total_requests",
    "FLOWCONTEXT_STREAMING_FINAL_WAIT_TIMEOUT_S": "streaming_final_wait_timeout_s",
    "FLOWCONTEXT_STREAMING_CANCEL_ON_SUPERSESSION": "streaming_cancel_on_supersession",
}


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key.strip() not in _ENV_TO_FIELD:
            raise ValueError(f"unsupported or malformed setting on line {line_number} of {path}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_settings(
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load defaults, a dotenv-style file, then process environment overrides."""

    source_file = env_file if env_file is not None else Path(".env")
    raw_values = _parse_env_file(source_file)
    process_environment = os.environ if environ is None else environ
    for env_name in _ENV_TO_FIELD:
        if env_name in process_environment:
            raw_values[env_name] = process_environment[env_name]
    values = {_ENV_TO_FIELD[key]: value for key, value in raw_values.items()}
    return Settings.model_validate(values)


def config_asset_status(settings: Settings) -> dict[str, object]:
    """Return read-only status for config-check and documentation."""

    return {
        "corpus_path": str(settings.corpus_path),
        "corpus_exists": settings.corpus_path.is_file(),
        "transcript_path": str(settings.transcript_path),
        "transcript_exists": settings.transcript_path.is_file(),
        "evaluation_path": str(settings.evaluation_path),
        "evaluation_exists": settings.evaluation_path.is_file(),
        "evaluation_development_path": str(settings.evaluation_development_path),
        "evaluation_development_exists": settings.evaluation_development_path.is_file(),
        "evaluation_held_out_path": str(settings.evaluation_held_out_path),
        "evaluation_held_out_exists": settings.evaluation_held_out_path.is_file(),
        "streaming_evaluation_development_path": str(settings.streaming_evaluation_development_path),
        "streaming_evaluation_development_exists": settings.streaming_evaluation_development_path.is_file(),
        "streaming_evaluation_held_out_path": str(settings.streaming_evaluation_held_out_path),
        "streaming_evaluation_held_out_exists": settings.streaming_evaluation_held_out_path.is_file(),
        "artifacts_dir": str(settings.artifacts_dir),
        "index_path": str(settings.index_path),
        "corpus_status": settings.corpus_status,
        "retrieval_backend": settings.retrieval_backend,
        "multi_intent_retrieval_mode": settings.multi_intent_retrieval_mode,
        "multi_intent_context_budget_tokens": settings.multi_intent_context_budget_tokens,
        "multi_intent_max_workers": settings.multi_intent_max_workers,
        "multi_intent_rrf_k": settings.multi_intent_rrf_k,
        "multi_intent_reranking_enabled": settings.multi_intent_reranking_enabled,
        "embedding_model": settings.embedding_model,
        "embedding_revision": settings.embedding_revision,
        "embedding_license": settings.embedding_license,
        "generation_backend": settings.generation_backend,
        "generation_provider": settings.generation_provider,
        "generation_model": settings.generation_model,
        "generation_api_key_env": settings.generation_api_key_env,
        "generation_api_key_configured": bool(os.environ.get(settings.generation_api_key_env)),
        "generation_timeout_s": settings.generation_timeout_s,
        "generation_max_retries": settings.generation_max_retries,
        "generation_max_repair_attempts": settings.generation_max_repair_attempts,
        "generation_input_price_per_million": settings.generation_input_price_per_million,
        "generation_output_price_per_million": settings.generation_output_price_per_million,
        "streaming_debounce_source_s": settings.streaming_debounce_source_s,
        "streaming_min_query_chars": settings.streaming_min_query_chars,
        "streaming_min_topic_terms": settings.streaming_min_topic_terms,
        "streaming_min_new_content_terms": settings.streaming_min_new_content_terms,
        "streaming_retrieve_on_correction": settings.streaming_retrieve_on_correction,
        "streaming_retrieve_on_constraint_change": settings.streaming_retrieve_on_constraint_change,
        "streaming_duplicate_query_suppression": settings.streaming_duplicate_query_suppression,
        "streaming_final_bypasses_debounce": settings.streaming_final_bypasses_debounce,
        "streaming_max_concurrency": settings.streaming_max_concurrency,
        "streaming_max_pending_requests": settings.streaming_max_pending_requests,
        "streaming_request_timeout_s": settings.streaming_request_timeout_s,
        "streaming_max_retries": settings.streaming_max_retries,
        "streaming_retry_backoff_s": settings.streaming_retry_backoff_s,
        "streaming_max_total_requests": settings.streaming_max_total_requests,
        "streaming_final_wait_timeout_s": settings.streaming_final_wait_timeout_s,
        "streaming_cancel_on_supersession": settings.streaming_cancel_on_supersession,
    }
