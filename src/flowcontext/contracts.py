"""Validated Phase 1 data contracts.

These are local, versioned contracts for this repository.  The supplied Theme 4
guide contains an illustrative output record, but does not provide an organiser-
mandated machine-readable API or schema.  No compatibility with such an API is
claimed here.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "flowcontext.phase1.v1"
PHASE2_SCHEMA_VERSION = "flowcontext.phase2.v1"
PHASE3_SCHEMA_VERSION = "flowcontext.phase3.v1"
PHASE3_EVALUATION_SCHEMA_VERSION = "flowcontext.phase3-evaluation.v1"
PHASE4_SCHEMA_VERSION = "flowcontext.phase4.v1"
RetrievalMode = Literal["dense", "lexical", "mock", "hybrid"]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ContractModel(BaseModel):
    """Base model used to reject accidental fields at contract boundaries."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


def _validate_hash(value: str) -> str:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
    return value


def _optional_non_blank(value: str | None) -> str | None:
    if value is not None and not value.strip():
        raise ValueError("value must not be blank when provided")
    return value


class TextSpan(ContractModel):
    """A character-offset reference into the original transcript or evidence text."""

    start: int = Field(ge=0)
    end: int = Field(ge=1)
    text: str = Field(min_length=1)

    _text_is_non_blank = field_validator("text")(_non_blank)

    @model_validator(mode="after")
    def span_has_positive_width(self) -> TextSpan:
        if self.end <= self.start:
            raise ValueError("text spans must have end greater than start")
        if self.end - self.start != len(self.text):
            raise ValueError("text span width must equal the supplied text length")
        return self

    @property
    def start_char(self) -> int:
        """Compatibility spelling for callers that use explicit char names."""
        return self.start

    @property
    def end_char(self) -> int:
        return self.end


class Usage(ContractModel):
    """Token accounting for retrieval or generation, with estimation status."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated: bool = True

    @model_validator(mode="after")
    def total_is_consistent(self) -> Usage:
        expected = self.input_tokens + self.output_tokens
        if self.total_tokens != expected:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        return self


class ChunkingConfig(ContractModel):
    """Deterministic chunking settings recorded in every index manifest."""

    strategy: Literal["paragraph_then_word_window"] = "paragraph_then_word_window"
    max_chars: int = Field(default=1200, ge=1, le=100_000)
    overlap_chars: int = Field(default=0, ge=0, le=99_999)

    @model_validator(mode="after")
    def overlap_is_smaller_than_chunk(self) -> ChunkingConfig:
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        return self


class EmbeddingConfig(ContractModel):
    """Embedding/index backend metadata, including reproducibility/licensing."""

    backend: Literal["dense", "lexical", "mock"]
    provider: str = Field(min_length=1)
    model_name: str | None = None
    revision: str | None = None
    license: str | None = None
    dimensions: int | None = Field(default=None, ge=1)
    normalize: bool = False
    model_card_url: str | None = None
    download_required: bool = False

    _provider_is_non_blank = field_validator("provider")(_non_blank)
    _optional_values_are_non_blank = field_validator("model_name", "revision", "license", "model_card_url")(
        _optional_non_blank
    )

    @model_validator(mode="after")
    def dense_metadata_is_complete(self) -> EmbeddingConfig:
        if self.backend == "dense":
            missing = [
                name
                for name in ("model_name", "revision", "license", "dimensions")
                if getattr(self, name) in (None, "")
            ]
            if missing:
                raise ValueError(f"dense embedding metadata is incomplete: {', '.join(missing)}")
        return self


class GenerationConfig(ContractModel):
    """Generation provider settings that are safe to persist or trace."""

    backend: Literal["mock", "openai_compatible"]
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = Field(default=30.0, ge=0.01, le=120.0, allow_inf_nan=False)
    max_retries: int = Field(default=2, ge=0, le=3)
    retry_backoff_s: float = Field(default=0.5, ge=0.0, le=10.0, allow_inf_nan=False)
    max_repair_attempts: int = Field(default=1, ge=0, le=2)
    max_output_tokens: int = Field(default=600, ge=1, le=4096)
    input_price_per_million: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    output_price_per_million: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)

    _values_are_non_blank = field_validator("provider", "model")(_non_blank)
    _optional_values_are_non_blank = field_validator("base_url", "api_key_env")(_optional_non_blank)

    @model_validator(mode="after")
    def provider_requirements_are_valid(self) -> GenerationConfig:
        if self.backend == "openai_compatible":
            missing = [
                name
                for name in ("base_url", "api_key_env")
                if getattr(self, name) in (None, "")
            ]
            if missing:
                raise ValueError(f"openai_compatible generation metadata is incomplete: {', '.join(missing)}")
            if not self.base_url.startswith(("http://", "https://")):
                raise ValueError("generation base_url must use http:// or https://")
        return self


class SourceVersion(ContractModel):
    """Content-addressed source provenance captured by an index manifest."""

    document_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    content_hash: str

    _ids_are_non_blank = field_validator("document_id", "source_location", "source_version")(_non_blank)
    _content_hash_is_sha256 = field_validator("content_hash")(_validate_hash)


class IndexManifest(ContractModel):
    """Deterministic build manifest used to detect stale indexes."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    manifest_version: Literal["flowcontext.index.v1"] = "flowcontext.index.v1"
    index_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    source_kind: Literal["official", "synthetic_fixture", "unknown"]
    source_format: Literal["jsonl"]
    source_path: str | None = None
    source_fingerprint: str
    source_versions: list[SourceVersion] = Field(min_length=1)
    document_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    build_fingerprint: str

    _ids_are_non_blank = field_validator("index_id", "corpus_id")(_non_blank)
    _fingerprints_are_sha256 = field_validator("source_fingerprint", "build_fingerprint")(_validate_hash)
    _source_path_is_non_blank = field_validator("source_path")(_optional_non_blank)

    @model_validator(mode="after")
    def manifest_counts_are_valid(self) -> IndexManifest:
        if len(self.source_versions) != self.document_count:
            raise ValueError("document_count must match source_versions length")
        if len({source.document_id for source in self.source_versions}) != self.document_count:
            raise ValueError("source_versions document IDs must be unique")
        return self


class TranscriptEvent(ContractModel):
    """One timestamped transcript update from a single utterance."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    session_id: str = Field(min_length=1, description="Ephemeral active conversation ID")
    utterance_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    sequence_number: int = Field(ge=0)
    source_timestamp_s: float = Field(ge=0, allow_inf_nan=False)
    text: str = Field(description="Incremental fragment or cumulative transcript")
    is_final: bool = Field(description="Final marker for the utterance")
    text_mode: Literal["incremental", "cumulative"] = Field(
        description="Whether text appends to or replaces the prior transcript"
    )

    _ids_are_non_blank = field_validator("session_id", "utterance_id", "event_id")(_non_blank)


class DocumentInput(ContractModel):
    """Validated ingestion input before derived hashes and chunks are created."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    document_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1, description="Path, URI, or other provenance")
    source_version: str | None = None
    title: str | None = None
    text: str = Field(min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section: str | None = None
    section_hierarchy: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _ids_are_non_blank = field_validator("document_id", "source_location")(_non_blank)
    _source_version_is_non_blank = field_validator("source_version")(_optional_non_blank)

    @field_validator("text")
    @classmethod
    def text_is_non_blank(cls, value: str) -> str:
        return _non_blank(value)

    @field_validator("section_hierarchy")
    @classmethod
    def section_values_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("section_hierarchy must not contain blank values")
        return values

    @model_validator(mode="after")
    def page_range_is_valid(self) -> DocumentInput:
        if self.page_start is not None and self.page_end is not None:
            if self.page_end < self.page_start:
                raise ValueError("page_end must be greater than or equal to page_start")
        return self


class Document(ContractModel):
    """A source document with a deterministic content hash."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    document_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    title: str | None = None
    content_hash: str
    text: str = Field(min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section: str | None = None
    section_hierarchy: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _ids_are_non_blank = field_validator("document_id", "source_location")(_non_blank)
    _source_version_is_non_blank = field_validator("source_version")(_non_blank)
    _content_hash_is_sha256 = field_validator("content_hash")(_validate_hash)

    @field_validator("section_hierarchy")
    @classmethod
    def section_values_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("section_hierarchy must not contain blank values")
        return values

    @model_validator(mode="after")
    def page_range_is_valid(self) -> Document:
        if self.page_start is not None and self.page_end is not None:
            if self.page_end < self.page_start:
                raise ValueError("page_end must be greater than or equal to page_start")
        expected_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError("content_hash must match the document text SHA-256 digest")
        return self


class Chunk(ContractModel):
    """A deterministic, provenance-preserving retrieval unit."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    content_hash: str
    text: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    title: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section: str | None = None
    section_hierarchy: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None

    _ids_are_non_blank = field_validator("chunk_id", "document_id", "source_location", "source_version")(_non_blank)
    _content_hash_is_sha256 = field_validator("content_hash")(_validate_hash)

    @field_validator("section_hierarchy")
    @classmethod
    def section_values_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("section_hierarchy must not contain blank values")
        return values

    @model_validator(mode="after")
    def page_range_is_valid(self) -> Chunk:
        if self.page_start is not None and self.page_end is not None:
            if self.page_end < self.page_start:
                raise ValueError("page_end must be greater than or equal to page_start")
        expected_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError("content_hash must match the chunk text SHA-256 digest")
        if self.embedding is not None and (
            not self.embedding or any(not math.isfinite(value) for value in self.embedding)
        ):
            raise ValueError("embedding must be a non-empty finite vector when provided")
        return self


class CorpusIndex(ContractModel):
    """Persisted corpus inventory used by the baseline retriever."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    corpus_id: str = Field(min_length=1)
    source_kind: Literal["official", "synthetic_fixture", "unknown"]
    manifest: IndexManifest
    documents: list[Document] = Field(min_length=1)
    chunks: list[Chunk] = Field(min_length=1)

    @model_validator(mode="after")
    def ids_and_provenance_are_consistent(self) -> CorpusIndex:
        document_ids = [document.document_id for document in self.documents]
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id values must be unique")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk_id values must be unique")
        if self.manifest.corpus_id != self.corpus_id:
            raise ValueError("manifest corpus_id must match index corpus_id")
        if self.manifest.source_kind != self.source_kind:
            raise ValueError("manifest source_kind must match index source_kind")
        if self.manifest.document_count != len(self.documents):
            raise ValueError("manifest document_count must match index documents")
        if self.manifest.chunk_count != len(self.chunks):
            raise ValueError("manifest chunk_count must match index chunks")
        expected_embedding_dimensions = self.manifest.embedding.dimensions
        if self.manifest.embedding.backend in {"dense", "mock"}:
            if expected_embedding_dimensions is None:
                raise ValueError("embedding-backed indexes must declare embedding dimensions")
            for chunk in self.chunks:
                if chunk.embedding is None or len(chunk.embedding) != expected_embedding_dimensions:
                    raise ValueError(
                        f"chunk {chunk.chunk_id!r} embedding does not match manifest dimensions"
                    )
        elif any(chunk.embedding is not None for chunk in self.chunks):
            raise ValueError("lexical indexes must not contain persisted embeddings")
        known_documents = set(document_ids)
        documents_by_id = {document.document_id: document for document in self.documents}
        for chunk in self.chunks:
            if chunk.document_id not in known_documents:
                raise ValueError(f"chunk {chunk.chunk_id!r} references an unknown document")
            document = documents_by_id[chunk.document_id]
            if chunk.source_location != document.source_location:
                raise ValueError(f"chunk {chunk.chunk_id!r} has inconsistent source_location provenance")
            if chunk.source_version != document.source_version:
                raise ValueError(f"chunk {chunk.chunk_id!r} has inconsistent source_version provenance")
        return self


class RetrievalHit(ContractModel):
    """One ranked result returned by a retrieval method."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    chunk_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    snippet_text: str = Field(min_length=1)
    rank: int = Field(ge=1)
    # Lexical overlap is non-negative, while dense cosine similarity may be
    # negative.  Preserve the backend's score rather than clipping it and
    # losing diagnostic meaning.
    score: float = Field(allow_inf_nan=False)
    retrieval_method: str = Field(min_length=1)

    # Phase 3 provenance is optional so Phase 1/2 artifacts and providers keep
    # their original shape.  A fused hit retains the contributing intent
    # ranks/scores instead of replacing the backend score with opaque text.
    intent_id: str | None = None
    intent_ids: list[str] = Field(default_factory=list)
    intent_ranks: dict[str, int] = Field(default_factory=dict)
    intent_scores: dict[str, float] = Field(default_factory=dict)
    rrf_score: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    _values_are_non_blank = field_validator("chunk_id", "source_location", "snippet_text", "retrieval_method")(
        _non_blank
    )

    _intent_id_is_non_blank = field_validator("intent_id")(_optional_non_blank)

    @field_validator("intent_ids")
    @classmethod
    def intent_ids_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("intent_ids must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("intent_ids must be unique")
        return values

    @field_validator("intent_ranks")
    @classmethod
    def intent_ranks_are_valid(cls, values: dict[str, int]) -> dict[str, int]:
        if any(not key.strip() or value < 1 for key, value in values.items()):
            raise ValueError("intent_ranks must contain non-blank IDs and positive ranks")
        return values

    @field_validator("intent_scores")
    @classmethod
    def intent_scores_are_valid(cls, values: dict[str, float]) -> dict[str, float]:
        if any(not key.strip() or not math.isfinite(value) for key, value in values.items()):
            raise ValueError("intent_scores must contain non-blank IDs and finite scores")
        return values

    @model_validator(mode="after")
    def fused_intent_provenance_is_consistent(self) -> RetrievalHit:
        if self.intent_id is not None and self.intent_id not in self.intent_ids:
            raise ValueError("intent_id must be included in intent_ids when provided")
        if set(self.intent_ranks) - set(self.intent_ids):
            raise ValueError("intent_ranks keys must be included in intent_ids")
        if set(self.intent_scores) - set(self.intent_ids):
            raise ValueError("intent_scores keys must be included in intent_ids")
        return self


class EvidencePassage(ContractModel):
    """A retrieved passage quoted for generation; its text is untrusted data."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    chunk_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    text: str = Field(min_length=1)
    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)
    retrieval_method: str = Field(min_length=1)
    intent_id: str | None = None
    intent_ids: list[str] = Field(default_factory=list)
    intent_ranks: dict[str, int] = Field(default_factory=dict)
    intent_scores: dict[str, float] = Field(default_factory=dict)
    rrf_score: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _values_are_non_blank = field_validator("chunk_id", "source_location", "text", "retrieval_method")(
        _non_blank
    )

    _intent_id_is_non_blank = field_validator("intent_id")(_optional_non_blank)

    @field_validator("intent_ids")
    @classmethod
    def evidence_intent_ids_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("intent_ids must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("intent_ids must be unique")
        return values

    @model_validator(mode="after")
    def evidence_intent_provenance_is_consistent(self) -> EvidencePassage:
        if self.intent_id is not None and self.intent_id not in self.intent_ids:
            raise ValueError("intent_id must be included in intent_ids when provided")
        if set(self.intent_ranks) - set(self.intent_ids):
            raise ValueError("intent_ranks keys must be included in intent_ids")
        if set(self.intent_scores) - set(self.intent_ids):
            raise ValueError("intent_scores keys must be included in intent_ids")
        return self


class GenerationRequest(ContractModel):
    """Structured generation input containing only the configured corpus hits."""

    schema_version: Literal[SCHEMA_VERSION, PHASE3_SCHEMA_VERSION] = SCHEMA_VERSION
    query: str = Field(min_length=1)
    passages: list[EvidencePassage] = Field(min_length=1)
    repair_feedback: str | None = None
    attempt: int = Field(default=1, ge=1)
    intent_queries: list[str] = Field(default_factory=list)
    unsupported_intent_queries: list[str] = Field(default_factory=list)
    decomposed_intents: list[dict[str, Any]] = Field(default_factory=list)
    shared_constraints: list[dict[str, Any]] = Field(default_factory=list)
    intent_evidence: dict[str, list[EvidencePassage]] = Field(default_factory=dict)
    transcript_revision: int | None = None

    _query_is_non_blank = field_validator("query")(_non_blank)
    _repair_feedback_is_non_blank = field_validator("repair_feedback")(_optional_non_blank)

    @field_validator("intent_queries", "unsupported_intent_queries")
    @classmethod
    def intent_query_lists_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("intent query lists must not contain blank values")
        return values


class RetrievalResponse(ContractModel):
    """Validated CLI/API-neutral retrieval response with provenance snippets."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    query: str = Field(min_length=1)
    retrieval_backend: RetrievalMode
    index_id: str = Field(min_length=1)
    hits: list[RetrievalHit] = Field(default_factory=list)
    decomposition: dict[str, Any] | None = None
    retrieval_details: dict[str, Any] | None = None

    _values_are_non_blank = field_validator("query", "index_id")(_non_blank)


IntentSynthesisStatus = Literal["answered", "insufficient_evidence", "conflicting_evidence", "needs_clarification"]


class IntentStatusRecord(ContractModel):
    """Synthesis status and explanation for one sub-question/intent."""

    schema_version: Literal[PHASE3_SCHEMA_VERSION] = PHASE3_SCHEMA_VERSION
    intent_id: str = Field(min_length=1)
    status: IntentSynthesisStatus
    reason: str | None = None
    addressed_by_claim_ids: list[str] = Field(default_factory=list)

    _intent_id_is_non_blank = field_validator("intent_id")(_non_blank)
    _reason_is_non_blank = field_validator("reason")(_optional_non_blank)


class ClaimVerificationVerdict(ContractModel):
    """Verdict from automated semantic verification of an atomic claim."""

    schema_version: Literal[PHASE3_SCHEMA_VERSION] = PHASE3_SCHEMA_VERSION
    claim_id: str = Field(min_length=1)
    verdict: Literal["supported", "unsupported", "uncertain"]
    reason: str = Field(min_length=1)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    cross_entity_violation: bool = False
    contradiction_detected: bool = False

    _strings_are_non_blank = field_validator("claim_id", "reason")(_non_blank)


class SemanticVerificationReport(ContractModel):
    """Audit report for claim verification with model identity, latency, and explicit limitations."""

    schema_version: Literal[PHASE3_SCHEMA_VERSION] = PHASE3_SCHEMA_VERSION
    verifier_model: str = Field(min_length=1)
    latency_ms: float = Field(ge=0.0)
    verdicts: list[ClaimVerificationVerdict] = Field(default_factory=list)
    limitations: str = Field(min_length=1)
    usage: Usage = Field(default_factory=Usage)

    _strings_are_non_blank = field_validator("verifier_model", "limitations")(_non_blank)


class FactualClaim(ContractModel):
    """A claim whose supporting chunk IDs must be checked before presentation."""

    schema_version: Literal[SCHEMA_VERSION, PHASE3_SCHEMA_VERSION] = SCHEMA_VERSION
    claim_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    supporting_chunk_ids: list[str] = Field(min_length=1)
    intent_id: str | None = None
    intent_ids: list[str] = Field(default_factory=list)
    supporting_excerpts: list[str] = Field(default_factory=list)
    supporting_spans: list[TextSpan] = Field(default_factory=list)
    semantic_support: str | None = None

    _ids_are_non_blank = field_validator("claim_id", "claim_text")(_non_blank)
    _intent_id_is_non_blank = field_validator("intent_id")(_optional_non_blank)
    _semantic_support_is_non_blank = field_validator("semantic_support")(_optional_non_blank)

    @field_validator("supporting_chunk_ids")
    @classmethod
    def support_ids_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("supporting_chunk_ids must not contain blank IDs")
        if len(values) != len(set(values)):
            raise ValueError("supporting_chunk_ids must be unique per claim")
        return values

    @field_validator("supporting_excerpts")
    @classmethod
    def excerpts_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("supporting_excerpts must not contain blank strings")
        return values

    @model_validator(mode="after")
    def sync_intent_fields(self) -> FactualClaim:
        if self.intent_id and not self.intent_ids:
            self.intent_ids = [self.intent_id]
        elif self.intent_ids and not self.intent_id:
            self.intent_id = self.intent_ids[0]
        return self


class Answer(ContractModel):
    """Grounded answer state for complete-utterance baseline and unified Phase 3 synthesis."""

    schema_version: Literal[SCHEMA_VERSION, PHASE3_SCHEMA_VERSION] = SCHEMA_VERSION
    answer_text: str = Field(min_length=1)
    factual_claims: list[FactualClaim] = Field(default_factory=list)
    uncertainty: str = Field(min_length=1)
    answer_version: int = Field(ge=1)
    intent_statuses: list[IntentStatusRecord] = Field(default_factory=list)
    verification_audit: dict[str, Any] | None = None

    @field_validator("answer_text", "uncertainty")
    @classmethod
    def answer_values_are_non_blank(cls, value: str) -> str:
        return _non_blank(value)


class GenerationResult(ContractModel):
    """Raw structured-output text and usage returned by one provider call."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    raw_text: str = Field(min_length=1)
    usage: Usage = Field(default_factory=Usage)

    _raw_text_is_non_blank = field_validator("raw_text")(_non_blank)


class TraceError(ContractModel):
    """Structured error attached to an execution trace event."""

    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)

    _values_are_non_blank = field_validator("error_type", "message")(_non_blank)


class ExecutionTrace(ContractModel):
    """One structured event in a replay execution trace."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    trace_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    source_timestamp_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    monotonic_execution_time_s: float = Field(ge=0, allow_inf_nan=False)
    # Relative to the start of the replay.  This is populated for transcript
    # delivery events so source time and observed delivery time remain
    # separate, inspectable timing domains.
    actual_delivery_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    duration_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    model_identity: str = Field(min_length=1)
    usage: Usage = Field(default_factory=Usage)
    cost: float | Literal["unavailable"] = "unavailable"
    error: TraceError | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    _ids_are_non_blank = field_validator("trace_id", "run_id", "session_id", "event_type", "model_identity")(
        _non_blank
    )


class ReplayResult(ContractModel):
    """Output of one complete-utterance transcript replay."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    baseline_mode: Literal["wait_for_complete_utterance"]
    corpus_source_kind: Literal["official", "synthetic_fixture", "unknown"]
    retrieval_backend: RetrievalMode
    generation_backend: Literal["mock", "openai_compatible"]
    generation_model: str = Field(min_length=1)
    generation_status: Literal["success", "abstained", "skipped"]
    generation_usage: Usage
    generation_cost: float | Literal["unavailable"] = "unavailable"
    generation_attempts: int = Field(ge=0)
    generation_repair_attempts: int = Field(ge=0)
    execution_mode: Literal["realtime", "accelerated"]
    run_status: Literal["completed", "abstained", "failed"]
    final_event_id: str = Field(min_length=1)
    final_source_timestamp_s: float = Field(ge=0, allow_inf_nan=False)
    transcript_event_count: int = Field(ge=1)
    duplicate_event_count: int = Field(default=0, ge=0)
    source_duration_s: float = Field(ge=0, allow_inf_nan=False)
    execution_duration_ms: float = Field(ge=0, allow_inf_nan=False)
    complete_answer_latency_ms: float = Field(ge=0, allow_inf_nan=False)
    query: str = Field(min_length=1)
    retrieval_triggered: bool
    retrieval_hits: list[RetrievalHit] = Field(default_factory=list)
    decomposition: dict[str, Any] | None = None
    retrieval_details: dict[str, Any] | None = None
    answer: Answer
    traces: list[ExecutionTrace] = Field(min_length=1)

    # Common comparison/reporting fields.  They are optional/defaulted for
    # compatibility with Phase 1 artifacts, but new replays always populate
    # them.
    mode: Literal["baseline"] = "baseline"
    final_event_delivery_time_s: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    evidence_ready_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    generation_started_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    generation_completed_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    generation_first_content_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    retrieval_started_early: bool = False
    valid_evidence_ready_before_finalization: bool = False
    early_evidence_reused: bool = False
    answer_latency_from_final_event_delivery_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    full_interaction_duration_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    controller_overhead_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    retrieval_call_count: int = Field(default=0, ge=0)
    scheduled_request_count: int = Field(default=0, ge=0)
    superseded_request_count: int = Field(default=0, ge=0)
    cancelled_request_count: int = Field(default=0, ge=0)
    timed_out_request_count: int = Field(default=0, ge=0)
    retrieval_error_count: int = Field(default=0, ge=0)
    stale_result_discard_count: int = Field(default=0, ge=0)
    retrieval_usage: Usage = Field(default_factory=Usage)
    errors: list[TraceError] = Field(default_factory=list)
    corpus_id: str = "unknown"
    index_id: str = "unknown"
    retrieval_top_k: int = Field(default=5, ge=1)

    _ids_are_non_blank = field_validator("run_id", "session_id", "utterance_id", "final_event_id", "query")(
        _non_blank
    )


class ReplayRunManifest(ContractModel):
    """Secret-free metadata describing one transcript replay execution."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    manifest_version: Literal["flowcontext.replay.manifest.v1"] = "flowcontext.replay.manifest.v1"
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    created_at_utc: str = Field(min_length=1)
    code_revision: str | None = None
    configuration: dict[str, Any] = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    index_id: str = Field(min_length=1)
    corpus_source_kind: Literal["official", "synthetic_fixture", "unknown"]
    model_identities: dict[str, str] = Field(min_length=1)
    execution_mode: Literal["realtime", "accelerated"]
    run_status: Literal["completed", "abstained", "failed"]
    source_duration_s: float = Field(ge=0, allow_inf_nan=False)
    actual_execution_duration_ms: float = Field(ge=0, allow_inf_nan=False)
    complete_answer_latency_ms: float = Field(ge=0, allow_inf_nan=False)
    result_path: str = Field(min_length=1)
    trace_path: str = Field(min_length=1)
    environment: dict[str, str] = Field(min_length=1)
    secrets_excluded: Literal[True] = True
    mode: Literal["baseline"] = "baseline"

    _ids_are_non_blank = field_validator(
        "run_id",
        "session_id",
        "utterance_id",
        "created_at_utc",
        "corpus_id",
        "index_id",
        "result_path",
        "trace_path",
    )(_non_blank)
    _code_revision_is_non_blank = field_validator("code_revision")(_optional_non_blank)
    @field_validator("model_identities")
    @classmethod
    def model_identities_are_non_blank(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("model_identities keys and values must not be blank")
        return values


class StreamingDecisionConfig(ContractModel):
    """Explainable Phase 2 controller settings.

    The debounce value is applied to source-time gaps, not wall-clock time.
    Scheduler resource limits live in :class:`StreamingSchedulerConfig` so
    controller policy and execution policy can be changed independently.
    """

    schema_version: Literal[PHASE2_SCHEMA_VERSION] = PHASE2_SCHEMA_VERSION
    debounce_source_s: float = Field(default=0.35, ge=0.0, le=30.0, allow_inf_nan=False)
    min_query_chars: int = Field(default=8, ge=1, le=1000)
    min_topic_terms: int = Field(default=1, ge=1, le=20)
    min_new_content_terms: int = Field(default=1, ge=1, le=20)
    retrieve_on_correction: bool = True
    retrieve_on_constraint_change: bool = True
    duplicate_query_suppression: bool = True
    final_decision_bypasses_debounce: bool = True


class StreamingSchedulerConfig(ContractModel):
    """Bounded execution settings for asynchronous Phase 2 retrieval.

    ``max_pending_requests`` counts requests whose results have not reached a
    terminal state in this session/utterance, while a single coalesced latest
    request may wait outside that count. A running executor thread can remain
    alive after a timeout or cancellation request; revision checks, rather
    than cancellation, provide the correctness boundary.
    """

    schema_version: Literal[PHASE2_SCHEMA_VERSION] = PHASE2_SCHEMA_VERSION
    max_concurrency: int = Field(default=2, ge=1, le=16)
    max_pending_requests: int = Field(default=4, ge=1, le=64)
    request_timeout_s: float = Field(default=5.0, ge=0.01, le=120.0, allow_inf_nan=False)
    max_retries: int = Field(default=0, ge=0, le=3)
    retry_backoff_s: float = Field(default=0.05, ge=0.0, le=10.0, allow_inf_nan=False)
    max_total_requests: int = Field(default=16, ge=1, le=256)
    final_wait_timeout_s: float = Field(default=5.0, ge=0.01, le=120.0, allow_inf_nan=False)
    cancel_on_supersession: bool = True

    @model_validator(mode="after")
    def pending_limit_is_usable(self) -> StreamingSchedulerConfig:
        if self.max_pending_requests < 1:
            raise ValueError("max_pending_requests must be at least one")
        return self


class PreviousAnswerContext(ContractModel):
    """Session-scoped context used only to classify formatting-only turns."""

    schema_version: Literal[PHASE2_SCHEMA_VERSION] = PHASE2_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    answer_text: str = Field(min_length=1)
    answer_version: int = Field(ge=1)

    _ids_are_non_blank = field_validator("session_id")(_non_blank)
    _answer_text_is_non_blank = field_validator("answer_text")(_non_blank)


class StreamingDecision(ContractModel):
    """One explainable decision made from a transcript revision."""

    schema_version: Literal[PHASE2_SCHEMA_VERSION] = PHASE2_SCHEMA_VERSION
    decision_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    transcript_revision: int = Field(ge=0)
    decision: Literal["WAIT", "RETRIEVE", "SKIP"]
    reason_code: str = Field(min_length=1)
    proposed_query: str | None = None
    source_event_id: str = Field(min_length=1)
    source_timestamp_s: float = Field(ge=0, allow_inf_nan=False)
    monotonic_decision_time_s: float = Field(ge=0, allow_inf_nan=False)
    is_final_event: bool
    needs_previous_answer_context: bool = False
    previous_answer_context_used: bool = False
    coalesced_event_ids: list[str] = Field(min_length=1)

    _ids_are_non_blank = field_validator(
        "decision_id", "session_id", "utterance_id", "reason_code", "source_event_id"
    )(_non_blank)
    _proposed_query_is_non_blank = field_validator("proposed_query")(_optional_non_blank)

    @field_validator("coalesced_event_ids")
    @classmethod
    def coalesced_ids_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("coalesced_event_ids must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("coalesced_event_ids must be unique")
        return values

    @model_validator(mode="after")
    def query_only_for_retrieve(self) -> StreamingDecision:
        if self.decision == "RETRIEVE" and self.proposed_query is None:
            raise ValueError("RETRIEVE decisions require proposed_query")
        if self.decision != "RETRIEVE" and self.proposed_query is not None:
            raise ValueError("WAIT/SKIP decisions must not contain proposed_query")
        if self.needs_previous_answer_context and self.previous_answer_context_used:
            raise ValueError("a decision cannot both need and use previous answer context")
        return self


class StreamingRetrievalRequest(ContractModel):
    """A retrieval request produced by a controller decision.

    The revision is a stale-result guard. A later transcript revision may
    supersede this request before its result is accepted.
    """

    schema_version: Literal[PHASE2_SCHEMA_VERSION] = PHASE2_SCHEMA_VERSION
    request_id: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    transcript_revision: int = Field(ge=0)
    retrieval_revision: int = Field(ge=1)
    query: str = Field(min_length=1)
    source_timestamp_s: float = Field(ge=0, allow_inf_nan=False)
    monotonic_started_s: float = Field(ge=0, allow_inf_nan=False)
    # Relative-to-replay scheduling time is kept separately from the absolute
    # monotonic value above for concise, portable trace reporting.
    scheduled_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    _ids_are_non_blank = field_validator(
        "request_id", "decision_id", "session_id", "utterance_id", "query"
    )(_non_blank)


class StreamingRetrievalResult(ContractModel):
    """One retrieval attempt, including whether its evidence remains current."""

    schema_version: Literal[PHASE2_SCHEMA_VERSION] = PHASE2_SCHEMA_VERSION
    request_id: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    transcript_revision: int = Field(ge=0)
    retrieval_revision: int = Field(ge=1)
    query: str = Field(min_length=1)
    hits: list[RetrievalHit] = Field(default_factory=list)
    # Serialized revision-bound Phase 3 decomposition metadata.  It is kept
    # optional/dict-shaped here so the Phase 2 contract remains loadable for
    # historical artifacts while new runs can expose the full result.
    decomposition: dict[str, Any] | None = None
    retrieval_details: dict[str, Any] | None = None
    accepted: bool
    stale: bool
    source_timestamp_s: float = Field(ge=0, allow_inf_nan=False)
    monotonic_started_s: float = Field(ge=0, allow_inf_nan=False)
    monotonic_completed_s: float = Field(ge=0, allow_inf_nan=False)
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    status: Literal["completed", "failed", "timed_out", "superseded", "cancelled"] = "completed"
    attempts: int = Field(default=1, ge=0)
    superseded: bool = False
    cancellation_requested: bool = False
    cancellation_confirmed: bool = False
    error: TraceError | None = None
    is_final_event: bool = False
    scheduled_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    actual_started_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    actual_completed_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    evidence_ready_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    actual_duration_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    _ids_are_non_blank = field_validator(
        "request_id", "decision_id", "session_id", "utterance_id", "query"
    )(_non_blank)

    @model_validator(mode="after")
    def completed_time_is_after_start(self) -> StreamingRetrievalResult:
        if self.monotonic_completed_s < self.monotonic_started_s:
            raise ValueError("monotonic_completed_s must not precede monotonic_started_s")
        if self.accepted and self.stale:
            raise ValueError("a stale retrieval result cannot be accepted")
        if self.cancellation_confirmed and not self.cancellation_requested:
            raise ValueError("cancellation_confirmed requires cancellation_requested")
        if self.status in {"failed", "timed_out", "superseded", "cancelled"} and self.accepted:
            raise ValueError("only a completed retrieval can be accepted")
        return self


class StreamingReplayResult(ContractModel):
    """Phase 2 streaming retrieval replay plus final-only answer output."""

    schema_version: Literal[PHASE2_SCHEMA_VERSION] = PHASE2_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    controller_mode: Literal["explainable_streaming_controller"]
    corpus_source_kind: Literal["official", "synthetic_fixture", "unknown"]
    retrieval_backend: RetrievalMode
    execution_mode: Literal["realtime", "accelerated"]
    run_status: Literal["completed", "needs_context", "abstained", "failed"]
    final_event_id: str = Field(min_length=1)
    final_source_timestamp_s: float = Field(ge=0, allow_inf_nan=False)
    transcript_event_count: int = Field(ge=1)
    duplicate_event_count: int = Field(default=0, ge=0)
    source_duration_s: float = Field(ge=0, allow_inf_nan=False)
    execution_duration_ms: float = Field(ge=0, allow_inf_nan=False)
    final_query: str = Field(min_length=1)
    final_decision: StreamingDecision
    decisions: list[StreamingDecision] = Field(min_length=1)
    retrieval_results: list[StreamingRetrievalResult] = Field(default_factory=list)
    current_evidence_hits: list[RetrievalHit] = Field(default_factory=list)
    decomposition: dict[str, Any] | None = None
    retrieval_details: dict[str, Any] | None = None
    decomposition_history: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_attempt_count: int = Field(default=0, ge=0)
    early_retrieval_count: int = Field(default=0, ge=0)
    final_retrieval_count: int = Field(default=0, ge=0)
    suppressed_duplicate_query_count: int = Field(default=0, ge=0)
    stale_retrieval_count: int = Field(default=0, ge=0)
    unnecessary_retrieval_count: int = Field(default=0, ge=0)
    scheduled_request_count: int = Field(default=0, ge=0)
    cancelled_request_count: int = Field(default=0, ge=0)
    superseded_request_count: int = Field(default=0, ge=0)
    timed_out_request_count: int = Field(default=0, ge=0)
    generation_backend: Literal["mock", "openai_compatible"] = "mock"
    generation_model: str = "not_run"
    generation_status: Literal["success", "abstained", "skipped", "failed"] = "skipped"
    generation_usage: Usage = Field(default_factory=Usage)
    generation_cost: float | Literal["unavailable"] = "unavailable"
    generation_attempts: int = Field(default=0, ge=0)
    generation_repair_attempts: int = Field(default=0, ge=0)
    complete_answer_latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    answer: Answer | None = None
    traces: list[ExecutionTrace] = Field(min_length=1)

    mode: Literal["streaming"] = "streaming"
    final_event_delivery_time_s: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    evidence_ready_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    generation_started_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    generation_completed_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    generation_first_content_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    retrieval_started_early: bool = False
    valid_evidence_ready_before_finalization: bool = False
    early_evidence_reused: bool = False
    answer_latency_from_final_event_delivery_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    full_interaction_duration_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    controller_overhead_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    retrieval_call_count: int = Field(default=0, ge=0)
    retrieval_error_count: int = Field(default=0, ge=0)
    stale_result_discard_count: int = Field(default=0, ge=0)
    retrieval_usage: Usage = Field(default_factory=Usage)
    errors: list[TraceError] = Field(default_factory=list)
    corpus_id: str = "unknown"
    index_id: str = "unknown"
    retrieval_top_k: int = Field(default=5, ge=1)

    _ids_are_non_blank = field_validator(
        "run_id", "session_id", "utterance_id", "final_event_id", "final_query"
    )(_non_blank)


class StreamingRunManifest(ContractModel):
    """Secret-free metadata describing one Phase 2 controller replay."""

    schema_version: Literal[PHASE2_SCHEMA_VERSION] = PHASE2_SCHEMA_VERSION
    manifest_version: Literal["flowcontext.streaming-replay.manifest.v1"] = (
        "flowcontext.streaming-replay.manifest.v1"
    )
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    created_at_utc: str = Field(min_length=1)
    code_revision: str | None = None
    configuration: dict[str, Any] = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    index_id: str = Field(min_length=1)
    corpus_source_kind: Literal["official", "synthetic_fixture", "unknown"]
    model_identities: dict[str, str] = Field(min_length=1)
    execution_mode: Literal["realtime", "accelerated"]
    run_status: Literal["completed", "needs_context", "abstained", "failed"]
    source_duration_s: float = Field(ge=0, allow_inf_nan=False)
    actual_execution_duration_ms: float = Field(ge=0, allow_inf_nan=False)
    result_path: str = Field(min_length=1)
    trace_path: str = Field(min_length=1)
    environment: dict[str, str] = Field(min_length=1)
    secrets_excluded: Literal[True] = True
    mode: Literal["streaming"] = "streaming"

    _ids_are_non_blank = field_validator(
        "run_id",
        "session_id",
        "utterance_id",
        "created_at_utc",
        "corpus_id",
        "index_id",
        "result_path",
        "trace_path",
    )(_non_blank)
    _code_revision_is_non_blank = field_validator("code_revision")(_optional_non_blank)

    @field_validator("model_identities")
    @classmethod
    def model_identities_are_non_blank(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("model_identities keys and values must not be blank")
        return values


class EvaluationCase(ContractModel):
    """Fixture or organiser-supplied expectations for one replay case.

    The original ``expected_citation_chunk_ids`` field is retained for the
    small legacy fixture.  New evaluation assets should use
    ``expected_evidence_chunk_ids`` and embed their transcript events directly
    in the case record.
    """

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    asset_status: Literal["official", "synthetic_fixture", "unknown"]
    split: Literal["development", "held_out"] = "development"
    scenario_group: str = "legacy"
    transcript: list[TranscriptEvent] = Field(default_factory=list)
    answerability: Literal["answerable", "unanswerable", "partially_answerable"] = "answerable"
    expected_answer_substrings: list[str] = Field(default_factory=list)
    expected_evidence_chunk_ids: list[str] = Field(default_factory=list)
    expected_citation_chunk_ids: list[str] = Field(default_factory=list)
    relevance_labels: dict[str, Literal["relevant", "not_relevant"]] = Field(default_factory=dict)
    relevance_label_status: Literal[
        "provisional_generated", "human_review_pending", "human_reviewed"
    ] = "provisional_generated"
    answer_support_rubric: list[str] = Field(default_factory=list)
    capabilities_deferred: list[str] = Field(default_factory=list)
    case_notes: list[str] = Field(default_factory=list)
    require_wait_until_final: bool = True

    _ids_are_non_blank = field_validator("case_id", "session_id")(_non_blank)

    @field_validator("scenario_group")
    @classmethod
    def scenario_group_is_non_blank(cls, value: str) -> str:
        return _non_blank(value)

    @field_validator("expected_evidence_chunk_ids", "expected_citation_chunk_ids")
    @classmethod
    def expected_ids_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("expected evidence/citation IDs must not be blank")
        if len(values) != len(set(values)):
            raise ValueError("expected evidence/citation IDs must be unique")
        return values

    @field_validator("relevance_labels")
    @classmethod
    def relevance_label_ids_are_non_blank(
        cls,
        values: dict[str, Literal["relevant", "not_relevant"]],
    ) -> dict[str, Literal["relevant", "not_relevant"]]:
        if any(not key.strip() for key in values):
            raise ValueError("relevance label IDs must not be blank")
        return values

    @field_validator("expected_answer_substrings", "answer_support_rubric", "capabilities_deferred", "case_notes")
    @classmethod
    def case_text_lists_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("case text lists must not contain blank values")
        return values

    @model_validator(mode="after")
    def transcript_is_isolated_when_present(self) -> EvaluationCase:
        if self.transcript:
            sessions = {event.session_id for event in self.transcript}
            utterances = {event.utterance_id for event in self.transcript}
            if sessions != {self.session_id}:
                raise ValueError("evaluation transcript session IDs must match session_id")
            if len(utterances) != 1:
                raise ValueError("an evaluation case must contain one utterance")
            if not any(event.is_final for event in self.transcript):
                raise ValueError("an evaluation transcript must contain a final event")
        if self.relevance_labels:
            relevant_ids = {
                chunk_id for chunk_id, label in self.relevance_labels.items() if label == "relevant"
            }
            expected_ids = set(self.expected_evidence_chunk_ids or self.expected_citation_chunk_ids)
            if relevant_ids != expected_ids:
                raise ValueError(
                    "relevance_labels relevant IDs must match expected_evidence_chunk_ids"
                )
        return self


class EvaluationCaseResult(ContractModel):
    """Transparent per-case outcome separating ID validity from semantic support."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    split: Literal["development", "held_out"] = "development"
    answerability: Literal["answerable", "unanswerable", "partially_answerable"] = "answerable"
    answer_substrings_passed: bool
    missing_answer_substrings: list[str] = Field(default_factory=list)
    expected_evidence_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_labels_available: bool = False
    retrieval_recall_at_k: dict[str, float] = Field(default_factory=dict)
    retrieval_reciprocal_rank: float | None = Field(default=None, ge=0, le=1)
    citation_ids_valid: bool
    citation_ids_checked: int = Field(default=0, ge=0)
    unknown_citation_ids: list[str] = Field(default_factory=list)
    semantic_support_evaluated: bool = False
    claim_support_status: Literal["not_evaluated", "human_reviewed", "model_judged"] = "not_evaluated"
    missing_expected_citations: list[str] = Field(default_factory=list)
    waited_until_final: bool
    abstained: bool = False
    generation_status: Literal["success", "abstained", "skipped"] = "abstained"
    run_status: Literal["completed", "abstained", "failed"] = "abstained"
    retrieval_latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    answer_latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    generation_usage: Usage = Field(default_factory=Usage)
    generation_cost: float | Literal["unavailable"] = "unavailable"
    trace_complete: bool = False
    missing_trace_events: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    passed: bool
    notes: list[str] = Field(default_factory=list)

    _case_id_is_non_blank = field_validator("case_id")(_non_blank)


class EvaluationReport(ContractModel):
    """Evaluation output with explicit fixture/benchmark provenance."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    evaluation_label: str = Field(min_length=1)
    asset_status: Literal["official", "synthetic_fixture", "unknown"]
    retrieval_backend: RetrievalMode = "lexical"
    generation_backend: Literal["mock", "openai_compatible"] = "mock"
    generation_model: str = Field(default="unknown", min_length=1)
    generation_status: Literal["success", "abstained", "skipped"] = "abstained"
    execution_mode: Literal["realtime", "accelerated"] = "accelerated"
    semantic_support_evaluated: bool = False
    competition_performance_claim: bool = False
    passed: bool
    cases: list[EvaluationCaseResult] = Field(min_length=1)
    metrics: dict[str, float] = Field(default_factory=dict)

    _label_is_non_blank = field_validator("evaluation_label")(_non_blank)


class LatencySummary(ContractModel):
    """Sample-counted latency summary using the inclusive linear percentile."""

    sample_count: int = Field(ge=0)
    p50_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    p95_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class ClaimSupportReview(ContractModel):
    """Claim support status, kept separate from citation-ID validity."""

    status: Literal["not_evaluated", "human_reviewed", "model_judged"] = "not_evaluated"
    procedure: str = Field(min_length=1)
    reviewed_claim_count: int = Field(default=0, ge=0)
    supported_claim_count: int = Field(default=0, ge=0)
    unsupported_claim_count: int = Field(default=0, ge=0)
    uncertain_claim_count: int = Field(default=0, ge=0)
    judge_model: str | None = None
    judge_rubric: str | None = None
    limitations: list[str] = Field(default_factory=list)


class EvaluationSection(ContractModel):
    """One explicitly labelled evaluation result section."""

    section_id: Literal["mock", "fixture", "real_model"]
    status: Literal["measured", "not_verified", "blocked", "failed"]
    result_group: str = Field(min_length=1)
    asset_status: Literal["official", "synthetic_fixture", "unknown"]
    corpus_id: str | None = None
    index_id: str | None = None
    corpus_source_kind: Literal["official", "synthetic_fixture", "unknown"]
    retrieval_backend: RetrievalMode | None = None
    generation_backend: Literal["mock", "openai_compatible"] | None = None
    generation_model: str | None = None
    execution_mode: Literal["realtime", "accelerated"] | None = None
    warm_cold_condition: str = Field(min_length=1)
    case_count: int = Field(ge=0)
    answerable_case_count: int = Field(default=0, ge=0)
    unanswerable_case_count: int = Field(default=0, ge=0)
    case_results: list[EvaluationCaseResult] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    retrieval_latency: LatencySummary = Field(default_factory=LatencySummary)
    answer_latency: LatencySummary = Field(default_factory=LatencySummary)
    generation_usage: Usage = Field(default_factory=Usage)
    generation_cost: float | Literal["unavailable"] = "unavailable"
    claim_support: ClaimSupportReview = Field(
        default_factory=lambda: ClaimSupportReview(
            procedure="Human reviewer reads each factual claim with its cited chunk and labels supported, unsupported, or uncertain; citation-ID validity alone is not support evidence."
        )
    )
    error_count: int = Field(default=0, ge=0)
    trace_complete_count: int = Field(default=0, ge=0)
    trace_required_event_set: list[str] = Field(default_factory=list)
    early_retrieval: Literal["absent"] = "absent"
    limitations: list[str] = Field(default_factory=list)


class EvaluationSuiteReport(ContractModel):
    """Measured Phase 1 evaluation report with fixture/mock/real separation."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    report_version: Literal["flowcontext.evaluation.v1"] = "flowcontext.evaluation.v1"
    evaluation_label: str = Field(min_length=1)
    asset_status: Literal["official", "synthetic_fixture", "unknown"]
    selected_split: Literal["development", "held_out", "all"]
    development_case_count: int = Field(ge=0)
    held_out_case_count: int = Field(ge=0)
    labels_status: Literal["provisional_generated", "human_review_pending", "human_reviewed"]
    competition_performance_claim: Literal[False] = False
    passed: bool
    created_at_utc: str = Field(min_length=1)
    code_revision: str | None = None
    configuration: dict[str, Any] = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    index_id: str = Field(min_length=1)
    model_identities: dict[str, str] = Field(min_length=1)
    execution_mode: Literal["realtime", "accelerated"]
    warm_cold_condition: str = Field(min_length=1)
    environment: dict[str, str] = Field(min_length=1)
    hardware: dict[str, str] = Field(min_length=1)
    sections: dict[str, EvaluationSection] = Field(min_length=1)
    official_assets_available: bool = False
    limitations: list[str] = Field(default_factory=list)

    _label_is_non_blank = field_validator("evaluation_label")(_non_blank)
    _created_at_is_non_blank = field_validator("created_at_utc")(_non_blank)
    _code_revision_is_non_blank = field_validator("code_revision")(_optional_non_blank)

    @field_validator("model_identities")
    @classmethod
    def identity_values_are_non_blank(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("model_identities keys and values must not be blank")
        return values

    @model_validator(mode="after")
    def required_sections_are_present(self) -> EvaluationSuiteReport:
        missing = {"mock", "fixture", "real_model"} - set(self.sections)
        if missing:
            raise ValueError("evaluation report must contain mock, fixture, and real_model sections")
        return self


class StreamingEvaluationCase(ContractModel):
    """Pre-labelled Phase 2 streaming case metadata.

    Eligibility and the earliest reasonable retrieval point are deliberately
    stored with the case, before the controller is run.  This prevents the
    measured denominator from moving with policy behavior.  Labels in the
    local suite are provisional engineering labels, not human verification.
    """

    schema_version: Literal[PHASE2_SCHEMA_VERSION] = PHASE2_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    asset_status: Literal["official", "synthetic_fixture", "unknown"]
    split: Literal["development", "held_out"]
    scenario_group: str = Field(min_length=1)
    transcript: list[TranscriptEvent] = Field(min_length=1)
    eligible_for_early_retrieval: bool
    eligibility_reason: str = Field(min_length=1)
    earliest_reasonable_retrieval_event_id: str | None = None
    earliest_reasonable_retrieval_source_timestamp_s: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    expected_final_evidence_chunk_ids: list[str] = Field(default_factory=list)
    relevance_labels: dict[str, Literal["relevant", "not_relevant"]] = Field(default_factory=dict)
    relevance_label_status: Literal[
        "provisional_generated", "human_review_pending", "human_reviewed"
    ] = "provisional_generated"
    expected_no_retrieval: bool = False
    expected_final_decision: Literal["WAIT", "RETRIEVE", "SKIP"] | None = None
    expected_final_reason_codes: list[str] = Field(default_factory=list)
    retrieval_behavior: Literal["normal", "delay", "failure", "timeout", "session_close"] = "normal"
    simulated_retrieval_delay_s: float = Field(default=0.0, ge=0, le=2.0, allow_inf_nan=False)
    retrieval_timeout_s: float = Field(default=0.05, ge=0.01, le=2.0, allow_inf_nan=False)
    previous_answer_context: PreviousAnswerContext | None = None
    concurrency_group: str | None = None
    compare_without_suppression: bool = False
    case_notes: list[str] = Field(default_factory=list)

    _ids_are_non_blank = field_validator("case_id", "session_id", "scenario_group")(_non_blank)
    _optional_ids_are_non_blank = field_validator(
        "earliest_reasonable_retrieval_event_id", "concurrency_group"
    )(_optional_non_blank)

    @field_validator("expected_final_evidence_chunk_ids")
    @classmethod
    def expected_evidence_ids_are_unique(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("expected final evidence IDs must not be blank")
        if len(values) != len(set(values)):
            raise ValueError("expected final evidence IDs must be unique")
        return values

    @field_validator("relevance_labels")
    @classmethod
    def relevance_ids_are_non_blank(
        cls, values: dict[str, Literal["relevant", "not_relevant"]]
    ) -> dict[str, Literal["relevant", "not_relevant"]]:
        if any(not key.strip() for key in values):
            raise ValueError("relevance label IDs must not be blank")
        return values

    @field_validator("expected_final_reason_codes", "case_notes")
    @classmethod
    def case_lists_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("case metadata lists must not contain blank values")
        return values

    @model_validator(mode="after")
    def case_structure_is_valid(self) -> StreamingEvaluationCase:
        sessions = {event.session_id for event in self.transcript}
        utterances = {event.utterance_id for event in self.transcript}
        if sessions != {self.session_id}:
            raise ValueError("streaming evaluation transcript session IDs must match session_id")
        if len(utterances) != 1:
            raise ValueError("a streaming evaluation case must contain one utterance")
        final_events = [event for event in self.transcript if event.is_final]
        if len(final_events) != 1:
            raise ValueError("a streaming evaluation case must contain exactly one final event")
        if self.previous_answer_context is not None and self.previous_answer_context.session_id != self.session_id:
            raise ValueError("previous answer context must belong to the case session")
        if self.eligible_for_early_retrieval:
            if self.earliest_reasonable_retrieval_event_id is None:
                raise ValueError("eligible cases require an earliest reasonable event ID")
            event_by_id = {event.event_id: event for event in self.transcript}
            earliest = event_by_id.get(self.earliest_reasonable_retrieval_event_id)
            if earliest is None or earliest.is_final:
                raise ValueError("earliest reasonable retrieval event must be a non-final case event")
            if self.earliest_reasonable_retrieval_source_timestamp_s is None:
                raise ValueError("eligible cases require earliest reasonable source time")
        elif self.earliest_reasonable_retrieval_event_id is not None:
            raise ValueError("ineligible cases must not define an early retrieval event")
        relevant_ids = {
            chunk_id for chunk_id, label in self.relevance_labels.items() if label == "relevant"
        }
        if relevant_ids != set(self.expected_final_evidence_chunk_ids):
            raise ValueError("relevance label relevant IDs must match expected final evidence IDs")
        if self.expected_no_retrieval and self.expected_final_evidence_chunk_ids:
            raise ValueError("no-retrieval cases must not require final evidence IDs")
        if self.retrieval_behavior == "delay" and self.simulated_retrieval_delay_s <= 0:
            raise ValueError("delay cases require a positive simulated retrieval delay")
        if self.retrieval_behavior == "timeout" and self.simulated_retrieval_delay_s <= self.retrieval_timeout_s:
            raise ValueError("timeout cases require delay greater than retrieval timeout")
        return self


class StreamingModeEvaluationResult(ContractModel):
    """Matched baseline or streaming outcome for one Phase 2 case."""

    schema_version: Literal[PHASE2_SCHEMA_VERSION] = PHASE2_SCHEMA_VERSION
    mode: Literal["baseline", "streaming"]
    execution_mode: Literal["realtime", "accelerated"]
    run_status: Literal["completed", "needs_context", "abstained", "failed", "closed", "not_run"]
    trace_complete: bool = False
    missing_trace_events: list[str] = Field(default_factory=list)
    trace_event_count: int = Field(default=0, ge=0)
    final_query: str | None = None
    final_decision: Literal["WAIT", "RETRIEVE", "SKIP"] | None = None
    decision_reasons: list[str] = Field(default_factory=list)
    retrieval_started_early: bool = False
    valid_evidence_ready_before_finalization: bool = False
    early_evidence_reused: bool = False
    retrieval_call_count: int = Field(default=0, ge=0)
    scheduled_request_count: int = Field(default=0, ge=0)
    superseded_request_count: int = Field(default=0, ge=0)
    cancelled_request_count: int = Field(default=0, ge=0)
    timed_out_request_count: int = Field(default=0, ge=0)
    retrieval_error_count: int = Field(default=0, ge=0)
    stale_result_discard_count: int = Field(default=0, ge=0)
    stale_result_accepted_count: int = Field(default=0, ge=0)
    final_evidence_chunk_ids: list[str] = Field(default_factory=list)
    expected_final_evidence_chunk_ids: list[str] = Field(default_factory=list)
    # Explicit comparison aliases retained in the report so a reader does
    # not have to infer which list is retrieved evidence versus the labels.
    retrieved_ids: list[str] = Field(default_factory=list)
    relevant_ids: list[str] = Field(default_factory=list)
    request_status: str = "not_scheduled"
    request_statuses: list[str] = Field(default_factory=list)
    request_queries: list[str] = Field(default_factory=list)
    request_details: list[dict[str, Any]] = Field(default_factory=list)
    reuse_decision: str = "not_applicable"
    reuse_validation: dict[str, Any] = Field(default_factory=dict)
    scoring_denominator: int = Field(default=0, ge=0)
    quality_scored: bool = False
    quality_exclusion_reason: str | None = None
    end_to_end_success: bool = False
    final_evidence_recall_at_k: dict[str, float] = Field(default_factory=dict)
    final_evidence_mrr: float | None = Field(default=None, ge=0, le=1)
    answer_latency_from_final_event_delivery_ms: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    full_interaction_duration_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    controller_overhead_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    timing_sample_count: int = Field(default=0, ge=0)
    generation_status: str = "not_run"
    generation_usage: Usage = Field(default_factory=Usage)
    retrieval_usage: Usage = Field(default_factory=Usage)
    generation_cost: float | Literal["unavailable"] = "unavailable"
    cost_availability: Literal["available", "unavailable", "not_applicable"] = "unavailable"
    errors: list[str] = Field(default_factory=list)
    trace_timeline: list[dict[str, Any]] = Field(default_factory=list)


class StreamingEvaluationCaseResult(ContractModel):
    """Transparent paired result; semantic shortcomings remain visible."""

    schema_version: Literal[PHASE2_SCHEMA_VERSION] = PHASE2_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    split: Literal["development", "held_out"]
    scenario_group: str = Field(min_length=1)
    asset_status: Literal["official", "synthetic_fixture", "unknown"]
    labels_status: Literal["provisional_generated", "human_review_pending", "human_reviewed"]
    baseline: StreamingModeEvaluationResult | None = None
    streaming: StreamingModeEvaluationResult | None = None
    expected_behavior_met: bool
    failure_flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    _ids_are_non_blank = field_validator("case_id", "scenario_group")(_non_blank)


class StreamingEvaluationSection(ContractModel):
    """One provenance-separated Phase 2 evaluation source/condition."""

    schema_version: Literal[PHASE2_SCHEMA_VERSION] = PHASE2_SCHEMA_VERSION
    section_id: Literal["fixture", "simulated_delay", "real_backend", "official_assets"]
    status: Literal["measured", "not_verified", "blocked", "failed"]
    result_group: str = Field(min_length=1)
    asset_status: Literal["official", "synthetic_fixture", "unknown"]
    corpus_id: str | None = None
    index_id: str | None = None
    corpus_source_kind: Literal["official", "synthetic_fixture", "unknown"]
    retrieval_backend: RetrievalMode | None = None
    generation_backend: Literal["mock", "openai_compatible"] | None = None
    generation_model: str | None = None
    execution_mode: Literal["realtime", "accelerated"] | None = None
    case_count: int = Field(ge=0)
    case_results: list[StreamingEvaluationCaseResult] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    timing: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    cost_availability: str = Field(min_length=1)
    trace_coverage: dict[str, int] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class StreamingEvaluationReport(ContractModel):
    """Machine-readable Phase 2 streaming audit, not an official benchmark."""

    schema_version: Literal[PHASE2_SCHEMA_VERSION] = PHASE2_SCHEMA_VERSION
    report_version: Literal["flowcontext.streaming-evaluation.v1"] = (
        "flowcontext.streaming-evaluation.v1"
    )
    evaluation_label: str = Field(min_length=1)
    asset_status: Literal["official", "synthetic_fixture", "unknown"]
    selected_split: Literal["development", "held_out", "all"]
    development_case_count: int = Field(ge=0)
    held_out_case_count: int = Field(ge=0)
    measured_case_count: int = Field(ge=0)
    labels_status: Literal["provisional_generated", "human_review_pending", "human_reviewed"]
    competition_performance_claim: Literal[False] = False
    passed: bool
    created_at_utc: str = Field(min_length=1)
    code_revision: str | None = None
    configuration: dict[str, Any] = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    index_id: str = Field(min_length=1)
    model_identities: dict[str, str] = Field(min_length=1)
    execution_mode: Literal["realtime", "accelerated"]
    warm_cold_condition: str = Field(min_length=1)
    environment: dict[str, str] = Field(min_length=1)
    hardware: dict[str, str] = Field(min_length=1)
    guide_early_retrieval_target: float = Field(default=0.8, ge=0, le=1)
    guide_early_retrieval_target_met: bool
    false_trigger_threshold: str = Field(min_length=1)
    metrics: dict[str, float] = Field(default_factory=dict)
    sections: dict[str, StreamingEvaluationSection] = Field(min_length=4)
    duplicate_suppression_comparison: dict[str, Any] = Field(default_factory=dict)
    official_assets_available: bool = False
    limitations: list[str] = Field(default_factory=list)

    _label_is_non_blank = field_validator("evaluation_label", "created_at_utc")(_non_blank)
    _code_revision_is_non_blank = field_validator("code_revision")(_optional_non_blank)

    @field_validator("model_identities")
    @classmethod
    def streaming_identity_values_are_non_blank(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("model identity keys and values must not be blank")
        return values

    @model_validator(mode="after")
    def required_streaming_sections_are_present(self) -> StreamingEvaluationReport:
        required = {"fixture", "simulated_delay", "real_backend", "official_assets"}
        missing = required - set(self.sections)
        if missing:
            raise ValueError(
                "streaming evaluation report must contain fixture, simulated_delay, "
                "real_backend, and official_assets sections"
            )
        return self


class DecompositionConstraint(ContractModel):
    """One constraint copied from the parent transcript."""

    constraint_id: str = Field(min_length=1)
    kind: Literal[
        "entity",
        "quantity",
        "date",
        "location",
        "negation",
        "comparison",
        "relationship",
        "temporal",
        "other",
    ]
    value: str = Field(min_length=1)
    source_span: TextSpan
    operator: str | None = None
    normalized_value: str | None = None

    _ids_and_values_are_non_blank = field_validator("constraint_id", "value")(_non_blank)
    _optional_values_are_non_blank = field_validator("operator", "normalized_value")(
        _optional_non_blank
    )


class DecompositionIntent(ContractModel):
    """One retrieval question and its intent-local constraints."""

    intent_id: str = Field(min_length=1)
    ordinal: int = Field(ge=1)
    query: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    source_span: TextSpan | None = None
    constraints: list[DecompositionConstraint] = Field(default_factory=list)
    relationship: Literal["independent", "dependent", "comparison"] = "independent"

    _values_are_non_blank = field_validator("intent_id", "query", "source_text")(_non_blank)

    @property
    def intent_specific_constraints(self) -> list[DecompositionConstraint]:
        """Readable alias used by decomposition consumers."""

        return self.constraints


class DecompositionDependency(ContractModel):
    """A relation between two questions in one parent request."""

    dependency_id: str = Field(min_length=1)
    prerequisite_intent_id: str = Field(min_length=1)
    dependent_intent_id: str = Field(min_length=1)
    relation: Literal["depends_on", "compares_with", "refines"]
    description: str = Field(min_length=1)
    source_span: TextSpan | None = None

    _values_are_non_blank = field_validator(
        "dependency_id",
        "prerequisite_intent_id",
        "dependent_intent_id",
        "description",
    )(_non_blank)


class DecompositionAmbiguity(ContractModel):
    """An unresolved reference or interpretation retained for grounding."""

    ambiguity_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source_span: TextSpan
    candidates: list[str] = Field(default_factory=list)
    resolution: str | None = None

    _values_are_non_blank = field_validator("ambiguity_id", "description")(_non_blank)
    _resolution_is_non_blank = field_validator("resolution")(_optional_non_blank)


class DecompositionResult(ContractModel):
    """Validated, revision-bound decomposition of one transcript revision.

    This contract is intentionally independent of a retrieval result.  A plan
    may be a successful model/rule decomposition or an explicitly labelled
    original-query fallback after provider failure; fallback is never encoded
    as successful decomposition.
    """

    schema_version: Literal[PHASE3_SCHEMA_VERSION] = PHASE3_SCHEMA_VERSION
    transcript_revision: int = Field(default=0, ge=0)
    original_transcript: str = Field(min_length=1)
    intents: list[DecompositionIntent] = Field(min_length=1, max_length=8)
    boundary_count: int = Field(default=0, ge=0)
    shared_constraints: list[DecompositionConstraint] = Field(default_factory=list)
    ambiguities: list[DecompositionAmbiguity] = Field(default_factory=list)
    dependencies: list[DecompositionDependency] = Field(default_factory=list)
    decomposition_method: str = Field(min_length=1)
    status: Literal["success", "fallback"] = "success"
    failure_reason: str | None = None
    provider: str | None = None
    provider_model: str | None = None
    latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    usage: Usage = Field(default_factory=Usage)
    attempts: int = Field(default=1, ge=0)
    repair_attempts: int = Field(default=0, ge=0)
    preserved_intent_ids: list[str] = Field(default_factory=list)
    superseded_intent_ids: list[str] = Field(default_factory=list)

    _original_transcript_is_non_blank = field_validator("original_transcript")(_non_blank)
    _method_is_non_blank = field_validator("decomposition_method")(_non_blank)
    _optional_values_are_non_blank = field_validator(
        "failure_reason", "provider", "provider_model"
    )(_optional_non_blank)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_parent_query_input(cls, values: Any) -> Any:
        """Keep the pre-contract constructor spelling source-compatible.

        ``parent_query`` remains a read-only property, but older callers may
        still construct a ``MultiIntentPlan`` with that field name.  Translate
        it before the strict contract rejects unknown fields; the serialized
        contract continues to expose only ``original_transcript``.
        """

        if isinstance(values, dict) and "parent_query" in values:
            values = dict(values)
            values.setdefault("original_transcript", values.pop("parent_query"))
        return values

    @model_validator(mode="after")
    def decomposition_references_are_valid(self) -> DecompositionResult:
        intent_ids = [intent.intent_id for intent in self.intents]
        if len(intent_ids) != len(set(intent_ids)):
            raise ValueError("intent IDs must be unique within a decomposition")
        if [intent.ordinal for intent in self.intents] != list(range(1, len(self.intents) + 1)):
            raise ValueError("intent ordinals must be contiguous and ordered")
        if self.boundary_count != max(0, len(self.intents) - 1):
            raise ValueError("boundary_count must match the number of intent boundaries")
        for intent in self.intents:
            if intent.source_span is None:
                raise ValueError("every intent must include a source_span")
            if self.original_transcript[intent.source_span.start:intent.source_span.end] != intent.source_span.text:
                raise ValueError("intent source_span must reference original_transcript")
            if intent.source_text != intent.source_span.text:
                raise ValueError("intent source_text must equal its source_span text")
        constraint_ids = [
            constraint.constraint_id
            for constraint in self.shared_constraints
        ] + [
            constraint.constraint_id
            for intent in self.intents
            for constraint in intent.constraints
        ]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint IDs must be unique within a decomposition")
        all_constraints = list(self.shared_constraints) + [
            constraint for intent in self.intents for constraint in intent.constraints
        ]
        for constraint in all_constraints:
            if self.original_transcript[constraint.source_span.start:constraint.source_span.end] != constraint.source_span.text:
                raise ValueError("constraint source_span must reference original_transcript")
            if constraint.value.casefold() not in constraint.source_span.text.casefold():
                raise ValueError("constraint value must be copied from its source span")
        for ambiguity in self.ambiguities:
            if self.original_transcript[ambiguity.source_span.start:ambiguity.source_span.end] != ambiguity.source_span.text:
                raise ValueError("ambiguity source_span must reference original_transcript")
        for dependency in self.dependencies:
            if dependency.source_span is not None and self.original_transcript[dependency.source_span.start:dependency.source_span.end] != dependency.source_span.text:
                raise ValueError("dependency source_span must reference original_transcript")
        dependency_ids = [dependency.dependency_id for dependency in self.dependencies]
        if len(dependency_ids) != len(set(dependency_ids)):
            raise ValueError("dependency IDs must be unique within a decomposition")
        for dependency in self.dependencies:
            if dependency.prerequisite_intent_id not in intent_ids:
                raise ValueError("dependency prerequisite must reference an intent")
            if dependency.dependent_intent_id not in intent_ids:
                raise ValueError("dependency dependent intent must reference an intent")
            if dependency.prerequisite_intent_id == dependency.dependent_intent_id:
                raise ValueError("an intent cannot depend on itself")
        dependency_graph: dict[str, list[str]] = {intent_id: [] for intent_id in intent_ids}
        for dependency in self.dependencies:
            if dependency.relation in {"depends_on", "refines"}:
                dependency_graph[dependency.prerequisite_intent_id].append(
                    dependency.dependent_intent_id
                )
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(intent_id: str) -> None:
            if intent_id in visiting:
                raise ValueError("decomposition dependencies must be acyclic")
            if intent_id in visited:
                return
            visiting.add(intent_id)
            for dependent_id in dependency_graph[intent_id]:
                visit(dependent_id)
            visiting.remove(intent_id)
            visited.add(intent_id)

        for intent_id in intent_ids:
            visit(intent_id)
        if self.status == "fallback" and self.failure_reason is None:
            raise ValueError("fallback decompositions require failure_reason")
        if self.status == "success" and self.failure_reason is not None:
            raise ValueError("successful decompositions must not contain failure_reason")
        return self

    @property
    def parent_query(self) -> str:
        """Phase 3 adapter spelling retained for existing retrieval code."""

        return self.original_transcript

    @property
    def original_transcript_revision(self) -> int:
        return self.transcript_revision

    @property
    def sub_questions(self) -> list[str]:
        return [intent.query for intent in self.intents]

    @property
    def intent_ids(self) -> list[str]:
        """Stable IDs in the same order as :attr:`sub_questions`."""

        return [intent.intent_id for intent in self.intents]

    @property
    def is_multi_intent(self) -> bool:
        return len(self.intents) > 1

    @property
    def intent_specific_constraints(self) -> dict[str, list[DecompositionConstraint]]:
        return {intent.intent_id: intent.constraints for intent in self.intents}


class DecompositionRequest(ContractModel):
    """Provider input for one bounded decomposition attempt."""

    schema_version: Literal[PHASE3_SCHEMA_VERSION] = PHASE3_SCHEMA_VERSION
    transcript_revision: int = Field(ge=0)
    original_transcript: str = Field(min_length=1)
    max_intents: int = Field(default=6, ge=1, le=8)
    attempt: int = Field(default=1, ge=1)
    repair_feedback: str | None = None

    _transcript_is_non_blank = field_validator("original_transcript")(_non_blank)
    _repair_feedback_is_non_blank = field_validator("repair_feedback")(_optional_non_blank)


class Phase3ExpectedConstraint(ContractModel):
    """One external evaluation label for a shared or intent-local constraint."""

    kind: str = Field(min_length=1)
    value: str = Field(min_length=1)
    scope: Literal["shared", "intent"] = "intent"

    _values_are_non_blank = field_validator("kind", "value")(_non_blank)


class Phase3ExpectedIntent(ContractModel):
    """External gold label for one information need in a Phase 3 case.

    These labels are deliberately separate from the application decomposition
    contract.  The evaluator compares a produced plan with this record; the
    runtime never imports case IDs, expected answers, or expected chunk IDs.
    """

    intent_key: str = Field(min_length=1)
    expected_relationship: Literal["single", "independent", "dependent", "comparison"]
    source_hint: str = Field(min_length=1)
    answerable: bool
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevance_labels: dict[str, Literal["relevant", "not_relevant"]] = Field(default_factory=dict)
    expected_answer_substrings: list[str] = Field(default_factory=list)
    expected_constraints: list[Phase3ExpectedConstraint] = Field(default_factory=list)

    _values_are_non_blank = field_validator("intent_key", "source_hint")(_non_blank)

    @field_validator("relevant_chunk_ids")
    @classmethod
    def relevant_ids_are_unique_and_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("relevant_chunk_ids must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("relevant_chunk_ids must be unique")
        return values

    @field_validator("expected_answer_substrings")
    @classmethod
    def answer_labels_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("expected_answer_substrings must not contain blank values")
        return values

    @model_validator(mode="after")
    def relevance_labels_match_relevant_ids(self) -> Phase3ExpectedIntent:
        relevant = {
            chunk_id
            for chunk_id, label in self.relevance_labels.items()
            if label == "relevant"
        }
        if relevant != set(self.relevant_chunk_ids):
            raise ValueError(
                "relevance_labels with label relevant must match relevant_chunk_ids"
            )
        if not self.answerable and (self.relevant_chunk_ids or self.expected_answer_substrings):
            raise ValueError("unanswerable intents must not have answer evidence labels")
        return self


class Phase3LabelReviewStatus(ContractModel):
    """Review provenance for each family of external Phase 3 labels."""

    intent_labels: Literal[
        "provisional_generated", "model_reviewed", "human_review_pending", "human_reviewed"
    ] = "provisional_generated"
    relevance_labels: Literal[
        "provisional_generated", "model_reviewed", "human_review_pending", "human_reviewed"
    ] = "provisional_generated"
    answer_expectations: Literal[
        "provisional_generated", "model_reviewed", "human_review_pending", "human_reviewed"
    ] = "provisional_generated"
    reviewer: str | None = None
    review_notes: list[str] = Field(default_factory=list)

    _reviewer_is_non_blank = field_validator("reviewer")(_optional_non_blank)

    @field_validator("review_notes")
    @classmethod
    def review_notes_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("review_notes must not contain blank values")
        return values


class Phase3EvaluationCase(ContractModel):
    """Dedicated Phase 3 evaluation case loaded from external JSONL labels."""

    schema_version: Literal[PHASE3_EVALUATION_SCHEMA_VERSION] = PHASE3_EVALUATION_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    split: Literal["development", "held_out"]
    scenario_group: str = Field(min_length=1)
    variant_family: str = Field(min_length=1)
    evaluation_role: Literal[
        "historical_baseline",
        "diagnostic_regression",
        "untouched_generalization",
    ] = "historical_baseline"
    asset_status: Literal["official", "synthetic_fixture", "unknown"]
    transcript: list[TranscriptEvent] = Field(min_length=1)
    answerability: Literal["answerable", "unanswerable", "partially_answerable"]
    expected_intents: list[Phase3ExpectedIntent] = Field(min_length=1, max_length=8)
    shared_constraints: list[Phase3ExpectedConstraint] = Field(default_factory=list)
    expected_answer_substrings: list[str] = Field(default_factory=list)
    require_early_retrieval: bool = False
    controller_overrides: dict[str, Any] = Field(default_factory=dict)
    retrieval_behavior: Literal["normal", "delay", "failure", "timeout", "session_close"] = "normal"
    retrieval_delay_s: float = Field(default=0.0, ge=0, le=2.0, allow_inf_nan=False)
    retrieval_timeout_s: float = Field(default=0.05, ge=0.01, le=2.0, allow_inf_nan=False)
    provider_failure_stage: Literal["none", "decomposition", "generation"] = "none"
    label_review_status: Phase3LabelReviewStatus = Field(default_factory=Phase3LabelReviewStatus)
    implementation_change_ids: list[str] = Field(default_factory=list)
    case_notes: list[str] = Field(default_factory=list)

    _ids_are_non_blank = field_validator("case_id", "session_id", "scenario_group", "variant_family")(
        _non_blank
    )

    @field_validator("expected_answer_substrings", "implementation_change_ids", "case_notes")
    @classmethod
    def case_lists_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Phase 3 case lists must not contain blank values")
        return values

    @model_validator(mode="after")
    def case_structure_is_valid(self) -> Phase3EvaluationCase:
        sessions = {event.session_id for event in self.transcript}
        utterances = {event.utterance_id for event in self.transcript}
        if sessions != {self.session_id}:
            raise ValueError("Phase 3 transcript session IDs must match case session_id")
        if len(utterances) != 1:
            raise ValueError("a Phase 3 case must contain one utterance")
        final_events = [event for event in self.transcript if event.is_final]
        if len(final_events) != 1:
            raise ValueError("a Phase 3 case must contain exactly one final event")
        intent_keys = [intent.intent_key for intent in self.expected_intents]
        if len(intent_keys) != len(set(intent_keys)):
            raise ValueError("Phase 3 expected intent keys must be unique")
        if any(
            constraint.scope != "intent"
            for intent in self.expected_intents
            for constraint in intent.expected_constraints
        ):
            raise ValueError("expected intent constraints must have scope='intent'")
        if any(constraint.scope != "shared" for constraint in self.shared_constraints):
            raise ValueError("shared_constraints must have scope='shared'")
        if self.answerability == "answerable" and not all(
            intent.answerable for intent in self.expected_intents
        ):
            raise ValueError("answerable cases must not contain unanswerable expected intents")
        if self.answerability == "unanswerable" and any(
            intent.answerable for intent in self.expected_intents
        ):
            raise ValueError("unanswerable cases must not contain answerable expected intents")
        if self.answerability == "partially_answerable":
            if not any(intent.answerable for intent in self.expected_intents):
                raise ValueError("partially answerable cases require an answerable intent")
            if not any(not intent.answerable for intent in self.expected_intents):
                raise ValueError("partially answerable cases require an unsupported intent")
        if self.require_early_retrieval and not any(not event.is_final for event in self.transcript):
            raise ValueError("cases requiring early retrieval need a non-final event")
        if self.retrieval_behavior == "delay" and self.retrieval_delay_s <= 0:
            raise ValueError("delay cases require a positive retrieval_delay_s")
        if self.retrieval_behavior == "timeout" and self.retrieval_delay_s <= self.retrieval_timeout_s:
            raise ValueError("timeout cases require delay greater than retrieval_timeout_s")
        return self


class Phase4ConstraintRecord(ContractModel):
    """A session-scoped requested constraint, deliberately not corpus evidence."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    record_id: str = Field(min_length=1)
    constraint: DecompositionConstraint
    constraint_id: str = Field(min_length=1)
    origin: Literal["user_initial", "user_follow_up", "inherited_context"]
    origin_turn: int = Field(ge=0)
    origin_utterance_id: str = Field(min_length=1)
    origin_revision: int = Field(ge=0)
    scope: Literal["shared", "intent"] = "intent"
    intent_id: str | None = None
    status: Literal["active", "superseded", "removed"] = "active"
    # This literal is intentional: a request constraint is never silently
    # promoted to a fact merely because retrieval later returns a match.
    corpus_fact: Literal[False] = False
    resolution_source: str | None = None

    _ids_are_non_blank = field_validator(
        "record_id", "constraint_id", "origin_utterance_id"
    )(_non_blank)
    _resolution_source_is_non_blank = field_validator("resolution_source")(
        _optional_non_blank
    )

    @model_validator(mode="after")
    def constraint_record_is_consistent(self) -> Phase4ConstraintRecord:
        if self.constraint_id != self.constraint.constraint_id:
            raise ValueError("constraint_id must match the wrapped decomposition constraint")
        if self.scope == "intent" and self.intent_id is None:
            raise ValueError("intent-scoped constraints require an intent_id")
        if self.scope == "shared" and self.intent_id is not None:
            raise ValueError("shared constraints must not have an intent_id")
        return self


class Phase4IntentRecord(ContractModel):
    """An active or historical intent with the turn that introduced it."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    intent: DecompositionIntent
    origin_turn: int = Field(ge=0)
    origin_utterance_id: str = Field(min_length=1)
    origin_revision: int = Field(ge=0)
    topic_key: str = Field(min_length=1)
    status: Literal["active", "superseded"] = "active"

    _ids_are_non_blank = field_validator("origin_utterance_id", "topic_key")(_non_blank)

    @property
    def intent_id(self) -> str:
        return self.intent.intent_id


class Phase4EvidenceRecord(ContractModel):
    """Retrieval provenance bound to one session/corpus/index identity."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    evidence_id: str = Field(min_length=1)
    passage: EvidencePassage
    session_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    transcript_revision: int = Field(ge=0)
    retrieval_revision: int = Field(ge=0)
    corpus_id: str = Field(min_length=1)
    index_id: str = Field(min_length=1)
    intent_ids: list[str] = Field(default_factory=list)
    dependency_ids: list[str] = Field(default_factory=list)
    status: Literal["current", "superseded", "stale"] = "current"
    reused_from_evidence_id: str | None = None
    reuse_reason: str | None = None

    _ids_are_non_blank = field_validator(
        "evidence_id", "session_id", "utterance_id", "corpus_id", "index_id"
    )(_non_blank)
    _reused_from_is_non_blank = field_validator("reused_from_evidence_id")(
        _optional_non_blank
    )
    _reuse_reason_is_non_blank = field_validator("reuse_reason")(_optional_non_blank)

    @field_validator("intent_ids", "dependency_ids")
    @classmethod
    def evidence_id_lists_are_valid(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("evidence ID lists must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("evidence ID lists must be unique")
        return values

    @model_validator(mode="after")
    def passage_identity_is_consistent(self) -> Phase4EvidenceRecord:
        passage_intents = set(self.passage.intent_ids)
        if passage_intents and not passage_intents <= set(self.intent_ids):
            raise ValueError("passage intent provenance must be included in the evidence record")
        if self.passage.intent_id is not None and self.passage.intent_id not in self.intent_ids:
            raise ValueError("passage intent_id must be included in the evidence record")
        return self


class Phase4ClaimDependency(ContractModel):
    """One explicit claim dependency; IDs are not treated as support verdicts."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    dependency_id: str = Field(min_length=1)
    dependency_type: Literal["evidence", "constraint", "entity", "claim"]
    target_id: str = Field(min_length=1)
    relation: Literal["supports", "depends_on", "refines", "compares_with"] = "depends_on"
    target_value: str | None = None

    _ids_are_non_blank = field_validator("dependency_id", "target_id")(_non_blank)
    _target_value_is_non_blank = field_validator("target_value")(_optional_non_blank)


class Phase4ClaimRecord(ContractModel):
    """A versioned claim with evidence and non-evidence dependencies."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    record_id: str = Field(min_length=1)
    claim: FactualClaim
    claim_id: str = Field(min_length=1)
    answer_version: int = Field(ge=1)
    intent_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    dependencies: list[Phase4ClaimDependency] = Field(default_factory=list)
    claim_revision: int = Field(ge=0)
    semantic_support_status: Literal["unreviewed", "supported", "unsupported", "uncertain"] = "unreviewed"
    status: Literal["current", "non_current", "superseded"] = "current"

    _ids_are_non_blank = field_validator("record_id", "claim_id")(_non_blank)

    @field_validator("intent_ids", "supporting_evidence_ids")
    @classmethod
    def claim_id_lists_are_valid(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("claim ID lists must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("claim ID lists must be unique")
        return values

    @model_validator(mode="after")
    def claim_record_is_consistent(self) -> Phase4ClaimRecord:
        if self.claim_id != self.claim.claim_id:
            raise ValueError("claim_id must match the wrapped factual claim")
        if not self.intent_ids and self.claim.intent_ids:
            self.intent_ids = list(self.claim.intent_ids)
        if not self.intent_ids:
            raise ValueError("claim records must identify at least one intent")
        if not self.supporting_evidence_ids:
            raise ValueError("claim records must identify supporting evidence records")
        claim_intents = set(self.claim.intent_ids)
        if self.intent_ids and claim_intents and not claim_intents <= set(self.intent_ids):
            raise ValueError("claim intent provenance must be included in the claim record")
        return self


class Phase4ClaimChange(ContractModel):
    """Append-only comparison of one claim across answer publications."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    change_type: Literal["added", "removed", "modified", "preserved"]
    claim_id: str = Field(min_length=1)
    previous_claim_id: str | None = None
    current_claim_id: str | None = None
    previous_record_id: str | None = None
    current_record_id: str | None = None
    reason: str = Field(min_length=1)

    _ids_are_non_blank = field_validator(
        "claim_id",
        "previous_claim_id",
        "current_claim_id",
        "previous_record_id",
        "current_record_id",
    )(_optional_non_blank)
    _reason_is_non_blank = field_validator("reason")(_non_blank)

    @model_validator(mode="after")
    def claim_change_has_a_side(self) -> Phase4ClaimChange:
        if self.change_type == "added" and self.current_record_id is None:
            raise ValueError("added claim changes require a current record")
        if self.change_type == "removed" and self.previous_record_id is None:
            raise ValueError("removed claim changes require a previous record")
        if self.change_type in {"modified", "preserved"} and (
            self.previous_record_id is None or self.current_record_id is None
        ):
            raise ValueError("modified and preserved claim changes require both records")
        return self


class Phase4AnswerVersion(ContractModel):
    """An immutable, inspectable publication envelope around a grounded Answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    answer_version: int = Field(ge=1)
    answer: Answer
    claim_record_ids: list[str] = Field(default_factory=list)
    state_revision: int = Field(ge=0)
    created_at_utc: str = Field(min_length=1)
    immutable: Literal[True] = True
    parent_version: int | None = Field(default=None, ge=1)
    triggering_turn: int = Field(default=0, ge=0)
    triggering_utterance_id: str = "unknown"
    triggering_patch_id: str | None = None
    change_kind: Literal["initial", "factual_update", "presentation"] = "initial"
    constraint_change_ids: list[str] = Field(default_factory=list)
    evidence_added_ids: list[str] = Field(default_factory=list)
    evidence_removed_ids: list[str] = Field(default_factory=list)
    reused_evidence_ids: list[str] = Field(default_factory=list)
    claim_changes: list[Phase4ClaimChange] = Field(default_factory=list)
    retrieval_usage: Usage = Field(default_factory=Usage)
    generation_usage: Usage = Field(default_factory=Usage)
    presentation_usage: Usage = Field(default_factory=Usage)
    retrieval_call_count: int = Field(default=0, ge=0)
    retrieval_attempt_count: int = Field(default=0, ge=0)
    generation_attempts: int = Field(default=0, ge=0)
    generation_repair_attempts: int = Field(default=0, ge=0)
    retrieval_model_identity: str = "unknown"
    generation_execution_mode: Literal["rule_based", "mock_provider", "real_provider"] = "rule_based"
    generation_provider: str = "unknown"
    generation_model: str = "unknown"
    presentation_execution_mode: Literal["rule_based", "mock_provider", "real_provider"] = "rule_based"
    presentation_provider: str = "flowcontext.rule_based"
    status: Literal["completed", "partial", "failed", "superseded"] = "completed"
    failure_reason: str | None = None
    unresolved_intent_ids: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)

    _created_at_is_non_blank = field_validator(
        "created_at_utc", "triggering_utterance_id", "retrieval_model_identity",
        "generation_provider", "generation_model", "presentation_provider",
    )(_non_blank)
    _optional_values_are_non_blank = field_validator("triggering_patch_id", "failure_reason")(
        _optional_non_blank
    )

    @field_validator(
        "constraint_change_ids",
        "evidence_added_ids",
        "evidence_removed_ids",
        "reused_evidence_ids",
        "unresolved_intent_ids",
        "unresolved_questions",
    )
    @classmethod
    def answer_version_lists_are_valid(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("answer version lists must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("answer version lists must be unique")
        return values

    @model_validator(mode="after")
    def answer_version_is_consistent(self) -> Phase4AnswerVersion:
        if self.answer.answer_version != self.answer_version:
            raise ValueError("answer.answer_version must match the publication version")
        return self


class Phase4RevisionRecord(ContractModel):
    """Append-only audit record for a state transition or publication."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    revision_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    state_revision: int = Field(ge=0)
    transcript_revision: int = Field(ge=0)
    retrieval_revision: int = Field(ge=0)
    supersedes: list[int] = Field(default_factory=list)
    patch_id: str | None = None
    changed_intent_ids: list[str] = Field(default_factory=list)
    preserved_intent_ids: list[str] = Field(default_factory=list)
    discarded_result_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    created_at_utc: str = Field(min_length=1)
    directly_affected_intent_ids: list[str] = Field(default_factory=list)
    invalidated_claim_ids: list[str] = Field(default_factory=list)
    preserved_claim_ids: list[str] = Field(default_factory=list)
    reused_evidence_ids: list[str] = Field(default_factory=list)
    retrieval_reasons: dict[str, str] = Field(default_factory=dict)
    dependency_uncertain: bool = False
    dependency_uncertainty_reasons: list[str] = Field(default_factory=list)
    answer_version: int | None = Field(default=None, ge=1)
    parent_answer_version: int | None = Field(default=None, ge=1)
    publication_status: Literal["completed", "partial", "failed", "superseded"] | None = None
    claim_changes: list[Phase4ClaimChange] = Field(default_factory=list)
    evidence_added_ids: list[str] = Field(default_factory=list)
    evidence_removed_ids: list[str] = Field(default_factory=list)
    generation_usage: Usage = Field(default_factory=Usage)
    retrieval_usage: Usage = Field(default_factory=Usage)
    presentation_usage: Usage = Field(default_factory=Usage)
    retrieval_call_count: int = Field(default=0, ge=0)
    generation_attempts: int = Field(default=0, ge=0)

    _ids_are_non_blank = field_validator(
        "revision_id", "session_id", "utterance_id", "reason", "created_at_utc"
    )(_non_blank)
    _patch_id_is_non_blank = field_validator("patch_id")(_optional_non_blank)

    @field_validator(
        "changed_intent_ids",
        "preserved_intent_ids",
        "discarded_result_ids",
        "directly_affected_intent_ids",
        "invalidated_claim_ids",
        "preserved_claim_ids",
        "reused_evidence_ids",
        "dependency_uncertainty_reasons",
        "evidence_added_ids",
        "evidence_removed_ids",
    )
    @classmethod
    def revision_id_lists_are_valid(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("revision ID lists must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("revision ID lists must be unique")
        return values

    @field_validator("retrieval_reasons")
    @classmethod
    def revision_retrieval_reasons_are_valid(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("retrieval reasons must contain non-blank keys and values")
        return values


class Phase4ReferenceResolution(ContractModel):
    """An unambiguous contextual reference resolution, never an inferred fact."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    reference_text: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    entity_text: str = Field(min_length=1)
    source_intent_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    _values_are_non_blank = field_validator(
        "reference_text", "entity_id", "entity_text", "source_intent_id", "rationale"
    )(_non_blank)


class Phase4Clarification(ContractModel):
    """A targeted user question that leaves active semantic state unchanged."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    clarification_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reason_code: Literal["ambiguous_reference", "ambiguous_change", "missing_entity"]
    candidate_entity_ids: list[str] = Field(default_factory=list)
    related_intent_ids: list[str] = Field(default_factory=list)
    created_revision: int = Field(ge=0)
    status: Literal["pending", "resolved", "withdrawn"] = "pending"

    _ids_are_non_blank = field_validator("clarification_id", "question")(_non_blank)

    @field_validator("candidate_entity_ids", "related_intent_ids")
    @classmethod
    def clarification_ids_are_valid(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("clarification IDs must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("clarification IDs must be unique")
        return values


class Phase4PendingUpdate(ContractModel):
    """An accepted semantic/formatting request waiting for a later answer step."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    update_id: str = Field(min_length=1)
    patch_id: str = Field(min_length=1)
    base_revision: int = Field(ge=0)
    triggering_turn: int = Field(default=0, ge=0)
    triggering_utterance_id: str = "unknown"
    requested_kind: Literal[
        "add_constraint",
        "replace_constraint",
        "remove_constraint",
        "add_question",
        "reformat_answer",
        "change_topic",
    ]
    reason: str = Field(min_length=1)
    target_intent_ids: list[str] = Field(default_factory=list)
    requires_retrieval: bool
    format_instruction: str | None = None
    constraint_change_ids: list[str] = Field(default_factory=list)
    created_at_utc: str = Field(min_length=1)
    status: Literal["pending", "applied", "blocked"] = "pending"
    candidate_revision: int | None = Field(default=None, ge=1)
    selective_plan_id: str | None = None
    directly_affected_intent_ids: list[str] = Field(default_factory=list)
    invalidated_intent_ids: list[str] = Field(default_factory=list)
    invalidated_claim_ids: list[str] = Field(default_factory=list)
    preserved_claim_ids: list[str] = Field(default_factory=list)
    invalidation_reasons: dict[str, str] = Field(default_factory=dict)
    invalidated_evidence_ids: list[str] = Field(default_factory=list)
    reused_evidence_ids: list[str] = Field(default_factory=list)
    retrieval_intent_ids: list[str] = Field(default_factory=list)
    retrieval_reasons: dict[str, str] = Field(default_factory=dict)
    task_ids: list[str] = Field(default_factory=list)
    discarded_result_ids: list[str] = Field(default_factory=list)
    dependency_uncertain: bool = False
    dependency_uncertainty_reasons: list[str] = Field(default_factory=list)

    _ids_are_non_blank = field_validator(
        "update_id", "patch_id", "reason", "created_at_utc", "triggering_utterance_id"
    )(_non_blank)
    _format_instruction_is_non_blank = field_validator("format_instruction")(
        _optional_non_blank
    )
    _selective_plan_is_non_blank = field_validator("selective_plan_id")(_optional_non_blank)

    @field_validator(
        "target_intent_ids",
        "directly_affected_intent_ids",
        "invalidated_intent_ids",
        "invalidated_claim_ids",
        "preserved_claim_ids",
        "invalidated_evidence_ids",
        "reused_evidence_ids",
        "retrieval_intent_ids",
        "task_ids",
        "discarded_result_ids",
        "constraint_change_ids",
        "dependency_uncertainty_reasons",
    )
    @classmethod
    def pending_update_id_lists_are_valid(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("pending update ID lists must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("pending update ID lists must be unique")
        return values

    @field_validator("retrieval_reasons")
    @classmethod
    def pending_update_retrieval_reasons_are_valid(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("retrieval reasons must contain non-blank keys and values")
        return values

    @field_validator("invalidation_reasons")
    @classmethod
    def pending_update_invalidation_reasons_are_valid(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("invalidation reasons must contain non-blank keys and values")
        return values


class Phase4PendingRequest(ContractModel):
    """Bounded request bookkeeping; it contains no cross-session user profile."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    request_id: str = Field(min_length=1)
    kind: Literal["follow_up_patch", "answer_publication", "presentation", "clarification"]
    base_revision: int = Field(ge=0)
    status: Literal["pending", "completed", "superseded"] = "pending"

    _request_id_is_non_blank = field_validator("request_id")(_non_blank)


class Phase4DecisionStep(ContractModel):
    """Small explainable trace step retained on every proposed patch."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    stage: Literal[
        "classification",
        "reference_resolution",
        "target_selection",
        "materialization",
        "dependency_propagation",
    ]
    rule: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    matched_text: str | None = None
    target_ids: list[str] = Field(default_factory=list)

    _values_are_non_blank = field_validator("rule", "detail")(_non_blank)
    _matched_text_is_non_blank = field_validator("matched_text")(_optional_non_blank)


class Phase4SelectiveRetrievalTask(ContractModel):
    """One targeted retrieval work item bound to a candidate session revision."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    patch_id: str = Field(min_length=1)
    candidate_state_revision: int = Field(ge=1)
    candidate_transcript_revision: int = Field(ge=0)
    candidate_retrieval_revision: int = Field(ge=1)
    intent_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    status: Literal[
        "queued", "running", "completed", "failed", "superseded", "discarded"
    ] = "queued"
    scheduler_request_id: str | None = None
    attempts: int = Field(default=0, ge=0)
    evidence_ids: list[str] = Field(default_factory=list)
    discarded_result_ids: list[str] = Field(default_factory=list)
    error: str | None = None

    _ids_are_non_blank = field_validator(
        "task_id", "session_id", "patch_id", "intent_id", "query", "reason"
    )(_non_blank)
    _optional_ids_are_non_blank = field_validator("scheduler_request_id", "error")(
        _optional_non_blank
    )

    @field_validator("evidence_ids", "discarded_result_ids")
    @classmethod
    def retrieval_task_id_lists_are_valid(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("retrieval task ID lists must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("retrieval task ID lists must be unique")
        return values


class Phase4SelectiveUpdatePlan(ContractModel):
    """Audit record for dependency analysis and targeted retrieval decisions."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    plan_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    patch_id: str = Field(min_length=1)
    base_revision: int = Field(ge=0)
    candidate_revision: int = Field(ge=1)
    transcript_revision: int = Field(ge=0)
    retrieval_revision: int = Field(ge=1)
    directly_affected_intent_ids: list[str] = Field(default_factory=list)
    invalidated_intent_ids: list[str] = Field(default_factory=list)
    new_intent_ids: list[str] = Field(default_factory=list)
    affected_entity_ids: list[str] = Field(default_factory=list)
    invalidated_claim_ids: list[str] = Field(default_factory=list)
    preserved_claim_ids: list[str] = Field(default_factory=list)
    invalidation_reasons: dict[str, str] = Field(default_factory=dict)
    invalidated_evidence_ids: list[str] = Field(default_factory=list)
    reused_evidence_ids: list[str] = Field(default_factory=list)
    retrieval_intent_ids: list[str] = Field(default_factory=list)
    retrieval_reasons: dict[str, str] = Field(default_factory=dict)
    dependency_uncertain: bool = False
    dependency_uncertainty_reasons: list[str] = Field(default_factory=list)
    discarded_result_ids: list[str] = Field(default_factory=list)
    retrieval_tasks: list[Phase4SelectiveRetrievalTask] = Field(default_factory=list)
    status: Literal["pending", "running", "completed", "blocked", "superseded"] = "pending"
    created_at_utc: str = Field(min_length=1)
    retrieval_usage: Usage = Field(default_factory=Usage)
    retrieval_call_count: int = Field(default=0, ge=0)
    retrieval_attempt_count: int = Field(default=0, ge=0)

    _ids_are_non_blank = field_validator(
        "plan_id", "session_id", "patch_id", "created_at_utc"
    )(_non_blank)

    @field_validator(
        "directly_affected_intent_ids",
        "invalidated_intent_ids",
        "new_intent_ids",
        "affected_entity_ids",
        "invalidated_claim_ids",
        "preserved_claim_ids",
        "invalidated_evidence_ids",
        "reused_evidence_ids",
        "retrieval_intent_ids",
        "dependency_uncertainty_reasons",
        "discarded_result_ids",
    )
    @classmethod
    def selective_plan_id_lists_are_valid(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("selective plan ID lists must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("selective plan ID lists must be unique")
        return values

    @field_validator("retrieval_reasons")
    @classmethod
    def selective_plan_retrieval_reasons_are_valid(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("retrieval reasons must contain non-blank keys and values")
        return values

    @field_validator("invalidation_reasons")
    @classmethod
    def selective_plan_invalidation_reasons_are_valid(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("invalidation reasons must contain non-blank keys and values")
        return values

    @model_validator(mode="after")
    def selective_tasks_match_plan(self) -> Phase4SelectiveUpdatePlan:
        task_ids = [task.task_id for task in self.retrieval_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("selective retrieval task IDs must be unique within a plan")
        for task in self.retrieval_tasks:
            if (
                task.session_id != self.session_id
                or task.patch_id != self.patch_id
                or task.candidate_state_revision != self.candidate_revision
                or task.candidate_transcript_revision != self.transcript_revision
                or task.candidate_retrieval_revision != self.retrieval_revision
            ):
                raise ValueError("selective retrieval task does not match its candidate plan")
        if set(self.retrieval_intent_ids) != {task.intent_id for task in self.retrieval_tasks}:
            raise ValueError("retrieval_intent_ids must match the targeted task intents")
        if set(self.retrieval_reasons) != set(self.retrieval_intent_ids):
            raise ValueError("every targeted retrieval intent must have a reason")
        return self


class Phase4FollowUpRequest(ContractModel):
    """Input for interpretation; the caller supplies the exact base revision."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    base_revision: int = Field(ge=0)
    transcript_revision: int | None = Field(default=None, ge=0)

    _ids_are_non_blank = field_validator("session_id", "utterance_id", "text")(_non_blank)


class Phase4ProposedPatch(ContractModel):
    """Validated, non-mutating proposal against one exact session revision."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    patch_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    source_text: str = Field(min_length=1)
    base_revision: int = Field(ge=0)
    proposed_revision: int = Field(ge=1)
    transcript_revision: int = Field(ge=0)
    retrieval_revision: int = Field(ge=0)
    classification: Literal[
        "add_constraint",
        "replace_constraint",
        "remove_constraint",
        "add_question",
        "reformat_answer",
        "change_topic",
        "clarification_required",
    ]
    execution_mode: Literal["rule_based", "mock_provider", "real_provider"]
    provider_identity: str = Field(min_length=1)
    provider_model: str = Field(min_length=1)
    target_intent_ids: list[str] = Field(default_factory=list)
    intent_replacements: dict[str, str] = Field(default_factory=dict)
    added_constraints: list[Phase4ConstraintRecord] = Field(default_factory=list)
    replacement_constraints: dict[str, Phase4ConstraintRecord] = Field(default_factory=dict)
    removed_constraint_ids: list[str] = Field(default_factory=list)
    new_intents: list[Phase4IntentRecord] = Field(default_factory=list)
    reference_resolutions: list[Phase4ReferenceResolution] = Field(default_factory=list)
    topic_key: str | None = None
    format_instruction: str | None = None
    clarification: Phase4Clarification | None = None
    decision_trace: list[Phase4DecisionStep] = Field(default_factory=list)

    _ids_are_non_blank = field_validator(
        "patch_id", "session_id", "utterance_id", "source_text", "provider_identity", "provider_model"
    )(_non_blank)
    _topic_key_is_non_blank = field_validator("topic_key")(_optional_non_blank)
    _format_instruction_is_non_blank = field_validator("format_instruction")(
        _optional_non_blank
    )

    @field_validator("target_intent_ids", "removed_constraint_ids")
    @classmethod
    def patch_id_lists_are_valid(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("patch ID lists must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("patch ID lists must be unique")
        return values

    @model_validator(mode="after")
    def patch_revision_and_classification_are_consistent(self) -> Phase4ProposedPatch:
        if self.proposed_revision != self.base_revision + 1:
            raise ValueError("proposed_revision must be exactly one after base_revision")
        if self.classification == "clarification_required":
            if self.clarification is None:
                raise ValueError("clarification_required patches require clarification")
            if self.added_constraints or self.replacement_constraints or self.removed_constraint_ids or self.new_intents:
                raise ValueError("clarification patches must not contain semantic mutations")
        elif self.clarification is not None:
            raise ValueError("non-clarification patches must not contain a clarification")
        if self.classification == "reformat_answer" and not self.format_instruction:
            raise ValueError("reformat patches require format_instruction")
        if self.classification == "change_topic" and not self.topic_key:
            raise ValueError("change_topic patches require topic_key")
        if any(not key.strip() or not value.strip() for key, value in self.intent_replacements.items()):
            raise ValueError("intent_replacements must contain non-blank IDs")
        if len(self.intent_replacements) != len(set(self.intent_replacements.values())):
            raise ValueError("intent_replacements must not map multiple old IDs to one new ID")
        return self


class Phase4SessionState(ContractModel):
    """Bounded, session-only state for follow-up interpretation and updates."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    active_topic: str = Field(min_length=1)
    active_task: str = Field(min_length=1)
    current_utterance_id: str = Field(min_length=1)
    transcript_revision: int = Field(ge=0)
    state_revision: int = Field(ge=0)
    retrieval_revision: int = Field(ge=0)
    corpus_id: str = Field(min_length=1)
    index_id: str = Field(min_length=1)
    active_intents: list[Phase4IntentRecord] = Field(default_factory=list)
    intent_history: list[Phase4IntentRecord] = Field(default_factory=list)
    constraint_records: list[Phase4ConstraintRecord] = Field(default_factory=list)
    current_constraint_record_ids: list[str] = Field(default_factory=list)
    evidence_records: list[Phase4EvidenceRecord] = Field(default_factory=list)
    claim_records: list[Phase4ClaimRecord] = Field(default_factory=list)
    answer_versions: list[Phase4AnswerVersion] = Field(default_factory=list)
    current_answer_version: int | None = Field(default=None, ge=1)
    current_evidence_ids: list[str] = Field(default_factory=list)
    current_claim_record_ids: list[str] = Field(default_factory=list)
    pending_clarification: Phase4Clarification | None = None
    pending_update: Phase4PendingUpdate | None = None
    pending_requests: list[Phase4PendingRequest] = Field(default_factory=list)
    superseded_requests: list[Phase4PendingRequest] = Field(default_factory=list)
    generation_status: Literal["idle", "pending", "published", "blocked"] = "idle"
    answer_status: Literal[
        "unavailable", "current", "stale", "presentation_pending", "partial", "failed"
    ] = "unavailable"
    trace_ids: list[str] = Field(default_factory=list)
    revision_history: list[Phase4RevisionRecord] = Field(default_factory=list)
    selective_update_plans: list[Phase4SelectiveUpdatePlan] = Field(default_factory=list)

    _ids_are_non_blank = field_validator(
        "session_id", "active_topic", "active_task", "current_utterance_id", "corpus_id", "index_id"
    )(_non_blank)

    @field_validator("current_constraint_record_ids", "current_evidence_ids", "current_claim_record_ids", "trace_ids")
    @classmethod
    def state_id_lists_are_valid(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("state ID lists must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("state ID lists must be unique")
        return values

    @model_validator(mode="after")
    def state_references_are_valid(self) -> Phase4SessionState:
        intent_ids = {record.intent_id for record in self.active_intents}
        if len(intent_ids) != len(self.active_intents):
            raise ValueError("active intent IDs must be unique")
        constraint_record_ids = [record.record_id for record in self.constraint_records]
        if len(constraint_record_ids) != len(set(constraint_record_ids)):
            raise ValueError("constraint record IDs must be unique")
        current_constraints = {
            record.record_id: record
            for record in self.constraint_records
            if record.record_id in set(self.current_constraint_record_ids)
        }
        if set(current_constraints) != set(self.current_constraint_record_ids):
            raise ValueError("current constraint IDs must reference constraint records")
        if any(record.status != "active" for record in current_constraints.values()):
            raise ValueError("current constraint IDs must reference active constraints")
        if any(
            record.scope == "intent" and record.intent_id not in intent_ids
            for record in current_constraints.values()
        ):
            raise ValueError("current intent constraints must reference active intents")
        evidence_ids = [record.evidence_id for record in self.evidence_records]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique")
        evidence_by_id = {record.evidence_id: record for record in self.evidence_records}
        if not set(self.current_evidence_ids) <= set(evidence_by_id):
            raise ValueError("current evidence IDs must reference evidence records")
        if any(evidence_by_id[item].status != "current" for item in self.current_evidence_ids):
            raise ValueError("current evidence IDs must reference current evidence")
        if any(
            evidence_by_id[item].session_id != self.session_id
            or evidence_by_id[item].utterance_id != self.current_utterance_id
            or evidence_by_id[item].transcript_revision != self.transcript_revision
            or evidence_by_id[item].retrieval_revision != self.retrieval_revision
            or evidence_by_id[item].corpus_id != self.corpus_id
            or evidence_by_id[item].index_id != self.index_id
            for item in self.current_evidence_ids
        ):
            raise ValueError(
                "current evidence must match session, utterance, revision, corpus, and index identity"
            )
        if any(
            set(evidence_by_id[item].intent_ids) - intent_ids
            for item in self.current_evidence_ids
        ):
            raise ValueError("current evidence must reference active intents")
        claim_record_ids = [record.record_id for record in self.claim_records]
        if len(claim_record_ids) != len(set(claim_record_ids)):
            raise ValueError("claim record IDs must be unique")
        claims_by_id = {record.record_id: record for record in self.claim_records}
        if not set(self.current_claim_record_ids) <= set(claims_by_id):
            raise ValueError("current claim IDs must reference claim records")
        if any(claims_by_id[item].status != "current" for item in self.current_claim_record_ids):
            raise ValueError("current claim IDs must reference current claims")
        evidence_ids = set(self.current_evidence_ids)
        if any(
            set(claims_by_id[item].supporting_evidence_ids) - evidence_ids
            for item in self.current_claim_record_ids
        ):
            raise ValueError("current claims must reference current evidence records")
        if any(
            set(claims_by_id[item].intent_ids) - intent_ids
            for item in self.current_claim_record_ids
        ):
            raise ValueError("current claims must reference active intents")
        version_ids = [version.answer_version for version in self.answer_versions]
        if len(version_ids) != len(set(version_ids)):
            raise ValueError("answer versions must be unique")
        plan_ids = [plan.plan_id for plan in self.selective_update_plans]
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("selective update plan IDs must be unique")
        for plan in self.selective_update_plans:
            if plan.session_id != self.session_id:
                raise ValueError("selective update plan belongs to another session")
        if self.pending_update is not None and self.pending_update.selective_plan_id is not None:
            if self.pending_update.selective_plan_id not in set(plan_ids):
                raise ValueError("pending update must reference a recorded selective update plan")
            pending_plan = next(
                plan
                for plan in self.selective_update_plans
                if plan.plan_id == self.pending_update.selective_plan_id
            )
            if (
                self.pending_update.candidate_revision is not None
                and self.pending_update.candidate_revision != pending_plan.candidate_revision
            ):
                raise ValueError("pending update candidate revision must match its selective plan")
        versions = set(version_ids)
        if self.current_answer_version is not None and self.current_answer_version not in versions:
            raise ValueError("current_answer_version must reference an answer version")
        if self.answer_status in {"current", "partial", "failed", "presentation_pending"} and self.current_answer_version is None:
            raise ValueError("a published answer status requires a current answer version")
        if self.answer_status == "unavailable" and self.current_answer_version is not None:
            raise ValueError("unavailable answer status must not retain an answer pointer")
        if self.answer_status == "stale" and self.current_answer_version is None and self.current_claim_record_ids:
            raise ValueError("stale answer claims require a historical answer pointer")
        if self.current_claim_record_ids:
            if self.current_answer_version is None:
                raise ValueError("current claims require a current answer version")
            if any(
                claims_by_id[item].answer_version != self.current_answer_version
                for item in self.current_claim_record_ids
            ):
                raise ValueError("current claims must belong to the current answer version")
        if self.pending_clarification is not None and self.pending_clarification.status != "pending":
            raise ValueError("pending_clarification must have status=pending")
        return self

    @property
    def current_constraints(self) -> list[Phase4ConstraintRecord]:
        """Active constraints in stable insertion order."""

        wanted = set(self.current_constraint_record_ids)
        return [record for record in self.constraint_records if record.record_id in wanted]

    @property
    def current_evidence(self) -> list[Phase4EvidenceRecord]:
        wanted = set(self.current_evidence_ids)
        return [record for record in self.evidence_records if record.evidence_id in wanted]

    @property
    def current_claims(self) -> list[Phase4ClaimRecord]:
        wanted = set(self.current_claim_record_ids)
        return [record for record in self.claim_records if record.record_id in wanted]

    @property
    def current_answer(self) -> Phase4AnswerVersion | None:
        """Return the pointed-to publication, including when marked historical/stale."""

        if self.current_answer_version is None:
            return None
        return next(
            (
                version
                for version in self.answer_versions
                if version.answer_version == self.current_answer_version
            ),
            None,
        )

    @property
    def usable_answer(self) -> Phase4AnswerVersion | None:
        """Return content that may be presented for the current request."""

        if self.answer_status not in {"current", "partial", "presentation_pending"}:
            return None
        return self.current_answer


class Phase4ReplayTurn(ContractModel):
    """One JSONL turn consumed by the Phase 4 executable replay harness."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    utterance_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    _ids_are_non_blank = field_validator("session_id", "utterance_id", "text")(_non_blank)


class Phase4ReplayResult(ContractModel):
    """Inspectable output of the Phase 4 replay pipeline."""

    schema_version: Literal[PHASE4_SCHEMA_VERSION] = PHASE4_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    index_id: str = Field(min_length=1)
    corpus_source_kind: Literal["official", "synthetic_fixture", "unknown"]
    retrieval_backend: str = Field(min_length=1)
    generation_execution_mode: Literal["rule_based", "mock_provider", "real_provider"]
    generation_provider: str = Field(min_length=1)
    generation_model: str = Field(min_length=1)
    run_status: Literal["completed", "partial", "failed"]
    steps: list[dict[str, Any]] = Field(default_factory=list)
    state: Phase4SessionState
    retrieval_call_count: int = Field(default=0, ge=0)
    retrieval_attempt_count: int = Field(default=0, ge=0)
    generation_call_count: int = Field(default=0, ge=0)
    generation_attempt_count: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)

    _ids_are_non_blank = field_validator(
        "session_id", "corpus_id", "index_id", "retrieval_backend",
        "generation_provider", "generation_model",
    )(_non_blank)


# Public short names for callers that do not need the phase prefix.
SessionState = Phase4SessionState
ProposedPatch = Phase4ProposedPatch
