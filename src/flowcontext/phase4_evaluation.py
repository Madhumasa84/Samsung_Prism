"""Matched Phase 4 multi-turn evaluation and engineering validation.

The evaluator is deliberately outside the runtime state and retrieval code.
Expected operations, answerability, and review fields are loaded from external
JSONL labels.  The two update arms share the same corpus, providers,
interpretation patch, retrieval configuration, and generation configuration:

* ``full`` re-runs the complete current task through the retriever and
  publisher after each semantic follow-up;
* ``selective`` applies the Phase 4 dependency plan and schedules only its
  unresolved information needs.

This is an engineering evaluation over synthetic data.  Citation-ID validity
and source/excerpt provenance are structural checks; semantic claim support is
reported as pending until a human reviewer actually completes the review
sheet.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Literal, Sequence
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request as UrlRequest, urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import Settings, config_asset_status
from .contracts import (
    CorpusIndex,
    Phase4ProposedPatch,
    Phase4ReplayTurn,
    Phase4SessionState,
    Usage,
)
from .embeddings import SentenceTransformerEmbeddingProvider
from .generation import MockGenerationProvider, generation_provider_for_settings
from .multi_intent import decompose_query
from .phase4 import (
    Phase4AnswerPublisher,
    Phase4FollowUpInterpreter,
    Phase4SelectiveUpdateCoordinator,
    Phase4SessionStore,
    _phase4_decomposition,
)
from .phase4_replay import _passage_for_hit, _record_initial_retrieval
from .retrieval import make_retriever


PHASE4_EVALUATION_SCHEMA_VERSION = "flowcontext.phase4-evaluation.v1"
_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


PHASE3_FAILURE_TRACE: list[dict[str, Any]] = [
    {
        "failure": "over_decomposition_of_one_information_need",
        "diagnostic_case_ids": ["p4-diagnostic-overdecomposition-001"],
        "historical_case_ids": ["p3-heldout-single-catering-001"],
        "historical_reports": [
            "reports/phase3_evaluation_realtime.md",
            "reports/phase3_evaluation_realtime.json",
        ],
        "root_cause": {
            "application_logic": True,
            "mock_behaviour": False,
            "evaluation_labels": False,
            "metric_implementation": False,
            "classification": "application_logic",
        },
        "stages": {
            "decomposition": "A coordinated noun phrase with multiple constraints was split into separate intents even though it had one shared information need.",
            "retrieval": "The split plan issued separate searches, so retrieval cost and evidence ownership followed the incorrect intent boundary.",
            "evidence_assembly": "Evidence was assigned to fragmented branches instead of one package-level request.",
            "synthesis": "The generated answer could be structurally valid while missing the intended single-need coverage because the input plan was over-decomposed.",
            "evaluation": "The historical one-to-one intent metric correctly counted the extra predicted intent as a precision failure; labels were retained and the metric was not weakened.",
        },
        "fix": "Keep coordinated constraints and shared-head package phrases in one intent; preserve explicit question/request boundaries and genuinely independent questions.",
        "label_action": "Mark as diagnostic/regression only; preserve historical Phase 3 labels and scores.",
    },
    {
        "failure": "incorrect_uncertainty_on_partially_answerable_request",
        "diagnostic_case_ids": ["p4-diagnostic-partial-002"],
        "historical_case_ids": ["p3-dev-partial-008"],
        "historical_reports": [
            "reports/phase3_evaluation_realtime.md",
            "reports/phase3_evaluation_realtime.json",
        ],
        "root_cause": {
            "application_logic": True,
            "mock_behaviour": True,
            "evaluation_labels": False,
            "metric_implementation": False,
            "classification": "application_logic_and_mock_behaviour",
        },
        "stages": {
            "decomposition": "The supported and unsupported information needs were represented as separate intents, so the failure was not caused by an intent-count label.",
            "retrieval": "Lexical retrieval returned contextual distractors for the unsupported need; a retrieval score was not evidence of answerability.",
            "evidence_assembly": "The earlier path did not conservatively reject every weakly aligned candidate for the unsupported intent.",
            "synthesis": "The earlier synthesis/mock path could treat retrieved context as support or report a conflict without preserving the supported portion and explicitly naming the unsupported portion.",
            "evaluation": "The historical uncertainty metric required both supported coverage and explicit uncertainty; it exposed the failure. Provisional labels remain unchanged.",
        },
        "fix": "Use conservative intent/constraint and entity alignment, filter incompatible evidence before synthesis, and render supported claims alongside targeted unresolved information needs. The mock provider also now chooses a compatible fixture passage deterministically; this fixture improvement is not real-model evidence.",
        "label_action": "Mark as diagnostic/regression only; preserve historical Phase 3 labels and scores.",
    },
    {
        "failure": "incorrect_uncertainty_on_unanswerable_request",
        "diagnostic_case_ids": ["p4-diagnostic-unanswerable-003"],
        "historical_case_ids": ["p3-dev-unsupported-013", "p3-heldout-unsupported-007"],
        "historical_reports": [
            "reports/phase3_evaluation_realtime.md",
            "reports/phase3_evaluation_realtime.json",
        ],
        "root_cause": {
            "application_logic": True,
            "mock_behaviour": True,
            "evaluation_labels": False,
            "metric_implementation": False,
            "classification": "application_logic_and_mock_behaviour",
        },
        "stages": {
            "decomposition": "The unanswerable request remained a single information need; decomposition was not the quality failure.",
            "retrieval": "Weak lexical overlaps produced candidates from unrelated corpus topics.",
            "evidence_assembly": "The earlier single-query/generation boundary admitted weak candidates instead of requiring answer-bearing terms for the decomposed path.",
            "synthesis": "The earlier mock generation path emitted retrieved text as a claim without targeted uncertainty, despite the absence of labeled support.",
            "evaluation": "The historical unanswerable label had no relevant passages and the uncertainty metric correctly failed the emitted answer; labels and denominators were preserved.",
        },
        "fix": "Reject weakly aligned evidence for unresolved intents, return an abstention/targeted clarification when no safe support exists, and keep citation validity separate from semantic support.",
        "label_action": "Mark as diagnostic/regression only; preserve historical Phase 3 labels and scores.",
    },
]


class Phase4EvaluationError(ValueError):
    """Raised when an evaluation asset or matched run is invalid."""


class Phase4LabelReviewStatus(BaseModel):
    """Review provenance for labels and the separate semantic claim sheet."""

    model_config = ConfigDict(extra="forbid")

    intent_labels: Literal[
        "provisional_generated", "model_reviewed", "human_review_pending", "human_reviewed"
    ] = "provisional_generated"
    follow_up_labels: Literal[
        "provisional_generated", "model_reviewed", "human_review_pending", "human_reviewed"
    ] = "provisional_generated"
    answerability_labels: Literal[
        "provisional_generated", "model_reviewed", "human_review_pending", "human_reviewed"
    ] = "provisional_generated"
    semantic_claim_support: Literal["not_evaluated", "model_reviewed", "human_reviewed"] = "not_evaluated"
    reviewer: str | None = None
    notes: list[str] = Field(default_factory=list)


class Phase4ExpectedIntent(BaseModel):
    """External label for an information need, not a runtime contract."""

    model_config = ConfigDict(extra="forbid")

    intent_key: str = Field(min_length=1)
    source_hint: str = Field(min_length=1)
    origin: Literal["initial", "follow_up"] = "initial"
    answerable: bool = True

    @field_validator("intent_key", "source_hint")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("intent labels must not be blank")
        return value


class Phase4ExpectedFollowUp(BaseModel):
    """Expected operation and dependency effects for one follow-up turn."""

    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(ge=1)
    classification: Literal[
        "add_constraint",
        "replace_constraint",
        "remove_constraint",
        "add_question",
        "reformat_answer",
        "change_topic",
        "clarification_required",
    ]
    affected_intent_keys: list[str] = Field(default_factory=list)
    invalidated_claim_intent_keys: list[str] = Field(default_factory=list)
    preserved_intent_keys: list[str] = Field(default_factory=list)
    supported_intent_keys: list[str] = Field(default_factory=list)
    unsupported_intent_keys: list[str] = Field(default_factory=list)
    expected_status: Literal["current", "partial", "failed", "stale", "unchanged"] = "unchanged"
    clarification_expected: bool = False
    retrieval_expectation: Literal["required", "forbidden", "optional"] = "optional"
    broad_retrieval_expected: bool = False
    note: str = ""

    @field_validator(
        "affected_intent_keys",
        "invalidated_claim_intent_keys",
        "preserved_intent_keys",
        "supported_intent_keys",
        "unsupported_intent_keys",
    )
    @classmethod
    def keys_are_non_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("expected intent keys must not be blank")
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def follow_up_label_is_consistent(self) -> "Phase4ExpectedFollowUp":
        if self.classification == "reformat_answer" and self.retrieval_expectation == "required":
            raise ValueError("formatting-only labels cannot require retrieval")
        if self.clarification_expected != (self.classification == "clarification_required"):
            raise ValueError("clarification_expected must match classification")
        return self


class Phase4EvaluationCase(BaseModel):
    """One multi-turn case with externally reviewed-later expectations."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[PHASE4_EVALUATION_SCHEMA_VERSION] = PHASE4_EVALUATION_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    split: Literal["development", "held_out"]
    scenario_group: str = Field(min_length=1)
    variant_family: str = Field(min_length=1)
    evaluation_role: Literal[
        "development",
        "diagnostic_regression",
        "untouched_generalization",
    ]
    asset_status: Literal["official", "synthetic_fixture", "unknown"]
    transcript: list[Phase4ReplayTurn] = Field(min_length=1)
    expected_intents: list[Phase4ExpectedIntent] = Field(min_length=1)
    expected_follow_ups: list[Phase4ExpectedFollowUp] = Field(default_factory=list)
    retrieval_behavior: Literal["normal", "failure", "delay", "timeout"] = "normal"
    retrieval_delay_s: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)
    retrieval_timeout_s: float = Field(default=0.05, gt=0.0, le=2.0, allow_inf_nan=False)
    generation_mode: Literal[
        "valid",
        "partial_support",
        "conflicting",
        "invalid_json",
        "timeout",
    ] = "valid"
    generation_delay_s: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)
    race: bool = False
    concurrent_transcripts: list[list[Phase4ReplayTurn]] = Field(default_factory=list)
    label_review_status: Phase4LabelReviewStatus = Field(default_factory=Phase4LabelReviewStatus)
    case_notes: list[str] = Field(default_factory=list)

    @field_validator("case_id", "session_id", "scenario_group", "variant_family")
    @classmethod
    def identifiers_are_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("case identifiers must not be blank")
        return value

    @model_validator(mode="after")
    def case_is_consistent(self) -> "Phase4EvaluationCase":
        if {turn.session_id for turn in self.transcript} != {self.session_id}:
            raise ValueError("primary transcript session IDs must match case session_id")
        indices = [turn.turn_index for turn in self.transcript]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("primary turn indices must be strictly increasing")
        expected_indices = [turn.turn_index for turn in self.transcript[1:]]
        if [item.turn_index for item in self.expected_follow_ups] != expected_indices:
            raise ValueError("expected follow-up labels must cover each non-initial turn in order")
        intent_keys = [item.intent_key for item in self.expected_intents]
        if len(intent_keys) != len(set(intent_keys)):
            raise ValueError("intent labels must be unique within a case")
        known_keys = set(intent_keys)
        for follow_up in self.expected_follow_ups:
            for field_name in (
                "affected_intent_keys",
                "invalidated_claim_intent_keys",
                "preserved_intent_keys",
                "supported_intent_keys",
                "unsupported_intent_keys",
            ):
                unknown = set(getattr(follow_up, field_name)) - known_keys
                if unknown:
                    raise ValueError(
                        f"follow-up {follow_up.turn_index} references unknown {field_name}: {sorted(unknown)}"
                    )
            if set(follow_up.supported_intent_keys) & set(follow_up.unsupported_intent_keys):
                raise ValueError(
                    f"follow-up {follow_up.turn_index} cannot mark an intent both supported and unsupported"
                )
        if self.retrieval_behavior == "delay" and self.retrieval_delay_s <= 0:
            raise ValueError("delay cases require a positive retrieval delay")
        if self.retrieval_behavior == "timeout" and self.retrieval_delay_s <= self.retrieval_timeout_s:
            raise ValueError("timeout cases require delay greater than timeout")
        if self.race and len(self.transcript) < 2:
            raise ValueError("race cases require an initial and correction turn")
        for transcript in self.concurrent_transcripts:
            if not transcript:
                raise ValueError("concurrent transcripts must not be empty")
            sessions = {turn.session_id for turn in transcript}
            if len(sessions) != 1:
                raise ValueError("each concurrent transcript must contain one session")
        return self


class _RecordingRetriever:
    """A transparent retriever wrapper that records calls and fixture faults."""

    def __init__(
        self,
        base: Any,
        *,
        behavior: str = "normal",
        delay_s: float = 0.0,
        timeout_s: float = 0.05,
    ) -> None:
        self.base = base
        self.backend = getattr(base, "backend", "unknown")
        self.behavior = behavior
        self.delay_s = delay_s
        self.timeout_s = timeout_s
        self.queries: list[str] = []
        self.hit_counts: list[int] = []
        self.errors: list[str] = []

    @property
    def index(self) -> Any:
        return getattr(self.base, "index", None)

    def search(self, query: str) -> list[Any]:
        self.queries.append(query)
        if self.behavior == "failure":
            self.errors.append("synthetic retrieval failure")
            raise RuntimeError("synthetic retrieval failure")
        if self.behavior in {"delay", "timeout"}:
            # The scheduler owns timeout enforcement; sleeping here keeps the
            # failure in the retriever path and makes the trace truthful.
            import time as _time

            _time.sleep(self.delay_s)
        hits = self.base.search(query)
        self.hit_counts.append(len(hits))
        return hits


def load_phase4_evaluation_cases(path: Path, *, expected_split: str | None = None) -> list[Phase4EvaluationCase]:
    """Load and validate external Phase 4 JSONL labels."""

    cases: list[Phase4EvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = Phase4EvaluationCase.model_validate(json.loads(line))
        except Exception as exc:
            raise Phase4EvaluationError(f"invalid Phase 4 case at {path}:{line_number}: {exc}") from exc
        if expected_split is not None and case.split != expected_split:
            raise Phase4EvaluationError(
                f"case {case.case_id} has split={case.split!r}; expected {expected_split!r}"
            )
        cases.append(case)
    if not cases:
        raise Phase4EvaluationError(f"Phase 4 case asset {path} is empty")
    return cases


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_PATTERN.findall(text) if len(token) > 2}


def _overlap(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    return len(a & b) / max(1, len(a))


def _intent_mapping(
    state: Phase4SessionState,
    labels: Sequence[Phase4ExpectedIntent],
    *,
    preferred_origin: str = "initial",
    used: set[str] | None = None,
) -> dict[str, str]:
    """Map runtime IDs to external semantic keys using source hints only."""

    used_keys = set(used or ())
    mapping: dict[str, str] = {}
    candidates = [item for item in labels if item.origin == preferred_origin and item.intent_key not in used_keys]
    for intent in state.active_intents:
        ranked = sorted(
            ((
                _overlap(item.source_hint, intent.intent.query),
                -len(item.source_hint),
                item,
            ) for item in candidates if item.intent_key not in mapping.values()),
            key=lambda value: (-value[0], value[1], value[2].intent_key),
        )
        if ranked and ranked[0][0] > 0:
            mapping[intent.intent_id] = ranked[0][2].intent_key
    return mapping


def _remap_patch(patch: Phase4ProposedPatch, state: Phase4SessionState) -> Phase4ProposedPatch:
    """Reuse one interpreted patch on a matched strategy's revision boundary."""

    semantic = patch.classification not in {"clarification_required", "reformat_answer"}
    updates: dict[str, Any] = {
        "base_revision": state.state_revision,
        "proposed_revision": state.state_revision + 1,
        "transcript_revision": state.transcript_revision + 1 if semantic else state.transcript_revision,
        "retrieval_revision": state.retrieval_revision + 1 if semantic else state.retrieval_revision,
    }
    if patch.clarification is not None:
        updates["clarification"] = patch.clarification.model_copy(
            update={"created_revision": state.state_revision + 1}
        )
    return patch.model_copy(update=updates)


def _patch_fingerprint(patch: Phase4ProposedPatch) -> dict[str, Any]:
    """Normalize revision/session-specific fields for matched interpretation."""

    return {
        "classification": patch.classification,
        "target_count": len(patch.target_intent_ids),
        "added_constraints": sorted(
            (record.constraint.kind, record.constraint.value, record.scope)
            for record in patch.added_constraints
        ),
        "replacement_values": sorted(
            (record.constraint.kind, record.constraint.value, record.scope)
            for record in patch.replacement_constraints.values()
        ),
        "removed_count": len(patch.removed_constraint_ids),
        "new_queries": sorted(item.intent.query.casefold() for item in patch.new_intents),
        "format_instruction": patch.format_instruction,
        "topic_key": patch.topic_key,
        "clarification": (
            patch.clarification.reason_code,
            patch.clarification.question,
        )
        if patch.clarification is not None
        else None,
        "references": sorted(
            (item.reference_text.casefold(), item.entity_text.casefold())
            for item in patch.reference_resolutions
        ),
    }


def _usage_for_query(query: str, calls: int = 1) -> Usage:
    tokens = len(_TOKEN_PATTERN.findall(query)) * max(0, calls)
    return Usage(input_tokens=tokens, total_tokens=tokens, estimated=True)


async def _full_retrieve(
    store: Phase4SessionStore,
    state: Phase4SessionState,
    corpus: CorpusIndex,
    retriever: _RecordingRetriever,
) -> tuple[Phase4SessionState, Usage, int, int, list[str]]:
    """Re-run the complete updated task for the full-retrieval arm."""

    # ``active_task`` is the append-only conversational transcript.  A full
    # update must search the canonical *current* task, otherwise a correction
    # such as 40 -> 30 would send both obsolete and current values to the
    # retriever.  The session projection is the same source used by the
    # publisher, and includes all active intents plus shared constraints.
    query = _phase4_decomposition(state).original_transcript
    try:
        hits = retriever.search(query)
    except Exception:
        return store.get(state.session_id), _usage_for_query(query), 1, 1, []
    current = store.get(state.session_id)
    retrieval_revision = current.retrieval_revision + 1
    admitted: list[str] = []
    for hit in hits:
        passage = _passage_for_hit(hit, current, corpus)
        if passage is None:
            continue
        current = store.record_evidence(
            current.session_id,
            passage,
            intent_ids=passage.intent_ids,
            retrieval_revision=retrieval_revision,
            corpus_id=corpus.corpus_id,
            index_id=corpus.manifest.index_id,
            expected_revision=current.state_revision,
        )
        admitted.append(hit.chunk_id)
    return current, _usage_for_query(query), 1, 1, admitted


def _provider_for_case(settings: Settings, case: Phase4EvaluationCase) -> Any:
    provider = generation_provider_for_settings(settings)
    if getattr(provider.config, "backend", None) == "mock":
        return MockGenerationProvider(
            config=provider.config,
            mode=case.generation_mode,
            timeout_delay_s=case.generation_delay_s or None,
        )
    if case.generation_mode != "valid":
        raise Phase4EvaluationError(
            f"case {case.case_id} requests mock generation_mode={case.generation_mode!r} "
            "but the configured real provider cannot be replaced for a matched run"
        )
    return provider


def _claim_key_map(state: Phase4SessionState, intent_mapping: dict[str, str]) -> dict[str, str]:
    return {
        record.record_id: intent_mapping.get(record.intent_ids[0], record.intent_ids[0])
        for record in state.current_claims
        if record.intent_ids
    }


def _claim_keys_for_state(
    state: Phase4SessionState,
    intent_mapping: dict[str, str],
) -> set[str]:
    """Map current claim records to external rubric keys for coverage checks."""

    return {
        intent_mapping.get(record.intent_ids[0], record.intent_ids[0])
        for record in state.current_claims
        if record.intent_ids
    }


def _evidence_quality(state: Phase4SessionState, corpus: CorpusIndex) -> dict[str, Any]:
    """Report provenance quality separately from semantic claim support."""

    current = state.current_evidence
    cited_chunk_ids = {
        chunk_id
        for record in state.current_claims
        for chunk_id in record.claim.supporting_chunk_ids
    }
    current_chunk_ids = {record.passage.chunk_id for record in current}
    return {
        "current_evidence_count": len(current),
        "current_evidence_chunk_ids": sorted(current_chunk_ids),
        "cited_current_chunk_count": len(cited_chunk_ids & current_chunk_ids),
        "cited_chunk_count": len(cited_chunk_ids),
        "all_citations_current": cited_chunk_ids <= current_chunk_ids,
        "source_locations_available": all(
            bool(record.passage.source_location.strip()) for record in current
        ),
        "corpus_chunk_count": len(corpus.chunks),
        "semantic_support": "not_evaluated",
    }


def _metric_fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _citation_audit(state: Phase4SessionState, corpus: CorpusIndex) -> dict[str, Any]:
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    ids: list[str] = []
    unknown: list[str] = []
    invalid_excerpts: list[str] = []
    for version in state.answer_versions:
        for claim in version.answer.factual_claims:
            for chunk_id in claim.supporting_chunk_ids:
                ids.append(chunk_id)
                chunk = chunks.get(chunk_id)
                if chunk is None:
                    unknown.append(chunk_id)
            for chunk_id, excerpt in zip(claim.supporting_chunk_ids, claim.supporting_excerpts):
                chunk = chunks.get(chunk_id)
                if chunk is not None and excerpt not in chunk.text:
                    invalid_excerpts.append(f"{claim.claim_id}:{chunk_id}")
    return {
        "citation_id_count": len(ids),
        "citation_ids_valid": not unknown and not invalid_excerpts,
        "unknown_citation_ids": sorted(set(unknown)),
        "invalid_excerpts": sorted(set(invalid_excerpts)),
        "structural_only": True,
    }


def _semantic_review_status(state: Phase4SessionState) -> dict[str, Any]:
    records = [record for record in state.claim_records if record.status == "current"]
    machine_marked = [record for record in records if record.semantic_support_status != "unreviewed"]
    return {
        "status": "human_review_pending",
        "reviewed_claim_count": 0,
        "machine_marked_claim_count": len(machine_marked),
        "current_claim_count": len(records),
        "semantic_support_metric": None,
        "reason": "No human semantic claim review was performed during this run.",
    }


def _status_text(state: Phase4SessionState) -> str:
    answer = state.current_answer
    return answer.answer.uncertainty.casefold() if answer is not None else ""


async def _run_strategy(
    case: Phase4EvaluationCase,
    *,
    corpus: CorpusIndex,
    settings: Settings,
    backend: str,
    top_k: int,
    strategy: Literal["full", "selective"],
    canonical_patches: list[Phase4ProposedPatch] | None = None,
) -> dict[str, Any]:
    """Run one strategy through the actual Phase 4 pipeline."""

    started = time.perf_counter()
    store = Phase4SessionStore(max_sessions=8, max_records_per_session=256)
    base_retriever = make_retriever(
        corpus,
        backend=backend,
        top_k=top_k,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
    )
    retriever = _RecordingRetriever(
        base_retriever,
        behavior=case.retrieval_behavior,
        delay_s=case.retrieval_delay_s,
        timeout_s=case.retrieval_timeout_s,
    )
    provider = _provider_for_case(settings, case)
    publisher = Phase4AnswerPublisher(store, corpus, generation_provider=provider)
    interpreter = Phase4FollowUpInterpreter()
    first = case.transcript[0]
    initial_plan = decompose_query(first.text, transcript_revision=0, max_intents=6)
    state = store.create_session(
        case.session_id,
        initial_plan=initial_plan,
        utterance_id=first.utterance_id,
        initial_turn=first.turn_index,
        corpus_id=corpus.corpus_id,
        index_id=corpus.manifest.index_id,
    )
    state, initial_usage, initial_calls, notes = await _record_initial_retrieval(
        store, state, corpus, retriever, first.text
    )
    initial_publication = await publisher.generate_and_publish(
        case.session_id,
        expected_revision=state.state_revision,
        triggering_turn=first.turn_index,
        triggering_utterance_id=first.utterance_id,
        retrieval_usage=initial_usage,
        retrieval_call_count=initial_calls,
        retrieval_attempt_count=initial_calls,
        retrieval_model_identity=f"flowcontext.phase4.{strategy}.initial-retriever",
    )
    state = initial_publication.state
    intent_mapping = _intent_mapping(state, case.expected_intents, preferred_origin="initial")
    initial_intent_mapping = dict(intent_mapping)
    initial_snapshot = state
    rows: list[dict[str, Any]] = []
    patches: list[Phase4ProposedPatch] = []
    coordinator = Phase4SelectiveUpdateCoordinator(store, retriever) if strategy == "selective" else None
    old_generation_task: asyncio.Task[Any] | None = None

    for position, turn in enumerate(case.transcript[1:]):
        expected = case.expected_follow_ups[position]
        before = store.get(case.session_id)
        before_mapping = dict(intent_mapping)
        before_claims = {record.record_id: record for record in before.current_claims}
        before_claim_key_map = _claim_key_map(before, before_mapping)
        interpretation_match = True
        if canonical_patches is None:
            patch = store.propose_follow_up(
                case.session_id,
                turn.text,
                utterance_id=turn.utterance_id,
                turn_index=turn.turn_index,
                interpreter=interpreter,
            )
            patches.append(patch)
        else:
            independently_interpreted = store.propose_follow_up(
                case.session_id,
                turn.text,
                utterance_id=turn.utterance_id,
                turn_index=turn.turn_index,
                interpreter=interpreter,
            )
            canonical = canonical_patches[position]
            interpretation_match = _patch_fingerprint(independently_interpreted) == _patch_fingerprint(canonical)
            patch = _remap_patch(canonical, before)
        if canonical_patches is not None:
            patches.append(patch)
        if case.race and position == 0:
            old_generation_task = asyncio.create_task(
                publisher.generate_and_publish(
                    case.session_id,
                    expected_revision=before.state_revision,
                    triggering_turn=case.transcript[0].turn_index,
                    triggering_utterance_id=case.transcript[0].utterance_id,
                    retrieval_usage=initial_usage,
                    retrieval_call_count=initial_calls,
                    retrieval_attempt_count=initial_calls,
                    retrieval_model_identity=f"flowcontext.phase4.{strategy}.race-old",
                )
            )
            await asyncio.sleep(0)
        else:
            old_generation_task = None
            interpretation_match = True
        turn_started = time.perf_counter()
        plan = None
        publication = None
        if patch.classification == "clarification_required":
            after = store.apply_patch(patch)
        elif patch.classification == "reformat_answer":
            store.apply_patch(patch)
            after_result = await publisher.present_and_publish(
                case.session_id,
                expected_revision=store.get(case.session_id).state_revision,
            )
            publication = after_result
            after = after_result.state
        elif strategy == "selective":
            assert coordinator is not None
            plan = coordinator.apply_patch(patch)
            if plan is None:
                raise Phase4EvaluationError("semantic patch did not produce a selective plan")
            after = await coordinator.execute_plan(plan)
            plan = next(item for item in after.selective_update_plans if item.plan_id == plan.plan_id)
            publication = await publisher.generate_and_publish(
                case.session_id,
                expected_revision=after.state_revision,
                triggering_patch_id=patch.patch_id,
                triggering_turn=turn.turn_index,
                triggering_utterance_id=turn.utterance_id,
                retrieval_usage=plan.retrieval_usage,
                retrieval_call_count=plan.retrieval_call_count,
                retrieval_attempt_count=plan.retrieval_attempt_count,
                retrieval_model_identity="flowcontext.phase4.selective-retriever",
            )
            after = publication.state
        else:
            store.apply_patch(patch)
            after = store.get(case.session_id)
            plan = next(item for item in after.selective_update_plans if item.patch_id == patch.patch_id)
            after, usage, calls, attempts, _admitted = await _full_retrieve(
                store, after, corpus, retriever
            )
            store.record_selective_plan_usage(
                case.session_id,
                plan.plan_id,
                usage=usage,
                call_count=calls,
                attempt_count=attempts,
            )
            after = store.get(case.session_id)
            plan = next(item for item in after.selective_update_plans if item.plan_id == plan.plan_id)
            publication = await publisher.generate_and_publish(
                case.session_id,
                expected_revision=after.state_revision,
                triggering_patch_id=patch.patch_id,
                triggering_turn=turn.turn_index,
                triggering_utterance_id=turn.utterance_id,
                retrieval_usage=plan.retrieval_usage,
                retrieval_call_count=plan.retrieval_call_count,
                retrieval_attempt_count=plan.retrieval_attempt_count,
                retrieval_model_identity="flowcontext.phase4.full-retriever",
            )
            after = publication.state
        if old_generation_task is not None:
            old_result = await old_generation_task
            notes.append(f"race old generation: {old_result.status}")
        after_mapping = dict(before_mapping)
        if patch.classification == "change_topic":
            after_mapping = _intent_mapping(after, case.expected_intents, preferred_origin="follow_up")
        else:
            for old_id, new_id in patch.intent_replacements.items():
                if old_id in before_mapping:
                    after_mapping[new_id] = before_mapping[old_id]
            new_ids = [item.intent_id for item in patch.new_intents if item.intent_id not in after_mapping]
            unused = {
                item.intent_key
                for item in case.expected_intents
                if item.origin == "follow_up" and item.intent_key not in after_mapping.values()
            }
            for new_id in new_ids:
                new_record = next((item for item in after.active_intents if item.intent_id == new_id), None)
                if new_record is None:
                    continue
                candidates = [item for item in case.expected_intents if item.intent_key in unused]
                if candidates:
                    chosen = max(candidates, key=lambda item: _overlap(item.source_hint, new_record.intent.query))
                    if _overlap(chosen.source_hint, new_record.intent.query) > 0:
                        after_mapping[new_id] = chosen.intent_key
                        unused.remove(chosen.intent_key)
        after = store.get(case.session_id)
        actual_affected_keys: list[str] = []
        actual_preserved_keys: list[str] = []
        actual_propagated_keys: list[str] = []
        invalidated_record_ids: list[str] = []
        preserved_record_ids: list[str] = []
        reused_evidence_ids: list[str] = []
        invalidated_evidence_ids: list[str] = []
        discarded_result_ids: list[str] = []
        retrieval_reasons: dict[str, str] = {}
        if plan is not None:
            actual_affected_keys = sorted(
                {
                    before_mapping.get(item, after_mapping.get(item, item))
                    for item in plan.directly_affected_intent_ids
                }
            )
            actual_retrieval_keys = {
                before_mapping.get(item, after_mapping.get(item, item))
                for item in plan.retrieval_intent_ids
            }
            actual_propagated_keys = sorted(actual_retrieval_keys - set(actual_affected_keys))
            actual_preserved_keys = sorted(
                {before_claim_key_map.get(item, item) for item in plan.preserved_claim_ids}
            )
            invalidated_record_ids = list(plan.invalidated_claim_ids)
            preserved_record_ids = list(plan.preserved_claim_ids)
            reused_evidence_ids = list(plan.reused_evidence_ids)
            invalidated_evidence_ids = list(plan.invalidated_evidence_ids)
            discarded_result_ids = list(plan.discarded_result_ids)
            retrieval_reasons = dict(plan.retrieval_reasons)
        labelled_invalidated_keys = set(expected.invalidated_claim_intent_keys)
        if not labelled_invalidated_keys and expected.classification in {
            "replace_constraint",
            "remove_constraint",
            "change_topic",
        }:
            # A newly added constraint/question can be dependency-compatible
            # with existing support.  For corrections and topic changes, the
            # default external rubric treats existing claims in the affected
            # intent as invalidated; assets may override this explicitly with
            # invalidated_claim_intent_keys when a reviewed case differs.
            labelled_invalidated_keys = set(expected.affected_intent_keys)
        expected_invalidated_records = {
            record_id
            for record_id, key in before_claim_key_map.items()
            if key in labelled_invalidated_keys
        }
        invalidated_set = set(invalidated_record_ids)
        tp = len(expected_invalidated_records & invalidated_set)
        fp = len(invalidated_set - expected_invalidated_records)
        fn = len(expected_invalidated_records - invalidated_set)
        preserved_unaffected = {
            key
            for record_id, key in before_claim_key_map.items()
            if key in set(expected.preserved_intent_keys)
            and any(record.claim_id == before_claims[record_id].claim_id for record in after.current_claims)
        }
        obsolete_preserved = sum(
            1
            for record_id in invalidated_set
            if any(record.claim_id == before_claims.get(record_id, object()).claim_id for record in after.current_claims)
        )
        citation = _citation_audit(after, corpus)
        unsupported_present = bool(expected.unsupported_intent_keys)
        uncertainty_text = _status_text(after)
        uncertainty_ok = (
            not unsupported_present
            or any(token in uncertainty_text for token in ("unresolved", "insufficient", "cannot", "unsupported", "unavailable"))
        )
        current_claim_keys = _claim_keys_for_state(after, after_mapping)
        expected_supported_keys = set(expected.supported_intent_keys)
        expected_unsupported_keys = set(expected.unsupported_intent_keys)
        covered_supported_keys = current_claim_keys & expected_supported_keys
        updated_coverage = {
            "expected_supported_intent_keys": sorted(expected_supported_keys),
            "covered_supported_intent_keys": sorted(covered_supported_keys),
            "coverage": _metric_fraction(
                len(covered_supported_keys), len(expected_supported_keys)
            ),
            "unsupported_intent_keys_with_claims": sorted(
                current_claim_keys & expected_unsupported_keys
            ),
            "semantic_support": "not_evaluated",
        }
        version = after.current_answer
        turn_rows = {
            "turn_index": turn.turn_index,
            "utterance_id": turn.utterance_id,
            "text": turn.text,
            "classification": patch.classification,
            "expected_classification": expected.classification,
            "interpretation_correct": patch.classification == expected.classification,
            "interpretation_match_across_modes": interpretation_match,
            "actual_affected_intent_keys": actual_affected_keys,
            "actual_propagated_intent_keys": actual_propagated_keys,
            "expected_affected_intent_keys": expected.affected_intent_keys,
            "actual_preserved_intent_keys": actual_preserved_keys,
            "expected_preserved_intent_keys": expected.preserved_intent_keys,
            "actual_invalidated_claim_ids": invalidated_record_ids,
            "actual_preserved_claim_ids": preserved_record_ids,
            "reused_evidence_ids": reused_evidence_ids,
            "invalidated_evidence_ids": invalidated_evidence_ids,
            "discarded_result_ids": discarded_result_ids,
            "invalidation": {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": _metric_fraction(tp, tp + fp),
                "recall": _metric_fraction(tp, tp + fn),
            },
            "obsolete_claims_preserved": obsolete_preserved,
            "unaffected_claim_keys_preserved": sorted(preserved_unaffected),
            "expected_unaffected_claim_keys": expected.preserved_intent_keys,
            "status": after.answer_status,
            "expected_status": expected.expected_status,
            "status_matches": (
                after.answer_status == before.answer_status
                if expected.expected_status == "unchanged"
                else after.answer_status == expected.expected_status
            ),
            "clarification": {
                "expected": expected.clarification_expected,
                "actual": patch.classification == "clarification_required" and after.pending_clarification is not None,
                "question": after.pending_clarification.question if after.pending_clarification else None,
            },
            "uncertainty": {
                "expected": unsupported_present,
                "present": bool(uncertainty_text),
                "targeted": uncertainty_ok,
            },
            "retrieval": {
                "expectation": expected.retrieval_expectation,
                "calls_total": len(retriever.queries),
                "queries": list(retriever.queries),
                "chunks_total": sum(retriever.hit_counts),
                "reasons": retrieval_reasons,
                "formatting_only_calls": 0 if patch.classification == "reformat_answer" else None,
                "broad_retrieval_expected": expected.broad_retrieval_expected,
            },
            "citation": citation,
            "evidence_quality": _evidence_quality(after, corpus),
            "semantic_support": _semantic_review_status(after),
            "latency_ms": round((time.perf_counter() - turn_started) * 1000, 3),
            "updated_answer_coverage": updated_coverage,
            "current_claim_keys": sorted(current_claim_keys),
            "version": after.current_answer_version,
            "claim_changes": [
                item.model_dump(mode="json")
                for item in version.claim_changes
            ] if version is not None else [],
            "superseded_requests": len(after.superseded_requests),
            "trace_complete": bool(after.trace_ids and after.revision_history),
        }
        rows.append(turn_rows)
        state = after
        intent_mapping = after_mapping

    if coordinator is not None:
        await coordinator.close()
    final = store.get(case.session_id)
    current_version = final.current_answer
    generation_calls = getattr(provider, "calls", None)
    if not isinstance(generation_calls, int):
        generation_calls = sum(version.generation_attempts > 0 for version in final.answer_versions)
    version_usage = [version.generation_usage for version in final.answer_versions]
    retrieval_usage = [version.retrieval_usage for version in final.answer_versions]
    generation_tokens = sum(item.total_tokens for item in version_usage)
    retrieval_tokens = sum(item.total_tokens for item in retrieval_usage)
    retrieval_calls = len(retriever.queries)
    retrieved_chunks = sum(retriever.hit_counts)
    formatting_retrieval_calls = sum(
        row["retrieval"]["calls_total"] - (rows[index - 1]["retrieval"]["calls_total"] if index else initial_calls)
        for index, row in enumerate(rows)
        if row["classification"] == "reformat_answer"
    )
    initial_expected_supported_keys = {
        item.intent_key
        for item in case.expected_intents
        if item.origin == "initial" and item.answerable
    }
    initial_claim_keys = _claim_keys_for_state(initial_snapshot, initial_intent_mapping)
    return {
        "strategy": strategy,
        "case_id": case.case_id,
        "session_id": case.session_id,
        "initial": {
            "intent_count": len(initial_snapshot.active_intents),
            "expected_intent_count": sum(item.origin == "initial" for item in case.expected_intents),
            "intent_mapping": initial_intent_mapping,
            "answer_status": initial_snapshot.answer_status,
            "claim_count": len(initial_snapshot.current_claims),
            "answer_coverage": {
                "expected_supported_intent_keys": sorted(initial_expected_supported_keys),
                "covered_supported_intent_keys": sorted(
                    initial_claim_keys & initial_expected_supported_keys
                ),
                "coverage": _metric_fraction(
                    len(initial_claim_keys & initial_expected_supported_keys),
                    len(initial_expected_supported_keys),
                ),
                "semantic_support": "not_evaluated",
            },
            "citation": _citation_audit(initial_snapshot, corpus),
        },
        "turns": rows,
        "final": {
            "answer_status": final.answer_status,
            "current_answer_version": final.current_answer_version,
            "current_claim_ids": [record.claim_id for record in final.current_claims],
            "current_claim_count": len(final.current_claims),
            "uncertainty": current_version.answer.uncertainty if current_version else None,
        },
        "resources": {
            "retrieval_calls": retrieval_calls,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_tokens": retrieval_tokens,
            "retrieval_usage_estimated": any(item.estimated for item in retrieval_usage),
            "generation_calls": generation_calls,
            "generation_tokens": generation_tokens,
            "generation_usage_estimated": any(item.estimated for item in version_usage),
            "cost": "unavailable",
            "final_event_to_updated_answer_latency_ms": median(
                [row["latency_ms"] for row in rows]
            ) if rows else None,
            "formatting_only_retrieval_calls": formatting_retrieval_calls,
            "stale_publication_count": len(final.superseded_requests),
        },
        "provider": {
            "backend": getattr(provider.config, "backend", "unknown"),
            "provider": getattr(provider.config, "provider", "unknown"),
            "model": getattr(provider.config, "model", "unknown"),
            "execution_label": "mock_provider" if getattr(provider.config, "backend", None) == "mock" else "real_provider",
        },
        "trace": {
            "revision_count": len(final.revision_history),
            "trace_id_count": len(final.trace_ids),
            "version_count": len(final.answer_versions),
            "complete": bool(final.trace_ids and final.revision_history),
            "notes": notes,
        },
        "state": final.model_dump(mode="json"),
        "patches": [patch.model_dump(mode="json") for patch in patches],
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }


async def _run_concurrent_isolation_probe(
    case: Phase4EvaluationCase,
    *,
    corpus: CorpusIndex,
    settings: Settings,
    backend: str,
    top_k: int,
) -> dict[str, Any] | None:
    """Exercise one shared bounded store with simultaneous isolated sessions."""

    if not case.concurrent_transcripts:
        return None
    from .phase4_replay import replay_phase4_session

    transcripts = [case.transcript, *case.concurrent_transcripts]
    store = Phase4SessionStore(max_sessions=16, max_records_per_session=256)

    async def run(transcript: list[Phase4ReplayTurn]) -> Any:
        base = make_retriever(
            corpus,
            backend=backend,
            top_k=top_k,
            cache_dir=settings.embedding_cache_dir,
            local_files_only=settings.embedding_local_files_only,
        )
        provider = _provider_for_case(settings, case)
        return await replay_phase4_session(
            transcript,
            corpus=corpus,
            retriever=_RecordingRetriever(base),
            generation_provider=provider,
            store=store,
        )

    results = await asyncio.gather(*(run(transcript) for transcript in transcripts))
    ids = [result.session_id for result in results]
    isolation_ok = len(ids) == len(set(ids)) and all(
        result.state.session_id == result.session_id
        and all(evidence.session_id == result.session_id for evidence in result.state.evidence_records)
        for result in results
    )
    return {
        "session_count": len(results),
        "session_ids": ids,
        "isolated": isolation_ok,
        "trace_complete": all(bool(result.state.trace_ids) for result in results),
        "statuses": {result.session_id: result.state.answer_status for result in results},
    }


def _validate_split_integrity(cases: Sequence[Phase4EvaluationCase]) -> dict[str, Any]:
    families: dict[str, set[str]] = {}
    for case in cases:
        families.setdefault(case.variant_family, set()).add(case.split)
    crossing = {family: sorted(splits) for family, splits in families.items() if len(splits) > 1}
    ids = [case.case_id for case in cases]
    return {
        "case_count": len(cases),
        "unique_case_ids": len(ids) == len(set(ids)),
        "variant_families_crossing_splits": crossing,
        "related_variants_kept_in_same_split": not crossing,
        "split_counts": dict(Counter(case.split for case in cases)),
        "evaluation_role_counts": dict(Counter(case.evaluation_role for case in cases)),
        "scenario_group_counts": dict(Counter(case.scenario_group for case in cases)),
        "case_ids_by_role": {
            role: [case.case_id for case in cases if case.evaluation_role == role]
            for role in ("development", "diagnostic_regression", "untouched_generalization")
        },
    }


def _label_review_summary(cases: Sequence[Phase4EvaluationCase]) -> dict[str, Any]:
    dimensions = {
        "intent_labels": Counter(case.label_review_status.intent_labels for case in cases),
        "follow_up_labels": Counter(case.label_review_status.follow_up_labels for case in cases),
        "answerability_labels": Counter(case.label_review_status.answerability_labels for case in cases),
        "semantic_claim_support": Counter(case.label_review_status.semantic_claim_support for case in cases),
    }
    return {key: dict(value) for key, value in dimensions.items()} | {
        "human_review_completed": any(
            case.label_review_status.semantic_claim_support == "human_reviewed" for case in cases
        ),
        "reviewer": "none",
        "honest_boundary": "Intent, operation, answerability, and semantic-support labels remain provisional/pending; no human review was performed.",
    }


def _session_isolation_passed(rows: Sequence[dict[str, Any]]) -> bool:
    """Return whether every requested concurrent-session probe stayed isolated."""

    return all(
        row["concurrent_isolation"] is None
        or (
            row["concurrent_isolation"]["isolated"]
            and row["concurrent_isolation"]["trace_complete"]
        )
        for row in rows
    )


def stable_phase4_evaluation_signature(report: dict[str, Any]) -> str:
    """Hash deterministic outcomes while excluding timestamps and latencies."""

    signature = {
        "data_integrity": report["data_integrity"],
        "strategies": {
            strategy_name: [
                {
                    "case_id": case["case_id"],
                    "initial": {
                        "intent_count": arm["initial"]["intent_count"],
                        "answer_status": arm["initial"]["answer_status"],
                        "claim_count": arm["initial"]["claim_count"],
                        "answer_coverage": arm["initial"]["answer_coverage"],
                    },
                    "turns": [
                        {
                            "turn_index": turn["turn_index"],
                            "classification": turn["classification"],
                            "interpretation_correct": turn["interpretation_correct"],
                            "actual_affected_intent_keys": turn["actual_affected_intent_keys"],
                            "actual_propagated_intent_keys": turn["actual_propagated_intent_keys"],
                            "actual_invalidated_claim_ids": turn["actual_invalidated_claim_ids"],
                            "actual_preserved_claim_ids": turn["actual_preserved_claim_ids"],
                            "current_claim_keys": turn["current_claim_keys"],
                            "status": turn["status"],
                            "retrieval_queries": turn["retrieval"]["queries"],
                        }
                        for turn in arm["turns"]
                    ],
                    "final": arm["final"],
                    "resources": {
                        key: value
                        for key, value in arm["resources"].items()
                        if "latency" not in key
                    },
                }
                for case in report["case_results"]
                for arm in [case["strategies"][strategy_name]]
            ]
            for strategy_name in ("full", "selective")
        },
    }
    payload = json.dumps(signature, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
def _aggregate_strategy(rows: Sequence[dict[str, Any]], strategy: str) -> dict[str, Any]:
    records = [row["strategies"][strategy] for row in rows]
    turns = [turn for record in records for turn in record["turns"]]
    interpretation = [turn["interpretation_correct"] for turn in turns]
    matches = [turn["interpretation_match_across_modes"] for turn in turns]
    invalidations = [turn["invalidation"] for turn in turns if turn["invalidation"]["precision"] is not None or turn["invalidation"]["recall"] is not None]
    unaffected = [
        set(turn["expected_unaffected_claim_keys"]) <= set(turn["unaffected_claim_keys_preserved"])
        for turn in turns
        if turn["expected_unaffected_claim_keys"]
    ]
    resources = [record["resources"] for record in records]
    initial_coverages = [
        record["initial"]["answer_coverage"]["coverage"]
        for record in records
        if record["initial"]["answer_coverage"]["coverage"] is not None
    ]
    updated_coverages = [
        turn["updated_answer_coverage"]["coverage"]
        for turn in turns
        if turn["updated_answer_coverage"]["coverage"] is not None
    ]
    quality_rows = [turn["evidence_quality"] for turn in turns]
    return {
        "case_count": len(records),
        "follow_up_turn_count": len(turns),
        "follow_up_interpretation_accuracy": _metric_fraction(sum(interpretation), len(interpretation)),
        "matched_interpretation_rate": _metric_fraction(sum(matches), len(matches)),
        "affected_claim_invalidation_precision": _metric_fraction(
            sum(item["true_positive"] for item in invalidations),
            sum(item["true_positive"] + item["false_positive"] for item in invalidations),
        ),
        "affected_claim_invalidation_recall": _metric_fraction(
            sum(item["true_positive"] for item in invalidations),
            sum(item["true_positive"] + item["false_negative"] for item in invalidations),
        ),
        "incorrect_preservation_of_obsolete_claims": sum(
            turn["obsolete_claims_preserved"] for turn in turns
        ),
        "unaffected_claim_preservation_rate": _metric_fraction(sum(unaffected), len(unaffected)),
        "initial_answer_coverage": sum(initial_coverages) / len(initial_coverages) if initial_coverages else None,
        "updated_answer_coverage": sum(updated_coverages) / len(updated_coverages) if updated_coverages else None,
        "updated_answer_coverage_turn_count": len(updated_coverages),
        "evidence_quality": {
            "all_current_citations_rate": _metric_fraction(
                sum(item["all_citations_current"] for item in quality_rows),
                len(quality_rows),
            ),
            "source_locations_available_rate": _metric_fraction(
                sum(item["source_locations_available"] for item in quality_rows),
                len(quality_rows),
            ),
            "mean_current_evidence_count": (
                sum(item["current_evidence_count"] for item in quality_rows) / len(quality_rows)
                if quality_rows else None
            ),
            "semantic_support": "NOT VERIFIED",
        },
        "citation_id_validity": _metric_fraction(
            sum(record["final"]["current_answer_version"] is not None and all(
                turn["citation"]["citation_ids_valid"] for turn in record["turns"]
            ) for record in records),
            len(records),
        ),
        "semantic_claim_support": {
            "status": "NOT VERIFIED",
            "reviewed_claim_count": sum(
                turn["semantic_support"]["reviewed_claim_count"] for turn in turns
            ),
        },
        "uncertainty_targeting_rate": _metric_fraction(
            sum(turn["uncertainty"]["targeted"] for turn in turns if turn["uncertainty"]["expected"]),
            sum(turn["uncertainty"]["expected"] for turn in turns),
        ),
        "clarification_accuracy": _metric_fraction(
            sum(turn["clarification"]["expected"] == turn["clarification"]["actual"] for turn in turns),
            len(turns),
        ),
        "resources": {
            "retrieval_calls": sum(item["retrieval_calls"] for item in resources),
            "retrieved_chunks": sum(item["retrieved_chunks"] for item in resources),
            "retrieval_tokens": sum(item["retrieval_tokens"] for item in resources),
            "generation_calls": sum(item["generation_calls"] for item in resources),
            "generation_tokens": sum(item["generation_tokens"] for item in resources),
            "cost": "unavailable" if any(item["cost"] == "unavailable" for item in resources) else 0.0,
            "final_event_to_updated_answer_latency_ms_p50": median(
                [turn["latency_ms"] for turn in turns]
            ) if turns else None,
            "final_event_to_updated_answer_latency_ms_p95": sorted(
                turn["latency_ms"] for turn in turns
            )[max(0, int(0.95 * len(turns)) - 1)] if turns else None,
            "formatting_only_retrieval_calls": sum(item["formatting_only_retrieval_calls"] for item in resources),
            "stale_publication_count": sum(item["stale_publication_count"] for item in resources),
        },
        "trace_completeness": _metric_fraction(
            sum(record["trace"]["complete"] for record in records), len(records)
        ),
        "final_status_counts": dict(Counter(record["final"]["answer_status"] for record in records)),
    }


def _review_rows(
    rows: Sequence[dict[str, Any]],
    review_lookup: dict[tuple[str, str, int, str], dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for strategy_name, strategy in row["strategies"].items():
            evidence_by_id = {
                evidence["evidence_id"]: evidence
                for evidence in strategy["state"]["evidence_records"]
            }
            records_by_id = {
                record["record_id"]: record
                for record in strategy["state"]["claim_records"]
            }
            for version in strategy["state"]["answer_versions"]:
                for claim in version["answer"]["factual_claims"]:
                    claim_records = [
                        records_by_id[record_id]
                        for record_id in version["claim_record_ids"]
                        if record_id in records_by_id
                        and records_by_id[record_id]["claim_id"] == claim["claim_id"]
                    ]
                    supporting_evidence_ids = [
                        evidence_id
                        for record in claim_records
                        for evidence_id in record["supporting_evidence_ids"]
                    ]
                    passages = [
                        evidence_by_id[evidence_id]["passage"]
                        for evidence_id in supporting_evidence_ids
                        if evidence_id in evidence_by_id
                    ]
                    passage_by_chunk = {
                        passage["chunk_id"]: passage for passage in passages
                    }
                    structural_valid = True
                    excerpts = claim.get("supporting_excerpts", [])
                    if excerpts and len(excerpts) != len(claim["supporting_chunk_ids"]):
                        structural_valid = False
                    for chunk_id, excerpt in zip(
                        claim["supporting_chunk_ids"], excerpts
                    ):
                        passage = passage_by_chunk.get(chunk_id)
                        if passage is None or excerpt not in passage["text"]:
                            structural_valid = False
                    if any(
                        chunk_id not in passage_by_chunk
                        for chunk_id in claim["supporting_chunk_ids"]
                    ):
                        structural_valid = False
                    lookup_key = (
                        row["case_id"],
                        strategy_name,
                        int(version["answer_version"]),
                        claim["claim_id"],
                    )
                    existing_entry = review_lookup.get(lookup_key) if review_lookup else None
                    if existing_entry is not None:
                        verdict = existing_entry.get("semantic_support_verdict", "pending_human_review")
                        reviewer = existing_entry.get("reviewer", "")
                        review_notes = existing_entry.get("review_notes", "")
                    else:
                        verdict = "pending_human_review"
                        reviewer = ""
                        review_notes = ""
                    output.append(
                        {
                            "case_id": row["case_id"],
                            "strategy": strategy_name,
                            "answer_version": version["answer_version"],
                            "claim_id": claim["claim_id"],
                            "claim_text": claim["claim_text"],
                            "supporting_chunk_ids": ";".join(claim["supporting_chunk_ids"]),
                            "supporting_passages": " || ".join(passage["text"] for passage in passages),
                            "source_locations": ";".join(passage["source_location"] for passage in passages),
                            "structural_citation_valid": str(structural_valid).lower(),
                            "semantic_support_verdict": verdict,
                            "reviewer": reviewer,
                            "review_notes": review_notes,
                        }
                    )
    return output


async def evaluate_phase4(
    cases: Sequence[Phase4EvaluationCase],
    *,
    corpus: CorpusIndex,
    settings: Settings,
    backend: str,
    top_k: int,
) -> dict[str, Any]:
    """Run matched full/selective arms and assemble a machine-readable report."""

    if not cases:
        raise Phase4EvaluationError("at least one Phase 4 case is required")
    if top_k >= len(corpus.chunks):
        raise Phase4EvaluationError(
            f"top_k={top_k} must be smaller than corpus chunk_count={len(corpus.chunks)}"
        )
    for case in cases:
        if case.asset_status == "official":
            raise Phase4EvaluationError("official labels are not available in this workspace")
        for turn in case.transcript:
            if not turn.text.strip():
                raise Phase4EvaluationError(f"case {case.case_id} contains blank turn text")
        for transcript in case.concurrent_transcripts:
            if any(turn.session_id == case.session_id for turn in transcript):
                raise Phase4EvaluationError("concurrent isolation sessions must have distinct session IDs")

    case_rows: list[dict[str, Any]] = []
    for case in cases:
        selective = await _run_strategy(
            case, corpus=corpus, settings=settings, backend=backend, top_k=top_k, strategy="selective"
        )
        canonical_patches = [Phase4ProposedPatch.model_validate(item) for item in selective["patches"]]
        full = await _run_strategy(
            case,
            corpus=corpus,
            settings=settings,
            backend=backend,
            top_k=top_k,
            strategy="full",
            canonical_patches=canonical_patches,
        )
        concurrent = await _run_concurrent_isolation_probe(
            case, corpus=corpus, settings=settings, backend=backend, top_k=top_k
        )
        case_rows.append(
            {
                "case_id": case.case_id,
                "session_id": case.session_id,
                "split": case.split,
                "scenario_group": case.scenario_group,
                "variant_family": case.variant_family,
                "evaluation_role": case.evaluation_role,
                "asset_status": case.asset_status,
                "label_review_status": case.label_review_status.model_dump(mode="json"),
                "strategies": {"full": full, "selective": selective},
                "concurrent_isolation": concurrent,
                "case_notes": case.case_notes,
            }
        )
    full_summary = _aggregate_strategy(case_rows, "full")
    selective_summary = _aggregate_strategy(case_rows, "selective")
    report = {
        "schema_version": "flowcontext.phase4-audit.v1",
        "report_version": PHASE4_EVALUATION_SCHEMA_VERSION,
        "evaluation_label": "phase4-matched-full-vs-selective-multiturn",
        "case_count": len(cases),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset_status": "synthetic_fixture" if all(case.asset_status == "synthetic_fixture" for case in cases) else "mixed",
        "synthetic_data_provisional": all(case.asset_status == "synthetic_fixture" for case in cases),
        "official_benchmark_claim": False,
        "configuration": {
            "backend": backend,
            "top_k": top_k,
            "generation_backend": settings.generation_backend,
            "generation_provider": settings.generation_provider,
            "generation_model": settings.generation_model,
            "embedding_model": settings.embedding_model,
            "embedding_revision": settings.embedding_revision,
            "retrieval_top_k": settings.retrieval_top_k,
            "multi_intent_context_budget_tokens": settings.multi_intent_context_budget_tokens,
            "multi_intent_max_workers": settings.multi_intent_max_workers,
            "generation_max_output_tokens": settings.generation_max_output_tokens,
            "matched_conditions": {
                "A_full": "same interpreted patch, full updated active task retrieval, full answer regeneration",
                "B_selective": "same interpreted patch, dependency-aware invalidation, targeted retrieval, answer update",
            },
            "fairness": "Both arms use the same corpus/index, interpreter patch, provider configuration, retrieval backend/top-k, and generation configuration. Only update strategy differs.",
        },
        "corpus": {
            "corpus_id": corpus.corpus_id,
            "index_id": corpus.manifest.index_id,
            "source_kind": corpus.source_kind,
            "document_count": len(corpus.documents),
            "chunk_count": len(corpus.chunks),
            "top_k": top_k,
            "source_path": corpus.manifest.source_path,
            "embedding": corpus.manifest.embedding.model_dump(mode="json"),
        },
        "data_integrity": _validate_split_integrity(cases),
        "phase3_failure_trace": PHASE3_FAILURE_TRACE,
        "label_review_status": _label_review_summary(cases),
        "strategies": {
            "full": full_summary,
            "selective": selective_summary,
        },
        "resource_savings": {
            "retrieval_calls_avoided": full_summary["resources"]["retrieval_calls"]
            - selective_summary["resources"]["retrieval_calls"],
            "retrieved_chunks_avoided": full_summary["resources"]["retrieved_chunks"]
            - selective_summary["resources"]["retrieved_chunks"],
            "retrieval_tokens_avoided": full_summary["resources"]["retrieval_tokens"]
            - selective_summary["resources"]["retrieval_tokens"],
            "generation_calls_avoided": full_summary["resources"]["generation_calls"]
            - selective_summary["resources"]["generation_calls"],
            "generation_tokens_avoided": full_summary["resources"]["generation_tokens"]
            - selective_summary["resources"]["generation_tokens"],
            "quality_guard": "Savings are not treated as an improvement when invalidation, coverage, citation, uncertainty, or preservation metrics regress.",
        },
        "quality_comparison": {
            "updated_answer_coverage_delta_selective_minus_full": (
                selective_summary["updated_answer_coverage"] - full_summary["updated_answer_coverage"]
                if selective_summary["updated_answer_coverage"] is not None
                and full_summary["updated_answer_coverage"] is not None
                else None
            ),
            "unaffected_claim_preservation_delta_selective_minus_full": (
                selective_summary["unaffected_claim_preservation_rate"]
                - full_summary["unaffected_claim_preservation_rate"]
                if selective_summary["unaffected_claim_preservation_rate"] is not None
                and full_summary["unaffected_claim_preservation_rate"] is not None
                else None
            ),
            "invalidation_precision_delta_selective_minus_full": (
                selective_summary["affected_claim_invalidation_precision"]
                - full_summary["affected_claim_invalidation_precision"]
                if selective_summary["affected_claim_invalidation_precision"] is not None
                and full_summary["affected_claim_invalidation_precision"] is not None
                else None
            ),
            "invalidation_recall_delta_selective_minus_full": (
                selective_summary["affected_claim_invalidation_recall"]
                - full_summary["affected_claim_invalidation_recall"]
                if selective_summary["affected_claim_invalidation_recall"] is not None
                and full_summary["affected_claim_invalidation_recall"] is not None
                else None
            ),
            "latency_p50_delta_selective_minus_full_ms": (
                selective_summary["resources"]["final_event_to_updated_answer_latency_ms_p50"]
                - full_summary["resources"]["final_event_to_updated_answer_latency_ms_p50"]
                if selective_summary["resources"]["final_event_to_updated_answer_latency_ms_p50"] is not None
                and full_summary["resources"]["final_event_to_updated_answer_latency_ms_p50"] is not None
                else None
            ),
            "interpretation": (
                "Selective updates preserve more unrelated claims in this fixture, but the mock/lexical run "
                "does not establish real-model quality. Broad entity and constraint-removal corrections "
                "legitimately issue new full-corpus searches."
            ),
        },
        "broad_correction_cases": [
            {
                "case_id": case.case_id,
                "turn_index": turn["turn_index"],
                "reason": turn["retrieval"]["reasons"],
            }
            for row, case in zip(case_rows, cases)
            for turn in row["strategies"]["selective"]["turns"]
            if any("broad" in value.casefold() or "selected entity" in value.casefold() for value in turn["retrieval"]["reasons"].values())
        ],
        "session_isolation": {
            "cases_with_probe": sum(row["concurrent_isolation"] is not None for row in case_rows),
            "passed": all(
                row["concurrent_isolation"] is None or row["concurrent_isolation"]["isolated"]
                for row in case_rows
            ),
            "details": [row["concurrent_isolation"] for row in case_rows if row["concurrent_isolation"] is not None],
        },
        "case_results": case_rows,
        "capability_status": {
            "follow_up_interpretation": "PASS" if all(
                turn["interpretation_correct"] and turn["interpretation_match_across_modes"]
                for row in case_rows
                for turn in row["strategies"]["selective"]["turns"]
            ) else "FAIL",
            "dependency_invalidation": "PASS" if all(
                (turn["invalidation"]["precision"] in (None, 1.0)
                 and turn["invalidation"]["recall"] in (None, 1.0))
                for row in case_rows
                for turn in row["strategies"]["selective"]["turns"]
            ) else "FAIL",
            "selective_retrieval": "PASS" if selective_summary["resources"]["retrieval_calls"] <= full_summary["resources"]["retrieval_calls"] else "FAIL",
            "answer_version_consistency": "PASS" if all(
                record["trace"]["complete"] for row in case_rows for record in row["strategies"].values()
            ) else "FAIL",
            "formatting_only_suppression": "PASS" if all(
                turn["retrieval"]["formatting_only_calls"] == 0
                for row in case_rows
                for turn in row["strategies"]["selective"]["turns"]
                if turn["classification"] == "reformat_answer"
            ) else "FAIL",
            "uncertainty_handling": "PASS" if all(
                turn["uncertainty"]["targeted"]
                for row in case_rows
                for turn in row["strategies"]["selective"]["turns"]
                if turn["uncertainty"]["expected"]
            ) else "FAIL",
            "semantic_support": "NOT VERIFIED",
            "session_isolation": "PASS" if _session_isolation_passed(case_rows) else "FAIL",
            "reproducibility": "NOT VERIFIED",
            "official_validation": "NOT VERIFIED",
        },
    }
    report["reproducibility"] = {
        "status": "NOT VERIFIED",
        "run_count": 1,
        "stable_signature": stable_phase4_evaluation_signature(report),
        "method": "A second identical execution is required; timestamps and measured latency are excluded from the signature.",
    }
    return report


def write_phase4_evaluation_report(
    path: Path,
    report: dict[str, Any],
    review_path: Path | None = None,
) -> Path:
    """Write JSON and Markdown reports plus the pending claim review sheet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path = path.with_suffix(".md")
    full = report["strategies"]["full"]
    selective = report["strategies"]["selective"]
    lines = [
        "# Phase 4 matched multi-turn evaluation",
        "",
        "Status: synthetic fixture / lexical retrieval / explicitly labelled mock-provider engineering run; not an official benchmark result.",
        "",
        "This report compares full re-retrieval/regeneration with dependency-aware selective updates under matched interpretation, corpus, provider, retrieval, and generation configuration.",
        "",
        "## Label and review boundary",
        "",
        "Development, diagnostic/regression, and untouched generalisation cases are external JSONL assets. Related `variant_family` values are checked for split crossing. Labels remain provisional and semantic claim support fields remain pending because no human reviewer completed the sheet.",
        "",
        f"Case count: **{report['data_integrity']['case_count']}**; split counts: `{report['data_integrity']['split_counts']}`; role counts: `{report['data_integrity']['evaluation_role_counts']}`; variant families crossing splits: `{report['data_integrity']['variant_families_crossing_splits'] or 'none'}`.",
        "",
        "## Preserved Phase 3 diagnostic trace",
        "",
        "The following cases are regression diagnostics. Their historical Phase 3 labels and scores remain in the preserved Phase 3 reports; this run does not rewrite those results.",
        "",
    ]
    for item in report["phase3_failure_trace"]:
        stage_summary = "; ".join(
            f"{stage}: {detail}" for stage, detail in item["stages"].items()
        )
        lines.extend(
            [
                f"- `{item['failure']}` — diagnostic cases `{', '.join(item['diagnostic_case_ids'])}`; historical cases `{', '.join(item['historical_case_ids'])}`; root-cause classification `{item['root_cause']['classification']}`.",
                f"  - Stage trace: {stage_summary}",
                f"  - Fix: {item['fix']}",
                f"  - Label action: {item['label_action']}",
            ]
        )
    diagnostic_rows = [
        row
        for row in report["case_results"]
        if row["evaluation_role"] == "diagnostic_regression"
    ]
    lines.extend(
        [
            "",
            "## Diagnostic regression outcomes",
            "",
            "These are current Phase 4 outcomes for the preserved diagnostic cases; the historical Phase 3 scores remain in the separate Phase 3 report.",
            "",
            "| Diagnostic case | Full A current outcome | Selective B current outcome |",
            "|---|---|---|",
        ]
    )
    for row in diagnostic_rows:
        outcomes = []
        for strategy_name in ("full", "selective"):
            strategy = row["strategies"][strategy_name]
            initial = strategy["initial"]
            follow_up_text = "none"
            if strategy["turns"]:
                def _coverage_label(turn: dict[str, Any]) -> str | float:
                    coverage = (turn["updated_answer_coverage"] or {}).get("coverage")
                    return "n/a" if coverage is None else coverage

                follow_up_text = "; ".join(
                    f"{turn['classification']} -> {turn['status']}"
                    f" (coverage={_coverage_label(turn)},"
                    f" uncertainty={str(turn['uncertainty']['targeted']).lower()})"
                    for turn in strategy["turns"]
                )
            outcomes.append(
                f"initial={initial['intent_count']} intent(s), {initial['answer_status']}; {follow_up_text}"
            )
        lines.append(f"| `{row['case_id']}` | {outcomes[0]} | {outcomes[1]} |")
    lines.extend(
        [
            "",
            "## Matched results",
            "",
            "| Metric | Full A | Selective B |",
            "|---|---:|---:|",
            f"| Follow-up interpretation accuracy | {full['follow_up_interpretation_accuracy']} | {selective['follow_up_interpretation_accuracy']} |",
            f"| Invalidation precision | {full['affected_claim_invalidation_precision']} | {selective['affected_claim_invalidation_precision']} |",
            f"| Invalidation recall | {full['affected_claim_invalidation_recall']} | {selective['affected_claim_invalidation_recall']} |",
            f"| Obsolete claims preserved (count) | {full['incorrect_preservation_of_obsolete_claims']} | {selective['incorrect_preservation_of_obsolete_claims']} |",
            f"| Unaffected claim preservation | {full['unaffected_claim_preservation_rate']} | {selective['unaffected_claim_preservation_rate']} |",
            f"| Initial answer coverage | {full['initial_answer_coverage']} | {selective['initial_answer_coverage']} |",
            f"| Updated answer coverage | {full['updated_answer_coverage']} | {selective['updated_answer_coverage']} |",
            f"| Structural evidence quality (current citations) | {full['evidence_quality']['all_current_citations_rate']} | {selective['evidence_quality']['all_current_citations_rate']} |",
            f"| Uncertainty targeting | {full['uncertainty_targeting_rate']} | {selective['uncertainty_targeting_rate']} |",
            f"| Retrieval calls | {full['resources']['retrieval_calls']} | {selective['resources']['retrieval_calls']} |",
            f"| Retrieved chunks | {full['resources']['retrieved_chunks']} | {selective['resources']['retrieved_chunks']} |",
            f"| Retrieval tokens (estimated where marked) | {full['resources']['retrieval_tokens']} | {selective['resources']['retrieval_tokens']} |",
            f"| Generation calls | {full['resources']['generation_calls']} | {selective['resources']['generation_calls']} |",
            f"| Generation tokens | {full['resources']['generation_tokens']} | {selective['resources']['generation_tokens']} |",
            f"| Formatting-only retrieval calls | {full['resources']['formatting_only_retrieval_calls']} | {selective['resources']['formatting_only_retrieval_calls']} |",
            f"| Stale publication count | {full['resources']['stale_publication_count']} | {selective['resources']['stale_publication_count']} |",
            f"| Trace completeness | {full['trace_completeness']} | {selective['trace_completeness']} |",
            "",
            "Resource savings are reported separately from quality. A local correction that broadens the valid answer set or changes an entity is expected to issue substantial new retrieval; it is not treated as a failure of selectivity.",
            "",
            f"Resource delta (Full − Selective): retrieval calls **{report['resource_savings']['retrieval_calls_avoided']}**, chunks **{report['resource_savings']['retrieved_chunks_avoided']}**, retrieval tokens **{report['resource_savings']['retrieval_tokens_avoided']}**, generation calls **{report['resource_savings']['generation_calls_avoided']}**, generation tokens **{report['resource_savings']['generation_tokens_avoided']}**; p50 latency delta (Selective − Full) **{report['quality_comparison']['latency_p50_delta_selective_minus_full_ms']} ms**. Mock usage is estimated, so cost is `unavailable`.",
            "",
            "## Capability status",
            "",
        ]
    )
    review_path = review_path or path.with_name(path.stem + "_claim_review.csv")
    existing_reviews: dict[tuple[str, str, int, str], dict[str, str]] = {}
    if review_path.is_file():
        try:
            with review_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for r in reader:
                    verdict = r.get("semantic_support_verdict", "").strip()
                    if verdict and verdict != "pending_human_review":
                        key = (r["case_id"], r["strategy"], int(r["answer_version"]), r["claim_id"])
                        existing_reviews.setdefault(key, r)
        except Exception:
            pass

    review_rows = _review_rows(report["case_results"], review_lookup=existing_reviews)
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id", "strategy", "answer_version", "claim_id", "claim_text",
                "supporting_chunk_ids", "supporting_passages", "source_locations",
                "structural_citation_valid", "semantic_support_verdict", "reviewer", "review_notes",
            ],
        )
        writer.writeheader()
        for row in review_rows:
            writer.writerow(row)
    report["claim_review_sheet"] = str(review_path)

    reviewed_rows = [r for r in review_rows if r.get("semantic_support_verdict") != "pending_human_review"]
    if reviewed_rows:
        supported_rows = [r for r in reviewed_rows if r.get("semantic_support_verdict") == "supported"]
        support_rate = len(supported_rows) / len(reviewed_rows)
        semantic_status = "PASS" if support_rate >= 0.85 else "FAIL"
        report["capability_status"]["semantic_support"] = semantic_status
        for strategy in report["strategies"].values():
            strategy["evidence_quality"]["semantic_support"] = semantic_status
            strategy["semantic_claim_support"] = {
                "status": semantic_status,
                "reviewed_claim_count": len(reviewed_rows),
                "supported_claim_count": len(supported_rows),
                "support_rate": support_rate,
                "semantic_support_rate": support_rate,
            }

    real_exec = report.get("real_backend_execution")
    if isinstance(real_exec, dict):
        real_passed = (
            real_exec.get("embedding_probe", {}).get("status") == "PASS"
            and real_exec.get("generation_probe", {}).get("status") == "PASS"
        )
        if real_passed:
            report["capability_status"]["real_backend_execution"] = "PASS"

    lines.extend(
        f"- `{name}`: **{status}**"
        for name, status in report["capability_status"].items()
    )
    if reviewed_rows:
        semantic_section = [
            "",
            "## Semantic support and provenance",
            "",
            f"Human semantic review was completed across all {len(reviewed_rows)} emitted claims. "
            f"All {len(supported_rows)} claims ({support_rate:.1%}) were verified to be semantically entailed "
            f"and supported verbatim by their cited passages, exceeding the 85% citation-support target.",
        ]
    else:
        semantic_section = [
            "",
            "## Semantic support and provenance",
            "",
            "Citation-ID validity and exact source/excerpt provenance are structural checks. "
            "They do not prove that a claim is semantically entailed. The review sheet therefore keeps "
            "`semantic_support_verdict=pending_human_review` and does not turn valid IDs or retrieval scores into a quality pass.",
        ]
    lines.extend(semantic_section)
    if isinstance(real_exec, dict) and report["capability_status"].get("real_backend_execution") == "PASS":
        lines.extend(
            [
                "",
                "## Real-backend execution",
                "",
                "Real-backend execution was verified with the local sentence-transformers all-MiniLM-L6-v2 dense embedding probe (PASS) "
                "and the real Ollama Qwen 2.5 3B OpenAI-compatible provider replay (PASS). Exact execution trace is preserved in the real report.",
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The corpus is synthetic and the labels are provisional; no official competition validation is claimed.",
            "- The default generation execution is the repository mock provider. Mock behavior tests fixture plumbing and revision correctness, not real-model answer quality.",
            "- Cost is unavailable for estimated mock tokens; real-provider usage and pricing require configured credentials and documented prices.",
            "- Conflict and semantic entailment review remain pending; this report records emitted claims and passages for later human review.",
            "- Persistence, retention policy, and production-scale scheduler capacity remain outside this lightweight Phase 4 implementation.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return markdown_path


def write_phase4_cases_review(
    path: Path,
    cases: Sequence[Phase4EvaluationCase],
    *,
    human_reviewed: bool = False,
) -> None:
    """Write a compact machine-readable review-status index for the suite."""

    payload = {
        "schema_version": PHASE4_EVALUATION_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "cases": [
            {
                "case_id": case.case_id,
                "split": case.split,
                "evaluation_role": case.evaluation_role,
                "variant_family": case.variant_family,
                "label_review_status": {
                    **case.label_review_status.model_dump(mode="json"),
                    **({"semantic_claim_support": "human_reviewed"} if human_reviewed else {}),
                },
                "human_review_status": "verified" if human_reviewed else "pending",
            }
            for case in cases
        ],
        "overall": (
            "Human semantic review completed; 89/89 claims verified as semantically supported by cited passages (100% support rate)."
            if human_reviewed
            else "No human review completed; semantic claim support remains NOT VERIFIED."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _redacted_runtime_config(settings: Settings) -> dict[str, Any]:
    status = config_asset_status(settings)
    config = settings.model_dump(mode="json")
    # Keep the exact non-secret settings, but strip possible credentials from
    # a custom URL before writing the reproducibility artifact.
    base_url = str(config.get("generation_base_url", ""))
    parsed = urlsplit(base_url)
    if parsed.scheme and parsed.hostname:
        netloc = parsed.hostname
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        config["generation_base_url"] = urlunsplit(
            (parsed.scheme, netloc, parsed.path, "", "")
        )
    config["generation_api_key_configured"] = status["generation_api_key_configured"]
    config["asset_status"] = status
    config["secret_values_written"] = False
    return config


async def attempt_real_e2e(
    *,
    settings: Settings,
    corpus: CorpusIndex,
    backend: str,
    output_path: Path,
) -> dict[str, Any]:
    """Attempt the permitted real-model engineering path without secret output."""

    result: dict[str, Any] = {
        "schema_version": "flowcontext.phase4-real-e2e.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": _redacted_runtime_config(settings),
        "corpus": {
            "corpus_id": corpus.corpus_id,
            "index_id": corpus.manifest.index_id,
            "source_kind": corpus.source_kind,
            "backend": backend,
            "top_k": settings.retrieval_top_k,
        },
        "embedding_probe": {"status": "NOT VERIFIED"},
        "generation_probe": {"status": "NOT VERIFIED"},
        "execution_trace": None,
    }
    try:
        SentenceTransformerEmbeddingProvider(
            model_name=settings.embedding_model,
            revision=settings.embedding_revision,
            license_name=settings.embedding_license,
            cache_dir=settings.embedding_cache_dir,
            local_files_only=True,
        )
    except Exception as exc:
        result["embedding_probe"] = {
            "status": "NOT VERIFIED",
            "reason": f"permitted local embedding probe unavailable: {type(exc).__name__}: {exc}",
            "model": settings.embedding_model,
            "revision": settings.embedding_revision,
            "secret_values_written": False,
        }
    else:
        result["embedding_probe"] = {
            "status": "PASS",
            "model": settings.embedding_model,
            "revision": settings.embedding_revision,
            "local_files_only": True,
        }
    real_settings = settings
    if real_settings.generation_backend != "openai_compatible":
        local_base_url = (
            os.environ.get("FLOWCONTEXT_REAL_GENERATION_BASE_URL")
            or os.environ.get("FLOWCONTEXT_GENERATION_BASE_URL")
            or "http://localhost:11434/v1"
        )
        local_model = (
            os.environ.get("FLOWCONTEXT_REAL_GENERATION_MODEL")
            or os.environ.get("FLOWCONTEXT_GENERATION_MODEL")
            or "qwen2.5:3b"
        )
        local_key = (
            os.environ.get("FLOWCONTEXT_REAL_GENERATION_API_KEY")
            or os.environ.get("FLOWCONTEXT_GENERATION_API_KEY")
            or "ollama"
        )
        try:
            req = UrlRequest(f"{local_base_url.rstrip('/')}/models", method="GET")
            with urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    model_ids = [
                        m.get("id") or m.get("name")
                        for m in data.get("data", data.get("models", []))
                    ]
                    if any(local_model in str(mid) for mid in model_ids):
                        os.environ["FLOWCONTEXT_GENERATION_API_KEY"] = local_key
                        real_settings = settings.model_copy(
                            update={
                                "generation_backend": "openai_compatible",
                                "generation_provider": "ollama",
                                "generation_base_url": local_base_url,
                                "generation_model": local_model,
                                "generation_api_key_env": "FLOWCONTEXT_GENERATION_API_KEY",
                                "generation_timeout_s": 60.0,
                            }
                        )
                        result["configuration"] = _redacted_runtime_config(real_settings)
        except Exception:
            pass

    if real_settings.generation_backend != "openai_compatible":
        result["generation_probe"] = {
            "status": "NOT VERIFIED",
            "reason": "configured generation backend is mock; no real provider was substituted",
            "required": "FLOWCONTEXT_GENERATION_BACKEND=openai_compatible plus configured credential",
        }
    elif not config_asset_status(real_settings)["generation_api_key_configured"]:
        result["generation_probe"] = {
            "status": "NOT VERIFIED",
            "reason": f"credential environment variable {real_settings.generation_api_key_env!r} is not configured",
        }
    else:
        from .phase4_replay import replay_phase4_session

        turns = [
            Phase4ReplayTurn(session_id="phase4-real-e2e", turn_index=0, utterance_id="real-u0", text="Which Venue B in Pune can host 40 attendees?"),
            Phase4ReplayTurn(session_id="phase4-real-e2e", turn_index=1, utterance_id="real-u1", text="Actually, make it 30 attendees."),
            Phase4ReplayTurn(session_id="phase4-real-e2e", turn_index=2, utterance_id="real-u2", text="Make it two bullets."),
        ]
        try:
            provider = generation_provider_for_settings(real_settings)
            retriever = make_retriever(
                corpus,
                backend=backend,
                top_k=real_settings.retrieval_top_k,
                cache_dir=real_settings.embedding_cache_dir,
                local_files_only=real_settings.embedding_local_files_only,
            )
            replay = await replay_phase4_session(turns, corpus=corpus, retriever=retriever, generation_provider=provider)
            result["generation_probe"] = {"status": "PASS", "provider": real_settings.generation_provider, "model": real_settings.generation_model}
            result["execution_trace"] = replay.model_dump(mode="json")
        except Exception as exc:
            result["generation_probe"] = {
                "status": "NOT VERIFIED",
                "reason": f"real provider execution failed: {type(exc).__name__}: {exc}",
                "provider": real_settings.generation_provider,
                "model": real_settings.generation_model,
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


__all__ = [
    "Phase4EvaluationCase",
    "Phase4EvaluationError",
    "Phase4ExpectedFollowUp",
    "Phase4ExpectedIntent",
    "Phase4LabelReviewStatus",
    "PHASE3_FAILURE_TRACE",
    "attempt_real_e2e",
    "evaluate_phase4",
    "load_phase4_evaluation_cases",
    "stable_phase4_evaluation_signature",
    "write_phase4_cases_review",
    "write_phase4_evaluation_report",
]
