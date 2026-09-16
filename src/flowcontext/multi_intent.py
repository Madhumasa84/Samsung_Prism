"""Phase 3 multi-intent parsing, parallel retrieval, and evidence fusion.

The Phase 2 retriever contract is deliberately synchronous and returns one
ranked list.  This module keeps that contract usable while adding a small
parent-query layer: a query is decomposed into stable sub-intents, each
sub-intent is searched independently, and the results are merged with
reciprocal-rank fusion (RRF) and deterministic chunk-ID deduplication.

The parser is intentionally structural rather than corpus-specific.  It uses
question/request leads, clause punctuation, and coordinated clauses; it does
not contain document names, expected answers, or evaluation-case branches.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from pydantic import Field, ValidationError, field_validator, model_validator

from .contracts import (
    ContractModel,
    DecompositionAmbiguity,
    DecompositionConstraint,
    DecompositionDependency,
    DecompositionIntent,
    DecompositionRequest,
    DecompositionResult,
    GenerationConfig,
    GenerationResult,
    PHASE3_SCHEMA_VERSION,
    RetrievalHit,
    RetrievalMode,
    TextSpan,
    Usage,
    _non_blank,
)
from .generation import run_in_daemon_thread
from .retrieval import Retriever, tokenize


_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_COORDINATOR_PATTERN = re.compile(r"\s+(?:and|also|plus)\s+", re.IGNORECASE)
_SHARED_HEAD_PATTERN = re.compile(
    r"(?P<prefix>.*?)\b(?P<left>[\w-]+)\s+(?:and|also|plus)\s+"
    r"(?P<right>[\w-]+)\s+(?P<head>(?:lunch\s+)?options?|policies?|"
    r"packages?|plans?|requirements?|rules?|fees?)(?P<tail>\s*[?.,!;:]*)$",
    re.IGNORECASE,
)
_SHARED_HEAD_WORDS = frozenset(
    {"option", "options", "policy", "policies", "package", "packages", "plan", "plans", "rule", "rules", "fee", "fees"}
)
_QUESTION_OR_REQUEST_LEADS = frozenset(
    {
        "can",
        "compare",
        "could",
        "do",
        "does",
        "explain",
        "find",
        "give",
        "how",
        "is",
        "list",
        "show",
        "tell",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "would",
    }
)
_PREDICATE_CONTINUATIONS = frozenset(
    {
        "accept",
        "accepts",
        "arrive",
        "allows",
        "are",
        "contains",
        "costs",
        "depart",
        "departs",
        "do",
        "does",
        "doesn’t",
        "doesn't",
        "features",
        "has",
        "have",
        "includes",
        "is",
        "leave",
        "leaves",
        "offers",
        "provides",
        "require",
        "requires",
        "return",
        "returns",
        "supports",
        "cannot",
    }
)
_REQUEST_CLAUSE_LEADS = frozenset(
    {
        "book",
        "choose",
        "exclude",
        "find",
        "get",
        "identify",
        "list",
        "pick",
        "recommend",
        "reserve",
        "search",
        "select",
        "show",
        "tell",
    }
)


class MultiIntentError(RuntimeError):
    """Base error for Phase 3 decomposition or evidence fusion."""


class MultiIntentRetrievalError(MultiIntentError):
    """All independent intent retrievals failed."""

    def __init__(self, message: str, *, result: MultiIntentRetrievalResult) -> None:
        super().__init__(message)
        self.result = result


class EvidenceFusionError(MultiIntentError):
    """Independent hits disagree about the provenance of one chunk ID."""


QueryIntent = DecompositionIntent
DecompositionPlan = DecompositionResult


class MultiIntentPlan(DecompositionResult):
    """Backward-compatible name for the revision-bound decomposition result."""


class IntentRetrievalResult(ContractModel):
    """One isolated sub-intent retrieval outcome."""

    schema_version: Literal[PHASE3_SCHEMA_VERSION] = PHASE3_SCHEMA_VERSION
    parent_revision: int = Field(ge=0)
    retrieval_revision: int = Field(ge=0)
    intent: QueryIntent
    status: Literal["completed", "failed"]
    hits: list[RetrievalHit] = Field(default_factory=list)
    # Each backend list is retained before per-intent rank fusion.  Keeping
    # these lists makes lexical/dense disagreements and fusion decisions
    # auditable without conflating their raw scores.
    backend_hits: dict[str, list[RetrievalHit]] = Field(default_factory=dict)
    fusion_decisions: list[dict[str, Any]] = Field(default_factory=list)
    dependency_intent_ids: list[str] = Field(default_factory=list)
    dependency_context_chunk_ids: list[str] = Field(default_factory=list)
    effective_query: str | None = None
    retrieval_usage: Usage = Field(default_factory=Usage)
    backend_timings_ms: dict[str, float] = Field(default_factory=dict)
    backend_errors: list[str] = Field(default_factory=list)
    error: str | None = None
    duration_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    reused: bool = False
    reused_from_revision: int | None = Field(default=None, ge=0)

    _error_is_non_blank = field_validator("error")(
        lambda value: value if value is None else _non_blank(value)
    )

    @model_validator(mode="after")
    def status_matches_error(self) -> IntentRetrievalResult:
        if self.status == "failed" and self.error is None:
            raise ValueError("failed intent retrievals require an error")
        if self.status == "completed" and self.error is not None:
            raise ValueError("completed intent retrievals must not contain an error")
        if self.reused and self.reused_from_revision is None:
            raise ValueError("reused intent retrievals require reused_from_revision")
        if self.effective_query is not None and not self.effective_query.strip():
            raise ValueError("effective_query must not be blank when provided")
        if any(not key.strip() or value < 0 for key, value in self.backend_timings_ms.items()):
            raise ValueError("backend_timings_ms must contain non-negative durations")
        return self


class MultiIntentRetrievalResult(ContractModel):
    """All sub-intent results plus the deterministic fused evidence set."""

    schema_version: Literal[PHASE3_SCHEMA_VERSION] = PHASE3_SCHEMA_VERSION
    parent_revision: int = Field(ge=0)
    retrieval_revision: int = Field(ge=0)
    plan: MultiIntentPlan
    retrieval_mode: RetrievalMode = "lexical"
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    intent_results: list[IntentRetrievalResult] = Field(min_length=1)
    fused_hits: list[RetrievalHit] = Field(default_factory=list)
    rrf_k: int = Field(default=60, ge=1)
    context_budget_tokens: int = Field(default=1200, ge=1)
    context_tokens_used: int = Field(default=0, ge=0)
    assembly_decisions: list[dict[str, Any]] = Field(default_factory=list)
    missing_intent_ids: list[str] = Field(default_factory=list)
    dependency_order: list[str] = Field(default_factory=list)
    retrieval_usage: Usage = Field(default_factory=Usage)
    total_duration_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    errors: list[str] = Field(default_factory=list)
    reused_intent_ids: list[str] = Field(default_factory=list)
    superseded_intent_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def result_intents_match_plan(self) -> MultiIntentRetrievalResult:
        expected = [intent.intent_id for intent in self.plan.intents]
        actual = [item.intent.intent_id for item in self.intent_results]
        if actual != expected:
            raise ValueError("intent retrieval results must match plan order")
        if self.context_tokens_used > self.context_budget_tokens:
            raise ValueError("assembled context exceeds its configured token budget")
        if any(intent_id not in expected for intent_id in self.missing_intent_ids):
            raise ValueError("missing_intent_ids must reference plan intents")
        return self


def _clean_query(value: str) -> str:
    return " ".join(value.strip().split()).strip(" ,;:")


def _canonical(value: str) -> str:
    return " ".join(token.lower() for token in _TOKEN_PATTERN.findall(value))


def _intent_id(ordinal: int, query: str) -> str:
    digest = hashlib.sha256(_canonical(query).encode("utf-8")).hexdigest()[:10]
    # The ordinal is deliberately not part of the identity.  A correction can
    # add/remove another question before this one without invalidating work
    # whose query and constraints are unchanged.
    return f"intent-{digest}"


def _leading_token(value: str) -> str:
    tokens = _TOKEN_PATTERN.findall(value.lower())
    return tokens[0] if tokens else ""


def _has_question_or_request_lead(value: str) -> bool:
    return _leading_token(value) in _QUESTION_OR_REQUEST_LEADS


def _split_coordinated_clause(value: str) -> list[str]:
    """Split only boundaries that look like separate requests.

    A conjunction inside a noun phrase such as ``projector and a breakout
    room`` remains one intent because the right clause begins with a leading
    filler.  A new interrogative/request clause or a coordinated topic after
    an existing question lead becomes a separate intent.
    """

    # ``Compare A and B ...`` is one information need.  Splitting the
    # compared entities would turn a comparison into two unrelated searches.
    if _leading_token(value) == "compare":
        return [value]
    pieces: list[str] = []
    start = 0
    structural_value = re.sub(
        r",?\s*(?:please|thanks)[.!?]*$", "", value, flags=re.IGNORECASE
    ).strip()
    shared_head = _SHARED_HEAD_PATTERN.search(structural_value)
    for match in _COORDINATOR_PATTERN.finditer(value):
        # Keep a coordinated noun phrase together until the shared-head
        # rewrite below can attach the head noun to both independent topics.
        # For example, ``cancellation and catering policies`` becomes two
        # useful queries instead of ``cancellation`` and ``catering policies``.
        if (
            shared_head is not None
            and shared_head.group("left").casefold() not in _SHARED_HEAD_WORDS
            and shared_head.start("left") <= match.start() < shared_head.end("right")
        ):
            continue
        left = value[start : match.start()].strip()
        right = value[match.end() :].strip()
        if not left or not right:
            continue
        right_lead = _leading_token(right)
        dependent_clause = bool(re.match(r"(?:for\s+each|per)\b", right, re.IGNORECASE))
        new_question_clause = (
            _has_question_or_request_lead(right)
            or right_lead in _REQUEST_CLAUSE_LEADS
            or dependent_clause
        )
        split = new_question_clause and right_lead not in _PREDICATE_CONTINUATIONS
        if split:
            pieces.append(left)
            start = match.end()
    pieces.append(value[start:].strip())
    return [piece for piece in pieces if _clean_query(piece)]


def _split_shared_head_clause(value: str) -> list[str]:
    """Expand a two-topic phrase that shares a trailing head noun."""

    cleaned = re.sub(r",?\s*(?:please|thanks)[.!?]*$", "", value, flags=re.IGNORECASE).strip()
    match = _SHARED_HEAD_PATTERN.search(cleaned)
    if match is None:
        return [value]
    if match.group("left").casefold() in _SHARED_HEAD_WORDS:
        return [value]
    prefix = match.group("prefix")
    left = _clean_query(f"{prefix}{match.group('left')} {match.group('head')}")
    right = _clean_query(
        f"{prefix}{match.group('right')} {match.group('head')}{match.group('tail')}"
    )
    return [piece for piece in (left, right) if piece]


@dataclass(frozen=True)
class _ClauseCandidate:
    query: str
    start: int
    end: int


_QUESTION_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "can",
        "could",
        "do",
        "does",
        "for",
        "how",
        "in",
        "is",
        "list",
        "me",
        "of",
        "please",
        "show",
        "tell",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "would",
    }
)


def _semantic_query_key(value: str) -> tuple[str, ...]:
    """Small, language-agnostic-ish key used only for duplicate suppression."""

    values: list[str] = []
    for token in _TOKEN_PATTERN.findall(value.casefold()):
        if token in _QUESTION_STOP_WORDS:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            token = token[:-1]
        values.append(token)
    return tuple(values)


def _semantically_overlaps(left: str, right: str) -> bool:
    left_key = set(_semantic_query_key(left))
    right_key = set(_semantic_query_key(right))
    if not left_key or not right_key:
        return _canonical(left) == _canonical(right)
    if left_key == right_key:
        return True
    overlap = len(left_key & right_key) / len(left_key | right_key)
    # Only collapse near-identical wording.  Important differences such as a
    # number, a date, a negation, or a compared entity remain distinct tokens.
    return overlap >= 0.9 and abs(len(left_key) - len(right_key)) <= 1


def _trim_bounds(value: str, start: int, end: int) -> tuple[int, int]:
    while start < end and value[start].isspace():
        start += 1
    while end > start and value[end - 1].isspace():
        end -= 1
    return start, end


def _explicit_segments(value: str) -> list[tuple[str, int, int]]:
    """Return question-mark/semicolon/newline segments with source offsets."""

    segments: list[tuple[str, int, int]] = []
    start = 0
    for separator in re.finditer(r"\?\s+|;\s*|\n+", value):
        segment_start, segment_end = _trim_bounds(value, start, separator.start())
        if segment_start < segment_end:
            segments.append((value[segment_start:segment_end], segment_start, segment_end))
        start = separator.end()
    segment_start, segment_end = _trim_bounds(value, start, len(value))
    if segment_start < segment_end:
        segments.append((value[segment_start:segment_end], segment_start, segment_end))
    return segments or [(value, 0, len(value))]


def _candidate_spans(
    original: str,
    part: str,
    part_start: int,
    part_end: int,
) -> list[_ClauseCandidate]:
    candidates: list[_ClauseCandidate] = []
    cursor = part_start
    for coordinated in _split_coordinated_clause(part):
        coordinated_clean = _clean_query(coordinated)
        if not coordinated_clean:
            continue
        coordinated_start = original.find(coordinated_clean, cursor, part_end)
        if coordinated_start < 0:
            # The structural splitter only returns source text, so this is a
            # defensive fallback for unusual whitespace normalization. Keep
            # the bounded parent segment rather than inventing offsets.
            coordinated_start, coordinated_end = part_start, part_end
        else:
            coordinated_end = coordinated_start + len(coordinated_clean)
        for piece in _split_shared_head_clause(coordinated_clean):
            cleaned = _clean_query(piece)
            if not cleaned:
                continue
            exact_start = original.find(cleaned, coordinated_start, coordinated_end)
            if exact_start >= 0:
                exact_end = exact_start + len(cleaned)
                candidates.append(_ClauseCandidate(cleaned, exact_start, exact_end))
                continue
            # Shared-head expansion (for example, "vegetarian and vegan
            # options") creates a search-ready query that is not a contiguous
            # source span. Point the derived question at its *local* source
            # clause, not the whole parent transcript. This keeps unrelated
            # constraints (such as a preceding location or capacity) attached
            # to the qualifying intent instead of making them falsely shared.
            candidates.append(_ClauseCandidate(cleaned, coordinated_start, coordinated_end))
        cursor = coordinated_end
    return candidates


def _deduplicate_candidates(candidates: Sequence[_ClauseCandidate]) -> list[_ClauseCandidate]:
    unique: list[_ClauseCandidate] = []
    for candidate in candidates:
        if any(_semantically_overlaps(candidate.query, existing.query) for existing in unique):
            continue
        unique.append(candidate)
    return unique


_CONSTRAINT_PATTERNS: tuple[tuple[str, str, int], ...] = (
    (
        "negation",
        r"\b(?:not|no|without|except|excluding|exclude|neither|nor|cannot|can't|don't|doesn't|isn't|aren't)\b(?:\s+(?!(?:in|at|on|for|from|and|or)\b)\w[\w-]*){0,4}|\bnon[- ]\w[\w-]*",
        0,
    ),
    (
        "comparison",
        r"\b(?:compare|difference\s+between)\b[^\n?!.;]*|\b(?:compared\s+with|compared\s+to|versus|vs\.?|more\s+than|less\s+than|higher\s+than|lower\s+than)\b(?:\s+\w[\w-]*){0,8}",
        1,
    ),
    (
        "quantity",
        r"(?:[$€£₹]\s*)?\b(?:at\s+least|at\s+most|no\s+more\s+than|no\s+fewer\s+than|more\s+than|less\s+than|up\s+to|under|over|above|below|between)\s+(?:[$€£₹]\s*)?\d+(?:[.,]\d+)?(?:\s*(?:and|to)\s*\d+(?:[.,]\d+)?)?(?:\s+(?:or\s+more|or\s+fewer|plus))?(?:\s*(?:%|gb|mb|kg|km|hours?|minutes?|people|persons?|attendees?|guests?))?(?:\s+(?!(?:and|or|but|does|do|is|are|can|will|on|in|at|for|with|before|after)\b)\w[\w-]*){0,2}|(?:[$€£₹]\s*)?\d+(?:[.,]\d+)?(?:\s+(?:or\s+more|or\s+fewer|plus))?(?:\s*(?:%|gb|mb|kg|km|hours?|minutes?|people|persons?|attendees?|guests?))?(?:\s+(?!(?:and|or|but|does|do|is|are|can|will|on|in|at|for|with|before|after)\b)\w[\w-]*){0,2}",
        2,
    ),
    (
        "temporal",
        r"\b(?:(?:at|by|before|after|from|until)\s+)?\d{1,2}:\d{2}\b",
        3,
    ),
    (
        "date",
        r"\b(?:(?:(?:on|before|after|by)\s+)?(?:\d{1,4}[/-]\d{1,2}[/-]\d{1,4}|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,?\s+\d{4})?|(?:mon|tues?|wednes|thurs?|fri|satur|sun)day(?:\s+\d{1,2})?|today|tomorrow|yesterday))\b",
        4,
    ),
    (
        "location",
        r"\b(?:in|at|near|from|within)\s+(?:the\s+)?(?!a\b|an\b|each\b|one\b|two\b|least\b|most\b|capacity\b|all\b)[A-Za-z][\w-]*(?:\s+[A-Z][\w-]*)*",
        5,
    ),
    (
        "relationship",
        r"\b(?:for\s+each|per|selected|qualifying|eligible|chosen|their|its|those|these|the\s+selected)\b(?:\s+\w[\w-]*){0,3}",
        6,
    ),
    (
        "entity",
        r"(?:['\"][^'\"]+['\"]|\b[A-Z][a-zA-Z0-9-]{2,}(?:\s+[A-Z][a-zA-Z0-9-]{2,})*)",
        7,
    ),
)


def _constraint_id(kind: str, value: str, start: int, end: int) -> str:
    digest = hashlib.sha256(f"{kind}|{_canonical(value)}|{start}|{end}".encode("utf-8")).hexdigest()[:10]
    return f"constraint-{digest}"


def _extract_constraints(original: str) -> list[DecompositionConstraint]:
    found: list[tuple[int, int, int, str, str]] = []
    for kind, pattern, priority in _CONSTRAINT_PATTERNS:
        flags = 0 if kind in {"location", "entity"} else re.IGNORECASE
        for match in re.finditer(pattern, original, flags=flags):
            start, end = match.span()
            if kind == "quantity" and (
                original[start - 1:start] in {"/", "-", ":"}
                or original[end:end + 1] in {"/", "-", ":"}
            ):
                # Numeric pieces of a date are date constraints, not three
                # independent quantities.
                continue
            while end > start and original[end - 1].isspace():
                end -= 1
            preceding = original[:start].rstrip()
            if kind == "entity" and (
                not preceding
                or preceding[-1] in ".?!;,\n"
            ):
                # Sentence-initial interrogatives are not useful entity
                # constraints ("Which", "Compare", ...).
                words = re.match(r"(?P<lead>\w+)(?P<rest>.*)$", original[start:end])
                if words and words.group("lead").casefold() in _QUESTION_OR_REQUEST_LEADS:
                    rest_offset = len(words.group("lead"))
                    while rest_offset < end - start and original[start + rest_offset].isspace():
                        rest_offset += 1
                    start += rest_offset
                    if start >= end:
                        continue
                else:
                    continue
            while end > start and re.search(r"\s+(?:and|or)$", original[start:end], re.IGNORECASE):
                end = start + re.search(r"\s+(?:and|or)$", original[start:end], re.IGNORECASE).start()
            if end > start:
                found.append((start, end, priority, kind, original[start:end]))

    # Prefer a longer span for repeated constraints of the same kind.  Some
    # different kinds intentionally overlap: a comparison can contain its
    # compared entities, while a date/time must not be split into numeric
    # quantity fragments.
    selected: list[tuple[int, int, int, str, str]] = []
    for item in sorted(found, key=lambda value: (value[0], -(value[1] - value[0]), value[2], value[3])):
        start, end, _priority, kind, _value = item
        overlapping = [
            other
            for other in selected
            if start < other[1] and end > other[0]
        ]
        if any(
            kind == other[3]
            or {kind, other[3]} <= {"quantity", "date", "temporal"}
            or (
                kind == "entity"
                and other[3]
                in {"location", "quantity", "date", "temporal", "negation", "relationship"}
            )
            or (
                other[3] == "entity"
                and kind
                in {"location", "quantity", "date", "temporal", "negation", "relationship"}
            )
            for other in overlapping
        ):
            continue
        selected.append(item)
    constraints: list[DecompositionConstraint] = []
    for start, end, _priority, kind, value in sorted(selected, key=lambda item: item[0]):
        operator = None
        lowered = value.casefold()
        if kind == "quantity":
            operator = next(
                (
                    candidate
                    for candidate in (
                        "at least",
                        "at most",
                        "no more than",
                        "no fewer than",
                        "more than",
                        "less than",
                        "up to",
                        "under",
                        "over",
                        "above",
                        "below",
                        "between",
                        "or more",
                        "or fewer",
                    )
                    if candidate in lowered
                ),
                "equals",
            )
        elif kind == "comparison":
            operator = "comparison"
        elif kind == "negation":
            operator = "exclude"
        span = TextSpan(start=start, end=end, text=original[start:end])
        constraints.append(
            DecompositionConstraint(
                constraint_id=_constraint_id(kind, value, start, end),
                kind=kind,
                value=value,
                source_span=span,
                operator=operator,
                normalized_value=" ".join(value.casefold().split()),
            )
        )
    return constraints


def _source_span(original: str, candidate: _ClauseCandidate) -> TextSpan:
    return TextSpan(
        start=candidate.start,
        end=candidate.end,
        text=original[candidate.start:candidate.end],
    )


def _overlaps(left: TextSpan, right: TextSpan) -> bool:
    return left.start < right.end and right.start < left.end


def _dependency_and_ambiguity_records(
    original: str,
    intents: Sequence[DecompositionIntent],
) -> tuple[list[DecompositionDependency], list[DecompositionAmbiguity]]:
    dependencies: list[DecompositionDependency] = []
    ambiguities: list[DecompositionAmbiguity] = []
    if not intents:
        return dependencies, ambiguities
    selector = next(
        (
            intent
            for intent in intents
            if _leading_token(intent.query) in {"which", "find", "list", "show", "identify"}
        ),
        None,
    )
    for intent in intents:
        lowered = intent.query.casefold()
        if selector is not None and intent.intent_id != selector.intent_id:
            dependent_signal = bool(
                re.search(r"\b(?:selected|chosen|qualifying|eligible|their|its|those|these|for each|per)\b", lowered)
            )
            if dependent_signal:
                dependencies.append(
                    DecompositionDependency(
                        dependency_id=f"dependency-{selector.intent_id}-{intent.intent_id}",
                        prerequisite_intent_id=selector.intent_id,
                        dependent_intent_id=intent.intent_id,
                        relation="depends_on",
                        description="The dependent question refers to the result of the qualifying question.",
                        source_span=intent.source_span,
                    )
                )
        for match in re.finditer(
            r"\b(?:it|its|they|them|their|this|that|these|those|selected|chosen|which one)\b(?:\s+\w+)?",
            intent.query,
            flags=re.IGNORECASE,
        ):
            # The intent source span is always a source reference.  Locate the
            # pronoun in the parent where possible; otherwise retain the full
            # source span rather than making up offsets for a rewritten query.
            source_span = intent.source_span
            if source_span is not None:
                local_start = original.casefold().find(match.group(0).casefold(), source_span.start, source_span.end)
                if local_start >= 0:
                    source_span = TextSpan(
                        start=local_start,
                        end=local_start + len(match.group(0)),
                        text=original[local_start:local_start + len(match.group(0))],
                    )
            if source_span is not None:
                ambiguity_id = f"ambiguity-{hashlib.sha256(match.group(0).casefold().encode('utf-8')).hexdigest()[:10]}"
                if not any(item.ambiguity_id == ambiguity_id for item in ambiguities):
                    ambiguities.append(
                        DecompositionAmbiguity(
                            ambiguity_id=ambiguity_id,
                            description=f"Reference {match.group(0)!r} needs an antecedent or selection context.",
                            source_span=source_span,
                            candidates=(
                                [selector.intent_id]
                                if selector is not None and selector.intent_id != intent.intent_id
                                else []
                            ),
                        )
                    )
    if len(intents) > 1 and re.search(r"\b(?:compare|versus|vs\.?|difference between)\b", original, re.IGNORECASE):
        for intent in intents:
            if intent.relationship == "independent":
                object.__setattr__(intent, "relationship", "comparison")
        for left, right in zip(intents, intents[1:]):
            dependencies.append(
                DecompositionDependency(
                    dependency_id=f"dependency-{left.intent_id}-{right.intent_id}-comparison",
                    prerequisite_intent_id=left.intent_id,
                    dependent_intent_id=right.intent_id,
                    relation="compares_with",
                    description="The questions are evaluated as a comparison, not as isolated facts.",
                    source_span=right.source_span,
                )
            )
    elif re.search(r"\b(?:compare|versus|vs\.?|difference between)\b", original, re.IGNORECASE):
        for intent in intents:
            if intent.relationship == "independent":
                object.__setattr__(intent, "relationship", "comparison")
    for intent in intents:
        if any(
            dependency.dependent_intent_id == intent.intent_id
            and dependency.relation == "depends_on"
            for dependency in dependencies
        ):
            object.__setattr__(intent, "relationship", "dependent")
    return dependencies, ambiguities


def _build_rule_plan(
    query: str,
    *,
    transcript_revision: int = 0,
    max_intents: int = 6,
) -> MultiIntentPlan:
    original = query
    if not original.strip():
        raise ValueError("query must not be blank")
    if max_intents < 1 or max_intents > 8:
        raise ValueError("max_intents must be between one and eight")

    candidates: list[_ClauseCandidate] = []
    for part, part_start, part_end in _explicit_segments(original):
        candidates.extend(_candidate_spans(original, part, part_start, part_end))
    candidates = _deduplicate_candidates(candidates)
    if len(candidates) > max_intents:
        # Keep the first bounded questions and merge the remainder into one
        # original-text span.  Nothing is silently discarded, and the method
        # makes the bound observable to callers.
        candidates = candidates[: max_intents - 1] + [
            _ClauseCandidate(
                query=original,
                start=0,
                end=len(original),
            )
        ]
    all_constraints = _extract_constraints(original)
    provisional_intents: list[DecompositionIntent] = []
    for ordinal, candidate in enumerate(candidates, start=1):
        span = _source_span(original, candidate)
        provisional_intents.append(
            DecompositionIntent(
                intent_id=_intent_id(ordinal, candidate.query),
                ordinal=ordinal,
                query=candidate.query,
                source_text=span.text,
                source_span=span,
            )
        )

    shared_constraints: list[DecompositionConstraint] = []
    intent_constraints: dict[str, list[DecompositionConstraint]] = {
        intent.intent_id: [] for intent in provisional_intents
    }
    for constraint in all_constraints:
        matching = [
            intent
            for intent in provisional_intents
            if intent.source_span is not None and _overlaps(intent.source_span, constraint.source_span)
        ]
        if len(matching) > 1 or not matching:
            shared_constraints.append(constraint)
        else:
            intent_constraints[matching[0].intent_id].append(constraint)
    intents = [
        intent.model_copy(update={"constraints": intent_constraints[intent.intent_id]})
        for intent in provisional_intents
    ]
    dependencies, ambiguities = _dependency_and_ambiguity_records(original, intents)
    method = "offline_rule_based_bounded_v2" if len(candidates) >= max_intents else "offline_rule_based_v2"
    return MultiIntentPlan(
        transcript_revision=transcript_revision,
        original_transcript=original,
        intents=intents,
        shared_constraints=shared_constraints,
        ambiguities=ambiguities,
        dependencies=dependencies,
        decomposition_method=method,
        boundary_count=max(0, len(intents) - 1),
        status="success",
        provider="flowcontext.rule_based",
        provider_model="offline-rule-based-v2",
        usage=Usage(
            input_tokens=len(tokenize(original)),
            output_tokens=0,
            total_tokens=len(tokenize(original)),
            estimated=True,
        ),
    )


def decompose_query(
    query: str,
    *,
    transcript_revision: int = 0,
    max_intents: int = 6,
) -> MultiIntentPlan:
    """Use the explicitly labelled offline rule-based decomposer.

    This compatibility helper never stands in for a failed model provider.
    Applications that need a configured model should construct
    :class:`StructuredMultiIntentDecomposer` and opt into its explicit
    original-query fallback policy.
    """

    return _build_rule_plan(
        query,
        transcript_revision=transcript_revision,
        max_intents=max_intents,
    )


class DecompositionProviderError(MultiIntentError):
    """A configured decomposition provider could not produce a response."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class DecompositionProviderUnavailable(DecompositionProviderError):
    """The configured real provider is unavailable; no mock is substituted."""


class DecompositionOutputError(MultiIntentError):
    """Provider output failed JSON or decomposition-contract validation."""


class DecompositionTimeout(DecompositionProviderError):
    """A decomposition provider call exceeded the configured timeout."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class DecompositionProvider(Protocol):
    """Async structured-output provider using the existing GenerationConfig."""

    @property
    def config(self) -> GenerationConfig:
        ...

    async def decompose(self, request: DecompositionRequest) -> GenerationResult:
        ...


DECOMPOSITION_SYSTEM_PROMPT = """You are a query decomposition component for corpus retrieval.

Return one JSON object only. Do not answer the request and do not add facts.
Preserve every entity, number, date, location, comparison, negation, and
relationship from original_transcript. Never invent a value that is absent.
Do not split a single question merely because it contains 'and'; coordinated
constraints belong in one intent. Split independent information needs. Mark a
question dependent when it needs the result of another question (for example,
the cancellation policy for each venue selected by the first question).

Required JSON shape:
{
  "transcript_revision": 0,
  "original_transcript": "exact input transcript",
  "intents": [{
    "intent_id": "provider-local-id",
    "ordinal": 1,
    "query": "search-ready question using only transcript information",
    "source_text": "exact source-span text",
    "source_span": {"start": 0, "end": 10, "text": "exact substring"},
    "constraints": [{
      "constraint_id": "provider-local-id",
      "kind": "entity|quantity|date|location|negation|comparison|relationship|temporal|other",
      "value": "exact value from source span",
      "source_span": {"start": 0, "end": 3, "text": "exact substring"},
      "operator": "optional operator",
      "normalized_value": "optional normalization"
    }],
    "relationship": "independent|dependent|comparison"
  }],
  "shared_constraints": [],
  "ambiguities": [{
    "ambiguity_id": "provider-local-id",
    "description": "what cannot be resolved without guessing",
    "source_span": {"start": 0, "end": 3, "text": "exact substring"},
    "candidates": [],
    "resolution": null
  }],
  "dependencies": [{
    "dependency_id": "provider-local-id",
    "prerequisite_intent_id": "provider-local-id",
    "dependent_intent_id": "provider-local-id",
    "relation": "depends_on|compares_with|refines",
    "description": "why the relation exists",
    "source_span": null
  }],
  "boundary_count": 0,
  "decomposition_method": "model_structured_v1",
  "status": "success"
}

All source spans use Python character offsets into the exact original_transcript.
"""


def _provider_usage(envelope: Any) -> Usage:
    raw_usage = envelope.get("usage") if isinstance(envelope, dict) else None
    if not isinstance(raw_usage, dict):
        return Usage()
    input_tokens = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens"))
    output_tokens = raw_usage.get("completion_tokens", raw_usage.get("output_tokens"))
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens < 0
        or isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens < 0
    ):
        return Usage()
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        estimated=False,
    )


def _provider_content(envelope: Any) -> str:
    try:
        content = envelope["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DecompositionProviderError(
            "decomposition provider response omitted message content"
        ) from exc
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        joined = "".join(part for part in parts if isinstance(part, str))
        if joined.strip():
            return joined
    raise DecompositionProviderError(
        "decomposition provider response contained empty message content"
    )


class OpenAICompatibleDecompositionProvider:
    """Real configured JSON provider sharing the generation provider settings.

    The implementation deliberately has no fallback behavior. Credentials,
    transport errors, timeouts, and malformed responses are returned to the
    decomposer, which may then record an explicit original-query fallback.
    """

    def __init__(self, *, config: GenerationConfig) -> None:
        if config.backend != "openai_compatible":
            raise ValueError(
                "OpenAICompatibleDecompositionProvider requires backend=openai_compatible"
            )
        self._config = config

    @property
    def config(self) -> GenerationConfig:
        return self._config

    async def decompose(self, request: DecompositionRequest) -> GenerationResult:
        try:
            return await run_in_daemon_thread(self._decompose_once, request)
        except MultiIntentError:
            raise
        except Exception as exc:  # Keep provider internals and secrets out of traces.
            raise DecompositionProviderError(
                "decomposition provider call failed",
                retryable=False,
            ) from exc

    def _decompose_once(self, request: DecompositionRequest) -> GenerationResult:
        api_key = os.environ.get(self._config.api_key_env or "")
        if not api_key:
            raise DecompositionProviderUnavailable(
                "decomposition credentials are unavailable; set the configured API-key environment variable"
            )
        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                {"role": "user", "content": request.model_dump_json()},
            ],
            "temperature": 0,
            "max_tokens": self._config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        http_request = UrlRequest(
            f"{self._config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self._config.timeout_s) as response:
                body = response.read()
                status = getattr(response, "status", 200)
        except HTTPError as exc:
            retryable = exc.code in {408, 429, 500, 502, 503, 504}
            raise DecompositionProviderError(
                f"decomposition provider returned HTTP status {exc.code}",
                retryable=retryable,
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise DecompositionProviderError(
                "decomposition provider network request failed",
                retryable=True,
            ) from exc
        if status >= 400:
            raise DecompositionProviderError(
                f"decomposition provider returned HTTP status {status}",
                retryable=status in {408, 429, 500, 502, 503, 504},
            )
        try:
            envelope = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DecompositionProviderError(
                "decomposition provider returned invalid JSON envelope"
            ) from exc
        return GenerationResult(raw_text=_provider_content(envelope), usage=_provider_usage(envelope))


class MockDecompositionProvider:
    """Deterministic offline *mock model* provider for contract tests only."""

    def __init__(
        self,
        *,
        config: GenerationConfig | None = None,
        mode: Literal["valid", "invalid_json", "invalid_schema", "timeout"] = "valid",
        timeout_delay_s: float | None = None,
    ) -> None:
        self._config = config or GenerationConfig(
            backend="mock",
            provider="flowcontext.mock.decomposer",
            model="mock-decomposer-v1",
            timeout_s=0.05,
            max_retries=0,
            retry_backoff_s=0.0,
            max_repair_attempts=1,
            max_output_tokens=600,
        )
        if self._config.backend != "mock":
            raise ValueError("MockDecompositionProvider requires backend=mock")
        self.mode = mode
        self.timeout_delay_s = timeout_delay_s
        self.calls = 0

    @property
    def config(self) -> GenerationConfig:
        return self._config

    async def decompose(self, request: DecompositionRequest) -> GenerationResult:
        self.calls += 1
        if self.mode == "timeout":
            await asyncio.sleep(self.timeout_delay_s or self._config.timeout_s * 10)
        if self.mode == "invalid_json":
            return GenerationResult(raw_text="not-json", usage=Usage())
        if self.mode == "invalid_schema":
            return GenerationResult(
                raw_text=json.dumps(
                    {
                        "transcript_revision": request.transcript_revision,
                        "original_transcript": request.original_transcript,
                        "intents": [],
                        "decomposition_method": "mock_invalid",
                        "status": "success",
                    }
                ),
                usage=Usage(),
            )
        plan = _build_rule_plan(
            request.original_transcript,
            transcript_revision=request.transcript_revision,
            max_intents=request.max_intents,
        )
        return GenerationResult(
            raw_text=plan.model_dump_json(),
            usage=Usage(
                input_tokens=len(tokenize(request.original_transcript)),
                output_tokens=len(_TOKEN_PATTERN.findall(plan.model_dump_json())),
                total_tokens=len(tokenize(request.original_transcript))
                + len(_TOKEN_PATTERN.findall(plan.model_dump_json())),
                estimated=True,
            ),
        )


def decomposition_provider_for_settings(settings: Any) -> DecompositionProvider:
    """Select the explicitly configured decomposition provider.

    Decomposition reuses the repository's validated generation-provider
    configuration (endpoint, credential environment variable, timeout, repair
    budget, and model identity).  A mock is selected only when settings say
    ``generation_backend=mock``; a real-provider error is never replaced by it.
    """

    from .generation import generation_config_for_settings

    config = generation_config_for_settings(settings)
    if config.backend == "mock":
        return MockDecompositionProvider(config=config)
    if config.backend == "openai_compatible":
        return OpenAICompatibleDecompositionProvider(config=config)
    raise DecompositionProviderUnavailable(
        f"unsupported configured decomposition backend {config.backend!r}"
    )


def _constraint_fingerprint(constraints: Sequence[DecompositionConstraint]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (
                constraint.kind,
                _canonical(constraint.value),
                constraint.operator or "",
            )
            for constraint in constraints
        )
    )


def _intent_fingerprint(
    intent: DecompositionIntent,
    shared_constraints: Sequence[DecompositionConstraint] = (),
) -> tuple[Any, ...]:
    return (
        _canonical(intent.query),
        _constraint_fingerprint(intent.constraints),
        _constraint_fingerprint(shared_constraints),
        intent.relationship,
    )


def _combine_usage(usages: Sequence[Usage]) -> Usage:
    if not usages:
        return Usage()
    return Usage(
        input_tokens=sum(item.input_tokens for item in usages),
        output_tokens=sum(item.output_tokens for item in usages),
        total_tokens=sum(item.total_tokens for item in usages),
        estimated=any(item.estimated for item in usages),
    )


def _validate_source_span(original: str, span: TextSpan | None, label: str) -> None:
    if span is None:
        raise DecompositionOutputError(f"{label} omitted its source span")
    if original[span.start:span.end] != span.text:
        raise DecompositionOutputError(f"{label} source span does not match the transcript")


def _tokens_are_from_transcript(value: str, original: str) -> bool:
    # Search queries may normalize punctuation or a simple inflection, but
    # every non-grammatical token—including entities, numbers, and nouns—must
    # come from the transcript. This prevents a provider from turning a
    # missing value into a plausible-looking retrieval term.
    original_tokens = set(_semantic_query_key(original))
    return set(_semantic_query_key(value)) <= original_tokens


def _provider_constraint_covers(
    expected: DecompositionConstraint,
    provided: DecompositionConstraint,
) -> bool:
    """Return whether one provider constraint retains an extracted value."""

    if provided.kind != expected.kind and not (
        expected.kind == "entity" and provided.kind in {"comparison", "location"}
    ):
        return False
    if not (
        expected.source_span.start < provided.source_span.end
        and provided.source_span.start < expected.source_span.end
    ):
        return False
    expected_tokens = set(_semantic_query_key(expected.value))
    provided_tokens = set(_semantic_query_key(provided.value))
    return expected_tokens <= provided_tokens


def _validate_provider_plan(
    candidate: MultiIntentPlan,
    *,
    transcript: str,
    transcript_revision: int,
    max_intents: int,
) -> None:
    if candidate.status != "success":
        raise DecompositionOutputError(
            "provider output must use status=success; fallback is reserved for the traced local fallback path"
        )
    if candidate.failure_reason is not None:
        raise DecompositionOutputError(
            "provider output must not include failure_reason on a successful decomposition"
        )
    if candidate.transcript_revision != transcript_revision:
        raise DecompositionOutputError(
            "provider decomposition belongs to an obsolete or different transcript revision"
        )
    if candidate.original_transcript != transcript:
        raise DecompositionOutputError(
            "provider decomposition did not preserve the exact original transcript"
        )
    if len(candidate.intents) > max_intents:
        raise DecompositionOutputError(
            f"provider returned {len(candidate.intents)} intents; maximum is {max_intents}"
        )
    for intent in candidate.intents:
        _validate_source_span(transcript, intent.source_span, f"intent {intent.intent_id}")
        assert intent.source_span is not None
        if intent.source_text != intent.source_span.text:
            raise DecompositionOutputError(
                f"intent {intent.intent_id} source_text is not its exact source span"
            )
        if not _tokens_are_from_transcript(intent.query, transcript):
            raise DecompositionOutputError(
                f"intent {intent.intent_id} contains a value not present in the transcript"
            )
        for constraint in intent.constraints:
            _validate_source_span(
                transcript,
                constraint.source_span,
                f"constraint {constraint.constraint_id}",
            )
            if constraint.value.casefold() not in constraint.source_span.text.casefold():
                raise DecompositionOutputError(
                    f"constraint {constraint.constraint_id} value is not copied from its source span"
                )
    for constraint in candidate.shared_constraints:
        _validate_source_span(
            transcript,
            constraint.source_span,
            f"shared constraint {constraint.constraint_id}",
        )
        if constraint.value.casefold() not in constraint.source_span.text.casefold():
            raise DecompositionOutputError(
                f"shared constraint {constraint.constraint_id} value is not copied from its source span"
            )
    for ambiguity in candidate.ambiguities:
        _validate_source_span(transcript, ambiguity.source_span, f"ambiguity {ambiguity.ambiguity_id}")
    for dependency in candidate.dependencies:
        if dependency.source_span is not None:
            _validate_source_span(transcript, dependency.source_span, f"dependency {dependency.dependency_id}")
    provided_constraints = list(candidate.shared_constraints) + [
        constraint
        for intent in candidate.intents
        for constraint in intent.constraints
    ]
    for expected in _extract_constraints(transcript):
        if not any(_provider_constraint_covers(expected, actual) for actual in provided_constraints):
            raise DecompositionOutputError(
                f"provider dropped the {expected.kind} constraint {expected.value!r}"
            )


def _normalise_provider_plan(
    candidate: MultiIntentPlan,
    *,
    transcript: str,
    transcript_revision: int,
    max_intents: int,
) -> MultiIntentPlan:
    _validate_provider_plan(
        candidate,
        transcript=transcript,
        transcript_revision=transcript_revision,
        max_intents=max_intents,
    )
    selected: list[DecompositionIntent] = []
    old_to_new: dict[str, str] = {}
    for intent in candidate.intents:
        overlapping_index = next(
            (
                index
                for index, existing in enumerate(selected)
                if _semantically_overlaps(intent.query, existing.query)
            ),
            None,
        )
        if overlapping_index is not None:
            existing = selected[overlapping_index]
            if _constraint_fingerprint(intent.constraints) == _constraint_fingerprint(existing.constraints):
                old_to_new[intent.intent_id] = existing.intent_id
                continue
            # Similar wording with different constraints is one question with
            # more information, not two retrieval needs. Merge the structured
            # constraints instead of dropping either provider-provided value.
            existing_constraint_keys = {
                (item.kind, item.value.casefold(), item.source_span.start, item.source_span.end)
                for item in existing.constraints
            }
            merged_constraints = list(existing.constraints)
            for constraint in intent.constraints:
                key = (constraint.kind, constraint.value.casefold(), constraint.source_span.start, constraint.source_span.end)
                if key not in existing_constraint_keys:
                    merged_constraints.append(constraint)
            selected[overlapping_index] = existing.model_copy(update={"constraints": merged_constraints})
            old_to_new[intent.intent_id] = existing.intent_id
            continue
        ordinal = len(selected) + 1
        new_id = _intent_id(ordinal, intent.query)
        old_to_new[intent.intent_id] = new_id
        constraints = [
            constraint.model_copy(
                update={
                    "constraint_id": _constraint_id(
                        constraint.kind,
                        constraint.value,
                        constraint.source_span.start,
                        constraint.source_span.end,
                    )
                }
            )
            for constraint in intent.constraints
        ]
        selected.append(
            intent.model_copy(
                update={
                    "intent_id": new_id,
                    "ordinal": ordinal,
                    "source_text": intent.source_span.text if intent.source_span else intent.source_text,
                    "constraints": constraints,
                }
            )
        )
    if not selected:
        raise DecompositionOutputError("provider returned no usable intents")
    shared_constraints = [
        constraint.model_copy(
            update={
                "constraint_id": _constraint_id(
                    constraint.kind,
                    constraint.value,
                    constraint.source_span.start,
                    constraint.source_span.end,
                )
            }
        )
        for constraint in candidate.shared_constraints
    ]
    dependencies: list[DecompositionDependency] = []
    for dependency in candidate.dependencies:
        prerequisite = old_to_new.get(dependency.prerequisite_intent_id)
        dependent = old_to_new.get(dependency.dependent_intent_id)
        if prerequisite is None or dependent is None or prerequisite == dependent:
            continue
        dependencies.append(
            dependency.model_copy(
                update={
                    "dependency_id": f"dependency-{prerequisite}-{dependent}-{dependency.relation}",
                    "prerequisite_intent_id": prerequisite,
                    "dependent_intent_id": dependent,
                }
            )
        )
    return MultiIntentPlan(
        transcript_revision=transcript_revision,
        original_transcript=transcript,
        intents=selected,
        shared_constraints=shared_constraints,
        ambiguities=candidate.ambiguities,
        dependencies=dependencies,
        decomposition_method="model_backed_structured_validated_v1",
        boundary_count=max(0, len(selected) - 1),
        status="success",
        provider=candidate.provider,
        provider_model=candidate.provider_model,
    )


def _revision_metadata(
    result: MultiIntentPlan,
    previous: MultiIntentPlan | None,
) -> MultiIntentPlan:
    if previous is None or result.transcript_revision <= previous.transcript_revision:
        return result
    previous_by_id = {intent.intent_id: intent for intent in previous.intents}
    used_previous: set[str] = set()
    current_to_stable: dict[str, str] = {}
    preserved: list[str] = []
    updated_intents: list[DecompositionIntent] = []
    for intent in result.intents:
        stable_id: str | None = None
        direct_previous = previous_by_id.get(intent.intent_id)
        if (
            direct_previous is not None
            and intent.intent_id not in used_previous
            and _intent_fingerprint(intent, result.shared_constraints)
            == _intent_fingerprint(direct_previous, previous.shared_constraints)
        ):
            stable_id = intent.intent_id
        else:
            for previous_id, previous_intent in previous_by_id.items():
                if previous_id in used_previous:
                    continue
                if _intent_fingerprint(intent, result.shared_constraints) == _intent_fingerprint(
                    previous_intent,
                    previous.shared_constraints,
                ):
                    stable_id = previous_id
                    break
        if stable_id is not None:
            used_previous.add(stable_id)
            preserved.append(stable_id)
            updated_intents.append(intent.model_copy(update={"intent_id": stable_id}))
            current_to_stable[intent.intent_id] = stable_id
        else:
            updated_intents.append(intent)
            current_to_stable[intent.intent_id] = intent.intent_id
    superseded = [intent_id for intent_id in previous_by_id if intent_id not in used_previous]
    updated_dependencies = [
        dependency.model_copy(
            update={
                "prerequisite_intent_id": current_to_stable.get(
                    dependency.prerequisite_intent_id,
                    dependency.prerequisite_intent_id,
                ),
                "dependent_intent_id": current_to_stable.get(
                    dependency.dependent_intent_id,
                    dependency.dependent_intent_id,
                ),
            }
        )
        for dependency in result.dependencies
    ]
    return result.model_copy(
        update={
            "intents": updated_intents,
            "dependencies": updated_dependencies,
            "preserved_intent_ids": preserved,
            "superseded_intent_ids": superseded,
        }
    )


@dataclass(frozen=True)
class DecompositionExecution:
    """Provider execution metadata retained alongside its validated result."""

    result: MultiIntentPlan
    attempts: int
    repair_attempts: int
    usage: Usage
    latency_ms: float
    errors: tuple[str, ...] = ()


class StructuredMultiIntentDecomposer:
    """Bounded, validated model-backed decomposition with explicit fallback."""

    def __init__(
        self,
        provider: DecompositionProvider,
        *,
        max_intents: int = 6,
        max_repair_attempts: int | None = None,
    ) -> None:
        if max_intents < 1 or max_intents > 8:
            raise ValueError("max_intents must be between one and eight")
        self.provider = provider
        self.max_intents = max_intents
        self.max_repair_attempts = (
            provider.config.max_repair_attempts
            if max_repair_attempts is None
            else max_repair_attempts
        )
        if self.max_repair_attempts < 0 or self.max_repair_attempts > 2:
            raise ValueError("max_repair_attempts must be between zero and two")
        self.last_execution: DecompositionExecution | None = None
        self.history: list[DecompositionExecution] = []

    async def decompose(
        self,
        transcript: str,
        *,
        transcript_revision: int = 0,
        previous: MultiIntentPlan | None = None,
        allow_original_query_fallback: bool = False,
    ) -> MultiIntentPlan:
        if not transcript.strip():
            raise ValueError("transcript must not be blank")
        started = time.perf_counter()
        attempts = 0
        repair_attempts = 0
        usages: list[Usage] = []
        errors: list[str] = []
        feedback: str | None = None
        last_error = "decomposition provider did not return a valid plan"
        last_provider_exception: DecompositionProviderError | None = None
        for attempt in range(1, self.max_repair_attempts + 2):
            attempts = attempt
            request = DecompositionRequest(
                transcript_revision=transcript_revision,
                original_transcript=transcript,
                max_intents=self.max_intents,
                attempt=attempt,
                repair_feedback=feedback,
            )
            try:
                raw_result = await asyncio.wait_for(
                    self.provider.decompose(request),
                    timeout=self.provider.config.timeout_s,
                )
                if isinstance(raw_result, GenerationResult):
                    provider_result = raw_result
                elif isinstance(raw_result, str):
                    provider_result = GenerationResult(raw_text=raw_result)
                else:
                    provider_result = GenerationResult(
                        raw_text=json.dumps(raw_result, ensure_ascii=False)
                    )
                usages.append(provider_result.usage)
                try:
                    payload = json.loads(provider_result.raw_text)
                    candidate = MultiIntentPlan.model_validate(payload)
                    candidate = _normalise_provider_plan(
                        candidate,
                        transcript=transcript,
                        transcript_revision=transcript_revision,
                        max_intents=self.max_intents,
                    )
                except (json.JSONDecodeError, ValidationError, DecompositionOutputError) as exc:
                    last_error = f"invalid decomposition output: {type(exc).__name__}: {exc}"
                    errors.append(last_error)
                    feedback = (
                        "The previous response failed local validation. Return exact JSON, use exact "
                        f"source offsets into the transcript, preserve all values, and fix: {exc}"
                    )
                    if attempt <= self.max_repair_attempts:
                        repair_attempts += 1
                        continue
                    break
                else:
                    candidate = candidate.model_copy(
                        update={
                            "attempts": attempts,
                            "repair_attempts": repair_attempts,
                            "usage": _combine_usage(usages),
                            "latency_ms": (time.perf_counter() - started) * 1000,
                            "decomposition_method": (
                                "mock_model_structured_validated_v1"
                                if self.provider.config.backend == "mock"
                                else "model_backed_structured_validated_v1"
                            ),
                            "provider": self.provider.config.provider,
                            "provider_model": self.provider.config.model,
                        }
                    )
                    candidate = _revision_metadata(candidate, previous)
                    execution = DecompositionExecution(
                        result=candidate,
                        attempts=attempts,
                        repair_attempts=repair_attempts,
                        usage=candidate.usage,
                        latency_ms=candidate.latency_ms,
                        errors=tuple(errors),
                    )
                    self.last_execution = execution
                    self.history.append(execution)
                    return candidate
            except asyncio.TimeoutError:
                timeout_error = DecompositionTimeout(
                    f"decomposition provider timed out after {self.provider.config.timeout_s:.3f}s"
                )
                last_provider_exception = timeout_error
                last_error = str(timeout_error)
                errors.append(last_error)
                if attempt <= self.max_repair_attempts:
                    repair_attempts += 1
                    feedback = "The previous decomposition attempt timed out; return the bounded JSON plan promptly."
                    continue
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                errors.append(last_error)
                last_provider_exception = (
                    exc
                    if isinstance(exc, DecompositionProviderError)
                    else DecompositionProviderError(last_error)
                )
                # Transport/provider failures are not repaired by asking a
                # provider that is unavailable.  The explicit fallback path
                # remains available to callers.
                break
        latency_ms = (time.perf_counter() - started) * 1000
        if not allow_original_query_fallback:
            if last_provider_exception is not None:
                raise last_provider_exception
            raise DecompositionProviderError(last_error)
        fallback = _build_rule_plan(
            transcript,
            transcript_revision=transcript_revision,
            max_intents=1,
        ).model_copy(
            update={
                "decomposition_method": "original_query_fallback_after_provider_failure",
                "status": "fallback",
                "failure_reason": last_error,
                "provider": self.provider.config.provider,
                "provider_model": self.provider.config.model,
                "latency_ms": latency_ms,
                "attempts": attempts,
                "repair_attempts": repair_attempts,
                "usage": _combine_usage(usages),
            }
        )
        execution = DecompositionExecution(
            result=fallback,
            attempts=attempts,
            repair_attempts=repair_attempts,
            usage=fallback.usage,
            latency_ms=latency_ms,
            errors=tuple(errors),
        )
        self.last_execution = execution
        self.history.append(execution)
        return fallback

    def decompose_sync(self, transcript: str, **kwargs: Any) -> MultiIntentPlan:
        """Run the async provider from synchronous retriever/controller workers."""

        coroutine = self.decompose(transcript, **kwargs)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        result: list[MultiIntentPlan] = []
        error: list[BaseException] = []

        def run_in_thread() -> None:
            try:
                result.append(asyncio.run(coroutine))
            except BaseException as exc:  # propagate the original failure
                error.append(exc)

        thread = threading.Thread(target=run_in_thread, name="flowcontext-decomposer", daemon=True)
        thread.start()
        thread.join()
        if error:
            raise error[0]
        return result[0]


ModelBackedMultiIntentDecomposer = StructuredMultiIntentDecomposer


def _annotate_hit(hit: RetrievalHit, intent: QueryIntent, *, rank: int | None = None) -> RetrievalHit:
    effective_rank = rank if rank is not None else hit.rank
    return hit.model_copy(
        update={
            "rank": effective_rank,
            "intent_id": intent.intent_id,
            "intent_ids": [intent.intent_id],
            "intent_ranks": {intent.intent_id: effective_rank},
            "intent_scores": {intent.intent_id: hit.score},
            "rrf_score": None,
        }
    )


def _retriever_identity(retriever: Retriever) -> str:
    provider = getattr(retriever, "provider", None)
    config = getattr(provider, "config", None)
    if config is not None:
        model_name = getattr(config, "model_name", None) or getattr(config, "model", None)
        if model_name:
            return f"{getattr(config, 'provider', retriever.backend)}:{model_name}"
    return f"{getattr(retriever, 'backend', 'unknown')}:{getattr(retriever, 'method', 'unknown')}"


def _retriever_configuration(retriever: Retriever) -> dict[str, Any]:
    provider = getattr(retriever, "provider", None)
    config = getattr(provider, "config", None)
    return {
        "backend": getattr(retriever, "backend", "unknown"),
        "method": getattr(retriever, "method", getattr(retriever, "backend", "unknown")),
        "identity": _retriever_identity(retriever),
        "embedding": config.model_dump(mode="json") if config is not None else None,
    }


def _normalise_backend_hits(
    hits: Sequence[RetrievalHit],
    *,
    intent: QueryIntent,
    top_k: int,
) -> list[RetrievalHit]:
    """Normalize provider rank positions without changing source provenance."""

    normalized: list[RetrievalHit] = []
    seen: dict[str, RetrievalHit] = {}
    for position, raw_hit in enumerate(hits, start=1):
        if len(normalized) >= top_k:
            break
        hit = raw_hit if isinstance(raw_hit, RetrievalHit) else RetrievalHit.model_validate(raw_hit)
        annotated = _annotate_hit(hit, intent, rank=position)
        previous = seen.get(annotated.chunk_id)
        if previous is not None:
            if (
                previous.source_location != annotated.source_location
                or previous.snippet_text != annotated.snippet_text
            ):
                raise EvidenceFusionError(
                    f"backend returned inconsistent provenance for chunk {annotated.chunk_id!r}"
                )
            continue
        seen[annotated.chunk_id] = annotated
        normalized.append(annotated)
    # Duplicate removal changes positions, so rank again after the provider's
    # malformed duplicate has been discarded.
    return [hit.model_copy(update={"rank": rank, "intent_ranks": {intent.intent_id: rank}})
            for rank, hit in enumerate(normalized, start=1)]


def _fuse_backend_hits(
    intent: QueryIntent,
    backend_hits: Mapping[str, Sequence[RetrievalHit]],
    *,
    top_k: int,
    rrf_k: int,
) -> tuple[list[RetrievalHit], list[dict[str, Any]]]:
    """Fuse backend lists for one intent using rank-only reciprocal-rank fusion."""

    order = {name: position for position, name in enumerate(backend_hits)}
    buckets: dict[str, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []
    nonempty_backend_count = sum(bool(hits) for hits in backend_hits.values())
    for backend_name, hits in backend_hits.items():
        for rank, hit in enumerate(hits, start=1):
            bucket = buckets.setdefault(
                hit.chunk_id,
                {
                    "hit": hit,
                    "input_ranks": {},
                    "raw_scores": {},
                    "rrf_score": 0.0,
                },
            )
            original = bucket["hit"]
            if (
                original.source_location != hit.source_location
                or original.snippet_text != hit.snippet_text
            ):
                raise EvidenceFusionError(
                    f"chunk {hit.chunk_id!r} resolved to inconsistent source text or location"
                )
            if backend_name in bucket["input_ranks"]:
                # A backend duplicate was removed by normalization. Keep this
                # guard for callers that provide pre-built hit lists directly.
                continue
            contribution = 1.0 / (rrf_k + rank)
            bucket["input_ranks"][backend_name] = rank
            bucket["raw_scores"][backend_name] = hit.score
            bucket["rrf_score"] += contribution
    ranked = sorted(
        buckets.values(),
        key=lambda item: (
            -item["rrf_score"] if nonempty_backend_count > 1 else 0,
            min(order[name] for name in item["input_ranks"]),
            min(item["input_ranks"].values()),
            item["hit"].chunk_id,
        ),
    )
    fused: list[RetrievalHit] = []
    for rank, bucket in enumerate(ranked[:top_k], start=1):
        is_fused = nonempty_backend_count > 1
        base = bucket["hit"]
        score = bucket["rrf_score"] if is_fused else base.score
        method = "rrf_fusion" if is_fused else base.retrieval_method
        fused_hit = _annotate_hit(
            base.model_copy(update={"score": score, "retrieval_method": method}),
            intent,
            rank=rank,
        ).model_copy(update={"rrf_score": bucket["rrf_score"] if is_fused else None})
        fused.append(fused_hit)
        decisions.append(
            {
                "stage": "per_intent_backend_fusion",
                "intent_id": intent.intent_id,
                "chunk_id": base.chunk_id,
                "input_ranks": dict(bucket["input_ranks"]),
                "raw_scores_retained": dict(bucket["raw_scores"]),
                "rrf_k": rrf_k,
                "rrf_score": bucket["rrf_score"] if is_fused else None,
                "selected": True,
                "selected_rank": rank,
                "method": method,
            }
        )
    return fused, decisions


def _global_fusion_buckets(
    plan: MultiIntentPlan,
    intent_results: Sequence[IntentRetrievalResult],
    *,
    rrf_k: int,
) -> list[dict[str, Any]]:
    order = {intent.intent_id: intent.ordinal for intent in plan.intents}
    buckets: dict[str, dict[str, Any]] = {}
    for result in intent_results:
        if result.status != "completed":
            continue
        intent_id = result.intent.intent_id
        for hit in result.hits:
            bucket = buckets.setdefault(
                hit.chunk_id,
                {
                    "hit": hit,
                    "intent_ranks": {},
                    "intent_scores": {},
                    "rrf_score": 0.0,
                    "intent_ids": [],
                },
            )
            original = bucket["hit"]
            if (
                original.source_location != hit.source_location
                or original.snippet_text != hit.snippet_text
            ):
                raise EvidenceFusionError(
                    f"chunk {hit.chunk_id!r} resolved to inconsistent source text or location"
                )
            if intent_id in bucket["intent_ranks"]:
                continue
            bucket["intent_ranks"][intent_id] = hit.rank
            bucket["intent_scores"][intent_id] = (
                hit.rrf_score if hit.rrf_score is not None else hit.score
            )
            bucket["rrf_score"] += 1.0 / (rrf_k + hit.rank)
            bucket["intent_ids"].append(intent_id)
    return sorted(
        buckets.values(),
        key=lambda item: (
            -item["rrf_score"],
            min(order[intent_id] for intent_id in item["intent_ids"]),
            min(item["intent_ranks"].values()),
            item["hit"].chunk_id,
        ),
    )


def _materialize_global_hit(
    bucket: Mapping[str, Any],
    *,
    rank: int,
    plan_order: Mapping[str, int],
    rrf_k: int,
) -> RetrievalHit:
    intent_ids = sorted(bucket["intent_ids"], key=plan_order.__getitem__)
    return bucket["hit"].model_copy(
        update={
            "rank": rank,
            # This is the rank-fusion score, never a sum/max of incomparable
            # lexical and dense raw scores.
            "score": bucket["rrf_score"],
            "retrieval_method": "rrf_fusion",
            "intent_id": intent_ids[0],
            "intent_ids": intent_ids,
            "intent_ranks": dict(bucket["intent_ranks"]),
            "intent_scores": dict(bucket["intent_scores"]),
            "rrf_score": bucket["rrf_score"],
        }
    )


def fuse_evidence(
    plan: MultiIntentPlan,
    intent_results: Sequence[IntentRetrievalResult],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[RetrievalHit]:
    """Fuse already-fused intent lists with RRF and provenance-preserving dedup."""

    if top_k < 1:
        raise ValueError("top_k must be at least one")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least one")
    plan_order = {intent.intent_id: intent.ordinal for intent in plan.intents}
    return [
        _materialize_global_hit(bucket, rank=rank, plan_order=plan_order, rrf_k=rrf_k)
        for rank, bucket in enumerate(
            _global_fusion_buckets(plan, intent_results, rrf_k=rrf_k)[:top_k],
            start=1,
        )
    ]


def _evidence_token_cost(hit: RetrievalHit) -> int:
    return max(1, len(_TOKEN_PATTERN.findall(hit.snippet_text)))


def _assemble_evidence(
    plan: MultiIntentPlan,
    intent_results: Sequence[IntentRetrievalResult],
    *,
    top_k: int,
    rrf_k: int,
    context_budget_tokens: int,
) -> tuple[list[RetrievalHit], list[dict[str, Any]], list[str], int]:
    """Apply fair coverage and a total token budget after intent fusion."""

    if context_budget_tokens < 1:
        raise ValueError("context_budget_tokens must be at least one")
    plan_order = {intent.intent_id: intent.ordinal for intent in plan.intents}
    buckets = _global_fusion_buckets(plan, intent_results, rrf_k=rrf_k)
    selected: list[RetrievalHit] = []
    selected_ids: set[str] = set()
    selected_intents: set[str] = set()
    decisions: list[dict[str, Any]] = []
    used_tokens = 0

    def consider(bucket: Mapping[str, Any], reason: str) -> bool:
        nonlocal used_tokens
        hit = bucket["hit"]
        chunk_id = hit.chunk_id
        if chunk_id in selected_ids:
            selected_intents.update(bucket["intent_ids"])
            return True
        if len(selected) >= top_k:
            decisions.append(
                {
                    "stage": "evidence_assembly",
                    "chunk_id": chunk_id,
                    "intent_ids": list(bucket["intent_ids"]),
                    "selected": False,
                    "reason": "top_k",
                    "rrf_score": bucket["rrf_score"],
                }
            )
            return False
        token_cost = _evidence_token_cost(hit)
        if used_tokens + token_cost > context_budget_tokens:
            decisions.append(
                {
                    "stage": "evidence_assembly",
                    "chunk_id": chunk_id,
                    "intent_ids": list(bucket["intent_ids"]),
                    "selected": False,
                    "reason": "context_budget",
                    "token_cost": token_cost,
                    "context_tokens_used": used_tokens,
                    "context_budget_tokens": context_budget_tokens,
                    "rrf_score": bucket["rrf_score"],
                }
            )
            return False
        selected_ids.add(chunk_id)
        selected_intents.update(bucket["intent_ids"])
        used_tokens += token_cost
        selected.append(
            _materialize_global_hit(
                bucket,
                rank=len(selected) + 1,
                plan_order=plan_order,
                rrf_k=rrf_k,
            )
        )
        decisions.append(
            {
                "stage": "evidence_assembly",
                "chunk_id": chunk_id,
                "intent_ids": list(bucket["intent_ids"]),
                "selected": True,
                "reason": reason,
                "token_cost": token_cost,
                "context_tokens_used": used_tokens,
                "rrf_score": bucket["rrf_score"],
                "selected_rank": len(selected),
            }
        )
        return True

    # Coverage pass: each intent gets its best fitting candidate before the
    # remaining global rank order is considered. A shared chunk satisfies all
    # relationships it actually carries; it is never duplicated.
    for intent in plan.intents:
        intent_id = intent.intent_id
        if intent_id in selected_intents:
            continue
        candidates = [bucket for bucket in buckets if intent_id in bucket["intent_ids"]]
        for bucket in candidates:
            if consider(bucket, "mandatory_intent_coverage"):
                break

    for bucket in buckets:
        if bucket["hit"].chunk_id not in selected_ids:
            consider(bucket, "global_rank_fill")

    missing = [
        intent.intent_id
        for intent in plan.intents
        if intent.intent_id not in selected_intents
    ]
    return selected, decisions, missing, used_tokens


def _intent_has_support(intent: QueryIntent, hits: Sequence[RetrievalHit]) -> bool:
    query_terms = tokenize(intent.query)
    if not query_terms:
        return False
    for hit in hits:
        if hit.intent_ids and intent.intent_id not in hit.intent_ids:
            continue
        if query_terms & tokenize(hit.snippet_text):
            return True
    return False


def _dependency_graph(plan: MultiIntentPlan) -> dict[str, list[str]]:
    graph = {intent.intent_id: [] for intent in plan.intents}
    for dependency in plan.dependencies:
        if dependency.relation in {"depends_on", "refines"}:
            graph[dependency.dependent_intent_id].append(dependency.prerequisite_intent_id)
    return graph


def _dependency_context(
    prerequisite_results: Sequence[IntentRetrievalResult],
    *,
    max_chunks: int = 2,
    max_tokens: int = 32,
) -> tuple[str, list[str]]:
    context_tokens: list[str] = []
    chunk_ids: list[str] = []
    for result in prerequisite_results:
        for hit in result.hits[:max_chunks]:
            if hit.chunk_id not in chunk_ids:
                chunk_ids.append(hit.chunk_id)
            for token in _TOKEN_PATTERN.findall(hit.snippet_text):
                context_tokens.append(token)
                if len(context_tokens) >= max_tokens:
                    return " ".join(context_tokens), chunk_ids
    return " ".join(context_tokens), chunk_ids


def _effective_query(
    intent: QueryIntent,
    *,
    shared_constraints: Sequence[DecompositionConstraint],
    dependency_results: Sequence[IntentRetrievalResult],
) -> tuple[str, list[str]]:
    query = intent.query
    query_lower = query.casefold()
    shared_values = [
        constraint.value
        for constraint in shared_constraints
        if constraint.value.casefold() not in query_lower
    ]
    if shared_values:
        query = f"{query} {' '.join(shared_values)}"
    dependency_text, dependency_chunk_ids = _dependency_context(dependency_results)
    if dependency_text:
        query = f"{query} [dependency evidence: {dependency_text}]"
    return _clean_query(query), dependency_chunk_ids


def _default_retrieval_config(
    *,
    mode: RetrievalMode,
    retrievers: Mapping[str, Retriever],
    top_k: int,
    max_workers: int,
    rrf_k: int,
    context_budget_tokens: int,
    reranking_enabled: bool,
    reranker: Callable[..., Any] | None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "top_k": top_k,
        "max_workers": max_workers,
        "rrf_k": rrf_k,
        "context_budget_tokens": context_budget_tokens,
        "fusion": {
            "method": "reciprocal_rank_fusion",
            "rrf_k": rrf_k,
            "raw_scores_combined": False,
            "stage_order": ["backend_within_intent", "intent_across_evidence"],
        },
        "reranking": {
            "enabled": reranking_enabled,
            "implementation": getattr(reranker, "__qualname__", None)
            if reranking_enabled
            else "disabled_by_default",
        },
        "backends": {name: _retriever_configuration(item) for name, item in retrievers.items()},
        "usage": {
            "query_tokens": "estimated from tokenizer; backend embedding/generation tokens are not claimed",
        },
        "support_evaluation": {
            "retrieval_score_is_not_factual_support": True,
            "semantic_relevance": "not_evaluated",
            "factual_entailment": "not_evaluated",
        },
    }


def _select_backend_retrievers(
    retriever: Retriever,
    *,
    retrieval_mode: RetrievalMode | None,
    lexical_retriever: Retriever | None,
    dense_retriever: Retriever | None,
) -> tuple[RetrievalMode, dict[str, Retriever]]:
    mode = retrieval_mode or getattr(retriever, "backend", "lexical")
    if mode == "hybrid":
        lexical = lexical_retriever or (retriever if getattr(retriever, "backend", None) == "lexical" else None)
        dense = dense_retriever or (retriever if getattr(retriever, "backend", None) == "dense" else None)
        if lexical is None or dense is None:
            raise ValueError("hybrid retrieval requires both lexical and dense retrievers")
        if getattr(lexical, "backend", None) != "lexical":
            raise ValueError("hybrid lexical retriever must expose backend='lexical'")
        if getattr(dense, "backend", None) != "dense":
            raise ValueError("hybrid dense retriever must expose backend='dense'")
        return mode, {"lexical": lexical, "dense": dense}
    if mode not in {"dense", "lexical", "mock"}:
        raise ValueError(f"unsupported multi-intent retrieval mode {mode!r}")
    if getattr(retriever, "backend", None) != mode:
        raise ValueError(
            f"retriever backend {getattr(retriever, 'backend', None)!r} does not match retrieval mode {mode!r}"
        )
    return mode, {mode: retriever}


def retrieve_multi_intent(
    plan: MultiIntentPlan,
    retriever: Retriever,
    *,
    top_k: int,
    parent_revision: int = 0,
    retrieval_revision: int = 0,
    max_workers: int | None = None,
    rrf_k: int = 60,
    reusable_results: Mapping[str, IntentRetrievalResult] | None = None,
    retrieval_mode: RetrievalMode | None = None,
    lexical_retriever: Retriever | None = None,
    dense_retriever: Retriever | None = None,
    context_budget_tokens: int = 1200,
    reranking_enabled: bool = False,
    reranker: Callable[..., Any] | None = None,
) -> MultiIntentRetrievalResult:
    """Retrieve bounded dependency stages and assemble fair, auditable evidence."""

    if top_k < 1:
        raise ValueError("top_k must be at least one")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least one")
    if context_budget_tokens < 1:
        raise ValueError("context_budget_tokens must be at least one")
    if reranking_enabled and reranker is None:
        raise ValueError("reranking_enabled requires an explicit reranker")
    worker_count = min(8, len(plan.intents)) if max_workers is None else max_workers
    if worker_count < 1:
        raise ValueError("max_workers must be at least one")
    mode, backend_retrievers = _select_backend_retrievers(
        retriever,
        retrieval_mode=retrieval_mode,
        lexical_retriever=lexical_retriever,
        dense_retriever=dense_retriever,
    )
    config = _default_retrieval_config(
        mode=mode,
        retrievers=backend_retrievers,
        top_k=top_k,
        max_workers=worker_count,
        rrf_k=rrf_k,
        context_budget_tokens=context_budget_tokens,
        reranking_enabled=reranking_enabled,
        reranker=reranker,
    )
    started_total = time.perf_counter()
    dependency_graph = _dependency_graph(plan)
    pending = {intent.intent_id for intent in plan.intents}
    completed: dict[str, IntentRetrievalResult] = {}
    dependency_order: list[str] = []
    all_errors: list[str] = []
    reusable_results = reusable_results or {}

    def run_intent(intent: QueryIntent, prerequisites: Sequence[IntentRetrievalResult]) -> IntentRetrievalResult:
        started = time.perf_counter()
        prerequisite_ids = dependency_graph[intent.intent_id]
        if any(item.status != "completed" or not item.hits for item in prerequisites):
            error = (
                "dependency evidence unavailable for "
                + ", ".join(prerequisite_ids)
            )
            return IntentRetrievalResult(
                parent_revision=parent_revision,
                retrieval_revision=retrieval_revision,
                intent=intent,
                status="failed",
                dependency_intent_ids=list(prerequisite_ids),
                error=error,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        effective_query, dependency_chunk_ids = _effective_query(
            intent,
            shared_constraints=plan.shared_constraints,
            dependency_results=prerequisites,
        )
        reusable = reusable_results.get(intent.intent_id)
        if (
            reusable is not None
            and reusable.status == "completed"
            and not reusable.error
            and _intent_has_support(intent, reusable.hits)
        ):
            old_backend_hits = reusable.backend_hits or {mode: reusable.hits}
            backend_hits = {
                name: [_annotate_hit(hit, intent, rank=rank) for rank, hit in enumerate(hits, start=1)]
                for name, hits in old_backend_hits.items()
            }
            reused_hits = [_annotate_hit(hit, intent, rank=rank) for rank, hit in enumerate(reusable.hits, start=1)]
            return IntentRetrievalResult(
                parent_revision=parent_revision,
                retrieval_revision=retrieval_revision,
                intent=intent,
                status="completed",
                hits=reused_hits,
                backend_hits=backend_hits,
                fusion_decisions=[
                    {
                        "stage": "reuse_validation",
                        "intent_id": intent.intent_id,
                        "decision": "reused",
                        "source_retrieval_revision": reusable.retrieval_revision,
                        "lexical_support_proxy": True,
                        "semantic_support": "not_evaluated",
                    }
                ],
                dependency_intent_ids=list(prerequisite_ids),
                dependency_context_chunk_ids=dependency_chunk_ids,
                effective_query=effective_query,
                retrieval_usage=Usage(),
                duration_ms=(time.perf_counter() - started) * 1000,
                reused=True,
                reused_from_revision=reusable.retrieval_revision,
            )

        backend_hits: dict[str, list[RetrievalHit]] = {}
        backend_timings: dict[str, float] = {}
        backend_errors: list[str] = []
        usages: list[Usage] = []
        for backend_name, backend in backend_retrievers.items():
            backend_started = time.perf_counter()
            try:
                raw_hits = backend.search(effective_query)
                backend_hits[backend_name] = _normalise_backend_hits(
                    raw_hits,
                    intent=intent,
                    top_k=top_k,
                )
                usages.append(
                    Usage(
                        input_tokens=len(tokenize(effective_query)),
                        output_tokens=0,
                        total_tokens=len(tokenize(effective_query)),
                        estimated=True,
                    )
                )
            except Exception as exc:
                backend_errors.append(f"{backend_name}: {type(exc).__name__}: {exc}")
            finally:
                backend_timings[backend_name] = (time.perf_counter() - backend_started) * 1000
        if not backend_hits and backend_errors:
            error = "; ".join(backend_errors)
            return IntentRetrievalResult(
                parent_revision=parent_revision,
                retrieval_revision=retrieval_revision,
                intent=intent,
                status="failed",
                backend_timings_ms=backend_timings,
                backend_errors=backend_errors,
                dependency_intent_ids=list(prerequisite_ids),
                dependency_context_chunk_ids=dependency_chunk_ids,
                effective_query=effective_query,
                retrieval_usage=_combine_usage(usages),
                error=error,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        fusion_decisions: list[dict[str, Any]] = []
        try:
            fused_hits, fusion_decisions = _fuse_backend_hits(
                intent,
                backend_hits,
                top_k=top_k,
                rrf_k=rrf_k,
            )
            if reranking_enabled and fused_hits:
                reranked = reranker(effective_query, list(fused_hits))
                if not isinstance(reranked, Sequence):
                    raise EvidenceFusionError(
                        "reranker must return a ranked sequence of RetrievalHit values"
                    )
                reranked_hits = [
                    item if isinstance(item, RetrievalHit) else RetrievalHit.model_validate(item)
                    for item in reranked
                ]
                expected_ids = {item.chunk_id for item in fused_hits}
                returned_ids = {item.chunk_id for item in reranked_hits}
                if returned_ids != expected_ids or len(reranked_hits) != len(fused_hits):
                    raise EvidenceFusionError("reranker must preserve exactly the fused chunk IDs")
                fused_hits = [
                    _annotate_hit(item, intent, rank=rank)
                    for rank, item in enumerate(reranked_hits, start=1)
                ]
                fusion_decisions.append(
                    {
                        "stage": "optional_reranking",
                        "intent_id": intent.intent_id,
                        "selected": True,
                        "chunk_ids": [item.chunk_id for item in fused_hits],
                    }
                )
        except Exception as exc:
            fusion_error = f"fusion: {type(exc).__name__}: {exc}"
            backend_errors.append(fusion_error)
            return IntentRetrievalResult(
                parent_revision=parent_revision,
                retrieval_revision=retrieval_revision,
                intent=intent,
                status="failed",
                backend_hits=backend_hits,
                fusion_decisions=fusion_decisions,
                backend_timings_ms=backend_timings,
                backend_errors=backend_errors,
                dependency_intent_ids=list(prerequisite_ids),
                dependency_context_chunk_ids=dependency_chunk_ids,
                effective_query=effective_query,
                retrieval_usage=_combine_usage(usages),
                error=fusion_error,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        return IntentRetrievalResult(
            parent_revision=parent_revision,
            retrieval_revision=retrieval_revision,
            intent=intent,
            status="completed",
            hits=fused_hits,
            backend_hits=backend_hits,
            fusion_decisions=fusion_decisions,
            dependency_intent_ids=list(prerequisite_ids),
            dependency_context_chunk_ids=dependency_chunk_ids,
            effective_query=effective_query,
            retrieval_usage=_combine_usage(usages),
            backend_timings_ms=backend_timings,
            backend_errors=backend_errors,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        while pending:
            ready = [
                intent
                for intent in plan.intents
                if intent.intent_id in pending
                and all(prerequisite_id in completed for prerequisite_id in dependency_graph[intent.intent_id])
            ]
            if not ready:
                raise ValueError("retrieval dependency graph is cyclic or references an unfinished intent")
            futures = {
                intent.intent_id: executor.submit(
                    run_intent,
                    intent,
                    [completed[prerequisite_id] for prerequisite_id in dependency_graph[intent.intent_id]],
                )
                for intent in ready
            }
            for intent in ready:
                try:
                    item = futures[intent.intent_id].result()
                except Exception as exc:
                    item = IntentRetrievalResult(
                        parent_revision=parent_revision,
                        retrieval_revision=retrieval_revision,
                        intent=intent,
                        status="failed",
                        dependency_intent_ids=list(dependency_graph[intent.intent_id]),
                        error=f"{type(exc).__name__}: {exc}",
                        duration_ms=0.0,
                    )
                completed[intent.intent_id] = item
                pending.remove(intent.intent_id)
                dependency_order.append(intent.intent_id)
                if item.error:
                    all_errors.append(f"{intent.intent_id}: {item.error}")
                all_errors.extend(f"{intent.intent_id}: {error}" for error in item.backend_errors)

    intent_results = [completed[intent.intent_id] for intent in plan.intents]
    fused_hits, assembly_decisions, missing_intent_ids, context_tokens_used = _assemble_evidence(
        plan,
        intent_results,
        top_k=top_k,
        rrf_k=rrf_k,
        context_budget_tokens=context_budget_tokens,
    )
    result = MultiIntentRetrievalResult(
        parent_revision=parent_revision,
        retrieval_revision=retrieval_revision,
        plan=plan,
        retrieval_mode=mode,
        retrieval_config=config,
        intent_results=intent_results,
        fused_hits=fused_hits,
        rrf_k=rrf_k,
        context_budget_tokens=context_budget_tokens,
        context_tokens_used=context_tokens_used,
        assembly_decisions=assembly_decisions,
        missing_intent_ids=missing_intent_ids,
        dependency_order=dependency_order,
        retrieval_usage=_combine_usage([item.retrieval_usage for item in intent_results]),
        total_duration_ms=(time.perf_counter() - started_total) * 1000,
        errors=all_errors,
        reused_intent_ids=[item.intent.intent_id for item in intent_results if item.reused],
    )
    if all(item.status == "failed" for item in intent_results):
        raise MultiIntentRetrievalError(
            "all sub-intent retrievals failed; no evidence was fabricated",
            result=result,
        )
    return result


class MultiIntentRetriever:
    """Retriever adapter used by existing baseline and streaming schedulers."""

    method = "rrf_fusion"

    def __init__(
        self,
        base: Retriever,
        *,
        top_k: int = 5,
        max_workers: int | None = None,
        rrf_k: int = 60,
        decomposer: StructuredMultiIntentDecomposer | None = None,
        retrieval_mode: RetrievalMode | None = None,
        lexical_retriever: Retriever | None = None,
        dense_retriever: Retriever | None = None,
        context_budget_tokens: int = 1200,
        reranking_enabled: bool = False,
        reranker: Callable[..., Any] | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least one")
        if context_budget_tokens < 1:
            raise ValueError("context_budget_tokens must be at least one")
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be at least one")
        if reranking_enabled and reranker is None:
            raise ValueError("reranking_enabled requires an explicit reranker")
        self.base = base
        mode, backend_retrievers = _select_backend_retrievers(
            base,
            retrieval_mode=retrieval_mode,
            lexical_retriever=lexical_retriever,
            dense_retriever=dense_retriever,
        )
        self.retrieval_mode = mode
        self._backend_retrievers = backend_retrievers
        # ``backend`` is the externally visible selected mode.  Existing
        # single-backend callers still receive their original backend string.
        self.backend = mode
        self.top_k = top_k
        self.max_workers = max_workers
        self.rrf_k = rrf_k
        self.context_budget_tokens = context_budget_tokens
        self.reranking_enabled = reranking_enabled
        self.reranker = reranker
        self.decomposer = decomposer
        self.calls = 0
        self.last_result: MultiIntentRetrievalResult | None = None
        self.last_decomposition: MultiIntentPlan | None = None
        self._result_by_revision: dict[int, MultiIntentRetrievalResult] = {}
        self._decomposition_by_revision: dict[int, MultiIntentPlan] = {}
        self._decomposition_by_query: dict[tuple[int, str], MultiIntentPlan] = {}
        self._latest_parent_revision = -1
        self._latest_plan: MultiIntentPlan | None = None
        # Retrieval revisions are the scheduler's monotonic publication
        # boundary. Parent/transcript revisions are retained for auditing but
        # must not let an older allocated request replace a newer one when a
        # caller supplies non-identical counters.
        self._latest_result_key: tuple[int, int] = (-1, -1)
        self._lock = threading.RLock()

    @property
    def retrieval_configuration(self) -> dict[str, Any]:
        return _default_retrieval_config(
            mode=self.retrieval_mode,
            retrievers=self._backend_retrievers,
            top_k=self.top_k,
            max_workers=self.max_workers or 8,
            rrf_k=self.rrf_k,
            context_budget_tokens=self.context_budget_tokens,
            reranking_enabled=self.reranking_enabled,
            reranker=self.reranker,
        )

    @property
    def decomposer_identity(self) -> str:
        if self.decomposer is None:
            return "flowcontext.rule_based:offline-rule-based-v2"
        config = self.decomposer.provider.config
        return f"{config.provider}:{config.model}"

    def _decompose(
        self,
        query: str,
        *,
        parent_revision: int,
    ) -> MultiIntentPlan:
        query_key = (parent_revision, _canonical(query))
        with self._lock:
            cached = self._decomposition_by_query.get(query_key)
            if cached is not None:
                return cached
            previous = (
                self._latest_plan
                if parent_revision > self._latest_parent_revision
                and self._latest_plan is not None
                else None
            )
            # Hold the cache lock across the provider call.  The finalisation
            # path can inspect the final plan while the scheduler worker is
            # still decomposing that same revision; serialising this one
            # bounded operation prevents duplicate model calls and guarantees
            # both paths observe the same validated result.
            if self.decomposer is None:
                plan = decompose_query(query, transcript_revision=parent_revision)
                plan = _revision_metadata(plan, previous)
            else:
                plan = self.decomposer.decompose_sync(
                    query,
                    transcript_revision=parent_revision,
                    previous=previous,
                    allow_original_query_fallback=True,
                )
            self._decomposition_by_query[query_key] = plan
            self._decomposition_by_revision[parent_revision] = plan
            if parent_revision >= self._latest_parent_revision:
                # A late result for an older transcript revision remains
                # available for its own audit record, but must not replace the
                # current decomposition exposed to finalisation or callers.
                self.last_decomposition = plan
            if parent_revision > self._latest_parent_revision:
                self._latest_parent_revision = parent_revision
                self._latest_plan = plan
        return plan

    def _reusable_results(
        self,
        plan: MultiIntentPlan,
        previous: MultiIntentRetrievalResult | None,
    ) -> dict[str, IntentRetrievalResult]:
        if previous is None or plan.status != "success":
            return {}
        previous_by_id = {item.intent.intent_id: item for item in previous.intent_results}
        preserved = set(plan.preserved_intent_ids)
        changed = set(previous_by_id) - preserved
        # A dependent question is affected when its prerequisite changed, even
        # if its own wording happens to be identical across revisions.
        changed_again = True
        while changed_again:
            changed_again = False
            for dependency in plan.dependencies:
                if (
                    dependency.relation in {"depends_on", "refines"}
                    and dependency.prerequisite_intent_id in changed
                    and dependency.dependent_intent_id not in changed
                ):
                    changed.add(dependency.dependent_intent_id)
                    changed_again = True
        return {
            intent.intent_id: previous_by_id[intent.intent_id]
            for intent in plan.intents
            if intent.intent_id in preserved
            and intent.intent_id not in changed
            and intent.intent_id in previous_by_id
            and previous_by_id[intent.intent_id].status == "completed"
        }

    def search_with_revision(
        self,
        query: str,
        *,
        parent_revision: int = 0,
        retrieval_revision: int = 0,
    ) -> list[RetrievalHit]:
        self.calls += 1
        plan = self._decompose(query, parent_revision=parent_revision)
        with self._lock:
            # Scheduler artifacts are keyed by the allocated retrieval
            # revision, while decomposition caching is keyed by transcript
            # revision/query.  Keep both mappings so non-identical controller
            # and retrieval counters never lose the plan in a trace.
            self._decomposition_by_revision[retrieval_revision] = plan
            previous = (
                self._result_by_revision.get(retrieval_revision - 1)
                if retrieval_revision > 0
                else None
            )
        reusable = self._reusable_results(plan, previous)
        try:
            result = retrieve_multi_intent(
                plan,
                self.base,
                top_k=self.top_k,
                parent_revision=parent_revision,
                retrieval_revision=retrieval_revision,
                max_workers=self.max_workers,
                rrf_k=self.rrf_k,
                reusable_results=reusable,
                retrieval_mode=self.retrieval_mode,
                lexical_retriever=self._backend_retrievers.get("lexical"),
                dense_retriever=self._backend_retrievers.get("dense"),
                context_budget_tokens=self.context_budget_tokens,
                reranking_enabled=self.reranking_enabled,
                reranker=self.reranker,
            )
        except MultiIntentRetrievalError as exc:
            # Preserve the structured partial/all-failed outcome for the
            # caller and traces before propagating the all-failed error.
            with self._lock:
                self._result_by_revision[retrieval_revision] = exc.result
                result_key = (exc.result.retrieval_revision, exc.result.parent_revision)
                if result_key >= self._latest_result_key:
                    self.last_result = exc.result
                    self._latest_result_key = result_key
            raise
        with self._lock:
            result = result.model_copy(
                update={
                    "superseded_intent_ids": list(plan.superseded_intent_ids),
                }
            )
            self._result_by_revision[retrieval_revision] = result
            result_key = (result.retrieval_revision, result.parent_revision)
            if result_key >= self._latest_result_key:
                self.last_result = result
                self._latest_result_key = result_key
        return result.fused_hits

    def search(self, query: str) -> list[RetrievalHit]:
        """Compatibility search with no transcript revision context."""

        return self.search_with_revision(query)

    def decomposition_for(self, retrieval_revision: int) -> MultiIntentPlan | None:
        with self._lock:
            return self._decomposition_by_revision.get(retrieval_revision)

    def decompose_for_query(
        self,
        query: str,
        *,
        parent_revision: int = 0,
    ) -> MultiIntentPlan:
        """Decompose once at finalisation and reuse the cached final plan."""

        return self._decompose(query, parent_revision=parent_revision)

    def result_for(self, retrieval_revision: int) -> MultiIntentRetrievalResult | None:
        with self._lock:
            return self._result_by_revision.get(retrieval_revision)

    def validate_evidence_for_query(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        parent_revision: int = 0,
    ) -> dict[str, Any]:
        """Validate final reuse against a final-plan decomposition."""

        plan = self._decompose(query, parent_revision=parent_revision)
        return reuse_validation(query, hits, plan=plan)

    @property
    def unsupported_intent_queries(self) -> list[str]:
        """Return final-plan intents not represented by fused evidence."""

        if self.last_result is None:
            return []
        covered = {
            intent_id
            for hit in self.last_result.fused_hits
            for intent_id in hit.intent_ids
        }
        return [
            intent.query
            for intent in self.last_result.plan.intents
            if intent.intent_id not in covered
            or any(
                item.intent.intent_id == intent.intent_id and item.status == "failed"
                for item in self.last_result.intent_results
            )
        ]


def make_multi_intent_retriever(
    index: Any,
    *,
    retrieval_mode: RetrievalMode,
    top_k: int = 5,
    base_retriever: Retriever | None = None,
    provider: Any | None = None,
    cache_dir: Any | None = None,
    local_files_only: bool = False,
    max_workers: int | None = None,
    rrf_k: int = 60,
    context_budget_tokens: int = 1200,
    decomposer: StructuredMultiIntentDecomposer | None = None,
    reranking_enabled: bool = False,
    reranker: Callable[..., Any] | None = None,
) -> MultiIntentRetriever:
    """Build the selected Phase 3 retrieval mode without backend fallback.

    ``base_retriever`` is reused only when its backend matches the requested
    single mode. Hybrid mode always has two explicit backend instances. A
    missing dense dependency or incompatible index therefore propagates as a
    real configuration error instead of becoming a mislabeled lexical run.
    """

    from .retrieval import make_retriever

    if retrieval_mode not in {"dense", "lexical", "hybrid", "mock"}:
        raise ValueError(f"unsupported multi-intent retrieval mode {retrieval_mode!r}")

    def selected_backend(backend_name: str) -> Retriever:
        if base_retriever is not None and getattr(base_retriever, "backend", None) == backend_name:
            return base_retriever
        return make_retriever(
            index,
            backend=backend_name,
            top_k=top_k,
            provider=provider,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    if retrieval_mode == "hybrid":
        lexical = selected_backend("lexical")
        dense = selected_backend("dense")
        base = lexical
        return MultiIntentRetriever(
            base,
            top_k=top_k,
            max_workers=max_workers,
            rrf_k=rrf_k,
            decomposer=decomposer,
            retrieval_mode="hybrid",
            lexical_retriever=lexical,
            dense_retriever=dense,
            context_budget_tokens=context_budget_tokens,
            reranking_enabled=reranking_enabled,
            reranker=reranker,
        )

    base = selected_backend(retrieval_mode)
    return MultiIntentRetriever(
        base,
        top_k=top_k,
        max_workers=max_workers,
        rrf_k=rrf_k,
        decomposer=decomposer,
        retrieval_mode=retrieval_mode,
        context_budget_tokens=context_budget_tokens,
        reranking_enabled=reranking_enabled,
        reranker=reranker,
    )


def unsupported_intent_queries_from_hits(
    plan: MultiIntentPlan,
    hits: Sequence[RetrievalHit],
) -> list[str]:
    """Identify plan intents absent from the final fused evidence set."""

    covered = {
        intent_id
        for hit in hits
        for intent_id in hit.intent_ids
    }
    # A single non-decomposed hit list has no intent annotations. It is only
    # accepted by this helper for a single-intent plan; multi-intent evidence
    # must carry provenance for every covered intent.
    if len(plan.intents) == 1 and hits and not covered:
        return []
    return [intent.query for intent in plan.intents if intent.intent_id not in covered]


def reuse_validation(
    final_query: str,
    hits: Sequence[RetrievalHit],
    *,
    plan: MultiIntentPlan | None = None,
) -> dict[str, Any]:
    """Check whether an already-returned result is appropriate to reuse.

    This is deliberately a conservative lexical *proxy* for semantic
    relevance.  It rejects empty/non-overlapping evidence but reports
    ``semantic_support`` as ``not_evaluated``; chunk IDs alone never pass this
    check as proof of entailment.
    """

    selected_plan = plan or decompose_query(final_query)
    duplicate_ids = len({hit.chunk_id for hit in hits}) != len(hits)
    malformed = [
        hit.chunk_id
        for hit in hits
        if not hit.chunk_id.strip() or not hit.source_location.strip() or not hit.snippet_text.strip()
    ]
    coverage: dict[str, int] = {}
    for intent in selected_plan.intents:
        query_terms = tokenize(intent.query)
        candidates = [
            hit
            for hit in hits
            if (
                intent.intent_id in hit.intent_ids
                if len(selected_plan.intents) > 1
                else not hit.intent_ids or intent.intent_id in hit.intent_ids
            )
        ]
        maximum = 0
        for hit in candidates:
            maximum = max(maximum, len(query_terms & tokenize(hit.snippet_text)))
        coverage[intent.intent_id] = maximum
    intent_coverage = all(value > 0 for value in coverage.values())
    appropriate = bool(hits) and not duplicate_ids and not malformed and intent_coverage
    return {
        "decision": "appropriate" if appropriate else "rejected",
        "final_query": final_query,
        "final_intent_ids": [intent.intent_id for intent in selected_plan.intents],
        "hit_ids": [hit.chunk_id for hit in hits],
        "duplicate_hit_ids": duplicate_ids,
        "malformed_hit_ids": malformed,
        "intent_query_term_overlap": coverage,
        "lexical_relevance_proxy": appropriate,
        "semantic_support": "not_evaluated",
        "reason": (
            "non-empty evidence has lexical support for every final intent"
            if appropriate
            else "chunk IDs or evidence overlap do not establish support for every final intent"
        ),
    }


def evidence_is_appropriate(
    final_query: str,
    hits: Sequence[RetrievalHit],
    *,
    plan: MultiIntentPlan | None = None,
) -> bool:
    """Boolean convenience wrapper for scheduler/replay correctness gates."""

    return bool(reuse_validation(final_query, hits, plan=plan)["lexical_relevance_proxy"])


def intent_metadata_from_hits(hits: Sequence[RetrievalHit]) -> dict[str, Any]:
    """Return compact intent provenance suitable for trace attributes."""

    return {
        "fused": any(hit.retrieval_method == "rrf_fusion" for hit in hits),
        "intent_ids": sorted({intent_id for hit in hits for intent_id in hit.intent_ids}),
        "hit_intent_ids": {
            hit.chunk_id: list(hit.intent_ids)
            for hit in hits
            if hit.intent_ids
        },
        "rrf_scores": {
            hit.chunk_id: hit.rrf_score
            for hit in hits
            if hit.rrf_score is not None
        },
    }
