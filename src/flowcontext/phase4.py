"""Phase 4 session state, follow-up interpretation, and selective updates.

Semantic follow-ups are applied as candidate revisions.  Dependency-aware
invalidation keeps unaffected claim/evidence branches alive, while the
selective-update coordinator sends only unresolved intent queries to the
existing bounded retrieval scheduler.  Answer generation remains an explicit
publication step; this module never turns retrieval provenance into a support
verdict by itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence

from .contracts import (
    Answer,
    CorpusIndex,
    DecompositionConstraint,
    DecompositionIntent,
    DecompositionResult,
    EvidencePassage,
    FactualClaim,
    IntentStatusRecord,
    RetrievalHit,
    Phase4AnswerVersion,
    Phase4ClaimChange,
    Phase4ClaimDependency,
    Phase4ClaimRecord,
    Phase4Clarification,
    Phase4ConstraintRecord,
    Phase4DecisionStep,
    Phase4EvidenceRecord,
    Phase4FollowUpRequest,
    Phase4IntentRecord,
    Phase4PendingRequest,
    Phase4PendingUpdate,
    Phase4ProposedPatch,
    Phase4ReferenceResolution,
    Phase4RevisionRecord,
    Phase4SelectiveRetrievalTask,
    Phase4SelectiveUpdatePlan,
    Phase4SessionState,
    StreamingDecision,
    StreamingRetrievalRequest,
    StreamingRetrievalResult,
    TextSpan,
    Usage,
)
from .multi_intent import (
    StructuredMultiIntentDecomposer,
    decompose_query,
    intent_evidence_alignment,
)


class Phase4Error(RuntimeError):
    """Base error for session-state and follow-up operations."""


class SessionNotFoundError(Phase4Error):
    """The requested session is not held by this process."""


class SessionExistsError(Phase4Error):
    """A session ID is already active in the bounded store."""


class PatchConflictError(Phase4Error):
    """A proposal or application does not match the exact current revision."""


class PatchValidationError(Phase4Error):
    """A patch would mutate state without a validated target or operation."""


_TOKEN_PATTERN = re.compile(r"[\w][\w'-]*", re.UNICODE)
_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*)*\b")
_REFERENCE_PATTERN = re.compile(
    r"\b(?:that|this|the same|same|it|its|there|those|these|their|selected|chosen|the other)"
    r"(?:\s+(?:venue|hall|plan|option|one|location))?\b",
    re.IGNORECASE,
)
_QUESTION_PATTERN = re.compile(
    r"^\s*(?:and\s+|also[,:]?\s+)?"
    r"(?:what|which|where|when|why|how|who|can|could|do|does|is|are|will|would|tell|show|list)\b",
    re.IGNORECASE,
)
_REPLACEMENT_PATTERN = re.compile(
    r"\b(?:actually|instead|correct(?:ion)?|replace|change|switch|make\s+it|use)\b",
    re.IGNORECASE,
)
_REMOVE_PATTERN = re.compile(
    r"\b(?:remove|drop|delete|discard|ignore|no\s+longer|stop\s+requiring)\b",
    re.IGNORECASE,
)
_ADD_PATTERN = re.compile(
    r"\b(?:also|add|include|require|requires|must|with|plus|in addition|make sure|exclude|without|do\s+not|don't)\b",
    re.IGNORECASE,
)
_TOPIC_PATTERN = re.compile(
    r"(?:\bnew\s+topic\b|\bswitch\s+to\b|\bdifferent\s+topic\b|\bchange\s+topic\b|"
    r"\blet(?:'|’)s\s+(?:talk|discuss)\s+about\b|\bnow\s+tell\s+me\s+about\b|"
    r"^\s*(?:instead|alternatively|rather)\b(?=.*\b(?:what|which|where|when|why|how|who|can|could|does|is|are|will|would)\b))",
    re.IGNORECASE,
)
_TOPIC_LEAD_PATTERN = re.compile(
    r"^\s*(?:(?:instead|alternatively|rather)\s*[,;:]?\s*|"
    r"(?:switch|move|change)\s+to\s+|"
    r"(?:new|different)\s+topic\s*[,;:]?\s*|"
    r"change\s+topic\s*(?:about|to)?\s*[,;:]?\s*|"
    r"let(?:'|’)s\s+(?:talk|discuss)\s+about\s+"
    r")",
    re.IGNORECASE,
)
_FORMAT_PATTERN = re.compile(
    r"\b(?:reformat|format|rewrite|put|present|show|give)\b.*\b(?:answer|response|result)\b|"
    r"\b(?:bullets?(?:\s+points?)?|table|concise|brief|shorter|summary|summarize|translate|translation)\b",
    re.IGNORECASE,
)
_GENERIC_ENTITY_WORDS = frozenset(
    {
        "which",
        "what",
        "where",
        "when",
        "why",
        "how",
        "compare",
        "find",
        "show",
        "tell",
        "list",
        "venue",
        "venues",
        "hall",
        "plan",
        "option",
        "options",
    }
)
_TOPIC_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "please",
        "tell",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(*parts: object, length: int = 12) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _clone(model: Any) -> Any:
    """Round-trip a contract so callers never receive store-owned objects."""

    return type(model).model_validate(model.model_dump())


def _canonical(value: str) -> str:
    return " ".join(token.casefold() for token in _TOKEN_PATTERN.findall(value))


def _topic_key(text: str) -> str:
    terms = [
        token.casefold()
        for token in _TOKEN_PATTERN.findall(text)
        if token.casefold() not in _TOPIC_STOP_WORDS
    ]
    return " ".join(terms[:8]) or "unspecified"


def _clean_query(text: str) -> str:
    return " ".join(text.strip().split()).strip(" ,;:")


def _phase4_decomposition(state: Phase4SessionState) -> DecompositionResult:
    """Rebuild a valid synthesis contract from the session's active intents.

    Intent and constraint records retain spans from their origin turns.  A
    follow-up can make those spans non-contiguous in ``active_task``, so the
    publication boundary creates a fresh, local transcript solely for the
    current synthesis request.  It does not rewrite the stored origin spans.
    """

    intent_chunks: list[str] = []
    for record in state.active_intents:
        chunk = record.intent.query.strip()
        missing_values = [
            constraint.value
            for constraint in record.intent.constraints
            if constraint.value.casefold() not in chunk.casefold()
        ]
        if missing_values:
            chunk = f"{chunk}; " + "; ".join(dict.fromkeys(missing_values))
        intent_chunks.append(chunk)
    if not intent_chunks:
        intent_chunks = [state.active_task]
    if state.current_constraints:
        shared_values = [
            record.constraint.value
            for record in state.current_constraints
            if record.scope == "shared"
            and not any(record.constraint.value.casefold() in chunk.casefold() for chunk in intent_chunks)
        ]
        if shared_values:
            intent_chunks[0] = f"{intent_chunks[0]}; " + "; ".join(dict.fromkeys(shared_values))
    original = " ".join(intent_chunks)
    cursor = 0
    intents: list[DecompositionIntent] = []
    for ordinal, record in enumerate(state.active_intents, start=1):
        chunk = intent_chunks[ordinal - 1]
        start = original.find(chunk, cursor)
        if start < 0:
            start = cursor
        end = start + len(chunk)
        cursor = end
        chunk_span = TextSpan(start=start, end=end, text=chunk)
        constraints: list[DecompositionConstraint] = []
        for constraint in record.intent.constraints:
            offset = chunk.casefold().find(constraint.value.casefold())
            if offset < 0:
                # The value was appended above when it was not in the origin
                # query; this fallback keeps the local contract valid even if
                # a provider supplied a differently-cased value.
                source_span = chunk_span
            else:
                source_text = chunk[offset : offset + len(constraint.value)]
                source_span = TextSpan(
                    start=start + offset,
                    end=start + offset + len(source_text),
                    text=source_text,
                )
            constraints.append(constraint.model_copy(update={"source_span": source_span}))
        intents.append(
            record.intent.model_copy(
                update={
                    "ordinal": ordinal,
                    "query": chunk,
                    "source_text": chunk,
                    "source_span": chunk_span,
                    "constraints": constraints,
                }
            )
        )
    shared_constraints: list[DecompositionConstraint] = []
    for constraint in state.current_constraints:
        if constraint.scope != "shared":
            continue
        offset = original.casefold().find(constraint.constraint.value.casefold())
        if offset < 0:
            source_span = TextSpan(start=0, end=len(original), text=original)
        else:
            source_text = original[offset : offset + len(constraint.constraint.value)]
            source_span = TextSpan(start=offset, end=offset + len(source_text), text=source_text)
        shared_constraints.append(
            constraint.constraint.model_copy(update={"source_span": source_span})
        )
    return DecompositionResult(
        transcript_revision=state.transcript_revision,
        original_transcript=original,
        intents=intents,
        boundary_count=max(0, len(intents) - 1),
        shared_constraints=shared_constraints,
        decomposition_method="phase4_session_state",
        status="success",
        provider="flowcontext.session_state",
        provider_model="phase4-publication-v1",
        attempts=1,
    )


def _claim_signature(claim: FactualClaim) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Compare material claim content while ignoring record/version identity."""

    return (
        _canonical(claim.claim_text),
        tuple(claim.supporting_chunk_ids),
        tuple(sorted(claim.intent_ids)),
    )


def _claim_display_text(claim: FactualClaim) -> str:
    citations = " ".join(f"[{chunk_id}]" for chunk_id in claim.supporting_chunk_ids)
    return f"{claim.claim_text} {citations}".strip()


def _render_presentation(
    answer: Answer,
    claims: Sequence[FactualClaim],
    instruction: str,
) -> str:
    """Render only existing claim text, citations, and uncertainty."""

    if not claims:
        raise PatchValidationError("presentation requires at least one current factual claim")
    displays = [_claim_display_text(claim) for claim in claims]
    lowered = instruction.casefold()
    if lowered.startswith("bullets"):
        count_match = re.search(r":(\d+)$", lowered)
        requested_count = int(count_match.group(1)) if count_match else len(displays)
        requested_count = max(1, min(requested_count, len(displays)))
        groups: list[list[str]] = [[] for _ in range(requested_count)]
        for index, display in enumerate(displays):
            groups[min(requested_count - 1, index * requested_count // len(displays))].append(display)
        rendered = "\n".join(f"- {' '.join(group)}" for group in groups if group)
    elif lowered == "table":
        rendered = "| Answer | Sources |\n|---|---|\n" + "\n".join(
            f"| {claim.claim_text} | {', '.join(f'[{chunk_id}]' for chunk_id in claim.supporting_chunk_ids)} |"
            for claim in claims
        )
    elif lowered == "concise":
        rendered = " ".join(displays)
    elif lowered in {"requested_format", "translate"}:
        rendered = answer.answer_text
    else:
        rendered = answer.answer_text
    if answer.uncertainty.strip():
        rendered += f"\n\nUncertainty: {answer.uncertainty.strip()}"
    return rendered


@dataclass(frozen=True)
class Phase4PublicationResult:
    """Outcome of a revision-bound factual or presentation publication."""

    state: Phase4SessionState
    published: bool
    status: str
    request_id: str | None = None
    outcome: Any | None = None
    reason: str | None = None


def _entity_mentions(text: str, *, excluded_terms: set[str] | None = None) -> list[str]:
    excluded = {item.casefold() for item in (excluded_terms or set())}
    result: list[str] = []
    for match in _ENTITY_PATTERN.finditer(text):
        value = match.group(0).strip(" ,.;:!?'")
        words = value.casefold().split()
        while words and words[0] in {"which", "what", "where", "when", "why", "how", "compare", "find", "show", "tell", "list"}:
            words.pop(0)
        if not words:
            continue
        value = " ".join(value.split()[-len(words) :])
        if not value or value.casefold() in _GENERIC_ENTITY_WORDS:
            continue
        if len(words) == 1 and (words[0] in _GENERIC_ENTITY_WORDS or words[0] in excluded):
            continue
        if value.casefold() not in {item.casefold() for item in result}:
            result.append(value)
    return result


def _span_for(text: str, value: str) -> TextSpan:
    start = text.casefold().find(value.casefold())
    if start < 0:
        start = 0
        value = text
    return TextSpan(start=start, end=start + len(value), text=text[start : start + len(value)])


def _constraint_record_id(
    constraint: DecompositionConstraint,
    *,
    origin_turn: int,
    origin_revision: int,
    scope: str,
    intent_id: str | None,
) -> str:
    return "constraint-record-" + _digest(
        constraint.constraint_id,
        constraint.kind,
        constraint.value,
        origin_turn,
        origin_revision,
        scope,
        intent_id or "shared",
    )


def _intent_id_for_revision(
    intent: DecompositionIntent,
    constraints: Sequence[DecompositionConstraint],
    shared_constraints: Sequence[DecompositionConstraint] = (),
) -> str:
    content = "|".join(
        [
            _canonical(intent.query),
            intent.relationship,
            *sorted(f"{item.kind}:{_canonical(item.value)}" for item in constraints),
            *sorted(f"shared:{item.kind}:{_canonical(item.value)}" for item in shared_constraints),
        ]
    )
    return "intent-" + _digest(content, length=10)


def _make_constraint_record(
    constraint: DecompositionConstraint,
    *,
    origin: str,
    origin_turn: int,
    origin_utterance_id: str,
    origin_revision: int,
    scope: str,
    intent_id: str | None,
    resolution_source: str | None = None,
) -> Phase4ConstraintRecord:
    return Phase4ConstraintRecord(
        record_id=_constraint_record_id(
            constraint,
            origin_turn=origin_turn,
            origin_revision=origin_revision,
            scope=scope,
            intent_id=intent_id,
        ),
        constraint=constraint,
        constraint_id=constraint.constraint_id,
        origin=origin,
        origin_turn=origin_turn,
        origin_utterance_id=origin_utterance_id,
        origin_revision=origin_revision,
        scope=scope,
        intent_id=intent_id,
        resolution_source=resolution_source,
    )


def _rebind_constraint_record(
    record: Phase4ConstraintRecord,
    *,
    scope: str,
    intent_id: str | None,
) -> Phase4ConstraintRecord:
    """Keep a replacement in the scope of the record it supersedes."""

    return record.model_copy(
        update={
            "record_id": _constraint_record_id(
                record.constraint,
                origin_turn=record.origin_turn,
                origin_revision=record.origin_revision,
                scope=scope,
                intent_id=intent_id,
            ),
            "scope": scope,
            "intent_id": intent_id,
        }
    )


def _make_intent_record(
    intent: DecompositionIntent,
    *,
    origin_turn: int,
    origin_utterance_id: str,
    origin_revision: int,
    topic_key: str,
) -> Phase4IntentRecord:
    return Phase4IntentRecord(
        intent=intent,
        origin_turn=origin_turn,
        origin_utterance_id=origin_utterance_id,
        origin_revision=origin_revision,
        topic_key=topic_key,
    )


def _generic_constraint(text: str) -> DecompositionConstraint | None:
    """Retain a non-typed requirement without inventing a corpus fact."""

    match = re.search(
        r"\b(?:also\s+)?(?:add|include|require|requires|must\s+have|make\s+sure\s+there\s+is)\s+"
        r"(?P<value>.+?)(?:[.!?])?$",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.search(r"\bwith\s+(?P<value>.+?)(?:[.!?])?$", text, flags=re.IGNORECASE)
    if match is None:
        return None
    value = match.group("value").strip()
    if not value:
        return None
    start = match.start("value")
    end = start + len(value)
    span = TextSpan(start=start, end=end, text=text[start:end])
    return DecompositionConstraint(
        constraint_id="constraint-" + _digest("other", _canonical(value), start, end, length=10),
        kind="other",
        value=value,
        source_span=span,
        normalized_value=_canonical(value),
    )


class Phase4FollowUpInterpreter:
    """Classify a follow-up and build a non-mutating revision-bound patch.

    The default path is explicitly rule-based.  When a configured
    :class:`StructuredMultiIntentDecomposer` is supplied, it is used only for
    model-backed intent/constraint materialization.  Its existing provider
    configuration determines whether the trace says ``mock_provider`` or
    ``real_provider``; no real-provider failure is relabelled as a mock.
    """

    def __init__(self, decomposer: StructuredMultiIntentDecomposer | Any | None = None) -> None:
        if decomposer is not None and not hasattr(decomposer, "decompose_sync"):
            # Accept the existing DecompositionProvider protocol directly as
            # well as its retry/validation wrapper for ergonomic integration.
            decomposer = StructuredMultiIntentDecomposer(decomposer)
        self.decomposer = decomposer

    @property
    def execution_mode(self) -> str:
        if self.decomposer is None:
            return "rule_based"
        return "mock_provider" if self.decomposer.provider.config.backend == "mock" else "real_provider"

    @property
    def provider_identity(self) -> tuple[str, str]:
        if self.decomposer is None:
            return "flowcontext.rule_based", "offline-follow-up-v1"
        config = self.decomposer.provider.config
        return config.provider, config.model

    def _plan(self, text: str, revision: int) -> DecompositionResult:
        if self.decomposer is None:
            return decompose_query(text, transcript_revision=revision, max_intents=6)
        # The existing structured decomposer owns timeout, validation, retry,
        # and provider identity.  There is intentionally no local mock
        # substitution if a real provider fails.
        return self.decomposer.decompose_sync(text, transcript_revision=revision)

    def interpret(
        self,
        state: Phase4SessionState,
        request: Phase4FollowUpRequest,
    ) -> Phase4ProposedPatch:
        if request.session_id != state.session_id:
            raise PatchConflictError("follow-up session does not match session state")
        if request.base_revision != state.state_revision:
            raise PatchConflictError(
                f"follow-up targets revision {request.base_revision}, current revision is {state.state_revision}"
            )
        if request.transcript_revision is not None and request.transcript_revision <= state.transcript_revision:
            raise PatchConflictError("follow-up transcript_revision must advance the session transcript")

        text = _clean_query(request.text)
        if not text:
            raise PatchValidationError("follow-up text must not be blank")
        proposed_revision = state.state_revision + 1
        transcript_revision = request.transcript_revision or state.transcript_revision + 1
        request = request.model_copy(update={"transcript_revision": transcript_revision})
        provider_identity, provider_model = self.provider_identity
        trace: list[Phase4DecisionStep] = []

        def base_patch(**updates: Any) -> Phase4ProposedPatch:
            semantic = updates.get("classification") not in {"reformat_answer", "clarification_required"}
            effective_transcript_revision = transcript_revision if semantic else state.transcript_revision
            return Phase4ProposedPatch(
                patch_id="patch-" + _digest(state.session_id, request.utterance_id, request.base_revision, text),
                session_id=state.session_id,
                utterance_id=request.utterance_id,
                turn_index=request.turn_index,
                source_text=text,
                base_revision=request.base_revision,
                proposed_revision=proposed_revision,
                transcript_revision=effective_transcript_revision,
                retrieval_revision=state.retrieval_revision + (1 if semantic else 0),
                execution_mode=self.execution_mode,
                provider_identity=provider_identity,
                provider_model=provider_model,
                decision_trace=trace,
                **updates,
            )

        # Formatting is deliberately checked before ordinary constraint
        # parsing, but a formatting marker does not swallow a second factual
        # question.  A combined turn is left for the user to disambiguate so
        # neither the presentation request nor the new information need is
        # silently dropped.
        if _FORMAT_PATTERN.search(text):
            if self._contains_factual_question(text):
                trace.append(
                    Phase4DecisionStep(
                        stage="classification",
                        rule="combined_presentation_and_question",
                        detail=(
                            "The turn contains both a presentation request and a new information need; "
                            "no formatting-only patch is proposed."
                        ),
                        matched_text=text,
                    )
                )
                clarification = self._clarification(
                    state,
                    reason_code="ambiguous_change",
                    question=(
                        "This combines a presentation change with a new factual question. "
                        "Which new question should I answer before formatting the answer?"
                    ),
                )
                return base_patch(
                    classification="clarification_required",
                    clarification=clarification,
                )
            instruction = self._format_instruction(text)
            trace.append(
                Phase4DecisionStep(
                    stage="classification",
                    rule="format_request",
                    detail="The turn requests a presentation change and names no semantic corpus change.",
                    matched_text=text,
                )
            )
            return base_patch(classification="reformat_answer", format_instruction=instruction)

        # An explicit topic switch takes precedence over pronoun resolution;
        # the new topic must not inherit old constraints.
        if _TOPIC_PATTERN.search(text):
            trace.append(
                Phase4DecisionStep(
                    stage="classification",
                    rule="explicit_topic_switch",
                    detail="A topic-switch marker starts a fresh active task.",
                    matched_text=_TOPIC_PATTERN.search(text).group(0),
                    target_ids=[item.intent_id for item in state.active_intents],
                )
            )
            # Topic markers are discourse instructions, not answer-bearing
            # corpus terms.  Strip only a recognized leading marker before
            # decomposition so lexical qualification cannot reject an
            # otherwise supported new topic because it lacks words such as
            # "instead" or "switch".
            topic_text = _clean_query(_TOPIC_LEAD_PATTERN.sub("", text)) or text
            plan = self._plan(topic_text, transcript_revision)
            new_intents, added = self._materialize_plan(
                plan,
                state=state,
                request=request,
                topic_key=_topic_key(topic_text),
            )
            return base_patch(
                classification="change_topic",
                target_intent_ids=[item.intent_id for item in state.active_intents],
                new_intents=new_intents,
                added_constraints=added,
                topic_key=_topic_key(topic_text),
            )

        resolutions, reference_error = self._resolve_references(state, text, trace)
        if reference_error is not None:
            return base_patch(
                classification="clarification_required",
                clarification=reference_error,
            )

        # Corrections/replacements are checked before additions.  "Actually,
        # make it 40 attendees" must not leave the old quantity active.
        if _REPLACEMENT_PATTERN.search(text):
            return self._replacement_patch(
                state,
                request,
                text,
                trace,
                resolutions,
                base_patch,
                transcript_revision,
            )

        if _REMOVE_PATTERN.search(text):
            return self._removal_patch(
                state,
                request,
                text,
                trace,
                resolutions,
                base_patch,
                transcript_revision,
            )

        if _QUESTION_PATTERN.search(text) or text.endswith("?"):
            return self._question_patch(
                state,
                request,
                text,
                trace,
                resolutions,
                base_patch,
                transcript_revision,
            )

        if _ADD_PATTERN.search(text) or self._looks_like_constraint_fragment(text):
            return self._addition_patch(
                state,
                request,
                text,
                trace,
                resolutions,
                base_patch,
                transcript_revision,
            )

        trace.append(
            Phase4DecisionStep(
                stage="classification",
                rule="ambiguous_change",
                detail="The turn does not identify a safe semantic operation.",
                matched_text=text,
            )
        )
        clarification = self._clarification(
            state,
            reason_code="ambiguous_change",
            question="Should I add a constraint, change an existing constraint, ask another question, or change topic?",
        )
        return base_patch(classification="clarification_required", clarification=clarification)

    def _format_instruction(self, text: str) -> str:
        lowered = text.casefold()
        if "table" in lowered:
            return "table"
        if "bullet" in lowered:
            count = re.search(r"\b(\d+)\s+bullets?\b", lowered)
            if count is None:
                number_words = {
                    "one": "1",
                    "two": "2",
                    "three": "3",
                    "four": "4",
                    "five": "5",
                    "six": "6",
                    "seven": "7",
                    "eight": "8",
                }
                word_count = re.search(
                    r"\b(one|two|three|four|five|six|seven|eight)\s+bullets?\b",
                    lowered,
                )
                if word_count is not None:
                    return f"bullets:{number_words[word_count.group(1)]}"
            return f"bullets:{count.group(1)}" if count else "bullets"
        if "translat" in lowered:
            language = re.search(
                r"\b(?:to|into)\s+([a-z][a-z -]{1,30})\b",
                lowered,
            )
            return f"translate:{language.group(1).strip()}" if language else "translate"
        if any(word in lowered for word in ("concise", "brief", "shorter", "summary", "summarize")):
            return "concise"
        return "requested_format"

    @staticmethod
    def _contains_factual_question(text: str) -> bool:
        """Detect a second information need without treating format words as one."""

        interrogative = re.compile(
            r"\b(?:what|which|where|when|why|how|who|can|could|does|is|are|will|would)\b",
            re.IGNORECASE,
        )
        if not interrogative.search(text):
            return False
        if "?" in text:
            return True
        return bool(
            re.search(
                r"\b(?:and|also|plus|then|as\s+well\s+as)\s+"
                r"(?:what|which|where|when|why|how|who|can|could|does|is|are|will|would)\b",
                text,
                re.IGNORECASE,
            )
        )

    def _looks_like_constraint_fragment(self, text: str) -> bool:
        return bool(
            re.match(
                r"^\s*(?:in|at|near|on|before|after|by|for|with|without|under|over|between)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _context_entities(
        self,
        state: Phase4SessionState,
    ) -> list[tuple[str, str, str]]:
        location_terms = {
            token
            for record in state.current_constraints
            if record.constraint.kind == "location"
            for token in _TOKEN_PATTERN.findall(record.constraint.value.casefold())
        }
        candidates: dict[str, tuple[str, str]] = {}
        for record in state.current_constraints:
            if record.constraint.kind == "entity":
                value = record.constraint.value
                candidates.setdefault(_canonical(value), (value, record.intent_id or ""))
        for intent_record in state.active_intents:
            for value in _entity_mentions(intent_record.intent.query):
                if _canonical(value) in location_terms:
                    continue
                candidates.setdefault(_canonical(value), (value, intent_record.intent_id))
        for claim_record in state.current_claims:
            for dependency in claim_record.dependencies:
                if dependency.dependency_type == "entity" and dependency.target_value:
                    candidates.setdefault(
                        _canonical(dependency.target_value),
                        (dependency.target_value, claim_record.intent_ids[0] if claim_record.intent_ids else ""),
                    )
        return [
            ("entity-" + _digest(key, length=10), value, intent_id)
            for key, (value, intent_id) in candidates.items()
            if value.casefold() not in _GENERIC_ENTITY_WORDS
        ]

    def _resolve_references(
        self,
        state: Phase4SessionState,
        text: str,
        trace: list[Phase4DecisionStep],
    ) -> tuple[list[Phase4ReferenceResolution], Phase4Clarification | None]:
        matches = list(_REFERENCE_PATTERN.finditer(text))
        if not matches:
            return [], None
        entities = self._context_entities(state)
        unique_entities = {entity_id: (value, intent_id) for entity_id, value, intent_id in entities}
        if len(unique_entities) == 1:
            entity_id, (entity_text, source_intent_id) = next(iter(unique_entities.items()))
            resolutions = [
                Phase4ReferenceResolution(
                    reference_text=match.group(0),
                    entity_id=entity_id,
                    entity_text=entity_text,
                    source_intent_id=source_intent_id or state.active_intents[0].intent_id,
                    rationale="Exactly one active entity candidate is available in this session.",
                )
                for match in matches
            ]
            trace.append(
                Phase4DecisionStep(
                    stage="reference_resolution",
                    rule="single_entity_context",
                    detail="Context resolved the reference without choosing between multiple entities.",
                    matched_text=matches[0].group(0),
                    target_ids=[entity_id],
                )
            )
            return resolutions, None
        if unique_entities:
            labels = [value for value, _intent_id in unique_entities.values()]
            question = (
                f"Which entity do you mean by {matches[0].group(0)!r}: "
                + ", ".join(labels)
                + "?"
            )
            trace.append(
                Phase4DecisionStep(
                    stage="reference_resolution",
                    rule="multiple_entity_candidates",
                    detail="The reference has multiple active entity candidates; no state mutation is proposed.",
                    matched_text=matches[0].group(0),
                    target_ids=list(unique_entities),
                )
            )
            return [], self._clarification(
                state,
                reason_code="ambiguous_reference",
                question=question,
                candidate_entity_ids=list(unique_entities),
            )
        question = f"Which entity do you mean by {matches[0].group(0)!r}?"
        trace.append(
            Phase4DecisionStep(
                stage="reference_resolution",
                rule="missing_entity_context",
                detail="The reference has no entity antecedent in the active session state.",
                matched_text=matches[0].group(0),
            )
        )
        return [], self._clarification(
            state,
            reason_code="missing_entity",
            question=question,
        )

    def _clarification(
        self,
        state: Phase4SessionState,
        *,
        reason_code: str,
        question: str,
        candidate_entity_ids: Sequence[str] = (),
    ) -> Phase4Clarification:
        return Phase4Clarification(
            clarification_id="clarification-" + _digest(state.session_id, state.state_revision, question),
            question=question,
            reason_code=reason_code,
            candidate_entity_ids=list(candidate_entity_ids),
            related_intent_ids=[item.intent_id for item in state.active_intents],
            created_revision=state.state_revision + 1,
        )

    def _target_intents(
        self,
        state: Phase4SessionState,
        text: str,
        resolutions: Sequence[Phase4ReferenceResolution],
        trace: list[Phase4DecisionStep],
    ) -> tuple[list[str], Phase4Clarification | None]:
        if resolutions:
            target = [resolutions[0].source_intent_id]
        elif len(state.active_intents) == 1:
            target = [state.active_intents[0].intent_id]
        elif re.search(r"\b(?:all|both|each|every)\b", text, flags=re.IGNORECASE):
            target = [item.intent_id for item in state.active_intents]
        else:
            return [], self._clarification(
                state,
                reason_code="ambiguous_change",
                question="Which existing question should this change apply to?",
                candidate_entity_ids=[item.intent_id for item in state.active_intents],
            )
        trace.append(
            Phase4DecisionStep(
                stage="target_selection",
                rule="follow_up_target_selection",
                detail="Selected the existing intent(s) that the proposed change addresses.",
                target_ids=target,
            )
        )
        return target, None

    def _plan_constraints(
        self,
        plan: DecompositionResult,
        *,
        request: Phase4FollowUpRequest,
        text: str,
        intent_id: str,
        include_generic: bool = False,
    ) -> list[Phase4ConstraintRecord]:
        records: list[Phase4ConstraintRecord] = []
        for constraint in plan.shared_constraints:
            records.append(
                _make_constraint_record(
                    constraint,
                    origin="user_follow_up",
                    origin_turn=request.turn_index,
                    origin_utterance_id=request.utterance_id,
                    origin_revision=request.transcript_revision or request.base_revision + 1,
                    scope="shared",
                    intent_id=None,
                )
            )
        for intent in plan.intents:
            if intent.intent_id != intent_id and len(plan.intents) > 1:
                continue
            for constraint in intent.constraints:
                records.append(
                    _make_constraint_record(
                        # One follow-up constraint can be applied to several
                        # existing intents.  Give each intent-local copy a
                        # distinct identity so the candidate decomposition
                        # remains valid and later invalidation can address
                        # the branches independently.
                        constraint.model_copy(
                            update={
                                "constraint_id": "constraint-" + _digest(
                                    constraint.constraint_id,
                                    intent_id,
                                    constraint.kind,
                                    _canonical(constraint.value),
                                )
                            }
                        ),
                        origin="user_follow_up",
                        origin_turn=request.turn_index,
                        origin_utterance_id=request.utterance_id,
                        origin_revision=request.transcript_revision or request.base_revision + 1,
                        scope="intent",
                        intent_id=intent_id,
                    )
                )
        if include_generic and not any(record.constraint.kind == "other" for record in records):
            generic = _generic_constraint(text)
            if generic is not None:
                records.append(
                    _make_constraint_record(
                        generic,
                        origin="user_follow_up",
                        origin_turn=request.turn_index,
                        origin_utterance_id=request.utterance_id,
                        origin_revision=request.transcript_revision or request.base_revision + 1,
                        scope="intent",
                        intent_id=intent_id,
                    )
                )
        return records

    def _materialize_plan(
        self,
        plan: DecompositionResult,
        *,
        state: Phase4SessionState,
        request: Phase4FollowUpRequest,
        topic_key: str,
    ) -> tuple[list[Phase4IntentRecord], list[Phase4ConstraintRecord]]:
        intent_records: list[Phase4IntentRecord] = []
        constraints: list[Phase4ConstraintRecord] = []
        for intent in plan.intents:
            intent_records.append(
                _make_intent_record(
                    intent,
                    origin_turn=request.turn_index,
                    origin_utterance_id=request.utterance_id,
                    origin_revision=request.transcript_revision or request.base_revision + 1,
                    topic_key=topic_key,
                )
            )
            constraints.extend(
                self._plan_constraints(
                    plan,
                    request=request,
                    text=plan.original_transcript,
                    intent_id=intent.intent_id,
                )
            )
        # The shared constraints were included once per plan intent above;
        # de-duplicate by record ID while preserving first occurrence.
        constraints = list({record.record_id: record for record in constraints}.values())
        return intent_records, constraints

    def _question_patch(
        self,
        state: Phase4SessionState,
        request: Phase4FollowUpRequest,
        text: str,
        trace: list[Phase4DecisionStep],
        resolutions: Sequence[Phase4ReferenceResolution],
        base_patch: Any,
        transcript_revision: int,
    ) -> Phase4ProposedPatch:
        plan = self._plan(text, transcript_revision)
        if not plan.intents:
            raise PatchValidationError("follow-up question produced no intent")
        active_ids = {item.intent_id for item in state.active_intents}
        new_records: list[Phase4IntentRecord] = []
        constraint_records: dict[str, Phase4ConstraintRecord] = {}
        for plan_intent in plan.intents:
            intent = plan_intent
            inherited: list[Phase4ConstraintRecord] = []
            if resolutions:
                resolution = resolutions[0]
                rewritten = re.sub(
                    re.escape(resolution.reference_text),
                    resolution.entity_text,
                    intent.query,
                    flags=re.IGNORECASE,
                )
                span = _span_for(text, resolution.reference_text)
                inherited_constraint = DecompositionConstraint(
                    constraint_id="constraint-" + _digest(
                        "inherited-entity",
                        resolution.entity_id,
                        intent.intent_id,
                        span.text,
                    ),
                    kind="entity",
                    value=resolution.entity_text,
                    source_span=span,
                    normalized_value=_canonical(resolution.entity_text),
                )
                intent = intent.model_copy(
                    update={
                        "query": rewritten,
                        "constraints": [*intent.constraints, inherited_constraint],
                    }
                )
                inherited.append(
                    _make_constraint_record(
                        inherited_constraint,
                        origin="inherited_context",
                        origin_turn=request.turn_index,
                        origin_utterance_id=request.utterance_id,
                        origin_revision=request.transcript_revision or request.base_revision + 1,
                        scope="intent",
                        intent_id=intent.intent_id,
                        resolution_source=resolution.entity_id,
                    )
                )
            if intent.intent_id in active_ids:
                intent = intent.model_copy(
                    update={
                        "intent_id": "intent-" + _digest(
                            intent.intent_id,
                            request.turn_index,
                            request.utterance_id,
                        )
                    }
                )
                inherited = [
                    item.model_copy(
                        update={
                            "record_id": item.record_id + "@" + intent.intent_id,
                            "intent_id": intent.intent_id,
                        }
                    )
                    for item in inherited
                ]
            record = _make_intent_record(
                intent,
                origin_turn=request.turn_index,
                origin_utterance_id=request.utterance_id,
                origin_revision=request.transcript_revision or request.base_revision + 1,
                topic_key=state.active_topic,
            )
            new_records.append(record)
            for constraint in self._plan_constraints(
                plan,
                request=request,
                text=text,
                intent_id=intent.intent_id,
            ):
                constraint_records[constraint.record_id] = constraint
            for constraint in inherited:
                constraint_records[constraint.record_id] = constraint
            if resolutions:
                trace.append(
                    Phase4DecisionStep(
                        stage="materialization",
                        rule="contextual_entity_inheritance",
                        detail="The resolved entity is carried as context, not as corpus verification.",
                        matched_text=resolutions[0].reference_text,
                        target_ids=[resolutions[0].entity_id, intent.intent_id],
                    )
                )
        trace.append(
            Phase4DecisionStep(
                stage="classification",
                rule="new_question",
                detail="The turn has interrogative/request boundaries; each independent question is retained.",
                matched_text=text,
                target_ids=[record.intent_id for record in new_records],
            )
        )
        return base_patch(
            classification="add_question",
            new_intents=new_records,
            added_constraints=list(constraint_records.values()),
            reference_resolutions=list(resolutions),
        )

    def _addition_patch(
        self,
        state: Phase4SessionState,
        request: Phase4FollowUpRequest,
        text: str,
        trace: list[Phase4DecisionStep],
        resolutions: Sequence[Phase4ReferenceResolution],
        base_patch: Any,
        transcript_revision: int,
    ) -> Phase4ProposedPatch:
        targets, clarification = self._target_intents(state, text, resolutions, trace)
        if clarification is not None:
            return base_patch(classification="clarification_required", clarification=clarification)
        plan = self._plan(text, transcript_revision)
        records: list[Phase4ConstraintRecord] = []
        # "all/both/each" denotes a shared constraint; otherwise a single
        # target receives an intent-local constraint.
        # ``shared`` is a global scope in the state contract.  When a user
        # names only a subset of several intents, keep one intent-local
        # record per target rather than silently applying a global constraint
        # to unrelated intents.
        shared = len(targets) > 1 and len(targets) == len(state.active_intents)
        for target in targets:
            records.extend(
                self._plan_constraints(
                    plan,
                    request=request,
                    text=text,
                    intent_id=target,
                    include_generic=True,
                )
            )
        if shared:
            shared_records: dict[tuple[str, str], Phase4ConstraintRecord] = {}
            for record in records:
                key = (record.constraint.kind, _canonical(record.constraint.value))
                shared_record = record.model_copy(
                    update={
                        "record_id": _constraint_record_id(
                            record.constraint,
                            origin_turn=request.turn_index,
                            origin_revision=request.transcript_revision or request.base_revision + 1,
                            scope="shared",
                            intent_id=None,
                        ),
                        "scope": "shared",
                        "intent_id": None,
                    }
                )
                shared_records.setdefault(key, shared_record)
            records = list(shared_records.values())
        else:
            records = list({record.record_id: record for record in records}.values())
        if not records:
            clarification = self._clarification(
                state,
                reason_code="ambiguous_change",
                question="What exact constraint should I add? Please include its value, unit, date, or entity.",
            )
            return base_patch(classification="clarification_required", clarification=clarification)
        replacements, new_intents = self._revision_intents(
            state,
            request=request,
            text=text,
            target_ids=targets,
            additions=records,
            replacements={},
            removed_ids=set(),
        )
        trace.append(
            Phase4DecisionStep(
                stage="classification",
                rule="add_constraint",
                detail="Typed values and generic requirements are retained without verifying them against the corpus.",
                matched_text=text,
                target_ids=targets,
            )
        )
        return base_patch(
            classification="add_constraint",
            target_intent_ids=targets,
            intent_replacements=replacements,
            added_constraints=records,
            new_intents=new_intents,
            reference_resolutions=list(resolutions),
        )

    def _replacement_patch(
        self,
        state: Phase4SessionState,
        request: Phase4FollowUpRequest,
        text: str,
        trace: list[Phase4DecisionStep],
        resolutions: Sequence[Phase4ReferenceResolution],
        base_patch: Any,
        transcript_revision: int,
    ) -> Phase4ProposedPatch:
        targets, clarification = self._target_intents(state, text, resolutions, trace)
        if clarification is not None:
            return base_patch(classification="clarification_required", clarification=clarification)
        plan = self._plan(text, transcript_revision)
        # An entity correction expressed through a reference (for example,
        # "use Venue A for that venue") can target one intent syntactically
        # while several active questions are about the same selected entity.
        # Propagate the correction to those explicitly matching entity
        # constraints so policy, price, availability, and capacity branches
        # do not continue to ask about the old entity.  Unrelated intents are
        # left untouched; an absent or non-unique antecedent was already
        # handled by reference clarification above.
        incoming_entity_constraints = [
            constraint
            for plan_intent in plan.intents
            for constraint in plan_intent.constraints
            if constraint.kind == "entity"
        ]
        if incoming_entity_constraints:
            old_entity_values = {
                _canonical(record.constraint.value)
                for record in state.current_constraints
                if record.constraint.kind == "entity"
                and (record.scope == "shared" or record.intent_id in targets)
            }
            propagated_targets = [
                intent_record.intent_id
                for intent_record in state.active_intents
                if intent_record.intent_id not in targets
                and any(
                    _canonical(record.constraint.value) in old_entity_values
                    for record in state.current_constraints
                    if record.constraint.kind == "entity"
                    and (record.scope == "shared" or record.intent_id == intent_record.intent_id)
                )
            ]
            if propagated_targets:
                targets = list(dict.fromkeys([*targets, *propagated_targets]))
                trace.append(
                    Phase4DecisionStep(
                        stage="dependency_propagation",
                        rule="same_entity_constraint_propagation",
                        detail=(
                            "The entity correction is applied to active intents that explicitly "
                            "depend on the same selected entity."
                        ),
                        matched_text=text,
                        target_ids=propagated_targets,
                    )
                )
        incoming: list[Phase4ConstraintRecord] = []
        for target in targets:
            incoming.extend(
                self._plan_constraints(
                    plan,
                    request=request,
                    text=text,
                    intent_id=target,
                    include_generic=False,
                )
            )
        if not incoming:
            clarification = self._clarification(
                state,
                reason_code="ambiguous_change",
                question="What value should replace the existing constraint? Please include the new value and unit.",
            )
            return base_patch(classification="clarification_required", clarification=clarification)
        active = {record.record_id: record for record in state.current_constraints}
        replacement_map: dict[str, Phase4ConstraintRecord] = {}
        for target in targets:
            candidates = [
                record
                for record in active.values()
                if record.scope == "shared" or record.intent_id == target
            ]
            for new_record in [
                item
                for item in incoming
                if item.scope == "shared" or item.intent_id == target
            ]:
                same_kind = [item for item in candidates if item.constraint.kind == new_record.constraint.kind]
                if len(same_kind) == 1:
                    old_record = same_kind[0]
                    replacement_map[old_record.record_id] = _rebind_constraint_record(
                        new_record,
                        scope=old_record.scope,
                        intent_id=old_record.intent_id,
                    )
                elif len(same_kind) == 0 and len(candidates) == 1:
                    old_record = candidates[0]
                    replacement_map[old_record.record_id] = _rebind_constraint_record(
                        new_record,
                        scope=old_record.scope,
                        intent_id=old_record.intent_id,
                    )
                else:
                    options = ", ".join(item.constraint.value for item in candidates) or "the existing constraints"
                    clarification = self._clarification(
                        state,
                        reason_code="ambiguous_change",
                        question=f"Which constraint should I replace ({options})?",
                    )
                    return base_patch(classification="clarification_required", clarification=clarification)
        if not replacement_map:
            clarification = self._clarification(
                state,
                reason_code="ambiguous_change",
                question="I could not match the replacement to an existing constraint. Which constraint did you mean?",
            )
            return base_patch(classification="clarification_required", clarification=clarification)
        if any(
            next(record for record in state.current_constraints if record.record_id == old_id).scope
            == "shared"
            for old_id in replacement_map
        ):
            # A shared constraint is a dependency of every active intent.  A
            # reference may identify one representative intent, but it must
            # not leave the other intents attached to the old shared value.
            targets = [item.intent_id for item in state.active_intents]
        replacements, new_intents = self._revision_intents(
            state,
            request=request,
            text=text,
            target_ids=targets,
            additions=[],
            replacements=replacement_map,
            removed_ids=set(),
        )
        trace.append(
            Phase4DecisionStep(
                stage="classification",
                rule="replace_constraint",
                detail="The new value replaces the matched constraint; the old record remains historical.",
                matched_text=text,
                target_ids=list(replacement_map),
            )
        )
        return base_patch(
            classification="replace_constraint",
            target_intent_ids=targets,
            intent_replacements=replacements,
            replacement_constraints=replacement_map,
            new_intents=new_intents,
            reference_resolutions=list(resolutions),
        )

    def _removal_patch(
        self,
        state: Phase4SessionState,
        request: Phase4FollowUpRequest,
        text: str,
        trace: list[Phase4DecisionStep],
        resolutions: Sequence[Phase4ReferenceResolution],
        base_patch: Any,
        transcript_revision: int,
    ) -> Phase4ProposedPatch:
        targets, clarification = self._target_intents(state, text, resolutions, trace)
        if clarification is not None:
            return base_patch(classification="clarification_required", clarification=clarification)
        lowered = _canonical(text)
        request_terms = set(lowered.split())
        removable_stop_words = {
            "a",
            "an",
            "and",
            "at",
            "be",
            "for",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "the",
            "to",
            "with",
            "constraint",
            "requirement",
        }
        candidates = [
            record
            for record in state.current_constraints
            if (record.scope == "shared" or record.intent_id in targets)
            and (
                _canonical(record.constraint.value) in lowered
                or (
                    set(_canonical(record.constraint.value).split()) - removable_stop_words
                    and set(_canonical(record.constraint.value).split()) - removable_stop_words
                    <= request_terms - removable_stop_words
                )
                or (
                    set(_canonical(record.constraint.value).split()) - removable_stop_words
                )
                & (request_terms - removable_stop_words)
            )
        ]
        if len(candidates) == 0:
            # A request may use a noun that was recorded as an `other`
            # constraint; the token check above catches it.  If there are
            # several constraints, guessing is unsafe, so ask instead of
            # deleting a random one.
            candidates = [
                record
                for record in state.current_constraints
                if record.scope == "shared" or record.intent_id in targets
            ]
        if len(candidates) != 1:
            options = ", ".join(item.constraint.value for item in candidates[:5]) or "the existing constraints"
            clarification = self._clarification(
                state,
                reason_code="ambiguous_change",
                question=f"Which constraint should I remove ({options})?",
            )
            return base_patch(classification="clarification_required", clarification=clarification)
        if candidates[0].scope == "shared":
            # Removing a shared constraint broadens every intent that used it,
            # even when the user named only one representative question.
            targets = [item.intent_id for item in state.active_intents]
        removed = {candidates[0].record_id}
        replacements, new_intents = self._revision_intents(
            state,
            request=request,
            text=text,
            target_ids=targets,
            additions=[],
            replacements={},
            removed_ids=removed,
        )
        trace.append(
            Phase4DecisionStep(
                stage="classification",
                rule="remove_constraint",
                detail="The exact matched constraint is marked removed; no other constraint is touched.",
                matched_text=text,
                target_ids=[candidates[0].record_id],
            )
        )
        return base_patch(
            classification="remove_constraint",
            target_intent_ids=targets,
            intent_replacements=replacements,
            removed_constraint_ids=list(removed),
            new_intents=new_intents,
            reference_resolutions=list(resolutions),
        )

    def _revision_intents(
        self,
        state: Phase4SessionState,
        *,
        request: Phase4FollowUpRequest,
        text: str,
        target_ids: Sequence[str],
        additions: Sequence[Phase4ConstraintRecord],
        replacements: Mapping[str, Phase4ConstraintRecord],
        removed_ids: set[str],
    ) -> tuple[dict[str, str], list[Phase4IntentRecord]]:
        current = {record.record_id: record for record in state.current_constraints}
        replacement_by_old = dict(replacements)
        by_target: dict[str, list[Phase4ConstraintRecord]] = {target: [] for target in target_ids}
        for record in current.values():
            if record.intent_id in by_target and record.record_id not in removed_ids and record.record_id not in replacement_by_old:
                by_target[record.intent_id or ""].append(record)
        for old_id, new_record in replacement_by_old.items():
            old = current[old_id]
            if old.intent_id in by_target:
                by_target[old.intent_id or ""].append(new_record)
        for record in additions:
            if record.scope == "intent" and record.intent_id in by_target:
                by_target[record.intent_id or ""].append(record)
        replacements_by_intent: dict[str, str] = {}
        new_records: list[Phase4IntentRecord] = []
        old_by_id = {record.intent_id: record for record in state.active_intents}
        for old_id in target_ids:
            old_record = old_by_id[old_id]
            local_constraints = [record.constraint for record in by_target.get(old_id, []) if record.scope == "intent"]
            revised_query = old_record.intent.query
            for old_constraint_id, new_record in replacement_by_old.items():
                old_constraint = current[old_constraint_id]
                if old_constraint.intent_id == old_id:
                    revised_query = revised_query.replace(old_constraint.constraint.value, new_record.constraint.value)
            for record in additions:
                if record.scope == "intent" and record.intent_id == old_id:
                    if record.constraint.value.casefold() not in revised_query.casefold():
                        revised_query = f"{revised_query.rstrip(' ?')}; {record.constraint.value}"
            for removed_id in removed_ids:
                old_constraint = current.get(removed_id)
                if old_constraint is not None and old_constraint.intent_id == old_id:
                    revised_query = revised_query.replace(old_constraint.constraint.value, "")
            revised_query = _clean_query(revised_query)
            revised_intent_id = _intent_id_for_revision(
                old_record.intent,
                local_constraints,
                [
                    record.constraint
                    for record in additions
                    if record.scope == "shared"
                ],
            )
            # A shared-constraint change is still a semantic revision even
            # though the intent-local constraint list is unchanged.  Keep the
            # revision mapping fresh so stale claims cannot attach to it.
            shared_change = any(
                current.get(constraint_id) is not None
                and current[constraint_id].scope == "shared"
                for constraint_id in (*removed_ids, *replacement_by_old)
            ) or any(record.scope == "shared" for record in additions)
            if revised_intent_id == old_id or shared_change:
                revised_intent_id = "intent-" + _digest(
                    "revision",
                    old_id,
                    request.turn_index,
                    request.utterance_id,
                    _canonical(text),
                )
            revised_intent = old_record.intent.model_copy(
                update={
                    "intent_id": revised_intent_id,
                    "query": revised_query,
                    "constraints": local_constraints,
                }
            )
            replacements_by_intent[old_id] = revised_intent.intent_id
            new_records.append(
                _make_intent_record(
                    revised_intent,
                    origin_turn=request.turn_index,
                    origin_utterance_id=request.utterance_id,
                    origin_revision=request.transcript_revision or request.base_revision + 1,
                    topic_key=old_record.topic_key,
                )
            )
        return replacements_by_intent, new_records


class Phase4SessionStore:
    """Thread-safe bounded in-memory session store with explicit clearing."""

    def __init__(self, *, max_sessions: int = 32, max_records_per_session: int = 256) -> None:
        if max_sessions < 1 or max_records_per_session < 1:
            raise ValueError("Phase 4 memory bounds are too small")
        self.max_sessions = max_sessions
        self.max_records_per_session = max_records_per_session
        self._sessions: OrderedDict[str, Phase4SessionState] = OrderedDict()
        self._lock = threading.RLock()

    def _get_internal(self, session_id: str) -> Phase4SessionState:
        try:
            state = self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(f"unknown session {session_id!r}") from exc
        self._sessions.move_to_end(session_id)
        return state

    def _bounded(self, state: Phase4SessionState) -> Phase4SessionState:
        limit = self.max_records_per_session

        def keep(items: list[Any], current_ids: set[str], id_getter: Any) -> list[Any]:
            if len(items) <= limit:
                return items
            current = [item for item in items if id_getter(item) in current_ids]
            remainder = [item for item in items if id_getter(item) not in current_ids]
            return (remainder[-max(0, limit - len(current)) :] + current)[-limit:]

        publication_claim_ids: set[str] = set()
        if state.current_answer_version is not None:
            publication_claim_ids = {
                claim_id
                for version in state.answer_versions
                if version.answer_version == state.current_answer_version
                for claim_id in version.claim_record_ids
            }
        claim_keep_ids = set(state.current_claim_record_ids) | publication_claim_ids
        claims = keep(state.claim_records, claim_keep_ids, lambda item: item.record_id)
        publication_evidence_ids = {
            evidence_id
            for claim in claims
            if claim.record_id in publication_claim_ids
            for evidence_id in claim.supporting_evidence_ids
        }
        evidence = keep(
            state.evidence_records,
            set(state.current_evidence_ids) | publication_evidence_ids,
            lambda item: item.evidence_id,
        )
        revisions = state.revision_history[-limit:]
        traces = state.trace_ids[-limit:]
        requests = state.pending_requests[-limit:]
        superseded = state.superseded_requests[-limit:]
        plans = list(state.selective_update_plans)
        if len(plans) > limit:
            pending_plan_id = (
                state.pending_update.selective_plan_id
                if state.pending_update is not None
                else None
            )
            kept_plans = plans[-limit:]
            if pending_plan_id is not None and all(
                plan.plan_id != pending_plan_id for plan in kept_plans
            ):
                pending_plan = next(
                    plan for plan in plans if plan.plan_id == pending_plan_id
                )
                kept_plans = [pending_plan, *kept_plans[:-1]]
            plans = kept_plans
        versions = list(state.answer_versions)
        if len(versions) > limit:
            kept_versions = versions[-limit:]
            if state.current_answer_version is not None and all(
                version.answer_version != state.current_answer_version for version in kept_versions
            ):
                current = next(
                    version for version in versions if version.answer_version == state.current_answer_version
                )
                kept_versions = [current, *kept_versions[:-1]]
            versions = kept_versions
        intent_history = state.intent_history[-limit:]
        publication_constraint_ids = {
            dependency.target_id
            for claim in claims
            if claim.record_id in publication_claim_ids
            for dependency in claim.dependencies
            if dependency.dependency_type in {"constraint", "entity"}
        }
        constraints = keep(
            state.constraint_records,
            set(state.current_constraint_record_ids) | publication_constraint_ids,
            lambda item: item.record_id,
        )
        return state.model_copy(
            update={
                "evidence_records": evidence,
                "claim_records": claims,
                "answer_versions": versions,
                "revision_history": revisions,
                "trace_ids": traces,
                "pending_requests": requests,
                "superseded_requests": superseded,
                "intent_history": intent_history,
                "constraint_records": constraints,
                "selective_update_plans": plans,
            }
        )

    def _assert_active_bounds(self, state: Phase4SessionState) -> None:
        """Reject growth that cannot be safely evicted without losing context."""

        limit = self.max_records_per_session
        if len(state.active_intents) > limit:
            raise PatchValidationError(
                "session active-intent bound reached; clear the session before adding more questions"
            )
        if len(state.current_constraints) > limit:
            raise PatchValidationError(
                "session active-constraint bound reached; clear the session before adding more constraints"
            )
        if len(state.current_evidence_ids) > limit or len(state.current_claim_record_ids) > limit:
            raise PatchValidationError(
                "session active-evidence/claim bound reached; clear the session before publishing more state"
            )
        if state.pending_update is not None and len(state.pending_update.task_ids) > limit:
            raise PatchValidationError(
                "session pending-retrieval bound reached; clear the session before adding more work"
            )

    @staticmethod
    def _supersede_outstanding_selective_plans(
        state: Phase4SessionState,
    ) -> list[Phase4SelectiveUpdatePlan]:
        """Mark work obsolete when any newer state-boundary turn is accepted."""

        plans: list[Phase4SelectiveUpdatePlan] = []
        for plan in state.selective_update_plans:
            if plan.status not in {"pending", "running"}:
                plans.append(plan)
                continue
            tasks = [
                task.model_copy(update={"status": "superseded"})
                if task.status in {"queued", "running"}
                else task
                for task in plan.retrieval_tasks
            ]
            plans.append(plan.model_copy(update={"status": "superseded", "retrieval_tasks": tasks}))
        return plans

    def _write(self, state: Phase4SessionState) -> Phase4SessionState:
        self._assert_active_bounds(state)
        bounded = self._bounded(state)
        self._sessions[state.session_id] = _clone(bounded)
        self._sessions.move_to_end(state.session_id)
        return _clone(bounded)

    def create_session(
        self,
        session_id: str,
        *,
        initial_plan: DecompositionResult | None = None,
        active_topic: str | None = None,
        active_task: str | None = None,
        utterance_id: str = "utterance-0",
        initial_turn: int = 0,
        corpus_id: str = "unknown",
        index_id: str = "unknown",
    ) -> Phase4SessionState:
        if not session_id.strip():
            raise ValueError("session_id must not be blank")
        if initial_plan is None:
            if not active_task:
                raise ValueError("create_session requires initial_plan or active_task")
            initial_plan = decompose_query(active_task, transcript_revision=0, max_intents=6)
        if session_id in self._sessions:
            raise SessionExistsError(f"session {session_id!r} already exists")
        topic = active_topic or _topic_key(initial_plan.original_transcript)
        task = active_task or initial_plan.original_transcript
        intents = [
            _make_intent_record(
                intent,
                origin_turn=initial_turn,
                origin_utterance_id=utterance_id,
                origin_revision=initial_plan.transcript_revision,
                topic_key=topic,
            )
            for intent in initial_plan.intents
        ]
        constraints: list[Phase4ConstraintRecord] = []
        current_constraint_ids: list[str] = []
        for constraint in initial_plan.shared_constraints:
            record = _make_constraint_record(
                constraint,
                origin="user_initial",
                origin_turn=initial_turn,
                origin_utterance_id=utterance_id,
                origin_revision=initial_plan.transcript_revision,
                scope="shared",
                intent_id=None,
            )
            constraints.append(record)
            current_constraint_ids.append(record.record_id)
        for intent in initial_plan.intents:
            for constraint in intent.constraints:
                record = _make_constraint_record(
                    constraint,
                    origin="user_initial",
                    origin_turn=initial_turn,
                    origin_utterance_id=utterance_id,
                    origin_revision=initial_plan.transcript_revision,
                    scope="intent",
                    intent_id=intent.intent_id,
                )
                constraints.append(record)
                current_constraint_ids.append(record.record_id)
        state = Phase4SessionState(
            session_id=session_id,
            active_topic=topic,
            active_task=task,
            current_utterance_id=utterance_id,
            transcript_revision=initial_plan.transcript_revision,
            state_revision=0,
            retrieval_revision=0,
            corpus_id=corpus_id,
            index_id=index_id,
            active_intents=intents,
            constraint_records=constraints,
            current_constraint_record_ids=current_constraint_ids,
        )
        self._assert_active_bounds(state)
        with self._lock:
            if session_id in self._sessions:
                raise SessionExistsError(f"session {session_id!r} already exists")
            self._sessions[session_id] = _clone(state)
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        return _clone(state)

    def get(self, session_id: str) -> Phase4SessionState:
        with self._lock:
            return _clone(self._get_internal(session_id))

    def session_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions)

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear_all(self) -> None:
        with self._lock:
            self._sessions.clear()

    def propose_follow_up(
        self,
        session_id: str,
        text: str,
        *,
        utterance_id: str,
        turn_index: int,
        base_revision: int | None = None,
        transcript_revision: int | None = None,
        interpreter: Phase4FollowUpInterpreter | None = None,
    ) -> Phase4ProposedPatch:
        with self._lock:
            state = self._get_internal(session_id)
            expected = state.state_revision if base_revision is None else base_revision
            request = Phase4FollowUpRequest(
                session_id=session_id,
                utterance_id=utterance_id,
                turn_index=turn_index,
                text=text,
                base_revision=expected,
                transcript_revision=transcript_revision,
            )
            return (interpreter or Phase4FollowUpInterpreter()).interpret(_clone(state), request)

    def apply_patch(self, patch: Phase4ProposedPatch) -> Phase4SessionState:
        with self._lock:
            state = self._get_internal(patch.session_id)
            if patch.base_revision != state.state_revision:
                raise PatchConflictError(
                    f"patch targets revision {patch.base_revision}, current revision is {state.state_revision}"
                )
            if patch.proposed_revision != state.state_revision + 1:
                raise PatchConflictError("patch proposed_revision is not the next state revision")
            self._validate_patch_against_state(state, patch)
            if patch.classification == "clarification_required":
                updated = self._apply_clarification(state, patch)
            elif patch.classification == "reformat_answer":
                updated = self._apply_formatting(state, patch)
            else:
                updated = self._apply_semantic_patch(state, patch)
            return self._write(updated)

    def _validate_patch_against_state(
        self,
        state: Phase4SessionState,
        patch: Phase4ProposedPatch,
    ) -> None:
        """Reject a hand-built patch that bypasses interpreter target checks."""

        active_intent_ids = {item.intent_id for item in state.active_intents}
        current_constraint_ids = {item.record_id for item in state.current_constraints}
        if not set(patch.target_intent_ids) <= active_intent_ids:
            raise PatchValidationError("patch targets an inactive or unknown intent")
        if not set(patch.intent_replacements) <= active_intent_ids:
            raise PatchValidationError("patch replaces an inactive or unknown intent")
        if not set(patch.removed_constraint_ids) <= current_constraint_ids:
            raise PatchValidationError("patch removes an inactive or unknown constraint")
        if not set(patch.replacement_constraints) <= current_constraint_ids:
            raise PatchValidationError("patch replaces an inactive or unknown constraint")
        if set(patch.intent_replacements.values()) & active_intent_ids:
            raise PatchValidationError("patch replacement intent IDs must be fresh")
        new_intent_ids = {item.intent_id for item in patch.new_intents}
        if not set(patch.intent_replacements.values()) <= new_intent_ids:
            raise PatchValidationError("patch intent replacement map must reference new intents")
        if patch.classification == "clarification_required":
            if (
                patch.target_intent_ids
                or patch.intent_replacements
                or patch.reference_resolutions
                or patch.topic_key
                or patch.format_instruction
            ):
                raise PatchValidationError("clarification patches must not select or mutate semantic state")
        elif patch.classification == "reformat_answer":
            if (
                patch.target_intent_ids
                or patch.intent_replacements
                or patch.added_constraints
                or patch.replacement_constraints
                or patch.removed_constraint_ids
                or patch.new_intents
                or patch.reference_resolutions
                or patch.topic_key
            ):
                raise PatchValidationError("formatting patches must not contain semantic mutations")
        elif patch.classification == "add_question":
            if patch.intent_replacements or patch.replacement_constraints or patch.removed_constraint_ids:
                raise PatchValidationError("add_question patches must not revise or remove constraints")
        elif patch.classification == "add_constraint":
            if not patch.added_constraints or patch.replacement_constraints or patch.removed_constraint_ids:
                raise PatchValidationError("add_constraint patches require additions only")
        elif patch.classification == "replace_constraint":
            if not patch.replacement_constraints or patch.added_constraints or patch.removed_constraint_ids:
                raise PatchValidationError("replace_constraint patches require replacement records only")
        elif patch.classification == "remove_constraint":
            if not patch.removed_constraint_ids or patch.added_constraints or patch.replacement_constraints:
                raise PatchValidationError("remove_constraint patches require removal IDs only")
        elif patch.classification == "change_topic":
            if patch.intent_replacements or patch.replacement_constraints or patch.removed_constraint_ids:
                raise PatchValidationError("change_topic patches must replace the active task as a whole")
        if patch.classification in {"add_constraint", "replace_constraint", "remove_constraint"}:
            if not patch.target_intent_ids or not patch.intent_replacements:
                raise PatchValidationError("constraint patches must identify revised target intents")
            shared_constraint_change = any(
                record.scope == "shared"
                for record in state.current_constraints
                if record.record_id in set(patch.removed_constraint_ids)
                or record.record_id in set(patch.replacement_constraints)
            ) or any(record.scope == "shared" for record in patch.added_constraints)
            if shared_constraint_change and set(patch.target_intent_ids) != active_intent_ids:
                raise PatchValidationError(
                    "shared constraint changes must revise every active intent"
                )
        if patch.classification == "add_question" and not patch.new_intents:
            raise PatchValidationError("add_question patches require at least one new intent")
        if patch.classification == "change_topic":
            if set(patch.target_intent_ids) != active_intent_ids:
                raise PatchValidationError("topic changes must supersede the complete active intent set")
            if not patch.new_intents:
                raise PatchValidationError("topic changes require a new active intent")
        if patch.classification not in {"clarification_required", "reformat_answer"}:
            if patch.transcript_revision <= state.transcript_revision:
                raise PatchValidationError("semantic patches must advance transcript_revision")
            if patch.retrieval_revision <= state.retrieval_revision:
                raise PatchValidationError("semantic patches must advance retrieval_revision")
        elif patch.classification == "reformat_answer":
            if patch.retrieval_revision != state.retrieval_revision:
                raise PatchValidationError("formatting patches must not advance retrieval_revision")

    def _project_semantic_transition(
        self,
        state: Phase4SessionState,
        patch: Phase4ProposedPatch,
    ) -> dict[str, Any]:
        """Materialise the candidate intents/constraints without touching evidence."""

        active_intents = list(state.active_intents)
        intent_history = list(state.intent_history)
        old_active_by_id = {item.intent_id: item for item in active_intents}
        changed_ids = set(patch.intent_replacements)
        changed_ids.update(patch.target_intent_ids if patch.classification == "change_topic" else ())

        if patch.classification == "change_topic":
            intent_history.extend(
                item.model_copy(update={"status": "superseded"}) for item in active_intents
            )
            active_intents = list(patch.new_intents)
        else:
            if patch.intent_replacements:
                active_intents = [
                    item
                    for item in active_intents
                    if item.intent_id not in patch.intent_replacements
                ]
                for old_id in patch.intent_replacements:
                    intent_history.append(
                        old_active_by_id[old_id].model_copy(update={"status": "superseded"})
                    )
                active_intents.extend(patch.new_intents)
            if patch.classification == "add_question":
                active_intents.extend(patch.new_intents)

        constraints = list(state.constraint_records)
        current_constraint_ids = list(state.current_constraint_record_ids)
        current_by_id = {record.record_id: record for record in state.current_constraints}
        removed = set(patch.removed_constraint_ids)
        replacements = dict(patch.replacement_constraints)

        if patch.classification == "change_topic":
            current_ids = set(current_constraint_ids)
            constraints = [
                record.model_copy(update={"status": "superseded"})
                if record.record_id in current_ids
                else record
                for record in constraints
            ]
            current_constraint_ids = []
            for added in patch.added_constraints:
                constraints.append(added.model_copy(update={"status": "active"}))
                current_constraint_ids.append(added.record_id)
        else:
            # A revised intent gets fresh association records for carried
            # constraints.  This keeps old dependencies historical while the
            # user's unchanged value remains active on the new intent.
            for old_intent_id, new_intent_id in patch.intent_replacements.items():
                for old_record in list(current_by_id.values()):
                    if old_record.intent_id != old_intent_id:
                        continue
                    if old_record.record_id in removed or old_record.record_id in replacements:
                        continue
                    carried = old_record.model_copy(
                        update={
                            "record_id": old_record.record_id + "@" + new_intent_id,
                            "intent_id": new_intent_id,
                            "status": "active",
                        }
                    )
                    constraints.append(carried)
                    current_constraint_ids.append(carried.record_id)
                    constraints = [
                        item.model_copy(update={"status": "superseded"})
                        if item.record_id == old_record.record_id
                        else item
                        for item in constraints
                    ]
                    current_constraint_ids = [
                        item
                        for item in current_constraint_ids
                        if item != old_record.record_id
                    ]
            for old_id in (*removed, *replacements.keys()):
                if old_id in current_by_id:
                    constraints = [
                        item.model_copy(
                            update={
                                "status": "removed"
                                if old_id in removed
                                else "superseded"
                            }
                        )
                        if item.record_id == old_id
                        else item
                        for item in constraints
                    ]
                    current_constraint_ids = [
                        item for item in current_constraint_ids if item != old_id
                    ]
            for old_id, replacement in replacements.items():
                new_intent_id = patch.intent_replacements.get(
                    current_by_id[old_id].intent_id or "",
                    current_by_id[old_id].intent_id,
                )
                replacement = replacement.model_copy(update={"intent_id": new_intent_id})
                constraints.append(replacement)
                current_constraint_ids.append(replacement.record_id)
            for added in patch.added_constraints:
                new_intent_id = patch.intent_replacements.get(
                    added.intent_id or "", added.intent_id
                )
                added = (
                    added.model_copy(update={"intent_id": new_intent_id})
                    if added.scope == "intent"
                    else added
                )
                constraints.append(added)
                current_constraint_ids.append(added.record_id)

        active_topic = (
            patch.topic_key or _topic_key(patch.source_text)
            if patch.classification == "change_topic"
            else state.active_topic
        )
        active_task = (
            patch.source_text
            if patch.classification == "change_topic"
            else f"{state.active_task.rstrip()} {patch.source_text}".strip()
        )
        return {
            "active_intents": active_intents,
            "intent_history": intent_history,
            "constraints": constraints,
            "current_constraint_ids": list(dict.fromkeys(current_constraint_ids)),
            "old_to_new_intent": dict(patch.intent_replacements),
            "changed_intent_ids": changed_ids,
            "active_topic": active_topic,
            "active_task": active_task,
        }

    @staticmethod
    def _constraint_is_covered(
        constraint: DecompositionConstraint,
        passage_text: str,
    ) -> bool:
        """Use exact typed-value coverage as a reuse guard, not as support proof."""

        passage_terms = {
            token.casefold() for token in _TOKEN_PATTERN.findall(passage_text)
        }
        value_terms = {
            token.casefold() for token in _TOKEN_PATTERN.findall(constraint.value)
        }
        # Operators and grammatical prepositions are not identifying values;
        # numbers, units, dates, negations, and entity words remain required.
        ignored = {
            "a",
            "an",
            "and",
            "at",
            "be",
            "for",
            "in",
            "is",
            "least",
            "make",
            "must",
            "of",
            "on",
            "require",
            "required",
            "the",
            "to",
            "with",
        }
        value_terms -= ignored
        return bool(value_terms) and value_terms <= passage_terms

    def _evidence_can_reuse_for_intent(
        self,
        evidence: Phase4EvidenceRecord,
        intent: Phase4IntentRecord,
        constraints: Sequence[Phase4ConstraintRecord],
        *,
        force_retrieval: bool,
    ) -> bool:
        if force_retrieval:
            return False
        relevant = [
            record
            for record in constraints
            if record.status == "active"
            and (record.scope == "shared" or record.intent_id == intent.intent_id)
        ]
        # A no-constraint reassessment can reuse an already selected passage;
        # a new constraint must be represented structurally in that passage.
        return all(
            self._constraint_is_covered(item.constraint, evidence.passage.text)
            for item in relevant
        )

    def _analyze_invalidation(
        self,
        state: Phase4SessionState,
        patch: Phase4ProposedPatch,
        projection: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Compute direct and transitive invalidation before changing records."""

        old_intent_ids = {item.intent_id for item in state.active_intents}
        active_intents: list[Phase4IntentRecord] = projection["active_intents"]
        active_intent_ids = {item.intent_id for item in active_intents}
        old_to_new: dict[str, str] = dict(projection["old_to_new_intent"])
        if patch.classification == "change_topic":
            directly_affected = set(old_intent_ids)
        elif patch.classification == "add_question":
            directly_affected = set()
        else:
            directly_affected = set(patch.target_intent_ids) | set(old_to_new)

        # Shared constraints are dependencies of every active intent.  Keep
        # this derivation defensive as well as relying on patch validation, so
        # hand-built callers cannot accidentally preserve a sibling branch.
        shared_constraint_change = any(
            record.scope == "shared"
            for record in state.current_constraints
            if record.record_id in set(patch.removed_constraint_ids)
            or record.record_id in set(patch.replacement_constraints)
        ) or any(record.scope == "shared" for record in patch.added_constraints)
        if shared_constraint_change and patch.classification != "add_question":
            directly_affected = set(old_intent_ids)

        new_intent_ids = active_intent_ids - old_intent_ids
        if patch.classification == "add_question":
            directly_reported = list(patch.new_intents[i].intent_id for i in range(len(patch.new_intents)))
        else:
            directly_reported = [
                item.intent_id for item in state.active_intents if item.intent_id in directly_affected
            ]
        if patch.classification == "change_topic":
            invalidated_intents = set(old_intent_ids)
        else:
            invalidated_intents = set(directly_affected)

        current_constraints = {item.record_id: item for item in state.current_constraints}
        projected_constraints: list[Phase4ConstraintRecord] = [
            item
            for item in projection["constraints"]
            if item.record_id in set(projection["current_constraint_ids"])
        ]
        projected_by_id = {item.record_id: item for item in projected_constraints}
        constraint_map: dict[str, str | None] = {}
        for old_id, old_record in current_constraints.items():
            if old_id in patch.removed_constraint_ids:
                constraint_map[old_id] = None
            elif old_id in patch.replacement_constraints:
                replacement = patch.replacement_constraints[old_id]
                constraint_map[old_id] = replacement.record_id
            elif old_record.intent_id in old_to_new:
                carried_id = old_id + "@" + old_to_new[old_record.intent_id]
                constraint_map[old_id] = carried_id
            else:
                constraint_map[old_id] = old_id

        changed_constraint_ids = set(patch.removed_constraint_ids)
        changed_constraint_ids.update(patch.replacement_constraints)
        changed_constraint_ids.update(item.record_id for item in patch.added_constraints)
        entity_scope_old = (
            old_intent_ids
            if patch.classification == "change_topic"
            else directly_affected
        )
        entity_scope_new = (
            active_intent_ids
            if patch.classification == "change_topic"
            else {
                old_to_new.get(intent_id, intent_id)
                for intent_id in directly_affected
            }
        )
        old_entity_values: set[str] = {
            _canonical(record.constraint.value)
            for record in current_constraints.values()
            if record.constraint.kind == "entity"
            and (record.scope == "shared" or record.intent_id in entity_scope_old)
        }
        new_entity_values: set[str] = set()
        affected_entity_ids: set[str] = set()
        entity_constraint_kinds = {"entity"}
        for old_id in (*patch.removed_constraint_ids, *patch.replacement_constraints):
            old = current_constraints.get(old_id)
            if old is not None and old.constraint.kind in entity_constraint_kinds:
                affected_entity_ids.add(old.record_id)
        for record in projected_constraints:
            if record.constraint.kind in entity_constraint_kinds:
                if record.scope == "shared" or record.intent_id in entity_scope_new:
                    new_entity_values.add(_canonical(record.constraint.value))
                if record.record_id in changed_constraint_ids:
                    affected_entity_ids.add(record.record_id)
        entity_changed = bool(old_entity_values != new_entity_values) and bool(
            old_entity_values or new_entity_values
        )
        if patch.classification == "change_topic":
            entity_changed = bool(old_entity_values or new_entity_values)
        if entity_changed:
            for record in current_constraints.values():
                if record.constraint.kind in entity_constraint_kinds:
                    affected_entity_ids.add(record.record_id)
            for record in projected_constraints:
                if record.constraint.kind in entity_constraint_kinds:
                    affected_entity_ids.add(record.record_id)

        intent_by_id = {item.intent_id: item for item in active_intents}
        reusable_by_evidence: dict[str, dict[str, str]] = {}
        invalidated_evidence_ids: set[str] = set()
        reused_evidence_ids: set[str] = set()
        sufficient_by_intent: dict[str, bool] = {}
        uncertainty_reasons: list[str] = []
        current_evidence = state.current_evidence
        known_constraint_ids = set(current_constraints)
        for old_id in directly_affected:
            new_id = old_to_new.get(old_id)
            if new_id is None or new_id not in intent_by_id:
                sufficient_by_intent[old_id] = False
                continue
            force = (
                patch.classification in {"remove_constraint", "change_topic"}
                or entity_changed
            )
            sufficient_by_intent[old_id] = any(
                old_id in evidence.intent_ids
                and self._evidence_can_reuse_for_intent(
                    evidence,
                    intent_by_id[new_id],
                    projected_constraints,
                    force_retrieval=force,
                )
                for evidence in current_evidence
            )

        for evidence in current_evidence:
            retained: dict[str, str] = {}
            dropped = False
            unknown_dependencies = [
                dependency
                for dependency in evidence.dependency_ids
                if dependency not in known_constraint_ids
            ]
            if unknown_dependencies:
                # Evidence dependency IDs are intentionally opaque at the
                # contract boundary.  If one cannot be matched to a current
                # constraint, do not guess that the passage is still valid.
                invalidated_evidence_ids.add(evidence.evidence_id)
                uncertainty_reasons.append(
                    f"{evidence.evidence_id}: evidence dependency coverage was not matchable; invalidated conservatively"
                )
                continue
            dependency_invalid = any(
                dependency in changed_constraint_ids
                or constraint_map.get(dependency, dependency) is None
                or (entity_changed and dependency in affected_entity_ids)
                for dependency in evidence.dependency_ids
            )
            if dependency_invalid:
                # Evidence-level dependency metadata is allowed to be
                # broader than the claim that consumed it.  If it names a
                # changed constraint/entity, retain no part of this passage
                # unless a later retrieval re-establishes the dependency.
                invalidated_evidence_ids.add(evidence.evidence_id)
                continue
            for old_id in evidence.intent_ids:
                if old_id not in old_intent_ids:
                    dropped = True
                    continue
                new_id = old_to_new.get(old_id, old_id)
                if new_id not in active_intent_ids:
                    dropped = True
                    continue
                if old_id in directly_affected:
                    force = (
                        patch.classification in {"remove_constraint", "change_topic"}
                        or entity_changed
                    )
                    if not self._evidence_can_reuse_for_intent(
                        evidence,
                        intent_by_id[new_id],
                        projected_constraints,
                        force_retrieval=force,
                    ):
                        dropped = True
                        continue
                retained[old_id] = new_id
            if retained:
                reusable_by_evidence[evidence.evidence_id] = retained
                reused_evidence_ids.add(evidence.evidence_id)
            if dropped or not retained:
                invalidated_evidence_ids.add(evidence.evidence_id)

        invalidated_claim_ids: set[str] = set()
        invalidation_reasons: dict[str, str] = {}
        claims = state.current_claims
        for claim in claims:
            claim_intents = set(claim.intent_ids)
            reason: str | None = None
            if claim_intents & directly_affected:
                affected_claim_intents = claim_intents & directly_affected
                can_reassess = all(
                    sufficient_by_intent.get(intent_id, False)
                    for intent_id in affected_claim_intents
                ) and all(
                    all(
                        intent_id in reusable_by_evidence.get(evidence_id, {})
                        for intent_id in affected_claim_intents
                    )
                    for evidence_id in claim.supporting_evidence_ids
                )
                if not can_reassess:
                    reason = "claim belongs to a directly changed intent without sufficient evidence"
            elif any(
                dependency.dependency_type in {"constraint", "entity"}
                and (
                    dependency.target_id in changed_constraint_ids
                    or dependency.target_id in affected_entity_ids
                    or (
                        dependency.target_value is not None
                        and _canonical(dependency.target_value) in old_entity_values
                    )
                    or (
                        entity_changed
                        and dependency.dependency_type == "entity"
                        and dependency.target_value is None
                        and dependency.target_id not in current_constraints
                        and dependency.target_id not in affected_entity_ids
                    )
                )
                for dependency in claim.dependencies
            ):
                reason = "claim depends on a changed constraint or selected entity"
                if any(
                    dependency.dependency_type == "entity"
                    and dependency.target_value is None
                    and dependency.target_id not in current_constraints
                    and dependency.target_id not in affected_entity_ids
                    for dependency in claim.dependencies
                ):
                    uncertainty_reasons.append(
                        f"{claim.record_id}: entity dependency identity was not matchable; invalidated conservatively"
                    )
            elif any(
                evidence_id in invalidated_evidence_ids
                or not all(
                    intent_id in reusable_by_evidence.get(evidence_id, {})
                    for intent_id in claim_intents
                    if intent_id in directly_affected
                )
                for evidence_id in claim.supporting_evidence_ids
            ):
                reason = "claim support cannot be carried to the candidate revision"
            if reason is not None:
                invalidated_claim_ids.add(claim.record_id)
                invalidation_reasons[claim.record_id] = reason
                if not claim.dependencies:
                    uncertainty_reasons.append(
                        f"{claim.record_id}: dependency coverage was absent; invalidated conservatively"
                    )

        # Explicit claim edges form a small dependency graph.  A claim that
        # summarizes or compares an invalidated claim cannot survive merely
        # because its own citation ID still exists.
        changed_again = True
        while changed_again:
            changed_again = False
            for claim in claims:
                if claim.record_id in invalidated_claim_ids:
                    continue
                for dependency in claim.dependencies:
                    if dependency.dependency_type != "claim":
                        continue
                    if dependency.target_id in invalidated_claim_ids or any(
                        item.claim_id == dependency.target_id
                        for item in claims
                        if item.record_id in invalidated_claim_ids
                    ):
                        invalidated_claim_ids.add(claim.record_id)
                        invalidation_reasons[claim.record_id] = (
                            "dependent claim was invalidated"
                        )
                        changed_again = True
                        break

        # A dependent claim can point to an entity without naming the intent
        # that selected it.  Bring that active intent into targeted retrieval
        # conservatively rather than carrying a policy/price/availability
        # claim across an entity replacement.
        entity_dependent_intents: set[str] = set()
        if affected_entity_ids or entity_changed:
            for claim in claims:
                if any(
                    dependency.dependency_type == "entity"
                    and (
                        dependency.target_id in affected_entity_ids
                        or (
                            dependency.target_value is not None
                            and _canonical(dependency.target_value) in old_entity_values
                        )
                        or (
                            entity_changed
                            and dependency.target_value is None
                            and dependency.target_id not in current_constraints
                        )
                    )
                    for dependency in claim.dependencies
                ):
                    entity_dependent_intents.update(claim.intent_ids)

        retrieval_reasons: dict[str, str] = {}
        retrieval_intent_ids: set[str] = set()
        for old_id in directly_affected:
            new_id = old_to_new.get(old_id)
            if new_id is None or new_id not in active_intent_ids:
                continue
            if entity_changed:
                retrieval_intent_ids.add(new_id)
                retrieval_reasons[new_id] = "selected entity changed; reassess entity-bound facts"
            elif patch.classification == "remove_constraint":
                retrieval_intent_ids.add(new_id)
                retrieval_reasons[new_id] = "constraint removal broadens the valid answer set"
            elif not sufficient_by_intent.get(old_id, False):
                retrieval_intent_ids.add(new_id)
                retrieval_reasons[new_id] = "changed constraint is not structurally covered by current evidence"
        for new_id in new_intent_ids:
            if new_id not in old_to_new.values():
                retrieval_intent_ids.add(new_id)
                retrieval_reasons[new_id] = "new information need has no prior evidence"
        for old_id in entity_dependent_intents:
            new_id = old_to_new.get(old_id, old_id)
            if new_id in active_intent_ids:
                retrieval_intent_ids.add(new_id)
                retrieval_reasons.setdefault(
                    new_id,
                    "a dependent claim selected or referenced an entity whose identity changed",
                )
        preserved_claim_ids: set[str] = set()
        for claim in claims:
            if claim.record_id in invalidated_claim_ids:
                continue
            valid = True
            for evidence_id in claim.supporting_evidence_ids:
                retained = reusable_by_evidence.get(evidence_id, {})
                if not retained or any(
                    intent_id in claim.intent_ids and intent_id not in retained
                    for intent_id in claim.intent_ids
                ):
                    valid = False
                    break
            for dependency in claim.dependencies:
                if dependency.dependency_type in {"constraint", "entity"}:
                    mapped = constraint_map.get(dependency.target_id, dependency.target_id)
                    if mapped is None or mapped not in projected_by_id:
                        valid = False
                        break
                elif dependency.dependency_type == "claim":
                    target = next(
                        (
                            item
                            for item in claims
                            if item.record_id == dependency.target_id
                            or item.claim_id == dependency.target_id
                        ),
                        None,
                    )
                    if target is None:
                        valid = False
                        uncertainty_reasons.append(
                            f"{claim.record_id}: claim dependency identity was not matchable; invalidated conservatively"
                        )
                        break
                    if target.record_id in invalidated_claim_ids:
                        valid = False
                        break
            if valid:
                preserved_claim_ids.add(claim.record_id)
            else:
                invalidated_claim_ids.add(claim.record_id)
                invalidation_reasons[claim.record_id] = (
                    "claim dependency is not present in the candidate revision"
                )

        # Evidence/dependency checks above can discover another invalidated
        # claim after the first graph walk.  Close the claim graph to a fixed
        # point before deciding retrieval scope and preserved IDs.
        changed_again = True
        while changed_again:
            changed_again = False
            for claim in claims:
                if claim.record_id in invalidated_claim_ids:
                    continue
                for dependency in claim.dependencies:
                    if dependency.dependency_type != "claim":
                        continue
                    target = next(
                        (
                            item
                            for item in claims
                            if item.record_id == dependency.target_id
                            or item.claim_id == dependency.target_id
                        ),
                        None,
                    )
                    if target is None or target.record_id in invalidated_claim_ids:
                        invalidated_claim_ids.add(claim.record_id)
                        invalidation_reasons[claim.record_id] = (
                            "dependent claim was invalidated"
                            if target is not None
                            else "claim dependency identity was not matchable; invalidated conservatively"
                        )
                        changed_again = True
                        break
            if changed_again:
                preserved_claim_ids.difference_update(invalidated_claim_ids)

        for claim_id in invalidated_claim_ids:
            claim = next(item for item in claims if item.record_id == claim_id)
            for old_id in claim.intent_ids:
                new_id = old_to_new.get(old_id, old_id)
                if new_id in active_intent_ids and old_id not in directly_affected:
                    retrieval_intent_ids.add(new_id)
                    retrieval_reasons.setdefault(
                        new_id,
                        "a dependent claim was invalidated and needs reassessment",
                    )

        task_intents = [item for item in active_intents if item.intent_id in retrieval_intent_ids]
        tasks = [
            Phase4SelectiveRetrievalTask(
                task_id="task-" + _digest(patch.patch_id, item.intent_id, "targeted-retrieval"),
                session_id=state.session_id,
                patch_id=patch.patch_id,
                candidate_state_revision=patch.proposed_revision,
                candidate_transcript_revision=patch.transcript_revision,
                candidate_retrieval_revision=patch.retrieval_revision,
                intent_id=item.intent_id,
                query=item.intent.query,
                reason=retrieval_reasons[item.intent_id],
            )
            for item in task_intents
        ]
        return {
            "directly_affected_intent_ids": list(dict.fromkeys(directly_reported)),
            "invalidated_intent_ids": list(
                item.intent_id for item in state.active_intents if item.intent_id in invalidated_intents
            ),
            "new_intent_ids": list(item.intent_id for item in active_intents if item.intent_id in new_intent_ids),
            "affected_entity_ids": sorted(affected_entity_ids),
            "invalidated_claim_ids": list(
                item.record_id for item in claims if item.record_id in invalidated_claim_ids
            ),
            "preserved_claim_ids": list(
                item.record_id for item in claims if item.record_id in preserved_claim_ids
            ),
            "invalidated_evidence_ids": list(
                item.evidence_id
                for item in current_evidence
                if item.evidence_id in invalidated_evidence_ids
            ),
            "reused_evidence_ids": list(
                item.evidence_id
                for item in current_evidence
                if item.evidence_id in reused_evidence_ids
            ),
            "retrieval_intent_ids": [item.intent_id for item in task_intents],
            "retrieval_reasons": retrieval_reasons,
            "dependency_uncertain": bool(uncertainty_reasons),
            "dependency_uncertainty_reasons": list(dict.fromkeys(uncertainty_reasons)),
            # Invalidation IDs describe historical evidence/claims, not
            # discarded scheduler results.  Actual result IDs are appended
            # only when a late or failed scheduler result is observed.
            "discarded_result_ids": [],
            "tasks": tasks,
            "reusable_by_evidence": reusable_by_evidence,
            "constraint_map": constraint_map,
            "old_to_new_intent": old_to_new,
            "projection": projection,
            "invalidation_reasons": invalidation_reasons,
        }

    def _base_transition(
        self,
        state: Phase4SessionState,
        patch: Phase4ProposedPatch,
        *,
        discarded_result_ids: Sequence[str],
        reason: str,
        changed_intent_ids: Sequence[str],
        preserved_intent_ids: Sequence[str],
        retrieval_revision: int,
        transcript_revision: int,
        directly_affected_intent_ids: Sequence[str] = (),
        invalidated_claim_ids: Sequence[str] = (),
        preserved_claim_ids: Sequence[str] = (),
        reused_evidence_ids: Sequence[str] = (),
        retrieval_reasons: Mapping[str, str] | None = None,
        dependency_uncertain: bool = False,
        dependency_uncertainty_reasons: Sequence[str] = (),
    ) -> dict[str, Any]:
        revision = patch.proposed_revision
        revision_record = Phase4RevisionRecord(
            revision_id="revision-" + _digest(state.session_id, revision, patch.patch_id),
            session_id=state.session_id,
            utterance_id=patch.utterance_id,
            state_revision=revision,
            transcript_revision=transcript_revision,
            retrieval_revision=retrieval_revision,
            supersedes=[state.state_revision],
            patch_id=patch.patch_id,
            changed_intent_ids=list(changed_intent_ids),
            preserved_intent_ids=list(preserved_intent_ids),
            discarded_result_ids=list(discarded_result_ids),
            reason=reason,
            created_at_utc=_now(),
            directly_affected_intent_ids=list(directly_affected_intent_ids),
            invalidated_claim_ids=list(invalidated_claim_ids),
            preserved_claim_ids=list(preserved_claim_ids),
            reused_evidence_ids=list(reused_evidence_ids),
            retrieval_reasons=dict(retrieval_reasons or {}),
            dependency_uncertain=dependency_uncertain,
            dependency_uncertainty_reasons=list(dependency_uncertainty_reasons),
        )
        completed_request = Phase4PendingRequest(
            request_id=patch.patch_id,
            kind="clarification" if patch.classification == "clarification_required" else "follow_up_patch",
            base_revision=patch.base_revision,
            status="completed",
        )
        superseded_requests = list(state.superseded_requests)
        pending_requests = list(state.pending_requests)
        for pending_request in pending_requests:
            if pending_request.status == "pending":
                superseded_requests.append(
                    pending_request.model_copy(update={"status": "superseded"})
                )
        pending_requests = [
            request
            for request in pending_requests
            if request.status != "pending"
        ]
        if state.pending_update is not None:
            superseded_requests.append(
                Phase4PendingRequest(
                    request_id=state.pending_update.patch_id,
                    kind="follow_up_patch",
                    base_revision=state.pending_update.base_revision,
                    status="superseded",
                )
            )
        if state.pending_clarification is not None:
            superseded_requests.append(
                Phase4PendingRequest(
                    request_id=state.pending_clarification.clarification_id,
                    kind="clarification",
                    base_revision=state.pending_clarification.created_revision - 1,
                    status="superseded",
                )
            )
        return {
            "state_revision": revision,
            "transcript_revision": transcript_revision,
            "retrieval_revision": retrieval_revision,
            # Presentation-only and clarification turns do not create a new
            # corpus query.  Keep the utterance identity that current
            # evidence was retrieved for; the incoming turn remains
            # traceable in the revision record and patch itself.
            "current_utterance_id": (
                patch.utterance_id
                if patch.classification not in {"reformat_answer", "clarification_required"}
                else state.current_utterance_id
            ),
            "pending_requests": [*pending_requests, completed_request],
            "superseded_requests": superseded_requests,
            "revision_history": [*state.revision_history, revision_record],
            "trace_ids": [*state.trace_ids, patch.patch_id],
        }

    def _apply_clarification(
        self,
        state: Phase4SessionState,
        patch: Phase4ProposedPatch,
    ) -> Phase4SessionState:
        assert patch.clarification is not None
        clarification = patch.clarification.model_copy(update={"created_revision": patch.proposed_revision})
        updates = self._base_transition(
            state,
            patch,
            discarded_result_ids=[],
            reason="clarification requested; active semantic state preserved",
            changed_intent_ids=[],
            preserved_intent_ids=[item.intent_id for item in state.active_intents],
            retrieval_revision=state.retrieval_revision,
            transcript_revision=state.transcript_revision,
        )
        updates.update(
            {
                "pending_clarification": clarification,
                "pending_update": None,
                "selective_update_plans": self._supersede_outstanding_selective_plans(state),
                "generation_status": "blocked",
                "answer_status": state.answer_status,
            }
        )
        return state.model_copy(update=updates)

    def _apply_formatting(
        self,
        state: Phase4SessionState,
        patch: Phase4ProposedPatch,
    ) -> Phase4SessionState:
        assert patch.format_instruction is not None
        if state.usable_answer is None:
            clarification = Phase4Clarification(
                clarification_id="clarification-" + _digest(
                    state.session_id, state.state_revision, "presentation-context"
                ),
                question=(
                    "What completed answer should I format? Please provide the original question "
                    "or answer context."
                ),
                reason_code="ambiguous_change",
                related_intent_ids=[item.intent_id for item in state.active_intents],
                created_revision=patch.proposed_revision,
            )
            pending_without_context = Phase4PendingUpdate(
                update_id="update-" + _digest(patch.patch_id, "format-missing-context"),
                patch_id=patch.patch_id,
                base_revision=patch.base_revision,
                triggering_turn=patch.turn_index,
                triggering_utterance_id=patch.utterance_id,
                requested_kind="reformat_answer",
                reason="formatting requested without a usable answer; clarification is pending",
                target_intent_ids=[item.intent_id for item in state.active_intents],
                requires_retrieval=False,
                format_instruction=patch.format_instruction,
                constraint_change_ids=[],
                created_at_utc=_now(),
                status="blocked",
            )
            updates = self._base_transition(
                state,
                patch,
                discarded_result_ids=[],
                reason="formatting requested without a usable answer; context clarification required",
                changed_intent_ids=[],
                preserved_intent_ids=[item.intent_id for item in state.active_intents],
                retrieval_revision=state.retrieval_revision,
                transcript_revision=state.transcript_revision,
            )
            updates.update(
                {
                    # Keep the requested presentation operation inspectable
                    # while the clarification makes clear that no answer was
                    # silently formatted.
                    "pending_update": pending_without_context,
                    "pending_clarification": clarification,
                    "selective_update_plans": self._supersede_outstanding_selective_plans(state),
                    "generation_status": "blocked",
                    "answer_status": state.answer_status,
                }
            )
            return state.model_copy(update=updates)
        pending = Phase4PendingUpdate(
            update_id="update-" + _digest(patch.patch_id, "format"),
            patch_id=patch.patch_id,
            base_revision=patch.base_revision,
            triggering_turn=patch.turn_index,
            triggering_utterance_id=patch.utterance_id,
            requested_kind="reformat_answer",
            reason="presentation-only follow-up; claims and evidence are not changed",
            target_intent_ids=[item.intent_id for item in state.active_intents],
            requires_retrieval=False,
            format_instruction=patch.format_instruction,
            constraint_change_ids=[],
            created_at_utc=_now(),
        )
        updates = self._base_transition(
            state,
            patch,
            discarded_result_ids=[],
            reason="formatting request; existing answer publication preserved",
            changed_intent_ids=[],
            preserved_intent_ids=[item.intent_id for item in state.active_intents],
            retrieval_revision=state.retrieval_revision,
            transcript_revision=state.transcript_revision,
        )
        updates.update(
            {
                "pending_update": pending,
                "pending_clarification": None,
                "selective_update_plans": self._supersede_outstanding_selective_plans(state),
                "generation_status": "pending",
                "answer_status": "presentation_pending",
            }
        )
        return state.model_copy(update=updates)

    def _apply_semantic_patch(
        self,
        state: Phase4SessionState,
        patch: Phase4ProposedPatch,
    ) -> Phase4SessionState:
        projection = self._project_semantic_transition(state, patch)
        analysis = self._analyze_invalidation(state, patch, projection)
        active_intents: list[Phase4IntentRecord] = projection["active_intents"]
        intent_history: list[Phase4IntentRecord] = projection["intent_history"]
        constraints: list[Phase4ConstraintRecord] = projection["constraints"]
        current_constraint_ids: list[str] = projection["current_constraint_ids"]
        current_evidence_ids = list(state.current_evidence_ids)
        current_claim_ids = list(state.current_claim_record_ids)
        reused_by_evidence: dict[str, dict[str, str]] = analysis["reusable_by_evidence"]
        old_to_new_intent: dict[str, str] = analysis["old_to_new_intent"]
        constraint_map: dict[str, str | None] = analysis["constraint_map"]
        active_intent_ids = {item.intent_id for item in active_intents}
        current_constraint_set = set(current_constraint_ids)

        evidence_records: list[Phase4EvidenceRecord] = []
        reused_evidence_record_ids: dict[str, str] = {}
        for record in state.evidence_records:
            if record.evidence_id not in set(current_evidence_ids):
                evidence_records.append(record)
                continue
            retained = reused_by_evidence.get(record.evidence_id)
            if not retained:
                evidence_records.append(record.model_copy(update={"status": "superseded"}))
                continue
            mapped_intents = list(dict.fromkeys(retained.values()))
            mapped_passage_intents = [
                old_to_new_intent.get(item, item)
                for item in (record.passage.intent_ids or record.intent_ids)
                if item in retained
            ]
            mapped_passage_intents = list(dict.fromkeys(mapped_passage_intents or mapped_intents))
            mapped_primary = (
                old_to_new_intent.get(record.passage.intent_id, record.passage.intent_id)
                if record.passage.intent_id in retained
                else (mapped_passage_intents[0] if mapped_passage_intents else None)
            )
            remapped_passage = record.passage.model_copy(
                update={
                    "intent_id": mapped_primary,
                    "intent_ids": mapped_passage_intents,
                    "intent_ranks": {
                        old_to_new_intent.get(key, key): value
                        for key, value in record.passage.intent_ranks.items()
                        if key in retained
                    },
                    "intent_scores": {
                        old_to_new_intent.get(key, key): value
                        for key, value in record.passage.intent_scores.items()
                        if key in retained
                    },
                }
            )
            remapped_dependencies = [
                mapped
                for dependency in record.dependency_ids
                for mapped in [constraint_map.get(dependency, dependency)]
                if mapped is not None and mapped in current_constraint_set
            ]
            reused_id = "evidence-reuse-" + _digest(
                record.evidence_id,
                patch.proposed_revision,
                ",".join(mapped_intents),
            )
            reused_evidence_record_ids[record.evidence_id] = reused_id
            evidence_records.append(
                record.model_copy(update={"status": "superseded"})
            )
            evidence_records.append(
                record.model_copy(
                    update={
                        "evidence_id": reused_id,
                        "passage": remapped_passage,
                        "utterance_id": patch.utterance_id,
                        "transcript_revision": patch.transcript_revision,
                        "retrieval_revision": patch.retrieval_revision,
                        "intent_ids": mapped_intents,
                        "dependency_ids": list(dict.fromkeys(remapped_dependencies)),
                        "status": "current",
                        "reused_from_evidence_id": record.evidence_id,
                        "reuse_reason": "dependency-compatible evidence rebound to the candidate revision",
                    }
                )
            )

        preserved_old_claim_ids = set(analysis["preserved_claim_ids"])
        invalidated_claim_ids = set(analysis["invalidated_claim_ids"])
        preserved_record_id_by_old: dict[str, str] = {
            record_id: f"{next(item.claim_id for item in state.claim_records if item.record_id == record_id)}@r{patch.proposed_revision}"
            for record_id in preserved_old_claim_ids
        }
        claim_records: list[Phase4ClaimRecord] = []
        for record in state.claim_records:
            if record.record_id not in set(current_claim_ids):
                claim_records.append(record)
                continue
            if record.record_id in invalidated_claim_ids:
                claim_records.append(record.model_copy(update={"status": "superseded"}))
                continue
            if record.record_id not in preserved_old_claim_ids:
                claim_records.append(record.model_copy(update={"status": "superseded"}))
                continue
            # Keep the original record resolvable by its parent answer
            # version.  The rebinding below is a new candidate record, not an
            # in-place edit of historical support.
            claim_records.append(record.model_copy(update={"status": "non_current"}))
            mapped_intents = list(
                dict.fromkeys(
                    old_to_new_intent.get(item, item)
                    for item in record.intent_ids
                    if old_to_new_intent.get(item, item) in active_intent_ids
                )
            )
            remapped_claim = record.claim.model_copy(
                update={
                    "intent_id": mapped_intents[0] if mapped_intents else record.claim.intent_id,
                    "intent_ids": mapped_intents,
                }
            )
            remapped_support = [
                reused_evidence_record_ids[item]
                for item in record.supporting_evidence_ids
                if item in reused_evidence_record_ids
            ]
            remapped_dependencies: list[Phase4ClaimDependency] = []
            for dependency in record.dependencies:
                if dependency.dependency_type == "evidence":
                    target_id = reused_evidence_record_ids.get(dependency.target_id)
                elif dependency.dependency_type in {"constraint", "entity"}:
                    target_id = constraint_map.get(dependency.target_id, dependency.target_id)
                elif dependency.dependency_type == "claim":
                    target_id = preserved_record_id_by_old.get(
                        dependency.target_id,
                        next(
                            (
                                preserved_record_id_by_old[item.record_id]
                                for item in state.claim_records
                                if item.claim_id == dependency.target_id
                                and item.record_id in preserved_record_id_by_old
                            ),
                            dependency.target_id,
                        ),
                    )
                else:
                    target_id = dependency.target_id
                if target_id is None:
                    continue
                remapped_dependencies.append(
                    dependency.model_copy(update={"target_id": target_id})
                )
            claim_records.append(
                record.model_copy(
                    update={
                        "record_id": preserved_record_id_by_old[record.record_id],
                        "claim": remapped_claim,
                        "answer_version": state.current_answer_version,
                        "intent_ids": mapped_intents,
                        "supporting_evidence_ids": list(dict.fromkeys(remapped_support)),
                        "dependencies": remapped_dependencies,
                        "claim_revision": patch.proposed_revision,
                        "status": "current",
                    }
                )
            )

        preserved_current_claim_ids = [
            preserved_record_id_by_old[item.record_id]
            for item in state.current_claims
            if item.record_id in preserved_old_claim_ids
        ]
        current_reused_evidence_ids = list(reused_evidence_record_ids.values())
        plan_id = "plan-" + _digest(state.session_id, patch.patch_id, "selective-update")
        tasks: list[Phase4SelectiveRetrievalTask] = analysis["tasks"]
        plan = Phase4SelectiveUpdatePlan(
            plan_id=plan_id,
            session_id=state.session_id,
            patch_id=patch.patch_id,
            base_revision=patch.base_revision,
            candidate_revision=patch.proposed_revision,
            transcript_revision=patch.transcript_revision,
            retrieval_revision=patch.retrieval_revision,
            directly_affected_intent_ids=analysis["directly_affected_intent_ids"],
            invalidated_intent_ids=analysis["invalidated_intent_ids"],
            new_intent_ids=analysis["new_intent_ids"],
            affected_entity_ids=analysis["affected_entity_ids"],
            invalidated_claim_ids=analysis["invalidated_claim_ids"],
            preserved_claim_ids=analysis["preserved_claim_ids"],
            invalidation_reasons=analysis["invalidation_reasons"],
            invalidated_evidence_ids=analysis["invalidated_evidence_ids"],
            reused_evidence_ids=analysis["reused_evidence_ids"],
            retrieval_intent_ids=analysis["retrieval_intent_ids"],
            retrieval_reasons=analysis["retrieval_reasons"],
            dependency_uncertain=analysis["dependency_uncertain"],
            dependency_uncertainty_reasons=analysis["dependency_uncertainty_reasons"],
            discarded_result_ids=analysis["discarded_result_ids"],
            retrieval_tasks=tasks,
            status="pending" if tasks else "completed",
            created_at_utc=_now(),
        )
        plans = list(state.selective_update_plans)
        if state.pending_update is not None and state.pending_update.selective_plan_id is not None:
            old_plan_id = state.pending_update.selective_plan_id
            plans = [
                old_plan.model_copy(
                    update={
                        "status": "superseded",
                        "retrieval_tasks": [
                            task.model_copy(update={"status": "superseded"})
                            if task.status in {"queued", "running"}
                            else task
                            for task in old_plan.retrieval_tasks
                        ],
                    }
                )
                if old_plan.plan_id == old_plan_id
                else old_plan
                for old_plan in plans
            ]
        plans.append(plan)
        pending = Phase4PendingUpdate(
            update_id="update-" + _digest(patch.patch_id, "semantic"),
            patch_id=patch.patch_id,
            base_revision=patch.base_revision,
            triggering_turn=patch.turn_index,
            triggering_utterance_id=patch.utterance_id,
            requested_kind=patch.classification,
            reason=(
                "semantic state changed; targeted retrieval is pending"
                if tasks
                else "existing evidence is sufficient for the changed intents; answer reassessment is pending"
            ),
            target_intent_ids=[item.intent_id for item in patch.new_intents] or patch.target_intent_ids,
            requires_retrieval=bool(tasks),
            constraint_change_ids=list(
                dict.fromkeys(
                    [
                        *patch.removed_constraint_ids,
                        *patch.replacement_constraints.keys(),
                        *[item.record_id for item in patch.replacement_constraints.values()],
                        *[item.record_id for item in patch.added_constraints],
                    ]
                )
            ),
            created_at_utc=_now(),
            candidate_revision=patch.proposed_revision,
            selective_plan_id=plan_id,
            directly_affected_intent_ids=analysis["directly_affected_intent_ids"],
            invalidated_intent_ids=analysis["invalidated_intent_ids"],
            invalidated_claim_ids=analysis["invalidated_claim_ids"],
            preserved_claim_ids=analysis["preserved_claim_ids"],
            invalidation_reasons=analysis["invalidation_reasons"],
            invalidated_evidence_ids=analysis["invalidated_evidence_ids"],
            reused_evidence_ids=analysis["reused_evidence_ids"],
            retrieval_intent_ids=analysis["retrieval_intent_ids"],
            retrieval_reasons=analysis["retrieval_reasons"],
            task_ids=[task.task_id for task in tasks],
            discarded_result_ids=analysis["discarded_result_ids"],
            dependency_uncertain=analysis["dependency_uncertain"],
            dependency_uncertainty_reasons=analysis["dependency_uncertainty_reasons"],
        )
        discarded_result_ids = list(
            dict.fromkeys(
                [
                    *analysis["discarded_result_ids"],
                    *(state.pending_update.task_ids if state.pending_update else []),
                ]
            )
        )
        updates = self._base_transition(
            state,
            patch,
            discarded_result_ids=discarded_result_ids,
            reason=(
                f"applied {patch.classification}; dependency-aware invalidation preserved "
                f"{len(analysis['preserved_claim_ids'])} claims and queued "
                f"{len(tasks)} targeted retrieval task(s)"
            ),
            changed_intent_ids=list(projection["changed_intent_ids"]),
            preserved_intent_ids=[
                item.intent_id
                for item in state.active_intents
                if item.intent_id not in set(analysis["directly_affected_intent_ids"])
            ],
            retrieval_revision=patch.retrieval_revision,
            transcript_revision=patch.transcript_revision,
            directly_affected_intent_ids=analysis["directly_affected_intent_ids"],
            invalidated_claim_ids=analysis["invalidated_claim_ids"],
            preserved_claim_ids=analysis["preserved_claim_ids"],
            reused_evidence_ids=analysis["reused_evidence_ids"],
            retrieval_reasons=analysis["retrieval_reasons"],
            dependency_uncertain=analysis["dependency_uncertain"],
            dependency_uncertainty_reasons=analysis["dependency_uncertainty_reasons"],
        )
        updates.update(
            {
                "active_topic": projection["active_topic"],
                "active_task": projection["active_task"],
                "active_intents": active_intents,
                "intent_history": intent_history,
                "constraint_records": constraints,
                "current_constraint_record_ids": list(dict.fromkeys(current_constraint_ids)),
                "evidence_records": evidence_records,
                "current_evidence_ids": current_reused_evidence_ids,
                "claim_records": claim_records,
                "current_claim_record_ids": preserved_current_claim_ids,
                "current_answer_version": state.current_answer_version,
                "pending_clarification": None,
                "pending_update": pending,
                "selective_update_plans": plans,
                "generation_status": "pending",
                "answer_status": "stale" if state.current_answer_version is not None else "unavailable",
            }
        )
        return state.model_copy(update=updates)

    def _selective_plan_state_update(
        self,
        state: Phase4SessionState,
        plan: Phase4SelectiveUpdatePlan,
    ) -> Phase4SessionState:
        """Persist scheduler bookkeeping without advancing the semantic revision."""

        plans = [
            plan if item.plan_id == plan.plan_id else item
            for item in state.selective_update_plans
        ]
        if not any(item.plan_id == plan.plan_id for item in state.selective_update_plans):
            raise PatchValidationError("selective update plan is not recorded in this session")
        pending = state.pending_update
        if pending is not None and pending.selective_plan_id == plan.plan_id:
            active_tasks = {
                "queued",
                "running",
            }
            requires_retrieval = any(
                task.status in active_tasks for task in plan.retrieval_tasks
            ) or any(task.status == "failed" for task in plan.retrieval_tasks)
            pending = pending.model_copy(
                update={
                    "requires_retrieval": requires_retrieval,
                    "reason": (
                        "targeted retrieval is pending"
                        if requires_retrieval
                        else "targeted retrieval phase completed; answer reassessment remains explicit"
                    ),
                    "task_ids": [task.task_id for task in plan.retrieval_tasks],
                    "discarded_result_ids": list(dict.fromkeys(plan.discarded_result_ids)),
                }
            )
        return self._write(
            state.model_copy(
                update={
                    "selective_update_plans": plans,
                    "pending_update": pending,
                    "trace_ids": list(dict.fromkeys([*state.trace_ids, plan.plan_id])),
                }
            )
        )

    def record_selective_plan_usage(
        self,
        session_id: str,
        plan_id: str,
        *,
        usage: Usage,
        call_count: int,
        attempt_count: int,
    ) -> Phase4SessionState:
        """Persist scheduler usage for one plan without changing its candidate revision."""

        if call_count < 0 or attempt_count < 0:
            raise PatchValidationError("selective retrieval usage counts must be non-negative")
        with self._lock:
            state = self._get_internal(session_id)
            plan = next(
                (item for item in state.selective_update_plans if item.plan_id == plan_id),
                None,
            )
            if plan is None:
                raise PatchValidationError("selective update plan is not recorded in this session")
            updated = plan.model_copy(
                update={
                    "retrieval_usage": usage,
                    "retrieval_call_count": call_count,
                    "retrieval_attempt_count": attempt_count,
                }
            )
            return self._selective_plan_state_update(state, updated)

    @staticmethod
    def _updated_plan_status(
        plan: Phase4SelectiveUpdatePlan,
    ) -> str:
        if plan.status == "superseded":
            return "superseded"
        statuses = {task.status for task in plan.retrieval_tasks}
        if statuses & {"queued", "running"}:
            return "running"
        if "failed" in statuses or "discarded" in statuses:
            return "blocked"
        return "completed"

    def record_selective_task_scheduled(
        self,
        session_id: str,
        plan_id: str,
        task_id: str,
        scheduler_request_id: str,
    ) -> Phase4SessionState:
        """Bind a Phase 4 task to the existing scheduler request identity."""

        with self._lock:
            state = self._get_internal(session_id)
            plan = next(
                (item for item in state.selective_update_plans if item.plan_id == plan_id),
                None,
            )
            if plan is None:
                raise PatchValidationError("selective update plan is not recorded in this session")
            tasks = [
                task.model_copy(
                    update={
                        "status": "running",
                        "scheduler_request_id": scheduler_request_id,
                    }
                )
                if task.task_id == task_id
                else task
                for task in plan.retrieval_tasks
            ]
            if all(task.task_id != task_id for task in plan.retrieval_tasks):
                raise PatchValidationError("selective retrieval task is not recorded in this plan")
            updated = plan.model_copy(
                update={
                    "retrieval_tasks": tasks,
                    "status": self._updated_plan_status(
                        plan.model_copy(update={"retrieval_tasks": tasks})
                    ),
                }
            )
            return self._selective_plan_state_update(state, updated)

    def record_selective_task_result(
        self,
        session_id: str,
        plan_id: str,
        task_id: str,
        *,
        status: str,
        attempts: int,
        evidence_ids: Sequence[str] = (),
        discarded_result_ids: Sequence[str] = (),
        error: str | None = None,
    ) -> Phase4SessionState:
        """Persist terminal task state, including failures and stale results."""

        with self._lock:
            state = self._get_internal(session_id)
            plan = next(
                (item for item in state.selective_update_plans if item.plan_id == plan_id),
                None,
            )
            if plan is None:
                raise PatchValidationError("selective update plan is not recorded in this session")
            task = next(
                (item for item in plan.retrieval_tasks if item.task_id == task_id),
                None,
            )
            if task is None:
                raise PatchValidationError("selective retrieval task is not recorded in this plan")
            if status not in {"completed", "failed", "superseded", "discarded"}:
                raise PatchValidationError("invalid terminal selective retrieval task status")
            updated_tasks = [
                item.model_copy(
                    update={
                        "status": status,
                        "attempts": attempts,
                        "evidence_ids": list(dict.fromkeys(evidence_ids)),
                        "discarded_result_ids": list(dict.fromkeys(discarded_result_ids)),
                        "error": error,
                    }
                )
                if item.task_id == task_id
                else item
                for item in plan.retrieval_tasks
            ]
            updated = plan.model_copy(
                update={
                    "retrieval_tasks": updated_tasks,
                    "discarded_result_ids": list(
                        dict.fromkeys([*plan.discarded_result_ids, *discarded_result_ids])
                    ),
                    "status": self._updated_plan_status(
                        plan.model_copy(update={"retrieval_tasks": updated_tasks})
                    ),
                }
            )
            return self._selective_plan_state_update(state, updated)

    def record_discarded_selective_result(
        self,
        session_id: str,
        plan_id: str,
        result_id: str,
        *,
        task_id: str | None = None,
        reason: str = "result belongs to a superseded candidate revision",
    ) -> Phase4SessionState:
        """Keep a late result in the audit plan while forbidding publication."""

        if not result_id.strip() or not reason.strip():
            raise PatchValidationError("discarded result ID and reason must not be blank")
        with self._lock:
            state = self._get_internal(session_id)
            plan = next(
                (item for item in state.selective_update_plans if item.plan_id == plan_id),
                None,
            )
            if plan is None:
                raise PatchValidationError("selective update plan is not recorded in this session")
            updated_tasks = [
                item.model_copy(
                    update={
                        "status": "discarded"
                        if task_id is not None and item.task_id == task_id
                        else item.status,
                        "discarded_result_ids": list(
                            dict.fromkeys([*item.discarded_result_ids, result_id])
                        )
                        if task_id is not None and item.task_id == task_id
                        else item.discarded_result_ids,
                        "error": reason
                        if task_id is not None and item.task_id == task_id
                        else item.error,
                    }
                )
                for item in plan.retrieval_tasks
            ]
            updated = plan.model_copy(
                update={
                    "retrieval_tasks": updated_tasks,
                    "discarded_result_ids": list(
                        dict.fromkeys([*plan.discarded_result_ids, result_id])
                    ),
                    # A superseded plan remains superseded; otherwise this
                    # result is a blocked terminal branch, never a success.
                    "status": (
                        "superseded"
                        if plan.status == "superseded"
                        else self._updated_plan_status(
                            plan.model_copy(update={"retrieval_tasks": updated_tasks})
                        )
                    ),
                }
            )
            pending = state.pending_update
            if pending is not None and pending.selective_plan_id == plan_id:
                pending = pending.model_copy(
                    update={
                        "discarded_result_ids": list(
                            dict.fromkeys([*pending.discarded_result_ids, result_id])
                        )
                    }
                )
            return self._write(
                state.model_copy(
                    update={
                        "selective_update_plans": [
                            updated if item.plan_id == plan_id else item
                            for item in state.selective_update_plans
                        ],
                        "pending_update": pending,
                        "trace_ids": list(dict.fromkeys([*state.trace_ids, result_id])),
                    }
                )
            )

    def record_selective_evidence(
        self,
        session_id: str,
        task: Phase4SelectiveRetrievalTask,
        passage: EvidencePassage,
        *,
        expected_revision: int | None = None,
    ) -> Phase4SessionState:
        """Record one accepted targeted passage without invalidating sibling work."""

        with self._lock:
            state = self._get_internal(session_id)
            if expected_revision is not None and expected_revision != state.state_revision:
                raise PatchConflictError("targeted evidence belongs to an obsolete session revision")
            if state.state_revision != task.candidate_state_revision:
                raise PatchConflictError("targeted evidence belongs to an obsolete session revision")
            if state.transcript_revision != task.candidate_transcript_revision:
                raise PatchConflictError("targeted evidence transcript revision is stale")
            if state.retrieval_revision != task.candidate_retrieval_revision:
                raise PatchConflictError("targeted evidence retrieval revision is stale")
            plan = next(
                (
                    item
                    for item in state.selective_update_plans
                    if item.patch_id == task.patch_id
                    and item.candidate_revision == task.candidate_state_revision
                ),
                None,
            )
            if plan is None or plan.status == "superseded":
                raise PatchConflictError("targeted evidence belongs to a superseded update plan")
            if task.intent_id not in {item.intent_id for item in state.active_intents}:
                raise PatchValidationError("targeted evidence references an inactive intent")
            normalized_passage = passage.model_copy(
                update={
                    "intent_id": task.intent_id,
                    "intent_ids": [task.intent_id],
                    "intent_ranks": {},
                    "intent_scores": {},
                }
            )
            evidence_id = "evidence-targeted-" + _digest(
                session_id,
                task.task_id,
                normalized_passage.chunk_id,
            )
            dependency_ids = [
                record.record_id
                for record in state.current_constraints
                if record.scope == "shared" or record.intent_id == task.intent_id
            ]
            record = Phase4EvidenceRecord(
                evidence_id=evidence_id,
                passage=normalized_passage,
                session_id=session_id,
                utterance_id=state.current_utterance_id,
                transcript_revision=state.transcript_revision,
                retrieval_revision=state.retrieval_revision,
                corpus_id=state.corpus_id,
                index_id=state.index_id,
                intent_ids=[task.intent_id],
                dependency_ids=dependency_ids,
                status="current",
            )
            evidence = [
                item for item in state.evidence_records if item.evidence_id != evidence_id
            ]
            evidence.append(record)
            task_evidence_ids = list(
                dict.fromkeys([*next(
                    item.evidence_ids
                    for item in plan.retrieval_tasks
                    if item.task_id == task.task_id
                ), evidence_id])
            )
            updated_tasks = [
                item.model_copy(update={"evidence_ids": task_evidence_ids})
                if item.task_id == task.task_id
                else item
                for item in plan.retrieval_tasks
            ]
            updated_plan = plan.model_copy(
                update={
                    "retrieval_tasks": updated_tasks,
                    "status": self._updated_plan_status(
                        plan.model_copy(update={"retrieval_tasks": updated_tasks})
                    ),
                }
            )
            updated_state = state.model_copy(
                update={
                    "evidence_records": evidence,
                    "current_evidence_ids": list(
                        dict.fromkeys([*state.current_evidence_ids, evidence_id])
                    ),
                    "selective_update_plans": [
                        updated_plan if item.plan_id == plan.plan_id else item
                        for item in state.selective_update_plans
                    ],
                    "trace_ids": list(dict.fromkeys([*state.trace_ids, evidence_id])),
                    "generation_status": "pending",
                }
            )
            return self._write(updated_state)

    def record_evidence(
        self,
        session_id: str,
        passage: EvidencePassage,
        *,
        intent_ids: Sequence[str] | None = None,
        dependency_ids: Sequence[str] = (),
        retrieval_revision: int | None = None,
        corpus_id: str | None = None,
        index_id: str | None = None,
        expected_revision: int | None = None,
    ) -> Phase4SessionState:
        with self._lock:
            state = self._get_internal(session_id)
            if expected_revision is not None and expected_revision != state.state_revision:
                raise PatchConflictError("evidence result belongs to an obsolete session revision")
            if corpus_id is not None and corpus_id != state.corpus_id:
                raise PatchValidationError("evidence corpus_id does not match the session corpus")
            if index_id is not None and index_id != state.index_id:
                raise PatchValidationError("evidence index_id does not match the session index")
            effective_intents = list(intent_ids or passage.intent_ids)
            if not effective_intents:
                if len(state.active_intents) == 1:
                    effective_intents = [state.active_intents[0].intent_id]
                else:
                    effective_intents = [item.intent_id for item in state.active_intents]
            active_ids = {item.intent_id for item in state.active_intents}
            if not set(effective_intents) <= active_ids:
                raise PatchValidationError("evidence references an inactive intent")
            effective_revision = state.retrieval_revision if retrieval_revision is None else retrieval_revision
            if effective_revision < state.retrieval_revision:
                raise PatchConflictError("evidence retrieval_revision is stale")
            old_current = list(state.current_evidence_ids)
            evidence = list(state.evidence_records)
            if effective_revision > state.retrieval_revision:
                evidence = [
                    item.model_copy(update={"status": "superseded"})
                    if item.evidence_id in set(old_current)
                    else item
                    for item in evidence
                ]
                old_current = []
            evidence_id = "evidence-" + _digest(
                state.session_id,
                state.transcript_revision,
                effective_revision,
                passage.chunk_id,
            )
            record = Phase4EvidenceRecord(
                evidence_id=evidence_id,
                passage=passage,
                session_id=state.session_id,
                utterance_id=state.current_utterance_id,
                transcript_revision=state.transcript_revision,
                retrieval_revision=effective_revision,
                corpus_id=state.corpus_id,
                index_id=state.index_id,
                intent_ids=list(dict.fromkeys(effective_intents)),
                dependency_ids=list(dict.fromkeys(dependency_ids)),
            )
            evidence = [item for item in evidence if item.evidence_id != evidence_id]
            evidence.append(record)
            claims = [
                item.model_copy(update={"status": "non_current"})
                if item.record_id in set(state.current_claim_record_ids)
                else item
                for item in state.claim_records
            ]
            updates = {
                "state_revision": state.state_revision + 1,
                "retrieval_revision": effective_revision,
                "evidence_records": evidence,
                "current_evidence_ids": [*old_current, evidence_id],
                "claim_records": claims,
                "current_claim_record_ids": [],
                "pending_update": state.pending_update,
                "generation_status": "pending",
                "trace_ids": [*state.trace_ids, evidence_id],
                "revision_history": [
                    *state.revision_history,
                    Phase4RevisionRecord(
                        revision_id="revision-" + _digest(state.session_id, state.state_revision + 1, evidence_id),
                        session_id=state.session_id,
                        utterance_id=state.current_utterance_id,
                        state_revision=state.state_revision + 1,
                        transcript_revision=state.transcript_revision,
                        retrieval_revision=effective_revision,
                        supersedes=[state.state_revision],
                        discarded_result_ids=list(state.current_claim_record_ids),
                        reason="evidence recorded; no semantic support verdict inferred",
                        created_at_utc=_now(),
                    ),
                ],
            }
            return self._write(state.model_copy(update=updates))

    def begin_answer_generation(
        self,
        session_id: str,
        *,
        request_id: str,
        expected_revision: int,
        request_kind: Literal["answer_publication", "presentation"] = "answer_publication",
    ) -> Phase4SessionState:
        """Register factual or presentation work without advancing the revision."""

        if not request_id.strip():
            raise PatchValidationError("answer generation request ID must not be blank")
        with self._lock:
            state = self._get_internal(session_id)
            if expected_revision != state.state_revision:
                raise PatchConflictError("answer generation belongs to an obsolete session revision")
            request = Phase4PendingRequest(
                request_id=request_id,
                kind=request_kind,
                base_revision=expected_revision,
                status="pending",
            )
            pending = [
                item for item in state.pending_requests if item.request_id != request_id
            ]
            return self._write(
                state.model_copy(
                    update={
                        "pending_requests": [*pending, request],
                        "generation_status": "pending",
                    }
                )
            )

    def record_discarded_generation(
        self,
        session_id: str,
        request_id: str,
        *,
        reason: str = "generation result belongs to a superseded session revision",
        request_kind: Literal["answer_publication", "presentation"] = "answer_publication",
    ) -> Phase4SessionState:
        """Retain a late generation result in bounded audit state without publishing it."""

        if not request_id.strip() or not reason.strip():
            raise PatchValidationError("generation request ID and reason must not be blank")
        with self._lock:
            state = self._get_internal(session_id)
            pending = [
                item for item in state.pending_requests if item.request_id != request_id
            ]
            known_superseded = {item.request_id for item in state.superseded_requests}
            superseded = list(state.superseded_requests)
            if request_id not in known_superseded:
                superseded.append(
                    Phase4PendingRequest(
                        request_id=request_id,
                        kind=request_kind,
                        base_revision=state.state_revision,
                        status="superseded",
                    )
                )
            return self._write(
                state.model_copy(
                    update={
                        "pending_requests": pending,
                        "superseded_requests": superseded,
                        "trace_ids": list(dict.fromkeys([*state.trace_ids, request_id, reason])),
                    }
                )
            )

    def publish_answer(
        self,
        session_id: str,
        answer: Answer,
        *,
        extra_dependencies: Mapping[str, Sequence[Phase4ClaimDependency]] | None = None,
        expected_revision: int | None = None,
        triggering_patch_id: str | None = None,
        triggering_turn: int | None = None,
        triggering_utterance_id: str | None = None,
        change_kind: str | None = None,
        status: str | None = None,
        failure_reason: str | None = None,
        constraint_change_ids: Sequence[str] = (),
        retrieval_usage: Usage | None = None,
        generation_usage: Usage | None = None,
        presentation_usage: Usage | None = None,
        retrieval_call_count: int = 0,
        retrieval_attempt_count: int = 0,
        generation_attempts: int = 0,
        generation_repair_attempts: int = 0,
        retrieval_model_identity: str = "unknown",
        generation_execution_mode: str = "rule_based",
        generation_provider: str = "unknown",
        generation_model: str = "unknown",
        presentation_execution_mode: str = "rule_based",
        presentation_provider: str = "flowcontext.rule_based",
        unresolved_intent_ids: Sequence[str] = (),
        unresolved_questions: Sequence[str] = (),
        generation_request_id: str | None = None,
        rendered_answer_text: str | None = None,
    ) -> Phase4SessionState:
        """Atomically publish a candidate answer against one exact state revision.

        All candidate claim records, status text, dependency links, and the
        immutable version envelope are constructed before ``_write``.  A late
        caller therefore either commits a complete compatible version or gets
        ``PatchConflictError`` without changing the session.
        """

        with self._lock:
            state = self._get_internal(session_id)
            if expected_revision is not None and expected_revision != state.state_revision:
                raise PatchConflictError("answer publication belongs to an obsolete session revision")
            if status not in {None, "completed", "partial", "failed", "superseded"}:
                raise PatchValidationError("invalid answer publication status")
            if change_kind not in {None, "initial", "factual_update", "presentation"}:
                raise PatchValidationError("invalid answer publication kind")
            if generation_execution_mode not in {"rule_based", "mock_provider", "real_provider"}:
                raise PatchValidationError("invalid generation execution mode")
            if presentation_execution_mode not in {"rule_based", "mock_provider", "real_provider"}:
                raise PatchValidationError("invalid presentation execution mode")
            if status == "superseded":
                raise PatchConflictError("superseded answer work cannot be published")
            return self._publish_answer_locked(
                state,
                answer,
                extra_dependencies=extra_dependencies or {},
                triggering_patch_id=triggering_patch_id,
                triggering_turn=triggering_turn,
                triggering_utterance_id=triggering_utterance_id,
                change_kind=change_kind,
                status=status,
                failure_reason=failure_reason,
                constraint_change_ids=constraint_change_ids,
                retrieval_usage=retrieval_usage or Usage(),
                generation_usage=generation_usage or Usage(),
                presentation_usage=presentation_usage or Usage(),
                retrieval_call_count=retrieval_call_count,
                retrieval_attempt_count=retrieval_attempt_count,
                generation_attempts=generation_attempts,
                generation_repair_attempts=generation_repair_attempts,
                retrieval_model_identity=retrieval_model_identity,
                generation_execution_mode=generation_execution_mode,
                generation_provider=generation_provider,
                generation_model=generation_model,
                presentation_execution_mode=presentation_execution_mode,
                presentation_provider=presentation_provider,
                unresolved_intent_ids=unresolved_intent_ids,
                unresolved_questions=unresolved_questions,
                generation_request_id=generation_request_id,
                rendered_answer_text=rendered_answer_text,
            )

    def _publish_answer_locked(
        self,
        state: Phase4SessionState,
        answer: Answer,
        *,
        extra_dependencies: Mapping[str, Sequence[Phase4ClaimDependency]],
        triggering_patch_id: str | None,
        triggering_turn: int | None,
        triggering_utterance_id: str | None,
        change_kind: str | None,
        status: str | None,
        failure_reason: str | None,
        constraint_change_ids: Sequence[str],
        retrieval_usage: Usage,
        generation_usage: Usage,
        presentation_usage: Usage,
        retrieval_call_count: int,
        retrieval_attempt_count: int,
        generation_attempts: int,
        generation_repair_attempts: int,
        retrieval_model_identity: str,
        generation_execution_mode: str,
        generation_provider: str,
        generation_model: str,
        presentation_execution_mode: str,
        presentation_provider: str,
        unresolved_intent_ids: Sequence[str],
        unresolved_questions: Sequence[str],
        generation_request_id: str | None,
        rendered_answer_text: str | None,
    ) -> Phase4SessionState:
        from .synthesis import render_unified_answer, validate_citations_and_excerpts

        revision = state.state_revision + 1
        next_version = max((item.answer_version for item in state.answer_versions), default=0) + 1
        parent_version = state.current_answer_version
        effective_kind = change_kind or ("initial" if parent_version is None else "factual_update")
        if effective_kind == "presentation" and parent_version is None:
            raise PatchValidationError("presentation publication requires an existing answer")
        current_evidence = state.current_evidence
        evidence_by_chunk: dict[str, Phase4EvidenceRecord] = {}
        for evidence in current_evidence:
            # The latest admitted record is the only active support selected
            # for a duplicated corpus chunk.
            evidence_by_chunk[evidence.passage.chunk_id] = evidence
        supplied_passages = {
            chunk_id: evidence.passage for chunk_id, evidence in evidence_by_chunk.items()
        }

        parent_records_by_id = {
            item.record_id: item for item in state.claim_records
        }
        parent_version_record_ids: list[str] = []
        if parent_version is not None:
            parent_publication = next(
                (
                    version
                    for version in state.answer_versions
                    if version.answer_version == parent_version
                ),
                None,
            )
            if parent_publication is not None:
                parent_version_record_ids = list(parent_publication.claim_record_ids)
        parent_records = [
            parent_records_by_id[record_id]
            for record_id in parent_version_record_ids
            if record_id in parent_records_by_id
        ]
        if not parent_records:
            parent_records = list(state.current_claims)
        parent_by_claim_id = {record.claim_id: record for record in parent_records}
        current_records = list(state.current_claims)
        current_by_claim_id = {record.claim_id: record for record in current_records}
        known_claim_ids = {item.claim_id for item in state.claim_records}
        active_intent_ids = {item.intent_id for item in state.active_intents}

        entries: list[tuple[FactualClaim, Phase4ClaimRecord | None, Phase4ClaimRecord | None, str]] = []
        matched_current_record_ids: set[str] = set()
        skipped_claim_reasons: list[str] = []
        occupied_claim_ids: set[str] = set()

        def normalize_claim(claim: FactualClaim) -> FactualClaim:
            supporting = [
                evidence_by_chunk[chunk_id]
                for chunk_id in claim.supporting_chunk_ids
                if chunk_id in evidence_by_chunk
            ]
            if len(supporting) != len(claim.supporting_chunk_ids):
                missing = sorted(
                    set(claim.supporting_chunk_ids) - set(evidence_by_chunk)
                )
                raise PatchValidationError(
                    f"claim {claim.claim_id!r} cites evidence that is not current in this session: {missing}"
                )
            intent_ids = list(dict.fromkeys(claim.intent_ids))
            if not intent_ids:
                inferred = list(
                    dict.fromkeys(
                        intent_id
                        for evidence in supporting
                        for intent_id in evidence.intent_ids
                    )
                )
                intent_ids = inferred if inferred else (
                    [state.active_intents[0].intent_id] if len(state.active_intents) == 1 else []
                )
            if not intent_ids or not set(intent_ids) <= active_intent_ids:
                raise PatchValidationError(
                    f"claim {claim.claim_id!r} must identify only active intent IDs"
                )
            return claim.model_copy(
                update={
                    "intent_id": intent_ids[0],
                    "intent_ids": intent_ids,
                }
            )

        incoming_claims: list[tuple[FactualClaim, str]] = []
        for raw_claim in answer.factual_claims:
            claim = normalize_claim(raw_claim)
            if claim.semantic_support in {"unsupported", "uncertain"}:
                skipped_claim_reasons.append(
                    f"{claim.claim_id}: semantic support was {claim.semantic_support}; claim omitted"
                )
                continue
            incoming_claims.append((claim, raw_claim.claim_id))
        if incoming_claims:
            validate_citations_and_excerpts(
                [claim for claim, _original_id in incoming_claims],
                supplied_passages,
            )

        for claim, original_claim_id in incoming_claims:
            signature = _claim_signature(claim)
            same_current = next(
                (
                    record
                    for record in current_records
                    if record.record_id not in matched_current_record_ids
                    and _claim_signature(record.claim) == signature
                ),
                None,
            )
            if same_current is not None:
                normalized_claim = claim.model_copy(update={"claim_id": same_current.claim_id})
                entries.append(
                    (
                        normalized_claim,
                        same_current,
                        parent_by_claim_id.get(same_current.claim_id, same_current),
                        original_claim_id,
                    )
                )
                matched_current_record_ids.add(same_current.record_id)
                occupied_claim_ids.add(same_current.claim_id)
                continue

            previous = current_by_claim_id.get(claim.claim_id) or parent_by_claim_id.get(claim.claim_id)
            normalized_claim = claim
            if (
                claim.claim_id in known_claim_ids
                or claim.claim_id in occupied_claim_ids
            ):
                normalized_claim = claim.model_copy(
                    update={
                        "claim_id": "claim-" + _digest(
                            "material-change",
                            claim.claim_id,
                            next_version,
                            claim.claim_text,
                            ",".join(claim.supporting_chunk_ids),
                            ",".join(claim.intent_ids),
                        )
                    }
                )
            entries.append((normalized_claim, None, previous, original_claim_id))
            occupied_claim_ids.add(normalized_claim.claim_id)

        # A generator may return only affected claims.  Current claims that
        # were preserved by dependency invalidation are therefore carried into
        # the candidate explicitly, never implicitly by reusing the old answer.
        for record in current_records:
            if record.record_id in matched_current_record_ids:
                continue
            entries.append(
                (
                    record.claim,
                    record,
                    parent_by_claim_id.get(record.claim_id, record),
                    record.claim_id,
                )
            )
            occupied_claim_ids.add(record.claim_id)

        entry_claims = [entry[0] for entry in entries]
        status_by_intent = {
            item.intent_id: item
            for item in answer.intent_statuses
            if item.intent_id in active_intent_ids
        }
        claims_by_intent: dict[str, list[FactualClaim]] = {}
        for claim in entry_claims:
            for intent_id in claim.intent_ids:
                claims_by_intent.setdefault(intent_id, []).append(claim)
        intent_statuses: list[IntentStatusRecord] = []
        effective_unresolved: set[str] = set(unresolved_intent_ids) & active_intent_ids
        pending_affected_intents: set[str] = set()
        if state.pending_update is not None:
            pending_affected_intents.update(state.pending_update.directly_affected_intent_ids)
            pending_affected_intents.update(state.pending_update.invalidated_intent_ids)
            pending_affected_intents.update(state.pending_update.target_intent_ids)
            pending_affected_intents.update(state.pending_update.retrieval_intent_ids)
        for intent_record in state.active_intents:
            intent_id = intent_record.intent_id
            provided = status_by_intent.get(intent_id)
            claims_for_intent = claims_by_intent.get(intent_id, [])
            if provided is not None and provided.status != "answered":
                if claims_for_intent and intent_id not in pending_affected_intents and intent_id not in effective_unresolved:
                    # A failed reassessment may return an insufficient status
                    # for every intent.  That must not erase a valid sibling
                    # branch that the candidate did not affect.
                    current_status = IntentStatusRecord(
                        intent_id=intent_id,
                        status="answered",
                        reason="Previously validated claim preserved while another intent was reassessed.",
                        addressed_by_claim_ids=[claim.claim_id for claim in claims_for_intent],
                    )
                else:
                    current_status = provided.model_copy(
                        update={"addressed_by_claim_ids": [claim.claim_id for claim in claims_for_intent]}
                    )
            elif claims_for_intent:
                current_status = IntentStatusRecord(
                    intent_id=intent_id,
                    status="answered",
                    reason=(provided.reason if provided is not None else "Supported claim retained or re-evaluated."),
                    addressed_by_claim_ids=[claim.claim_id for claim in claims_for_intent],
                )
            else:
                current_status = IntentStatusRecord(
                    intent_id=intent_id,
                    status="insufficient_evidence",
                    reason=(
                        provided.reason
                        if provided is not None and provided.reason
                        else "No supported claim is available for this information need."
                    ),
                )
            if current_status.status != "answered":
                effective_unresolved.add(intent_id)
            intent_statuses.append(current_status)
        effective_unresolved_questions = list(dict.fromkeys(
            [
                *unresolved_questions,
                *[
                    record.intent.query
                    for record in state.active_intents
                    if record.intent_id in effective_unresolved
                ],
            ]
        ))
        if skipped_claim_reasons:
            failure_reason = failure_reason or "; ".join(skipped_claim_reasons)

        if rendered_answer_text is None:
            rendered_answer_text = render_unified_answer(
                entry_claims,
                intent_statuses,
                [record.intent for record in state.active_intents],
                [],
            )
        uncertainty_parts = [answer.uncertainty.strip()]
        if effective_unresolved_questions:
            uncertainty_parts.append(
                "Unresolved information needs: "
                + "; ".join(effective_unresolved_questions)
                + "."
            )
        if skipped_claim_reasons:
            uncertainty_parts.append("Claims omitted during publication: " + "; ".join(skipped_claim_reasons) + ".")
        if any(
            claim.semantic_support in {None, ""}
            for claim in entry_claims
        ):
            uncertainty_parts.append(
                "Citation IDs and exact excerpts were structurally validated; semantic support "
                "was not independently established for every claim."
            )
        normalized_answer = answer.model_copy(
            update={
                "answer_text": rendered_answer_text,
                "factual_claims": entry_claims,
                "uncertainty": " ".join(dict.fromkeys(part for part in uncertainty_parts if part)),
                "answer_version": next_version,
                "intent_statuses": intent_statuses,
            }
        )

        claim_records: list[Phase4ClaimRecord] = [
            item.model_copy(update={"status": "non_current"})
            if item.record_id in set(state.current_claim_record_ids)
            else item
            for item in state.claim_records
        ]
        new_records: list[Phase4ClaimRecord] = []
        origin_by_new_record: dict[str, Phase4ClaimRecord | None] = {}
        for claim, source_record, previous_record, original_claim_id in entries:
            supporting = [evidence_by_chunk[item] for item in claim.supporting_chunk_ids]
            carried_claim_dependencies = self._carried_claim_dependencies(
                source_record,
                current_records,
            )
            explicit_dependencies = [
                *carried_claim_dependencies,
                *extra_dependencies.get(original_claim_id, ()),
                *extra_dependencies.get(claim.claim_id, ()),
            ]
            dependencies = self._claim_dependencies(
                state,
                claim,
                supporting,
                explicit_dependencies,
                {item[0].claim_id for item in entries} | known_claim_ids | {
                    item.record_id for item in state.claim_records
                },
            )
            semantic_status = (
                claim.semantic_support
                if claim.semantic_support in {"supported", "unsupported", "uncertain"}
                else source_record.semantic_support_status
                if source_record is not None
                else "unreviewed"
            )
            record = Phase4ClaimRecord(
                record_id=f"{claim.claim_id}@v{next_version}",
                claim=claim,
                claim_id=claim.claim_id,
                answer_version=next_version,
                intent_ids=list(claim.intent_ids),
                supporting_evidence_ids=[item.evidence_id for item in supporting],
                dependencies=dependencies,
                claim_revision=revision,
                semantic_support_status=semantic_status,
            )
            new_records.append(record)
            origin_by_new_record[record.record_id] = previous_record

        old_by_record_id = {
            record.record_id: record
            for record in parent_records
        }
        old_used: set[str] = set()
        claim_changes: list[Phase4ClaimChange] = []
        for record in new_records:
            previous = origin_by_new_record.get(record.record_id)
            if previous is None:
                previous = old_by_record_id.get(record.claim_id)
            if previous is not None:
                old_used.add(previous.record_id)
                unchanged = _claim_signature(previous.claim) == _claim_signature(record.claim)
                change_type = "preserved" if unchanged else "modified"
                claim_changes.append(
                    Phase4ClaimChange(
                        change_type=change_type,
                        claim_id=record.claim_id,
                        previous_claim_id=previous.claim_id,
                        current_claim_id=record.claim_id,
                        previous_record_id=previous.record_id,
                        current_record_id=record.record_id,
                        reason=(
                            "claim content and corpus support remained compatible"
                            if unchanged
                            else "claim content, intent, or corpus support changed"
                        ),
                    )
                )
            else:
                claim_changes.append(
                    Phase4ClaimChange(
                        change_type="added",
                        claim_id=record.claim_id,
                        current_claim_id=record.claim_id,
                        current_record_id=record.record_id,
                        reason="newly supported or reassessed information need",
                    )
                )
        for previous in parent_records:
            if previous.record_id not in old_used:
                claim_changes.append(
                    Phase4ClaimChange(
                        change_type="removed",
                        claim_id=previous.claim_id,
                        previous_claim_id=previous.claim_id,
                        previous_record_id=previous.record_id,
                        reason="claim was not valid for the candidate revision",
                    )
                )

        old_evidence_ids = {
            evidence_id
            for record in parent_records
            for evidence_id in record.supporting_evidence_ids
        }
        new_evidence_ids = set(state.current_evidence_ids)
        reused_current_ids = {
            record.evidence_id
            for record in current_evidence
            if record.reused_from_evidence_id is not None
        }
        publication_status = status
        if publication_status is None:
            publication_status = (
                "partial"
                if effective_unresolved and new_records
                else "failed"
                if effective_unresolved
                else "completed"
            )
        if publication_status == "failed" and not failure_reason:
            failure_reason = "No supported claim was available for the requested revision."
        final_failure_reason = failure_reason if publication_status in {"partial", "failed"} else None
        answer_version = Phase4AnswerVersion(
            answer_version=next_version,
            answer=normalized_answer,
            claim_record_ids=[item.record_id for item in new_records],
            state_revision=revision,
            created_at_utc=_now(),
            parent_version=parent_version,
            triggering_turn=triggering_turn if triggering_turn is not None else 0,
            triggering_utterance_id=triggering_utterance_id or state.current_utterance_id,
            triggering_patch_id=triggering_patch_id,
            change_kind=effective_kind,
            constraint_change_ids=list(dict.fromkeys(constraint_change_ids)),
            evidence_added_ids=sorted(new_evidence_ids - old_evidence_ids),
            evidence_removed_ids=sorted(old_evidence_ids - new_evidence_ids),
            reused_evidence_ids=sorted(reused_current_ids),
            claim_changes=claim_changes,
            retrieval_usage=retrieval_usage,
            generation_usage=generation_usage,
            presentation_usage=presentation_usage,
            retrieval_call_count=retrieval_call_count,
            retrieval_attempt_count=retrieval_attempt_count,
            generation_attempts=generation_attempts,
            generation_repair_attempts=generation_repair_attempts,
            retrieval_model_identity=retrieval_model_identity,
            generation_execution_mode=generation_execution_mode,
            generation_provider=generation_provider,
            generation_model=generation_model,
            presentation_execution_mode=presentation_execution_mode,
            presentation_provider=presentation_provider,
            status=publication_status,
            failure_reason=final_failure_reason,
            unresolved_intent_ids=list(dict.fromkeys(effective_unresolved)),
            unresolved_questions=effective_unresolved_questions,
        )
        pending_requests = list(state.pending_requests)
        if generation_request_id is not None:
            pending_requests = [
                item for item in pending_requests if item.request_id != generation_request_id
            ]
            pending_requests.append(
                Phase4PendingRequest(
                    request_id=generation_request_id,
                    kind=(
                        "presentation"
                        if effective_kind == "presentation"
                        else "answer_publication"
                    ),
                    base_revision=state.state_revision,
                    status="completed",
                )
            )
        revision_record = Phase4RevisionRecord(
            revision_id="revision-" + _digest(state.session_id, revision, f"answer-v{next_version}"),
            session_id=state.session_id,
            utterance_id=triggering_utterance_id or state.current_utterance_id,
            state_revision=revision,
            transcript_revision=state.transcript_revision,
            retrieval_revision=state.retrieval_revision,
            supersedes=[state.state_revision],
            patch_id=triggering_patch_id,
            changed_intent_ids=[
                item.intent_id
                for item in state.active_intents
                if item.intent_id in effective_unresolved
            ],
            preserved_intent_ids=[
                item.intent_id
                for item in state.active_intents
                if item.intent_id not in effective_unresolved
            ],
            discarded_result_ids=list(
                dict.fromkeys(
                    [
                        *(
                            state.pending_update.discarded_result_ids
                            if state.pending_update is not None
                            else []
                        ),
                    ]
                )
            ),
            reason=(
                "presentation-only answer version published from current valid claims"
                if effective_kind == "presentation"
                else "factual answer version published from preserved and reassessed claims"
            ),
            created_at_utc=_now(),
            directly_affected_intent_ids=(
                state.pending_update.directly_affected_intent_ids
                if state.pending_update is not None
                else []
            ),
            invalidated_claim_ids=(
                state.pending_update.invalidated_claim_ids
                if state.pending_update is not None
                else []
            ),
            preserved_claim_ids=(
                state.pending_update.preserved_claim_ids
                if state.pending_update is not None
                else [
                    change.previous_record_id
                    for change in claim_changes
                    if change.change_type == "preserved" and change.previous_record_id is not None
                ]
            ),
            reused_evidence_ids=sorted(reused_current_ids),
            retrieval_reasons=(
                state.pending_update.retrieval_reasons
                if state.pending_update is not None
                else {}
            ),
            dependency_uncertain=bool(
                state.pending_update.dependency_uncertain
                if state.pending_update is not None
                else False
            ),
            dependency_uncertainty_reasons=(
                state.pending_update.dependency_uncertainty_reasons
                if state.pending_update is not None
                else []
            ),
            answer_version=next_version,
            parent_answer_version=parent_version,
            publication_status=publication_status,
            claim_changes=claim_changes,
            evidence_added_ids=answer_version.evidence_added_ids,
            evidence_removed_ids=answer_version.evidence_removed_ids,
            generation_usage=generation_usage,
            retrieval_usage=retrieval_usage,
            presentation_usage=presentation_usage,
            retrieval_call_count=retrieval_call_count,
            generation_attempts=generation_attempts,
        )
        updates = {
            "state_revision": revision,
            "answer_versions": [*state.answer_versions, answer_version],
            "current_answer_version": next_version,
            "claim_records": [*claim_records, *new_records],
            "current_claim_record_ids": [item.record_id for item in new_records],
            "pending_update": None,
            "pending_clarification": None,
            "pending_requests": pending_requests,
            "generation_status": "published",
            "answer_status": (
                "current"
                if publication_status == "completed"
                else "partial"
                if publication_status == "partial"
                else "failed"
            ),
            "trace_ids": list(dict.fromkeys([*state.trace_ids, f"answer-v{next_version}"])),
            "revision_history": [*state.revision_history, revision_record],
        }
        return self._write(state.model_copy(update=updates))

    @staticmethod
    def _carried_claim_dependencies(
        source_record: Phase4ClaimRecord | None,
        current_records: Sequence[Phase4ClaimRecord],
    ) -> list[Phase4ClaimDependency]:
        """Carry only claim-to-claim edges whose target remains current."""

        if source_record is None:
            return []
        current_by_id = {item.claim_id: item for item in current_records}
        current_by_record = {item.record_id: item for item in current_records}
        carried: list[Phase4ClaimDependency] = []
        for dependency in source_record.dependencies:
            if dependency.dependency_type != "claim":
                continue
            target = current_by_record.get(dependency.target_id) or current_by_id.get(dependency.target_id)
            if target is None:
                continue
            carried.append(dependency.model_copy(update={"target_id": target.claim_id}))
        return carried

    def publish_presentation(
        self,
        session_id: str,
        *,
        expected_revision: int | None = None,
        rendered_answer_text: str | None = None,
        presentation_usage: Usage | None = None,
        presentation_execution_mode: str = "rule_based",
        presentation_provider: str = "flowcontext.rule_based",
        generation_request_id: str | None = None,
    ) -> Phase4SessionState:
        """Publish a presentation-only version from the current valid claims.

        This method never consults the corpus or retrieval scheduler.  A
        caller may provide a provider-produced translation, but the provider
        result must retain every current citation tag before it reaches the
        atomic answer publication boundary.
        """

        with self._lock:
            state = self._get_internal(session_id)
            if expected_revision is not None and expected_revision != state.state_revision:
                raise PatchConflictError("presentation publication belongs to an obsolete revision")
            current_version = state.usable_answer
            if current_version is None or not state.current_claims:
                raise PatchValidationError("presentation requires a usable current answer")
            pending = state.pending_update
            if pending is None or pending.requested_kind != "reformat_answer":
                raise PatchValidationError("no presentation update is pending")
            answer = current_version.answer.model_copy(
                update={"factual_claims": [record.claim for record in state.current_claims]}
            )
            if rendered_answer_text is None:
                if pending.format_instruction.startswith("translate"):
                    raise PatchValidationError(
                        "translation presentation requires a translation-capable provider"
                    )
                rendered_answer_text = _render_presentation(
                    answer,
                    [record.claim for record in state.current_claims],
                    pending.format_instruction or "requested_format",
                )
            citation_ids = {
                f"[{chunk_id}]"
                for record in state.current_claims
                for chunk_id in record.claim.supporting_chunk_ids
            }
            cited_text = set(re.findall(r"\[([^\]]+)\]", rendered_answer_text))
            expected_ids = {
                chunk_id.strip("[]")
                for chunk_id in citation_ids
            }
            unknown_ids = cited_text - expected_ids
            if unknown_ids:
                raise PatchValidationError(
                    "presentation output introduced unknown citation IDs: "
                    + ", ".join(sorted(unknown_ids))
                )
            if not expected_ids <= cited_text:
                raise PatchValidationError(
                    "presentation output dropped one or more current citation IDs"
                )
            state = self.publish_answer(
                session_id,
                answer,
                expected_revision=state.state_revision,
                triggering_patch_id=pending.patch_id,
                triggering_turn=pending.triggering_turn,
                triggering_utterance_id=pending.triggering_utterance_id,
                change_kind="presentation",
                status="completed",
                presentation_usage=presentation_usage or Usage(),
                retrieval_usage=Usage(),
                generation_usage=Usage(),
                retrieval_call_count=0,
                retrieval_attempt_count=0,
                generation_attempts=0,
                generation_repair_attempts=0,
                retrieval_model_identity=current_version.retrieval_model_identity,
                generation_execution_mode="rule_based",
                generation_provider="flowcontext.presentation",
                generation_model="none",
                presentation_execution_mode=presentation_execution_mode,
                presentation_provider=presentation_provider,
                unresolved_intent_ids=current_version.unresolved_intent_ids,
                unresolved_questions=current_version.unresolved_questions,
                generation_request_id=generation_request_id,
                rendered_answer_text=rendered_answer_text,
            )
            return state

    def _claim_dependencies(
        self,
        state: Phase4SessionState,
        claim: FactualClaim,
        supporting: Sequence[Phase4EvidenceRecord],
        extra: Sequence[Phase4ClaimDependency],
        known_claim_ids: set[str],
    ) -> list[Phase4ClaimDependency]:
        dependencies: list[Phase4ClaimDependency] = []
        for evidence in supporting:
            dependencies.append(
                Phase4ClaimDependency(
                    dependency_id="dependency-" + _digest("evidence", evidence.evidence_id, claim.claim_id),
                    dependency_type="evidence",
                    target_id=evidence.evidence_id,
                    relation="supports",
                )
            )
        claim_intents = set(claim.intent_ids)
        for constraint in state.current_constraints:
            if constraint.scope == "shared" or constraint.intent_id in claim_intents:
                dependencies.append(
                    Phase4ClaimDependency(
                        dependency_id="dependency-" + _digest(
                            "entity" if constraint.constraint.kind == "entity" else "constraint",
                            constraint.record_id,
                            claim.claim_id,
                        ),
                        dependency_type=(
                            "entity" if constraint.constraint.kind == "entity" else "constraint"
                        ),
                        target_id=constraint.record_id,
                        relation="depends_on",
                        target_value=constraint.constraint.value,
                    )
                )
        for dependency in extra:
            if dependency.dependency_type == "claim" and dependency.target_id not in known_claim_ids:
                raise PatchValidationError(
                    f"claim dependency {dependency.target_id!r} is not an existing or published claim"
                )
            if dependency.dependency_id not in {item.dependency_id for item in dependencies}:
                dependencies.append(dependency)
        return dependencies


class Phase4AnswerPublisher:
    """Run revision-bound synthesis and publish one complete answer envelope.

    The publisher is deliberately a thin bridge: retrieval is owned by the
    selective-update coordinator, generation by the existing generation
    provider abstraction, and the store is the only atomic publication
    boundary.  A provider result is never allowed to choose the session
    revision that it may update.
    """

    def __init__(
        self,
        store: Phase4SessionStore,
        corpus: CorpusIndex,
        *,
        generation_provider: Any | None = None,
        verifier: Any | None = None,
    ) -> None:
        from .generation import MockGenerationProvider

        self.store = store
        self.corpus = corpus
        self.generation_provider = generation_provider or MockGenerationProvider()
        self.verifier = verifier

    @staticmethod
    def _provider_metadata(provider: Any) -> tuple[str, str, str]:
        config = getattr(provider, "config", None)
        backend = getattr(config, "backend", None)
        if backend == "mock":
            mode = "mock_provider"
        elif backend == "openai_compatible":
            mode = "real_provider"
        else:
            raise PatchValidationError(
                "provider execution metadata must declare backend=mock or backend=openai_compatible"
            )
        provider_name = str(getattr(config, "provider", "")).strip()
        model = str(getattr(config, "model", "")).strip()
        if not provider_name or not model:
            raise PatchValidationError("provider execution metadata must include provider and model")
        return mode, provider_name, model

    def _hits_from_state(self, state: Phase4SessionState) -> list[RetrievalHit]:
        if state.corpus_id != self.corpus.corpus_id:
            raise PatchValidationError(
                "answer generation corpus identity does not match the session corpus"
            )
        if state.index_id != self.corpus.manifest.index_id:
            raise PatchValidationError(
                "answer generation index identity does not match the session index"
            )
        chunks = {chunk.chunk_id: chunk for chunk in self.corpus.chunks}
        hits: list[RetrievalHit] = []
        for evidence in state.current_evidence:
            passage = evidence.passage
            chunk = chunks.get(passage.chunk_id)
            if chunk is None:
                raise PatchValidationError(
                    f"active evidence {evidence.evidence_id!r} references an unknown corpus chunk"
                )
            if (
                evidence.corpus_id != state.corpus_id
                or evidence.index_id != state.index_id
                or passage.source_location != chunk.source_location
                or passage.text != chunk.text
            ):
                raise PatchValidationError(
                    f"active evidence {evidence.evidence_id!r} failed structural corpus provenance validation"
                )
            intent_ids = list(dict.fromkeys(passage.intent_ids or evidence.intent_ids))
            intent_id = passage.intent_id or (intent_ids[0] if len(intent_ids) == 1 else None)
            hits.append(
                RetrievalHit(
                    chunk_id=passage.chunk_id,
                    source_location=passage.source_location,
                    snippet_text=passage.text,
                    rank=passage.rank,
                    score=passage.score,
                    retrieval_method=passage.retrieval_method,
                    intent_id=intent_id,
                    intent_ids=intent_ids,
                    intent_ranks=dict(passage.intent_ranks),
                    intent_scores=dict(passage.intent_scores),
                    rrf_score=passage.rrf_score,
                )
            )
        return hits

    @staticmethod
    def _failure_answer(state: Phase4SessionState, reason: str) -> Answer:
        statuses = [
            IntentStatusRecord(
                intent_id=record.intent_id,
                status="insufficient_evidence",
                reason=reason,
            )
            for record in state.active_intents
        ]
        question_text = "; ".join(record.intent.query for record in state.active_intents)
        return Answer(
            answer_text=(
                "The requested information could not be safely reassessed from the configured corpus."
                + (f" Requested information needs: {question_text}." if question_text else "")
            ),
            factual_claims=[],
            uncertainty=f"{reason} No unsupported claim was returned.",
            answer_version=1,
            intent_statuses=statuses,
        )

    @staticmethod
    def _request_id(
        state: Phase4SessionState,
        *,
        expected_revision: int,
        triggering_utterance_id: str | None,
        request_id: str | None,
    ) -> str:
        return request_id or (
            "generation-"
            + _digest(
                state.session_id,
                expected_revision,
                triggering_utterance_id or state.current_utterance_id,
                state.pending_update.patch_id if state.pending_update else "initial",
            )
        )

    @staticmethod
    def _trigger_metadata(
        state: Phase4SessionState,
        *,
        triggering_patch_id: str | None,
        triggering_turn: int | None,
        triggering_utterance_id: str | None,
    ) -> tuple[str | None, int, str]:
        pending = state.pending_update
        return (
            triggering_patch_id or (pending.patch_id if pending else None),
            triggering_turn if triggering_turn is not None else (pending.triggering_turn if pending else 0),
            triggering_utterance_id or (pending.triggering_utterance_id if pending else state.current_utterance_id),
        )

    async def generate_and_publish(
        self,
        session_id: str,
        *,
        expected_revision: int | None = None,
        triggering_patch_id: str | None = None,
        triggering_turn: int | None = None,
        triggering_utterance_id: str | None = None,
        retrieval_usage: Usage | None = None,
        retrieval_call_count: int = 0,
        retrieval_attempt_count: int = 0,
        retrieval_model_identity: str = "unknown",
        request_id: str | None = None,
    ) -> Phase4PublicationResult:
        """Generate from the exact candidate revision and publish if still current."""

        from .generation import generate_grounded_answer

        state = self.store.get(session_id)
        expected = state.state_revision if expected_revision is None else expected_revision
        if expected != state.state_revision:
            return Phase4PublicationResult(
                state=state,
                published=False,
                status="superseded",
                reason="answer generation target is not the current session revision",
            )
        mode, provider_name, provider_model = self._provider_metadata(self.generation_provider)
        request = self._request_id(
            state,
            expected_revision=expected,
            triggering_utterance_id=triggering_utterance_id,
            request_id=request_id,
        )
        self.store.begin_answer_generation(
            session_id,
            request_id=request,
            expected_revision=expected,
        )

        def is_superseded() -> bool:
            try:
                current = self.store.get(session_id)
            except SessionNotFoundError:
                return True
            return current.state_revision != expected

        try:
            hits = self._hits_from_state(state)
            decomposition = _phase4_decomposition(state)
            evidence_intent_ids = {
                intent_id
                for evidence in state.current_evidence
                for intent_id in evidence.intent_ids
            }
            unsupported = [
                record.intent.query
                for record in state.active_intents
                if record.intent_id not in evidence_intent_ids
            ]
            outcome = await generate_grounded_answer(
                state.active_task,
                hits,
                self.corpus,
                self.generation_provider,
                decomposition=decomposition,
                unsupported_intent_queries=unsupported,
                verifier=self.verifier,
                is_superseded=is_superseded,
                transcript_revision=state.transcript_revision,
            )
        except Exception as exc:
            if is_superseded():
                current = self.store.record_discarded_generation(
                    session_id,
                    request,
                    reason="generation failed after its candidate revision was superseded",
                )
                return Phase4PublicationResult(
                    state=current,
                    published=False,
                    status="superseded",
                    request_id=request,
                    reason="generation work belonged to an obsolete revision",
                )
            # A provider or structural generation failure is itself a
            # publishable candidate: preserved claims are carried by the store,
            # while affected intents become explicit unresolved information.
            failure = self._failure_answer(state, f"Answer generation failed: {type(exc).__name__}: {exc}")
            failure_state = self.store.get(session_id)
            failure_patch_id, failure_turn, failure_utterance_id = self._trigger_metadata(
                failure_state,
                triggering_patch_id=triggering_patch_id,
                triggering_turn=triggering_turn,
                triggering_utterance_id=triggering_utterance_id,
            )
            try:
                published = self.store.publish_answer(
                    session_id,
                    failure,
                    expected_revision=expected,
                    triggering_patch_id=failure_patch_id,
                    triggering_turn=failure_turn,
                    triggering_utterance_id=failure_utterance_id,
                    change_kind="initial" if state.current_answer_version is None else "factual_update",
                    failure_reason=str(exc),
                    constraint_change_ids=(
                        failure_state.pending_update.constraint_change_ids
                        if failure_state.pending_update is not None
                        else []
                    ),
                    retrieval_usage=retrieval_usage or Usage(),
                    retrieval_call_count=retrieval_call_count,
                    retrieval_attempt_count=retrieval_attempt_count,
                    generation_attempts=getattr(exc, "attempts", 0),
                    retrieval_model_identity=retrieval_model_identity,
                    generation_execution_mode=mode,
                    generation_provider=provider_name,
                    generation_model=provider_model,
                    generation_request_id=request,
                )
            except (PatchConflictError, PatchValidationError) as publication_error:
                current = self.store.record_discarded_generation(
                    session_id,
                    request,
                    reason=f"failed answer publication was not current: {publication_error}",
                )
                return Phase4PublicationResult(
                    state=current,
                    published=False,
                    status="superseded" if isinstance(publication_error, PatchConflictError) else "failed",
                    request_id=request,
                    reason=str(publication_error),
                )
            return Phase4PublicationResult(
                state=published,
                published=True,
                status=published.answer_status,
                request_id=request,
                reason=str(exc),
            )

        if outcome.status == "stale_rejected" or is_superseded():
            current = self.store.record_discarded_generation(
                session_id,
                request,
                reason="generation result was returned for a superseded candidate revision",
            )
            return Phase4PublicationResult(
                state=current,
                published=False,
                status="superseded",
                request_id=request,
                outcome=outcome,
                reason=outcome.error_message or "candidate revision was superseded",
            )

        patch_id, turn, utterance_id = self._trigger_metadata(
            self.store.get(session_id),
            triggering_patch_id=triggering_patch_id,
            triggering_turn=triggering_turn,
            triggering_utterance_id=triggering_utterance_id,
        )
        try:
            published = self.store.publish_answer(
                session_id,
                outcome.answer,
                expected_revision=expected,
                triggering_patch_id=patch_id,
                triggering_turn=turn,
                triggering_utterance_id=utterance_id,
                change_kind="initial" if state.current_answer_version is None else "factual_update",
                failure_reason=outcome.error_message,
                retrieval_usage=retrieval_usage or Usage(),
                generation_usage=outcome.usage,
                retrieval_call_count=retrieval_call_count,
                retrieval_attempt_count=retrieval_attempt_count,
                generation_attempts=outcome.attempts,
                generation_repair_attempts=outcome.repair_attempts,
                retrieval_model_identity=retrieval_model_identity,
                generation_execution_mode=mode,
                generation_provider=provider_name,
                generation_model=provider_model,
                generation_request_id=request,
            )
        except (PatchConflictError, PatchValidationError) as exc:
            current = self.store.record_discarded_generation(
                session_id,
                request,
                reason=f"answer publication rejected: {exc}",
            )
            return Phase4PublicationResult(
                state=current,
                published=False,
                status="superseded" if isinstance(exc, PatchConflictError) else "failed",
                request_id=request,
                outcome=outcome,
                reason=str(exc),
            )
        return Phase4PublicationResult(
            state=published,
            published=True,
            status=published.answer_status,
            request_id=request,
            outcome=outcome,
        )

    @staticmethod
    async def _translation_output(provider: Any, answer: Answer, language: str) -> tuple[str, Usage]:
        method = getattr(provider, "translate", None)
        if not callable(method):
            raise PatchValidationError("translation requires a provider with a translate method")
        try:
            value = method(answer=answer, language=language)
        except TypeError:
            value = method(answer, language)
        if inspect.isawaitable(value):
            value = await value
        usage = Usage()
        if isinstance(value, tuple) and len(value) == 2:
            value, possible_usage = value
            if isinstance(possible_usage, Usage):
                usage = possible_usage
            else:
                usage = Usage.model_validate(possible_usage)
        if isinstance(value, Answer):
            value = value.answer_text
        elif not isinstance(value, str):
            value = getattr(value, "answer_text", getattr(value, "text", value))
        if not isinstance(value, str) or not value.strip():
            raise PatchValidationError("translation provider returned empty presentation text")
        return value.strip(), usage

    async def present_and_publish(
        self,
        session_id: str,
        *,
        expected_revision: int | None = None,
        presentation_provider: Any | None = None,
        request_id: str | None = None,
    ) -> Phase4PublicationResult:
        """Publish a formatting-only version without retrieval or generation."""

        state = self.store.get(session_id)
        expected = state.state_revision if expected_revision is None else expected_revision
        if expected != state.state_revision:
            raise PatchConflictError("presentation target is not the current session revision")
        pending = state.pending_update
        answer = state.usable_answer
        if pending is None or pending.requested_kind != "reformat_answer":
            raise PatchValidationError("no presentation update is pending")
        if answer is None or not state.current_claims:
            current = self.store.get(session_id)
            return Phase4PublicationResult(
                state=current,
                published=False,
                status="failed",
                reason="No usable answer exists; context is required before reformatting.",
            )
        request = request_id or (
            "presentation-"
            + _digest(session_id, expected, pending.patch_id, pending.format_instruction)
        )
        self.store.begin_answer_generation(
            session_id,
            request_id=request,
            expected_revision=expected,
            request_kind="presentation",
        )
        mode = "rule_based"
        provider_name = "flowcontext.presentation"
        presentation_usage = Usage()
        try:
            instruction = pending.format_instruction or "requested_format"
            if instruction.casefold().startswith("translate"):
                if presentation_provider is None:
                    raise PatchValidationError(
                        "translation presentation requires a translation-capable provider"
                    )
                config = getattr(presentation_provider, "config", None)
                backend = getattr(config, "backend", None)
                if backend == "mock":
                    mode = "mock_provider"
                elif backend == "openai_compatible":
                    mode = "real_provider"
                else:
                    raise PatchValidationError(
                        "translation provider execution metadata must declare backend=mock or backend=openai_compatible"
                    )
                provider_name = str(getattr(config, "provider", "")).strip()
                if not provider_name:
                    raise PatchValidationError("translation provider metadata must include provider")
                language = instruction.partition(":")[2].strip() or "the requested language"
                rendered, presentation_usage = await self._translation_output(
                    presentation_provider,
                    answer.answer,
                    language,
                )
            else:
                rendered = _render_presentation(
                    answer.answer,
                    [record.claim for record in state.current_claims],
                    instruction,
                )
            if self.store.get(session_id).state_revision != expected:
                raise PatchConflictError("presentation result belongs to an obsolete revision")
            published = self.store.publish_presentation(
                session_id,
                expected_revision=expected,
                rendered_answer_text=rendered,
                presentation_usage=presentation_usage,
                presentation_execution_mode=mode,
                presentation_provider=provider_name,
                generation_request_id=request,
            )
        except (PatchConflictError, PatchValidationError) as exc:
            current = self.store.record_discarded_generation(
                session_id,
                request,
                reason=f"presentation publication rejected: {exc}",
                request_kind="presentation",
            )
            return Phase4PublicationResult(
                state=current,
                published=False,
                status="superseded" if isinstance(exc, PatchConflictError) else "failed",
                request_id=request,
                reason=str(exc),
            )
        return Phase4PublicationResult(
            state=published,
            published=True,
            status=published.answer_status,
            request_id=request,
        )


class _Phase4SchedulerController:
    """Small controller bridge that gives the Phase 2 scheduler Phase 4 guards."""

    def __init__(
        self,
        *,
        session_id: str,
        known_chunk_locations: Mapping[str, str] | None = None,
    ) -> None:
        self.session_id = session_id
        self._known_chunk_locations = dict(known_chunk_locations or {})
        self._candidate_state_revision = 0
        self._candidate_transcript_revision = 0
        self._candidate_retrieval_revision = 0
        self._work_counter = 0
        self._request_candidates: dict[str, int] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def transcript_revision(self) -> int:
        return self._candidate_transcript_revision

    @property
    def latest_retrieval_revision(self) -> int:
        return self._work_counter

    @property
    def current_text(self) -> str:
        return ""

    def advance_candidate(
        self,
        *,
        state_revision: int,
        transcript_revision: int,
        retrieval_revision: int,
    ) -> None:
        if state_revision < self._candidate_state_revision:
            raise PatchConflictError("scheduler candidate revision moved backwards")
        self._candidate_state_revision = state_revision
        self._candidate_transcript_revision = transcript_revision
        self._candidate_retrieval_revision = retrieval_revision

    def begin_retrieval(
        self,
        decision: StreamingDecision,
        *,
        query_override: str | None = None,
    ) -> StreamingRetrievalRequest:
        if self._closed:
            raise PatchConflictError("Phase 4 scheduler controller is closed")
        if decision.session_id != self.session_id:
            raise PatchConflictError("retrieval decision belongs to another session")
        query = query_override or decision.proposed_query
        if query is None or not query.strip():
            raise PatchValidationError("targeted retrieval requires a non-empty query")
        if decision.transcript_revision != self._candidate_transcript_revision:
            raise PatchConflictError("retrieval decision does not match the candidate transcript revision")
        self._work_counter += 1
        request_id = "phase4-retrieval-" + _digest(
            self.session_id,
            decision.decision_id,
            self._work_counter,
            query,
        )
        self._request_candidates[request_id] = self._candidate_state_revision
        started = time.monotonic()
        return StreamingRetrievalRequest(
            request_id=request_id,
            decision_id=decision.decision_id,
            session_id=self.session_id,
            utterance_id=decision.utterance_id,
            transcript_revision=decision.transcript_revision,
            retrieval_revision=self._work_counter,
            query=query,
            source_timestamp_s=decision.source_timestamp_s,
            monotonic_started_s=started,
        )

    def complete_retrieval(
        self,
        request: StreamingRetrievalRequest,
        hits: Sequence[RetrievalHit],
        *,
        monotonic_completed_s: float | None = None,
    ) -> StreamingRetrievalResult:
        if request.session_id != self.session_id:
            raise PatchConflictError("retrieval result belongs to another session")
        for hit in hits:
            expected_location = self._known_chunk_locations.get(hit.chunk_id)
            if expected_location is not None and expected_location != hit.source_location:
                raise PatchValidationError(
                    f"retrieval hit {hit.chunk_id!r} has an inconsistent source location"
                )
        completed = monotonic_completed_s if monotonic_completed_s is not None else time.monotonic()
        stale = self._request_candidates.get(request.request_id) != self._candidate_state_revision
        return StreamingRetrievalResult(
            request_id=request.request_id,
            decision_id=request.decision_id,
            session_id=request.session_id,
            utterance_id=request.utterance_id,
            transcript_revision=request.transcript_revision,
            retrieval_revision=request.retrieval_revision,
            query=request.query,
            hits=list(hits),
            accepted=not stale,
            stale=stale,
            source_timestamp_s=request.source_timestamp_s,
            monotonic_started_s=request.monotonic_started_s,
            monotonic_completed_s=max(completed, request.monotonic_started_s),
            duration_ms=max(0.0, (completed - request.monotonic_started_s) * 1000),
        )

    def close(self) -> None:
        self._closed = True


@dataclass
class _Phase4Runtime:
    controller: _Phase4SchedulerController
    scheduler: Any
    request_by_task: dict[str, str] = field(default_factory=dict)
    consumers_by_task: dict[str, asyncio.Task[None]] = field(default_factory=dict)


class Phase4SelectiveUpdateCoordinator:
    """Apply candidate patches and execute only their unresolved retrieval tasks.

    ``apply_patch`` is synchronous at the state boundary.  If called inside an
    asyncio loop it dispatches work immediately; callers outside a loop can
    call :meth:`execute_plan` later.  Retrieval itself uses
    :class:`AsyncRetrievalScheduler`, including its bounded executor, retry,
    cancellation optimisation, and late-result observer.
    """

    def __init__(
        self,
        store: Phase4SessionStore,
        retriever: Any,
        *,
        scheduler_config: Any | None = None,
        retrieval_model_identity: str = "flowcontext.phase4.selective-retrieval.v1",
        traces: Any | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.scheduler_config = scheduler_config
        self.retrieval_model_identity = retrieval_model_identity
        self.traces = traces
        self._runtimes: OrderedDict[str, _Phase4Runtime] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def session_ids(self) -> list[str]:
        with self._lock:
            return list(self._runtimes)

    def _known_chunk_locations(self) -> dict[str, str]:
        candidates: list[Any] = [self.retriever]
        base = getattr(self.retriever, "base", None)
        if base is not None:
            candidates.append(base)
        for candidate in candidates:
            index = getattr(candidate, "index", None)
            chunks = getattr(index, "chunks", None)
            if chunks is not None:
                return {chunk.chunk_id: chunk.source_location for chunk in chunks}
        return {}

    def _trace(self, session_id: str, event_type: str, **attributes: Any) -> None:
        trace = self.traces
        if trace is None:
            return
        trace.add(event_type, model_identity=self.retrieval_model_identity, attributes=attributes)

    def _runtime_for(self, state: Phase4SessionState) -> _Phase4Runtime:
        from .scheduler import AsyncRetrievalScheduler

        with self._lock:
            runtime = self._runtimes.get(state.session_id)
            if runtime is not None:
                self._runtimes.move_to_end(state.session_id)
                return runtime
            controller = _Phase4SchedulerController(
                session_id=state.session_id,
                known_chunk_locations=self._known_chunk_locations(),
            )
            controller.advance_candidate(
                state_revision=state.state_revision,
                transcript_revision=state.transcript_revision,
                retrieval_revision=state.retrieval_revision,
            )
            trace = self.traces
            if trace is None:
                from .trace import TraceCollector

                trace = TraceCollector(
                    f"phase4-{state.session_id}",
                    state.session_id,
                    self.retrieval_model_identity,
                )
            scheduler = AsyncRetrievalScheduler(
                controller=controller,
                retriever=self.retriever,
                traces=trace,
                retrieval_model_identity=self.retrieval_model_identity,
                config=self.scheduler_config,
            )
            runtime = _Phase4Runtime(controller=controller, scheduler=scheduler)
            self._runtimes[state.session_id] = runtime
            self._runtimes.move_to_end(state.session_id)
            while len(self._runtimes) > self.store.max_sessions:
                _evicted_session_id, evicted = self._runtimes.popitem(last=False)
                evicted.controller.close()
                # ``_runtime_for`` is reached from an active event loop when
                # a scheduler can exist.  Closing the executor asynchronously
                # prevents LRU eviction from leaking worker threads.
                asyncio.create_task(evicted.scheduler.close())
            return runtime

    @staticmethod
    def _decision_for_task(
        state: Phase4SessionState,
        task: Phase4SelectiveRetrievalTask,
    ) -> StreamingDecision:
        return StreamingDecision(
            decision_id=task.task_id,
            session_id=state.session_id,
            utterance_id=state.current_utterance_id,
            transcript_revision=task.candidate_transcript_revision,
            decision="RETRIEVE",
            reason_code="phase4_targeted_retrieval",
            proposed_query=task.query,
            source_event_id=task.patch_id,
            source_timestamp_s=float(task.candidate_transcript_revision),
            monotonic_decision_time_s=time.monotonic(),
            is_final_event=True,
            coalesced_event_ids=[task.patch_id],
        )

    def _dispatch_plan(
        self,
        state: Phase4SessionState,
        plan: Phase4SelectiveUpdatePlan,
        runtime: _Phase4Runtime,
    ) -> None:
        runtime.controller.advance_candidate(
            state_revision=plan.candidate_revision,
            transcript_revision=plan.transcript_revision,
            retrieval_revision=plan.retrieval_revision,
        )
        for task in plan.retrieval_tasks:
            if task.task_id in runtime.request_by_task:
                continue
            decision = self._decision_for_task(state, task)
            try:
                request = runtime.scheduler.schedule(
                    decision,
                    allow_parallel_queries=True,
                )
            except Exception as exc:
                self.store.record_selective_task_result(
                    state.session_id,
                    plan.plan_id,
                    task.task_id,
                    status="failed",
                    attempts=0,
                    error=f"scheduler rejected targeted work: {exc}",
                )
                self._trace(
                    state.session_id,
                    "phase4_selective_retrieval_schedule_failed",
                    plan_id=plan.plan_id,
                    task_id=task.task_id,
                    reason=str(exc),
                )
                continue
            runtime.request_by_task[task.task_id] = request.request_id
            self.store.record_selective_task_scheduled(
                state.session_id,
                plan.plan_id,
                task.task_id,
                request.request_id,
            )
            consumer = asyncio.create_task(
                self._consume_task_result(
                    state.session_id,
                    plan.plan_id,
                    task.task_id,
                    runtime,
                ),
                name=f"flowcontext-phase4-consume-{task.task_id}",
            )
            runtime.consumers_by_task[task.task_id] = consumer
            self._trace(
                state.session_id,
                "phase4_selective_retrieval_scheduled",
                plan_id=plan.plan_id,
                task_id=task.task_id,
                intent_id=task.intent_id,
                query=task.query,
                reason=task.reason,
                candidate_revision=task.candidate_state_revision,
                retrieval_revision=task.candidate_retrieval_revision,
            )

    def apply_patch(
        self,
        patch: Phase4ProposedPatch,
    ) -> Phase4SelectiveUpdatePlan | None:
        """Apply a validated patch and return its selective update plan."""

        state = self.store.apply_patch(patch)
        runtime: _Phase4Runtime | None = None
        with self._lock:
            existing = self._runtimes.get(state.session_id)
            if existing is not None:
                # Advance the bridge before requesting cancellation.  A worker
                # that ignores cancellation is therefore stale immediately.
                existing.controller.advance_candidate(
                    state_revision=state.state_revision,
                    transcript_revision=state.transcript_revision,
                    retrieval_revision=state.retrieval_revision,
                )
                existing.scheduler.supersede_all(
                    reason="new Phase 4 candidate revision superseded prior work"
                )
                runtime = existing
        if patch.classification in {"clarification_required", "reformat_answer"}:
            return None
        plan = next(
            item
            for item in state.selective_update_plans
            if item.patch_id == patch.patch_id
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return plan
        if runtime is None:
            runtime = self._runtime_for(state)
        self._dispatch_plan(state, plan, runtime)
        return plan

    async def execute_plan(
        self,
        plan: Phase4SelectiveUpdatePlan,
        *,
        timeout_s: float | None = None,
    ) -> Phase4SessionState:
        """Dispatch queued work and await only this candidate's task set."""

        state = self.store.get(plan.session_id)
        if state.state_revision != plan.candidate_revision:
            self.store.record_discarded_selective_result(
                plan.session_id,
                plan.plan_id,
                plan.plan_id,
                reason="candidate plan was superseded before execution",
            )
            return self.store.get(plan.session_id)
        runtime = self._runtime_for(state)
        self._dispatch_plan(state, plan, runtime)
        current_plan = next(
            item
            for item in self.store.get(plan.session_id).selective_update_plans
            if item.plan_id == plan.plan_id
        )
        consumers = [
            consumer
            for task_id, consumer in runtime.consumers_by_task.items()
            if any(task.task_id == task_id for task in current_plan.retrieval_tasks)
        ]
        if consumers:
            group = asyncio.gather(*consumers, return_exceptions=True)
            if timeout_s is None:
                await group
            else:
                try:
                    await asyncio.wait_for(asyncio.shield(group), timeout=max(0.0, timeout_s))
                except asyncio.TimeoutError:
                    self._trace(
                        plan.session_id,
                        "phase4_selective_retrieval_wait_timeout",
                        plan_id=plan.plan_id,
                        timeout_s=timeout_s,
                    )
        plan_request_ids = {
            task.scheduler_request_id
            for task in current_plan.retrieval_tasks
            if task.scheduler_request_id is not None
        }
        plan_results = [
            result
            for result in runtime.scheduler.results
            if result.request_id in plan_request_ids
        ]
        usage_calls = [
            len(_TOKEN_PATTERN.findall(result.query))
            for result in plan_results
            for _ in range(result.attempts)
        ]
        usage = Usage(
            input_tokens=sum(usage_calls),
            output_tokens=0,
            total_tokens=sum(usage_calls),
            estimated=True,
        )
        try:
            self.store.record_selective_plan_usage(
                plan.session_id,
                plan.plan_id,
                usage=usage,
                call_count=sum(result.attempts for result in plan_results),
                attempt_count=sum(result.attempts for result in plan_results),
            )
        except PatchValidationError:
            # The bounded store may have evicted an old audit plan after a
            # newer turn; retrieval completion remains harmless in that case.
            pass
        return self.store.get(plan.session_id)

    async def apply_patch_and_wait(
        self,
        patch: Phase4ProposedPatch,
        *,
        timeout_s: float | None = None,
    ) -> Phase4SessionState:
        """Convenience method for a patch followed by targeted retrieval."""

        plan = self.apply_patch(patch)
        if plan is None:
            return self.store.get(patch.session_id)
        await self.execute_plan(plan, timeout_s=timeout_s)
        return self.store.get(patch.session_id)

    async def _consume_task_result(
        self,
        session_id: str,
        plan_id: str,
        task_id: str,
        runtime: _Phase4Runtime,
    ) -> None:
        """Consume a result, treating an explicitly cleared session as closed."""

        try:
            await self._consume_task_result_impl(session_id, plan_id, task_id, runtime)
        except SessionNotFoundError:
            # ``Phase4SessionStore.clear`` is intentionally independent of
            # this optional runtime coordinator.  A worker finishing after a
            # direct clear is therefore harmless and cannot recreate state.
            return

    async def _consume_task_result_impl(
        self,
        session_id: str,
        plan_id: str,
        task_id: str,
        runtime: _Phase4Runtime,
    ) -> None:
        request_id = runtime.request_by_task[task_id]
        result = await runtime.scheduler.wait_for_request(request_id)
        if result is None:
            self.store.record_selective_task_result(
                session_id,
                plan_id,
                task_id,
                status="failed",
                attempts=0,
                error="scheduler produced no terminal result within its wait bound",
            )
            return
        task = next(
            task
            for task in self.store.get(session_id).selective_update_plans
            if task.plan_id == plan_id
            for task in task.retrieval_tasks
            if task.task_id == task_id
        )
        if not result.accepted or result.stale or result.status != "completed":
            self.store.record_discarded_selective_result(
                session_id,
                plan_id,
                result.request_id,
                task_id=task_id,
                reason=(
                    "late retrieval result rejected for obsolete candidate revision"
                    if result.stale or result.status == "superseded"
                    else f"targeted retrieval did not complete: {result.status}"
                ),
            )
            self._trace(
                session_id,
                "phase4_selective_retrieval_discarded",
                plan_id=plan_id,
                task_id=task_id,
                result_id=result.request_id,
                stale=result.stale,
                status=result.status,
            )
            return
        evidence_ids: list[str] = []
        try:
            intent_record = next(
                item
                for item in self.store.get(session_id).active_intents
                if item.intent_id == task.intent_id
            )
            for hit in result.hits:
                alignment = intent_evidence_alignment(intent_record.intent, hit.snippet_text)
                if not alignment["aligned"]:
                    self._trace(
                        session_id,
                        "phase4_selective_candidate_rejected",
                        plan_id=plan_id,
                        task_id=task_id,
                        result_id=result.request_id,
                        chunk_id=hit.chunk_id,
                        reason=alignment["reason"],
                        semantic_support="not_evaluated",
                    )
                    continue
                passage = EvidencePassage(
                    chunk_id=hit.chunk_id,
                    source_location=hit.source_location,
                    text=hit.snippet_text,
                    rank=hit.rank,
                    score=hit.score,
                    retrieval_method=hit.retrieval_method,
                    intent_id=task.intent_id,
                    intent_ids=[task.intent_id],
                    metadata={"scheduler_request_id": result.request_id},
                )
                before = self.store.get(session_id)
                after = self.store.record_selective_evidence(
                    session_id,
                    task,
                    passage,
                    expected_revision=task.candidate_state_revision,
                )
                new_ids = set(after.current_evidence_ids) - set(before.current_evidence_ids)
                evidence_ids.extend(new_ids)
        except (PatchConflictError, PatchValidationError) as exc:
            self.store.record_discarded_selective_result(
                session_id,
                plan_id,
                result.request_id,
                task_id=task_id,
                reason=(
                    "session revision changed before targeted evidence admission"
                    if isinstance(exc, PatchConflictError)
                    else f"targeted evidence was rejected before admission: {exc}"
                ),
            )
            self._trace(
                session_id,
                "phase4_selective_retrieval_discarded",
                plan_id=plan_id,
                task_id=task_id,
                result_id=result.request_id,
                stale=isinstance(exc, PatchConflictError),
                status=(
                    "obsolete_session_revision"
                    if isinstance(exc, PatchConflictError)
                    else "evidence_rejected"
                ),
            )
            return
        self.store.record_selective_task_result(
            session_id,
            plan_id,
            task_id,
            status="completed",
            attempts=result.attempts,
            evidence_ids=evidence_ids,
        )
        self._trace(
            session_id,
            "phase4_selective_retrieval_completed",
            plan_id=plan_id,
            task_id=task_id,
            result_id=result.request_id,
            hit_count=len(result.hits),
            evidence_ids=evidence_ids,
            candidate_revision=task.candidate_state_revision,
        )

    async def close_session(self, session_id: str, *, clear_state: bool = False) -> None:
        """Close scheduler work for one session and optionally clear state."""

        with self._lock:
            runtime = self._runtimes.pop(session_id, None)
        if runtime is not None:
            runtime.controller.close()
            await runtime.scheduler.close()
        if clear_state:
            self.store.clear(session_id)

    async def close(self) -> None:
        """Close all bounded scheduler runtimes without creating persistence."""

        for session_id in list(self.session_ids):
            await self.close_session(session_id)

def interpret_follow_up(
    state: Phase4SessionState,
    request: Phase4FollowUpRequest,
    *,
    decomposer: StructuredMultiIntentDecomposer | Any | None = None,
) -> Phase4ProposedPatch:
    """Interpret one request without mutating the supplied state."""

    return Phase4FollowUpInterpreter(decomposer).interpret(state, request)


# Short aliases keep the implementation convenient for callers while the
# serialized contracts retain their explicit Phase 4 names.
SessionState = Phase4SessionState
ProposedPatch = Phase4ProposedPatch
FollowUpInterpreter = Phase4FollowUpInterpreter
SessionStore = Phase4SessionStore
SelectiveUpdatePlan = Phase4SelectiveUpdatePlan
SelectiveRetrievalTask = Phase4SelectiveRetrievalTask
SelectiveUpdateCoordinator = Phase4SelectiveUpdateCoordinator
Phase4SelectiveRetrievalCoordinator = Phase4SelectiveUpdateCoordinator
