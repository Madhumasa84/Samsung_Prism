"""Executable Phase 4 session replay and answer-publication workflow.

This module is intentionally small.  It assembles the existing retriever,
bounded Phase 4 scheduler, generation provider, and session store so CLI
examples produce inspectable state rather than hard-coded demonstration text.
The default generation path is the repository's explicitly labelled offline
mock provider; callers may inject a real provider for an integration run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Sequence

from .contracts import (
    CorpusIndex,
    EvidencePassage,
    Phase4ReplayResult,
    Phase4ReplayTurn,
    Phase4SessionState,
    RetrievalHit,
    Usage,
)
from .multi_intent import intent_evidence_alignment, decompose_query
from .phase4 import (
    PatchValidationError,
    Phase4AnswerPublisher,
    Phase4FollowUpInterpreter,
    Phase4SelectiveUpdateCoordinator,
    Phase4SessionStore,
)


def load_phase4_turns(path: Path) -> list[Phase4ReplayTurn]:
    """Load one session's compact JSONL replay turns."""

    import json

    turns: list[Phase4ReplayTurn] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            turns.append(Phase4ReplayTurn.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid Phase 4 replay turn at {path}:{line_number}: {exc}") from exc
    if not turns:
        raise ValueError(f"Phase 4 replay file {path} contains no turns")
    return turns


def _retrieval_usage(query: str, calls: int = 1) -> Usage:
    tokens = len(query.split()) * max(0, calls)
    return Usage(input_tokens=tokens, output_tokens=0, total_tokens=tokens, estimated=True)


def _intent_ids_for_hit(
    hit: RetrievalHit,
    state: Phase4SessionState,
) -> list[str]:
    direct = list(dict.fromkeys(hit.intent_ids or ([hit.intent_id] if hit.intent_id else [])))
    active = {item.intent_id for item in state.active_intents}
    direct = [item for item in direct if item in active]
    if direct:
        return direct
    if len(state.active_intents) == 1:
        intent = state.active_intents[0]
        return (
            [intent.intent_id]
            if intent_evidence_alignment(intent.intent, hit.snippet_text)["aligned"]
            else []
        )
    # A plain retriever has no intent labels.  Use the same structural
    # alignment gate as synthesis; a score or a shared chunk ID alone is not
    # enough to attach support to every active intent.
    return [
        intent.intent_id
        for intent in state.active_intents
        if intent_evidence_alignment(intent.intent, hit.snippet_text)["aligned"]
    ]


def _passage_for_hit(
    hit: RetrievalHit,
    state: Phase4SessionState,
    corpus: CorpusIndex,
) -> EvidencePassage | None:
    chunk = next((item for item in corpus.chunks if item.chunk_id == hit.chunk_id), None)
    if chunk is None or hit.source_location != chunk.source_location or hit.snippet_text != chunk.text:
        raise PatchValidationError(
            f"retrieval hit {hit.chunk_id!r} failed corpus provenance validation"
        )
    intent_ids = _intent_ids_for_hit(hit, state)
    if not intent_ids:
        return None
    return EvidencePassage(
        chunk_id=hit.chunk_id,
        source_location=hit.source_location,
        text=hit.snippet_text,
        rank=hit.rank,
        score=hit.score,
        retrieval_method=hit.retrieval_method,
        intent_id=intent_ids[0] if len(intent_ids) == 1 else None,
        intent_ids=intent_ids,
        intent_ranks=dict(hit.intent_ranks),
        intent_scores=dict(hit.intent_scores),
        rrf_score=hit.rrf_score,
        metadata=dict(chunk.metadata),
    )


async def _record_initial_retrieval(
    store: Phase4SessionStore,
    state: Phase4SessionState,
    corpus: CorpusIndex,
    retriever: Any,
    query: str,
) -> tuple[Phase4SessionState, Usage, int, list[str]]:
    """Run one real retriever call and admit only structurally valid passages."""

    notes: list[str] = []
    try:
        # Initial replay retrieval is a single bounded call.  Keep it on the
        # caller's event-loop turn so the CLI does not create an unmanaged
        # default executor thread; follow-up retrieval uses the scheduler's
        # explicitly bounded executor.
        hits = retriever.search(query)
    except Exception as exc:
        notes.append(f"initial retrieval failed: {type(exc).__name__}: {exc}")
        return state, _retrieval_usage(query), 1, notes
    state = store.get(state.session_id)
    retrieval_revision = state.retrieval_revision + 1
    for hit in hits:
        passage = _passage_for_hit(hit, state, corpus)
        if passage is None:
            notes.append(f"candidate {hit.chunk_id} was not intent-aligned and was not admitted")
            continue
        state = store.record_evidence(
            state.session_id,
            passage,
            intent_ids=passage.intent_ids,
            retrieval_revision=retrieval_revision,
            corpus_id=corpus.corpus_id,
            index_id=corpus.manifest.index_id,
            expected_revision=state.state_revision,
        )
        # One retrieval call can yield several chunks.  Keep all records on
        # the same retrieval revision for a coherent initial evidence set.
    return state, _retrieval_usage(query), 1, notes


def _step(
    turn: Phase4ReplayTurn,
    state: Phase4SessionState,
    *,
    classification: str,
    patch: Any | None = None,
    plan: Any | None = None,
    publication: Any | None = None,
) -> dict[str, Any]:
    return {
        "turn_index": turn.turn_index,
        "utterance_id": turn.utterance_id,
        "text": turn.text,
        "classification": classification,
        "patch": patch.model_dump(mode="json") if patch is not None else None,
        "plan": plan.model_dump(mode="json") if plan is not None else None,
        "publication": {
            "published": publication.published,
            "status": publication.status,
            "request_id": publication.request_id,
            "reason": publication.reason,
        }
        if publication is not None
        else None,
        "state_revision": state.state_revision,
        "answer_status": state.answer_status,
        "current_answer_version": state.current_answer_version,
        "current_claim_ids": [item.claim_id for item in state.current_claims],
    }


def _result(
    state: Phase4SessionState,
    *,
    corpus: CorpusIndex,
    retriever: Any,
    publisher: Phase4AnswerPublisher,
    steps: list[dict[str, Any]],
    notes: list[str],
) -> Phase4ReplayResult:
    config = getattr(publisher.generation_provider, "config", None)
    backend = getattr(config, "backend", None)
    execution_mode = (
        "mock_provider"
        if backend == "mock"
        else "real_provider"
        if backend == "openai_compatible"
        else "rule_based"
    )
    provider = getattr(config, "provider", "unknown")
    model = getattr(config, "model", "unknown")
    run_status = (
        "completed"
        if state.answer_status == "current"
        else "partial"
        if state.answer_status == "partial"
        else "failed"
    )
    retrieval_calls = sum(
        version.retrieval_call_count
        for version in state.answer_versions
        if version.change_kind != "presentation"
    )
    retrieval_attempts = sum(
        version.retrieval_attempt_count
        for version in state.answer_versions
        if version.change_kind != "presentation"
    )
    generation_attempts = sum(version.generation_attempts for version in state.answer_versions)
    generation_calls = sum(
        1
        for request in state.pending_requests + state.superseded_requests
        if request.kind == "answer_publication"
    )
    return Phase4ReplayResult(
        session_id=state.session_id,
        corpus_id=corpus.corpus_id,
        index_id=corpus.manifest.index_id,
        corpus_source_kind=corpus.source_kind,
        retrieval_backend=getattr(retriever, "backend", "unknown"),
        generation_execution_mode=execution_mode,
        generation_provider=provider,
        generation_model=model,
        run_status=run_status,
        steps=steps,
        state=state,
        retrieval_call_count=retrieval_calls,
        retrieval_attempt_count=retrieval_attempts,
        generation_call_count=generation_calls,
        generation_attempt_count=generation_attempts,
        notes=notes,
    )


async def replay_phase4_session(
    turns: Sequence[Phase4ReplayTurn],
    *,
    corpus: CorpusIndex,
    retriever: Any,
    generation_provider: Any | None = None,
    follow_up_interpreter: Phase4FollowUpInterpreter | None = None,
    store: Phase4SessionStore | None = None,
    race_follow_up: Phase4ReplayTurn | None = None,
    race_delay_s: float = 0.0,
) -> Phase4ReplayResult:
    """Execute an initial answer and follow-ups through the Phase 4 pipeline.

    ``race_follow_up`` is an optional replay-only concurrency exercise.  It
    starts the older answer generation task, applies that turn through the
    normal patch/retrieval/publication path, and then awaits the old task so a
    late result is observed and rejected by revision protection.
    """

    if not turns:
        raise ValueError("Phase 4 replay requires at least one turn")
    first = turns[0]
    if any(turn.session_id != first.session_id for turn in turns):
        raise ValueError("a Phase 4 replay may contain only one session")
    if any(turn.turn_index <= first.turn_index for turn in turns[1:]):
        raise ValueError("Phase 4 replay turn indices must increase")
    if race_follow_up is not None and race_follow_up.session_id != first.session_id:
        raise ValueError("race follow-up must belong to the replay session")
    store = store or Phase4SessionStore(max_sessions=8, max_records_per_session=256)
    initial_plan = decompose_query(first.text, transcript_revision=0, max_intents=6)
    state = store.create_session(
        first.session_id,
        initial_plan=initial_plan,
        utterance_id=first.utterance_id,
        initial_turn=first.turn_index,
        corpus_id=corpus.corpus_id,
        index_id=corpus.manifest.index_id,
    )
    state, initial_usage, initial_calls, notes = await _record_initial_retrieval(
        store,
        state,
        corpus,
        retriever,
        first.text,
    )
    publisher = Phase4AnswerPublisher(store, corpus, generation_provider=generation_provider)
    initial_publication = await publisher.generate_and_publish(
        first.session_id,
        expected_revision=state.state_revision,
        triggering_turn=first.turn_index,
        triggering_utterance_id=first.utterance_id,
        retrieval_usage=initial_usage,
        retrieval_call_count=initial_calls,
        retrieval_attempt_count=initial_calls,
        retrieval_model_identity="flowcontext.phase4.replay.retriever",
    )
    state = initial_publication.state
    steps = [_step(first, state, classification="initial", publication=initial_publication)]
    interpreter = follow_up_interpreter or Phase4FollowUpInterpreter()
    coordinator = Phase4SelectiveUpdateCoordinator(store, retriever)

    async def apply_follow_up(turn: Phase4ReplayTurn) -> tuple[Phase4SessionState, dict[str, Any]]:
        nonlocal notes
        patch = store.propose_follow_up(
            turn.session_id,
            turn.text,
            utterance_id=turn.utterance_id,
            turn_index=turn.turn_index,
            interpreter=interpreter,
        )
        publication = None
        plan = None
        if patch.classification == "clarification_required":
            state_after = store.apply_patch(patch)
        elif patch.classification == "reformat_answer":
            coordinator.apply_patch(patch)
            state_after = store.get(turn.session_id)
            publication = await publisher.present_and_publish(
                turn.session_id,
                expected_revision=state_after.state_revision,
            )
            state_after = publication.state
        else:
            plan = coordinator.apply_patch(patch)
            if plan is None:
                raise PatchValidationError("semantic Phase 4 patch did not create a selective plan")
            state_after = await coordinator.execute_plan(plan)
            stored_plan = next(
                item for item in state_after.selective_update_plans if item.plan_id == plan.plan_id
            )
            plan = stored_plan
            publication = await publisher.generate_and_publish(
                turn.session_id,
                expected_revision=state_after.state_revision,
                triggering_patch_id=patch.patch_id,
                triggering_turn=turn.turn_index,
                triggering_utterance_id=turn.utterance_id,
                retrieval_usage=stored_plan.retrieval_usage,
                retrieval_call_count=stored_plan.retrieval_call_count,
                retrieval_attempt_count=stored_plan.retrieval_attempt_count,
                retrieval_model_identity="flowcontext.phase4.selective-retriever",
            )
            state_after = publication.state
        record = _step(
            turn,
            state_after,
            classification=patch.classification,
            patch=patch,
            plan=plan,
            publication=publication,
        )
        return state_after, record

    if race_follow_up is not None:
        old_generation = asyncio.create_task(
            publisher.generate_and_publish(
                first.session_id,
                expected_revision=state.state_revision,
                triggering_turn=first.turn_index,
                triggering_utterance_id=first.utterance_id,
                retrieval_usage=initial_usage,
                retrieval_call_count=initial_calls,
                retrieval_attempt_count=initial_calls,
                retrieval_model_identity="flowcontext.phase4.replay.retriever",
            )
        )
        # Let the older task register its revision-bound request before the
        # correction is applied.  The generation provider may then be
        # cancellation-resistant while the store still rejects its result.
        await asyncio.sleep(0)
        if race_delay_s > 0:
            await asyncio.sleep(race_delay_s)
        state, race_step = await apply_follow_up(race_follow_up)
        steps.append(race_step)
        old_result = await old_generation
        steps.insert(
            1,
            {
                "turn_index": first.turn_index,
                "utterance_id": first.utterance_id,
                "text": first.text,
                "classification": "superseded_initial_generation",
                "publication": {
                    "published": old_result.published,
                    "status": old_result.status,
                    "request_id": old_result.request_id,
                    "reason": old_result.reason,
                },
                "state_revision": old_result.state.state_revision,
                "answer_status": old_result.state.answer_status,
                "current_answer_version": old_result.state.current_answer_version,
                "current_claim_ids": [item.claim_id for item in old_result.state.current_claims],
            },
        )
    else:
        for turn in turns[1:]:
            state, turn_step = await apply_follow_up(turn)
            steps.append(turn_step)

    await coordinator.close()
    final_state = store.get(first.session_id)
    return _result(
        final_state,
        corpus=corpus,
        retriever=retriever,
        publisher=publisher,
        steps=steps,
        notes=notes,
    )


__all__ = ["load_phase4_turns", "replay_phase4_session"]
