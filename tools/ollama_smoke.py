"""Real local-model checks using the same streaming and Phase 4 replay paths.

Run from the repo root: uv run --extra demo python tools/ollama_smoke.py
This never pulls models. Install the requested Ollama model before running.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from flowcontext.config import Settings
from flowcontext.contracts import TranscriptEvent
from flowcontext.generation import generation_provider_for_settings
from flowcontext.ingestion import CorpusIngestor, load_document_inputs
from flowcontext.phase4 import Phase4SessionStore
from flowcontext.phase4_replay import load_phase4_turns, replay_phase4_session
from flowcontext.retrieval import make_retriever
from flowcontext.streaming import replay_streaming_transcript

FILES = {'B': 'late-constraint', 'C': 'entity-correction', 'D': 'formatting',
         'E': 'partial-unsupported', 'F': 'race'}


def answer_checks(answer, corpus):
    chunks = {chunk.chunk_id: chunk.text for chunk in corpus.chunks}
    claims = answer.factual_claims if answer else []
    return {
        'has_grounded_claims': bool(claims),
        'citation_ids_in_corpus': all(cid in chunks for claim in claims for cid in claim.supporting_chunk_ids),
        'claims_are_cited_excerpts': all(
            any(claim.claim_text in chunks.get(cid, '') for cid in claim.supporting_chunk_ids)
            for claim in claims
        ),
        'supporting_excerpts_in_cited_passages': all(
            any(excerpt in chunks.get(cid, '') for cid in claim.supporting_chunk_ids)
            for claim in claims for excerpt in claim.supporting_excerpts
        ),
    }


async def check_scenario(scenario, settings, corpus):
    provider = generation_provider_for_settings(settings)
    retriever = make_retriever(corpus, backend='lexical')
    started = time.perf_counter()
    if scenario == 'A':
        query = 'Which Venue B in Pune can host 40 attendees? What are the catering options?'
        events = [TranscriptEvent(session_id='ollama-a', utterance_id='u-0', event_id=f'e-{i}',
                                  sequence_number=i, source_timestamp_s=i * 0.8, text=query,
                                  is_final=bool(i), text_mode='cumulative') for i in range(2)]
        result = await replay_streaming_transcript(
            events, corpus=corpus, backend='lexical', retriever=retriever,
            multi_intent=True, retrieval_mode='lexical', execution_mode='realtime',
            generation_provider=provider,
        )
        checks = answer_checks(result.answer, corpus)
        checks.update(real_generation_success=result.generation_status == 'success',
                      early_evidence_reused=result.early_evidence_reused)
        checks['all_supported_intents_answered'] = bool(result.answer and result.answer.intent_statuses) and all(
            status.status == 'answered' for status in result.answer.intent_statuses
        )
        answer = result.answer
        attempts = result.generation_attempts
    else:
        turns = load_phase4_turns(ROOT / 'examples/replay' / f'phase4-{FILES[scenario]}.jsonl')
        race = None
        if scenario == 'F':
            race, turns = turns[1], [turns[0], *turns[2:]]
        result = await replay_phase4_session(
            turns, corpus=corpus, retriever=retriever, generation_provider=provider,
            store=Phase4SessionStore(), race_follow_up=race,
        )
        version = result.state.current_answer
        checks = answer_checks(version.answer if version else None, corpus)
        checks['real_provider_reported'] = result.generation_execution_mode == 'real_provider'
        checks['answer_published'] = version is not None and result.run_status in {'completed', 'partial'}
        answer = version.answer if version else None
        if scenario in 'BCD':
            checks['all_supported_intents_answered'] = bool(answer and answer.intent_statuses) and all(
                status.status == 'answered' for status in answer.intent_statuses
            )
        if scenario == 'D':
            checks['presentation_has_no_retrieval_or_generation'] = bool(
                version and version.change_kind == 'presentation'
                and version.retrieval_call_count == 0 and version.generation_attempts == 0
            )
        if scenario == 'E':
            checks['unsupported_constraint_is_partial'] = result.run_status == 'partial'
            checks['supported_intent_preserved'] = bool(answer and any(
                status.status == 'answered' for status in answer.intent_statuses
            ))
        if scenario == 'F':
            checks['superseded_request_recorded'] = bool(result.state.superseded_requests)
            checks['stale_publication_rejected'] = any(
                step.get('publication') and not step['publication']['published']
                and step['classification'] == 'superseded_initial_generation' for step in result.steps
            )
        attempts = result.generation_attempt_count
    return {'scenario': scenario, 'run_status': result.run_status,
            'replay_wall_clock_s': round(time.perf_counter() - started, 3),
            'generation_attempts': attempts, 'checks': checks, 'passed': all(checks.values()),
            'intent_statuses': [status.model_dump(mode='json') for status in answer.intent_statuses] if answer else [],
            'result': result.model_dump(mode='json')}


async def run(args):
    settings = Settings(generation_backend='openai_compatible', generation_provider='ollama',
                        generation_model=args.model, generation_base_url=args.base_url,
                        generation_timeout_s=args.timeout, generation_max_retries=0,
                        generation_max_output_tokens=1200, generation_max_repair_attempts=1)
    # Validate the local endpoint before doing any provider work.
    generation_provider_for_settings(settings)
    corpus = CorpusIngestor().ingest(load_document_inputs(ROOT / 'data/synthetic/phase4_documents.jsonl'),
                                    'synthetic_fixture')
    report = {'model': args.model, 'provider': 'ollama', 'retrieval_backend': 'lexical',
              'generation_execution_mode': 'real_provider', 'corpus_source_kind': 'synthetic_fixture',
              'official_benchmark': False, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'generation_adapter_sha256': hashlib.sha256((ROOT / 'src/flowcontext/generation.py').read_bytes()).hexdigest(),
              'scenarios': []}
    for scenario in args.scenarios:
        entry = await check_scenario(scenario, settings, corpus)
        report['scenarios'].append(entry)
        failed = [key for key, value in entry['checks'].items() if not value]
        print(f"{scenario}: {'PASS' if entry['passed'] else 'FAIL'} · {entry['run_status']} · "
              f"{entry['replay_wall_clock_s']:.3f}s replay wall-clock · failed checks: {failed}", flush=True)
        # Persist after each scenario so an interrupted run retains completed checks.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Detailed local results: {args.output}', flush=True)
    return 0 if all(entry['passed'] for entry in report['scenarios']) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='qwen2.5:3b')
    parser.add_argument('--base-url', default='http://127.0.0.1:11434/v1')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--scenarios', nargs='+', choices=list('ABCDEF'), default=list('ABCDEF'))
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/ollama-smoke.json')
    return asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    raise SystemExit(main())
