"""Command-line entry points for the Phase 1/2/3 corpus and replay workflows."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .config import Settings, config_asset_status, load_settings
from .contracts import PreviousAnswerContext, RetrievalResponse, TranscriptEvent
from .embeddings import DenseRetrievalUnavailable, EmbeddingError
from .evaluation import (
    EvaluationError,
    evaluate,
    evaluate_suite,
    load_evaluation_cases,
    write_report,
)
from .generation import generation_provider_for_settings
from .indexing import RetrievalBackend, build_index_from_source, inspect_corpus
from .ingestion import (
    IngestionError,
    StaleIndexError,
    assert_index_matches_source,
    load_index,
)
from .replay import (
    ReplayError,
    build_run_manifest,
    load_replay,
    load_transcript,
    replay_transcript,
    write_replay,
    write_run_manifest,
)
from .retrieval import RetrievalError, make_retriever
from .multi_intent import (
    MultiIntentRetriever,
    StructuredMultiIntentDecomposer,
    decomposition_provider_for_settings,
    make_multi_intent_retriever,
)
from .streaming import (
    StreamingControllerError,
    build_streaming_run_manifest,
    replay_streaming_transcript,
    streaming_config_from_settings,
    write_streaming_replay,
    write_streaming_run_manifest,
)
from .scheduler import scheduler_config_from_settings
from .streaming_evaluation import (
    StreamingEvaluationError,
    evaluate_streaming_suite,
    load_streaming_evaluation_cases,
    write_streaming_evaluation_report,
)
from .phase3_evaluation import (
    Phase3EvaluationError,
    build_phase3_comparison,
    evaluate_multi_intent_cases,
    load_historical_report,
    write_phase3_comparison_report,
)
from .trace import write_trace_jsonl


def _settings(env_file: str | None, overrides: dict[str, object] | None = None) -> Settings:
    settings = load_settings(Path(env_file) if env_file else None)
    if not overrides:
        return settings
    values = settings.model_dump(mode="python")
    values.update(overrides)
    return Settings.model_validate(values)


def _settings_for_args(args: argparse.Namespace) -> Settings:
    overrides: dict[str, object] = {}
    for argument_name, setting_name in (
        ("max_chars", "chunk_max_chars"),
        ("overlap_chars", "chunk_overlap_chars"),
        ("top_k", "retrieval_top_k"),
        ("embedding_model", "embedding_model"),
        ("embedding_revision", "embedding_revision"),
        ("embedding_license", "embedding_license"),
        ("embedding_dimensions", "embedding_dimensions"),
        ("context_budget_tokens", "multi_intent_context_budget_tokens"),
        ("multi_intent_max_workers", "multi_intent_max_workers"),
        ("multi_intent_rrf_k", "multi_intent_rrf_k"),
        ("streaming_debounce_source_s", "streaming_debounce_source_s"),
        ("streaming_min_query_chars", "streaming_min_query_chars"),
        ("streaming_min_topic_terms", "streaming_min_topic_terms"),
        ("streaming_min_new_content_terms", "streaming_min_new_content_terms"),
        ("streaming_max_concurrency", "streaming_max_concurrency"),
        ("streaming_max_pending_requests", "streaming_max_pending_requests"),
        ("streaming_request_timeout_s", "streaming_request_timeout_s"),
        ("streaming_max_retries", "streaming_max_retries"),
        ("streaming_retry_backoff_s", "streaming_retry_backoff_s"),
        ("streaming_max_total_requests", "streaming_max_total_requests"),
        ("streaming_final_wait_timeout_s", "streaming_final_wait_timeout_s"),
    ):
        value = getattr(args, argument_name, None)
        if value is not None:
            overrides[setting_name] = value
    backend = getattr(args, "backend", None)
    if backend is not None:
        overrides["retrieval_backend"] = backend
    retrieval_mode = getattr(args, "retrieval_mode", None)
    if retrieval_mode is not None:
        overrides["multi_intent_retrieval_mode"] = retrieval_mode
    source_kind = getattr(args, "source_kind", None)
    if source_kind is not None:
        overrides["corpus_status"] = source_kind
    if getattr(args, "local_files_only", False):
        overrides["embedding_local_files_only"] = True
    for argument_name, setting_name in (
        ("streaming_retrieve_on_correction", "streaming_retrieve_on_correction"),
        ("streaming_retrieve_on_constraint_change", "streaming_retrieve_on_constraint_change"),
        ("streaming_duplicate_query_suppression", "streaming_duplicate_query_suppression"),
        ("streaming_final_bypasses_debounce", "streaming_final_bypasses_debounce"),
        ("streaming_cancel_on_supersession", "streaming_cancel_on_supersession"),
    ):
        value = getattr(args, argument_name, None)
        if value is not None:
            overrides[setting_name] = value
    return _settings(args.env_file, overrides)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", default=None, help="dotenv-style config file; defaults to .env")


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", dest="input_path", default=None, help="supported corpus source JSONL")
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--overlap-chars", type=int, default=None)


def _add_multi_intent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "lexical", "hybrid", "mock"],
        default=None,
        help="Phase 3 retrieval mode; hybrid uses rank-based lexical+dense fusion",
    )
    parser.add_argument(
        "--context-budget-tokens",
        dest="context_budget_tokens",
        type=int,
        default=None,
        help="total Phase 3 evidence context budget",
    )
    parser.add_argument(
        "--multi-intent-max-workers",
        dest="multi_intent_max_workers",
        type=int,
        default=None,
        help="bounded concurrent independent-intent workers",
    )
    parser.add_argument(
        "--multi-intent-rrf-k",
        dest="multi_intent_rrf_k",
        type=int,
        default=None,
        help="RRF smoothing constant for Phase 3 fusion",
    )


def _selected_multi_intent_mode(
    args: argparse.Namespace,
    settings: Settings,
    backend: str,
) -> str:
    """Resolve Phase 3 mode while keeping explicit legacy --backend behavior."""

    return (
        args.retrieval_mode
        if getattr(args, "retrieval_mode", None) is not None
        else backend
        if getattr(args, "backend", None) is not None
        else settings.multi_intent_retrieval_mode
    )


def _base_backend_for_multi_intent(
    args: argparse.Namespace,
    settings: Settings,
    index,
    backend: str,
) -> str:
    """Choose the construction backend without defeating an explicit Phase 3 mode."""

    if not getattr(args, "multi_intent", False):
        return backend
    mode = _selected_multi_intent_mode(args, settings, backend)
    if mode != "hybrid":
        return mode
    if getattr(args, "backend", None) is not None:
        return args.backend
    indexed_backend = index.manifest.embedding.backend
    return indexed_backend if indexed_backend in {"dense", "lexical", "mock"} else backend


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common(parser)
    _add_source_arguments(parser)
    parser.add_argument("--output", dest="output_path", default=None)
    parser.add_argument("--source-kind", choices=["official", "synthetic_fixture", "unknown"], default=None)
    parser.add_argument("--backend", choices=["dense", "lexical", "mock"], default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-revision", default=None)
    parser.add_argument("--embedding-license", default=None)
    parser.add_argument("--embedding-dimensions", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace a stale index atomically")


def _add_streaming_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--streaming-debounce-source-s",
        dest="streaming_debounce_source_s",
        type=float,
        default=None,
        help="source-time coalescing window for revised partial queries",
    )
    parser.add_argument(
        "--streaming-min-query-chars",
        dest="streaming_min_query_chars",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--streaming-min-topic-terms",
        dest="streaming_min_topic_terms",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--streaming-min-new-content-terms",
        dest="streaming_min_new_content_terms",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--streaming-retrieve-on-correction",
        dest="streaming_retrieve_on_correction",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--streaming-retrieve-on-constraint-change",
        dest="streaming_retrieve_on_constraint_change",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--streaming-duplicate-query-suppression",
        dest="streaming_duplicate_query_suppression",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--streaming-final-bypasses-debounce",
        dest="streaming_final_bypasses_debounce",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--streaming-max-concurrency",
        dest="streaming_max_concurrency",
        type=int,
        default=None,
        help="maximum simultaneous executor-backed retrieval calls",
    )
    parser.add_argument(
        "--streaming-max-pending-requests",
        dest="streaming_max_pending_requests",
        type=int,
        default=None,
        help="maximum active retrieval request states before latest-query coalescing",
    )
    parser.add_argument(
        "--streaming-request-timeout-s",
        dest="streaming_request_timeout_s",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--streaming-max-retries",
        dest="streaming_max_retries",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--streaming-retry-backoff-s",
        dest="streaming_retry_backoff_s",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--streaming-max-total-requests",
        dest="streaming_max_total_requests",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--streaming-final-wait-timeout-s",
        dest="streaming_final_wait_timeout_s",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--streaming-cancel-on-supersession",
        dest="streaming_cancel_on_supersession",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlowContext Theme 4 Phase 1/2 CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    config_parser = commands.add_parser("config-check", help="validate config and report asset presence")
    _add_common(config_parser)

    inspect_parser = commands.add_parser(
        "inspect-corpus", help="inspect supported source format, metadata, and extraction health"
    )
    _add_common(inspect_parser)
    _add_source_arguments(inspect_parser)
    inspect_parser.add_argument("--output", dest="output_path", default=None)

    build_parser_command = commands.add_parser("build-index", help="build or safely reuse a corpus index")
    _add_build_arguments(build_parser_command)

    # Keep the original Phase 1 command as a compatibility alias while using
    # exactly the same implementation and dense-by-default behavior.
    ingest_parser = commands.add_parser("ingest", help="compatibility alias for build-index")
    _add_build_arguments(ingest_parser)

    retrieve_parser = commands.add_parser("retrieve", help="retrieve ranked snippets from an index")
    _add_common(retrieve_parser)
    retrieve_parser.add_argument("--index", "--corpus", dest="index_path", required=True)
    retrieve_parser.add_argument("--query", required=True)
    retrieve_parser.add_argument("--backend", choices=["dense", "lexical", "mock"], default=None)
    retrieve_parser.add_argument("--top-k", type=int, default=None)
    retrieve_parser.add_argument("--source", dest="source_path", default=None, help="source JSONL for freshness check")
    retrieve_parser.add_argument("--output", dest="output_path", default=None)
    retrieve_parser.add_argument("--local-files-only", action="store_true")
    retrieve_parser.add_argument(
        "--multi-intent",
        action="store_true",
        help="decompose the query and fuse per-intent retrieval with RRF",
    )
    _add_multi_intent_arguments(retrieve_parser)

    replay_parser = commands.add_parser("replay", help="replay a transcript and retrieve from an index")
    _add_common(replay_parser)
    replay_parser.add_argument("--transcript", dest="transcript_path", default=None)
    replay_parser.add_argument("--index", "--corpus", dest="index_path", required=True)
    replay_parser.add_argument("--backend", choices=["dense", "lexical", "mock"], default=None)
    replay_parser.add_argument("--source", dest="source_path", default=None, help="source JSONL for freshness check")
    replay_parser.add_argument("--output", dest="output_path", default=None)
    replay_parser.add_argument(
        "--trace-output",
        dest="trace_output_path",
        default=None,
        help="JSONL trace path; defaults beside --output with .traces.jsonl suffix",
    )
    replay_parser.add_argument(
        "--manifest-output",
        dest="manifest_output_path",
        default=None,
        help="run manifest path; defaults beside --output with .manifest.json suffix",
    )
    replay_parser.add_argument("--run-id", default=None)
    replay_parser.add_argument("--top-k", type=int, default=None)
    replay_parser.add_argument("--local-files-only", action="store_true")
    replay_parser.add_argument(
        "--mode",
        choices=["baseline", "streaming"],
        default="baseline",
        help="baseline waits for final; streaming enables Phase 2 decisions and early retrieval",
    )
    replay_parser.add_argument(
        "--multi-intent",
        action="store_true",
        help="enable Phase 3 decomposition, parallel intent retrieval, and RRF fusion",
    )
    _add_multi_intent_arguments(replay_parser)
    replay_parser.add_argument(
        "--previous-answer",
        dest="previous_answer_path",
        default=None,
        help="optional same-session completed replay result for formatting-only context",
    )
    _add_streaming_policy_arguments(replay_parser)
    replay_parser.add_argument(
        "--execution-mode",
        choices=["realtime", "accelerated"],
        default="accelerated",
        help="replay source-time gaps or consume events without waiting",
    )

    evaluate_parser = commands.add_parser("evaluate", help="evaluate one replay against local expectation JSONL")
    _add_common(evaluate_parser)
    evaluate_parser.add_argument("--run", dest="run_path", required=True)
    evaluate_parser.add_argument("--gold", dest="gold_path", default=None)
    evaluate_parser.add_argument("--corpus", dest="corpus_path", required=True)
    evaluate_parser.add_argument("--output", dest="output_path", default=None)

    evaluate_suite_parser = commands.add_parser(
        "evaluate-suite",
        help="run a fixed development or held-out evaluation split through the baseline",
    )
    _add_common(evaluate_suite_parser)
    evaluate_suite_parser.add_argument("--cases", dest="cases_path", default=None)
    evaluate_suite_parser.add_argument(
        "--split",
        choices=["development", "held_out", "all"],
        default="development",
    )
    evaluate_suite_parser.add_argument("--corpus", dest="index_path", required=True)
    evaluate_suite_parser.add_argument("--source", dest="source_path", default=None)
    evaluate_suite_parser.add_argument("--backend", choices=["dense", "lexical", "mock"], default=None)
    evaluate_suite_parser.add_argument("--top-k", type=int, default=None)
    evaluate_suite_parser.add_argument("--local-files-only", action="store_true")
    evaluate_suite_parser.add_argument("--execution-mode", choices=["realtime", "accelerated"], default="accelerated")
    evaluate_suite_parser.add_argument("--run-id-prefix", default="evaluation")
    evaluate_suite_parser.add_argument("--output", dest="output_path", default=None)

    streaming_evaluate_parser = commands.add_parser(
        "evaluate-streaming",
        help="run the matched Phase 2 baseline/streaming audit suite",
    )
    _add_common(streaming_evaluate_parser)
    streaming_evaluate_parser.add_argument("--cases", dest="cases_path", default=None)
    streaming_evaluate_parser.add_argument(
        "--split",
        choices=["development", "held_out", "all"],
        default="development",
    )
    streaming_evaluate_parser.add_argument("--corpus", dest="index_path", required=True)
    streaming_evaluate_parser.add_argument("--source", dest="source_path", default=None)
    streaming_evaluate_parser.add_argument("--backend", choices=["dense", "lexical", "mock"], default=None)
    streaming_evaluate_parser.add_argument("--top-k", type=int, default=None)
    streaming_evaluate_parser.add_argument("--local-files-only", action="store_true")
    streaming_evaluate_parser.add_argument(
        "--execution-mode", choices=["realtime", "accelerated"], default="accelerated"
    )
    streaming_evaluate_parser.add_argument("--run-id-prefix", default="streaming-evaluation")
    streaming_evaluate_parser.add_argument(
        "--compare-disabled-scheduling",
        action="store_true",
        help="rerun the marked coalescing case with duplicate suppression and debounce disabled",
    )
    streaming_evaluate_parser.add_argument("--output", dest="output_path", default=None)

    phase3_parser = commands.add_parser(
        "evaluate-phase3",
        help="investigate Phase 2 matched results and measure opt-in multi-intent retrieval",
    )
    _add_common(phase3_parser)
    phase3_parser.add_argument(
        "--split",
        choices=["development", "held_out", "all"],
        default="all",
        help="split for the Phase 2 comparison and Phase 3 answer cases",
    )
    phase3_parser.add_argument("--corpus", dest="index_path", required=True)
    phase3_parser.add_argument("--source", dest="source_path", default=None)
    phase3_parser.add_argument("--backend", choices=["dense", "lexical", "mock"], default=None)
    phase3_parser.add_argument("--top-k", type=int, default=None)
    phase3_parser.add_argument("--local-files-only", action="store_true")
    phase3_parser.add_argument(
        "--execution-mode", choices=["realtime", "accelerated"], default="realtime"
    )
    phase3_parser.add_argument(
        "--phase2-cases",
        dest="phase2_cases_path",
        default=None,
        help="optional Phase 2 JSONL asset; defaults to the configured development/held-out files",
    )
    phase3_parser.add_argument(
        "--phase3-development-cases",
        dest="phase3_development_cases_path",
        default=None,
        help="optional frozen Phase 3 development cases; defaults to the Phase 1 development file",
    )
    phase3_parser.add_argument(
        "--phase3-held-out-cases",
        dest="phase3_held_out_cases_path",
        default=None,
        help="optional frozen Phase 3 held-out cases; defaults to the Phase 1 held-out file",
    )
    # Keep the earlier Phase 3 decomposition/generation comparison intact;
    # this command writes the retrieval/evidence follow-up under a distinct
    # default artifact name. Callers may still choose an explicit path.
    phase3_parser.add_argument(
        "--output",
        dest="output_path",
        default="reports/phase3_retrieval_comparison_realtime.json",
    )
    _add_multi_intent_arguments(phase3_parser)

    answer_parser = commands.add_parser(
        "answer",
        help="run one complete final utterance or print an existing replay answer",
    )
    _add_common(answer_parser)
    answer_source = answer_parser.add_mutually_exclusive_group(required=True)
    answer_source.add_argument("--run", dest="run_path", help="existing replay artifact to inspect")
    answer_source.add_argument("--query", help="complete utterance to run as one final transcript event")
    answer_parser.add_argument("--index", "--corpus", dest="index_path", default=None)
    answer_parser.add_argument("--source", dest="source_path", default=None)
    answer_parser.add_argument("--backend", choices=["dense", "lexical", "mock"], default=None)
    answer_parser.add_argument("--top-k", type=int, default=None)
    answer_parser.add_argument("--local-files-only", action="store_true")
    answer_parser.add_argument(
        "--multi-intent",
        action="store_true",
        help="enable Phase 3 decomposition, parallel intent retrieval, and RRF fusion",
    )
    _add_multi_intent_arguments(answer_parser)
    answer_parser.add_argument("--session-id", default="answer-session")
    answer_parser.add_argument("--utterance-id", default="answer-utterance")
    answer_parser.add_argument("--run-id", default="answer-run")
    answer_parser.add_argument("--execution-mode", choices=["realtime", "accelerated"], default="accelerated")
    answer_parser.add_argument("--trace-output", dest="trace_output_path", default=None)
    answer_parser.add_argument("--manifest-output", dest="manifest_output_path", default=None)
    answer_parser.add_argument("--output", dest="output_path", default=None)

    smoke_parser = commands.add_parser(
        "smoke", help="run offline contract, config, lexical/mock fixture, and evaluation checks"
    )
    _add_common(smoke_parser)

    dense_smoke_parser = commands.add_parser(
        "dense-smoke", help="run a real dense model smoke test; exit 3 when dependencies/model are unavailable"
    )
    _add_common(dense_smoke_parser)
    dense_smoke_parser.add_argument("--input", dest="input_path", default=None)
    dense_smoke_parser.add_argument("--query", default="workshop venue cancellation catering")
    dense_smoke_parser.add_argument("--top-k", type=int, default=None)
    dense_smoke_parser.add_argument("--local-files-only", action="store_true")
    return parser


def _json_print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def command_config_check(args: argparse.Namespace) -> int:
    settings = _settings(args.env_file)
    status = config_asset_status(settings)
    status.update(
        {
            "app_name": settings.app_name,
            "chunk_max_chars": settings.chunk_max_chars,
            "chunk_overlap_chars": settings.chunk_overlap_chars,
            "retrieval_backend": settings.retrieval_backend,
            "retrieval_top_k": settings.retrieval_top_k,
            "embedding_model": settings.embedding_model,
            "embedding_revision": settings.embedding_revision,
            "embedding_license": settings.embedding_license,
            "embedding_dimensions": settings.embedding_dimensions,
            "offline_model_call": False,
        }
    )
    _json_print(status)
    return 0


def command_inspect_corpus(args: argparse.Namespace) -> int:
    settings = _settings_for_args(args)
    source_path = Path(args.input_path) if args.input_path else settings.corpus_path
    result = inspect_corpus(source_path, settings=settings)
    if args.output_path:
        _write_json(Path(args.output_path), result)
        result["output"] = args.output_path
    _json_print(result)
    return 0


def command_build_index(args: argparse.Namespace) -> int:
    settings = _settings_for_args(args)
    source_path = Path(args.input_path) if args.input_path else settings.corpus_path
    output_path = Path(args.output_path) if args.output_path else settings.index_path
    selected_backend: RetrievalBackend = args.backend or settings.retrieval_backend
    index, elapsed, status = build_index_from_source(
        source_path,
        settings=settings,
        backend=selected_backend,
        force=args.force,
        output_path=output_path,
    )
    _json_print(
        {
            "status": status,
            "output": str(output_path),
            "index_id": index.manifest.index_id,
            "build_fingerprint": index.manifest.build_fingerprint,
            "source_fingerprint": index.manifest.source_fingerprint,
            "selected_backend": index.manifest.embedding.backend,
            "documents": len(index.documents),
            "chunks": len(index.chunks),
            "chunking": index.manifest.chunking.model_dump(mode="json"),
            "embedding": index.manifest.embedding.model_dump(mode="json"),
            "indexing_seconds": round(elapsed, 6),
        }
    )
    return 0


def _load_query_index(args: argparse.Namespace, settings: Settings):
    index = load_index(Path(args.index_path))
    if args.source_path:
        assert_index_matches_source(
            index,
            Path(args.source_path),
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
    return index


def command_retrieve(args: argparse.Namespace) -> int:
    settings = _settings_for_args(args)
    index = _load_query_index(args, settings)
    backend = args.backend or settings.retrieval_backend
    base_backend = _base_backend_for_multi_intent(args, settings, index, backend)
    base_retriever = make_retriever(
        index,
        backend=base_backend,
        top_k=settings.retrieval_top_k,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
    )
    retriever = base_retriever
    if args.multi_intent:
        retriever = make_multi_intent_retriever(
            index,
            retrieval_mode=_selected_multi_intent_mode(args, settings, backend),
            base_retriever=base_retriever,
            top_k=settings.retrieval_top_k,
            max_workers=settings.multi_intent_max_workers,
            rrf_k=settings.multi_intent_rrf_k,
            context_budget_tokens=settings.multi_intent_context_budget_tokens,
            decomposer=StructuredMultiIntentDecomposer(
                decomposition_provider_for_settings(settings)
            ),
            cache_dir=settings.embedding_cache_dir,
            local_files_only=settings.embedding_local_files_only,
        )
    hits = retriever.search(args.query)
    response = RetrievalResponse(
        query=args.query,
        retrieval_backend=retriever.backend,
        index_id=index.manifest.index_id,
        hits=hits,
        decomposition=(
            retriever.last_decomposition.model_dump(mode="json")
            if isinstance(retriever, MultiIntentRetriever)
            and retriever.last_decomposition is not None
            else None
        ),
        retrieval_details=(
            retriever.last_result.model_dump(mode="json")
            if isinstance(retriever, MultiIntentRetriever)
            and retriever.last_result is not None
            else None
        ),
    )
    if args.output_path:
        _write_json(Path(args.output_path), response.model_dump(mode="json"))
    _json_print(response.model_dump(mode="json"))
    return 0


def _previous_answer_context(path: Path, session_id: str) -> PreviousAnswerContext:
    """Load only an existing answer from the same session boundary."""

    result = load_replay(path)
    if result.session_id != session_id:
        raise StreamingControllerError(
            "--previous-answer must come from the same session as the streaming transcript"
        )
    if result.run_status != "completed" or result.generation_status != "success":
        raise StreamingControllerError(
            "--previous-answer must be a completed successful Phase 1 answer"
        )
    return PreviousAnswerContext(
        session_id=result.session_id,
        answer_text=result.answer.answer_text,
        answer_version=result.answer.answer_version,
    )


def command_streaming_replay(
    args: argparse.Namespace,
    settings: Settings,
    index,
    transcript_path: Path,
    output_path: Path,
    trace_path: Path,
    manifest_path: Path,
    backend: str,
) -> int:
    events = load_transcript(transcript_path)
    previous_context = (
        _previous_answer_context(Path(args.previous_answer_path), events[0].session_id)
        if args.previous_answer_path
        else None
    )
    result = asyncio.run(
        replay_streaming_transcript(
            events,
            corpus=index,
            top_k=settings.retrieval_top_k,
            backend=backend,
            run_id=args.run_id,
            model_identity="flowcontext.streaming-controller.v1",
            embedding_cache_dir=settings.embedding_cache_dir,
            embedding_local_files_only=settings.embedding_local_files_only,
            execution_mode=args.execution_mode,
            controller_config=streaming_config_from_settings(settings),
            scheduler_config=scheduler_config_from_settings(settings),
            previous_answer_context=previous_context,
            generation_provider=generation_provider_for_settings(settings),
            decomposition_provider=(
                decomposition_provider_for_settings(settings)
                if args.multi_intent
                else None
            ),
            multi_intent=args.multi_intent,
            retrieval_mode=(
                _selected_multi_intent_mode(args, settings, backend)
                if args.multi_intent
                else None
            ),
            context_budget_tokens=settings.multi_intent_context_budget_tokens,
            multi_intent_max_workers=settings.multi_intent_max_workers,
            multi_intent_rrf_k=settings.multi_intent_rrf_k,
            multi_intent_reranking_enabled=settings.multi_intent_reranking_enabled,
        )
    )
    write_streaming_replay(output_path, result)
    write_trace_jsonl(trace_path, result.traces)
    manifest = build_streaming_run_manifest(
        result,
        corpus=index,
        settings=settings,
        transcript_path=transcript_path,
        index_path=Path(args.index_path),
        result_path=output_path,
        trace_path=trace_path,
    )
    write_streaming_run_manifest(manifest_path, manifest)
    _json_print(
        {
            "status": result.run_status,
            "output": str(output_path),
            "trace_output": str(trace_path),
            "manifest_output": str(manifest_path),
            "run_id": result.run_id,
            "mode": "streaming",
            "retrieval_backend": result.retrieval_backend,
            "execution_mode": result.execution_mode,
            "top_k": result.retrieval_top_k,
            "corpus_id": result.corpus_id,
            "index_id": result.index_id,
            "final_query": result.final_query,
            "final_decision": result.final_decision.model_dump(mode="json"),
            "retrieval_attempt_count": result.retrieval_attempt_count,
            "early_retrieval_count": result.early_retrieval_count,
            "final_retrieval_count": result.final_retrieval_count,
            "suppressed_duplicate_query_count": result.suppressed_duplicate_query_count,
            "stale_retrieval_count": result.stale_retrieval_count,
            "unnecessary_retrieval_count": result.unnecessary_retrieval_count,
            "scheduled_request_count": result.scheduled_request_count,
            "cancelled_request_count": result.cancelled_request_count,
            "superseded_request_count": result.superseded_request_count,
            "timed_out_request_count": result.timed_out_request_count,
            "retrieval_call_count": result.retrieval_call_count,
            "retrieval_error_count": result.retrieval_error_count,
            "stale_result_discard_count": result.stale_result_discard_count,
            "retrieval_started_early": result.retrieval_started_early,
            "valid_evidence_ready_before_finalization": result.valid_evidence_ready_before_finalization,
            "early_evidence_reused": result.early_evidence_reused,
            "answer_latency_from_final_event_delivery_ms": result.answer_latency_from_final_event_delivery_ms,
            "full_interaction_duration_ms": result.full_interaction_duration_ms,
            "controller_overhead_ms": result.controller_overhead_ms,
            "final_event_delivery_time_s": result.final_event_delivery_time_s,
            "evidence_ready_time_s": result.evidence_ready_time_s,
            "generation_started_time_s": result.generation_started_time_s,
            "generation_completed_time_s": result.generation_completed_time_s,
            "generation_first_content_time_s": result.generation_first_content_time_s,
            "retrieval_usage": result.retrieval_usage.model_dump(mode="json"),
            "error_count": len(result.errors),
            "execution_duration_ms": result.execution_duration_ms,
            "trace_events": len(result.traces),
            "generation": {
                "backend": result.generation_backend,
                "model": result.generation_model,
                "status": result.generation_status,
                "attempts": result.generation_attempts,
                "usage": result.generation_usage.model_dump(mode="json"),
                "cost": result.generation_cost,
            },
        }
    )
    return 1 if result.run_status == "failed" else 0


def command_replay(args: argparse.Namespace) -> int:
    settings = _settings_for_args(args)
    index = _load_query_index(args, settings)
    transcript_path = Path(args.transcript_path) if args.transcript_path else settings.transcript_path
    backend = args.backend or settings.retrieval_backend
    if args.mode == "streaming":
        output_path = (
            Path(args.output_path)
            if args.output_path
            else settings.artifacts_dir / "streaming-replay.json"
        )
        trace_path = (
            Path(args.trace_output_path)
            if args.trace_output_path
            else output_path.with_name(f"{output_path.stem}.traces.jsonl")
        )
        manifest_path = (
            Path(args.manifest_output_path)
            if args.manifest_output_path
            else output_path.with_name(f"{output_path.stem}.manifest.json")
        )
        if output_path.resolve() in {trace_path.resolve(), manifest_path.resolve()}:
            raise ReplayError("replay, trace, and manifest outputs must be different files")
        return command_streaming_replay(
            args,
            settings,
            index,
            transcript_path,
            output_path,
            trace_path,
            manifest_path,
            backend,
        )
    if args.previous_answer_path:
        baseline_events = load_transcript(transcript_path)
        previous_context = _previous_answer_context(
            Path(args.previous_answer_path), baseline_events[0].session_id
        )
    else:
        baseline_events = load_transcript(transcript_path)
        previous_context = None
    output_path = Path(args.output_path) if args.output_path else settings.artifacts_dir / "replay.json"
    trace_path = (
        Path(args.trace_output_path)
        if args.trace_output_path
        else output_path.with_name(f"{output_path.stem}.traces.jsonl")
    )
    manifest_path = (
        Path(args.manifest_output_path)
        if args.manifest_output_path
        else output_path.with_name(f"{output_path.stem}.manifest.json")
    )
    if output_path.resolve() in {trace_path.resolve(), manifest_path.resolve()}:
        raise ReplayError("replay, trace, and manifest outputs must be different files")
    generation_provider = generation_provider_for_settings(settings)
    result = asyncio.run(
        replay_transcript(
            baseline_events,
            corpus=index,
            top_k=settings.retrieval_top_k,
            backend=backend,
            generation_provider=generation_provider,
            run_id=args.run_id,
            model_identity=settings.model_identity,
            embedding_cache_dir=settings.embedding_cache_dir,
            embedding_local_files_only=settings.embedding_local_files_only,
            execution_mode=args.execution_mode,
            previous_answer_context=previous_context,
            multi_intent=args.multi_intent,
            decomposition_provider=(
                decomposition_provider_for_settings(settings)
                if args.multi_intent
                else None
            ),
            retrieval_mode=(
                _selected_multi_intent_mode(args, settings, backend)
                if args.multi_intent
                else None
            ),
            context_budget_tokens=settings.multi_intent_context_budget_tokens,
            multi_intent_max_workers=settings.multi_intent_max_workers,
            multi_intent_rrf_k=settings.multi_intent_rrf_k,
            multi_intent_reranking_enabled=settings.multi_intent_reranking_enabled,
        )
    )
    write_replay(output_path, result)
    write_trace_jsonl(trace_path, result.traces)
    manifest = build_run_manifest(
        result,
        corpus=index,
        settings=settings,
        transcript_path=transcript_path,
        index_path=Path(args.index_path),
        result_path=output_path,
        trace_path=trace_path,
    )
    write_run_manifest(manifest_path, manifest)
    _json_print(
        {
            "status": result.run_status,
            "output": str(output_path),
            "run_id": result.run_id,
            "mode": "baseline",
            "retrieval_backend": result.retrieval_backend,
            "top_k": result.retrieval_top_k,
            "corpus_id": result.corpus_id,
            "index_id": result.index_id,
            "generation_backend": result.generation_backend,
            "generation_model": result.generation_model,
            "generation_status": result.generation_status,
            "generation_attempts": result.generation_attempts,
            "generation_repair_attempts": result.generation_repair_attempts,
            "generation_usage": result.generation_usage.model_dump(mode="json"),
            "generation_cost": result.generation_cost,
            "execution_mode": result.execution_mode,
            "run_status": result.run_status,
            "source_duration_s": result.source_duration_s,
            "execution_duration_ms": result.execution_duration_ms,
            "complete_answer_latency_ms": result.complete_answer_latency_ms,
            "answer_latency_from_final_event_delivery_ms": result.answer_latency_from_final_event_delivery_ms,
            "full_interaction_duration_ms": result.full_interaction_duration_ms,
            "controller_overhead_ms": result.controller_overhead_ms,
            "final_event_delivery_time_s": result.final_event_delivery_time_s,
            "evidence_ready_time_s": result.evidence_ready_time_s,
            "generation_started_time_s": result.generation_started_time_s,
            "generation_completed_time_s": result.generation_completed_time_s,
            "generation_first_content_time_s": result.generation_first_content_time_s,
            "retrieval_started_early": result.retrieval_started_early,
            "valid_evidence_ready_before_finalization": result.valid_evidence_ready_before_finalization,
            "early_evidence_reused": result.early_evidence_reused,
            "retrieval_call_count": result.retrieval_call_count,
            "scheduled_request_count": result.scheduled_request_count,
            "superseded_request_count": result.superseded_request_count,
            "cancelled_request_count": result.cancelled_request_count,
            "timed_out_request_count": result.timed_out_request_count,
            "retrieval_error_count": result.retrieval_error_count,
            "stale_result_discard_count": result.stale_result_discard_count,
            "retrieval_usage": result.retrieval_usage.model_dump(mode="json"),
            "error_count": len(result.errors),
            "duplicate_event_count": result.duplicate_event_count,
            "query": result.query,
            "retrieval_triggered": result.retrieval_triggered,
            "retrieval_hit_count": len(result.retrieval_hits),
            "answer_version": result.answer.answer_version,
            "trace_events": len(result.traces),
            "trace_output": str(trace_path),
            "manifest_output": str(manifest_path),
        }
    )
    return 1 if result.run_status == "failed" else 0


def command_evaluate(args: argparse.Namespace) -> int:
    settings = _settings(args.env_file)
    gold_path = Path(args.gold_path) if args.gold_path else settings.evaluation_path
    report = evaluate(
        load_evaluation_cases(gold_path),
        load_replay(Path(args.run_path)),
        load_index(Path(args.corpus_path)),
    )
    output_path = Path(args.output_path) if args.output_path else settings.artifacts_dir / "evaluation.json"
    write_report(output_path, report)
    _json_print(report.model_dump(mode="json"))
    return 0 if report.passed else 1


def command_evaluate_suite(args: argparse.Namespace) -> int:
    settings = _settings_for_args(args)
    index = _load_query_index(args, settings)
    evaluation_asset: str
    if args.cases_path:
        evaluation_asset = str(Path(args.cases_path))
        cases = load_evaluation_cases(
            Path(args.cases_path),
            expected_split=None if args.split == "all" else args.split,
        )
    elif args.split == "development":
        evaluation_asset = str(settings.evaluation_development_path)
        cases = load_evaluation_cases(settings.evaluation_development_path, expected_split="development")
    elif args.split == "held_out":
        evaluation_asset = str(settings.evaluation_held_out_path)
        cases = load_evaluation_cases(settings.evaluation_held_out_path, expected_split="held_out")
    else:
        evaluation_asset = (
            f"development={settings.evaluation_development_path};"
            f"held_out={settings.evaluation_held_out_path}"
        )
        cases = [
            *load_evaluation_cases(settings.evaluation_development_path, expected_split="development"),
            *load_evaluation_cases(settings.evaluation_held_out_path, expected_split="held_out"),
        ]
    selected_backend = args.backend or settings.retrieval_backend
    report = asyncio.run(
        evaluate_suite(
            cases,
            corpus=index,
            settings=settings,
            backend=selected_backend,
            top_k=settings.retrieval_top_k,
            execution_mode=args.execution_mode,
            run_id_prefix=args.run_id_prefix,
            evaluation_asset=evaluation_asset,
        )
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else settings.artifacts_dir / f"evaluation-{args.split}.json"
    )
    write_report(output_path, report)
    _json_print(
        {
            "status": "PASS" if report.passed else "FAIL",
            "output": str(output_path),
            "evaluation_label": report.evaluation_label,
            "selected_split": report.selected_split,
            "passed": report.passed,
            "asset_status": report.asset_status,
            "competition_performance_claim": report.competition_performance_claim,
            "sections": {
                key: {
                    "status": section.status,
                    "result_group": section.result_group,
                    "case_count": section.case_count,
                    "metrics": section.metrics,
                    "retrieval_latency": section.retrieval_latency.model_dump(mode="json"),
                    "answer_latency": section.answer_latency.model_dump(mode="json"),
                    "generation_usage": section.generation_usage.model_dump(mode="json"),
                    "generation_cost": section.generation_cost,
                    "error_count": section.error_count,
                    "trace_complete_count": section.trace_complete_count,
                }
                for key, section in report.sections.items()
            },
        }
    )
    return 0 if report.passed else 1


def command_evaluate_streaming(args: argparse.Namespace) -> int:
    settings = _settings_for_args(args)
    index = _load_query_index(args, settings)
    if args.cases_path:
        evaluation_asset = str(Path(args.cases_path))
        cases = load_streaming_evaluation_cases(
            Path(args.cases_path),
            expected_split=None if args.split == "all" else args.split,
        )
    elif args.split == "development":
        evaluation_asset = str(settings.streaming_evaluation_development_path)
        cases = load_streaming_evaluation_cases(
            settings.streaming_evaluation_development_path,
            expected_split="development",
        )
    elif args.split == "held_out":
        evaluation_asset = str(settings.streaming_evaluation_held_out_path)
        cases = load_streaming_evaluation_cases(
            settings.streaming_evaluation_held_out_path,
            expected_split="held_out",
        )
    else:
        evaluation_asset = (
            f"development={settings.streaming_evaluation_development_path};"
            f"held_out={settings.streaming_evaluation_held_out_path}"
        )
        cases = [
            *load_streaming_evaluation_cases(
                settings.streaming_evaluation_development_path,
                expected_split="development",
            ),
            *load_streaming_evaluation_cases(
                settings.streaming_evaluation_held_out_path,
                expected_split="held_out",
            ),
        ]
    selected_backend = args.backend or settings.retrieval_backend
    report = asyncio.run(
        evaluate_streaming_suite(
            cases,
            corpus=index,
            settings=settings,
            backend=selected_backend,
            top_k=settings.retrieval_top_k,
            execution_mode=args.execution_mode,
            run_id_prefix=args.run_id_prefix,
            compare_without_suppression=args.compare_disabled_scheduling,
        )
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else settings.artifacts_dir / f"streaming-evaluation-{args.split}.json"
    )
    write_streaming_evaluation_report(output_path, report)
    _json_print(
        {
            "status": "PASS" if report.passed else "FAIL",
            "output": str(output_path),
            "evaluation_asset": evaluation_asset,
            "evaluation_label": report.evaluation_label,
            "selected_split": report.selected_split,
            "execution_mode": report.execution_mode,
            "passed": report.passed,
            "competition_performance_claim": report.competition_performance_claim,
            "guide_early_retrieval_target": report.guide_early_retrieval_target,
            "guide_early_retrieval_target_met": report.guide_early_retrieval_target_met,
            "metrics": report.metrics,
            "duplicate_suppression_comparison": report.duplicate_suppression_comparison,
            "sections": {
                key: {
                    "status": section.status,
                    "case_count": section.case_count,
                    "metrics": section.metrics,
                    "trace_coverage": section.trace_coverage,
                    "cost_availability": section.cost_availability,
                }
                for key, section in report.sections.items()
            },
        }
    )
    return 0 if report.passed else 1


def command_evaluate_phase3(args: argparse.Namespace) -> int:
    """Run the denominator-corrected Phase 2 audit plus Phase 3 evaluation."""

    settings = _settings_for_args(args)
    index = _load_query_index(args, settings)
    selected_backend = args.backend or settings.retrieval_backend

    if args.phase2_cases_path:
        phase2_cases = load_streaming_evaluation_cases(Path(args.phase2_cases_path))
    elif args.split == "development":
        phase2_cases = load_streaming_evaluation_cases(
            settings.streaming_evaluation_development_path,
            expected_split="development",
        )
    elif args.split == "held_out":
        phase2_cases = load_streaming_evaluation_cases(
            settings.streaming_evaluation_held_out_path,
            expected_split="held_out",
        )
    else:
        phase2_cases = [
            *load_streaming_evaluation_cases(
                settings.streaming_evaluation_development_path,
                expected_split="development",
            ),
            *load_streaming_evaluation_cases(
                settings.streaming_evaluation_held_out_path,
                expected_split="held_out",
            ),
        ]

    phase3_development_path = Path(
        args.phase3_development_cases_path or settings.evaluation_development_path
    )
    phase3_held_out_path = Path(
        args.phase3_held_out_cases_path or settings.evaluation_held_out_path
    )
    if args.split == "development":
        phase3_cases = load_evaluation_cases(phase3_development_path, expected_split="development")
    elif args.split == "held_out":
        phase3_cases = load_evaluation_cases(phase3_held_out_path, expected_split="held_out")
    else:
        phase3_cases = [
            *load_evaluation_cases(phase3_development_path, expected_split="development"),
            *load_evaluation_cases(phase3_held_out_path, expected_split="held_out"),
        ]

    phase2_report = asyncio.run(
        evaluate_streaming_suite(
            phase2_cases,
            corpus=index,
            settings=settings,
            backend=selected_backend,
            top_k=settings.retrieval_top_k,
            execution_mode=args.execution_mode,
            run_id_prefix="phase3-phase2-audit",
        )
    )
    phase3_results = asyncio.run(
        evaluate_multi_intent_cases(
            phase3_cases,
            corpus=index,
            settings=settings,
            backend=selected_backend,
            top_k=settings.retrieval_top_k,
            execution_mode=args.execution_mode,
            retrieval_mode=(
                _selected_multi_intent_mode(args, settings, selected_backend)
            ),
            context_budget_tokens=settings.multi_intent_context_budget_tokens,
            multi_intent_max_workers=settings.multi_intent_max_workers,
            multi_intent_rrf_k=settings.multi_intent_rrf_k,
        )
    )
    output_path = Path(args.output_path)
    report = build_phase3_comparison(
        phase2_report=phase2_report,
        phase2_cases=phase2_cases,
        phase3_results=phase3_results,
        corpus=index,
        settings=settings,
        backend=selected_backend,
        top_k=settings.retrieval_top_k,
        execution_mode=args.execution_mode,
        historical_report=load_historical_report(Path("reports/phase2_streaming_evaluation.json")),
        retrieval_mode=_selected_multi_intent_mode(args, settings, selected_backend),
    )
    markdown_path = write_phase3_comparison_report(output_path, report)
    _json_print(
        {
            "status": "PASS",
            "output": str(output_path),
            "markdown_output": str(markdown_path),
            "selected_split": args.split,
            "phase2_case_count": len(phase2_cases),
            "phase3_case_count": len(phase3_cases),
            "matched_successful_case_count": report["historical_gap"]["matched_successful_case_count"],
            "matched_successful_baseline_recall_at_5": report["historical_gap"]["matched_successful_baseline_recall_at_5"],
            "matched_successful_streaming_recall_at_5": report["historical_gap"]["matched_successful_streaming_recall_at_5"],
            "non_useful_early_reuse_category_counts": report["phase2_investigation"]["overall_non_useful_category_counts"],
            "real_backend": report["sections"]["real_backend"]["status"],
            "official_assets": report["sections"]["official_assets"]["status"],
        }
    )
    return 0


def command_answer(args: argparse.Namespace) -> int:
    if args.run_path:
        result = load_replay(Path(args.run_path))
        payload = {
            "run_id": result.run_id,
            "session_id": result.session_id,
            "utterance_id": result.utterance_id,
            "mode": result.mode,
            "execution_mode": result.execution_mode,
            "run_status": result.run_status,
            "generation_status": result.generation_status,
            "generation_backend": result.generation_backend,
            "generation_model": result.generation_model,
            "generation_usage": result.generation_usage.model_dump(mode="json"),
            "generation_cost": result.generation_cost,
            "answer": result.answer.model_dump(mode="json"),
        }
        if args.output_path:
            _write_json(Path(args.output_path), payload)
            payload["output"] = args.output_path
        _json_print(payload)
        return 1 if result.run_status == "failed" else 0

    if not args.index_path:
        raise ReplayError("answer --query requires --index/--corpus")
    settings = _settings_for_args(args)
    index = _load_query_index(args, settings)
    backend = args.backend or settings.retrieval_backend
    output_path = Path(args.output_path) if args.output_path else settings.artifacts_dir / "answer.json"
    trace_path = (
        Path(args.trace_output_path)
        if args.trace_output_path
        else output_path.with_name(f"{output_path.stem}.traces.jsonl")
    )
    manifest_path = (
        Path(args.manifest_output_path)
        if args.manifest_output_path
        else output_path.with_name(f"{output_path.stem}.manifest.json")
    )
    if output_path.resolve() in {trace_path.resolve(), manifest_path.resolve()}:
        raise ReplayError("answer, trace, and manifest outputs must be different files")
    final_event = TranscriptEvent(
        session_id=args.session_id,
        utterance_id=args.utterance_id,
        event_id=f"{args.utterance_id}-final",
        sequence_number=0,
        source_timestamp_s=0.0,
        text=args.query,
        is_final=True,
        text_mode="cumulative",
    )
    result = asyncio.run(
        replay_transcript(
            [final_event],
            corpus=index,
            top_k=settings.retrieval_top_k,
            backend=backend,
            generation_provider=generation_provider_for_settings(settings),
            run_id=args.run_id,
            model_identity=settings.model_identity,
            embedding_cache_dir=settings.embedding_cache_dir,
            embedding_local_files_only=settings.embedding_local_files_only,
            execution_mode=args.execution_mode,
            multi_intent=args.multi_intent,
            decomposition_provider=(
                decomposition_provider_for_settings(settings)
                if args.multi_intent
                else None
            ),
            retrieval_mode=(
                _selected_multi_intent_mode(args, settings, backend)
                if args.multi_intent
                else None
            ),
            context_budget_tokens=settings.multi_intent_context_budget_tokens,
            multi_intent_max_workers=settings.multi_intent_max_workers,
            multi_intent_rrf_k=settings.multi_intent_rrf_k,
            multi_intent_reranking_enabled=settings.multi_intent_reranking_enabled,
        )
    )
    write_replay(output_path, result)
    write_trace_jsonl(trace_path, result.traces)
    manifest = build_run_manifest(
        result,
        corpus=index,
        settings=settings,
        transcript_path=Path("<inline-final-query>"),
        index_path=Path(args.index_path),
        result_path=output_path,
        trace_path=trace_path,
    )
    write_run_manifest(manifest_path, manifest)
    _json_print(
        {
            "status": result.run_status,
            "output": str(output_path),
            "trace_output": str(trace_path),
            "manifest_output": str(manifest_path),
            "run_id": result.run_id,
            "mode": "baseline",
            "retrieval_backend": result.retrieval_backend,
            "generation_backend": result.generation_backend,
            "generation_model": result.generation_model,
            "generation_status": result.generation_status,
            "generation_usage": result.generation_usage.model_dump(mode="json"),
            "generation_cost": result.generation_cost,
            "execution_mode": result.execution_mode,
            "complete_answer_latency_ms": result.complete_answer_latency_ms,
            "answer_latency_from_final_event_delivery_ms": result.answer_latency_from_final_event_delivery_ms,
            "full_interaction_duration_ms": result.full_interaction_duration_ms,
            "retrieval_hit_count": len(result.retrieval_hits),
            "answer_version": result.answer.answer_version,
            "answer": result.answer.model_dump(mode="json"),
        }
    )
    return 1 if result.run_status == "failed" else 0


def _fixture_contract_check() -> bool:
    try:
        TranscriptEvent(
            session_id="smoke-session",
            utterance_id="smoke-utterance",
            event_id="smoke-event",
            sequence_number=0,
            source_timestamp_s=0.0,
            text="hello",
            is_final=False,
            text_mode="not-a-mode",
        )
    except ValidationError:
        return True
    return False


def command_smoke(args: argparse.Namespace) -> int:
    settings = _settings(args.env_file)
    asset_status = config_asset_status(settings)
    if not all(asset_status[key] for key in ("corpus_exists", "transcript_exists", "evaluation_exists")):
        raise IngestionError("default synthetic fixture assets are missing; config-check reported their paths")

    # The offline smoke check deliberately selects lexical and mock separately;
    # it does not pretend that either is a real dense model run.
    fixture_settings = _settings(args.env_file, {"retrieval_backend": "lexical"})
    lexical_index, _, _ = build_index_from_source(
        fixture_settings.corpus_path,
        settings=fixture_settings,
        backend="lexical",
    )
    result = asyncio.run(
        replay_transcript(
            load_transcript(fixture_settings.transcript_path),
            corpus=lexical_index,
            top_k=fixture_settings.retrieval_top_k,
            backend="lexical",
            generation_provider=generation_provider_for_settings(fixture_settings),
            run_id="smoke-run",
            model_identity=fixture_settings.model_identity,
        )
    )
    report = evaluate(load_evaluation_cases(fixture_settings.evaluation_path), result, lexical_index)
    known_chunks = {chunk.chunk_id: chunk for chunk in lexical_index.chunks}
    hits_resolve = all(
        hit.chunk_id in known_chunks and hit.source_location == known_chunks[hit.chunk_id].source_location
        for hit in result.retrieval_hits
    )

    mock_index, _, _ = build_index_from_source(
        fixture_settings.corpus_path,
        settings=fixture_settings,
        backend="mock",
    )
    mock_retriever = make_retriever(mock_index, backend="mock", top_k=fixture_settings.retrieval_top_k)
    mock_hits = mock_retriever.search("workshop venue")
    mock_resolves = bool(mock_hits) and all(
        hit.chunk_id in {chunk.chunk_id for chunk in mock_index.chunks} for hit in mock_hits
    )
    streaming_result = asyncio.run(
        replay_streaming_transcript(
            load_transcript(Path("examples/streaming/early-retrieval.jsonl")),
            corpus=lexical_index,
            backend="lexical",
            run_id="smoke-streaming-run",
            controller_config=streaming_config_from_settings(fixture_settings),
            scheduler_config=scheduler_config_from_settings(fixture_settings),
            generation_provider=generation_provider_for_settings(fixture_settings),
        )
    )
    streaming_trace_types = {trace.event_type for trace in streaming_result.traces}
    checks = {
        "config_valid": True,
        "invalid_contract_rejected": _fixture_contract_check(),
        "lexical_fixture_indexed": bool(lexical_index.documents and lexical_index.chunks),
        "lexical_fixture_evaluation_passed": report.passed,
        "lexical_hits_resolve_to_source": hits_resolve,
        "mock_generation_completed": (
            result.generation_backend == "mock" and result.generation_status == "success"
        ),
        "generation_citations_validated": any(
            trace.attributes.get("citation_ids_validated") is True for trace in result.traces
        ),
        "mock_dense_fixture_check": mock_resolves,
        "streaming_controller_completed": streaming_result.run_status == "completed",
        "streaming_early_retrieval_observed": streaming_result.early_retrieval_count >= 1,
        "streaming_final_decision_recorded": streaming_result.final_decision.is_final_event,
        "streaming_trace_complete": {
            "streaming_replay_started",
            "transcript_event_received",
            "streaming_decision",
            "utterance_finalized",
            "streaming_retrieval_scheduled",
            "streaming_retrieval_started",
            "streaming_retrieval_completed",
            "streaming_answer_completed",
            "streaming_replay_completed",
        }.issubset(streaming_trace_types)
        and (
            "streaming_generation_started" in streaming_trace_types
            or "streaming_generation_skipped" in streaming_trace_types
        ),
    }
    passed = all(checks.values())
    dense_dependency_available = importlib.util.find_spec("sentence_transformers") is not None
    _json_print(
        {
            "status": "PASS" if passed else "FAIL",
            "mode": "offline",
            "fixture_backend": "lexical",
            "mock_backend": "hash_embedding",
            "dense_backend": {
                "status": "available_for_dense_smoke" if dense_dependency_available else "blocked",
                "reason": None
                if dense_dependency_available
                else "sentence-transformers is not installed in the target environment",
            },
            "competition_performance_claim": False,
            "phase2": {
                "mode": "streaming_controller_async_scheduler",
                "execution_mode": "accelerated",
                "generation": {
                    "backend": streaming_result.generation_backend,
                    "model": streaming_result.generation_model,
                    "status": streaming_result.generation_status,
                },
                "early_retrieval_count": streaming_result.early_retrieval_count,
                "retrieval_attempt_count": streaming_result.retrieval_attempt_count,
                "scheduled_request_count": streaming_result.scheduled_request_count,
                "unnecessary_retrieval_count": streaming_result.unnecessary_retrieval_count,
                "retrieval_started_early": streaming_result.retrieval_started_early,
                "valid_evidence_ready_before_finalization": streaming_result.valid_evidence_ready_before_finalization,
                "early_evidence_reused": streaming_result.early_evidence_reused,
            },
            "checks": checks,
        }
    )
    return 0 if passed else 1


def command_dense_smoke(args: argparse.Namespace) -> int:
    settings = _settings(args.env_file, {"retrieval_backend": "dense"})
    source_path = Path(args.input_path) if args.input_path else settings.corpus_path
    try:
        index, elapsed, _ = build_index_from_source(
            source_path,
            settings=settings,
            backend="dense",
        )
        retriever = make_retriever(
            index,
            backend="dense",
            top_k=args.top_k if args.top_k is not None else settings.retrieval_top_k,
            cache_dir=settings.embedding_cache_dir,
            local_files_only=settings.embedding_local_files_only,
        )
        response = RetrievalResponse(
            query=args.query,
            retrieval_backend=retriever.backend,
            index_id=index.manifest.index_id,
            hits=retriever.search(args.query),
        )
    except DenseRetrievalUnavailable as exc:
        _json_print(
            {
                "status": "BLOCKED",
                "selected_backend": "dense",
                "reason": str(exc),
                "model": settings.embedding_model,
                "revision": settings.embedding_revision,
                "license": settings.embedding_license,
                "competition_performance_claim": False,
            }
        )
        return 3
    _json_print(
        {
            "status": "PASS",
            "selected_backend": response.retrieval_backend,
            "model": index.manifest.embedding.model_dump(mode="json"),
            "documents": len(index.documents),
            "chunks": len(index.chunks),
            "indexing_seconds": round(elapsed, 6),
            "query": response.query,
            "hits": response.model_dump(mode="json")["hits"],
            "competition_performance_claim": False,
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "config-check":
            return command_config_check(args)
        if args.command == "inspect-corpus":
            return command_inspect_corpus(args)
        if args.command in {"build-index", "ingest"}:
            return command_build_index(args)
        if args.command == "retrieve":
            return command_retrieve(args)
        if args.command == "replay":
            return command_replay(args)
        if args.command == "evaluate":
            return command_evaluate(args)
        if args.command == "evaluate-suite":
            return command_evaluate_suite(args)
        if args.command == "evaluate-streaming":
            return command_evaluate_streaming(args)
        if args.command == "evaluate-phase3":
            return command_evaluate_phase3(args)
        if args.command == "answer":
            return command_answer(args)
        if args.command == "smoke":
            return command_smoke(args)
        if args.command == "dense-smoke":
            return command_dense_smoke(args)
    except (
        DenseRetrievalUnavailable,
        EmbeddingError,
        EvaluationError,
        Phase3EvaluationError,
        StreamingEvaluationError,
        IngestionError,
        RetrievalError,
        ReplayError,
        StaleIndexError,
        ValueError,
        ValidationError,
    ) as exc:
        if isinstance(exc, (DenseRetrievalUnavailable, EmbeddingError)):
            _json_print(
                {
                    "status": "BLOCKED",
                    "reason": str(exc),
                    "selected_backend": "dense",
                    "fallback": "none",
                    "execution_mode": getattr(args, "execution_mode", None),
                }
            )
            return 3
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
