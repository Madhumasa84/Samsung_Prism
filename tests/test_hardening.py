"""Regression coverage for persistence, freshness, and resource-boundary hardening."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from flowcontext import cli
from flowcontext.config import load_settings
from flowcontext.contracts import EvidencePassage, GenerationConfig, GenerationRequest
from flowcontext.generation import GenerationProviderError, OpenAICompatibleGenerationProvider
from flowcontext.indexing import build_index_from_source
from flowcontext.ingestion import IngestionError, StaleIndexError, load_document_inputs
from flowcontext.phase4 import Phase4Error, Phase4SessionStore
from flowcontext.storage import atomic_write_text


class _FakeResponse:
    status = 200

    def __init__(self, body: bytes, content_length: str | None = None) -> None:
        self._body = body
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int = -1) -> bytes:
        return self._body


class _StatLyingPath:
    """Path facade used to exercise growth between stat and read."""

    suffix = ".jsonl"

    def __init__(self, path: Path) -> None:
        self.path = path

    def is_file(self) -> bool:
        return True

    def stat(self) -> SimpleNamespace:
        return SimpleNamespace(st_size=0)

    def open(self, *args: object, **kwargs: object):
        return self.path.open(*args, **kwargs)

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

    def __str__(self) -> str:
        return str(self.path)


class HardeningTests(unittest.TestCase):
    def test_atomic_write_preserves_previous_complete_artifact_on_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.json"
            target.write_text('{"status":"old"}\n', encoding="utf-8")
            with mock.patch("flowcontext.storage.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic_write_text(target, '{"status":"new"}\n')
            self.assertEqual(target.read_text(encoding="utf-8"), '{"status":"old"}\n')
            self.assertEqual(list(Path(directory).glob(".result.json.*.tmp")), [])

    def test_corpus_loader_rejects_oversized_input_and_lines(self) -> None:
        payload = {
            "document_id": "doc-1",
            "source_location": "fixture://doc-1",
            "text": "small document",
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "docs.jsonl"
            source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaises(IngestionError):
                load_document_inputs(source, max_bytes=4)
            with self.assertRaises(IngestionError):
                load_document_inputs(source, max_line_bytes=4)

    def test_corpus_loader_bounds_read_when_file_grows_after_stat(self) -> None:
        payload = {
            "document_id": "doc-1",
            "source_location": "fixture://doc-1",
            "text": "document larger than the limit",
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "docs.jsonl"
            source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaises(IngestionError):
                load_document_inputs(_StatLyingPath(source), max_bytes=8)

    def test_query_index_uses_embedded_source_fingerprint_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "docs.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "document_id": "doc-1",
                        "source_location": "fixture://doc-1",
                        "text": "Venue A has parking.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            index_path = root / "index.json"
            settings = load_settings(environ={}).model_copy(update={"retrieval_backend": "lexical"})
            build_index_from_source(
                source,
                settings=settings,
                backend="lexical",
                output_path=index_path,
            )
            args = SimpleNamespace(index_path=str(index_path), source_path=None)
            loaded = cli._load_query_index(args, settings)
            self.assertEqual(loaded.manifest.source_path, str(source))
            source.write_text(
                json.dumps(
                    {
                        "document_id": "doc-1",
                        "source_location": "fixture://doc-1",
                        "text": "Venue A has no parking.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(StaleIndexError):
                cli._load_query_index(args, settings)

    def test_generation_request_limit_fails_before_network_call(self) -> None:
        config = GenerationConfig(
            backend="openai_compatible",
            provider="test-provider",
            model="test-model",
            base_url="https://example.invalid/v1",
            api_key_env="FLOWCONTEXT_TEST_KEY",
            max_request_bytes=100,
            max_response_bytes=1024,
        )
        provider = OpenAICompatibleGenerationProvider(config=config)
        passage = EvidencePassage(
            chunk_id="chunk-1",
            source_location="fixture://doc-1",
            text="x" * 500,
            rank=1,
            score=1.0,
            retrieval_method="test",
        )
        request = GenerationRequest(query="question", passages=[passage])
        with mock.patch.dict(os.environ, {"FLOWCONTEXT_TEST_KEY": "secret"}), mock.patch(
            "flowcontext.generation.urlopen"
        ) as urlopen:
            with self.assertRaises(GenerationProviderError):
                provider._generate_once(request)
        urlopen.assert_not_called()

    def test_generation_response_limit_rejects_declared_oversize(self) -> None:
        config = GenerationConfig(
            backend="openai_compatible",
            provider="test-provider",
            model="test-model",
            base_url="https://example.invalid/v1",
            api_key_env="FLOWCONTEXT_TEST_KEY",
            max_request_bytes=1024 * 1024,
            max_response_bytes=4,
        )
        provider = OpenAICompatibleGenerationProvider(config=config)
        passage = EvidencePassage(
            chunk_id="chunk-1",
            source_location="fixture://doc-1",
            text="supported passage",
            rank=1,
            score=1.0,
            retrieval_method="test",
        )
        request = GenerationRequest(query="question", passages=[passage])
        for content_length in ("5", None):
            with self.subTest(content_length=content_length), mock.patch.dict(
                os.environ, {"FLOWCONTEXT_TEST_KEY": "secret"}
            ), mock.patch(
                "flowcontext.generation.urlopen",
                return_value=_FakeResponse(b"12345", content_length=content_length),
            ):
                with self.assertRaises(GenerationProviderError):
                    provider._generate_once(request)

    def test_durable_session_store_rejects_tampered_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "phase4-sessions.json"
            snapshot.write_text(
                '{"schema_version":"flowcontext.phase4-session-store.v1",'
                '"sessions":[{"session_id":"tampered"}]}\n',
                encoding="utf-8",
            )
            with self.assertRaises(Phase4Error):
                Phase4SessionStore(storage_path=snapshot)

    def test_phase4_review_status_defaults_beside_report(self) -> None:
        args = cli.build_parser().parse_args(
            ["evaluate-phase4", "--corpus", "index.json", "--output", "reports/run.json"]
        )
        self.assertIsNone(args.review_status_output)
        self.assertEqual(
            cli._phase4_review_status_path(args),
            Path("reports/run_label_review_status.json"),
        )
        args.review_status_output = "data/evaluation/explicit.json"
        self.assertEqual(
            cli._phase4_review_status_path(args),
            Path("data/evaluation/explicit.json"),
        )


if __name__ == "__main__":
    unittest.main()
