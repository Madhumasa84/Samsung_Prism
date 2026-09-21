"""Focused command-line regression checks."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from flowcontext import cli
from flowcontext.contracts import EmbeddingConfig


class CliTests(unittest.TestCase):
    def test_dense_smoke_local_files_only_reaches_embedding_settings(self) -> None:
        args = cli.build_parser().parse_args(["dense-smoke", "--local-files-only"])
        observed: dict[str, object] = {}
        index = SimpleNamespace(
            manifest=SimpleNamespace(
                index_id="test-index",
                embedding=EmbeddingConfig(
                    backend="dense",
                    provider="sentence-transformers",
                    model_name="test-model",
                    revision="test-revision",
                    license="apache-2.0",
                    dimensions=2,
                    normalize=True,
                    download_required=True,
                ),
            ),
            documents=[object()],
            chunks=[object()],
        )
        retriever = SimpleNamespace(backend="dense", search=lambda query: [])

        def fake_build_index(source_path, *, settings, backend):
            observed["settings"] = settings
            observed["backend"] = backend
            return index, 0.0, "built"

        with (
            mock.patch.object(cli, "build_index_from_source", side_effect=fake_build_index),
            mock.patch.object(cli, "make_retriever", return_value=retriever),
        ):
            result = cli.command_dense_smoke(args)

        self.assertEqual(result, 0)
        self.assertEqual(observed["backend"], "dense")
        self.assertTrue(observed["settings"].embedding_local_files_only)


if __name__ == "__main__":
    unittest.main()
