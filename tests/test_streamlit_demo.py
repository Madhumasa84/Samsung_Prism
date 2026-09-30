"""Submission demo startup and playback checks (requires the demo extra)."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

st = pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo_checkout(tmp_path, monkeypatch):
    shutil.copyfile(ROOT / 'app.py', tmp_path / 'app.py')
    (tmp_path / 'src').symlink_to(ROOT / 'src', target_is_directory=True)
    (tmp_path / 'data').symlink_to(ROOT / 'data', target_is_directory=True)
    (tmp_path / 'artifacts').mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('FLOWCONTEXT_RETRIEVAL_BACKEND', 'lexical')
    monkeypatch.setenv('FLOWCONTEXT_MULTI_INTENT_RETRIEVAL_MODE', 'lexical')
    monkeypatch.setenv('FLOWCONTEXT_GENERATION_BACKEND', 'mock')
    monkeypatch.setenv('FLOWCONTEXT_INDEX_PATH', 'artifacts/corpus-index.json')
    st.cache_resource.clear()
    yield tmp_path
    st.cache_resource.clear()


def build_displayed_index(app, checkout):
    command = shlex.split(next(block.value for block in app.code if 'build-index' in block.value))
    assert command[:3] == ['uv', 'run', 'flowcontext']
    env = dict(os.environ, PYTHONPATH=str(checkout / 'src'))
    subprocess.run([sys.executable, '-m', 'flowcontext.cli', *command[3:]],
                   cwd=checkout, env=env, check=True, capture_output=True, text=True)


def test_missing_index_instructions_build_a_working_index(demo_checkout):
    app = AppTest.from_file(str(demo_checkout / 'app.py'), default_timeout=30).run()
    assert not app.exception
    assert any('Index could not be loaded' in item.value for item in app.error)
    assert any('Build an index once' in item.value for item in app.info)
    build_displayed_index(app, demo_checkout)
    app.run()
    assert not app.error and not app.exception
    assert any(button.label == 'Play' for button in app.button)


def test_formatting_and_controls_work_without_evaluation_archive(demo_checkout):
    app = AppTest.from_file(str(demo_checkout / 'app.py'), default_timeout=30).run()
    build_displayed_index(app, demo_checkout)
    app.run()
    app.sidebar.radio[0].set_value('D').run()
    app.button[0].click().run()
    assert not app.error and not app.exception
    version = app.session_state['run']['result'].state.answer_versions[-1]
    assert version.change_kind == 'presentation'
    assert version.retrieval_call_count == 0
    assert version.generation_attempts == 0
    app.button[1].click().run()
    assert not app.session_state['playing']
    generation = next(select for select in app.sidebar.selectbox if select.label == 'Generation')
    generation.set_value('Local Ollama').run()
    assert app.session_state['run'] is None
    assert not app.error and not app.exception
    assert next(field.value for field in app.sidebar.text_input if field.label == 'Ollama model') == 'qwen2.5:3b'
    app.button[2].click().run()
    assert app.session_state['run'] is None
    assert app.session_state['upto'] == 0


def test_editable_request_runs_and_changes_clear_previous_result(demo_checkout):
    app = AppTest.from_file(str(demo_checkout / 'app.py'), default_timeout=30).run()
    build_displayed_index(app, demo_checkout)
    app.run()
    query = 'Which venue in Pune hosts 40 people and what catering options exist?'
    app.text_area[0].set_value(query)
    next(control for control in app.sidebar.radio if control.label == 'Execution').set_value('accelerated')
    app.run()
    app.button[0].click().run()
    assert not app.error and not app.exception
    assert app.session_state['run']['events'][-1].text == query
    app.text_area[0].set_value(query.replace('40', '30')).run()
    assert app.session_state['run'] is None
    assert not app.session_state['playing']
