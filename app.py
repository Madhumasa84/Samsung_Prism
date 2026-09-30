"""FlowContext — Samsung PRISM Theme 4 Streamlit demo.

A thin UI over the existing ``src/flowcontext`` backend. Nothing here
re-implements retrieval, decomposition, session updates, or generation:
every number, intent, chunk ID, score, claim and answer shown on screen is
read from the objects returned by FlowContext's own replay functions.

Run:
    uv sync --locked --python 3.11 --extra dense --extra demo
    uv run streamlit run app.py

Read-only by design: the app never writes indexes, corpus files, or
evaluation artifacts. Phase 4 sessions use an in-memory session store.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:  # works even if the project is not installed
    sys.path.insert(0, str(ROOT / "src"))

from flowcontext.config import load_settings  # noqa: E402
from flowcontext.config import Settings  # noqa: E402
from flowcontext.contracts import Phase4ReplayTurn, TranscriptEvent  # noqa: E402
from flowcontext.generation import (  # noqa: E402
    MockGenerationProvider,
    generation_provider_for_settings,
)
from flowcontext.ingestion import assert_index_matches_source, load_index  # noqa: E402
from flowcontext.multi_intent import decomposition_provider_for_settings  # noqa: E402
from flowcontext.phase4 import Phase4SessionStore  # noqa: E402
from flowcontext.phase4_replay import load_phase4_turns, replay_phase4_session  # noqa: E402
from flowcontext.replay import load_transcript  # noqa: E402
from flowcontext.retrieval import make_retriever  # noqa: E402
from flowcontext.scheduler import scheduler_config_from_settings  # noqa: E402
from flowcontext.streaming import (  # noqa: E402
    replay_streaming_transcript,
    streaming_config_from_settings,
)

try:  # backend's own completeness rule for streaming traces
    from flowcontext.streaming_evaluation import _trace_complete as _backend_trace_complete
except Exception:  # pragma: no cover - optional
    _backend_trace_complete = None

os.chdir(ROOT)  # FlowContext resolves relative paths (.env, artifacts/, data/) from repo root

# ---------------------------------------------------------------------------
# Scenario catalogue. Only *inputs* are defined here; all outputs come from the
# backend. Phase 4 turn text is taken, in order of preference, from a turns
# JSONL in the repo, else from the recorded replay artifact's step inputs,
# else from the default text below (identical to those artifacts' inputs).
# ---------------------------------------------------------------------------
SCENARIOS: dict[str, dict[str, Any]] = {
    "A": {
        "title": "A — Incremental multi-intent utterance / early retrieval",
        "kind": "streaming",
        "default_query": "Which venue in Pune hosts 30 people and what catering options exist?",
    },
    "B": {
        "title": "B — Late constraint addition",
        "kind": "phase4",
        "artifact": "artifacts/my-late-constraint.json",
        "session_id": "phase4-late-constraint",
        "default_turns": [
            "Which Venue B in Pune can host 40 attendees? What are the catering options?",
            "Actually, make it 30 attendees.",
        ],
    },
    "C": {
        "title": "C — Entity correction",
        "kind": "phase4",
        "artifact": "artifacts/my-entity-correction.json",
        "session_id": "phase4-entity-correction",
        "default_turns": [
            "Which Venue B in Pune can host 40 attendees?",
            "Actually, use Venue A instead and make it 30 attendees.",
        ],
    },
    "D": {
        "title": "D — Zero-retrieval formatting",
        "kind": "phase4",
        "artifact": "artifacts/my-formatting.json",
        "session_id": "phase4-formatting",
        "default_turns": [
            "Which Venue B in Pune can host 40 attendees? What are the catering options?",
            "Make it two bullets.",
        ],
    },
    "E": {
        "title": "E — Partial / unsupported constraint",
        "kind": "phase4",
        "artifact": "artifacts/my-partial-unsupported.json",
        "session_id": "phase4-partial-unsupported",
        "default_turns": [
            "Which Venue B in Pune can host 40 attendees? What are the catering options?",
            "Actually, make it 50 attendees and keep the catering question.",
        ],
    },
    "F": {
        "title": "F (optional) — Correction during generation / stale publication",
        "kind": "phase4",
        "race": True,
        "artifact": "artifacts/my-race.json",
        "session_id": "phase4-race",
        "default_turns": [
            "Which Venue B in Pune can host 40 attendees? What are the catering options?",
            "Actually, make it 30 attendees.",
        ],
    },
}

STAGE_LABELS_A = [
    "Listening / semantic instability",
    "Semantic stability detected",
    "Speculative retrieval triggered",
    "Multi-intent decomposition",
    "Final utterance",
    "Grounded synthesis",
]

# ---------------------------------------------------------------------------
# Page + light styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FlowContext — Streaming Live RAG", page_icon="🎙️", layout="wide")
st.markdown(
    """
<style>
.fc-stage {border-left: 4px solid #1428A0; padding: .45rem .8rem; margin: .35rem 0;
           background: rgba(20,40,160,.05); border-radius: 0 6px 6px 0;}
.fc-stage.done {border-left-color: #0a8f5a; background: rgba(10,143,90,.06);}
.fc-stage.warn {border-left-color: #c77700; background: rgba(199,119,0,.07);}
.fc-stage.pending {opacity: .35;}
.fc-badge {display:inline-block; padding: 2px 10px; border-radius: 12px; font-size: .78rem;
           font-weight: 600; margin-right: 6px;}
.fc-mock {background:#fff3cd; color:#7a5a00;}
.fc-real {background:#d1f2e1; color:#0a5c38;}
.fc-muted {color: #6b7280; font-size: .85rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Settings, index, providers (cached, read-only)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_settings():
    return load_settings()  # .env + process environment, exactly as the CLI does


@st.cache_resource(show_spinner="Loading corpus index…")
def get_index(index_path: str):
    index = load_index(Path(index_path))
    note = "no source file found; freshness not checked"
    src = index.manifest.source_path
    if src:
        candidates = [Path(src), Path(index_path).parent / src]
        found = next((c for c in candidates if c.is_file()), None)
        if found is not None:
            s = get_settings()
            assert_index_matches_source(
                index,
                found,
                max_chars=s.chunk_max_chars,
                overlap_chars=s.chunk_overlap_chars,
                max_bytes=s.corpus_max_bytes,
                max_documents=s.corpus_max_documents,
                max_line_bytes=s.corpus_max_line_bytes,
            )
            note = f"verified fresh against {found}"
    return index, note


@st.cache_resource(show_spinner="Preparing retriever…")
def get_retriever(index_path: str, backend: str, top_k: int):
    index, _ = get_index(index_path)
    s = get_settings()
    return make_retriever(
        index,
        backend=backend,
        top_k=top_k,
        cache_dir=s.embedding_cache_dir,
        local_files_only=s.embedding_local_files_only,
    )


def generation_mode(settings) -> dict[str, Any]:
    """Describe the configured generation backend without revealing secrets."""
    backend = settings.generation_backend
    return {
        "backend": backend,
        "provider": settings.generation_provider,
        "model": settings.generation_model,
        "is_real": backend == "openai_compatible",
        "api_key_env": settings.generation_api_key_env,
        "api_key_configured": bool(os.environ.get(settings.generation_api_key_env)),
    }


def demo_generation_settings(cfg: dict[str, Any]) -> Settings:
    settings = get_settings()
    source = cfg.get("generation_source", "Configured")
    overrides: dict[str, Any] = {}
    if source == "Mock":
        overrides = {"generation_backend": "mock", "generation_provider": "flowcontext.mock",
                     "generation_model": "mock-grounded-v1"}
    elif source == "Local Ollama":
        overrides = {
            "generation_backend": "openai_compatible", "generation_provider": "ollama",
            "generation_model": cfg["ollama_model"], "generation_base_url": cfg["ollama_url"],
            "generation_timeout_s": cfg["ollama_timeout_s"], "generation_max_retries": 0,
            "generation_max_output_tokens": 1200,
        }
    return Settings.model_validate({**settings.model_dump(), **overrides})


def discover_indexes() -> list[str]:
    found = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "artifacts").glob("*index*.json"))
    return found or ["artifacts/corpus-index.json"]


def discover_files(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        out.extend(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern) if p.is_file())
    return sorted(set(out))


def index_chunk_map(index) -> dict[str, Any]:
    return {c.chunk_id: c for c in index.chunks}


def dump(obj: Any) -> Any:
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj


# ---------------------------------------------------------------------------
# Scenario inputs
# ---------------------------------------------------------------------------


def scripted_transcript(query: str, word_interval_s: float, end_pause_s: float, session_id: str):
    """Build cumulative TranscriptEvents for a scripted utterance (demo INPUT only).

    Words arrive every ``word_interval_s`` of *source* time; the final marker
    arrives ``end_pause_s`` after the last word (end-of-speech pause). These
    are input timestamps, not measurements.
    """
    words = query.split()
    events = []
    for i in range(1, len(words) + 1):
        events.append(
            TranscriptEvent(
                session_id=session_id,
                utterance_id=f"{session_id}-u",
                event_id=f"{session_id}-{i:03d}",
                sequence_number=i - 1,
                source_timestamp_s=round(word_interval_s * (i - 1), 3),
                text=" ".join(words[:i]).rstrip("?.!"),
                is_final=False,
                text_mode="cumulative",
            )
        )
    events.append(
        TranscriptEvent(
            session_id=session_id,
            utterance_id=f"{session_id}-u",
            event_id=f"{session_id}-final",
            sequence_number=len(words),
            source_timestamp_s=round(word_interval_s * (len(words) - 1) + end_pause_s, 3),
            text=query,
            is_final=True,
            text_mode="cumulative",
        )
    )
    return events


def phase4_turns_for(key: str, turns_file: str | None) -> tuple[list[Phase4ReplayTurn], str]:
    spec = SCENARIOS[key]
    if turns_file:
        return load_phase4_turns(ROOT / turns_file), f"turns file `{turns_file}`"
    artifact = ROOT / spec["artifact"]
    if artifact.is_file():
        try:
            data = json.loads(artifact.read_text(encoding="utf-8"))
            steps = [s for s in data.get("steps", []) if s.get("classification") != "superseded_initial_generation"]
            seen, turns = set(), []
            for s in steps:
                if s["turn_index"] in seen:
                    continue
                seen.add(s["turn_index"])
                turns.append(
                    Phase4ReplayTurn(
                        session_id=data["session_id"],
                        turn_index=s["turn_index"],
                        utterance_id=s["utterance_id"],
                        text=s["text"],
                    )
                )
            turns.sort(key=lambda t: t.turn_index)
            if turns:
                return turns, f"turn inputs of `{spec['artifact']}` (outputs are recomputed live)"
        except Exception:
            pass
    turns = [
        Phase4ReplayTurn(session_id=spec["session_id"], turn_index=i, utterance_id=f"u-{i}", text=t)
        for i, t in enumerate(spec["default_turns"])
    ]
    return turns, "built-in default turn text"


# ---------------------------------------------------------------------------
# Backend execution (the only place FlowContext is invoked)
# ---------------------------------------------------------------------------


def run_scenario_a(cfg: dict[str, Any]) -> dict[str, Any]:
    settings = demo_generation_settings(cfg)
    index, _ = get_index(cfg["index_path"])
    if cfg["transcript_source"] == "scripted":
        events = scripted_transcript(cfg["query"], cfg["word_interval_s"], cfg["end_pause_s"], "demo-a")
    else:
        events = load_transcript(ROOT / cfg["transcript_source"])
    t0 = time.perf_counter()
    result = asyncio.run(
        replay_streaming_transcript(
            events,
            corpus=index,
            top_k=cfg["top_k"],
            backend=cfg["base_backend"],
            run_id="streamlit-demo-a",
            model_identity="flowcontext.streaming-controller.v1",
            embedding_cache_dir=settings.embedding_cache_dir,
            embedding_local_files_only=settings.embedding_local_files_only,
            execution_mode=cfg["execution_mode"],
            controller_config=streaming_config_from_settings(settings),
            scheduler_config=scheduler_config_from_settings(settings),
            generation_provider=generation_provider_for_settings(settings),
            decomposition_provider=decomposition_provider_for_settings(settings) if cfg["multi_intent"] else None,
            multi_intent=cfg["multi_intent"],
            retrieval_mode=cfg["retrieval_mode"] if cfg["multi_intent"] else None,
            context_budget_tokens=settings.multi_intent_context_budget_tokens,
            multi_intent_max_workers=settings.multi_intent_max_workers,
            multi_intent_rrf_k=settings.multi_intent_rrf_k,
            multi_intent_reranking_enabled=settings.multi_intent_reranking_enabled,
        )
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    return {"kind": "streaming", "result": result, "events": events, "app_wall_ms": wall_ms}


def run_phase4(key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    settings = demo_generation_settings(cfg)
    index, _ = get_index(cfg["index_path"])
    retriever = get_retriever(cfg["index_path"], cfg["base_backend"], cfg["top_k"])
    turns, turn_source = phase4_turns_for(key, cfg.get("turns_file"))
    provider = generation_provider_for_settings(settings)
    race_follow_up, replay_turns = None, turns
    if SCENARIOS[key].get("race"):
        if len(turns) < 2:
            raise ValueError("race scenario needs an initial turn and one correction turn")
        race_follow_up, replay_turns = turns[1], [turns[0], *turns[2:]]
        if provider.config.backend == "mock":  # same as `flowcontext phase4-replay --race`
            provider = MockGenerationProvider(config=provider.config, timeout_delay_s=0.05)
    store = Phase4SessionStore(max_sessions=8, max_records_per_session=256)  # in-memory, isolated per run
    t0 = time.perf_counter()
    result = asyncio.run(
        replay_phase4_session(
            replay_turns,
            corpus=index,
            retriever=retriever,
            generation_provider=provider,
            store=store,
            race_follow_up=race_follow_up,
        )
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    return {"kind": "phase4", "result": result, "turns": turns, "turn_source": turn_source, "app_wall_ms": wall_ms}


# ---------------------------------------------------------------------------
# Stage builders — turn backend output into an ordered list of reveal steps
# ---------------------------------------------------------------------------


def stages_for_streaming(run: dict[str, Any]) -> list[dict[str, Any]]:
    r = run["result"]
    traces = list(r.traces)
    final_trace = next((t for t in traces if t.event_type == "final_event_delivered"), None)
    final_exec = final_trace.monotonic_execution_time_s if final_trace else None
    stages = []
    for t in traces:
        et = t.event_type
        attrs = t.attributes or {}
        label = None
        if et == "transcript_event_received":
            label = "Listening"
        elif et == "streaming_decision":
            dec = attrs.get("decision")
            label = "Semantic stability detected" if dec == "RETRIEVE" else "Listening / semantic instability"
        elif et in ("streaming_retrieval_scheduled", "streaming_retrieval_started"):
            early = final_exec is None or t.monotonic_execution_time_s < final_exec
            label = "Speculative retrieval triggered" if early else "Final retrieval"
        elif et.startswith("streaming_retrieval_") or et == "streaming_decomposition_obsolete":
            label = "Retrieval lifecycle"
        elif "decomposition" in et:
            label = "Multi-intent decomposition"
        elif et in ("final_event_delivered", "utterance_finalized"):
            label = "Final utterance"
        elif et in ("streaming_evidence_ready", "streaming_generation_started"):
            label = "Evidence ready / generation"
        elif et in ("streaming_generation_completed", "streaming_answer_completed"):
            label = "Grounded synthesis"
        else:
            label = "Run bookkeeping"
        stages.append({"label": label, "trace": t})
    return stages


def stages_for_phase4(run: dict[str, Any]) -> list[dict[str, Any]]:
    r = run["result"]
    versions = {v.answer_version: v for v in r.state.answer_versions}
    stages: list[dict[str, Any]] = []
    for step in r.steps:
        cls = step["classification"]
        ver = step.get("current_answer_version")
        if cls == "initial":
            stages.append({"label": f"Answer v{ver}", "step": step, "version": versions.get(ver)})
            continue
        if cls == "superseded_initial_generation":
            stages.append({"label": "Older generation result arrives late", "step": step})
            continue
        patch, plan = step.get("patch") or {}, step.get("plan")
        stages.append({"label": "Follow-up interpretation", "step": step, "patch": patch})
        if plan:
            stages.append({"label": "Affected intent identified", "step": step, "plan": plan})
            stages.append({"label": "Affected evidence / claims invalidated", "step": step, "plan": plan})
            stages.append({"label": "Targeted retrieval", "step": step, "plan": plan})
        else:
            stages.append({"label": "No semantic change — presentation only", "step": step, "patch": patch})
        stages.append({"label": f"Answer v{ver}", "step": step, "version": versions.get(ver)})
    return stages


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def stage_box(label: str, body_md: str = "", css: str = "done") -> None:
    st.markdown(f"<div class='fc-stage {css}'><b>{label}</b></div>", unsafe_allow_html=True)
    if body_md:
        st.markdown(body_md)


def render_provenance(answer: dict[str, Any], chunk_map: dict[str, Any], claim_records: list[dict] | None = None,
                      evidence_records: list[dict] | None = None) -> None:
    claims = answer.get("factual_claims") or []
    if not claims:
        st.info("No factual claims in this answer version.")
        return
    ev_by_id = {e["evidence_id"]: e for e in (evidence_records or [])}
    rec_by_claim: dict[str, dict] = {}
    for rec in claim_records or []:
        if rec.get("answer_version") == answer.get("answer_version"):
            rec_by_claim[rec["claim_id"]] = rec
    rows = []
    for c in claims:
        rec = rec_by_claim.get(c["claim_id"], {})
        for cid in c.get("supporting_chunk_ids", []):
            chunk = chunk_map.get(cid)
            ev_ids = [
                e for e in rec.get("supporting_evidence_ids", [])
                if ev_by_id.get(e, {}).get("passage", {}).get("chunk_id") == cid
            ]
            rows.append({
                "Claim ID": c["claim_id"],
                "Citation (as rendered)": f"[{cid}]",
                "Evidence ID": ", ".join(ev_ids) or "—",
                "Document ID": chunk.document_id if chunk else "not in loaded index",
                "Chunk ID": cid,
                "Support": c.get("semantic_support", "—"),
            })
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption("FlowContext cites by chunk ID; there is no separate citation-ID namespace, so none is invented here.")
    for c in claims:
        for cid in c.get("supporting_chunk_ids", []):
            chunk = chunk_map.get(cid)
            with st.expander(f"Supporting passage — {cid}"):
                st.markdown(f"**Claim:** {c['claim_text']}")
                if chunk:
                    st.markdown(f"**Corpus passage** (`{chunk.source_location}`):")
                    st.write(chunk.text)
                else:
                    st.warning("Chunk not present in the loaded index; passage cannot be shown.")


def render_uncertainty(answer: dict[str, Any], version: Any | None = None) -> None:
    statuses = answer.get("intent_statuses") or []
    unsupported = [s for s in statuses if s.get("status") != "answered"]
    unresolved = list(getattr(version, "unresolved_questions", []) or []) if version is not None else []
    if unsupported or unresolved:
        st.warning(
            "The requested information could not be verified from the available corpus evidence for "
            "the item(s) below. No unsupported claim was generated."
        )
        if unsupported:
            st.dataframe(
                [{"Intent": s["intent_id"], "Status": s["status"], "Backend reason": s.get("reason")} for s in unsupported],
                width="stretch", hide_index=True,
            )
        for q in unresolved:
            st.markdown(f"- Unresolved: _{q}_")
    if answer.get("uncertainty"):
        with st.expander("Backend uncertainty statement"):
            st.write(answer["uncertainty"])


def render_hits(hits: list[dict[str, Any]], chunk_map: dict[str, Any]) -> None:
    if not hits:
        st.info("No retrieval hits were returned.")
        return
    rows = []
    for h in hits:
        chunk = chunk_map.get(h["chunk_id"])
        rows.append({
            "Rank": h.get("rank"),
            "Chunk ID": h["chunk_id"],
            "Document ID": chunk.document_id if chunk else "—",
            "Method": h.get("retrieval_method"),
            "Score": h.get("score"),
            "RRF score": h.get("rrf_score"),
            "Per-intent scores": json.dumps(h.get("intent_scores") or {}) if h.get("intent_scores") else "—",
            "Intent(s)": ", ".join(h.get("intent_ids") or ([h["intent_id"]] if h.get("intent_id") else [])) or "—",
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption("Scores are shown exactly as returned; empty cells mean the backend did not report that value.")


def render_streaming(run: dict[str, Any], upto: int, chunk_map: dict[str, Any]) -> None:
    r = run["result"]
    stages = stages_for_streaming(run)
    shown = stages[:upto]
    events = run["events"]
    final_trace = next((t for t in r.traces if t.event_type == "final_event_delivered"), None)
    first_early = next(
        (t for t in r.traces if t.event_type == "streaming_retrieval_started"
         and (final_trace is None or t.monotonic_execution_time_s < final_trace.monotonic_execution_time_s)),
        None,
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Live transcript")
        latest_src = max((s["trace"].source_timestamp_s or 0.0) for s in shown) if shown else -1
        heard = [e for e in events if e.source_timestamp_s <= latest_src]
        text = heard[-1].text if heard else ""
        is_final = bool(heard and heard[-1].is_final)
        st.markdown(f"### {'✅' if is_final else '🎙️'} {text or '…'}")
        st.caption(f"{len(heard)}/{len(events)} transcript events delivered (source time {max(latest_src, 0):.2f}s)")

        st.subheader("Pipeline stages reached")
        reached = {s["label"] for s in shown}
        for label in STAGE_LABELS_A:
            css = "done" if label in reached else "pending"
            stage_box(label, css=css)

    with right:
        st.subheader("Event log (FlowContext traces)")
        rows = []
        for s in shown[-25:]:
            t = s["trace"]
            a = t.attributes or {}
            rows.append({
                "exec ms": round(t.monotonic_execution_time_s * 1000, 2),
                "source s": t.source_timestamp_s,
                "event": t.event_type,
                "detail": " · ".join(str(x) for x in (a.get("decision"), a.get("reason_code"), a.get("proposed_query") or a.get("query")) if x),
            })
        st.dataframe(rows, width="stretch", hide_index=True, height=420)

    if upto < len(stages):
        return

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieval started before final", "YES" if r.retrieval_started_early else "NO")
    c2.metric("Evidence ready before final", "YES" if r.valid_evidence_ready_before_finalization else "NO")
    c3.metric("Early evidence reused", "YES" if r.early_evidence_reused else "NO")
    c4.metric("Early / final retrievals", f"{r.early_retrieval_count} / {r.final_retrieval_count}")
    if first_early is not None and final_trace is not None:
        st.success(
            f"First speculative retrieval started at {first_early.monotonic_execution_time_s*1000:.1f} ms "
            f"(source t={first_early.source_timestamp_s}s); final utterance delivered at "
            f"{final_trace.monotonic_execution_time_s*1000:.1f} ms (source t={final_trace.source_timestamp_s}s). "
            "Both timestamps are read from the backend trace."
        )

    st.subheader("Controller decisions")
    st.dataframe(
        [{"source s": d.source_timestamp_s, "decision": d.decision, "reason": d.reason_code,
          "proposed query": d.proposed_query, "final": d.is_final_event} for d in r.decisions],
        width="stretch", hide_index=True,
    )

    st.subheader("Multi-intent decomposition")
    decomp = r.decomposition if isinstance(r.decomposition, dict) else dump(r.decomposition) if r.decomposition else None
    if decomp:
        st.caption(f"method: `{decomp.get('decomposition_method')}` · provider: `{decomp.get('provider')}` · status: `{decomp.get('status')}`")
        st.dataframe(
            [{"#": i.get("ordinal"), "Intent ID": i["intent_id"], "Sub-query": i["query"], "Relationship": i.get("relationship"),
              "Constraints": ", ".join(f"{c['kind']}={c['value']}" for c in i.get("constraints", [])) or "—"}
             for i in decomp.get("intents", [])],
            width="stretch", hide_index=True,
        )
    else:
        st.info("Multi-intent mode was off or the backend returned no decomposition.")

    st.subheader("Evidence used for the answer")
    render_hits([dump(h) for h in r.current_evidence_hits], chunk_map)

    st.subheader("Grounded answer")
    answer = dump(r.answer)
    st.write(answer.get("answer_text", ""))
    render_provenance(answer, chunk_map)
    render_uncertainty(answer)

    st.subheader("Telemetry")
    trace_ok, missing = (None, [])
    if _backend_trace_complete is not None:
        try:
            trace_ok, missing = _backend_trace_complete(
                r.traces, mode="streaming", expected_no_retrieval=False, run_status=r.run_status
            )
        except Exception:
            trace_ok = None
    tele = {
        "Run status": r.run_status,
        "Trace events": len(r.traces),
        "Trace coverage (backend rule)": ("complete" if trace_ok else f"missing: {missing}") if trace_ok is not None else "unavailable",
        "Retrieval trigger (RETRIEVE) decisions": sum(1 for d in r.decisions if d.decision == "RETRIEVE"),
        "Retrieval calls": r.retrieval_call_count,
        "Scheduled / superseded / cancelled": f"{r.scheduled_request_count} / {r.superseded_request_count} / {r.cancelled_request_count}",
        "Stale results discarded": r.stale_result_discard_count,
        "Generation backend / status": f"{r.generation_backend} / {r.generation_status}",
        "Answer latency after final (ms)": round(r.answer_latency_from_final_event_delivery_ms, 2)
        if r.answer_latency_from_final_event_delivery_ms is not None else "unavailable",
        "Generation tokens (cost proxy)": f"{r.generation_usage.total_tokens}{' (estimated)' if r.generation_usage.estimated else ''}",
        "Retrieval tokens (cost proxy)": f"{r.retrieval_usage.total_tokens}{' (estimated)' if r.retrieval_usage.estimated else ''}",
        "Provider billing": r.generation_cost if r.generation_cost != "unavailable" else "unavailable (no billing data)",
        "App-measured wall clock of replay call (ms)": round(run["app_wall_ms"], 1),
    }
    st.dataframe([{"Metric": k, "Value": str(v)} for k, v in tele.items()], width="stretch", hide_index=True)


def render_phase4(key: str, run: dict[str, Any], upto: int, chunk_map: dict[str, Any]) -> None:
    r = run["result"]
    state = dump(r.state)
    stages = stages_for_phase4(run)
    st.caption(f"Turn inputs from {run['turn_source']}.")

    st.subheader("Session timeline")
    for idx, s in enumerate(stages):
        if idx >= upto:
            stage_box(s["label"], css="pending")
            continue
        step, label = s["step"], s["label"]
        if "version" in s:
            v = s["version"]
            if v is None:
                stage_box(label, "_Version not found in session state._", "warn")
                continue
            vd = dump(v)
            css = "warn" if step.get("answer_status") == "partial" else "done"
            stage_box(label, f"> **Turn {step['turn_index']}:** “{step['text']}”", css)
            m = st.columns(5)
            m[0].metric("Change kind", vd["change_kind"])
            m[1].metric("Retrieval calls", vd["retrieval_call_count"])
            m[2].metric("Generation attempts", vd["generation_attempts"])
            m[3].metric("Status", vd["status"])
            m[4].metric("Parent version", vd["parent_version"] if vd["parent_version"] is not None else "—")
            st.write(vd["answer"]["answer_text"])
            if vd.get("claim_changes"):
                st.dataframe(
                    [{"Change": c["change_type"], "Claim": c["claim_id"], "Reason": c.get("reason")} for c in vd["claim_changes"]],
                    width="stretch", hide_index=True,
                )
            if vd.get("evidence_removed_ids") or vd.get("evidence_added_ids"):
                st.markdown(
                    f"Evidence removed: `{', '.join(vd['evidence_removed_ids']) or 'none'}` · "
                    f"added: `{', '.join(vd['evidence_added_ids']) or 'none'}` · "
                    f"reused: `{', '.join(vd['reused_evidence_ids']) or 'none'}`"
                )
        elif label == "Follow-up interpretation":
            p = s["patch"]
            stage_box(label, f"> “{step['text']}” → classification **`{p.get('classification')}`** "
                             f"({p.get('execution_mode')}, `{p.get('provider_model')}`)")
            if p.get("decision_trace"):
                st.dataframe(
                    [{"stage": d["stage"], "rule": d["rule"], "detail": d["detail"]} for d in p["decision_trace"]],
                    width="stretch", hide_index=True,
                )
            if p.get("replacement_constraints"):
                st.markdown("Replacement constraints: " + ", ".join(
                    f"`{rc['constraint']['kind']}={rc['constraint']['value']}`" for rc in p["replacement_constraints"].values()))
        elif label == "No semantic change — presentation only":
            p = s["patch"]
            stage_box(label, f"Classification `{p.get('classification')}` · new intents: "
                             f"**{len(p.get('new_intents') or []) or 'NONE'}** · format instruction: "
                             f"`{json.dumps(p.get('format_instruction'))}`")
        elif label == "Affected intent identified":
            pl = s["plan"]
            stage_box(label, f"Directly affected: `{', '.join(pl['directly_affected_intent_ids']) or 'none'}` · "
                             f"new intents: `{', '.join(pl['new_intent_ids']) or 'none'}` · "
                             f"entities: `{', '.join(pl['affected_entity_ids']) or 'none'}`")
        elif label == "Affected evidence / claims invalidated":
            pl = s["plan"]
            stage_box(label)
            st.dataframe([
                {"Item": "Invalidated claims", "IDs": ", ".join(pl["invalidated_claim_ids"]) or "NONE"},
                {"Item": "Preserved claims", "IDs": ", ".join(pl["preserved_claim_ids"]) or "NONE"},
                {"Item": "Invalidated evidence", "IDs": ", ".join(pl["invalidated_evidence_ids"]) or "NONE"},
                {"Item": "Reused evidence", "IDs": ", ".join(pl["reused_evidence_ids"]) or "NONE"},
            ], width="stretch", hide_index=True)
            for cid, why in (pl.get("invalidation_reasons") or {}).items():
                st.markdown(f"- `{cid}`: {why}")
        elif label == "Targeted retrieval":
            pl = s["plan"]
            tasks = pl.get("retrieval_tasks") or []
            found = sum(len(t.get("evidence_ids") or []) for t in tasks)
            css = "warn" if tasks and found == 0 else "done"
            stage_box(label, f"Targeted retrieval calls: **{pl['retrieval_call_count']}** · full corpus rerun: **NO** "
                             f"(only intents `{', '.join(pl['retrieval_intent_ids']) or 'none'}` were queried)", css)
            if tasks:
                st.dataframe(
                    [{"Intent": t["intent_id"], "Query": t["query"], "Reason": t["reason"], "Status": t["status"],
                      "Evidence admitted": len(t.get("evidence_ids") or []), "Error": t.get("error") or ""} for t in tasks],
                    width="stretch", hide_index=True,
                )
                if found == 0:
                    st.warning("No supporting evidence found for the targeted intent — nothing was fabricated.")
        elif label == "Older generation result arrives late":
            pub = step.get("publication") or {}
            stage_box(label, f"Publication: **{pub.get('status')}** · published: `{pub.get('published')}` · "
                             f"reason: {pub.get('reason') or '—'}", "warn")

    if upto < len(stages):
        return

    st.divider()
    current = next((v for v in r.state.answer_versions if v.answer_version == r.state.current_answer_version), None)
    st.subheader(f"Current answer — v{r.state.current_answer_version} ({r.state.answer_status})")
    if current is not None:
        cur = dump(current)
        st.write(cur["answer"]["answer_text"])
        st.subheader("Citations and provenance")
        render_provenance(cur["answer"], chunk_map, state["claim_records"], state["evidence_records"])
        render_uncertainty(cur["answer"], current)

    st.subheader("Telemetry")
    pubs = [s.get("publication") or {} for s in r.steps]
    stale = sum(1 for p in pubs if p and not p.get("published"))
    tele = {
        "Run status": r.run_status,
        "Answer versions": " → ".join(f"v{v.answer_version} ({v.change_kind})" for v in r.state.answer_versions),
        "Retrieval calls (non-presentation versions)": r.retrieval_call_count,
        "Generation calls / attempts": f"{r.generation_call_count} / {r.generation_attempt_count}",
        "Superseded requests": len(r.state.superseded_requests),
        "Stale (rejected) publications": stale,
        "Full corpus rerun on follow-up": "NO (selective plan only)" if any(s.get("plan") for s in r.steps) else "n/a",
        "Retrieval tokens (cost proxy)": sum(v.retrieval_usage.total_tokens for v in r.state.answer_versions),
        "Generation tokens (cost proxy)": sum(v.generation_usage.total_tokens for v in r.state.answer_versions),
        "Update latency": "not reported by Phase 4 backend",
        "App-measured wall clock of replay call (ms)": round(run["app_wall_ms"], 1),
        "Retrieval backend": r.retrieval_backend,
        "Generation mode": f"{r.generation_execution_mode} ({r.generation_provider} / {r.generation_model})",
    }
    st.dataframe([{"Metric": k, "Value": str(v)} for k, v in tele.items()], width="stretch", hide_index=True)
    if r.notes:
        with st.expander("Backend notes"):
            for n in r.notes:
                st.markdown(f"- {n}")


# ---------------------------------------------------------------------------
# Session state + controls
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("run", None)
ss.setdefault("run_key", None)
ss.setdefault("upto", 0)
ss.setdefault("playing", False)
ss.setdefault("error", None)


def reset_state() -> None:
    ss.run, ss.run_key, ss.upto, ss.playing, ss.error = None, None, 0, False, None


try:
    settings = get_settings()
except Exception as exc:  # malformed .env etc.
    st.error(f"FlowContext configuration could not be loaded: {exc}")
    st.stop()
gen = generation_mode(settings)

# ---- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Demo controls")
    key = st.radio("Scenario", list(SCENARIOS), format_func=lambda k: SCENARIOS[k]["title"], key="scenario")
    spec = SCENARIOS[key]

    st.subheader("Answer generation")
    generation_sources = ["Configured", "Mock", "Local Ollama"]
    generation_source = st.selectbox("Generation", generation_sources,
                                    index=2 if settings.generation_provider == "ollama" else 0)
    generation_cfg: dict[str, Any] = {"generation_source": generation_source}
    if generation_source == "Local Ollama":
        generation_cfg["ollama_model"] = st.text_input(
            "Ollama model", settings.generation_model if settings.generation_provider == "ollama" else "qwen2.5:3b")
        generation_cfg["ollama_url"] = st.text_input(
            "Local Ollama server", settings.generation_base_url if settings.generation_provider == "ollama"
            else "http://127.0.0.1:11434/v1")
        generation_cfg["ollama_timeout_s"] = st.slider("Model request timeout (seconds)", 30, 120, 120, 10)
        st.caption("Start Ollama and install the selected model. Local generation does not require an API key.")

    st.subheader("Corpus & retrieval")
    indexes = discover_indexes()
    default_idx = indexes.index(str(settings.index_path)) if str(settings.index_path) in indexes else 0
    index_path = st.selectbox("Index", indexes, index=default_idx)
    backend_opts = ["dense", "lexical"]
    base_backend = st.selectbox(
        "Retrieval backend", backend_opts,
        index=backend_opts.index(settings.retrieval_backend) if settings.retrieval_backend in backend_opts else 0,
        help="Dense uses the index's persisted embeddings; no silent fallback to lexical.",
    )
    top_k = st.number_input("top-k", 1, 20, settings.retrieval_top_k)

    cfg: dict[str, Any] = {**generation_cfg, "index_path": index_path,
                           "base_backend": base_backend, "top_k": int(top_k)}
    if spec["kind"] == "streaming":
        st.subheader("Scenario A input")
        transcripts = discover_files(["examples/**/*.jsonl", "data/**/transcript*.jsonl"])
        cfg["transcript_source"] = st.selectbox(
            "Transcript", ["scripted"] + transcripts,
            format_func=lambda v: "Scripted utterance (below)" if v == "scripted" else v,
        )
        if cfg["transcript_source"] == "scripted":
            cfg["query"] = st.text_area("Utterance", spec["default_query"])
            cfg["word_interval_s"] = st.slider("Source time per word (s)", 0.1, 1.0, 0.35, 0.05)
            cfg["end_pause_s"] = st.slider("End-of-speech pause before final marker (s)", 0.0, 2.0, 0.8, 0.1)
        cfg["multi_intent"] = st.checkbox("Multi-intent decomposition", True)
        mode_opts = ["dense", "lexical", "hybrid"]
        cfg["retrieval_mode"] = st.selectbox(
            "Multi-intent retrieval mode", mode_opts,
            index=mode_opts.index(settings.multi_intent_retrieval_mode) if settings.multi_intent_retrieval_mode in mode_opts else 0,
            help="hybrid = dense + lexical fused with RRF (k from FLOWCONTEXT_MULTI_INTENT_RRF_K).",
        )
        cfg["execution_mode"] = st.radio(
            "Execution", ["realtime", "accelerated"], horizontal=True,
            help="realtime waits real source-time gaps, so speculative retrieval can finish before the final event.",
        )
    else:
        turn_files = discover_files(["examples/**/*.jsonl", "data/**/phase4*.jsonl"])
        tf = st.selectbox("Turns source", ["auto"] + turn_files,
                          format_func=lambda v: "auto (artifact inputs → defaults)" if v == "auto" else v)
        cfg["turns_file"] = None if tf == "auto" else tf

    st.subheader("Playback")
    speed = st.select_slider("Speed", [0.25, 0.5, 1.0, 2.0, 4.0], value=1.0)

    st.divider()
    try:
        gen = generation_mode(demo_generation_settings(cfg))
    except ValueError as exc:
        st.error(f"Generation settings are invalid: {exc}")
        st.stop()
    badge = "fc-real" if gen["is_real"] else "fc-mock"
    label = "REAL LLM" if gen["is_real"] else "MOCK GENERATION"
    st.markdown(f"<span class='fc-badge {badge}'>{label}</span>", unsafe_allow_html=True)
    st.caption(f"backend `{gen['backend']}` · provider `{gen['provider']}` · model `{gen['model']}`")
    if gen["provider"] == "ollama":
        st.caption("Local Ollama · no API key required · answers still undergo grounding validation.")
    elif gen["is_real"]:
        st.caption(f"API key env `{gen['api_key_env']}`: {'configured' if gen['api_key_configured'] else 'NOT set'}")
    else:
        st.caption("Answers come from FlowContext's deterministic mock provider — not a real LLM.")

run_key = json.dumps({"scenario": key, **cfg}, sort_keys=True, default=str)
if ss.run_key is not None and ss.run_key != run_key:
    reset_state()  # any scenario/config change → no state leakage

# ---- Header ----------------------------------------------------------------
st.title("FlowContext — Streaming Live RAG")
st.caption("Samsung PRISM Theme 4 · every value below is produced by the FlowContext backend at run time")
st.markdown(f"## {spec['title']}")

try:
    index_obj, freshness = get_index(index_path)
    chunk_map = index_chunk_map(index_obj)
    emb = index_obj.manifest.embedding
    st.caption(
        f"Index `{index_obj.manifest.index_id}` · corpus `{index_obj.corpus_id}` ({index_obj.source_kind}) · "
        f"{len(index_obj.chunks)} chunks · embeddings: `{emb.backend}` "
        f"{emb.model_name or ''} {('· ' + str(emb.dimensions) + ' dims') if emb.dimensions else ''} · {freshness}"
    )
except Exception as exc:
    st.error(f"Index could not be loaded: {exc}")
    if not (ROOT / index_path).is_file():
        st.info("Build an index once from the repository root, then reload this page.")
        st.code(shlex.join([
            "uv", "run", "flowcontext", "build-index",
            "--input", "data/synthetic/phase4_documents.jsonl",
            "--output", index_path, "--backend", base_backend,
            "--source-kind", "synthetic_fixture",
        ]), language="bash")
        st.caption("Choose lexical for setup without a model download. See the README for CPU dense setup.")
    st.stop()

b1, b2, b3, _ = st.columns([1, 1, 1, 5])
play = b1.button("▶ Play", type="primary", width="stretch")
pause = b2.button("⏸ Pause", width="stretch")
reset = b3.button("↺ Reset", width="stretch")

if reset:
    reset_state()
if pause:
    ss.playing = False
if play:
    ss.error = None
    if ss.run is None:
        try:
            with st.spinner("Running FlowContext…"):
                ss.run = run_scenario_a(cfg) if spec["kind"] == "streaming" else run_phase4(key, cfg)
            ss.run_key = run_key
            ss.upto = 0
        except Exception as exc:  # retrieval / generation / config failures surface, never faked
            ss.error = f"{type(exc).__name__}: {exc}"
            ss.run = None
    if ss.run is not None:
        total_now = len(stages_for_streaming(ss.run) if ss.run["kind"] == "streaming" else stages_for_phase4(ss.run))
        if ss.upto >= total_now:
            ss.upto = 0  # Play after the end replays the same recorded run from the start
        ss.playing = True

if ss.error:
    st.error(f"Backend run failed — nothing is displayed instead of real output.\n\n`{ss.error}`")
    if "dense" in ss.error.lower() or "sentence" in ss.error.lower() or "embedding" in ss.error.lower():
        st.info("Dense retrieval needs `uv sync --extra dense` and the embedding model available locally "
                "(or network access). Choose the lexical backend explicitly if that is intended.")

if ss.run is None:
    st.info("Press **Play** to execute this scenario through the FlowContext backend.")
    st.stop()

run = ss.run
stages = stages_for_streaming(run) if run["kind"] == "streaming" else stages_for_phase4(run)
total = len(stages)
st.progress(min(ss.upto, total) / max(total, 1), text=f"Stage {min(ss.upto, total)} of {total}")

if run["kind"] == "streaming":
    render_streaming(run, ss.upto, chunk_map)
else:
    render_phase4(key, run, ss.upto, chunk_map)

# ---- Playback loop -----------------------------------------------------
if ss.playing and ss.upto < total:
    if run["kind"] == "streaming":
        # step size keeps a ~60-80 event trace watchable; delay follows real trace gaps (clamped)
        cur = stages[ss.upto]["trace"]
        nxt = stages[min(ss.upto + 1, total - 1)]["trace"]
        gap = max(0.0, (nxt.source_timestamp_s or 0) - (cur.source_timestamp_s or 0))
        delay = min(max(gap, 0.12), 0.8) / speed
    else:
        delay = 1.4 / speed
    time.sleep(delay)
    ss.upto += 1
    if ss.upto >= total:
        ss.playing = False
    st.rerun()
