"""Structured execution trace collection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal, Sequence

from .contracts import ExecutionTrace, TraceError, Usage


class TraceCollector:
    """Create monotonic, relative trace events for one run/session."""

    def __init__(self, run_id: str, session_id: str, model_identity: str) -> None:
        self.run_id = run_id
        self.session_id = session_id
        self.model_identity = model_identity
        self._started = time.monotonic()
        self._counter = 0
        self.events: list[ExecutionTrace] = []

    def elapsed_s(self) -> float:
        """Return a replay-relative monotonic timestamp.

        Absolute monotonic clock values are useful internally for ordering but
        are needlessly machine-specific in reports.  All public timing fields
        emitted by this collector use this relative domain.
        """

        return max(0.0, time.monotonic() - self._started)

    def add(
        self,
        event_type: str,
        *,
        source_timestamp_s: float | None = None,
        duration_ms: float = 0,
        usage: Usage | None = None,
        cost: float | Literal["unavailable"] = "unavailable",
        model_identity: str | None = None,
        error: TraceError | None = None,
        attributes: dict[str, Any] | None = None,
        actual_delivery_time_s: float | None = None,
    ) -> ExecutionTrace:
        self._counter += 1
        execution_time_s = self.elapsed_s()
        event = ExecutionTrace(
            trace_id=f"{self.run_id}-trace-{self._counter:04d}",
            run_id=self.run_id,
            session_id=self.session_id,
            event_type=event_type,
            source_timestamp_s=source_timestamp_s,
            monotonic_execution_time_s=execution_time_s,
            actual_delivery_time_s=actual_delivery_time_s,
            duration_ms=duration_ms,
            model_identity=model_identity or self.model_identity,
            usage=usage or Usage(),
            cost=cost,
            error=error,
            attributes=attributes or {},
        )
        self.events.append(event)
        return event

    def add_error(
        self,
        event_type: str,
        error: Exception,
        *,
        source_timestamp_s: float | None = None,
        model_identity: str | None = None,
        attributes: dict[str, Any] | None = None,
        actual_delivery_time_s: float | None = None,
    ) -> ExecutionTrace:
        return self.add(
            event_type,
            source_timestamp_s=source_timestamp_s,
            model_identity=model_identity,
            error=TraceError(error_type=type(error).__name__, message=str(error)),
            attributes=attributes,
            actual_delivery_time_s=actual_delivery_time_s,
        )


def write_trace_jsonl(path: Path, events: Sequence[ExecutionTrace]) -> None:
    """Write one validated trace event per JSONL line."""

    if not events:
        raise ValueError("cannot write an empty trace file")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(event.model_dump_json() + "\n" for event in events),
        encoding="utf-8",
    )
