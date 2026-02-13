"""Event-based trace models for debug pipeline output."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass
class TraceEvent:
    """One chronological debug trace event."""

    sequence: int
    timestamp: float
    stage: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventTrace:
    """Chronological event log for pipeline debugging."""

    events: list[TraceEvent] = field(default_factory=list)
    _next_sequence: int = 0

    def emit(self, event_type: str, stage: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append(
            TraceEvent(
                sequence=self._next_sequence,
                timestamp=time.time(),
                stage=stage,
                event_type=event_type,
                payload=payload or {},
            )
        )
        self._next_sequence += 1

    def find_events(self, event_type: str) -> list[TraceEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def latest_payload(self, event_type: str) -> dict[str, Any] | None:
        for event in reversed(self.events):
            if event.event_type == event_type:
                return event.payload
        return None


def _truncate_text(value: str, max_text_len: int) -> str:
    if len(value) <= max_text_len:
        return value
    return value[:max_text_len] + f"... [truncated, total {len(value)} chars]"


def _to_jsonable(value: Any, max_text_len: int) -> Any:
    if isinstance(value, BaseModel):
        return _to_jsonable(value.model_dump(), max_text_len)
    if isinstance(value, str):
        return _truncate_text(value, max_text_len)
    if isinstance(value, list):
        return [_to_jsonable(v, max_text_len) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v, max_text_len) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v, max_text_len) for k, v in value.items()}
    return value


def event_trace_to_jsonable(trace: EventTrace, max_text_len: int = 2000) -> dict[str, Any]:
    """Convert EventTrace into compact step-first JSON payload."""

    skip_steps = {"docling_export", "graph_export", "visualization"}

    def _step_name(event: TraceEvent) -> str:
        if event.event_type == "page_markdown_extracted":
            return "docling_conversion"
        if event.stage == "graph_conversion":
            return "graph_mapping"
        if event.stage == "extraction":
            return "data_extraction"
        if event.stage == "docling_export":
            return "docling_export"
        if event.stage == "visualization":
            return "visualization"
        if event.stage == "export":
            return "graph_export"
        return event.stage

    sorted_events = sorted(trace.events, key=lambda e: e.sequence)
    ordered_step_names: list[str] = []
    steps_by_name: dict[str, dict[str, Any]] = {}

    for event in sorted_events:
        payload = _to_jsonable(event.payload, max_text_len)
        step_name = _step_name(event)

        if step_name not in steps_by_name:
            ordered_step_names.append(step_name)
            steps_by_name[step_name] = {
                "started_at": event.timestamp,
                "finished_at": event.timestamp,
                "artifacts": {},
                "had_failure": False,
            }

        step_obj = steps_by_name[step_name]
        step_obj["finished_at"] = event.timestamp
        if "failed" in event.event_type:
            step_obj["had_failure"] = True
        artifacts = step_obj["artifacts"]

        if event.event_type == "page_markdown_extracted":
            artifacts.setdefault("pages", []).append(payload)
        elif event.event_type in ("extraction_completed", "extraction_failed"):
            artifacts.setdefault("extractions", []).append(payload)
        elif event.event_type == "structured_output_fallback_triggered":
            artifacts.setdefault("fallbacks", []).append(payload)
        elif event.event_type == "staged_trace_emitted":
            artifacts.setdefault("staged_traces", []).append(payload)
        elif event.event_type == "graph_created":
            artifacts["graph"] = payload
        elif event.event_type == "export_written":
            artifacts.setdefault("exports", []).append(payload)
        elif event.event_type == "pipeline_started":
            artifacts["start"] = payload
        elif event.event_type == "pipeline_finished":
            artifacts["finish"] = payload
        elif event.event_type == "pipeline_failed":
            artifacts["failure"] = payload

    docling_conversion = steps_by_name.get("docling_conversion", {})
    data_extraction = steps_by_name.get("data_extraction", {})
    graph_mapping = steps_by_name.get("graph_mapping", {})

    page_count = len((docling_conversion.get("artifacts") or {}).get("pages", []))
    extraction_artifacts = (data_extraction.get("artifacts") or {}).get("extractions", [])
    extraction_success = any(
        isinstance(item, dict) and item.get("error") in (None, "") for item in extraction_artifacts
    )
    fallback_used = bool((data_extraction.get("artifacts") or {}).get("fallbacks"))
    graph_payload = (graph_mapping.get("artifacts") or {}).get("graph", {})
    node_count = graph_payload.get("node_count", 0) if isinstance(graph_payload, dict) else 0
    edge_count = graph_payload.get("edge_count", 0) if isinstance(graph_payload, dict) else 0

    overall_processing_time = 0.0
    if sorted_events:
        overall_processing_time = max(0.0, sorted_events[-1].timestamp - sorted_events[0].timestamp)

    steps_out: list[dict[str, Any]] = []
    for step_name in ordered_step_names:
        if step_name in skip_steps:
            continue
        step_obj = steps_by_name[step_name]
        processing_time = max(0.0, step_obj["finished_at"] - step_obj["started_at"])
        status = "failed" if step_obj.get("had_failure") else "success"
        artifacts = step_obj.get("artifacts", {})
        if step_name == "pipeline":
            # Keep high-signal pipeline context only; summary already carries final counts.
            start_payload = artifacts.get("start", {})
            failure_payload = artifacts.get("failure")
            artifacts = {
                "mode": start_payload.get("mode"),
                "source": start_payload.get("source"),
                "processing_mode": start_payload.get("processing_mode"),
                "backend": start_payload.get("backend"),
                "inference": start_payload.get("inference"),
                "debug": start_payload.get("debug"),
            }
            if failure_payload is not None:
                artifacts["failure"] = failure_payload
        steps_out.append(
            {
                "name": step_name,
                "runtime_seconds": round(processing_time, 4),
                "status": status,
                "artifacts": artifacts,
            }
        )

    out: dict[str, Any] = {
        "summary": {
            "runtime_seconds": round(overall_processing_time, 4),
            "page_count": page_count,
            "extraction_success": extraction_success,
            "fallback_used": fallback_used,
            "node_count": node_count,
            "edge_count": edge_count,
        },
        "steps": steps_out,
    }
    return out
