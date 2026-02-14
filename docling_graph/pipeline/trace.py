"""Event-based trace models for debug pipeline output."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

EXCLUDED_EXPORTED_STEPS = {"docling_export", "graph_export", "visualization"}
STAGE_EXPORT_NAME_ALIASES = {
    "extraction": "data_extraction",
    "graph_conversion": "graph_mapping",
    "export": "graph_export",
}
EVENT_EXPORT_NAME_ALIASES = {
    # Conversion artifacts are emitted from extraction strategies; exported as a distinct step.
    "page_markdown_extracted": "docling_conversion",
    "docling_conversion_completed": "docling_conversion",
}


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


@dataclass
class _StepAccumulator:
    """Internal aggregation bucket for one exported trace step."""

    first_timestamp: float
    last_timestamp: float
    artifacts: dict[str, Any] = field(default_factory=dict)
    had_failure: bool = False

    def add_event(self, event_type: str, timestamp: float) -> None:
        self.last_timestamp = timestamp
        if "failed" in event_type:
            self.had_failure = True


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


def _resolve_export_step_name(event: TraceEvent) -> str:
    """Map internal event/stage names to exported step names."""
    if event.event_type in EVENT_EXPORT_NAME_ALIASES:
        return EVENT_EXPORT_NAME_ALIASES[event.event_type]
    return STAGE_EXPORT_NAME_ALIASES.get(event.stage, event.stage)


def _add_event_artifact(step: _StepAccumulator, event: TraceEvent, payload: dict[str, Any]) -> None:
    """Accumulate canonical artifacts for a step from one event payload."""
    artifacts = step.artifacts
    if event.event_type == "page_markdown_extracted":
        artifacts.setdefault("pages", []).append(payload)
    elif event.event_type == "docling_conversion_completed":
        artifacts["conversion"] = payload
    elif event.event_type in ("extraction_completed", "extraction_failed"):
        artifacts.setdefault("extractions", []).append(payload)
    elif event.event_type == "structured_output_fallback_triggered":
        artifacts.setdefault("fallbacks", []).append(payload)
    elif event.event_type == "staged_trace_emitted":
        artifacts.setdefault("staged_traces", []).append(payload)
    elif event.event_type == "delta_trace_emitted":
        artifacts.setdefault("delta_traces", []).append(payload)
    elif event.event_type == "delta_failed_then_direct_fallback":
        artifacts.setdefault("delta_fallbacks", []).append(payload)
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


def _compute_step_runtime_seconds(step_name: str, step: _StepAccumulator) -> float:
    """Compute step runtime in seconds, with extraction fallback from payload durations."""
    runtime_seconds = max(0.0, step.last_timestamp - step.first_timestamp)

    # data_extraction often has a single event, so timestamp delta can be 0.
    # Use extraction payload durations when present.
    if step_name == "data_extraction" and runtime_seconds == 0.0:
        extraction_times = [
            float(item["extraction_time"])
            for item in step.artifacts.get("extractions", [])
            if isinstance(item, dict) and item.get("extraction_time") is not None
        ]
        if extraction_times:
            runtime_seconds = sum(extraction_times)
    if step_name == "docling_conversion":
        conversion = step.artifacts.get("conversion", {})
        if isinstance(conversion, dict):
            conversion_runtime = conversion.get("runtime_seconds")
            if conversion_runtime is not None:
                # Event timestamp deltas can be near-zero even when conversion took time.
                runtime_seconds = max(runtime_seconds, float(conversion_runtime))
    return runtime_seconds


def _summarize_pipeline_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    """Keep only high-signal pipeline context in exported artifacts."""
    start_payload = artifacts.get("start", {})
    failure_payload = artifacts.get("failure")
    out = {
        "mode": start_payload.get("mode"),
        "source": start_payload.get("source"),
        "processing_mode": start_payload.get("processing_mode"),
        "backend": start_payload.get("backend"),
        "inference": start_payload.get("inference"),
        "debug": start_payload.get("debug"),
    }
    if failure_payload is not None:
        out["failure"] = failure_payload
    return out


def event_trace_to_jsonable(trace: EventTrace, max_text_len: int = 2000) -> dict[str, Any]:
    """Convert EventTrace into compact step-first JSON payload."""

    sorted_events = sorted(trace.events, key=lambda e: e.sequence)
    ordered_step_names: list[str] = []
    steps_by_name: dict[str, _StepAccumulator] = {}

    for event in sorted_events:
        payload = _to_jsonable(event.payload, max_text_len)
        step_name = _resolve_export_step_name(event)

        if step_name not in steps_by_name:
            ordered_step_names.append(step_name)
            steps_by_name[step_name] = _StepAccumulator(
                first_timestamp=event.timestamp,
                last_timestamp=event.timestamp,
            )

        step = steps_by_name[step_name]
        step.add_event(event.event_type, event.timestamp)
        _add_event_artifact(step, event, payload)

    docling_conversion = steps_by_name.get("docling_conversion")
    data_extraction = steps_by_name.get("data_extraction")
    graph_mapping = steps_by_name.get("graph_mapping")

    page_count = len((docling_conversion.artifacts if docling_conversion else {}).get("pages", []))
    extraction_artifacts = (data_extraction.artifacts if data_extraction else {}).get(
        "extractions", []
    )
    extraction_success = any(
        isinstance(item, dict) and item.get("error") in (None, "") for item in extraction_artifacts
    )
    fallback_used = bool((data_extraction.artifacts if data_extraction else {}).get("fallbacks"))
    graph_payload = (graph_mapping.artifacts if graph_mapping else {}).get("graph", {})
    node_count = graph_payload.get("node_count", 0) if isinstance(graph_payload, dict) else 0
    edge_count = graph_payload.get("edge_count", 0) if isinstance(graph_payload, dict) else 0

    overall_runtime_seconds = 0.0
    if sorted_events:
        overall_runtime_seconds = max(0.0, sorted_events[-1].timestamp - sorted_events[0].timestamp)

    steps_out: list[dict[str, Any]] = []
    for step_name in ordered_step_names:
        if step_name in EXCLUDED_EXPORTED_STEPS:
            continue
        step = steps_by_name[step_name]
        runtime_seconds = _compute_step_runtime_seconds(step_name, step)
        status = "failed" if step.had_failure else "success"
        artifacts = step.artifacts
        if step_name == "pipeline":
            artifacts = _summarize_pipeline_artifacts(artifacts)
        steps_out.append(
            {
                "name": step_name,
                "runtime_seconds": round(runtime_seconds, 4),
                "status": status,
                "artifacts": artifacts,
            }
        )

    out: dict[str, Any] = {
        "summary": {
            "runtime_seconds": round(overall_runtime_seconds, 4),
            "page_count": page_count,
            "extraction_success": extraction_success,
            "fallback_used": fallback_used,
            "node_count": node_count,
            "edge_count": edge_count,
        },
        "steps": steps_out,
    }
    return out
