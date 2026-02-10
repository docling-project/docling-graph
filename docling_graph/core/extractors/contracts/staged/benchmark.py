"""
Quality benchmark helpers for staged extraction comparisons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunMetrics:
    nodes: int
    edges: int
    extracted_models: int
    model: str
    provider: str


@dataclass(frozen=True)
class QualityDelta:
    nodes_delta: int
    edges_delta: int
    nodes_ratio: float
    edges_ratio: float


def load_run_metrics(run_dir: Path) -> RunMetrics:
    metadata_path = run_dir / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    extraction = payload.get("config", {}).get("extraction", {})
    results = payload.get("results", {})
    return RunMetrics(
        nodes=int(results.get("nodes", 0)),
        edges=int(results.get("edges", 0)),
        extracted_models=int(results.get("extracted_models", 0)),
        model=str(extraction.get("model", "")),
        provider=str(extraction.get("provider", "")),
    )


def compare_metrics(standard_run: RunMetrics, advanced_run: RunMetrics) -> QualityDelta:
    nodes_ratio = (
        float(standard_run.nodes) / float(advanced_run.nodes) if advanced_run.nodes > 0 else 0.0
    )
    edges_ratio = (
        float(standard_run.edges) / float(advanced_run.edges) if advanced_run.edges > 0 else 0.0
    )
    return QualityDelta(
        nodes_delta=standard_run.nodes - advanced_run.nodes,
        edges_delta=standard_run.edges - advanced_run.edges,
        nodes_ratio=nodes_ratio,
        edges_ratio=edges_ratio,
    )


def summarize_comparison(standard_dir: Path, advanced_dir: Path) -> dict[str, Any]:
    standard_metrics = load_run_metrics(standard_dir)
    advanced_metrics = load_run_metrics(advanced_dir)
    delta = compare_metrics(standard_metrics, advanced_metrics)
    return {
        "standard": standard_metrics,
        "advanced": advanced_metrics,
        "delta": delta,
    }

