"""Unit tests for the benchmark aggregation example script (Example 17).

Covers the aligned-rung surfacing (the eval's fair rung under identity drift
was previously dropped by the rollup, misreporting a 0.008 dense-vs-direct edge
gap as 0.116), the per-rep skeleton-size column, and skipped-cell rendering.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "examples"
    / "scripts"
    / "17_benchmark_aggregate.py"
)


@pytest.fixture(scope="module")
def aggmod():
    spec = importlib.util.spec_from_file_location("benchmark_aggregate", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval_payload() -> dict:
    per_class = {
        "strict": {"tp": 1, "fp": 1, "fn": 1, "p": 0.5, "r": 0.5, "f1": 0.5},
        "relaxed": {"tp": 2, "fp": 0, "fn": 0, "p": 1.0, "r": 1.0, "f1": 1.0},
    }
    return {
        "nodes": {"A": per_class},
        "edges": {"HAS": per_class},
        "micro": {
            "nodes": {
                "strict": {"f1": 0.5},
                "relaxed": {"f1": 1.0},
                "aligned": {"tp": 3, "fp": 0, "fn": 0, "p": 1.0, "r": 1.0, "f1": 0.9},
            },
            "edges": {
                "strict": {"f1": 0.5},
                "relaxed": {"f1": 1.0},
                "aligned": {"tp": 3, "fp": 0, "fn": 0, "p": 1.0, "r": 1.0, "f1": 0.8},
            },
        },
    }


def _write_run(tmp_path: Path, with_aligned: bool = True) -> Path:
    run_dir = tmp_path / "doc" / "dense" / "markdown" / "rep1" / "run_x"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"processing_time_seconds": 10, "results": {"nodes": 3, "edges": 2}}),
        encoding="utf-8",
    )
    payload = _eval_payload()
    if not with_aligned:
        payload["micro"]["nodes"].pop("aligned")
        payload["micro"]["edges"].pop("aligned")
    (run_dir / "eval.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def test_run_record_surfaces_aligned_micro(aggmod, tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    row = aggmod.run_record(run_dir, "doc", "dense", "markdown", 1)
    assert row["node_aligned"]["f1"] == 0.9
    assert row["edge_aligned"]["f1"] == 0.8
    # strict/relaxed still re-summed from per-class sections
    assert row["edge_strict"]["tp"] == 1


def test_run_record_tolerates_missing_aligned(aggmod, tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, with_aligned=False)
    row = aggmod.run_record(run_dir, "doc", "dense", "markdown", 1)
    assert "node_aligned" not in row
    assert "edge_aligned" not in row


def test_cell_summary_aligned_mean_and_skeleton_sizes(aggmod) -> None:
    runs = [
        {
            "node_strict": {"f1": 0.4},
            "node_relaxed": {"f1": 0.5},
            "node_aligned": {"f1": 0.6},
            "edge_strict": {"f1": 0.2},
            "edge_relaxed": {"f1": 0.3},
            "edge_aligned": {"f1": 0.5},
            "skeleton_nodes": 657,
        },
        {
            "node_strict": {"f1": 0.2},
            "node_relaxed": {"f1": 0.3},
            "edge_strict": {"f1": 0.2},
            "edge_relaxed": {"f1": 0.3},
            "skeleton_nodes": 299,
        },
    ]
    summary = aggmod._cell_summary(runs)
    assert summary["node_f1_aligned"] == 0.6  # mean over runs that HAVE the rung
    assert summary["edge_f1_aligned"] == 0.5
    assert summary["skeleton_nodes"] == [657, 299]


def test_rungs_cell_formatting(aggmod) -> None:
    assert (
        aggmod._rungs(
            {"edge_f1_strict": 0.258, "edge_f1_relaxed": 0.318, "edge_f1_aligned": 0.53}, "edge"
        )
        == "0.258 (0.318 / 0.53)"
    )
    assert (
        aggmod._rungs(
            {"edge_f1_strict": 0.5, "edge_f1_relaxed": None, "edge_f1_aligned": None}, "edge"
        )
        == "0.5 (— / —)"
    )
    assert aggmod._rungs({"edge_f1_strict": None}, "edge") == "—"


def test_failed_record_passes_status_through(aggmod, tmp_path: Path) -> None:
    cell = tmp_path / "doc" / "direct" / "markdown" / "rep1"
    cell.mkdir(parents=True)
    (cell / "cell_status.json").write_text(
        json.dumps({"status": "skipped", "reason": "skipped_by_config"}), encoding="utf-8"
    )
    record = aggmod._failed_record(cell, "doc", "direct", "markdown", 1)
    assert record["status"] == "skipped"
    assert record["failure_reason"] == "skipped_by_config"


def test_render_shows_failed_and_skipped_labels(aggmod) -> None:
    records = [
        {
            "doc": "report",
            "contract": "direct",
            "format": "markdown",
            "rep": 1,
            "run": None,
            "status": "failed",
            "failure_reason": "extract_failed",
        },
        {
            "doc": "report",
            "contract": "direct",
            "format": "doclang",
            "rep": 1,
            "run": None,
            "status": "skipped",
            "failure_reason": "skipped_by_config",
        },
    ]
    report = aggmod.render_markdown(records)
    assert "FAILED (extract_failed)" in report
    assert "SKIPPED (skipped_by_config)" in report
