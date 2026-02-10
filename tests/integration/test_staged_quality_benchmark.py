from __future__ import annotations

import json
from pathlib import Path

import pytest

from docling_graph.core.extractors.contracts.staged import summarize_comparison


@pytest.mark.integration
def test_staged_quality_benchmark_harness(tmp_path: Path):
    standard_dir = tmp_path / "standard"
    advanced_dir = tmp_path / "advanced"
    standard_dir.mkdir()
    advanced_dir.mkdir()

    standard_metadata = {
        "config": {"extraction": {"model": "ibm-granite/granite-4.0-1b", "provider": "vllm"}},
        "results": {"nodes": 9, "edges": 8, "extracted_models": 1},
    }
    advanced_metadata = {
        "config": {"extraction": {"model": "mistral/mistral-medium-latest", "provider": "mistral"}},
        "results": {"nodes": 29, "edges": 41, "extracted_models": 1},
    }
    (standard_dir / "metadata.json").write_text(json.dumps(standard_metadata), encoding="utf-8")
    (advanced_dir / "metadata.json").write_text(json.dumps(advanced_metadata), encoding="utf-8")

    summary = summarize_comparison(standard_dir, advanced_dir)

    assert summary["standard"].nodes == 9
    assert summary["advanced"].nodes == 29
    assert summary["delta"].nodes_delta == -20
    assert round(summary["delta"].nodes_ratio, 3) == round(9 / 29, 3)
