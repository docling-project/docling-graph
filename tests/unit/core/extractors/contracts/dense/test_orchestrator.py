"""
Unit tests for dense extraction orchestrator (Phase 1 and Phase 2, including parallel).
"""

from typing import Any

import pytest

from docling_graph.core.extractors.contracts.dense.catalog import NodeCatalog, NodeSpec
from docling_graph.core.extractors.contracts.dense.orchestrator import (
    DenseOrchestrator,
    DenseOrchestratorConfig,
    chunk_batches_by_token_limit,
    normalize_skeleton_node,
    prune_barren_branches,
)
from tests.fixtures.sample_templates.test_template import SampleInvoice


def test_normalize_skeleton_node_derives_parent_from_ancestry():
    """When raw has ancestry with two elements, normalized node has parent equal to last element."""
    allowed = {"", "studies[]"}
    raw = {
        "path": "studies[]",
        "ids": {"study_id": "S1"},
        "parent": {"path": "studies[]", "ids": {"study_id": "S0"}},
        "ancestry": [
            {"path": "", "ids": {}},
            {"path": "studies[]", "ids": {"study_id": "S0"}},
        ],
    }
    norm = normalize_skeleton_node(raw, allowed)
    assert norm is not None
    assert norm["path"] == "studies[]"
    assert norm["ids"] == {"study_id": "S1"}
    assert norm["parent"] == {"path": "studies[]", "ids": {"study_id": "S0"}}
    assert "ancestry" not in norm


def test_normalize_skeleton_node_strips_extra_keys():
    """Normalized node contains only path, ids, parent (and optional _source_batch_index)."""
    allowed = {""}
    raw = {
        "path": "",
        "ids": {},
        "parent": None,
        "reported_findings": ["should be stripped"],
    }
    norm = normalize_skeleton_node(raw, allowed)
    assert norm is not None
    assert set(norm.keys()) == {"path", "ids", "parent"}
    assert "reported_findings" not in norm


def test_normalize_skeleton_node_adds_source_batch_index_when_provided():
    """When source_batch_index is passed, normalized node includes _source_batch_index."""
    allowed = {""}
    raw = {"path": "", "ids": {}, "parent": None}
    norm = normalize_skeleton_node(raw, allowed, source_batch_index=2)
    assert norm is not None
    assert norm.get("_source_batch_index") == 2


def test_chunk_batches_produces_multiple_batches_for_parallel():
    """With small max_batch_tokens, multiple chunks yield multiple batches."""
    chunks = ["chunk one", "chunk two", "chunk three", "chunk four", "chunk five"]
    token_counts = [40, 40, 40, 40, 40]
    batches = chunk_batches_by_token_limit(
        chunks, token_counts, max_batch_tokens=50
    )
    assert len(batches) >= 2


def test_orchestrator_parallel_workers_returns_non_null_root():
    """With parallel_workers=2 and multiple skeleton batches, orchestrator runs and returns a non-None root."""
    # Build chunks so we get multiple skeleton batches (small token limit)
    chunks = ["Document chunk A. " * 20, "Document chunk B. " * 20, "Document chunk C. " * 20]
    token_counts = [50, 50, 50]
    full_markdown = "\n\n".join(chunks)

    skeleton_responses = [
        {"nodes": [{"path": "", "ids": {"invoice_number": "INV-1"}, "parent": None}]}
        for _ in range(5)
    ]
    fill_responses = [
        {"items": [{"invoice_number": "INV-1", "date": "2024-01-01", "total_amount": 100.0, "vendor_name": "Acme", "items": []}]}
    ]
    skeleton_idx = [0]
    fill_idx = [0]

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        if "dense_skeleton" in context:
            i = skeleton_idx[0]
            skeleton_idx[0] += 1
            return skeleton_responses[i % len(skeleton_responses)]
        if "dense_fill" in context:
            i = fill_idx[0]
            fill_idx[0] += 1
            return fill_responses[i % len(fill_responses)]
        return None

    config = DenseOrchestratorConfig(
        parallel_workers=2,
        skeleton_batch_tokens=50,
        fill_nodes_cap=5,
        quality_min_instances=1,
    )
    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleInvoice,
        config=config,
    )
    root = orch.run(
        chunks=chunks,
        chunk_metadata=[{"token_count": tc} for tc in token_counts],
        full_markdown=full_markdown,
        context="test",
    )
    assert root is not None
    assert "invoice_number" in root or "vendor_name" in root


def test_prune_barren_branches_removes_barren_branch_keeps_non_barren():
    """prune_barren_branches removes branch nodes that are childless and barren; leaves non-barren and leaf paths unchanged."""
    catalog = NodeCatalog(
        nodes=[
            NodeSpec(path="", node_type="Root", id_fields=[], parent_path="", field_name="", is_list=False),
            NodeSpec(path="studies[]", node_type="Study", id_fields=["study_id"], parent_path="", field_name="studies", is_list=True),
            NodeSpec(path="studies[].experiments[]", node_type="Experiment", id_fields=["exp_id"], parent_path="studies[]", field_name="experiments", is_list=True),
        ]
    )
    root = {
        "studies": [
            {"study_id": "S1", "objective": None, "experiments": []},
            {"study_id": "S2", "objective": "Real study", "experiments": []},
        ]
    }
    out = prune_barren_branches(root, catalog)
    assert len(out["studies"]) == 1
    assert out["studies"][0]["study_id"] == "S2"
    assert out["studies"][0]["objective"] == "Real study"
