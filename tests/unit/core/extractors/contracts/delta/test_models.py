"""Tests for delta contract Pydantic models (DeltaGraph, DeltaNode, etc.)."""

import pytest

from docling_graph.core.extractors.contracts.delta.models import (
    DeltaGraph,
    DeltaNode,
    DeltaParentRef,
    DeltaRelationship,
)


def test_delta_node_accepts_string_parent() -> None:
    """LLM often returns parent as string (e.g. \"BillingDocument\"); must coerce to DeltaParentRef."""
    node = DeltaNode.model_validate({
        "path": "seller",
        "ids": {"name": "Acme"},
        "parent": "BillingDocument",  # string instead of {"path": "", "ids": {}}
        "properties": {},
    })
    assert node.parent is not None
    assert node.parent.path == "BillingDocument"
    assert node.parent.ids == {}


def test_delta_node_accepts_numeric_ids() -> None:
    """LLM often returns line_number as integer; must coerce to string."""
    node = DeltaNode.model_validate({
        "path": "line_items[]",
        "ids": {"line_number": 1},  # int instead of str
        "parent": {"path": "", "ids": {}},
        "properties": {"name": "Item"},
    })
    assert node.ids == {"line_number": "1"}


def test_delta_graph_validates_real_failed_batch_shape() -> None:
    """Payload that previously failed validation (string parents, int ids) now validates."""
    raw = {
        "nodes": [
            {"path": "", "node_type": "BillingDocument", "ids": {"document_number": "3139"}, "parent": None, "properties": {"document_type": "Invoice"}},
            {"path": "seller", "node_type": "Party", "ids": {"name": "Robert Schneider AG"}, "parent": "BillingDocument", "properties": {}},
            {"path": "line_items.1", "node_type": "LineItem", "ids": {"line_number": 1}, "parent": "BillingDocument", "properties": {}},
        ],
        "relationships": [],
    }
    graph = DeltaGraph.model_validate(raw)
    assert len(graph.nodes) == 3
    assert graph.nodes[1].parent is not None
    assert graph.nodes[1].parent.path == "BillingDocument"
    assert graph.nodes[2].ids == {"line_number": "1"}


def test_delta_parent_ref_ids_coerced_to_str() -> None:
    """DeltaParentRef ids values are coerced to string."""
    ref = DeltaParentRef.model_validate({"path": "line_items[]", "ids": {"line_number": 2}})
    assert ref.ids == {"line_number": "2"}


def test_delta_relationship_source_target_ids_coerced() -> None:
    """Relationship source_ids/target_ids numeric values coerced to string."""
    rel = DeltaRelationship.model_validate({
        "edge_label": "has",
        "source_path": "",
        "source_ids": {"document_number": "1"},
        "target_path": "line_items[]",
        "target_ids": {"line_number": 1},
    })
    assert rel.target_ids == {"line_number": "1"}
