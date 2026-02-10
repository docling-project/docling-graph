"""
Unit tests for materialize_nested_edges (Phase 2).
"""

import pytest

from docling_graph.core.extractors.contracts.staged.prompts import TemplateGraphMetadata
from docling_graph.core.extractors.contracts.staged.materialize_nested_edges import (
    materialize_nested_edges,
)


def _metadata(line_items_targets: list[tuple[str, list[str]]] | None = None) -> TemplateGraphMetadata:
    nested = {"line_items": line_items_targets or []}
    return TemplateGraphMetadata(
        root_identity_fields=[],
        root_edge_fields={},
        root_entity_identity_fields={},
        nested_entity_identity_fields={},
        nested_edge_targets=nested,
    )


def test_materializes_item_from_sibling_scalars():
    """BillingDocument-like: line_items with item=null and item_code/description -> item dict with item_code and optionally name."""
    data = {
        "line_items": [
            {"line_number": "1", "item_code": "SKU-001", "description": "Widget A", "item": None},
            {"line_number": "2", "item_code": "SKU-002", "item": None},
        ],
    }
    meta = _metadata([("item", ["item_code"])])
    materialize_nested_edges(data, meta)
    assert data["line_items"][0]["item"] == {"item_code": "SKU-001"}
    assert data["line_items"][1]["item"] == {"item_code": "SKU-002"}


def test_materializes_item_with_name_fallback_from_description():
    """When target has name and element has description, use description for name."""
    data = {
        "line_items": [
            {"line_number": "1", "item_code": "SKU-001", "description": "Widget A", "item": None},
        ],
    }
    meta = _metadata([("item", ["item_code", "name"])])
    materialize_nested_edges(data, meta)
    assert data["line_items"][0]["item"]["item_code"] == "SKU-001"
    assert data["line_items"][0]["item"]["name"] == "Widget A"


def test_no_overwrite_when_item_already_set():
    """Do not overwrite when nested object is already populated."""
    data = {
        "line_items": [
            {"line_number": "1", "item_code": "SKU-001", "item": {"item_code": "OTHER", "name": "Custom"}},
        ],
    }
    meta = _metadata([("item", ["item_code"])])
    materialize_nested_edges(data, meta)
    assert data["line_items"][0]["item"] == {"item_code": "OTHER", "name": "Custom"}


def test_missing_sibling_scalar_no_materialization():
    """When no sibling scalar for identity, do not materialize for that element."""
    data = {
        "line_items": [
            {"line_number": "1", "item": None},
        ],
    }
    meta = _metadata([("item", ["item_code"])])
    materialize_nested_edges(data, meta)
    assert data["line_items"][0]["item"] is None


def test_empty_metadata_no_op():
    """When nested_edge_targets is empty, data is unchanged."""
    data = {"line_items": [{"item": None, "item_code": "X"}]}
    meta = TemplateGraphMetadata([], {}, {}, {}, {})
    materialize_nested_edges(data, meta)
    assert data["line_items"][0]["item"] is None


def test_empty_dict_nested_treated_as_empty():
    """Nested field that is {} is treated as empty and can be materialized."""
    data = {
        "line_items": [
            {"line_number": "1", "item_code": "SKU-1", "item": {}},
        ],
    }
    meta = _metadata([("item", ["item_code"])])
    materialize_nested_edges(data, meta)
    assert data["line_items"][0]["item"] == {"item_code": "SKU-1"}
