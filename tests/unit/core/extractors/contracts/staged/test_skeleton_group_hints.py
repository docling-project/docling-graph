"""
Unit tests for skeleton and group prompt nested edge hints (Phase 4).
"""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.extractors.contracts import staged


def _edge(label: str, **kwargs: Any) -> Any:
    return Field(json_schema_extra={"edge_label": label}, **kwargs)


class Item(BaseModel):
    model_config = ConfigDict(graph_id_fields=["item_code"])
    item_code: str
    name: str | None = None


class LineItem(BaseModel):
    line_number: str
    item_code: str | None = None
    item: Item | None = _edge("REFERENCES_ITEM", default=None)


class BillingLikeTemplate(BaseModel):
    document_number: str
    line_items: list[LineItem] = Field(default_factory=list)


def test_skeleton_prompt_includes_nested_hint_when_line_items_has_item_target():
    """For template with line_items and LineItem.item -> Item with graph_id_fields=['item_code'], skeleton prompt includes line_items, item, item_code."""
    schema_json = json.dumps(BillingLikeTemplate.model_json_schema(), indent=2)
    metadata = staged.get_template_graph_metadata(BillingLikeTemplate, schema_json)
    assert "line_items" in metadata.nested_edge_targets
    assert metadata.nested_edge_targets["line_items"] == [("item", ["item_code"])]

    prompt = staged.get_skeleton_prompt(
        markdown_content="doc",
        schema_json=schema_json,
        anchor_fields=["document_number", "line_items"],
        nested_edge_hints=metadata.nested_edge_targets,
    )
    system = prompt["system"]
    assert "line_items" in system
    assert "item" in system
    assert "item_code" in system
    assert "For each element of" in system


def test_skeleton_prompt_backward_compat_without_hints():
    """Existing callers not passing nested_edge_hints get unchanged prompt (no nested instruction)."""
    schema_json = json.dumps(BillingLikeTemplate.model_json_schema(), indent=2)
    prompt = staged.get_skeleton_prompt(
        markdown_content="doc",
        schema_json=schema_json,
        anchor_fields=["document_number", "line_items"],
        # nested_edge_hints omitted
    )
    assert "For each element of" not in prompt["system"]


def test_group_prompt_includes_nested_hint_for_focus_field():
    schema_json = json.dumps(BillingLikeTemplate.model_json_schema(), indent=2)
    metadata = staged.get_template_graph_metadata(BillingLikeTemplate, schema_json)
    prompt = staged.get_group_prompt(
        markdown_content="doc",
        schema_json=schema_json,
        group_name="g1",
        focus_fields=["line_items"],
        nested_edge_hints=metadata.nested_edge_targets,
    )
    system = prompt["system"]
    assert "line_items" in system
    assert "item" in system
    assert "item_code" in system
