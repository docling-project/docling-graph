"""
Unit tests for staged contract prompt/schema helpers.
"""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from docling_graph.core.extractors.contracts import staged


def _schema() -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": {
                "document_number": {"type": "string"},
                "issue_date": {"type": "string"},
                "seller": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
                "line_items": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"description": {"type": "string"}}},
                },
            },
            "required": ["document_number", "seller"],
        },
        indent=2,
    )


def test_get_root_field_groups_splits_object_fields():
    groups = staged.get_root_field_groups(_schema(), max_fields_per_group=2)
    assert ["seller"] in groups
    assert ["line_items"] in groups
    assert any("document_number" in g for g in groups)


class Seller(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str


class InvoiceTemplate(BaseModel):
    model_config = ConfigDict(graph_id_fields=["document_number"])
    document_number: str
    issue_date: str | None = None
    seller: Seller = Field(..., json_schema_extra={"edge_label": "ISSUED_BY"})
    line_items: list[dict[str, Any]] = Field(default_factory=list)


def test_get_template_graph_metadata_uses_graph_fields_and_edges():
    schema_json = json.dumps(InvoiceTemplate.model_json_schema(), indent=2)
    metadata = staged.get_template_graph_metadata(InvoiceTemplate, schema_json)
    assert "document_number" in metadata.root_identity_fields
    assert metadata.root_edge_fields.get("seller") == "ISSUED_BY"
    assert metadata.root_entity_identity_fields.get("seller") == ["name"]


def test_plan_extraction_passes_disjoint_groups():
    schema_json = json.dumps(InvoiceTemplate.model_json_schema(), indent=2)
    plan = staged.plan_extraction_passes(schema_json, max_fields_per_group=2, max_skeleton_fields=2)
    grouped_fields = [field for group in plan.groups for field in group]
    assert len(grouped_fields) == len(set(grouped_fields))
    for field in plan.skeleton_fields:
        assert field not in grouped_fields


def test_build_root_subschema_keeps_selected_properties():
    subschema = staged.build_root_subschema(_schema(), ["document_number", "seller"])
    parsed = json.loads(subschema)
    assert set(parsed["properties"].keys()) == {"document_number", "seller"}
    assert parsed["required"] == ["document_number", "seller"]


def test_detect_quality_issues_flags_placeholders_and_missing_required():
    candidate = {
        "document_number": "INV-SAMPLE-001",
        "issue_date": "2026-01-01",
        "seller": {},
    }
    issues = staged.detect_quality_issues(candidate, _schema())
    assert "document_number" in issues
    assert "seller" in issues


def test_assess_quality_reports_nested_issue_paths():
    candidate = {
        "document_number": "INV-2026-01",
        "seller": {"name": "unknown"},
        "line_items": [{"description": "sample value"}],
    }
    report = staged.assess_quality(candidate, _schema(), max_depth=4)
    paths = [issue.field_path for issue in report.issues]
    assert ("seller", "name") in paths
    assert ("line_items", 0, "description") in paths

