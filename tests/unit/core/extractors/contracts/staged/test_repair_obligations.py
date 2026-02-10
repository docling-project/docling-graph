"""
Unit tests for nested path obligations and repair prompt checklist (Phase 3).
"""

from docling_graph.core.extractors.contracts.staged.prompts import (
    QualityReport,
    QualityIssue,
    TemplateGraphMetadata,
    get_repair_prompt,
)


def test_nested_path_obligations_dedupes_and_returns_identity_fields():
    """Quality report with issues at (line_items, 0, item) and (line_items, 1, item) yields one obligation line_items[].item."""
    report = QualityReport(
        issues=[
            QualityIssue(("line_items", 0, "item"), reason="empty", severity="warning"),
            QualityIssue(("line_items", 1, "item"), reason="empty", severity="warning"),
        ]
    )
    metadata = TemplateGraphMetadata(
        root_identity_fields=[],
        root_edge_fields={},
        root_entity_identity_fields={},
        nested_entity_identity_fields={},
        nested_edge_targets={"line_items": [("item", ["item_code"])]},
    )
    obligations = report.nested_path_obligations(metadata)
    assert len(obligations) == 1
    pattern, ids = obligations[0]
    assert pattern == "line_items[].item"
    assert ids == ["item_code"]


def test_nested_path_obligations_empty_metadata_returns_empty():
    report = QualityReport(
        issues=[QualityIssue(("line_items", 0, "item"), reason="empty", severity="warning")]
    )
    metadata = TemplateGraphMetadata([], {}, {}, {}, {})
    assert report.nested_path_obligations(metadata) == []


def test_repair_prompt_contains_checklist_when_nested_obligations():
    prompt = get_repair_prompt(
        markdown_content="doc",
        schema_json="{}",
        failed_fields=["line_items"],
        nested_obligations=[("line_items[].item", ["item_code"])],
    )
    system = prompt["system"]
    assert "You must populate the following nested structures" in system
    assert "line_items[].item" in system
    assert "item_code" in system


def test_repair_prompt_no_checklist_when_nested_obligations_empty():
    prompt = get_repair_prompt(
        markdown_content="doc",
        schema_json="{}",
        failed_fields=["line_items"],
        nested_obligations=[],
    )
    assert "You must populate the following nested structures" not in prompt["system"]
