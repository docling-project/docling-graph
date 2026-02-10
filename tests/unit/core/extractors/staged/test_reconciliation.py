from docling_graph.core.extractors.contracts.staged.reconciliation import (
    ReconciliationPolicy,
    merge_pass_output,
)


def test_merge_pass_output_prefers_non_placeholder_scalar():
    merged = {"title": "unknown"}
    pass_output = {"title": "Real Title"}
    result = merge_pass_output(
        merged,
        pass_output,
        context_tag="test",
        policy=ReconciliationPolicy(prefer_non_placeholder=True),
    )
    assert result["title"] == "Real Title"


def test_merge_pass_output_repair_override_roots():
    merged = {"line_items": [{"description": "old"}]}
    pass_output = {"line_items": [{"description": "new"}]}
    result = merge_pass_output(
        merged,
        pass_output,
        context_tag="repair",
        policy=ReconciliationPolicy(repair_override_roots={"line_items"}),
    )
    assert result["line_items"] == [{"description": "new"}]

