from docling_graph.core.extractors.contracts.delta.prompts import get_delta_batch_prompt


def test_prompt_mentions_cross_batch_identifier_stability() -> None:
    prompt = get_delta_batch_prompt(
        batch_markdown="--- CHUNK 1 ---\ntext",
        schema_semantic_guide="guide",
        path_catalog_block="catalog",
        batch_index=0,
        total_batches=2,
    )
    assert "across the entire document" in prompt["system"]
    assert "Do not use class names or slash-separated paths" in prompt["system"]
    assert "repeat the same identity values in flat properties" in prompt["system"]
    assert "CHF 3360.00' -> 3360.00" in prompt["system"]
    assert "ONLY put identity fields in ids" in prompt["system"]
    assert "Do not create synthetic pseudo-path nodes" in prompt["system"]
    assert "Example good root scalar placement" in prompt["user"]
    assert "<root_field>" in prompt["user"]
    assert "total_amount" not in prompt["user"]
