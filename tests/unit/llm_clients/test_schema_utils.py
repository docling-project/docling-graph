from docling_graph.llm_clients.schema_utils import (
    build_compact_semantic_guide,
    normalize_schema_for_response_format,
)


def test_normalize_schema_for_object_top_level():
    schema = {"title": "X", "type": "object", "properties": {"name": {"type": "string"}}}
    out = normalize_schema_for_response_format(schema, top_level="object", name="test_schema")
    assert out["name"] == "test_schema"
    assert out["strict"] is True
    assert out["schema"]["type"] == "object"
    assert "title" not in out["schema"]


def test_normalize_schema_wraps_array_top_level():
    schema = {"type": "object", "properties": {"id": {"type": "string"}}}
    out = normalize_schema_for_response_format(schema, top_level="array")
    assert out["schema"]["type"] == "array"
    assert out["schema"]["items"]["type"] == "object"


def test_build_compact_semantic_guide_includes_required_description_and_enum():
    schema = {
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {
                "type": "string",
                "description": "Current lifecycle state",
                "enum": ["draft", "published"],
                "examples": ["draft"],
            }
        },
    }
    guide = build_compact_semantic_guide(schema)
    assert "status" in guide
    assert "required" in guide
    assert "Current lifecycle state" in guide
    assert "draft" in guide
