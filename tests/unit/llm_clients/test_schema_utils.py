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


def test_normalize_schema_dereferences_recursive_root():
    """A recursive root model (an inverse reference edge back to the root)
    makes pydantic emit a bare top-level $ref into $defs; providers need a
    typed top-level object, so the root def is inlined and $defs kept for
    the inner self-references."""
    schema = {
        "$defs": {
            "Paper": {
                "type": "object",
                "title": "Paper",
                "properties": {
                    "title": {"type": "string"},
                    "author": {"$ref": "#/$defs/Author"},
                },
                "required": ["title"],
            },
            "Author": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "authored": {"$ref": "#/$defs/Paper"},  # back to the root
                },
                "required": ["name"],
            },
        },
        "$ref": "#/$defs/Paper",
    }
    normalized = normalize_schema_for_response_format(schema)
    assert normalized["strict"] is True
    top = normalized["schema"]
    assert top["type"] == "object"
    assert "title" in top["properties"]
    assert "$ref" not in top  # dereferenced
    # Inner self-references still resolve through the retained $defs.
    assert top["$defs"]["Author"]["properties"]["authored"]["$ref"] == "#/$defs/Paper"
    assert "Paper" in top["$defs"]
