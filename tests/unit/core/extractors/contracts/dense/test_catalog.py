"""Unit tests for the dense extraction node catalog builder."""

import json
from typing import Dict, List
from unittest.mock import patch

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

import docling_graph.core.extractors.contracts.dense.catalog as catalog_mod
from docling_graph.core.extractors.contracts.dense.catalog import (
    NodeCatalog,
    NodeSpec,
    _field_aliases,
    _get_id_fields,
    _is_component,
    _is_entity,
    _schema_hints_for_model,
    _unwrap_model_from_annotation,
    bottom_up_path_order,
    build_node_catalog,
    build_projected_fill_schema,
    build_skeleton_semantic_guide,
    get_model_for_path,
    skeleton_output_schema,
)


def test_unwrap_model_from_annotation_returns_none_for_generic_without_model() -> None:
    assert _unwrap_model_from_annotation(Dict[str, int]) is None
    assert _unwrap_model_from_annotation(list[int]) is None


class _NonDictConfigModel:
    """Duck-typed stand-in whose model_config is not a dict.

    Real pydantic BaseModel subclasses always normalize model_config to a
    plain dict, so the "not isinstance(cfg, dict)" defensive branches in
    catalog.py can only be reached with a non-BaseModel object like this.
    """

    model_config = "not-a-dict"


def test_get_id_fields_returns_empty_when_config_not_dict() -> None:
    assert _get_id_fields(_NonDictConfigModel) == []


def test_get_id_fields_filters_non_string_entries() -> None:
    class M(BaseModel):
        model_config = ConfigDict(graph_id_fields=["a", 1, "b", None])
        a: str = "x"
        b: str = "y"

    assert _get_id_fields(M) == ["a", "b"]


def test_is_entity_defaults_true_when_config_not_dict() -> None:
    assert _is_entity(_NonDictConfigModel) is True


def test_is_entity_false_when_is_entity_config_false() -> None:
    class Comp(BaseModel):
        model_config = ConfigDict(is_entity=False)
        text: str = "x"

    assert _is_entity(Comp) is False


def test_is_entity_true_when_id_fields_present() -> None:
    class Entity(BaseModel):
        model_config = ConfigDict(graph_id_fields=["name"])
        name: str = "x"

    assert _is_entity(Entity) is True


def test_is_component_false_when_config_not_dict() -> None:
    assert _is_component(_NonDictConfigModel) is False


def test_is_component_true_when_is_entity_false() -> None:
    class Comp(BaseModel):
        model_config = ConfigDict(is_entity=False)
        text: str = "x"

    assert _is_component(Comp) is True


def test_is_component_false_for_plain_entity() -> None:
    class Entity(BaseModel):
        model_config = ConfigDict(graph_id_fields=["name"])
        name: str = "x"

    assert _is_component(Entity) is False


def test_field_aliases_collects_alias_and_choices_and_string_validation_alias() -> None:
    class M(BaseModel):
        a: str = Field(alias="alpha", default="x")
        b: str = Field(validation_alias=AliasChoices("beta", "bee"), default="y")
        c: str = Field(validation_alias="charlie", default="z")
        d: str = "w"

    fields = M.model_fields
    assert _field_aliases("a", fields["a"]) == ["alpha"]
    assert _field_aliases("b", fields["b"]) == ["bee", "beta"]
    assert _field_aliases("c", fields["c"]) == ["charlie"]
    assert _field_aliases("d", fields["d"]) == []


def test_schema_hints_for_model_handles_exception() -> None:
    class Weird:
        __doc__ = None

        @classmethod
        def model_json_schema(cls) -> dict:
            raise ValueError("boom")

    desc, hint = _schema_hints_for_model(Weird, [])
    assert desc == ""
    assert hint == ""


def test_schema_hints_for_model_skips_missing_field_and_uses_examples() -> None:
    class M(BaseModel):
        """Some description."""

        invoice_number: str = Field(
            description="inv",
            json_schema_extra={"examples": ["INV-1", "INV-2", "INV-3", "INV-4"]},
        )
        other: str = "x"

    desc, hint = _schema_hints_for_model(M, ["invoice_number", "missing_field"])
    assert desc == "Some description."
    assert hint == " e.g. invoice_number: 'INV-1', 'INV-2', 'INV-3'"


def test_node_spec_to_dict_roundtrip() -> None:
    spec = NodeSpec(
        path="items[]",
        node_type="Item",
        id_fields=["name"],
        kind="entity",
        parent_path="",
        field_name="items",
        is_list=True,
        description="desc",
        example_hint="hint",
    )
    assert spec.to_dict() == {
        "path": "items[]",
        "node_type": "Item",
        "id_fields": ["name"],
        "kind": "entity",
        "parent_path": "",
        "field_name": "items",
        "is_list": True,
        "description": "desc",
        "example_hint": "hint",
    }


def test_node_catalog_paths() -> None:
    catalog = NodeCatalog(
        nodes=[
            NodeSpec(path="", node_type="Root"),
            NodeSpec(path="items[]", node_type="Item"),
        ]
    )
    assert catalog.paths() == ["", "items[]"]


class _Comp(BaseModel):
    model_config = ConfigDict(is_entity=False)
    text: str = "x"


class _SubEntity(BaseModel):
    model_config = ConfigDict(graph_id_fields=["sub_id"])
    sub_id: str = "s1"


class _Entity(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str = "n1"
    sub: _SubEntity  # required (no default) -> exercises "required" filtering


class _Root(BaseModel):
    labeled_comp: _Comp | None = Field(default=None, json_schema_extra={"edge_label": "HAS_COMP"})
    unlabeled_comp: _Comp | None = None
    unlabeled_comp_list: List[_Comp] = Field(default_factory=list)
    ents: List[_Entity] = Field(default_factory=list, alias="entities")


def test_build_node_catalog_covers_component_and_nested_entity_branches() -> None:
    catalog = build_node_catalog(_Root)
    by_path = {n.path: n for n in catalog.nodes}

    assert "" in by_path
    assert by_path[""].kind == "entity"

    # Single component field with an edge_label produces a node (kind=component).
    assert "labeled_comp" in by_path
    assert by_path["labeled_comp"].kind == "component"
    assert by_path["labeled_comp"].is_list is False

    # Components without an edge_label are skipped (no node created), whether
    # scalar or list-typed, but the walk still recurses into their fields.
    assert "unlabeled_comp" not in by_path
    assert "unlabeled_comp_list[]" not in by_path

    # List-of-entity produces a "[]" node, and nested entity fields within it
    # get their parent_path set to the enclosing list node's path.
    assert "ents[]" in by_path
    assert by_path["ents[]"].is_list is True
    assert "ents[].sub" in by_path
    assert by_path["ents[].sub"].parent_path == "ents[]"

    # Aliased fields discovered during the walk are recorded in field_aliases.
    assert catalog.field_aliases.get("entities") == "ents"


def test_get_model_for_path_resolves_nested_list_path() -> None:
    model = get_model_for_path(_Root, "ents[].sub")
    assert model is _SubEntity


def test_get_model_for_path_returns_none_for_unknown_path() -> None:
    assert get_model_for_path(_Root, "does.not.exist") is None


def test_build_projected_fill_schema_returns_empty_object_when_model_missing() -> None:
    catalog = build_node_catalog(_Root)
    fake_spec = NodeSpec(path="nonexistent", node_type="X")
    assert build_projected_fill_schema(_Root, fake_spec, catalog) == "{}"


def test_build_projected_fill_schema_dumps_schema_when_no_properties_dict() -> None:
    catalog = build_node_catalog(_Root)
    spec = next(n for n in catalog.nodes if n.path == "")

    class FakeModel:
        @classmethod
        def model_json_schema(cls) -> dict:
            return {"type": "string"}

    with patch.object(catalog_mod, "get_model_for_path", return_value=FakeModel):
        result = build_projected_fill_schema(_Root, spec, catalog)

    assert result == '{\n  "type": "string"\n}'


def test_build_projected_fill_schema_excludes_child_path_fields() -> None:
    catalog = build_node_catalog(_Root)
    spec = next(n for n in catalog.nodes if n.path == "")
    schema = json.loads(build_projected_fill_schema(_Root, spec, catalog))
    props = schema["properties"]
    assert "ents" not in props
    assert "labeled_comp" not in props
    assert "unlabeled_comp" in props


def test_build_projected_fill_schema_filters_required_list_for_child_field() -> None:
    """_Entity.sub is both required (no default) and a nested child path, so
    it must be dropped from both properties and the required list."""
    catalog = build_node_catalog(_Root)
    spec = next(n for n in catalog.nodes if n.path == "ents[]")
    schema = json.loads(build_projected_fill_schema(_Root, spec, catalog))
    assert "sub" not in schema["properties"]
    assert "sub" not in schema.get("required", [])
    assert "name" in schema["properties"]


def test_build_skeleton_semantic_guide_lists_paths_and_ids() -> None:
    catalog = build_node_catalog(_Root)
    guide = build_skeleton_semantic_guide(catalog)
    assert '""' in guide
    assert "ents[] (_Entity) ids=[name]" in guide
    assert "none (use ids={})" in guide


def test_skeleton_output_schema_shape() -> None:
    schema = skeleton_output_schema(["", "ents[]"])
    assert schema["type"] == "object"
    assert schema["properties"]["nodes"]["items"]["properties"]["path"]["enum"] == [
        "",
        "ents[]",
    ]
    assert schema["required"] == ["nodes"]


def test_bottom_up_path_order_sorts_deepest_first() -> None:
    catalog = build_node_catalog(_Root)
    order = bottom_up_path_order(catalog)
    assert order.index("ents[].sub") < order.index("ents[]")
    assert order.index("ents[]") < order.index("")
