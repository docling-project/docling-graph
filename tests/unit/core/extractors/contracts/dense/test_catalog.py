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

    # Components NEVER become catalog paths — with or without an edge_label.
    # They are identity-less value objects Phase 1 cannot discover; they stay
    # in the parent's fill schema and are embedded inline by the converter.
    assert "labeled_comp" not in by_path
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
    # Components are filled inline with their parent, edge-labeled or not.
    assert "labeled_comp" in props
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


class _Plan(BaseModel):
    """A subscribable plan."""

    model_config = ConfigDict(graph_id_fields=["name"])
    name: str
    included: List[_Entity] = Field(
        default_factory=list,
        json_schema_extra={"edge_label": "INCLUDES", "graph_reference": True},
    )
    featured: _Entity | None = Field(default=None, json_schema_extra={"graph_reference": True})
    anonymous_refs: List[_Comp] = Field(
        default_factory=list, json_schema_extra={"graph_reference": True}
    )


class _PlanRoot(BaseModel):
    plans: List[_Plan] = Field(default_factory=list)
    ents: List[_Entity] = Field(default_factory=list)


def test_reference_fields_produce_no_catalog_paths() -> None:
    """Reference fields (and their subtrees) are absent from the catalog: they
    are filled by the parent, not skeleton-discovered."""
    catalog = build_node_catalog(_PlanRoot)
    paths = set(catalog.paths())
    assert "plans[]" in paths
    assert "ents[]" in paths
    assert "plans[].included[]" not in paths
    assert "plans[].included[].sub" not in paths
    assert "plans[].featured" not in paths


def test_reference_marker_on_identity_less_target_is_ignored() -> None:
    """graph_reference on a component target is meaningless -> normal handling
    (walked through, no path, still absent because components have no paths)."""
    catalog = build_node_catalog(_PlanRoot)
    assert "plans[].anonymous_refs[]" not in set(catalog.paths())
    # The projection must NOT rewrite the field either: it keeps its full schema.
    spec = next(n for n in catalog.nodes if n.path == "plans[]")
    schema = json.loads(build_projected_fill_schema(_PlanRoot, spec, catalog))
    anon = schema["properties"]["anonymous_refs"]
    assert "Identity-only reference" not in json.dumps(anon)


def test_projected_fill_schema_inlines_reference_fields_id_only() -> None:
    """The parent's fill schema re-includes reference fields projected to the
    target's graph_id_fields (list and scalar forms), self-contained (no $ref)."""
    catalog = build_node_catalog(_PlanRoot)
    spec = next(n for n in catalog.nodes if n.path == "plans[]")
    schema = json.loads(build_projected_fill_schema(_PlanRoot, spec, catalog))
    included = schema["properties"]["included"]
    assert included["type"] == "array"
    item = included["items"]
    assert set(item["properties"].keys()) == {"name"}
    assert item["required"] == ["name"]
    assert "Identity-only reference to a _Entity" in item["description"]
    assert "$ref" not in json.dumps(included)
    featured = schema["properties"]["featured"]
    assert set(featured["properties"].keys()) == {"name"}


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


def test_path_has_reference_fields_detects_reference_lists():
    """Paths whose fill schema carries id-only reference projections are flagged
    (they get per-parent fill batches to prevent membership dumping)."""
    from typing import List

    from pydantic import BaseModel, Field

    from docling_graph.core.extractors.contracts.dense.catalog import (
        build_node_catalog,
        path_has_reference_fields,
    )

    class Tag(BaseModel):
        name: str
        model_config = {"graph_id_fields": ["name"]}

    class Group(BaseModel):
        name: str
        tags: List[Tag] = Field(
            default_factory=list,
            json_schema_extra={"edge_label": "HAS_TAG", "graph_reference": True},
        )
        model_config = {"graph_id_fields": ["name"]}

    class RootDoc(BaseModel):
        title: str = ""
        groups: List[Group] = Field(default_factory=list)
        model_config = {"graph_id_fields": ["title"]}

    catalog = build_node_catalog(RootDoc)
    by_path = {s.path: s for s in catalog.nodes}
    assert path_has_reference_fields(RootDoc, by_path["groups[]"]) is True
    assert path_has_reference_fields(RootDoc, by_path[""]) is False
