"""
Unit tests for the node/edge catalog builder.
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from docling_graph.core.extractors.contracts.staged.catalog import (
    NodeCatalog,
    build_discovery_schema,
    build_node_catalog,
    flat_nodes_to_path_lists,
    get_allowed_paths_for_primary_paths,
    get_discovery_prompt,
    get_id_pass_shards,
    get_id_pass_shards_v2,
    get_identity_paths,
    get_model_for_path,
    merge_and_dedupe_flat_nodes,
    validate_id_pass_skeleton_response,
    write_catalog_artifact,
    write_id_pass_artifact,
)
from tests.fixtures.sample_templates.test_template import (
    SampleCompany,
    SampleInvoice,
    SamplePerson,
)


def test_build_node_catalog_simple_invoice():
    """SampleInvoice has no nested entities or edges; only root node."""
    catalog = build_node_catalog(SampleInvoice)
    assert len(catalog.nodes) >= 1
    root = next((n for n in catalog.nodes if n.path == ""), None)
    assert root is not None
    assert root.node_type == "SampleInvoice"
    assert root.id_fields == ["invoice_number"]
    assert root.kind == "entity"
    assert root.parent_path == ""
    assert root.field_name == ""
    assert root.is_list is False
    assert len(catalog.edges) == 0


def test_build_node_catalog_company_with_employees():
    """SampleCompany has root + employees[] list (entity at any depth)."""
    catalog = build_node_catalog(SampleCompany)
    paths = catalog.paths()
    assert "" in paths
    assert "employees[]" in paths
    root = next((n for n in catalog.nodes if n.path == ""), None)
    assert root is not None
    assert root.node_type == "SampleCompany"
    emp = next((n for n in catalog.nodes if n.path == "employees[]"), None)
    assert emp is not None
    assert emp.node_type == "SamplePerson"
    assert emp.id_fields == ["email"]
    assert emp.kind == "entity"
    assert emp.parent_path == ""
    assert emp.field_name == "employees"
    assert emp.is_list is True


def test_catalog_serialization_roundtrip():
    """to_dict is JSON-serializable and paths() matches nodes; includes description and example_hint."""
    catalog = build_node_catalog(SampleCompany)
    d = catalog.to_dict()
    assert "nodes" in d and "edges" in d
    assert len(d["nodes"]) == len(catalog.nodes)
    assert catalog.paths() == [n["path"] for n in d["nodes"]]
    for n in d["nodes"]:
        assert "description" in n and "example_hint" in n


def test_write_catalog_artifact():
    """write_catalog_artifact creates node_catalog.json in debug dir."""
    catalog = build_node_catalog(SampleCompany)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_catalog_artifact(catalog, tmp)
        assert path == str(Path(tmp) / "node_catalog.json")
        assert Path(path).exists()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "nodes" in data and "edges" in data
        assert any(n["path"] == "employees[]" for n in data["nodes"])


def test_discovery_prompt_shape():
    """Discovery prompt has system and user keys and mentions path/ids/parent skeleton."""
    catalog = build_node_catalog(SampleCompany)
    prompt = get_discovery_prompt("Document text.", catalog)
    assert "system" in prompt and "user" in prompt
    assert "employees[]" in prompt["system"] or "employees[]" in prompt["user"]
    assert "ids" in prompt["system"] or "ids" in prompt["user"]
    assert "parent" in prompt["system"] or "parent" in prompt["user"]
    assert "not the class name" in prompt["system"].lower()


def test_catalog_includes_schema_descriptions_and_examples():
    """When the Pydantic model has description and Field(examples=...), catalog carries them into the prompt."""

    class ModelWithHints(BaseModel):
        """Root entity for testing schema hints."""

        model_config = {"graph_id_fields": ["code"]}
        code: str = Field(description="Unique code", examples=["A1", "B2", "C3"])

    catalog = build_node_catalog(ModelWithHints)
    root = next((n for n in catalog.nodes if n.path == ""), None)
    assert root is not None
    # Model docstring or schema description may appear in description
    assert (
        "testing" in root.description.lower()
        or "entity" in root.description.lower()
        or root.description != ""
    )
    # id_field 'code' has examples in schema -> example_hint should be set
    assert "code" in root.example_hint and ("A1" in root.example_hint or "B2" in root.example_hint)


def test_discovery_prompt_includes_list_path_shared_child_rule():
    """Prompt must tell model to output one node per (parent, child) pair so shared children appear under each parent."""
    catalog = build_node_catalog(SampleCompany)
    prompt = get_discovery_prompt("Document text.", catalog)
    # System: list-path rule (same child under multiple parents → output once per parent)
    assert (
        "one node per" in prompt["system"]
        and "parent" in prompt["system"]
        and "child" in prompt["system"]
    )
    assert "same child" in prompt["system"] or "once per parent" in prompt["system"]
    # User: explicit instruction for shared children
    assert "nested list paths" in prompt["user"] and "one node per parent" in prompt["user"]
    assert "same path and ids" in prompt["user"] or "once per parent" in prompt["user"]


def test_discovery_prompt_explicitly_handles_no_id_paths():
    """Prompt should explicitly state ids={} for paths without id_fields."""

    class Address(BaseModel):
        street: str = ""
        model_config = ConfigDict(is_entity=False)

    class Employee(BaseModel):
        email: str
        addresses: list[Address] = Field(
            default_factory=list, json_schema_extra={"edge_label": "HAS_ADDRESS"}
        )
        model_config = ConfigDict(graph_id_fields=["email"])

    class Company(BaseModel):
        company_name: str
        employees: list[Employee] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=["company_name"])

    catalog = build_node_catalog(Company)
    prompt = get_discovery_prompt("Document text.", catalog)
    assert "ids must be {}" in prompt["system"]
    assert "Every NON-ROOT node must have parent" in prompt["system"]


def test_catalog_with_edge_label_populates_edges():
    """When template has json_schema_extra edge_label on list or scalar, catalog.edges is populated."""

    class Address(BaseModel):
        street: str = ""
        model_config = ConfigDict(is_entity=False)

    class Employee(BaseModel):
        email: str
        addresses: list[Address] = Field(
            default_factory=list, json_schema_extra={"edge_label": "HAS_ADDRESS"}
        )
        model_config = ConfigDict(graph_id_fields=["email"])

    class Company(BaseModel):
        company_name: str
        employees: list[Employee] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=["company_name"])

    catalog = build_node_catalog(Company)
    assert len(catalog.edges) >= 1
    edge_labels = [e.edge_label for e in catalog.edges]
    assert "HAS_ADDRESS" in edge_labels


def test_catalog_scalar_component_with_edge_label_adds_edge():
    """Scalar (non-list) component field with edge_label hits the scalar edge branch in build_node_catalog."""

    class Location(BaseModel):
        model_config = ConfigDict(is_entity=False)
        city: str = ""

    class Company(BaseModel):
        model_config = ConfigDict(graph_id_fields=["company_name"])
        company_name: str = ""
        headquarters: Location | None = Field(
            default=None,
            json_schema_extra={"edge_label": "HEADQUARTERS_AT"},
        )

    catalog = build_node_catalog(Company)
    edge_labels = [e.edge_label for e in catalog.edges]
    assert "HEADQUARTERS_AT" in edge_labels
    assert any(e.target_path == "headquarters" for e in catalog.edges)


def test_catalog_build_node_catalog_else_branch_component_list_no_edge_label():
    """Else branch (253-264): list field of component without edge_label; traverse, no edge appended."""

    class Note(BaseModel):
        model_config = ConfigDict(is_entity=False)
        text: str = ""

    class Doc(BaseModel):
        model_config = ConfigDict(graph_id_fields=["doc_id"])
        doc_id: str = ""
        notes: list[Note] = Field(default_factory=list)  # no edge_label

    catalog = build_node_catalog(Doc)
    assert len(catalog.nodes) >= 1
    assert any(n.path == "" for n in catalog.nodes)
    edge_targets = [e.target_path for e in catalog.edges]
    assert "notes[]" not in edge_targets


def test_catalog_build_node_catalog_else_branch_component_scalar_no_edge_label():
    """Else branch (266-274): scalar component field without edge_label; traverse, no edge appended."""

    class Meta(BaseModel):
        model_config = ConfigDict(is_entity=False)
        version: str = ""

    class Doc(BaseModel):
        model_config = ConfigDict(graph_id_fields=["doc_id"])
        doc_id: str = ""
        meta: Meta | None = None  # scalar component, no edge_label

    catalog = build_node_catalog(Doc)
    assert len(catalog.nodes) >= 1
    edge_targets = [e.target_path for e in catalog.edges]
    assert "meta" not in edge_targets


def test_catalog_collects_field_aliases() -> None:
    class Person(BaseModel):
        model_config = ConfigDict(graph_id_fields=["name"])
        name: str

    class AliasRoot(BaseModel):
        model_config = ConfigDict(graph_id_fields=["document_number"])
        document_number: str
        line_items: list[Person] = Field(
            default_factory=list,
            validation_alias=AliasChoices("lineItems", "line_items"),
        )

    catalog = build_node_catalog(AliasRoot)
    assert "lineItems" in catalog.field_aliases
    assert catalog.field_aliases["lineItems"] == "line_items"


def test_validate_id_pass_skeleton_response_success():
    """Skeleton response (path, ids, parent) validates and keeps real ids."""
    catalog = build_node_catalog(SampleInvoice)
    data = {"nodes": [{"path": "", "ids": {"invoice_number": "INV-1"}, "parent": None}]}
    ok, errs, flat_nodes, counts = validate_id_pass_skeleton_response(data, catalog)
    assert ok is True
    assert len(errs) == 0
    assert len(flat_nodes) == 1
    assert flat_nodes[0]["path"] == ""
    assert flat_nodes[0]["ids"] == {"invoice_number": "INV-1"}
    assert counts.get("") == 1


def test_validate_id_pass_skeleton_response_flat_nested_with_parent():
    """Nested instance with parent path+ids validates and keeps parent refs."""
    catalog = build_node_catalog(SampleCompany)
    data = {
        "nodes": [
            {"path": "", "ids": {"company_name": "Acme"}, "parent": None},
            {
                "path": "employees[]",
                "ids": {"email": "a@b.com"},
                "parent": {"path": "", "ids": {"company_name": "Acme"}},
            },
        ]
    }
    ok, _errs, flat_nodes, counts = validate_id_pass_skeleton_response(data, catalog)
    assert ok is True
    assert len(flat_nodes) == 2
    assert counts[""] == 1
    assert counts["employees[]"] == 1
    assert flat_nodes[1]["parent"] is not None
    assert flat_nodes[1]["parent"]["path"] == ""
    assert flat_nodes[1]["ids"] == {"email": "a@b.com"}


def test_validate_id_pass_skeleton_response_rejects_invalid_path():
    """Unknown path in response causes validation failure."""
    catalog = build_node_catalog(SampleCompany)
    data = {
        "nodes": [
            {"path": "", "ids": {"company_name": "Acme"}, "parent": None},
            {
                "path": "unknown[]",
                "ids": {"x": "1"},
                "parent": {"path": "", "ids": {"company_name": "Acme"}},
            },
        ]
    }
    ok, errs, _flat_nodes, _counts = validate_id_pass_skeleton_response(data, catalog)
    assert ok is False
    assert any("invalid path" in e for e in errs)


def test_validate_id_pass_skeleton_response_rejects_missing_id_fields():
    """Missing ids for a path is rejected."""
    catalog = build_node_catalog(SampleCompany)
    data = {
        "nodes": [
            {"path": "", "ids": {"company_name": "Acme"}, "parent": None},
            {
                "path": "employees[]",
                "ids": {},
                "parent": {"path": "", "ids": {"company_name": "Acme"}},
            },
        ]
    }
    ok, errs, _flat_nodes, _counts = validate_id_pass_skeleton_response(data, catalog)
    assert ok is False
    assert any("missing id field" in e for e in errs)


def test_validate_id_pass_skeleton_response_rejects_wrong_parent_path():
    """Child parent.path must match catalog parent_path."""
    catalog = build_node_catalog(SampleCompany)
    data = {
        "nodes": [
            {"path": "", "ids": {"company_name": "Acme"}, "parent": None},
            {
                "path": "employees[]",
                "ids": {"email": "a@b.com"},
                "parent": {"path": "employees[]", "ids": {"email": "a@b.com"}},
            },
        ]
    }
    ok, errs, _, _ = validate_id_pass_skeleton_response(data, catalog)
    assert ok is False
    assert any("must equal catalog parent_path" in e for e in errs)


def test_validate_id_pass_skeleton_response_rejects_orphan_parent_reference():
    """Parent reference must point to an existing node in the same payload."""
    catalog = build_node_catalog(SampleCompany)
    data = {
        "nodes": [
            {"path": "", "ids": {"company_name": "Acme"}, "parent": None},
            {
                "path": "employees[]",
                "ids": {"email": "a@b.com"},
                "parent": {"path": "", "ids": {"company_name": "MissingCompany"}},
            },
        ]
    }
    ok, errs, _, _ = validate_id_pass_skeleton_response(data, catalog)
    assert ok is False
    assert any("parent reference not found" in e for e in errs)


def test_flat_nodes_to_path_lists():
    """flat_nodes_to_path_lists groups by path."""
    flat = [
        {
            "path": "offres[]",
            "ids": {"nom": "A"},
            "provenance": "p1",
            "parent": {"path": "", "ids": {}},
        },
        {
            "path": "offres[]",
            "ids": {"nom": "B"},
            "provenance": "p2",
            "parent": {"path": "", "ids": {}},
        },
    ]
    grouped = flat_nodes_to_path_lists(flat)
    assert list(grouped.keys()) == ["offres[]"]
    assert len(grouped["offres[]"]) == 2
    assert grouped["offres[]"][0]["ids"] == {"nom": "A"}


def test_get_id_pass_shards_single_when_zero():
    """get_id_pass_shards with shard_size 0 returns one shard with all paths and parent closure."""
    catalog = build_node_catalog(SampleCompany)
    shards = get_id_pass_shards(catalog, 0)
    assert len(shards) == 1
    primary, allowed = shards[0]
    assert set(primary) == set(catalog.paths())
    assert "" in allowed
    assert "employees[]" in allowed
    assert set(allowed) >= set(primary)


def test_get_id_pass_shards_splits_by_size():
    """get_id_pass_shards with small shard_size returns multiple shards; each allowed_paths includes parents."""
    catalog = build_node_catalog(SampleCompany)
    shards = get_id_pass_shards(catalog, 1)
    assert len(shards) >= 1
    for primary_paths, allowed_paths in shards:
        assert len(primary_paths) >= 1
        assert set(allowed_paths) >= set(primary_paths)
        assert "" in allowed_paths or "" not in primary_paths  # root in allowed when needed


def test_get_allowed_paths_for_primary_paths_includes_parents():
    """get_allowed_paths_for_primary_paths returns primary paths plus parent chain."""
    catalog = build_node_catalog(SampleCompany)
    allowed = get_allowed_paths_for_primary_paths(catalog, ["employees[]"])
    assert "employees[]" in allowed
    assert "" in allowed
    assert len(allowed) >= 2


def test_get_model_for_path_returns_model_for_root_and_list_paths():
    """get_model_for_path returns the Pydantic model for a catalog path."""
    catalog = build_node_catalog(SampleCompany)
    assert catalog.paths()  # ensure we have paths
    root_model = get_model_for_path(SampleCompany, "")
    assert root_model is SampleCompany
    emp_model = get_model_for_path(SampleCompany, "employees[]")
    assert emp_model is SamplePerson
    missing = get_model_for_path(SampleCompany, "nonexistent_path")
    assert missing is None


def test_build_discovery_schema_with_empty_allowed_paths():
    """build_discovery_schema with empty allowed_paths produces valid schema with empty path enum."""
    catalog = build_node_catalog(SampleInvoice)
    schema_str = build_discovery_schema(catalog, allowed_paths=[])
    data = json.loads(schema_str)
    path_enum = data["$defs"]["node_instance"]["properties"]["path"].get("enum")
    assert path_enum == []


def test_build_discovery_schema_restricts_paths():
    """build_discovery_schema with allowed_paths restricts path enum; has path, ids, parent only."""
    catalog = build_node_catalog(SampleCompany)
    allowed = ["", "employees[]"]
    schema_str = build_discovery_schema(catalog, allowed_paths=allowed)
    schema = json.loads(schema_str)
    props = schema["$defs"]["node_instance"]["properties"]
    assert set(props["path"]["enum"]) == set(allowed)
    assert "ids" in props
    assert "parent" in props
    assert "index" not in props
    assert "provenance" not in props


def test_merge_and_dedupe_flat_nodes():
    """merge_and_dedupe_flat_nodes merges shard results and dedupes by (path, ids); per_path_counts correct."""
    catalog = build_node_catalog(SampleCompany)
    list1 = [
        {"path": "", "ids": {"company_name": "A"}, "provenance": "p0", "parent": None},
        {
            "path": "employees[]",
            "ids": {"email": "e1"},
            "provenance": "p1",
            "parent": {"path": "", "ids": {"company_name": "A"}},
        },
    ]
    list2 = [
        {
            "path": "",
            "ids": {"company_name": "A"},
            "provenance": "dup",
            "parent": None,
        },  # duplicate root
        {
            "path": "employees[]",
            "ids": {"email": "e2"},
            "provenance": "p2",
            "parent": {"path": "", "ids": {"company_name": "A"}},
        },
    ]
    merged, per_path_counts = merge_and_dedupe_flat_nodes([list1, list2], catalog)
    assert len(merged) == 3  # root once, employees e1 and e2
    assert per_path_counts.get("", 0) == 1
    assert per_path_counts.get("employees[]", 0) == 2


def test_merge_and_dedupe_collapses_id_variants_run1_run_1():
    """Identifier variants (run_1, run1, Run-1) collapse to one node via canonicalize_identity_for_dedup."""

    class Run(BaseModel):
        run_id: str = ""
        model_config = ConfigDict(graph_id_fields=["run_id"])

    catalog = build_node_catalog(Run)
    list1 = [{"path": "", "ids": {"run_id": "run1"}, "parent": None}]
    list2 = [{"path": "", "ids": {"run_id": "run_1"}, "parent": None}]
    merged, per_path_counts = merge_and_dedupe_flat_nodes([list1, list2], catalog)
    assert len(merged) == 1
    assert per_path_counts.get("", 0) == 1
    # First occurrence wins; node keeps one of the raw ids
    assert merged[0]["ids"]["run_id"] in ("run1", "run_1")


def test_merge_and_dedupe_keeps_multiple_component_instances_without_id_fields():
    """Component paths with empty id_fields keep all instances (no collapse by empty ids)."""

    class Address(BaseModel):
        street: str = ""
        model_config = ConfigDict(is_entity=False)

    class Employee(BaseModel):
        name: str = ""
        email: str
        addresses: list[Address] = Field(
            default_factory=list, json_schema_extra={"edge_label": "HAS_ADDRESS"}
        )
        model_config = ConfigDict(graph_id_fields=["email"])

    class Company(BaseModel):
        company_name: str
        employees: list[Employee] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=["company_name"])

    catalog = build_node_catalog(Company)
    list1 = [
        {"path": "", "ids": {"company_name": "Acme"}, "parent": None},
        {
            "path": "employees[]",
            "ids": {"email": "a@b.com"},
            "parent": {"path": "", "ids": {"company_name": "Acme"}},
        },
        {
            "path": "employees[].addresses[]",
            "ids": {},
            "parent": {"path": "employees[]", "ids": {"email": "a@b.com"}},
        },
    ]
    list2 = [
        {"path": "", "ids": {"company_name": "Acme"}, "parent": None},
        {
            "path": "employees[]",
            "ids": {"email": "b@b.com"},
            "parent": {"path": "", "ids": {"company_name": "Acme"}},
        },
        {
            "path": "employees[].addresses[]",
            "ids": {},
            "parent": {"path": "employees[]", "ids": {"email": "b@b.com"}},
        },
    ]
    merged, per_path_counts = merge_and_dedupe_flat_nodes([list1, list2], catalog)
    assert per_path_counts.get("", 0) == 1
    assert per_path_counts.get("employees[]", 0) == 2
    assert per_path_counts.get("employees[].addresses[]", 0) == 2
    address_nodes = [n for n in merged if n.get("path") == "employees[].addresses[]"]
    assert len(address_nodes) == 2
    keys = [n.get("__instance_key") for n in address_nodes]
    assert all(isinstance(k, str) and k for k in keys)
    assert len(set(keys)) == 2


def test_merge_and_dedupe_flat_nodes_list_under_list_keeps_one_per_parent_child():
    """For list-under-list paths, same (path, ids) under different parents yields one descriptor per (parent, child)."""
    from docling_graph.core.extractors.contracts.staged.catalog import (
        _is_list_under_list,
        build_node_catalog,
    )

    class Item(BaseModel):
        id: str = ""
        model_config = ConfigDict(graph_id_fields=["id"])

    class Section(BaseModel):
        name: str = ""
        items: list[Item] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=["name"])

    class RootWithSections(BaseModel):
        ref: str = ""
        sections: list[Section] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=["ref"])

    catalog = build_node_catalog(RootWithSections)
    grandchild_spec = next(
        s for s in catalog.nodes if s.path != "" and s.parent_path != "" and s.path.endswith("[]")
    )
    assert _is_list_under_list(grandchild_spec.path, grandchild_spec), (
        "sections[].items[] is list-under-list"
    )
    # Same child (id="x") under two parents (sections A and B) — must keep both descriptors
    flat = [
        {"path": "", "ids": {"ref": "R"}, "parent": None},
        {"path": "sections[]", "ids": {"name": "A"}, "parent": {"path": "", "ids": {"ref": "R"}}},
        {"path": "sections[]", "ids": {"name": "B"}, "parent": {"path": "", "ids": {"ref": "R"}}},
        {
            "path": "sections[].items[]",
            "ids": {"id": "x"},
            "parent": {"path": "sections[]", "ids": {"name": "A"}},
        },
        {
            "path": "sections[].items[]",
            "ids": {"id": "x"},
            "parent": {"path": "sections[]", "ids": {"name": "B"}},
        },
    ]
    merged, per_path_counts = merge_and_dedupe_flat_nodes([flat], catalog)
    assert per_path_counts.get("sections[].items[]", 0) == 2
    items = [n for n in merged if n.get("path") == "sections[].items[]"]
    assert len(items) == 2
    parent_id_vals = []
    for n in items:
        pid = (n.get("parent") or {}).get("ids")
        parent_id_vals.append(tuple(pid.values()) if isinstance(pid, dict) else ())
    assert ("A",) in parent_id_vals and ("B",) in parent_id_vals


def test_staged_list_under_list_dedup_then_merge_produces_shared_child():
    """ID-pass-style flat list with same child under two parents → dedup keeps both → merge attaches under each parent."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import merge_filled_into_root

    class Item(BaseModel):
        id: str = ""
        model_config = ConfigDict(graph_id_fields=["id"])

    class Section(BaseModel):
        name: str = ""
        items: list[Item] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=["name"])

    class RootWithSections(BaseModel):
        ref: str = ""
        sections: list[Section] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=["ref"])

    catalog = build_node_catalog(RootWithSections)
    flat = [
        {"path": "", "ids": {"ref": "R"}, "parent": None},
        {"path": "sections[]", "ids": {"name": "A"}, "parent": {"path": "", "ids": {"ref": "R"}}},
        {"path": "sections[]", "ids": {"name": "B"}, "parent": {"path": "", "ids": {"ref": "R"}}},
        {
            "path": "sections[].items[]",
            "ids": {"id": "x"},
            "parent": {"path": "sections[]", "ids": {"name": "A"}},
        },
        {
            "path": "sections[].items[]",
            "ids": {"id": "x"},
            "parent": {"path": "sections[]", "ids": {"name": "B"}},
        },
    ]
    merged_flat, _ = merge_and_dedupe_flat_nodes([flat], catalog)
    path_to_descriptors = flat_nodes_to_path_lists(merged_flat)
    path_filled = {
        "": [{"ref": "R"}],
        "sections[]": [{"name": "A"}, {"name": "B"}],
        "sections[].items[]": [{"id": "x"}, {"id": "x"}],
    }
    merged = merge_filled_into_root(path_filled, path_to_descriptors, catalog)
    assert merged.get("sections") is not None
    assert len(merged["sections"]) == 2
    assert (
        len(merged["sections"][0]["items"]) == 1 and merged["sections"][0]["items"][0]["id"] == "x"
    )
    assert (
        len(merged["sections"][1]["items"]) == 1 and merged["sections"][1]["items"][0]["id"] == "x"
    )


def test_write_id_pass_artifact():
    """write_id_pass_artifact writes id_pass.json with nodes array and per_path_counts."""
    with tempfile.TemporaryDirectory() as tmp:
        path = write_id_pass_artifact(
            {"nodes": [{"path": "", "ids": {"invoice_number": "INV-1"}, "parent": None}]},
            {"": 1},
            tmp,
        )
        assert path.endswith("id_pass.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "nodes" in data and "per_path_counts" in data
        assert isinstance(data["nodes"], list)
        assert data["per_path_counts"][""] == 1


def test_catalog_orchestrator_end_to_end():
    """Catalog orchestrator: mock LLM returns flat id_pass then fill; merge produces root dict."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import (
        CatalogOrchestrator,
        CatalogOrchestratorConfig,
    )

    call_log: list[tuple[str, str]] = []

    def mock_llm(
        prompt: dict, schema_json: str, context: str, **kwargs: object
    ) -> dict | list | None:
        call_log.append((context, schema_json[:80]))
        if "catalog_id_pass" in context:
            return {
                "nodes": [{"path": "", "ids": {"invoice_number": "INV-001"}, "parent": None}],
            }
        if "fill_" in context:
            return [
                {
                    "invoice_number": "INV-001",
                    "date": "2024-01-15",
                    "total_amount": 100.0,
                    "vendor_name": "Acme",
                    "items": [],
                }
            ]
        return None

    with tempfile.TemporaryDirectory() as tmp:
        config = CatalogOrchestratorConfig(max_nodes_per_call=5, parallel_workers=1)
        schema_json = '{"type":"object","properties":{"invoice_number":{},"date":{},"total_amount":{},"vendor_name":{},"items":{}}}'
        orch = CatalogOrchestrator(
            llm_call_fn=mock_llm,
            schema_json=schema_json,
            template=SampleInvoice,
            config=config,
            debug_dir=tmp,
        )
        result = orch.extract(markdown="Invoice INV-001...", context="test")
    assert result is not None
    assert result.get("invoice_number") == "INV-001"
    assert "date" in result or "total_amount" in result or "vendor_name" in result
    assert len(call_log) >= 2
    assert any("catalog_id_pass" in c[0] for c in call_log)
    assert any("fill_" in c[0] for c in call_log)


def test_catalog_orchestrator_list_under_list_fill_reuse():
    """With list-under-list, same child under two parents: fill is called once per unique (path, ids), then expanded."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import (
        CatalogOrchestrator,
        CatalogOrchestratorConfig,
    )

    class Item(BaseModel):
        id: str = ""
        model_config = ConfigDict(graph_id_fields=["id"])

    class Section(BaseModel):
        name: str = ""
        items: list[Item] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=["name"])

    class RootWithSections(BaseModel):
        ref: str = ""
        sections: list[Section] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=["ref"])

    fill_call_count = 0

    def mock_llm(
        prompt: dict, schema_json: str, context: str, **kwargs: object
    ) -> dict | list | None:
        nonlocal fill_call_count
        if "catalog_id_pass" in context:
            return {
                "nodes": [
                    {"path": "", "ids": {"ref": "R"}, "parent": None},
                    {
                        "path": "sections[]",
                        "ids": {"name": "A"},
                        "parent": {"path": "", "ids": {"ref": "R"}},
                    },
                    {
                        "path": "sections[]",
                        "ids": {"name": "B"},
                        "parent": {"path": "", "ids": {"ref": "R"}},
                    },
                    {
                        "path": "sections[].items[]",
                        "ids": {"id": "x"},
                        "parent": {"path": "sections[]", "ids": {"name": "A"}},
                    },
                    {
                        "path": "sections[].items[]",
                        "ids": {"id": "x"},
                        "parent": {"path": "sections[]", "ids": {"name": "B"}},
                    },
                ]
            }
        if "fill_" in context:
            fill_call_count += 1
            # Fill order is deepest first: sections[].items[], then sections[], then root
            if "id" in schema_json and "name" not in schema_json and "ref" not in schema_json:
                return [{"id": "x"}]
            if "sections" in schema_json or "name" in schema_json:
                return [{"name": "A"}, {"name": "B"}]
            return [{"ref": "R"}]
        return None

    with tempfile.TemporaryDirectory() as tmp:
        config = CatalogOrchestratorConfig(max_nodes_per_call=5, parallel_workers=1)
        schema_json = '{"type":"object","properties":{"ref":{},"sections":{}}}'
        orch = CatalogOrchestrator(
            llm_call_fn=mock_llm,
            schema_json=schema_json,
            template=RootWithSections,
            config=config,
            debug_dir=tmp,
        )
        result = orch.extract(
            markdown="Doc with sections A and B, each with item x.", context="test"
        )
    assert result is not None
    assert result.get("ref") == "R"
    assert result.get("sections") is not None
    assert len(result["sections"]) == 2
    assert len(result["sections"][0].get("items", [])) == 1
    assert result["sections"][0]["items"][0].get("id") == "x"
    assert len(result["sections"][1].get("items", [])) == 1
    assert result["sections"][1]["items"][0].get("id") == "x"
    # One fill call for root, one for sections (2 items), one for items (1 unique, expanded to 2)
    assert fill_call_count == 3


def test_merge_filled_into_root_nested_by_parent_id():
    """Merge attaches nested list items to parent by parent path+ids."""
    from docling_graph.core.extractors.contracts.staged.catalog import build_node_catalog
    from docling_graph.core.extractors.contracts.staged.orchestrator import merge_filled_into_root

    catalog = build_node_catalog(SampleCompany)
    path_filled = {
        "": [{"company_name": "Acme", "industry": "Tech"}],
        "employees[]": [
            {"email": "a@b.com", "first_name": "Alice"},
            {"email": "b@b.com", "first_name": "Bob"},
        ],
    }
    path_descriptors = {
        "": [{"path": "", "ids": {}, "provenance": "p0", "parent": None}],
        "employees[]": [
            {
                "path": "employees[]",
                "ids": {"email": "a@b.com"},
                "provenance": "p1",
                "parent": {"path": "", "ids": {}},
            },
            {
                "path": "employees[]",
                "ids": {"email": "b@b.com"},
                "provenance": "p2",
                "parent": {"path": "", "ids": {}},
            },
        ],
    }
    merged = merge_filled_into_root(path_filled, path_descriptors, catalog)
    assert merged.get("company_name") == "Acme"
    assert "employees" in merged
    assert len(merged["employees"]) == 2
    assert merged["employees"][0]["email"] == "a@b.com"
    assert merged["employees"][1]["email"] == "b@b.com"


def test_merge_filled_into_root_preserves_component_instances_without_id_fields():
    """No-id component instances attach without being collapsed by lookup key collisions."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import merge_filled_into_root

    class Address(BaseModel):
        street: str = ""
        model_config = ConfigDict(is_entity=False)

    class Employee(BaseModel):
        email: str
        addresses: list[Address] = Field(
            default_factory=list, json_schema_extra={"edge_label": "HAS_ADDRESS"}
        )
        model_config = ConfigDict(graph_id_fields=["email"])

    class Company(BaseModel):
        company_name: str
        employees: list[Employee] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=["company_name"])

    catalog = build_node_catalog(Company)
    flat_nodes = [
        {"path": "", "ids": {"company_name": "Acme"}, "parent": None},
        {
            "path": "employees[]",
            "ids": {"email": "a@b.com"},
            "parent": {"path": "", "ids": {"company_name": "Acme"}},
        },
        {
            "path": "employees[]",
            "ids": {"email": "b@b.com"},
            "parent": {"path": "", "ids": {"company_name": "Acme"}},
        },
        {
            "path": "employees[].addresses[]",
            "ids": {},
            "__instance_key": "employees[].addresses[]#0",
            "parent": {"path": "employees[]", "ids": {"email": "a@b.com"}},
        },
        {
            "path": "employees[].addresses[]",
            "ids": {},
            "__instance_key": "employees[].addresses[]#1",
            "parent": {"path": "employees[]", "ids": {"email": "b@b.com"}},
        },
    ]
    path_descriptors = flat_nodes_to_path_lists(flat_nodes)
    path_filled = {
        "": [{"company_name": "Acme"}],
        "employees[]": [{"email": "a@b.com"}, {"email": "b@b.com"}],
        "employees[].addresses[]": [{"street": "Rue A"}, {"street": "Rue B"}],
    }
    merged = merge_filled_into_root(path_filled, path_descriptors, catalog)
    assert len(merged["employees"]) == 2
    assert merged["employees"][0]["addresses"][0]["street"] == "Rue A"
    assert merged["employees"][1]["addresses"][0]["street"] == "Rue B"


def test_merge_filled_into_root_shared_child_under_multiple_parents():
    """When same child identity appears under multiple parents (ID pass one-per-parent), merge attaches it under each parent."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import merge_filled_into_root

    class Item(BaseModel):
        id: str = ""
        model_config = ConfigDict(graph_id_fields=["id"])

    class Section(BaseModel):
        name: str = ""
        items: list[Item] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=["name"])

    class RootWithSections(BaseModel):
        ref: str = ""
        sections: list[Section] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=["ref"])

    catalog = build_node_catalog(RootWithSections)
    # Same child (id="x") under two different parents (sections A and B)
    path_descriptors = {
        "": [{"path": "", "ids": {"ref": "R"}, "parent": None}],
        "sections[]": [
            {
                "path": "sections[]",
                "ids": {"name": "A"},
                "parent": {"path": "", "ids": {"ref": "R"}},
            },
            {
                "path": "sections[]",
                "ids": {"name": "B"},
                "parent": {"path": "", "ids": {"ref": "R"}},
            },
        ],
        "sections[].items[]": [
            {
                "path": "sections[].items[]",
                "ids": {"id": "x"},
                "parent": {"path": "sections[]", "ids": {"name": "A"}},
            },
            {
                "path": "sections[].items[]",
                "ids": {"id": "x"},
                "parent": {"path": "sections[]", "ids": {"name": "B"}},
            },
        ],
    }
    path_filled = {
        "": [{"ref": "R"}],
        "sections[]": [{"name": "A"}, {"name": "B"}],
        "sections[].items[]": [{"id": "x"}, {"id": "x"}],
    }
    merged = merge_filled_into_root(path_filled, path_descriptors, catalog)
    assert merged.get("sections") is not None
    assert len(merged["sections"]) == 2
    assert len(merged["sections"][0]["items"]) == 1
    assert len(merged["sections"][1]["items"]) == 1
    assert merged["sections"][0]["items"][0]["id"] == "x"
    assert merged["sections"][1]["items"][0]["id"] == "x"


def test_merge_filled_into_root_canonical_lookup_resolves_parent_id_variants():
    """After canonical dedup, child with parent ref run_1 finds parent kept as run1 (same canonical key)."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import merge_filled_into_root

    class RunDetail(BaseModel):
        detail_id: str = ""
        model_config = ConfigDict(graph_id_fields=["detail_id"])

    class Run(BaseModel):
        run_id: str = ""
        details: list[RunDetail] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=["run_id"])

    class Root(BaseModel):
        runs: list[Run] = Field(default_factory=list)
        model_config = ConfigDict(graph_id_fields=[])

    catalog = build_node_catalog(Root)
    # One run descriptor (run1) after dedup; child descriptor references parent as run_1
    path_descriptors = {
        "": [{"path": "", "ids": {}, "parent": None}],
        "runs[]": [
            {"path": "runs[]", "ids": {"run_id": "run1"}, "parent": {"path": "", "ids": {}}},
        ],
        "runs[].details[]": [
            {
                "path": "runs[].details[]",
                "ids": {"detail_id": "d1"},
                "parent": {"path": "runs[]", "ids": {"run_id": "run_1"}},
            },
        ],
    }
    path_filled = {
        "": [{}],
        "runs[]": [{"run_id": "run1", "details": []}],
        "runs[].details[]": [{"detail_id": "d1"}],
    }
    merge_stats: dict[str, int] = {}
    merged = merge_filled_into_root(path_filled, path_descriptors, catalog, stats=merge_stats)
    assert merged.get("runs") is not None
    assert len(merged["runs"]) == 1
    assert merged["runs"][0].get("run_id") == "run1"
    assert merged["runs"][0].get("details") is not None
    assert len(merged["runs"][0]["details"]) == 1
    assert merged["runs"][0]["details"][0].get("detail_id") == "d1"
    assert merge_stats.get("parent_lookup_miss", 0) == 0


def test_fill_pass_order_bottom_up():
    """Fill pass runs in bottom-up order: employees[] (leaf) before root."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import (
        CatalogOrchestrator,
        CatalogOrchestratorConfig,
    )

    def mock_llm(
        prompt: dict, schema_json: str, context: str, **kwargs: object
    ) -> dict | list | None:
        if "catalog_id_pass" in context:
            return {
                "nodes": [
                    {"path": "", "ids": {"company_name": "Acme"}, "parent": None},
                    {
                        "path": "employees[]",
                        "ids": {"email": "a@b.com"},
                        "parent": {"path": "", "ids": {"company_name": "Acme"}},
                    },
                ],
            }
        if "fill_call_0" in context:
            return [{"email": "a@b.com", "first_name": "Alice"}]
        if "fill_call_1" in context:
            return [{"company_name": "Acme", "industry": "Tech"}]
        return None

    with tempfile.TemporaryDirectory() as tmp:
        config = CatalogOrchestratorConfig(max_nodes_per_call=5, parallel_workers=1)
        schema_json = (
            '{"type":"object","properties":{"company_name":{},"industry":{},"employees":{}}}'
        )
        orch = CatalogOrchestrator(
            llm_call_fn=mock_llm,
            schema_json=schema_json,
            template=SampleCompany,
            config=config,
            debug_dir=tmp,
        )
        orch.extract(markdown="Acme Corp...", context="test")
        trace_path = Path(tmp) / "staged_trace.json"
        assert trace_path.exists()
        with open(trace_path, encoding="utf-8") as f:
            trace = json.load(f)
    fill_batches = trace.get("fill_batches", [])
    paths_in_order = [b["path"] for b in fill_batches]
    assert "employees[]" in paths_in_order and "" in paths_in_order
    assert paths_in_order.index("employees[]") < paths_in_order.index("")


def test_detect_merge_conflicts_returns_empty():
    """_detect_merge_conflicts returns [] (no conflicts detected yet)."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import _detect_merge_conflicts

    assert _detect_merge_conflicts({}) == []
    assert _detect_merge_conflicts({"company_name": "Acme", "employees": []}) == []


def test_maybe_resolve_conflicts_returns_merged_when_no_conflicts():
    """_maybe_resolve_conflicts returns merged unchanged when no conflicts detected."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import _maybe_resolve_conflicts

    catalog = build_node_catalog(SampleCompany)
    merged = {"company_name": "Acme", "employees": []}
    out = _maybe_resolve_conflicts(merged, catalog, lambda *a: None, "test")
    assert out == merged


def test_id_pass_shards_can_run_in_parallel_with_fill_workers():
    """ID pass shards run in parallel when parallel_workers > 1; results are merged in shard order."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import (
        CatalogOrchestrator,
        CatalogOrchestratorConfig,
    )

    id_contexts: list[str] = []

    def mock_llm(
        prompt: dict, schema_json: str, context: str, **kwargs: object
    ) -> dict | list | None:
        if "catalog_id_pass_shard_" in context and "_split" not in context:
            id_contexts.append(context)
            return {"nodes": [{"path": "", "ids": {"company_name": "Acme"}, "parent": None}]}
        if "fill_call_" in context:
            return [{"company_name": "Acme", "industry": "Tech"}]
        return None

    with tempfile.TemporaryDirectory() as tmp:
        config = CatalogOrchestratorConfig(
            max_nodes_per_call=5,
            parallel_workers=4,
            id_shard_size=1,
        )
        schema_json = (
            '{"type":"object","properties":{"company_name":{},"industry":{},"employees":{}}}'
        )
        orch = CatalogOrchestrator(
            llm_call_fn=mock_llm,
            schema_json=schema_json,
            template=SampleCompany,
            config=config,
            debug_dir=tmp,
        )
        result = orch.extract(markdown="Acme Corp...", context="test")

    shard_indexes = [
        int(c.split("catalog_id_pass_shard_")[1])
        for c in id_contexts
        if "catalog_id_pass_shard_" in c and "_split" not in c
    ]
    # With id_shard_size=1 we get 2 identity shards (root + one other); all shards were invoked
    assert len(shard_indexes) >= 1
    assert set(shard_indexes) == set(range(len(shard_indexes)))
    assert result is not None


def test_fill_pass_accepts_wrapped_items_response():
    """Fill pass unwraps LLM response when it is an object with 'items' (object-root schema)."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import (
        CatalogOrchestrator,
        CatalogOrchestratorConfig,
    )

    def mock_llm(
        prompt: dict, schema_json: str, context: str, **kwargs: object
    ) -> dict | list | None:
        if "catalog_id_pass" in context:
            return {"nodes": [{"path": "", "ids": {"company_name": "Acme"}, "parent": None}]}
        if "fill_call_" in context:
            return {"items": [{"company_name": "Acme", "industry": "Tech"}]}
        return None

    with tempfile.TemporaryDirectory() as tmp:
        config = CatalogOrchestratorConfig(max_nodes_per_call=5, parallel_workers=1)
        schema_json = (
            '{"type":"object","properties":{"company_name":{},"industry":{},"employees":{}}}'
        )
        orch = CatalogOrchestrator(
            llm_call_fn=mock_llm,
            schema_json=schema_json,
            template=SampleCompany,
            config=config,
            debug_dir=tmp,
        )
        result = orch.extract(markdown="Acme Corp...", context="test")
    assert result is not None
    assert result.get("company_name") == "Acme"
    assert result.get("industry") == "Tech"


def test_get_identity_paths_returns_root_and_id_entities_only():
    """Identity paths keep root and ID-bearing entities for minimal ID pass."""
    catalog = build_node_catalog(SampleCompany)
    paths = get_identity_paths(catalog)
    assert "" in paths
    assert "employees[]" in paths


def test_get_id_pass_shards_v2_root_first_and_parent_complete():
    """V2 shards should keep parent closure and put root shard first."""
    catalog = build_node_catalog(SampleCompany)
    shards = get_id_pass_shards_v2(catalog, shard_size=1, identity_only=True, root_first=True)
    assert len(shards) >= 1
    first_primary, first_allowed = shards[0]
    assert "" in first_primary
    assert first_allowed == first_primary
    for primary_paths, allowed_paths in shards:
        assert set(allowed_paths) == set(primary_paths)
        assert "" in allowed_paths


def test_discovery_prompt_compact_omits_schema_block():
    """Compact prompt mode should avoid embedding large schema in user prompt."""
    catalog = build_node_catalog(SampleCompany)
    prompt = get_discovery_prompt(
        "Document text.",
        catalog,
        compact=True,
        include_schema_in_user=False,
    )
    assert "ID pass only" in prompt["system"]
    assert "=== SCHEMA ===" not in prompt["user"]
    assert "=== ALLOWED PATHS ===" in prompt["user"]


def test_orchestrator_quality_gate_fails_without_root_instance():
    """When ID pass has no valid root, orchestrator should return None for fallback."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import (
        CatalogOrchestrator,
        CatalogOrchestratorConfig,
    )

    def mock_llm(
        prompt: dict, schema_json: str, context: str, **kwargs: object
    ) -> dict | list | None:
        if "catalog_id_pass" in context:
            # Missing root by design; validation will reject and leave sparse ID map
            return {
                "nodes": [
                    {
                        "path": "employees[]",
                        "ids": {"email": "a@b.com"},
                        "parent": {"path": "", "ids": {"company_name": "Acme"}},
                    }
                ]
            }
        if "fill_call_" in context:
            return [{}]
        return None

    with tempfile.TemporaryDirectory() as tmp:
        orch = CatalogOrchestrator(
            llm_call_fn=mock_llm,
            schema_json='{"type":"object"}',
            template=SampleCompany,
            config=CatalogOrchestratorConfig(),
            debug_dir=tmp,
        )
        result = orch.extract(markdown="Acme", context="test")
    assert result is None


def test_orchestrator_sanitizes_root_overfill_nested_children():
    """Root fill should not override nested child paths discovered/fill separately."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import (
        CatalogOrchestrator,
        CatalogOrchestratorConfig,
    )

    def mock_llm(
        prompt: dict, schema_json: str, context: str, **kwargs: object
    ) -> dict | list | None:
        if "catalog_id_pass" in context:
            return {
                "nodes": [
                    {"path": "", "ids": {"company_name": "Acme"}, "parent": None},
                    {
                        "path": "employees[]",
                        "ids": {"email": "good@acme.com"},
                        "parent": {"path": "", "ids": {"company_name": "Acme"}},
                    },
                ]
            }
        if "fill_call_0" in context:
            return [{"email": "good@acme.com", "first_name": "Good"}]
        if "fill_call_1" in context:
            # Nested employees payload should be ignored by projected fill/sanitization.
            return [{"company_name": "Acme", "employees": [{"email": "evil@acme.com"}]}]
        return None

    with tempfile.TemporaryDirectory() as tmp:
        orch = CatalogOrchestrator(
            llm_call_fn=mock_llm,
            schema_json='{"type":"object"}',
            template=SampleCompany,
            config=CatalogOrchestratorConfig(),
            debug_dir=tmp,
        )
        result = orch.extract(markdown="Acme", context="test")

    assert result is not None
    assert result.get("company_name") == "Acme"
    assert result.get("employees")
    assert result["employees"][0]["email"] == "good@acme.com"
