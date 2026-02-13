"""
Unit tests for the node/edge catalog builder.
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

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
    merge_and_dedupe_flat_nodes,
    validate_id_pass_skeleton_response,
    write_catalog_artifact,
    write_id_pass_artifact,
)
from tests.fixtures.sample_templates.test_template import SampleCompany, SampleInvoice


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


def test_id_pass_shards_run_sequentially_even_with_fill_workers():
    """ID pass shards run sequentially; workers are used only for fill calls."""
    from docling_graph.core.extractors.contracts.staged.orchestrator import (
        CatalogOrchestrator,
        CatalogOrchestratorConfig,
    )

    id_contexts: list[str] = []

    def mock_llm(
        prompt: dict, schema_json: str, context: str, **kwargs: object
    ) -> dict | list | None:
        if "catalog_id_pass_shard_" in context:
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
        orch.extract(markdown="Acme Corp...", context="test")

    shard_indexes = [int(c.split("catalog_id_pass_shard_")[1]) for c in id_contexts]
    assert shard_indexes == sorted(shard_indexes)


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
