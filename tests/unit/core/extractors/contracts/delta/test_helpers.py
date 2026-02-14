from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.extractors.contracts.delta.catalog import build_delta_node_catalog
from docling_graph.core.extractors.contracts.delta.helpers import (
    build_dedup_policy,
    flatten_node_properties,
    merge_delta_graphs,
    sanitize_batch_echo_from_graph,
)


class Person(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str
    title: str | None = None


class RootDoc(BaseModel):
    model_config = ConfigDict(graph_id_fields=["document_number"])
    document_number: str
    people: list[Person] = Field(default_factory=list)


def test_build_dedup_policy_uses_catalog_identity_fields() -> None:
    catalog = build_delta_node_catalog(RootDoc)
    policy = build_dedup_policy(catalog)
    assert policy[""].identity_fields == ("document_number",)
    assert policy["people[]"].identity_fields == ("name",)


def test_flatten_node_properties_drops_nested_values() -> None:
    props = {
        "name": "Neo4j",
        "meta": {"nested": True},
        "tags": ["graph", {"bad": "object"}],
    }
    flattened = flatten_node_properties(props)
    assert "meta" not in flattened
    assert flattened["tags"] == ["graph"]


def test_merge_delta_graphs_uses_path_policy_for_dedup() -> None:
    catalog = build_delta_node_catalog(RootDoc)
    policy = build_dedup_policy(catalog)
    merged = merge_delta_graphs(
        [
            {
                "nodes": [
                    {
                        "path": "people[]",
                        "ids": {"name": "Alice"},
                        "properties": {"title": "Engineer"},
                    }
                ],
                "relationships": [],
            },
            {
                "nodes": [
                    {
                        "path": "people[]",
                        "ids": {"name": "Alice"},
                        "properties": {"title": ""},
                    }
                ],
                "relationships": [],
            },
        ],
        dedup_policy=policy,
    )
    assert len(merged["nodes"]) == 1
    assert merged["nodes"][0]["properties"]["title"] == "Engineer"


def test_merge_delta_graphs_prefers_richer_non_empty_string_values() -> None:
    catalog = build_delta_node_catalog(RootDoc)
    policy = build_dedup_policy(catalog)
    merged = merge_delta_graphs(
        [
            {
                "nodes": [
                    {
                        "path": "people[]",
                        "ids": {"name": "Alice"},
                        "properties": {"title": "Mgr"},
                    }
                ],
                "relationships": [],
            },
            {
                "nodes": [
                    {
                        "path": "people[]",
                        "ids": {"name": "Alice"},
                        "properties": {"title": "Senior Manager"},
                    }
                ],
                "relationships": [],
            },
        ],
        dedup_policy=policy,
    )
    assert merged["nodes"][0]["properties"]["title"] == "Senior Manager"
    assert merged["__merge_stats"]["property_conflicts"] >= 1


def test_merge_delta_graphs_no_identity_nodes_do_not_collapse() -> None:
    merged = merge_delta_graphs(
        [
            {
                "nodes": [
                    {"path": "misc[]", "ids": {}, "properties": {}},
                    {"path": "misc[]", "ids": {}, "properties": {}},
                ],
                "relationships": [],
            }
        ],
        dedup_policy=None,
    )
    assert len(merged["nodes"]) == 2


def test_merge_delta_graphs_tolerates_string_provenance_value() -> None:
    """When __property_provenance has a string value (malformed IR), merge coerces to list and appends."""
    catalog = build_delta_node_catalog(RootDoc)
    policy = build_dedup_policy(catalog)
    # First node: empty title so merge code does not set provenance; keep malformed string provenance.
    # Second node: same identity, non-empty title so we merge and hit the append path.
    merged = merge_delta_graphs(
        [
            {
                "nodes": [
                    {
                        "path": "people[]",
                        "ids": {"name": "Alice"},
                        "properties": {"title": ""},
                        "__property_provenance": {"title": "batch_0"},  # malformed: should be list
                    }
                ],
                "relationships": [],
            },
            {
                "nodes": [
                    {
                        "path": "people[]",
                        "ids": {"name": "Alice"},
                        "properties": {"title": "Senior Engineer"},
                    }
                ],
                "relationships": [],
            },
        ],
        dedup_policy=policy,
    )
    assert len(merged["nodes"]) == 1
    assert merged["nodes"][0]["properties"]["title"] == "Senior Engineer"
    prov = merged["nodes"][0].get("__property_provenance", {})
    assert isinstance(prov.get("title"), list)
    assert "batch_0" in prov["title"]


def test_sanitize_batch_echo_from_graph_clears_echoed_batch_labels() -> None:
    graph = {
        "nodes": [
            {
                "path": "",
                "ids": {},
                "properties": {
                    "reference_document": "Delta extraction batch 25/49",
                    "title": "Real Title",
                },
            },
            {
                "path": "offres[]",
                "ids": {"nom": "Delta extraction batch 28/49."},
                "properties": {"nom": "Delta extraction batch 28/49."},
            },
        ],
        "relationships": [],
    }
    sanitize_batch_echo_from_graph(graph)
    assert graph["nodes"][0]["properties"]["reference_document"] == ""
    assert graph["nodes"][0]["properties"]["title"] == "Real Title"
    assert graph["nodes"][1]["properties"]["nom"] == ""
    assert graph["nodes"][1]["ids"]["nom"] == ""  # ids sanitized too
