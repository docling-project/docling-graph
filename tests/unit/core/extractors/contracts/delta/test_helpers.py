from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.extractors.contracts.delta.catalog import build_delta_node_catalog
from docling_graph.core.extractors.contracts.delta.helpers import (
    build_dedup_policy,
    filter_entity_nodes_by_identity,
    flatten_node_properties,
    merge_delta_graphs,
    node_identity_key,
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


class SubItem(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str


class Item(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str
    subitems: list[SubItem] = Field(default_factory=list)


class RootWithItems(BaseModel):
    model_config = ConfigDict(graph_id_fields=["doc_id"])
    doc_id: str
    items: list[Item] = Field(default_factory=list)


def test_node_identity_key_list_path_different_parents_different_keys() -> None:
    """List-item nodes (path ending with []) with same ids but different parents get different keys."""
    catalog = build_delta_node_catalog(RootWithItems)
    policy = build_dedup_policy(catalog)
    list_path = "items[].subitems[]"
    node_p1 = {
        "path": list_path,
        "ids": {"name": "A"},
        "parent": {"path": "items[]", "ids": {"name": "P1"}},
        "properties": {},
    }
    node_p2 = {
        "path": list_path,
        "ids": {"name": "A"},
        "parent": {"path": "items[]", "ids": {"name": "P2"}},
        "properties": {},
    }
    key1 = node_identity_key(node_p1, dedup_policy=policy)
    key2 = node_identity_key(node_p2, dedup_policy=policy)
    assert key1 != key2
    assert key1[0] == list_path and key2[0] == list_path


def test_node_identity_key_non_list_path_same_ids_different_parents_same_key() -> None:
    """Non-list path nodes with same ids but different parents get the same key (no parent scoping)."""
    catalog = build_delta_node_catalog(RootWithItems)
    policy = build_dedup_policy(catalog)
    # Use a path that does not end with [] (e.g. root or a scalar entity path if catalog had one).
    # Root path "" with same ids -> one key; we need "different parents" - for root, parent is absent.
    # So use items[] (list path) but we want "same key when path does not end with []".
    # Catalog has "", "items[]", "items[].subitems[]". So non-list path is "".
    # Root node: path "", ids {"doc_id": "D1"}, no parent. Another root with same ids -> same key.
    # For "different parents" with non-list path we need a path that doesn't end with [].
    # Invent a policy entry for a fake scalar path "item" so we can test.
    policy_with_fake = {**policy, "item": policy["items[]"]}
    non_list_path = "item"
    node_p1 = {
        "path": non_list_path,
        "ids": {"name": "A"},
        "parent": {"path": "items[]", "ids": {"name": "P1"}},
        "properties": {},
    }
    node_p2 = {
        "path": non_list_path,
        "ids": {"name": "A"},
        "parent": {"path": "items[]", "ids": {"name": "P2"}},
        "properties": {},
    }
    key1 = node_identity_key(node_p1, dedup_policy=policy_with_fake)
    key2 = node_identity_key(node_p2, dedup_policy=policy_with_fake)
    assert key1 == key2


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


def test_merge_delta_graphs_keeps_list_items_with_same_ids_under_different_parents_separate() -> (
    None
):
    """List-item nodes (path ending []) with same ids but different parent identities are not merged."""
    catalog = build_delta_node_catalog(RootWithItems)
    policy = build_dedup_policy(catalog)
    merged = merge_delta_graphs(
        [
            {
                "nodes": [
                    {
                        "path": "items[].subitems[]",
                        "ids": {"name": "A"},
                        "parent": {"path": "items[]", "ids": {"name": "P1"}},
                        "properties": {"title": "first"},
                    }
                ],
                "relationships": [],
            },
            {
                "nodes": [
                    {
                        "path": "items[].subitems[]",
                        "ids": {"name": "A"},
                        "parent": {"path": "items[]", "ids": {"name": "P2"}},
                        "properties": {"title": "second"},
                    }
                ],
                "relationships": [],
            },
        ],
        dedup_policy=policy,
    )
    assert len(merged["nodes"]) == 2


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


def test_merge_delta_graphs_canonicalizes_identity_and_acronym_keys() -> None:
    class Offer(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str

    class OfferDoc(BaseModel):
        model_config = ConfigDict(graph_id_fields=["reference_document"])
        reference_document: str
        offres: list[Offer] = Field(default_factory=list)

    catalog = build_delta_node_catalog(OfferDoc)
    policy = build_dedup_policy(catalog)
    merged = merge_delta_graphs(
        [
            {
                "nodes": [
                    {
                        "path": "offres[]",
                        "ids": {"nom": "PROPRIÉTAIRE NON OCCUPANT"},
                        "properties": {"nom": "PROPRIÉTAIRE NON OCCUPANT"},
                    }
                ],
                "relationships": [],
            },
            {
                "nodes": [
                    {
                        "path": "offres[]",
                        "ids": {"nom": "PNO"},
                        "properties": {"nom": "PNO"},
                    }
                ],
                "relationships": [],
            },
        ],
        dedup_policy=policy,
    )
    assert len(merged["nodes"]) == 1


def test_merge_delta_graphs_dedups_relationships_with_canonicalized_endpoint_ids() -> None:
    class Offer(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str

    class OfferDoc(BaseModel):
        model_config = ConfigDict(graph_id_fields=["reference_document"])
        reference_document: str
        offres: list[Offer] = Field(default_factory=list)

    catalog = build_delta_node_catalog(OfferDoc)
    policy = build_dedup_policy(catalog)
    merged = merge_delta_graphs(
        [
            {
                "nodes": [],
                "relationships": [
                    {
                        "edge_label": "AOFFRE",
                        "source_path": "",
                        "source_ids": {"reference_document": "CGV-MRH-2023"},
                        "target_path": "offres[]",
                        "target_ids": {"nom": "PROPRIÉTAIRE NON OCCUPANT"},
                        "properties": {},
                    }
                ],
            },
            {
                "nodes": [],
                "relationships": [
                    {
                        "edge_label": "AOFFRE",
                        "source_path": "",
                        "source_ids": {"reference_document": "CGV-MRH-2023"},
                        "target_path": "offres[]",
                        "target_ids": {"nom": "PNO"},
                        "properties": {},
                    }
                ],
            },
        ],
        dedup_policy=policy,
    )
    assert len(merged["relationships"]) == 1


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


def test_catalog_populates_identity_example_values_for_list_entity() -> None:
    """Catalog should set identity_example_values from Field examples for list-entity paths."""

    class Offer(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str = Field(..., examples=["ESSENTIELLE", "CONFORT"])

    class Doc(BaseModel):
        offres: list[Offer] = Field(
            default_factory=list,
            description="Offers",
            examples=[[{"nom": "ESSENTIELLE"}, {"nom": "CONFORT"}, {"nom": "CONFORT PLUS"}]],
        )

    catalog = build_delta_node_catalog(Doc)
    offres_spec = next((n for n in catalog.nodes if n.path == "offres[]"), None)
    assert offres_spec is not None
    assert getattr(offres_spec, "identity_example_values", None) is not None
    vals = offres_spec.identity_example_values
    assert "ESSENTIELLE" in vals
    assert "CONFORT" in vals
    assert "CONFORT PLUS" in vals


def test_filter_entity_nodes_by_identity_allows_allowlist_value() -> None:
    """Nodes whose identity is in the schema allowlist are kept."""

    class Offer(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str

    class Doc(BaseModel):
        offres: list[Offer] = Field(
            default_factory=list,
            examples=[[{"nom": "ESSENTIELLE"}, {"nom": "CONFORT"}]],
        )

    catalog = build_delta_node_catalog(Doc)
    policy = build_dedup_policy(catalog)
    graph = {
        "nodes": [
            {
                "path": "offres[]",
                "ids": {"nom": "ESSENTIELLE"},
                "properties": {"nom": "ESSENTIELLE"},
            },
            {"path": "offres[]", "ids": {"nom": "CONFORT"}, "properties": {"nom": "CONFORT"}},
        ],
        "relationships": [],
    }
    out, stats = filter_entity_nodes_by_identity(graph, catalog, policy, enabled=True, strict=False)
    assert len(out["nodes"]) == 2
    assert stats["identity_filter_dropped"] == 0


def test_filter_entity_nodes_by_identity_drops_section_title_when_not_strict() -> None:
    """When not strict, nodes with section-title-like identity are dropped."""

    class Offer(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str

    class Doc(BaseModel):
        offres: list[Offer] = Field(
            default_factory=list,
            examples=[[{"nom": "ESSENTIELLE"}, {"nom": "CONFORT"}]],
        )

    catalog = build_delta_node_catalog(Doc)
    policy = build_dedup_policy(catalog)
    graph = {
        "nodes": [
            {
                "path": "offres[]",
                "ids": {"nom": "ESSENTIELLE"},
                "properties": {"nom": "ESSENTIELLE"},
            },
            {
                "path": "offres[]",
                "ids": {"nom": "LA PRESCRIPTION"},
                "properties": {"nom": "LA PRESCRIPTION"},
            },
            {
                "path": "offres[]",
                "ids": {"nom": "LE TRAITEMENT DE VOS RÉCLAMATIONS"},
                "properties": {},
            },
        ],
        "relationships": [],
    }
    out, stats = filter_entity_nodes_by_identity(graph, catalog, policy, enabled=True, strict=False)
    assert len(out["nodes"]) == 1
    assert out["nodes"][0]["ids"]["nom"] == "ESSENTIELLE"
    assert stats["identity_filter_dropped"] == 2
    assert stats["identity_filter_dropped_by_path"].get("offres[]", 0) == 2


def test_filter_entity_nodes_by_identity_strict_drops_non_allowlist() -> None:
    """When strict=True, any identity not in allowlist is dropped."""

    class Offer(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str

    class Doc(BaseModel):
        offres: list[Offer] = Field(
            default_factory=list,
            examples=[[{"nom": "ESSENTIELLE"}, {"nom": "CONFORT"}]],
        )

    catalog = build_delta_node_catalog(Doc)
    policy = build_dedup_policy(catalog)
    graph = {
        "nodes": [
            {"path": "offres[]", "ids": {"nom": "ESSENTIELLE"}, "properties": {}},
            {"path": "offres[]", "ids": {"nom": "OtherFormula"}, "properties": {}},
        ],
        "relationships": [],
    }
    out, stats = filter_entity_nodes_by_identity(graph, catalog, policy, enabled=True, strict=True)
    assert len(out["nodes"]) == 1
    assert out["nodes"][0]["ids"]["nom"] == "ESSENTIELLE"
    assert stats["identity_filter_dropped"] == 1


def test_filter_entity_nodes_by_identity_disabled_keeps_all_nodes() -> None:
    """When enabled=False, no nodes are dropped."""

    class Offer(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str

    class Doc(BaseModel):
        offres: list[Offer] = Field(
            default_factory=list,
            examples=[[{"nom": "ESSENTIELLE"}]],
        )

    catalog = build_delta_node_catalog(Doc)
    policy = build_dedup_policy(catalog)
    graph = {
        "nodes": [
            {"path": "offres[]", "ids": {"nom": "LA PRESCRIPTION"}, "properties": {}},
        ],
        "relationships": [],
    }
    out, stats = filter_entity_nodes_by_identity(
        graph, catalog, policy, enabled=False, strict=False
    )
    assert len(out["nodes"]) == 1
    assert stats["identity_filter_dropped"] == 0


def test_filter_entity_nodes_by_identity_strict_drops_non_allowlist_only_when_strict_true() -> None:
    """When strict=True, non-allowlist nodes are dropped; when strict=False, only section-title heuristic applies."""

    class Offer(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str

    class Doc(BaseModel):
        offres: list[Offer] = Field(
            default_factory=list,
            examples=[[{"nom": "ESSENTIELLE"}, {"nom": "CONFORT"}]],
        )

    catalog = build_delta_node_catalog(Doc)
    policy = build_dedup_policy(catalog)
    graph = {
        "nodes": [
            {"path": "offres[]", "ids": {"nom": "ESSENTIELLE"}, "properties": {}},
            {"path": "offres[]", "ids": {"nom": "Option Dépannage"}, "properties": {}},
            {"path": "offres[]", "ids": {"nom": "EXCLUSIONS COMMUNES"}, "properties": {}},
        ],
        "relationships": [],
    }
    out_strict, stats_strict = filter_entity_nodes_by_identity(
        graph, catalog, policy, enabled=True, strict=True
    )
    assert len(out_strict["nodes"]) == 1
    assert out_strict["nodes"][0]["ids"]["nom"] == "ESSENTIELLE"
    assert stats_strict["identity_filter_dropped"] == 2
    out_heuristic, stats_heuristic = filter_entity_nodes_by_identity(
        graph, catalog, policy, enabled=True, strict=False
    )
    assert out_heuristic["nodes"][0]["ids"]["nom"] == "ESSENTIELLE"
    assert stats_heuristic["identity_filter_dropped"] >= 1


def test_filter_entity_nodes_by_identity_coerces_list_nom_to_string_for_allowlist() -> None:
    """When LLM returns nom as a list, first string element is used for allowlist check; with strict=True non-allowlist is dropped."""

    class Offer(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str

    class Doc(BaseModel):
        offres: list[Offer] = Field(
            default_factory=list,
            examples=[[{"nom": "ESSENTIELLE"}, {"nom": "CONFORT"}]],
        )

    catalog = build_delta_node_catalog(Doc)
    policy = build_dedup_policy(catalog)
    graph = {
        "nodes": [
            {"path": "offres[]", "ids": {}, "properties": {"nom": ["ESSENTIELLE"]}},
            {"path": "offres[]", "ids": {"nom": ["Other"]}, "properties": {}},
        ],
        "relationships": [],
    }
    out, stats = filter_entity_nodes_by_identity(graph, catalog, policy, enabled=True, strict=True)
    assert len(out["nodes"]) == 1
    assert out["nodes"][0]["properties"]["nom"] == ["ESSENTIELLE"]
    assert stats["identity_filter_dropped"] == 1
