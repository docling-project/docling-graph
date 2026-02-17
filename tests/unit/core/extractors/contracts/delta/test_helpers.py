from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.extractors.contracts.delta.catalog import build_delta_node_catalog
from docling_graph.core.extractors.contracts.delta.helpers import (
    build_dedup_policy,
    chunk_batches_by_token_limit,
    ensure_root_node,
    filter_entity_nodes_by_identity,
    flatten_node_properties,
    merge_delta_graphs,
    node_identity_key,
    per_path_counts,
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


def test_flatten_node_properties_normalize_list_skips_dict_values() -> None:
    """_normalize_list skips dict items (elif isinstance(value, dict): continue)."""
    props = {"key": [{"nested": 1}, "ok"]}
    flattened = flatten_node_properties(props)
    assert flattened["key"] == ["ok"]


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


def test_chunk_batches_by_token_limit_fallback_when_token_counts_shorter_than_chunks() -> None:
    """When token_counts has fewer elements than chunks, missing indices use max(1, len(chunk.split()))."""
    chunks = ["one", "two words", "three word chunk"]
    token_counts = [1, 2]  # shorter than chunks
    batches = chunk_batches_by_token_limit(chunks, token_counts, max_batch_tokens=10)
    assert len(batches) >= 1
    all_chunks = [chunk for batch in batches for _, chunk, _ in batch]
    assert all_chunks == chunks


def test_chunk_batches_by_token_limit_uses_token_counts_when_same_length() -> None:
    """When token_counts has same length as chunks, token_counts[idx] is used (not fallback)."""
    chunks = ["short", "medium length", "very long chunk here"]
    token_counts = [1, 2, 4]
    batches = chunk_batches_by_token_limit(chunks, token_counts, max_batch_tokens=5)
    assert len(batches) >= 1
    all_chunks = [chunk for batch in batches for _, chunk, _ in batch]
    assert all_chunks == chunks
    # First batch should use token_counts: 1+2=3 <= 5, +4 would be 7 > 5 so new batch
    assert len(batches[0]) == 2
    assert batches[0][0][2] == 1 and batches[0][1][2] == 2


def test_chunk_batches_by_token_limit_raises_when_max_zero() -> None:
    """chunk_batches_by_token_limit raises ValueError when max_batch_tokens <= 0."""
    import pytest

    with pytest.raises(ValueError, match="max_batch_tokens must be > 0"):
        chunk_batches_by_token_limit(["a"], [1], max_batch_tokens=0)


def test_filter_entity_nodes_by_identity_removes_relationships_to_dropped_nodes() -> None:
    """When a node is dropped as section title, relationships referencing it are removed."""

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
    # Node 1 kept, Node 2 dropped (section-title-like). Rel from 1 to 2 should be removed.
    graph = {
        "nodes": [
            {"path": "offres[]", "ids": {"nom": "ESSENTIELLE"}, "properties": {}},
            {
                "path": "offres[]",
                "ids": {"nom": "LA PRESCRIPTION ET LE TRAITEMENT DES RÉCLAMATIONS"},
                "properties": {},
            },
        ],
        "relationships": [
            {
                "source_path": "offres[]",
                "source_ids": {"nom": "ESSENTIELLE"},
                "target_path": "offres[]",
                "target_ids": {"nom": "LA PRESCRIPTION ET LE TRAITEMENT DES RÉCLAMATIONS"},
                "label": "REL",
            },
        ],
    }
    out, stats = filter_entity_nodes_by_identity(graph, catalog, policy, enabled=True, strict=False)
    assert len(out["nodes"]) == 1
    assert stats["identity_filter_dropped"] == 1
    assert len(out["relationships"]) == 0


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


def test_catalog_populates_identity_example_values_from_child_model_scalar_examples() -> None:
    """Catalog should set identity_example_values from child model's ID field scalar examples."""

    class Study(BaseModel):
        model_config = ConfigDict(graph_id_fields=["study_id"])
        study_id: str = Field(
            ...,
            examples=["3.1", "STUDY-BINDER-MW", "STUDY-SECTION-3.1"],
        )
        objective: str = ""

    class Doc(BaseModel):
        studies: list[Study] = Field(default_factory=list, description="Studies")

    catalog = build_delta_node_catalog(Doc)
    studies_spec = next((n for n in catalog.nodes if n.path == "studies[]"), None)
    assert studies_spec is not None
    vals = getattr(studies_spec, "identity_example_values", None)
    assert vals is not None
    assert "3.1" in vals
    assert "STUDY-BINDER-MW" in vals
    assert "STUDY-SECTION-3.1" in vals


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
    """Identity filter only drops when value looks like a section title; allowlist is not used for dropping."""

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
    # Neither value looks like a section title, so both are kept (allowlist is not used for dropping).
    assert len(out["nodes"]) == 2
    assert stats["identity_filter_dropped"] == 0


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
    """Only section-title heuristic applies; values matching section patterns are dropped (strict and non-strict same)."""

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
    # "EXCLUSIONS COMMUNES" matches section-title pattern; ESSENTIELLE and Option Dépannage kept (Option Dépannage may not match after NFKD).
    assert len(out_strict["nodes"]) >= 1
    assert out_strict["nodes"][0]["ids"]["nom"] == "ESSENTIELLE"
    assert stats_strict["identity_filter_dropped"] >= 1
    out_heuristic, stats_heuristic = filter_entity_nodes_by_identity(
        graph, catalog, policy, enabled=True, strict=False
    )
    assert out_heuristic["nodes"][0]["ids"]["nom"] == "ESSENTIELLE"
    assert stats_heuristic["identity_filter_dropped"] >= 1


def test_filter_entity_nodes_by_identity_coerces_list_nom_to_string_for_allowlist() -> None:
    """When LLM returns nom as a list, first string element is used for section-title check; allowlist is not used for dropping."""

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
    # Neither "ESSENTIELLE" nor "Other" looks like a section title, so both nodes are kept.
    assert len(out["nodes"]) == 2
    assert stats["identity_filter_dropped"] == 0


def test_filter_entity_nodes_by_identity_coerces_list_with_dict_item_via_nom() -> None:
    """When primary identity is a list containing a dict, _coerce_identity_to_str extracts from dict (nom or first value)."""

    class Offer(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str

    class Doc(BaseModel):
        offres: list[Offer] = Field(default_factory=list)

    catalog = build_delta_node_catalog(Doc)
    policy = build_dedup_policy(catalog)
    graph = {
        "nodes": [
            {"path": "offres[]", "ids": {"nom": [{"nom": "ExtractedName"}]}, "properties": {}},
        ],
        "relationships": [],
    }
    out, stats = filter_entity_nodes_by_identity(graph, catalog, policy, enabled=True, strict=False)
    assert len(out["nodes"]) == 1
    assert stats["identity_filter_dropped"] == 0


def test_filter_entity_nodes_by_identity_keeps_non_dict_relationships_when_dropping_nodes() -> None:
    """When relationships list contains non-dict entries (e.g. None), they are kept in kept_rels."""

    class Offer(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str

    class Doc(BaseModel):
        offres: list[Offer] = Field(default_factory=list)

    catalog = build_delta_node_catalog(Doc)
    policy = build_dedup_policy(catalog)
    graph = {
        "nodes": [
            {"path": "offres[]", "ids": {"nom": "ESSENTIELLE"}, "properties": {}},
            {
                "path": "offres[]",
                "ids": {"nom": "SECTION TITLE WITH ENOUGH CAPS TO TRIGGER DROP"},
                "properties": {},
            },
        ],
        "relationships": [
            {
                "source_path": "offres[]",
                "source_ids": {"nom": "ESSENTIELLE"},
                "target_path": "offres[]",
                "target_ids": {"nom": "SECTION TITLE WITH ENOUGH CAPS TO TRIGGER DROP"},
                "properties": {},
            },
            None,
            "not a dict",
        ],
    }
    out, stats = filter_entity_nodes_by_identity(graph, catalog, policy, enabled=True, strict=False)
    assert stats["identity_filter_dropped"] == 1
    # Valid dict rel references dropped node so removed; non-dict entries are kept
    assert len(out["relationships"]) == 2
    assert None in out["relationships"]
    assert "not a dict" in out["relationships"]


def test_ensure_root_node_adds_root_when_missing_but_has_root_children() -> None:
    """When graph has root-level children but no root node, ensure_root_node adds one so quality gate passes."""
    merged_graph = {
        "nodes": [
            {
                "path": "authors[]",
                "ids": {"full_name": "Alice"},
                "parent": {"path": "", "ids": {}},
                "properties": {},
            },
        ],
        "relationships": [],
    }
    assert per_path_counts(merged_graph["nodes"]).get("", 0) == 0
    ensure_root_node(merged_graph)
    counts = per_path_counts(merged_graph["nodes"])
    assert counts.get("", 1) == 1
    root_nodes = [
        n for n in merged_graph["nodes"] if isinstance(n, dict) and str(n.get("path") or "") == ""
    ]
    assert len(root_nodes) == 1
    assert root_nodes[0]["path"] == ""
    assert root_nodes[0]["ids"] == {}
    assert root_nodes[0]["parent"] is None


def test_ensure_root_node_does_nothing_when_root_exists() -> None:
    """When graph already has a root node, ensure_root_node does not add another."""
    merged_graph = {
        "nodes": [
            {"path": "", "ids": {"document_number": "DOC-1"}, "parent": None, "properties": {}},
            {"path": "authors[]", "ids": {}, "parent": {"path": "", "ids": {}}, "properties": {}},
        ],
        "relationships": [],
    }
    ensure_root_node(merged_graph)
    assert per_path_counts(merged_graph["nodes"]).get("", 0) == 1


def test_ensure_root_node_does_nothing_when_no_root_children() -> None:
    """When graph has no root-level children, ensure_root_node does not add a root."""
    merged_graph = {
        "nodes": [
            {
                "path": "nested[]",
                "ids": {},
                "parent": {"path": "other", "ids": {}},
                "properties": {},
            },
        ],
        "relationships": [],
    }
    ensure_root_node(merged_graph)
    assert per_path_counts(merged_graph["nodes"]).get("", 0) == 0
