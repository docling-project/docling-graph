from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.extractors.contracts.delta.catalog import build_delta_node_catalog
from docling_graph.core.extractors.contracts.delta.helpers import (
    build_dedup_policy,
    merge_delta_graphs,
)
from docling_graph.core.extractors.contracts.delta.ir_normalizer import (
    DeltaIrNormalizerConfig,
    normalize_delta_ir_batch_results,
)
from docling_graph.core.extractors.contracts.delta.schema_mapper import (
    project_graph_to_template_root,
)


class Item(BaseModel):
    model_config = ConfigDict(graph_id_fields=["item_code"])
    item_code: str
    name: str | None = None


class LineItem(BaseModel):
    model_config = ConfigDict(graph_id_fields=["line_number"])
    line_number: str
    item: Item | None = None


class Invoice(BaseModel):
    model_config = ConfigDict(graph_id_fields=["document_number"])
    document_number: str
    line_items: list[LineItem] = Field(default_factory=list)


class SubItem(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str


class ItemWithSubitems(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str
    subitems: list[SubItem] = Field(default_factory=list)


class RootWithItems(BaseModel):
    model_config = ConfigDict(graph_id_fields=["doc_id"])
    doc_id: str
    items: list[ItemWithSubitems] = Field(default_factory=list)


def test_projection_attaches_same_list_item_identity_under_different_parents_to_each_parent() -> (
    None
):
    """List-item nodes with same path+ids but different parents appear under each parent after projection."""
    merged_graph = {
        "nodes": [
            {
                "path": "",
                "ids": {"doc_id": "D1"},
                "properties": {"doc_id": "D1"},
            },
            {
                "path": "items[]",
                "ids": {"name": "P1"},
                "parent": {"path": "", "ids": {}},
                "properties": {"name": "P1"},
            },
            {
                "path": "items[]",
                "ids": {"name": "P2"},
                "parent": {"path": "", "ids": {}},
                "properties": {"name": "P2"},
            },
            {
                "path": "items[].subitems[]",
                "ids": {"name": "A"},
                "parent": {"path": "items[]", "ids": {"name": "P1"}},
                "properties": {"name": "A"},
            },
            {
                "path": "items[].subitems[]",
                "ids": {"name": "A"},
                "parent": {"path": "items[]", "ids": {"name": "P2"}},
                "properties": {"name": "A"},
            },
        ],
        "relationships": [],
    }
    merged_root, merge_stats = project_graph_to_template_root(merged_graph, RootWithItems)
    assert merge_stats.get("parent_lookup_miss", 0) == 0
    assert len(merged_root["items"]) == 2
    assert merged_root["items"][0]["name"] == "P1"
    assert merged_root["items"][1]["name"] == "P2"
    assert len(merged_root["items"][0]["subitems"]) == 1
    assert len(merged_root["items"][1]["subitems"]) == 1
    assert merged_root["items"][0]["subitems"][0]["name"] == "A"
    assert merged_root["items"][1]["subitems"][0]["name"] == "A"


def test_projection_links_children_without_parent_lookup_miss_after_inference() -> None:
    catalog = build_delta_node_catalog(Invoice)
    policy = build_dedup_policy(catalog)

    batch_results = [
        {
            "nodes": [
                {
                    "path": "document",
                    "ids": {"document_number": "INV-42"},
                    "properties": {"document_number": "INV-42"},
                },
                {
                    "path": "document.line_items.1",
                    "ids": {"line_number": "1"},
                    "parent": {"path": "document", "ids": {"document_number": "INV-42"}},
                    "properties": {"line_number": "1"},
                },
                {
                    "path": "document.line_items.1.item",
                    "ids": {"item_code": "SKU-1"},
                    "parent": {"path": "document.line_items", "ids": {}},
                    "properties": {"item_code": "SKU-1", "name": "Keyboard"},
                },
            ],
            "relationships": [],
        }
    ]

    normalized, _stats = normalize_delta_ir_batch_results(
        batch_results=batch_results,
        batch_plan=[[(0, "chunk", 10)]],
        chunk_metadata=[{"page_numbers": [1], "token_count": 10}],
        catalog=catalog,
        dedup_policy=policy,
        config=DeltaIrNormalizerConfig(),
    )
    merged_graph = merge_delta_graphs(normalized, dedup_policy=policy)
    merged_root, merge_stats = project_graph_to_template_root(merged_graph, Invoice)

    assert merge_stats.get("parent_lookup_miss", 0) == 0
    assert merged_root["line_items"][0]["item"]["item_code"] == "SKU-1"


def test_projection_salvages_orphans_instead_of_dropping() -> None:
    merged_graph = {
        "nodes": [
            {
                "path": "",
                "ids": {"document_number": "INV-99"},
                "properties": {"document_number": "INV-99"},
            },
            {
                "path": "line_items[].item",
                "ids": {"item_code": "SKU-X"},
                "parent": {"path": "line_items[]", "ids": {"line_number": "404"}},
                "properties": {"item_code": "SKU-X", "name": "Standalone"},
            },
        ],
        "relationships": [],
    }
    merged_root, merge_stats = project_graph_to_template_root(merged_graph, Invoice)

    assert merge_stats.get("parent_lookup_miss", 0) == 1
    assert merge_stats.get("orphan_attached", 0) == 1
    assert merged_root.get("__orphans__")


def test_projection_injects_entity_ids_when_properties_empty() -> None:
    merged_graph = {
        "nodes": [
            {
                "path": "",
                "ids": {"document_number": "INV-100"},
                "properties": {"document_number": "INV-100"},
            },
            {
                "path": "line_items[]",
                "ids": {"line_number": "1"},
                "parent": {"path": "", "ids": {}},
                "properties": {},
            },
            {
                "path": "line_items[].item",
                "ids": {"item_code": "SKU-100"},
                "parent": {"path": "line_items[]", "ids": {"line_number": "1"}},
                "properties": {},
            },
        ],
        "relationships": [],
    }
    merged_root, _merge_stats = project_graph_to_template_root(merged_graph, Invoice)
    assert merged_root["line_items"][0]["line_number"] == "1"
    assert merged_root["line_items"][0]["item"]["item_code"] == "SKU-100"


def test_projection_repairs_local_id_parent_lookup_when_off_by_one() -> None:
    merged_graph = {
        "nodes": [
            {
                "path": "",
                "ids": {"document_number": "INV-101"},
                "properties": {"document_number": "INV-101"},
            },
            {
                "path": "line_items[]",
                "ids": {"line_number": "1"},
                "parent": {"path": "", "ids": {}},
                "properties": {"line_number": "1"},
            },
            {
                "path": "line_items[].item",
                "ids": {"item_code": "SKU-101"},
                "parent": {"path": "line_items[]", "ids": {"line_number": "0"}},
                "properties": {"item_code": "SKU-101"},
            },
        ],
        "relationships": [],
    }
    merged_root, merge_stats = project_graph_to_template_root(merged_graph, Invoice)
    assert merge_stats.get("parent_lookup_repaired_local_id", 0) == 1
    assert merge_stats.get("parent_lookup_miss", 0) == 0
    assert merged_root["line_items"][0]["item"]["item_code"] == "SKU-101"


def test_projection_repairs_parent_lookup_by_position_when_parent_ids_missing() -> None:
    merged_graph = {
        "nodes": [
            {
                "path": "",
                "ids": {"document_number": "INV-102"},
                "properties": {"document_number": "INV-102"},
            },
            {
                "path": "line_items[]",
                "ids": {"line_number": "1"},
                "parent": {"path": "", "ids": {}},
                "properties": {"line_number": "1"},
            },
            {
                "path": "line_items[]",
                "ids": {"line_number": "2"},
                "parent": {"path": "", "ids": {}},
                "properties": {"line_number": "2"},
            },
            {
                "path": "line_items[].item",
                "ids": {"item_code": "SKU-POS-1"},
                "parent": {"path": "line_items[]", "ids": {}},
                "properties": {"item_code": "SKU-POS-1"},
            },
            {
                "path": "line_items[].item",
                "ids": {"item_code": "SKU-POS-2"},
                "parent": {"path": "line_items[]", "ids": {}},
                "properties": {"item_code": "SKU-POS-2"},
            },
        ],
        "relationships": [],
    }
    merged_root, merge_stats = project_graph_to_template_root(merged_graph, Invoice)
    assert merge_stats.get("parent_lookup_repaired_positional", 0) >= 2
    assert merged_root["line_items"][0]["item"]["item_code"] == "SKU-POS-1"
    assert merged_root["line_items"][1]["item"]["item_code"] == "SKU-POS-2"


def test_projection_repairs_parent_lookup_by_canonical_id() -> None:
    merged_graph = {
        "nodes": [
            {
                "path": "",
                "ids": {"document_number": "INV-103"},
                "properties": {"document_number": "INV-103"},
            },
            {
                "path": "line_items[]",
                "ids": {"line_number": "LIGNE-A"},
                "parent": {"path": "", "ids": {}},
                "properties": {"line_number": "LIGNE-A"},
            },
            {
                "path": "line_items[]",
                "ids": {"line_number": "LIGNE-B"},
                "parent": {"path": "", "ids": {}},
                "properties": {"line_number": "LIGNE-B"},
            },
            {
                "path": "line_items[].item",
                "ids": {"item_code": "SKU-CANON"},
                "parent": {"path": "line_items[]", "ids": {"line_number": "ligne-a"}},
                "properties": {"item_code": "SKU-CANON"},
            },
        ],
        "relationships": [],
    }
    _merged_root, merge_stats = project_graph_to_template_root(merged_graph, Invoice)
    assert merge_stats.get("parent_lookup_repaired_canonical_id", 0) == 1
    assert merge_stats.get("parent_lookup_miss", 0) == 0
