"""Tests for reattach_orphans (parent-identity and single-candidate reattachment)."""

from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.extractors.contracts.delta.catalog import (
    DeltaNodeCatalog,
    build_delta_node_catalog,
    reattach_orphans,
)


class Widget(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str


class Box(BaseModel):
    model_config = ConfigDict(graph_id_fields=["box_id"])
    box_id: str
    widgets: list[Widget] = Field(default_factory=list)


class RootWithBoxes(BaseModel):
    model_config = ConfigDict(graph_id_fields=["doc_id"])
    doc_id: str
    boxes: list[Box] = Field(default_factory=list)


def test_reattach_with_parent_ids_one_matching_candidate() -> None:
    """Orphan with parent_ids that matches exactly one parent is reattached to that parent."""
    catalog = build_delta_node_catalog(RootWithBoxes)
    merged_root: dict = {
        "doc_id": "D1",
        "boxes": [
            {"box_id": "B1", "widgets": []},
            {"box_id": "B2", "widgets": []},
        ],
        "__orphans__": [
            {
                "path": "boxes[].widgets[]",
                "parent_path": "boxes[]",
                "parent_ids": {"box_id": "B1"},
                "data": {"name": "W1"},
            },
        ],
    }
    reattach_orphans(merged_root, catalog)
    assert len(merged_root["__orphans__"]) == 0
    assert len(merged_root["boxes"][0]["widgets"]) == 1
    assert merged_root["boxes"][0]["widgets"][0]["name"] == "W1"
    assert len(merged_root["boxes"][1]["widgets"]) == 0


def test_reattach_parent_ids_zero_matches_stays_orphan() -> None:
    """Orphan with parent_ids that match no parent remains in __orphans__."""
    catalog = build_delta_node_catalog(RootWithBoxes)
    merged_root = {
        "doc_id": "D1",
        "boxes": [
            {"box_id": "B1", "widgets": []},
            {"box_id": "B2", "widgets": []},
        ],
        "__orphans__": [
            {
                "path": "boxes[].widgets[]",
                "parent_path": "boxes[]",
                "parent_ids": {"box_id": "B99"},
                "data": {"name": "W1"},
            },
        ],
    }
    reattach_orphans(merged_root, catalog)
    assert len(merged_root["__orphans__"]) == 1
    assert merged_root["__orphans__"][0]["data"]["name"] == "W1"
    assert len(merged_root["boxes"][0]["widgets"]) == 0
    assert len(merged_root["boxes"][1]["widgets"]) == 0


def test_reattach_parent_ids_multiple_matches_stays_orphan() -> None:
    """Orphan with parent_ids that match more than one parent remains in __orphans__."""
    catalog = build_delta_node_catalog(RootWithBoxes)
    merged_root = {
        "doc_id": "D1",
        "boxes": [
            {"box_id": "B1", "widgets": []},
            {"box_id": "B1", "widgets": []},
        ],
        "__orphans__": [
            {
                "path": "boxes[].widgets[]",
                "parent_path": "boxes[]",
                "parent_ids": {"box_id": "B1"},
                "data": {"name": "W1"},
            },
        ],
    }
    reattach_orphans(merged_root, catalog)
    assert len(merged_root["__orphans__"]) == 1
    assert len(merged_root["boxes"][0]["widgets"]) == 0
    assert len(merged_root["boxes"][1]["widgets"]) == 0


def test_reattach_no_parent_ids_single_candidate_reattaches() -> None:
    """Orphan without parent_ids and exactly one candidate at parent_path is reattached."""
    catalog = build_delta_node_catalog(RootWithBoxes)
    merged_root = {
        "doc_id": "D1",
        "boxes": [{"box_id": "B1", "widgets": []}],
        "__orphans__": [
            {
                "path": "boxes[].widgets[]",
                "parent_path": "boxes[]",
                "data": {"name": "W1"},
            },
        ],
    }
    reattach_orphans(merged_root, catalog)
    assert len(merged_root["__orphans__"]) == 0
    assert len(merged_root["boxes"][0]["widgets"]) == 1
    assert merged_root["boxes"][0]["widgets"][0]["name"] == "W1"


def test_reattach_no_parent_ids_two_candidates_stays_orphan() -> None:
    """Orphan without parent_ids and two candidates at parent_path stays orphan."""
    catalog = build_delta_node_catalog(RootWithBoxes)
    merged_root = {
        "doc_id": "D1",
        "boxes": [
            {"box_id": "B1", "widgets": []},
            {"box_id": "B2", "widgets": []},
        ],
        "__orphans__": [
            {
                "path": "boxes[].widgets[]",
                "parent_path": "boxes[]",
                "data": {"name": "W1"},
            },
        ],
    }
    reattach_orphans(merged_root, catalog)
    assert len(merged_root["__orphans__"]) == 1
    assert len(merged_root["boxes"][0]["widgets"]) == 0
    assert len(merged_root["boxes"][1]["widgets"]) == 0


def test_reattach_id_canonicalization_matches() -> None:
    """Orphan parent_ids and candidate id with different case but same canonical form match."""
    catalog = build_delta_node_catalog(RootWithBoxes)
    merged_root = {
        "doc_id": "D1",
        "boxes": [
            {"box_id": "B1", "widgets": []},
            {"box_id": "B2", "widgets": []},
        ],
        "__orphans__": [
            {
                "path": "boxes[].widgets[]",
                "parent_path": "boxes[]",
                "parent_ids": {"box_id": "b1"},
                "data": {"name": "W1"},
            },
        ],
    }
    reattach_orphans(merged_root, catalog)
    assert len(merged_root["__orphans__"]) == 0
    assert len(merged_root["boxes"][0]["widgets"]) == 1
    assert merged_root["boxes"][0]["widgets"][0]["name"] == "W1"
    assert len(merged_root["boxes"][1]["widgets"]) == 0


def test_reattach_defensive_key_handling_alias_maps_to_canonical() -> None:
    """Orphan parent_ids with key that is an alias for canonical id_field still matches parent."""
    catalog = build_delta_node_catalog(RootWithBoxes)
    alias_catalog = DeltaNodeCatalog(
        nodes=catalog.nodes,
        field_aliases={"Box_ID": "box_id"},
    )
    merged_root = {
        "doc_id": "D1",
        "boxes": [
            {"box_id": "B1", "widgets": []},
            {"box_id": "B2", "widgets": []},
        ],
        "__orphans__": [
            {
                "path": "boxes[].widgets[]",
                "parent_path": "boxes[]",
                "parent_ids": {"Box_ID": "B1"},
                "data": {"name": "W1"},
            },
        ],
    }
    reattach_orphans(merged_root, alias_catalog)
    assert len(merged_root["__orphans__"]) == 0
    assert len(merged_root["boxes"][0]["widgets"]) == 1
    assert merged_root["boxes"][0]["widgets"][0]["name"] == "W1"
    assert len(merged_root["boxes"][1]["widgets"]) == 0


def test_reattach_parent_ids_with_extra_alias_key_not_in_id_fields() -> None:
    """When parent_ids has a key that is an alias but canonical not in parent id_fields, it is skipped."""
    catalog = build_delta_node_catalog(RootWithBoxes)
    # Alias "box_id" -> "box_id", and a spurious alias "other" -> "other" (not in id_fields)
    alias_catalog = DeltaNodeCatalog(
        nodes=catalog.nodes,
        field_aliases={"Box_ID": "box_id", "other": "other"},
    )
    merged_root = {
        "doc_id": "D1",
        "boxes": [{"box_id": "B1", "widgets": []}],
        "__orphans__": [
            {
                "path": "boxes[].widgets[]",
                "parent_path": "boxes[]",
                "parent_ids": {"Box_ID": "B1", "other": "ignored"},
                "data": {"name": "W2"},
            },
        ],
    }
    reattach_orphans(merged_root, alias_catalog)
    assert len(merged_root["__orphans__"]) == 0
    assert len(merged_root["boxes"][0]["widgets"]) == 1
    assert merged_root["boxes"][0]["widgets"][0]["name"] == "W2"


def test_reattach_non_dict_orphan_entry_stays_in_orphans() -> None:
    """Non-dict entries in __orphans__ are left in still_orphans (defensive branch)."""
    catalog = build_delta_node_catalog(RootWithBoxes)
    merged_root = {
        "doc_id": "D1",
        "boxes": [{"box_id": "B1", "widgets": []}],
        "__orphans__": [
            None,
            "not a dict",
            {
                "path": "boxes[].widgets[]",
                "parent_path": "boxes[]",
                "parent_ids": {"box_id": "B1"},
                "data": {"name": "W1"},
            },
        ],
    }
    reattach_orphans(merged_root, catalog)
    assert len(merged_root["__orphans__"]) == 2
    assert None in merged_root["__orphans__"]
    assert "not a dict" in merged_root["__orphans__"]
    assert len(merged_root["boxes"][0]["widgets"]) == 1
    assert merged_root["boxes"][0]["widgets"][0]["name"] == "W1"
