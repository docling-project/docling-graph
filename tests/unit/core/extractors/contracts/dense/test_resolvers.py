"""Unit tests for the fuzzy dedupe resolver (aggressive mode) and the
deterministic containment merge (standard mode)."""

import re
import unicodedata
from typing import Any

from docling_graph.core.extractors.contracts.dense.resolvers import (
    merge_contained_skeleton_nodes,
    resolve_skeleton_nodes,
)


def _key(node: dict[str, Any]) -> tuple[Any, ...]:
    ids = node.get("ids") or {}
    return (node.get("path"), tuple(sorted((k, str(v).lower()) for k, v in ids.items())))


def _canon_key(node: dict[str, Any]) -> tuple[Any, ...]:
    """Mirror identity_pairs canonicalization (casefold + alphanumeric only)."""

    def canon(value: Any) -> str:
        text = unicodedata.normalize("NFKD", str(value)).casefold()
        return re.sub(r"[^a-z0-9]", "", text)

    ids = node.get("ids") or {}
    return (node.get("path"), tuple(sorted((k, canon(v)) for k, v in ids.items())))


def test_ocr_accent_noise_merges_and_parents_remap() -> None:
    nodes = [
        {"path": "items[]", "ids": {"name": "Responsabilité Civile"}, "parent": None},
        {"path": "items[]", "ids": {"name": "Responsabilite Civile"}, "parent": None},
        {
            "path": "items[].parts[]",
            "ids": {"part": "p1"},
            "parent": {"path": "items[]", "ids": {"name": "Responsabilite Civile"}},
        },
    ]
    kept, stats = resolve_skeleton_nodes(nodes, _key)
    assert stats["merged_count"] == 1
    items = [n for n in kept if n["path"] == "items[]"]
    assert len(items) == 1
    assert items[0]["ids"] == {"name": "Responsabilité Civile"}
    child = next(n for n in kept if n["path"] == "items[].parts[]")
    assert child["parent"]["ids"] == {"name": "Responsabilité Civile"}


def test_numeric_parameter_variants_never_merge() -> None:
    """Digit runs are precise discriminators: near-identical strings that differ
    numerically denote distinct entities and must survive aggressive dedupe."""
    nodes = [
        {"path": "batches[]", "ids": {"batch_id": "LFP_20vol_5wtPVDF"}, "parent": None},
        {"path": "batches[]", "ids": {"batch_id": "LFP_30vol_5wtPVDF"}, "parent": None},
        {"path": "sections[]", "ids": {"title": "Article 5"}, "parent": None},
        {"path": "sections[]", "ids": {"title": "Article 6"}, "parent": None},
    ]
    kept, stats = resolve_skeleton_nodes(nodes, _key)
    assert stats["merged_count"] == 0
    assert len(kept) == 4


def test_exact_key_duplicates_are_left_to_exact_dedup() -> None:
    """Case-only variants share a canonical key; the fuzzy pass skips them."""
    nodes = [
        {"path": "items[]", "ids": {"name": "Vol et Vandalisme"}, "parent": None},
        {"path": "items[]", "ids": {"name": "Vol et vandalisme"}, "parent": None},
    ]
    kept, stats = resolve_skeleton_nodes(nodes, _key)
    assert stats["merged_count"] == 0
    assert len(kept) == 2


def test_different_paths_never_merge() -> None:
    nodes = [
        {"path": "authors[]", "ids": {"name": "J. Smith"}, "parent": None},
        {"path": "reviewers[]", "ids": {"name": "J. Smith"}, "parent": None},
    ]
    kept, stats = resolve_skeleton_nodes(nodes, _key)
    assert stats["merged_count"] == 0
    assert len(kept) == 2


def test_containment_merges_qualifier_variants_keeping_base() -> None:
    """Q2: a descriptive superset id merges into its canonical base (shorter kept)."""
    nodes = [
        {"path": "components[]", "ids": {"material_name": "LiFePO4 (LFP)"}, "parent": None},
        {
            "path": "components[]",
            "ids": {"material_name": "nanoscaled LiFePO4 (LFP)"},
            "parent": None,
        },
        {"path": "components[]", "ids": {"material_name": "PVDF"}, "parent": None},
        {
            "path": "components[]",
            "ids": {"material_name": "Polyvinylidene difluoride (PVDF)"},
            "parent": None,
        },
    ]
    kept, stats = merge_contained_skeleton_nodes(nodes, _canon_key)
    assert stats["merged_count"] == 2
    assert sorted(n["ids"]["material_name"] for n in kept) == ["LiFePO4 (LFP)", "PVDF"]


def test_containment_remaps_child_parent_to_base() -> None:
    nodes = [
        {"path": "components[]", "ids": {"material_name": "LiFePO4"}, "parent": None},
        {"path": "components[]", "ids": {"material_name": "nanoscaled LiFePO4"}, "parent": None},
        {
            "path": "components[].measurements[]",
            "ids": {"m": "d50"},
            "parent": {"path": "components[]", "ids": {"material_name": "nanoscaled LiFePO4"}},
        },
    ]
    kept, stats = merge_contained_skeleton_nodes(nodes, _canon_key)
    assert stats["merged_count"] == 1
    child = next(n for n in kept if n["path"] == "components[].measurements[]")
    assert child["parent"]["ids"] == {"material_name": "LiFePO4"}


def test_containment_respects_digit_signature() -> None:
    """A superset that differs numerically is a different entity, never merged."""
    nodes = [
        {"path": "sections[]", "ids": {"title": "Article 5"}, "parent": None},
        {"path": "sections[]", "ids": {"title": "Article 50"}, "parent": None},
    ]
    kept, stats = merge_contained_skeleton_nodes(nodes, _canon_key)
    assert stats["merged_count"] == 0
    assert len(kept) == 2


def test_containment_skips_ambiguous_multi_base_superset() -> None:
    """A superset containing two distinct shorter ids is ambiguous -> left alone."""
    nodes = [
        {"path": "items[]", "ids": {"name": "alpha"}, "parent": None},
        {"path": "items[]", "ids": {"name": "beta"}, "parent": None},
        {"path": "items[]", "ids": {"name": "alphabeta"}, "parent": None},
    ]
    kept, stats = merge_contained_skeleton_nodes(nodes, _canon_key)
    assert stats["merged_count"] == 0
    assert len(kept) == 3


def test_containment_skips_nodes_without_ids() -> None:
    """A node with no usable ids has empty canonical text and is never a superset."""
    nodes = [
        {"path": "items[]", "ids": {}, "parent": None},
        {"path": "items[]", "ids": {"name": "widget"}, "parent": None},
    ]
    kept, stats = merge_contained_skeleton_nodes(nodes, _canon_key)
    assert stats["merged_count"] == 0
    assert len(kept) == 2


def test_containment_key_fn_with_unkeyable_marker_is_never_a_containment_candidate() -> None:
    """A positional-fallback key (the orchestrator's "__key" marker for id-less
    siblings) must never be treated as containment text, even if it happens to
    look like a substring of another node's id."""

    def key_with_unkeyable(node: dict) -> tuple:
        ids = node.get("ids") or {}
        if not ids:
            return (node.get("path"), (("__key", str(id(node))),))
        return _canon_key(node)

    nodes = [
        {"path": "items[]", "ids": {}, "parent": None},
        {"path": "items[]", "ids": {"name": "widget assembly"}, "parent": None},
    ]
    kept, stats = merge_contained_skeleton_nodes(nodes, key_with_unkeyable)
    assert stats["merged_count"] == 0
    assert len(kept) == 2


def test_containment_key_fn_returning_malformed_key_is_treated_as_unkeyable() -> None:
    """A key_fn that doesn't return a (path, pairs) tuple never crashes the merge."""

    def bad_key_fn(node: dict) -> str:
        return "not-a-tuple"

    nodes = [
        {"path": "items[]", "ids": {"name": "widget"}, "parent": None},
        {"path": "items[]", "ids": {"name": "widget assembly"}, "parent": None},
    ]
    kept, stats = merge_contained_skeleton_nodes(nodes, bad_key_fn)
    assert stats["merged_count"] == 0
    assert len(kept) == 2


def test_containment_merge_unions_existing_source_bookkeeping() -> None:
    """Absorbing a superset's provenance bookkeeping into the kept base node."""
    nodes = [
        {
            "path": "components[]",
            "ids": {"material_name": "LiFePO4"},
            "parent": None,
            "_source_batch_indexes": [0],
            "_source_chunk_ids": [1],
        },
        {
            "path": "components[]",
            "ids": {"material_name": "nanoscaled LiFePO4"},
            "parent": None,
            "_source_batch_indexes": [1],
            "_source_chunk_ids": [2],
        },
    ]
    kept, stats = merge_contained_skeleton_nodes(nodes, _canon_key)
    assert stats["merged_count"] == 1
    base = kept[0]
    assert base["_source_batch_indexes"] == [0, 1]
    assert base["_source_chunk_ids"] == [1, 2]
