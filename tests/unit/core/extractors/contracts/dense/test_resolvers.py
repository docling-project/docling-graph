"""Unit tests for the fuzzy dedupe resolver (aggressive mode)."""

from typing import Any

from docling_graph.core.extractors.contracts.dense.resolvers import resolve_skeleton_nodes


def _key(node: dict[str, Any]) -> tuple[Any, ...]:
    ids = node.get("ids") or {}
    return (node.get("path"), tuple(sorted((k, str(v).lower()) for k, v in ids.items())))


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
