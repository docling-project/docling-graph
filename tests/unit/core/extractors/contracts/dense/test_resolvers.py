"""Unit tests for the fuzzy dedupe resolver (aggressive mode) and the
containment proposal pass (standard mode, LLM-confirmed)."""

from typing import Any

from docling_graph.core.extractors.contracts.dense.resolvers import (
    propose_containment_groups,
    resolve_skeleton_nodes,
)


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


def test_containment_proposes_qualifier_variants_with_base_as_keep() -> None:
    """A descriptive superset id is proposed as an alias of its shorter base."""
    groups = propose_containment_groups(
        {
            "components[]": [
                {"material_name": "LiFePO4 (LFP)"},
                {"material_name": "nanoscaled LiFePO4 (LFP)"},
                {"material_name": "PVDF"},
                {"material_name": "Polyvinylidene difluoride (PVDF)"},
            ]
        }
    )
    assert {(g["path"], g["keep"], tuple(g["merge"])) for g in groups} == {
        ("components[]", 0, (1,)),
        ("components[]", 2, (3,)),
    }


def test_containment_proposes_tier_pairs_for_llm_review() -> None:
    """Tier names ARE mechanically proposed — the reconciliation LLM (with the
    tier guard in its prompt) is the layer that must reject them. Nothing is
    merged by the proposal itself."""
    groups = propose_containment_groups({"offres[]": [{"nom": "CONFORT"}, {"nom": "CONFORT PLUS"}]})
    assert groups == [{"path": "offres[]", "keep": 0, "merge": [1]}]


def test_containment_respects_digit_signature() -> None:
    """A superset that differs numerically is a different entity, never proposed."""
    groups = propose_containment_groups(
        {"sections[]": [{"title": "Article 5"}, {"title": "Article 50"}]}
    )
    assert groups == []


def test_containment_skips_ambiguous_multi_base_superset() -> None:
    """A superset containing two distinct shorter ids is ambiguous -> not proposed."""
    groups = propose_containment_groups(
        {"items[]": [{"name": "alpha"}, {"name": "beta"}, {"name": "alphabeta"}]}
    )
    assert groups == []


def test_containment_skips_instances_without_ids() -> None:
    """Empty ids have empty canonical text and are never proposed on either side."""
    groups = propose_containment_groups({"items[]": [{}, {"name": "widget"}, {"name": None}]})
    assert groups == []


def test_containment_short_bases_are_ignored() -> None:
    """Bases below the minimum containment length are too common to be a signal."""
    groups = propose_containment_groups({"items[]": [{"name": "ab"}, {"name": "absolute"}]})
    assert groups == []


def test_containment_groups_stay_within_one_path() -> None:
    groups = propose_containment_groups(
        {
            "authors[]": [{"name": "Smith"}],
            "reviewers[]": [{"name": "Smith et al."}],
        }
    )
    assert groups == []


def test_containment_groups_consolidate_multiple_supersets_per_base() -> None:
    groups = propose_containment_groups(
        {
            "items[]": [
                {"name": "widget"},
                {"name": "blue widget"},
                {"name": "premium widget"},
            ]
        }
    )
    assert groups == [{"path": "items[]", "keep": 0, "merge": [1, 2]}]
