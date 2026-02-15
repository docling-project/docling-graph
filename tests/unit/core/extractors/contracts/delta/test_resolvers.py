from docling_graph.core.extractors.contracts.delta.helpers import DedupPolicy
from docling_graph.core.extractors.contracts.delta.resolvers import (
    DeltaResolverConfig,
    resolve_post_merge_graph,
)


def _policy() -> dict[str, DedupPolicy]:
    return {
        "entity[]": DedupPolicy(
            path="entity[]",
            node_type="Entity",
            identity_fields=tuple(),
            fallback_text_fields=("name",),
            allowed_match_fields=("name",),
            is_entity=True,
        )
    }


def test_fuzzy_resolver_merges_similar_nodes() -> None:
    graph = {
        "nodes": [
            {"path": "entity[]", "ids": {}, "properties": {"name": "Neo4j"}},
            {"path": "entity[]", "ids": {}, "properties": {"name": "Neo4j database"}},
        ],
        "relationships": [],
    }
    out, stats = resolve_post_merge_graph(
        graph,
        dedup_policy=_policy(),
        config=DeltaResolverConfig(enabled=True, mode="fuzzy", fuzzy_threshold=0.4),
    )
    assert len(out["nodes"]) == 1
    assert stats["merged_count"] == 1


def test_resolver_does_not_merge_different_ids_by_default() -> None:
    """Resolver does not merge when both nodes have non-empty distinct ids (default strict behavior)."""
    policy = {
        "entity[]": DedupPolicy(
            path="entity[]",
            node_type="Entity",
            identity_fields=("name",),
            fallback_text_fields=("name",),
            allowed_match_fields=("name",),
            is_entity=True,
        )
    }
    graph = {
        "nodes": [
            {
                "path": "entity[]",
                "ids": {"name": "STUDY-BINDER-MW"},
                "properties": {"name": "Neo4j"},
            },
            {
                "path": "entity[]",
                "ids": {"name": "3"},
                "properties": {"name": "Neo4j"},
            },
        ],
        "relationships": [],
    }
    out, stats = resolve_post_merge_graph(
        graph,
        dedup_policy=policy,
        config=DeltaResolverConfig(enabled=True, mode="fuzzy", fuzzy_threshold=0.6),
    )
    assert len(out["nodes"]) == 2
    assert stats["merged_count"] == 0


def test_resolver_merges_same_content_different_ids_when_allowed() -> None:
    """With allow_merge_different_ids=True, resolver merges same content even when ids differ; id_remap keeps primary's ids."""
    policy = {
        "entity[]": DedupPolicy(
            path="entity[]",
            node_type="Entity",
            identity_fields=("name",),
            fallback_text_fields=("name",),
            allowed_match_fields=("name",),
            is_entity=True,
        )
    }
    graph = {
        "nodes": [
            {
                "path": "entity[]",
                "ids": {"name": "STUDY-BINDER-MW"},
                "properties": {"name": "Neo4j"},
            },
            {
                "path": "entity[]",
                "ids": {"name": "3"},
                "properties": {"name": "Neo4j"},
            },
        ],
        "relationships": [],
    }
    out, stats = resolve_post_merge_graph(
        graph,
        dedup_policy=policy,
        config=DeltaResolverConfig(
            enabled=True, mode="fuzzy", fuzzy_threshold=0.6, allow_merge_different_ids=True
        ),
    )
    assert len(out["nodes"]) == 1
    assert stats["merged_count"] == 1
    assert out["nodes"][0]["ids"]["name"] == "STUDY-BINDER-MW"


def test_resolver_does_not_merge_different_content() -> None:
    """Resolver does not merge when content similarity is below threshold."""
    policy = {
        "entity[]": DedupPolicy(
            path="entity[]",
            node_type="Entity",
            identity_fields=("name",),
            fallback_text_fields=("name",),
            allowed_match_fields=("name",),
            is_entity=True,
        )
    }
    graph = {
        "nodes": [
            {"path": "entity[]", "ids": {"name": "A"}, "properties": {"name": "Neo4j"}},
            {
                "path": "entity[]",
                "ids": {"name": "B"},
                "properties": {"name": "TigerGraph"},
            },
        ],
        "relationships": [],
    }
    out, stats = resolve_post_merge_graph(
        graph,
        dedup_policy=policy,
        config=DeltaResolverConfig(enabled=True, mode="fuzzy", fuzzy_threshold=0.95),
    )
    assert len(out["nodes"]) == 2
    assert stats["merged_count"] == 0


def test_resolver_merges_case_and_diacritic_variants_in_ids() -> None:
    policy = {
        "entity[]": DedupPolicy(
            path="entity[]",
            node_type="Entity",
            identity_fields=("name",),
            fallback_text_fields=("name",),
            allowed_match_fields=("name",),
            is_entity=True,
        )
    }
    graph = {
        "nodes": [
            {"path": "entity[]", "ids": {"name": "PROPRIETAIRE NON OCCUPANT"}, "properties": {}},
            {"path": "entity[]", "ids": {"name": "Proprietaire Non Occupant"}, "properties": {}},
        ],
        "relationships": [],
    }
    out, stats = resolve_post_merge_graph(
        graph,
        dedup_policy=policy,
        config=DeltaResolverConfig(enabled=True, mode="fuzzy", fuzzy_threshold=0.99),
    )
    assert len(out["nodes"]) == 1
    assert stats["merged_count"] == 1


def test_resolver_merges_acronym_with_full_name_in_ids() -> None:
    """PNO (abbreviation) and Propriétaire Non Occupant should merge via acronym matching."""
    policy = {
        "offres[]": DedupPolicy(
            path="offres[]",
            node_type="Offre",
            identity_fields=("nom",),
            fallback_text_fields=("nom",),
            allowed_match_fields=("nom",),
            is_entity=True,
        )
    }
    graph = {
        "nodes": [
            {"path": "offres[]", "ids": {"nom": "PROPRIÉTAIRE NON OCCUPANT"}, "properties": {}},
            {"path": "offres[]", "ids": {"nom": "PNO"}, "properties": {}},
        ],
        "relationships": [],
    }
    out, stats = resolve_post_merge_graph(
        graph,
        dedup_policy=policy,
        config=DeltaResolverConfig(enabled=True, mode="fuzzy", fuzzy_threshold=0.99),
    )
    assert len(out["nodes"]) == 1
    assert stats["merged_count"] == 1


def test_resolver_remaps_relationship_endpoints_after_merge() -> None:
    policy = {
        "entity[]": DedupPolicy(
            path="entity[]",
            node_type="Entity",
            identity_fields=tuple(),
            fallback_text_fields=("name",),
            allowed_match_fields=("name",),
            is_entity=True,
        )
    }
    graph = {
        "nodes": [
            {"path": "entity[]", "ids": {"name": "Neo4j"}, "properties": {"name": "Neo4j"}},
            {
                "path": "entity[]",
                "ids": {"name": "Neo4j DB"},
                "properties": {"name": "Neo4j"},
            },
        ],
        "relationships": [
            {
                "edge_label": "RELATED_TO",
                "source_path": "entity[]",
                "source_ids": {"name": "Neo4j DB"},
                "target_path": "entity[]",
                "target_ids": {"name": "Neo4j"},
                "properties": {},
            }
        ],
    }
    out, stats = resolve_post_merge_graph(
        graph,
        dedup_policy=policy,
        config=DeltaResolverConfig(enabled=True, mode="fuzzy", fuzzy_threshold=0.7),
    )

    assert len(out["nodes"]) == 1
    assert out["relationships"][0]["source_ids"] == {"name": "Neo4j"}
    assert stats["merge_tiers"]["resolver_fuzzy"] == 1


def test_resolver_merges_root_children_even_when_root_parent_ids_differ() -> None:
    policy = {
        "offres[]": DedupPolicy(
            path="offres[]",
            node_type="Offre",
            identity_fields=("nom",),
            fallback_text_fields=("nom",),
            allowed_match_fields=("nom", "title"),
            is_entity=True,
        )
    }
    graph = {
        "nodes": [
            {
                "path": "offres[]",
                "ids": {"nom": "PNO"},
                "parent": {"path": "", "ids": {"reference_document": "A"}},
                "properties": {"title": "Formula"},
            },
            {
                "path": "offres[]",
                "ids": {"nom": "Propriétaire Non Occupant"},
                "parent": {"path": "", "ids": {"reference_document": "B"}},
                "properties": {"title": "Formula"},
            },
        ],
        "relationships": [],
    }
    out, stats = resolve_post_merge_graph(
        graph,
        dedup_policy=policy,
        config=DeltaResolverConfig(enabled=True, mode="chain", fuzzy_threshold=0.9),
    )
    assert len(out["nodes"]) == 1
    assert stats["merge_tiers"]["resolver_identity"] == 1
