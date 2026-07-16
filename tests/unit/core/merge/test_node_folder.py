"""Unit tests for the per-field fold rules (core.merge.node_folder, design §5.3/§5.7)."""

from docling_graph.core.merge.node_folder import fold_edge, fold_edge_attrs, fold_node_attrs
from docling_graph.core.merge.policy import MergePolicy
from docling_graph.core.utils.alias_reconciler import _attr_richness
from docling_graph.core.utils.graph_cleaner import GraphCleaner

POLICY = MergePolicy()


def test_rule1_meta_attrs_never_copied():
    survivor = {"id": "A", "label": "X", "type": "entity", "__class__": "X", "name": "n"}
    incoming = {
        "id": "B",
        "label": "Y",
        "type": "other",
        "__class__": "Y",
        "__provenance__": {"chunks": [1]},
        "merged_aliases": [{"id": "Z"}],
        "merged_from": [{"document_id": "d"}],
        "__conflicts__": [{"field": "x"}],
    }
    conflicts = fold_node_attrs(survivor, incoming, POLICY, "src")
    assert conflicts == []
    assert survivor == {"id": "A", "label": "X", "type": "entity", "__class__": "X", "name": "n"}


def test_rule2_empty_never_clobbers():
    survivor = {"id": "A", "name": "kept", "tags": ["a"], "meta": {"k": 1}}
    incoming = {"name": None, "tags": [], "meta": {}, "other": ""}
    assert fold_node_attrs(survivor, incoming, POLICY, "src") == []
    assert survivor == {"id": "A", "name": "kept", "tags": ["a"], "meta": {"k": 1}}


def test_rule3_fill_empty():
    survivor = {"id": "A", "name": "", "role": None, "tags": []}
    incoming = {"name": "Ada", "role": "physicist", "tags": ["x"], "extra": 5}
    assert fold_node_attrs(survivor, incoming, POLICY, "src") == []
    assert survivor["name"] == "Ada"
    assert survivor["role"] == "physicist"
    assert survivor["tags"] == ["x"]
    assert survivor["extra"] == 5


def test_rule4_equal_values_are_noop_even_for_combine_fields():
    # Equality short-circuits BEFORE the sentence merge, keeping folds
    # byte-idempotent (merge_descriptions would strip/re-truncate).
    text = "First sentence.  "
    survivor = {"id": "A", "description": text}
    incoming = {"description": text}
    assert fold_node_attrs(survivor, incoming, POLICY, "src") == []
    assert survivor["description"] == text


def test_rule5_combine_fields_sentence_merge():
    survivor = {"id": "A", "description": "Pioneer of radioactivity research."}
    incoming = {
        "description": "Pioneer of radioactivity research. First woman to win a Nobel Prize."
    }
    assert fold_node_attrs(survivor, incoming, POLICY, "src") == []
    assert (
        survivor["description"]
        == "Pioneer of radioactivity research. First woman to win a Nobel Prize."
    )


def test_rule6_scalar_list_union_preserves_order():
    survivor = {"id": "A", "tags": ["b", "a"]}
    incoming = {"tags": ["a", "c", "b", "d"]}
    assert fold_node_attrs(survivor, incoming, POLICY, "src") == []
    assert survivor["tags"] == ["b", "a", "c", "d"]


def test_rule7_list_of_dicts_content_hash_dedup():
    survivor = {"id": "A", "measurements": [{"name": "w", "value": 1}]}
    incoming = {
        "measurements": [
            {"value": 1, "name": "w"},  # same content, different key order -> deduped
            {"name": "h", "value": 2},
        ]
    }
    assert fold_node_attrs(survivor, incoming, POLICY, "src") == []
    assert survivor["measurements"] == [{"name": "w", "value": 1}, {"name": "h", "value": 2}]


def test_rule8_conflict_keep_first_records_and_keeps_survivor():
    survivor = {"id": "A", "role": "physicist"}
    incoming = {"role": "chemist"}
    conflicts = fold_node_attrs(survivor, incoming, POLICY, "doc-b")
    assert survivor["role"] == "physicist"
    assert "__conflicts__" not in survivor
    assert conflicts == [
        {
            "node": "A",
            "field": "role",
            "kept": "physicist",
            "dropped": "chemist",
            "dropped_source": "doc-b",
        }
    ]


def test_rule8_conflict_keep_all_stores_suppressed_value():
    policy = MergePolicy(conflicts="keep-all")
    survivor = {"id": "A", "role": "physicist"}
    conflicts = fold_node_attrs(survivor, {"role": "chemist"}, policy, "doc-b")
    assert len(conflicts) == 1
    assert survivor["role"] == "physicist"
    assert survivor["__conflicts__"] == [{"field": "role", "value": "chemist", "source": "doc-b"}]
    # Folding the same conflict again does not duplicate the record.
    fold_node_attrs(survivor, {"role": "chemist"}, policy, "doc-b")
    assert len(survivor["__conflicts__"]) == 1


def test_conflicts_attr_is_invisible_to_richness_and_content_hash():
    """keep-all suppressed values are audit records: they must not tip the
    richer-survivor ranking nor split the cleaner's content-duplicate hash."""
    bare = {"id": "A", "label": "X", "type": "entity", "__class__": "X", "name": "n"}
    audited = {
        **bare,
        "__conflicts__": [{"field": "role", "value": "x", "source": "s"}],
        "merged_from": [{"document_id": "d", "source": "s"}],
    }
    assert _attr_richness(audited) == _attr_richness(bare)
    cleaner = GraphCleaner(verbose=False)
    assert cleaner._compute_content_hash(audited, "A") == cleaner._compute_content_hash(bare, "A")


def test_dict_conflict_keeps_survivor():
    survivor = {"id": "A", "address": {"city": "Paris"}}
    conflicts = fold_node_attrs(survivor, {"address": {"city": "Lyon"}}, POLICY, "src")
    assert survivor["address"] == {"city": "Paris"}
    assert len(conflicts) == 1 and conflicts[0]["field"] == "address"


def test_fold_does_not_alias_incoming_values():
    incoming = {"tags": ["x"], "items": [{"a": 1}]}
    survivor = {"id": "A", "tags": None, "items": []}
    fold_node_attrs(survivor, incoming, POLICY, "src")
    survivor["tags"].append("mutated")
    survivor["items"][0]["a"] = 999
    assert incoming["tags"] == ["x"]
    assert incoming["items"] == [{"a": 1}]


# ------------------------------------------------------------------- edges


def test_fold_edge_same_label_fills_and_unions():
    survivor = {"label": "INCLUT", "keywords": ["a"], "weight": None}
    incoming = {"label": "INCLUT", "keywords": ["b", "a"], "weight": 2}
    record = fold_edge("U", "V", survivor, incoming, POLICY, "doc-b")
    assert record is None
    assert survivor["keywords"] == ["a", "b"]
    assert survivor["weight"] == 2
    assert "also_labels" not in survivor


def test_fold_edge_different_label_keeps_first_and_records():
    survivor = {"label": "EMPLOYS", "keywords": ["a"]}
    incoming = {"label": "HIRES", "keywords": ["b"]}
    record = fold_edge("U", "V", survivor, incoming, POLICY, "doc-b")
    assert survivor["label"] == "EMPLOYS"
    assert survivor["also_labels"] == ["HIRES"]
    assert survivor["keywords"] == ["a"]  # attrs of a different relation are not folded
    assert record == {
        "source": "U",
        "kept_label": "EMPLOYS",
        "dropped_labels": ["HIRES"],
        "target": "V",
        "dropped_source": "doc-b",
    }


def test_fold_edge_carries_prior_also_labels():
    survivor = {"label": "EMPLOYS"}
    incoming = {"label": "EMPLOYS", "also_labels": ["HIRES", "EMPLOYS"]}
    assert fold_edge("U", "V", survivor, incoming, POLICY, "src") is None
    assert survivor["also_labels"] == ["HIRES"]  # kept label never listed twice


def test_fold_edge_attrs_conflicting_scalars_keep_survivor_silently():
    survivor = {"label": "REL", "weight": 1}
    fold_edge_attrs(survivor, {"label": "REL", "weight": 9}, POLICY)
    assert survivor["weight"] == 1
