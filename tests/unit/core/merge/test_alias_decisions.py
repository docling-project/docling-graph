"""Unit tests for human alias decisions (--alias-decisions closure, design §5.6)."""

import json
from pathlib import Path
from typing import Any

import networkx as nx

from docling_graph.core.merge import MergePolicy, merge_graphs
from docling_graph.core.provenance.identity import PROVENANCE_NODE_ATTR

_ID_FIELDS = {"Garantie": ["nom"], "Offre": ["nom"]}


def _node(cls: str, **attrs: Any) -> dict[str, Any]:
    return {"label": cls, "type": "entity", "__class__": cls, **attrs}


def _alias_graphs() -> tuple[nx.DiGraph, nx.DiGraph]:
    """Input A: short table label; input B: full section title of the same thing."""
    g1 = nx.DiGraph()
    g1.add_node("G_short", id="G_short", **_node("Garantie", nom="Attentat"))
    g1.add_node("O_e", id="O_e", **_node("Offre", nom="ESSENTIELLE"))
    g1.add_edge("O_e", "G_short", label="INCLUT")
    g1.graph["id_fields_map"] = dict(_ID_FIELDS)
    g2 = nx.DiGraph()
    g2.add_node(
        "G_long",
        id="G_long",
        **_node(
            "Garantie",
            nom="Attentat et actes de terrorisme",
            description="Dommages matériels directs causés par un attentat.",
        ),
    )
    g2.add_node("O_c", id="O_c", **_node("Offre", nom="CONFORT"))
    g2.add_edge("O_c", "G_long", label="INCLUT")
    g2.graph["id_fields_map"] = dict(_ID_FIELDS)
    return g1, g2


def _decisions_file(tmp_path: Path, decisions: list[dict[str, Any]]) -> Path:
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps({"alias_candidates": decisions}), encoding="utf-8")
    return path


_NO_REKEY = {"rekey": False}


def test_candidates_are_proposed_not_merged_by_default():
    g1, g2 = _alias_graphs()
    merged, report = merge_graphs([g1, g2], policy=MergePolicy(**_NO_REKEY))
    assert "G_short" in merged and "G_long" in merged  # propose-only, nothing merged
    assert len(report.alias_candidates) == 1
    stub = report.alias_candidates[0]
    assert stub["class"] == "Garantie"
    assert stub["keep_id"] == "G_short"
    assert stub["merge_ids"] == ["G_long"]
    assert stub["confirm"] is False
    assert stub["similarity"] is None or 0.0 <= stub["similarity"] <= 1.0
    assert report.alias_stats["candidates"] == 1
    assert report.alias_stats["merged"] == 0


def test_confirmed_decision_merges_into_richer_node(tmp_path):
    g1, g2 = _alias_graphs()
    decisions = _decisions_file(
        tmp_path,
        [{"class": "Garantie", "keep_id": "G_short", "merge_ids": ["G_long"], "confirm": True}],
    )
    merged, report = merge_graphs(
        [g1, g2], policy=MergePolicy(alias_decisions=decisions, **_NO_REKEY)
    )
    assert report.alias_stats["merged"] == 1
    # The attribute-richer node survives (description-bearing long form).
    assert "G_long" in merged and "G_short" not in merged
    assert merged.has_edge("O_e", "G_long")  # edges redirected
    aliases = merged.nodes["G_long"]["merged_aliases"]
    assert aliases and aliases[0]["nom"] == "Attentat"
    assert report.ignored_alias_decisions == []


def test_confirmed_merge_keeps_per_document_provenance(tmp_path):
    """The alias fold must wrap cross-document views, never blend ledger-local
    chunk ids under the first document's id."""
    g1, g2 = _alias_graphs()
    g1.nodes["G_short"][PROVENANCE_NODE_ATTR] = {
        "document_id": "doc-a",
        "match": "verbatim",
        "chunks": [2],
        "pages": [1],
    }
    g2.nodes["G_long"][PROVENANCE_NODE_ATTR] = {
        "document_id": "doc-b",
        "match": "verbatim",
        "chunks": [5],
        "pages": [3],
    }
    decisions = _decisions_file(
        tmp_path,
        [{"class": "Garantie", "keep_id": "G_short", "merge_ids": ["G_long"], "confirm": True}],
    )
    merged, report = merge_graphs(
        [g1, g2], policy=MergePolicy(alias_decisions=decisions, **_NO_REKEY)
    )
    assert report.alias_stats["merged"] == 1
    view = merged.nodes["G_long"][PROVENANCE_NODE_ATTR]
    assert view["multi_document"] is True
    by_doc = {s["document_id"]: s for s in view["sources"]}
    assert by_doc["doc-a"]["chunks"] == [2]
    assert by_doc["doc-b"]["chunks"] == [5]


def test_unconfirmed_stub_is_inert(tmp_path):
    g1, g2 = _alias_graphs()
    decisions = _decisions_file(
        tmp_path,
        [{"class": "Garantie", "keep_id": "G_short", "merge_ids": ["G_long"], "confirm": False}],
    )
    merged, report = merge_graphs(
        [g1, g2], policy=MergePolicy(alias_decisions=decisions, **_NO_REKEY)
    )
    assert "G_short" in merged and "G_long" in merged
    assert report.alias_stats["merged"] == 0


def test_unknown_node_ids_are_ignored_and_reported(tmp_path):
    g1, g2 = _alias_graphs()
    decisions = _decisions_file(
        tmp_path,
        [
            {
                "class": "Garantie",
                "keep_id": "Garantie_doesnotexist",
                "merge_ids": ["G_long"],
                "confirm": True,
            },
            {
                "class": "Garantie",
                "keep_id": "G_short",
                "merge_ids": ["Garantie_alsomissing"],
                "confirm": True,
            },
        ],
    )
    merged, report = merge_graphs(
        [g1, g2], policy=MergePolicy(alias_decisions=decisions, **_NO_REKEY)
    )
    assert "G_short" in merged and "G_long" in merged
    assert report.alias_stats["merged"] == 0
    reasons = {d["reason"] for d in report.ignored_alias_decisions}
    assert reasons == {
        "keep_id not present in the merged graph",
        "merge id not present in the merged graph",
    }


def test_unproposed_pair_is_ignored_like_a_hallucinated_llm_pairing(tmp_path):
    """Confirming two unrelated offers (no containment) is silently vetoed."""
    g1, g2 = _alias_graphs()
    decisions = _decisions_file(
        tmp_path,
        [{"class": "Offre", "keep_id": "O_e", "merge_ids": ["O_c"], "confirm": True}],
    )
    merged, report = merge_graphs(
        [g1, g2], policy=MergePolicy(alias_decisions=decisions, **_NO_REKEY)
    )
    assert "O_e" in merged and "O_c" in merged
    assert report.alias_stats["merged"] == 0
    assert report.ignored_alias_decisions == [
        {
            "class": "Offre",
            "keep_id": "O_e",
            "merge_ids": ["O_c"],
            "reason": "pair was not proposed in this run",
        }
    ]


def test_sibling_co_occurrence_veto_still_fires_on_confirmed_pairs(tmp_path):
    """Human confirmation does not bypass the reconciler's sibling veto: two
    offers enumerated side by side under one root are never aliases."""
    g = nx.DiGraph()
    g.add_node("root", id="root", **_node("Root", rid="R"))
    g.add_node("O_c", id="O_c", **_node("Offre", nom="CONFORT"))
    g.add_node("O_cp", id="O_cp", **_node("Offre", nom="CONFORT PLUS"))
    g.add_edge("root", "O_c", label="AOFFRE")
    g.add_edge("root", "O_cp", label="AOFFRE")
    g.graph["id_fields_map"] = {"Offre": ["nom"]}

    decisions = _decisions_file(
        tmp_path,
        [{"class": "Offre", "keep_id": "O_c", "merge_ids": ["O_cp"], "confirm": True}],
    )
    merged, report = merge_graphs([g], policy=MergePolicy(alias_decisions=decisions, **_NO_REKEY))
    assert report.alias_stats["vetoed_sibling"] == 1
    assert report.alias_stats["merged"] == 0
    assert "O_c" in merged and "O_cp" in merged
    # The vetoed confirmation is surfaced, not silently dropped (the user
    # confirmed it; re-running the same decisions file cannot change a veto).
    assert report.ignored_alias_decisions == [
        {
            "class": "Offre",
            "keep_id": "O_c",
            "merge_ids": ["O_cp"],
            "reason": "vetoed by reconciliation guards (e.g. sibling co-occurrence)",
        }
    ]


def test_decisions_accept_bare_list_files(tmp_path):
    g1, g2 = _alias_graphs()
    path = tmp_path / "bare.json"
    path.write_text(
        json.dumps(
            [{"class": "Garantie", "keep_id": "G_short", "merge_ids": ["G_long"], "confirm": True}]
        ),
        encoding="utf-8",
    )
    _merged, report = merge_graphs([g1, g2], policy=MergePolicy(alias_decisions=path, **_NO_REKEY))
    assert report.alias_stats["merged"] == 1
