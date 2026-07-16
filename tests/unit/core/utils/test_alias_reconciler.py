"""Unit tests for graph-level alias reconciliation (utils.alias_reconciler)."""

from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.utils.alias_reconciler import (
    _META_ATTRS,
    _attr_richness,
    containment_groups,
    id_fields_by_class,
    propose_alias_candidates,
    reconcile_graph_aliases,
)


def _node(cls: str, **attrs: Any) -> dict[str, Any]:
    return {"label": cls, "type": "entity", "__class__": cls, **attrs}


def _cgv_like_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("G_short", id="G_short", **_node("Garantie", nom="Attentat"))
    graph.add_node(
        "G_long",
        id="G_long",
        **_node(
            "Garantie",
            nom="Attentat et actes de terrorisme",
            description="Dommages matériels directs causés par un attentat.",
        ),
    )
    graph.add_node("O1", id="O1", **_node("Offre", nom="ESSENTIELLE"))
    graph.add_node("root", id="root", **_node("AssuranceMRH", assureur="X"))
    graph.add_edge("O1", "G_short", label="INCLUTGARANTIE")
    graph.add_edge("root", "G_long", label="AGARANTIE")
    graph.add_edge("root", "O1", label="AOFFRE")
    return graph


_ID_FIELDS = {"Garantie": ["nom"], "Offre": ["nom"], "AssuranceMRH": ["reference_document"]}


def test_containment_groups_unique_base_and_digit_guard() -> None:
    assert containment_groups(["attentat", "attentatetactesdeterrorisme"]) == {0: [1]}
    assert containment_groups(["article5", "article50"]) == {}
    assert containment_groups(["alpha", "beta", "alphabeta"]) == {}
    assert containment_groups(["", "whatever"]) == {}


def test_propose_alias_candidates_finds_table_vs_section_pair() -> None:
    graph = _cgv_like_graph()
    node_ids, displays, groups = propose_alias_candidates(graph, _ID_FIELDS)
    assert groups == [{"class": "Garantie", "keep": 0, "merge": [1]}]
    assert node_ids["Garantie"] == ["G_short", "G_long"]
    assert displays["Garantie"] == ["Attentat", "Attentat et actes de terrorisme"]


def test_reconcile_merges_confirmed_pair_into_richer_node() -> None:
    graph = _cgv_like_graph()

    def confirm(**kwargs: Any) -> dict[str, Any]:
        assert "CONTAINMENT CANDIDATES" in kwargs["prompt"]["user"]
        return {"merges": [{"class": "Garantie", "keep": 0, "merge": [1]}]}

    stats = reconcile_graph_aliases(graph, _ID_FIELDS, llm_call_fn=confirm)
    assert stats["merged"] == 1
    # The attribute-richer node (the long-form one with a description) survives.
    assert "G_long" in graph and "G_short" not in graph
    assert graph.nodes["G_long"]["nom"] == "Attentat et actes de terrorisme"
    # The INCLUT edge was re-pointed onto the survivor.
    assert graph.has_edge("O1", "G_long")
    assert graph.edges["O1", "G_long"]["label"] == "INCLUTGARANTIE"
    # The absorbed identity is recorded.
    aliases = graph.nodes["G_long"]["merged_aliases"]
    assert aliases and aliases[0]["nom"] == "Attentat"


def test_reconcile_protects_tier_pairs_when_llm_declines() -> None:
    graph = nx.DiGraph()
    graph.add_node("O1", id="O1", **_node("Offre", nom="CONFORT"))
    graph.add_node("O2", id="O2", **_node("Offre", nom="CONFORT PLUS"))

    stats = reconcile_graph_aliases(graph, _ID_FIELDS, llm_call_fn=lambda **_kwargs: {"merges": []})
    assert stats["candidates"] == 1  # mechanically proposed...
    assert stats["merged"] == 0  # ...but the LLM tier guard rejected it
    assert "O1" in graph and "O2" in graph


def test_reconcile_ignores_unproposed_llm_pairings() -> None:
    """A hallucinated merge of two unrelated instances is silently dropped."""
    graph = nx.DiGraph()
    graph.add_node("O1", id="O1", **_node("Offre", nom="ESSENTIELLE"))
    graph.add_node("O2", id="O2", **_node("Offre", nom="CONFORT"))
    graph.add_node("O3", id="O3", **_node("Offre", nom="CONFORT PLUS"))

    def hallucinate(**_kwargs: Any) -> dict[str, Any]:
        return {"merges": [{"class": "Offre", "keep": 0, "merge": [1]}]}  # ESSENTIELLE~CONFORT

    stats = reconcile_graph_aliases(graph, _ID_FIELDS, llm_call_fn=hallucinate)
    assert stats["merged"] == 0
    assert set(graph.nodes) == {"O1", "O2", "O3"}


def test_reconcile_without_llm_is_propose_only() -> None:
    graph = _cgv_like_graph()
    stats = reconcile_graph_aliases(graph, _ID_FIELDS, llm_call_fn=None)
    assert stats["candidates"] == 1
    assert stats["merged"] == 0
    assert "G_short" in graph and "G_long" in graph


def test_reconcile_survives_llm_exception() -> None:
    graph = _cgv_like_graph()

    def boom(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("provider down")

    stats = reconcile_graph_aliases(graph, _ID_FIELDS, llm_call_fn=boom)
    assert stats["merged"] == 0
    assert "G_short" in graph and "G_long" in graph


def test_merge_audit_attrs_are_meta_not_content() -> None:
    """__conflicts__ and merged_from (graph-merge audit records) are framework
    attrs: they never count toward attribute richness."""
    assert {"__conflicts__", "merged_from"} <= _META_ATTRS
    bare = _node("Garantie", nom="Attentat")
    audited = {
        **bare,
        "__conflicts__": [{"field": "nom", "dropped": "x"}],
        "merged_from": [{"document_id": "doc-a"}],
    }
    assert _attr_richness(audited) == _attr_richness(bare)


def test_reconcile_never_copies_merge_audit_attrs() -> None:
    """An absorbed node's __conflicts__/merged_from must neither tip the
    survivor choice nor be copied onto the survivor."""
    graph = _cgv_like_graph()
    graph.nodes["G_short"]["__conflicts__"] = [{"field": "nom", "dropped": "x"}]
    graph.nodes["G_short"]["merged_from"] = [{"document_id": "doc-a"}]

    def confirm(**_kwargs: Any) -> dict[str, Any]:
        return {"merges": [{"class": "Garantie", "keep": 0, "merge": [1]}]}

    stats = reconcile_graph_aliases(graph, _ID_FIELDS, llm_call_fn=confirm)
    assert stats["merged"] == 1
    # Without meta-attr registration the audit attrs would have made G_short
    # the "richer" survivor; the description-bearing node must still win.
    assert "G_long" in graph and "G_short" not in graph
    assert "__conflicts__" not in graph.nodes["G_long"]
    assert "merged_from" not in graph.nodes["G_long"]


def test_id_fields_by_class_walks_nested_models() -> None:
    class Bien(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str

    class Garantie(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str
        biens: list[Bien] = Field(default_factory=list)

    class Root(BaseModel):
        garanties: list[Garantie] = Field(default_factory=list)

    root = Root(garanties=[Garantie(nom="G", biens=[Bien(nom="B")])])
    mapping = id_fields_by_class([root])
    assert mapping["Garantie"] == ["nom"]
    assert mapping["Bien"] == ["nom"]
    assert mapping["Root"] == []


def test_converter_runs_alias_pass_with_llm_fn() -> None:
    """End-to-end through GraphConverter: duplicate-named entities merge when
    the (mocked) LLM confirms the containment candidate."""
    from docling_graph.core.converters.graph_converter import GraphConverter

    class Garantie(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str
        description: str | None = None

    class Offre(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str
        incluses: list[Garantie] = Field(
            default_factory=list, json_schema_extra={"edge_label": "INCLUT"}
        )

    class Root(BaseModel):
        model_config = ConfigDict(graph_id_fields=[])
        offres: list[Offre] = Field(default_factory=list)
        garanties: list[Garantie] = Field(default_factory=list)

    root = Root(
        offres=[Offre(nom="ESSENTIELLE", incluses=[Garantie(nom="Attentat")])],
        garanties=[Garantie(nom="Attentat et actes de terrorisme", description="Détail complet.")],
    )

    def confirm(**kwargs: Any) -> dict[str, Any]:
        return {"merges": [{"class": "Garantie", "keep": 0, "merge": [1]}]}

    converter = GraphConverter(validate_graph=False, alias_llm_fn=confirm)
    graph, _meta = converter.pydantic_list_to_graph([root])
    garanties = [n for n, d in graph.nodes(data=True) if d.get("__class__") == "Garantie"]
    assert len(garanties) == 1
    survivor = graph.nodes[garanties[0]]
    assert survivor["nom"] == "Attentat et actes de terrorisme"
    offre = next(n for n, d in graph.nodes(data=True) if d.get("__class__") == "Offre")
    assert graph.has_edge(offre, garanties[0])


def test_reconcile_vetoes_sibling_co_occurrence_shared_parent() -> None:
    """Two same-class nodes hanging off one neighbor under one label are
    enumerated siblings (root -AOFFRE-> CONFORT and CONFORT PLUS) — the merge
    is vetoed even when the confirm LLM approves it."""
    graph = nx.DiGraph()
    graph.add_node("r", __class__="Root", id="r", label="Root", rid="R")
    graph.add_node("a", __class__="Offre", id="a", label="Offre", nom="CONFORT")
    graph.add_node("b", __class__="Offre", id="b", label="Offre", nom="CONFORT PLUS")
    graph.add_edge("r", "a", label="AOFFRE")
    graph.add_edge("r", "b", label="AOFFRE")
    stats = reconcile_graph_aliases(
        graph,
        {"Offre": ["nom"]},
        llm_call_fn=lambda **kw: {"merges": [{"class": "Offre", "keep": 0, "merge": [1]}]},
    )
    assert stats["vetoed_sibling"] == 1
    assert stats["merged"] == 0
    assert graph.number_of_nodes() == 3


def test_reconcile_vetoes_sibling_co_occurrence_shared_child() -> None:
    """The veto also fires on a shared same-label CHILD (both nodes -INCLUT->
    the same target)."""
    graph = nx.DiGraph()
    graph.add_node("a", __class__="Offre", id="a", label="Offre", nom="CONFORT")
    graph.add_node("b", __class__="Offre", id="b", label="Offre", nom="CONFORT PLUS")
    graph.add_node("g", __class__="Garantie", id="g", label="Garantie", nom="Incendie")
    graph.add_edge("a", "g", label="INCLUT")
    graph.add_edge("b", "g", label="INCLUT")
    stats = reconcile_graph_aliases(
        graph,
        {"Offre": ["nom"]},
        llm_call_fn=lambda **kw: {"merges": [{"class": "Offre", "keep": 0, "merge": [1]}]},
    )
    assert stats["vetoed_sibling"] == 1
    assert stats["merged"] == 0


def test_reconcile_merges_non_sibling_alias_and_repoints_all_edges() -> None:
    """A true alias pair with disjoint neighbors still merges, and the absorbed
    node's in AND out edges land on the survivor with labels preserved (the
    edge-union invariant)."""
    graph = nx.DiGraph()
    graph.add_node("short", __class__="Garantie", id="short", label="Garantie", nom="Attentat")
    graph.add_node(
        "long",
        __class__="Garantie",
        id="long",
        label="Garantie",
        nom="Attentat et actes de terrorisme",
        texte="detail",
    )
    graph.add_node("o", __class__="Offre", id="o", label="Offre", nom="ESSENTIELLE")
    graph.add_node("e", __class__="Exclusion", id="e", label="Exclusion", cid="E1")
    graph.add_edge("o", "short", label="INCLUT")
    graph.add_edge("long", "e", label="AEXCLUSION")
    stats = reconcile_graph_aliases(
        graph,
        {"Garantie": ["nom"]},
        llm_call_fn=lambda **kw: {"merges": [{"class": "Garantie", "keep": 0, "merge": [1]}]},
    )
    assert stats["merged"] == 1
    assert stats["vetoed_sibling"] == 0
    survivor = "long" if "long" in graph else "short"
    assert graph.has_edge("o", survivor)
    assert graph["o"][survivor]["label"] == "INCLUT"
    assert graph.has_edge(survivor, "e")
    assert graph[survivor]["e"]["label"] == "AEXCLUSION"
