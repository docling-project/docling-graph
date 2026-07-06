"""Unit tests for graph-level alias reconciliation (utils.alias_reconciler)."""

from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.utils.alias_reconciler import (
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
