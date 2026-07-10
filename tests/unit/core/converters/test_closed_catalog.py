"""Closed-catalog reference-edge guard (GraphConverter).

A field declared reference_closed_catalog promises its targets form a FIXED
catalog canonically defined elsewhere in the schema: a target that exists ONLY
through such references is a hallucinated catalog member — its edges (and the
orphaned target) are dropped. A target independently anchored by any other edge
keeps everything. When EVERY member of the class is closed-catalog-only, the
canonical catalog was not extracted and enforcement refuses to wipe the class.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.converters.graph_converter import GraphConverter


class Bien(BaseModel):
    model_config = ConfigDict(graph_id_fields=["nom"])
    nom: str


class Exclusion(BaseModel):
    model_config = ConfigDict(graph_id_fields=["cid"])
    cid: str
    biens_exclus: list[Bien] = Field(
        default_factory=list,
        json_schema_extra={"edge_label": "EXCLUTBIEN", "reference_closed_catalog": True},
    )


class Garantie(BaseModel):
    model_config = ConfigDict(graph_id_fields=["nom"])
    nom: str
    biens_couverts: list[Bien] = Field(
        default_factory=list, json_schema_extra={"edge_label": "COUVREBIEN"}
    )


class Contrat(BaseModel):
    model_config = ConfigDict(graph_id_fields=["ref"])
    ref: str
    garanties: list[Garantie] = Field(default_factory=list, json_schema_extra={"edge_label": "AG"})
    exclusions: list[Exclusion] = Field(
        default_factory=list, json_schema_extra={"edge_label": "AE"}
    )


def test_unanchored_targets_dropped_anchored_targets_kept() -> None:
    contrat = Contrat(
        ref="C",
        garanties=[Garantie(nom="Vol", biens_couverts=[Bien(nom="Piscine")])],
        exclusions=[Exclusion(cid="E1", biens_exclus=[Bien(nom="Piscine"), Bien(nom="bateaux")])],
    )
    graph, _ = GraphConverter().pydantic_list_to_graph([contrat])
    biens = sorted(d["nom"] for _, d in graph.nodes(data=True) if d.get("__class__") == "Bien")
    assert biens == ["Piscine"]
    labels = [d["label"] for _, _, d in graph.edges(data=True)]
    # Piscine keeps BOTH its edges (anchored member); bateaux lost its only one.
    assert labels.count("COUVREBIEN") == 1
    assert labels.count("EXCLUTBIEN") == 1
    assert graph.graph["closed_catalog_drops"] == {"EXCLUTBIEN": 1}


def test_all_unanchored_class_is_skipped_not_wiped() -> None:
    contrat = Contrat(
        ref="C",
        exclusions=[Exclusion(cid="E1", biens_exclus=[Bien(nom="bateaux"), Bien(nom="remorques")])],
    )
    graph, _ = GraphConverter().pydantic_list_to_graph([contrat])
    biens = sorted(d["nom"] for _, d in graph.nodes(data=True) if d.get("__class__") == "Bien")
    assert biens == ["bateaux", "remorques"]
    assert "closed_catalog_drops" not in graph.graph


def test_marker_is_stripped_from_surviving_edges() -> None:
    contrat = Contrat(
        ref="C",
        garanties=[Garantie(nom="Vol", biens_couverts=[Bien(nom="Piscine")])],
        exclusions=[Exclusion(cid="E1", biens_exclus=[Bien(nom="Piscine")])],
    )
    graph, _ = GraphConverter().pydantic_list_to_graph([contrat])
    assert not any("_closed_catalog" in d for _, _, d in graph.edges(data=True))


def test_stale_marker_from_digraph_attr_merge_does_not_count() -> None:
    """nx.DiGraph merges attr dicts when one (source, target) pair is re-added
    under a second label; the label-scoped marker must not poison the survivor."""
    import networkx as nx

    converter = GraphConverter(auto_cleanup=False, validate_graph=False)
    graph = nx.DiGraph()
    graph.add_node("c", __class__="Exclusion", id="c", label="Exclusion", cid="E1")
    graph.add_node("b", __class__="Bien", id="b", label="Bien", nom="Piscine")
    graph.add_node("b2", __class__="Bien", id="b2", label="Bien", nom="bateaux")
    # contaminated survivor: COUVREBIEN label with a stale EXCLUTBIEN marker
    graph.add_edge("c", "b", label="COUVREBIEN", _closed_catalog="EXCLUTBIEN")
    graph.add_edge("c", "b2", label="EXCLUTBIEN", _closed_catalog="EXCLUTBIEN")
    converter._enforce_closed_catalogs(graph)
    assert "b" in graph  # contaminated edge is NOT treated as closed-catalog
    assert graph.has_edge("c", "b")
    assert "b2" not in graph  # genuinely marked-only target dropped
    assert not any("_closed_catalog" in d for _, _, d in graph.edges(data=True))
