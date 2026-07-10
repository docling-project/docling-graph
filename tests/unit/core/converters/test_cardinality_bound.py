"""Template-declared graph_max_instances cardinality bound (GraphConverter).

The bound is a structural safety rail against discovery spam: past the bound,
the least-supported instances of the class are demoted (removed with incident
edges, recorded under graph.graph["demoted_nodes"]). Ranking is filled-first —
provenance-chunk-first provably buries true instances under alias-merged junk.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.converters.graph_converter import (
    GraphConverter,
    _collect_cardinality_bounds,
)


class Segment(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"], graph_max_instances=2)
    name: str
    description: str | None = None
    revenue: float | None = None


class Report(BaseModel):
    model_config = ConfigDict(graph_id_fields=["title"])
    title: str
    segments: list[Segment] = Field(
        default_factory=list, json_schema_extra={"edge_label": "HAS_SEGMENT"}
    )


def _report(segments: list[Segment]) -> Report:
    return Report(title="FY", segments=segments)


def test_bound_demotes_least_filled_surplus_with_audit_trail() -> None:
    report = _report(
        [
            Segment(name="Software", description="d", revenue=1.0),
            Segment(name="Consulting", description="d"),
            Segment(name="Total"),
            Segment(name="Net income"),
        ]
    )
    graph, _ = GraphConverter().pydantic_list_to_graph([report])
    kept = sorted(d["name"] for _, d in graph.nodes(data=True) if d.get("__class__") == "Segment")
    assert kept == ["Consulting", "Software"]
    demoted = graph.graph["demoted_nodes"]
    assert {d["identity"]["name"] for d in demoted} == {"Total", "Net income"}
    assert all(d["reason"] == "cardinality_bound" for d in demoted)
    # incident edges went with the demoted nodes
    labels = [d["label"] for _, _, d in graph.edges(data=True)]
    assert labels.count("HAS_SEGMENT") == 2


def test_bound_noop_at_or_under_bound() -> None:
    report = _report([Segment(name="Software"), Segment(name="Consulting")])
    graph, _ = GraphConverter().pydantic_list_to_graph([report])
    assert "demoted_nodes" not in graph.graph
    assert sum(1 for _, d in graph.nodes(data=True) if d.get("__class__") == "Segment") == 2


def test_kill_switch_disables_enforcement() -> None:
    report = _report([Segment(name=f"S{i}") for i in range(5)])
    graph, _ = GraphConverter(enforce_cardinality_bounds=False).pydantic_list_to_graph([report])
    assert "demoted_nodes" not in graph.graph
    assert sum(1 for _, d in graph.nodes(data=True) if d.get("__class__") == "Segment") == 5


def test_ranking_prefers_filled_over_provenance_chunks() -> None:
    """A junk node with heavy provenance-chunk support must NOT outrank a
    filled true instance (the 'Total'/'Other' alias-union trap)."""
    report = _report(
        [
            Segment(name="Software", description="real", revenue=1.0),
            Segment(name="Consulting", description="real", revenue=2.0),
            Segment(name="Total"),
        ]
    )
    converter = GraphConverter(
        auto_cleanup=False, validate_graph=False, enforce_cardinality_bounds=False
    )
    graph, _ = converter.pydantic_list_to_graph([report])
    # Inflate the junk node's provenance, then run enforcement directly.
    total_id = next(n for n, d in graph.nodes(data=True) if d.get("name") == "Total")
    graph.nodes[total_id]["__provenance__"] = {"chunks": list(range(300))}
    graph.graph.pop("demoted_nodes", None)
    converter._enforce_cardinality_bounds(graph, {"Segment": 2}, {"Report"}, {"Segment": ["name"]})
    kept = sorted(d["name"] for _, d in graph.nodes(data=True) if d.get("__class__") == "Segment")
    assert kept == ["Consulting", "Software"]


def test_invalid_bound_is_ignored() -> None:
    class Bad(BaseModel):
        model_config = ConfigDict(graph_id_fields=["name"], graph_max_instances=0)
        name: str

    class Root(BaseModel):
        model_config = ConfigDict(graph_id_fields=["rid"])
        rid: str
        bads: list[Bad] = Field(default_factory=list)

    root = Root(rid="R", bads=[Bad(name=f"b{i}") for i in range(4)])
    assert _collect_cardinality_bounds([root]) == {}
    graph, _ = GraphConverter().pydantic_list_to_graph([root])
    assert sum(1 for _, d in graph.nodes(data=True) if d.get("__class__") == "Bad") == 4


def test_collect_bounds_walks_nested_models() -> None:
    report = _report([Segment(name="Software")])
    assert _collect_cardinality_bounds([report]) == {"Segment": 2}
