"""Load exported knowledge graphs (graph.json) back into NetworkX.

Exact mirror of ``JSONExporter._graph_to_dict``: nodes carry ``id`` plus
attributes (the ``id`` attribute is part of the node contract and is
preserved), edges carry ``source``/``target`` plus attributes, and format-v2
graph-level metadata is restored from the top-level ``"graph"`` key (absent on
v1 exports — tolerated). Values live in the flattened JSON space (dates/tuples
were serialized by ``json_serializable``), which is exactly what fill-empty
merge folding operates on.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from ...exceptions import ConfigurationError
from ...logging_utils import get_component_logger
from ..provenance.models import ProvenanceLedger

logger = get_component_logger("GraphImporter", __name__)

# The ExportStage layout: graph.json lives under <run_dir>/docling_graph/.
_RUN_DIR_CANDIDATES = ("docling_graph/graph.json", "graph.json")

# Lossy export formats: CSV stringifies nested dicts through pandas and Cypher
# stringifies everything, so round-tripping them would corrupt attribute types.
_LOSSY_SUFFIXES = {".csv", ".cypher"}


def resolve_graph_path(input_path: Path) -> Path:
    """Resolve a merge input argument to a concrete graph.json path.

    Directories are searched for the ``ExportStage`` layout
    (``docling_graph/graph.json``, then ``graph.json``); ``.json`` files are
    used as-is. CSV/Cypher exports are rejected with a pointer to graph.json.
    """
    if input_path.is_dir():
        for relative in _RUN_DIR_CANDIDATES:
            candidate = input_path / relative
            if candidate.is_file():
                return candidate
        raise ConfigurationError(
            f"No graph export found under directory: {input_path}",
            details={
                "directory": str(input_path),
                "expected": " or ".join(_RUN_DIR_CANDIDATES),
            },
        )
    if input_path.suffix.lower() in _LOSSY_SUFFIXES:
        raise ConfigurationError(
            f"{input_path.suffix} exports are lossy and cannot be merged",
            details={
                "path": str(input_path),
                "hint": "Point at the run's graph.json (docling_graph/graph.json) instead.",
            },
        )
    if not input_path.is_file():
        raise ConfigurationError(
            f"Merge input not found: {input_path}",
            details={"path": str(input_path)},
        )
    return input_path


def load_graph_json(path: Path) -> nx.DiGraph:
    """Load one exported graph.json into an ``nx.DiGraph``.

    Raises:
        ConfigurationError: When the file is not a docling-graph export
            (missing top-level ``nodes``/``edges``), is empty, or is
            structurally corrupt (malformed records, dangling edge endpoints).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigurationError(
            f"Cannot read graph export: {path}", details={"path": str(path)}, cause=e
        ) from e

    if not isinstance(data, dict) or "nodes" not in data or "edges" not in data:
        found = sorted(data.keys()) if isinstance(data, dict) else type(data).__name__
        raise ConfigurationError(
            "Not a docling-graph export: expected top-level 'nodes' and 'edges' keys "
            "(the shape JSONExporter writes)",
            details={"path": str(path), "found": found},
        )

    nodes = data["nodes"]
    edges = data["edges"]
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ConfigurationError(
            "Malformed graph export: 'nodes' and 'edges' must be lists",
            details={"path": str(path)},
        )
    if not nodes:
        # The exporter refuses to write empty graphs, so this is a hand-made file.
        raise ConfigurationError(
            "Graph export contains no nodes (docling-graph never writes empty exports)",
            details={"path": str(path)},
        )

    graph = nx.DiGraph()
    graph_meta = data.get("graph")  # format-v2 metadata; absent on v1 exports
    if isinstance(graph_meta, dict):
        graph.graph.update(graph_meta)

    for raw_node in nodes:
        if not isinstance(raw_node, dict) or "id" not in raw_node:
            raise ConfigurationError(
                "Malformed graph export: every node needs an 'id'",
                details={"path": str(path), "node": str(raw_node)[:200]},
            )
        attrs = dict(raw_node)
        # The exporter writes id redundantly (node key + attr); keep the attr —
        # the node contract has it (GraphConverter node_attrs).
        graph.add_node(attrs["id"], **attrs)

    for raw_edge in edges:
        if not isinstance(raw_edge, dict) or "source" not in raw_edge or "target" not in raw_edge:
            raise ConfigurationError(
                "Malformed graph export: every edge needs 'source' and 'target'",
                details={"path": str(path), "edge": str(raw_edge)[:200]},
            )
        attrs = dict(raw_edge)
        source = attrs.pop("source")
        target = attrs.pop("target")
        if source not in graph or target not in graph:
            raise ConfigurationError(
                "Malformed graph export: edge endpoint is not an exported node",
                details={"path": str(path), "source": str(source), "target": str(target)},
            )
        graph.add_edge(source, target, **attrs)

    logger.info(
        "Loaded graph export %s: %s nodes, %s edges (%s)",
        path,
        graph.number_of_nodes(),
        graph.number_of_edges(),
        str(graph.graph.get("format") or "v1, no graph metadata"),
    )
    return graph


def load_sibling_ledger(graph_path: Path) -> ProvenanceLedger | None:
    """Load the provenance.json the ExportStage writes next to graph.json.

    Missing or unreadable ledgers degrade to ``None`` (merge proceeds without
    document identity for that input).
    """
    ledger_path = graph_path.parent / "provenance.json"
    if not ledger_path.is_file():
        return None
    try:
        return ProvenanceLedger.model_validate_json(ledger_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Ignoring unreadable provenance ledger %s: %s", ledger_path, e)
        return None


def load_graph_input(input_path: Path) -> tuple[nx.DiGraph, ProvenanceLedger | None, Path]:
    """Resolve + load one merge input: (graph, sibling ledger or None, graph path)."""
    graph_path = resolve_graph_path(input_path)
    return load_graph_json(graph_path), load_sibling_ledger(graph_path), graph_path
