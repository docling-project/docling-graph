"""Configuration classes for graph conversion, export, and visualization."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal


@dataclass(frozen=True)
class GraphConfig:
    """Internal Constants."""

    # Node ID generation
    NODE_ID_HASH_LENGTH: Final[int] = 12

    # Serialization
    MAX_STRING_LENGTH: Final[int] = 1000
    TRUNCATE_SUFFIX: Final[str] = "..."

    """Configuration Options."""

    # Edge options
    add_reverse_edges: bool = False

    # Trigger graph validation
    validate_graph: bool = True


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for graph export."""

    # CSV export
    CSV_ENCODING: str = "utf-8"
    CSV_NODE_FILENAME: str = "nodes.csv"
    CSV_EDGE_FILENAME: str = "edges.csv"

    # Cypher export
    CYPHER_ENCODING: str = "utf-8"
    CYPHER_FILENAME: str = "graph.cypher"
    CYPHER_BATCH_SIZE: int = 1000

    # JSON export
    JSON_ENCODING: str = "utf-8"
    JSON_INDENT: int = 2
    JSON_FILENAME: str = "graph.json"

    # General
    ENSURE_ASCII: bool = False
