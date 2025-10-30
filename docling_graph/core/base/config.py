"""Configuration classes for graph conversion, export, and visualization."""

from dataclasses import dataclass, field
from typing import Final, Literal
from pathlib import Path


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


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for graph visualization."""

    # Property display
    MAX_TOOLTIP_LENGTH: int = 80
    MAX_PROPERTIES_DISPLAY: int = 5

    # Interactive visualization (Pyvis)
    INTERACTIVE_HEIGHT: str = "calc(100vh - 20px)"
    INTERACTIVE_WIDTH: str = "100%"
    INTERACTIVE_BGCOLOR: str = "#ffffff"
    INTERACTIVE_FONT_COLOR: str = "#2C3E50"
    INTERACTIVE_DIRECTED: bool = True
    INTERACTIVE_CDN_RESOURCES: str = "remote"

    # Static visualization (Matplotlib)
    STATIC_NODE_SIZE: int = 4000
    STATIC_NODE_COLOR: str = "#4A90E2"
    STATIC_NODE_EDGE_COLOR: str = "#2C3E50"
    STATIC_NODE_EDGE_WIDTH: float = 2.5

    STATIC_EDGE_COLOR: str = "#95A5A6"
    STATIC_EDGE_WIDTH: float = 2.5
    STATIC_ARROW_STYLE: str = "-|>"
    STATIC_ARROW_SIZE: int = 30

    STATIC_EDGE_LABEL_COLOR: str = "#B35045"
    STATIC_EDGE_LABEL_FONT_SIZE: int = 12
    STATIC_EDGE_LABEL_FONT_WEIGHT: str = "bold"

    STATIC_NODE_LABEL_FONT_SIZE: int = 12
    STATIC_NODE_LABEL_FONT_WEIGHT: str = "bold"
    STATIC_NODE_LABEL_COLOR: str = "white"

    # Output formats
    STATIC_FORMATS: Final[tuple[str, ...]] = ("png", "svg", "pdf")
    DEFAULT_STATIC_FORMAT: str = "png"
