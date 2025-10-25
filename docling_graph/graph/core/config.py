"""Configuration classes for graph conversion, export, and visualization."""

from dataclasses import dataclass, field
from typing import Final, Literal
from pathlib import Path


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for graph conversion."""

    # Node ID generation
    NODE_ID_HASH_LENGTH: Final[int] = 12

    # Edge options
    add_reverse_edges: bool = False

    # Serialization
    MAX_STRING_LENGTH: Final[int] = 1000
    TRUNCATE_SUFFIX: Final[str] = "..."


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for graph visualization."""

    # Property display
    MAX_TOOLTIP_LENGTH: int = 80
    MAX_PROPERTIES_DISPLAY: int = 5

    # Static visualization (Matplotlib)
    STATIC_NODE_SIZE: int = 3000
    STATIC_NODE_COLOR: str = "lightblue"
    STATIC_EDGE_COLOR: str = "gray"
    STATIC_FONT_SIZE: int = 10
    STATIC_FIGURE_SIZE: tuple[int, int] = (16, 12)
    STATIC_DPI: int = 300

    # Interactive visualization (Pyvis)
    INTERACTIVE_HEIGHT: str = "750px"
    INTERACTIVE_WIDTH: str = "100%"
    INTERACTIVE_BGCOLOR: str = "#ffffff"
    INTERACTIVE_FONT_COLOR: str = "#000000"

    # Output formats
    STATIC_FORMATS: Final[tuple[str, ...]] = ("png", "svg", "pdf")
    DEFAULT_STATIC_FORMAT: str = "png"


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
