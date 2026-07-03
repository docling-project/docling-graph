# Converters API


## Overview

Graph conversion from Pydantic models to NetworkX graphs.

**Module:** `docling_graph.core.converters`

---

## GraphConverter

Main class for converting Pydantic models to knowledge graphs. Stateless and thread-safe — all conversion state is passed through method parameters, not held on the instance.

```python
class GraphConverter:
    """Converts Pydantic models to NetworkX graphs with enhanced features."""

    def __init__(
        self,
        config: GraphConfig | None = None,
        add_reverse_edges: bool = False,
        validate_graph: bool = True,
        registry: NodeIDRegistry | None = None,
        auto_cleanup: bool = True,
    ) -> None:
        """
        Args:
            config: Graph configuration (optional).
            add_reverse_edges: Create bidirectional edges.
            validate_graph: Validate structure after conversion.
            registry: Shared NodeIDRegistry for cross-batch ID consistency.
                A new registry is created per instance if omitted.
            auto_cleanup: Remove phantom nodes, duplicates, and orphaned
                edges after conversion.
        """
```

### Methods

#### pydantic_list_to_graph()

```python
def pydantic_list_to_graph(
    self,
    model_instances: List[BaseModel],
    provenance_binder: Callable[[nx.DiGraph, List[BaseModel]], None] | None = None,
) -> tuple[nx.DiGraph, GraphMetadata]:
    """
    Convert a list of Pydantic models to a NetworkX graph.

    Args:
        model_instances: Pydantic model instances to convert.
        provenance_binder: Optional callable that annotates nodes with
            __provenance__ after edges are created but before cleanup.
            See Data Grounding & Provenance.

    Returns:
        Tuple of (graph, metadata).
    """
```

**Example:**

```python
from docling_graph.core import GraphConverter

converter = GraphConverter()
graph, metadata = converter.pydantic_list_to_graph(models)

print(f"Nodes: {metadata.node_count}")
print(f"Edges: {metadata.edge_count}")
```

---

## NodeIDRegistry

Deterministic node ID registry for cross-batch consistency: the same entity always gets the same node ID, even when extracted in different batches.

```python
class NodeIDRegistry:
    """Global registry that maps entity fingerprints to stable node IDs."""

    def get_node_id(self, model_instance: BaseModel, auto_register: bool = True) -> str:
        """
        Get or create a deterministic node ID for a model instance.

        Fingerprint is derived from graph_id_fields (entities) or all
        non-empty fields (components, is_entity=False). ID format:
        "{ClassName}_{fingerprint}".
        """

    def register_batch(self, models: list[BaseModel]) -> None:
        """Register every model in a batch to pre-populate the registry."""

    def get_stats(self) -> dict:
        """Registry statistics: total_entities, classes (per-class counts)."""
```

**Features:**

- Deterministic ID generation (content hash of identity fields)
- Collision detection (raises if a fingerprint maps to two different classes)
- Cross-batch consistency when a registry instance is shared
- `graph_id_fields` support (falls back to content-based hashing for components)

**Example:**

```python
from docling_graph.core.converters.node_id_registry import NodeIDRegistry

registry = NodeIDRegistry()
node_id = registry.get_node_id(person_model)
```

---

## GraphConfig

Graph conversion configuration — a frozen dataclass, not a Pydantic model.

```python
@dataclass(frozen=True)
class GraphConfig:
    """Internal constants and configuration options for graph conversion."""

    NODE_ID_HASH_LENGTH: Final[int] = 12
    MAX_STRING_LENGTH: Final[int] = 1000
    TRUNCATE_SUFFIX: Final[str] = "..."

    add_reverse_edges: bool = False
    validate_graph: bool = True
```

The companion `ExportConfig` dataclass (same module) holds the real output filenames:

```python
@dataclass(frozen=True)
class ExportConfig:
    """Configuration for graph export."""

    CSV_NODE_FILENAME: str = "nodes.csv"
    CSV_EDGE_FILENAME: str = "edges.csv"
    CYPHER_FILENAME: str = "graph.cypher"
    JSON_FILENAME: str = "graph.json"
```

```python
from docling_graph.core import GraphConfig, ExportConfig
```

---

## Related APIs

- **[Graph Management](../fundamentals/graph-management/graph-conversion.md)** - Usage guide
- **[Provenance](provenance.md)** - `provenance_binder` and the `__provenance__` node attribute
- **[Exporters](exporters.md)** - Export graphs