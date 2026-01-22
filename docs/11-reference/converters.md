# Converters API

**Navigation:** [← Exceptions](exceptions.md) | [Next: Extractors →](extractors.md)

---

## Overview

Graph conversion from Pydantic models to NetworkX graphs.

**Module:** `docling_graph.core.converters`

---

## GraphConverter

Main class for converting Pydantic models to knowledge graphs.

```python
class GraphConverter:
    """Convert Pydantic models to NetworkX graphs."""
    
    def __init__(
        self,
        config: Optional[GraphConverterConfig] = None,
        node_id_registry: Optional[NodeIDRegistry] = None
    ):
        """Initialize converter."""
```

### Methods

#### convert()

```python
def convert(
    self,
    models: List[BaseModel],
    reverse_edges: bool = False
) -> nx.MultiDiGraph:
    """
    Convert Pydantic models to graph.
    
    Args:
        models: List of Pydantic model instances
        reverse_edges: Create reverse edges
        
    Returns:
        NetworkX MultiDiGraph
    """
```

**Example:**

```python
from docling_graph.core.converters import GraphConverter

converter = GraphConverter()
graph = converter.convert(models, reverse_edges=False)

print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
```

---

## NodeIDRegistry

Registry for stable node ID generation.

```python
class NodeIDRegistry:
    """Generate and track stable node IDs."""
    
    def get_or_create_id(
        self,
        model: BaseModel,
        node_type: str
    ) -> str:
        """
        Get or create stable ID for model.
        
        Args:
            model: Pydantic model instance
            node_type: Type of node
            
        Returns:
            Stable node ID
        """
```

**Features:**
- Deterministic ID generation
- Collision detection
- Cross-batch consistency
- Graph ID field support

**Example:**

```python
from docling_graph.core.converters import NodeIDRegistry

registry = NodeIDRegistry()
node_id = registry.get_or_create_id(person_model, "Person")
```

---

## GraphConverterConfig

Configuration for graph conversion.

```python
class GraphConverterConfig(BaseModel):
    """Configuration for graph converter."""
    
    reverse_edges: bool = False
    include_metadata: bool = True
```

---

## Related APIs

- **[Graph Management](../06-graph-management/graph-conversion.md)** - Usage guide
- **[Exporters](exporters.md)** - Export graphs

---

**Navigation:** [← Exceptions](exceptions.md) | [Next: Extractors →](extractors.md)