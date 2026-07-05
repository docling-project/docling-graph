# Graph Conversion


## Overview

**Graph conversion** transforms Pydantic models into NetworkX directed graphs, creating nodes for entities and edges for relationships. This is the foundation of knowledge graph creation.

**In this guide:**

- Conversion process
- Node and edge creation
- Node ID registry
- Graph validation
- Automatic cleanup

---

## Conversion Process

### High-Level Flow

--8<-- "docs/assets/flowcharts/conversion_process.md"

---

## GraphConverter

### Basic Usage

```python
from docling_graph.core import GraphConverter

# Create converter
converter = GraphConverter()

# Convert models to graph
graph, metadata = converter.pydantic_list_to_graph(models)

print(f"Created graph with {metadata.node_count} nodes and {metadata.edge_count} edges")
```

### With Configuration

```python
from docling_graph.core import GraphConverter

converter = GraphConverter(
    add_reverse_edges=False,  # Don't create bidirectional edges
    validate_graph=True,      # Validate structure
    auto_cleanup=True         # Remove phantom nodes
)

graph, metadata = converter.pydantic_list_to_graph(models)
```

---

## Node Creation

### What Becomes a Node?

**Entities** (models with `is_entity=True`) become nodes:

```python
from pydantic import BaseModel

# ✅ Becomes a node
class Organization(BaseModel):
    name: str
    model_config = {"is_entity": True}  # Default

# ❌ Does NOT become a node
class Address(BaseModel):
    street: str
    city: str
    model_config = {"is_entity": False}  # Component
```

### Node Structure

```python
# Node in graph
{
    "id": "organization_acme_corp",
    "label": "Organization",
    "type": "entity",
    "__class__": "Organization",
    "name": "Acme Corp",
    "address": None,  # Reference to nested entity
    "__provenance__": {"document_id": "...", "match": "verbatim", "chunks": [2], "pages": [1]},
}
```

`__provenance__` is added by the [data-grounding binder](provenance.md) when `provenance` is not `"off"` (the default). Being a dunder attribute, it can never collide with a template field. See [Data Grounding & Provenance](provenance.md).

---

## Edge Creation

### Automatic Edge Generation

Edges are created automatically from model relationships:

```python
class BillingDocument(BaseModel):
    document_no: str
    issued_by: Organization  # Creates edge: BillingDocument -> Organization
    line_items: List[LineItem]  # Creates edges: BillingDocument -> LineItem (multiple)
```

### Edge Structure

```python
# Edge in graph
{
    "source": "invoice_001",
    "target": "organization_acme_corp",
    "label": "issued_by",
    "properties": {}
}
```

### Custom Edge Labels

```python
from pydantic import BaseModel, Field

class BillingDocument(BaseModel):
    issued_by: Organization = Field(
        json_schema_extra={"edge_label": "ISSUED_BY"}
    )
```

**Result:** Edge label becomes `ISSUED_BY` instead of `issued_by`

---

## Node ID Registry

### What is Node ID Registry?

The **NodeIDRegistry** ensures consistent, deterministic node IDs across multiple extractions.

### How It Works

```python
# Same entity always gets same ID
org1 = Organization(name="Acme Corp")
org2 = Organization(name="Acme Corp")

# Both get ID: "organization_acme_corp"
id1 = registry.get_node_id(org1)
id2 = registry.get_node_id(org2)

assert id1 == id2  # True
```

### ID Generation

The registry's `get_node_id()` derives an ID from a content fingerprint — a `blake2b` hash of the model's identity fields, truncated to 16 hex characters — and formats it as `"{ClassName}_{fingerprint}"`:

- **Entities** (have `graph_id_fields` in `model_config`): the fingerprint is computed from those fields only, so two instances with the same identity field values always get the same ID regardless of other differing fields.
- **Components** (`is_entity=False`, no `graph_id_fields`): the fingerprint is computed from *all* non-empty scalar fields, so identical components are deduplicated by content.

This logic is internal to `NodeIDRegistry` — to change how a model's identity is derived, set `graph_id_fields` on that model's `model_config` rather than subclassing the registry (see [Key Concepts: Node ID Generation](../../introduction/key-concepts.md#node-id-generation)).

---

## Provenance Binding

`pydantic_list_to_graph()` accepts an optional `provenance_binder` — a callable that annotates each node with a `__provenance__` attribute after edges are created but **before** automatic cleanup, so that when cleanup merges duplicate nodes, their provenance views are unioned rather than one being lost.

```python
def pydantic_list_to_graph(
    self,
    model_instances: list[BaseModel],
    provenance_binder: Callable[[nx.DiGraph, list[BaseModel]], None] | None = None,
) -> tuple[nx.DiGraph, GraphMetadata]:
```

The pipeline builds this closure automatically (using the **same** `NodeIDRegistry` instance the converter uses, so binding can never disagree on node IDs) whenever `provenance` is not `"off"`; `GraphConverter` itself has no dependency on the provenance module. See [Data Grounding & Provenance](provenance.md) for what the binder does and the resulting node attribute shape.

---

## Graph Validation

### Automatic Validation

Validation checks graph structure:

```python
converter = GraphConverter(validate_graph=True)
graph, metadata = converter.pydantic_list_to_graph(models)

# Validates:
# - No isolated nodes
# - Valid node IDs
# - Valid edge connections
# - No self-loops (optional)
```

### Manual Validation

```python
from docling_graph.core.utils import validate_graph_structure

try:
    validate_graph_structure(graph, raise_on_error=True)
    print("✅ Graph structure valid")
except ValueError as e:
    print(f"❌ Validation failed: {e}")
```

---

## Automatic Cleanup

### What Gets Cleaned?

Automatic cleanup removes:

1. **Phantom nodes** - Nodes with no data
2. **Duplicate nodes** - Same entity multiple times
3. **Orphaned edges** - Edges to non-existent nodes
4. **Empty attributes** - Null or empty values

### Configuration

```python
converter = GraphConverter(
    auto_cleanup=True  # Enable cleanup (default)
)

graph, metadata = converter.pydantic_list_to_graph(models)
```

### Manual Cleanup

```python
from docling_graph.core.utils import GraphCleaner

cleaner = GraphCleaner(verbose=True)
cleaned_graph = cleaner.clean_graph(graph)

print(f"Removed {graph.number_of_nodes() - cleaned_graph.number_of_nodes()} phantom nodes")
```

### What was lost when a phantom is removed

Removing a phantom node also drops every edge that pointed to or from it — which means a *relationship* was lost, not just a node. `GraphCleaner` records each dropped edge as `{"source", "label", "target"}`:

```python
cleaner.clean_graph(graph)
for rel in cleaner.last_dropped_relationships:
    print(f"{rel['source']} -[{rel['label']}]-> {rel['target']}")
```

The same list is stashed on the graph itself (`graph.graph["dropped_relationships"]`), which is how the markdown report's **Dropped Relationships** section surfaces it without extra plumbing — useful for spotting which specific relationship a run silently lost, not just how many.

---

## Complete Examples

### 📍 Basic Conversion

```python
from docling_graph.core import GraphConverter
from my_templates import BillingDocument, Organization, LineItem

# Create sample models
models = [
    BillingDocument(
        document_no="INV-001",
        issued_by=Organization(name="Acme Corp"),
        line_items=[
            LineItem(description="Product A", total=100),
            LineItem(description="Product B", total=200)
        ],
        total=300
    )
]

# Convert to graph
converter = GraphConverter()
graph, metadata = converter.pydantic_list_to_graph(models)

print(f"Nodes: {metadata.node_count}")
print(f"Edges: {metadata.edge_count}")
print(f"Node types: {metadata.node_types}")
```

### 📍 With Reverse Edges

```python
from docling_graph.core import GraphConverter

# Create bidirectional edges
converter = GraphConverter(add_reverse_edges=True)
graph, metadata = converter.pydantic_list_to_graph(models)

# Original edge: BillingDocument -> Organization (ISSUED_BY)
# Reverse edge: Organization -> Invoice (reverse_ISSUED_BY)

print(f"Total edges (with reverse): {metadata.edge_count}")
```

### 📍 Shared Registry for Batches

```python
from docling_graph.core import GraphConverter
from docling_graph.core.converters.node_id_registry import NodeIDRegistry

# Create shared registry
registry = NodeIDRegistry()

# Convert first batch
converter1 = GraphConverter(registry=registry)
graph1, _ = converter1.pydantic_list_to_graph(batch1_models)

# Convert second batch (same registry)
converter2 = GraphConverter(registry=registry)
graph2, _ = converter2.pydantic_list_to_graph(batch2_models)

# Same entities get same IDs across batches
print(f"Registry has {registry.get_stats()['total_entities']} unique entities")
```

### 📍 Custom Configuration

`add_reverse_edges` and `validate_graph` are set directly on `GraphConverter` — they are read from these constructor arguments, not from the `config` object's matching-named fields:

```python
from docling_graph.core import GraphConverter

converter = GraphConverter(add_reverse_edges=True, validate_graph=True)
graph, metadata = converter.pydantic_list_to_graph(models)
```

`GraphConfig` itself mainly carries fixed internal constants (`NODE_ID_HASH_LENGTH`, `MAX_STRING_LENGTH`, `TRUNCATE_SUFFIX`); pass a custom instance only if you need to override those:

```python
from docling_graph.core import GraphConverter, GraphConfig

converter = GraphConverter(config=GraphConfig(), add_reverse_edges=True)
```

---

## Graph Metadata

### Metadata Structure

`GraphMetadata` is a frozen Pydantic model (`docling_graph.core.converters.models`), not a plain dataclass:

```python
class GraphMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_count: int
    edge_count: int
    node_types: Dict[str, int] = {}
    edge_types: Dict[str, int] = {}
    created_at: datetime = ...  # defaults to now(), UTC
    source_models: int
    average_degree: float | None = None
```

### Using Metadata

```python
graph, metadata = converter.pydantic_list_to_graph(models)

print(f"Graph Statistics:")
print(f"  Nodes: {metadata.node_count}")
print(f"  Edges: {metadata.edge_count}")
print(f"  Source models: {metadata.source_models}")
if metadata.average_degree is not None:
    print(f"  Avg degree: {metadata.average_degree:.2f}")

print(f"\nNode Types:")
for node_type, count in metadata.node_types.items():
    print(f"  {node_type}: {count}")

print(f"\nEdge Types:")
for edge_type, count in metadata.edge_types.items():
    print(f"  {edge_type}: {count}")
```

---

## Advanced Features

### Reverse Edges

Create bidirectional relationships:

```python
converter = GraphConverter(add_reverse_edges=True)
graph, metadata = converter.pydantic_list_to_graph(models)

# For each edge A -> B, creates B -> A
# Useful for graph traversal in both directions
```

### Custom Node Identity

`NodeIDRegistry` has no public subclass hook for ID generation — identity is controlled entirely by `graph_id_fields` on the model's `model_config` (see [ID Generation](#id-generation) above), not by overriding registry internals:

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    model_config = {"graph_id_fields": ["invoice_number"]}  # drives the ID fingerprint

    invoice_number: str
    total: float
```

To share ID assignment across separate `GraphConverter` calls (e.g. batch runs), pass the **same** `NodeIDRegistry` instance — see [Shared Registry for Batches](#shared-registry-for-batches) above.

---

## Performance Optimization

### Batch Processing

```python
# Process large model lists efficiently
converter = GraphConverter(auto_cleanup=True)

# Convert in single call (efficient)
graph, metadata = converter.pydantic_list_to_graph(all_models)

# Don't convert one by one (inefficient)
# for model in models:
#     graph, _ = converter.pydantic_list_to_graph([model])
```

### Memory Management

```python
# For very large graphs
converter = GraphConverter(
    auto_cleanup=True,  # Remove unnecessary nodes
    validate_graph=False  # Skip validation for speed
)

graph, metadata = converter.pydantic_list_to_graph(models)

# NodeIDRegistry has no in-place reset; start a new one (and a new
# GraphConverter) for the next, unrelated document instead of reusing it.
```

---

## Troubleshooting

### 🐛 Empty Graph

**Solution:**
```python
# Check if models have entities
for model in models:
    if hasattr(model, 'model_config'):
        is_entity = model.model_config.get('is_entity', True)
        print(f"{model.__class__.__name__}: is_entity={is_entity}")
```

### 🐛 Missing Edges

**Solution:**
```python
# Ensure relationships are defined
class BillingDocument(BaseModel):
    issued_by: Organization  # Must be typed as entity
    # Not: issued_by: dict  # Won't create edge
```

### 🐛 Duplicate Nodes

**Solution:**
```python
# Enable auto cleanup
converter = GraphConverter(auto_cleanup=True)
graph, metadata = converter.pydantic_list_to_graph(models)
```

### 🐛 Validation Fails

**Solution:**
```python
# Check graph structure
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")

# Inspect nodes
for node_id, data in list(graph.nodes(data=True))[:5]:
    print(f"Node: {node_id}, Data: {data}")
```

---

## Best Practices

### 👍 Use Shared Registry for Batches

```python
# ✅ Good - Consistent IDs across batches
registry = NodeIDRegistry()

for batch in batches:
    converter = GraphConverter(registry=registry)
    graph, _ = converter.pydantic_list_to_graph(batch)
```

### 👍 Enable Auto Cleanup

```python
# ✅ Good - Clean graphs
converter = GraphConverter(auto_cleanup=True)
```

### 👍 Validate in Development

```python
# ✅ Good - Catch issues early
converter = GraphConverter(validate_graph=True)
```

### 👍 Disable Validation in Production

```python
# ✅ Good - Faster in production
converter = GraphConverter(validate_graph=False)
```

---

## Next Steps

Now that you understand graph conversion:

1. **[Data Grounding & Provenance](provenance.md)** - Trace nodes back to source chunks and pages
2. **[Export Formats](export-formats.md)** - Export graphs to CSV, Cypher, JSON
3. **[Visualization](visualization.md)** - Visualize your graphs
4. **[Neo4j Integration](neo4j-integration.md)** - Import into Neo4j