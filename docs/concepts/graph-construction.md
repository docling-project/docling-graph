# Graph Construction

This document explains how Docling Graph converts validated Pydantic models into NetworkX directed graphs with semantic relationships.

## Overview

Graph construction is the process of transforming structured Pydantic objects into a knowledge graph where:

- **Nodes** represent entities and components
- **Edges** represent relationships between nodes
- **Attributes** store rich metadata on nodes and edges

**Location**: `docling_graph/core/converters/graph_converter.py`

## Core Concepts

### Nodes

Every Pydantic model instance becomes a node in the graph:

```python
class Person(BaseModel):
    model_config = ConfigDict(
        is_entity=True,
        graph_id_fields=["name", "date_of_birth"]
    )
    name: str
    date_of_birth: str
    email: str

# Creates node:
# ID: "Person_JohnDoe_1990-01-15"
# Attributes: {name: "John Doe", date_of_birth: "1990-01-15", email: "john@example.com"}
```

### Edges

Relationships defined with the `edge()` helper become graph edges:

```python
class Document(BaseModel):
    issued_by: Organization = edge(
        label="ISSUED_BY",
        description="Organization that issued this document"
    )

# Creates edge:
# Source: Document node
# Target: Organization node
# Label: "ISSUED_BY"
```

### Node IDs

Node IDs are generated using two strategies:

#### 1. Entity Nodes (with graph_id_fields)

```python
class Person(BaseModel):
    model_config = ConfigDict(
        graph_id_fields=["first_name", "last_name", "dob"]
    )
    first_name: str = "John"
    last_name: str = "Doe"
    dob: str = "1990-01-15"

# Generated ID: "Person_John_Doe_1990-01-15"
# Format: {ClassName}_{field1}_{field2}_{field3}
```

#### 2. Component Nodes (content-based)

```python
class Address(BaseModel):
    model_config = ConfigDict(is_entity=False)
    street: str = "123 Main St"
    city: str = "Boston"

# Generated ID: hash of all field values
# Format: {ClassName}_{content_hash}
# Same content → Same ID (deduplication)
```

## Graph Converter

### Basic Usage

```python
from docling_graph.core import GraphConverter

# Create converter
converter = GraphConverter(
    add_reverse_edges=False,
    validate_graph=True
)

# Convert models to graph
models = [person1, person2, organization]
graph, metadata = converter.pydantic_list_to_graph(models)

# Result:
# - graph: NetworkX DiGraph
# - metadata: GraphMetadata with statistics
```

### Configuration Options

```python
class GraphConverter:
    def __init__(
        self,
        add_reverse_edges: bool = False,  # Add bidirectional edges
        validate_graph: bool = True,      # Validate structure
        registry: NodeIDRegistry = None   # Shared ID registry
    ):
        ...
```

#### add_reverse_edges

Creates bidirectional relationships:

```python
# Original edge
Document --ISSUED_BY--> Organization

# With add_reverse_edges=True
Document --ISSUED_BY--> Organization
Document <--ISSUES-- Organization
```

#### validate_graph

Performs structural validation:
- Checks for orphaned nodes
- Validates edge connectivity
- Ensures ID uniqueness

#### registry

Shared registry for deterministic IDs across batches:

```python
# Create shared registry
registry = NodeIDRegistry()

# Use in multiple conversions
converter1 = GraphConverter(registry=registry)
converter2 = GraphConverter(registry=registry)

# Same entities get same IDs across conversions
```

## Node ID Registry

**Location**: `docling_graph/core/converters/node_id_registry.py`

The registry ensures stable, deterministic node IDs:

### Features

- **Deterministic**: Same input → same ID
- **Collision Detection**: Prevents ID conflicts
- **Cross-Batch Consistency**: Maintains IDs across multiple extractions
- **Type-Safe**: Tracks node types and metadata

### Usage

```python
from docling_graph.core.converters import NodeIDRegistry

registry = NodeIDRegistry()

# Register a node
node_id = registry.register_node(
    model_instance=person,
    model_class=Person,
    id_fields=["name", "dob"]
)

# Check if node exists
if registry.has_node(node_id):
    print("Node already registered")

# Get node metadata
metadata = registry.get_node_metadata(node_id)
```

## Entity vs Component

### Entity Nodes

**Characteristics**:
- Unique, identifiable objects
- Use `graph_id_fields` for stable IDs
- Track individually in the graph

**Configuration**:
```python
class Person(BaseModel):
    model_config = ConfigDict(
        is_entity=True,  # Explicit (optional)
        graph_id_fields=["name", "dob"]
    )
```

**Example**:
```python
# Two persons with same name but different DOB
person1 = Person(name="John Doe", dob="1990-01-15")
person2 = Person(name="John Doe", dob="1985-05-20")

# Result: 2 separate nodes
# - Person_JohnDoe_1990-01-15
# - Person_JohnDoe_1985-05-20
```

### Component Nodes

**Characteristics**:
- Value objects
- Deduplicated by content
- Shared across entities

**Configuration**:
```python
class Address(BaseModel):
    model_config = ConfigDict(is_entity=False)
    street: str
    city: str
```

**Example**:
```python
# Two persons at same address
address = Address(street="123 Main St", city="Boston")
person1 = Person(name="John", address=address)
person2 = Person(name="Jane", address=address)

# Result: 3 nodes
# - Person_John
# - Person_Jane
# - Address_{hash} (shared by both persons)
```

## Edge Creation

### Edge Definition

Use the `edge()` helper in Pydantic models:

```python
def edge(label: str, **kwargs: Any) -> Any:
    """Create a Field with edge metadata."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)
```

### Single Relationships

```python
class Document(BaseModel):
    # Required single edge
    issued_by: Organization = edge(
        label="ISSUED_BY",
        description="Issuing organization"
    )
    
    # Optional single edge
    verified_by: Optional[Person] = edge(
        label="VERIFIED_BY",
        description="Verifying person"
    )
```

### List Relationships

```python
class Document(BaseModel):
    # One-to-many relationship
    authors: List[Person] = edge(
        label="HAS_AUTHOR",
        default_factory=list,
        description="Document authors"
    )
```

### Edge Labels

**Conventions**:
- Use ALL_CAPS with underscores
- Use descriptive verb phrases
- Be consistent across templates

**Common patterns**:
```python
# Authorship/Ownership
ISSUED_BY, CREATED_BY, OWNED_BY

# Recipients
SENT_TO, ADDRESSED_TO, DELIVERED_TO

# Location
LOCATED_AT, LIVES_AT, BASED_AT

# Composition
CONTAINS_ITEM, HAS_COMPONENT, INCLUDES_PART

# Membership
BELONGS_TO, PART_OF, MEMBER_OF

# Processes
HAS_PROCESS_STEP, HAS_EVALUATION, HAS_MEASUREMENT
```

## Graph Structure

### Example Graph

Given these models:

```python
class Author(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str
    affiliation: str

class Paper(BaseModel):
    model_config = ConfigDict(graph_id_fields=["title"])
    title: str
    authors: List[Author] = edge(label="HAS_AUTHOR", default_factory=list)
    year: int

# Instances
author1 = Author(name="Dr. Smith", affiliation="MIT")
author2 = Author(name="Dr. Jones", affiliation="Stanford")
paper = Paper(
    title="Advanced AI",
    authors=[author1, author2],
    year=2024
)
```

Resulting graph:

```
Nodes:
  Paper_AdvancedAI
    - title: "Advanced AI"
    - year: 2024
  
  Author_DrSmith
    - name: "Dr. Smith"
    - affiliation: "MIT"
  
  Author_DrJones
    - name: "Dr. Jones"
    - affiliation: "Stanford"

Edges:
  Paper_AdvancedAI --HAS_AUTHOR--> Author_DrSmith
  Paper_AdvancedAI --HAS_AUTHOR--> Author_DrJones
```

### NetworkX Representation

```python
import networkx as nx

# Access nodes
for node_id, attrs in graph.nodes(data=True):
    print(f"Node: {node_id}")
    print(f"Attributes: {attrs}")

# Access edges
for source, target, attrs in graph.edges(data=True):
    print(f"Edge: {source} --{attrs['label']}--> {target}")

# Query graph
# Find all authors of a paper
paper_id = "Paper_AdvancedAI"
authors = list(graph.successors(paper_id))

# Find all papers by an author
author_id = "Author_DrSmith"
papers = [n for n in graph.predecessors(author_id)]
```

## Advanced Features

### Nested Relationships

```python
class Component(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str
    material: Material = edge(label="USES_MATERIAL")

class Assembly(BaseModel):
    model_config = ConfigDict(graph_id_fields=["id"])
    id: str
    components: List[Component] = edge(
        label="HAS_COMPONENT",
        default_factory=list
    )

# Creates multi-level graph:
# Assembly --HAS_COMPONENT--> Component --USES_MATERIAL--> Material
```

### Deduplication

Components with identical content are automatically deduplicated:

```python
# Same address used by multiple people
address = Address(street="123 Main St", city="Boston")

person1 = Person(name="John", address=address)
person2 = Person(name="Jane", address=address)

# Result: Only 1 Address node, shared by both persons
# Person_John --LIVES_AT--> Address_{hash}
# Person_Jane --LIVES_AT--> Address_{hash}
```

### Reverse Edges

Enable bidirectional traversal:

```python
converter = GraphConverter(add_reverse_edges=True)

# Original
Document --ISSUED_BY--> Organization

# With reverse edges
Document --ISSUED_BY--> Organization
Document <--ISSUES-- Organization

# Allows queries in both directions
# "What did Organization X issue?"
# "Who issued Document Y?"
```

## Graph Metadata

The converter returns metadata about the generated graph:

```python
@dataclass
class GraphMetadata:
    node_count: int           # Total nodes
    edge_count: int           # Total edges
    node_types: Dict[str, int]  # Count by type
    edge_types: Dict[str, int]  # Count by label
    source_model_count: int   # Input models
    
graph, metadata = converter.pydantic_list_to_graph(models)

print(f"Nodes: {metadata.node_count}")
print(f"Edges: {metadata.edge_count}")
print(f"Node types: {metadata.node_types}")
# Output: {'Person': 2, 'Organization': 1, 'Address': 1}
```

## Validation

### Automatic Validation

When `validate_graph=True`:

```python
converter = GraphConverter(validate_graph=True)

# Checks performed:
# 1. All edges connect to existing nodes
# 2. No orphaned nodes (unless root)
# 3. Node IDs are unique
# 4. Edge labels are present
```

### Manual Validation

```python
from docling_graph.core.utils import validate_graph_structure

# Validate after construction
is_valid, errors = validate_graph_structure(graph)

if not is_valid:
    for error in errors:
        print(f"Validation error: {error}")
```

## Performance Considerations

### Memory Usage

- **Small graphs** (< 1000 nodes): Negligible
- **Medium graphs** (1000-10000 nodes): ~10-50 MB
- **Large graphs** (> 10000 nodes): Consider batch processing

### Optimization Tips

1. **Reuse Registry**: Share `NodeIDRegistry` across batches
2. **Disable Validation**: For trusted data, set `validate_graph=False`
3. **Batch Processing**: Process large document sets in chunks
4. **Lazy Loading**: Don't load entire graph into memory if not needed

## Common Patterns

### Pattern 1: Document with Entities

```python
class Document(BaseModel):
    model_config = ConfigDict(graph_id_fields=["doc_id"])
    doc_id: str
    issuer: Organization = edge(label="ISSUED_BY")
    recipient: Person = edge(label="SENT_TO")
    items: List[LineItem] = edge(label="CONTAINS_ITEM", default_factory=list)
```

### Pattern 2: Hierarchical Structure

```python
class Section(BaseModel):
    model_config = ConfigDict(graph_id_fields=["title"])
    title: str
    subsections: List["Section"] = edge(label="HAS_SUBSECTION", default_factory=list)
```

### Pattern 3: Many-to-Many Relationships

```python
class Student(BaseModel):
    model_config = ConfigDict(graph_id_fields=["student_id"])
    student_id: str
    courses: List["Course"] = edge(label="ENROLLED_IN", default_factory=list)

class Course(BaseModel):
    model_config = ConfigDict(graph_id_fields=["course_id"])
    course_id: str
    students: List[Student] = edge(label="HAS_STUDENT", default_factory=list)
```

## Troubleshooting

### Issue: Duplicate Nodes

**Symptom**: Same entity appears multiple times

**Solution**: Ensure `graph_id_fields` are set correctly

```python
# Wrong: No graph_id_fields
class Person(BaseModel):
    name: str

# Right: With graph_id_fields
class Person(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str
```

### Issue: Missing Edges

**Symptom**: Relationships not appearing in graph

**Solution**: Use `edge()` helper for relationship fields

```python
# Wrong: Regular field
class Document(BaseModel):
    issuer: Organization

# Right: Edge field
class Document(BaseModel):
    issuer: Organization = edge(label="ISSUED_BY")
```

### Issue: ID Collisions

**Symptom**: Different entities get same ID

**Solution**: Add more fields to `graph_id_fields`

```python
# Collision risk: Only name
model_config = ConfigDict(graph_id_fields=["name"])

# Better: Name + DOB
model_config = ConfigDict(graph_id_fields=["name", "date_of_birth"])
```

## Next Steps

- Learn about [Export Formats](export-formats.md)
- Understand [Pydantic Templates](pydantic-templates.md)
- Explore [Configuration System](configuration.md)