# Custom Exporters


## Overview

Create custom exporters to output knowledge graphs in specialized formats for your specific use case or database system.

**What You'll Learn:**
- Exporter protocol implementation
- Graph data access
- Custom format generation
- Registration and usage
- Testing strategies

**Prerequisites:**
- Understanding of [Graph Management](../../fundamentals/graph-management/index.md)
- Familiarity with NetworkX graphs
- Knowledge of target output format

---

## Exporter Protocol

All exporters must implement the `BaseExporter` protocol:

```python
from pathlib import Path
from typing import Any
import networkx as nx

class BaseExporter:
    """Base class for graph exporters."""
    
    def __init__(self, graph: nx.MultiDiGraph, output_dir: Path):
        """
        Initialize exporter.
        
        Args:
            graph: NetworkX graph to export
            output_dir: Directory for output files
        """
        self.graph = graph
        self.output_dir = output_dir
    
    def export(self) -> None:
        """Export the graph to the target format."""
        raise NotImplementedError
```

---

## Complete Exporter Example

### GraphML Exporter

```python
"""
Custom exporter for GraphML format.
GraphML is an XML-based format for graphs.
"""

from pathlib import Path
from typing import Any
import networkx as nx
from docling_graph.core.exporters.base import BaseExporter
from docling_graph.exceptions import GraphError

class GraphMLExporter(BaseExporter):
    """
    Export knowledge graph to GraphML format.
    
    GraphML is widely supported by graph visualization tools
    like Gephi, Cytoscape, and yEd.
    
    Args:
        graph: NetworkX graph to export
        output_dir: Directory for output files
        pretty_print: Whether to format XML with indentation
    """
    
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        output_dir: Path,
        pretty_print: bool = True
    ):
        super().__init__(graph, output_dir)
        self.pretty_print = pretty_print
    
    def export(self) -> None:
        """
        Export graph to GraphML format.
        
        Creates a .graphml file in the output directory.
        
        Raises:
            GraphError: If export fails
        """
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Define output path
            output_path = self.output_dir / "graph.graphml"
            
            # Export using NetworkX
            nx.write_graphml(
                self.graph,
                str(output_path),
                prettyprint=self.pretty_print
            )
            
            print(f"✅ GraphML exported to {output_path}")
            
        except Exception as e:
            raise GraphError(
                "GraphML export failed",
                details={"output_dir": str(self.output_dir)},
                cause=e
            )
    
    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics for the export."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "edge_types": self._count_edge_types()
        }
    
    def _count_node_types(self) -> dict[str, int]:
        """Count nodes by type."""
        type_counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts
    
    def _count_edge_types(self) -> dict[str, int]:
        """Count edges by type."""
        type_counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get("type", "Unknown")
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        return type_counts
```

### Usage

```python
"""Use custom GraphML exporter."""

from pathlib import Path
import networkx as nx
from my_exporters import GraphMLExporter

# Assume you have a graph from the pipeline
graph: nx.MultiDiGraph = ...  # From pipeline

# Create exporter
exporter = GraphMLExporter(
    graph=graph,
    output_dir=Path("outputs/graphml"),
    pretty_print=True
)

# Export
exporter.export()

# Get statistics
stats = exporter.get_statistics()
print(f"Exported {stats['num_nodes']} nodes and {stats['num_edges']} edges")
```

---

## Advanced Exporter Example

### RDF/Turtle Exporter

```python
"""
Export knowledge graph to RDF Turtle format.
Useful for semantic web applications and triple stores.
"""

from pathlib import Path
from typing import Any
import networkx as nx
from docling_graph.core.exporters.base import BaseExporter
from docling_graph.exceptions import GraphError

class TurtleExporter(BaseExporter):
    """
    Export knowledge graph to RDF Turtle format.
    
    Args:
        graph: NetworkX graph to export
        output_dir: Directory for output files
        namespace: Base namespace URI for entities
        prefixes: Additional namespace prefixes
    """
    
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        output_dir: Path,
        namespace: str = "http://example.org/kg/",
        prefixes: dict[str, str] | None = None
    ):
        super().__init__(graph, output_dir)
        self.namespace = namespace
        self.prefixes = prefixes or {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#"
        }
    
    def export(self) -> None:
        """Export graph to Turtle format."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / "graph.ttl"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write prefixes
                self._write_prefixes(f)
                f.write("\n")
                
                # Write nodes (entities)
                self._write_nodes(f)
                f.write("\n")
                
                # Write edges (relationships)
                self._write_edges(f)
            
            print(f"✅ Turtle RDF exported to {output_path}")
            
        except Exception as e:
            raise GraphError(
                "Turtle export failed",
                details={"output_dir": str(self.output_dir)},
                cause=e
            )
    
    def _write_prefixes(self, f: Any) -> None:
        """Write namespace prefixes."""
        f.write(f"@prefix : <{self.namespace}> .\n")
        for prefix, uri in self.prefixes.items():
            f.write(f"@prefix {prefix}: <{uri}> .\n")
    
    def _write_nodes(self, f: Any) -> None:
        """Write node definitions."""
        for node_id, data in self.graph.nodes(data=True):
            # Create URI for node
            node_uri = self._create_uri(node_id)
            
            # Write type
            node_type = data.get("type", "Entity")
            f.write(f"{node_uri} rdf:type :{node_type} ;\n")
            
            # Write properties
            properties = []
            for key, value in data.items():
                if key not in ["type", "id"]:
                    prop_line = self._format_property(key, value)
                    if prop_line:
                        properties.append(prop_line)
            
            # Write properties with proper punctuation
            for i, prop in enumerate(properties):
                if i < len(properties) - 1:
                    f.write(f"    {prop} ;\n")
                else:
                    f.write(f"    {prop} .\n")
            
            f.write("\n")
    
    def _write_edges(self, f: Any) -> None:
        """Write edge definitions."""
        for source, target, data in self.graph.edges(data=True):
            source_uri = self._create_uri(source)
            target_uri = self._create_uri(target)
            edge_type = data.get("type", "relatedTo")
            
            f.write(f"{source_uri} :{edge_type} {target_uri} .\n")
    
    def _create_uri(self, node_id: str) -> str:
        """Create URI for a node."""
        # Clean node ID for URI
        clean_id = node_id.replace(" ", "_").replace("/", "_")
        return f":{clean_id}"
    
    def _format_property(self, key: str, value: Any) -> str | None:
        """Format a property for Turtle output."""
        if value is None:
            return None
        
        # Handle different value types
        if isinstance(value, bool):
            return f':{key} "{str(value).lower()}"^^xsd:boolean'
        elif isinstance(value, int):
            return f':{key} "{value}"^^xsd:integer'
        elif isinstance(value, float):
            return f':{key} "{value}"^^xsd:decimal'
        elif isinstance(value, str):
            # Escape quotes in strings
            escaped = value.replace('"', '\\"')
            return f':{key} "{escaped}"'
        else:
            # Convert to string for other types
            return f':{key} "{str(value)}"'
```

---

## Integration with Pipeline

### Method 1: Post-Processing

```python
"""Export after pipeline completes."""

from pathlib import Path
from docling_graph import PipelineConfig
from my_exporters import GraphMLExporter, TurtleExporter

# Run pipeline
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    output_dir="outputs"
)
config.run()

# Load the generated graph
import json
graph_path = Path("outputs/graph.json")
with open(graph_path) as f:
    graph_data = json.load(f)

# Convert to NetworkX graph
import networkx as nx
graph = nx.node_link_graph(graph_data)

# Export to custom formats
GraphMLExporter(graph, Path("outputs/graphml")).export()
TurtleExporter(graph, Path("outputs/turtle")).export()
```

### Method 2: Custom Pipeline Stage

```python
"""Add custom export as pipeline stage."""

from docling_graph.pipeline.stages import PipelineStage
from docling_graph.pipeline.context import PipelineContext
from my_exporters import GraphMLExporter

class CustomExportStage(PipelineStage):
    """Custom export stage."""
    
    def execute(self, context: PipelineContext) -> None:
        """Execute custom export."""
        if context.graph is None:
            return
        
        # Export to GraphML
        exporter = GraphMLExporter(
            graph=context.graph,
            output_dir=context.output_dir / "graphml"
        )
        exporter.export()
        
        print("✅ Custom export complete")

# Use in custom pipeline orchestration
# (Requires modifying pipeline code)
```

---

## Testing Custom Exporters

### Unit Tests

```python
"""Test custom exporter."""

import pytest
from pathlib import Path
import networkx as nx
from my_exporters import GraphMLExporter

@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    G = nx.MultiDiGraph()
    
    # Add nodes
    G.add_node("person_1", type="Person", name="John", age=30)
    G.add_node("org_1", type="Organization", name="ACME Corp")
    
    # Add edge
    G.add_edge("person_1", "org_1", type="WORKS_AT")
    
    return G

def test_exporter_initialization(sample_graph, tmp_path):
    """Test exporter can be initialized."""
    exporter = GraphMLExporter(
        graph=sample_graph,
        output_dir=tmp_path
    )
    assert exporter.graph == sample_graph
    assert exporter.output_dir == tmp_path

def test_export_creates_file(sample_graph, tmp_path):
    """Test export creates output file."""
    exporter = GraphMLExporter(
        graph=sample_graph,
        output_dir=tmp_path
    )
    exporter.export()
    
    output_file = tmp_path / "graph.graphml"
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_export_valid_format(sample_graph, tmp_path):
    """Test exported file is valid GraphML."""
    exporter = GraphMLExporter(
        graph=sample_graph,
        output_dir=tmp_path
    )
    exporter.export()
    
    # Try to read it back
    output_file = tmp_path / "graph.graphml"
    loaded_graph = nx.read_graphml(str(output_file))
    
    assert loaded_graph.number_of_nodes() == 2
    assert loaded_graph.number_of_edges() == 1

def test_statistics(sample_graph, tmp_path):
    """Test statistics generation."""
    exporter = GraphMLExporter(
        graph=sample_graph,
        output_dir=tmp_path
    )
    
    stats = exporter.get_statistics()
    
    assert stats["num_nodes"] == 2
    assert stats["num_edges"] == 1
    assert "Person" in stats["node_types"]
    assert "Organization" in stats["node_types"]
```

---

## Best Practices

### 1. Handle Errors Gracefully

```python
# ✅ Good - Structured error handling
from docling_graph.exceptions import GraphError

def export(self):
    try:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Export logic...
    except IOError as e:
        raise GraphError("File write failed", cause=e)
    except Exception as e:
        raise GraphError("Export failed", cause=e)

# ❌ Avoid - Silent failures
def export(self):
    try:
        # Export logic...
        pass
    except:
        pass  # Error ignored!
```

### 2. Validate Graph Data

```python
# ✅ Good - Validate before export
def export(self):
    if self.graph.number_of_nodes() == 0:
        raise GraphError("Cannot export empty graph")
    
    # Check for required attributes
    for node_id, data in self.graph.nodes(data=True):
        if "type" not in data:
            raise GraphError(
                f"Node {node_id} missing 'type' attribute"
            )
    
    # Proceed with export...

# ❌ Avoid - No validation
def export(self):
    # Export without checks
    pass
```

### 3. Provide Progress Feedback

```python
# ✅ Good - Progress updates
def export(self):
    total_nodes = self.graph.number_of_nodes()
    print(f"Exporting {total_nodes} nodes...")
    
    # Export nodes
    for i, (node_id, data) in enumerate(self.graph.nodes(data=True)):
        self._export_node(node_id, data)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total_nodes} nodes")
    
    print("✅ Export complete")

# ❌ Avoid - No feedback
def export(self):
    # Silent export
    pass
```

### 4. Make Exporters Configurable

```python
# ✅ Good - Configurable options
class MyExporter(BaseExporter):
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        output_dir: Path,
        include_metadata: bool = True,
        compress: bool = False,
        encoding: str = "utf-8"
    ):
        super().__init__(graph, output_dir)
        self.include_metadata = include_metadata
        self.compress = compress
        self.encoding = encoding

# ❌ Avoid - Hardcoded behavior
class MyExporter(BaseExporter):
    def __init__(self, graph, output_dir):
        super().__init__(graph, output_dir)
        # No configuration options
```

---

## Common Export Formats

### JSON-LD

```python
"""Export to JSON-LD for semantic web."""

def export_jsonld(self) -> None:
    """Export to JSON-LD format."""
    output = {
        "@context": {
            "@vocab": self.namespace,
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        },
        "@graph": []
    }
    
    # Add nodes
    for node_id, data in self.graph.nodes(data=True):
        node_obj = {
            "@id": node_id,
            "@type": data.get("type", "Entity")
        }
        # Add properties
        for key, value in data.items():
            if key not in ["type", "id"]:
                node_obj[key] = value
        output["@graph"].append(node_obj)
    
    # Write to file
    import json
    output_path = self.output_dir / "graph.jsonld"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
```

### DOT (Graphviz)

```python
"""Export to DOT format for Graphviz."""

def export_dot(self) -> None:
    """Export to DOT format."""
    from networkx.drawing.nx_pydot import write_dot
    
    output_path = self.output_dir / "graph.dot"
    write_dot(self.graph, str(output_path))
    
    print(f"✅ DOT exported to {output_path}")
    print("  Visualize with: dot -Tpng graph.dot -o graph.png")
```

---

## Next Steps

1. **[Custom Stages →](custom-stages.md)** - Add pipeline stages
2. **[Testing →](testing.md)** - Test your exporter
3. **[Graph Management →](../../fundamentals/graph-management/index.md)** - Learn about graphs

---

## Related Documentation

- **[Export Formats](../../fundamentals/graph-management/export-formats.md)** - Built-in formats
- **[Graph Conversion](../../fundamentals/graph-management/graph-conversion.md)** - How graphs are built
- **[Exporters API](../../reference/exporters.md)** - API reference