# Exporters API


## Overview

Graph export formats for knowledge graphs.

**Module:** `docling_graph.core.exporters`

---

## Base Exporter

### BaseExporter

Base class for all exporters.

```python
class BaseExporter:
    """Base class for graph exporters."""
    
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        output_dir: Path
    ):
        """
        Initialize exporter.
        
        Args:
            graph: NetworkX graph to export
            output_dir: Output directory
        """
        self.graph = graph
        self.output_dir = output_dir
    
    def export(self) -> None:
        """Export graph to target format."""
        raise NotImplementedError
```

---

## CSV Exporter

### CSVExporter

Export graphs to Neo4j-compatible CSV format.

```python
class CSVExporter(BaseExporter):
    """Export graph to CSV format."""
    
    def export(self) -> None:
        """
        Export to CSV files.
        
        Creates:
            - nodes.csv: Node data
            - edges.csv: Edge data
        """
```

**Output Format:**

**nodes.csv:**
```csv
id,label,type,property1,property2,...
node_1,John Doe,Person,30,john@example.com
```

**edges.csv:**
```csv
source,target,type
node_1,node_2,WORKS_AT
```

**Example:**

```python
from docling_graph.core.exporters import CSVExporter
from pathlib import Path

exporter = CSVExporter(graph, Path("outputs"))
exporter.export()

# Files created:
# - outputs/nodes.csv
# - outputs/edges.csv
```

---

## Cypher Exporter

### CypherExporter

Export graphs to Cypher script format.

```python
class CypherExporter(BaseExporter):
    """Export graph to Cypher script."""
    
    def export(self) -> None:
        """
        Export to Cypher script.
        
        Creates:
            - graph.cypher: Cypher CREATE statements
        """
```

**Output Format:**

```cypher
CREATE (n1:Person {name: "John Doe", age: 30})
CREATE (n2:Organization {name: "ACME Corp"})
CREATE (n1)-[:WORKS_AT]->(n2)
```

**Example:**

```python
from docling_graph.core.exporters import CypherExporter
from pathlib import Path

exporter = CypherExporter(graph, Path("outputs"))
exporter.export()

# File created: outputs/graph.cypher
```

---

## JSON Exporter

### JSONExporter

Export graphs to JSON format.

```python
class JSONExporter(BaseExporter):
    """Export graph to JSON format."""
    
    def export(self) -> None:
        """
        Export to JSON.
        
        Creates:
            - graph.json: NetworkX node-link format
        """
```

**Output Format:**

```json
{
  "directed": true,
  "multigraph": true,
  "nodes": [
    {"id": "node_1", "type": "Person", "name": "John"}
  ],
  "links": [
    {"source": "node_1", "target": "node_2", "type": "WORKS_AT"}
  ]
}
```

**Example:**

```python
from docling_graph.core.exporters import JSONExporter
from pathlib import Path

exporter = JSONExporter(graph, Path("outputs"))
exporter.export()

# File created: outputs/graph.json
```

---

## Docling Exporter

### DoclingExporter

Export Docling document outputs.

```python
class DoclingExporter:
    """Export Docling document outputs."""
    
    def export(
        self,
        document: Any,
        output_dir: Path,
        export_json: bool = True,
        export_markdown: bool = True,
        export_per_page: bool = False
    ) -> None:
        """
        Export Docling outputs.
        
        Args:
            document: Docling document
            output_dir: Output directory
            export_json: Export JSON
            export_markdown: Export markdown
            export_per_page: Export per-page markdown
        """
```

**Creates:**
- `docling_document.json` - Docling JSON
- `markdown/full_document.md` - Full markdown
- `markdown/pages/page_N.md` - Per-page markdown

---

## Custom Exporters

Create custom exporters by extending `BaseExporter`:

```python
from docling_graph.core.exporters import BaseExporter
from pathlib import Path
import networkx as nx

class MyExporter(BaseExporter):
    """Custom exporter."""
    
    def export(self) -> None:
        """Export to custom format."""
        output_file = self.output_dir / "custom.txt"
        
        with open(output_file, 'w') as f:
            f.write(f"Nodes: {self.graph.number_of_nodes()}\n")
            f.write(f"Edges: {self.graph.number_of_edges()}\n")
```

See [Custom Exporters Guide](../usage/advanced/custom-exporters.md) for details.

---

## Related APIs

- **[Export Formats](../fundamentals/graph-management/export-formats.md)** - Usage guide
- **[Custom Exporters](../usage/advanced/custom-exporters.md)** - Create exporters
- **[Converters](converters.md)** - Graph conversion