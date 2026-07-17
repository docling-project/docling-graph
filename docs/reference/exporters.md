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
class JSONExporter:
    """Export graph to JSON format."""

    def __init__(self, config: ExportConfig | None = None) -> None:
        """
        Initialize exporter.

        Args:
            config: Export configuration. Uses defaults if None.
        """

    def export(self, graph: nx.DiGraph, output_path: Path) -> None:
        """
        Export graph to JSON.

        Args:
            graph: NetworkX directed graph to export.
            output_path: File path where to save JSON.

        Raises:
            ValueError: If graph is empty.
        """
```

**Output Format:**

Docling-graph's own shape — *not* NetworkX node-link. Edges live under `edges` (not `links`), and graph-level metadata (the format-v2 identity contract: `id_fields_map`, template name/schema hash) lives under a top-level `graph` key, absent on v1 exports.

```json
{
  "nodes": [
    {"id": "node_1", "type": "entity", "label": "John"}
  ],
  "edges": [
    {"source": "node_1", "target": "node_2", "label": "works_at"}
  ],
  "metadata": {"node_count": 2, "edge_count": 1},
  "graph": {"format": "v2"}
}
```

**Example:**

```python
from docling_graph.core.exporters import JSONExporter
from pathlib import Path

exporter = JSONExporter()
exporter.export(graph, Path("outputs/graph.json"))
```

---

### graph_to_dict

Serialize a graph to the canonical `graph.json` shape without touching the filesystem.

```python
def graph_to_dict(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Serialize a graph to the canonical docling-graph JSON shape.

    Args:
        graph: NetworkX directed graph.

    Returns:
        {"nodes": [...], "edges": [...], "metadata": {...}, "graph": {...}}
    """
```

The round-trip inverse is [`load_graph_from_dict()`](#load_graph_from_dict). Use the pair when the graph or payload is already in memory (an HTTP handler, a notebook) and a temp file would be pure overhead.

**Example:**

```python
from docling_graph.core.exporters import graph_to_dict
from docling_graph.core.importers import load_graph_from_dict

payload = graph_to_dict(graph)
restored = load_graph_from_dict(payload)
```

!!! warning "Output holds live Python values"

    The returned dict is not directly JSON-encodable — a `date` stays a `date`, a `Decimal` stays a `Decimal`. Pass `json_serializable` as `default=`, the way `JSONExporter` does internally:

    ```python
    import json
    from docling_graph.core.utils.string_formatter import json_serializable

    encoded = json.dumps(graph_to_dict(graph), default=json_serializable)
    ```

    Encoding flattens those values (`date` → ISO string, `Decimal` → float); passing the dict straight to `load_graph_from_dict()` keeps them live.

---

## Graph Importer

### load_graph_from_dict

Load an in-memory `graph.json` object back into a graph. The read-side inverse of [`graph_to_dict()`](#graph_to_dict).

**Module:** `docling_graph.core.importers`

```python
def load_graph_from_dict(data: dict, *, source: str = "<dict>") -> nx.DiGraph:
    """
    Load an already-parsed graph.json object into an nx.DiGraph.

    Args:
        data: A docling-graph graph export object.
        source: Label used in error messages (a path, URL, or "<dict>").

    Raises:
        ConfigurationError: When data is not a docling-graph export, is empty,
            or is structurally corrupt (malformed records, dangling edges).
    """
```

Validation is identical to the file-based `load_graph_json()`, which is a thin wrapper over this function. Node `id`s must be JSON scalars; graph-level metadata under `graph` is restored when present and tolerated when absent (v1 exports).

**Example:**

```python
from docling_graph.core.importers import load_graph_from_dict
from docling_graph.exceptions import ConfigurationError

try:
    graph = load_graph_from_dict(request_body, source="POST /graphs")
except ConfigurationError as e:
    print(f"Rejected: {e.message} ({e.details})")
```

**Related file-based loaders** (`docling_graph.core.importers`):

| Function | Description |
|:---------|:------------|
| `load_graph_json(path)` | Read and parse one `graph.json` file into a graph |
| `resolve_graph_path(path)` | Resolve a directory to its `graph.json` (rejects lossy CSV/Cypher) |
| `load_sibling_ledger(path)` | Load the `provenance.json` written next to `graph.json` |
| `load_graph_input(path)` | Resolve + load one merge input: `(graph, ledger \| None, path)` |

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