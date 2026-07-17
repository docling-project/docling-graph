# Export Formats


## Overview

**Export formats** determine how your knowledge graph is saved and shared. Docling Graph supports CSV, Cypher, and JSON formats, each optimized for different use cases.

**In this guide:**

- CSV format (spreadsheets, analysis)
- Cypher format (Neo4j import)
- JSON format (programmatic access)
- Format selection criteria
- Integration examples

---

## Format Comparison

| Format | Best For | Output | Use Case |
|:-------|:---------|:-------|:---------|
| **CSV** | Analysis, spreadsheets | `nodes.csv`, `edges.csv` | Excel, Pandas, SQL |
| **Cypher** | Graph databases | `graph.cypher` | Neo4j import |
| **JSON** | APIs, processing | `graph.json` | Python, JavaScript |

---

## Provenance in Exports

When [data grounding](provenance.md) is enabled (the default, `provenance="standard"`), every entity node carries a `__provenance__` attribute. Each exporter serializes it appropriately for its format:

| Format | How `__provenance__` appears |
|--------|-------------------------------|
| **JSON** (`graph.json`) | Native nested object, same as any other node field. |
| **CSV** (`nodes.csv`) | A JSON **string** column — parse with `json.loads(row["__provenance__"])`. |
| **Cypher** (`graph.cypher`) | A string node property containing escaped JSON, e.g. `CREATE (n:Invoice {..., __provenance__: "{\"match\": \"verbatim\", ...}"})`. |

The full ledger — including chunk **text**, so you can trace a node straight to its source snippet — is always written separately as `provenance.json`, regardless of `export_format`. See [Data Grounding & Provenance](provenance.md) for the complete schema.

```python
import pandas as pd
import json

nodes = pd.read_csv("outputs/.../docling_graph/nodes.csv")
nodes["__provenance__"] = nodes["__provenance__"].apply(json.loads)

verbatim = nodes[nodes["__provenance__"].apply(lambda p: p.get("match") == "verbatim")]
print(f"{len(verbatim)} nodes grounded to an exact page")
```

---

## CSV Export

### What is CSV Export?

**CSV export** creates separate files for nodes and edges in comma-separated format, perfect for spreadsheet analysis and SQL databases.

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    export_format="csv",  # CSV export (default)
    output_dir="outputs"
)

run_pipeline(config)
```

### Output Files

```
outputs/{document}_{timestamp}/
├── metadata.json                # Pipeline metadata (results.nodes, results.edges, ...)
└── docling_graph/
    ├── nodes.csv                 # All nodes with properties
    ├── edges.csv                 # All edges with relationships
    ├── graph.json                # Same graph in JSON (always written alongside CSV/Cypher)
    ├── graph.html                # Interactive visualization
    └── report.md                 # Summary report
```

---

### nodes.csv Format

```csv
id,label,type,__class__,invoice_number,total,name,street,city
invoice_001,Invoice,entity,Invoice,INV-001,1000,,,
org_acme,Organization,entity,Organization,,,Acme Corp,,
addr_123,Address,entity,Address,,,,123 Main St,Paris
```

**Columns:**

- `id`: Unique node identifier
- `label`: Node type/class
- `type`: Always "entity"
- `__class__`: Python class name
- Additional columns for each property

---

### edges.csv Format

```csv
source,target,label
invoice_001,org_acme,issued_by
org_acme,addr_123,located_at
invoice_001,item_001,contains_item
```

**Columns:**

- `source`: Source node ID
- `target`: Target node ID
- `label`: Relationship type

---

### Manual CSV Export

```python
from pathlib import Path

from docling_graph.core.exporters import CSVExporter
from docling_graph.core import GraphConverter

# Convert models to graph
converter = GraphConverter()
graph, metadata = converter.pydantic_list_to_graph(models)

# Export to CSV
exporter = CSVExporter()
exporter.export(graph, Path("csv_output"))

print("Exported to csv_output/nodes.csv and csv_output/edges.csv")
```

---

### Using CSV with Pandas

```python
import pandas as pd

# Load CSV files
nodes = pd.read_csv("outputs/nodes.csv")
edges = pd.read_csv("outputs/edges.csv")

# Analyze nodes
print(f"Total nodes: {len(nodes)}")
print(f"Node types:\n{nodes['label'].value_counts()}")

# Analyze edges
print(f"Total edges: {len(edges)}")
print(f"Edge types:\n{edges['label'].value_counts()}")

# Filter specific node type
invoices = nodes[nodes['label'] == 'Invoice']
print(f"Found {len(invoices)} invoices")
```

---

### Using CSV with SQL

```python
import sqlite3
import pandas as pd

# Load CSV
nodes = pd.read_csv("outputs/nodes.csv")
edges = pd.read_csv("outputs/edges.csv")

# Create database
conn = sqlite3.connect("graph.db")

# Import to SQL
nodes.to_sql("nodes", conn, if_exists="replace", index=False)
edges.to_sql("edges", conn, if_exists="replace", index=False)

# Query
result = pd.read_sql("""
    SELECT n.label, COUNT(*) as count
    FROM nodes n
    GROUP BY n.label
""", conn)

print(result)
```

---

## Cypher Export

### What is Cypher Export?

**Cypher export** generates Cypher statements for direct import into Neo4j graph databases.

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    export_format="cypher",  # Cypher export
    output_dir="outputs"
)

run_pipeline(config)
```

### Output Files

```
outputs/{document}_{timestamp}/
├── metadata.json                # Pipeline metadata (results.nodes, results.edges, ...)
└── docling_graph/
    ├── graph.cypher               # Cypher statements
    ├── graph.json                 # Same graph in JSON (always written alongside CSV/Cypher)
    ├── graph.html                 # Interactive visualization
    └── report.md                  # Summary report
```

---

### graph.cypher Format

```cypher
// Cypher script generated by docling-graph
// Import this into Neo4j

// --- Create Nodes ---
CREATE (invoice_001:Invoice {invoice_number: "INV-001", total: 1000, node_id: "invoice_001"})
CREATE (org_acme:Organization {name: "Acme Corp", node_id: "org_acme"})
CREATE (addr_123:Address {street: "123 Main St", city: "Paris", node_id: "addr_123"})

// --- Create Relationships ---
MATCH (invoice_001), (org_acme)
CREATE (invoice_001)-[:ISSUED_BY]->(org_acme)

MATCH (org_acme), (addr_123)
CREATE (org_acme)-[:LOCATED_AT]->(addr_123)
```

---

### Manual Cypher Export

```python
from pathlib import Path

from docling_graph.core.exporters import CypherExporter
from docling_graph.core import GraphConverter

# Convert models to graph
converter = GraphConverter()
graph, metadata = converter.pydantic_list_to_graph(models)

# Export to Cypher
exporter = CypherExporter()
exporter.export(graph, Path("outputs/graph.cypher"))

print("Exported to outputs/graph.cypher")
```

---

### Importing to Neo4j

#### Method 1: cypher-shell

```bash
# Import using cypher-shell
cat outputs/graph.cypher | cypher-shell -u neo4j -p password

# Or with file
cypher-shell -u neo4j -p password -f outputs/graph.cypher
```

#### Method 2: Neo4j Browser

1. Open Neo4j Browser (http://localhost:7474)
2. Copy contents of `graph.cypher`
3. Paste into query editor
4. Execute

#### Method 3: Python Driver

```python
from neo4j import GraphDatabase

# Connect to Neo4j
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

# Read Cypher file
with open("outputs/graph.cypher") as f:
    cypher_script = f.read()

# Execute
with driver.session() as session:
    session.run(cypher_script)

driver.close()
print("Imported to Neo4j")
```

---

## JSON Export

### What is JSON Export?

**JSON export** (`graph.json`) is always written alongside CSV or Cypher, providing structured data for programmatic access. There is no separate "extracted models" JSON file — the models' field values are the node properties in `graph.json`.

### Output Files

```
outputs/{document}_{timestamp}/
├── metadata.json                 # Pipeline metadata (results.nodes, results.edges, ...)
└── docling_graph/
    └── graph.json                  # Nodes + edges + metadata (always written)
```

---

### graph.json Format

Node properties are **flat** — template field values sit directly on the node dict alongside `id`/`label`/`type`/`__class__`, not nested under a `"properties"` key:

```json
{
  "nodes": [
    {
      "id": "invoice_001",
      "label": "Invoice",
      "type": "entity",
      "__class__": "Invoice",
      "invoice_number": "INV-001",
      "total": 1000
    },
    {
      "id": "org_acme",
      "label": "Organization",
      "type": "entity",
      "__class__": "Organization",
      "name": "Acme Corp"
    }
  ],
  "edges": [
    {
      "source": "invoice_001",
      "target": "org_acme",
      "label": "issued_by"
    }
  ],
  "metadata": {
    "node_count": 2,
    "edge_count": 1
  }
}
```

---

### Manual JSON Export

```python
from pathlib import Path

from docling_graph.core.exporters import JSONExporter
from docling_graph.core import GraphConverter

# Convert models to graph
converter = GraphConverter()
graph, metadata = converter.pydantic_list_to_graph(models)

# Export to JSON
exporter = JSONExporter()
exporter.export(graph, Path("outputs/graph.json"))

print("Exported to outputs/graph.json")
```

---

### Using JSON in Python

```python
import json

# Load graph data
with open("outputs/.../docling_graph/graph.json") as f:
    graph_data = json.load(f)

# Access nodes (field values are flat on each node dict)
for node in graph_data["nodes"]:
    print(f"{node['label']}: {node['id']}")

# Access edges
for edge in graph_data["edges"]:
    print(f"{edge['source']} --[{edge['label']}]--> {edge['target']}")

# Filter by type
invoices = [n for n in graph_data["nodes"] if n["label"] == "Invoice"]
print(f"Found {len(invoices)} invoices")
```

---

### In-Memory Round Trip (no files)

`JSONExporter` writes to disk and `load_graph_json()` reads from disk. When you already hold the graph or the payload in memory — serving a graph over HTTP, accepting one in a request body — use the file-free pair instead of spooling through a temp file:

```python
from docling_graph.core.exporters import graph_to_dict
from docling_graph.core.importers import load_graph_from_dict

payload = graph_to_dict(graph)      # graph -> dict, same shape as graph.json
graph = load_graph_from_dict(payload)  # dict -> graph
```

`load_graph_from_dict()` accepts an optional `source=` label used in error messages (a path, URL, or the default `"<dict>"`), and raises `ConfigurationError` when the payload is not a docling-graph export, is empty, or is structurally corrupt (malformed records, dangling edge endpoints):

```python
from docling_graph.exceptions import ConfigurationError

try:
    graph = load_graph_from_dict(request_body, source="POST /graphs")
except ConfigurationError as e:
    print(f"Rejected: {e.message} ({e.details})")
```

!!! warning "`graph_to_dict()` output is not directly JSON-encodable"

    The returned dict holds **live** Python values — a `date` stays a `date`, a `Decimal` stays a `Decimal`. Pass `json_serializable` when encoding, exactly as `JSONExporter` does internally:

    ```python
    import json
    from docling_graph.core.utils.string_formatter import json_serializable

    encoded = json.dumps(graph_to_dict(graph), default=json_serializable)
    ```

    This matters for the round trip: handing the dict straight back to `load_graph_from_dict()` preserves the live objects, but going through JSON flattens them (`date` → ISO string, `Decimal` → float). The graph round-trips structurally either way; attribute *types* only survive if you skip the encode step.

---

## Format Selection

### Decision Matrix

| Use Case | Recommended Format | Reason |
|:---------|:------------------|:-------|
| **Excel analysis** | CSV | Direct import to Excel |
| **Neo4j database** | Cypher | Direct import |
| **Python processing** | JSON | Easy to parse |
| **SQL database** | CSV | Standard import |
| **Data science** | CSV | Pandas compatible |
| **API integration** | JSON | Standard format |
| **Graph queries** | Cypher | Neo4j native |

---

### By Tool

| Tool | Format | Import Method |
|:-----|:-------|:--------------|
| **Excel** | CSV | File → Open |
| **Neo4j** | Cypher | cypher-shell |
| **Python** | JSON | json.load() |
| **Pandas** | CSV | pd.read_csv() |
| **SQL** | CSV | COPY/LOAD DATA |
| **Power BI** | CSV | Get Data |
| **Tableau** | CSV | Connect to File |

---

## Complete Examples

### 📍 CSV for Analysis

```python
from docling_graph import run_pipeline, PipelineConfig
import pandas as pd

# Extract and export to CSV
config = PipelineConfig(
    source="invoices.pdf",
    template="templates.BillingDocument",
    export_format="csv",
    output_dir="analysis"
)

run_pipeline(config)

# Analyze with Pandas
nodes = pd.read_csv("analysis/nodes.csv")
edges = pd.read_csv("analysis/edges.csv")

# Calculate statistics
print(f"Total invoices: {len(nodes[nodes['label'] == 'Invoice'])}")
print(f"Total organizations: {len(nodes[nodes['label'] == 'Organization'])}")
print(f"Total relationships: {len(edges)}")

# Export summary
summary = nodes.groupby('label').size()
summary.to_csv("analysis/summary.csv")
```

### 📍 Cypher for Neo4j

```python
from docling_graph import run_pipeline, PipelineConfig
import subprocess

# Extract and export to Cypher
config = PipelineConfig(
    source="contracts.pdf",
    template="templates.Contract",
    export_format="cypher",
    output_dir="neo4j_import"
)

run_pipeline(config)

# Import to Neo4j
result = subprocess.run([
    "cypher-shell",
    "-u", "neo4j",
    "-p", "password",
    "-f", "neo4j_import/graph.cypher"
], capture_output=True, text=True)

if result.returncode == 0:
    print("✅ Successfully imported to Neo4j")
else:
    print(f"❌ Import failed: {result.stderr}")
```

### 📍 JSON for API

```python
from docling_graph import run_pipeline, PipelineConfig
import json
import requests

# Extract and export
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    export_format="csv",  # JSON always generated alongside it
    dump_to_disk=True,
    output_dir="api_data"
)

context = run_pipeline(config)

# Load the written graph.json (or read context.knowledge_graph directly, no file needed)
graph_json_path = context.output_manager.get_docling_graph_dir() / "graph.json"
with open(graph_json_path) as f:
    data = json.load(f)

# Send to API
response = requests.post(
    "https://api.example.com/invoices",
    json=data,
    headers={"Content-Type": "application/json"}
)

print(f"API response: {response.status_code}")
```

---

## Best Practices

### 👍 Choose Format by Use Case

```python
# ✅ Good - Match format to use case
if use_case == "neo4j":
    export_format = "cypher"
elif use_case == "analysis":
    export_format = "csv"
else:
    export_format = "csv"  # Default
```

### 👍 Organize Output Directories

```python
# ✅ Good - Structured outputs
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"exports/{export_format}/{timestamp}"

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    export_format=export_format,
    output_dir=output_dir
)
```

### 👍 Validate Exports

```python
# ✅ Good - Check exports exist
import os

run_pipeline(config)

if export_format == "csv":
    assert os.path.exists(f"{output_dir}/nodes.csv")
    assert os.path.exists(f"{output_dir}/edges.csv")
elif export_format == "cypher":
    assert os.path.exists(f"{output_dir}/graph.cypher")

print("✅ Exports validated")
```

---

## Troubleshooting

### 🐛 Empty CSV Files

**Solution:**
```python
# Check if graph has nodes (metadata.json is at the document dir root,
# one level above docling_graph/)
import json

with open("outputs/.../metadata.json") as f:
    metadata = json.load(f)

if metadata["results"]["nodes"] == 0:
    print("No nodes in graph - check extraction")
```

### 🐛 Cypher Import Fails

**Solution:**
```bash
# Check Cypher syntax
head -20 outputs/graph.cypher

# Test connection
cypher-shell -u neo4j -p password "RETURN 1"

# Import with error logging
cat outputs/graph.cypher | cypher-shell -u neo4j -p password 2>&1 | tee import.log
```

### 🐛 JSON Parsing Error

**Solution:**
```python
# Validate JSON
import json

try:
    with open("outputs/.../docling_graph/graph.json") as f:
        data = json.load(f)
    print("✅ Valid JSON")
except json.JSONDecodeError as e:
    print(f"❌ Invalid JSON: {e}")
```

---

## Next Steps

Now that you understand export formats:

1. **[Data Grounding & Provenance](provenance.md)** - Trace nodes back to source chunks and pages
2. **[Visualization](visualization.md)** - Visualize your graphs
3. **[Neo4j Integration](neo4j-integration.md)** - Deep dive into Neo4j
4. **[Graph Analysis](graph-analysis.md)** - Analyze graph structure