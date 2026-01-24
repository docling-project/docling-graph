# Export Configuration


## Overview

Export configuration controls how Docling Graph outputs extracted data and knowledge graphs. Multiple export formats are supported, allowing you to integrate with various downstream tools and databases.

**In this guide:**
- Export formats (CSV, Cypher, JSON)
- Output directory structure
- Format-specific options
- Export best practices
- Integration scenarios

---

## Export Formats

### Quick Comparison

| Format | Best For | File Type | Use Case |
|:-------|:---------|:----------|:---------|
| **CSV** | Analysis, spreadsheets | `.csv` | Data analysis, Excel |
| **Cypher** | Neo4j import | `.cypher` | Graph database import |
| **JSON** | APIs, processing | `.json` | Programmatic access |

---

## CSV Export

### What is CSV Export?

CSV export creates **separate CSV files** for nodes and edges, making it easy to analyze data in spreadsheets or import into databases.

### Configuration

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_format="csv"  # CSV export (default)
)
```

### Output Structure

```
outputs/
├── nodes.csv          # All nodes
├── edges.csv          # All edges
├── graph_stats.json   # Graph statistics
└── visualization.html # Interactive visualization
```

### CSV Files Format

#### nodes.csv

```csv
node_id,node_type,properties
invoice_001,Invoice,"{""invoice_number"": ""INV-001"", ""total"": 1000}"
org_acme,Organization,"{""name"": ""Acme Corp""}"
addr_123,Address,"{""street"": ""123 Main St"", ""city"": ""Paris""}"
```

#### edges.csv

```csv
source_id,edge_type,target_id
invoice_001,ISSUED_BY,org_acme
org_acme,LOCATED_AT,addr_123
```

### When to Use CSV

✅ **Use CSV when:**
- Analyzing data in Excel/spreadsheets
- Importing into SQL databases
- Need human-readable format
- Performing data analysis
- Creating reports

❌ **Don't use CSV when:**
- Need direct Neo4j import
- Want programmatic access
- Require nested structures
- Need type preservation

---

## Cypher Export

### What is Cypher Export?

Cypher export creates **Cypher statements** for direct import into Neo4j graph databases.

### Configuration

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_format="cypher"  # Cypher export
)
```

### Output Structure

```
outputs/
├── graph.cypher       # Cypher statements
├── graph_stats.json   # Graph statistics
└── visualization.html # Interactive visualization
```

### Cypher File Format

```cypher
// Create nodes
CREATE (:Invoice {invoice_number: "INV-001", total: 1000, node_id: "invoice_001"})
CREATE (:Organization {name: "Acme Corp", node_id: "org_acme"})
CREATE (:Address {street: "123 Main St", city: "Paris", node_id: "addr_123"})

// Create relationships
MATCH (a {node_id: "invoice_001"}), (b {node_id: "org_acme"})
CREATE (a)-[:ISSUED_BY]->(b)

MATCH (a {node_id: "org_acme"}), (b {node_id: "addr_123"})
CREATE (a)-[:LOCATED_AT]->(b)
```

### When to Use Cypher

✅ **Use Cypher when:**
- Importing into Neo4j
- Building graph databases
- Need graph queries
- Want graph visualization
- Performing graph analytics

❌ **Don't use Cypher when:**
- Not using Neo4j
- Need tabular format
- Want simple data analysis
- Require Excel compatibility

### Neo4j Import

```bash
# Import into Neo4j
cat outputs/graph.cypher | cypher-shell -u neo4j -p password

# Or use Neo4j Browser
# 1. Open Neo4j Browser
# 2. Copy contents of graph.cypher
# 3. Paste and execute
```

---

## JSON Export

### What is JSON Export?

JSON export is **always generated** alongside CSV or Cypher, providing structured data for programmatic access.

### Output Structure

```
outputs/
├── extracted_data.json  # Extracted Pydantic models
├── graph_data.json      # Graph structure
├── graph_stats.json     # Graph statistics
└── ...
```

### JSON Files Format

#### extracted_data.json

```json
{
  "models": [
    {
      "invoice_number": "INV-001",
      "total": 1000,
      "issued_by": {
        "name": "Acme Corp",
        "located_at": {
          "street": "123 Main St",
          "city": "Paris"
        }
      }
    }
  ]
}
```

#### graph_data.json

```json
{
  "nodes": [
    {
      "id": "invoice_001",
      "type": "Invoice",
      "properties": {
        "invoice_number": "INV-001",
        "total": 1000
      }
    }
  ],
  "edges": [
    {
      "source": "invoice_001",
      "type": "ISSUED_BY",
      "target": "org_acme"
    }
  ]
}
```

### When to Use JSON

✅ **Use JSON when:**
- Building APIs
- Programmatic processing
- Need nested structures
- Want type preservation
- Integrating with applications

---

## Output Directory Structure

### Default Structure

```
outputs/
├── extracted_data.json      # Pydantic models
├── graph_data.json          # Graph structure
├── graph_stats.json         # Statistics
├── nodes.csv                # Nodes (CSV format)
├── edges.csv                # Edges (CSV format)
├── graph.cypher             # Cypher (Cypher format)
├── visualization.html       # Interactive viz
├── report.md                # Markdown report
├── docling_document.json    # Docling output
├── document.md              # Markdown export
└── pages/                   # Per-page exports
    ├── page_001.md
    ├── page_002.md
    └── ...
```

### Custom Output Directory

```python
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    output_dir="my_results/invoice_001"  # Custom directory
)
```

**Output:** `my_results/invoice_001/nodes.csv`, etc.

---

## Complete Configuration Examples

### Example 1: CSV Export with Full Outputs

```python
config = PipelineConfig(
    source="invoice.pdf",
    template="my_templates.Invoice",
    
    # CSV export
    export_format="csv",
    
    # Enable all exports
    export_docling=True,
    export_docling_json=True,
    export_markdown=True,
    
    # Custom output directory
    output_dir="results/invoice_001"
)
```

### Example 2: Cypher Export for Neo4j

```python
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    
    # Cypher export for Neo4j
    export_format="cypher",
    
    # Minimal other exports
    export_docling=False,
    export_markdown=False,
    
    output_dir="neo4j_import"
)
```

### Example 3: Batch Processing with Organized Outputs

```python
import os
from pathlib import Path

for doc_path in Path("documents").glob("*.pdf"):
    # Create output directory per document
    output_dir = f"results/{doc_path.stem}"
    
    config = PipelineConfig(
        source=str(doc_path),
        template="my_templates.Invoice",
        export_format="csv",
        output_dir=output_dir
    )
    
    config.run()
```

---

## Graph Statistics

### graph_stats.json

Always generated, contains graph metrics:

```json
{
  "node_count": 15,
  "edge_count": 18,
  "node_types": {
    "Invoice": 1,
    "Organization": 2,
    "Address": 3,
    "LineItem": 9
  },
  "edge_types": {
    "ISSUED_BY": 1,
    "SENT_TO": 1,
    "LOCATED_AT": 5,
    "CONTAINS_ITEM": 9,
    "HAS_TOTAL": 2
  },
  "avg_degree": 2.4,
  "density": 0.17
}
```

### Using Statistics

```python
import json

# Load statistics
with open("outputs/graph_stats.json") as f:
    stats = json.load(f)

print(f"Nodes: {stats['node_count']}")
print(f"Edges: {stats['edge_count']}")
print(f"Node types: {stats['node_types']}")
```

---

## Visualization

### Interactive HTML Visualization

Always generated: `outputs/visualization.html`

**Features:**
- Interactive graph visualization
- Node and edge inspection
- Zoom and pan
- Search functionality
- Export to image

**Open in browser:**
```bash
# Open visualization
open outputs/visualization.html  # macOS
xdg-open outputs/visualization.html  # Linux
start outputs/visualization.html  # Windows
```

### Markdown Report

Always generated: `outputs/report.md`

**Contains:**
- Extraction summary
- Graph statistics
- Node and edge counts
- Processing time
- Configuration used

---

## Export Format Selection

### By Use Case

| Use Case | Recommended Format | Reason |
|:---------|:------------------|:-------|
| **Data Analysis** | CSV | Excel/spreadsheet compatible |
| **Graph Database** | Cypher | Direct Neo4j import |
| **API Integration** | JSON | Programmatic access |
| **Reporting** | CSV + Markdown | Human-readable |
| **Machine Learning** | JSON | Structured data |
| **Visualization** | Any | HTML viz always generated |

### By Tool

| Tool | Format | Import Method |
|:-----|:-------|:--------------|
| **Excel** | CSV | Open directly |
| **Neo4j** | Cypher | cypher-shell or Browser |
| **Python** | JSON | json.load() |
| **Pandas** | CSV | pd.read_csv() |
| **SQL Database** | CSV | COPY or LOAD DATA |
| **Power BI** | CSV | Import data |

---

## Advanced Export Options

### Reverse Edges

Create bidirectional relationships:

```python
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_format="cypher",
    reverse_edges=True  # Create reverse relationships
)
```

**Effect:**
```cypher
// Original
CREATE (a)-[:ISSUED_BY]->(b)

// With reverse_edges=True
CREATE (a)-[:ISSUED_BY]->(b)
CREATE (b)-[:ISSUES]->(a)  # Reverse edge added
```

---

## Integration Scenarios

### Scenario 1: Excel Analysis

```python
# Export for Excel analysis
config = PipelineConfig(
    source="invoices.pdf",
    template="my_templates.Invoice",
    export_format="csv",
    output_dir="excel_analysis"
)

config.run()

# Open in Excel
# File -> Open -> excel_analysis/nodes.csv
```

### Scenario 2: Neo4j Graph Database

```python
# Export for Neo4j
config = PipelineConfig(
    source="documents.pdf",
    template="my_templates.Invoice",
    export_format="cypher",
    output_dir="neo4j_import"
)

config.run()

# Import to Neo4j
# cat neo4j_import/graph.cypher | cypher-shell
```

### Scenario 3: Python Data Processing

```python
# Export and process in Python
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_format="csv",
    output_dir="python_processing"
)

config.run()

# Load and process
import pandas as pd

nodes = pd.read_csv("python_processing/nodes.csv")
edges = pd.read_csv("python_processing/edges.csv")

print(f"Total nodes: {len(nodes)}")
print(f"Total edges: {len(edges)}")
```

### Scenario 4: API Integration

```python
# Export for API
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_format="csv",  # Format doesn't matter, JSON always generated
    output_dir="api_data"
)

config.run()

# Load JSON for API
import json

with open("api_data/extracted_data.json") as f:
    data = json.load(f)

# Send to API
# requests.post("https://api.example.com/invoices", json=data)
```

---

## Best Practices

### 1. Choose Format by Use Case

```python
# ✅ Good - Match format to use case
if use_case == "neo4j":
    export_format = "cypher"
elif use_case == "excel":
    export_format = "csv"
else:
    export_format = "csv"  # Default

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_format=export_format
)
```

### 2. Organize Output Directories

```python
# ✅ Good - Organized structure
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/{document_type}/{timestamp}"

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    output_dir=output_dir
)
```

### 3. Enable Useful Exports

```python
# ✅ Good - Enable what you need
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_format="csv",
    
    # Enable for debugging
    export_markdown=True,
    
    # Disable if not needed
    export_docling=False,
    export_docling_json=False
)
```

### 4. Check Output Files

```python
# ✅ Good - Verify outputs
config.run()

import os
output_dir = config.output_dir

# Check files exist
assert os.path.exists(f"{output_dir}/nodes.csv")
assert os.path.exists(f"{output_dir}/edges.csv")
assert os.path.exists(f"{output_dir}/graph_stats.json")

print("✓ All outputs generated successfully")
```

---

## Troubleshooting

### Issue: Output Directory Not Created

**Solution:**
```python
# Ensure parent directory exists
import os
os.makedirs("results/invoices", exist_ok=True)

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    output_dir="results/invoices/invoice_001"
)
```

### Issue: CSV Files Empty

**Solution:**
```python
# Check extraction succeeded
config.run()

# Verify graph has nodes
import json
with open("outputs/graph_stats.json") as f:
    stats = json.load(f)
    
if stats["node_count"] == 0:
    print("Warning: No nodes extracted")
```

### Issue: Cypher Import Fails

**Solution:**
```bash
# Check Cypher syntax
cat outputs/graph.cypher | head -20

# Import with error handling
cat outputs/graph.cypher | cypher-shell -u neo4j -p password 2>&1 | tee import.log
```

---

## Next Steps

Now that you understand export configuration:

1. **[Configuration Examples →](configuration-examples.md)** - Complete scenarios
2. **[Model Configuration](model-configuration.md)** - Model settings
3. **[Graph Management](../graph-management/index.md)** - Working with graphs

---

## Quick Reference

### CSV Export (Default)

```python
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_format="csv"
)
```

**Output:** `nodes.csv`, `edges.csv`, JSON files

### Cypher Export

```python
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_format="cypher"
)
```

**Output:** `graph.cypher`, JSON files

### Custom Output Directory

```python
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    output_dir="my_results/doc_001"
)
```