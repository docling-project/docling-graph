# Export Formats

Docling Graph supports multiple export formats to integrate with different downstream systems and use cases. This document explains each format and when to use it.

## Overview

| Format | Purpose | Best For |
|:-------|:--------|:---------|
| **CSV** | Neo4j admin import | Database ingestion |
| **Cypher** | Bulk graph creation | Neo4j scripting |
| **JSON** | General-purpose data | API integration, archival |
| **Docling JSON** | Original document | Document preservation |
| **Markdown** | Human-readable text | Documentation, review |
| **HTML** | Interactive visualization | Exploration, presentation |

## CSV Export

### Overview

Exports graph data as CSV files compatible with Neo4j's admin import tool.

**Location**: `docling_graph/core/exporters/csv_exporter.py`

### Output Files

```
output_dir/
├── nodes.csv          # All nodes with properties
└── edges.csv          # All edges with properties
```

### Node CSV Format

```csv
:ID,name:string,age:int,email:string,:LABEL
Person_JohnDoe_1990-01-15,John Doe,34,john@example.com,Person
Person_JaneSmith_1985-06-20,Jane Smith,39,jane@example.com,Person
Organization_AcmeCorp,Acme Corp,,,Organization
```

**Columns**:
- `:ID` - Unique node identifier
- Property columns with type suffixes (`:string`, `:int`, `:float`, `:boolean`)
- `:LABEL` - Node type/class

### Edge CSV Format

```csv
:START_ID,:END_ID,:TYPE
Document_INV001,Organization_AcmeCorp,ISSUED_BY
Document_INV001,Person_JohnDoe_1990-01-15,SENT_TO
```

**Columns**:
- `:START_ID` - Source node ID
- `:END_ID` - Target node ID
- `:TYPE` - Edge label/relationship type

### Configuration

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    export_format="csv",  # Export as CSV
    output_dir="outputs"
)
```

### Neo4j Import

```bash
# Using Neo4j admin import tool
neo4j-admin database import full \
    --nodes=outputs/nodes.csv \
    --relationships=outputs/edges.csv \
    --database=neo4j
```

### Advantages

✅ **Fast Import**: Optimized for bulk loading  
✅ **Standard Format**: Works with Neo4j admin tools  
✅ **Type Safety**: Explicit type annotations  
✅ **Scalable**: Handles large graphs efficiently  

### Limitations

⚠️ **Neo4j Specific**: Primarily for Neo4j databases  
⚠️ **Requires Admin**: Needs Neo4j admin access  
⚠️ **Offline Import**: Database must be stopped  

### Use Cases

- Initial database population
- Bulk data migration
- Large-scale graph imports
- Production deployments

## Cypher Export

### Overview

Generates Cypher scripts for creating nodes and relationships in Neo4j.

**Location**: `docling_graph/core/exporters/cypher_exporter.py`

### Output File

```
output_dir/
└── document_graph.cypher
```

### Cypher Script Format

```cypher
// Create nodes
CREATE (:Person {
    id: 'Person_JohnDoe_1990-01-15',
    name: 'John Doe',
    age: 34,
    email: 'john@example.com'
});

CREATE (:Organization {
    id: 'Organization_AcmeCorp',
    name: 'Acme Corp'
});

// Create relationships
MATCH (a:Document {id: 'Document_INV001'})
MATCH (b:Organization {id: 'Organization_AcmeCorp'})
CREATE (a)-[:ISSUED_BY]->(b);

MATCH (a:Document {id: 'Document_INV001'})
MATCH (b:Person {id: 'Person_JohnDoe_1990-01-15'})
CREATE (a)-[:SENT_TO]->(b);
```

### Configuration

```python
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    export_format="cypher",  # Export as Cypher
    output_dir="outputs"
)
```

### Neo4j Execution

```bash
# Using cypher-shell
cat outputs/document_graph.cypher | cypher-shell -u neo4j -p password

# Or in Neo4j Browser
# Copy and paste the script content
```

### Advantages

✅ **Online Import**: Database can remain running  
✅ **Flexible**: Easy to modify scripts  
✅ **Readable**: Human-readable format  
✅ **Incremental**: Can add to existing data  

### Limitations

⚠️ **Slower**: Less efficient than CSV import  
⚠️ **Memory**: Large scripts may cause issues  
⚠️ **Transaction Size**: May need to split large scripts  

### Use Cases

- Development and testing
- Incremental updates
- Small to medium graphs
- When database downtime is not acceptable

## JSON Export

### Overview

Exports graph data as JSON for general-purpose use.

**Location**: `docling_graph/core/exporters/json_exporter.py`

### Output File

```
output_dir/
└── document_graph.json
```

### JSON Format

```json
{
  "nodes": [
    {
      "id": "Person_JohnDoe_1990-01-15",
      "type": "Person",
      "properties": {
        "name": "John Doe",
        "age": 34,
        "email": "john@example.com",
        "date_of_birth": "1990-01-15"
      }
    },
    {
      "id": "Organization_AcmeCorp",
      "type": "Organization",
      "properties": {
        "name": "Acme Corp",
        "tax_id": "123456789"
      }
    }
  ],
  "edges": [
    {
      "source": "Document_INV001",
      "target": "Organization_AcmeCorp",
      "label": "ISSUED_BY",
      "properties": {}
    },
    {
      "source": "Document_INV001",
      "target": "Person_JohnDoe_1990-01-15",
      "label": "SENT_TO",
      "properties": {}
    }
  ],
  "metadata": {
    "node_count": 3,
    "edge_count": 2,
    "node_types": {
      "Person": 1,
      "Organization": 1,
      "Document": 1
    },
    "edge_types": {
      "ISSUED_BY": 1,
      "SENT_TO": 1
    }
  }
}
```

### Configuration

```python
# JSON is always exported alongside other formats
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    export_format="csv",  # Primary format
    output_dir="outputs"
    # JSON is automatically created
)
```

### Advantages

✅ **Universal**: Works with any system  
✅ **Structured**: Easy to parse and process  
✅ **Complete**: Includes all graph data  
✅ **Portable**: Platform-independent  

### Use Cases

- API integration
- Data archival
- Custom processing pipelines
- Cross-platform data exchange
- Web applications

## Docling Export

### Overview

Exports the original Docling document structure and markdown representations.

**Location**: `docling_graph/core/exporters/docling_exporter.py`

### Output Files

```
output_dir/
├── document.json           # Full Docling document structure
├── document.md            # Complete markdown
└── document_pages/        # Per-page markdown (optional)
    ├── page_1.md
    ├── page_2.md
    └── page_3.md
```

### Docling JSON Format

Contains complete document structure:
- Document metadata
- Layout information
- Tables and figures
- Text content
- Page boundaries

### Configuration

```python
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    export_docling=True,              # Enable Docling export
    export_docling_json=True,         # Export JSON structure
    export_markdown=True,             # Export full markdown
    export_per_page_markdown=False,   # Export per-page markdown
    output_dir="outputs"
)
```

### Advantages

✅ **Complete**: Preserves all document information  
✅ **Layout**: Maintains document structure  
✅ **Tables**: Preserves table formatting  
✅ **Figures**: Includes figure references  

### Use Cases

- Document archival
- Layout analysis
- Table extraction
- Multi-format conversion
- Document comparison

## HTML Visualization

### Overview

Generates interactive HTML visualization using Cytoscape.js.

**Location**: `docling_graph/core/visualizers/interactive_visualizer.py`

### Output File

```
output_dir/
└── document_graph.html
```

### Features

- **Interactive**: Click, drag, zoom
- **Node Details**: Click nodes to see properties
- **Edge Labels**: Hover to see relationships
- **Search**: Find specific nodes
- **Layout**: Automatic graph layout
- **Styling**: Color-coded by node type

### Configuration

```python
# HTML visualization is always generated
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    output_dir="outputs"
    # HTML is automatically created
)
```

### Usage

```bash
# Open in browser
open outputs/document_graph.html

# Or serve with Python
python -m http.server 8000
# Navigate to http://localhost:8000/outputs/document_graph.html
```

### Advantages

✅ **Visual**: Easy to understand graph structure  
✅ **Interactive**: Explore relationships dynamically  
✅ **Standalone**: No dependencies, works offline  
✅ **Shareable**: Easy to share with stakeholders  

### Use Cases

- Graph exploration
- Presentations
- Documentation
- Quality assurance
- Stakeholder communication

## Markdown Report

### Overview

Generates detailed markdown reports with node and edge information.

**Location**: `docling_graph/core/visualizers/report_generator.py`

### Output Files

```
output_dir/
└── document_report/
    ├── index.md           # Overview and statistics
    ├── nodes.md          # All nodes with properties
    └── edges.md          # All edges with relationships
```

### Report Structure

#### index.md
```markdown
# Graph Report

## Statistics
- Total Nodes: 10
- Total Edges: 15
- Node Types: Person (3), Organization (2), Document (5)
- Edge Types: ISSUED_BY (5), SENT_TO (5), HAS_AUTHOR (5)

## Source Information
- Source Models: 1
- Processing Mode: many-to-one
- Extraction Backend: llm
```

#### nodes.md
```markdown
# Nodes

## Person_JohnDoe_1990-01-15
**Type**: Person

**Properties**:
- name: John Doe
- age: 34
- email: john@example.com
- date_of_birth: 1990-01-15

---

## Organization_AcmeCorp
**Type**: Organization

**Properties**:
- name: Acme Corp
- tax_id: 123456789
```

#### edges.md
```markdown
# Edges

## ISSUED_BY
- Document_INV001 → Organization_AcmeCorp
- Document_INV002 → Organization_AcmeCorp

## SENT_TO
- Document_INV001 → Person_JohnDoe_1990-01-15
- Document_INV002 → Person_JaneSmith_1985-06-20
```

### Advantages

✅ **Readable**: Human-friendly format  
✅ **Searchable**: Easy to grep and search  
✅ **Versionable**: Works with Git  
✅ **Detailed**: Complete property information  

### Use Cases

- Documentation
- Code reviews
- Audit trails
- Knowledge base
- Team collaboration

## Choosing Export Formats

### Decision Matrix

```
┌─────────────────────────────────────────────────────────┐
│ Use Case                                                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Production Database    →  CSV (bulk import)           │
│  Development/Testing    →  Cypher (incremental)        │
│  API Integration        →  JSON                        │
│  Exploration            →  HTML Visualization          │
│  Documentation          →  Markdown Report             │
│  Archival               →  Docling JSON + JSON         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Multiple Formats

You can use multiple formats simultaneously:

```python
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    export_format="csv",              # Primary: CSV for Neo4j
    export_docling=True,              # Also: Docling document
    export_markdown=True,             # Also: Markdown
    output_dir="outputs"
    # JSON and HTML are always created
)
```

## Best Practices

### For Production

1. **Use CSV**: Fastest for bulk imports
2. **Validate First**: Test with small datasets
3. **Backup**: Keep original exports
4. **Version**: Track export versions

### For Development

1. **Use Cypher**: Easier to modify and test
2. **Use HTML**: Visual verification
3. **Use Markdown**: Documentation
4. **Keep JSON**: Debugging and analysis

### For Integration

1. **Use JSON**: Universal format
2. **Document Schema**: Provide JSON schema
3. **Version API**: Track format changes
4. **Test Parsing**: Validate with consumers

## Troubleshooting

### CSV Import Fails

**Problem**: Neo4j import errors

**Solutions**:
- Check CSV format matches Neo4j version
- Verify node IDs are unique
- Ensure all referenced nodes exist
- Check for special characters in data

### Cypher Script Too Large

**Problem**: Memory errors or timeouts

**Solutions**:
- Split into smaller batches
- Use CSV import instead
- Increase Neo4j memory settings
- Use PERIODIC COMMIT in Cypher

### JSON Too Large

**Problem**: File size issues

**Solutions**:
- Stream processing instead of loading all
- Compress JSON files
- Split into multiple files
- Use database export instead

## Next Steps

- Learn about [Graph Construction](graph-construction.md)
- Understand [Configuration System](configuration.md)
- Explore [Architecture](architecture.md)