# Neo4j Integration


## Overview

**Neo4j integration** enables you to import knowledge graphs into Neo4j graph database for powerful querying, analysis, and visualization using Cypher query language.

**In this guide:**
- Neo4j setup
- Cypher import
- Query examples
- Best practices
- Troubleshooting

---

## Why Neo4j?

### Benefits

✅ **Graph-native database**
- Optimized for graph queries
- Fast relationship traversal
- ACID transactions

✅ **Cypher query language**
- Intuitive pattern matching
- Powerful aggregations
- Path finding algorithms

✅ **Visualization**
- Built-in graph browser
- Interactive exploration
- Custom styling

✅ **Scalability**
- Handles millions of nodes
- Distributed architecture
- High performance

---

## Neo4j Setup

### Installation

#### Option 1: Neo4j Desktop (Recommended)

```bash
# Download from https://neo4j.com/download/
# Install and create a new database
# Default credentials: neo4j/neo4j (change on first login)
```

#### Option 2: Docker

```bash
# Run Neo4j in Docker
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest

# Access at http://localhost:7474
```

#### Option 3: Cloud (Neo4j Aura)

```bash
# Sign up at https://neo4j.com/cloud/aura/
# Create free instance
# Note connection URI and credentials
```

---

### Verify Installation

```bash
# Check Neo4j is running
curl http://localhost:7474

# Test cypher-shell
cypher-shell -u neo4j -p password "RETURN 1"
```

---

## Exporting for Neo4j

### Generate Cypher Script

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    export_format="cypher",  # Generate Cypher script
    output_dir="neo4j_import"
)

run_pipeline(config)

# Generates: neo4j_import/graph.cypher
```

---

## Importing to Neo4j

### Method 1: cypher-shell (Recommended)

```bash
# Import Cypher script
cat neo4j_import/graph.cypher | cypher-shell -u neo4j -p password

# Or with file
cypher-shell -u neo4j -p password -f neo4j_import/graph.cypher

# With error logging
cat neo4j_import/graph.cypher | cypher-shell -u neo4j -p password 2>&1 | tee import.log
```

---

### Method 2: Neo4j Browser

1. Open Neo4j Browser (http://localhost:7474)
2. Login with credentials
3. Open `graph.cypher` file
4. Copy contents
5. Paste into query editor
6. Click "Run" or press Ctrl+Enter

---

### Method 3: Python Driver

```python
from neo4j import GraphDatabase

# Connect to Neo4j
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

# Read Cypher file
with open("neo4j_import/graph.cypher") as f:
    cypher_script = f.read()

# Execute
with driver.session() as session:
    session.run(cypher_script)

driver.close()
print("✅ Imported to Neo4j")
```

---

### Method 4: Automated Import

```python
from docling_graph import run_pipeline, PipelineConfig
import subprocess

# Extract and export
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
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

---

## Querying Neo4j

### Basic Queries

#### Count Nodes

```cypher
// Count all nodes
MATCH (n)
RETURN count(n) as total_nodes

// Count by type
MATCH (n)
RETURN labels(n) as type, count(n) as count
ORDER BY count DESC
```

#### Count Relationships

```cypher
// Count all relationships
MATCH ()-[r]->()
RETURN count(r) as total_relationships

// Count by type
MATCH ()-[r]->()
RETURN type(r) as relationship_type, count(r) as count
ORDER BY count DESC
```

---

### Finding Nodes

#### Find Specific Node

```cypher
// Find invoice by number
MATCH (i:BillingDocument {document_no: "INV-001"})
RETURN i

// Find organization by name
MATCH (o:Organization {name: "Acme Corp"})
RETURN o
```

#### Find All of Type

```cypher
// Find all invoices
MATCH (i:BillingDocument)
RETURN i
LIMIT 10

// Find all organizations
MATCH (o:Organization)
RETURN o.name, o.address
```

---

### Relationship Queries

#### Direct Relationships

```cypher
// Find who issued an invoice
MATCH (i:BillingDocument {document_no: "INV-001"})-[:ISSUED_BY]->(o:Organization)
RETURN i.document_no, o.name

// Find all line items in an invoice
MATCH (i:BillingDocument)-[:CONTAINS_LINE]->(item:LineItem)
WHERE i.document_no = "INV-001"
RETURN item.description, item.total
```

#### Multi-Hop Relationships

```cypher
// Find invoice -> organization -> address
MATCH (i:BillingDocument)-[:ISSUED_BY]->(o:Organization)-[:LOCATED_AT]->(a:Address)
RETURN i.document_no, o.name, a.city

// Find all paths between two nodes
MATCH path = (start:BillingDocument)-[*..3]-(end:Address)
WHERE start.document_no = "INV-001"
RETURN path
```

---

### Aggregation Queries

#### Sum and Average

```cypher
// Total invoice amount
MATCH (i:BillingDocument)
RETURN sum(i.total) as total_amount

// Average invoice amount
MATCH (i:BillingDocument)
RETURN avg(i.total) as average_amount

// Count invoices per organization
MATCH (o:Organization)<-[:ISSUED_BY]-(i:BillingDocument)
RETURN o.name, count(i) as invoice_count
ORDER BY invoice_count DESC
```

---

### Pattern Matching

#### Complex Patterns

```cypher
// Find invoices with specific pattern
MATCH (i:BillingDocument)-[:ISSUED_BY]->(o:Organization),
      (i)-[:SENT_TO]->(c:Organization),
      (i)-[:CONTAINS_LINE]->(item:LineItem)
WHERE i.total > 1000
RETURN i, o, c, collect(item) as items

// Find organizations that both issue and receive invoices
MATCH (o:Organization)<-[:ISSUED_BY]-(i1:BillingDocument),
      (o)<-[:SENT_TO]-(i2:BillingDocument)
RETURN o.name, count(DISTINCT i1) as issued, count(DISTINCT i2) as received
```

---

## Complete Examples

### 📍 Import and Query

```python
from docling_graph import run_pipeline, PipelineConfig
from neo4j import GraphDatabase

# 1. Extract and export
config = PipelineConfig(
    source="invoices.pdf",
    template="templates.BillingDocument",
    export_format="cypher",
    output_dir="neo4j_data"
)

run_pipeline(config)

# 2. Import to Neo4j
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

with open("neo4j_data/graph.cypher") as f:
    cypher_script = f.read()

with driver.session() as session:
    session.run(cypher_script)

# 3. Query
with driver.session() as session:
    result = session.run("""
        MATCH (i:BillingDocument)
        RETURN i.document_no, i.total
        ORDER BY i.total DESC
        LIMIT 5
    """)
    
    for record in result:
        print(f"{record['i.document_no']}: ${record['i.total']}")

driver.close()
```

### 📍 Batch Import

```python
from docling_graph import run_pipeline, PipelineConfig
from pathlib import Path
import subprocess

# Process multiple documents
for pdf_file in Path("documents").glob("*.pdf"):
    print(f"Processing {pdf_file.name}")
    
    # Extract
    config = PipelineConfig(
        source=str(pdf_file),
        template="templates.BillingDocument",
        export_format="cypher",
        output_dir=f"neo4j_batch/{pdf_file.stem}"
    )
    
    run_pipeline(config)
    
    # Import
    cypher_file = f"neo4j_batch/{pdf_file.stem}/graph.cypher"
    subprocess.run([
        "cypher-shell",
        "-u", "neo4j",
        "-p", "password",
        "-f", cypher_file
    ])

print("✅ Batch import complete")
```

### 📍 Query and Export

```python
from neo4j import GraphDatabase
import pandas as pd

# Connect
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

# Query
with driver.session() as session:
    result = session.run("""
        MATCH (i:BillingDocument)-[:ISSUED_BY]->(o:Organization)
        RETURN i.document_no as invoice,
               o.name as organization,
               i.total as amount
        ORDER BY i.total DESC
    """)
    
    # Convert to DataFrame
    df = pd.DataFrame([dict(record) for record in result])
    
    # Export
    df.to_csv("invoice_summary.csv", index=False)
    print(f"Exported {len(df)} records")

driver.close()
```

---

## Best Practices

### 👍 Clear Database Before Import

```cypher
// Delete all nodes and relationships
MATCH (n)
DETACH DELETE n

// Verify empty
MATCH (n)
RETURN count(n)
```

### 👍 Create Indexes

```cypher
// Create index on invoice number
CREATE INDEX document_no_idx FOR (i:BillingDocument) ON (i.document_no)

// Create index on organization name
CREATE INDEX org_name_idx FOR (o:Organization) ON (o.name)

// List indexes
SHOW INDEXES
```

### 👍 Use Constraints

```cypher
// Unique constraint on invoice number
CREATE CONSTRAINT invoice_unique FOR (i:BillingDocument) REQUIRE i.document_no IS UNIQUE

// Existence constraint
CREATE CONSTRAINT document_no_exists FOR (i:BillingDocument) REQUIRE i.document_no IS NOT NULL
```

### 👍 Batch Imports

```python
# ✅ Good - Import in batches
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# Process in batches
batch_size = 1000
for i in range(0, len(statements), batch_size):
    batch = statements[i:i+batch_size]
    
    with driver.session() as session:
        for statement in batch:
            session.run(statement)
    
    print(f"Imported batch {i//batch_size + 1}")

driver.close()
```

---

## Troubleshooting

### 🐛 Connection Refused

**Solution:**
```bash
# Check Neo4j is running
docker ps | grep neo4j

# Or check service
systemctl status neo4j

# Restart if needed
docker restart neo4j
```

### 🐛 Authentication Failed

**Solution:**
```bash
# Reset password
cypher-shell -u neo4j -p neo4j
# Then change password when prompted

# Or set in Docker
docker run -e NEO4J_AUTH=neo4j/newpassword neo4j
```

### 🐛 Import Fails

**Solution:**
```bash
# Check Cypher syntax
head -20 neo4j_import/graph.cypher

# Test small portion
head -100 neo4j_import/graph.cypher | cypher-shell -u neo4j -p password

# Check logs
docker logs neo4j
```

### 🐛 Slow Queries

**Solution:**
```cypher
// Create indexes
CREATE INDEX FOR (i:BillingDocument) ON (i.document_no)

// Use EXPLAIN to analyze
EXPLAIN MATCH (i:BillingDocument) WHERE i.total > 1000 RETURN i

// Use PROFILE for detailed analysis
PROFILE MATCH (i:BillingDocument) WHERE i.total > 1000 RETURN i
```

---

## Advanced Topics

### Graph Algorithms

```cypher
// Find shortest path
MATCH path = shortestPath(
    (start:BillingDocument {document_no: "INV-001"})-[*]-(end:Address)
)
RETURN path

// PageRank (requires APOC or GDS)
CALL gds.pageRank.stream('myGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC
```

### Full-Text Search

```cypher
// Create full-text index
CREATE FULLTEXT INDEX organization_search FOR (o:Organization) ON EACH [o.name, o.description]

// Search
CALL db.index.fulltext.queryNodes('organization_search', 'Acme')
YIELD node, score
RETURN node.name, score
```

---

## Next Steps

Now that you understand Neo4j integration:

1. **[Graph Analysis →](graph-analysis.md)** - Analyze graph structure
2. **[CLI Guide →](../../usage/cli/index.md)** - Use command-line tools
3. **[API Reference →](../../usage/api/index.md)** - Programmatic access