# Graph Analysis

**Navigation:** [← Neo4j Integration](neo4j-integration.md) | [Next: CLI Guide →](../07-cli/index.md)

---

## Overview

**Graph analysis** helps you understand the structure, quality, and characteristics of your knowledge graphs through metrics, statistics, and validation.

**In this guide:**
- Graph metrics
- Quality checks
- Connectivity analysis
- Performance optimization
- Validation techniques

---

## Graph Metrics

### Basic Metrics

```python
from docling_graph.core.converters import GraphConverter

# Create graph
converter = GraphConverter()
graph, metadata = converter.pydantic_list_to_graph(models)

# Basic metrics
print(f"Nodes: {metadata.node_count}")
print(f"Edges: {metadata.edge_count}")
print(f"Density: {metadata.density:.3f}")
print(f"Avg degree: {metadata.avg_degree:.2f}")
```

---

### Node Metrics

#### Node Count by Type

```python
# Node type distribution
for node_type, count in metadata.node_types.items():
    percentage = (count / metadata.node_count) * 100
    print(f"{node_type}: {count} ({percentage:.1f}%)")
```

#### Node Degree

```python
import networkx as nx

# Calculate node degrees
degrees = dict(graph.degree())

# Statistics
avg_degree = sum(degrees.values()) / len(degrees)
max_degree = max(degrees.values())
min_degree = min(degrees.values())

print(f"Average degree: {avg_degree:.2f}")
print(f"Max degree: {max_degree}")
print(f"Min degree: {min_degree}")

# Find high-degree nodes (hubs)
hubs = [(node, deg) for node, deg in degrees.items() if deg > avg_degree * 2]
print(f"Hub nodes: {len(hubs)}")
```

---

### Edge Metrics

#### Edge Count by Type

```python
# Edge type distribution
for edge_type, count in metadata.edge_types.items():
    percentage = (count / metadata.edge_count) * 100
    print(f"{edge_type}: {count} ({percentage:.1f}%)")
```

#### Edge Density

```python
# Graph density (actual edges / possible edges)
density = metadata.density

if density < 0.1:
    print("Sparse graph")
elif density < 0.5:
    print("Medium density graph")
else:
    print("Dense graph")
```

---

## Connectivity Analysis

### Connected Components

```python
import networkx as nx

# Find connected components
components = list(nx.weakly_connected_components(graph))

print(f"Connected components: {len(components)}")
print(f"Largest component: {len(max(components, key=len))} nodes")

# Check if graph is connected
is_connected = nx.is_weakly_connected(graph)
print(f"Graph is connected: {is_connected}")
```

### Isolated Nodes

```python
# Find isolated nodes (no connections)
isolated = [node for node, degree in graph.degree() if degree == 0]

print(f"Isolated nodes: {len(isolated)}")
if isolated:
    print("Warning: Graph has isolated nodes")
    for node in isolated[:5]:
        print(f"  - {node}")
```

---

## Quality Checks

### Validation

```python
from docling_graph.core.utils import validate_graph_structure

try:
    validate_graph_structure(graph, raise_on_error=True)
    print("✓ Graph structure valid")
except ValueError as e:
    print(f"✗ Validation failed: {e}")
```

### Completeness Check

```python
def check_completeness(graph, metadata):
    """Check graph completeness."""
    issues = []
    
    # Check for nodes
    if metadata.node_count == 0:
        issues.append("No nodes in graph")
    
    # Check for edges
    if metadata.edge_count == 0:
        issues.append("No edges in graph")
    
    # Check for isolated nodes
    isolated = [n for n, d in graph.degree() if d == 0]
    if isolated:
        issues.append(f"{len(isolated)} isolated nodes")
    
    # Check node attributes
    nodes_without_label = [
        n for n, data in graph.nodes(data=True)
        if 'label' not in data
    ]
    if nodes_without_label:
        issues.append(f"{len(nodes_without_label)} nodes without labels")
    
    return issues

# Run check
issues = check_completeness(graph, metadata)
if issues:
    print("Graph issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✓ Graph is complete")
```

---

## Complete Examples

### Example 1: Comprehensive Analysis

```python
from docling_graph.core.converters import GraphConverter
import networkx as nx

# Create graph
converter = GraphConverter()
graph, metadata = converter.pydantic_list_to_graph(models)

print("=== Graph Analysis ===\n")

# Basic metrics
print("Basic Metrics:")
print(f"  Nodes: {metadata.node_count}")
print(f"  Edges: {metadata.edge_count}")
print(f"  Density: {metadata.density:.3f}")
print(f"  Avg degree: {metadata.avg_degree:.2f}\n")

# Node types
print("Node Types:")
for node_type, count in sorted(metadata.node_types.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / metadata.node_count) * 100
    print(f"  {node_type}: {count} ({percentage:.1f}%)")

# Edge types
print("\nEdge Types:")
for edge_type, count in sorted(metadata.edge_types.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / metadata.edge_count) * 100
    print(f"  {edge_type}: {count} ({percentage:.1f}%)")

# Connectivity
print("\nConnectivity:")
components = list(nx.weakly_connected_components(graph))
print(f"  Connected components: {len(components)}")
print(f"  Largest component: {len(max(components, key=len))} nodes")

# Quality
print("\nQuality:")
isolated = [n for n, d in graph.degree() if d == 0]
print(f"  Isolated nodes: {len(isolated)}")
print(f"  Graph is connected: {nx.is_weakly_connected(graph)}")
```

### Example 2: Batch Analysis

```python
from docling_graph import PipelineConfig
from pathlib import Path
import json
import pandas as pd

# Analyze multiple documents
results = []

for pdf_file in Path("documents").glob("*.pdf"):
    # Process document
    output_dir = f"analysis/{pdf_file.stem}"
    
    config = PipelineConfig(
        source=str(pdf_file),
        template="my_templates.Invoice",
        output_dir=output_dir
    )
    
    config.run()
    
    # Load statistics
    with open(f"{output_dir}/graph_stats.json") as f:
        stats = json.load(f)
    
    results.append({
        "document": pdf_file.name,
        "nodes": stats["node_count"],
        "edges": stats["edge_count"],
        "density": stats["density"],
        "avg_degree": stats["avg_degree"]
    })

# Create summary
df = pd.DataFrame(results)
print("\n=== Batch Analysis Summary ===")
print(df.describe())

# Export
df.to_csv("batch_analysis.csv", index=False)
```

### Example 3: Quality Report

```python
from docling_graph.core.converters import GraphConverter
import networkx as nx

def generate_quality_report(graph, metadata):
    """Generate comprehensive quality report."""
    
    report = {
        "basic_metrics": {
            "nodes": metadata.node_count,
            "edges": metadata.edge_count,
            "density": metadata.density,
            "avg_degree": metadata.avg_degree
        },
        "quality_checks": {},
        "warnings": []
    }
    
    # Check 1: Empty graph
    if metadata.node_count == 0:
        report["warnings"].append("Graph is empty")
        return report
    
    # Check 2: Isolated nodes
    isolated = [n for n, d in graph.degree() if d == 0]
    report["quality_checks"]["isolated_nodes"] = len(isolated)
    if isolated:
        report["warnings"].append(f"{len(isolated)} isolated nodes found")
    
    # Check 3: Connectivity
    is_connected = nx.is_weakly_connected(graph)
    report["quality_checks"]["is_connected"] = is_connected
    if not is_connected:
        components = list(nx.weakly_connected_components(graph))
        report["warnings"].append(f"Graph has {len(components)} disconnected components")
    
    # Check 4: Node attributes
    nodes_without_label = sum(1 for _, data in graph.nodes(data=True) if 'label' not in data)
    report["quality_checks"]["nodes_without_label"] = nodes_without_label
    if nodes_without_label > 0:
        report["warnings"].append(f"{nodes_without_label} nodes missing labels")
    
    # Check 5: Self-loops
    self_loops = list(nx.selfloop_edges(graph))
    report["quality_checks"]["self_loops"] = len(self_loops)
    if self_loops:
        report["warnings"].append(f"{len(self_loops)} self-loops found")
    
    return report

# Generate report
converter = GraphConverter()
graph, metadata = converter.pydantic_list_to_graph(models)

report = generate_quality_report(graph, metadata)

# Print report
print("=== Quality Report ===\n")
print("Basic Metrics:")
for key, value in report["basic_metrics"].items():
    print(f"  {key}: {value}")

print("\nQuality Checks:")
for key, value in report["quality_checks"].items():
    print(f"  {key}: {value}")

if report["warnings"]:
    print("\nWarnings:")
    for warning in report["warnings"]:
        print(f"  ⚠ {warning}")
else:
    print("\n✓ No quality issues found")
```

---

## Advanced Analysis

### Centrality Measures

```python
import networkx as nx

# Degree centrality
degree_centrality = nx.degree_centrality(graph)
top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

print("Top 5 nodes by degree centrality:")
for node, centrality in top_nodes:
    print(f"  {node}: {centrality:.3f}")

# Betweenness centrality (for undirected view)
undirected = graph.to_undirected()
betweenness = nx.betweenness_centrality(undirected)
top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]

print("\nTop 5 nodes by betweenness centrality:")
for node, centrality in top_between:
    print(f"  {node}: {centrality:.3f}")
```

### Path Analysis

```python
import networkx as nx

# Average shortest path length (for connected graphs)
if nx.is_weakly_connected(graph):
    avg_path_length = nx.average_shortest_path_length(graph.to_undirected())
    print(f"Average shortest path length: {avg_path_length:.2f}")

# Diameter (longest shortest path)
if nx.is_weakly_connected(graph):
    diameter = nx.diameter(graph.to_undirected())
    print(f"Graph diameter: {diameter}")
```

---

## Performance Optimization

### Graph Size Optimization

```python
# Check graph size
import sys

graph_size = sys.getsizeof(graph)
print(f"Graph size: {graph_size / 1024:.2f} KB")

# Optimize by removing unnecessary attributes
def optimize_graph(graph):
    """Remove unnecessary node attributes."""
    for node, data in graph.nodes(data=True):
        # Keep only essential attributes
        essential = ['id', 'label', 'type']
        to_remove = [k for k in data.keys() if k not in essential and data[k] is None]
        for key in to_remove:
            del data[key]
    
    return graph

optimized = optimize_graph(graph.copy())
optimized_size = sys.getsizeof(optimized)
print(f"Optimized size: {optimized_size / 1024:.2f} KB")
print(f"Reduction: {(1 - optimized_size/graph_size) * 100:.1f}%")
```

---

## Best Practices

### 1. Always Validate

```python
# ✅ Good - Validate after creation
from docling_graph.core.utils import validate_graph_structure

try:
    validate_graph_structure(graph, raise_on_error=True)
except ValueError as e:
    print(f"Validation failed: {e}")
```

### 2. Check Statistics

```python
# ✅ Good - Review statistics
if metadata.node_count == 0:
    print("Warning: Empty graph")

if metadata.edge_count == 0:
    print("Warning: No relationships")

if metadata.density < 0.01:
    print("Warning: Very sparse graph")
```

### 3. Monitor Quality

```python
# ✅ Good - Regular quality checks
isolated = [n for n, d in graph.degree() if d == 0]
if len(isolated) > metadata.node_count * 0.1:
    print(f"Warning: {len(isolated)} isolated nodes (>10%)")
```

---

## Troubleshooting

### Issue: Low Density

**Solution:**
```python
# Check if entities are properly connected
# Ensure relationships are defined in Pydantic models

class Invoice(BaseModel):
    issued_by: Organization  # Creates edge
    line_items: List[LineItem]  # Creates edges
```

### Issue: Many Isolated Nodes

**Solution:**
```python
# Enable auto cleanup
converter = GraphConverter(auto_cleanup=True)
graph, metadata = converter.pydantic_list_to_graph(models)

# Or manually remove isolated nodes
isolated = [n for n, d in graph.degree() if d == 0]
graph.remove_nodes_from(isolated)
```

### Issue: Disconnected Components

**Solution:**
```python
# Find largest component
import networkx as nx

components = list(nx.weakly_connected_components(graph))
largest = max(components, key=len)

# Extract largest component
subgraph = graph.subgraph(largest).copy()
print(f"Largest component: {len(subgraph.nodes())} nodes")
```

---

## Next Steps

Now that you understand graph analysis:

1. **[CLI Guide →](../07-cli/index.md)** - Use command-line tools
2. **[API Reference →](../08-api/index.md)** - Programmatic access
3. **[Examples →](../09-examples/index.md)** - Real-world examples

---

## Quick Reference

### Basic Metrics

```python
print(f"Nodes: {metadata.node_count}")
print(f"Edges: {metadata.edge_count}")
print(f"Density: {metadata.density:.3f}")
```

### Connectivity

```python
import networkx as nx
is_connected = nx.is_weakly_connected(graph)
components = list(nx.weakly_connected_components(graph))
```

### Quality Check

```python
isolated = [n for n, d in graph.degree() if d == 0]
print(f"Isolated nodes: {len(isolated)}")
```

### Validation

```python
from docling_graph.core.utils import validate_graph_structure
validate_graph_structure(graph, raise_on_error=True)
```

---

**Navigation:** [← Neo4j Integration](neo4j-integration.md) | [Next: CLI Guide →](../07-cli/index.md)