# inspect Command

**Navigation:** [← convert Command](convert-command.md) | [Next: CLI Recipes →](cli-recipes.md)

---

## Overview

The `inspect` command creates **interactive HTML visualizations** of your knowledge graphs that open in your browser.

**Key Features:**
- Interactive node/edge exploration
- CSV and JSON import
- Self-contained HTML output
- Automatic browser opening
- Shareable visualizations

---

## Basic Usage

```bash
uv run docling-graph inspect PATH [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `PATH` | Path to graph data (directory for CSV, file for JSON) |

### Example

```bash
# Visualize CSV output
uv run docling-graph inspect outputs/

# Visualize JSON output
uv run docling-graph inspect outputs/graph.json --format json
```

---

## Input Formats

### CSV Format (Default)

For CSV format, provide a **directory** containing:
- `nodes.csv` - Node data
- `edges.csv` - Edge data

```bash
uv run docling-graph inspect outputs/
```

**Directory structure:**
```
outputs/
├── nodes.csv
├── edges.csv
└── ... (other files)
```

---

### JSON Format

For JSON format, provide a **file path** to the graph JSON:

```bash
uv run docling-graph inspect outputs/graph.json --format json
```

---

## Options

### Input Format

```bash
--format {csv|json}
```

**Default:** `csv`

**Example:**
```bash
# CSV format (default)
uv run docling-graph inspect outputs/

# JSON format
uv run docling-graph inspect outputs/graph.json --format json
```

---

### Output File

```bash
--output PATH
```

Specify where to save the HTML visualization.

**Default:** Temporary file

**Example:**
```bash
# Save to specific location
uv run docling-graph inspect outputs/ \
    --output visualization.html

# Save with timestamp
uv run docling-graph inspect outputs/ \
    --output "viz_$(date +%Y%m%d_%H%M%S).html"
```

---

### Browser Control

```bash
--open / --no-open
```

Control whether to automatically open the visualization in your browser.

**Default:** `--open` (opens automatically)

**Example:**
```bash
# Open automatically (default)
uv run docling-graph inspect outputs/

# Don't open browser
uv run docling-graph inspect outputs/ \
    --no-open \
    --output visualization.html
```

---

## Complete Examples

### Example 1: Quick Visualization

```bash
# Convert document
uv run docling-graph convert invoice.pdf \
    --template "templates.Invoice" \
    --output-dir "outputs/invoice"

# Visualize immediately
uv run docling-graph inspect outputs/invoice/
```

### Example 2: Save for Later

```bash
# Create visualization without opening
uv run docling-graph inspect outputs/ \
    --output graph_viz.html \
    --no-open

# Open later
open graph_viz.html  # macOS
xdg-open graph_viz.html  # Linux
start graph_viz.html  # Windows
```

### Example 3: JSON Format

```bash
# Visualize JSON graph
uv run docling-graph inspect outputs/graph.json \
    --format json \
    --output interactive_graph.html
```

### Example 4: Batch Visualization

```bash
# Create visualizations for multiple outputs
for dir in outputs/*/; do
    name=$(basename "$dir")
    uv run docling-graph inspect "$dir" \
        --output "visualizations/${name}.html" \
        --no-open
done

echo "Created visualizations in visualizations/"
```

### Example 5: Share Visualization

```bash
# Create self-contained HTML
uv run docling-graph inspect outputs/ \
    --output shared_graph.html \
    --no-open

# Share the HTML file
# The file contains all data and can be opened anywhere
```

---

## Interactive Features

### Node Exploration

**Click on a node to:**
- View node properties
- Highlight connected edges
- See relationship details
- Filter by node type

### Edge Exploration

**Click on an edge to:**
- View relationship type
- See source and target nodes
- View edge properties

### Graph Navigation

**Controls:**
- **Zoom:** Mouse wheel or pinch
- **Pan:** Click and drag
- **Reset:** Double-click background
- **Search:** Use search box to find nodes

### Layout Options

**Available layouts:**
- **Force-directed:** Automatic positioning
- **Hierarchical:** Top-down structure
- **Circular:** Nodes in a circle
- **Grid:** Regular grid layout

---

## Output Structure

### HTML File Contents

The generated HTML file is **self-contained** and includes:
- Complete graph data
- Interactive visualization library
- Styling and controls
- No external dependencies

**File size:** Typically 100KB - 2MB depending on graph size

### Sharing

```bash
# Create visualization
uv run docling-graph inspect outputs/ \
    --output graph.html \
    --no-open

# Share via email, cloud storage, or web hosting
# Recipients can open directly in any modern browser
```

---

## Validation

### CSV Validation

The command validates that required files exist:

```bash
uv run docling-graph inspect outputs/
```

**Checks:**
- Directory exists
- `nodes.csv` exists
- `edges.csv` exists

**Error if missing:**
```
[bold red]Error:[/bold red] nodes.csv not found in outputs/
```

---

### JSON Validation

```bash
uv run docling-graph inspect graph.json --format json
```

**Checks:**
- File exists
- File has `.json` extension
- Valid JSON format

**Error if invalid:**
```
[bold red]Error:[/bold red] For JSON format, path must be a .json file
```

---

## Troubleshooting

### Issue: Files Not Found

**Error:**
```
[bold red]Error:[/bold red] nodes.csv not found in outputs/
```

**Solution:**
```bash
# Check directory contents
ls outputs/

# Ensure convert completed successfully
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --output-dir "outputs"

# Then inspect
uv run docling-graph inspect outputs/
```

---

### Issue: Browser Doesn't Open

**Error:**
```
Browser failed to open
```

**Solution:**
```bash
# Save to file and open manually
uv run docling-graph inspect outputs/ \
    --output graph.html \
    --no-open

# Open manually
open graph.html  # macOS
xdg-open graph.html  # Linux
start graph.html  # Windows
```

---

### Issue: Large Graph Performance

**Problem:** Visualization is slow with large graphs

**Solution:**
```bash
# Filter graph before visualization
# Use Python to create smaller subset

# Or use Neo4j for large graphs
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --export-format cypher

# Import to Neo4j and use Neo4j Browser
```

---

## Integration Workflows

### Workflow 1: Development Cycle

```bash
# 1. Convert document
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --output-dir "test_output"

# 2. Inspect results
uv run docling-graph inspect test_output/

# 3. Iterate on template
# Edit templates/invoice.py

# 4. Re-convert and inspect
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --output-dir "test_output"

uv run docling-graph inspect test_output/
```

---

### Workflow 2: Batch Processing with Visualization

```bash
#!/bin/bash
# process_and_visualize.sh

INPUT_DIR="documents"
OUTPUT_BASE="outputs"
VIZ_DIR="visualizations"

mkdir -p "$VIZ_DIR"

for pdf in "$INPUT_DIR"/*.pdf; do
    name=$(basename "$pdf" .pdf)
    output_dir="$OUTPUT_BASE/$name"
    
    echo "Processing: $name"
    
    # Convert
    uv run docling-graph convert "$pdf" \
        --template "templates.Invoice" \
        --output-dir "$output_dir"
    
    # Visualize
    uv run docling-graph inspect "$output_dir" \
        --output "$VIZ_DIR/${name}.html" \
        --no-open
    
    echo "Completed: $name"
done

echo "All visualizations saved to $VIZ_DIR/"
```

---

### Workflow 3: Quality Assurance

```bash
# Convert with verbose logging
uv run docling-graph --verbose convert document.pdf \
    --template "templates.Invoice" \
    --output-dir "qa_output"

# Inspect graph structure
uv run docling-graph inspect qa_output/

# Check statistics
cat qa_output/graph_stats.json

# Review markdown report
cat qa_output/markdown_report.md
```

---

## Comparison with Other Tools

### inspect vs Neo4j Browser

| Feature | inspect | Neo4j Browser |
|---------|---------|---------------|
| Setup | No setup required | Requires Neo4j installation |
| Sharing | Self-contained HTML | Requires Neo4j access |
| Performance | Good for small/medium graphs | Excellent for large graphs |
| Querying | Basic filtering | Full Cypher queries |
| Best for | Quick visualization, sharing | Production, complex queries |

### When to Use inspect

✅ **Use inspect for:**
- Quick visualization during development
- Sharing results with non-technical users
- Small to medium graphs (<1000 nodes)
- No database setup required

❌ **Use Neo4j for:**
- Large graphs (>1000 nodes)
- Complex queries
- Production deployments
- Team collaboration

---

## Best Practices

### 1. Save Important Visualizations

```bash
# ✅ Good - Save with descriptive name
uv run docling-graph inspect outputs/ \
    --output "invoice_001_graph.html" \
    --no-open

# ❌ Avoid - Temporary files get lost
uv run docling-graph inspect outputs/
```

### 2. Organize Visualizations

```bash
# ✅ Good - Organized structure
mkdir -p visualizations/invoices
uv run docling-graph inspect outputs/invoice_001/ \
    --output "visualizations/invoices/invoice_001.html" \
    --no-open

# ❌ Avoid - Cluttered directory
uv run docling-graph inspect outputs/ \
    --output "viz1.html" \
    --no-open
```

### 3. Use for Development

```bash
# ✅ Good - Quick feedback loop
uv run docling-graph convert test.pdf -t "templates.Invoice" -o "test"
uv run docling-graph inspect test/

# ✅ Good - Iterate quickly
# Edit template, re-run, inspect
```

---

## Next Steps

1. **[CLI Recipes →](cli-recipes.md)** - Common CLI patterns
2. **[Visualization Guide →](../06-graph-management/visualization.md)** - Advanced visualization
3. **[Neo4j Integration →](../06-graph-management/neo4j-integration.md)** - Database visualization

---

## Quick Reference

### Basic Commands

```bash
# Visualize CSV (default)
uv run docling-graph inspect outputs/

# Visualize JSON
uv run docling-graph inspect graph.json --format json

# Save to file
uv run docling-graph inspect outputs/ --output viz.html

# Don't open browser
uv run docling-graph inspect outputs/ --no-open
```

### Common Patterns

```bash
# Convert and inspect
uv run docling-graph convert doc.pdf -t "templates.Invoice" -o "out"
uv run docling-graph inspect out/

# Batch visualization
for dir in outputs/*/; do
    uv run docling-graph inspect "$dir" \
        --output "viz/$(basename $dir).html" \
        --no-open
done

# Share visualization
uv run docling-graph inspect outputs/ \
    --output shared.html \
    --no-open
```

### All Options

```bash
uv run docling-graph inspect PATH \
    --format {csv|json} \
    --output PATH \
    --open / --no-open
```

---

**Navigation:** [← convert Command](convert-command.md) | [Next: CLI Recipes →](cli-recipes.md)