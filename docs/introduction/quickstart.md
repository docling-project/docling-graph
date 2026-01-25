# Quickstart


## Overview

Get started with docling-graph in **5 minutes** by extracting structured data from a simple invoice.

**What You'll Learn:**
- Basic template creation
- Running your first extraction
- Viewing results

**Prerequisites:**
- Python 3.10+
- `uv` package manager
- A sample invoice (PDF or image)

---

## Step 1: Installation

```bash
# Install docling-graph with all features
uv sync --extra all

# Verify installation
uv run docling-graph --version
```

---

## Step 2: Create a Template

Create a file `simple_invoice.py`:

```python
"""Simple invoice template for quickstart."""

from pydantic import BaseModel, Field

class SimpleInvoice(BaseModel):
    """A simple invoice model."""
    
    invoice_number: str = Field(
        description="The unique invoice identifier",
        examples=["INV-001", "2024-001"]
    )
    
    date: str = Field(
        description="Invoice date in any format",
        examples=["2024-01-15", "January 15, 2024"]
    )
    
    total: float = Field(
        description="Total amount to be paid",
        examples=[1234.56, 999.99]
    )
    
    currency: str = Field(
        description="Currency code",
        examples=["USD", "EUR", "GBP"]
    )
```

---

## Step 3: Run Extraction

### Option A: Using CLI

```bash
# Process invoice
uv run docling-graph convert invoice.pdf \
    --template "simple_invoice.SimpleInvoice" \
    --output-dir "quickstart_output"
```

### Option B: Using Python API

Create `run_quickstart.py`:

```python
"""Quickstart extraction script."""

from docling_graph import PipelineConfig

# Configure pipeline
config = PipelineConfig(
    source="invoice.pdf",
    template="simple_invoice.SimpleInvoice",
    output_dir="quickstart_output"
)

# Run extraction
print("Processing invoice...")
config.run()
print("✅ Complete! Check quickstart_output/")
```

Run it:

```bash
uv run python run_quickstart.py
```

---

## Step 4: View Results

### Inspect Graph Visually

```bash
# Open interactive visualization
uv run docling-graph inspect quickstart_output/
```

This opens an HTML visualization in your browser showing:
- Extracted nodes (invoice data)
- Relationships (if any)
- Interactive exploration

### View CSV Data

```bash
# View nodes
cat quickstart_output/nodes.csv

# View edges
cat quickstart_output/edges.csv
```

**Example nodes.csv:**
```csv
id,label,type,invoice_number,date,total,currency
invoice_1,SimpleInvoice,SimpleInvoice,INV-001,2024-01-15,1234.56,USD
```

### View Statistics

```bash
# View graph statistics
cat quickstart_output/graph_stats.json
```

**Example output:**
```json
{
  "node_count": 1,
  "edge_count": 0,
  "density": 0.0,
  "avg_degree": 0.0,
  "node_types": {
    "SimpleInvoice": 1
  },
  "edge_types": {}
}
```

---

## Complete Example

Here's everything together:

### 1. Create Template

**File:** `simple_invoice.py`

```python
from pydantic import BaseModel, Field

class SimpleInvoice(BaseModel):
    """A simple invoice model."""
    
    invoice_number: str = Field(
        description="The unique invoice identifier",
        examples=["INV-001", "2024-001"]
    )
    
    date: str = Field(
        description="Invoice date",
        examples=["2024-01-15"]
    )
    
    total: float = Field(
        description="Total amount",
        examples=[1234.56]
    )
    
    currency: str = Field(
        description="Currency code",
        examples=["USD", "EUR"]
    )
```

### 2. Run Extraction

**CLI:**
```bash
uv run docling-graph convert invoice.pdf \
    --template "simple_invoice.SimpleInvoice" \
    --output-dir "quickstart_output"
```

**Python:**
```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="invoice.pdf",
    template="simple_invoice.SimpleInvoice",
    output_dir="quickstart_output"
)
config.run()
```

### 3. View Results

```bash
# Interactive visualization
uv run docling-graph inspect quickstart_output/

# View data
cat quickstart_output/nodes.csv
cat quickstart_output/graph_stats.json
```

---

## Expected Output Structure

```
quickstart_output/
├── nodes.csv                    # Extracted data
├── edges.csv                    # Relationships (empty for simple model)
├── graph.json                   # Complete graph
├── graph_stats.json             # Statistics
├── graph_visualization.html     # Interactive viz
├── markdown_report.md           # Summary report
└── full_document.md             # Markdown export
```

---

## Troubleshooting

### Issue: Template Not Found

**Error:**
```
ModuleNotFoundError: No module named 'simple_invoice'
```

**Solution:**
```bash
# Ensure template is in current directory
ls simple_invoice.py

# Or use absolute path
uv run docling-graph convert invoice.pdf \
    --template "$(pwd)/simple_invoice.SimpleInvoice"
```

### Issue: No Data Extracted

**Problem:** Empty nodes.csv

**Solution:**
1. Check template descriptions are clear
2. Verify document is readable
3. Try with verbose logging:

```bash
uv run docling-graph --verbose convert invoice.pdf \
    --template "simple_invoice.SimpleInvoice"
```

### Issue: API Key Error

**Error:**
```
ConfigurationError: API key not found
```

**Solution:**
```bash
# Use local inference (default)
uv run docling-graph convert invoice.pdf \
    --template "simple_invoice.SimpleInvoice" \
    --inference local

# Or set API key for remote
export MISTRAL_API_KEY="your-key"
```

---

## Next Steps

### Improve Your Template

Add more fields:

```python
class ImprovedInvoice(BaseModel):
    """Improved invoice with more fields."""
    
    invoice_number: str = Field(description="Invoice number")
    date: str = Field(description="Invoice date")
    total: float = Field(description="Total amount")
    currency: str = Field(description="Currency")
    
    # New fields
    issuer_name: str = Field(
        description="Company that issued the invoice",
        examples=["Acme Corp", "ABC Company"]
    )
    
    client_name: str = Field(
        description="Client receiving the invoice",
        examples=["John Doe", "XYZ Inc"]
    )
    
    subtotal: float = Field(
        description="Amount before tax",
        examples=[1000.00]
    )
    
    tax_amount: float = Field(
        description="Tax amount",
        examples=[234.56]
    )
```

### Add Relationships

Create nested entities:

```python
class Address(BaseModel):
    """Address component."""
    street: str
    city: str
    postal_code: str

class Organization(BaseModel):
    """Organization entity."""
    name: str
    address: Address

def edge(label: str, **kwargs):
    """Helper for graph edges."""
    from pydantic import Field
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

class Invoice(BaseModel):
    """Invoice with relationships."""
    invoice_number: str
    total: float
    issued_by: Organization = edge(label="ISSUED_BY")
```

### Try Different Backends

```bash
# VLM for images (faster)
uv run docling-graph convert invoice.jpg \
    --template "simple_invoice.SimpleInvoice" \
    --backend vlm

# LLM for complex documents
uv run docling-graph convert invoice.pdf \
    --template "simple_invoice.SimpleInvoice" \
    --backend llm \
    --inference remote
```

---

## Learn More

### Complete Examples

- **[Invoice Extraction →](../usage/examples/invoice-extraction.md)** - Full invoice with relationships
- **[Research Paper →](../usage/examples/research-paper.md)** - Complex scientific documents
- **[ID Card →](../usage/examples/id-card.md)** - Vision-based extraction

### Documentation

- **[Schema Definition →](../fundamentals/schema-definition/index.md)** - Template creation guide
- **[CLI Reference →](../usage/cli/index.md)** - All CLI commands
- **[Python API →](../usage/api/index.md)** - Programmatic usage

### Advanced Topics

- **[Custom Backends →](../usage/advanced/custom-backends.md)** - Create custom extractors
- **[Performance Tuning →](../usage/advanced/performance-tuning.md)** - Optimize processing
- **[Testing →](../usage/advanced/testing.md)** - Test your templates

---

## Quick Reference

### Minimal Template

```python
from pydantic import BaseModel, Field

class MyTemplate(BaseModel):
    field1: str = Field(description="Description")
    field2: float = Field(description="Description")
```

### Run Extraction

```bash
# CLI
uv run docling-graph convert doc.pdf -t "template.MyTemplate"

# Python
from docling_graph import PipelineConfig
config = PipelineConfig(source="doc.pdf", template="template.MyTemplate")
config.run()
```

### View Results

```bash
# Visualize
uv run docling-graph inspect outputs/

# View data
cat outputs/nodes.csv
```

---

## Summary

You've learned:
<br>✅ How to create a simple Pydantic template
<br>✅ How to run extraction (CLI and Python)
<br>✅ How to view and inspect results
<br>✅ Basic troubleshooting

**Time taken:** ~5 minutes

**Next:** Try the [Invoice Extraction](../usage/examples/invoice-extraction.md) example for a more complete workflow!