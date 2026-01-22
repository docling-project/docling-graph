# Examples

**Navigation:** [← Batch Processing](../08-api/batch-processing.md) | [Next: Advanced Topics →](../10-advanced/index.md)

---

## Overview

This section provides **complete, end-to-end examples** for common document processing scenarios. Each example includes:

- Complete Pydantic template
- Step-by-step processing guide
- CLI and Python API usage
- Expected outputs
- Troubleshooting tips

---

## Available Examples

### 1. [Quickstart](quickstart.md)
**5-Minute Introduction**

Get started quickly with a simple invoice extraction example.

- **Document Type:** Invoice (PDF/Image)
- **Time:** 5 minutes
- **Backend:** VLM or LLM

---

### 2. [Invoice Extraction](invoice-extraction.md)
**Complete Invoice Processing**

Extract structured data from invoices including issuer, client, line items, and totals.

- **Document Type:** Invoice (PDF/JPG)
- **Features:** Nested entities, relationships, validation
- **Backend:** VLM (recommended) or LLM

**What You'll Learn:**
- Creating entity and component models
- Defining graph relationships
- Using edge() helper
- Handling addresses and line items

---

### 3. [Research Paper](research-paper.md)
**Scientific Document Analysis**

Extract complex research data including experiments, measurements, and results.

- **Document Type:** Research Paper (PDF)
- **Features:** Complex ontology, enums, validators, measurements
- **Backend:** LLM with chunking

**What You'll Learn:**
- Complex template design
- Enum normalization
- Custom validators
- Measurement parsing
- Multi-page consolidation

---

### 4. [ID Card](id-card.md)
**Identity Document Extraction**

Extract personal information from ID cards and identity documents.

- **Document Type:** ID Card (Image)
- **Features:** Date parsing, address validation, person entities
- **Backend:** VLM (recommended)

**What You'll Learn:**
- Vision-based extraction
- Date field handling
- Address parsing
- Field validators
- Graph ID configuration

---

### 5. [Insurance Policy](insurance-policy.md)
**Insurance Document Processing**

Extract policy details, coverage information, and terms from insurance documents.

- **Document Type:** Insurance Policy (PDF)
- **Features:** Policy terms, coverage details, beneficiaries
- **Backend:** LLM

**What You'll Learn:**
- Financial document processing
- Policy structure modeling
- Coverage relationships
- Term extraction

---

## Quick Comparison

| Example | Document Type | Complexity | Backend | Time | Key Features |
|---------|---------------|------------|---------|------|--------------|
| [Quickstart](quickstart.md) | Invoice | + | Any | 5 min | Quick start |
| [Invoice](invoice-extraction.md) | Invoice | ++ | VLM/LLM | 15 min | Nested entities |
| [Research](research-paper.md) | Paper | +++ | LLM | 30 min | Complex ontology |
| [ID Card](id-card.md) | ID Card | ++ | VLM | 15 min | Vision extraction |
| [Insurance](insurance-policy.md) | Policy | ++ | LLM | 20 min | Financial docs |

---

## Example Structure

Each example follows this structure:

### 1. Overview
- Document type and use case
- What you'll learn
- Prerequisites

### 2. Template Definition
- Complete Pydantic models
- Field descriptions
- Relationship definitions

### 3. Processing
- CLI commands
- Python API code
- Configuration options

### 4. Results
- Expected output structure
- Graph visualization
- Statistics

### 5. Troubleshooting
- Common issues
- Solutions
- Tips

---

## Getting Started

### Prerequisites

```bash
# Install docling-graph
uv sync --extra all

# Verify installation
uv run docling-graph --version
```

### Choose Your Example

**New to docling-graph?**
→ Start with [Quickstart](quickstart.md)

**Processing invoices?**
→ See [Invoice Extraction](invoice-extraction.md)

**Working with research papers?**
→ See [Research Paper](research-paper.md)

**Extracting from ID cards?**
→ See [ID Card](id-card.md)

**Processing insurance documents?**
→ See [Insurance Policy](insurance-policy.md)

---

## Example Templates

All example templates are available in the repository:

```
docs/examples/templates/
├── invoice.py           # Invoice extraction
├── rheology_research.py # Research paper
├── id_card.py          # ID card extraction
└── insurance.py        # Insurance policy
```

### Using Example Templates

```bash
# Clone repository
git clone https://github.com/DS4SD/docling-graph.git
cd docling-graph

# Use example template
uv run docling-graph convert document.pdf \
    --template "docs.examples.templates.invoice.Invoice"
```

---

## Common Patterns

### Pattern 1: Simple Entity Extraction

```python
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    """Simple invoice model."""
    invoice_number: str = Field(description="Invoice number")
    total: float = Field(description="Total amount")
    date: str = Field(description="Invoice date")
```

### Pattern 2: Nested Entities

```python
class Address(BaseModel):
    """Address component."""
    street: str
    city: str
    postal_code: str

class Organization(BaseModel):
    """Organization entity."""
    name: str
    address: Address  # Nested component
```

### Pattern 3: Graph Relationships

```python
def edge(label: str, **kwargs):
    """Helper for graph edges."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

class Invoice(BaseModel):
    """Invoice with relationships."""
    invoice_number: str
    issued_by: Organization = edge(label="ISSUED_BY")
    sent_to: Client = edge(label="SENT_TO")
```

---

## Best Practices

### 1. Start Simple

```python
# ✅ Good - Start with basic fields
class Invoice(BaseModel):
    invoice_number: str
    total: float

# ❌ Avoid - Too complex initially
class Invoice(BaseModel):
    invoice_number: str
    total: float
    line_items: List[LineItem]
    issued_by: Organization
    sent_to: Client
    # ... 20 more fields
```

### 2. Add Descriptions

```python
# ✅ Good - Clear descriptions
invoice_number: str = Field(
    description="The unique invoice identifier",
    examples=["INV-001", "2024-001"]
)

# ❌ Avoid - No guidance
invoice_number: str
```

### 3. Use Examples

```python
# ✅ Good - Concrete examples
total: float = Field(
    description="Total amount",
    examples=[1234.56, 999.99]
)

# ❌ Avoid - Abstract examples
total: float = Field(
    description="Total amount",
    examples=["amount"]
)
```

---

## Testing Your Templates

### Quick Test

```bash
# Test with example document
uv run docling-graph convert test.pdf \
    --template "my_templates.Invoice" \
    --output-dir "test_output"

# Inspect results
uv run docling-graph inspect test_output/
```

### Iterative Development

```python
# 1. Start simple
class Invoice(BaseModel):
    invoice_number: str
    total: float

# 2. Test extraction
config = PipelineConfig(
    source="test.pdf",
    template=Invoice
)
config.run()

# 3. Review results
# 4. Add more fields
# 5. Repeat
```

---

## Next Steps

1. **[Quickstart →](quickstart.md)** - Get started in 5 minutes
2. **[Invoice Extraction →](invoice-extraction.md)** - Complete invoice example
3. **[Advanced Topics →](../10-advanced/index.md)** - Custom backends and more

---

## Additional Resources

### Documentation
- [Schema Definition](../03-schema-definition/index.md) - Template creation guide
- [CLI Reference](../07-cli/index.md) - Command-line usage
- [Python API](../08-api/index.md) - Programmatic usage

### Example Scripts
- `docs/examples/scripts/` - Python scripts
- `docs/examples/templates/` - Template files
- `docs/examples/data/` - Sample documents

### Community
- [GitHub Issues](https://github.com/DS4SD/docling-graph/issues) - Report issues
- [Discussions](https://github.com/DS4SD/docling-graph/discussions) - Ask questions

---

## Quick Reference

### Run Example

```bash
# CLI
uv run docling-graph convert document.pdf \
    --template "docs.examples.templates.invoice.Invoice"

# Python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="docs.examples.templates.invoice.Invoice"
)
config.run()
```

### View Results

```bash
# Inspect graph
uv run docling-graph inspect outputs/

# View statistics
cat outputs/graph_stats.json

# Read CSV
cat outputs/nodes.csv
cat outputs/edges.csv
```

---

**Navigation:** [← Batch Processing](../08-api/batch-processing.md) | [Next: Advanced Topics →](../10-advanced/index.md)