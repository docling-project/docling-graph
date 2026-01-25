# Quickstart


## Overview

Get started with docling-graph in **5 minutes** by extracting structured data from a simple billing document.

**What You'll Learn:**
- Basic template creation
- Running your first extraction
- Viewing results

**Prerequisites:**
- Python 3.10+
- `uv` package manager
- A sample billing document (PDF or image)

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

Create a file `simple_billing_doc.py`:

```python
"""Simple billing document template for quickstart."""

from pydantic import BaseModel, Field

class SimpleBillingDoc(BaseModel):
    """A simple billing document model."""
    
    document_no: str = Field(
        description="The unique document identifier",
        examples=["INV-001", "2024-001"]
    )
    
    date: str = Field(
        description="Document date in any format",
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
# Process billing document
uv run docling-graph convert billing_doc.pdf \
    --template "simple_billing_doc.SimpleBillingDoc" \
    --output-dir "quickstart_output"
```

### Option B: Using Python API

Create `run_quickstart.py`:

```python
"""Quickstart extraction script."""

from docling_graph import run_pipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    source="billing_doc.pdf",
    template="simple_billing_doc.SimpleBillingDoc",
    output_dir="quickstart_output"
)

# Run extraction
print("Processing billing document...")
run_pipeline(config)
print("✅ Complete! Check quickstart_output/")
```

Run it:

```bash
uv run python run_quickstart.py
```

---

## Step 4: View Results

### Interactive Visualization

```bash
uv run docling-graph inspect quickstart_output/
```

Opens an HTML visualization showing extracted nodes, relationships, and interactive exploration.

### CSV Data

```bash
cat quickstart_output/nodes.csv
cat quickstart_output/edges.csv
```

## Expected Output Structure

```
quickstart_output/
├── metadata.json                # Pipeline metadata
├── docling/                     # Docling conversion output
│   ├── document.json            # Docling format
│   └── document.md              # Markdown export
└── docling_graph/               # Graph outputs
    ├── graph.json               # Complete graph
    ├── nodes.csv                # Extracted data
    ├── edges.csv                # Relationships
    ├── graph.html               # Interactive visualization
    └── report.md                # Summary report
```

---

## Troubleshooting

**Template Not Found:**
```bash
# Ensure template is in current directory or use absolute path
uv run docling-graph convert billing_doc.pdf --template "$(pwd)/simple_billing_doc.SimpleBillingDoc"
```

**No Data Extracted:**
```bash
# Use verbose logging to debug
uv run docling-graph --verbose convert billing_doc.pdf --template simple_billing_doc.SimpleBillingDoc
```

**API Key Error:**
```bash
# Use local inference (default) or set API key
export MISTRAL_API_KEY='your-key'
```


### Improve Your Template

Add more fields:

```python
class ImprovedBillingDoc(BaseModel):
    """Improved billing document with more fields."""

    document_no: str = Field(description="Document number")
    date: str = Field(description="Document date")
    total: float = Field(description="Total amount")
    currency: str = Field(description="Currency")
    
    # New fields
    issuer_name: str = Field(
        description="Company that issued the document",
        examples=["Acme Corp", "ABC Company"]
    )
    
    client_name: str = Field(
        description="Client receiving the document",
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

class BillingDoc(BaseModel):
    """Billing document with relationships."""
    document_no: str
    total: float
    issued_by: Organization = edge(label="ISSUED_BY")
```

### Try Different Backends

```bash
# VLM for images (faster)
uv run docling-graph convert billing_doc.jpg \
    --template "simple_billing_doc.SimpleBillingDoc" \
    --backend vlm

# LLM for complex documents
uv run docling-graph convert billing_doc.pdf \
    --template "simple_billing_doc.SimpleBillingDoc" \
    --backend llm \
    --inference remote
```

---

## Learn More

- **[Billing Document Extraction](../usage/examples/billing-document.md)** - Full billing document with relationships
- **[Schema Definition](../fundamentals/schema-definition/index.md)** - Template creation guide
- **[CLI Reference](../usage/cli/index.md)** - All CLI commands