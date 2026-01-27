# DoclingDocument Input Example


## Overview

This example demonstrates how to process pre-converted DoclingDocument JSON files, enabling reprocessing of documents without re-running OCR or document conversion.

**Time:** 10 minutes

---

## Use Case: BillingDocument Reprocessing

Reprocess a previously converted invoice document with a different template or extraction strategy, without re-running expensive OCR operations.

### Document Source

**File:** `invoice_docling.json`

**Type:** DoclingDocument JSON

**Content:** Pre-processed invoice with structure, text, and layout information.

---

## Creating DoclingDocument Files

### Method 1: Export from Docling Graph

```bash
# First run: Convert PDF and export DoclingDocument
uv run docling-graph convert invoice.pdf \
    --template "templates.billing_document.BillingDocument" \
    --export-docling-json

# This creates: outputs/invoice_docling.json
```

### Method 2: Use Docling Directly

```python
from docling.document_converter import DocumentConverter

# Convert document with Docling
converter = DocumentConverter()
result = converter.convert("invoice.pdf")

# Export DoclingDocument
with open("invoice_docling.json", "w") as f:
    f.write(result.document.model_dump_json(indent=2))
```

### Method 3: Custom Pipeline

```python
from docling_core.types.doc import DoclingDocument
import json

# Create custom DoclingDocument
doc = DoclingDocument(
    schema_name="DoclingDocument",
    version="1.0.0",
    name="custom_invoice",
    # ... add pages, body, furniture
)

# Save to JSON
with open("custom_docling.json", "w") as f:
    json.dump(doc.model_dump(), f, indent=2)
```

---

## Template Definition

We'll use an invoice template for this example.

```python
from pydantic import BaseModel, Field
from docling_graph.utils import edge

class Address(BaseModel):
    """Address component."""
    model_config = {'is_entity': False}
    
    street: str = Field(description="Street address")
    city: str = Field(description="City")
    postal_code: str = Field(description="Postal code")
    country: str = Field(description="Country")

class Company(BaseModel):
    """Company entity."""
    model_config = {
        'is_entity': True,
        'graph_id_fields': ['name']
    }
    
    name: str = Field(description="Company name")
    address: Address = Field(description="Company address")
    tax_id: str | None = Field(default=None, description="Tax ID")

class LineItem(BaseModel):
    """Invoice line item."""
    model_config = {'is_entity': False}
    
    description: str = Field(description="Item description")
    quantity: float = Field(description="Quantity")
    unit_price: float = Field(description="Unit price")
    total: float = Field(description="Line total")

class BillingDocument(BaseModel):
    """Complete invoice structure."""
    model_config = {'is_entity': True}
    
    document_no: str = Field(description="Invoice number")
    date: str = Field(description="Invoice date")
    issuer: Company = edge("ISSUED_BY", description="Issuing company")
    client: Company = edge("BILLED_TO", description="Client company")
    line_items: list[LineItem] = Field(description="Invoice line items")
    subtotal: float = Field(description="Subtotal amount")
    tax: float = Field(description="Tax amount")
    total: float = Field(description="Total amount")
```

**Save as:** `templates/billing_document.py`

---

## Processing with CLI

### Basic DoclingDocument Processing

```bash
# Process DoclingDocument JSON
uv run docling-graph convert invoice_docling.json \
    --template "templates.billing_document.BillingDocument" \
    --backend llm \
    --inference remote
```

### Reprocess with Different Template

```bash
# First extraction
uv run docling-graph convert invoice.pdf \
    --template "templates.billing_document.BasicInvoice" \
    --export-docling-json

# Reprocess with detailed template (no OCR needed)
uv run docling-graph convert outputs/invoice_docling.json \
    --template "templates.billing_document.DetailedInvoice" \
    --output-dir "outputs/detailed"
```

### Batch Reprocessing

```bash
# Reprocess multiple DoclingDocument files
for file in outputs/*_docling.json; do
    uv run docling-graph convert "$file" \
        --template "templates.billing_document.BillingDocument" \
        --output-dir "outputs/reprocessed"
done
```

---

## Processing with Python API

### Basic Usage

```python
from docling_graph import run_pipeline, PipelineConfig
from templates.billing_document import BillingDocument

# Configure pipeline for DoclingDocument input
config = PipelineConfig(
    source="invoice_docling.json",
    template=Invoice,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one"
)

# Run pipeline (skips document conversion)
context = run_pipeline(config)
```

### Two-Stage Processing

```python
import json
from docling_graph import run_pipeline, PipelineConfig
from templates.billing_document import BasicInvoice, DetailedInvoice

# Stage 1: Initial extraction with basic template
stage1_config = PipelineConfig(
    source="invoice.pdf",
    template=BasicInvoice,
    backend="llm",
    inference="remote",
    export_docling_json=True
)
stage1_context = run_pipeline(stage1_config)

# Stage 2: Detailed extraction from DoclingDocument
docling_json = json.dumps(stage1_context.docling_document.export_to_dict())
stage2_config = PipelineConfig(
    source=docling_json,
    template=DetailedInvoice,
    backend="llm",
    inference="remote"
)
run_pipeline(stage2_config)
```

### Batch Reprocessing

```python
import json
from docling_graph import run_pipeline, PipelineConfig
from templates.billing_document import BillingDocument

# Example: DoclingDocument JSON strings from a datastore
docling_documents = [
    "...docling_json_1...",
    "...docling_json_2...",
]

for doc_json in docling_documents:
    print("Reprocessing DoclingDocument...")
    
    config = PipelineConfig(
        source=doc_json,
        template=Invoice,
        backend="llm",
        inference="remote"
    )
    
    try:
        run_pipeline(config)
        print("✅ Completed")
    except Exception as e:
        print(f"❌ Failed: {e}")
```

---

## Expected Output

### Graph Structure

```
Invoice (root node)
├── ISSUED_BY → Company (Acme Corp)
│   └── address (embedded)
├── BILLED_TO → Company (Client Inc)
│   └── address (embedded)
└── line_items (list)
    ├── LineItem 1
    ├── LineItem 2
    └── LineItem 3
```

### Processing Benefits

**With DoclingDocument Input:**
- ⚡ **Faster**: Skips OCR and document conversion
- 💰 **Cheaper**: No OCR processing costs
- 🔄 **Reusable**: Process same document with different templates
- 🎯 **Consistent**: Same document structure every time

**Comparison:**

| Operation | PDF Input | DoclingDocument Input |
|-----------|-----------|----------------------|
| OCR | ✅ Required | ❌ Skipped |
| Conversion | ✅ Required | ❌ Skipped |
| Extraction | ✅ Yes | ✅ Yes |
| Graph Build | ✅ Yes | ✅ Yes |
| **Time** | ~30s | ~5s |

---

## DoclingDocument Structure

### Required Fields

```json
{
  "schema_name": "DoclingDocument",  // Required
  "version": "1.0.0",                // Required
  "name": "document_name",
  "pages": {
    "0": {
      "page_no": 0,
      "size": {"width": 612, "height": 792}
    }
  },
  "body": {
    "self_ref": "#/body",
    "children": []
  },
  "furniture": {}
}
```

### Validation

The pipeline validates:
<br>✅ `schema_name` must be "DoclingDocument"
<br>✅ `version` field must be present
<br>✅ Valid JSON structure
<br>✅ Required fields present

---

## Troubleshooting

### 🐛 Invalid Schema

**Error:**
```
ValidationError: schema_name must be 'DoclingDocument', got 'CustomDocument'
```

**Solution:**
```json
{
  "schema_name": "DoclingDocument",  // Must be exactly this
  "version": "1.0.0",
  ...
}
```

### 🐛 Missing Version

**Error:**
```
ValidationError: Missing required field: version
```

**Solution:**
```json
{
  "schema_name": "DoclingDocument",
  "version": "1.0.0",  // Add version field
  ...
}
```

### 🐛 Invalid JSON

**Error:**
```
ValidationError: Invalid JSON in DoclingDocument file
```

**Solution:**
```bash
# Validate JSON syntax
python -m json.tool invoice_docling.json

# Or use jq
jq . invoice_docling.json
```

### 🐛 File Not Found

**Error:**
```
ConfigurationError: File not found: invoice_docling.json
```

**Solution:**
```bash
# Check file exists
ls -la invoice_docling.json

# Check file path
pwd
# Use absolute path if needed
uv run docling-graph convert /full/path/to/invoice_docling.json ...
```

---

## Best Practices

### 👍 Version Your DoclingDocuments

```python
# Add version to filename
doc_file = f"invoice_v{version}_docling.json"

# Or in metadata
doc = DoclingDocument(
    schema_name="DoclingDocument",
    version="1.0.0",
    name=f"invoice_v{version}",
    ...
)
```

### 👍 Store Metadata

```json
{
  "schema_name": "DoclingDocument",
  "version": "1.0.0",
  "name": "invoice_001",
  "metadata": {
    "source_file": "invoice.pdf",
    "processed_date": "2024-01-15",
    "ocr_engine": "docling",
    "template_version": "2.0"
  },
  ...
}
```

### 👍 Validate Before Processing

```python
from docling_graph.core.input.validators import DoclingDocumentValidator
import json

# Validate DoclingDocument
validator = DoclingDocumentValidator()

with open("invoice_docling.json") as f:
    content = f.read()

try:
    validator.validate(content)
    print("✅ Valid DoclingDocument")
except ValidationError as e:
    print(f"❌ Invalid: {e.message}")
```

### 👍 Archive Original PDFs

```python
from pathlib import Path
import shutil

# Keep original PDF alongside DoclingDocument
pdf_file = Path("invoice.pdf")
docling_file = Path("invoice_docling.json")
archive_dir = Path("archive")

# Archive structure
archive_dir.mkdir(exist_ok=True)
shutil.copy(pdf_file, archive_dir / pdf_file.name)
shutil.copy(docling_file, archive_dir / docling_file.name)
```

---

## Advanced Usage

### Custom DoclingDocument Creation

```python
from docling_core.types.doc import DoclingDocument, Page, Size
import json

# Create custom DoclingDocument
doc = DoclingDocument(
    schema_name="DoclingDocument",
    version="1.0.0",
    name="custom_invoice",
    pages={
        "0": Page(
            page_no=0,
            size=Size(width=612, height=792)
        )
    },
    body={
        "self_ref": "#/body",
        "children": []
    },
    furniture={}
)

# Save
with open("custom_docling.json", "w") as f:
    json.dump(doc.model_dump(), f, indent=2)
```

### Merging Multiple DoclingDocuments

```python
from docling_core.types.doc import DoclingDocument
import json

# Load multiple documents
docs = []
for file in ["doc1_docling.json", "doc2_docling.json"]:
    with open(file) as f:
        docs.append(json.load(f))

# Merge (simplified example)
merged = docs[0].copy()
for doc in docs[1:]:
    # Merge pages, body, etc.
    merged["pages"].update(doc["pages"])

# Save merged document
with open("merged_docling.json", "w") as f:
    json.dump(merged, f, indent=2)
```

### Extracting Specific Pages

```python
import json

# Load DoclingDocument
with open("multi_page_docling.json") as f:
    doc = json.load(f)

# Extract specific pages
pages_to_keep = ["0", "2", "4"]  # Keep pages 0, 2, 4
doc["pages"] = {
    k: v for k, v in doc["pages"].items()
    if k in pages_to_keep
}

# Save filtered document
with open("filtered_docling.json", "w") as f:
    json.dump(doc, f, indent=2)
```

---

## Use Cases

### 1. Template Experimentation

Test different templates without re-running OCR:

```python
templates = [
    "templates.billing_document.BasicInvoice",
    "templates.billing_document.DetailedInvoice",
    "templates.billing_document.MinimalInvoice"
]

for template in templates:
    config = PipelineConfig(
        source="invoice_docling.json",
        template=template
    )
    run_pipeline(config)
```

### 2. A/B Testing Extraction Strategies

```python
# Test different backends
for backend in ["llm", "vlm"]:
    config = PipelineConfig(
        source="invoice_docling.json",
        template=Invoice,
        backend=backend
    )
    run_pipeline(config)
```

### 3. Incremental Processing

```python
# Process in stages
stages = [
    ("basic", BasicTemplate),
    ("detailed", DetailedTemplate),
    ("enriched", EnrichedTemplate)
]

source = "invoice_docling.json"
for stage_name, template in stages:
    config = PipelineConfig(
        source=source,
        template=template
    )
    run_pipeline(config)
```

---

## Next Steps

- **[Input Formats Guide](../../fundamentals/pipeline-configuration/input-formats.md)** - Complete input format reference
- **[Examples Index](index.md)** - Browse all examples