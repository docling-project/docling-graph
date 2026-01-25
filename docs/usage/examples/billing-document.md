# Billing Document Extraction


## Overview

Extract complete structured data from billing documents (invoices, credit notes, receipts, etc.) including parties, line items, taxes, and payment information.

**Document Type:** Billing Documents (PDF/JPG)  
**Time:** 15 minutes  
**Backend:** VLM (recommended) or LLM

---

## Prerequisites

```bash
# Install with all features
uv sync --extra all

# Verify installation
uv run docling-graph --version
```

---

## Template Reference

The `BillingDocument` template is a comprehensive schema located at:
**`docs/examples/templates/billing_document.py`**

### Key Features

- **Multiple Document Types**: Invoice, Credit Note, Debit Note, Pro Forma, Receipt
- **Comprehensive Party Support**: Issuer, buyer, payee, tax representative
- **Multi-level Charges**: Document, line, and price level allowances/charges
- **Tax Handling**: VAT categories, rates, exemptions
- **Payment Methods**: Bank transfer, card, direct debit, QR codes
- **Standards Compliant**: EN 16931, Peppol BIS, UBL

### Root Model

```python
from examples.templates.billing_document import BillingDocument

# The root entity with document_no as unique identifier
class BillingDocument(BaseModel):
    """Root billing document entity."""
    model_config = ConfigDict(graph_id_fields=["document_no"])
    
    document_no: str  # Primary identifier (e.g., "INV-2024-001")
    document_type: DocumentType  # INVOICE, CREDIT_NOTE, etc.
    issue_date: date | None
    due_date: date | None
    currency: str | None  # ISO 4217 code
    
    # Relationships (edges)
    issuer: Party  # Who issued the document
    buyer: Party | None  # Who receives it
    lines: List[DocumentLine]  # Line items
    totals: DocumentTotals  # Financial totals
    payment: Settlement | None  # Payment info
    # ... and more
```

---

## Usage Examples

### CLI - Process Image

```bash
# Process billing document image with VLM
uv run docling-graph convert "https://upload.wikimedia.org/wikipedia/commons/9/9f/Swiss_QR-Bill_example.jpg" \
    --template "docs.examples.templates.billing_document.BillingDocument" \
    --backend vlm \
    --processing-mode one-to-one \
    --output-dir "outputs/billing_doc"
```

### CLI - Process PDF

```bash
# Process PDF with LLM
uv run docling-graph convert billing_document.pdf \
    --template "docs.examples.templates.billing_document.BillingDocument" \
    --backend llm \
    --inference remote \
    --output-dir "outputs/billing_doc"
```

### Python API

**File:** `process_billing_doc.py`

```python
"""Process billing document using Python API."""

from docling_graph import PipelineConfig

config = PipelineConfig(
    source="https://upload.wikimedia.org/wikipedia/commons/9/9f/Swiss_QR-Bill_example.jpg",
    template="docs.examples.templates.billing_document.BillingDocument",
    backend="vlm",
    inference="local",
    processing_mode="one-to-one",
    model_override="mistral-small-latest",
    output_dir="outputs/billing_doc"
)

# Run extraction
print("Processing billing document...")
config.run()
print("✅ Complete! Check outputs/billing_doc/")
```

**Run:**
```bash
uv run python process_billing_doc.py
```

---

## Expected Output

### Graph Structure

```
BillingDocument (root)
  ├─ ISSUED_BY → Party (Seller/Supplier)
  │   └─ LOCATED_AT → PostalAddress
  ├─ SENT_TO → Party (Buyer/Customer)
  │   └─ LOCATED_AT → PostalAddress
  ├─ CONTAINS_LINE → DocumentLine (multiple)
  │   ├─ HAS_ITEM → Item
  │   ├─ HAS_PRICE → Price
  │   └─ HAS_TAX → TaxDetail
  ├─ HAS_TOTALS → DocumentTotals
  ├─ HAS_TAX_SUMMARY → TaxSummary
  └─ HAS_SETTLEMENT → Settlement
      └─ HAS_PAYMENT_INSTRUCTION → PaymentInstruction
          └─ HAS_BANK_ACCOUNT → BankAccount
```

### Files Generated

**outputs/billing_doc/docling_graph/**

- `nodes.csv` - All entities and components
- `edges.csv` - Relationships between nodes
- `graph.json` - Complete graph structure
- `graph.html` - Interactive visualization
- `report.md` - Extraction statistics

### Sample nodes.csv

```csv
id,label,type,document_no,document_type,issue_date,total
doc_1,BillingDocument,entity,INV-2024-001,Invoice,2024-01-15,1075.00
party_1,Party,entity,,,Acme Corp,
party_2,Party,entity,,,Client Inc,
line_1,DocumentLine,component,,,,50.00
```

### Sample edges.csv

```csv
source,target,type
doc_1,party_1,ISSUED_BY
doc_1,party_2,SENT_TO
doc_1,line_1,CONTAINS_LINE
doc_1,totals_1,HAS_TOTALS
```

---

## Visualization

```bash
# Open interactive visualization
uv run docling-graph inspect outputs/billing_doc/
```

**Features:**
- Interactive node exploration
- Relationship filtering
- Property inspection
- Export capabilities

---

## Advanced Usage

### Export as Cypher for Neo4j

```bash
# Export as Cypher script
uv run docling-graph convert billing_document.pdf \
    --template "docs.examples.templates.billing_document.BillingDocument" \
    --export-format cypher \
    --output-dir "outputs/neo4j"

# Import to Neo4j
cat outputs/neo4j/docling_graph/graph.cypher | cypher-shell -u neo4j -p password
```

### Batch Processing

```python
"""Process multiple billing documents."""

from pathlib import Path
from docling_graph import PipelineConfig

documents = [
    "https://example.com/invoice1.pdf",
    "https://example.com/invoice2.pdf",
    "https://example.com/credit_note1.pdf",
]

for doc in documents:
    doc_name = Path(doc).stem
    config = PipelineConfig(
        source=doc,
        template="docs.examples.templates.billing_document.BillingDocument",
        backend="llm",
        output_dir=f"outputs/batch/{doc_name}"
    )
    
    try:
        config.run()
        print(f"✅ {doc_name}")
    except Exception as e:
        print(f"❌ {doc_name}: {e}")
```

---

## Document Types Supported

The `BillingDocument` template supports multiple document types:

| Type | Description | Use Case |
|------|-------------|----------|
| **INVOICE** | Standard invoice | Sales, services |
| **CREDIT_NOTE** | Credit memo | Returns, corrections |
| **DEBIT_NOTE** | Debit memo | Additional charges |
| **PRO_FORMA** | Pro forma invoice | Customs, quotes |
| **RECEIPT** | Payment receipt | Proof of payment |

The `document_type` field automatically normalizes various input formats.

---

## Key Fields Reference

### Core Document Fields

```python
document_no: str          # "INV-2024-001" (required, unique ID)
document_type: DocumentType  # INVOICE, CREDIT_NOTE, etc.
issue_date: date | None   # Document issue date
due_date: date | None     # Payment due date
currency: str | None      # "EUR", "USD", "GBP" (ISO 4217)
```

### Party Information

```python
issuer: Party             # Seller/supplier (required)
buyer: Party | None       # Customer/buyer
payee: Party | None       # Payment recipient (if different)
tax_representative: Party | None  # Tax representative
```

### Financial Information

```python
lines: List[DocumentLine]  # Line items with products/services
totals: DocumentTotals     # Net, tax, gross amounts
tax_summary: TaxSummary | None  # Tax breakdown by category
payment: Settlement | None  # Payment terms and instructions
```

---

## Best Practices

### Field Descriptions

```python
# ✅ Good - Specific and clear
document_no: str = Field(
    description="Human-readable document number. Look for 'Invoice No', 'Document Number', etc.",
    examples=["INV-2024-001", "2024-INV-12345", "REC-001"]
)

# ❌ Avoid - Vague
document_no: str = Field(description="Document number")
```

### Required vs Optional

```python
# Required fields
document_no: str  # Always needed for identification
issuer: Party     # Always present
totals: DocumentTotals  # Always present

# Optional fields
buyer: Party | None = None  # May not be present
due_date: date | None = None  # Not all documents have due dates
```

### Validation

The template includes built-in validators:

- Currency format validation (ISO 4217)
- Date order validation (issue_date < due_date)
- Enum normalization (handles various input formats)
- Positive amount validation

---

## Troubleshooting

### Common Issues

**"Field document_no is required"**
→ Ensure the document has a visible document number

**"Currency must be 3 uppercase letters"**
→ Use ISO 4217 codes: EUR, USD, GBP (not €, $, £)

**"Cannot normalize enum value"**
→ Check DocumentType values match: INVOICE, CREDIT_NOTE, etc.

### Improving Extraction Quality

1. **Use VLM for images** - Better layout understanding
2. **Provide clear examples** - Add diverse examples to field descriptions
3. **Use vision pipeline** - For complex layouts: `--docling-config vision`
4. **Enable chunking** - For large documents: `--use-chunking`

---

## Related Examples

- **[ID Card Extraction](id-card.md)** - Identity documents
- **[Insurance Policy](insurance-policy.md)** - Legal documents
- **[Batch Processing](../api/batch-processing.md)** - Multiple documents

---

## Additional Resources

### Documentation

- **[Schema Definition](../../fundamentals/schema-definition/index.md)** - Template creation guide
- **[Graph Management](../../fundamentals/graph-management/index.md)** - Working with graphs
- **[Neo4j Integration](../../fundamentals/graph-management/neo4j-integration.md)** - Database import

### Template Source

- **Full Template**: `docs/examples/templates/billing_document.py`
- **1998 lines** with comprehensive field definitions
- **EN 16931, Peppol BIS, UBL compliant**

---

## Next Steps

1. **Try the example** - Process a sample billing document
2. **Customize template** - Adapt for your specific needs
3. **Integrate with Neo4j** - Build a document knowledge base
4. **Automate workflows** - Set up batch processing pipelines