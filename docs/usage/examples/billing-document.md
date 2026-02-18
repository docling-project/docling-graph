# Billing Document Extraction

## Overview

Extract complete structured data from billing documents (invoices, credit notes, receipts, etc.) including parties, line items, taxes, and payment information.

**Document Type:** Billing Documents (PDF/JPG)  
**Time:** 15 minutes  
**Backend:** VLM (recommended) or LLM

---

## Prerequisites

```bash
# Install
pip install docling-graph

# Verify installation
uv run docling-graph --version
```

---

## Template Reference

The `BillingDocument` template is a streamlined schema located at:
**`docs/examples/templates/billing_document.py`**

### Key Features

- **Multiple Document Types**: Invoice, Credit Note, Debit Note, Receipt
- **Simplified Structure**: 10 core classes (reduced from 40+)
- **Embedded Fields**: Contact info and totals directly in parent classes
- **Clear Extraction Prompts**: Each field has "LOOK FOR", "EXTRACT", and "EXAMPLES" sections
- **Essential Tax Handling**: VAT, GST, Sales Tax support
- **Payment Methods**: Bank transfer, card, cash, direct debit

### Root Model

```python
from examples.templates.billing_document import BillingDocument

# The root entity with document_number as unique identifier
class BillingDocument(BaseModel):
    """Root billing document entity."""
    model_config = ConfigDict(graph_id_fields=["document_number"])
    
    # Core fields
    document_number: str  # Primary identifier (e.g., "INV-2024-001")
    document_type: DocumentType  # INVOICE, CREDIT_NOTE, RECEIPT, etc.
    issue_date: date | None
    due_date: date | None
    currency: str | None  # ISO 4217 code (EUR, USD, GBP)
    
    # Financial totals (embedded)
    subtotal: float | None
    discount_total: float | None
    tax_total: float | None
    total_amount: float | None
    balance_due: float | None
    
    # Relationships (edges)
    seller: Party  # Who issued the document
    buyer: Party | None  # Who receives it
    line_items: List[LineItem]  # Line items
    taxes: List[Tax]  # Tax breakdown
    payment: Payment | None  # Payment info
    delivery: Delivery | None  # Delivery info
    references: List[DocumentReference]  # Related documents
```

### Simplified Party Model

```python
class Party(BaseModel):
    """Party with embedded contact and address information."""
    model_config = ConfigDict(graph_id_fields=["name", "tax_id"])
    
    name: str  # Company/person name
    tax_id: str | None  # VAT/Tax ID
    
    # Contact info (embedded)
    email: str | None
    phone: str | None
    website: str | None
    
    # Address (embedded)
    street: str | None
    city: str | None
    postal_code: str | None
    country: str | None
```

### Simplified LineItem Model

```python
class LineItem(BaseModel):
    """Line item with embedded price and quantity."""
    model_config = ConfigDict(graph_id_fields=["line_number", "item_code"])
    
    line_number: str  # Line position
    description: str | None
    
    # Quantity and price (embedded)
    quantity: float | None
    unit: str | None  # EA, KG, HUR, etc.
    unit_price: float | None
    discount_percent: float | None
    line_total: float | None
    
    # Relationships
    item: Item | None  # Product/service reference
    tax: Tax | None  # Tax for this line
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

from docling_graph import PipelineConfig, run_pipeline

config = PipelineConfig(
    source="https://upload.wikimedia.org/wikipedia/commons/9/9f/Swiss_QR-Bill_example.jpg",
    template="docs.examples.templates.billing_document.BillingDocument",
    backend="vlm",
    inference="local",
    processing_mode="one-to-one"
)

# Run extraction
print("Processing billing document...")
context = run_pipeline(config)
graph = context.knowledge_graph
print(f"✅ Complete! Extracted {graph.number_of_nodes()} nodes")
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
  │   ├─ name: "Acme Corp"
  │   ├─ email: "contact@acme.com"
  │   ├─ street: "123 Main St"
  │   └─ city: "Paris"
  ├─ BILLED_TO → Party (Buyer/Customer)
  │   ├─ name: "Client Inc"
  │   └─ email: "billing@client.com"
  ├─ CONTAINS_LINE → LineItem (multiple)
  │   ├─ line_number: "1"
  │   ├─ quantity: 10.0
  │   ├─ unit_price: 50.00
  │   ├─ REFERENCES_ITEM → Item
  │   └─ HAS_TAX → Tax
  ├─ HAS_TAX → Tax (document-level)
  └─ HAS_PAYMENT_INFO → Payment
      ├─ method: "Bank Transfer"
      ├─ iban: "FR76..."
      └─ due_date: "2024-02-15"
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
id,label,type,document_number,document_type,issue_date,total_amount,name,email,city
doc_1,BillingDocument,entity,INV-2024-001,Invoice,2024-01-15,1075.00,,,
party_1,Party,entity,,,,,Acme Corp,contact@acme.com,Paris
party_2,Party,entity,,,,,Client Inc,billing@client.com,London
line_1,LineItem,entity,1,,,50.00,,,
item_1,Item,entity,,,,,Widget Pro,,
```

### Sample edges.csv

```csv
source,target,label
doc_1,party_1,ISSUED_BY
doc_1,party_2,BILLED_TO
doc_1,line_1,CONTAINS_LINE
line_1,item_1,REFERENCES_ITEM
line_1,tax_1,HAS_TAX
doc_1,payment_1,HAS_PAYMENT_INFO
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
from docling_graph import PipelineConfig, run_pipeline

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
        backend="llm"
    )
    
    try:
        run_pipeline(config)
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
| **RECEIPT** | Payment receipt | Proof of payment |
| **OTHER** | Other billing docs | Custom types |

The `document_type` field automatically normalizes various input formats.

---

## Key Fields Reference

### Core Document Fields

```python
document_number: str          # "INV-2024-001" (required, unique ID)
document_type: DocumentType   # INVOICE, CREDIT_NOTE, RECEIPT, etc.
issue_date: date | None       # Document issue date
due_date: date | None         # Payment due date
currency: str | None          # "EUR", "USD", "GBP" (ISO 4217)
notes: str | None             # General notes or remarks
```

### Financial Totals (Embedded)

```python
subtotal: float | None        # Subtotal before tax and discounts
discount_total: float | None  # Total discount amount
tax_total: float | None       # Total tax amount
total_amount: float | None    # Final total (including tax)
amount_paid: float | None     # Amount already paid
balance_due: float | None     # Remaining balance
```

### Party Information

```python
seller: Party                 # Seller/supplier (required)
buyer: Party | None           # Customer/buyer
```

**Party fields:**
```python
name: str                     # Company/person name
tax_id: str | None            # VAT/Tax ID
email: str | None             # Email address
phone: str | None             # Phone number
website: str | None           # Website URL
street: str | None            # Street address
city: str | None              # City
postal_code: str | None       # Postal/ZIP code
country: str | None           # Country name or code
```

### Line Items

```python
line_items: List[LineItem]    # Line items with products/services
```

**LineItem fields:**
```python
line_number: str              # Line position (required)
description: str | None       # Item description
quantity: float | None        # Quantity
unit: str | None              # Unit of measure (EA, KG, etc.)
unit_price: float | None      # Price per unit
discount_percent: float | None # Discount percentage
line_total: float | None      # Total for this line
item: Item | None             # Product/service reference
tax: Tax | None               # Tax for this line
```

### Tax Information

```python
taxes: List[Tax]              # Tax breakdown
```

**Tax fields:**
```python
tax_type: TaxType             # VAT, GST, SALES_TAX, OTHER
rate_percent: float | None    # Tax rate (e.g., 20.0)
taxable_amount: float | None  # Amount on which tax is calculated
tax_amount: float | None      # Calculated tax amount
exemption_reason: str | None  # Exemption reason if applicable
```

### Payment Information

```python
payment: Payment | None       # Payment details
```

**Payment fields:**
```python
method: PaymentMethod         # BANK_TRANSFER, CARD, CASH, etc.
due_date: date | None         # Payment due date
terms: str | None             # Payment terms (e.g., "Net 30")
bank_name: str | None         # Bank name
iban: str | None              # IBAN
bic: str | None               # BIC/SWIFT code
reference: str | None         # Payment reference
```

---

## Best Practices

### Field Descriptions with Extraction Hints

The simplified template includes enhanced extraction prompts:

```python
# ✅ Good - Specific with visual cues
document_number: str = Field(
    ...,
    description=(
        "Invoice/document number (primary identifier). "
        "LOOK FOR: Large, bold text in header, 'Invoice No', 'Invoice Number', "
        "'Receipt No', 'Facture No' labels (usually top right). "
        "EXTRACT: Complete number including prefixes/suffixes. "
        "EXAMPLES: 'INV-2024-001', '2024-INV-12345', 'REC-001'"
    ),
    examples=["INV-2024-001", "2024-INV-12345", "REC-001"],
)

# ❌ Avoid - Vague
document_number: str = Field(description="Document number")
```

### Required vs Optional

```python
# Required fields
document_number: str          # Always needed for identification
seller: Party                 # Always present

# Optional fields
buyer: Party | None = None    # May not be present
due_date: date | None = None  # Not all documents have due dates
payment: Payment | None = None # Not all documents have payment info
```

### Validation

The template includes essential validators:

- Currency format validation (ISO 4217)
- Enum normalization (handles various input formats)
- Automatic currency symbol conversion (€ → EUR, $ → USD, £ → GBP)

---

## Troubleshooting

### Common Issues

**"Field document_number is required"**
→ Ensure the document has a visible document number

**"Currency must be 3 uppercase letters"**
→ Use ISO 4217 codes: EUR, USD, GBP (symbols are auto-converted)

**"Cannot normalize enum value"**
→ Check DocumentType values match: INVOICE, CREDIT_NOTE, RECEIPT, OTHER

### Improving Extraction Quality

1. **Use VLM for images** - Better layout understanding
2. **Provide clear examples** - Template includes diverse examples
3. **Use vision pipeline** - For complex layouts: `--docling-config vision`
4. **Enable chunking** - For large documents: `--use-chunking`

---

## Template Simplification (v2.0.0)

The template has been significantly simplified:

- **Reduced from 2230 lines to 717 lines** (68% reduction)
- **Reduced from 40+ classes to 10 core classes**
- **Embedded contact info** - Email, phone, address directly in Party
- **Embedded totals** - Financial totals directly in BillingDocument
- **Simplified line items** - Direct fields instead of nested objects
- **Better extraction prompts** - Clear "LOOK FOR", "EXTRACT", "EXAMPLES" sections

See `BILLING_DOCUMENT_CHANGELOG.md` for detailed migration guide.

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
- **717 lines** (simplified from 2230)
- **10 core classes** with clear extraction prompts
- **Changelog**: `docs/examples/templates/BILLING_DOCUMENT_CHANGELOG.md`

---

## Next Steps

1. **Try the example** - Process a sample billing document
2. **Customize template** - Adapt for your specific needs
3. **Integrate with Neo4j** - Build a document knowledge base
4. **Automate workflows** - Set up batch processing pipelines