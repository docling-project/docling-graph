# Invoice Extraction


## Overview

Extract complete structured data from invoices including issuer, client, line items, and financial details.

**Document Type:** Invoice (PDF/JPG)  
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

## Template Definition

### Complete Template

**File:** `invoice_template.py`

```python
"""
Complete invoice extraction template.
Demonstrates entities, components, and relationships.
"""

from typing import List
from pydantic import BaseModel, ConfigDict, Field

# Edge helper
def edge(label: str, **kwargs):
    """Helper to create graph edges."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

# --- Components (is_entity=False) ---

class Address(BaseModel):
    """Physical address component."""
    
    model_config = ConfigDict(is_entity=False)
    
    street: str = Field(
        description="Street name and number",
        examples=["123 Main St", "456 Oak Avenue"]
    )
    
    city: str = Field(
        description="City name",
        examples=["New York", "San Francisco"]
    )
    
    postal_code: str = Field(
        description="Postal or ZIP code",
        examples=["10001", "94102"]
    )
    
    country: str | None = Field(
        default=None,
        description="Country name or code",
        examples=["USA", "United States"]
    )

class LineItem(BaseModel):
    """Individual line item in invoice."""
    
    model_config = ConfigDict(is_entity=False)
    
    description: str = Field(
        description="Product or service description",
        examples=["Web Development Services", "Consulting Hours"]
    )
    
    quantity: float = Field(
        description="Quantity of items",
        examples=[1.0, 10.0, 100.0]
    )
    
    unit: str | None = Field(
        default=None,
        description="Unit of measurement",
        examples=["hours", "items", "pcs"]
    )
    
    unit_price: float = Field(
        description="Price per unit",
        examples=[50.00, 100.00, 25.50]
    )
    
    total: float = Field(
        description="Total price (quantity × unit_price)",
        examples=[500.00, 1000.00, 2550.00]
    )

# --- Entities (is_entity=True, default) ---

class Organization(BaseModel):
    """Organization entity (issuer)."""
    
    name: str = Field(
        description="Legal organization name",
        examples=["Acme Corporation", "ABC Services Inc"]
    )
    
    phone: str | None = Field(
        default=None,
        description="Contact phone number",
        examples=["+1-555-0100", "(555) 123-4567"]
    )
    
    email: str | None = Field(
        default=None,
        description="Contact email",
        examples=["billing@acme.com", "info@abc.com"]
    )
    
    website: str | None = Field(
        default=None,
        description="Company website",
        examples=["www.acme.com", "https://abc.com"]
    )
    
    # Relationship to address
    located_at: Address = edge(label="LOCATED_AT")

class Client(BaseModel):
    """Client entity (recipient)."""
    
    name: str = Field(
        description="Client name (person or organization)",
        examples=["John Doe", "XYZ Company"]
    )
    
    phone: str | None = Field(
        default=None,
        description="Client phone number",
        examples=["+1-555-0200"]
    )
    
    email: str | None = Field(
        default=None,
        description="Client email",
        examples=["john@example.com"]
    )
    
    # Relationship to address
    lives_at: Address = edge(label="LIVES_AT")

# --- Root Entity ---

class Invoice(BaseModel):
    """Root invoice entity."""
    
    invoice_number: str = Field(
        description="Unique invoice identifier",
        examples=["INV-001", "2024-001"]
    )
    
    date: str = Field(
        description="Invoice date (any format)",
        examples=["2024-01-15", "January 15, 2024"]
    )
    
    currency: str = Field(
        description="Currency code",
        examples=["USD", "EUR", "GBP"]
    )
    
    subtotal: float = Field(
        description="Amount before tax",
        examples=[1000.00, 5000.00]
    )
    
    vat_rate: float | str | None = Field(
        default=None,
        description="VAT/tax rate percentage",
        examples=["7.5", 10.0, "8.5%"]
    )
    
    vat_amount: float = Field(
        description="Total tax amount",
        examples=[75.00, 500.00]
    )
    
    total: float = Field(
        description="Final total amount",
        examples=[1075.00, 5500.00]
    )
    
    # Relationships
    issued_by: Organization = edge(label="ISSUED_BY")
    sent_to: Client = edge(label="SENT_TO")
    contains_items: List[LineItem] = edge(label="CONTAINS_ITEM")
```

---

## Processing

### Using CLI

```bash
# Process invoice with VLM (recommended for images)
uv run docling-graph convert invoice.jpg \
    --template "invoice_template.Invoice" \
    --backend vlm \
    --processing-mode one-to-one \
    --output-dir "outputs/invoice"

# Process PDF with LLM
uv run docling-graph convert invoice.pdf \
    --template "invoice_template.Invoice" \
    --backend llm \
    --inference remote \
    --output-dir "outputs/invoice"
```

### Using Python API

**File:** `process_invoice.py`

```python
"""Process invoice using Python API."""

import os
from docling_graph import run_pipeline, PipelineConfig

# Set API key if using remote
os.environ["MISTRAL_API_KEY"] = "your-key"

# Configure pipeline
config = PipelineConfig(
    source="invoice.pdf",
    template="invoice_template.Invoice",
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-small-latest",
    output_dir="outputs/invoice"
)

# Run extraction
print("Processing invoice...")
run_pipeline(config)
print("✅ Complete! Check outputs/invoice/")
```

Run it:

```bash
uv run python process_invoice.py
```

---

## Expected Results

### Graph Structure

```
Invoice (INV-001)
├── ISSUED_BY → Organization (Acme Corp)
│   └── LOCATED_AT → Address (123 Main St, NYC)
├── SENT_TO → Client (John Doe)
│   └── LIVES_AT → Address (456 Oak Ave, SF)
└── CONTAINS_ITEM → LineItem (Web Development)
    └── CONTAINS_ITEM → LineItem (Consulting)
```

### Nodes CSV

**outputs/invoice/nodes.csv:**
```csv
id,label,type,invoice_number,date,total,currency
invoice_1,INV-001,Invoice,INV-001,2024-01-15,1075.00,USD
org_1,Acme Corp,Organization,,,
client_1,John Doe,Client,,,
addr_1,123 Main St,Address,,,
addr_2,456 Oak Ave,Address,,,
item_1,Web Development,LineItem,,,
item_2,Consulting,LineItem,,,
```

### Edges CSV

**outputs/invoice/edges.csv:**
```csv
source,target,type
invoice_1,org_1,ISSUED_BY
invoice_1,client_1,SENT_TO
invoice_1,item_1,CONTAINS_ITEM
invoice_1,item_2,CONTAINS_ITEM
org_1,addr_1,LOCATED_AT
client_1,addr_2,LIVES_AT
```

### Statistics

**outputs/invoice/graph_stats.json:**
```json
{
  "node_count": 7,
  "edge_count": 6,
  "density": 0.143,
  "avg_degree": 1.714,
  "node_types": {
    "Invoice": 1,
    "Organization": 1,
    "Client": 1,
    "Address": 2,
    "LineItem": 2
  },
  "edge_types": {
    "ISSUED_BY": 1,
    "SENT_TO": 1,
    "CONTAINS_ITEM": 2,
    "LOCATED_AT": 1,
    "LIVES_AT": 1
  }
}
```

---

## Visualization

### Interactive HTML

```bash
# Open visualization
uv run docling-graph inspect outputs/invoice/
```

**Features:**
- Click nodes to see properties
- Hover over edges to see relationships
- Zoom and pan
- Search nodes
- Filter by type

### Neo4j Import

```bash
# Export as Cypher
uv run docling-graph convert invoice.pdf \
    --template "invoice_template.Invoice" \
    --export-format cypher \
    --output-dir "outputs/neo4j"

# Import to Neo4j
cat outputs/neo4j/graph.cypher | cypher-shell -u neo4j -p password
```

---

## Template Breakdown

### 1. Components vs Entities

```python
# Component (embedded, no separate node)
class Address(BaseModel):
    model_config = ConfigDict(is_entity=False)
    street: str
    city: str

# Entity (separate node in graph)
class Organization(BaseModel):
    # is_entity=True by default
    name: str
    located_at: Address  # Creates edge to Address
```

### 2. Edge Definitions

```python
def edge(label: str, **kwargs):
    """Creates graph relationship."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

class Invoice(BaseModel):
    # Creates ISSUED_BY edge from Invoice to Organization
    issued_by: Organization = edge(label="ISSUED_BY")
```

### 3. Optional Fields

```python
# Required field
name: str = Field(description="Required name")

# Optional field
phone: str | None = Field(
    default=None,
    description="Optional phone"
)
```

### 4. Lists

```python
# List of line items
contains_items: List[LineItem] = edge(label="CONTAINS_ITEM")

# Creates multiple edges:
# Invoice --CONTAINS_ITEM--> LineItem1
# Invoice --CONTAINS_ITEM--> LineItem2
```

---

## Customization

### Add More Fields

```python
class Invoice(BaseModel):
    # Existing fields...
    
    # Add payment terms
    payment_terms: str | None = Field(
        default=None,
        description="Payment terms",
        examples=["Net 30", "Due on receipt"]
    )
    
    # Add due date
    due_date: str | None = Field(
        default=None,
        description="Payment due date",
        examples=["2024-02-15"]
    )
    
    # Add notes
    notes: str | None = Field(
        default=None,
        description="Additional notes or comments"
    )
```

### Add Validation

```python
from pydantic import field_validator

class Invoice(BaseModel):
    subtotal: float
    vat_amount: float
    total: float
    
    @field_validator('total')
    @classmethod
    def validate_total(cls, v, info):
        """Ensure total = subtotal + vat_amount."""
        subtotal = info.data.get('subtotal', 0)
        vat = info.data.get('vat_amount', 0)
        expected = subtotal + vat
        
        if abs(v - expected) > 0.01:  # Allow small rounding
            raise ValueError(
                f"Total {v} doesn't match subtotal + VAT ({expected})"
            )
        return v
```

---

## Troubleshooting

### 🐛 Missing Line Items

Line items not extracted

**Solution:**
```python
# Make line items optional
contains_items: List[LineItem] = edge(
    label="CONTAINS_ITEM",
    default_factory=list  # Empty list if none found
)
```

### 🐛 Address Not Parsed

Address fields empty

**Solution:**
```python
# Make address fields optional
class Address(BaseModel):
    street: str | None = Field(default=None, ...)
    city: str | None = Field(default=None, ...)
    postal_code: str | None = Field(default=None, ...)
```

### 🐛 Wrong Currency

Currency extracted incorrectly

**Solution:**
```python
# Add examples and validation
currency: str = Field(
    description="Three-letter currency code (ISO 4217)",
    examples=["USD", "EUR", "GBP", "CHF"],
    pattern="^[A-Z]{3}$"  # Enforce 3 uppercase letters
)
```

---

## Best Practices

### 👍 Clear Descriptions

```python
# ✅ Good - Specific and clear
invoice_number: str = Field(
    description="The unique invoice identifier, typically alphanumeric",
    examples=["INV-001", "2024-001", "ABC123"]
)

# ❌ Avoid - Vague
invoice_number: str = Field(description="Invoice number")
```

### 👍 Concrete Examples

```python
# ✅ Good - Real examples
total: float = Field(
    description="Total amount to be paid",
    examples=[1234.56, 999.99, 5000.00]
)

# ❌ Avoid - Abstract examples
total: float = Field(
    description="Total amount",
    examples=["amount", "total"]
)
```

### 👍 Appropriate Optionality

```python
# ✅ Good - Core fields required, details optional
class Invoice(BaseModel):
    invoice_number: str  # Required
    total: float  # Required
    notes: str | None = Field(default=None)  # Optional
```

---

## Next Steps

1. **[Research Paper →](research-paper.md)** - Complex scientific documents
2. **[ID Card →](id-card.md)** - Vision-based extraction
3. **[Schema Definition →](../../fundamentals/schema-definition/index.md)** - Advanced templates