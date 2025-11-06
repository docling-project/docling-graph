# **Complete Guide to Creating Pydantic Templates for Knowledge Graph Extraction**

This guide will walk you through creating Pydantic templates that are optimized for LLM-based document extraction and automatic conversion to knowledge graphs. The key is to **be explicit**, **provide examples**, and **think about the graph structure** while designing your models. More examples could be found in the `tempaltes` directory.


## **Table of Contents**

1. [Core Concepts](#1-core-concepts)
2. [Template Structure](#2-template-structure)
3. [The Edge Helper Function](#3-the-edge-helper-function)
4. [Entity vs Component Classification](#4-entity-vs-component-classification)
5. [Field Definition Best Practices](#5-field-definition-best-practices)
6. [Advanced Features](#6-advanced-features)
7. [Complete Template Examples](#7-complete-template-examples)
8. [Common Patterns](#8-common-patterns)


## **1. Core Concepts**

### **Why Pydantic for Document Extraction?**

Your system uses Pydantic models for three key purposes:

1. **LLM Guidance**: Descriptions and examples guide the LLM to extract accurate data
2. **Validation**: Field validators ensure data quality
3. **Graph Structure**: The models define nodes and edges for your knowledge graph

### **Key Terminology**

| Term | Definition | Example |
| :-- | :-- | :-- |
| **Entity** | A unique, identifiable object in your graph | `Person`, `Organization`, `Invoice` |
| **Component** | A value object that's deduplicated by content | `Address`, `MonetaryAmount` |
| **Node** | Any Pydantic model that becomes a graph node | All entities and components |
| **Edge** | A relationship between nodes | `ISSUED_BY`, `LIVES_AT`, `LOCATED_AT` |
| **graph_id_fields** | Fields used to create stable, unique node IDs | `['name']`, `['document_number']` |



## **2. Template Structure**

### **Basic File Organization**

```python
"""
Brief description of what this template extracts.
Mention the document type and any key features.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, List, Any
from datetime import date  # Import types as needed

# --- Edge Helper Function ---
def Edge(label: str, **kwargs: Any) -> Any:
    """Helper function to create a Pydantic Field with edge metadata."""
    return Field(..., json_schema_extra={'edge_label': label}, **kwargs)

# --- Reusable Components ---
# Value objects like Address, MonetaryAmount

# --- Reusable Entities ---
# Common entities like Person, Organization

# --- Document-Specific Models ---
# Models unique to this document type

# --- Root Document Model ---
# The main entry point (Invoice, IDCard, etc.)
```


### **Import Checklist**

```python
# Always needed
from pydantic import BaseModel, Field
from typing import Optional, List, Any

# Common additions
from pydantic import ConfigDict, field_validator  # For config and validation
from datetime import date, datetime               # For date fields
from decimal import Decimal                       # For precise monetary values
import re                                          # For regex validators
```



## **3. The Edge Helper Function**

### **Definition** (Copy this to every template)

```python
def Edge(label: str, **kwargs: Any) -> Any:
    """Helper function to create a Pydantic Field with edge metadata."""
    return Field(..., json_schema_extra={'edge_label': label}, **kwargs)
```


### **How It Works**

The `Edge()` function marks a field as a **relationship** rather than a simple attribute. The `graph_converter.py` uses the `edge_label` metadata to create edges in your graph.

### **Usage Patterns**

```python
# Required relationship (field must be present)
issued_by: Organization = Edge(
    label="ISSUED_BY",
    description="The organization that issued this invoice"
)

# Optional relationship
lives_at: Optional[Address] = Edge(
    label="LIVES_AT",
    description="Physical address (e.g., home address)"
)

# List of relationships (one-to-many)
contains_items: List[LineItem] = Edge(
    label="CONTAINS_ITEM",
    default_factory=list,  # REQUIRED for lists
    description="Line items in the invoice"
)
```


### **Edge Label Conventions**

Use descriptive, all-caps labels with underscores:

- **GOOD**: `ISSUED_BY`, `LIVES_AT`, `LOCATED_AT`, `CONTAINS_ITEM`
- **GOOD**: `BELONGS_TO`, `HAS_GUARANTEE`, `OFFERS_PLAN`
- **BAD**: `issuedBy`, `located-at`, `has guarantee`


## **4. Entity vs Component Classification**

### **The Critical Distinction**

| Aspect | Entity | Component |
| :-- | :-- | :-- |
| **What** | Unique, identifiable objects | Value objects, deduplicated by content |
| **Examples** | `Person`, `Organization`, `Invoice` | `Address`, `MonetaryAmount` |
| **graph_id_fields** | Required | Not used |
| **is_entity** | `True` (default) | `False` (must set explicitly) |
| **Deduplication** | By ID fields | By all fields (content-based) |

### **Entity Pattern**

```python
class Person(BaseModel):
    """
    A person entity.
    Uniquely identified by name and date of birth.
    """
    model_config = ConfigDict(
        graph_id_fields=['first_name', 'last_name', 'date_of_birth']
    )
    
    first_name: Optional[str] = Field(...)
    last_name: Optional[str] = Field(...)
    date_of_birth: Optional[date] = Field(...)
    # ... other fields
```


### **Component Pattern**

```python
class Address(BaseModel):
    """
    A physical address component.
    Deduplicated by content - same address = same node.
    """
    model_config = ConfigDict(is_entity=False)
    
    street_address: Optional[str] = Field(...)
    city: Optional[str] = Field(...)
    postal_code: Optional[str] = Field(...)
    country: Optional[str] = Field(...)
```


### **When to Use Each**

**Use Entity for:**

- Things you want to track individually (people, organizations, documents)
- Objects with unique identifiers
- Data where duplicates should be separate nodes

**Use Component for:**

- Data that should be shared when identical (addresses, amounts, dates)
- Value objects without identity
- Data where duplicates should be merged


## **5. Field Definition Best Practices**

### **The Anatomy of a Field**

```python
field_name: Type = Field(
    default_or_ellipsis,              # ... = required, None = optional
    description="Clear, detailed description for the LLM",
    examples=["Example 1", "Example 2", "Example 3"]
)
```


### **Required vs Optional**

```python
# Required field - LLM MUST extract this
document_number: str = Field(
    ...,  # Three dots = required
    description="The unique document ID",
    examples=["INV-2024-001", "DOC123"]
)

# Optional field - may be missing
phone: Optional[str] = Field(
    None,  # or default=None
    description="Contact phone number",
    examples=["+33 1 23 45 67 89"]
)

# Optional with default
status: str = Field(
    "pending",  # Default value
    description="Current status",
    examples=["pending", "approved", "rejected"]
)
```


### **Writing Effective Descriptions**

**Good descriptions guide the LLM:**

```python
# GOOD - Clear, specific, with guidance
date_of_birth: Optional[date] = Field(
    None,
    description=(
        "The person's date of birth. "
        "Look for text like 'Date of birth', 'Date de naiss.', or 'Born on'. "
        "Parse formats like 'DD MM YYYY' or 'DDMMYYYY' and normalize to YYYY-MM-DD."
    ),
    examples=["1990-05-15", "1985-12-20"]
)

# BAD - Too vague
date_of_birth: Optional[date] = Field(None, description="Birth date")
```


### **Examples: The Secret Weapon**

**Provide 2-5 realistic examples per field:**

```python
# GOOD - Diverse, realistic examples
email: Optional[str] = Field(
    None,
    description="Contact email address",
    examples=[
        "jean.dupont@email.com",
        "contact@company.fr",
        "info@organization.org"
    ]
)

# BAD - Generic or single example
email: Optional[str] = Field(
    None,
    description="Email",
    examples=["test@test.com"]
)
```

**For lists, show list examples:**

```python
# GOOD - Shows list structure
guarantees: Optional[List[str]] = Field(
    default_factory=list,
    description="List of coverage guarantees",
    examples=[
        ["Fire protection", "Water damage", "Theft"],
        ["Basic coverage", "Extended warranty"],
        ["Liability", "Property damage", "Medical expenses"]
    ]
)
```



## **6. Advanced Features**

### **Field Validators**

Use validators to ensure data quality:

```python
class MonetaryAmount(BaseModel):
    """Monetary value with validation."""
    
    value: float = Field(
        ...,
        description="Numeric amount",
        examples=[500.00, 1250.50]
    )
    
    currency: Optional[str] = Field(
        None,
        description="ISO 4217 currency code",
        examples=["EUR", "USD", "CHF"]
    )
    
    @field_validator('value')
    @classmethod
    def validate_positive(cls, v):
        """Ensure amount is non-negative."""
        if v < 0:
            raise ValueError('Monetary amount must be non-negative')
        return v
    
    @field_validator('currency')
    @classmethod
    def validate_currency_format(cls, v):
        """Ensure currency is 3 uppercase letters."""
        if v and not (len(v) == 3 and v.isupper()):
            raise ValueError('Currency must be 3 uppercase letters (ISO 4217)')
        return v
```


### **Pre-validators (mode='before')**

Transform data before validation:

```python
@field_validator('given_names', mode='before')
def ensure_list(cls, v):
    """Convert string to list if needed."""
    if isinstance(v, str):
        return [v]
    return v

@field_validator('email', mode='before')
@classmethod
def normalize_email(cls, v):
    """Convert email to lowercase."""
    if v:
        return v.lower().strip()
    return v
```


### **String Representations**

Add `__str__` for debugging and logging:

```python
class Person(BaseModel):
    first_name: Optional[str] = Field(...)
    last_name: Optional[str] = Field(...)
    
    def __str__(self):
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p)

class Address(BaseModel):
    street_address: Optional[str] = Field(...)
    city: Optional[str] = Field(...)
    postal_code: Optional[str] = Field(...)
    country: Optional[str] = Field(...)
    
    def __str__(self):
        parts = [self.street_address, self.city, self.postal_code, self.country]
        return ", ".join(p for p in parts if p)
```



## **7. Complete Template Examples**

### **Example 1: Simple Invoice Template**

```python
"""
Pydantic template for invoice extraction.
Extracts issuer, client, line items, and financial totals.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Any
from datetime import date

# --- Edge Helper Function ---
def Edge(label: str, **kwargs: Any) -> Any:
    """Helper function to create a Pydantic Field with edge metadata."""
    return Field(..., json_schema_extra={'edge_label': label}, **kwargs)

# --- Component: Address ---
class Address(BaseModel):
    """Physical address component (deduplicated by content)."""
    model_config = ConfigDict(is_entity=False)
    
    street: str = Field(
        description="Street name and number",
        examples=["123 Main St", "45 Avenue des Champs-Élysées"]
    )
    postal_code: str = Field(
        description="Postal or ZIP code",
        examples=["75001", "10001"]
    )
    city: str = Field(
        description="City name",
        examples=["Paris", "New York"]
    )
    country: Optional[str] = Field(
        default=None,
        description="Country code or name",
        examples=["FR", "US", "France"]
    )
    
    def __str__(self):
        parts = [self.street, self.postal_code, self.city, self.country]
        return ", ".join(p for p in parts if p)

# --- Entity: Organization ---
class Organization(BaseModel):
    """Organization entity (unique by name)."""
    model_config = ConfigDict(graph_id_fields=['name'])
    
    name: str = Field(
        description="Legal name of the organization",
        examples=["Acme Corp", "TechStart SAS", "Global Industries Inc"]
    )
    email: Optional[str] = Field(
        default=None,
        description="Contact email",
        examples=["contact@acme.com", "info@techstart.fr"]
    )
    phone: Optional[str] = Field(
        default=None,
        description="Contact phone number",
        examples=["+33 1 23 45 67 89", "+1 555-123-4567"]
    )
    
    # Edge to Address
    located_at: Address = Edge(
        label="LOCATED_AT",
        description="Organization's physical address"
    )
    
    def __str__(self):
        return self.name

# --- Entity: Person ---
class Person(BaseModel):
    """Person entity (unique by full name)."""
    model_config = ConfigDict(graph_id_fields=['full_name'])
    
    full_name: str = Field(
        description="Full name of the person",
        examples=["Jean Dupont", "Maria Garcia", "John Smith"]
    )
    email: Optional[str] = Field(
        default=None,
        description="Contact email",
        examples=["jean.dupont@email.com"]
    )
    
    # Edge to Address
    lives_at: Address = Edge(
        label="LIVES_AT",
        description="Person's residential address"
    )
    
    def __str__(self):
        return self.full_name

# --- Component: LineItem ---
class LineItem(BaseModel):
    """Invoice line item (not deduplicated)."""
    
    description: str = Field(
        description="Product or service description",
        examples=["Professional services", "Software license", "Consulting hours"]
    )
    quantity: float = Field(
        description="Quantity ordered",
        examples=[1.0, 10.0, 40.5]
    )
    unit_price: float = Field(
        description="Price per unit",
        examples=[99.99, 1500.00, 75.50]
    )
    total: float = Field(
        description="Total for this line (quantity × unit_price)",
        examples=[99.99, 15000.00, 3057.75]
    )

# --- Root Document: Invoice ---
class Invoice(BaseModel):
    """Root invoice document entity."""
    model_config = ConfigDict(graph_id_fields=['invoice_number'])
    
    invoice_number: str = Field(
        description="Unique invoice identifier",
        examples=["INV-2024-001", "F20240515", "123456"]
    )
    invoice_date: date = Field(
        description="Date the invoice was issued (YYYY-MM-DD)",
        examples=["2024-10-15", "2024-01-20"]
    )
    currency: str = Field(
        description="Currency code (ISO 4217)",
        examples=["EUR", "USD", "CHF"]
    )
    subtotal: float = Field(
        description="Total before tax",
        examples=[1000.00, 5432.10]
    )
    tax_amount: float = Field(
        description="Total tax amount",
        examples=[200.00, 1086.42]
    )
    total: float = Field(
        description="Final total amount due",
        examples=[1200.00, 6518.52]
    )
    
    # Edges
    issued_by: Organization = Edge(
        label="ISSUED_BY",
        description="Organization that issued the invoice"
    )
    sent_to: Person = Edge(
        label="SENT_TO",
        description="Person receiving the invoice"
    )
    contains_items: List[LineItem] = Edge(
        label="CONTAINS_ITEM",
        default_factory=list,
        description="Line items on the invoice"
    )
    
    def __str__(self):
        return f"Invoice {self.invoice_number}"
```


### **Example 2: Insurance Policy Template**

```python
"""
Pydantic template for insurance policy documents.
Extracts policy holder, insurer, coverage details, and guarantees.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, List, Any
from datetime import date

# --- Edge Helper Function ---
def Edge(label: str, **kwargs: Any) -> Any:
    """Helper function to create a Pydantic Field with edge metadata."""
    return Field(..., json_schema_extra={'edge_label': label}, **kwargs)

# --- Component: MonetaryAmount ---
class MonetaryAmount(BaseModel):
    """Monetary value component."""
    model_config = ConfigDict(is_entity=False)
    
    value: float = Field(
        ...,
        description="Numeric amount",
        examples=[500.00, 150000.00, 75.50]
    )
    currency: Optional[str] = Field(
        None,
        description="ISO 4217 currency code",
        examples=["EUR", "USD", "CHF"]
    )
    
    @field_validator('value')
    @classmethod
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v
    
    def __str__(self):
        return f"{self.value} {self.currency or ''}".strip()

# --- Component: Address ---
class Address(BaseModel):
    """Physical address component."""
    model_config = ConfigDict(is_entity=False)
    
    street_address: Optional[str] = Field(
        None,
        description="Street name and number",
        examples=["123 Rue de Rivoli", "45 Oak Street"]
    )
    city: Optional[str] = Field(
        None,
        description="City name",
        examples=["Paris", "Lyon", "New York"]
    )
    postal_code: Optional[str] = Field(
        None,
        description="Postal code",
        examples=["75001", "69002", "10001"]
    )
    country: Optional[str] = Field(
        None,
        description="Country",
        examples=["France", "USA", "FR"]
    )
    
    def __str__(self):
        parts = [self.street_address, self.city, self.postal_code, self.country]
        return ", ".join(p for p in parts if p)

# --- Entity: Person ---
class Person(BaseModel):
    """Person entity."""
    model_config = ConfigDict(graph_id_fields=['first_name', 'last_name', 'date_of_birth'])
    
    first_name: Optional[str] = Field(
        None,
        description="First name(s)",
        examples=["Jean", "Marie", "Pierre"]
    )
    last_name: Optional[str] = Field(
        None,
        description="Last name (surname)",
        examples=["Dupont", "Martin", "Bernard"]
    )
    date_of_birth: Optional[date] = Field(
        None,
        description="Date of birth (YYYY-MM-DD)",
        examples=["1985-03-12", "1990-06-20"]
    )
    
    # Edge to Address
    addresses: List[Address] = Edge(
        label="LIVES_AT",
        default_factory=list,
        description="Residential addresses"
    )
    
    def __str__(self):
        return f"{self.first_name} {self.last_name}".strip()

# --- Entity: Organization ---
class Organization(BaseModel):
    """Organization entity."""
    model_config = ConfigDict(graph_id_fields=['name'])
    
    name: str = Field(
        ...,
        description="Legal organization name",
        examples=["AXA Assurance", "Allianz France", "MAIF"]
    )
    tax_id: Optional[str] = Field(
        None,
        description="Tax ID or registration number",
        examples=["572 093 920", "FR12345678901"]
    )
    
    # Edge to Address
    addresses: List[Address] = Edge(
        label="LOCATED_AT",
        default_factory=list,
        description="Organization addresses"
    )
    
    def __str__(self):
        return self.name

# --- Entity: Coverage ---
class Coverage(BaseModel):
    """Insurance coverage/guarantee entity."""
    model_config = ConfigDict(graph_id_fields=['name'])
    
    name: str = Field(
        ...,
        description="Name of the coverage",
        examples=[
            "Fire and allied perils",
            "Water damage",
            "Theft protection",
            "Civil liability"
        ]
    )
    description: Optional[str] = Field(
        None,
        description="Detailed description of what's covered",
        examples=[
            "Covers damage to property caused by fire, explosion, or lightning",
            "Protection against water damage from pipes and plumbing"
        ]
    )
    coverage_limit: Optional[MonetaryAmount] = Field(
        None,
        description="Maximum coverage amount"
    )
    deductible: Optional[MonetaryAmount] = Field(
        None,
        description="Deductible amount (franchise)"
    )
    
    def __str__(self):
        return self.name

# --- Root Document: InsurancePolicy ---
class InsurancePolicy(BaseModel):
    """Root insurance policy document."""
    model_config = ConfigDict(graph_id_fields=['policy_number'])
    
    policy_number: str = Field(
        ...,
        description="Unique policy identifier",
        examples=["POL-2024-001", "12345678", "FR2024XYZ"]
    )
    policy_type: Optional[str] = Field(
        None,
        description="Type of insurance",
        examples=["Home insurance", "Auto insurance", "Health insurance"]
    )
    effective_date: Optional[date] = Field(
        None,
        description="Date policy becomes effective (YYYY-MM-DD)",
        examples=["2024-01-01", "2024-07-15"]
    )
    expiry_date: Optional[date] = Field(
        None,
        description="Date policy expires (YYYY-MM-DD)",
        examples=["2025-01-01", "2025-07-15"]
    )
    premium: Optional[MonetaryAmount] = Field(
        None,
        description="Premium amount"
    )
    
    # Edges
    policy_holder: Person = Edge(
        label="HELD_BY",
        description="Person who holds the policy"
    )
    insurer: Organization = Edge(
        label="ISSUED_BY",
        description="Insurance company"
    )
    coverages: List[Coverage] = Edge(
        label="INCLUDES_COVERAGE",
        default_factory=list,
        description="Coverage items included"
    )
    
    def __str__(self):
        return f"Policy {self.policy_number}"
```



## **8. Common Patterns**

### **Pattern 1: Nested Lists**

When you have lists of complex objects:

```python
class Beneficiary(BaseModel):
    """A person who benefits from the policy."""
    full_name: str = Field(...)
    relationship: Optional[str] = Field(
        None,
        examples=["Spouse", "Child", "Parent"]
    )
    percentage: Optional[float] = Field(
        None,
        description="Percentage of benefit (0-100)",
        examples=[50.0, 100.0, 25.0]
    )

class Policy(BaseModel):
    policy_number: str = Field(...)
    
    # List of beneficiaries
    beneficiaries: List[Beneficiary] = Edge(
        label="HAS_BENEFICIARY",
        default_factory=list,
        description="List of policy beneficiaries"
    )
```


### **Pattern 2: Optional Edges**

When relationships might not exist:

```python
class Document(BaseModel):
    document_id: str = Field(...)
    
    # Optional single relationship
    verified_by: Optional[Person] = Edge(
        label="VERIFIED_BY",
        description="Person who verified this document, if any"
    )
```


### **Pattern 3: Multiple Addresses**

Common for organizations and people:

```python
class Organization(BaseModel):
    model_config = ConfigDict(graph_id_fields=['name'])
    
    name: str = Field(...)
    
    # Multiple addresses
    addresses: List[Address] = Edge(
        label="LOCATED_AT",
        default_factory=list,
        description="Can include headquarters, branch offices, etc."
    )
```


### **Pattern 4: Conditional Fields**

When field requirements depend on document type:

```python
class Document(BaseModel):
    document_type: str = Field(
        description="Type of document",
        examples=["Invoice", "Receipt", "Credit Note"]
    )
    
    # Only for invoices
    payment_terms: Optional[str] = Field(
        None,
        description="Payment terms (only for invoices)",
        examples=["Net 30", "Due on receipt", "Net 60"]
    )
    
    # Only for credit notes
    original_invoice: Optional[str] = Field(
        None,
        description="Reference to original invoice (only for credit notes)",
        examples=["INV-2024-001"]
    )
```


### **Pattern 5: Enumerated Values**

For fields with limited options:

```python
from typing import Literal

class Person(BaseModel):
    first_name: str = Field(...)
    last_name: str = Field(...)
    
    # Using Literal for strict validation
    gender: Optional[Literal["M", "F", "Other"]] = Field(
        None,
        description="Gender",
        examples=["M", "F", "Other"]
    )
    
    # Or use validator for flexible input
    @field_validator('gender', mode='before')
    @classmethod
    def normalize_gender(cls, v):
        if v:
            v_upper = v.upper()
            if v_upper in ['M', 'MALE', 'H', 'HOMME']:
                return 'M'
            elif v_upper in ['F', 'FEMALE', 'FEMME']:
                return 'F'
        return v
```



## **Quick Reference Checklist**

When creating a new template:

- Import necessary modules (BaseModel, Field, ConfigDict, etc.)
- Define `Edge()` helper function
- Identify components (set `is_entity=False`)
- Identify entities (set `graph_id_fields=[...]`)
- Add clear descriptions to ALL fields
- Add 2-5 realistic examples per field
- Use `Edge()` for relationships
- Add validators where needed
- Add `__str__` methods for debugging
- Test with sample documents


## **Testing Your Template**

Create a simple test to verify your template works:

```python
# test_invoice.py
from invoice import Invoice, Organization, Person, Address, LineItem
from datetime import date

# Create test data
test_invoice = Invoice(
    invoice_number="TEST-001",
    invoice_date=date(2024, 10, 25),
    currency="EUR",
    subtotal=1000.00,
    tax_amount=200.00,
    total=1200.00,
    issued_by=Organization(
        name="Test Company",
        email="contact@test.com",
        phone="+33 1 23 45 67 89",
        located_at=Address(
            street="123 Test St",
            postal_code="75001",
            city="Paris",
            country="FR"
        )
    ),
    sent_to=Person(
        full_name="John Doe",
        email="john@example.com",
        lives_at=Address(
            street="456 Client Ave",
            postal_code="75002",
            city="Paris",
            country="FR"
        )
    ),
    contains_items=[
        LineItem(
            description="Consulting services",
            quantity=10.0,
            unit_price=100.00,
            total=1000.00
        )
    ]
)

# Verify it works
print(test_invoice)
print(test_invoice.model_dump_json(indent=2))
```