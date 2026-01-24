# Advanced Patterns


## Overview

This guide covers advanced Pydantic patterns for complex document structures, reusable components, and sophisticated validation scenarios. These patterns are drawn from production templates across multiple domains.

**In this guide:**
- Flexible measurement models
- Nested list patterns with edges
- Multiple address support
- Optional edges and conditional fields
- Reusable component library

---

## Pattern 1: Flexible Measurement with Range Support

### The Challenge

Scientific and technical documents often contain measurements in various formats:
- Single values: "25°C", "1.6 mPa.s"
- Ranges: "80-90°C", "1.5-2.0 mm"
- Text values: "High", "Low", "Stable"

### The Solution

```python
from typing import Union, Optional, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Measurement(BaseModel):
    """
    Flexible measurement supporting single values, ranges, or text.
    Can represent '25°C', '1.6 mPa.s', '80-90°C', or 'High'.
    """
    model_config = ConfigDict(is_entity=False)
    
    name: str = Field(
        description="Name of the measured property",
        examples=["Temperature", "Viscosity", "pH", "Concentration"]
    )
    
    text_value: Optional[str] = Field(
        default=None,
        description="Textual value if not numerical",
        examples=["High", "Low", "Stable", "Increasing"]
    )
    
    numeric_value: Optional[Union[float, int]] = Field(
        default=None,
        description="Single numerical value",
        examples=[25.0, 1.6, 8.2]
    )
    
    numeric_value_min: Optional[Union[float, int]] = Field(
        default=None,
        description="Minimum value for range measurements",
        examples=[80.0, 1.5]
    )
    
    numeric_value_max: Optional[Union[float, int]] = Field(
        default=None,
        description="Maximum value for range measurements",
        examples=[90.0, 2.0]
    )
    
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement",
        examples=["°C", "mPa.s", "wt%", "kg"]
    )
    
    condition: Optional[str] = Field(
        default=None,
        description="Measurement conditions or context",
        examples=["at 25°C", "after 24h", "under normal pressure"]
    )
    
    @model_validator(mode="after")
    def validate_value_consistency(self) -> Self:
        """Ensure value fields don't conflict."""
        has_single = self.numeric_value is not None
        has_min = self.numeric_value_min is not None
        has_max = self.numeric_value_max is not None
        
        # Reject ambiguous cases
        if has_single and has_min and has_max:
            raise ValueError(
                "Cannot specify all three: numeric_value, "
                "numeric_value_min, and numeric_value_max"
            )
        
        # Allow implicit range: if numeric_value + min/max, treat as range
        if has_single and (has_min or has_max):
            if has_max and not has_min:
                # Treat numeric_value as min
                self.numeric_value_min = self.numeric_value
                self.numeric_value = None
            elif has_min and not has_max:
                # Treat numeric_value as max
                self.numeric_value_max = self.numeric_value
                self.numeric_value = None
        
        return self
```

### Usage Examples

```python
# Single value
temp = Measurement(
    name="Temperature",
    numeric_value=25.0,
    unit="°C"
)

# Range
temp_range = Measurement(
    name="Temperature",
    numeric_value_min=80.0,
    numeric_value_max=90.0,
    unit="°C"
)

# Text value
quality = Measurement(
    name="Quality",
    text_value="High"
)

# With condition
viscosity = Measurement(
    name="Viscosity",
    numeric_value=1.6,
    unit="mPa.s",
    condition="at 25°C"
)
```

---

## Pattern 2: Nested List with Edges

### The Challenge

Complex documents have nested structures where list items themselves have relationships:

```
Assembly
  └─ Components (list)
      ├─ Material (edge)
      ├─ Role
      └─ Amount (edge)
```

### The Solution

```python
from typing import Any, List
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum

def edge(label: str, **kwargs: Any) -> Any:
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

class RoleEnum(str, Enum):
    PRIMARY = "Primary"
    SECONDARY = "Secondary"
    ADDITIVE = "Additive"

class Material(BaseModel):
    """Material entity."""
    model_config = ConfigDict(graph_id_fields=["name"])
    
    name: str = Field(
        description="Material name",
        examples=["Steel", "Aluminum", "Polymer"]
    )
    
    grade: Optional[str] = Field(
        None,
        description="Material grade or specification",
        examples=["304", "6061", "ABS"]
    )

class Component(BaseModel):
    """Component with material, role, and amount."""
    model_config = ConfigDict(graph_id_fields=["material", "role"])
    
    # Edge to Material entity
    material: Material = edge(
        label="USES_MATERIAL",
        description="The material used in this component"
    )
    
    role: RoleEnum = Field(
        description="Function of this component",
        examples=["Primary", "Secondary", "Additive"]
    )
    
    # Edge to Measurement component
    amount: Optional[Measurement] = edge(
        label="HAS_AMOUNT",
        description="Amount specification"
    )

class Assembly(BaseModel):
    """Root assembly containing components."""
    model_config = ConfigDict(graph_id_fields=["assembly_id"])
    
    assembly_id: str = Field(...)
    
    # List edge to Component entities
    components: List[Component] = edge(
        label="HAS_COMPONENT",
        default_factory=list,
        description="List of components in this assembly"
    )
```

### Graph Structure

```
Assembly-001
  ├─ HAS_COMPONENT → Component-1
  │                   ├─ USES_MATERIAL → Material(Steel)
  │                   └─ HAS_AMOUNT → Measurement(12.0 kg)
  └─ HAS_COMPONENT → Component-2
                      ├─ USES_MATERIAL → Material(Aluminum)
                      └─ HAS_AMOUNT → Measurement(5.0 kg)
```

---

## Pattern 3: Multiple Address Support

### The Challenge

Entities often have multiple addresses (home, work, billing, shipping):

```python
class Entity(BaseModel):
    """Entity that may have multiple addresses."""
    model_config = ConfigDict(graph_id_fields=["name"])
    
    name: str = Field(...)
    
    # Support multiple addresses
    addresses: List[Address] = edge(
        label="LOCATED_AT",
        default_factory=list,
        description="Physical addresses (headquarters, branches, etc.)"
    )
```

### Enhanced with Address Types

```python
from enum import Enum

class AddressType(str, Enum):
    HOME = "Home"
    WORK = "Work"
    BILLING = "Billing"
    SHIPPING = "Shipping"
    HEADQUARTERS = "Headquarters"
    BRANCH = "Branch"

class TypedAddress(BaseModel):
    """Address with type classification."""
    model_config = ConfigDict(is_entity=False)
    
    address_type: AddressType = Field(
        description="Type of address",
        examples=["Home", "Work", "Billing"]
    )
    
    street_address: Optional[str] = Field(None)
    city: Optional[str] = Field(None)
    postal_code: Optional[str] = Field(None)
    country: Optional[str] = Field(None)

class Organization(BaseModel):
    """Organization with typed addresses."""
    model_config = ConfigDict(graph_id_fields=["name"])
    
    name: str = Field(...)
    
    addresses: List[TypedAddress] = edge(
        label="LOCATED_AT",
        default_factory=list,
        description="Physical addresses with types"
    )
```

---

## Pattern 4: Optional Edges

### The Challenge

Some relationships are conditional - they may or may not exist:

```python
class Document(BaseModel):
    """Document that may or may not have a verifier."""
    model_config = ConfigDict(graph_id_fields=["document_id"])
    
    document_id: str = Field(...)
    
    # Required edge
    issued_by: Organization = edge(
        label="ISSUED_BY",
        description="Organization that issued this document"
    )
    
    # Optional edge - document may not be verified
    verified_by: Optional[Person] = edge(
        label="VERIFIED_BY",
        description="Person who verified this document, if verified"
    )
    
    # Optional edge - document may not be approved
    approved_by: Optional[Person] = edge(
        label="APPROVED_BY",
        description="Person who approved this document, if approved"
    )
```

### Graph Behavior

```
# Verified document
Document-001 --ISSUED_BY--> Org-A
Document-001 --VERIFIED_BY--> Person-A

# Unverified document
Document-002 --ISSUED_BY--> Org-B
# No VERIFIED_BY edge
```

---

## Pattern 5: Conditional Fields with Validators

### The Challenge

Some fields are only relevant for certain document types:

```python
class Document(BaseModel):
    """Document with type-specific fields."""
    
    document_type: str = Field(
        description="Type of document",
        examples=["Invoice", "Receipt", "Credit Note"]
    )
    
    # Field only relevant for invoices
    payment_terms: Optional[str] = Field(
        None,
        description="Payment terms (primarily for invoices)",
        examples=["Net 30", "Due on receipt", "Net 60"]
    )
    
    # Field only relevant for credit notes
    original_document_ref: Optional[str] = Field(
        None,
        description="Reference to original document (for credit notes)",
        examples=["INV-2024-001", "DOC-123456"]
    )
    
    @model_validator(mode="after")
    def validate_type_specific_fields(self) -> Self:
        """Validate fields based on document type."""
        if self.document_type == "Invoice":
            if not self.payment_terms:
                # Could warn or set default
                self.payment_terms = "Net 30"
        
        if self.document_type == "Credit Note":
            if not self.original_document_ref:
                raise ValueError(
                    "original_document_ref required for Credit Note"
                )
        
        return self
```

---

## Pattern 6: Polymorphic Fields

### The Challenge

A field might accept multiple types:

```python
class FlexibleValue(BaseModel):
    """Value that can be numeric or textual."""
    
    value: Union[str, int, float] = Field(
        ...,
        description=(
            "Value can be numeric (int/float) or textual. "
            "Extract as-is: '100', '25.5', or 'High'."
        ),
        examples=[100, 25.5, "High", "Medium"]
    )
    
    @field_validator("value", mode="before")
    @classmethod
    def coerce_value(cls, v: Any) -> Any:
        """Try to convert to number if possible."""
        if isinstance(v, str):
            # Try numeric conversion
            try:
                if "." in v:
                    return float(v)
                return int(v)
            except ValueError:
                # Keep as string
                return v
        return v
```

---

## Pattern 7: Hierarchical Structures

### The Challenge

Documents with nested sections or chapters:

```python
class Section(BaseModel):
    """Document section."""
    model_config = ConfigDict(graph_id_fields=["section_number"])
    
    section_number: str = Field(...)
    title: str = Field(...)
    content: str = Field(...)
    
    # Recursive: sections can contain subsections
    subsections: List["Section"] = edge(
        label="HAS_SUBSECTION",
        default_factory=list,
        description="Nested subsections"
    )

class Document(BaseModel):
    """Document with hierarchical structure."""
    model_config = ConfigDict(graph_id_fields=["document_id"])
    
    document_id: str = Field(...)
    
    sections: List[Section] = edge(
        label="HAS_SECTION",
        default_factory=list,
        description="Top-level sections"
    )
```

**Note:** Pydantic requires forward references for recursive models. Use string quotes for the type hint.

---

## Pattern 8: Reusable Component Library

### Common Components

Build a library of reusable components:

```python
# --- Address Component ---
class Address(BaseModel):
    """Physical address component (deduplicated by content)."""
    model_config = ConfigDict(is_entity=False)
    
    street_address: Optional[str] = Field(None)
    city: Optional[str] = Field(None)
    state_or_province: Optional[str] = Field(None)
    postal_code: Optional[str] = Field(None)
    country: Optional[str] = Field(None)
    
    def __str__(self) -> str:
        parts = [
            self.street_address,
            self.city,
            self.state_or_province,
            self.postal_code,
            self.country
        ]
        return ", ".join(p for p in parts if p)

# --- Monetary Amount Component ---
class MonetaryAmount(BaseModel):
    """Monetary value with currency (deduplicated by content)."""
    model_config = ConfigDict(is_entity=False)
    
    value: float = Field(...)
    currency: Optional[str] = Field(None)
    
    @field_validator("value")
    @classmethod
    def validate_positive(cls, v: Any) -> Any:
        if v < 0:
            raise ValueError("Amount must be non-negative")
        return v
    
    @field_validator("currency")
    @classmethod
    def validate_currency_format(cls, v: Any) -> Any:
        if v and not (len(v) == 3 and v.isupper()):
            raise ValueError("Currency must be 3 uppercase letters (ISO 4217)")
        return v
    
    def __str__(self) -> str:
        return f"{self.value} {self.currency or ''}".strip()

# --- Contact Information Component ---
class ContactInfo(BaseModel):
    """Contact information component."""
    model_config = ConfigDict(is_entity=False)
    
    email: Optional[str] = Field(None)
    phone: Optional[str] = Field(None)
    website: Optional[str] = Field(None)
    
    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v: Any) -> Any:
        if v:
            return v.lower().strip()
        return v

# --- Date Range Component ---
class DateRange(BaseModel):
    """Date range component."""
    model_config = ConfigDict(is_entity=False)
    
    start_date: Optional[date] = Field(None)
    end_date: Optional[date] = Field(None)
    
    @model_validator(mode="after")
    def validate_date_order(self) -> Self:
        if self.start_date and self.end_date:
            if self.end_date < self.start_date:
                raise ValueError("end_date must be after start_date")
        return self
```

### Common Entities

```python
# --- Person Entity ---
class Person(BaseModel):
    """Person entity (unique by name and date of birth)."""
    model_config = ConfigDict(
        graph_id_fields=["first_name", "last_name", "date_of_birth"]
    )
    
    first_name: Optional[str] = Field(None)
    last_name: Optional[str] = Field(None)
    date_of_birth: Optional[date] = Field(None)
    
    contact: Optional[ContactInfo] = Field(None)
    
    addresses: List[Address] = edge(
        label="LIVES_AT",
        default_factory=list,
        description="Residential addresses"
    )
    
    def __str__(self) -> str:
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p) or "Unknown"

# --- Organization Entity ---
class Organization(BaseModel):
    """Organization entity (unique by name)."""
    model_config = ConfigDict(graph_id_fields=["name"])
    
    name: str = Field(...)
    tax_id: Optional[str] = Field(None)
    
    contact: Optional[ContactInfo] = Field(None)
    
    addresses: List[Address] = edge(
        label="LOCATED_AT",
        default_factory=list,
        description="Business addresses"
    )
    
    def __str__(self) -> str:
        return self.name
```

---

## Pattern 9: String Representations

### Purpose

Add `__str__` methods for debugging, logging, and visualization:

```python
# Simple concatenation
class Person(BaseModel):
    first_name: Optional[str] = Field(...)
    last_name: Optional[str] = Field(...)
    
    def __str__(self) -> str:
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p) or "Unknown"

# With list handling
class Person(BaseModel):
    given_names: Optional[List[str]] = Field(...)
    last_name: Optional[str] = Field(...)
    
    def __str__(self) -> str:
        first_names = " ".join(self.given_names) if self.given_names else ""
        parts = [first_names, self.last_name]
        return " ".join(p for p in parts if p) or "Unknown"

# Address formatting
class Address(BaseModel):
    street_address: Optional[str] = Field(...)
    city: Optional[str] = Field(...)
    postal_code: Optional[str] = Field(...)
    country: Optional[str] = Field(...)
    
    def __str__(self) -> str:
        parts = [self.street_address, self.city, self.postal_code, self.country]
        return ", ".join(p for p in parts if p)

# Value with unit
class MonetaryAmount(BaseModel):
    value: float = Field(...)
    currency: Optional[str] = Field(None)
    
    def __str__(self) -> str:
        return f"{self.value} {self.currency or ''}".strip()

# With identifier
class Document(BaseModel):
    document_type: str = Field(...)
    document_number: str = Field(...)
    
    def __str__(self) -> str:
        return f"{self.document_type} {self.document_number}"
```

---

## Pattern 10: Template Composition

### The Challenge

Large templates can become unwieldy. Break them into modules:

```python
# common_components.py
"""Reusable components for all templates."""

class Address(BaseModel):
    """Physical address component."""
    model_config = ConfigDict(is_entity=False)
    # ... fields

class MonetaryAmount(BaseModel):
    """Monetary value component."""
    model_config = ConfigDict(is_entity=False)
    # ... fields

# common_entities.py
"""Reusable entities for all templates."""

from .common_components import Address

class Person(BaseModel):
    """Person entity."""
    model_config = ConfigDict(graph_id_fields=["first_name", "last_name"])
    # ... fields
    addresses: List[Address] = edge(label="LIVES_AT", default_factory=list)

class Organization(BaseModel):
    """Organization entity."""
    model_config = ConfigDict(graph_id_fields=["name"])
    # ... fields
    addresses: List[Address] = edge(label="LOCATED_AT", default_factory=list)

# invoice_template.py
"""Invoice-specific template."""

from .common_components import Address, MonetaryAmount
from .common_entities import Person, Organization

class LineItem(BaseModel):
    """Invoice line item."""
    # ... fields

class Invoice(BaseModel):
    """Invoice document."""
    model_config = ConfigDict(graph_id_fields=["invoice_number"])
    # ... fields
    issued_by: Organization = edge(label="ISSUED_BY")
    sent_to: Person = edge(label="SENT_TO")
```

---

## Next Steps

Now that you understand advanced patterns:

1. **[Best Practices →](best-practices.md)** - Complete template checklist
2. **[Examples](../../usage/examples/index.md)** - See patterns in production templates
3. **[Pipeline Configuration](../pipeline-configuration/index.md)** - Configure extraction

---

## Quick Reference

### Pattern Checklist

- [ ] Flexible measurements for scientific data
- [ ] Nested lists with edges for complex structures
- [ ] Multiple addresses with optional types
- [ ] Optional edges for conditional relationships
- [ ] Conditional fields with validators
- [ ] Polymorphic fields for flexible values
- [ ] Hierarchical structures for nested content
- [ ] Reusable component library
- [ ] String representations for all entities
- [ ] Template composition for large projects