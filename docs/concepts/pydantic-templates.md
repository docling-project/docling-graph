# Pydantic Templates

Pydantic templates are the foundation of Docling Graph, serving three critical purposes: guiding LLM extraction, validating data, and defining graph structure.

## Overview

A Pydantic template is a Python class that inherits from `BaseModel` and defines:

1. **Extraction Schema**: What data to extract from documents
2. **Validation Rules**: How to validate and normalize extracted data
3. **Graph Structure**: How entities and relationships map to nodes and edges

## Why Pydantic?

Pydantic provides:

- **Type Safety**: Strong typing with Python type hints
- **Validation**: Automatic data validation and coercion
- **Documentation**: Field descriptions guide LLM extraction
- **Serialization**: Easy conversion to/from JSON
- **IDE Support**: Autocomplete and type checking

## Basic Template Structure

### Minimal Example

```python
from pydantic import BaseModel, Field
from typing import Optional

class Person(BaseModel):
    """A person entity."""
    model_config = {
        'is_entity': True,
        'graph_id_fields': ['name']
    }
    
    name: str = Field(
        description="Person's full name",
        examples=["John Doe", "Jane Smith"]
    )
    
    age: Optional[int] = Field(
        None,
        description="Person's age in years",
        examples=[25, 30, 45]
    )
```

### Complete Example

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any, List, Optional
from datetime import date

# Edge helper function (required)
def edge(label: str, **kwargs: Any) -> Any:
    """Create a Field with edge metadata for graph relationships."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

class Address(BaseModel):
    """Physical address component."""
    model_config = ConfigDict(is_entity=False)
    
    street: Optional[str] = Field(
        None,
        description="Street address",
        examples=["123 Main St", "456 Oak Ave"]
    )
    city: Optional[str] = Field(
        None,
        description="City name",
        examples=["Boston", "New York"]
    )
    
    def __str__(self) -> str:
        parts = [self.street, self.city]
        return ", ".join(p for p in parts if p)

class Person(BaseModel):
    """Person entity with validation."""
    model_config = ConfigDict(
        graph_id_fields=["name", "date_of_birth"]
    )
    
    name: str = Field(
        description="Person's full name",
        examples=["John Doe", "Jane Smith"]
    )
    
    date_of_birth: Optional[date] = Field(
        None,
        description="Date of birth in YYYY-MM-DD format",
        examples=["1990-01-15", "1985-06-20"]
    )
    
    email: Optional[str] = Field(
        None,
        description="Email address",
        examples=["john@example.com", "jane@company.com"]
    )
    
    address: Optional[Address] = Field(
        None,
        description="Residential address"
    )
    
    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v: Any) -> Any:
        """Convert email to lowercase."""
        if v:
            return v.lower().strip()
        return v
    
    def __str__(self) -> str:
        return self.name
```

## Model Configuration

### ConfigDict Options

```python
from pydantic import ConfigDict

model_config = ConfigDict(
    # Graph-specific options
    is_entity=True,                    # Mark as entity (default: True if graph_id_fields present)
    graph_id_fields=["field1", "field2"],  # Fields for stable node IDs
    
    # Pydantic options
    validate_assignment=True,          # Validate on attribute assignment
    arbitrary_types_allowed=True,      # Allow custom types
    str_strip_whitespace=True,         # Strip whitespace from strings
    use_enum_values=True,              # Use enum values instead of enum objects
)
```

### Entity vs Component

#### Entity (Unique, Identifiable)

```python
class Person(BaseModel):
    """Entity: Unique person tracked individually."""
    model_config = ConfigDict(
        graph_id_fields=["name", "date_of_birth"]
    )
    
    name: str
    date_of_birth: date
```

**Result**: Each person gets a unique node ID based on name + DOB

#### Component (Value Object)

```python
class Address(BaseModel):
    """Component: Deduplicated by content."""
    model_config = ConfigDict(is_entity=False)
    
    street: str
    city: str
```

**Result**: Identical addresses share the same node

## Field Definitions

### Field Types

```python
from typing import List, Optional, Union
from datetime import date, datetime
from enum import Enum

class MyModel(BaseModel):
    # Required fields
    required_str: str = Field(...)
    required_int: int = Field(...)
    
    # Optional fields
    optional_str: Optional[str] = Field(None)
    optional_int: Optional[int] = Field(None)
    
    # Lists
    string_list: List[str] = Field(default_factory=list)
    object_list: List[Address] = Field(default_factory=list)
    
    # Dates
    birth_date: Optional[date] = Field(None)
    timestamp: Optional[datetime] = Field(None)
    
    # Enums
    status: StatusEnum = Field(...)
    
    # Union types
    value: Union[str, int, float] = Field(...)
```

### Field Parameters

```python
field_name: str = Field(
    default=...,  # ... = required, None = optional, or actual default value
    
    # LLM guidance
    description="Detailed description for LLM extraction",
    examples=["Example 1", "Example 2", "Example 3"],
    
    # Validation
    min_length=1,
    max_length=100,
    ge=0,  # Greater than or equal
    le=100,  # Less than or equal
    pattern=r"^\d{3}-\d{3}-\d{4}$",  # Regex pattern
    
    # Metadata
    alias="alternativeName",  # Alternative field name
    title="Field Title",
    deprecated=True,
)
```

### Description Best Practices

**Good descriptions**:
- Are specific and detailed
- Include extraction hints (field names, patterns)
- Provide parsing instructions
- Guide the LLM on ambiguous cases

```python
# EXCELLENT
date_of_birth: Optional[date] = Field(
    None,
    description=(
        "Person's date of birth. Look for 'Date of birth', 'DOB', "
        "'Born on', or 'Date de naissance'. Parse formats like "
        "'DD/MM/YYYY' or 'DD-MM-YYYY' and normalize to YYYY-MM-DD."
    ),
    examples=["1990-05-15", "1985-12-20"]
)

# POOR
date_of_birth: Optional[date] = Field(None, description="Birth date")
```

### Examples Best Practices

Provide 2-5 diverse, realistic examples:

```python
# For simple fields
email: Optional[str] = Field(
    None,
    description="Email address",
    examples=[
        "john.doe@email.com",
        "contact@company.fr",
        "info@organization.org"
    ]
)

# For lists
tags: List[str] = Field(
    default_factory=list,
    description="Document tags or categories",
    examples=[
        ["finance", "invoice", "2024"],
        ["research", "chemistry", "battery"],
        ["legal", "contract"]
    ]
)

# For nested objects
components: List[Component] = Field(
    default_factory=list,
    description="List of components",
    examples=[
        [
            {
                "name": "Battery Cell",
                "material": {"name": "Lithium", "grade": "99.9%"},
                "quantity": 100
            }
        ]
    ]
)
```

## Relationships and Edges

### Edge Helper Function

**Required in every template**:

```python
from typing import Any

def edge(label: str, **kwargs: Any) -> Any:
    """Create a Field with edge metadata for graph relationships."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)
```

### Single Relationships

```python
class Document(BaseModel):
    # Required single edge
    issued_by: Organization = edge(
        label="ISSUED_BY",
        description="Organization that issued this document"
    )
    
    # Optional single edge
    verified_by: Optional[Person] = edge(
        label="VERIFIED_BY",
        description="Person who verified this document"
    )
```

### List Relationships

```python
class Document(BaseModel):
    # One-to-many relationship
    authors: List[Person] = edge(
        label="HAS_AUTHOR",
        default_factory=list,  # REQUIRED for lists
        description="Document authors"
    )
    
    # Optional list relationship
    reviewers: List[Person] = edge(
        label="REVIEWED_BY",
        default_factory=list,
        description="People who reviewed this document"
    )
```

### Edge Label Conventions

Use descriptive, ALL_CAPS labels:

```python
# Authorship/Ownership
ISSUED_BY, CREATED_BY, OWNED_BY, AUTHORED_BY

# Recipients
SENT_TO, ADDRESSED_TO, DELIVERED_TO

# Location
LOCATED_AT, LIVES_AT, BASED_AT

# Composition
CONTAINS_ITEM, HAS_COMPONENT, INCLUDES_PART, HAS_SECTION

# Membership
BELONGS_TO, PART_OF, MEMBER_OF

# Processes
HAS_PROCESS_STEP, HAS_EVALUATION, HAS_MEASUREMENT

# Temporal
FOLLOWS, PRECEDES, OCCURS_DURING
```

## Validation

### Field Validators

```python
from pydantic import field_validator

class Person(BaseModel):
    email: Optional[str] = Field(None)
    age: Optional[int] = Field(None)
    
    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v: Any) -> Any:
        """Convert email to lowercase and strip whitespace."""
        if v:
            return v.lower().strip()
        return v
    
    @field_validator("age")
    @classmethod
    def validate_age(cls, v: Any) -> Any:
        """Ensure age is reasonable."""
        if v is not None and (v < 0 or v > 150):
            raise ValueError("Age must be between 0 and 150")
        return v
```

### Model Validators

```python
from pydantic import model_validator
from typing_extensions import Self

class Measurement(BaseModel):
    numeric_value: Optional[float] = Field(None)
    numeric_value_min: Optional[float] = Field(None)
    numeric_value_max: Optional[float] = Field(None)
    
    @model_validator(mode="after")
    def validate_value_consistency(self) -> Self:
        """Ensure value fields are used consistently."""
        has_single = self.numeric_value is not None
        has_range = (self.numeric_value_min is not None or 
                     self.numeric_value_max is not None)
        
        if has_single and has_range:
            raise ValueError(
                "Cannot specify both numeric_value and range values"
            )
        
        return self
```

### Pre-validators

Use `mode="before"` for transformations before type coercion:

```python
@field_validator("given_names", mode="before")
@classmethod
def ensure_list(cls, v: Any) -> Any:
    """Convert comma-separated string to list."""
    if isinstance(v, str):
        if "," in v:
            return [name.strip() for name in v.split(",")]
        return [v]
    return v
```

## String Representations

Add `__str__` methods for human-readable output:

```python
class Person(BaseModel):
    first_name: Optional[str] = Field(None)
    last_name: Optional[str] = Field(None)
    
    def __str__(self) -> str:
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p) or "Unknown"

class Address(BaseModel):
    street: Optional[str] = Field(None)
    city: Optional[str] = Field(None)
    postal_code: Optional[str] = Field(None)
    
    def __str__(self) -> str:
        parts = [self.street, self.city, self.postal_code]
        return ", ".join(p for p in parts if p)

class MonetaryAmount(BaseModel):
    value: float = Field(...)
    currency: Optional[str] = Field(None)
    
    def __str__(self) -> str:
        return f"{self.value} {self.currency or ''}".strip()
```

## Template Organization

### File Structure

```python
"""
Template for extracting invoice data.
Defines entities, components, and relationships for invoice documents.
"""

# --- Required Imports ---
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any, List, Optional
from datetime import date

# --- Edge Helper Function ---
def edge(label: str, **kwargs: Any) -> Any:
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

# --- Reusable Components ---
class Address(BaseModel):
    """Physical address component."""
    model_config = ConfigDict(is_entity=False)
    # ... fields ...

class MonetaryAmount(BaseModel):
    """Monetary value component."""
    model_config = ConfigDict(is_entity=False)
    # ... fields ...

# --- Reusable Entities ---
class Organization(BaseModel):
    """Organization entity."""
    model_config = ConfigDict(graph_id_fields=["name"])
    # ... fields ...

class Person(BaseModel):
    """Person entity."""
    model_config = ConfigDict(graph_id_fields=["name", "date_of_birth"])
    # ... fields ...

# --- Domain-Specific Models ---
class LineItem(BaseModel):
    """Invoice line item."""
    model_config = ConfigDict(graph_id_fields=["description", "amount"])
    # ... fields ...

# --- Root Document Model ---
class Invoice(BaseModel):
    """Complete invoice document."""
    model_config = ConfigDict(graph_id_fields=["invoice_number"])
    # ... fields ...
```

## Common Patterns

### Pattern 1: Document with Entities

```python
class Invoice(BaseModel):
    """Invoice document."""
    model_config = ConfigDict(graph_id_fields=["invoice_number"])
    
    invoice_number: str = Field(...)
    issue_date: date = Field(...)
    
    issuer: Organization = edge(label="ISSUED_BY")
    recipient: Person = edge(label="SENT_TO")
    items: List[LineItem] = edge(
        label="CONTAINS_ITEM",
        default_factory=list
    )
```

### Pattern 2: Hierarchical Structure

```python
class Section(BaseModel):
    """Document section."""
    model_config = ConfigDict(graph_id_fields=["title"])
    
    title: str = Field(...)
    content: str = Field(...)
    subsections: List["Section"] = edge(
        label="HAS_SUBSECTION",
        default_factory=list
    )
```

### Pattern 3: Flexible Measurement

```python
class Measurement(BaseModel):
    """Flexible measurement supporting single values or ranges."""
    model_config = ConfigDict(is_entity=False)
    
    name: str = Field(...)
    numeric_value: Optional[float] = Field(None)
    numeric_value_min: Optional[float] = Field(None)
    numeric_value_max: Optional[float] = Field(None)
    unit: Optional[str] = Field(None)
    
    @model_validator(mode="after")
    def validate_values(self) -> Self:
        has_single = self.numeric_value is not None
        has_range = (self.numeric_value_min is not None or 
                     self.numeric_value_max is not None)
        if has_single and has_range:
            raise ValueError("Cannot specify both single and range values")
        return self
```

## Testing Templates

### Basic Test

```python
# test_template.py
from my_template import Invoice, Organization, Person

def test_invoice_creation():
    invoice = Invoice(
        invoice_number="INV-001",
        issue_date="2024-01-15",
        issuer=Organization(name="Acme Corp"),
        recipient=Person(name="John Doe", date_of_birth="1990-01-15")
    )
    
    assert invoice.invoice_number == "INV-001"
    assert invoice.issuer.name == "Acme Corp"
    print(invoice.model_dump_json(indent=2))
```

### Validation Test

```python
def test_validation():
    # Should raise validation error
    try:
        person = Person(
            name="John Doe",
            email="INVALID EMAIL"  # Should fail validation
        )
    except ValueError as e:
        print(f"Validation error: {e}")
```

## Best Practices Checklist

When creating templates, ensure:

- [ ] All necessary imports included
- [ ] `edge()` helper function defined
- [ ] Entities have `graph_id_fields`
- [ ] Components have `is_entity=False`
- [ ] Field descriptions are detailed and LLM-friendly
- [ ] 2-5 realistic examples per field
- [ ] Validators for data quality
- [ ] Edge labels are descriptive and ALL_CAPS
- [ ] List edges use `default_factory=list`
- [ ] `__str__` methods for entities
- [ ] Docstrings for all models
- [ ] Proper type hints (Optional, List, Union)

## Complete Example Template

See the [complete guide](../guides/create_pydantic_templates_for_kg_extraction.md) for a comprehensive template creation tutorial with advanced patterns and best practices.

## Next Steps

- Review [example templates](../examples/templates/)
- Learn about [Graph Construction](graph-construction.md)
- Understand [Extraction Backends](extraction-backends.md)
- Explore [Processing Strategies](processing-strategies.md)