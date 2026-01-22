# Template Basics

**Navigation:** [← Schema Definition](index.md) | [Next: Entities vs Components →](entities-vs-components.md)

---

## Overview

Every Pydantic template for Docling Graph follows a **standard structure** with required imports, helper functions, and organization patterns. This ensures consistency and compatibility with the extraction pipeline.

**In this guide:**
- Required imports and their purposes
- The mandatory `edge()` helper function
- Standard file organization
- Docstring conventions

---

## Required Imports

### Standard Import Block

Every template **must** include this import structure:

```python
"""
Brief description of what this template extracts.
Mention the document type and key domain features.
"""

from typing import Any, List, Optional, Union, Self, Type
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from datetime import date, datetime  # Include based on domain needs
from enum import Enum  # Include if using enums
import re  # Include if using regex validators
```

### Import Breakdown

| Import | Purpose | When to Use |
|:-------|:--------|:------------|
| `Any, List, Optional, Union` | Type hints for fields | Always |
| `Self, Type` | Advanced type hints for validators | When using validators |
| `BaseModel` | Base class for all models | Always |
| `ConfigDict` | Model configuration (graph_id_fields, is_entity) | Always |
| `Field` | Field definitions with metadata | Always |
| `field_validator` | Single-field validation | When validating individual fields |
| `model_validator` | Cross-field validation | When validating multiple fields together |
| `date, datetime` | Date/time types | For temporal data |
| `Enum` | Enumerated types | For controlled vocabularies |
| `re` | Regular expressions | For pattern matching in validators |

### Example: Minimal Template

```python
"""
Invoice extraction template.
Extracts invoice data including issuer, client, and line items.
"""

from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field

# This is the minimal import set for a basic template
```

### Example: Full-Featured Template

```python
"""
Research paper extraction template.
Extracts scientific experiments, measurements, and materials.
"""

from typing import Any, List, Optional, Union, Self, Type
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from datetime import date
from enum import Enum
import re

# This includes all common imports for complex templates
```

---

## The Edge Helper Function

### Required Definition

This function **must be defined identically** in every template:

```python
def edge(label: str, **kwargs: Any) -> Any:
    """
    Helper function to create a Pydantic Field with edge metadata.
    The 'edge_label' defines the type of relationship in the knowledge graph.
    """
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)
```

### Critical Rules

✅ **DO:**
- Use lowercase `edge` (not `Edge` or `EDGE`)
- Return `Field(...)` with `json_schema_extra={"edge_label": label}`
- Accept `**kwargs` to pass through additional Field parameters
- Include the docstring

❌ **DON'T:**
- Change the function signature
- Modify the `json_schema_extra` structure
- Remove `**kwargs` support

### Why This Function?

The `edge()` helper serves two purposes:

1. **Marks relationships** - Tells the graph converter this field is an edge
2. **Provides metadata** - The `edge_label` becomes the relationship type in the graph

### Usage Examples

```python
# Required single relationship
issued_by: Organization = edge(
    label="ISSUED_BY",
    description="The organization that issued this document"
)

# Optional single relationship
verified_by: Optional[Person] = edge(
    label="VERIFIED_BY",
    description="Person who verified this document, if applicable"
)

# Required list relationship (one-to-many)
contains_items: List[LineItem] = edge(
    label="CONTAINS_ITEM",
    default_factory=list,  # REQUIRED for lists
    description="Line items contained in this document"
)

# Optional list relationship
addresses: List[Address] = edge(
    label="LOCATED_AT",
    default_factory=list,
    description="Physical addresses for this entity"
)
```

**Important:** For list edges, you **must** provide `default_factory=list` in the `edge()` call.

---

## Standard File Organization

### Recommended Structure

Organize your template in this exact order:

```python
"""
Template docstring describing purpose and domain
"""

# --- 1. Required Imports ---
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any, List, Optional
# ... additional imports

# --- 2. Edge Helper Function ---
def edge(label: str, **kwargs: Any) -> Any:
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

# --- 3. Helper Functions (if needed) ---
# Normalization, parsing, or utility functions

# --- 4. Reusable Components ---
# Value objects with is_entity=False

# --- 5. Reusable Entities ---
# Common entities like Person, Organization, Address

# --- 6. Domain-Specific Models ---
# Models unique to this document type

# --- 7. Root Document Model ---
# The main entry point (last in file)
```

### Example: Invoice Template Structure

```python
"""
Invoice extraction template.
Extracts structured data from invoice documents.
"""

# --- 1. Imports ---
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field

# --- 2. Edge Helper ---
def edge(label: str, **kwargs: Any) -> Any:
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

# --- 3. Helper Functions ---
# (none needed for this simple template)

# --- 4. Components ---
class Address(BaseModel):
    """Physical address component."""
    model_config = ConfigDict(is_entity=False)
    # ... fields

class MonetaryAmount(BaseModel):
    """Monetary value component."""
    model_config = ConfigDict(is_entity=False)
    # ... fields

# --- 5. Reusable Entities ---
class Organization(BaseModel):
    """Organization entity."""
    model_config = ConfigDict(graph_id_fields=["name"])
    # ... fields

class Person(BaseModel):
    """Person entity."""
    model_config = ConfigDict(graph_id_fields=["first_name", "last_name"])
    # ... fields

# --- 6. Domain-Specific Models ---
class LineItem(BaseModel):
    """Invoice line item."""
    # ... fields

# --- 7. Root Document ---
class Invoice(BaseModel):
    """Invoice document (root model)."""
    model_config = ConfigDict(graph_id_fields=["invoice_number"])
    # ... fields
```

---

## Docstring Standards

### Module Docstring

Every template file should start with a clear module docstring:

```python
"""
Pydantic templates for [Document Type] extraction.

These models extract [key information] from [document type] documents.
The schema is designed for automatic conversion to knowledge graphs.

Key entities:
- [Entity1]: [Description]
- [Entity2]: [Description]

Key relationships:
- [Entity1] --[RELATIONSHIP]--> [Entity2]
"""
```

### Model Docstrings

Each model should have a clear docstring:

```python
class Person(BaseModel):
    """
    A person entity.
    
    Uniquely identified by first name, last name, and date of birth.
    Represents individuals mentioned in documents.
    """
```

### Good vs Bad Docstrings

✅ **Good:**
```python
class Address(BaseModel):
    """
    Physical address component.
    
    Deduplicated by content - identical addresses share the same node.
    Used for both residential and business addresses.
    """
```

❌ **Bad:**
```python
class Address(BaseModel):
    """Address."""  # Too vague
```

---

## Complete Minimal Template

Here's a complete, minimal template showing all required elements:

```python
"""
Simple document extraction template.
Extracts basic document information.
"""

from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field

# --- Edge Helper Function (REQUIRED) ---
def edge(label: str, **kwargs: Any) -> Any:
    """Helper to create graph edges."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

# --- Component ---
class Address(BaseModel):
    """Physical address (value object)."""
    model_config = ConfigDict(is_entity=False)
    
    street: str = Field(
        description="Street name and number",
        examples=["123 Main St", "45 Rue de la Paix"]
    )
    city: str = Field(
        description="City name",
        examples=["Paris", "London"]
    )

# --- Entity ---
class Organization(BaseModel):
    """Organization entity."""
    model_config = ConfigDict(graph_id_fields=["name"])
    
    name: str = Field(
        description="Legal organization name",
        examples=["Acme Corp", "Tech Solutions Ltd"]
    )
    
    # Edge to Address
    located_at: Address = edge(
        label="LOCATED_AT",
        description="Organization's physical address"
    )

# --- Root Document ---
class Document(BaseModel):
    """Document entity (root model)."""
    model_config = ConfigDict(graph_id_fields=["document_id"])
    
    document_id: str = Field(
        description="Unique document identifier",
        examples=["DOC-2024-001", "12345"]
    )
    
    # Edge to Organization
    issued_by: Organization = edge(
        label="ISSUED_BY",
        description="Organization that issued this document"
    )
```

**This template includes:**
- ✅ Module docstring
- ✅ Required imports
- ✅ `edge()` helper function
- ✅ Component with `is_entity=False`
- ✅ Entity with `graph_id_fields`
- ✅ Root document model
- ✅ Clear field descriptions and examples
- ✅ Graph relationships via `edge()`

---

## Testing Your Template Structure

### Quick Validation

Test that your template is properly structured:

```python
# test_template_structure.py
from my_template import Document, Organization, Address

# 1. Check imports work
print("✓ Imports successful")

# 2. Check edge function exists
from my_template import edge
print("✓ edge() function defined")

# 3. Create test instance
doc = Document(
    document_id="TEST-001",
    issued_by=Organization(
        name="Test Corp",
        located_at=Address(
            street="123 Test St",
            city="Paris"
        )
    )
)
print("✓ Model instantiation works")

# 4. Check serialization
json_data = doc.model_dump_json(indent=2)
print("✓ JSON serialization works")
print(json_data)
```

### Run with uv

```bash
# Save test to file
uv run python test_template_structure.py
```

---

## Common Mistakes

### ❌ Mistake 1: Wrong edge() Definition

```python
# WRONG - Missing **kwargs
def edge(label: str) -> Any:
    return Field(..., json_schema_extra={"edge_label": label})

# WRONG - Wrong metadata key
def edge(label: str, **kwargs: Any) -> Any:
    return Field(..., json_schema_extra={"label": label}, **kwargs)

# CORRECT
def edge(label: str, **kwargs: Any) -> Any:
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)
```

### ❌ Mistake 2: Missing default_factory for Lists

```python
# WRONG - List edge without default_factory
items: List[Item] = edge(
    label="CONTAINS_ITEM",
    description="Items in document"
)

# CORRECT
items: List[Item] = edge(
    label="CONTAINS_ITEM",
    default_factory=list,  # Required!
    description="Items in document"
)
```

### ❌ Mistake 3: Inconsistent Organization

```python
# WRONG - Root model at the top
class Document(BaseModel):
    """Root document."""
    # ...

class Address(BaseModel):
    """Component used by Document."""
    # ...

# CORRECT - Components before entities, root at end
class Address(BaseModel):
    """Component."""
    # ...

class Document(BaseModel):
    """Root document."""
    # ...
```

---

## Next Steps

Now that you understand template basics:

1. **[Entities vs Components →](entities-vs-components.md)** - Learn the critical distinction
2. **[Field Definitions](field-definitions.md)** - Master field descriptions and examples
3. **[Example Templates](../09-examples/index.md)** - See complete working examples

---

## Quick Reference

### Required Elements Checklist

- [ ] Module docstring
- [ ] Standard imports (`BaseModel`, `ConfigDict`, `Field`, etc.)
- [ ] `edge()` helper function (exact definition)
- [ ] Organized structure (components → entities → root)
- [ ] Model docstrings
- [ ] Field descriptions and examples
- [ ] Proper `model_config` for all models

### File Template

```python
"""[Template description]"""

from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field

def edge(label: str, **kwargs: Any) -> Any:
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

# Components (is_entity=False)
# Entities (graph_id_fields=[...])
# Root document
```

---

**Navigation:** [← Schema Definition](index.md) | [Next: Entities vs Components →](entities-vs-components.md)