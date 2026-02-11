# Schema Definition: Pydantic Templates


> **Note**: The examples in this document use simplified field names and structures for teaching purposes. 
> The actual `BillingDocument` schema at `docs/examples/templates/billing_document.py` is more comprehensive 
> with 30+ classes, EN 16931/Peppol BIS compliance, and uses `CONTAINS_LINE` for line items.


## Overview

Pydantic templates are the **foundation** of knowledge graph extraction in Docling Graph. They serve three critical purposes:

1. **LLM Guidance** - Field descriptions and examples guide the language model to extract accurate, structured data
2. **Data Validation** - Field validators ensure data quality and consistency
3. **Graph Structure** - Models define nodes, edges, and relationships for the knowledge graph

## What You'll Learn

This section provides a complete guide to creating Pydantic templates optimized for LLM-based document extraction and automatic conversion to knowledge graphs.

## Quick Example

Here's a minimal template showing the key concepts:

```python
"""BillingDocument extraction template."""

from typing import Any, List
from pydantic import BaseModel, ConfigDict, Field

# Required: Edge helper function
def edge(label: str, **kwargs: Any) -> Any:
    """Helper to create graph edges."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

# Component: Deduplicated by content
class Address(BaseModel):
    """Physical address (value object)."""
    model_config = ConfigDict(is_entity=False)
    
    street: str = Field(
        description="Street name and number",
        examples=["123 Main St", "45 Avenue des Champs-Élysées"]
    )
    city: str = Field(
        description="City name",
        examples=["Paris", "London"]
    )

# Entity: Unique by name
class Organization(BaseModel):
    """Organization entity."""
    model_config = ConfigDict(graph_id_fields=["name"])
    
    name: str = Field(
        description="Legal organization name",
        examples=["Acme Corp", "Tech Solutions Ltd"]
    )
    
    # Edge to Address component
    located_at: Address = edge(
        label="LOCATED_AT",
        description="Organization's physical address"
    )

# Root document
class BillingDocument(BaseModel):
    """BillingDocument document."""
    model_config = ConfigDict(graph_id_fields=["document_no"])
    
    document_no: str = Field(
        description="Unique invoice identifier",
        examples=["INV-2024-001", "12345"]
    )
    
    # Edge to Organization entity
    issued_by: Organization = edge(
        label="ISSUED_BY",
        description="Organization that issued this invoice"
    )
```

**Key Concepts Shown:**
<br>✅ `edge()` helper function for relationships
<br>✅ Component with `is_entity=False` (Address)
<br>✅ Entity with `graph_id_fields` (Organization, Invoice)
<br>✅ Clear field descriptions and examples
<br>✅ Graph relationships via `edge()` calls

---

## Why Pydantic for Knowledge Graphs?

### 1. Type Safety and Validation

Pydantic provides automatic type checking and validation:

```python
class MonetaryAmount(BaseModel):
    value: float = Field(...)
    currency: str = Field(...)
    
    @field_validator("value")
    @classmethod
    def validate_positive(cls, v: Any) -> Any:
        if v < 0:
            raise ValueError("Amount must be non-negative")
        return v
```

### 2. LLM-Friendly Schema

Field descriptions and examples guide the LLM:

```python
date_of_birth: date = Field(
    description=(
        "Person's date of birth. "
        "Look for 'Date of birth', 'Date de naiss.', or 'Born on'. "
        "Parse formats like 'DD MM YYYY' and normalize to YYYY-MM-DD."
    ),
    examples=["1990-05-15", "1985-12-20"]
)
```

### 3. Automatic Graph Conversion

The pipeline automatically converts Pydantic models to knowledge graphs:

```
BillingDocument (node)
  ├─ ISSUED_BY → Organization (node)
  │               └─ LOCATED_AT → Address (node)
  └─ SENT_TO → Client (node)
                └─ LIVES_AT → Address (node)
```

---

## Core Terminology

| Term | Definition | Example |
|:-----|:-----------|:--------|
| **Entity** | Unique, identifiable object tracked individually | Person, Organization, Document |
| **Component** | Value object deduplicated by content | Address, MonetaryAmount, Measurement |
| **Node** | Any Pydantic model that becomes a graph node | All BaseModel subclasses |
| **Edge** | Relationship between nodes | `ISSUED_BY`, `LOCATED_AT`, `CONTAINS_LINE` |
| **graph_id_fields** | Fields used to create stable, unique node IDs | `["name"]`, `["first_name", "last_name"]` |

---

## Template Examples by Domain

Docling Graph includes production-ready templates for various domains:

### 📍 Invoice Template
- **Entities:** Invoice, Organization, Client
- **Components:** Address, LineItem
- **Use Case:** Financial document processing

### 🆔 ID Card Template
- **Entities:** IDCard, Person
- **Components:** Address
- **Use Case:** Identity document extraction

### 🔬 Rheology Research Template
- **Entities:** Research, Experiment, Material
- **Components:** Measurement, VibrationParameter
- **Use Case:** Scientific literature mining

### 🏥 Insurance Template
- **Entities:** InsuranceTerms, InsurancePlan, Guarantee
- **Components:** MonetaryAmount, Address
- **Use Case:** Insurance document analysis

**Location:** `docs/examples/templates/`

---

## Prerequisites

Before creating templates, ensure you have:
<br>✅ **Python 3.10+** installed
<br>✅ **Docling Graph** installed (`uv sync`)
<br>✅ **Basic Pydantic knowledge** (recommended but not required)
<br>✅ **Understanding of your domain** (document types, entities, relationships)

---

## Learning Path

### Beginner Path (Start Here)

1. **[Template Basics](template-basics.md)** - Learn file structure and imports
2. **[Entities vs Components](entities-vs-components.md)** - Understand the critical distinction
3. **[Field Definitions](field-definitions.md)** - Master field descriptions and examples
4. **[Best Practices](best-practices.md)** - Follow the checklist

### Advanced Path

1. **[Relationships](relationships.md)** - Complex edge patterns
2. **[Validation](validation.md)** - Custom validators and normalization
3. **[Advanced Patterns](advanced-patterns.md)** - Reusable components and complex structures
4. **[Schema design for staged extraction](staged-extraction-schema.md)** - Identity fields, linkage, and catalog constraints when using staged extraction

---

## Common Questions

### Q: Do I need to know Pydantic?

**A:** Basic knowledge helps, but this guide covers everything you need. Pydantic is intuitive and well-documented.

### Q: Can I use existing Pydantic models?

**A:** Yes! Add `graph_id_fields` or `is_entity=False` to `model_config`, and use the `edge()` helper for relationships.

### Q: How do I choose between Entity and Component?

**A:** Ask: "Should this be tracked individually?" If yes → Entity. If it's a shared value → Component. See [Entities vs Components](entities-vs-components.md).

### Q: What if my domain is complex?

**A:** Start simple with core entities, then add complexity. See [Advanced Patterns](advanced-patterns.md) for nested structures.

### Q: How do I design a schema for staged extraction?

**A:** Use [Schema design for staged extraction](staged-extraction-schema.md): ensure root and entities have required, extractable `graph_id_fields`, add examples for ID fields, and keep nesting depth manageable so the ID pass and quality gate succeed.

---

## Next Steps

Ready to create your first template?

1. **[Template Basics →](template-basics.md)** - Learn the required structure
2. **[Examples](../../introduction/quickstart.md)** - See complete working examples
3. **[Pipeline Configuration](../pipeline-configuration/index.md)** - Configure extraction after creating templates

---

## Additional Resources

- **[Pydantic Documentation](https://docs.pydantic.dev/)** - Official Pydantic docs
- **[Example Templates](../../usage/examples/index.md)** - Production-ready templates
- **[API Reference](../../reference/config.md)** - PipelineConfig and model details