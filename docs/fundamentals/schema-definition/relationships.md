# Relationships: Edge Definitions


## Overview

Relationships (edges) connect nodes in your knowledge graph. The `edge()` helper function marks fields as graph relationships and defines their labels. Well-designed edges create meaningful, queryable graph structures.

**In this guide:**
- Using the `edge()` function
- Edge label conventions
- Single vs list relationships
- Common edge patterns
- Relationship best practices

---

## Using the edge() Function

### Basic Syntax

```python
field_name: TargetType = edge(
    label="EDGE_LABEL",
    description="Description of the relationship",
    # Additional Field parameters...
)
```

### Required vs Optional Edges

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

**Critical Rule:** For list edges, you **must** provide `default_factory=list`.

---

## Edge Label Conventions

### Naming Standards

✅ **DO:**
- Use ALL_CAPS with underscores
- Use verb phrases that describe the relationship
- Choose descriptive, domain-appropriate verbs
- Be consistent across your template

❌ **DON'T:**
- Use camelCase, lowercase, or mixed case
- Use vague labels like "LINK" or "RELATED"
- Mix naming styles

### Good vs Bad Labels

```python
# ✅ Good - Clear, descriptive, consistent
issued_by: Organization = edge(label="ISSUED_BY")
sent_to: Client = edge(label="SENT_TO")
contains_items: List[Item] = edge(label="CONTAINS_ITEM")
located_at: Address = edge(label="LOCATED_AT")

# ❌ Bad - Inconsistent, vague
issued_by: Organization = edge(label="issuedBy")  # Wrong case
sent_to: Client = edge(label="sent-to")  # Wrong separator
contains_items: List[Item] = edge(label="has")  # Too vague
located_at: Address = edge(label="LINK")  # Not descriptive
```

---

## Common Edge Labels by Category

### Authorship & Ownership

```python
# Document creation
issued_by: Organization = edge(label="ISSUED_BY")
created_by: Person = edge(label="CREATED_BY")
authored_by: Person = edge(label="AUTHORED_BY")
owned_by: Organization = edge(label="OWNED_BY")
published_by: Organization = edge(label="PUBLISHED_BY")

# Verification & approval
verified_by: Person = edge(label="VERIFIED_BY")
approved_by: Person = edge(label="APPROVED_BY")
signed_by: Person = edge(label="SIGNED_BY")
```

### Recipients & Targets

```python
# Document recipients
sent_to: Client = edge(label="SENT_TO")
addressed_to: Person = edge(label="ADDRESSED_TO")
delivered_to: Organization = edge(label="DELIVERED_TO")
billed_to: Client = edge(label="BILLED_TO")

# Beneficiaries
insured_by: Person = edge(label="INSURED_BY")
covered_by: InsurancePlan = edge(label="COVERED_BY")
```

### Location & Physical Presence

```python
# Physical locations
located_at: Address = edge(label="LOCATED_AT")
lives_at: Address = edge(label="LIVES_AT")
based_at: Address = edge(label="BASED_AT")
manufactured_at: Address = edge(label="MANUFACTURED_AT")

# Geographic relationships
operates_in: Region = edge(label="OPERATES_IN")
ships_to: Country = edge(label="SHIPS_TO")
```

### Composition & Containment

```python
# Document structure
contains_item: List[LineItem] = edge(label="CONTAINS_ITEM")
has_component: List[Component] = edge(label="HAS_COMPONENT")
includes_part: List[Part] = edge(label="INCLUDES_PART")
composed_of: List[Material] = edge(label="COMPOSED_OF")

# Hierarchical relationships
has_section: List[Section] = edge(label="HAS_SECTION")
has_chapter: List[Chapter] = edge(label="HAS_CHAPTER")
```

### Membership & Association

```python
# Organizational relationships
belongs_to: Organization = edge(label="BELONGS_TO")
part_of: Group = edge(label="PART_OF")
member_of: Organization = edge(label="MEMBER_OF")
employed_by: Organization = edge(label="EMPLOYED_BY")

# Associations
affiliated_with: Organization = edge(label="AFFILIATED_WITH")
partnered_with: Organization = edge(label="PARTNERED_WITH")
```

### Services & Offerings

```python
# Insurance & coverage
has_guarantee: List[Guarantee] = edge(label="HAS_GUARANTEE")
offers_plan: List[Plan] = edge(label="OFFERS_PLAN")
provides_coverage: List[Coverage] = edge(label="PROVIDES_COVERAGE")

# Products & services
offers_product: List[Product] = edge(label="OFFERS_PRODUCT")
provides_service: List[Service] = edge(label="PROVIDES_SERVICE")
```

### Research & Scientific

```python
# Experiments & studies
has_experiment: List[Experiment] = edge(label="HAS_EXPERIMENT")
uses_material: Material = edge(label="USES_MATERIAL")
has_measurement: List[Measurement] = edge(label="HAS_MEASUREMENT")
has_result: List[Result] = edge(label="HAS_RESULT")

# Processes & methods
has_process_step: List[Step] = edge(label="HAS_PROCESS_STEP")
uses_method: Method = edge(label="USES_METHOD")
has_evaluation: Evaluation = edge(label="HAS_EVALUATION")
```

---

## Single vs List Relationships

### Single Relationships (One-to-One)

Use when an entity has **exactly one** or **at most one** related entity:

```python
class Invoice(BaseModel):
    """Invoice document."""
    model_config = ConfigDict(graph_id_fields=["invoice_number"])
    
    invoice_number: str = Field(...)
    
    # One invoice is issued by one organization
    issued_by: Organization = edge(
        label="ISSUED_BY",
        description="The organization that issued this invoice"
    )
    
    # One invoice is sent to one client
    sent_to: Client = edge(
        label="SENT_TO",
        description="The client receiving this invoice"
    )
```

**Graph Structure:**
```
Invoice-001 --ISSUED_BY--> Acme Corp
Invoice-001 --SENT_TO--> Client A
```

### List Relationships (One-to-Many)

Use when an entity can have **multiple** related entities:

```python
class Invoice(BaseModel):
    """Invoice document."""
    model_config = ConfigDict(graph_id_fields=["invoice_number"])
    
    invoice_number: str = Field(...)
    
    # One invoice contains many line items
    contains_items: List[LineItem] = edge(
        label="CONTAINS_ITEM",
        default_factory=list,  # Required!
        description="Line items in this invoice"
    )
```

**Graph Structure:**
```
Invoice-001 --CONTAINS_ITEM--> LineItem-1
Invoice-001 --CONTAINS_ITEM--> LineItem-2
Invoice-001 --CONTAINS_ITEM--> LineItem-3
```

### Optional Single Relationships

Use `Optional[Type]` for relationships that may not exist:

```python
class Document(BaseModel):
    """Document that may or may not have a verifier."""
    model_config = ConfigDict(graph_id_fields=["document_id"])
    
    document_id: str = Field(...)
    
    # Optional: document may not be verified
    verified_by: Optional[Person] = edge(
        label="VERIFIED_BY",
        description="Person who verified this document, if verified"
    )
```

---

## Edge Patterns and Examples

### Pattern 1: Bidirectional Relationships

Create meaningful relationships in both directions:

```python
class Organization(BaseModel):
    """Organization entity."""
    model_config = ConfigDict(graph_id_fields=["name"])
    
    name: str = Field(...)
    
    # Organization has employees
    employees: List[Person] = edge(
        label="EMPLOYS",
        default_factory=list,
        description="People employed by this organization"
    )

class Person(BaseModel):
    """Person entity."""
    model_config = ConfigDict(graph_id_fields=["first_name", "last_name"])
    
    first_name: str = Field(...)
    last_name: str = Field(...)
    
    # Person works for organization
    employer: Optional[Organization] = edge(
        label="EMPLOYED_BY",
        description="Organization employing this person"
    )
```

**Graph Structure:**
```
Organization --EMPLOYS--> Person
Person --EMPLOYED_BY--> Organization
```

### Pattern 2: Shared Components

Multiple entities can reference the same component:

```python
class Address(BaseModel):
    """Address component (shared)."""
    model_config = ConfigDict(is_entity=False)
    
    street: str = Field(...)
    city: str = Field(...)

class Person(BaseModel):
    """Person entity."""
    model_config = ConfigDict(graph_id_fields=["first_name", "last_name"])
    
    first_name: str = Field(...)
    last_name: str = Field(...)
    
    addresses: List[Address] = edge(
        label="LIVES_AT",
        default_factory=list,
        description="Residential addresses"
    )

class Organization(BaseModel):
    """Organization entity."""
    model_config = ConfigDict(graph_id_fields=["name"])
    
    name: str = Field(...)
    
    addresses: List[Address] = edge(
        label="LOCATED_AT",
        default_factory=list,
        description="Business addresses"
    )
```

**Graph Structure:**
```
Person-1 --LIVES_AT--> Address(123 Main St, Paris)
Person-2 --LIVES_AT--> Address(123 Main St, Paris)  # Same address node
Organization-1 --LOCATED_AT--> Address(123 Main St, Paris)  # Same address node
```

### Pattern 3: Nested Relationships

Edges can point to entities that have their own edges:

```python
class Material(BaseModel):
    """Material entity."""
    model_config = ConfigDict(graph_id_fields=["name"])
    
    name: str = Field(...)
    
    properties: List[MaterialProperty] = edge(
        label="HAS_PROPERTY",
        default_factory=list,
        description="Material properties"
    )

class Component(BaseModel):
    """Component entity."""
    model_config = ConfigDict(graph_id_fields=["component_id"])
    
    component_id: str = Field(...)
    
    material: Material = edge(
        label="USES_MATERIAL",
        description="Material used in this component"
    )

class Assembly(BaseModel):
    """Assembly entity."""
    model_config = ConfigDict(graph_id_fields=["assembly_id"])
    
    assembly_id: str = Field(...)
    
    components: List[Component] = edge(
        label="HAS_COMPONENT",
        default_factory=list,
        description="Components in this assembly"
    )
```

**Graph Structure:**
```
Assembly --HAS_COMPONENT--> Component --USES_MATERIAL--> Material --HAS_PROPERTY--> Property
```

### Pattern 4: Multiple Edge Types to Same Entity

An entity can have multiple types of relationships to the same target type:

```python
class Document(BaseModel):
    """Document entity."""
    model_config = ConfigDict(graph_id_fields=["document_id"])
    
    document_id: str = Field(...)
    
    # Different relationship types to Person
    created_by: Person = edge(
        label="CREATED_BY",
        description="Person who created this document"
    )
    
    reviewed_by: Optional[Person] = edge(
        label="REVIEWED_BY",
        description="Person who reviewed this document"
    )
    
    approved_by: Optional[Person] = edge(
        label="APPROVED_BY",
        description="Person who approved this document"
    )
```

**Graph Structure:**
```
Document --CREATED_BY--> Person-A
Document --REVIEWED_BY--> Person-B
Document --APPROVED_BY--> Person-C
```

---

## Complete Example: Invoice Template

Here's a complete example showing various edge patterns:

```python
"""Invoice extraction template with comprehensive edge definitions."""

from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field

def edge(label: str, **kwargs: Any) -> Any:
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

# --- Components ---

class Address(BaseModel):
    """Physical address component."""
    model_config = ConfigDict(is_entity=False)
    
    street: str = Field(...)
    city: str = Field(...)
    postal_code: str = Field(...)

class MonetaryAmount(BaseModel):
    """Monetary value component."""
    model_config = ConfigDict(is_entity=False)
    
    value: float = Field(...)
    currency: str = Field(...)

# --- Entities ---

class Organization(BaseModel):
    """Organization entity."""
    model_config = ConfigDict(graph_id_fields=["name"])
    
    name: str = Field(...)
    tax_id: Optional[str] = Field(None)
    
    # Edge to Address component
    addresses: List[Address] = edge(
        label="LOCATED_AT",
        default_factory=list,
        description="Business addresses"
    )

class Client(BaseModel):
    """Client entity."""
    model_config = ConfigDict(graph_id_fields=["name"])
    
    name: str = Field(...)
    email: Optional[str] = Field(None)
    
    # Edge to Address component
    addresses: List[Address] = edge(
        label="LIVES_AT",
        default_factory=list,
        description="Client addresses"
    )

class LineItem(BaseModel):
    """Line item entity."""
    model_config = ConfigDict(graph_id_fields=["description", "unit_price"])
    
    description: str = Field(...)
    quantity: float = Field(...)
    unit_price: float = Field(...)
    
    # Edge to MonetaryAmount component
    total: MonetaryAmount = edge(
        label="HAS_TOTAL",
        description="Total amount for this line item"
    )

# --- Root Document ---

class Invoice(BaseModel):
    """Invoice document (root)."""
    model_config = ConfigDict(graph_id_fields=["invoice_number"])
    
    invoice_number: str = Field(...)
    date: str = Field(...)
    
    # Single edges to entities
    issued_by: Organization = edge(
        label="ISSUED_BY",
        description="Organization that issued this invoice"
    )
    
    sent_to: Client = edge(
        label="SENT_TO",
        description="Client receiving this invoice"
    )
    
    # List edge to entities
    contains_items: List[LineItem] = edge(
        label="CONTAINS_ITEM",
        default_factory=list,
        description="Line items in this invoice"
    )
    
    # Edge to component
    total_amount: MonetaryAmount = edge(
        label="HAS_TOTAL",
        description="Total invoice amount"
    )
```

**Resulting Graph:**
```
Invoice-001
  ├─ ISSUED_BY → Organization(Acme Corp)
  │               └─ LOCATED_AT → Address(123 Main St, Paris)
  ├─ SENT_TO → Client(John Doe)
  │             └─ LIVES_AT → Address(456 Oak Ave, London)
  ├─ CONTAINS_ITEM → LineItem-1
  │                   └─ HAS_TOTAL → MonetaryAmount(100, EUR)
  ├─ CONTAINS_ITEM → LineItem-2
  │                   └─ HAS_TOTAL → MonetaryAmount(200, EUR)
  └─ HAS_TOTAL → MonetaryAmount(300, EUR)
```

---

## Best Practices

### 1. Use Descriptive Labels

```python
# ✅ Good - Clear and specific
issued_by: Organization = edge(label="ISSUED_BY")
contains_items: List[Item] = edge(label="CONTAINS_ITEM")

# ❌ Bad - Vague
issued_by: Organization = edge(label="HAS")
contains_items: List[Item] = edge(label="RELATED_TO")
```

### 2. Be Consistent

```python
# ✅ Good - Consistent pattern
lives_at: Address = edge(label="LIVES_AT")
works_at: Address = edge(label="WORKS_AT")
located_at: Address = edge(label="LOCATED_AT")

# ❌ Bad - Inconsistent
lives_at: Address = edge(label="LIVES_AT")
works_at: Address = edge(label="WORKS_IN")
located_at: Address = edge(label="HAS_LOCATION")
```

### 3. Always Use default_factory for Lists

```python
# ✅ Good
items: List[Item] = edge(
    label="CONTAINS_ITEM",
    default_factory=list
)

# ❌ Bad - Missing default_factory
items: List[Item] = edge(label="CONTAINS_ITEM")
```

### 4. Provide Clear Descriptions

```python
# ✅ Good - Explains the relationship
issued_by: Organization = edge(
    label="ISSUED_BY",
    description="The organization that created and issued this document"
)

# ❌ Bad - No description
issued_by: Organization = edge(label="ISSUED_BY")
```

---

## Common Mistakes

### ❌ Mistake 1: Missing default_factory

```python
# Wrong
items: List[Item] = edge(label="CONTAINS_ITEM")

# Correct
items: List[Item] = edge(
    label="CONTAINS_ITEM",
    default_factory=list
)
```

### ❌ Mistake 2: Inconsistent Label Format

```python
# Wrong - Mixed formats
issued_by: Org = edge(label="issuedBy")
sent_to: Client = edge(label="SENT_TO")
has_items: List[Item] = edge(label="contains-item")

# Correct - Consistent ALL_CAPS_WITH_UNDERSCORES
issued_by: Org = edge(label="ISSUED_BY")
sent_to: Client = edge(label="SENT_TO")
has_items: List[Item] = edge(label="CONTAINS_ITEM")
```

### ❌ Mistake 3: Vague Labels

```python
# Wrong - Too vague
org: Organization = edge(label="HAS")
items: List[Item] = edge(label="RELATED")

# Correct - Descriptive
org: Organization = edge(label="ISSUED_BY")
items: List[Item] = edge(label="CONTAINS_ITEM")
```

---

## Next Steps

Now that you understand relationships:

1. **[Validation →](validation.md)** - Add validators for data quality
2. **[Advanced Patterns](advanced-patterns.md)** - Complex relationship patterns
3. **[Best Practices](best-practices.md)** - Complete template checklist

---

## Quick Reference

### Edge Definition Template

```python
# Single edge
field: TargetType = edge(
    label="EDGE_LABEL",
    description="Relationship description"
)

# Optional single edge
field: Optional[TargetType] = edge(
    label="EDGE_LABEL",
    description="Relationship description"
)

# List edge
field: List[TargetType] = edge(
    label="EDGE_LABEL",
    default_factory=list,  # Required!
    description="Relationship description"
)
```

### Common Label Patterns

- Authorship: `ISSUED_BY`, `CREATED_BY`, `AUTHORED_BY`
- Recipients: `SENT_TO`, `ADDRESSED_TO`, `DELIVERED_TO`
- Location: `LOCATED_AT`, `LIVES_AT`, `BASED_AT`
- Composition: `CONTAINS_ITEM`, `HAS_COMPONENT`, `INCLUDES_PART`
- Membership: `BELONGS_TO`, `PART_OF`, `MEMBER_OF`