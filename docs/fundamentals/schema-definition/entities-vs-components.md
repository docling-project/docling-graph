# Entities vs Components


## Overview

The **most critical decision** when designing a Pydantic template is classifying each model as either an **Entity** or a **Component**. This distinction fundamentally affects how your knowledge graph is constructed and how nodes are deduplicated.

**In this guide:**
- Understanding the Entity vs Component distinction
- When to use each classification
- How to configure `graph_id_fields` and `is_entity`
- Real-world examples and decision trees

---

## The Critical Distinction

### Quick Comparison

| Aspect | Entity | Component |
|:-------|:-------|:----------|
| **Purpose** | Unique, identifiable objects | Value objects, content-based deduplication |
| **Configuration** | `graph_id_fields=[...]` | `is_entity=False` |
| **Deduplication** | By specified ID fields | By all field values |
| **When to Use** | Track individually (people, documents, organizations) | Shared values (addresses, amounts, measurements) |
| **Graph Behavior** | One node per unique ID combination | One node per unique content combination |

### Visual Example

```
# Entity: Person (unique by name + DOB)
Person(first_name="John", last_name="Doe", dob="1990-01-01")
Person(first_name="John", last_name="Doe", dob="1990-01-01")
→ Creates 1 node (same ID fields)

Person(first_name="John", last_name="Doe", dob="1991-01-01")
→ Creates 2nd node (different DOB)

# Component: Address (unique by content)
Address(street="123 Main St", city="Paris")
Address(street="123 Main St", city="Paris")
→ Creates 1 node (identical content)

Address(street="123 Main St", city="London")
→ Creates 2nd node (different city)
```

---

## Entities: Unique, Identifiable Objects

### What is an Entity?

An **Entity** is a model that represents a **unique, identifiable object** that should be tracked individually in your knowledge graph. Entities are deduplicated based on specific identifying fields.

### Configuration

```python
class Person(BaseModel):
    """
    A person entity.
    Uniquely identified by first name, last name, and date of birth.
    """
    model_config = ConfigDict(graph_id_fields=["first_name", "last_name", "date_of_birth"])
    
    first_name: Optional[str] = Field(...)
    last_name: Optional[str] = Field(...)
    date_of_birth: Optional[date] = Field(...)
    email: Optional[str] = Field(None)  # Not part of ID
    phone: Optional[str] = Field(None)  # Not part of ID
```

**Key Points:**
- Use `graph_id_fields` to specify which fields form the unique identifier
- Only fields in `graph_id_fields` are used for deduplication
- Other fields can vary without creating new nodes

### When to Use Entities

Use entities for models that represent:

✅ **People** - Individuals with unique identities
```python
model_config = ConfigDict(graph_id_fields=["first_name", "last_name", "date_of_birth"])
```

✅ **Organizations** - Companies, institutions
```python
model_config = ConfigDict(graph_id_fields=["name"])
```

✅ **Documents** - Invoices, contracts, reports
```python
model_config = ConfigDict(graph_id_fields=["document_number"])
```

✅ **Products** - Items with SKUs or unique identifiers
```python
model_config = ConfigDict(graph_id_fields=["sku"])
```

✅ **Experiments** - Research experiments with IDs
```python
model_config = ConfigDict(graph_id_fields=["experiment_id"])
```

### Choosing graph_id_fields

Select fields that:
1. **Together form a natural unique identifier**
2. **Are stable** (don't change frequently)
3. **Are likely to be present** in extracted data

**Staged extraction:** When using `extraction_contract="staged"`, only paths with non-empty `graph_id_fields` are sent to the ID pass by default. Entities without `graph_id_fields` are skipped in identity-only mode. The root model must have `graph_id_fields` so at least one root instance can be discovered (required by the quality gate). Prefer required, short, extractable ID fields and add schema examples for better ID-pass prompts. See [Schema design for staged extraction](staged-extraction-schema.md).

#### Examples

```python
# Single field ID
class Organization(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str = Field(...)

# Multi-field ID
class Person(BaseModel):
    model_config = ConfigDict(
        graph_id_fields=["first_name", "last_name", "date_of_birth"]
    )
    first_name: Optional[str] = Field(...)
    last_name: Optional[str] = Field(...)
    date_of_birth: Optional[date] = Field(...)

# Complex ID
class Measurement(BaseModel):
    model_config = ConfigDict(
        graph_id_fields=["name", "text_value", "numeric_value", "unit"]
    )
    name: str = Field(...)
    text_value: Optional[str] = Field(None)
    numeric_value: Optional[float] = Field(None)
    unit: Optional[str] = Field(None)
```

### Entity Examples

#### 📍 Person Entity

```python
class Person(BaseModel):
    """
    Person entity.
    Uniquely identified by name and date of birth.
    """
    model_config = ConfigDict(
        graph_id_fields=["first_name", "last_name", "date_of_birth"]
    )
    
    first_name: Optional[str] = Field(
        None,
        description="Person's given name",
        examples=["Jean", "Maria", "John"]
    )
    
    last_name: Optional[str] = Field(
        None,
        description="Person's family name",
        examples=["Dupont", "Garcia", "Smith"]
    )
    
    date_of_birth: Optional[date] = Field(
        None,
        description="Date of birth in YYYY-MM-DD format",
        examples=["1985-03-12", "1990-06-20"]
    )
    
    # These fields don't affect identity
    email: Optional[str] = Field(None)
    phone: Optional[str] = Field(None)
    
    def __str__(self) -> str:
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p) or "Unknown"
```

**Graph Behavior:**
```
Person(first_name="John", last_name="Doe", dob="1990-01-01", email="john@email.com")
Person(first_name="John", last_name="Doe", dob="1990-01-01", email="john@work.com")
→ Same node (same ID fields, email difference ignored)
```

#### 📍 Document Entity

```python
class BillingDocument(BaseModel):
    """
    BillingDocument document entity.
    Uniquely identified by invoice number.
    """
    model_config = ConfigDict(graph_id_fields=["document_no"])
    
    document_no: str = Field(
        ...,
        description="Unique invoice identifier",
        examples=["INV-2024-001", "12345"]
    )
    
    date: Optional[str] = Field(None)
    total: Optional[float] = Field(None)
    
    def __str__(self) -> str:
        return f"Invoice {self.document_no}"
```

---

## Components: Value Objects

### What is a Component?

A **Component** is a model that represents a **value object** - it's deduplicated by its entire content. If two components have identical field values, they share the same graph node.

### Configuration

```python
class Address(BaseModel):
    """
    Physical address component.
    Deduplicated by content - identical addresses share the same node.
    """
    model_config = ConfigDict(is_entity=False)
    
    street_address: Optional[str] = Field(...)
    city: Optional[str] = Field(...)
    postal_code: Optional[str] = Field(...)
    country: Optional[str] = Field(...)
```

**Key Points:**
- Use `is_entity=False` to mark as component
- **All fields** are used for deduplication
- Identical content = same node

### When to Use Components

Use components for models that represent:

✅ **Addresses** - Physical locations
```python
model_config = ConfigDict(is_entity=False)
```

✅ **Monetary Amounts** - Values with currency
```python
model_config = ConfigDict(is_entity=False)
```

✅ **Measurements** - Quantities with units
```python
model_config = ConfigDict(is_entity=False)
```

✅ **Dates/Times** - Temporal values
```python
model_config = ConfigDict(is_entity=False)
```

✅ **Coordinates** - Geographic points
```python
model_config = ConfigDict(is_entity=False)
```

### Component Examples

#### 📍 Address Component

```python
class Address(BaseModel):
    """
    Physical address component.
    Deduplicated by content - identical addresses share the same node.
    """
    model_config = ConfigDict(is_entity=False)
    
    street_address: Optional[str] = Field(
        None,
        description="Street name and number",
        examples=["123 Main Street", "45 Avenue des Champs-Élysées"]
    )
    
    city: Optional[str] = Field(
        None,
        description="City name",
        examples=["Paris", "London", "New York"]
    )
    
    postal_code: Optional[str] = Field(
        None,
        description="Postal or ZIP code",
        examples=["75001", "SW1A 1AA", "10001"]
    )
    
    country: Optional[str] = Field(
        None,
        description="Country name or code",
        examples=["France", "FR", "United Kingdom"]
    )
    
    def __str__(self) -> str:
        parts = [self.street_address, self.city, self.postal_code, self.country]
        return ", ".join(p for p in parts if p)
```

**Graph Behavior:**
```
Address(street="123 Main St", city="Paris", postal_code="75001")
Address(street="123 Main St", city="Paris", postal_code="75001")
→ Same node (identical content)

Address(street="123 Main St", city="Paris", postal_code="75002")
→ Different node (postal code differs)
```

#### 📍 Monetary Amount Component

```python
class MonetaryAmount(BaseModel):
    """
    Monetary value with currency.
    Deduplicated by content - same value and currency share a node.
    """
    model_config = ConfigDict(is_entity=False)
    
    value: float = Field(
        ...,
        description="Numeric amount",
        examples=[500.00, 1250.50, 89.99]
    )
    
    currency: Optional[str] = Field(
        None,
        description="ISO 4217 currency code",
        examples=["EUR", "USD", "GBP"]
    )
    
    @field_validator("value")
    @classmethod
    def validate_positive(cls, v: Any) -> Any:
        if v < 0:
            raise ValueError("Amount must be non-negative")
        return v
    
    def __str__(self) -> str:
        return f"{self.value} {self.currency or ''}".strip()
```

**Graph Behavior:**
```
MonetaryAmount(value=100.00, currency="EUR")
MonetaryAmount(value=100.00, currency="EUR")
→ Same node (identical value and currency)

MonetaryAmount(value=100.00, currency="USD")
→ Different node (different currency)
```

---

## Decision Tree

Use this decision tree to classify your models:

--8<-- "docs/assets/flowcharts/model_decision_tree.md"

### Questions to Ask

1. **"Should this be tracked individually?"**
   - Yes → Likely an Entity
   - No → Likely a Component

2. **"If I see this twice with identical values, should it be one thing or two?"**
   - One thing → Component
   - Two things → Entity

3. **"Does this represent a unique object or a shared value?"**
   - Unique object → Entity
   - Shared value → Component

4. **"Would I want to query for all instances of this specific thing?"**
   - Yes → Entity
   - No → Component

---

## Real-World Examples

### 📍 Invoice Processing

```python
# ENTITY: BillingDocument (unique document)
class BillingDocument(BaseModel):
    model_config = ConfigDict(graph_id_fields=["document_no"])
    document_no: str = Field(...)
    # Each invoice is unique

# ENTITY: Organization (unique company)
class Organization(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str = Field(...)
    # Each organization is unique

# COMPONENT: Address (shared value)
class Address(BaseModel):
    model_config = ConfigDict(is_entity=False)
    street: str = Field(...)
    city: str = Field(...)
    # Multiple organizations can share the same address

# COMPONENT: MonetaryAmount (shared value)
class MonetaryAmount(BaseModel):
    model_config = ConfigDict(is_entity=False)
    value: float = Field(...)
    currency: str = Field(...)
    # Multiple invoices can have the same amount
```

**Graph Structure:**
```
BillingDocument-001 --ISSUED_BY--> Acme Corp --LOCATED_AT--> Address(123 Main St, Paris)
BillingDocument-002 --ISSUED_BY--> Tech Ltd --LOCATED_AT--> Address(123 Main St, Paris)
                                                      ↑ Same address node shared
```

### 📍 Rheology Research

```python
# ENTITY: Research (unique paper)
class Research(BaseModel):
    model_config = ConfigDict(graph_id_fields=["title"])
    title: str = Field(...)

# ENTITY: Experiment (unique experiment)
class Experiment(BaseModel):
    model_config = ConfigDict(graph_id_fields=["experiment_id"])
    experiment_id: str = Field(...)

# ENTITY: Material (unique material type)
class Material(BaseModel):
    model_config = ConfigDict(graph_id_fields=["material_type"])
    material_type: str = Field(...)

# COMPONENT: Measurement (shared value)
class Measurement(BaseModel):
    model_config = ConfigDict(is_entity=False)
    name: str = Field(...)
    value: float = Field(...)
    unit: str = Field(...)
    # Multiple experiments can have the same measurement
```

### 📍 ID Card

```python
# ENTITY: IDCard (unique document)
class IDCard(BaseModel):
    model_config = ConfigDict(graph_id_fields=["document_number"])
    document_number: str = Field(...)

# ENTITY: Person (unique individual)
class Person(BaseModel):
    model_config = ConfigDict(
        graph_id_fields=["given_names", "last_name", "date_of_birth"]
    )
    given_names: List[str] = Field(...)
    last_name: str = Field(...)
    date_of_birth: date = Field(...)

# COMPONENT: Address (shared value)
class Address(BaseModel):
    model_config = ConfigDict(is_entity=False)
    street_address: str = Field(...)
    city: str = Field(...)
    # Multiple people can live at the same address
```

---

## Common Patterns

### Pattern 1: Shared Addresses

**Scenario:** Multiple people or organizations at the same address.

**Solution:** Make Address a component.

```python
class Address(BaseModel):
    """Component - shared by multiple entities."""
    model_config = ConfigDict(is_entity=False)
    # ...

class Person(BaseModel):
    """Entity - unique individual."""
    model_config = ConfigDict(graph_id_fields=["first_name", "last_name"])
    # ...
    addresses: List[Address] = edge(label="LIVES_AT", default_factory=list)

class Organization(BaseModel):
    """Entity - unique company."""
    model_config = ConfigDict(graph_id_fields=["name"])
    # ...
    addresses: List[Address] = edge(label="LOCATED_AT", default_factory=list)
```

**Result:** Same address node is shared across multiple people/organizations.

---

## Staged extraction considerations

When using **staged extraction** (`extraction_contract="staged"`):

- **Entities** that should participate in the ID pass must have `graph_id_fields`. Components (`is_entity=False`) do not appear as separate identity paths when `staged_id_identity_only=True`.
- **Components** are included in the staged catalog only when the relationship uses `edge()` with an `edge_label`. Components without an edge label are not separate catalog nodes.
- **Parent linkage** in the fill/merge phase uses `(path, id_tuple)` and parent `(parent_path, parent_id_tuple)`. Ensure parent entities have stable, extractable `graph_id_fields` so references resolve and the quality gate (e.g. `staged_quality_max_parent_lookup_miss`) can pass.

For full guidance (identity choices, depth, troubleshooting), see [Schema design for staged extraction](staged-extraction-schema.md).

### Pattern 2: Measurements in Research

**Scenario:** Multiple experiments report the same measurement value.

**Solution:** Make Measurement a component.

```python
class Measurement(BaseModel):
    """Component - shared measurement value."""
    model_config = ConfigDict(is_entity=False)
    name: str = Field(...)
    value: float = Field(...)
    unit: str = Field(...)

class Experiment(BaseModel):
    """Entity - unique experiment."""
    model_config = ConfigDict(graph_id_fields=["experiment_id"])
    # ...
    measurements: List[Measurement] = Field(default_factory=list)
```

### Pattern 3: Line Items

**Scenario:** Invoice line items - should each be unique or shared?

**Decision:** Usually **neither** - line items are typically embedded data, not separate nodes.

```python
class LineItem(BaseModel):
    """Line item - embedded in invoice, not a separate node."""
    # No model_config needed - this won't become a node
    description: str = Field(...)
    quantity: float = Field(...)
    unit_price: float = Field(...)

class BillingDocument(BaseModel):
    """Entity - unique invoice."""
    model_config = ConfigDict(graph_id_fields=["document_no"])
    # ...
    # Use regular Field, not edge() - these are embedded
    items: List[LineItem] = Field(default_factory=list)
```

!!! note "Line items as nodes"
    If you want line items as nodes, use edge() and decide if they're entities or components.

---

## Common Mistakes

### ❌ Making Everything an Entity

```python
# WRONG - Address as entity
class Address(BaseModel):
    model_config = ConfigDict(graph_id_fields=["street", "city"])
    # This creates separate nodes for identical addresses
```

**Problem:** Identical addresses create separate nodes, losing the benefit of shared locations.

**Fix:** Make Address a component.

### ❌ Making Everything a Component

```python
# WRONG - Person as component
class Person(BaseModel):
    model_config = ConfigDict(is_entity=False)
    # This merges people with identical names
```

**Problem:** Two people with the same name become one node.

**Fix:** Make Person an entity with appropriate `graph_id_fields`.

### ❌ Wrong ID Fields

```python
# WRONG - Using non-stable fields
class Person(BaseModel):
    model_config = ConfigDict(graph_id_fields=["email"])
    # Email can change, creating duplicate nodes
```

**Problem:** When email changes, a new node is created for the same person.

**Fix:** Use stable fields like name + date of birth.

---

## Testing Your Classification

### Test 1: Deduplication Behavior

```python
# Test entity deduplication
person1 = Person(first_name="John", last_name="Doe", email="john@email.com")
person2 = Person(first_name="John", last_name="Doe", email="john@work.com")
# Should create 1 node (same ID fields)

# Test component deduplication
addr1 = Address(street="123 Main St", city="Paris")
addr2 = Address(street="123 Main St", city="Paris")
# Should create 1 node (identical content)

addr3 = Address(street="123 Main St", city="London")
# Should create 2nd node (different city)
```

### Test 2: Graph Structure

Run extraction and check the graph:

```bash
uv run docling-graph convert document.pdf \
    --template "my_template.MyTemplate" \
    --export-format csv \
    --output-dir test_output
```

Check `test_output/nodes.csv`:
- Entities should have one row per unique ID combination
- Components should have one row per unique content combination

---

## Next Steps

Now that you understand entities vs components:

1. **[Field Definitions →](field-definitions.md)** - Learn to write effective field descriptions
2. **[Relationships](relationships.md)** - Connect entities and components with edges
3. **[Advanced Patterns](advanced-patterns.md)** - Complex entity/component patterns