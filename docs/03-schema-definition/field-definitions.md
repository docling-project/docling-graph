# Field Definitions

**Navigation:** [← Entities vs Components](entities-vs-components.md) | [Next: Relationships →](relationships.md)

---

## Overview

Field definitions are where you **guide the LLM** to extract accurate data. Well-written field descriptions and examples are crucial for extraction quality. This guide covers best practices for defining fields that maximize LLM accuracy.

**In this guide:**
- Field anatomy and structure
- Required vs optional fields
- Writing effective descriptions
- Providing helpful examples
- Type hints and defaults

---

## Field Anatomy

### Basic Structure

```python
field_name: FieldType = Field(
    default_value,  # ... for required, None for optional, or a default
    description="Detailed, LLM-friendly description with extraction hints",
    examples=["Example 1", "Example 2", "Example 3"]  # 2-5 realistic examples
)
```

### Components Explained

| Component | Purpose | Required |
|:----------|:--------|:---------|
| `field_name` | Python variable name | Yes |
| `FieldType` | Type hint (str, int, date, etc.) | Yes |
| `Field(...)` | Pydantic field definition | Yes |
| `default_value` | Default or required marker (`...`) | Yes |
| `description` | LLM guidance text | Highly recommended |
| `examples` | Sample values | Highly recommended |

---

## Required vs Optional Fields

### Required Fields

Use `...` (ellipsis) to mark a field as required:

```python
document_id: str = Field(
    ...,  # Ellipsis = required
    description="Unique document identifier",
    examples=["DOC-2024-001", "INV-123456"]
)
```

**When to use:**
- Fields that must always be present
- Core identifying fields
- Fields critical for graph structure

### Optional Fields with None Default

Use `Optional[Type]` with `None` default:

```python
phone: Optional[str] = Field(
    None,  # None = optional
    description="Contact phone number",
    examples=["+33 1 23 45 67 89", "06 12 34 56 78"]
)
```

**When to use:**
- Fields that may not be present in all documents
- Supplementary information
- Fields that vary by document type

### Optional Fields with Custom Default

Provide a meaningful default value:

```python
status: str = Field(
    "pending",  # Custom default
    description="Current processing status",
    examples=["pending", "approved", "rejected"]
)

priority: int = Field(
    0,  # Numeric default
    description="Priority level (0-10)",
    examples=[0, 5, 10]
)
```

**When to use:**
- Fields with sensible fallback values
- Status or state fields
- Counters or flags

### Optional Lists

**Always use `default_factory=list`** for optional lists:

```python
items: List[Item] = Field(
    default_factory=list,  # Required for lists!
    description="List of items",
    examples=[[{"name": "Item1"}, {"name": "Item2"}]]
)
```

**Why:** Using `[]` as default creates a shared mutable object (Python gotcha).

---

## Writing Effective Descriptions

### The Golden Rules

1. **Be specific** - Tell the LLM exactly what to look for
2. **Include extraction hints** - Mention field names, patterns, synonyms
3. **Provide parsing instructions** - Explain how to normalize data
4. **Guide ambiguous cases** - Tell the LLM how to handle edge cases

### Excellent Description Pattern

```python
date_of_birth: Optional[date] = Field(
    None,
    description=(
        "The person's date of birth. "
        "Look for text like 'Date of birth', 'Date de naiss.', or 'Born on'. "
        "Parse formats like 'DD MM YYYY' or 'DDMMYYYY' and normalize to YYYY-MM-DD."
    ),
    examples=["1990-05-15", "1985-12-20", "1978-03-30"]
)
```

**What makes this excellent:**
- ✅ Clear purpose ("person's date of birth")
- ✅ Extraction hints (field name variations)
- ✅ Parsing instructions (format normalization)
- ✅ Multiple realistic examples

### Poor Description Examples

❌ **Too vague:**
```python
date_of_birth: Optional[date] = Field(None, description="Birth date")
```

❌ **No extraction hints:**
```python
email: Optional[str] = Field(None, description="Email address")
```

❌ **Missing parsing guidance:**
```python
amount: float = Field(..., description="The amount")
```

### Description Templates by Field Type

#### Text Fields

```python
name: str = Field(
    ...,
    description=(
        "Full legal name of the organization. "
        "Look for 'Company Name', 'Organization', or header text. "
        "Include legal suffixes like 'Ltd', 'Inc', 'SA'."
    ),
    examples=["Acme Corporation Ltd", "Tech Solutions Inc", "Global Industries SA"]
)
```

#### Numeric Fields

```python
amount: float = Field(
    ...,
    description=(
        "Total monetary amount. "
        "Extract numeric value only, removing currency symbols and commas. "
        "Convert formats like '1,234.56' to 1234.56."
    ),
    examples=[1234.56, 500.00, 89.99]
)
```

#### Date Fields

```python
issue_date: Optional[date] = Field(
    None,
    description=(
        "Date the document was issued. "
        "Look for 'Issue Date', 'Date d'émission', 'Issued on'. "
        "Parse various formats (DD/MM/YYYY, MM-DD-YYYY, YYYY-MM-DD) "
        "and normalize to YYYY-MM-DD."
    ),
    examples=["2024-01-15", "2023-12-20", "2024-03-01"]
)
```

#### Email Fields

```python
email: Optional[str] = Field(
    None,
    description=(
        "Contact email address. "
        "Look for text containing '@' symbol. "
        "Common labels: 'Email', 'E-mail', 'Contact', 'Courriel'. "
        "Normalize to lowercase."
    ),
    examples=["contact@company.com", "info@organization.fr", "support@business.co.uk"]
)
```

#### Phone Fields

```python
phone: Optional[str] = Field(
    None,
    description=(
        "Contact phone number. "
        "Look for 'Phone', 'Tel', 'Telephone', 'Mobile'. "
        "Preserve formatting (spaces, dashes, parentheses). "
        "Include country code if present."
    ),
    examples=["+33 1 23 45 67 89", "06 12 34 56 78", "+1 (555) 123-4567"]
)
```

#### List Fields

```python
guarantees: List[str] = Field(
    default_factory=list,
    description=(
        "List of coverage items or guarantees. "
        "Look for bullet points, numbered lists, or comma-separated items. "
        "Extract each item as a separate string. "
        "Common section headers: 'Coverage', 'Guarantees', 'Benefits'."
    ),
    examples=[
        ["Fire protection", "Water damage", "Theft"],
        ["Basic coverage", "Extended warranty"],
        ["Liability", "Property damage", "Personal injury"]
    ]
)
```

---

## Providing Helpful Examples

### The 2-5 Rule

Provide **2-5 diverse, realistic examples** per field:

```python
# Good: 3 diverse examples
currency: str = Field(
    ...,
    description="ISO 4217 currency code",
    examples=["EUR", "USD", "GBP"]
)

# Too few: Only 1 example
currency: str = Field(
    ...,
    description="ISO 4217 currency code",
    examples=["EUR"]  # Not enough variety
)

# Too many: Overwhelming
currency: str = Field(
    ...,
    description="ISO 4217 currency code",
    examples=["EUR", "USD", "GBP", "CHF", "JPY", "CNY", "AUD", "CAD"]  # Too many
)
```

### Example Diversity

Show different formats and edge cases:

```python
# Good: Shows format variations
date: str = Field(
    ...,
    description="Document date",
    examples=[
        "2024-01-15",      # ISO format
        "01/15/2024",      # US format
        "15.01.2024"       # European format
    ]
)

# Good: Shows value ranges
quantity: float = Field(
    ...,
    description="Item quantity",
    examples=[1.0, 28.5, 115.0]  # Small, medium, large
)
```

### Examples for Complex Types

#### List Examples

Show the **list structure**, not just individual items:

```python
# Correct: Show list structure
items: List[str] = Field(
    default_factory=list,
    description="List of item names",
    examples=[
        ["Item A", "Item B", "Item C"],
        ["Product 1", "Product 2"],
        ["Service X"]
    ]
)

# Wrong: Show individual items
items: List[str] = Field(
    default_factory=list,
    description="List of item names",
    examples=["Item A", "Item B", "Item C"]  # Not a list of lists!
)
```

#### Nested Object Examples

Show the complete structure:

```python
components: List[Component] = Field(
    default_factory=list,
    description="List of components with roles and amounts",
    examples=[
        [
            {
                "material": {"name": "Steel", "grade": "304"},
                "role": "Primary",
                "amount": {"value": 12.0, "unit": "kg"}
            },
            {
                "material": {"name": "Aluminum", "grade": "6061"},
                "role": "Secondary",
                "amount": {"value": 5.0, "unit": "kg"}
            }
        ]
    ]
)
```

---

## Type Hints and Defaults

### Common Type Patterns

```python
# Simple types
name: str = Field(...)
age: int = Field(...)
price: float = Field(...)
active: bool = Field(...)

# Optional types
email: Optional[str] = Field(None)
phone: Optional[int] = Field(None)

# Union types (multiple possible types)
value: Union[str, int, float] = Field(...)

# List types
tags: List[str] = Field(default_factory=list)
items: List[Item] = Field(default_factory=list)

# Date/time types
birth_date: date = Field(...)
created_at: datetime = Field(...)

# Enum types
status: StatusEnum = Field(...)
```

### Optional vs Union[Type, None]

These are equivalent:

```python
# Preferred (cleaner)
email: Optional[str] = Field(None)

# Equivalent (more explicit)
email: Union[str, None] = Field(None)
```

### Default Values by Type

```python
# Strings
name: str = Field("Unknown")
status: str = Field("pending")

# Numbers
count: int = Field(0)
price: float = Field(0.0)

# Booleans
active: bool = Field(True)
verified: bool = Field(False)

# Lists (always use default_factory)
items: List[str] = Field(default_factory=list)

# Dicts (always use default_factory)
metadata: Dict[str, Any] = Field(default_factory=dict)

# Dates (use callable)
created_at: datetime = Field(default_factory=datetime.now)
```

---

## Field Organization

### Grouping Related Fields

Organize fields logically within models:

```python
class Person(BaseModel):
    """Person entity."""
    model_config = ConfigDict(graph_id_fields=["first_name", "last_name"])
    
    # --- Identity Fields ---
    first_name: str = Field(...)
    last_name: str = Field(...)
    date_of_birth: Optional[date] = Field(None)
    
    # --- Contact Information ---
    email: Optional[str] = Field(None)
    phone: Optional[str] = Field(None)
    
    # --- Address ---
    addresses: List[Address] = edge(
        label="LIVES_AT",
        default_factory=list,
        description="Residential addresses"
    )
```

### Field Ordering Best Practices

1. **Required fields first**
2. **ID fields at the top** (for entities)
3. **Group related fields together**
4. **Edges at the end**

```python
class Invoice(BaseModel):
    """Invoice document."""
    model_config = ConfigDict(graph_id_fields=["invoice_number"])
    
    # 1. Required ID field
    invoice_number: str = Field(...)
    
    # 2. Required core fields
    date: str = Field(...)
    total: float = Field(...)
    
    # 3. Optional fields
    notes: Optional[str] = Field(None)
    payment_terms: Optional[str] = Field(None)
    
    # 4. Edges
    issued_by: Organization = edge(label="ISSUED_BY")
    sent_to: Client = edge(label="SENT_TO")
    contains_items: List[LineItem] = edge(
        label="CONTAINS_ITEM",
        default_factory=list
    )
```

---

## Advanced Field Patterns

### Multi-line Descriptions

For complex fields, use multi-line strings:

```python
# Using parentheses (preferred)
field: str = Field(
    ...,
    description=(
        "First line of description. "
        "Second line with more details. "
        "Third line with extraction hints."
    ),
    examples=["Example 1", "Example 2"]
)

# Using triple quotes (alternative)
field: str = Field(
    ...,
    description="""
        First line of description.
        Second line with more details.
        Third line with extraction hints.
    """.strip(),
    examples=["Example 1", "Example 2"]
)
```

### Conditional Fields

Document when fields are relevant:

```python
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
```

### Flexible Value Fields

Support multiple value types:

```python
value: Union[str, int, float] = Field(
    ...,
    description=(
        "Value can be numeric (int/float) or textual. "
        "Extract as-is: '100', '25.5', or 'High'."
    ),
    examples=[100, 25.5, "High", "Medium"]
)
```

---

## Testing Field Definitions

### Test 1: LLM Understanding

Ask yourself:
- "If I were an LLM, could I extract this field from the description alone?"
- "Are there ambiguous cases I haven't addressed?"
- "Do the examples cover the expected variety?"

### Test 2: Extraction Quality

Run extraction and check results:

```bash
uv run docling-graph convert document.pdf \
    --template "my_template.MyTemplate" \
    --output-dir test_output
```

Check `test_output/extracted_data.json`:
- Are fields extracted correctly?
- Are values normalized as expected?
- Are edge cases handled properly?

### Test 3: Example Coverage

Verify examples match real data:

```python
# Compare examples to actual extracted values
import json

with open("test_output/extracted_data.json") as f:
    data = json.load(f)

# Check if extracted values match example patterns
for item in data:
    print(f"Extracted: {item['field_name']}")
    # Does this match the examples you provided?
```

---

## Common Mistakes

### ❌ Mistake 1: Vague Descriptions

```python
# Bad
name: str = Field(..., description="Name")

# Good
name: str = Field(
    ...,
    description=(
        "Full legal name of the organization. "
        "Look for 'Company Name' or header text. "
        "Include legal suffixes like 'Ltd', 'Inc'."
    ),
    examples=["Acme Corp Ltd", "Tech Solutions Inc"]
)
```

### ❌ Mistake 2: Missing Examples

```python
# Bad
email: str = Field(..., description="Email address")

# Good
email: str = Field(
    ...,
    description="Contact email address",
    examples=["contact@company.com", "info@org.fr"]
)
```

### ❌ Mistake 3: Wrong List Defaults

```python
# Bad - Creates shared mutable object
items: List[str] = Field([], description="Items")

# Good - Uses factory
items: List[str] = Field(
    default_factory=list,
    description="Items"
)
```

### ❌ Mistake 4: Inconsistent Examples

```python
# Bad - Examples don't match description
currency: str = Field(
    ...,
    description="ISO 4217 currency code (3 uppercase letters)",
    examples=["Euro", "Dollar", "Pound"]  # Wrong format!
)

# Good - Examples match description
currency: str = Field(
    ...,
    description="ISO 4217 currency code (3 uppercase letters)",
    examples=["EUR", "USD", "GBP"]  # Correct format
)
```

---

## Next Steps

Now that you understand field definitions:

1. **[Relationships →](relationships.md)** - Connect models with edges
2. **[Validation](validation.md)** - Add validators for data quality
3. **[Best Practices](best-practices.md)** - Follow the complete checklist

---

## Quick Reference

### Field Definition Template

```python
field_name: FieldType = Field(
    default_or_required,  # ... or None or value
    description=(
        "Clear purpose. "
        "Extraction hints (field names, patterns). "
        "Parsing instructions (normalization)."
    ),
    examples=["Example 1", "Example 2", "Example 3"]
)
```

### Common Patterns

```python
# Required field
field: str = Field(..., description="...", examples=[...])

# Optional field
field: Optional[str] = Field(None, description="...", examples=[...])

# List field
field: List[str] = Field(default_factory=list, description="...", examples=[[...]])

# Field with default
field: str = Field("default", description="...", examples=[...])
```

---

**Navigation:** [← Entities vs Components](entities-vs-components.md) | [Next: Relationships →](relationships.md)