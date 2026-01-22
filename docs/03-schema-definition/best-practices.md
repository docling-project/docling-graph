# Best Practices and Checklist

**Navigation:** [← Advanced Patterns](advanced-patterns.md) | [Next: Pipeline Configuration →](../04-pipeline-configuration/index.md)

---

## Overview

This guide provides a comprehensive checklist and best practices for creating high-quality Pydantic templates. Use this as a final review before deploying your template.

**In this guide:**
- Complete template checklist
- Testing your template
- Common pitfalls to avoid
- Performance considerations
- Maintenance tips

---

## Complete Template Checklist

### ✅ Structure and Organization

- [ ] **Module docstring** - Clear description of template purpose
- [ ] **Standard imports** - All necessary imports included
- [ ] **Edge helper function** - Defined correctly with exact signature
- [ ] **Logical organization** - Components → Entities → Domain Models → Root
- [ ] **Model docstrings** - All models have clear docstrings
- [ ] **Consistent naming** - Follow Python naming conventions

### ✅ Entity Configuration

- [ ] **graph_id_fields defined** - All entities have appropriate ID fields
- [ ] **ID fields are stable** - Won't change frequently
- [ ] **ID fields are likely present** - Will be extracted from documents
- [ ] **Composite IDs make sense** - Multiple fields form natural unique identifier

### ✅ Component Configuration

- [ ] **is_entity=False set** - All components marked correctly
- [ ] **Appropriate for sharing** - Components represent value objects
- [ ] **Content-based deduplication** - All fields used for uniqueness

### ✅ Field Definitions

- [ ] **Clear descriptions** - LLM-friendly with extraction hints
- [ ] **Realistic examples** - 2-5 diverse examples per field
- [ ] **Proper type hints** - Optional, List, Union used correctly
- [ ] **Appropriate defaults** - Required (...), None, or meaningful defaults
- [ ] **List fields use default_factory** - Never use [] as default

### ✅ Edge Definitions

- [ ] **Descriptive labels** - ALL_CAPS_WITH_UNDERSCORES format
- [ ] **Consistent naming** - Same pattern across template
- [ ] **List edges have default_factory** - Required for list relationships
- [ ] **Clear descriptions** - Explain the relationship
- [ ] **Appropriate cardinality** - Single vs list chosen correctly

### ✅ Validation

- [ ] **Field validators** - Data quality checks where needed
- [ ] **Model validators** - Cross-field validation implemented
- [ ] **Clear error messages** - Specific, actionable errors
- [ ] **Handle None values** - Validators allow None for optional fields
- [ ] **Pre-validators for normalization** - mode='before' used appropriately

### ✅ String Representations

- [ ] **__str__ methods** - Defined for entities and key components
- [ ] **Handle None values** - String methods don't crash on None
- [ ] **Meaningful output** - Useful for debugging and logging

### ✅ Type Hints and Consistency

- [ ] **Proper type hints** - All fields have correct types
- [ ] **Consistent patterns** - Similar fields use similar patterns
- [ ] **No duplicate code** - Reusable components extracted

---

## Testing Your Template

### Test 1: Basic Instantiation

```python
# test_template_basic.py
from my_template import Document, Organization, Address

def test_basic_instantiation():
    """Test that models can be instantiated."""
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
    assert doc.document_id == "TEST-001"
    assert doc.issued_by.name == "Test Corp"
    print("✓ Basic instantiation works")

if __name__ == "__main__":
    test_basic_instantiation()
```

Run with:
```bash
uv run python test_template_basic.py
```

### Test 2: Validation

```python
# test_template_validation.py
from my_template import MonetaryAmount
import pytest

def test_positive_amount():
    """Test that negative amounts are rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        MonetaryAmount(value=-100, currency="EUR")
    print("✓ Validation works")

def test_valid_amount():
    """Test that positive amounts are accepted."""
    amount = MonetaryAmount(value=100, currency="EUR")
    assert amount.value == 100
    print("✓ Valid data accepted")

if __name__ == "__main__":
    test_positive_amount()
    test_valid_amount()
```

Run with:
```bash
uv run pytest test_template_validation.py -v
```

### Test 3: Serialization

```python
# test_template_serialization.py
from my_template import Document, Organization, Address
import json

def test_json_serialization():
    """Test that models can be serialized to JSON."""
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
    
    # Serialize to JSON
    json_str = doc.model_dump_json(indent=2)
    print("✓ JSON serialization works")
    print(json_str)
    
    # Deserialize from JSON
    json_data = json.loads(json_str)
    doc2 = Document(**json_data)
    assert doc2.document_id == doc.document_id
    print("✓ JSON deserialization works")

if __name__ == "__main__":
    test_json_serialization()
```

### Test 4: Edge Metadata

```python
# test_template_edges.py
from my_template import Document

def test_edge_metadata():
    """Test that edge metadata is present."""
    # Get field info
    fields = Document.model_fields
    
    # Check issued_by has edge metadata
    issued_by_field = fields["issued_by"]
    metadata = issued_by_field.json_schema_extra
    
    assert metadata is not None
    assert "edge_label" in metadata
    assert metadata["edge_label"] == "ISSUED_BY"
    print("✓ Edge metadata present")

if __name__ == "__main__":
    test_edge_metadata()
```

### Test 5: End-to-End Extraction

```python
# test_template_extraction.py
"""Test template with actual extraction."""

def test_extraction():
    """Test extraction with a sample document."""
    import subprocess
    
    result = subprocess.run([
        "uv", "run", "docling-graph", "convert",
        "test_document.pdf",
        "--template", "my_template.Document",
        "--output-dir", "test_output",
        "--backend", "llm",
        "--inference", "local"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Extraction failed: {result.stderr}"
    print("✓ End-to-end extraction works")

if __name__ == "__main__":
    test_extraction()
```

---

## Common Pitfalls to Avoid

### ❌ Pitfall 1: Wrong edge() Definition

```python
# WRONG - Missing **kwargs
def edge(label: str) -> Any:
    return Field(..., json_schema_extra={"edge_label": label})

# CORRECT
def edge(label: str, **kwargs: Any) -> Any:
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)
```

### ❌ Pitfall 2: Missing default_factory for Lists

```python
# WRONG
items: List[Item] = edge(label="CONTAINS_ITEM")

# CORRECT
items: List[Item] = edge(
    label="CONTAINS_ITEM",
    default_factory=list
)
```

### ❌ Pitfall 3: Mutable Default Values

```python
# WRONG - Shared mutable object
items: List[str] = Field([])

# CORRECT
items: List[str] = Field(default_factory=list)
```

### ❌ Pitfall 4: Vague Descriptions

```python
# WRONG
name: str = Field(..., description="Name")

# CORRECT
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

### ❌ Pitfall 5: Inconsistent Edge Labels

```python
# WRONG - Mixed formats
issued_by: Org = edge(label="issuedBy")
sent_to: Client = edge(label="SENT_TO")
has_items: List[Item] = edge(label="contains-item")

# CORRECT - Consistent ALL_CAPS_WITH_UNDERSCORES
issued_by: Org = edge(label="ISSUED_BY")
sent_to: Client = edge(label="SENT_TO")
has_items: List[Item] = edge(label="CONTAINS_ITEM")
```

### ❌ Pitfall 6: Wrong Entity/Component Classification

```python
# WRONG - Address as entity (creates duplicate nodes)
class Address(BaseModel):
    model_config = ConfigDict(graph_id_fields=["street", "city"])

# CORRECT - Address as component (shared nodes)
class Address(BaseModel):
    model_config = ConfigDict(is_entity=False)
```

### ❌ Pitfall 7: Unstable ID Fields

```python
# WRONG - Email can change
class Person(BaseModel):
    model_config = ConfigDict(graph_id_fields=["email"])

# CORRECT - Stable fields
class Person(BaseModel):
    model_config = ConfigDict(
        graph_id_fields=["first_name", "last_name", "date_of_birth"]
    )
```

### ❌ Pitfall 8: Missing Validators

```python
# WRONG - No validation
currency: str = Field(...)

# CORRECT - Validated
currency: str = Field(...)

@field_validator("currency")
@classmethod
def validate_currency(cls, v: Any) -> Any:
    if v and not (len(v) == 3 and v.isupper()):
        raise ValueError("Currency must be 3 uppercase letters")
    return v
```

---

## Performance Considerations

### 1. Keep Templates Focused

```python
# ✅ Good - Focused template
class Invoice(BaseModel):
    """Invoice document."""
    # Only invoice-related fields

# ❌ Bad - Kitchen sink template
class Document(BaseModel):
    """Generic document."""
    # Hundreds of fields for every document type
```

### 2. Use Appropriate Validators

```python
# ✅ Good - Simple validation
@field_validator("value")
@classmethod
def validate_positive(cls, v: Any) -> Any:
    if v < 0:
        raise ValueError("Must be non-negative")
    return v

# ❌ Bad - Complex validation in validator
@field_validator("value")
@classmethod
def validate_complex(cls, v: Any) -> Any:
    # Expensive database lookup
    # Complex calculations
    # Multiple API calls
    return v
```

### 3. Minimize Nested Depth

```python
# ✅ Good - Reasonable nesting (2-3 levels)
Invoice → LineItem → MonetaryAmount

# ❌ Bad - Excessive nesting (5+ levels)
Document → Section → Subsection → Paragraph → Sentence → Word
```

---

## Maintenance Tips

### 1. Version Your Templates

```python
"""
Invoice extraction template.

Version: 2.0.0
Last Updated: 2024-01-15
Changes:
  - Added payment_terms field
  - Updated Organization to include tax_id
  - Fixed email validation
"""
```

### 2. Document Breaking Changes

```python
"""
BREAKING CHANGES in v2.0.0:
- Renamed 'bill_no' to 'invoice_number'
- Changed 'date' from str to date type
- Removed deprecated 'legacy_field'
"""
```

### 3. Keep Examples Updated

```python
# ✅ Good - Current examples
invoice_number: str = Field(
    ...,
    description="Unique invoice identifier",
    examples=["INV-2024-001", "2024-INV-12345"]  # Current format
)

# ❌ Bad - Outdated examples
invoice_number: str = Field(
    ...,
    description="Unique invoice identifier",
    examples=["INV-2020-001", "2020-INV-12345"]  # Old format
)
```

### 4. Add Migration Guides

```python
"""
Migration from v1.x to v2.0:

1. Rename fields:
   - bill_no → invoice_number
   - client → sent_to

2. Update types:
   - date: str → date

3. Add required fields:
   - payment_terms (default: "Net 30")
"""
```

---

## Template Quality Checklist

### Before Deployment

- [ ] All tests pass
- [ ] Template validated with sample documents
- [ ] Edge metadata verified
- [ ] Documentation complete
- [ ] Examples realistic and current
- [ ] No TODO or FIXME comments
- [ ] Code reviewed by team
- [ ] Performance tested with large documents

### After Deployment

- [ ] Monitor extraction quality
- [ ] Collect feedback from users
- [ ] Track common extraction errors
- [ ] Update examples based on real data
- [ ] Refine descriptions based on LLM performance
- [ ] Version and document changes

---

## Quick Start Template

Use this as a starting point for new templates:

```python
"""
[Template Name] extraction template.

Extracts [key information] from [document type] documents.

Version: 1.0.0
Last Updated: [Date]
"""

from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- Edge Helper Function (REQUIRED) ---
def edge(label: str, **kwargs: Any) -> Any:
    """Helper to create graph edges."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

# --- Components ---
class Address(BaseModel):
    """Physical address component."""
    model_config = ConfigDict(is_entity=False)
    
    street: str = Field(
        description="Street name and number",
        examples=["123 Main St", "45 Rue de la Paix"]
    )
    city: str = Field(
        description="City name",
        examples=["Paris", "London"]
    )

# --- Entities ---
class Organization(BaseModel):
    """Organization entity."""
    model_config = ConfigDict(graph_id_fields=["name"])
    
    name: str = Field(
        description="Legal organization name",
        examples=["Acme Corp", "Tech Solutions Ltd"]
    )
    
    located_at: Address = edge(
        label="LOCATED_AT",
        description="Organization's physical address"
    )

# --- Root Document ---
class [DocumentName](BaseModel):
    """[Document type] document."""
    model_config = ConfigDict(graph_id_fields=["document_id"])
    
    document_id: str = Field(
        description="Unique document identifier",
        examples=["DOC-2024-001", "12345"]
    )
    
    issued_by: Organization = edge(
        label="ISSUED_BY",
        description="Organization that issued this document"
    )
```

---

## Next Steps

Congratulations! You've completed the Schema Definition guide. Now:

1. **[Pipeline Configuration →](../04-pipeline-configuration/index.md)** - Configure extraction settings
2. **[Examples](../09-examples/index.md)** - See complete working templates
3. **[Extraction Process](../05-extraction-process/index.md)** - Understand the extraction pipeline

---

## Additional Resources

### Documentation
- **[Pydantic Documentation](https://docs.pydantic.dev/)** - Official Pydantic docs
- **[Template Examples](../09-examples/index.md)** - Production-ready templates
- **[API Reference](../11-reference/config.md)** - Complete API documentation

### Example Templates
- **Invoice Template** - `docs/examples/templates/invoice.py`
- **ID Card Template** - `docs/examples/templates/id_card.py`
- **Research Paper Template** - `docs/examples/templates/rheology_research.py`
- **Insurance Template** - `docs/examples/templates/insurance.py`

### Community
- **GitHub Issues** - Report bugs or request features
- **Discussions** - Ask questions and share templates

---

## Final Checklist

Before moving to Pipeline Configuration, ensure:

- ✅ Template structure follows best practices
- ✅ All entities have appropriate `graph_id_fields`
- ✅ All components have `is_entity=False`
- ✅ Edge labels are consistent and descriptive
- ✅ Field descriptions are LLM-friendly
- ✅ Examples are realistic and diverse
- ✅ Validators ensure data quality
- ✅ Tests pass successfully
- ✅ Template tested with sample documents

---

**Navigation:** [← Advanced Patterns](advanced-patterns.md) | [Next: Pipeline Configuration →](../04-pipeline-configuration/index.md)