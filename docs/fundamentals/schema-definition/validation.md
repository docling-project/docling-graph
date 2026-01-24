# Validation and Normalization


## Overview

Validators ensure data quality and consistency in your extracted data. Pydantic provides powerful validation mechanisms that can transform, normalize, and validate field values before they're stored in your knowledge graph.

**In this guide:**
- Field validators for single-field validation
- Model validators for cross-field validation
- Pre-validators for data transformation
- Common validation patterns
- Normalization helpers

---

## Field Validators

### Basic Field Validator

Use `@field_validator` to validate individual fields:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Any

class MonetaryAmount(BaseModel):
    """Monetary value with validation."""
    model_config = ConfigDict(is_entity=False)
    
    value: float = Field(...)
    currency: Optional[str] = Field(None)
    
    @field_validator("value")
    @classmethod
    def validate_positive(cls, v: Any) -> Any:
        """Ensure value is non-negative."""
        if v < 0:
            raise ValueError("Monetary amount must be non-negative")
        return v
```

### Validator Anatomy

```python
@field_validator("field_name")  # Field to validate
@classmethod  # Must be classmethod
def validator_name(cls, v: Any) -> Any:  # Takes value, returns value
    """Docstring explaining validation."""
    # Validation logic
    if not valid:
        raise ValueError("Error message")
    return v  # Return (possibly modified) value
```

---

## Pre-Validators (mode='before')

### When to Use Pre-Validators

Use `mode='before'` to transform input **before** type coercion:

```python
@field_validator("email", mode="before")
@classmethod
def normalize_email(cls, v: Any) -> Any:
    """Convert email to lowercase and strip whitespace."""
    if v:
        return v.lower().strip()
    return v
```

**Use cases:**
- Normalizing strings (lowercase, strip whitespace)
- Converting types (string to list)
- Parsing complex formats
- Cleaning input data

### Pre-Validator Examples

#### Example 1: Email Normalization

```python
class Person(BaseModel):
    """Person with normalized email."""
    
    email: Optional[str] = Field(None)
    
    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v: Any) -> Any:
        """Convert email to lowercase and strip whitespace."""
        if v:
            return v.lower().strip()
        return v
```

**Input/Output:**
```python
Person(email="  John.Doe@EMAIL.COM  ")
# Result: email="john.doe@email.com"
```

#### Example 2: String to List Conversion

```python
class Person(BaseModel):
    """Person with flexible name input."""
    
    given_names: List[str] = Field(default_factory=list)
    
    @field_validator("given_names", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> Any:
        """Ensure given_names is always a list."""
        if isinstance(v, str):
            # Handle comma-separated names
            if "," in v:
                return [name.strip() for name in v.split(",")]
            return [v]
        return v
```

**Input/Output:**
```python
Person(given_names="John, Paul, George")
# Result: given_names=["John", "Paul", "George"]

Person(given_names="John")
# Result: given_names=["John"]

Person(given_names=["John", "Paul"])
# Result: given_names=["John", "Paul"]
```

#### Example 3: Phone Number Cleaning

```python
class Contact(BaseModel):
    """Contact with cleaned phone number."""
    
    phone: Optional[str] = Field(None)
    
    @field_validator("phone", mode="before")
    @classmethod
    def clean_phone(cls, v: Any) -> Any:
        """Remove non-numeric characters except + and spaces."""
        if v:
            # Keep only digits, +, and spaces
            import re
            return re.sub(r'[^\d\s+]', '', v)
        return v
```

**Input/Output:**
```python
Contact(phone="+33 (0)1-23-45-67-89")
# Result: phone="+33 01 23 45 67 89"
```

---

## Post-Validators (Default Mode)

### When to Use Post-Validators

Use default mode (or `mode='after'`) to validate **after** type coercion:

```python
@field_validator("currency")
@classmethod
def validate_currency_format(cls, v: Any) -> Any:
    """Ensure currency is 3 uppercase letters (ISO 4217)."""
    if v and not (len(v) == 3 and v.isupper()):
        raise ValueError("Currency must be 3 uppercase letters (ISO 4217)")
    return v
```

**Use cases:**
- Validating format constraints
- Checking value ranges
- Enforcing business rules
- Verifying data integrity

### Post-Validator Examples

#### Example 1: Currency Code Validation

```python
class MonetaryAmount(BaseModel):
    """Monetary amount with validated currency."""
    model_config = ConfigDict(is_entity=False)
    
    value: float = Field(...)
    currency: Optional[str] = Field(None)
    
    @field_validator("currency")
    @classmethod
    def validate_currency_format(cls, v: Any) -> Any:
        """Ensure currency is 3 uppercase letters."""
        if v and not (len(v) == 3 and v.isupper()):
            raise ValueError("Currency must be 3 uppercase letters (ISO 4217)")
        return v
```

#### Example 2: Range Validation

```python
class Product(BaseModel):
    """Product with validated quantity."""
    
    quantity: int = Field(...)
    
    @field_validator("quantity")
    @classmethod
    def validate_quantity_range(cls, v: Any) -> Any:
        """Ensure quantity is between 1 and 10000."""
        if v < 1:
            raise ValueError("Quantity must be at least 1")
        if v > 10000:
            raise ValueError("Quantity cannot exceed 10000")
        return v
```

#### Example 3: Email Format Validation

```python
class Contact(BaseModel):
    """Contact with validated email."""
    
    email: Optional[str] = Field(None)
    
    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: Any) -> Any:
        """Basic email format validation."""
        if v and "@" not in v:
            raise ValueError("Invalid email format")
        return v
```

---

## Model Validators

### When to Use Model Validators

Use `@model_validator` for **cross-field validation** - when validation depends on multiple fields:

```python
from pydantic import model_validator
from typing_extensions import Self

class Measurement(BaseModel):
    """Measurement with cross-field validation."""
    model_config = ConfigDict(is_entity=False)
    
    numeric_value: Optional[float] = Field(None)
    numeric_value_min: Optional[float] = Field(None)
    numeric_value_max: Optional[float] = Field(None)
    
    @model_validator(mode="after")
    def validate_value_consistency(self) -> Self:
        """Ensure value fields are used consistently."""
        has_single = self.numeric_value is not None
        has_min = self.numeric_value_min is not None
        has_max = self.numeric_value_max is not None
        
        if has_single and has_min and has_max:
            raise ValueError(
                "Cannot specify numeric_value, numeric_value_min, "
                "and numeric_value_max simultaneously"
            )
        
        return self
```

### Model Validator Examples

#### Example 1: Date Range Validation

```python
from datetime import date

class Event(BaseModel):
    """Event with validated date range."""
    
    start_date: Optional[date] = Field(None)
    end_date: Optional[date] = Field(None)
    
    @model_validator(mode="after")
    def validate_date_range(self) -> Self:
        """Ensure end_date is after start_date."""
        if self.start_date and self.end_date:
            if self.end_date < self.start_date:
                raise ValueError("end_date must be after start_date")
        return self
```

#### Example 2: Conditional Required Fields

```python
class Document(BaseModel):
    """Document with conditional validation."""
    
    document_type: str = Field(...)
    invoice_number: Optional[str] = Field(None)
    receipt_number: Optional[str] = Field(None)
    
    @model_validator(mode="after")
    def validate_document_numbers(self) -> Self:
        """Ensure appropriate number field is present."""
        if self.document_type == "invoice" and not self.invoice_number:
            raise ValueError("invoice_number required for invoice documents")
        if self.document_type == "receipt" and not self.receipt_number:
            raise ValueError("receipt_number required for receipt documents")
        return self
```

#### Example 3: Mutual Exclusivity

```python
class Payment(BaseModel):
    """Payment with mutually exclusive fields."""
    
    cash_amount: Optional[float] = Field(None)
    card_amount: Optional[float] = Field(None)
    check_amount: Optional[float] = Field(None)
    
    @model_validator(mode="after")
    def validate_single_payment_method(self) -> Self:
        """Ensure only one payment method is used."""
        methods = [
            self.cash_amount is not None,
            self.card_amount is not None,
            self.check_amount is not None
        ]
        if sum(methods) > 1:
            raise ValueError("Only one payment method can be specified")
        if sum(methods) == 0:
            raise ValueError("At least one payment method must be specified")
        return self
```

---

## Common Validation Patterns

### Pattern 1: Positive Number Validation

```python
@field_validator("amount", "quantity", "price")
@classmethod
def validate_positive(cls, v: Any) -> Any:
    """Ensure value is positive."""
    if v is not None and v < 0:
        raise ValueError(f"Value must be non-negative, got {v}")
    return v
```

### Pattern 2: String Length Validation

```python
@field_validator("postal_code")
@classmethod
def validate_postal_code_length(cls, v: Any) -> Any:
    """Ensure postal code is 5 digits."""
    if v and len(v) != 5:
        raise ValueError("Postal code must be 5 digits")
    return v
```

### Pattern 3: Enum-like Validation

```python
@field_validator("status")
@classmethod
def validate_status(cls, v: Any) -> Any:
    """Ensure status is one of allowed values."""
    allowed = ["pending", "approved", "rejected"]
    if v and v not in allowed:
        raise ValueError(f"Status must be one of {allowed}")
    return v
```

### Pattern 4: Pattern Matching

```python
import re

@field_validator("email")
@classmethod
def validate_email_pattern(cls, v: Any) -> Any:
    """Validate email format using regex."""
    if v:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError("Invalid email format")
    return v
```

---

## Enum Normalization Helper

### The Problem

Enums can be tricky with LLM extraction - the model might return various formats:

```python
from enum import Enum

class Status(str, Enum):
    PENDING = "Pending"
    APPROVED = "Approved"
    REJECTED = "Rejected"

# LLM might return: "pending", "PENDING", "Pending", "approved", etc.
```

### The Solution

Use a normalization helper:

```python
import re
from enum import Enum
from typing import Type, Any

def _normalize_enum(enum_cls: Type[Enum], v: Any) -> Any:
    """
    Accept enum instances, value strings, or member names.
    Handles various formats: 'VALUE', 'value', 'Value', 'VALUE_NAME'.
    Falls back to 'OTHER' member if present.
    """
    if isinstance(v, enum_cls):
        return v
    
    if isinstance(v, str):
        # Normalize to alphanumeric lowercase
        key = re.sub(r"[^A-Za-z0-9]+", "", v).lower()
        
        # Build mapping of normalized names/values to enum members
        mapping = {}
        for member in enum_cls:
            normalized_name = re.sub(r"[^A-Za-z0-9]+", "", member.name).lower()
            normalized_value = re.sub(r"[^A-Za-z0-9]+", "", member.value).lower()
            mapping[normalized_name] = member
            mapping[normalized_value] = member
        
        if key in mapping:
            return mapping[key]
        
        # Last attempt: direct value match
        try:
            return enum_cls(v)
        except Exception:
            # Safe fallback to OTHER if present
            if "OTHER" in enum_cls.__members__:
                return enum_cls.OTHER
            raise
    
    raise ValueError(f"Cannot normalize {v} to {enum_cls}")
```

### Usage Example

```python
class DocumentType(str, Enum):
    INVOICE = "Invoice"
    RECEIPT = "Receipt"
    CREDIT_NOTE = "Credit Note"
    OTHER = "Other"

class Document(BaseModel):
    """Document with normalized enum."""
    
    document_type: DocumentType = Field(...)
    
    @field_validator("document_type", mode="before")
    @classmethod
    def normalize_document_type(cls, v: Any) -> Any:
        return _normalize_enum(DocumentType, v)
```

**Handles all these inputs:**
```python
Document(document_type="invoice")  # → DocumentType.INVOICE
Document(document_type="INVOICE")  # → DocumentType.INVOICE
Document(document_type="Invoice")  # → DocumentType.INVOICE
Document(document_type="credit note")  # → DocumentType.CREDIT_NOTE
Document(document_type="unknown")  # → DocumentType.OTHER (fallback)
```

---

## Measurement Parsing Helper

### The Problem

LLMs might return measurements in various formats:

```
"1.6 mPa.s"
"2 mm"
"80-90 °C"
"High"
```

### The Solution

Use a parsing helper:

```python
import re
from typing import Any, Optional

def _parse_measurement_string(
    s: str,
    default_name: Optional[str] = None,
    strict: bool = False
) -> dict[str, Any]:
    """
    Parse measurement strings into structured dict.
    
    Examples:
        "1.6 mPa.s" → {numeric_value: 1.6, unit: "mPa.s"}
        "80-90 °C" → {numeric_value_min: 80, numeric_value_max: 90, unit: "°C"}
        "High" → {text_value: "High"}
    """
    if not isinstance(s, str):
        return s
    
    # Try to parse range (e.g., "80-90 °C")
    range_match = re.match(
        r"^\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*([^\d]+)?$",
        s
    )
    if range_match:
        min_val = float(range_match.group(1))
        max_val = float(range_match.group(2))
        unit = (range_match.group(3) or "").strip() or None
        return {
            "name": default_name or "Value",
            "numeric_value": None,
            "numeric_value_min": min_val,
            "numeric_value_max": max_val,
            "text_value": None,
            "unit": unit,
        }
    
    # Try to parse single value (e.g., "1.6 mPa.s")
    single_match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([^\d]+)?$", s)
    if single_match:
        num = float(single_match.group(1))
        unit = (single_match.group(2) or "").strip() or None
        return {
            "name": default_name or "Value",
            "numeric_value": num,
            "numeric_value_min": None,
            "numeric_value_max": None,
            "text_value": None,
            "unit": unit,
        }
    
    # No numeric part found
    if strict:
        raise ValueError(f"Cannot parse '{s}' as measurement")
    
    # Fallback: keep raw as text
    return {
        "name": default_name or "Value",
        "numeric_value": None,
        "numeric_value_min": None,
        "numeric_value_max": None,
        "text_value": s.strip(),
        "unit": None,
    }
```

### Usage Example

```python
class Measurement(BaseModel):
    """Flexible measurement model."""
    model_config = ConfigDict(is_entity=False)
    
    name: str = Field(...)
    numeric_value: Optional[float] = Field(None)
    numeric_value_min: Optional[float] = Field(None)
    numeric_value_max: Optional[float] = Field(None)
    text_value: Optional[str] = Field(None)
    unit: Optional[str] = Field(None)
    
    @field_validator("numeric_value", "numeric_value_min", "numeric_value_max", mode="before")
    @classmethod
    def parse_if_string(cls, v: Any, info: ValidationInfo) -> Any:
        """Parse measurement strings."""
        if isinstance(v, str):
            field_name = info.field_name
            parsed = _parse_measurement_string(v, default_name=field_name)
            return parsed.get(field_name)
        return v
```

---

## Best Practices

### 1. Validate Early

Use `mode='before'` for normalization, default mode for validation:

```python
@field_validator("email", mode="before")
@classmethod
def normalize_email(cls, v: Any) -> Any:
    """Normalize before validation."""
    if v:
        return v.lower().strip()
    return v

@field_validator("email")
@classmethod
def validate_email(cls, v: Any) -> Any:
    """Validate after normalization."""
    if v and "@" not in v:
        raise ValueError("Invalid email")
    return v
```

### 2. Provide Clear Error Messages

```python
# ✅ Good - Specific error message
@field_validator("quantity")
@classmethod
def validate_quantity(cls, v: Any) -> Any:
    if v < 1:
        raise ValueError(f"Quantity must be at least 1, got {v}")
    return v

# ❌ Bad - Vague error message
@field_validator("quantity")
@classmethod
def validate_quantity(cls, v: Any) -> Any:
    if v < 1:
        raise ValueError("Invalid quantity")
    return v
```

### 3. Handle None Values

```python
@field_validator("email")
@classmethod
def validate_email(cls, v: Any) -> Any:
    """Validate email, allowing None."""
    if v is None:
        return v  # Allow None for optional fields
    if "@" not in v:
        raise ValueError("Invalid email")
    return v
```

### 4. Use Type Guards

```python
@field_validator("value", mode="before")
@classmethod
def coerce_to_float(cls, v: Any) -> Any:
    """Convert string to float if needed."""
    if isinstance(v, str):
        try:
            return float(v.replace(",", ""))
        except ValueError:
            raise ValueError(f"Cannot convert '{v}' to float")
    return v
```

---

## Testing Validators

### Test Individual Validators

```python
# test_validators.py
from my_template import MonetaryAmount
import pytest

def test_positive_amount():
    """Test that negative amounts are rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        MonetaryAmount(value=-100, currency="EUR")

def test_valid_amount():
    """Test that positive amounts are accepted."""
    amount = MonetaryAmount(value=100, currency="EUR")
    assert amount.value == 100
```

### Test with uv

```bash
uv run pytest test_validators.py -v
```

---

## Next Steps

Now that you understand validation:

1. **[Advanced Patterns →](advanced-patterns.md)** - Complex validation patterns
2. **[Best Practices](best-practices.md)** - Complete template checklist
3. **[Examples](../../usage/examples/index.md)** - See validators in action

---

## Quick Reference

### Field Validator Template

```python
@field_validator("field_name", mode="before")  # or default mode
@classmethod
def validator_name(cls, v: Any) -> Any:
    """Docstring."""
    # Validation/transformation logic
    if not valid:
        raise ValueError("Error message")
    return v
```

### Model Validator Template

```python
@model_validator(mode="after")
def validator_name(self) -> Self:
    """Docstring."""
    # Cross-field validation
    if not valid:
        raise ValueError("Error message")
    return self
```