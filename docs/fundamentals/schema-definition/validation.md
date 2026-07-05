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

#### 📍 Email Normalization

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

#### 📍 String to List Conversion

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

#### 📍 Phone Number Cleaning

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

#### 📍 Currency Code Validation

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

#### 📍 Range Validation

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

#### 📍 Email Format Validation

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

#### 📍 Date Range Validation

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

#### 📍 Conditional Required Fields

```python
class Document(BaseModel):
    """Document with conditional validation."""
    
    document_type: str = Field(...)
    document_no: Optional[str] = Field(None)
    receipt_number: Optional[str] = Field(None)
    
    @model_validator(mode="after")
    def validate_document_numbers(self) -> Self:
        """Ensure appropriate number field is present."""
        if self.document_type == "invoice" and not self.document_no:
            raise ValueError("document_no required for invoice documents")
        if self.document_type == "receipt" and not self.receipt_number:
            raise ValueError("receipt_number required for receipt documents")
        return self
```

#### 📍 Mutual Exclusivity

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

#### 📍 Semantic sanity: clear wrong unit/amount

When a field pair has a clear semantic (e.g. amount + unit), use a **model_validator** to clear both if the unit indicates a different quantity type (e.g. temperature or viscosity mistaken for amount). Prefer **clearing** over raising so extraction still validates:

```python
import re
from typing_extensions import Self

class Component(BaseModel):
    """Component with amount; amount_unit must be a quantity unit, not a property unit."""
    amount_value: float | None = Field(None)
    amount_unit: str | None = Field(None)
    # ... other fields

    @model_validator(mode="after")
    def clear_amount_if_property_unit(self) -> Self:
        """Clear amount_value/amount_unit when unit indicates a property (temp, viscosity), not a quantity."""
        unit = self.amount_unit
        if not unit or not isinstance(unit, str):
            return self
        normalized = re.sub(r"[\s·]", "", unit.lower())
        # Forbidden: temperature, viscosity, pressure
        forbidden = ("°c", "k", "pa.s", "pas", "mpa.s", "mpas")
        if any(f in normalized for f in forbidden) or normalized == "pa":
            object.__setattr__(self, "amount_value", None)
            object.__setattr__(self, "amount_unit", None)
        return self
```

#### 📍 Deduplicate root-level list by key

For root-level list fields that have no identity (e.g. authors), chunked extraction can append the same item multiple times. Use a **model_validator** to keep first occurrence per key:

```python
from typing import List
from typing_extensions import Self

class PersonIdentity(BaseModel):
    full_name: str = Field(...)

class Document(BaseModel):
    """Root document."""
    authors: List[PersonIdentity] = Field(default_factory=list)
    # ... other fields

    @model_validator(mode="after")
    def deduplicate_authors_by_name(self) -> Self:
        """Keep first occurrence of each author per full_name (removes duplicates from chunked extraction)."""
        if not self.authors:
            return self
        seen: set[str] = set()
        unique: list[PersonIdentity] = []
        for a in self.authors:
            key = (getattr(a, "full_name", None) or "").strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(a)
        object.__setattr__(self, "authors", unique)
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
    DEBIT_NOTE = "Debit Note"
    PRO_FORMA = "Pro Forma"
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

### 👍 Validate Early

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

### 👍 Provide Clear Error Messages

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

### 👍 Handle None Values

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

### 👍 Use Type Guards

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

## Graceful Error Handling

### The Problem with Strict Validators

**Strict validators** that raise `ValueError` on invalid data can cause **complete extraction failure**:

```python
# ❌ Strict validator - causes extraction failure
@field_validator("value")
@classmethod
def validate_positive(cls, v: Any) -> Any:
    """Ensure amount is non-negative."""
    if v < 0:
        raise ValueError(f"Monetary amount must be non-negative, got {v}")
    return v
```

**What happens:**

- LLM extracts: `allowance_total: -258.12` (negative because it's a discount)
- Validator rejects: "Monetary amount must be non-negative"
- **Result:** Entire extraction fails, losing ALL extracted data

### The Solution: Lenient Validators

**Lenient validators** coerce invalid values instead of rejecting them:

```python
# ✅ Lenient validator - coerces instead of rejecting
import logging

logger = logging.getLogger(__name__)

@field_validator("value", mode="before")
@classmethod
def coerce_positive(cls, v: Any) -> Any:
    """
    Coerce negative values to positive (use absolute value).
    
    Allowances and discounts are often represented as negative in accounting,
    but should be stored as positive amounts. The charge_indicator field
    (in AllowanceCharge) indicates direction: false=allowance, true=charge.
    
    This validator is lenient - it coerces instead of rejecting to prevent
    extraction failures due to semantic differences in how amounts are represented.
    """
    if isinstance(v, (int, float)) and v < 0:
        logger.warning(
            f"Negative monetary value {v} coerced to positive {abs(v)}. "
            "Allowances/discounts should be positive amounts."
        )
        return abs(v)
    return v
```

**Benefits:**

- ✅ Extraction succeeds even with "invalid" data
- ✅ Data quality issues are logged for review
- ✅ 99% correct data is preserved instead of lost
- ✅ Semantic differences are handled gracefully

### When to Use Lenient Validators

Use lenient validators for:

1. **Semantic Variations**

   - Negative amounts for discounts/allowances
   - Lowercase currency codes (normalize to uppercase)
   - Different date formats (parse and normalize)

2. **Common LLM Mistakes**

   - Missing spaces in addresses
   - Wrong case in enums
   - Currency symbols instead of codes

3. **Non-Critical Validation**

   - Format preferences (3-letter currency codes)
   - Range constraints (quantity > 0)
   - Pattern matching (email format)

### When to Use Strict Validators

Use strict validators only for:

1. **Critical Data Integrity**

   - Required fields that must be present
   - Type safety (must be a number, not a string)
   - Business rules that cannot be violated

2. **Security Concerns**

   - SQL injection prevention
   - Path traversal prevention
   - XSS prevention

### Lenient Validator Patterns

#### Pattern 1: Coerce Negative to Positive

```python
@field_validator("value", mode="before")
@classmethod
def coerce_positive(cls, v: Any) -> Any:
    """Coerce negative values to positive."""
    if isinstance(v, (int, float)) and v < 0:
        logger.warning(f"Negative value {v} coerced to {abs(v)}")
        return abs(v)
    return v
```

#### Pattern 2: Normalize Case

```python
@field_validator("currency", mode="before")
@classmethod
def normalize_currency(cls, v: Any) -> Any:
    """Normalize currency to uppercase."""
    if v:
        v_upper = str(v).strip().upper()
        if len(v_upper) == 3 and v_upper.isalpha():
            return v_upper
        logger.warning(f"Currency '{v}' normalized to '{v_upper}'")
        return v_upper
    return v
```

#### Pattern 3: Handle Zero Values

```python
@field_validator("quantity", mode="before")
@classmethod
def handle_zero(cls, v: Any) -> Any:
    """Handle zero quantities by setting default."""
    if isinstance(v, (int, float)):
        if v == 0:
            logger.warning("Zero quantity detected, setting to 1 as default")
            return 1.0
        elif v < 0:
            logger.warning(f"Negative quantity {v} coerced to {abs(v)}")
            return abs(v)
    return v
```

#### Pattern 4: Symbol to Code Conversion

```python
@field_validator("currency", mode="before")
@classmethod
def convert_symbol(cls, v: Any) -> Any:
    """Convert currency symbols to ISO codes."""
    symbol_map = {
        "€": "EUR",
        "$": "USD",
        "£": "GBP",
        "¥": "JPY",
    }
    
    if v in symbol_map:
        logger.info(f"Currency symbol '{v}' converted to '{symbol_map[v]}'")
        return symbol_map[v]
    
    return v
```

### Logging Best Practices

Always log data quality issues:

```python
import logging

logger = logging.getLogger(__name__)

@field_validator("value", mode="before")
@classmethod
def coerce_positive(cls, v: Any) -> Any:
    """Coerce with logging."""
    if isinstance(v, (int, float)) and v < 0:
        # Log at WARNING level for data quality issues
        logger.warning(
            f"Data quality issue: Negative value {v} coerced to {abs(v)}. "
            f"Field: {cls.__name__}.value"
        )
        return abs(v)
    return v
```

**Log Levels:**

- `logger.info()` - Normal coercion (e.g., lowercase → uppercase)
- `logger.warning()` - Data quality issues (e.g., negative → positive)
- `logger.error()` - Serious issues that couldn't be fixed

### Complete Example: Lenient MonetaryAmount

```python
import logging
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

class MonetaryAmount(BaseModel):
    """Monetary amount with lenient validation."""
    model_config = ConfigDict(is_entity=False)
    
    value: float = Field(
        ...,
        description="Monetary amount (always positive)",
        examples=[100.00, 1250.50, 89.99]
    )
    
    currency: str | None = Field(
        None,
        description="ISO 4217 currency code (3 uppercase letters)",
        examples=["EUR", "USD", "GBP", "CHF"]
    )
    
    @field_validator("value", mode="before")
    @classmethod
    def coerce_positive(cls, v: Any) -> Any:
        """Coerce negative values to positive."""
        if isinstance(v, (int, float)) and v < 0:
            logger.warning(
                f"Negative monetary value {v} coerced to positive {abs(v)}. "
                "Allowances/discounts should be positive amounts."
            )
            return abs(v)
        return v
    
    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, v: Any) -> Any:
        """Normalize currency to ISO 4217 format."""
        if not v:
            return v
        
        # Symbol to code mapping
        symbol_map = {
            "€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY",
            "₹": "INR", "₽": "RUB", "₩": "KRW", "₪": "ILS",
        }
        
        v_str = str(v).strip()
        
        # Convert symbol to code
        if v_str in symbol_map:
            return symbol_map[v_str]
        
        # Normalize to uppercase
        v_upper = v_str.upper()
        
        # Validate format
        if len(v_upper) == 3 and v_upper.isalpha():
            return v_upper
        
        # Log warning but don't fail
        logger.warning(
            f"Currency '{v}' does not match ISO 4217 format. "
            f"Normalized to '{v_upper}' but may be invalid."
        )
        return v_upper if len(v_upper) == 3 else v_str
```

### Migration Guide: Strict → Lenient

**Before (Strict):**
```python
@field_validator("value")
@classmethod
def validate_positive(cls, v: Any) -> Any:
    if v < 0:
        raise ValueError(f"Must be non-negative, got {v}")
    return v
```

**After (Lenient):**
```python
@field_validator("value", mode="before")
@classmethod
def coerce_positive(cls, v: Any) -> Any:
    if isinstance(v, (int, float)) and v < 0:
        logger.warning(f"Negative value {v} coerced to {abs(v)}")
        return abs(v)
    return v
```

**Changes:**
1. Add `mode="before"` to validator decorator
2. Replace `raise ValueError` with coercion logic
3. Add `logger.warning()` for data quality tracking
4. Add type guard (`isinstance`) for safety
5. Update docstring to explain lenient behavior

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

1. **[Advanced Patterns](advanced-patterns.md)** - Complex validation patterns
2. **[Best Practices](best-practices.md)** - Complete template checklist
3. **[Examples](../../usage/examples/index.md)** - See validators in action