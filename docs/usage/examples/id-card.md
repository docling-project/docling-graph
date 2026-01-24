# ID Card Extraction


## Overview

Extract personal information from ID cards and identity documents using vision-based extraction.

**What You'll Learn:**
- Vision-based extraction (VLM)
- Date field handling
- Address parsing
- Field validators
- Graph ID configuration

**Document Type:** ID Card (Image)  
**Time:** 15 minutes  
**Backend:** VLM (recommended)

---

## Prerequisites

```bash
# Install with all features
uv sync --extra all
```

---

## Template Definition

### Complete Template

**File:** `id_card_template.py`

```python
"""
ID Card extraction template.
Demonstrates date parsing, validators, and graph IDs.
"""

import re
from datetime import date
from typing import List
from pydantic import BaseModel, ConfigDict, Field, field_validator

def edge(label: str, **kwargs):
    """Helper for graph edges."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

# --- Address Component ---

class Address(BaseModel):
    """Physical address."""
    
    model_config = ConfigDict(is_entity=False)
    
    street_address: str | None = Field(
        None,
        description="Street name and number",
        examples=["123 Main Street", "456 Oak Avenue"]
    )
    
    city: str | None = Field(
        None,
        description="City name",
        examples=["New York", "Los Angeles"]
    )
    
    state_or_province: str | None = Field(
        None,
        description="State or province",
        examples=["NY", "California"]
    )
    
    postal_code: str | None = Field(
        None,
        description="Postal or ZIP code",
        examples=["10001", "90210"]
    )
    
    country: str | None = Field(
        None,
        description="Country name",
        examples=["USA", "United States"]
    )

# --- Person Entity ---

class Person(BaseModel):
    """Person entity with unique identification."""
    
    # Graph ID: Unique by name + date of birth
    model_config = ConfigDict(
        graph_id_fields=["given_names", "last_name", "date_of_birth"]
    )
    
    given_names: List[str] | None = Field(
        default=None,
        description="List of given names (first names)",
        examples=[["John"], ["Mary", "Jane"], ["Pierre", "Louis"]]
    )
    
    last_name: str | None = Field(
        None,
        description="Family name (surname)",
        examples=["Smith", "Johnson", "Doe"]
    )
    
    alternate_name: str | None = Field(
        None,
        description="Alternate or maiden name",
        examples=["Doe", "MJ"]
    )
    
    date_of_birth: date | None = Field(
        None,
        description="Date of birth in YYYY-MM-DD format",
        examples=["1990-05-15", "1985-12-01"]
    )
    
    place_of_birth: str | None = Field(
        None,
        description="City and/or country of birth",
        examples=["New York, USA", "Paris, France"]
    )
    
    gender: str | None = Field(
        None,
        description="Gender",
        examples=["M", "F", "Male", "Female"]
    )
    
    # Relationship
    lives_at: Address | None = edge(
        label="LIVES_AT",
        description="Home address"
    )
    
    # --- Validators ---
    
    @field_validator("given_names", mode="before")
    @classmethod
    def ensure_list(cls, v):
        """Ensure given_names is a list."""
        if isinstance(v, str):
            # Handle comma or space separated
            if "," in v:
                return [name.strip() for name in v.split(",")]
            return [v]
        return v
    
    @field_validator("lives_at", mode="before")
    @classmethod
    def parse_address(cls, v):
        """Parse address string into Address object."""
        if v is None or isinstance(v, dict):
            return v
        
        if isinstance(v, str):
            # Simple parsing
            parts = [p.strip() for p in v.split(",")]
            return {
                "street_address": parts[0] if len(parts) > 0 else None,
                "city": parts[1] if len(parts) > 1 else None,
                "country": parts[-1] if len(parts) > 2 else None
            }
        return v

# --- Root Entity: IDCard ---

class IDCard(BaseModel):
    """Identity document."""
    
    # Graph ID: Unique by document number
    model_config = ConfigDict(graph_id_fields=["document_number"])
    
    document_number: str = Field(
        ...,
        description="Unique document identifier",
        examples=["A12345678", "123456789", "AB1234567"]
    )
    
    issuing_country: str | None = Field(
        None,
        description="Country that issued the document",
        examples=["USA", "France", "United Kingdom"]
    )
    
    issue_date: date | None = Field(
        None,
        description="Date document was issued (YYYY-MM-DD)",
        examples=["2023-10-20", "2020-05-15"]
    )
    
    expiry_date: date | None = Field(
        None,
        description="Date document expires (YYYY-MM-DD)",
        examples=["2033-10-19", "2030-05-14"]
    )
    
    # Relationship
    holder: Person = edge(
        label="BELONGS_TO",
        description="Person this ID belongs to"
    )
```

---

## Processing

### Using CLI

```bash
# Process ID card image with VLM
uv run docling-graph convert id_card.jpg \
    --template "id_card_template.IDCard" \
    --backend vlm \
    --processing-mode one-to-one \
    --docling-pipeline vision \
    --output-dir "outputs/id_card"
```

### Using Python API

```python
"""Process ID card."""

from docling_graph import PipelineConfig

config = PipelineConfig(
    source="id_card.jpg",
    template="id_card_template.IDCard",
    backend="vlm",
    inference="local",  # VLM only supports local
    processing_mode="one-to-one",
    docling_config="vision",
    output_dir="outputs/id_card"
)

print("Processing ID card...")
config.run()
print("✓ Complete!")
```

---

## Expected Results

### Graph Structure

```
IDCard (A12345678)
└── BELONGS_TO → Person (John Smith, 1990-05-15)
    └── LIVES_AT → Address (123 Main St, NYC)
```

### Nodes CSV

```csv
id,label,type,document_number,issuing_country,issue_date,expiry_date
id_1,A12345678,IDCard,A12345678,USA,2023-10-20,2033-10-19
person_1,John Smith,Person,,,,
addr_1,123 Main St,Address,,,,
```

### Edges CSV

```csv
source,target,type
id_1,person_1,BELONGS_TO
person_1,addr_1,LIVES_AT
```

---

## Key Features

### 1. Date Parsing

```python
# Pydantic automatically parses dates
date_of_birth: date | None = Field(
    None,
    description="Date in YYYY-MM-DD format"
)

# Accepts: "1990-05-15", "1990/05/15", "05-15-1990"
# Converts to: date(1990, 5, 15)
```

### 2. Graph ID Configuration

```python
# Person uniquely identified by name + DOB
model_config = ConfigDict(
    graph_id_fields=["given_names", "last_name", "date_of_birth"]
)

# Same person in multiple documents = same node
```

### 3. List Handling

```python
# Validator converts string to list
given_names: List[str] = Field(...)

@field_validator("given_names", mode="before")
@classmethod
def ensure_list(cls, v):
    if isinstance(v, str):
        return [v]  # "John" → ["John"]
    return v
```

### 4. Address Parsing

```python
# Validator parses address string
@field_validator("lives_at", mode="before")
@classmethod
def parse_address(cls, v):
    if isinstance(v, str):
        # "123 Main St, NYC, USA" → Address object
        parts = v.split(",")
        return {"street_address": parts[0], ...}
    return v
```

---

## Visualization

```bash
# Interactive visualization
uv run docling-graph inspect outputs/id_card/
```

**Features:**
- View extracted personal information
- See address relationships
- Verify dates are parsed correctly

---

## Customization

### Add More Fields

```python
class IDCard(BaseModel):
    document_number: str
    issuing_country: str | None
    issue_date: date | None
    expiry_date: date | None
    
    # Add document type
    document_type: str | None = Field(
        None,
        description="Type of ID document",
        examples=["Passport", "Driver License", "National ID"]
    )
    
    # Add nationality
    nationality: str | None = Field(
        None,
        description="Holder's nationality",
        examples=["American", "French", "British"]
    )
    
    holder: Person = edge(label="BELONGS_TO")
```

### Add Validation

```python
from pydantic import field_validator
from datetime import date

class IDCard(BaseModel):
    issue_date: date | None
    expiry_date: date | None
    
    @field_validator("expiry_date")
    @classmethod
    def validate_expiry(cls, v, info):
        """Ensure expiry date is after issue date."""
        issue = info.data.get("issue_date")
        if issue and v and v <= issue:
            raise ValueError("Expiry date must be after issue date")
        return v
```

---

## Troubleshooting

### Issue: Dates Not Parsed

**Problem:** Date fields are None or incorrect

**Solution:**
```python
# Make dates optional and add examples
date_of_birth: date | None = Field(
    None,
    description="Date of birth. Parse formats like DD/MM/YYYY, MM-DD-YYYY, YYYY-MM-DD",
    examples=["1990-05-15", "05/15/1990", "15-05-1990"]
)
```

### Issue: Name Parsing

**Problem:** Full name extracted as single string

**Solution:**
```python
# Add validator to split names
@field_validator("given_names", mode="before")
@classmethod
def split_names(cls, v):
    if isinstance(v, str):
        # "John Paul" → ["John", "Paul"]
        return v.split()
    return v
```

### Issue: Address Not Structured

**Problem:** Address extracted as single string

**Solution:**
```python
# Use validator to parse
@field_validator("lives_at", mode="before")
@classmethod
def parse_address(cls, v):
    if isinstance(v, str):
        # Extract postal code
        postal_match = re.search(r'\b(\d{5})\b', v)
        postal = postal_match.group(1) if postal_match else None
        
        return {
            "street_address": v.split(",")[0] if "," in v else v,
            "postal_code": postal
        }
    return v
```

---

## Best Practices

### 1. Use VLM for Images

```bash
# ✅ Good - VLM for image documents
uv run docling-graph convert id_card.jpg \
    --backend vlm

# ❌ Avoid - LLM for images (slower, less accurate)
uv run docling-graph convert id_card.jpg \
    --backend llm
```

### 2. Make Fields Optional

```python
# ✅ Good - Optional fields for incomplete data
class Person(BaseModel):
    given_names: List[str] | None = Field(default=None)
    last_name: str | None = Field(default=None)
    date_of_birth: date | None = Field(default=None)

# ❌ Avoid - Required fields that might be missing
class Person(BaseModel):
    given_names: List[str]  # Fails if not found
    last_name: str
```

### 3. Provide Date Format Examples

```python
# ✅ Good - Multiple format examples
date_of_birth: date | None = Field(
    None,
    description="Date of birth in various formats",
    examples=["1990-05-15", "05/15/1990", "15-05-1990"]
)
```

---

## Next Steps

1. **[Insurance Policy →](insurance-policy.md)** - Financial documents
2. **[Validation Guide →](../../fundamentals/schema-definition/validation.md)** - Advanced validators
3. **[VLM Backend →](../../fundamentals/extraction-process/extraction-backends.md)** - Vision models

---

## Quick Reference

### Process ID Card

```bash
# VLM (recommended)
uv run docling-graph convert id_card.jpg \
    -t "id_card_template.IDCard" \
    --backend vlm

# LLM (alternative)
uv run docling-graph convert id_card.pdf \
    -t "id_card_template.IDCard" \
    --backend llm
```

### View Results

```bash
uv run docling-graph inspect outputs/id_card/
cat outputs/id_card/nodes.csv
```

### Template Location

```
docs/examples/templates/id_card.py
```