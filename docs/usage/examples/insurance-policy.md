# Insurance Policy Extraction


## Overview

Extract structured information from insurance policy documents including coverage details, terms, and relationships.

**Document Type:** Insurance Policy (PDF)   
**Time:** 20 minutes  
**Backend:** LLM (recommended)

---

## Prerequisites

```bash
# Install
pip install docling-graph

# For remote API (recommended for complex documents)
export MISTRAL_API_KEY="your_key_here"
```

---

## Template Definition

### Complete Template

**File:** `insurance_template.py`

```python
"""
Insurance policy extraction template.
Demonstrates complex relationships and nested structures.
"""

from datetime import date
from decimal import Decimal
from typing import List
from pydantic import BaseModel, ConfigDict, Field

def edge(label: str, **kwargs):
    """Helper for graph edges."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

# --- Components (is_entity=False) ---

class Address(BaseModel):
    """Physical address."""
    
    model_config = ConfigDict(is_entity=False)
    
    street: str | None = Field(None, description="Street address")
    city: str | None = Field(None, description="City")
    state: str | None = Field(None, description="State or province")
    postal_code: str | None = Field(None, description="Postal code")
    country: str | None = Field(None, description="Country")

class MonetaryAmount(BaseModel):
    """Money value with currency."""
    
    model_config = ConfigDict(is_entity=False)
    
    amount: Decimal = Field(..., description="Numeric amount")
    currency: str = Field(default="USD", description="Currency code")

class DateRange(BaseModel):
    """Date range for coverage periods."""
    
    model_config = ConfigDict(is_entity=False)
    
    start_date: date = Field(..., description="Start date")
    end_date: date = Field(..., description="End date")

# --- Entities ---

class Person(BaseModel):
    """Person entity."""
    
    model_config = ConfigDict(graph_id_fields=["full_name", "date_of_birth"])
    
    full_name: str = Field(..., description="Full legal name")
    date_of_birth: date | None = Field(None, description="Date of birth")
    email: str | None = Field(None, description="Email address")
    phone: str | None = Field(None, description="Phone number")
    
    # Relationship
    address: Address | None = edge(
        label="LIVES_AT",
        description="Residential address"
    )

class Organization(BaseModel):
    """Insurance company or provider."""
    
    model_config = ConfigDict(graph_id_fields=["name"])
    
    name: str = Field(..., description="Organization name")
    registration_number: str | None = Field(None, description="Business registration")
    phone: str | None = Field(None, description="Contact phone")
    email: str | None = Field(None, description="Contact email")
    website: str | None = Field(None, description="Website URL")
    
    # Relationship
    headquarters: Address | None = edge(
        label="LOCATED_AT",
        description="Main office address"
    )

class Coverage(BaseModel):
    """Insurance coverage details."""
    
    model_config = ConfigDict(graph_id_fields=["coverage_type", "policy_number"])
    
    coverage_type: str = Field(
        ...,
        description="Type of coverage",
        examples=["Liability", "Collision", "Comprehensive", "Medical"]
    )
    
    policy_number: str = Field(..., description="Associated policy number")
    
    coverage_limit: MonetaryAmount | None = Field(
        None,
        description="Maximum coverage amount"
    )
    
    deductible: MonetaryAmount | None = Field(
        None,
        description="Deductible amount"
    )
    
    premium: MonetaryAmount | None = Field(
        None,
        description="Premium cost"
    )
    
    description: str | None = Field(
        None,
        description="Coverage description"
    )

class PolicyTerm(BaseModel):
    """Policy term or condition."""
    
    model_config = ConfigDict(is_entity=True)
    
    term_type: str = Field(
        ...,
        description="Type of term",
        examples=["Exclusion", "Condition", "Limitation", "Requirement"]
    )
    
    description: str = Field(..., description="Term description")
    
    applies_to: str | None = Field(
        None,
        description="What this term applies to"
    )

# --- Root Entity: InsurancePolicy ---

class InsurancePolicy(BaseModel):
    """Complete insurance policy document."""
    
    model_config = ConfigDict(graph_id_fields=["policy_number"])
    
    policy_number: str = Field(
        ...,
        description="Unique policy identifier",
        examples=["POL-2024-001234", "AUTO-12345"]
    )
    
    policy_type: str = Field(
        ...,
        description="Type of insurance policy",
        examples=["Auto", "Home", "Life", "Health", "Business"]
    )
    
    status: str | None = Field(
        None,
        description="Policy status",
        examples=["Active", "Pending", "Expired", "Cancelled"]
    )
    
    effective_period: DateRange = Field(
        ...,
        description="Policy effective dates"
    )
    
    total_premium: MonetaryAmount | None = Field(
        None,
        description="Total policy premium"
    )
    
    payment_frequency: str | None = Field(
        None,
        description="Payment schedule",
        examples=["Monthly", "Quarterly", "Annually"]
    )
    
    # Relationships
    policyholder: Person = edge(
        label="HELD_BY",
        description="Primary policyholder"
    )
    
    insurer: Organization = edge(
        label="ISSUED_BY",
        description="Insurance company"
    )
    
    coverages: List[Coverage] = edge(
        label="INCLUDES_COVERAGE",
        description="Coverage items"
    )
    
    beneficiaries: List[Person] | None = edge(
        label="BENEFITS",
        description="Policy beneficiaries",
        default=None
    )
    
    terms: List[PolicyTerm] | None = edge(
        label="HAS_TERM",
        description="Policy terms and conditions",
        default=None
    )
```

---

## Processing

### Using CLI

```bash
# Process with remote LLM (best for complex documents)
uv run docling-graph convert insurance_policy.pdf \
    --template "insurance_template.InsurancePolicy" \
    --backend llm \
    --inference remote \
    --provider mistral \
    --model mistral-small-latest \
    --processing-mode many-to-one \
    --output-dir "outputs/insurance"
```

### Using Python API

```python
"""Process insurance policy."""

from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="insurance_policy.pdf",
    template="insurance_template.InsurancePolicy",
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    model_override="mistral-small-latest",
    provider_override="mistral"
)

print("Processing insurance policy...")
run_pipeline(config)
print("✅ Complete!")
```

---

## Expected Results

### Graph Structure

```
InsurancePolicy (POL-2024-001234)
├── HELD_BY → Person (John Smith)
│   └── LIVES_AT → Address (123 Main St)
├── ISSUED_BY → Organization (ABC Insurance)
│   └── LOCATED_AT → Address (456 Corp Plaza)
├── INCLUDES_COVERAGE → Coverage (Liability)
├── INCLUDES_COVERAGE → Coverage (Collision)
├── INCLUDES_COVERAGE → Coverage (Comprehensive)
├── BENEFITS → Person (Jane Smith)
└── HAS_TERM → PolicyTerm (Exclusion)
```

### Nodes CSV

```csv
id,label,type,policy_number,policy_type,status
policy_1,POL-2024-001234,InsurancePolicy,POL-2024-001234,Auto,Active
person_1,John Smith,Person,,,
person_2,Jane Smith,Person,,,
org_1,ABC Insurance,Organization,,,
cov_1,Liability,Coverage,,,
cov_2,Collision,Coverage,,,
term_1,Exclusion,PolicyTerm,,,
```

### Edges CSV

```csv
source,target,type
policy_1,person_1,HELD_BY
policy_1,org_1,ISSUED_BY
policy_1,cov_1,INCLUDES_COVERAGE
policy_1,cov_2,INCLUDES_COVERAGE
policy_1,person_2,BENEFITS
policy_1,term_1,HAS_TERM
person_1,addr_1,LIVES_AT
org_1,addr_2,LOCATED_AT
```

---

## Key Features

### 1. Complex Nested Structures

```python
# Policy contains multiple coverages
coverages: List[Coverage] = edge(
    label="INCLUDES_COVERAGE",
    description="Coverage items"
)

# Each coverage has its own details
class Coverage(BaseModel):
    coverage_type: str
    coverage_limit: MonetaryAmount
    deductible: MonetaryAmount
```

### 2. Financial Data Handling

```python
# Use Decimal for precise amounts
class MonetaryAmount(BaseModel):
    amount: Decimal  # Not float!
    currency: str = "USD"

# In policy
total_premium: MonetaryAmount = Field(
    ...,
    description="Total premium with currency"
)
```

### 3. Date Ranges

```python
# Structured date range
class DateRange(BaseModel):
    start_date: date
    end_date: date

# In policy
effective_period: DateRange = Field(
    ...,
    description="Coverage period"
)
```

### 4. Multiple Relationships

```python
# One-to-one
policyholder: Person = edge(label="HELD_BY")

# One-to-many
coverages: List[Coverage] = edge(label="INCLUDES_COVERAGE")
beneficiaries: List[Person] = edge(label="BENEFITS")
```

---

## Visualization

```bash
# Interactive visualization
uv run docling-graph inspect outputs/insurance/
```

**Features:**
- View policy structure
- Explore coverage relationships
- See beneficiary connections
- Review terms and conditions

---

## Customization

### Add Vehicle Information

```python
class Vehicle(BaseModel):
    """Vehicle covered by policy."""
    
    model_config = ConfigDict(graph_id_fields=["vin"])
    
    vin: str = Field(..., description="Vehicle identification number")
    make: str | None = Field(None, description="Manufacturer")
    model: str | None = Field(None, description="Model name")
    year: int | None = Field(None, description="Model year")
    
class InsurancePolicy(BaseModel):
    # ... existing fields ...
    
    # Add vehicle relationship
    insured_vehicles: List[Vehicle] | None = edge(
        label="COVERS_VEHICLE",
        description="Vehicles covered by this policy",
        default=None
    )
```

### Add Claim History

```python
class Claim(BaseModel):
    """Insurance claim."""
    
    model_config = ConfigDict(graph_id_fields=["claim_number"])
    
    claim_number: str = Field(..., description="Claim ID")
    claim_date: date = Field(..., description="Date filed")
    claim_amount: MonetaryAmount = Field(..., description="Claim amount")
    status: str = Field(..., description="Claim status")
    description: str | None = Field(None, description="Claim details")

class InsurancePolicy(BaseModel):
    # ... existing fields ...
    
    # Add claims relationship
    claims: List[Claim] | None = edge(
        label="HAS_CLAIM",
        description="Claims filed under this policy",
        default=None
    )
```

---

## Troubleshooting

### 🐛 Coverage Not Extracted

Coverage list is empty

**Solution:**
```python
# Make coverages optional and add clear examples
coverages: List[Coverage] | None = edge(
    label="INCLUDES_COVERAGE",
    description="List of coverage types. Extract ALL coverages mentioned: Liability, Collision, Comprehensive, Medical, etc.",
    examples=[
        [
            {"coverage_type": "Liability", "coverage_limit": {"amount": 100000, "currency": "USD"}},
            {"coverage_type": "Collision", "deductible": {"amount": 500, "currency": "USD"}}
        ]
    ],
    default=None
)
```

### 🐛 Amounts Not Parsed

MonetaryAmount fields are None

**Solution:**
```python
# Add validator to parse currency strings
from pydantic import field_validator
import re

class MonetaryAmount(BaseModel):
    amount: Decimal
    currency: str = "USD"
    
    @field_validator("amount", mode="before")
    @classmethod
    def parse_amount(cls, v):
        if isinstance(v, str):
            # Remove currency symbols and commas
            v = re.sub(r'[$,]', '', v)
            return Decimal(v)
        return v
```

### 🐛 Dates Not Recognized

Date fields are None

**Solution:**
```python
# Add multiple date format examples
effective_period: DateRange = Field(
    ...,
    description="Policy effective dates. Parse formats like MM/DD/YYYY, YYYY-MM-DD, Month DD, YYYY",
    examples=[
        {"start_date": "2024-01-01", "end_date": "2025-01-01"},
        {"start_date": "01/01/2024", "end_date": "01/01/2025"}
    ]
)
```

---

## Best Practices

### 👍 Use Remote API for Complex Documents

```bash
# ✅ Good - Remote API for multi-page policies
uv run docling-graph convert policy.pdf \
    --backend llm \
    --inference remote

# ⚠️ Caution - Local models may struggle with complexity
uv run docling-graph convert policy.pdf \
    --backend llm \
    --inference local
```

### 👍 Use Decimal for Money

```python
# ✅ Good - Decimal for financial precision
from decimal import Decimal

class MonetaryAmount(BaseModel):
    amount: Decimal  # Exact precision

# ❌ Avoid - Float for money (rounding errors)
class MonetaryAmount(BaseModel):
    amount: float  # 0.1 + 0.2 = 0.30000000000000004
```

### 👍 Make Lists Optional

```python
# ✅ Good - Optional lists with defaults
beneficiaries: List[Person] | None = edge(
    label="BENEFITS",
    default=None
)

# ❌ Avoid - Required lists (fails if empty)
beneficiaries: List[Person] = edge(label="BENEFITS")
```

### 👍 Provide Clear Examples

```python
# ✅ Good - Detailed examples
coverage_type: str = Field(
    ...,
    description="Type of coverage",
    examples=[
        "Bodily Injury Liability",
        "Property Damage Liability",
        "Collision",
        "Comprehensive",
        "Medical Payments",
        "Uninsured Motorist"
    ]
)
```

---

## Advanced: Multi-Document Processing

Process multiple policies:

```bash
# Process all policies in directory
for policy in policies/*.pdf; do
    uv run docling-graph convert "$policy" \
        -t "insurance_template.InsurancePolicy" \
        --backend llm \
        --inference remote \
        --output-dir "outputs/$(basename "$policy" .pdf)"
done
```

Or use Python:

```python
"""Process multiple policies."""

from pathlib import Path
from docling_graph import run_pipeline, PipelineConfig

policies_dir = Path("policies")

for policy_file in policies_dir.glob("*.pdf"):
    print(f"Processing {policy_file.name}...")
    
    config = PipelineConfig(
        source=str(policy_file),
        template="insurance_template.InsurancePolicy",
        backend="llm",
        inference="remote"
    )
    
    run_pipeline(config)
    print(f"✅ {policy_file.name} complete!")
```

---

## Next Steps

1. **[Examples Index](index.md)** - See all examples
2. **[Graph Analysis →](../../fundamentals/graph-management/graph-analysis.md)** - Analyze extracted data
3. **[Neo4j Integration →](../../fundamentals/graph-management/neo4j-integration.md)** - Load into database