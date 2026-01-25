# Rheology Research Extraction


## Overview

Extract complex research data from scientific papers including experiments, measurements, materials, and results.

**Document Type:** Rheology Research (PDF)  
**Time:** 30 minutes  
**Backend:** LLM with chunking

---

## Prerequisites

```bash
# Install with remote API support
uv sync --extra remote

# Set API key
export MISTRAL_API_KEY="your-key"
```

---

## Template Overview

The rheology research template (`rheology_research.py`) includes:

- **Measurements** - Flexible value/unit pairs
- **Materials** - Granular material properties
- **Geometry** - Experimental setup
- **Vibration** - Vibration parameters
- **Simulation** - DEM simulation details
- **Results** - Rheological measurements
- **Experiments** - Complete experiment instances
- **Research** - Root document model

### Key Components

```python
# 1. Measurement Model
class Measurement(BaseModel):
    """Flexible measurement with value and unit."""
    name: str
    numeric_value: float | None = None
    text_value: str | None = None
    unit: str | None = None

# 2. Enum Types
class GeometryType(str, Enum):
    VANE_RHEOMETER = "Vane Rheometer"
    DOUBLE_PLATE = "Double Plate"
    CYLINDRICAL_CONTAINER = "Cylindrical Container"

# 3. Experiment Entity
class Experiment(BaseModel):
    experiment_id: str
    objective: str
    granular_material: GranularMaterial = edge("USES_MATERIAL")
    vibration_conditions: VibrationConditions = edge("HAS_VIBRATION")
    rheological_results: List[RheologicalResult] = edge("HAS_RESULT")

# 4. Root Model
class Research(BaseModel):
    title: str
    authors: List[str]
    experiments: List[Experiment] = edge("HAS_EXPERIMENT")
```

---

## Processing

### Using CLI

```bash
# Process rheology research with chunking
uv run docling-graph convert research.pdf \
    --template "docs.examples.templates.rheology_research.Research" \
    --backend llm \
    --inference remote \
    --provider mistral \
    --model mistral-large-latest \
    --processing-mode many-to-one \
    --use-chunking \
    --llm-consolidation \
    --docling-pipeline vision \
    --output-dir "outputs/research"
```

### Using Python API

```python
"""Process rheology research."""

import os
from docling_graph import run_pipeline, PipelineConfig

os.environ["MISTRAL_API_KEY"] = "your-key"

config = PipelineConfig(
    source="research.pdf",
    template="docs.examples.templates.rheology_research.Research",
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-large-latest",
    processing_mode="many-to-one",
    use_chunking=True,
    llm_consolidation=True,
    docling_config="vision",  # Better for complex layouts
    output_dir="outputs/research"
)

print("Processing rheology research (may take several minutes)...")
run_pipeline(config)
print("✅ Complete!")
```

---

## Expected Results

### Graph Structure

```
Research (Title)
├── HAS_EXPERIMENT → Experiment 1
│   ├── USES_MATERIAL → GranularMaterial
│   │   └── properties: [Measurement, Measurement]
│   ├── HAS_GEOMETRY → SystemGeometry
│   │   └── dimensions: [Measurement, Measurement]
│   ├── HAS_VIBRATION → VibrationConditions
│   │   ├── amplitude: Measurement
│   │   ├── frequency: Measurement
│   │   └── confining_pressure: Measurement
│   ├── HAS_SIMULATION → SimulationSetup
│   │   └── parameters: [Measurement, Measurement]
│   └── HAS_RESULT → RheologicalResult
│       └── measurement: Measurement
└── HAS_EXPERIMENT → Experiment 2
    └── ...
```

### Statistics

```json
{
  "node_count": 45,
  "edge_count": 38,
  "density": 0.019,
  "node_types": {
    "Research": 1,
    "Experiment": 3,
    "GranularMaterial": 3,
    "SystemGeometry": 3,
    "VibrationConditions": 3,
    "RheologicalResult": 12,
    "Measurement": 20
  }
}
```

---

## Key Features

### 1. Enum Normalization

```python
class GeometryType(str, Enum):
    VANE_RHEOMETER = "Vane Rheometer"
    CYLINDRICAL_CONTAINER = "Cylindrical Container"

# Validator accepts multiple formats
@field_validator("geometry_type", mode="before")
@classmethod
def normalize_enum(cls, v):
    # Accepts: "Vane Rheometer", "vane_rheometer", "VANE_RHEOMETER"
    return _normalize_enum(GeometryType, v)
```

### 2. Measurement Parsing

```python
# Parses strings like "1.6 mPa.s", "2 mm", "80-90 °C"
def _parse_measurement_string(s: str):
    # Single value: "1.6 mPa.s" → {numeric_value: 1.6, unit: "mPa.s"}
    # Range: "80-90 °C" → {numeric_value_min: 80, numeric_value_max: 90, unit: "°C"}
    ...
```

### 3. Flexible Measurements

```python
class Measurement(BaseModel):
    name: str
    numeric_value: float | None = None  # Single value
    numeric_value_min: float | None = None  # Range min
    numeric_value_max: float | None = None  # Range max
    text_value: str | None = None  # Qualitative
    unit: str | None = None
```

### 4. Nested Relationships

```python
class Experiment(BaseModel):
    # Direct edges
    granular_material: GranularMaterial = edge("USES_MATERIAL")
    
    # Nested properties (not separate nodes)
    key_findings: List[str] = Field(default_factory=list)
```

---

## Configuration Tips

### For Long Documents

```bash
# Enable chunking and consolidation
uv run docling-graph convert research.pdf \
    --template "templates.Research" \
    --use-chunking \
    --llm-consolidation \
    --processing-mode many-to-one
```

### For Complex Layouts

```bash
# Use vision pipeline for better table/figure handling
uv run docling-graph convert research.pdf \
    --template "templates.Research" \
    --docling-pipeline vision
```

### For Cost Optimization

```bash
# Use smaller model without consolidation
uv run docling-graph convert research.pdf \
    --template "templates.Research" \
    --model mistral-small-latest \
    --no-llm-consolidation
```

---

## Customization

### Simplify for Your Domain

```python
"""Simplified research template."""

from pydantic import BaseModel, Field
from typing import List

def edge(label: str, **kwargs):
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

class Measurement(BaseModel):
    """Simple measurement."""
    name: str
    value: str  # Keep as string for simplicity
    unit: str | None = None

class Experiment(BaseModel):
    """Simplified experiment."""
    title: str
    objective: str
    methods: str
    results: str
    measurements: List[Measurement] = Field(default_factory=list)

class Research(BaseModel):
    """Simplified rheology research."""
    title: str
    authors: List[str]
    abstract: str
    experiments: List[Experiment] = edge("HAS_EXPERIMENT")
```

---

## Troubleshooting

### 🐛 Extraction Takes Too Long

**Solution:**
```bash
# Disable consolidation for faster processing
uv run docling-graph convert research.pdf \
    --template "templates.Research" \
    --no-llm-consolidation

# Or use smaller model
--model mistral-small-latest
```

### 🐛 Missing Measurements

**Solution:**
```python
# Make measurements optional
measurements: List[Measurement] = Field(
    default_factory=list,
    description="List of measurements (optional)"
)
```

### 🐛 Enum Validation Errors

**Solution:**
```python
# Add OTHER option to enums
class GeometryType(str, Enum):
    VANE_RHEOMETER = "Vane Rheometer"
    OTHER = "Other"  # Fallback

# Or make enum optional
geometry_type: GeometryType | None = Field(default=None)
```

---

## Best Practices

### 👍 Start Simple, Add Complexity

```python
# Phase 1: Basic structure
class Research(BaseModel):
    title: str
    authors: List[str]
    abstract: str

# Phase 2: Add experiments
class Research(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    experiments: List[Experiment]

# Phase 3: Add measurements, validations, etc.
```

### 👍 Use Appropriate Chunking

```python
# For papers > 10 pages
config = PipelineConfig(
    source="long_paper.pdf",
    template="templates.Research",
    use_chunking=True,  # Essential
    llm_consolidation=True  # Better accuracy
)
```

### 👍 Provide Clear Examples

```python
# ✅ Good - Domain-specific examples
viscosity: Measurement = Field(
    description="Effective viscosity measurement",
    examples=[
        {"name": "Effective Viscosity", "numeric_value": 1.6, "unit": "mPa.s"}
    ]
)
```

---

## Next Steps

1. **[ID Card →](id-card.md)** - Vision-based extraction
2. **[Advanced Patterns →](../../fundamentals/schema-definition/advanced-patterns.md)** - Complex templates
3. **[Performance Tuning →](../advanced/performance-tuning.md)** - Optimization