# Configuration Basics

**Navigation:** [← Pipeline Configuration](index.md) | [Next: Backend Selection →](backend-selection.md)

---

## Overview

The `PipelineConfig` class is the foundation of Docling Graph configuration. It provides type-safe, validated configuration for all pipeline operations. This guide covers the fundamentals of creating and using pipeline configurations.

**In this guide:**
- PipelineConfig structure
- Required vs optional settings
- Creating configurations
- Configuration validation
- Common patterns

---

## PipelineConfig Structure

### Core Components

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    # Required (at runtime)
    source="document.pdf",
    template="my_templates.MyTemplate",
    
    # Backend settings
    backend="llm",              # "llm" or "vlm"
    inference="local",          # "local" or "remote"
    
    # Processing settings
    processing_mode="many-to-one",  # "one-to-one" or "many-to-one"
    docling_config="ocr",           # "ocr" or "vision"
    use_chunking=True,
    llm_consolidation=False,
    
    # Export settings
    export_format="csv",        # "csv" or "cypher"
    output_dir="outputs"
)
```

### Configuration Categories

| Category | Settings | Purpose |
|:---------|:---------|:--------|
| **Source** | `source`, `template` | What to extract |
| **Backend** | `backend`, `inference`, `models` | How to extract |
| **Processing** | `processing_mode`, `docling_config`, `use_chunking` | How to process |
| **Export** | `export_format`, `output_dir`, `export_*` | What to output |
| **Advanced** | `max_batch_size`, `reverse_edges`, `chunker_config` | Optimization |

---

## Required Settings

### 1. Source Document

```python
# File path (string or Path)
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice"
)

# Also accepts Path objects
from pathlib import Path
config = PipelineConfig(
    source=Path("documents/invoice.pdf"),
    template="my_templates.Invoice"
)
```

**Supported formats:**
- PDF documents
- Images (PNG, JPG, JPEG)
- DOCX files

### 2. Pydantic Template

```python
# Dotted path string (recommended)
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice"
)

# Direct class reference (alternative)
from my_templates import Invoice
config = PipelineConfig(
    source="document.pdf",
    template=Invoice
)
```

**Template must:**
- Be a valid Pydantic BaseModel
- Have proper `model_config` (graph_id_fields or is_entity)
- Include the `edge()` helper function
- Follow template best practices

---

## Backend Settings

### Backend Type

```python
# LLM backend (default) - for text extraction
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    backend="llm"
)

# VLM backend - for vision-based extraction
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    backend="vlm"
)
```

**See:** [Backend Selection](backend-selection.md) for detailed comparison.

### Inference Location

```python
# Local inference (default) - uses local GPU/CPU
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    inference="local"
)

# Remote inference - uses API providers
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    inference="remote"
)
```

**See:** [Model Configuration](model-configuration.md) for setup details.

### Model Overrides

```python
# Override default model
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    inference="remote",
    model_override="gpt-4-turbo",
    provider_override="openai"
)
```

---

## Processing Settings

### Processing Mode

```python
# Many-to-one (default) - whole document as single entity
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    processing_mode="many-to-one"
)

# One-to-one - process each page separately
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    processing_mode="one-to-one"
)
```

**See:** [Processing Modes](processing-modes.md) for when to use each.

### Docling Configuration

```python
# OCR pipeline (default) - traditional OCR
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    docling_config="ocr"
)

# Vision pipeline - VLM-based conversion
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    docling_config="vision"
)
```

**See:** [Docling Settings](docling-settings.md) for details.

### Chunking

```python
# With chunking (default) - splits large documents
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    use_chunking=True
)

# Without chunking - processes entire document
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    use_chunking=False
)
```

### LLM Consolidation

```python
# Without consolidation (default)
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    llm_consolidation=False
)

# With consolidation - merges results using LLM
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    llm_consolidation=True
)
```

---

## Export Settings

### Export Format

```python
# CSV format (default)
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_format="csv"
)

# Cypher format - for Neo4j import
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_format="cypher"
)
```

**See:** [Export Configuration](export-configuration.md) for format details.

### Output Directory

```python
# Default output directory
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    output_dir="outputs"
)

# Custom output directory
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    output_dir="my_results/invoice_001"
)
```

### Docling Exports

```python
# Control Docling document exports
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_docling=True,           # Export Docling document
    export_docling_json=True,      # Export as JSON
    export_markdown=True,          # Export as markdown
    export_per_page_markdown=False # Export per-page markdown
)
```

---

## Creating Configurations

### Method 1: Direct Instantiation

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    backend="llm",
    inference="remote"
)

# Run the pipeline
config.run()
```

### Method 2: Programmatic Building

```python
from docling_graph import PipelineConfig

# Start with defaults
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice"
)

# Modify as needed
if use_gpu:
    config.inference = "local"
    config.model_override = "ibm-granite/granite-4.0-1b"
else:
    config.inference = "remote"
    config.model_override = "gpt-4-turbo"

# Run
config.run()
```

### Method 3: From Dictionary

```python
from docling_graph import PipelineConfig

config_dict = {
    "source": "document.pdf",
    "template": "my_templates.Invoice",
    "backend": "llm",
    "inference": "remote"
}

config = PipelineConfig(**config_dict)
config.run()
```

### Method 4: Using run_pipeline

```python
from docling_graph import run_pipeline

# Pass config dict directly
run_pipeline({
    "source": "document.pdf",
    "template": "my_templates.Invoice",
    "backend": "llm",
    "inference": "remote"
})
```

---

## Configuration Validation

### Automatic Validation

PipelineConfig validates settings automatically:

```python
from docling_graph import PipelineConfig

# This raises ValueError
try:
    config = PipelineConfig(
        source="document.pdf",
        template="my_templates.Invoice",
        backend="vlm",
        inference="remote"  # ❌ VLM doesn't support remote
    )
except ValueError as e:
    print(f"Error: {e}")
    # Output: VLM backend currently only supports local inference
```

### Common Validation Errors

#### Error 1: VLM Remote Inference

```python
# ❌ Wrong
config = PipelineConfig(
    backend="vlm",
    inference="remote"
)

# ✅ Correct
config = PipelineConfig(
    backend="vlm",
    inference="local"
)
```

#### Error 2: Invalid Backend

```python
# ❌ Wrong
config = PipelineConfig(
    backend="gpt"  # Not a valid backend
)

# ✅ Correct
config = PipelineConfig(
    backend="llm"  # or "vlm"
)
```

#### Error 3: Invalid Processing Mode

```python
# ❌ Wrong
config = PipelineConfig(
    processing_mode="batch"  # Not valid
)

# ✅ Correct
config = PipelineConfig(
    processing_mode="many-to-one"  # or "one-to-one"
)
```

---

## Default Values Reference

### All Defaults

```python
PipelineConfig(
    # Required (no defaults)
    source="",
    template="",
    
    # Backend defaults
    backend="llm",
    inference="local",
    model_override=None,
    provider_override=None,
    
    # Processing defaults
    processing_mode="many-to-one",
    docling_config="ocr",
    use_chunking=True,
    llm_consolidation=False,
    max_batch_size=1,
    
    # Export defaults
    export_format="csv",
    export_docling=True,
    export_docling_json=True,
    export_markdown=True,
    export_per_page_markdown=False,
    
    # Graph defaults
    reverse_edges=False,
    
    # Output defaults
    output_dir="outputs"
)
```

### Model Defaults

```python
# LLM Local
default_model="ibm-granite/granite-4.0-1b"
provider="vllm"

# LLM Remote
default_model="mistral-small-latest"
provider="mistral"

# VLM Local
default_model="numind/NuExtract-2.0-8B"
provider="docling"
```

---

## Common Configuration Patterns

### Pattern 1: Quick Local Test

```python
# Minimal config for quick testing
config = PipelineConfig(
    source="test.pdf",
    template="my_templates.Invoice"
)
config.run()
```

### Pattern 2: Production Remote

```python
# Production config with remote API
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    inference="remote",
    model_override="gpt-4-turbo",
    provider_override="openai",
    output_dir="production_outputs"
)
config.run()
```

### Pattern 3: GPU-Accelerated Local

```python
# Local GPU extraction
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    inference="local",
    model_override="ibm-granite/granite-4.0-1b",
    provider_override="vllm"
)
config.run()
```

### Pattern 4: Vision-Based Extraction

```python
# VLM extraction for complex layouts
config = PipelineConfig(
    source="complex_layout.pdf",
    template="my_templates.Invoice",
    backend="vlm",
    inference="local",
    docling_config="vision"
)
config.run()
```

### Pattern 5: Page-by-Page Processing

```python
# Process each page separately
config = PipelineConfig(
    source="multi_page.pdf",
    template="my_templates.Invoice",
    processing_mode="one-to-one",
    export_per_page_markdown=True
)
config.run()
```

---

## Configuration Inspection

### View Configuration

```python
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    inference="remote"
)

# Print configuration
print(f"Backend: {config.backend}")
print(f"Inference: {config.inference}")
print(f"Processing mode: {config.processing_mode}")
print(f"Output dir: {config.output_dir}")
```

### Convert to Dictionary

```python
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice"
)

# Get as dictionary
config_dict = config.to_dict()
print(config_dict)
```

### Serialize to JSON

```python
import json

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice"
)

# Serialize
json_str = config.model_dump_json(indent=2)
print(json_str)
```

---

## Best Practices

### 1. Use Type Hints

```python
from docling_graph import PipelineConfig

# ✅ Good - Type hints help catch errors
config: PipelineConfig = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice"
)
```

### 2. Validate Early

```python
# ✅ Good - Create and validate before processing
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    backend="vlm",
    inference="local"
)

# Verify settings
assert config.backend == "vlm"
assert config.inference == "local"

# Then run
config.run()
```

### 3. Use Defaults

```python
# ✅ Good - Rely on sensible defaults
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice"
)

# ❌ Bad - Over-specify defaults
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    backend="llm",  # Already default
    inference="local",  # Already default
    processing_mode="many-to-one",  # Already default
    use_chunking=True,  # Already default
    # ... etc
)
```

### 4. Document Custom Configs

```python
# ✅ Good - Document why you override defaults
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    # Use remote API for better accuracy on complex documents
    inference="remote",
    model_override="gpt-4-turbo",
    # Disable chunking for short documents
    use_chunking=False
)
```

---

## Next Steps

Now that you understand configuration basics:

1. **[Backend Selection →](backend-selection.md)** - Choose between LLM and VLM
2. **[Model Configuration](model-configuration.md)** - Configure models
3. **[Configuration Examples](configuration-examples.md)** - See complete scenarios

---

## Quick Reference

### Minimal Configuration

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.MyTemplate"
)
config.run()
```

### Common Overrides

```python
# Remote API
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.MyTemplate",
    inference="remote",
    model_override="gpt-4-turbo"
)

# VLM extraction
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.MyTemplate",
    backend="vlm"
)

# Page-by-page
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.MyTemplate",
    processing_mode="one-to-one"
)
```

---

**Navigation:** [← Pipeline Configuration](index.md) | [Next: Backend Selection →](backend-selection.md)