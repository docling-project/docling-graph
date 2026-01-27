# PipelineConfig


## Overview

`PipelineConfig` is a **type-safe configuration class** built with Pydantic that provides validation, defaults, and IDE autocomplete for pipeline configuration.

**Key Features:**
- Type validation
- Default values
- IDE autocomplete
- Validation errors
- Convenience methods

---

## Basic Usage

```python
from docling_graph import run_pipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="remote"
)

# Run pipeline
run_pipeline(config)
```

---

## Constructor Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `str | Path` | Path to source document |
| `template` | `str | Type[BaseModel]` | Pydantic template (dotted path or class) |

### Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `Literal["llm", "vlm"]` | `"llm"` | Backend type |
| `inference` | `Literal["local", "remote"]` | `"local"` | Inference mode |
| `processing_mode` | `Literal["one-to-one", "many-to-one"]` | `"many-to-one"` | Processing strategy |

### Docling Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `docling_config` | `Literal["ocr", "vision"]` | `"ocr"` | Docling pipeline type |

### Model Overrides

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_override` | `str | None` | `None` | Override model name |
| `provider_override` | `str | None` | `None` | Override provider name |

### Extraction Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_chunking` | `bool` | `True` | Enable document chunking |
| `llm_consolidation` | `bool` | `False` | Enable LLM consolidation |
| `max_batch_size` | `int` | `1` | Maximum batch size |

### Debug Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `debug` | `bool` | `False` | Enable debug mode to save all intermediate extraction artifacts |

### Export Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dump_to_disk` | `bool | None` | `None` | Control file exports. `None`=auto (CLI=True, API=False), `True`=always, `False`=never |
| `export_format` | `Literal["csv", "cypher"]` | `"csv"` | Export format |
| `export_docling` | `bool` | `True` | Export Docling outputs |
| `export_docling_json` | `bool` | `True` | Export Docling JSON |
| `export_markdown` | `bool` | `True` | Export markdown |
| `export_per_page_markdown` | `bool` | `False` | Export per-page markdown |

### Graph Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reverse_edges` | `bool` | `False` | Create bidirectional edges |

### Output Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str | Path` | `"outputs"` | Output directory path |

### Models Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `ModelsConfig` | Default models | Models configuration |

---

## Methods

### run()

Execute the pipeline with this configuration.

```python
from docling_graph import run_pipeline

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument"
)

# Returns PipelineContext with results
context = run_pipeline(config)
graph = context.knowledge_graph
```

**Returns:** `PipelineContext` - Contains knowledge graph, Pydantic model, and other results

**Raises:** `PipelineError`, `ConfigurationError`, `ExtractionError`

!!! note "Accessing pipeline return values"
    Use run_pipeline(config) instead of config.run() to access return values.

---

### to_dict()

Convert configuration to dictionary format.

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument"
)

config_dict = config.to_dict()
print(config_dict)
# {
#     "source": "document.pdf",
#     "template": "templates.BillingDocument",
#     "backend": "llm",
#     ...
# }
```

**Returns:** `Dict[str, Any]`

---

## Complete Examples

### 📍 Minimal Config (API Mode)

```python
from docling_graph import run_pipeline, PipelineConfig

# Only required parameters - no file exports by default
config = PipelineConfig(
    source="invoice.pdf",
    template="templates.BillingDocument"
)

# Returns data in memory
context = run_pipeline(config)
graph = context.knowledge_graph
invoice = context.pydantic_model
```

### 📍 Debug Mode Enabled

```python
from docling_graph import run_pipeline, PipelineConfig

# Enable debug mode for troubleshooting
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    debug=True,  # Save all intermediate artifacts
    dump_to_disk=True,  # Also save final outputs
    output_dir="outputs/debug_run"
)

context = run_pipeline(config)

# Debug artifacts available at:
# outputs/debug_run/document_pdf_20260206_094500/debug/
print(f"Debug artifacts saved to: {context.output_dir}/debug/")
```

### Output Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str | Path` | `"outputs"` | Output directory path |

### 📍 Minimal Config With File Exports

```python
from docling_graph import run_pipeline, PipelineConfig

# Enable file exports
config = PipelineConfig(
    source="invoice.pdf",
    template="templates.BillingDocument",
    dump_to_disk=True,
    output_dir="outputs/invoice"
)

# Returns data AND writes files
context = run_pipeline(config)
```

### 📍 Remote LLM

```python
import os
from docling_graph import run_pipeline, PipelineConfig

# Set API key
os.environ["MISTRAL_API_KEY"] = "your-key"

# Configure for remote inference
config = PipelineConfig(
    source="research.pdf",
    template="templates.ScholarlyRheologyPaper",
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-large-latest",
    processing_mode="many-to-one",
    use_chunking=True,
    llm_consolidation=True
)

run_pipeline(config)
```

### 📍 Local VLM

```python
from docling_graph import run_pipeline, PipelineConfig

# VLM for form extraction
config = PipelineConfig(
    source="form.jpg",
    template="templates.IDCard",
    backend="vlm",
    inference="local",  # VLM only supports local
    processing_mode="one-to-one",
    docling_config="vision"
)

run_pipeline(config)
```

### 📍 Template as Class

```python
from pydantic import BaseModel, Field
from docling_graph import run_pipeline, PipelineConfig

# Define template inline
class Invoice(BaseModel):
    """Invoice template."""
    invoice_number: str = Field(description="Invoice number")
    total: float = Field(description="Total amount")

# Pass class directly
config = PipelineConfig(
    source="invoice.pdf",
    template=Invoice  # Class instead of string
)

run_pipeline(config)
```

### 📍 Custom Models Configuration

```python
from docling_graph import LLMConfig, ModelConfig, ModelsConfig, PipelineConfig, run_pipeline

# Custom models configuration
models = ModelsConfig(
    llm=LLMConfig(
        remote=ModelConfig(
            model="gpt-4o",
            provider="openai"
        )
    )
)

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="remote",
    models=models
)

run_pipeline(config)
```

For full registry and override details, see `docs/usage/api/llm-model-config.md`.

---

## Validation

### Automatic Validation

PipelineConfig validates parameters at creation:

```python
from docling_graph import run_pipeline, PipelineConfig

# This raises ValidationError
try:
    config = PipelineConfig(
        source="document.pdf",
        template="templates.BillingDocument",
        backend="invalid"  # Invalid value
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### VLM Constraints

VLM backend only supports local inference:

```python
from docling_graph import run_pipeline, PipelineConfig

# This raises ValidationError
try:
    config = PipelineConfig(
        source="document.pdf",
        template="templates.BillingDocument",
        backend="vlm",
        inference="remote"  # Not allowed for VLM
    )
except ValueError as e:
    print(f"VLM only supports local inference: {e}")
```

---

## Type Safety Benefits

### IDE Autocomplete

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",  # IDE suggests: "llm" | "vlm"
    inference="remote",  # IDE suggests: "local" | "remote"
    processing_mode="many-to-one"  # IDE suggests valid options
)
```

### Type Checking

```python
from docling_graph import run_pipeline, PipelineConfig

# mypy will catch this error
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    use_chunking="yes"  # Error: expected bool, got str
)
```

---

## Advanced Usage

### Programmatic Configuration

```python
from docling_graph import run_pipeline, PipelineConfig
from pathlib import Path

def create_config(source: str, template: str, use_remote: bool = False):
    """Factory function for creating configurations."""
    return PipelineConfig(
        source=source,
        template=template,
        backend="llm",
        inference="remote" if use_remote else "local",
        provider_override="mistral" if use_remote else "ollama"
    )

# Use factory
config = create_config("document.pdf", "templates.BillingDocument", use_remote=True)
run_pipeline(config)
```

### Configuration Templates

```python
from docling_graph import run_pipeline, PipelineConfig

# Base configuration
BASE_CONFIG = {
    "backend": "llm",
    "inference": "remote",
    "provider_override": "mistral",
    "use_chunking": True,
    "llm_consolidation": False
}

# Create specific configurations
def process_invoice(source: str):
    config = PipelineConfig(
        source=source,
        template="templates.BillingDocument",
        **BASE_CONFIG
    )
    run_pipeline(config)

def process_research(source: str):
    config = PipelineConfig(
        source=source,
        template="templates.ScholarlyRheologyPaper",
        **BASE_CONFIG,
        llm_consolidation=True  # Override for research
    )
    run_pipeline(config)
```

### Dynamic Configuration

```python
from docling_graph import run_pipeline, PipelineConfig
from pathlib import Path

def smart_config(source: str) -> PipelineConfig:
    """Create configuration based on document characteristics."""
    path = Path(source)
    file_size = path.stat().st_size
    
    # Choose settings based on file size
    if file_size < 1_000_000:  # < 1MB
        use_chunking = False
        processing = "one-to-one"
    else:
        use_chunking = True
        processing = "many-to-one"
    
    # Choose backend based on extension
    if path.suffix.lower() in ['.jpg', '.png']:
        backend = "vlm"
    else:
        backend = "llm"
    
    return PipelineConfig(
        source=source,
        template="templates.BillingDocument",
        backend=backend,
        processing_mode=processing,
        use_chunking=use_chunking
    )

# Use smart configuration
config = smart_config("document.pdf")
run_pipeline(config)
```

---

## Configuration Patterns

### Pattern 1: Environment-Based Configuration

```python
import os
from docling_graph import run_pipeline, PipelineConfig

def get_config(source: str, template: str) -> PipelineConfig:
    """Get configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return PipelineConfig(
            source=source,
            template=template,
            backend="llm",
            inference="remote",
            provider_override="mistral",
            model_override="mistral-large-latest",
            llm_consolidation=True
        )
    else:
        return PipelineConfig(
            source=source,
            template=template,
            backend="llm",
            inference="local",
            provider_override="ollama",
            llm_consolidation=False
        )

config = get_config("document.pdf", "templates.BillingDocument")
run_pipeline(config)
```

### Pattern 2: Configuration Builder

```python
from docling_graph import run_pipeline, PipelineConfig

class ConfigBuilder:
    """Builder pattern for PipelineConfig."""
    
    def __init__(self, source: str, template: str):
        self.config_dict = {
            "source": source,
            "template": template
        }
    
    def with_remote_llm(self, provider: str, model: str):
        self.config_dict.update({
            "backend": "llm",
            "inference": "remote",
            "provider_override": provider,
            "model_override": model
        })
        return self
    
    def with_chunking(self, enabled: bool = True):
        self.config_dict["use_chunking"] = enabled
        return self
    
    def with_consolidation(self, enabled: bool = True):
        self.config_dict["llm_consolidation"] = enabled
        return self
    
    def build(self) -> PipelineConfig:
        return PipelineConfig(**self.config_dict)

# Use builder
config = (ConfigBuilder("document.pdf", "templates.BillingDocument")
    .with_remote_llm("mistral", "mistral-large-latest")
    .with_chunking(True)
    .with_consolidation(True)
    .build())

run_pipeline(config)
```

---

## Best Practices

### 👍 Use Type-Safe Configuration

```python
# ✅ Good - Type-safe with validation
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm"  # Validated at creation
)

# ❌ Avoid - Dictionary without validation
config = {
    "source": "document.pdf",
    "template": "templates.BillingDocument",
    "backend": "invalid"  # No validation
}
```

### 👍 Use Defaults When Possible

```python
# ✅ Good - Rely on sensible defaults
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument"
    # Uses default backend, inference, etc.
)

# ❌ Avoid - Specifying every parameter
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="local",
    processing_mode="many-to-one",
    use_chunking=True,
    # ... all defaults
)
```

---

## Troubleshooting

### 🐛 Validation Error

**Error:**
```
ValidationError: 1 validation error for PipelineConfig
backend
  Input should be 'llm' or 'vlm'
```

**Solution:**
```python
# Use valid values
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm"  # Valid: "llm" or "vlm"
)
```

### 🐛 VLM Remote Inference

**Error:**
```
ValueError: VLM backend currently only supports local inference
```

**Solution:**
```python
# VLM only supports local
config = PipelineConfig(
    source="form.jpg",
    template="templates.IDCard",
    backend="vlm",
    inference="local"  # Must be local for VLM
)
```

---

## Next Steps

1. **[Programmatic Examples →](programmatic-examples.md)** - More code examples
2. **[Batch Processing →](batch-processing.md)** - Batch patterns
3. **[API Reference →](../../reference/config.md)** - Complete API docs