# Configuration API


## Overview

Type-safe configuration classes for the docling-graph pipeline.

**Module:** `docling_graph.config`

---

## PipelineConfig

Main configuration class for pipeline execution.

```python
class PipelineConfig(BaseModel):
    """Type-safe configuration for the docling-graph pipeline."""
```

### Constructor

```python
config = PipelineConfig(
    source: Union[str, Path] = "",
    template: Union[str, Type[BaseModel]] = "",
    backend: Literal["llm", "vlm"] = "llm",
    inference: Literal["local", "remote"] = "local",
    processing_mode: Literal["one-to-one", "many-to-one"] = "many-to-one",
    extraction_contract: Literal["direct", "staged"] = "direct",
    docling_config: Literal["ocr", "vision"] = "ocr",
    model_override: str | None = None,
    provider_override: str | None = None,
    models: ModelsConfig = ModelsConfig(),
    use_chunking: bool = True,
    chunk_max_tokens: int | None = None,
    debug: bool = False,
    max_batch_size: int = 1,
    dump_to_disk: bool | None = None,
    export_format: Literal["csv", "cypher"] = "csv",
    export_docling: bool = True,
    export_docling_json: bool = True,
    export_markdown: bool = True,
    export_per_page_markdown: bool = False,
    reverse_edges: bool = False,
    output_dir: Union[str, Path] = "outputs"
)
```

### Fields

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` or `Path` | Path to source document |
| `template` | `str` or `Type[BaseModel]` | Pydantic template class or dotted path |

#### Backend Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `"llm"` or `"vlm"` | `"llm"` | Extraction backend type |
| `inference` | `"local"` or `"remote"` | `"local"` | Inference location |
| `model_override` | `str` or `None` | `None` | Override default model |
| `provider_override` | `str` or `None` | `None` | Override default provider |
| `models` | `ModelsConfig` | `ModelsConfig()` | Model configurations |

#### Processing Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `processing_mode` | `"one-to-one"` or `"many-to-one"` | `"many-to-one"` | Processing strategy |
| `extraction_contract` | `"direct"` or `"staged"` | `"direct"` | LLM extraction contract (`staged` is optimized for weaker models in many-to-one mode) |
| `docling_config` | `"ocr"` or `"vision"` | `"ocr"` | Docling pipeline type |
| `use_chunking` | `bool` | `True` | Enable document chunking |
| `chunk_max_tokens` | `int` or `None` | `None` | Max tokens per chunk (default 512 when chunking) |
| `debug` | `bool` | `False` | Enable debug artifacts |
| `max_batch_size` | `int` | `1` | Maximum batch size |

#### Staged Tuning Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `staged_tuning_preset` | `"standard"` or `"advanced"` | `"standard"` | Preset for staged extraction defaults |
| `staged_pass_retries` | `int` or `None` | `None` | Retries per staged pass (`None` uses preset) |
| `staged_workers` | `int` or `None` | `None` | Fill-pass worker count only (`None` uses preset) |
| `staged_nodes_fill_cap` | `int` or `None` | `None` | Max node instances per fill-pass call (`None` uses preset) |
| `staged_id_shard_size` | `int` or `None` | `None` | Max paths per ID-pass call (`0` = no sharding, `None` uses preset) |

#### Export Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dump_to_disk` | `bool` or `None` | `None` | Control file exports. `None`=auto-detect (CLI=True, API=False), `True`=always export, `False`=never export |
| `export_format` | `"csv"` or `"cypher"` | `"csv"` | Graph export format |
| `export_docling` | `bool` | `True` | Export Docling outputs |
| `export_docling_json` | `bool` | `True` | Export Docling JSON |
| `export_markdown` | `bool` | `True` | Export markdown |
| `export_per_page_markdown` | `bool` | `False` | Export per-page markdown |

#### Graph Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reverse_edges` | `bool` | `False` | Create reverse edges |

#### Output Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | `str` or `Path` | `"outputs"` | Output directory path |

### Methods

#### run()

```python
def run(self) -> None
```

Execute the pipeline with this configuration.

**Example:**

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate"
)
run_pipeline(config)
```

#### to_dict()

```python
def to_dict(self) -> Dict[str, Any]
```

Convert configuration to dictionary format.

**Returns:** Dictionary with all configuration values

**Example:**

```python
config = PipelineConfig(source="doc.pdf", template="templates.MyTemplate")
config_dict = config.to_dict()
print(config_dict["backend"])  # "llm"
```

#### generate_yaml_dict()

```python
@classmethod
def generate_yaml_dict(cls) -> Dict[str, Any]
```

Generate YAML-compatible configuration dictionary with defaults.

**Returns:** Dictionary suitable for YAML serialization

---

## ModelsConfig

Configuration for all model types.

```python
class ModelsConfig(BaseModel):
    """Complete models configuration."""
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `llm` | `LLMConfig` | LLM model configuration |
| `vlm` | `VLMConfig` | VLM model configuration |

---

## LLMConfig

Configuration for LLM models.

```python
class LLMConfig(BaseModel):
    """LLM model configurations for local and remote inference."""
    
    local: ModelConfig = Field(default_factory=lambda: ModelConfig(
        model="ibm-granite/granite-4.0-1b",
        provider="vllm"
    ))
    remote: ModelConfig = Field(default_factory=lambda: ModelConfig(
        model="mistral-small-latest",
        provider="mistral"
    ))
```

### Fields

| Field | Type | Default Model | Default Provider |
|-------|------|---------------|------------------|
| `local` | `ModelConfig` | `ibm-granite/granite-4.0-1b` | `vllm` |
| `remote` | `ModelConfig` | `mistral-small-latest` | `mistral` |

---

## VLMConfig

Configuration for VLM models.

```python
class VLMConfig(BaseModel):
    """VLM model configuration."""
    
    local: ModelConfig = Field(default_factory=lambda: ModelConfig(
        model="numind/NuExtract-2.0-8B",
        provider="docling"
    ))
```

### Fields

| Field | Type | Default Model | Default Provider |
|-------|------|---------------|------------------|
| `local` | `ModelConfig` | `numind/NuExtract-2.0-8B` | `docling` |

!!! note "VLM inference"
    VLM only supports local inference.

---

## ModelConfig

Configuration for a specific model.

```python
class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    model: str = Field(..., description="Model name/path")
    provider: str = Field(..., description="Provider name")
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Model name or path |
| `provider` | `str` | Provider name (e.g., "vllm", "mistral") |

---

## BackendConfig

Configuration for extraction backend (internal use).

```python
class BackendConfig(BaseModel):
    """Configuration for an extraction backend."""
    
    provider: str = Field(..., description="Backend provider")
    model: str = Field(..., description="Model name or path")
    api_key: str | None = Field(None, description="API key")
    base_url: str | None = Field(None, description="Base URL")
```

---

## ExtractorConfig

Configuration for extraction strategy (internal use).

```python
class ExtractorConfig(BaseModel):
    """Configuration for the extraction strategy."""
    
    strategy: Literal["many-to-one", "one-to-one"] = Field(default="many-to-one")
    docling_config: Literal["ocr", "vision"] = Field(default="ocr")
    use_chunking: bool = Field(default=True)
    chunker_config: Dict[str, Any] | None = Field(default=None)
```

---

## Usage Examples

### Basic Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate"
)
run_pipeline(config)
```

### Custom Backend

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    backend="llm",
    inference="remote",
    model_override="gpt-4-turbo",
    provider_override="openai"
)
run_pipeline(config)
```

### Custom Processing

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    processing_mode="one-to-one",
    use_chunking=False
)
run_pipeline(config)
```

### Custom Export

```python
from docling_graph import run_pipeline

config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    dump_to_disk=True,  # Enable file exports
    export_format="cypher",
    export_docling_json=True,
    export_markdown=True,
    export_per_page_markdown=True,
    output_dir="custom_outputs"
)

# Returns data AND writes files
context = run_pipeline(config)
```

### API Mode (No File Exports)

```python
from docling_graph import run_pipeline

config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate"
    # dump_to_disk defaults to None (auto-detects as False for API)
    # debug defaults to False (no debug artifacts)
)

# Returns data only, no file exports, no debug artifacts
context = run_pipeline(config)
graph = context.knowledge_graph
```

### Debug Mode

```python
from docling_graph import run_pipeline
from pathlib import Path

# Enable debug mode for troubleshooting
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    debug=True,  # Enable debug mode
    dump_to_disk=True  # Also save final outputs
)

context = run_pipeline(config)

# Debug artifacts saved to outputs/{document}_{timestamp}/debug/
debug_dir = Path(context.output_dir) / "debug"
print(f"Debug artifacts saved to: {debug_dir}")

# Analyze staged debug artifacts
import json

# Core staged artifacts
for name in [
    "node_catalog.json",
    "id_pass.json",
    "fill_pass.json",
    "edges_pass.json",
    "merged_output.json",
    "staged_trace.json",
]:
    p = debug_dir / name
    print(name, "exists" if p.exists() else "missing")

# Example: inspect staged trace timings
with open(debug_dir / "staged_trace.json") as f:
    trace = json.load(f)
    print(trace.get("timings_seconds", {}))
```

### Explicit Control

```python
from docling_graph import run_pipeline

# Force file exports in API mode
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    dump_to_disk=True,
    output_dir="outputs"
)
context = run_pipeline(config)

# Force no file exports (even if output_dir is set)
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    dump_to_disk=False,
    output_dir="outputs"  # Ignored
)
context = run_pipeline(config)
```

### Complete Configuration

```python
from docling_graph import run_pipeline, PipelineConfig, LLMConfig, ModelConfig

config = PipelineConfig(
    # Source
    source="document.pdf",
    template="templates.MyTemplate",
    
    # Backend
    backend="llm",
    inference="remote",
    model_override="mistral-small-latest",
    provider_override="mistral",
    
    # Processing
    processing_mode="many-to-one",
    docling_config="ocr",
    use_chunking=True,
    max_batch_size=5,
    
    # Export
    export_format="csv",
    export_docling=True,
    export_docling_json=True,
    export_markdown=True,
    export_per_page_markdown=False,
    
    # Graph
    reverse_edges=False,
    
    # Output
    output_dir="outputs/custom"
)

run_pipeline(config)
```

---

## Validation

### Automatic Validation

Pydantic validates all fields automatically:

```python
# ✅ Valid
config = PipelineConfig(
    source="doc.pdf",
    template="templates.MyTemplate",
    backend="llm"
)

# ❌ Invalid - raises ValidationError
config = PipelineConfig(
    source="doc.pdf",
    template="templates.MyTemplate",
    backend="invalid"  # Not "llm" or "vlm"
)
```

### Custom Validation

VLM backend validation:

```python
# ❌ Invalid - VLM doesn't support remote
config = PipelineConfig(
    source="doc.pdf",
    template="templates.MyTemplate",
    backend="vlm",
    inference="remote"  # Raises ValueError
)

# ✅ Valid
config = PipelineConfig(
    source="doc.pdf",
    template="templates.MyTemplate",
    backend="vlm",
    inference="local"
)
```

---

## Type Safety

### Type Hints

All fields have proper type hints:

```python
from docling_graph import run_pipeline, PipelineConfig
from pathlib import Path

# Type checker knows these are valid
config = PipelineConfig(
    source="doc.pdf",  # str
    template="templates.MyTemplate",  # str
    backend="llm",  # Literal["llm", "vlm"]
    use_chunking=True  # bool
)

# Type checker knows output_dir is str
output: str = config.output_dir
```

### IDE Support

IDEs provide autocomplete and type checking:

```python
config = PipelineConfig(
    source="doc.pdf",
    template="templates.MyTemplate",
    backend="l"  # IDE suggests "llm"
)
```

---

## Default Values

All fields have sensible defaults:

```python
config = PipelineConfig(
    source="doc.pdf",
    template="templates.MyTemplate"
    # All other fields use defaults
)

print(config.backend)  # "llm"
print(config.inference)  # "local"
print(config.processing_mode)  # "many-to-one"
print(config.use_chunking)  # True
print(config.export_format)  # "csv"
```

---

## Related APIs

- **[Pipeline API](pipeline.md)** - run_pipeline() function
- **[Protocols](protocols.md)** - Protocol definitions
- **[Exceptions](exceptions.md)** - Validation errors

---

## See Also

- **[Configuration Guide](../fundamentals/pipeline-configuration/index.md)** - Configuration overview
- **[Python API](../usage/api/pipeline-config.md)** - Usage guide
- **[Examples](../usage/examples/index.md)** - Example configurations