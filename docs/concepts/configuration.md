# Configuration System

Docling Graph uses a flexible, type-safe configuration system built on Pydantic. This document explains how to configure the pipeline for different use cases.

## Overview

Configuration can be provided in three ways:

1. **Programmatic**: Using `PipelineConfig` class
2. **YAML File**: Using `config.yaml`
3. **CLI Arguments**: Command-line flags

**Location**: `docling_graph/config.py`

## PipelineConfig Class

### Basic Usage

```python
from docling_graph import PipelineConfig, run_pipeline

config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="llm",
    inference="remote",
    output_dir="outputs"
)

run_pipeline(config)
```

### Complete Configuration

```python
config = PipelineConfig(
    # Required
    source="document.pdf",
    template=YourTemplate,
    
    # Core Processing
    backend="llm",                    # "llm" or "vlm"
    inference="local",                # "local" or "remote"
    processing_mode="many-to-one",    # "one-to-one" or "many-to-one"
    
    # Docling Settings
    docling_config="ocr",             # "ocr" or "vision"
    
    # Model Selection
    model_override="gpt-4-turbo",     # Override default model
    provider_override="openai",       # Override default provider
    
    # Extraction Settings
    use_chunking=True,                # Enable document chunking
    llm_consolidation=False,          # Use LLM for consolidation
    max_batch_size=1,                 # Batch size for processing
    
    # Export Settings
    export_format="csv",              # "csv" or "cypher"
    export_docling=True,              # Export Docling document
    export_docling_json=True,         # Export Docling JSON
    export_markdown=True,             # Export markdown
    export_per_page_markdown=False,   # Export per-page markdown
    
    # Graph Settings
    reverse_edges=False,              # Add reverse edges
    
    # Output
    output_dir="outputs"              # Output directory
)
```

## Configuration Parameters

### Source and Template

```python
# Source document
source: str | Path = "document.pdf"

# Pydantic template (class or dotted path)
template: Type[BaseModel] | str = YourTemplate
# or
template: str = "module.path.YourTemplate"
```

### Backend Selection

```python
# Backend type
backend: Literal["llm", "vlm"] = "llm"

# Inference location
inference: Literal["local", "remote"] = "local"

# Note: VLM only supports local inference
```

### Processing Mode

```python
# How to handle multi-page documents
processing_mode: Literal["one-to-one", "many-to-one"] = "many-to-one"

# one-to-one: Each page → separate model
# many-to-one: All pages → single merged model
```

### Docling Configuration

```python
# Docling pipeline type
docling_config: Literal["ocr", "vision"] = "ocr"

# ocr: Traditional OCR pipeline (most accurate for standard documents)
# vision: Vision-Language Model pipeline (best for complex layouts)
```

### Model Selection

```python
# Override default model
model_override: str | None = "gpt-4-turbo"

# Override default provider
provider_override: str | None = "openai"

# Available providers:
# Local: "vllm", "ollama"
# Remote: "mistral", "openai", "gemini", "watsonx"
```

### Extraction Settings

```python
# Enable hybrid chunking
use_chunking: bool = True

# Use LLM for consolidation (many-to-one only)
llm_consolidation: bool = False

# Batch size for chunk processing
max_batch_size: int = 1
```

### Export Settings

```python
# Primary export format
export_format: Literal["csv", "cypher"] = "csv"

# Docling exports
export_docling: bool = True
export_docling_json: bool = True
export_markdown: bool = True
export_per_page_markdown: bool = False
```

### Graph Settings

```python
# Add reverse edges for bidirectional traversal
reverse_edges: bool = False
```

### Output Settings

```python
# Output directory
output_dir: str | Path = "outputs"
```

## YAML Configuration

### config.yaml Structure

```yaml
# Default settings
defaults:
  processing_mode: many-to-one
  backend: llm
  inference: local
  export_format: csv

# Docling pipeline configuration
docling:
  pipeline: ocr  # ocr | vision
  export:
    docling_json: true
    markdown: true
    per_page_markdown: false

# Model configurations
models:
  vlm:
    local:
      default_model: "numind/NuExtract-2.0-8B"
      provider: "docling"
  
  llm:
    local:
      default_model: "ibm-granite/granite-4.0-1b"
      provider: "vllm"
      providers:
        vllm:
          default_model: "ibm-granite/granite-4.0-1b"
          base_url: "http://localhost:8000/v1"
        ollama:
          default_model: "llama-3.1-8b"
    
    remote:
      default_model: "mistral-small-latest"
      provider: "mistral"
      providers:
        mistral:
          default_model: "mistral-small-latest"
        openai:
          default_model: "gpt-4-turbo"
        gemini:
          default_model: "gemini-2.5-flash"

# Output settings
output:
  directory: "outputs"
  create_visualizations: true
  create_markdown: true
```

### Using YAML Config

```python
import yaml
from docling_graph import PipelineConfig, run_pipeline

# Load YAML config
with open("config.yaml") as f:
    yaml_config = yaml.safe_load(f)

# Create PipelineConfig
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    **yaml_config["defaults"]
)

run_pipeline(config)
```

## CLI Configuration

### Command-Line Arguments

```bash
docling-graph convert document.pdf \
    --template "module.YourTemplate" \
    --backend llm \
    --inference remote \
    --provider openai \
    --model gpt-4-turbo \
    --processing-mode many-to-one \
    --use-chunking \
    --no-llm-consolidation \
    --export-format csv \
    --output-dir outputs
```

### Available CLI Flags

```bash
# Required
--template TEXT              Pydantic template (dotted path)

# Backend
--backend [llm|vlm]         Extraction backend
--inference [local|remote]  Inference location
--provider TEXT             LLM/VLM provider
--model TEXT                Model name/path

# Processing
--processing-mode [one-to-one|many-to-one]
--docling-config [ocr|vision]
--use-chunking / --no-chunking
--llm-consolidation / --no-llm-consolidation

# Export
--export-format [csv|cypher]
--export-docling / --no-export-docling
--export-markdown / --no-export-markdown
--per-page-markdown / --no-per-page-markdown

# Graph
--reverse-edges / --no-reverse-edges

# Output
--output-dir PATH           Output directory
```

## Model Configuration

### Default Models

```python
# VLM (local only)
models.vlm.local.default_model = "numind/NuExtract-2.0-8B"

# LLM (local)
models.llm.local.default_model = "ibm-granite/granite-4.0-1b"
models.llm.local.provider = "vllm"

# LLM (remote)
models.llm.remote.default_model = "mistral-small-latest"
models.llm.remote.provider = "mistral"
```

### Overriding Models

```python
# Override in code
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo"
)

# Override via CLI
docling-graph convert document.pdf \
    --provider openai \
    --model gpt-4-turbo
```

### Provider-Specific Models

```yaml
# In config.yaml
models:
  llm:
    remote:
      providers:
        openai:
          default_model: "gpt-4-turbo"
        mistral:
          default_model: "mistral-large-latest"
        gemini:
          default_model: "gemini-2.5-flash"
```

## Environment Variables

### API Keys

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Mistral
export MISTRAL_API_KEY="..."

# Google Gemini
export GEMINI_API_KEY="..."

# IBM WatsonX
export WATSONX_API_KEY="..."
export WATSONX_PROJECT_ID="..."
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
```

### Using .env File

```bash
# .env
OPENAI_API_KEY=sk-...
MISTRAL_API_KEY=...
GEMINI_API_KEY=...
```

```python
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# API keys are automatically picked up
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="openai"
)
```

## Configuration Presets

### Preset 1: Fast Local Processing

```python
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="vlm",
    inference="local",
    processing_mode="one-to-one",
    output_dir="outputs"
)
```

### Preset 2: High-Quality Remote

```python
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo",
    processing_mode="many-to-one",
    use_chunking=True,
    llm_consolidation=True,
    output_dir="outputs"
)
```

### Preset 3: Cost-Effective

```python
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-small-latest",
    processing_mode="many-to-one",
    use_chunking=True,
    llm_consolidation=False,  # Programmatic merge
    output_dir="outputs"
)
```

### Preset 4: Privacy-Focused

```python
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama-3.1-8b",
    processing_mode="many-to-one",
    use_chunking=True,
    output_dir="outputs"
)
```

## Validation

### Automatic Validation

PipelineConfig validates settings automatically:

```python
# This will raise ValidationError
config = PipelineConfig(
    backend="vlm",
    inference="remote"  # Error: VLM only supports local
)

# ValidationError: VLM backend currently only supports local inference
```

### Custom Validation

```python
from pydantic import ValidationError

try:
    config = PipelineConfig(
        source="document.pdf",
        template="invalid.path",  # Invalid template path
        backend="llm"
    )
except ValidationError as e:
    print(f"Configuration error: {e}")
```

## Best Practices

### 1. Use Type-Safe Config

```python
# Good: Type-safe
config = PipelineConfig(
    backend="llm",
    inference="remote"
)

# Avoid: Dictionary (no validation)
config_dict = {
    "backend": "llm",
    "inference": "remote"
}
```

### 2. Separate Configs by Environment

```python
# development.py
dev_config = PipelineConfig(
    backend="llm",
    inference="local",
    provider_override="ollama"
)

# production.py
prod_config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo"
)
```

### 3. Use Environment Variables for Secrets

```python
# Don't hardcode API keys
# Bad:
config = PipelineConfig(
    provider_override="openai",
    api_key="sk-..."  # Don't do this!
)

# Good: Use environment variables
# API keys are automatically loaded from environment
config = PipelineConfig(
    provider_override="openai"
)
```

### 4. Document Your Configs

```python
# config.py
"""
Configuration for invoice processing pipeline.

Uses OpenAI GPT-4 for high-quality extraction.
Processes documents in many-to-one mode with chunking.
"""

INVOICE_CONFIG = PipelineConfig(
    template=Invoice,
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo",
    processing_mode="many-to-one",
    use_chunking=True,
    output_dir="outputs/invoices"
)
```

## Troubleshooting

### Issue: Configuration Not Applied

**Problem**: Settings seem to be ignored

**Solution**: Check precedence order:
1. CLI arguments (highest priority)
2. PipelineConfig parameters
3. YAML config
4. Defaults (lowest priority)

### Issue: Model Not Found

**Problem**: "Model not found" error

**Solution**: Verify model name and provider:
```python
# Check available models in config.yaml
# Ensure model name matches exactly
config = PipelineConfig(
    provider_override="openai",
    model_override="gpt-4-turbo"  # Must match exactly
)
```

### Issue: API Key Not Found

**Problem**: "API key not set" error

**Solution**: Set environment variable:
```bash
export OPENAI_API_KEY="your-key"
# or use .env file
```

## Next Steps

- Learn about [Extraction Backends](extraction-backends.md)
- Understand [Processing Strategies](processing-strategies.md)
- Explore [Architecture](architecture.md)