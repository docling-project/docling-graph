# Configuration

Learn how to configure Docling Graph for your specific use case.

## Configuration Methods

### 1. Python Configuration

Use `PipelineConfig` for programmatic configuration:

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4",
    output_dir="outputs"
)
```

### 2. CLI Configuration

Use command-line arguments:

```bash
docling-graph convert document.pdf \
    --template "module.Template" \
    --backend llm \
    --inference remote \
    --provider openai \
    --model gpt-4 \
    --output-dir outputs
```

### 3. Configuration File

Create a YAML configuration file:

```yaml
# config.yaml
source: document.pdf
template: module.Template
backend: llm
inference: remote
provider_override: openai
model_override: gpt-4
output_dir: outputs
```

Load it in Python:

```python
from docling_graph import run_pipeline
from docling_graph.config import load_config

config = load_config("config.yaml")
run_pipeline(config)
```

## Configuration Parameters

### Core Parameters

#### source
- **Type**: `str` or `Path`
- **Required**: Yes
- **Description**: Path to the input document (PDF, image, etc.)

```python
source="document.pdf"
source="/path/to/document.pdf"
```

#### template
- **Type**: `Type[BaseModel]` or `str`
- **Required**: Yes
- **Description**: Pydantic model defining the extraction schema

```python
# As class
template=YourTemplate

# As dotted path (CLI)
template="module.submodule.YourTemplate"
```

#### output_dir
- **Type**: `str` or `Path`
- **Required**: Yes
- **Description**: Directory for output files

```python
output_dir="outputs"
output_dir="outputs/experiment_1"
```

### Backend Configuration

#### backend
- **Type**: `str`
- **Options**: `"vlm"`, `"llm"`
- **Default**: `"vlm"`
- **Description**: Extraction backend to use

```python
# Local VLM (Docling)
backend="vlm"

# LLM-based extraction
backend="llm"
```

#### inference
- **Type**: `str`
- **Options**: `"local"`, `"remote"`
- **Default**: `"remote"`
- **Description**: Inference mode (only for LLM backend)

```python
# Remote API
inference="remote"

# Local inference
inference="local"
```

### Provider Configuration

#### provider_override
- **Type**: `str` or `None`
- **Options**: `"openai"`, `"mistral"`, `"gemini"`, `"watsonx"`, `"ollama"`, `"vllm"`
- **Default**: `None`
- **Description**: Override the LLM provider

```python
# Remote providers
provider_override="openai"
provider_override="mistral"
provider_override="gemini"
provider_override="watsonx"

# Local providers
provider_override="ollama"
provider_override="vllm"
```

#### model_override
- **Type**: `str` or `None`
- **Default**: `None`
- **Description**: Override the model name

```python
# OpenAI models
model_override="gpt-4"
model_override="gpt-3.5-turbo"

# Mistral models
model_override="mistral-medium-latest"
model_override="mistral-large-latest"

# Gemini models
model_override="gemini-pro"

# Ollama models
model_override="llama2"
model_override="mistral"
```

### Processing Configuration

#### processing_mode
- **Type**: `str`
- **Options**: `"one-to-one"`, `"many-to-one"`
- **Default**: `"one-to-one"`
- **Description**: How to process multi-page documents

```python
# Process each page separately
processing_mode="one-to-one"

# Combine all pages into one extraction
processing_mode="many-to-one"
```

#### use_chunking
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable hybrid chunking for better context

```python
# Enable chunking
use_chunking=True

# Disable chunking
use_chunking=False
```

#### llm_consolidation
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use LLM to consolidate extracted data

```python
# LLM-based consolidation
llm_consolidation=True

# Programmatic consolidation
llm_consolidation=False
```

### Export Configuration

#### export_formats
- **Type**: `List[str]`
- **Options**: `"json"`, `"csv"`, `"cypher"`, `"docling"`, `"markdown"`
- **Default**: `["json", "csv", "cypher"]`
- **Description**: Output formats to generate

```python
# All formats
export_formats=["json", "csv", "cypher", "docling", "markdown"]

# Specific formats
export_formats=["json", "cypher"]
```

#### generate_visualization
- **Type**: `bool`
- **Default**: `True`
- **Description**: Generate interactive HTML visualization

```python
generate_visualization=True
```

#### generate_report
- **Type**: `bool`
- **Default**: `True`
- **Description**: Generate markdown report

```python
generate_report=True
```

## Advanced Configuration

### LLM Configuration

For fine-grained control over LLM behavior:

```python
from docling_graph.llm_clients import LLMConfig

llm_config = LLMConfig(
    provider="openai",
    model="gpt-4",
    temperature=0.1,
    max_tokens=4000,
    timeout=60,
    max_retries=3
)

config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="llm",
    llm_config=llm_config,
    output_dir="outputs"
)
```

### VLM Configuration

For VLM-specific settings:

```python
from docling_graph.config import VLMConfig

vlm_config = VLMConfig(
    use_ocr=True,
    extract_tables=True,
    extract_images=True
)

config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="vlm",
    vlm_config=vlm_config,
    output_dir="outputs"
)
```

### Chunking Configuration

Control chunking behavior:

```python
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="llm",
    use_chunking=True,
    chunk_size=1000,        # Characters per chunk
    chunk_overlap=200,      # Overlap between chunks
    output_dir="outputs"
)
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

### Other Settings

```bash
# Logging level
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR

# Cache directory
export DOCLING_CACHE_DIR="~/.cache/docling"

# Temporary directory
export TEMP_DIR="/tmp/docling"
```

## Configuration Examples

### Example 1: Simple Invoice Processing

```python
config = PipelineConfig(
    source="invoice.pdf",
    template=Invoice,
    backend="vlm",
    output_dir="outputs/invoices"
)
```

### Example 2: Research Paper with LLM

```python
config = PipelineConfig(
    source="paper.pdf",
    template=ResearchPaper,
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-medium-latest",
    processing_mode="many-to-one",
    use_chunking=True,
    llm_consolidation=False,
    output_dir="outputs/papers"
)
```

### Example 3: Local Processing with Ollama

```python
config = PipelineConfig(
    source="document.pdf",
    template=Document,
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama2",
    output_dir="outputs/local"
)
```

### Example 4: Batch Processing

```python
from pathlib import Path

for pdf in Path("documents").glob("*.pdf"):
    config = PipelineConfig(
        source=str(pdf),
        template=YourTemplate,
        backend="llm",
        output_dir=f"outputs/{pdf.stem}"
    )
    run_pipeline(config)
```

## Best Practices

### 1. Choose the Right Backend

- **VLM**: Best for structured documents (invoices, forms, ID cards)
- **LLM**: Best for unstructured text (research papers, reports, articles)

### 2. Optimize Processing Mode

- **one-to-one**: Better for documents with distinct page-level information
- **many-to-one**: Better for documents with continuous narrative

### 3. Use Chunking Wisely

- Enable for long documents (>10 pages)
- Disable for short, structured documents
- Adjust chunk size based on model context window

### 4. Select Appropriate Models

- **GPT-4**: Best quality, higher cost
- **GPT-3.5-turbo**: Good balance
- **Mistral Medium**: Cost-effective alternative
- **Local models**: Privacy-focused, no API costs

## Troubleshooting

### Configuration Validation

```python
from docling_graph import PipelineConfig

try:
    config = PipelineConfig(
        source="document.pdf",
        template=YourTemplate,
        output_dir="outputs"
    )
    print("Configuration valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = PipelineConfig(...)
run_pipeline(config)
```

## Next Steps

- [Quick Start](quickstart.md) - Get started with examples
- [Pydantic Templates](../guides/create_pydantic_templates_for_kg_extraction.md) - Create templates
- [API Reference](../api/config.md) - Complete configuration API