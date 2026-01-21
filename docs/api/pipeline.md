# Pipeline API

The pipeline module provides the main entry point for running document-to-graph conversions.

## run_pipeline

::: docling_graph.pipeline.run_pipeline
    options:
      show_source: true
      heading_level: 3

## PipelineConfig

::: docling_graph.config.PipelineConfig
    options:
      show_source: true
      heading_level: 3

## Usage Examples

### Basic Usage

```python
from docling_graph import run_pipeline, PipelineConfig
from your_templates import YourTemplate

config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="llm",
    output_dir="outputs"
)

run_pipeline(config)
```

### With Custom Configuration

```python
from docling_graph import run_pipeline, PipelineConfig
from docling_graph.llm_clients import LLMConfig

llm_config = LLMConfig(
    provider="openai",
    model="gpt-4",
    temperature=0.1,
    max_tokens=4000
)

config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="llm",
    llm_config=llm_config,
    processing_mode="many-to-one",
    use_chunking=True,
    output_dir="outputs"
)

run_pipeline(config)
```

### Batch Processing

```python
from pathlib import Path
from docling_graph import run_pipeline, PipelineConfig

documents = Path("documents").glob("*.pdf")

for doc in documents:
    config = PipelineConfig(
        source=str(doc),
        template=YourTemplate,
        backend="llm",
        output_dir=f"outputs/{doc.stem}"
    )
    
    try:
        run_pipeline(config)
        print(f"✓ Processed {doc.name}")
    except Exception as e:
        print(f"✗ Failed {doc.name}: {e}")
```

## See Also

- [Configuration](config.md) - Configuration options
- [Quick Start](../getting-started/quickstart.md) - Getting started guide
- [Examples](../examples/README.md) - More examples