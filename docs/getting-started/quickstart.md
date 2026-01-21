# Quick Start

Get up and running with Docling Graph in minutes.

## Prerequisites

- Python 3.10 or higher installed
- Docling Graph installed (see [Installation](installation.md))
- API keys configured (if using remote LLM providers)

## Your First Conversion

### 1. Using the CLI

The easiest way to get started is with the CLI:

```bash
# Initialize configuration (interactive wizard)
docling-graph init

# Convert a document
docling-graph convert document.pdf \
    --template "your_module.YourTemplate" \
    --output-dir outputs
```

### 2. Using Python

For programmatic control:

```python
from docling_graph import run_pipeline, PipelineConfig
from pydantic import BaseModel, Field

# Define your extraction template
class Person(BaseModel):
    """Person entity."""
    model_config = {
        'is_entity': True,
        'graph_id_fields': ['name']
    }
    
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's occupation")

# Configure pipeline
config = PipelineConfig(
    source="document.pdf",
    template=Person,
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4",
    output_dir="outputs"
)

# Run conversion
run_pipeline(config)
```

## Example: Extract Research Paper Information

Let's extract information from a research paper:

```python
from docling_graph import run_pipeline, PipelineConfig
from pydantic import BaseModel, Field
from typing import List

class Author(BaseModel):
    """Research paper author."""
    model_config = {
        'is_entity': True,
        'graph_id_fields': ['name']
    }
    
    name: str = Field(description="Author's full name")
    affiliation: str = Field(description="Author's institution")

class ResearchPaper(BaseModel):
    """Research paper information."""
    
    title: str = Field(description="Paper title")
    abstract: str = Field(description="Paper abstract")
    authors: List[Author] = Field(description="List of authors")
    keywords: List[str] = Field(description="Paper keywords")
    year: int = Field(description="Publication year")

# Configure and run
config = PipelineConfig(
    source="research_paper.pdf",
    template=ResearchPaper,
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-medium-latest",
    use_chunking=True,
    processing_mode="many-to-one",
    output_dir="outputs/research"
)

run_pipeline(config)
```

## Understanding the Output

After running the pipeline, you'll find several files in your output directory:

```
outputs/
├── graph.json              # Graph data in JSON format
├── nodes.csv              # Nodes for Neo4j import
├── edges.csv              # Edges for Neo4j import
├── cypher_script.cypher   # Cypher script for bulk import
├── visualization.html     # Interactive graph visualization
└── report.md             # Detailed markdown report
```

### Viewing the Graph

Open `visualization.html` in your browser to explore the interactive graph:

- Click nodes to see details
- Hover over edges to see relationships
- Use the search to find specific entities
- Zoom and pan to navigate

## Configuration Options

### Backend Options

```python
# Local VLM (Docling)
backend="vlm"

# LLM-based extraction
backend="llm"
```

### Inference Options

```python
# Remote API
inference="remote"
provider_override="openai"  # or "mistral", "gemini", "watsonx"

# Local inference
inference="local"
provider_override="ollama"  # or "vllm"
```

### Processing Modes

```python
# Process each page separately
processing_mode="one-to-one"

# Combine all pages into one extraction
processing_mode="many-to-one"
```

### Chunking Options

```python
# Enable hybrid chunking
use_chunking=True

# Disable chunking (process full document)
use_chunking=False

# LLM-based consolidation
llm_consolidation=True
```

## CLI Examples

### Basic Conversion

```bash
docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --output-dir outputs
```

### With Custom Configuration

```bash
docling-graph convert document.pdf \
    --template "templates.Research" \
    --backend llm \
    --inference remote \
    --provider openai \
    --model gpt-4 \
    --processing-mode many-to-one \
    --use-chunking \
    --output-dir outputs
```

### Inspect Results

```bash
docling-graph inspect outputs
```

## Common Patterns

### Pattern 1: Invoice Processing

```python
from docling_graph import run_pipeline, PipelineConfig
from templates.invoice import Invoice

config = PipelineConfig(
    source="invoice.pdf",
    template=Invoice,
    backend="vlm",  # VLM works well for structured documents
    output_dir="outputs/invoices"
)

run_pipeline(config)
```

### Pattern 2: Multi-Page Document

```python
config = PipelineConfig(
    source="report.pdf",
    template=Report,
    backend="llm",
    processing_mode="many-to-one",  # Combine all pages
    use_chunking=True,              # Use smart chunking
    llm_consolidation=False,        # Merge programmatically
    output_dir="outputs/reports"
)

run_pipeline(config)
```

### Pattern 3: Batch Processing

```python
from pathlib import Path

documents = Path("documents").glob("*.pdf")

for doc in documents:
    config = PipelineConfig(
        source=str(doc),
        template=YourTemplate,
        backend="llm",
        output_dir=f"outputs/{doc.stem}"
    )
    run_pipeline(config)
```

## Next Steps

- [Configuration Guide](configuration.md) - Detailed configuration options
- [Pydantic Templates](../guides/create_pydantic_templates_for_kg_extraction.md) - Create custom templates
- [Examples](../examples/README.md) - More example use cases
- [API Reference](../api/pipeline.md) - Complete API documentation

## Troubleshooting

### Common Issues

**Issue**: Import errors
```bash
# Solution: Install required extras
pip install docling-graph[all]
```

**Issue**: API authentication errors
```bash
# Solution: Set API keys
export OPENAI_API_KEY="your-key"
```

**Issue**: Out of memory errors
```bash
# Solution: Enable chunking
use_chunking=True
```

For more help, see our [GitHub Issues](https://github.com/IBM/docling-graph/issues).