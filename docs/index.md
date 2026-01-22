# Docling Graph Documentation

<p align="center">
  <img src="assets/logo.png" alt="Docling Graph" width="280"/>
</p>

Welcome to the **Docling Graph** documentation! This guide will help you transform unstructured documents into validated knowledge graphs using Pydantic templates and advanced extraction techniques.

---

## What is Docling Graph?

Docling Graph converts documents into validated **Pydantic** objects and then into **directed knowledge graphs**, with exports to CSV or Cypher and both static and interactive visualizations.

This transformation enables precise semantic relationships essential for complex domains like **chemistry, finance, and legal** where AI systems must understand exact entity connections rather than approximate text vectors, **enabling explainable reasoning over technical document collections**.

### Key Features

- **🧠 Flexible Extraction**: Local VLM (Docling) or LLM-based (vLLM, Ollama, Mistral, OpenAI, Gemini, WatsonX)
- **🔨 Smart Graph Construction**: Convert Pydantic models to NetworkX graphs with stable node IDs
- **📦 Multiple Export Formats**: CSV (Neo4j-compatible), Cypher scripts, JSON, Markdown
- **📊 Rich Visualizations**: Interactive HTML and detailed Markdown reports
- **⚙️ Type-Safe Configuration**: Pydantic-based configuration with validation
- **🎯 Pipeline Architecture**: Modular, stage-based processing with error handling

---

## Quick Navigation

### 🚀 Getting Started

<div class="grid cards" markdown>

- **[Installation →](02-installation/index.md)**

    Set up your environment with uv package manager
    
    ```bash
    uv sync --extra all
    ```

- **[Quick Start →](09-examples/quickstart.md)**

    Run your first extraction in 5 minutes
    
    ```bash
    uv run docling-graph init
    ```

- **[Architecture →](01-introduction/architecture-overview.md)**

    Understand the pipeline stages and components

- **[Key Concepts →](01-introduction/key-concepts.md)**

    Learn how documents flow through the system

</div>

### 📚 Core Documentation

<div class="grid cards" markdown>

- **[1. Introduction](01-introduction/index.md)**

    Overview, architecture, and core concepts

- **[2. Installation](02-installation/index.md)**

    Setup, GPU support, and troubleshooting

- **[3. Schema Definition](03-schema-definition/index.md)**

    Create Pydantic templates for extraction

- **[4. Pipeline Configuration](04-pipeline-configuration/index.md)**

    Configure backends, models, and processing

- **[5. Extraction Process](05-extraction-process/index.md)**

    Document conversion and extraction

- **[6. Graph Management](06-graph-management/index.md)**

    Export and visualize knowledge graphs

- **[7. CLI Reference](07-cli/index.md)**

    Command-line interface guide

- **[8. Python API](08-api/index.md)**

    Programmatic usage and integration

- **[9. Examples](09-examples/index.md)**

    Working code examples and templates

- **[10. Advanced Topics](10-advanced/index.md)**

    Performance, testing, and debugging

- **[11. API Reference](11-reference/index.md)**

    Detailed API documentation

- **[12. Development](12-development/index.md)**

    Contributing and development guide

</div>

---

## Quick Start Example

### Python API

```python
from docling_graph import PipelineConfig

# Create configuration
config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    output_dir="outputs/invoice"
)

# Run pipeline
config.run()
```

### CLI

```bash
# Initialize configuration
uv run docling-graph init

# Convert document
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --output-dir "outputs/invoice"

# Visualize results
uv run docling-graph inspect outputs/invoice
```

---

## Documentation Structure

The documentation follows the **docling-graph pipeline stages** for a logical, end-to-end learning experience:

### Stage 1: Understanding (Introduction)
Learn what Docling Graph is, how it works, and its architecture.

**[→ Go to Introduction](01-introduction/index.md)**

### Stage 2: Setup (Installation)
Install dependencies, configure your environment, and set up GPU support.

**[→ Go to Installation](02-installation/index.md)**

### Stage 3: Schema Design (Schema Definition)
Create Pydantic templates that define both extraction schema and graph structure.

**[→ Go to Schema Definition](03-schema-definition/index.md)**

### Stage 4: Configuration (Pipeline Configuration)
Configure backends, models, processing modes, and export options.

**[→ Go to Pipeline Configuration](04-pipeline-configuration/index.md)**

### Stage 5: Extraction (Extraction Process)
Convert documents, chunk content, and extract structured data.

**[→ Go to Extraction Process](05-extraction-process/index.md)**

### Stage 6: Output (Graph Management)
Export graphs to CSV/Cypher, visualize results, and integrate with Neo4j.

**[→ Go to Graph Management](06-graph-management/index.md)**

### Stage 7: Usage (CLI & API)
Use the command-line interface or Python API for your workflows.

**[→ CLI Reference](07-cli/index.md)** | **[→ Python API](08-api/index.md)**

### Stage 8: Learning (Examples)
Explore working examples and template gallery.

**[→ Go to Examples](09-examples/index.md)**

### Stage 9: Optimization (Advanced Topics)
Optimize performance, handle errors, and implement custom backends.

**[→ Go to Advanced Topics](10-advanced/index.md)**

### Stage 10: Reference (API Reference)
Detailed API documentation for all modules.

**[→ Go to API Reference](11-reference/index.md)**

### Stage 11: Contributing (Development)
Contribute to the project and understand the development workflow.

**[→ Go to Development](12-development/index.md)**

---

## Common Use Cases

### Extract Invoice Data

```python
from docling_graph import PipelineConfig
from templates import Invoice

config = PipelineConfig(
    source="invoice.pdf",
    template=Invoice,
    backend="llm",
    inference="remote"
)
config.run()
```

**[→ See Invoice Template](09-examples/invoice-extraction.md)**

### Process Research Papers

```python
config = PipelineConfig(
    source="research.pdf",
    template=ResearchPaper,
    backend="llm",
    processing_mode="many-to-one",
    use_chunking=True
)
config.run()
```

**[→ See Research Template](09-examples/research-paper.md)**

### Extract ID Card Information

```python
config = PipelineConfig(
    source="id_card.jpg",
    template=IDCard,
    backend="vlm",
    inference="local"
)
config.run()
```

**[→ See ID Card Template](09-examples/id-card.md)**

---

## Key Concepts

### Pydantic Templates

Templates define both the **extraction schema** and the **graph structure**:

```python
from pydantic import BaseModel, Field
from docling_graph.utils import edge

class Person(BaseModel):
    """Person entity."""
    model_config = {
        'is_entity': True,
        'graph_id_fields': ['last_name', 'date_of_birth']
    }
    
    first_name: str = Field(description="First name")
    last_name: str = Field(description="Last name")
    date_of_birth: str = Field(description="Date of birth (YYYY-MM-DD)")

class Organization(BaseModel):
    """Organization entity."""
    model_config = {'is_entity': True}
    
    name: str = Field(description="Organization name")
    employees: list[Person] = edge("EMPLOYS", description="Employees")
```

**[→ Learn More About Templates](03-schema-definition/index.md)**

### Pipeline Stages

1. **Template Loading**: Load and validate Pydantic templates
2. **Document Conversion**: Convert documents using Docling
3. **Chunking**: Split content into manageable pieces (optional)
4. **Extraction**: Extract structured data using VLM or LLM
5. **Consolidation**: Merge results (optional)
6. **Graph Construction**: Build NetworkX graph from Pydantic models
7. **Export**: Generate CSV, Cypher, JSON outputs
8. **Visualization**: Create interactive HTML and Markdown reports

**[→ Learn More About Key Concepts](01-introduction/key-concepts.md)**

### Extraction Backends

- **VLM (Vision-Language Model)**: Local extraction using Docling's NuExtract
- **LLM (Language Model)**: Text-based extraction using local (vLLM, Ollama) or remote APIs (Mistral, OpenAI, Gemini, WatsonX)

**[→ Learn More About Backends](05-extraction-process/extraction-backends.md)**

---

## Resources

### Documentation
- **[GitHub Repository](https://github.com/IBM/docling-graph)** - Source code and issues
- **[PyPI Package](https://pypi.org/project/docling-graph/)** - Install via pip/uv
- **[Contributing Guidelines](https://github.com/IBM/docling-graph/blob/main/.github/CONTRIBUTING.md)** - How to contribute

### Community
- **[GitHub Issues](https://github.com/IBM/docling-graph/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/IBM/docling-graph/discussions)** - Ask questions and share ideas

### Related Projects
- **[Docling](https://github.com/docling-project/docling)** - Document processing engine
- **[Pydantic](https://pydantic.dev)** - Data validation library
- **[NetworkX](https://networkx.org/)** - Graph library

---

## Next Steps

1. **[Install Docling Graph →](02-installation/index.md)**
2. **[Follow the Quick Start →](09-examples/quickstart.md)**
3. **[Create Your First Template →](03-schema-definition/index.md)**
4. **[Explore Examples →](09-examples/index.md)**

---

## Need Help?

- **Installation Issues**: See [Installation Guide](02-installation/index.md)
- **Template Questions**: See [Schema Definition](03-schema-definition/index.md)
- **Configuration Help**: See [Pipeline Configuration](04-pipeline-configuration/index.md)
- **Error Messages**: See [Error Handling](10-advanced/error-handling.md)

---

<p align="center">
  <strong>Ready to get started?</strong><br>
  <a href="02-installation/index.md">Install Docling Graph →</a>
</p>