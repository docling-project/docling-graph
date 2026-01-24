# Docling Graph Documentation

<p align="center">
  <img src="assets/logo.png" alt="Docling Graph" width="280"/>
</p>

## What is Docling Graph?

Docling-Graph turns documents into validated **Pydantic** objects, then builds a **directed knowledge graph** with explicit semantic relationships.

This transformation enables high-precision use cases in **chemistry, finance, and legal** domains, where AI must capture exact entity connections (compounds and reactions, instruments and dependencies, properties and measurements) **rather than rely on approximate text embeddings**.

This toolkit supports two extraction paths: **local VLM extraction** via Docling, and **LLM-based extraction** using either local runtimes (vLLM, Ollama) or API providers (Mistral, OpenAI, Gemini, IBM WatsonX), all orchestrated through a flexible, config-driven pipeline.

---

### Key Features

- **✍🏻 Multi-Format Input**: Ingest PDFs, images, URLs, raw text, Markdown and more.
- **🧠 Flexible Extraction:** VLM or LLM-based (vLLM, Ollama, Mistral, Gemini, WatsonX, etc.)
- **🔨 Smart Graphs:** Convert Pydantic models to NetworkX graphs with stable node IDs
- **📦 Multiple Export:** CSV (Neo4j-compatible), Cypher scripts, JSON, Markdown
- **📊 Rich Visualizations:** Interactive HTML and detailed Markdown reports
- **⚙️ Type-Safe Configuration:** Pydantic-based validation

---

## Quick Navigation

### 🚀 Getting Started

<div class="grid cards" markdown>

- **[Installation →](fundamentals/installation/index.md)**

    Set up your environment with uv package manager

- **[Quick Start →](introduction/quickstart.md)**

    Run your first extraction in 5 minutes

- **[Architecture →](introduction/architecture.md)**

    Understand the pipeline stages and components

- **[Key Concepts →](introduction/key-concepts.md)**

    Learn how documents flow through the system

</div>

### 📚 Core Documentation

<div class="grid cards" markdown>

- **[Introduction](introduction/index.md)**

    Overview, architecture, and core concepts

- **[Fundamentals](fundamentals/index.md)**

    Installation, schema definition, pipeline configuration, extraction, and more

- **[Usage](usage/index.md)**

    CLI reference, Python API, examples, and advanced topics

- **[Reference](reference/index.md)**

    Detailed API documentation

- **[Community](community/index.md)**

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

The documentation is organized into **5 main sections** for a streamlined learning experience:

### 1. Introduction
Learn what Docling Graph is, how it works, and its architecture.

**[→ Go to Introduction](introduction/index.md)**

### 2. Fundamentals
Master the core concepts: installation, schema definition, pipeline configuration, extraction, and graph management.

**[→ Go to Fundamentals](fundamentals/index.md)**

### 3. Usage
Learn to use Docling Graph through CLI, Python API, examples, and advanced techniques.

**[→ Go to Usage](usage/index.md)**

### 4. Reference
Detailed API documentation for all modules and components.

**[→ Go to Reference](reference/index.md)**

### 5. Community
Contribute to the project and understand the development workflow.

**[→ Go to Community](community/index.md)**

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

**[→ See Invoice Template](usage/examples/invoice-extraction.md)**

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

**[→ See Research Template](usage/examples/research-paper.md)**

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

**[→ See ID Card Template](usage/examples/id-card.md)**

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

**[→ Learn More About Templates](fundamentals/schema-definition/index.md)**

### Pipeline Stages

1. **Template Loading**: Load and validate Pydantic templates
2. **Document Conversion**: Convert documents using Docling
3. **Chunking**: Split content into manageable pieces (optional)
4. **Extraction**: Extract structured data using VLM or LLM
5. **Consolidation**: Merge results (optional)
6. **Graph Construction**: Build NetworkX graph from Pydantic models
7. **Export**: Generate CSV, Cypher, JSON outputs
8. **Visualization**: Create interactive HTML and Markdown reports

**[→ Learn More About Key Concepts](introduction/key-concepts.md)**

### Extraction Backends

- **VLM (Vision-Language Model)**: Local extraction using Docling's NuExtract
- **LLM (Language Model)**: Text-based extraction using local (vLLM, Ollama) or remote APIs (Mistral, OpenAI, Gemini, WatsonX)
- **Model Capabilities**: Automatic classification into SIMPLE/STANDARD/ADVANCED tiers for optimized extraction

**[→ Learn More About Backends](fundamentals/extraction-process/extraction-backends.md)**
**[→ Learn More About Model Capabilities](fundamentals/extraction-process/model-capabilities.md)**

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

1. **[Install Docling Graph →](fundamentals/installation/index.md)**
2. **[Follow the Quick Start →](introduction/quickstart.md)**
3. **[Create Your First Template →](fundamentals/schema-definition/index.md)**
4. **[Explore Examples →](usage/examples/index.md)**

---

## Need Help?

- **Installation Issues**: See [Installation Guide](fundamentals/installation/index.md)
- **Template Questions**: See [Schema Definition](fundamentals/schema-definition/index.md)
- **Configuration Help**: See [Pipeline Configuration](fundamentals/pipeline-configuration/index.md)
- **Error Messages**: See [Error Handling](usage/advanced/error-handling.md)

---

<p align="center">
  <strong>Ready to get started?</strong><br>
  <a href="fundamentals/installation/index.md">Install Docling Graph →</a>
</p>