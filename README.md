<p align="center"><br>
  <a href="https://github.com/IBM/docling-graph">
    <img loading="lazy" alt="Docling Graph" src="docs/assets/logo.png" width="280"/>
  </a>
</p>

# Docling Graph

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://ibm.github.io/docling-graph)
[![Docling](https://img.shields.io/badge/Docling-VLM-red)](https://github.com/docling-project/docling)
[![PyPI version](https://img.shields.io/pypi/v/docling-graph?include_prereleases)](https://pypi.org/project/docling-graph/)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-red)](https://networkx.org/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![Typer](https://img.shields.io/badge/Typer-CLI-purple)](https://typer.tiangolo.com/)
[![Rich](https://img.shields.io/badge/Rich-terminal-purple)](https://github.com/Textualize/rich)
[![vLLM](https://img.shields.io/badge/vLLM-compatible-brightgreen)](https://vllm.ai/)
[![Ollama](https://img.shields.io/badge/Ollama-compatible-brightgreen)](https://ollama.ai/)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-003778?logo=linuxfoundation&logoColor=fff&color=0094ff&labelColor=003778)](https://lfaidata.foundation/projects/)
[![License MIT](https://img.shields.io/github/license/IBM/docling-graph)](https://opensource.org/licenses/MIT)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/11598/badge)](https://www.bestpractices.dev/projects/11598)

Docling-Graph converts documents into validated **Pydantic** objects and then into a **directed knowledge graph**, with exports to CSV or Cypher and both static and interactive visualizations.

This transformation of unstructured documents into validated knowledge graphs with precise semantic relationships, essential for complex domains like **chemistry, finance, and legal** where AI systems must understand exact entity connections (e.g., chemical compounds and their reactions, financial instruments and their dependencies, physical properties and their measurements) rather than approximate text vectors, **enabling explainable reasoning over technical document collections**.

The toolkit supports two extraction families: **local VLM** via Docling and **LLM-based extraction** via local (vLLM, Ollama) or API providers (Mistral, OpenAI, Gemini, IBM WatsonX), all orchestrated by a flexible, config-driven pipeline.



## Key Capabilities

- **🧠 Extraction**:
  - Local `VLM` (Docling's information extraction pipeline - ideal for small documents with key-value focus)  
  - `LLM` (local via vLLM/Ollama or remote via Mistral/OpenAI/Gemini/IBM WatsonX API)
  - `Hybrid Chunking` Leveraging Docling's segmentation with semantic LLM chunking for more context-aware extraction
  - `Page-wise` or `whole-document` conversion strategies for flexible processing
- **🔨 Graph Construction**:
  - Markdown to Graph: Convert validated Pydantic instances to a `NetworkX DiGraph` with rich edge metadata and stable node IDs
  - Smart Merge: Combine multi-page documents into a single Pydantic instance for unified processing
  - Modular graph module with enhanced type safety and configuration
- **📦 Export**:
  - `Docling Document` exports (JSON format with full document structure)
  - `Markdown` exports (full document and per-page options)
  - `CSV` compatible with `Neo4j` admin import  
  - `Cypher` script generation for bulk ingestion
  - `JSON` export for general-purpose graph data
- **📊 Visualization**:
  - Interactive `HTML` visualization in full-page browser view with enhanced node/edge exploration
  - Detailed `MARKDOWN` report with graph nodes content and edges

### Coming Soon

* 🪜 **Multi-Stage Extraction:** Define `extraction_stage` in templates to control multi-pass extraction.
* 🧩 **Interactive Template Builder:** Guided workflows for building Pydantic templates.
* 🧬 **Ontology-Based Templates:** Match content to the best Pydantic template using semantic similarity.
* ✍🏻 **Flexible Inputs:** Accepts `text`, `markdown`, and `DoclingDocument` directly.
* + **Batch Optimization:** Faster GPU inference with better memory handling.
* 💾 **Graph Database Integration:** Export data straight into `Neo4j`, `ArangoDB`, and similar databases.



## Quick Start

### Requirements

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/IBM/docling-graph
cd docling-graph

# Install with uv (choose your option)
uv sync                    # Minimal: Core + VLM only
uv sync --extra all        # Full: All features
uv sync --extra local      # Local LLM (vLLM, Ollama)
uv sync --extra remote     # Remote APIs (Mistral, OpenAI, Gemini)
uv sync --extra watsonx    # IBM WatsonX support
```

For detailed installation instructions, see [Installation Guide](docs/02-installation/index.md).

### API Key Setup (Remote Inference)

```bash
export OPENAI_API_KEY="..."        # OpenAI
export MISTRAL_API_KEY="..."       # Mistral
export GEMINI_API_KEY="..."        # Google Gemini

# IBM WatsonX
export WATSONX_API_KEY="..."       # IBM WatsonX API Key
export WATSONX_PROJECT_ID="..."    # IBM WatsonX Project ID
export WATSONX_URL="..."           # IBM WatsonX URL (optional)
```

### Basic Usage

#### Python API

```python
from docling_graph import PipelineConfig
from docs.examples.templates.rheology_research import Research

# Create configuration
config = PipelineConfig(
    source="docs/examples/data/research_paper/rheology.pdf",
    template=Research,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    provider_override="mistral",
    model_override="mistral-medium-latest",
    use_chunking=True,
    output_dir="outputs/research"
)

# Run pipeline
config.run()
```

#### CLI

```bash
# Initialize configuration
uv run docling-graph init

# Convert document
uv run docling-graph convert "document.pdf" \
    --template "templates.MyTemplate" \
    --output-dir "outputs/my_graph"

# Visualize results
uv run docling-graph inspect outputs/my_graph
```

For more examples, see [Examples](docs/09-examples/index.md).



## Pydantic Templates

Templates define both the **extraction schema** and the resulting **graph structure**.

```python
from pydantic import BaseModel, Field
from docling_graph.utils import edge

class Person(BaseModel):
    """Person entity with stable ID."""
    model_config = {
        'is_entity': True,
        'graph_id_fields': ['last_name', 'date_of_birth']
    }
    
    first_name: str = Field(description="Person's first name")
    last_name: str = Field(description="Person's last name")
    date_of_birth: str = Field(description="Date of birth (YYYY-MM-DD)")

class Organization(BaseModel):
    """Organization entity."""
    model_config = {'is_entity': True}
    
    name: str = Field(description="Organization name")
    employees: list[Person] = edge("EMPLOYS", description="List of employees")
```

For complete guidance, see:
- [Schema Definition Guide](docs/03-schema-definition/index.md)
- [Pydantic Templates Tutorial](docs/03-schema-definition/pydantic-basics.md)
- [Example Templates](docs/examples/templates/)



## Documentation

Comprehensive documentation can be found on [Docling Graph's Page](https://ibm.github.io/docling-graph/).

### Documentation Structure

The documentation follows the docling-graph pipeline stages:

1. [Introduction](docs/01-introduction/index.md) - Overview and core concepts
2. [Installation](docs/02-installation/index.md) - Setup and environment configuration
3. [Schema Definition](docs/03-schema-definition/index.md) - Creating Pydantic templates
4. [Pipeline Configuration](docs/04-pipeline-configuration/index.md) - Configuring the extraction pipeline
5. [Extraction Process](docs/05-extraction-process/index.md) - Document conversion and extraction
6. [Graph Management](docs/06-graph-management/index.md) - Exporting and visualizing graphs
7. [CLI Reference](docs/07-cli/index.md) - Command-line interface guide
8. [Python API](docs/08-api/index.md) - Programmatic usage
9. [Examples](docs/09-examples/index.md) - Working code examples
10. [Advanced Topics](docs/10-advanced/index.md) - Performance, testing, error handling
11. [API Reference](docs/11-reference/index.md) - Detailed API documentation
12. [Development](docs/12-development/index.md) - Contributing and development guide



## Examples

Explore working examples in [docs/examples/](docs/examples/):

- **VLM Extraction**: [Image](docs/examples/scripts/01_vlm_from_image.py) | [PDF](docs/examples/scripts/02_vlm_from_pdf_page.py)
- **LLM Extraction**: [Remote API](docs/examples/scripts/03_llm_remote_api.py) | [Local Ollama](docs/examples/scripts/04_llm_local_ollama.py)
- **Advanced**: [Consolidation](docs/examples/scripts/05_llm_with_consolidation.py) | [One-to-One](docs/examples/scripts/06_llm_one_to_one.py)
- **CLI Recipes**: [Common Workflows](docs/examples/scripts/10_cli_recipes.md)

### Example Templates

- [Invoice](docs/examples/templates/invoice.py) - Financial document extraction
- [ID Card](docs/examples/templates/id_card.py) - Identity document parsing
- [Insurance](docs/examples/templates/insurance.py) - Insurance policy extraction
- [Research Paper](docs/examples/templates/rheology_research.py) - Scientific document analysis



## Contributing

We welcome contributions! Please see:

- [Contributing Guidelines](.github/CONTRIBUTING.md) - How to contribute
- [Development Guide](docs/12-development/index.md) - Development setup
- [GitHub Workflow](docs/12-development/github-workflow.md) - Branch strategy and CI/CD

### Development Setup

```bash
# Clone and setup
git clone https://github.com/IBM/docling-graph
cd docling-graph

# Install with dev dependencies
uv sync --extra all --extra dev

# Run Execute pre-commit checks
uv run pre-commit run --all-files
```



## License

MIT License - see [LICENSE](LICENSE) for details.



## Acknowledgments

- Powered by [Docling](https://github.com/docling-project/docling) for advanced document processing
- Uses [Pydantic](https://pydantic.dev) for data validation
- Graph generation powered by [NetworkX](https://networkx.org/)
- Visualizations powered by [Cytoscape.js](https://js.cytoscape.org/)
- CLI powered by [Typer](https://typer.tiangolo.com/) and [Rich](https://github.com/Textualize/rich)



## IBM ❤️ Open Source AI

Docling Graph has been brought to you by IBM.
