<p align="center"><br>
  <a href="https://github.com/docling-project/docling-graph">
    <img loading="lazy" alt="Docling Graph" src="docs/assets/logo.png" width="280"/>
  </a>
</p>

# Docling Graph

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://docling-project.github.io/docling-graph/)
[![PyPI version](https://img.shields.io/pypi/v/docling-graph?cacheSeconds=300)](https://pypi.org/project/docling-graph/)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License MIT](https://img.shields.io/github/license/docling-project/docling-graph)](https://opensource.org/licenses/MIT)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![Docling](https://img.shields.io/badge/Docling-VLM-red)](https://github.com/docling-project/docling)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-red)](https://networkx.org/)
[![Typer](https://img.shields.io/badge/Typer-CLI-purple)](https://typer.tiangolo.com/)
[![Rich](https://img.shields.io/badge/Rich-terminal-purple)](https://github.com/Textualize/rich)
[![vLLM](https://img.shields.io/badge/vLLM-compatible-brightgreen)](https://vllm.ai/)
[![Ollama](https://img.shields.io/badge/Ollama-compatible-brightgreen)](https://ollama.ai/)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/11598/badge)](https://www.bestpractices.dev/projects/11598)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-003778?logo=linuxfoundation&logoColor=fff&color=0094ff&labelColor=003778)](https://lfaidata.foundation/projects/)



Docling-Graph turns documents into validated **Pydantic** objects, then builds a **directed knowledge graph** with explicit semantic relationships.

This transformation enables high-precision use cases in **chemistry, finance, and legal** domains, where AI must capture exact entity connections (compounds and reactions, instruments and dependencies, properties and measurements) **rather than rely on approximate text embeddings**.

This toolkit supports two extraction paths: **local VLM extraction** via Docling, and **LLM-based extraction** routed through **LiteLLM** for local runtimes (vLLM, Ollama) and API providers (Mistral, OpenAI, Gemini, IBM watsonx), all orchestrated through a flexible, config-driven pipeline.



## Key Capabilities

- **✍🏻 Input formats:** [Docling](https://docling-project.github.io/docling/usage/supported_formats/)’s supported inputs: PDF, images, markdown, Office, HTML, and more.

- **🧠 Extraction:** [LLM](https://docling-project.github.io/docling-graph/fundamentals/pipeline-configuration/backend-selection/) or [VLM](https://docling-project.github.io/docling-graph/fundamentals/pipeline-configuration/backend-selection/) backends, with [chunking](https://docling-project.github.io/docling-graph/fundamentals/extraction-process/chunking-strategies/) and [processing modes](https://docling-project.github.io/docling-graph/fundamentals/pipeline-configuration/processing-modes/).

- **💎 Graphs:** Pydantic to [NetworkX](https://docling-project.github.io/docling-graph/fundamentals/graph-management/graph-conversion/) directed graphs with stable IDs, edge and [provenance](https://docling-project.github.io/docling-graph/fundamentals/graph-management/provenance/) metadata.

- **📦 Export:** [CSV](https://docling-project.github.io/docling-graph/fundamentals/graph-management/export-formats/#csv-export), [Cypher](https://docling-project.github.io/docling-graph/fundamentals/graph-management/export-formats/#cypher-export), and other KG-friendly formats.

- **🔍 Visualization:** [Interactive HTML](https://docling-project.github.io/docling-graph/fundamentals/graph-management/visualization/) and Markdown reports.

- **🐛 Trace capture:** [Debug exports](https://docling-project.github.io/docling-graph/usage/advanced/trace-data-debugging/) for extraction and fallback diagnostics.

### Latest Changes

- **✨ Dense extraction:** Advanced [skeleton-then-flesh](https://docling-project.github.io/docling-graph/fundamentals/extraction-process/dense-extraction/) extraction mode for complex documents.

- **📍 Data grounding:** Deterministic [provenance](https://docling-project.github.io/docling-graph/fundamentals/graph-management/provenance/) ledger with no extra LLM calls.

### Coming Soon

* 🦆 **DocLang Support:** Leverage the [DocLang](https://doclang.ai/) input format to unlock advanced document structure parsing.

* 🧩 **Interactive Template Builder:** Guided workflows for building Pydantic templates.

* 🔗 **Graph Fusion:** Combine and reconcile disparate knowledge graphs into a unified structure.

* 🧲 **Ontology-Based Templates:** Match content to the best Pydantic template using semantic similarity.



## Quick Start

### Requirements

- Python 3.10 or higher

### Installation

```bash
pip install docling-graph
```

This installs the core package with VLM support and LiteLLM for LLM providers. For detailed installation instructions (including optional extras and GPU setup), see [Installation Guide](https://docling-project.github.io/docling-graph/fundamentals/installation/).

### API Key Setup (Remote Inference)

```bash
export OPENAI_API_KEY="..."        # OpenAI
export MISTRAL_API_KEY="..."       # Mistral
export GEMINI_API_KEY="..."        # Google Gemini

# IBM watsonx
export WATSONX_API_KEY="..."       # IBM watsonx API Key
export WATSONX_PROJECT_ID="..."    # IBM watsonx Project ID
export WATSONX_URL="..."           # IBM watsonx URL (optional)
```

### Basic Usage

#### CLI

```bash
# Initialize configuration
docling-graph init

# Convert document from URL (each line except the last must end with \)
docling-graph convert "https://arxiv.org/pdf/2207.02720" \
    --template "docs.examples.templates.rheology_research.ScholarlyRheologyPaper" \
    --processing-mode "many-to-one" \
    --extraction-contract "dense" \
    --debug

# Visualize results
docling-graph inspect outputs
```

#### Python API - Default Behavior

```python
from docling_graph import run_pipeline, PipelineContext
from docs.examples.templates.rheology_research import ScholarlyRheologyPaper

# Create configuration
config = {
    "source": "https://arxiv.org/pdf/2207.02720",
    "template": ScholarlyRheologyPaper,
    "backend": "llm",
    "inference": "remote",
    "processing_mode": "many-to-one",
    "extraction_contract": "dense",
    "provider_override": "mistral",
    "model_override": "mistral-medium-latest",
    "structured_output": True,  # default
    "use_chunking": True,
}

# Run pipeline - returns data directly, no files written to disk
context: PipelineContext = run_pipeline(config)

# Access results
graph = context.knowledge_graph
models = context.extracted_models
metadata = context.graph_metadata

print(f"Extracted {len(models)} model(s)")
print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

Every node above also carries a deterministic `__provenance__` attribute by default (`provenance="standard"`), pointing back to the source chunk and page it was extracted from — no extra LLM calls involved. See [Data Grounding & Provenance](https://docling-project.github.io/docling-graph/fundamentals/graph-management/provenance/).

For debugging, use `--debug` with the CLI to save intermediate artifacts to disk; see [Trace Data & Debugging](https://docling-project.github.io/docling-graph/usage/advanced/trace-data-debugging/). For more examples, see [Examples](https://docling-project.github.io/docling-graph/usage/examples/).



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
- [Schema Definition Guide](https://docling-project.github.io/docling-graph/fundamentals/schema-definition/)
- [Template Basics](https://docling-project.github.io/docling-graph/fundamentals/schema-definition/template-basics/)
- [Example Templates](docs/examples/README.md)



## Documentation

Comprehensive documentation can be found on [Docling Graph's Page](https://docling-project.github.io/docling-graph/).

### Documentation Structure

The documentation follows the docling-graph pipeline stages:

1. [Introduction](https://docling-project.github.io/docling-graph/introduction/) - Overview and core concepts
2. [Installation](https://docling-project.github.io/docling-graph/fundamentals/installation/) - Setup and environment configuration
3. [Schema Definition](https://docling-project.github.io/docling-graph/fundamentals/schema-definition/) - Creating Pydantic templates
4. [Pipeline Configuration](https://docling-project.github.io/docling-graph/fundamentals/pipeline-configuration/) - Configuring the extraction pipeline
5. [Extraction Process](https://docling-project.github.io/docling-graph/fundamentals/extraction-process/) - Document conversion and extraction
6. [Graph Management](https://docling-project.github.io/docling-graph/fundamentals/graph-management/) - Converting, grounding, exporting, and visualizing graphs
7. [CLI Reference](https://docling-project.github.io/docling-graph/usage/cli/) - Command-line interface guide
8. [Python API](https://docling-project.github.io/docling-graph/usage/api/) - Programmatic usage
9. [Examples](https://docling-project.github.io/docling-graph/usage/examples/) - Working code examples
10. [Advanced Topics](https://docling-project.github.io/docling-graph/usage/advanced/) - Performance, testing, error handling
11. [API Reference](https://docling-project.github.io/docling-graph/reference/) - Detailed API documentation
12. [Community](https://docling-project.github.io/docling-graph/community/) - Contributing and development guide



## Contributing

We welcome contributions! Please see:

- [Contributing Guidelines](.github/CONTRIBUTING.md) - How to contribute
- [Development Guide](https://docling-project.github.io/docling-graph/community/) - Development setup

### Development Setup

```bash
# Clone and setup
git clone https://github.com/docling-project/docling-graph
cd docling-graph

# Install with dev dependencies
uv sync --extra dev

# Run Execute pre-commit checks
uv run pre-commit run --all-files
```



## License

MIT License - see [LICENSE](LICENSE) for details.



## Acknowledgments

Docling Graph builds on outstanding open-source projects:

- [Docling](https://github.com/docling-project/docling) - document conversion and VLM extraction
- [Pydantic](https://pydantic.dev) - schema definition and validation
- [NetworkX](https://networkx.org/) - graph construction and analysis
- [LiteLLM](https://github.com/BerriAI/litellm) - unified LLM provider interface
- [Cytoscape](https://js.cytoscape.org/) - interactive graph visualization



## IBM ❤️ Open Source AI

Docling Graph has been brought to you by IBM.
