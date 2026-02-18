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
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-003778?logo=linuxfoundation&logoColor=fff&color=0094ff&labelColor=003778)](https://lfaidata.foundation/projects/)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/11598/badge)](https://www.bestpractices.dev/projects/11598)



Docling-Graph turns documents into validated **Pydantic** objects, then builds a **directed knowledge graph** with explicit semantic relationships.

This transformation enables high-precision use cases in **chemistry, finance, and legal** domains, where AI must capture exact entity connections (compounds and reactions, instruments and dependencies, properties and measurements) **rather than rely on approximate text embeddings**.

This toolkit supports two extraction paths: **local VLM extraction** via Docling, and **LLM-based extraction** routed through **LiteLLM** for local runtimes (vLLM, Ollama) and API providers (Mistral, OpenAI, Gemini, IBM WatsonX), all orchestrated through a flexible, config-driven pipeline.



## Key Capabilities

- **✍🏻 Input formats** - [Docling](https://docling-project.github.io/docling/usage/supported_formats/)’s supported inputs: PDF, images, markdown, Office, HTML, and more.

- **🧠 Extraction** - [VLM](docs/fundamentals/pipeline-configuration/backend-selection.md) or [LLM](docs/fundamentals/pipeline-configuration/backend-selection.md), chunking and processing modes.

- **💎 Graphs** - Pydantic → [NetworkX](docs/fundamentals/graph-management/graph-conversion.md) directed graphs with stable IDs and edge metadata.

- **📦 Export** - [CSV](docs/fundamentals/graph-management/export-formats.md#csv-export), [Cypher](docs/fundamentals/graph-management/export-formats.md#cypher-export), and other KG-friendly formats.

- **🔍 Visualization** - [Interactive HTML](docs/fundamentals/graph-management/visualization.md) and Markdown reports.

### Latest Changes

- **🪜 Multi-pass extraction** - [Delta](docs/fundamentals/extraction-process/delta-extraction.md) and [staged](docs/fundamentals/extraction-process/staged-extraction.md) contracts (experimental).

- **📐 Structured extraction** - LLM output is schema-enforced by default; see [CLI](docs/usage/cli/convert-command.md#structured-output-mode) and [API](docs/usage/api/llm-model-config.md) to disable.

- **✨ LiteLLM** - Single [interface](docs/reference/llm-clients.md) for vLLM, OpenAI, Mistral, WatsonX, and more.

- **🐛 Trace capture** - [Debug exports](docs/usage/advanced/trace-data-debugging.md) for extraction and fallback diagnostics.

### Coming Soon

* 🧩 **Interactive Template Builder** - Guided workflows for building Pydantic templates.

* 🧲 **Ontology-Based Templates** - Match content to the best Pydantic template using semantic similarity.

* 💾 **Graph Database Integration** - Export data straight into `Neo4j`, `ArangoDB`, and similar databases.



## Quick Start

### Requirements

- Python 3.10 or higher

### Installation

```bash
pip install docling-graph
```

This installs the core package with VLM support and LiteLLM for LLM providers. For detailed installation instructions (including optional extras and GPU setup), see [Installation Guide](docs/fundamentals/installation/index.md).

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

#### CLI

```bash
# Initialize configuration
docling-graph init

# Convert document from URL (each line except the last must end with \)
docling-graph convert "https://arxiv.org/pdf/2207.02720" \
    --template "docs.examples.templates.rheology_research.ScholarlyRheologyPaper" \
    --processing-mode "many-to-one" \
    --extraction-contract "staged" \
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
    "extraction_contract": "staged",  # robust for smaller models
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

For debugging, use `--debug` with the CLI to save intermediate artifacts to disk; see [Trace Data & Debugging](docs/usage/advanced/trace-data-debugging.md). For more examples, see [Examples](docs/usage/examples/index.md).



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
- [Schema Definition Guide](docs/fundamentals/schema-definition/index.md)
- [Template Basics](docs/fundamentals/schema-definition/template-basics.md)
- [Example Templates](docs/examples/README.md)



## Documentation

Comprehensive documentation can be found on [Docling Graph's Page](https://ibm.github.io/docling-graph/).

### Documentation Structure

The documentation follows the docling-graph pipeline stages:

1. [Introduction](docs/introduction/index.md) - Overview and core concepts
2. [Installation](docs/fundamentals/installation/index.md) - Setup and environment configuration
3. [Schema Definition](docs/fundamentals/schema-definition/index.md) - Creating Pydantic templates
4. [Pipeline Configuration](docs/fundamentals/pipeline-configuration/index.md) - Configuring the extraction pipeline
5. [Extraction Process](docs/fundamentals/extraction-process/index.md) - Document conversion and extraction
6. [Graph Management](docs/fundamentals/graph-management/index.md) - Exporting and visualizing graphs
7. [CLI Reference](docs/usage/cli/index.md) - Command-line interface guide
8. [Python API](docs/usage/api/index.md) - Programmatic usage
9. [Examples](docs/usage/examples/index.md) - Working code examples
10. [Advanced Topics](docs/usage/advanced/index.md) - Performance, testing, error handling
11. [API Reference](docs/reference/index.md) - Detailed API documentation
12. [Community](docs/community/index.md) - Contributing and development guide



## Contributing

We welcome contributions! Please see:

- [Contributing Guidelines](.github/CONTRIBUTING.md) - How to contribute
- [Development Guide](docs/community/index.md) - Development setup

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
- [SpaCy](https://spacy.io/) - semantic entity resolution in delta extraction
- [Cytoscape](https://js.cytoscape.org/) - interactive graph visualization



## IBM ❤️ Open Source AI

Docling Graph has been brought to you by IBM.
