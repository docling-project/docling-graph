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
* ⚡ **Batch Optimization:** Faster GPU inference with better memory handling.
* 💾 **Graph Database Integration:** Export data straight into `Neo4j`, `ArangoDB`, and similar databases.



## Initial Setup

### Requirements

- Python 3.10 or higher
- UV package manager

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/IBM/docling-graph
cd docling-graph
```

#### 2. Install Dependencies

Choose the installation option that matches your use case:

| Option          | Command                   | Description                                                                |
| :---            | :---                      | :---                                                                       |
| **Minimal**     | `uv sync`                 | Includes core VLM features (Docling), **no** LLM inference                 |
| **Full**        | `uv sync --extra all`     | Includes **all** features, VLM, and all local/remote LLM providers         |
| **Local LLM**   | `uv sync --extra local`   | Adds support for vLLM and Ollama (requires GPU for vLLM)                   |
| **Remote API**  | `uv sync --extra remote`  | Adds support for Mistral, OpenAI, Gemini, and IBM WatsonX APIs            |
| **WatsonX**     | `uv sync --extra watsonx` | Adds support for IBM WatsonX foundation models (Granite, Llama, Mixtral)   |


#### 3. OPTIONAL - GPU Support (PyTorch)

Follow the steps in [this guide](docs/guides/setup_with_gpu_support.md) to install PyTorch with NVIDIA GPU (CUDA) support.



### API Key Setup (for Remote Inference)

If you're using remote/cloud inference, set your API keys for the providers you plan to use:

```bash
export OPENAI_API_KEY="..."        # OpenAI
export MISTRAL_API_KEY="..."       # Mistral
export GEMINI_API_KEY="..."        # Google Gemini

# IBM WatsonX
export WATSONX_API_KEY="..."       # IBM WatsonX API Key
export WATSONX_PROJECT_ID="..."    # IBM WatsonX Project ID
export WATSONX_URL="..."           # IBM WatsonX URL (optional, defaults to US South)
```

On Windows, replace `export` with `set` in Command Prompt or `$env:` in PowerShell.

Alternatively, add them to your `.env` file.

**Note:** For IBM WatsonX setup and available models, see the [WatsonX Integration Guide](docs/guides/watsonx_integration.md).



## Getting Started

Docling Graph is primarily driven by its **CLI**, but you can easily integrate the core pipeline into Python scripts.

### 1. Python Example

To run a conversion programmatically, you define a configuration dictionary and pass it to the `run_pipeline` function. This example uses a **remote LLM API** in a `many-to-one` mode for a single multi-page document:

```python
from docling_graph import run_pipeline, PipelineConfig
from docs.examples.templates.rheology_research import Research  # Pydantic model to use as an extraction template

# Create typed config
config = PipelineConfig(
    source="docs/examples/data/research_paper/rheology.pdf",
    template=Research,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    provider_override="mistral",              # Specify your preferred provider and ensure its API key is set
    model_override="mistral-medium-latest",   # Specify your preferred LLM model
    use_chunking=True,                        # Enable docling's hybrid chunker
    llm_consolidation=False,                  # If False, programmatically merge batch-extracted dictionaries
    output_dir="outputs/battery_research"
)

try:
    run_pipeline(config)
    print(f"\nExtraction complete! Graph data saved to: {config.output_dir}")
except Exception as e:
    print(f"An error occurred: {e}")
```


### 2. CLI Example

Use the command-line interface for quick conversions and inspections. The following command runs the conversion using the local VLM backend and outputs a graph ready for Neo4j import:

#### 2.1. Initialize Configuration

A wizard will walk you through setting up the right config for your use case.

```bash
uv run docling-graph init
```

**New in v0.3.0**: The init command now features **75-85% faster** performance with intelligent dependency caching! The first run checks for installed dependencies, but subsequent runs are nearly instant.

**Tip**: Use `uv run docling-graph --verbose init` for detailed logging during setup.


#### 2.2. Run Conversion

You can use `uv run docling-graph convert --help` to see the full list of available options and usage details.

```bash
# Basic usage
uv run docling-graph convert <SOURCE_FILE_PATH> --template "<TEMPLATE_DOTTED_PATH>" [OPTIONS]

# Example: Convert research paper
uv run docling-graph convert "docs/examples/data/research_paper/rheology.pdf" \
    --template "docs.examples.templates.rheology_research.Research" \
    --output-dir "outputs/battery_research" \
    --processing-mode "many-to-one" \
    --use-chunking \
    --no-llm-consolidation
```

**New CLI Features**:
- `--verbose` / `-v`: Enable detailed logging for debugging
- `--version`: Show version and exit
- Better error messages with actionable details

```bash
# Debug with verbose logging
uv run docling-graph --verbose convert document.pdf --template templates.Invoice

# Check version
uv run docling-graph --version
```

#### 2.3. Inspect Results

```bash
# Visualize the generated graph
uv run docling-graph inspect <CONVERT_OUTPUT_PATH> [OPTIONS]

# Example
uv run docling-graph inspect outputs/battery_research

# With custom output
uv run docling-graph inspect outputs/battery_research --output graph_viz.html
```



## Pydantic Templates

Templates are the foundation of Docling Graph, defining both the **extraction schema** and the resulting **graph structure**.

  * Use `is_entity=True` in `model_config` to explicitly mark a class as a graph node.
  * Leverage `model_config.graph_id_fields` to create stable, readable node IDs (natural keys).
  * Use the `Edge()` helper to define explicit relationships between entities.

**Example:**

```python
from pydantic import BaseModel, Field
from typing import Optional

class Person(BaseModel):
    """Person entity with stable ID based on name and DOB."""
    model_config = {
        'is_entity': True,
        'graph_id_fields': ['last_name', 'date_of_birth']
    }
    
    first_name: str = Field(description="Person's first name")
    last_name: str = Field(description="Person's last name")
    date_of_birth: str = Field(description="Date of birth (YYYY-MM-DD)")
```

Reference Pydantic [templates](docs/examples/templates) are available to help you get started quickly.

For complete guidance, see: [Pydantic Templates for Knowledge Graph Extraction](docs/guides/create_pydantic_templates_for_kg_extraction.md)



## Documentation

Comprehensive documentation can be found on [Docling Graph's Page](https://ibm.github.io/docling-graph/).

### Key Resources

- [Getting Started](https://ibm.github.io/docling-graph/getting-started/installation/) - Installation and quick start guides
- [Concepts](https://ibm.github.io/docling-graph/concepts/) - Core concepts and architecture
- [Guides](https://ibm.github.io/docling-graph/guides/create_pydantic_templates_for_kg_extraction/) - In-depth tutorials and best practices
- [Examples](https://ibm.github.io/docling-graph/examples/README/) - Working code examples
- [API Reference](https://ibm.github.io/docling-graph/api/pipeline/) - Detailed API documentation

You can also browse the documentation locally in the [`docs/`](docs/) directory.



## Examples

Get hands-on with Docling Graph [examples](docs/examples/scripts) to convert documents into knowledge graphs through `VLM` or `LLM`-based processing.

## License

MIT License - see [LICENSE](LICENSE) for details.



## Acknowledgments

- Powered by [Docling](https://github.com/docling-project/docling) for advanced document processing.
- Uses [Pydantic](https://pydantic.dev) for data validation.
- Graph generation powered by [NetworkX](https://networkx.org/).
- Visualizations powered by [Cytoscape.js](https://js.cytoscape.org/).
- CLI powered by [Typer](https://typer.tiangolo.com/) and [Rich](https://github.com/Textualize/rich).



## IBM ❤️ Open Source AI

Docling Graph has been brought to you by IBM.
