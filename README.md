<p align="center"><br>
  <a href="https://github.com/ayoub-ibm/docling-graph">
    <img loading="lazy" alt="Docling Graph" src="docs/assets/logo.png" width="300"/>
  </a>
</p>

# Docling Graph

[![Docs](https://img.shields.io/badge/Docs-coming%20soon-brightgreen)](https://github.com/ayoub-ibm/docling-graph)
[![Docling](https://img.shields.io/badge/Docling-VLM-red)](https://github.com/docling-project/docling)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-red)](https://networkx.org/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![Mypy](https://img.shields.io/badge/Type--checked-mypy-blue?logo=python&logoColor=white)](https://mypy.readthedocs.io/en/stable)
[![Typer](https://img.shields.io/badge/Typer-CLI-purple)](https://typer.tiangolo.com/)
[![Rich](https://img.shields.io/badge/Rich-terminal-purple)](https://github.com/Textualize/rich)
[![vLLM](https://img.shields.io/badge/vLLM-compatible-brightgreen)](https://vllm.ai/)
[![Ollama](https://img.shields.io/badge/Ollama-compatible-brightgreen)](https://ollama.ai/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License MIT](https://img.shields.io/github/license/ayoub-ibm/docling-graph)](https://opensource.org/licenses/MIT)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-003778?logo=linuxfoundation&logoColor=fff&color=0094ff&labelColor=003778)](https://lfaidata.foundation/projects/)

Docling-Graph converts documents into validated **Pydantic** objects and then into a **directed knowledge graph**, with exports to CSV or Cypher and both static and interactive visualizations.  

The toolkit supports two extraction families: **local VLM** via Docling and **LLM-based extraction** via local (vLLM, Ollama) or API providers (Mistral, OpenAI, Gemini), all orchestrated by a flexible, config-driven pipeline.



## Key Capabilities

- **🧠 Extraction**:
  - Local `VLM` (Docling's information extraction pipeline - ideal for small documents with key-value focus)  
  - `LLM` (local via vLLM/Ollama or remote via Mistral/OpenAI/Gemini API)  
  - Page-wise or whole-document conversion strategies
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

* ✂️ **Smart Chunking:** Structure-aware text splitting that keeps document logic intact.
* 🧬 **Ontology-Based Templates:** Match content to the best Pydantic template using semantic similarity.
* ✍🏻 **Flexible Inputs:** Accepts `text`, `markdown`, and `DoclingDocument` directly.
* 🧩 **Interactive Template Builder:** Guided workflows for building Pydantic templates.
* ⚡ **Batch Optimization:** Faster GPU inference with better memory handling.
* 💾 **Graph Database Integration:** Export data straight into `Neo4j`, `ArangoDB`, and similar databases.



## Initial Setup

### Requirements

- Python 3.10 or higher
- UV package manager

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/ayoub-ibm/docling-graph
cd docling-graph
```

#### 2. Install Dependencies

Choose the installation option that matches your use case:

| Option          | Command                   | Description                                                         |
| :---            | :---                      | :---                                                                |
| **Minimal**     | `uv sync`                 | Includes core VLM features (Docling), **no** LLM inference          |
| **Full**        | `uv sync --extra all`     | Includes **all** features, VLM, and all local/remote LLM providers  |
| **Local LLM**   | `uv sync --extra local`   | Adds support for vLLM and Ollama (requires GPU for vLLM)            |
| **Remote API**  | `uv sync --extra remote`  | Adds support for Mistral, OpenAI, and Google Gemini APIs            |


#### 3. OPTIONAL - GPU Support (PyTorch)

Follow the steps in [this guide](docs/guides/setup_with_gpu_support.md) to install PyTorch with NVIDIA GPU (CUDA) support.



### API Key Setup (for Remote Inference)

If you’re using **Option C** or **Option D** (remote/cloud inference), set your API keys for the providers you plan to use:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Mistral
export MISTRAL_API_KEY="..."

# Google Gemini
export GEMINI_API_KEY="..."
```

Alternatively, add them to your `.env` file.



## Getting Started

Docling Graph is primarily driven by its **CLI**, but you can easily integrate the core pipeline into Python scripts.

### 1. Python Example

To run a conversion programmatically, you define a configuration dictionary and pass it to the `run_pipeline` function. This example uses a **remote LLM API** in a `many-to-one` mode for a single multi-page document:

```python
from docling_graph import run_pipeline, PipelineConfig
from examples.templates.battery_research import Research  # Pydantic model to use as an extraction template

# Create typed config
config = PipelineConfig(
    source="examples/data/battery_research/bauer2014.pdf",
    template=Research,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    provider_override="mistral",              # Specify your preferred provider and ensure its API key is set
    model_override="mistral-medium-latest",   # Specify your preferred LLM model
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

A wizard will walk you through setting up the right configfor your use case.

```bash
uv run docling-graph init
```

#### 2.2. Run Conversion

You can use: `docling-graph convert --help` to see the full list of available options and usage details

```bash
# uv run docling-graph convert <SOURCE_FILE_PATH> --template "<TEMPLATE_DOTTED_PATH>" [OPTIONS]

uv run docling-graph convert "examples/data/battery_research/bauer2014.pdf" \
    --template "examples.templates.battery_research.Research" \
    --output-dir "outputs/battery_research"  \
    --processing-mode "many-to-one"
```

#### 2.3. Run Conversion

```bash
# uv run docling-graph inspect <CONVERT_OUTPUT_PATH> [OPTIONS]

uv run docling-graph inspect outputs/battery_research
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

For complete guidance, see: [Pydantic Templates for Knowledge Graph Extraction](docs/guides/pydantic_templates_for_knowledge_graph_extraction.md)



## Documentation

*Work In Progress*



## Examples

*Work In Progress*



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