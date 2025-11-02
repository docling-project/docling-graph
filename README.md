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

- **Extraction**:
  - Local `VLM` (Docling's vision pipeline leveraging Granite-Docling)  
  - `LLM` (local via vLLM/Ollama or remote via Mistral/OpenAI/Gemini API)  
  - Page-wise or whole-document conversion strategies
- **Graph Construction**:
  - Markdown to Graph: Convert validated Pydantic instances to a `NetworkX DiGraph` with rich edge metadata and stable node IDs
  - Smart Merge: Combine multi-page documents into a single Pydantic instance for unified processing
  - Modular graph module with enhanced type safety and configuration
- **Export**:
  - `Docling Document` exports (JSON format with full document structure)
  - `Markdown` exports (full document and per-page options)
  - `CSV` compatible with `Neo4j` admin import  
  - `Cypher` script generation for bulk ingestion
  - `JSON` export for general-purpose graph data
- **Visualization**:
  - Interactive `HTML` visualization in full-page browser view with enhanced node/edge exploration
  - Publication-grade static images (`PNG`, `SVG`, `PDF`)
  - Detailed `MARKDOWN` report with graph nodes content and edges



## Initial Setup

### Requirements

- Python 3.10 or higher
- UV package manager (pip install uv)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/ayoub-ibm/docling-graph
cd docling-graph
```

#### 2. Install Dependencies

Choose the installation option that matches your use case:

**Option A: Minimal Installation** (Docling features only, no LLM inference)
```bash
uv sync
```

**Option B: Full Installation** (All features and providers)
```bash
uv sync --extra all
```

**Option C: Local LLM Inference** (Run LLMs on your machine)
```bash
uv sync --extra local
```

Includes:
- `vllm` - Fast local LLM inference (requires GPU)
- `ollama` - Run open-source models locally

**Option D: Remote API Inference** (Use cloud-based LLM APIs)
```bash
uv sync --extra remote
```

Includes:
- `mistral` - Mistral AI API
- `openai` - OpenAI API
- `gemini` - Google Gemini API


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



## Quick Start

### 1. Initialize Configuration (CLI)

```bash
uv run docling-graph init
```

### 2. Run Conversion (CLI)

```bash
uv run docling-graph convert <SOURCE_FILE_PATH> --template "<TEMPLATE_PATH>" [OPTIONS]
```

### 2. Run Conversion (CLI)

```bash
uv run docling-graph inspect <CONVERT_OUTPUT_PATH>
```


## Pydantic Templates

Templates define the schema for extraction and graph structure:

- Available templates: **invoices**, **French ID cards**, **insurance terms**
- Edges can be **implicit** (nested BaseModels) or **explicit** (generic Edge type)

### Tips

- Use `model_config.graph_id_fields` for natural keys to ensure stable, readable node IDs
- Include examples and descriptions on fields for better LLM extraction
- Leverage `is_entity=True` in model_config to create nodes
- Use `Edge()` helper for explicit relationship definition

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