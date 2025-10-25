<p align="center"><br>
  <a href="https://github.com/ayoub-ibm/docling-graph">
    <img loading="lazy" alt="Docling Graph" src="docs/assets/logo.png" width="250"/>
  </a>
</p>

# Docling Graph

[![Docs](https://img.shields.io/badge/docs-coming%20soon-brightgreen)](https://github.com/ayoub-ibm/docling-graph)
[![Docling](https://img.shields.io/badge/Docling-VLM-red)](https://github.com/docling-project/docling)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-orange)](https://networkx.org/)
[![Typer](https://img.shields.io/badge/Typer-CLI-purple)](https://typer.tiangolo.com/)
[![Rich](https://img.shields.io/badge/Rich-terminal-cyan)](https://github.com/Textualize/rich)
[![Ollama](https://img.shields.io/badge/Ollama-compatible-black)](https://ollama.ai/)
[![Mistral AI](https://img.shields.io/badge/Mistral-API-ff7000)](https://mistral.ai/)
[![License MIT](https://img.shields.io/github/license/ayoub-ibm/docling-graph)](https://opensource.org/licenses/MIT)


Docling-Graph converts PDFs and images into validated **Pydantic** objects and then into a **directed knowledge graph**, with exports to CSV or Cypher and both static and interactive visualizations.  

The toolkit supports two extraction families: **local VLM** via Docling and **LLM-based extraction** via local (Ollama) or API providers (Mistral), all orchestrated by a flexible, config-driven pipeline.



## Design Philosophy

- **Separation of Concerns**: Decouples processing granularity, model family, and inference location—configurable independently via CLI and defaults in `config.yaml`.
- **Typed, Validated Outputs**: Uses Pydantic models with stable node IDs driven by optional `graph_id_fields`.
- **Robust Fallbacks**: Supports long documents with merging strategies, clear logs, Markdown reports, and exportable graphs (CSV, Cypher, HTML, PNG/SVG/PDF).



## Key Capabilities

- **Extraction**:
  - Local `VLM` (Docling’s VLM pipeline)  
  - `LLM` (local via Ollama or API via Mistral)  
  - Page-wise or whole-document strategies
- **Graph Construction**:
  - Markdown to Graph: Convert validated Pydantic instances to a `NetworkX DiGraph` with rich edge metadata and stable node IDs
  - Smart Merge: Combine multi-page documents into a single Pydantic instance for unified processing
- **Export**:
  - `CSV` compatible with `Neo4j` admin import  
  - `Cypher` script generation for bulk ingestion
- **Visualization**:
  - Interactive Pyvis `HTML` with improved tooltips and physics  
  - Publication-grade static images (`PNG`, `SVG`, `PDF`)



## Data Flow

1. **Input**: Document file (PDF, image)
2. **Conversion**: Docling converts to structured format
3. **Extraction**: LLM/VLM extracts data into Pydantic models
4. **Validation**: Pydantic validates against template schema
5. **Graph Creation**: GraphConverter builds NetworkX graph
6. **Export**: Multiple output formats generated



## Installation

Requirements:

- **Python >= 3.9** (OS independent)  
- Clone and install:
```bash
git clone https://github.com/ayoub-ibm/docling-graph
cd docling-graph
pip install -e .
```

Dependencies:

- Core: `docling[vlm]`, `pydantic`, `networkx`, `pymupdf`, `matplotlib`, `pyvis`, `rich`, `typer`
- Optional LLM clients: `ollama` (local), `mistralai` (API)



## Quick Start

1. Initialize configuration in the current directory:
```bash
docling-graph init
```

The interactive wizard will walk you through:
- **Processing Mode** – Choose between `one-to-one` (page-by-page) or `many-to-one` (entire document) processing  
- **Backend Type** – Select `llm` (Language Model) or `vlm` (Vision-Language Model) for extraction  
- **Inference Location** – Choose `local` (your machine) or `remote` (cloud APIs)  
- **Export Format** – Select `csv` or `cypher` for knowledge graph output  
- **Docling Pipeline** – Choose document processing pipeline (`ocr`, or `vision`)  
- **Model Configuration** – Select specific models based on your backend and inference choices  
- **Output Settings** – Configure output directory and visualization preferences  

Each option includes helpful descriptions and sensible defaults, making it easy to get started even if you're new to the tool.


2. Run conversion with your Pydantic template:
```bash
docling-graph convert <SOURCE> --template "<TEMPLATE_PATH>" [OPTIONS]
```



## Configuration

`docling-graph init` creates `config.yaml` with editable defaults:

```yaml
defaults:
  processing_mode: many-to-one    # one-to-one | many-to-one
  backend_type: llm               # llm | vlm
  inference: local                # local | remote
  export_format: csv              # csv | cypher

docling:
  pipeline: ocr                   # ocr | vision

models:
  vlm:
    local:
      default_model: "numind/NuExtract-2.0-8B"
      provider: "docling"

  llm:
    local:
      default_model: "llama3:8b-instruct"
      provider: "ollama"
    remote:
      default_model: "mistral-small-latest"
      provider: "mistral"

  providers:
    mistral:
      default_model: "mistral-small-latest"
    openai:
      default_model: "gpt-4-turbo"
    gemini:
      default_model: "gemini-2.5-flash"

output:
  default_directory: "outputs"
  create_visualizations: true
  create_markdown: true
```

**Notes**:

- VLM extraction is **local-only**; `--backend_type vlm` with `--inference remote` is invalid.
- `docling.pipeline` only affects the LLM path (default OCR vs VLM pipeline).



## CLI Usage

**Base command**:
```bash
docling-graph convert [OPTIONS] SOURCE
```

**Required options**:

- `--template, -t` : Dotted path to Pydantic template (e.g., `docling_graph.invoice.Invoice`)

**Optional dimensions**:

- `--processing-mode, -p` : `one-to-one` | `many-to-one`  
- `--backend_type, -b` : `llm` | `vlm`  
- `--inference, -i` : `local` | `remote`  
- `--docling-config, -d` : `default` (OCR) | `vlm` (VLM pipeline)  
- `--output-dir, -o` : Output directory (default: `outputs`)  
- `--model` : Override model name  
- `--provider` : Override provider  
- `--export-format, -e` : `csv` | `cypher`

The CLI validates all options and fails fast on unsupported or conflicting combinations.



## Usage Examples

- **Local VLM, per page** (one-to-one), VLM docling pipeline:
```bash
docling-graph convert data/invoices.pdf \
  --template "docling_graph.invoice.Invoice" \
  -p one-to-one b vlm -i local -d vlm -o outputs
```

- **Local LLM via Ollama, many-to-one**, default OCR pipeline:
```bash
docling-graph convert data/policy.pdf \
  --template "docling_graph.insurance.InsuranceTerms" \
  -p many-to-one -b llm -i local --provider ollama \
  --model "llama3:8b-instruct" -d default -o outputs
```

- **API LLM via Mistral, many-to-one**:
```bash
export MISTRAL_API_KEY="your_api_key_here"
docling-graph convert data/policy.pdf \
  --template "docling_graph.insurance.InsuranceTerms" \
  -p many-to-one -b llm -i remote --provider mistral -o outputs
```

- **Export graph as Cypher instead of CSV**:
```bash
docling-graph convert data/id_card.png \
  --template "docling_graph.id_card.IDCard" \
  -p many-to-one -b llm -i local -e cypher -o outputs
```



## Outputs

- **CSV Export**: `outputs/graph/nodes.csv` and `relationships.csv` compatible with Neo4j.  
- **Interactive HTML**: `outputs/<name>_graph.html`  
- **Static PNG**: `outputs/<name>_graph.png`  
- **Markdown Report**: `outputs/<name>_graph.md`  
- **Cypher Export**: `outputs/<name>_graph.cypher` (CREATE + MATCH statements)



## Architecture

- **CLI (Typer)**: `init` scaffolds config; `convert` orchestrates the pipeline.  
- **Pipeline**: Imports Pydantic template → resolves model/provider → runs extraction → converts to graph → exports + renders.  
- **Extractors**: Compose backend (VLM or LLM) with strategy (OneToOne / ManyToOne).  
- **Strategies**: One-to-one processes per page; many-to-one handles full-document extraction and merges intelligently.  
- **Backends**: VLM wraps Docling VLM; LLM prompts Markdown via Docling DocumentConverter.  
- **LLM clients**: Ollama (local) and Mistral (API).



## Pydantic Templates

- Templates available for **invoices**, **French ID cards**, **insurance terms**.  
- Edges can be **implicit** (nested BaseModels) or **explicit** (generic Edge type).  

**Tips**:

- Use `model_config.graph_id_fields` for natural keys to ensure stable node IDs.  
- Include examples and descriptions on fields for better LLM extraction.
- Please refer to the [Pydantic Templates for Knowledge Graph Extraction](docs/guides/pydantic_templates_for_knowledge_graph_extraction.md) guide for more details.



## Visualization

- **Static**: PNG, SVG, PDF with property boxes and layout heuristics.  
- **Interactive**: Pyvis HTML with improved tooltips and stabilized physics.

Parameters:

- `tooltip_max_length`: Maximum tooltip text length  
- `show_properties`: Show entity properties in static images



## Environment and Providers

- **Ollama (local LLM)**: Ensure `ollama serve` is running and model is pulled.  
- **Mistral (API LLM)**: Set `MISTRAL_API_KEY` in environment or `.env` file.  
- **VLM (Docling)**: Specify HF repo id (e.g., `numind/NuExtract-2.0-8B`).



## Troubleshooting

- **Config not found**: Run `docling-graph init`  
- **VLM with API not supported**: Use `--inference local` or `--backend_type llm`  
- **Template import failed**: Verify the dotted path  
- **Ollama connection error**: Ensure service is running and model is pulled  
- **Mistral key not set**: Export `MISTRAL_API_KEY` or add to `.env`



## TODO List

- **Expanded LLM Provider Support**: Add compatibility for additional LLM providers like WatsonX and vLLM.
- **Add Graph Database Connectivity**: Enable direct or batch loading of generated graphs into graph databases like Neo4j.
- **Implement docling-graph query Command**: Add a CLI feature that allows querying and interacting with generated document graphs, for example through LangChain.
- **Interactive Pydantic Model Generation**: Introduce prompts to help users generate Pydantic templates directly from documents, using the detailed guide from documentation.
- **Improve Component Instantiation**: Refactor class and model initialization to support user-defined components, allowing flexible injection or replacement of default implementations.
- **Refactor Batch Processing for Local GPU Inference**: Optimize batch handling to improve efficiency and performance during local inference on GPU.
- **Store Markdown Conversion Output**: Add functionality to save the output generated by the Docling converter in Markdown format for later use or reference.