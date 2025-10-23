# Docling-Graph: Document to Knowledge Graph Converter

Docling-Graph is a Python toolkit that leverages the **docling** library to convert unstructured documents (PDFs, images) into structured knowledge graphs.

It provides a flexible, config-driven pipeline to extract information from documents using your desired **Pydantic schema** and processing strategy. You can choose between:

- **Fast, Local VLM Extraction** (using docling's NuExtract pipeline)
- **Powerful LLM Extraction** (using local models via Ollama or remote APIs like Mistral, OpenAI, etc.)



## Core Processing Strategies

Docling-Graph is built around two core processing strategies:

- **One-to-One:** For documents that are batches of single-page items (e.g., a 10-page PDF containing 10 separate invoices).
- **Many-to-One:** For documents that are one logical entity spanning multiple pages (e.g., a 5-page insurance policy).



## Core Features

### Config-Driven Pipelines
Use `docling-graph init` to create a central YAML config. Run complex pipelines with a simple name (e.g., `many_to_one_api`).

### Flexible Processing Modes
- **One-to-One:** Processes each page of a document as a separate item.
- **Many-to-One:** Processes the entire document as a single, complete item.

### Multiple Extractor Types
- **local_vlm:** Uses the docling VLM pipeline (e.g., NuExtract) for direct, schema-driven extraction. Blazing fast, no LLM required.
- **local_llm:** Uses a local model (via Ollama) to extract from document markdown.
- **api:** Uses a remote LLM API (Mistral, OpenAI, etc.) for extraction.

### Intelligent Fallback Logic
The Many-to-One extractor automatically handles large documents. It attempts a single "fast path" call, but if the document exceeds the LLM's context window, it automatically falls back to a page-by-page extraction and intelligently merges the partial JSON results.

### Pydantic-Powered
Define your desired output using a Pydantic model. The extractor will return a validated object.

### Graph Generation & Visualization
Automatically converts your extracted Pydantic models into a **networkx** graph and generates a static `.png` visualization of the resulting graph.



## Quick Start

### 1. Installation

```bash
git clone https://github.com/ayoub-ibm/docling-graph.git
cd docling-graph
pip install -e .

# Install LLM clients as needed
pip install ollama mistralai
```

### 2. Initialize Project Config

```bash
docling-graph init
```

This creates a file `docling_graph_config.yaml` with 4 pre-defined pipelines. You can edit this file to set your preferred models and API providers.

### 3. Run a Pipeline

#### Example: Local One-to-One (Batch of Invoices)
```bash
docling-graph convert "data/my_invoices.pdf"   --template "templates.invoice.Invoice"   --pipeline "one_to_one_local"   --output-dir "outputs/"
```

#### Example: API Many-to-One (Multi-page Insurance Policy)
```bash
export MISTRAL_API_KEY="your_api_key_here"

docling-graph convert "data/my_policy.pdf"   --template "templates.insurance.HomeInsurance"   --pipeline "many_to_one_api"   --output-dir "outputs/"
```



## Configuration Example

```yaml
# docling_graph_config.yaml
pipelines:
  one_to_one_local:
    processing_mode: one_to_one
    extractor_type: local_vlm
    default_model: "numind/NuExtract-2.0-8B"

  one_to_one_api:
    processing_mode: one_to_one
    extractor_type: api
    provider: mistral
    default_model: "mistral-small-latest"

  many_to_one_local:
    processing_mode: many_to_one
    extractor_type: local_llm
    provider: ollama
    default_model: "llama3:8b"

  many_to_one_api:
    processing_mode: many_to_one
    extractor_type: api
    provider: mistral
    default_model: "mistral-large-latest"
```



## Pipeline Breakdown

### one_to_one_local
- **Use Case:** Batch of single-page items (invoices, ID cards).
- **Logic:** Uses `docling.DocumentExtractor` (NuExtract) per page.
- **Result:** List of Pydantic models.

### one_to_one_api
- **Use Case:** Same as above, but using an API-based LLM.
- **Logic:** Converts each page to Markdown and calls the LLM API.
- **Result:** List of Pydantic models.

### many_to_one_local
- **Use Case:** Multi-page logical document (reports, policies).
- **Logic:** Uses local Ollama model with hybrid fallback.
- **Result:** Single Pydantic model.

### many_to_one_api
- **Use Case:** Same as above, using remote API (Mistral, etc.).
- **Logic:** Uses remote LLM API with hybrid fallback.
- **Result:** Single Pydantic model.



## CLI Usage

```bash
docling-graph convert [OPTIONS] SOURCE
```

### Arguments
- `SOURCE`: Path to the source document.

### Options
| Option | Description |
|--------|--------------|
| `--template` | (Required) Pydantic model path (e.g., templates.invoice.Invoice). |
| `--pipeline` | (Required) Pipeline name (e.g., many_to_one_api). |
| `--config-file` | Config file path (default: `docling_graph_config.yaml`). |
| `--output-dir` | Output directory (default: `outputs/`). |
| `--model` | Override model name. |
| `--provider` | Override provider (e.g., openai). |



## Conceptual Design

### 1. Processing Modes
- **one_to_one:** Treats a 10-page PDF as 10 distinct documents.  
  _Good for invoices, receipts, ID cards._
- **many_to_one:** Treats a 10-page PDF as one logical document.  
  _Good for policies, reports, contracts._

### 2. Extractor Types
- **local_vlm:** Fast, schema-driven extraction via NuExtract.
- **local_llm:** Local LLM (Ollama) for Markdown-to-schema extraction.
- **api:** Remote LLM (Mistral, OpenAI, etc.) for complex documents.

### 3. Hybrid Fallback Logic
If the full document is too large for the model context window, the extractor:
1. Detects the overflow.
2. Switches to page-by-page extraction.
3. Merges partial JSON results into one validated Pydantic model.



## Adding Pydantic Templates

Create new files in the `templates/` directory (e.g., `templates/report.py`).  
Define your Pydantic `BaseModel`s and use the `Edge` helper function for relationships.

Run the CLI with:
```bash
docling-graph convert "data/my_doc.pdf"   --template "templates.report.YourModelName"
```



## Dependencies

| Category | Libraries |
|-----------|------------|
| Core | `docling[vlm]`, `typer[all]`, `rich`, `networkx`, `pyyaml` |
| Visualization | `matplotlib`, `pygraphviz` *(optional)* |
| LLM Clients | `ollama`, `mistralai`, `openai`, `google-generativeai` |
