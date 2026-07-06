# Docling Graph Examples

Example scripts and Pydantic templates for docling-graph. Run scripts from the **project root** with `uv run python docs/examples/scripts/...`.

## Project Structure

| Path | Description |
|------|-------------|
| `docs/examples/scripts/` | Python example scripts (01–16) |
| `docs/examples/templates/` | Pydantic templates (e.g. `billing_document.py`, `insurance_terms.py`, `rheology_research.py`) |

## Example Scripts (01–16)

### Getting Started

1. **`01_quickstart_vlm_image.py`** — VLM extraction from an invoice image
2. **`02_quickstart_llm_pdf.py`** — LLM extraction from a multi-page PDF (e.g. rheology)
3. **`03_url_processing.py`** — Process documents from URLs (e.g. arXiv)

### Core Features

4. **`04_input_formats.py`** — Text, Markdown, and DoclingDocument inputs
5. **`05_processing_modes.py`** — One-to-one vs many-to-one modes
6. **`06_export_formats.py`** — CSV, Cypher, and JSON exports
7. **`07_local_inference.py`** — Local inference with Ollama

### Optimization & Providers

8. **`08_chunking_consolidation.py`** — Chunking and merge behavior
9. **`09_batch_processing.py`** — Batch processing with error handling
10. **`10_provider_configs.py`** — OpenAI, Mistral, Gemini, watsonx
11. **`12_custom_llm_client.py`** — Custom LLM client (bring your own URL) with full pipeline
12. **`14_streaming_responses.py`** — Streaming LLM responses for real-time processing and progress feedback

### Data Grounding

13. **`15_provenance_grounding.py`** — Trace extracted nodes back to source chunks and pages via `__provenance__` and `provenance.json`

### Evaluation

14. **`16_extraction_evaluation.py`** — Score an extracted `graph.json` against a template-shaped ground-truth JSON (node/edge P/R/F1, attribute completeness, integrity). Domain-agnostic — works with any `(template, ground_truth.json, graph.json)` triple.

For CLI usage, see [CLI Reference](../usage/cli/index.md) and [convert command](../usage/cli/convert-command.md).

## Quick Start

```bash
# From project root: VLM from image
uv run python docs/examples/scripts/01_quickstart_vlm_image.py

# Or use the CLI
uv run docling-graph convert "https://upload.wikimedia.org/wikipedia/commons/9/9f/Swiss_QR-Bill_example.jpg" \
    --template "docs.examples.templates.billing_document.BillingDocument" \
    --backend "vlm"

# Score a completed run against a ground-truth JSON
uv run python docs/examples/scripts/16_extraction_evaluation.py \
    --graph outputs/RUN_DIR/docling_graph/graph.json \
    --truth ground_truth.json \
    --template "docs.examples.templates.insurance_terms.AssuranceMRH"
```

## Learning Path

1. Run **01** for a minimal VLM run, then **02** for LLM extraction.
2. Use **03–07** for input formats, processing modes, exports, and local inference.
3. Use **08–10** for chunking, batch runs, and multiple providers.