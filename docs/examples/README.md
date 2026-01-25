# Docling Graph Examples

Welcome to the `docling-graph` examples. This directory contains all the resources you need to get started.

## Project Structure

* `/examples/scripts/`: Python script examples and CLI recipes
* `/examples/data/`: Sample PDF and image files used by the scripts
* `/examples/templates/`: Pydantic schemas (e.g., `invoice.py`) that define what data to extract

## 10 High-Quality Example Scripts

All Python scripts are located in the `/examples/scripts/` folder and are designed to be **run from the project's root directory**.

### Beginner Level (Getting Started)

1. **`01_quickstart_vlm_image.py`**: Basic VLM extraction from invoice image - The "Hello World" example
2. **`02_quickstart_llm_pdf.py`**: Basic LLM extraction from multi-page research paper PDF
3. **`03_url_processing.py`**: Download and process documents directly from URLs (arXiv, etc.)

### Intermediate Level (Core Features)

4. **`04_input_formats.py`**: Process text, Markdown, and DoclingDocument formats
5. **`05_processing_modes.py`**: Compare one-to-one vs many-to-one processing modes
6. **`06_export_formats.py`**: Generate CSV, Cypher, and JSON exports for Neo4j
7. **`07_local_inference.py`**: Privacy-focused offline processing with Ollama

### Advanced Level (Optimization & Configuration)

8. **`08_chunking_consolidation.py`**: Compare programmatic merge vs LLM consolidation
9. **`09_batch_processing.py`**: Process multiple documents efficiently with error handling
10. **`10_provider_configs.py`**: Compare OpenAI, Mistral, Gemini, and WatsonX providers

### CLI Reference

11. **`11_cli_recipes.md`**: Complete CLI command reference for all examples above

## Quick Start

```bash
# Run the simplest example (VLM from image)
uv run python docs/examples/scripts/01_quickstart_vlm_image.py

# Or use CLI directly
uv run docling-graph convert "docs/examples/data/invoice/sample_invoice.jpg" \
    --template "docs.examples.templates.invoice.Invoice" \
    --backend "vlm"
```

## Learning Path

1. **Start Here**: Run `01_quickstart_vlm_image.py` to understand the basics
2. **Text Processing**: Try `02_quickstart_llm_pdf.py` for LLM-based extraction
3. **Explore Features**: Work through examples 03-07 to learn core capabilities
4. **Advanced Topics**: Examples 08-10 cover optimization and multi-provider setups
5. **CLI Reference**: Use `11_cli_recipes.md` for command-line usage

## Features Covered

| Feature | Example Scripts |
|---------|----------------|
| VLM Backend | 01, 06 |
| LLM Backend | 02, 03, 04, 05, 07, 08, 09, 10 |
| Local Inference | 05, 07 |
| Remote Inference | 02, 03, 08, 10 |
| URL Input | 03 |
| Text/Markdown Input | 04 |
| One-to-One Mode | 05 |
| Many-to-One Mode | 02, 03, 05, 08 |
| Chunking | 02, 08 |
| Consolidation | 08 |
| Export Formats | 06 |
| Batch Processing | 09 |
| Multi-Provider | 10 |