# Docling Graph Examples

Welcome to the `docling-graph` examples. This directory contains all the resources you need to get started.

## Project Structure

* `/examples/scripts/`: Python script examples and the CLI recipes.
* `/examples/data/`: Sample PDF and image files used by the scripts.
* `/examples/templates/`: The Pydantic schemas (e.g., `invoice.py`) that define what data to extract.

## 10 Core Recipes

All Python scripts are located in the `/examples/scripts/` folder and are designed to be **run from the project's root directory**.

1.  **`01_vlm_from_image.py`**: The "Hello World." Extracts from a **single image** using the **VLM**.
2.  **`02_vlm_from_pdf_page.py`**: Shows the **VLM** backend on a **single-page PDF**.
3.  **`03_llm_remote_api.py`**: Standard **LLM** workflow using a **remote API** (Mistral) on a multi-page PDF.
4.  **`04_llm_local_ollama.py`**: Uses a **local LLM** (Ollama) on the same PDF.
5.  **`05_llm_with_consolidation.py`**: Advanced merging strategy using an **LLM to consolidate** results.
6.  **`06_llm_one_to_one.py`**: Processes **each page individually** (`one-to-one`) and combines them into one graph.
7.  **`07_llm_no_chunking.py`**: Processes a short document in a **single pass** (disables chunking).
8.  **`08_llm_with_vision_config.py`**: Hybrid mode: uses the `vision` config for **layout-aware chunks** for the LLM.
9.  **`09_export_to_cypher.py`**: Shows how to change the **output format** to a **Cypher script** for Neo4j.
10. **`10_cli_recipes.md`**: A markdown file showing the **CLI (command-line) equivalents** for all the examples above.