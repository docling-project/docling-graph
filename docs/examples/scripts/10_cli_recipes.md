# Example 10: CLI Recipes

These recipes show how to run the `docling-graph convert` command from your **project's root directory**.

All paths (`--template`, source file, `--output-dir`) are relative to the root.

---

### Recipe 1: VLM from Image
(Python: `01_vlm_from_image.py`)

```bash
uv run docling-graph convert "docs/examples/data/invoice/sample_invoice.jpg" \
    --template "docs.examples.templates.invoice.Invoice" \
    --output-dir "outputs/cli_01" \
    --backend "vlm" \
    --processing-mode "one-to-one" \
    --docling-pipeline "vision"
````

-----

### Recipe 3: Remote LLM (Mistral)

(Python: `03_llm_remote_api.py`)

*(Requires `MISTRAL_API_KEY` env var)*

```bash
uv run docling-graph convert "docs/examples/data/battery_research/bauer2014.pdf" \
    --template "docs.examples.templates.battery_research.Research" \
    --output-dir "outputs/cli_03" \
    --backend "llm" \
    --inference "remote" \
    --provider "mistral" \
    --model "mistral-large-latest" \
    --processing-mode "many-to-one" \
    --use-chunking \
    --no-llm-consolidation
```

-----

### Recipe 4: Local LLM (Ollama)

(Python: `04_llm_local_ollama.py`)

*(Requires Ollama server to be running)*

```bash
uv run docling-graph convert "docs/examples/data/battery_research/bauer2014.pdf" \
    --template "docs.examples.templates.battery_research.Research" \
    --output-dir "outputs/cli_04" \
    --backend "llm" \
    --inference "local" \
    --provider "ollama" \
    --model "llama3:8b" \
    --processing-mode "many-to-one" \
    --use-chunking \
    --no-llm-consolidation
```

-----

### Recipe 5: LLM with Consolidation

(Python: `05_llm_with_consolidation.py`)

```bash
uv run docling-graph convert "docs/examples/data/battery_research/bauer2014.pdf" \
    --template "docs.examples.templates.battery_research.Research" \
    --output-dir "outputs/cli_05" \
    --backend "llm" \
    --inference "remote" \
    --provider "mistral" \
    --processing-mode "many-to-one" \
    --use-chunking \
    --llm-consolidation
```

-----

### Recipe 8: LLM with 'vision' Config (Hybrid)

(Python: `08_llm_with_vision_config.py`)

```bash
uv run docling-graph convert "docs/examples/data/battery_research/bauer2014.pdf" \
    --template "docs.examples.templates.battery_research.Research" \
    --output-dir "outputs/cli_08" \
    --backend "llm" \
    --inference "local" \
    --provider "ollama" \
    --model "llama3:8b" \
    --docling-pipeline "vision" \
    --processing-mode "many-to-one" \
    --use-chunking
```

-----

### Recipe 9: Export to Cypher

(Python: `09_export_to_cypher.py`)

```bash
uv run docling-graph convert "docs/examples/data/invoice/sample_invoice.jpg" \
    --template "docs.examples.templates.invoice.Invoice" \
    --output-dir "outputs/cli_09" \
    --backend "vlm" \
    --docling-pipeline "vision" \
    --export-format "cypher"
```