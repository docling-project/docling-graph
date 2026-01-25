# convert Command


## Overview

The `convert` command transforms documents into knowledge graphs using configurable extraction pipelines.

**Key Features:**
- Multiple backend support (LLM/VLM)
- Flexible processing modes
- Configurable chunking
- Multiple export formats
- Batch processing support

---

## Basic Usage

```bash
uv run docling-graph convert SOURCE --template TEMPLATE [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `SOURCE` | Path to document (PDF, JPG, PNG, TXT, MD), URL, or DoclingDocument JSON |
| `--template`, `-t` | Dotted path to Pydantic template |

### Examples

```bash
# PDF document
uv run docling-graph convert invoice.pdf \
    --template "my_templates.Invoice"

# Text file
uv run docling-graph convert notes.txt \
    --template "my_templates.Report" \
    --backend llm

# URL
uv run docling-graph convert https://example.com/doc.pdf \
    --template "my_templates.Invoice"

# Markdown file
uv run docling-graph convert README.md \
    --template "my_templates.Documentation" \
    --backend llm
```

---

## Core Options

### Backend Selection

```bash
--backend {llm|vlm}
```

**LLM (Language Model):**
- Best for text-heavy documents
- Supports chunking and consolidation
- Works with local and remote providers

**VLM (Vision-Language Model):**
- Best for forms and structured layouts
- Processes images directly
- Local inference only

**Example:**
```bash
# Use LLM backend
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --backend llm

# Use VLM backend
uv run docling-graph convert form.jpg \
    --template "templates.IDCard" \
    --backend vlm
```

---

### Inference Mode

```bash
--inference {local|remote}
```

**Local:**
- Run models on your machine
- Requires GPU for best performance
- No API costs

**Remote:**
- Use cloud API providers
- Requires API key
- Pay per request

**Example:**
```bash
# Local inference
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --inference local

# Remote inference
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --inference remote
```

---

### Processing Mode

```bash
--processing-mode {one-to-one|many-to-one}
```

**many-to-one (recommended):**
- Merge all pages into single graph
- Better for multi-page documents
- Enables consolidation

**one-to-one:**
- Create separate graph per page
- Better for independent pages
- Faster processing

**Example:**
```bash
# Merge all pages
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --processing-mode many-to-one

# Process pages separately
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --processing-mode one-to-one
```

---

## Model Configuration

### Provider Override

```bash
--provider PROVIDER
```

**Available providers:**
- **Local:** `vllm`, `ollama`
- **Remote:** `mistral`, `openai`, `gemini`, `watsonx`

### Model Override

```bash
--model MODEL
```

**Example:**
```bash
# Use specific model
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --provider mistral \
    --model mistral-large-latest
```

---

## Extraction Options

### Chunking

```bash
--use-chunking / --no-use-chunking
```

**Enable chunking for:**
- Large documents (>5 pages)
- Documents exceeding context limits
- Better extraction accuracy

**Disable chunking for:**
- Small documents
- When full context is needed
- Faster processing

**Example:**
```bash
# Enable chunking (default)
uv run docling-graph convert large_doc.pdf \
    --template "templates.Research" \
    --use-chunking

# Disable chunking
uv run docling-graph convert small_doc.pdf \
    --template "templates.Invoice" \
    --no-use-chunking
```

---

### LLM Consolidation

```bash
--llm-consolidation / --no-llm-consolidation
```

**Enable for:**
- Higher accuracy
- Complex merging scenarios
- When quality > speed

**Disable for:**
- Faster processing
- Lower API costs
- Simple documents

**Example:**
```bash
# Enable LLM consolidation
uv run docling-graph convert document.pdf \
    --template "templates.Research" \
    --llm-consolidation

# Disable (use programmatic merge)
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --no-llm-consolidation
```

---

## Docling Configuration

### Pipeline Selection

```bash
--docling-pipeline {ocr|vision}
```

**OCR Pipeline:**
- Traditional OCR approach
- Most accurate for standard documents
- Faster processing

**Vision Pipeline:**
- Uses Granite-Docling VLM
- Better for complex layouts
- Handles tables and figures better

**Example:**
```bash
# Use OCR pipeline (default)
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --docling-pipeline ocr

# Use vision pipeline
uv run docling-graph convert complex_doc.pdf \
    --template "templates.Research" \
    --docling-pipeline vision
```

---

## Export Options

### Export Format

```bash
--export-format {csv|cypher}
```

**CSV:**
- For Neo4j import
- Separate nodes.csv and edges.csv
- Easy to analyze

**Cypher:**
- Direct Neo4j execution
- Single .cypher file
- Ready to import

**Example:**
```bash
# Export as CSV
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --export-format csv

# Export as Cypher
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --export-format cypher
```

---

### Docling Exports

```bash
--export-docling-json / --no-docling-json
--export-markdown / --no-markdown
--export-per-page / --no-per-page
```

**Example:**
```bash
# Export all Docling outputs
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --export-docling-json \
    --export-markdown \
    --export-per-page

# Minimal exports
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --no-docling-json \
    --no-markdown \
    --no-per-page
```

---

## Graph Options

### Reverse Edges

```bash
--reverse-edges
```

Creates bidirectional relationships in the graph.

**Example:**
```bash
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --reverse-edges
```

---

## Output Options

### Output Directory

```bash
--output-dir PATH
```

**Default:** `outputs/`

**Example:**
```bash
# Custom output directory
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --output-dir "results/invoice_001"

# Organize by date
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --output-dir "outputs/$(date +%Y-%m-%d)"
```

---

## Complete Examples

### Example 1: Simple Invoice (VLM)

```bash
uv run docling-graph convert invoice.jpg \
    --template "templates.Invoice" \
    --backend vlm \
    --processing-mode one-to-one \
    --output-dir "outputs/invoice"
```

### Example 2: Research Paper (Remote LLM)

```bash
export MISTRAL_API_KEY="your-key"

uv run docling-graph convert research.pdf \
    --template "templates.Research" \
    --backend llm \
    --inference remote \
    --provider mistral \
    --model mistral-large-latest \
    --processing-mode many-to-one \
    --use-chunking \
    --llm-consolidation \
    --output-dir "outputs/research"
```

### Example 3: Local Processing (Ollama)

```bash
# Start Ollama server first
ollama serve

uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --backend llm \
    --inference local \
    --provider ollama \
    --model llama3:8b \
    --processing-mode many-to-one \
    --use-chunking \
    --output-dir "outputs/local"
```

### Example 4: Cypher Export for Neo4j

```bash
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --backend llm \
    --inference remote \
    --export-format cypher \
    --output-dir "outputs/neo4j"

# Import to Neo4j
cat outputs/neo4j/graph.cypher | cypher-shell
```

### Example 5: Minimal Processing

```bash
uv run docling-graph convert small_doc.pdf \
    --template "templates.Invoice" \
    --backend llm \
    --inference local \
    --no-use-chunking \
    --no-llm-consolidation \
    --no-docling-json \
    --no-markdown \
    --output-dir "outputs/minimal"
```

---

## Batch Processing

### Process Multiple Files

```bash
# Bash loop
for pdf in documents/*.pdf; do
    uv run docling-graph convert "$pdf" \
        --template "templates.Invoice" \
        --output-dir "outputs/$(basename $pdf .pdf)"
done
```

### Parallel Processing

```bash
# Using GNU parallel
ls documents/*.pdf | parallel -j 4 \
    uv run docling-graph convert {} \
        --template "templates.Invoice" \
        --output-dir "outputs/{/.}"
```

### Batch Script

```bash
#!/bin/bash
# batch_convert.sh

TEMPLATE="templates.Invoice"
INPUT_DIR="documents"
OUTPUT_BASE="outputs"

for file in "$INPUT_DIR"/*.pdf; do
    filename=$(basename "$file" .pdf)
    echo "Processing: $filename"
    
    uv run docling-graph convert "$file" \
        --template "$TEMPLATE" \
        --output-dir "$OUTPUT_BASE/$filename" \
        --backend llm \
        --inference remote
    
    echo "Completed: $filename"
done
```

---

## Configuration Priority

Options are resolved in this order (highest to lowest):

1. **Command-line arguments**
2. **config.yaml** (from `init`)
3. **Built-in defaults**

### Example

```yaml
# config.yaml
defaults:
  backend: llm
  inference: local
  processing_mode: many-to-one
```

```bash
# This uses remote inference (CLI overrides config)
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --inference remote
```

---

## Output Structure

```
outputs/
├── metadata.json                # Pipeline metadata
├── docling/                     # Docling conversion output
│   ├── document.json            # Docling format
│   └── document.md              # Markdown export
└── docling_graph/               # Graph outputs
    ├── graph.json               # Complete graph
    ├── nodes.csv                # Node data
    ├── edges.csv                # Edge data
    ├── graph.html               # Interactive visualization
    └── report.md                # Summary report
    └── ...
```

---

## Error Handling

### Configuration Errors

```bash
[red]Configuration Error:[/red] Invalid backend type: 'invalid'
```

**Solution:** Use `llm` or `vlm`

### Extraction Errors

```bash
[red]Extraction Error:[/red] Template not found: 'templates.Missing'
```

**Solution:** Check template path and ensure it's importable

### API Errors

```bash
[red]Pipeline Error:[/red] API key not found for provider: mistral
```

**Solution:**
```bash
export MISTRAL_API_KEY="your-key"
```

---

## Troubleshooting

### Issue: Template Not Found

**Error:**
```
ModuleNotFoundError: No module named 'templates'
```

**Solution:**
```bash
# Ensure template is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use absolute path
uv run docling-graph convert document.pdf \
    --template "my_project.templates.Invoice"
```

### Issue: Out of Memory

**Error:**
```
CUDA out of memory
```

**Solution:**
```bash
# Enable chunking
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --use-chunking

# Or use smaller model
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --model "ibm-granite/granite-4.0-1b"
```

### Issue: Slow Processing

**Solution:**
```bash
# Disable LLM consolidation
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --no-llm-consolidation

# Or disable chunking for small docs
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --no-use-chunking
```

---

## Best Practices

### 1. Use Configuration Files

```bash
# ✅ Good - Reusable configuration
uv run docling-graph init
uv run docling-graph convert document.pdf -t "templates.Invoice"

# ❌ Avoid - Repeating options
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --backend llm --inference remote --provider mistral
```

### 2. Organize Outputs

```bash
# ✅ Good - Organized by document
uv run docling-graph convert invoice_001.pdf \
    --template "templates.Invoice" \
    --output-dir "outputs/invoice_001"

# ❌ Avoid - Overwriting outputs
uv run docling-graph convert invoice_001.pdf \
    --template "templates.Invoice"
```

### 3. Use Appropriate Backend

```bash
# ✅ Good - VLM for forms
uv run docling-graph convert id_card.jpg \
    --template "templates.IDCard" \
    --backend vlm

# ✅ Good - LLM for documents
uv run docling-graph convert research.pdf \
    --template "templates.Research" \
    --backend llm
```

---

## Next Steps

1. **[inspect Command →](inspect-command.md)** - Visualize results
2. **[CLI Recipes →](cli-recipes.md)** - Common patterns
3. **[Examples →](../examples/index.md)** - Real-world examples

---

## Quick Reference

### Minimal Command

```bash
uv run docling-graph convert SOURCE -t TEMPLATE
```

### Common Options

```bash
# Backend and inference
--backend llm --inference remote

# Processing
--processing-mode many-to-one --use-chunking

# Export
--export-format csv --output-dir PATH

# Model
--provider mistral --model mistral-large-latest
```

### All Options

```bash
uv run docling-graph convert SOURCE \
    --template TEMPLATE \
    --backend {llm|vlm} \
    --inference {local|remote} \
    --processing-mode {one-to-one|many-to-one} \
    --docling-pipeline {ocr|vision} \
    --provider PROVIDER \
    --model MODEL \
    --use-chunking / --no-use-chunking \
    --llm-consolidation / --no-llm-consolidation \
    --export-format {csv|cypher} \
    --export-docling-json / --no-docling-json \
    --export-markdown / --no-markdown \
    --export-per-page / --no-per-page \
    --reverse-edges \
    --output-dir PATH
```