# CLI Recipes


## Overview

This guide provides **ready-to-use CLI recipes** for common document processing scenarios. All examples use `uv` and can be run from your project root.

---

## Quick Reference

| Recipe | Backend | Use Case |
|--------|---------|----------|
| [VLM from Image](#recipe-1-vlm-from-image) | VLM | Forms, ID cards, structured layouts |
| [VLM from PDF Page](#recipe-2-vlm-from-pdf-page) | VLM | Single-page PDFs |
| [Remote LLM](#recipe-3-remote-llm-mistral) | LLM (Remote) | Text-heavy documents, API-based |
| [Local LLM](#recipe-4-local-llm-ollama) | LLM (Local) | Privacy-focused, offline processing |
| [LLM with Consolidation](#recipe-5-llm-with-consolidation) | LLM (Remote) | High-accuracy extraction |
| [One-to-One Processing](#recipe-6-one-to-one-processing) | LLM | Independent pages |
| [No Chunking](#recipe-7-no-chunking) | LLM | Small documents |
| [Vision Pipeline](#recipe-8-vision-pipeline-hybrid) | LLM | Complex layouts with tables |
| [Cypher Export](#recipe-9-cypher-export) | Any | Neo4j import |
| [Batch Processing](#recipe-10-batch-processing) | Any | Multiple documents |

---

## Recipe 1: VLM from Image

**Use Case:** Extract structured data from images (forms, ID cards, invoices)

**Requirements:**
```bash
uv sync --extra all
```

**Command:**
```bash
uv run docling-graph convert "docs/examples/data/invoice/sample_invoice.jpg" \
    --template "docs.examples.templates.invoice.Invoice" \
    --output-dir "outputs/recipe_01" \
    --backend "vlm" \
    --processing-mode "one-to-one" \
    --docling-pipeline "vision"
```

**When to Use:**
<br>✅ Single-page forms
<br>✅ ID cards or badges
<br>✅ Structured layouts
<br>✅ Image files (JPG, PNG)

**Python Equivalent:** `examples/scripts/01_vlm_from_image.py`

---

## Recipe 2: VLM from PDF Page

**Use Case:** Extract from single-page PDFs using vision model

**Requirements:**
```bash
uv sync --extra all
```

**Command:**
```bash
uv run docling-graph convert "docs/examples/data/invoice/sample_invoice.pdf" \
    --template "docs.examples.templates.invoice.Invoice" \
    --output-dir "outputs/recipe_02" \
    --backend "vlm" \
    --processing-mode "one-to-one" \
    --docling-pipeline "vision"
```

**When to Use:**
<br>✅ Single-page PDFs
<br>✅ Forms in PDF format
<br>✅ High-quality scans

**Python Equivalent:** `examples/scripts/02_vlm_from_pdf_page.py`

---

## Recipe 3: Remote LLM (Mistral)

**Use Case:** Process documents using Mistral AI API

**Requirements:**
```bash
uv sync --extra remote
export MISTRAL_API_KEY="your-api-key"
```

**Command:**
```bash
uv run docling-graph convert "docs/examples/data/research_paper/rheology.pdf" \
    --template "docs.examples.templates.rheology_research.Research" \
    --output-dir "outputs/recipe_03" \
    --backend "llm" \
    --inference "remote" \
    --provider "mistral" \
    --model "mistral-large-latest" \
    --processing-mode "many-to-one" \
    --use-chunking \
    --no-llm-consolidation
```

**When to Use:**
<br>✅ Multi-page documents
<br>✅ Text-heavy content
<br>✅ No local GPU
<br>✅ Cloud-based processing

**Cost:** ~$0.01-0.10 per document (varies by model and length)

**Python Equivalent:** `examples/scripts/03_llm_remote_api.py`

---

## Recipe 4: Local LLM (Ollama)

**Use Case:** Process documents locally using Ollama

**Requirements:**
```bash
uv sync --extra local

# Start Ollama server
ollama serve

# Pull model
ollama pull llama3:8b
```

**Command:**
```bash
uv run docling-graph convert "docs/examples/data/research_paper/rheology.pdf" \
    --template "docs.examples.templates.rheology_research.Research" \
    --output-dir "outputs/recipe_04" \
    --backend "llm" \
    --inference "local" \
    --provider "ollama" \
    --model "llama3:8b" \
    --processing-mode "many-to-one" \
    --use-chunking \
    --no-llm-consolidation
```

**When to Use:**
<br>✅ Privacy-sensitive documents
<br>✅ Offline processing
<br>✅ No API costs
<br>✅ Local development

**Python Equivalent:** `examples/scripts/04_llm_local_ollama.py`

---

## Recipe 5: LLM with Consolidation

**Use Case:** High-accuracy extraction with LLM-based merging

**Requirements:**
```bash
uv sync --extra remote
export MISTRAL_API_KEY="your-api-key"
```

**Command:**
```bash
uv run docling-graph convert "docs/examples/data/research_paper/rheology.pdf" \
    --template "docs.examples.templates.rheology_research.Research" \
    --output-dir "outputs/recipe_05" \
    --backend "llm" \
    --inference "remote" \
    --provider "mistral" \
    --processing-mode "many-to-one" \
    --use-chunking \
    --llm-consolidation
```

**When to Use:**
<br>✅ Complex documents
<br>✅ Accuracy > speed
<br>✅ Conflicting information across pages
<br>✅ Quality matters more than cost

**Trade-offs:**
<br>⚠️ Slower processing
<br>⚠️ Higher API costs
<br>✅ Better accuracy

**Python Equivalent:** `examples/scripts/05_llm_with_consolidation.py`

---

## Recipe 6: One-to-One Processing

**Use Case:** Process each page independently

**Requirements:**
```bash
uv sync --extra remote
export MISTRAL_API_KEY="your-api-key"
```

**Command:**
```bash
uv run docling-graph convert "docs/examples/data/research_paper/rheology.pdf" \
    --template "docs.examples.templates.rheology_research.Research" \
    --output-dir "outputs/recipe_06" \
    --backend "llm" \
    --inference "remote" \
    --provider "mistral" \
    --processing-mode "one-to-one" \
    --use-chunking
```

**When to Use:**
<br>✅ Independent pages
<br>✅ Page-level analysis
<br>✅ Faster processing
<br>✅ Parallel processing possible

**Output:** Multiple graphs (one per page)

**Python Equivalent:** `examples/scripts/06_llm_one_to_one.py`

---

## Recipe 7: No Chunking

**Use Case:** Process small documents without chunking

**Requirements:**
```bash
uv sync --extra remote
export MISTRAL_API_KEY="your-api-key"
```

**Command:**
```bash
uv run docling-graph convert "docs/examples/data/invoice/sample_invoice.pdf" \
    --template "docs.examples.templates.invoice.Invoice" \
    --output-dir "outputs/recipe_07" \
    --backend "llm" \
    --inference "remote" \
    --provider "mistral" \
    --processing-mode "many-to-one" \
    --no-use-chunking
```

**When to Use:**
<br>✅ Small documents (<5 pages)
<br>✅ Documents within context limit
<br>✅ Faster processing
<br>✅ Simpler pipeline

**Python Equivalent:** `examples/scripts/07_llm_no_chunking.py`

---

## Recipe 8: Vision Pipeline (Hybrid)

**Use Case:** Use vision-based document conversion with LLM extraction

**Requirements:**
```bash
uv sync --extra local

# Start Ollama
ollama serve
ollama pull llama3:8b
```

**Command:**
```bash
uv run docling-graph convert "docs/examples/data/research_paper/rheology.pdf" \
    --template "docs.examples.templates.rheology_research.Research" \
    --output-dir "outputs/recipe_08" \
    --backend "llm" \
    --inference "local" \
    --provider "ollama" \
    --model "llama3:8b" \
    --docling-pipeline "vision" \
    --processing-mode "many-to-one" \
    --use-chunking
```

**When to Use:**
<br>✅ Complex layouts
<br>✅ Tables and figures
<br>✅ Mixed content types
<br>✅ Better layout preservation

**Python Equivalent:** `examples/scripts/08_llm_with_vision_config.py`

---

## Recipe 9: Cypher Export

**Use Case:** Export directly to Neo4j Cypher format

**Requirements:**
```bash
uv sync --extra all
```

**Command:**
```bash
uv run docling-graph convert "docs/examples/data/invoice/sample_invoice.jpg" \
    --template "docs.examples.templates.invoice.Invoice" \
    --output-dir "outputs/recipe_09" \
    --backend "vlm" \
    --docling-pipeline "vision" \
    --export-format "cypher"
```

**Import to Neo4j:**
```bash
# Import the generated Cypher script
cat outputs/recipe_09/graph.cypher | cypher-shell -u neo4j -p password
```

**When to Use:**
<br>✅ Direct Neo4j import
<br>✅ Graph database workflows
<br>✅ Production deployments

**Python Equivalent:** `examples/scripts/09_export_to_cypher.py`

---

## Recipe 10: Batch Processing

**Use Case:** Process multiple documents

**Requirements:**
```bash
uv sync --extra remote
export MISTRAL_API_KEY="your-api-key"
```

**Bash Script:**
```bash
#!/bin/bash
# batch_process.sh

TEMPLATE="docs.examples.templates.invoice.Invoice"
INPUT_DIR="documents"
OUTPUT_BASE="outputs"

for file in "$INPUT_DIR"/*.pdf; do
    filename=$(basename "$file" .pdf)
    echo "Processing: $filename"
    
    uv run docling-graph convert "$file" \
        --template "$TEMPLATE" \
        --output-dir "$OUTPUT_BASE/$filename" \
        --backend "llm" \
        --inference "remote" \
        --processing-mode "many-to-one"
    
    echo "Completed: $filename"
done

echo "All documents processed!"
```

**Parallel Processing:**
```bash
# Using GNU parallel (faster)
ls documents/*.pdf | parallel -j 4 \
    uv run docling-graph convert {} \
        --template "templates.Invoice" \
        --output-dir "outputs/{/.}" \
        --backend llm \
        --inference remote
```

---

## Advanced Recipes

### Recipe 11: Custom Model Configuration

```bash
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --backend "llm" \
    --inference "remote" \
    --provider "openai" \
    --model "gpt-4-turbo" \
    --output-dir "outputs/custom_model"
```

### Recipe 12: Minimal Export

```bash
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --backend "llm" \
    --inference "local" \
    --no-docling-json \
    --no-markdown \
    --no-per-page \
    --output-dir "outputs/minimal"
```

### Recipe 13: Reverse Edges

```bash
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --backend "llm" \
    --inference "remote" \
    --reverse-edges \
    --output-dir "outputs/bidirectional"
```

### Recipe 14: Development Workflow

```bash
# Enable verbose logging for debugging
uv run docling-graph --verbose convert document.pdf \
    --template "templates.Invoice" \
    --backend "llm" \
    --inference "local" \
    --output-dir "test_output"

# Inspect results
uv run docling-graph inspect test_output/

# Check statistics
cat test_output/graph_stats.json | jq
```

---

## Provider-Specific Recipes

### OpenAI

```bash
export OPENAI_API_KEY="your-key"

uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --backend "llm" \
    --inference "remote" \
    --provider "openai" \
    --model "gpt-4-turbo"
```

### Google Gemini

```bash
export GEMINI_API_KEY="your-key"

uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --backend "llm" \
    --inference "remote" \
    --provider "gemini" \
    --model "gemini-2.5-flash"
```

### IBM watsonx

```bash
export WATSONX_API_KEY="your-key"

uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --backend "llm" \
    --inference "remote" \
    --provider "watsonx" \
    --model "ibm/granite-13b-chat-v2"
```

### vLLM (Local GPU)

```bash
# Start vLLM server
uv run python -m vllm.entrypoints.openai.api_server \
    --model "ibm-granite/granite-4.0-1b" \
    --port 8000

# Use with docling-graph
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --backend "llm" \
    --inference "local" \
    --provider "vllm"
```

---

## Workflow Recipes

### Complete Development Workflow

```bash
# 1. Initialize configuration
uv run docling-graph init

# 2. Create template
cat > templates/my_template.py << 'EOF'
from pydantic import BaseModel, Field

class MyTemplate(BaseModel):
    """My custom template."""
    title: str = Field(description="Document title")
    content: str = Field(description="Main content")
EOF

# 3. Test extraction
uv run docling-graph convert test.pdf \
    --template "templates.my_template.MyTemplate" \
    --output-dir "test_output"

# 4. Inspect results
uv run docling-graph inspect test_output/

# 5. Iterate on template
# Edit templates/my_template.py

# 6. Re-run
uv run docling-graph convert test.pdf \
    --template "templates.my_template.MyTemplate" \
    --output-dir "test_output"
```

### Production Deployment

```bash
# 1. Use configuration file
uv run docling-graph init

# 2. Process documents
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --output-dir "production/$(date +%Y%m%d)"

# 3. Export to Neo4j
cat production/$(date +%Y%m%d)/graph.cypher | \
    cypher-shell -u neo4j -p password

# 4. Archive outputs
tar -czf "archive_$(date +%Y%m%d).tar.gz" production/$(date +%Y%m%d)
```

---

## Best Practices

### 1. Use Configuration Files

```bash
# ✅ Good - Reusable configuration
uv run docling-graph init
uv run docling-graph convert doc.pdf -t "templates.Invoice"

# ❌ Avoid - Repeating options
uv run docling-graph convert doc.pdf -t "templates.Invoice" \
    --backend llm --inference remote --provider mistral
```

### 2. Organize Outputs

```bash
# ✅ Good - Organized by document
uv run docling-graph convert invoice_001.pdf \
    --template "templates.Invoice" \
    --output-dir "outputs/invoices/invoice_001"

# ❌ Avoid - Overwriting outputs
uv run docling-graph convert invoice_001.pdf \
    --template "templates.Invoice"
```

### 3. Use Appropriate Backend

```bash
# ✅ Good - VLM for forms
uv run docling-graph convert form.jpg \
    --template "templates.IDCard" \
    --backend vlm

# ✅ Good - LLM for documents
uv run docling-graph convert research.pdf \
    --template "templates.Research" \
    --backend llm
```

---

## Next Steps

1. **[Python API →](../api/index.md)** - Programmatic usage
2. **[Examples →](../examples/index.md)** - Real-world examples
3. **[Advanced Topics →](../advanced/index.md)** - Custom backends

---

## Quick Reference

### Common Patterns

```bash
# VLM from image
uv run docling-graph convert image.jpg -t "templates.Form" --backend vlm

# Remote LLM
uv run docling-graph convert doc.pdf -t "templates.Invoice" \
    --backend llm --inference remote

# Local LLM
uv run docling-graph convert doc.pdf -t "templates.Invoice" \
    --backend llm --inference local --provider ollama

# With consolidation
uv run docling-graph convert doc.pdf -t "templates.Research" \
    --llm-consolidation

# Cypher export
uv run docling-graph convert doc.pdf -t "templates.Invoice" \
    --export-format cypher

# Batch processing
for pdf in docs/*.pdf; do
    uv run docling-graph convert "$pdf" -t "templates.Invoice" \
        --output-dir "outputs/$(basename $pdf .pdf)"
done
```