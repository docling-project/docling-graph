# CLI Recipes

This guide provides **ready-to-use CLI commands** for all example scripts. Run these from your project root directory.

---

## Quick Reference

| Recipe | Script | Backend | Use Case |
|--------|--------|---------|----------|
| [01: VLM from Image](#vlm-from-image) | `01_quickstart_vlm_image.py` | VLM | Forms, invoices, ID cards |
| [02: LLM from PDF](#llm-from-pdf) | `02_quickstart_llm_pdf.py` | LLM (Remote) | Rheology researchs, reports |
| [03: URL Processing](#url-processing) | `03_url_processing.py` | LLM (Remote) | Remote documents |
| [04: Input Formats](#input-formats) | `04_input_formats.py` | LLM | Text, Markdown, JSON |
| [05: Processing Modes](#processing-modes) | `05_processing_modes.py` | LLM (Local) | Mode comparison |
| [06: Export Formats](#export-formats) | `06_export_formats.py` | VLM | CSV, Cypher, JSON |
| [07: Local Inference](#local-inference) | `07_local_inference.py` | LLM (Local) | Offline processing |
| [08: Chunking](#chunking-consolidation) | `08_chunking_consolidation.py` | LLM (Remote) | Large documents |
| [09: Batch Processing](#batch-processing) | `09_batch_processing.py` | VLM | Multiple documents |
| [10: Multi-Provider](#multi-provider) | `10_provider_configs.py` | LLM (Remote) | Provider comparison |

---

## 📍 VLM from Image

**Python Script:** `01_quickstart_vlm_image.py`

**Use Case:** Extract structured data from invoice images

**CLI Command:**
```bash
docling-graph convert "docs/examples/data/invoice/sample_invoice.jpg" \
    --template "docs.examples.templates.billing_document.BillingDocument" \
    --output-dir "outputs/cli_01" \
    --backend "vlm" \
    --processing-mode "one-to-one" \
    --docling-pipeline "vision"
```

**When to Use:**

- ✅ Single-page forms or invoices
- ✅ ID cards, badges, receipts
- ✅ Image files (JPG, PNG)
- ✅ Structured layouts

---

## 📍 LLM from PDF

**Python Script:** `02_quickstart_llm_pdf.py`

**Use Case:** Extract from multi-page rheology researchs

**Prerequisites:**
```bash
pip install docling-graph
export MISTRAL_API_KEY="your-api-key"
```

**CLI Command:**
```bash
docling-graph convert "docs/examples/data/research_paper/rheology.pdf" \
    --template "docs.examples.templates.rheology_research.ScholarlyRheologyPaper" \
    --output-dir "outputs/cli_02" \
    --backend "llm" \
    --inference "remote" \
    --provider "mistral" \
    --model "mistral-large-latest" \
    --processing-mode "many-to-one" \
    --use-chunking
```

**When to Use:**

- ✅ Multi-page documents
- ✅ Text-heavy content
- ✅ Rheology researchs, reports
- ✅ Complex narratives

---

## 📍 URL Processing

**Python Script:** `03_url_processing.py`

**Use Case:** Download and process documents from URLs

**Prerequisites:**
```bash
pip install docling-graph
export MISTRAL_API_KEY="your-api-key"
```

**CLI Command:**
```bash
docling-graph convert "https://arxiv.org/pdf/2207.02720" \
    --template "docs.examples.templates.rheology_research.ScholarlyRheologyPaper" \
    --output-dir "outputs/cli_03" \
    --backend "llm" \
    --inference "remote" \
    --provider "mistral" \
    --model "mistral-large-latest" \
    --processing-mode "many-to-one" \
    --use-chunking
```

**When to Use:**

- ✅ arXiv papers
- ✅ Web-hosted PDFs
- ✅ Automated ingestion
- ✅ Remote document processing

---

## 📍 Input Formats

**Python Script:** `04_input_formats.py`

**Use Case:** Process text, Markdown, and DoclingDocument formats

**Text File:**
```bash
# Create sample text file
echo "Title: Sample Document
Summary: This is a test document.
Key Points:
- Point 1
- Point 2" > sample.txt

# Process text file
docling-graph convert "sample.txt" \
    --template "docs.examples.templates.simple.SimpleDocument" \
    --output-dir "outputs/cli_04_text" \
    --backend "llm" \
    --inference "remote" \
    --provider "mistral"
```

**Markdown File:**
```bash
# Process markdown file
docling-graph convert "README.md" \
    --template "docs.examples.templates.simple.SimpleDocument" \
    --output-dir "outputs/cli_04_markdown" \
    --backend "llm"
```

**When to Use:**

- ✅ Documentation files
- ✅ Plain text content
- ✅ Reprocessing (DoclingDocument)
- ✅ Skip OCR for speed

---

## 📍 Processing Modes

**Python Script:** `05_processing_modes.py`

**Use Case:** Compare one-to-one vs many-to-one modes

**Prerequisites:**
```bash
ollama serve
ollama pull llama3:8b
pip install docling-graph
```

**One-to-One Mode:**
```bash
docling-graph convert "docs/examples/data/id_card/multi_french_id_cards.pdf" \
    --template "docs.examples.templates.id_card.IDCard" \
    --output-dir "outputs/cli_05_one_to_one" \
    --backend "llm" \
    --inference "local" \
    --provider "ollama" \
    --model "llama3:8b" \
    --processing-mode "one-to-one" \
    --no-use-chunking
```

**Many-to-One Mode:**
```bash
docling-graph convert "docs/examples/data/id_card/multi_french_id_cards.pdf" \
    --template "docs.examples.templates.id_card.IDCard" \
    --output-dir "outputs/cli_05_many_to_one" \
    --backend "llm" \
    --inference "local" \
    --provider "ollama" \
    --model "llama3:8b" \
    --processing-mode "many-to-one" \
    --use-chunking
```

---

## 📍 Export Formats

**Python Script:** `06_export_formats.py`

**Use Case:** Generate different export formats for Neo4j

**CSV Export (Bulk Import):**
```bash
docling-graph convert "docs/examples/data/invoice/sample_invoice.jpg" \
    --template "docs.examples.templates.billing_document.BillingDocument" \
    --output-dir "outputs/cli_06_csv" \
    --backend "vlm" \
    --export-format "csv"
```

**Cypher Export (Script):**
```bash
docling-graph convert "docs/examples/data/invoice/sample_invoice.jpg" \
    --template "docs.examples.templates.billing_document.BillingDocument" \
    --output-dir "outputs/cli_06_cypher" \
    --backend "vlm" \
    --export-format "cypher"
```

**Neo4j Import:**
```bash
# CSV bulk import
neo4j-admin database import full \
    --nodes=outputs/cli_06_csv/docling_graph/nodes.csv \
    --relationships=outputs/cli_06_csv/docling_graph/edges.csv

# Cypher script
cat outputs/cli_06_cypher/docling_graph/graph.cypher | \
    cypher-shell -u neo4j -p password
```

---

## 📍 Local Inference

**Python Script:** `07_local_inference.py`

**Use Case:** Privacy-focused offline processing

**Prerequisites:**
```bash
ollama serve
ollama pull llama3:8b
pip install docling-graph
```

**CLI Command:**
```bash
docling-graph convert "docs/examples/data/research_paper/rheology.pdf" \
    --template "docs.examples.templates.rheology_research.ScholarlyRheologyPaper" \
    --output-dir "outputs/cli_07" \
    --backend "llm" \
    --inference "local" \
    --provider "ollama" \
    --model "llama3:8b" \
    --processing-mode "many-to-one" \
    --use-chunking
```

**When to Use:**

- ✅ Privacy-sensitive documents
- ✅ Offline processing
- ✅ No API costs
- ✅ Development and testing

---

## 📍 Chunking & Consolidation

**Python Script:** `08_chunking_consolidation.py`

**Use Case:** Compare consolidation strategies

**Prerequisites:**
```bash
pip install docling-graph
export MISTRAL_API_KEY="your-api-key"
```

**Programmatic Merge (Fast):**
```bash
docling-graph convert "docs/examples/data/research_paper/rheology.pdf" \
    --template "docs.examples.templates.rheology_research.ScholarlyRheologyPaper" \
    --output-dir "outputs/cli_08_programmatic" \
    --backend "llm" \
    --inference "remote" \
    --provider "mistral" \
    --processing-mode "many-to-one" \
    --use-chunking
```

---

## 📍 Batch Processing

**Python Script:** `09_batch_processing.py`

**Use Case:** Process multiple documents efficiently

**Bash Script:**
```bash
#!/bin/bash
# Process all invoices in a directory

for file in docs/examples/data/invoice/*.jpg; do
    filename=$(basename "$file" .jpg)
    echo "Processing $filename..."
    
    docling-graph convert "$file" \
        --template "docs.examples.templates.billing_document.BillingDocument" \
        --output-dir "outputs/cli_09/$filename" \
        --backend "vlm" \
        --processing-mode "one-to-one"
done

echo "Batch processing complete!"
```

---

## 📍 Multi-Provider

**Python Script:** `10_provider_configs.py`

**Use Case:** Compare different LLM providers

**Prerequisites:**
```bash
# Set API keys for providers you want to test
export OPENAI_API_KEY="sk-..."
export MISTRAL_API_KEY="..."
export GEMINI_API_KEY="..."
export WATSONX_API_KEY="..."
export WATSONX_PROJECT_ID="..."

pip install docling-graph
```

**OpenAI:**
```bash
docling-graph convert "docs/examples/data/research_paper/rheology.pdf" \
    --template "docs.examples.templates.rheology_research.ScholarlyRheologyPaper" \
    --output-dir "outputs/cli_10_openai" \
    --backend "llm" \
    --inference "remote" \
    --provider "openai" \
    --model "gpt-4-turbo-preview"
```

**Mistral:**
```bash
docling-graph convert "docs/examples/data/research_paper/rheology.pdf" \
    --template "docs.examples.templates.rheology_research.ScholarlyRheologyPaper" \
    --output-dir "outputs/cli_10_mistral" \
    --backend "llm" \
    --inference "remote" \
    --provider "mistral" \
    --model "mistral-large-latest"
```

**Gemini:**
```bash
docling-graph convert "docs/examples/data/research_paper/rheology.pdf" \
    --template "docs.examples.templates.rheology_research.ScholarlyRheologyPaper" \
    --output-dir "outputs/cli_10_gemini" \
    --backend "llm" \
    --inference "remote" \
    --provider "gemini" \
    --model "gemini-1.5-pro"
```

**WatsonX:**
```bash
docling-graph convert "docs/examples/data/research_paper/rheology.pdf" \
    --template "docs.examples.templates.rheology_research.ScholarlyRheologyPaper" \
    --output-dir "outputs/cli_10_watsonx" \
    --backend "llm" \
    --inference "remote" \
    --provider "watsonx" \
    --model "ibm/granite-4-h-small"
```

---

## Common Options

### Visualization
```bash
# View interactive graph
docling-graph inspect outputs/cli_01

# Open specific HTML file
open outputs/cli_01/docling_graph/graph.html
```

### Debugging
```bash
# Verbose output
docling-graph --verbose convert ...

# Check version
docling-graph --version

# Get help
docling-graph convert --help
```

### Configuration
```bash
# Initialize config file
docling-graph init

# Use custom config
docling-graph convert --config custom_config.yaml ...
```

---

## Troubleshooting

### 🐛 API Key Issues
```bash
# Check if key is set
echo $MISTRAL_API_KEY

# Set key for current session
export MISTRAL_API_KEY="your-key"

# Set permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export MISTRAL_API_KEY="your-key"' >> ~/.bashrc
```

### 🐛 Ollama Issues
```bash
# Check if Ollama is running
curl http://localhost:11434

# Start Ollama
ollama serve

# List available models
ollama list

# Pull a model
ollama pull llama3:8b
```

### 🐛 Installation Issues
```bash
# Reinstall
pip install --force-reinstall docling-graph

# Check Python version
python --version  # Should be 3.10+

# Verify installation
python -c "import docling_graph; print(docling_graph.__version__)"
```

---

## Next Steps

1. **[Python API →](../api/index.md)** - Programmatic usage
2. **[Examples →](../examples/index.md)** - Real-world examples
3. **[Advanced Topics →](../advanced/index.md)** - Custom backends