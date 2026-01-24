# Examples


## Overview

This section provides **complete, end-to-end examples** organized by both **input format** and **domain/use case**. Each example demonstrates how to process different types of documents through the Docling Graph pipeline.

**What's Covered:**
- Complete Pydantic templates
- CLI and Python API usage
- Expected outputs and graph structures
- Troubleshooting tips
- Best practices

---

## Quick Navigation

### By Input Format

| Example                                                | Input Type | Backend |
| ------------------------------------------------------ | ---------- | ------- |
| **[Quickstart](../../introduction/quickstart.md)**                        | PDF/Image  | VLM/LLM |
| **[URL](url-input.md)**                          | URL (PDF)  | LLM     |
| **[Markdown](markdown-input.md)**                | Markdown   | LLM     |
| **[DoclingDocument](docling-document-input.md)** | JSON       | LLM     |

### By Domain/Use Case

| Example                                         | Domain   | Input Type |
| ----------------------------------------------- | -------- | ---------- |
| **[Invoice Extraction](invoice-extraction.md)** | Business | PDF/Image  |
| **[ID Card](id-card.md)**                       | Identity | Image      |
| **[Insurance Policy](insurance-policy.md)**     | Legal    | PDF        |
| **[Research Paper](research-paper.md)**         | Academic | PDF        |

---

## Section 1: Examples by Input Format

Learn how to work with different input types and understand the pipeline's flexibility.

### 1. [Quickstart](../../introduction/quickstart.md)
**5-Minute Introduction**

Get started quickly with a simple document extraction example using traditional PDF or image inputs.

- **Input:** PDF or Image file
- **Use Case:** Invoice extraction
- **Backend:** VLM (recommended) or LLM
- **Features:** Basic extraction, graph visualization

**Perfect for:** First-time users wanting a quick introduction.

---

### 2. [URL Input](url-input.md)
**Processing Documents from URLs**

Learn how to process documents directly from URLs without manual downloads.

- **Input:** URL (e.g., `https://arxiv.org/pdf/2207.02720`)
- **Use Case:** Research paper analysis
- **Backend:** LLM
- **Features:** Automatic download, content type detection, remote processing

**What You'll Learn:**
- URL-based workflows
- Automatic content type detection
- Download configuration (timeout, size limits)
- Network error handling

**Perfect for:** Processing documents from web sources, automated pipelines.

---

### 3. [Markdown Input](markdown-input.md)
**Processing Markdown Documents**

Extract structured data from Markdown files like README.md or documentation.

- **Input:** Markdown file (`.md`)
- **Use Case:** Documentation analysis
- **Backend:** LLM (required)
- **Features:** Text-only processing, no OCR needed, fast extraction

**What You'll Learn:**
- Text-only extraction workflow
- Processing documentation
- Markdown structure preservation
- Batch processing multiple files

**Perfect for:** Documentation analysis, knowledge base extraction.

---

### 4. [DoclingDocument Input](docling-document-input.md)
**Reprocessing Pre-Converted Documents**

Use pre-processed DoclingDocument JSON files for fast reprocessing without OCR.

- **Input:** DoclingDocument JSON file
- **Use Case:** Invoice reprocessing
- **Backend:** LLM or VLM
- **Features:** Skip OCR, fast reprocessing, template experimentation

**What You'll Learn:**
- Creating DoclingDocument files
- Two-stage processing workflows
- Template experimentation
- Batch reprocessing

**Perfect for:** Reprocessing documents, A/B testing templates, incremental workflows.

---

## Section 2: Examples by Domain/Use Case

Explore complete, domain-specific examples with production-ready templates.

### 5. [Invoice Extraction](invoice-extraction.md)
**Complete Invoice Processing**

Extract structured data from invoices including issuer, client, line items, and totals.

- **Domain:** Business/Finance
- **Input:** PDF or Image
- **Backend:** VLM (recommended) or LLM
- **Features:** Nested entities, relationships, validation

**What You'll Learn:**
- Creating entity and component models
- Defining graph relationships
- Using edge() helper
- Handling addresses and line items

**Perfect for:** Business document processing, accounting automation.

---

### 6. [ID Card](id-card.md)
**Identity Document Extraction**

Extract personal information from ID cards and identity documents.

- **Domain:** Identity/Government
- **Input:** Image (photo of ID card)
- **Backend:** VLM (recommended)
- **Features:** Structured personal data, validation

**What You'll Learn:**
- Processing identity documents
- Handling personal information
- Data validation and formatting
- Privacy considerations

**Perfect for:** KYC processes, identity verification systems.

---

### 7. [Insurance Policy](insurance-policy.md)
**Legal Document Analysis**

Extract structured information from insurance policies and legal documents.

- **Domain:** Legal/Insurance
- **Input:** PDF (multi-page)
- **Backend:** LLM with chunking
- **Features:** Complex document structure, multi-page processing

**What You'll Learn:**
- Processing long documents
- Handling complex legal terminology
- Multi-page extraction strategies
- Document consolidation

**Perfect for:** Legal document processing, insurance automation.

---

### 8. [Research Paper](research-paper.md)
**Scientific Document Analysis**

Extract structured data from academic papers including authors, methodology, and findings.

- **Domain:** Academic/Research
- **Input:** PDF (scientific paper)
- **Backend:** LLM with chunking
- **Features:** Complex structure, citations, methodology

**What You'll Learn:**
- Processing scientific documents
- Extracting methodology and findings
- Handling citations and references
- Academic document structure

**Perfect for:** Research automation, literature review systems.

---

## Input Format Comparison

| Format | OCR Required | Processing Speed | Backend Support | Best For |
|--------|--------------|------------------|-----------------|----------|
| **PDF** | ✅ Yes | 🐢 Slow | LLM + VLM | Scanned documents, forms |
| **Image** | ✅ Yes | 🐢 Slow | LLM + VLM | Photos, scans |
| **URL** | Depends | ⚡ Variable | LLM + VLM | Remote documents |
| **Markdown** | ❌ No | ⚡ Fast | LLM only | Documentation, notes |
| **DoclingDocument** | ❌ No | ⚡ Very Fast | LLM only | Reprocessing, experimentation |

---

## Choosing the Right Example

### Start Here

**New to Docling Graph?**
→ Start with [Quickstart](../../introduction/quickstart.md)

### By Input Format

**Processing web documents?**
→ See [URL Input](url-input.md)

**Working with documentation?**
→ See [Markdown Input](markdown-input.md)

**Need to reprocess documents?**
→ See [DoclingDocument Input](docling-document-input.md)

### By Domain/Use Case

**Business Documents:**
- [Invoice Extraction](invoice-extraction.md) - Invoices, receipts, financial documents

**Identity Verification:**
- [ID Card](id-card.md) - ID cards, passports, identity documents

**Legal Documents:**
- [Insurance Policy](insurance-policy.md) - Policies, contracts, legal agreements

**Academic Research:**
- [Research Paper](research-paper.md) - Scientific papers, academic documents
- [URL Input](url-input.md) - Process from arXiv, PubMed, etc.

**Documentation:**
- [Markdown Input](markdown-input.md) - README files, project documentation

**Workflow Optimization:**
- [DoclingDocument Input](docling-document-input.md) - Fast reprocessing, template testing

---

## Common Workflows

### Workflow 1: URL → Extract → Visualize

```bash
# Download and process in one step
uv run docling-graph convert "https://arxiv.org/pdf/2207.02720" \
    --template "templates.research.Research" \
    --processing-mode "many-to-one"

# Visualize results
uv run docling-graph inspect outputs
```

### Workflow 2: PDF → DoclingDocument → Reprocess

```bash
# Step 1: Initial processing with DoclingDocument export
uv run docling-graph convert invoice.pdf \
    --template "templates.invoice.BasicInvoice" \
    --export-docling-json

# Step 2: Reprocess with different template (no OCR)
uv run docling-graph convert outputs/invoice_docling.json \
    --template "templates.invoice.DetailedInvoice"
```

### Workflow 3: Batch Markdown Processing

```bash
# Process all markdown files
for file in docs/**/*.md; do
    uv run docling-graph convert "$file" \
        --template "templates.documentation.Documentation" \
        --backend llm \
        --output-dir "outputs/$(basename $file .md)"
done
```

---

## Template Examples

All examples use Pydantic templates. Here's a quick reference:

### Simple Entity

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    """Person entity."""
    model_config = {
        'is_entity': True,
        'graph_id_fields': ['name']
    }
    
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")
```

### With Relationships

```python
from docling_graph.utils import edge

class Organization(BaseModel):
    """Organization with employees."""
    model_config = {'is_entity': True}
    
    name: str = Field(description="Organization name")
    employees: list[Person] = edge(
        "EMPLOYS",
        description="Organization employees"
    )
```

### Complete Examples

See individual example pages for complete, domain-specific templates.

---

## Additional Resources

### Documentation

- **[Input Formats Guide](../../fundamentals/pipeline-configuration/input-formats.md)** - Complete input format reference
- **[Backend Selection](../../fundamentals/pipeline-configuration/backend-selection.md)** - Choose LLM vs VLM
- **[Processing Modes](../../fundamentals/pipeline-configuration/processing-modes.md)** - One-to-one vs many-to-one

### API Reference

- **[PipelineConfig](../api/pipeline-config.md)** - Configuration options
- **[run_pipeline](../api/run-pipeline.md)** - Pipeline execution
- **[Batch Processing](../api/batch-processing.md)** - Process multiple documents

### Advanced Topics

- **[Performance Tuning](../advanced/performance-tuning.md)** - Optimize processing
- **[Error Handling](../advanced/error-handling.md)** - Handle failures gracefully
- **[Custom Backends](../advanced/custom-backends.md)** - Extend functionality

---

## Getting Help

### Common Issues

**"VLM backend does not support text-only inputs"**
→ Use `--backend llm` for Markdown and text files

**"URL download timeout"**
→ Increase timeout or download manually first

**"Text input is empty"**
→ Check file content and encoding

**"Invalid DoclingDocument schema"**
→ Verify `schema_name` and `version` fields

### Support

- **Documentation:** [https://ibm.github.io/docling-graph](https://ibm.github.io/docling-graph)
- **GitHub Issues:** [https://github.com/IBM/docling-graph/issues](https://github.com/IBM/docling-graph/issues)
- **Discussions:** [https://github.com/IBM/docling-graph/discussions](https://github.com/IBM/docling-graph/discussions)

---

## Next Steps

1. **Explore [Input Formats](../../fundamentals/pipeline-configuration/input-formats.md)** - Learn about all supported formats
2. **Read [Advanced Topics](../advanced/index.md)** - Optimize your workflows