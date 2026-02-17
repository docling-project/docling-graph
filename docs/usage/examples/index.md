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


---

## Quick Navigation

### By Input Format

| Example | Input Type | Backend |
|---------|------------|---------|
| [Quickstart](../../introduction/quickstart.md) | PDF/Image | VLM/LLM |
| [URL Input](url-input.md) | URL | LLM |
| [Markdown Input](markdown-input.md) | Markdown | LLM |
| [DoclingDocument Input](docling-document-input.md) | JSON | LLM |

### By Domain

| Example | Domain | Input |
|---------|--------|-------|
| [Billing Document Extraction](billing-document.md) | Business | PDF/Image |
| [ID Card](id-card.md) | Identity | Image |
| [Insurance Policy](insurance-policy.md) | Legal | PDF |
| [Rheology Research](rheology_research.md) | Academic | PDF |


| Format | OCR Required | Processing Speed | Backend Support | Best For |
|--------|--------------|------------------|-----------------|----------|
| **PDF** | ✅ Yes | 🐢 Slow | LLM + VLM | Scanned documents, forms |
| **Image** | ✅ Yes | 🐢 Slow | LLM + VLM | Photos, scans |
| **URL** | Depends | ⚡ Variable | LLM + VLM | Remote documents |
| **Markdown** | ❌ No | ⚡ Fast | LLM only | Documentation, notes |
| **DoclingDocument** | ❌ No | ⚡ Very Fast | LLM only | Reprocessing, experimentation |

---

## Choosing the Right Example

**New to Docling Graph?** → [Quickstart](../../introduction/quickstart.md)

**By Input Format:**
- Web documents → [URL Input](url-input.md)
- Documentation → [Markdown Input](markdown-input.md)
- Reprocessing → [DoclingDocument Input](docling-document-input.md)

**By Domain:**
- Business → [Billing Document Extraction](billing-document.md)
- Identity → [ID Card](id-card.md)
- Legal → [Insurance Policy](insurance-policy.md)
- Academic → [Rheology Research](rheology_research.md)


### Workflow 1: URL → Extract → Visualize

```bash
# Download and process in one step
uv run docling-graph convert "https://arxiv.org/pdf/2207.02720" \
    --template "docs.examples.templates.rheology_research.ScholarlyRheologyPaper" \
    --processing-mode "many-to-one"

# Visualize results
uv run docling-graph inspect outputs
```

### Workflow 2: PDF → DoclingDocument → Reprocess

```bash
# Step 1: Initial processing with DoclingDocument export
uv run docling-graph convert billing_doc.pdf \
    --template "templates.billing_document.BasicBillingDocument" \
    --export-docling-json

# Step 2: Reprocess with different template (no OCR)
uv run docling-graph convert outputs/billing_doc_docling.json \
    --template "templates.billing_document.DetailedBillingDocument"
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

### Simple Entity

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    """Person entity."""
    model_config = {'is_entity': True, 'graph_id_fields': ['name']}
    name: str = Field(description="Person's name")
```

### With Relationships

```python
from docling_graph.utils import edge

class Organization(BaseModel):
    name: str
    employees: list[Person] = edge("EMPLOYS")
```

See individual example pages for complete templates.


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
- **GitHub Issues:** [https://github.com/docling-project/docling-graph/issues](https://github.com/docling-project/docling-graph/issues)
- **Discussions:** [https://github.com/docling-project/docling-graph/discussions](https://github.com/docling-project/docling-graph/discussions)

---

## Next Steps

1. **Explore [Input Formats](../../fundamentals/pipeline-configuration/input-formats.md)** - Learn about all supported formats
2. **Read [Advanced Topics](../advanced/index.md)** - Optimize your workflows