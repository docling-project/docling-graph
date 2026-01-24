# Document Conversion


## Overview

**Document Conversion** is the first stage of the extraction pipeline, transforming raw PDFs and images into structured DoclingDocument format. This stage uses the Docling library to perform OCR, layout analysis, and content extraction.

**In this guide:**
- OCR vs Vision pipelines
- Layout analysis
- Table extraction
- Multi-language support
- Performance optimization

---

## Docling Pipelines

### Quick Comparison

| Pipeline | Best For | Speed | Accuracy | GPU Required |
|:---------|:---------|:------|:---------|:-------------|
| **OCR** | Standard documents | Fast | High | No |
| **Vision** | Complex layouts | Slower | Very High | Recommended |

---

## OCR Pipeline (Default)

### What is OCR Pipeline?

The **OCR (Optical Character Recognition) pipeline** is the default and most accurate for standard documents. It uses traditional OCR engines combined with layout analysis.

### Configuration

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    docling_config="ocr"  # Default
)
```

### Features

✅ **Strengths:**
- Fast processing
- High accuracy for text
- Excellent table extraction
- Multi-language support
- No GPU required

❌ **Limitations:**
- May struggle with complex layouts
- Less effective for handwriting
- Requires clear text

### When to Use OCR

Use OCR pipeline for:
- Standard business documents
- Invoices and forms
- Reports and contracts
- Documents with clear text
- Batch processing (faster)

---

## Vision Pipeline

### What is Vision Pipeline?

The **Vision pipeline** uses Vision-Language Models (VLMs) to understand document layout and content visually, similar to how humans read documents.

### Configuration

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    docling_config="vision"  # Vision pipeline
)
```

### Features

✅ **Strengths:**
- Excellent for complex layouts
- Handles handwriting better
- Understands visual context
- Better for images
- Robust to noise

❌ **Limitations:**
- Slower processing
- Requires more memory
- GPU recommended
- Higher resource usage

### When to Use Vision

Use Vision pipeline for:
- Complex layouts (magazines, brochures)
- Handwritten documents
- Low-quality scans
- Documents with images
- Visual-heavy content

---

## Document Processor

### Basic Usage

```python
from docling_graph.core.extractors import DocumentProcessor

# Initialize with OCR pipeline
processor = DocumentProcessor(docling_config="ocr")

# Convert document
document = processor.convert_to_docling_doc("document.pdf")

print(f"Converted {document.num_pages()} pages")
```

### With Vision Pipeline

```python
# Initialize with Vision pipeline
processor = DocumentProcessor(docling_config="vision")

# Convert document
document = processor.convert_to_docling_doc("complex_document.pdf")
```

---

## DoclingDocument Structure

### What is DoclingDocument?

A **DoclingDocument** is a structured representation of your document containing:
- Page information
- Layout elements
- Text content
- Tables
- Images
- Metadata

### Accessing Document Data

```python
# Get number of pages
num_pages = document.num_pages()

# Get page keys
page_keys = sorted(document.pages.keys())

# Access specific page
page = document.pages[page_keys[0]]

# Get document metadata
metadata = document.metadata
```

---

## Markdown Extraction

### Full Document Markdown

```python
# Extract complete document as markdown
full_markdown = processor.extract_full_markdown(document)

print(f"Document length: {len(full_markdown)} characters")
```

**Output example:**
```markdown
# Invoice

**Invoice Number:** INV-001
**Date:** 2024-01-15

## Items

| Description | Quantity | Price |
|-------------|----------|-------|
| Product A   | 2        | $50   |
| Product B   | 1        | $100  |

**Total:** $200
```

### Per-Page Markdown

```python
# Extract markdown for each page
page_markdowns = processor.extract_page_markdowns(document)

for i, page_md in enumerate(page_markdowns, 1):
    print(f"Page {i}: {len(page_md)} characters")
```

---

## Layout Analysis

### What is Layout Analysis?

Layout analysis identifies document structure:
- Headers and footers
- Sections and paragraphs
- Tables and lists
- Images and figures
- Captions and footnotes

### Accessing Layout Information

```python
# Get document structure
for page_no, page in document.pages.items():
    print(f"Page {page_no}:")
    
    # Access layout elements
    for element in page.elements:
        print(f"  - {element.type}: {element.text[:50]}")
```

---

## Table Extraction

### Automatic Table Detection

The OCR pipeline automatically detects and extracts tables:

```python
# Tables are preserved in markdown
markdown = processor.extract_full_markdown(document)

# Tables appear as markdown tables
# | Column 1 | Column 2 |
# |----------|----------|
# | Value 1  | Value 2  |
```

### Table Structure

Tables are extracted with:
- Column headers
- Row data
- Cell alignment
- Merged cells (when possible)

---

## Multi-Language Support

### Supported Languages

The OCR pipeline supports multiple languages:

```python
from docling_graph.core.extractors import DocumentProcessor

# Default: English and French
processor = DocumentProcessor(docling_config="ocr")

# Document will be processed with both languages
document = processor.convert_to_docling_doc("multilingual.pdf")
```

### Language Configuration

Currently configured for:
- English (en)
- French (fr)

**Note:** Language configuration is set in the DocumentProcessor initialization and can be extended by modifying the source code.

---

## Performance Optimization

### OCR Pipeline Optimization

```python
from docling.datamodel.accelerator_options import AcceleratorDevice

# The OCR pipeline is pre-configured with:
# - 4 threads for parallel processing
# - Auto device selection (GPU if available)
# - Optimized table structure matching
```

### Vision Pipeline Optimization

```python
# Vision pipeline automatically uses:
# - GPU acceleration (if available)
# - Optimized batch processing
# - Memory-efficient processing
```

---

## Complete Examples

### Example 1: Basic OCR Conversion

```python
from docling_graph.core.extractors import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(docling_config="ocr")

# Convert document
document = processor.convert_to_docling_doc("invoice.pdf")

# Extract markdown
markdown = processor.extract_full_markdown(document)

print(f"Converted {document.num_pages()} pages")
print(f"Markdown length: {len(markdown)} characters")
```

### Example 2: Vision Pipeline for Complex Layout

```python
from docling_graph.core.extractors import DocumentProcessor

# Initialize with vision pipeline
processor = DocumentProcessor(docling_config="vision")

# Convert complex document
document = processor.convert_to_docling_doc("magazine.pdf")

# Extract per-page markdown
pages = processor.extract_page_markdowns(document)

for i, page in enumerate(pages, 1):
    print(f"Page {i}: {len(page)} characters")
```

### Example 3: Batch Processing

```python
from docling_graph.core.extractors import DocumentProcessor
from pathlib import Path

# Initialize processor once
processor = DocumentProcessor(docling_config="ocr")

# Process multiple documents
for pdf_file in Path("documents").glob("*.pdf"):
    print(f"Processing {pdf_file.name}")
    
    document = processor.convert_to_docling_doc(str(pdf_file))
    markdown = processor.extract_full_markdown(document)
    
    # Save markdown
    output_file = pdf_file.with_suffix(".md")
    output_file.write_text(markdown)

# Cleanup resources
processor.cleanup()
```

---

## Integration with Pipeline

### Automatic Conversion

When using PipelineConfig, conversion happens automatically:

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    docling_config="ocr"  # Conversion happens automatically
)

config.run()
```

### Manual Conversion

For more control, use DocumentProcessor directly:

```python
from docling_graph.core.extractors import DocumentProcessor

# Manual conversion
processor = DocumentProcessor(docling_config="ocr")
document = processor.convert_to_docling_doc("document.pdf")

# Now use document for extraction
# ... extraction code ...

# Cleanup
processor.cleanup()
```

---

## Export Options

### Export Docling JSON

```python
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_docling_json=True  # Export DoclingDocument as JSON
)
```

**Output:** `outputs/document.json`

### Export Markdown

```python
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_markdown=True  # Export as markdown
)
```

**Output:** `outputs/document.md`

### Export Per-Page Markdown

```python
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_per_page_markdown=True  # Export each page
)
```

**Output:** `outputs/pages/page_001.md`, `page_002.md`, etc.

---

## Advanced Features

### Custom Pipeline Options

For advanced use cases, you can customize the pipeline:

```python
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

# Create custom options
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.ocr_options.lang = ["en", "de", "fr"]  # Multiple languages

# Set accelerator options
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=8,  # More threads
    device=AcceleratorDevice.CUDA  # Force GPU
)

# Note: This requires modifying DocumentProcessor source code
```

---

## Troubleshooting

### Issue: Conversion Fails

**Solution:**
```python
try:
    document = processor.convert_to_docling_doc("document.pdf")
except Exception as e:
    print(f"Conversion failed: {e}")
    # Check if file exists
    # Check if file is valid PDF
    # Try with different pipeline
```

### Issue: Poor OCR Quality

**Solution:**
```python
# Try Vision pipeline instead
processor = DocumentProcessor(docling_config="vision")
document = processor.convert_to_docling_doc("document.pdf")
```

### Issue: Slow Conversion

**Solution:**
```python
# Use OCR pipeline (faster)
processor = DocumentProcessor(docling_config="ocr")

# Or process in batches
# ... batch processing code ...
```

### Issue: Out of Memory

**Solution:**
```python
# Process pages individually
processor = DocumentProcessor(docling_config="ocr")
document = processor.convert_to_docling_doc("large_doc.pdf")

# Extract per-page to reduce memory
pages = processor.extract_page_markdowns(document)

# Process each page separately
for page in pages:
    # ... process page ...
    pass

# Cleanup
processor.cleanup()
```

---

## Best Practices

### 1. Choose the Right Pipeline

```python
# ✅ Good - Match pipeline to document type
if document_is_standard:
    docling_config = "ocr"  # Faster
else:
    docling_config = "vision"  # More accurate
```

### 2. Cleanup Resources

```python
# ✅ Good - Always cleanup
processor = DocumentProcessor(docling_config="ocr")
try:
    document = processor.convert_to_docling_doc("document.pdf")
    # ... process document ...
finally:
    processor.cleanup()
```

### 3. Reuse Processor for Batch Processing

```python
# ✅ Good - Reuse processor
processor = DocumentProcessor(docling_config="ocr")

for pdf_file in pdf_files:
    document = processor.convert_to_docling_doc(pdf_file)
    # ... process ...

processor.cleanup()
```

### 4. Export for Debugging

```python
# ✅ Good - Export markdown for inspection
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    export_markdown=True,  # Check conversion quality
    export_per_page_markdown=True  # Debug per page
)
```

---

## Next Steps

Now that you understand document conversion:

1. **[Chunking Strategies →](chunking-strategies.md)** - Learn intelligent document splitting
2. **[Extraction Backends →](extraction-backends.md)** - Choose LLM or VLM backend
3. **[Model Merging →](model-merging.md)** - Consolidate extractions

---

## Quick Reference

### OCR Pipeline (Default)

```python
processor = DocumentProcessor(docling_config="ocr")
document = processor.convert_to_docling_doc("document.pdf")
```

### Vision Pipeline

```python
processor = DocumentProcessor(docling_config="vision")
document = processor.convert_to_docling_doc("document.pdf")
```

### Extract Markdown

```python
# Full document
markdown = processor.extract_full_markdown(document)

# Per page
pages = processor.extract_page_markdowns(document)
```

### Cleanup

```python
processor.cleanup()
```