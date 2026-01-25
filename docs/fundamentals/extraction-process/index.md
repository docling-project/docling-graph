# The Extraction Process


## Overview

The **Extraction Process** is the core of Docling Graph, transforming raw documents into structured knowledge graphs through a multi-stage pipeline. This section explains each stage in detail.

**What you'll learn:**
- How documents are converted to structured format
- Intelligent chunking strategies
- Extraction backends (LLM vs VLM)
- Model merging and consolidation
- Pipeline orchestration

---

## The Four-Stage Pipeline

--8<-- "docs/assets/flowcharts/four_stage_pipeline.md"

### Stage 1: Document Conversion

**Purpose:** Convert PDF/images to structured Docling format

**Process:**
- OCR or Vision pipeline
- Layout analysis
- Table extraction
- Text extraction

**Output:** DoclingDocument with structure

**Learn more:** [Document Conversion →](document-conversion.md)

---

### Stage 2: Chunking

**Purpose:** Split document into optimal chunks for LLM processing

**Process:**
- Structure-aware splitting
- Token counting
- Semantic boundaries
- Context preservation

**Output:** List of contextualized chunks

**Learn more:** [Chunking Strategies →](chunking-strategies.md)

---

### Stage 3: Extraction

**Purpose:** Extract structured data using LLM/VLM

**Process:**
- Backend selection (LLM/VLM)
- Batch processing
- Schema validation
- Error handling

**Output:** List of Pydantic models

**Learn more:** [Extraction Backends →](extraction-backends.md)

---

### Stage 4: Merging

**Purpose:** Consolidate multiple extractions into single model

**Process:**
- Programmatic merging
- LLM consolidation (optional)
- Conflict resolution
- Validation

**Output:** Single consolidated model

**Learn more:** [Model Merging →](model-merging.md)

---

## Processing Modes

### Many-to-One (Default)

**Best for:** Most documents

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    processing_mode="many-to-one"  # Default
)
```

**Process:**
1. Convert entire document
2. Chunk intelligently
3. Extract from each chunk
4. Merge into single model

**Output:** 1 consolidated model

---

### One-to-One

**Best for:** Multi-page forms, page-specific data

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    processing_mode="one-to-one"
)
```

**Process:**
1. Convert entire document
2. Extract from each page
3. Return separate models

**Output:** N models (one per page)

---

## Backend Comparison

| Feature | LLM Backend | VLM Backend |
|:--------|:------------|:------------|
| **Input** | Markdown text | Images/PDFs |
| **Accuracy** | High for text | High for visuals |
| **Speed** | Fast | Slower |
| **Cost** | Low (local) | Medium |
| **Best For** | Text documents | Complex layouts |

---

## Pipeline Stages in Code

### Stage Overview

```python
from docling_graph.pipeline.stages import (
    TemplateLoadingStage,    # Load Pydantic template
    ExtractionStage,         # Extract data
    DoclingExportStage,      # Export Docling outputs
    GraphConversionStage,    # Convert to graph
    ExportStage,             # Export graph
    VisualizationStage       # Generate visualizations
)
```

### Orchestration

```python
from docling_graph.pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator(config)
context = orchestrator.run()

# Access results
print(f"Extracted {len(context.extracted_models)} models")
print(f"Graph has {context.graph_metadata.node_count} nodes")
```

---

## Extraction Flow

### Complete Flow Diagram

--8<-- "docs/assets/flowcharts/extraction_flow.md"

---

## Key Concepts

### 1. Document Conversion

Transform raw documents into structured format:

```python
from docling_graph.core.extractors import DocumentProcessor

processor = DocumentProcessor(docling_config="ocr")
document = processor.convert_to_docling_doc("document.pdf")
```

**Learn more:** [Document Conversion →](document-conversion.md)

---

### 2. Chunking

Split documents intelligently:

```python
from docling_graph.core.extractors import DocumentChunker

chunker = DocumentChunker(
    provider="mistral",
    max_tokens=4096
)
chunks = chunker.chunk_document(document)
```

**Learn more:** [Chunking Strategies →](chunking-strategies.md)

---

### 3. Extraction

Extract structured data:

```python
from docling_graph.core.extractors import ExtractorFactory

extractor = ExtractorFactory.create_extractor(
    processing_mode="many-to-one",
    backend_name="llm",
    llm_client=client
)
models, doc = extractor.extract(source, template)
```

**Learn more:** [Extraction Backends →](extraction-backends.md)

---

### 4. Merging

Consolidate multiple models:

```python
from docling_graph.core.utils import merge_pydantic_models

merged = merge_pydantic_models(models, template)
```

**Learn more:** [Model Merging →](model-merging.md)

---

## Performance Optimization

### Chunking vs No Chunking

| Approach | Speed | Accuracy | Memory | Best For |
|:---------|:------|:---------|:-------|:---------|
| **Chunking** | Fast | High | Low | Large docs |
| **No Chunking** | Slow | Medium | High | Small docs |

### Batch Processing

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    use_chunking=True,      # Enable chunking
    max_batch_size=5        # Process 5 chunks at once
)
```

---

## Error Handling

### Extraction Errors

```python
from docling_graph.exceptions import ExtractionError

try:
    run_pipeline(config)
except ExtractionError as e:
    print(f"Extraction failed: {e.message}")
    print(f"Details: {e.details}")
```

### Pipeline Errors

```python
from docling_graph.exceptions import PipelineError

try:
    run_pipeline(config)
except PipelineError as e:
    print(f"Pipeline failed at stage: {e.details['stage']}")
```

---

## Section Contents

### 1. [Document Conversion](document-conversion.md)
Learn how documents are converted to structured format using Docling pipelines.

**Topics:**
- OCR vs Vision pipelines
- Layout analysis
- Table extraction
- Multi-language support

---

### 2. [Chunking Strategies](chunking-strategies.md)
Understand intelligent document chunking for optimal LLM processing.

**Topics:**
- Structure-aware chunking
- Token management
- Semantic boundaries
- Provider-specific optimization

---

### 3. [Extraction Backends](extraction-backends.md)
Deep dive into LLM and VLM extraction backends.

**Topics:**
- LLM backend (text-based)
- VLM backend (vision-based)
- Backend selection
- Performance comparison

---

### 4. [Model Merging](model-merging.md)
Learn how multiple extractions are consolidated into single models.

**Topics:**
- Programmatic merging
- LLM consolidation
- Conflict resolution
- Validation strategies

---

### 5. [Batch Processing](batch-processing.md)
Optimize extraction with intelligent batching.

**Topics:**
- Chunk batching
- Context window management
- Adaptive batch sizing
- Performance tuning

---

### 6. Pipeline Orchestration
Understand how pipeline stages are coordinated through the extraction process.

**Topics:**
- Stage execution
- Context management
- Error handling
- Resource cleanup

---

## Quick Examples

### 📍 Basic Extraction

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="local"
)

run_pipeline(config)
```

### 📍 High-Accuracy Extraction

```python
config = PipelineConfig(
    source="complex_document.pdf",
    template="templates.ResearchPaper",
    backend="vlm",              # Vision backend
    processing_mode="one-to-one",
    docling_config="vision"     # Vision pipeline
)

run_pipeline(config)
```

### 📍 Optimized for Large Documents

```python
config = PipelineConfig(
    source="large_document.pdf",
    template="templates.Contract",
    backend="llm",
    use_chunking=True,          # Enable chunking
    llm_consolidation=True,     # Extra accuracy
    max_batch_size=3            # Smaller batches
)

run_pipeline(config)
```

---

## Best Practices

### 👍 Choose the Right Backend

```python
# ✅ Good - Match backend to document type
if document_has_complex_layout:
    backend = "vlm"
else:
    backend = "llm"
```

### 👍 Enable Chunking for Large Documents

```python
# ✅ Good - Use chunking for efficiency
config = PipelineConfig(
    source="large_doc.pdf",
    template="templates.BillingDocument",
    use_chunking=True  # Recommended
)
```

### 👍 Use LLM Consolidation for Accuracy

```python
# ✅ Good - Extra accuracy for critical data
config = PipelineConfig(
    source="contract.pdf",
    template="templates.Contract",
    llm_consolidation=True  # Higher accuracy
)
```

---

## Troubleshooting

### 🐛 Extraction Returns Empty Results

**Solution:**
```python
# Check document conversion
processor = DocumentProcessor()
document = processor.convert_to_docling_doc("document.pdf")
markdown = processor.extract_full_markdown(document)

if not markdown.strip():
    print("Document conversion failed")
```

### 🐛 Out of Memory

**Solution:**
```python
# Enable chunking and reduce batch size
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    use_chunking=True,
    max_batch_size=1  # Smaller batches
)
```

### 🐛 Slow Extraction

**Solution:**
```python
# Use local backend or disable consolidation
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="local",      # Faster
    llm_consolidation=False  # Skip extra pass
)
```

---

## Next Steps

Ready to dive deeper? Start with:

1. **[Document Conversion →](document-conversion.md)** - Learn about Docling pipelines
2. **[Chunking Strategies →](chunking-strategies.md)** - Optimize document splitting
3. **[Extraction Backends →](extraction-backends.md)** - Choose the right backend