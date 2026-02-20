# Extraction Backends


## Overview

**Extraction backends** are the engines that extract structured data from documents. Docling Graph supports two types: **LLM backends** (text-based) and **VLM backends** (vision-based).

**In this guide:**
- LLM vs VLM comparison
- Backend selection criteria
- Configuration and usage
- Extraction contracts (direct, staged, delta, dense)
- Performance optimization
- Error handling

---

## Backend Types

### Quick Comparison

| Feature | LLM Backend | VLM Backend |
|:--------|:------------|:------------|
| **Input** | Markdown text | Images/PDFs directly |
| **Processing** | Text-based | Vision-based |
| **Accuracy** | High for text | High for visuals |
| **Speed** | Fast | Slower |
| **Cost** | Low (local) / Medium (API) | Medium |
| **GPU** | Optional | Recommended |
| **Best For** | Standard documents | Complex layouts |

---

## LLM Backend

### What is LLM Backend?

The **LLM (Language Model) backend** processes documents as text, using markdown extracted from PDFs. It supports both local and remote models.

### Architecture

--8<-- "docs/assets/flowcharts/llm_backend.md"

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",           # LLM backend
    inference="local",       # or "remote"
    provider_override="ollama",
    model_override="llama3.1:8b"
)
```

---

### Extraction contracts (LLM backend)

For many-to-one extraction with the LLM backend you can choose:

- **direct** (default): Single-pass extraction; chunks are extracted and merged programmatically.
- **staged**: Multi-pass extraction (catalog → ID pass → fill pass → merge), better for complex nested templates and weaker models. See [Staged Extraction](staged-extraction.md).
- **delta**: Chunk → token-bounded batches → flat graph IR (nodes/relationships) → normalize → merge → projection; for long documents and graph-first extraction. Requires chunking. See [Delta Extraction](delta-extraction.md).
- **dense**: Two-phase skeleton-then-fill (Phase 1: identify all entities per path; Phase 2: fill each with full schema data). Requires chunking. See [Dense Extraction](dense-extraction.md).

---

### LLM Backend Features

#### ✅ Strengths

1. **Fast Processing**
   - Quick text extraction
   - Efficient chunking
   - Parallel processing

2. **Cost Effective**
   - Local models are free
   - Remote APIs are affordable
   - No GPU required (local)

3. **Flexible**
   - Multiple providers
   - Easy to switch models
   - API or local

4. **Accurate for Text**
   - Excellent for standard documents
   - Good table understanding
   - Strong reasoning

#### ❌ Limitations

1. **Text-Only**
   - No visual understanding
   - Relies on OCR quality
   - May miss layout cues

2. **Context Limits**
   - Requires chunking for large docs
   - May lose cross-page context
   - Needs merging

---

### Supported Providers

#### Local Providers

**Ollama:**
```python
config = PipelineConfig(
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.1:8b"
)
```

**vLLM:**
```python
config = PipelineConfig(
    backend="llm",
    inference="local",
    provider_override="vllm",
    model_override="ibm-granite/granite-4.0-1b"
)
```

#### Remote Providers

**Mistral AI:**
```python
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-large-latest"
)
```

**OpenAI:**
```python
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo"
)
```

**Google Gemini:**
```python
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="gemini",
    model_override="gemini-2.5-flash"
)
```

**IBM watsonx:**
```python
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="watsonx",
    model_override="ibm/granite-13b-chat-v2"
)
```

---

### LLM Backend Usage

#### Basic Extraction

```python
from docling_graph.core.extractors.backends import LlmBackend
from docling_graph.llm_clients import get_client
from docling_graph.llm_clients.config import resolve_effective_model_config

# Initialize client
effective = resolve_effective_model_config("ollama", "llama3.1:8b")
client = get_client("ollama")(model_config=effective)

# Create backend
backend = LlmBackend(llm_client=client)

# Extract from markdown
model = backend.extract_from_markdown(
    markdown="# BillingDocument\n\nInvoice Number: INV-001\nTotal: $1000",
    template=InvoiceTemplate,
    context="full document",
    is_partial=False
)

print(model)
```

#### With Consolidation

```python
# Extract from multiple chunks
models = []
for chunk in chunks:
    model = backend.extract_from_markdown(
        markdown=chunk,
        template=InvoiceTemplate,
        context=f"chunk {i}",
        is_partial=True
    )
    if model:
        models.append(model)

# Consolidate with LLM
from docling_graph.core.utils import merge_pydantic_models

programmatic_merge = merge_pydantic_models(models, InvoiceTemplate)

final_model = backend.consolidate_from_pydantic_models(
    raw_models=models,
    programmatic_model=programmatic_merge,
    template=InvoiceTemplate
)
```

---

## VLM Backend

### What is VLM Backend?

The **VLM (Vision-Language Model) backend** processes documents visually, understanding layout, images, and text together like a human would.

### Architecture

--8<-- "docs/assets/flowcharts/vlm_backend.md"

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="vlm",                      # VLM backend
    inference="local",                  # Only local supported
    model_override="numind/NuExtract-2.0-8B"
)
```

---

### VLM Backend Features

#### ✅ Strengths

1. **Visual Understanding**
   - Sees layout and structure
   - Understands images
   - Handles complex formats

2. **No Chunking Needed**
   - Processes pages directly
   - No context window limits
   - Simpler pipeline

3. **Robust to OCR Issues**
   - Doesn't rely on OCR
   - Handles poor quality
   - Better for handwriting

4. **Layout Aware**
   - Understands visual hierarchy
   - Recognizes forms
   - Detects tables visually

#### ❌ Limitations

1. **Slower**
   - More computation
   - GPU recommended
   - Longer processing time

2. **Local Only**
   - No remote API support
   - Requires local GPU
   - Higher resource usage

3. **Model Size**
   - Large models (2B-8B params)
   - More memory needed
   - Longer startup time

---

### Supported Models

**NuExtract 2.0 (Recommended):**
```python
# 2B model (faster, less accurate)
model_override="numind/NuExtract-2.0-2B"

# 8B model (slower, more accurate)
model_override="numind/NuExtract-2.0-8B"
```

---

### VLM Backend Usage

#### Basic Extraction

```python
from docling_graph.core.extractors.backends import VlmBackend

# Initialize backend
backend = VlmBackend(model_name="numind/NuExtract-2.0-8B")

# Extract from document
models = backend.extract_from_document(
    source="document.pdf",
    template=InvoiceTemplate
)

print(f"Extracted {len(models)} models")
```

#### With Pipeline

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="complex_form.pdf",
    template="templates.ApplicationForm",
    backend="vlm",
    inference="local",
    processing_mode="one-to-one"  # One model per page
)

run_pipeline(config)
```

---

## Backend Selection

### LLM Backend Criteria

- Document is text-heavy  
- Need fast processing  
- Want to use remote APIs  
- Processing many documents  
- Standard layout  
- Good OCR quality  

### VLM Backend Criteria

- Complex visual layout  
- Poor OCR quality  
- Handwritten content  
- Image-heavy documents  
- Form-based extraction  
- Have GPU available  

---

## Complete Examples

### 📍 LLM Backend (Local)

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="invoice.pdf",
    template="templates.BillingDocument",
    
    # LLM backend with Ollama
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.1:8b",
    
    # Optimized settings
    use_chunking=True,
    processing_mode="many-to-one",
    
    output_dir="outputs/llm_local"
)

run_pipeline(config)
```

### 📍 LLM Backend (Remote)

```python
from docling_graph import run_pipeline, PipelineConfig
import os

# Set API key
os.environ["MISTRAL_API_KEY"] = "your_api_key"

config = PipelineConfig(
    source="contract.pdf",
    template="templates.Contract",
    
    # LLM backend with Mistral API
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-large-latest",
    
    # High accuracy settings
    use_chunking=True,
    processing_mode="many-to-one",
    
    output_dir="outputs/llm_remote"
)

run_pipeline(config)
```

### 📍 VLM Backend

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="complex_form.pdf",
    template="templates.ApplicationForm",
    
    # VLM backend
    backend="vlm",
    inference="local",
    model_override="numind/NuExtract-2.0-8B",
    
    # VLM settings
    processing_mode="one-to-one",  # One model per page
    docling_config="vision",       # Vision pipeline
    use_chunking=False,            # VLM doesn't need chunking
    
    output_dir="outputs/vlm"
)

run_pipeline(config)
```

### 📍 Hybrid Approach

```python
from docling_graph import run_pipeline, PipelineConfig

def process_document(doc_path: str, doc_type: str):
    """Process document with appropriate backend."""
    
    if doc_type == "form":
        # Use VLM for forms
        backend = "vlm"
        inference = "local"
        processing_mode = "one-to-one"
    else:
        # Use LLM for standard docs
        backend = "llm"
        inference = "remote"
        processing_mode = "many-to-one"
    
    config = PipelineConfig(
        source=doc_path,
        template=f"templates.{doc_type.capitalize()}",
        backend=backend,
        inference=inference,
        processing_mode=processing_mode
    )
    
    run_pipeline(config)

# Process different document types
process_document("invoice.pdf", "invoice")  # LLM
process_document("form.pdf", "form")        # VLM
```

---

## Error Handling

### LLM Backend Errors

```python
from docling_graph.exceptions import ExtractionError

try:
    config = PipelineConfig(
        source="document.pdf",
        template="templates.BillingDocument",
        backend="llm",
        inference="remote"
    )
    run_pipeline(config)
    
except ExtractionError as e:
    print(f"Extraction failed: {e.message}")
    print(f"Details: {e.details}")
    
    # Fallback to local
    config = PipelineConfig(
        source="document.pdf",
        template="templates.BillingDocument",
        backend="llm",
        inference="local"
    )
    run_pipeline(config)
```

### VLM Backend Errors

```python
from docling_graph.exceptions import ExtractionError

try:
    config = PipelineConfig(
        source="document.pdf",
        template="templates.BillingDocument",
        backend="vlm"
    )
    run_pipeline(config)
    
except ExtractionError as e:
    print(f"VLM extraction failed: {e.message}")
    
    # Fallback to LLM
    config = PipelineConfig(
        source="document.pdf",
        template="templates.BillingDocument",
        backend="llm",
        inference="local"
    )
    run_pipeline(config)
```

---

## Best Practices

### 👍 Match Backend to Document Type

```python
# ✅ Good - Choose based on document
if document_is_form:
    backend = "vlm"
elif document_is_standard:
    backend = "llm"
```

### 👍 Use Local for Development

```python
# ✅ Good - Fast iteration
config = PipelineConfig(
    source="test.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="local"  # Fast for testing
)
```

### 👍 Use Remote for Production

```python
# ✅ Good - Reliable and scalable
config = PipelineConfig(
    source="production.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="remote"  # Reliable
)
```

### 👍 Cleanup Resources

```python
# ✅ Good - Always cleanup
from docling_graph.core.extractors.backends import VlmBackend

backend = VlmBackend(model_name="numind/NuExtract-2.0-8B")
try:
    models = backend.extract_from_document(source, template)
finally:
    backend.cleanup()  # Free GPU memory
```

!!! tip "Enhanced GPU Cleanup"
    VLM backend now includes enhanced GPU memory management:
    
    - **Model-to-CPU Transfer**: Moves model to CPU before deletion
    - **CUDA Cache Clearing**: Explicitly clears GPU cache
    - **Memory Tracking**: Logs memory usage before/after cleanup
    - **Multi-GPU Support**: Handles multiple GPU devices
    
    This ensures GPU memory is properly released, especially important for long-running processes.

### 👍 Use Real Tokenizers

```python
# ✅ Good - Accurate token counting
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.1:8b",
    use_chunking=True  # Uses real tokenizer with 20% safety margin
)
```

**Benefits:**
- Prevents context window overflows
- More efficient chunk packing
- Better resource utilization

---

## Troubleshooting

### 🐛 LLM Returns Empty Results

**Solution:**
```python
# Check markdown extraction
from docling_graph.core.extractors import DocumentProcessor

processor = DocumentProcessor()
document = processor.convert_to_docling_doc("document.pdf")
markdown = processor.extract_full_markdown(document)

if not markdown.strip():
    print("Markdown extraction failed")
```

### 🐛 VLM Out of Memory

**Solution:**
```python
# Use smaller model
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="vlm",
    model_override="numind/NuExtract-2.0-2B"  # Smaller model
)
```

### 🐛 Slow VLM Processing

**Solution:**
```python
# Switch to LLM for speed
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",  # Faster
    inference="local"
)
```

---

## Advanced Features

### Provider-Specific Batching

Different LLM providers have different optimal batching strategies:

```python
from docling_graph import run_pipeline, PipelineConfig

# OpenAI - Uses default 95% threshold
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo",
    use_chunking=True  # Automatically uses 95% threshold (default)
)

# Anthropic - Uses default 95% threshold
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="anthropic",
    model_override="claude-3-opus",
    use_chunking=True  # Automatically uses 95% threshold (default)
)

# Ollama - Uses default 95% threshold
config = PipelineConfig(
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.1:8b",
    use_chunking=True  # Automatically uses 95% threshold (default)
)
```

**Why Different Thresholds?**
- **OpenAI/Google**: Robust to near-limit contexts → aggressive batching
- **Anthropic**: More conservative → moderate batching
- **Ollama/Local**: Variable performance → conservative batching

---

## Next Steps

Now that you understand extraction backends:

1. **[Staged Extraction →](staged-extraction.md)** - Multi-pass extraction for complex templates
2. **[Delta Extraction →](delta-extraction.md)** - Chunk-based graph extraction for long documents
3. **[Dense Extraction →](dense-extraction.md)** - Skeleton-then-fill for granular structure and rich data
3. **[Model Merging →](model-merging.md)** - Learn how to consolidate extractions
4. **[Batch Processing →](batch-processing.md)** - Optimize chunk processing
5. **[Performance Tuning →](../../usage/advanced/performance-tuning.md)** - Advanced optimization