# Extraction Backends


## Overview

**Extraction backends** are the engines that extract structured data from documents. Docling Graph supports two types: **LLM backends** (text-based) and **VLM backends** (vision-based).

**In this guide:**
- LLM vs VLM comparison
- Backend selection criteria
- Configuration and usage
- Model capability tiers
- Performance optimization
- Error handling

!!! tip "New: Model Capability Detection"
    Docling Graph now automatically detects model capabilities and adapts prompts and consolidation strategies based on model size. See [Model Capabilities](model-capabilities.md) for details.

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
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    backend="llm",           # LLM backend
    inference="local",       # or "remote"
    provider_override="ollama",
    model_override="llama3.1:8b"
)
```

---

### Model Capability Detection

Docling Graph automatically detects model capabilities based on parameter count and adapts its behavior:

```python
from docling_graph import PipelineConfig

# Small model (1B-7B) - Uses SIMPLE tier
config = PipelineConfig(
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.2:3b"  # Automatically detected as SIMPLE
)

# Medium model (7B-13B) - Uses STANDARD tier
config = PipelineConfig(
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.1:8b"  # Automatically detected as STANDARD
)

# Large model (13B+) - Uses ADVANCED tier
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo"  # Automatically detected as ADVANCED
)
```

**Capability Tiers:**

| Tier | Model Size | Prompt Style | Consolidation |
|:-----|:-----------|:-------------|:--------------|
| **SIMPLE** | 1B-7B | Minimal instructions | Basic merge |
| **STANDARD** | 7B-13B | Balanced instructions | Standard merge |
| **ADVANCED** | 13B+ | Detailed instructions | Chain of Density |

See [Model Capabilities](model-capabilities.md) for complete details.

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
from docling_graph.llm_clients import OllamaClient

# Initialize client
client = OllamaClient(model="llama3.1:8b")

# Create backend
backend = LlmBackend(llm_client=client)

# Extract from markdown
model = backend.extract_from_markdown(
    markdown="# Invoice\n\nInvoice Number: INV-001\nTotal: $1000",
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

!!! info "Chain of Density Consolidation"
    For ADVANCED tier models (13B+), consolidation uses a multi-turn "Chain of Density" approach:
    
    1. **Initial Merge**: Create first consolidated version
    2. **Refinement**: Identify and resolve conflicts
    3. **Final Polish**: Ensure completeness and accuracy
    
    This produces higher quality results but uses more tokens. See [Model Capabilities](model-capabilities.md#chain-of-density-consolidation).

---

## VLM Backend

### What is VLM Backend?

The **VLM (Vision-Language Model) backend** processes documents visually, understanding layout, images, and text together like a human would.

### Architecture

--8<-- "docs/assets/flowcharts/vlm_backend.md"

### Configuration

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
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
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="complex_form.pdf",
    template="my_templates.ApplicationForm",
    backend="vlm",
    inference="local",
    processing_mode="one-to-one"  # One model per page
)

config.run()
```

---

## Backend Selection

### Decision Matrix

| Document Type | Recommended Backend | Reason |
|:--------------|:-------------------|:-------|
| **Standard invoices** | LLM | Fast, accurate for text |
| **Complex forms** | VLM | Better layout understanding |
| **Research papers** | LLM | Good for text-heavy docs |
| **Handwritten forms** | VLM | Handles handwriting better |
| **Scanned documents** | VLM | Robust to poor quality |
| **Multi-page contracts** | LLM | Efficient chunking |
| **Image-heavy docs** | VLM | Visual understanding |
| **Batch processing** | LLM | Faster throughput |

---

### Selection Criteria

#### Choose LLM Backend When:

✅ Document is text-heavy  
✅ Need fast processing  
✅ Want to use remote APIs  
✅ Processing many documents  
✅ Standard layout  
✅ Good OCR quality  

#### Choose VLM Backend When:

✅ Complex visual layout  
✅ Poor OCR quality  
✅ Handwritten content  
✅ Image-heavy documents  
✅ Form-based extraction  
✅ Have GPU available  

---

## Performance Comparison

### Speed Benchmark

| Backend | Document Type | Pages | Time | Throughput |
|:--------|:-------------|:------|:-----|:-----------|
| **LLM (Local)** | Invoice | 1 | 2s | 30 docs/min |
| **LLM (Remote)** | Invoice | 1 | 3s | 20 docs/min |
| **VLM (Local)** | Invoice | 1 | 8s | 7 docs/min |
| **LLM (Local)** | Contract | 10 | 15s | 4 docs/min |
| **VLM (Local)** | Contract | 10 | 60s | 1 doc/min |

**Note:** Times are approximate and vary by hardware.

---

### Accuracy Comparison

| Document Type | LLM Accuracy | VLM Accuracy | Winner |
|:--------------|:-------------|:-------------|:-------|
| **Standard invoice** | 95% | 93% | LLM |
| **Complex form** | 85% | 95% | VLM |
| **Handwritten** | 70% | 90% | VLM |
| **Research paper** | 92% | 88% | LLM |
| **Poor scan** | 75% | 88% | VLM |

---

## Complete Examples

### Example 1: LLM Backend (Local)

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="invoice.pdf",
    template="my_templates.Invoice",
    
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

config.run()
```

### Example 2: LLM Backend (Remote)

```python
from docling_graph import PipelineConfig
import os

# Set API key
os.environ["MISTRAL_API_KEY"] = "your_api_key"

config = PipelineConfig(
    source="contract.pdf",
    template="my_templates.Contract",
    
    # LLM backend with Mistral API
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-large-latest",
    
    # High accuracy settings
    use_chunking=True,
    llm_consolidation=True,
    processing_mode="many-to-one",
    
    output_dir="outputs/llm_remote"
)

config.run()
```

### Example 3: VLM Backend

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="complex_form.pdf",
    template="my_templates.ApplicationForm",
    
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

config.run()
```

### Example 4: Hybrid Approach

```python
from docling_graph import PipelineConfig

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
        template=f"my_templates.{doc_type.capitalize()}",
        backend=backend,
        inference=inference,
        processing_mode=processing_mode
    )
    
    config.run()

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
        template="my_templates.Invoice",
        backend="llm",
        inference="remote"
    )
    config.run()
    
except ExtractionError as e:
    print(f"Extraction failed: {e.message}")
    print(f"Details: {e.details}")
    
    # Fallback to local
    config = PipelineConfig(
        source="document.pdf",
        template="my_templates.Invoice",
        backend="llm",
        inference="local"
    )
    config.run()
```

### VLM Backend Errors

```python
from docling_graph.exceptions import ExtractionError

try:
    config = PipelineConfig(
        source="document.pdf",
        template="my_templates.Invoice",
        backend="vlm"
    )
    config.run()
    
except ExtractionError as e:
    print(f"VLM extraction failed: {e.message}")
    
    # Fallback to LLM
    config = PipelineConfig(
        source="document.pdf",
        template="my_templates.Invoice",
        backend="llm",
        inference="local"
    )
    config.run()
```

---

## Best Practices

### 1. Match Backend to Document Type

```python
# ✅ Good - Choose based on document
if document_is_form:
    backend = "vlm"
elif document_is_standard:
    backend = "llm"
```

### 2. Use Local for Development

```python
# ✅ Good - Fast iteration
config = PipelineConfig(
    source="test.pdf",
    template="my_templates.Invoice",
    backend="llm",
    inference="local"  # Fast for testing
)
```

### 3. Use Remote for Production

```python
# ✅ Good - Reliable and scalable
config = PipelineConfig(
    source="production.pdf",
    template="my_templates.Invoice",
    backend="llm",
    inference="remote"  # Reliable
)
```

### 4. Cleanup Resources

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

### 5. Use Real Tokenizers

```python
# ✅ Good - Accurate token counting
from docling_graph import PipelineConfig

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

### Issue: LLM Returns Empty Results

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

### Issue: VLM Out of Memory

**Solution:**
```python
# Use smaller model
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    backend="vlm",
    model_override="numind/NuExtract-2.0-2B"  # Smaller model
)
```

### Issue: Slow VLM Processing

**Solution:**
```python
# Switch to LLM for speed
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    backend="llm",  # Faster
    inference="local"
)
```

---

## Advanced Features

### Provider-Specific Batching

Different LLM providers have different optimal batching strategies:

```python
from docling_graph import PipelineConfig

# OpenAI - Aggressive batching (90% merge threshold)
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo",
    use_chunking=True  # Automatically uses 90% threshold
)

# Anthropic - Conservative batching (85% threshold)
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="anthropic",
    model_override="claude-3-opus",
    use_chunking=True  # Automatically uses 85% threshold
)

# Ollama - Very conservative (75% threshold)
config = PipelineConfig(
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.1:8b",
    use_chunking=True  # Automatically uses 75% threshold
)
```

**Why Different Thresholds?**
- **OpenAI/Google**: Robust to near-limit contexts → aggressive batching
- **Anthropic**: More conservative → moderate batching
- **Ollama/Local**: Variable performance → conservative batching

---

## Next Steps

Now that you understand extraction backends:

1. **[Model Capabilities →](model-capabilities.md)** - Learn about adaptive prompting
2. **[Model Merging →](model-merging.md)** - Learn how to consolidate extractions
3. **[Batch Processing →](batch-processing.md)** - Optimize chunk processing
4. **[Performance Tuning →](../../usage/advanced/performance-tuning.md)** - Advanced optimization

---

## Quick Reference

### LLM Backend (Local)

```python
config = PipelineConfig(
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.1:8b"
)
```

### LLM Backend (Remote)

```python
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-large-latest"
)
```

### VLM Backend

```python
config = PipelineConfig(
    backend="vlm",
    inference="local",
    model_override="numind/NuExtract-2.0-8B"
)
```