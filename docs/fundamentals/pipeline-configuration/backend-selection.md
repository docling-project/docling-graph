# Backend Selection: LLM vs VLM


## Overview

Docling Graph supports two extraction backends: **LLM (Language Model)** for text-based extraction and **VLM (Vision-Language Model)** for vision-based extraction. Choosing the right backend is crucial for extraction quality and performance.

**In this guide:**
- LLM vs VLM comparison
- When to use each backend
- Performance characteristics
- Cost considerations
- Switching between backends

---

## Backend Comparison

### Quick Comparison Table

| Aspect | LLM Backend | VLM Backend |
|:-------|:------------|:------------|
| **Input** | Markdown text | Document images |
| **Best For** | Text-heavy documents | Complex layouts, images |
| **Inference** | Local or Remote | Local only |
| **Speed** | Fast | Slower |
| **Accuracy** | High for text | Highest for complex layouts |
| **GPU Required** | Optional (remote) | Yes (local only) |
| **Cost** | Low (local) to Medium (remote) | Medium (GPU required) |
| **Setup** | Easy | Moderate |

---

## LLM Backend

### What is LLM Backend?

The LLM backend uses **language models** to extract structured data from **markdown text**. Documents are first converted to markdown using Docling, then processed by the LLM.

### Architecture

--8<-- "docs/assets/flowcharts/llm_backend.md"

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",  # LLM backend
    inference="local"  # or "remote"
)
```

### When to Use LLM

✅ **Use LLM when:**
- Documents are primarily text-based
- Layout is standard (invoices, contracts, reports)
- You need remote API support
- Cost efficiency is important
- You want fast processing
- You don't have GPU available (use remote)

❌ **Don't use LLM when:**
- Documents have complex visual layouts
- Images contain critical information
- Tables have complex structures
- Handwriting needs to be processed

### LLM Advantages

1. **Flexible Inference**
   - Local: Use your own GPU/CPU
   - Remote: Use cloud APIs (OpenAI, Mistral, Gemini)

2. **Fast Processing**
   - Quick markdown conversion
   - Efficient text processing
   - Parallel chunking support

3. **Cost Effective**
   - Local inference: Free (after GPU cost)
   - Remote inference: Pay per token
   - Generally cheaper than VLM

4. **Easy Setup**
   - No GPU required for remote
   - Simple API key configuration
   - Wide model selection

### LLM Limitations

1. **Text-Only Processing**
   - Loses visual information
   - May miss layout cues
   - Can't process images directly

2. **OCR Dependency**
   - Relies on Docling OCR quality
   - May struggle with poor scans
   - Handwriting not well supported

3. **Context Limits**
   - Large documents need chunking
   - May lose cross-page context
   - Requires consolidation for coherence

---

## VLM Backend

### What is VLM Backend?

The VLM backend uses **vision-language models** to extract structured data directly from **document images**. It processes visual information alongside text, understanding layout and structure.

### Architecture

--8<-- "docs/assets/flowcharts/vlm_backend.md"

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="vlm",  # VLM backend
    inference="local",  # VLM only supports local
    docling_config="vision"  # Optional: use vision pipeline
)
```

### When to Use VLM

✅ **Use VLM when:**
- Documents have complex visual layouts
- Images contain critical information
- Tables have intricate structures
- Forms have specific visual patterns
- Highest accuracy is required
- You have GPU available

❌ **Don't use VLM when:**
- Documents are simple text
- You need remote API support
- GPU is not available
- Processing speed is critical
- Cost is a major concern

### VLM Advantages

1. **Visual Understanding**
   - Processes layout and structure
   - Understands visual relationships
   - Handles complex tables
   - Processes embedded images

2. **Higher Accuracy**
   - Best for complex documents
   - Understands visual context
   - Fewer extraction errors
   - Better table handling

3. **No OCR Dependency**
   - Direct image processing
   - Better with poor scans
   - Handles handwriting better
   - Preserves visual information

### VLM Limitations

1. **Local Only**
   - Requires local GPU
   - No remote API support
   - Higher setup complexity
   - GPU memory requirements

2. **Slower Processing**
   - Image processing overhead
   - Larger model size
   - More GPU memory needed
   - Longer inference time

3. **Higher Cost**
   - GPU required
   - More expensive hardware
   - Higher power consumption
   - Larger storage needs

---

## Decision Matrix

### By Document Type

| Document Type | Recommended Backend | Reason |
|:--------------|:-------------------|:-------|
| **Invoices** | LLM | Standard layout, text-heavy |
| **Contracts** | LLM | Text-heavy, standard format |
| **Rheology Researchs** | LLM | Text-heavy, standard layout |
| **Forms** | VLM | Visual structure important |
| **ID Cards** | VLM | Visual layout critical |
| **Complex Tables** | VLM | Visual structure needed |
| **Handwritten** | VLM | Visual processing required |
| **Mixed Content** | VLM | Images and text combined |

### By Infrastructure

| Infrastructure | Recommended Backend | Configuration |
|:---------------|:-------------------|:--------------|
| **No GPU** | LLM Remote | `backend="llm", inference="remote"` |
| **CPU Only** | LLM Remote | `backend="llm", inference="remote"` |
| **GPU Available** | LLM or VLM Local | `backend="llm/vlm", inference="local"` |
| **Cloud/API** | LLM Remote | `backend="llm", inference="remote"` |

### By Priority

| Priority | Recommended Backend | Reason |
|:---------|:-------------------|:-------|
| **Speed** | LLM | Faster processing |
| **Accuracy** | VLM | Better visual understanding |
| **Cost** | LLM Local | No API costs |
| **Simplicity** | LLM Remote | Easy setup |
| **Offline** | LLM or VLM Local | No internet needed |

---

## Performance Comparison

### Processing Speed

```
Document: 10-page invoice PDF

LLM Local (GPU):     ~30 seconds
LLM Remote (API):    ~45 seconds
VLM Local (GPU):     ~90 seconds
```

### Accuracy Comparison

```
Document Type: Complex invoice with tables

LLM Accuracy:  92% field extraction
VLM Accuracy:  97% field extraction

Document Type: Simple text contract

LLM Accuracy:  98% field extraction
VLM Accuracy:  96% field extraction
```

### Cost Comparison

```
Processing 1000 documents:

LLM Local:     $0 (GPU amortized)
LLM Remote:    $50-200 (API costs)
VLM Local:     $0 (GPU amortized)
VLM Remote:    Not available
```

---

## Switching Between Backends

### From LLM to VLM

```python
# Original LLM config
config_llm = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="remote"
)

# Switch to VLM
config_vlm = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="vlm",  # Change backend
    inference="local",  # Must be local for VLM
    docling_config="vision"  # Optional: use vision pipeline
)
```

### From VLM to LLM

```python
# Original VLM config
config_vlm = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="vlm",
    inference="local"
)

# Switch to LLM
config_llm = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",  # Change backend
    inference="remote",  # Can now use remote
    model_override="gpt-4-turbo"  # Specify model
)
```

---

## Hybrid Approach

### Strategy 1: Document Type Based

```python
def get_config(document_path: str, document_type: str):
    """Choose backend based on document type."""
    if document_type in ["invoice", "contract", "report"]:
        # Use LLM for text-heavy documents
        return PipelineConfig(
            source=document_path,
            template="templates.BillingDocument",
            backend="llm",
            inference="remote"
        )
    else:
        # Use VLM for complex layouts
        return PipelineConfig(
            source=document_path,
            template="templates.Form",
            backend="vlm",
            inference="local"
        )
```

### Strategy 2: Fallback Pattern

```python
def extract_with_fallback(document_path: str):
    """Try LLM first, fallback to VLM if needed."""
    try:
        # Try LLM first (faster)
        config = PipelineConfig(
            source=document_path,
            template="templates.BillingDocument",
            backend="llm",
            inference="remote"
        )
        run_pipeline(config)
    except ExtractionError:
        # Fallback to VLM for better accuracy
        config = PipelineConfig(
            source=document_path,
            template="templates.BillingDocument",
            backend="vlm",
            inference="local"
        )
        run_pipeline(config)
```

---

## Backend-Specific Settings

### LLM-Specific Settings

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    
    # LLM-specific
    use_chunking=True,  # Split large documents
    max_batch_size=5  # Process multiple chunks
)
```

### VLM-Specific Settings

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="vlm",
    
    # VLM-specific
    docling_config="vision",  # Use vision pipeline
    processing_mode="one-to-one"  # Process pages individually
)
```

---

## Common Questions

**Q: Can I use VLM with remote inference?**

**A:** No, VLM currently only supports local inference. Use LLM backend for remote API support.

**Q: Which backend is more accurate?**

**A:** VLM is generally more accurate for complex layouts and visual documents. LLM is more accurate for simple text documents.

**Q: Which backend is faster?**

**A:** LLM is faster, especially with remote APIs. VLM requires more processing time due to image analysis.

**Q: Can I switch backends mid-project?**

**A:** Yes, backends are interchangeable. Just change the `backend` parameter in your config.

**Q: Do I need different templates for different backends?**

**A:** No, the same Pydantic template works with both backends.

---

## Next Steps

Now that you understand backend selection:

1. **[Model Configuration →](model-configuration.md)** - Configure models for your chosen backend
2. **[Processing Modes](processing-modes.md)** - Choose processing strategy
3. **[Configuration Examples](configuration-examples.md)** - See complete scenarios