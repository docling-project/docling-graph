# Chunking Strategies


## Overview

**Chunking** is the process of intelligently splitting documents into optimal pieces for LLM processing. Docling Graph uses **structure-aware chunking** that preserves document semantics, tables, and hierarchies.

**In this guide:**
- Why chunking matters
- Structure-aware vs naive chunking
- Real tokenizer integration
- Token management with safety margins
- Schema-aware chunking
- Provider-specific optimization
- Performance tuning

!!! tip "New: Real Tokenizer Integration"
    Docling Graph now uses real tokenizers for accurate token counting instead of character-based heuristics. This prevents context window overflows and enables more efficient chunk packing with a 20% safety margin.

---

## Why Chunking Matters

### The Context Window Problem

LLMs have limited context windows:

| Provider | Model | Context Limit |
|:---------|:------|:--------------|
| **OpenAI** | GPT-4 Turbo | 128K tokens |
| **Mistral** | Mistral Large | 32K tokens |
| **Ollama** | Llama 3.1 8B | 8K tokens |
| **IBM** | Granite 4.0 | 8K tokens |

**Problem:** Most documents exceed these limits.

**Solution:** Intelligent chunking.

---

## Chunking Approaches

### ❌ Naive Chunking

```python
# ❌ Bad - Breaks tables and structure
def naive_chunk(text, max_chars=1000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
```

**Problems:**
- Breaks tables mid-row
- Splits lists
- Ignores semantic boundaries
- Loses context

---

### ✅ Structure-Aware Chunking

```python
# ✅ Good - Preserves structure
from docling_graph.core.extractors import DocumentChunker

chunker = DocumentChunker(
    provider="mistral",
    max_tokens=4096
)

chunks = chunker.chunk_document(document)
```

**Benefits:**
- Preserves tables
- Keeps lists intact
- Respects sections
- Maintains context

---

## DocumentChunker

### Basic Usage

```python
import json

from docling_graph.core.extractors import DocumentChunker, DocumentProcessor
from my_templates import ContractTemplate

# Initialize processor
processor = DocumentProcessor(docling_config="ocr")
document = processor.convert_to_docling_doc("document.pdf")

# Initialize chunker
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=4096
)

# Chunk document
chunks = chunker.chunk_document(document)

print(f"Created {len(chunks)} chunks")
```

---

## Configuration Options

### By Provider

```python
# Automatic configuration for provider
chunker = DocumentChunker(
    provider="mistral",  # Auto-configures for Mistral
    merge_peers=True
)
```

**Supported providers:**
- `mistral` - Mistral AI models
- `openai` - OpenAI models
- `ollama` - Ollama local models
- `watsonx` - IBM watsonx models
- `google` - Google Gemini models

---

### Custom Tokenizer

```python
# Use specific tokenizer
chunker = DocumentChunker(
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
    max_tokens=4096,
    merge_peers=True
)
```

---

### Custom Max Tokens

```python
# Override max tokens
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=8000,  # Custom limit
    merge_peers=True
)
```

---

## Structure Preservation

### What Gets Preserved?

The HybridChunker preserves:

1. **Tables** - Never split across chunks
2. **Lists** - Kept intact
3. **Sections** - With headers
4. **Hierarchies** - Parent-child relationships
5. **Semantic boundaries** - Natural breaks

### Example: Table Preservation

**Input document:**
```markdown
# Sales Report

| Product | Q1 | Q2 | Q3 | Q4 |
|---------|----|----|----|----|
| A       | 10 | 15 | 20 | 25 |
| B       | 5  | 10 | 15 | 20 |
```

**Chunking result:**
```python
# ✅ Table stays together in one chunk
chunks = [
    "# Sales Report\n\n| Product | Q1 | Q2 | Q3 | Q4 |\n..."
]
```

---

## Context Enrichment

### What is Context Enrichment?

Chunks are **contextualized** with metadata:
- Section headers
- Parent sections
- Document structure
- Page numbers

### Example

**Original text:**
```
Product A costs $50.
```

**Contextualized chunk:**
```
# BillingDocument INV-001
## Line Items
### Product Details

Product A costs $50.
```

**Why it matters:** LLM understands context better.

---

## Real Tokenizer Integration

### Accurate Token Counting

Docling Graph uses **real tokenizers** instead of character-based heuristics:

```python
from docling_graph.core.extractors import DocumentChunker

# ✅ Good - Real tokenizer (accurate)
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=4096
)
# Uses Mistral's actual tokenizer for precise counting

# ❌ Old approach - Character heuristic (inaccurate)
# estimated_tokens = len(text) / 4  # Rough approximation
```

**Benefits:**

| Feature | Character Heuristic | Real Tokenizer |
|:--------|:-------------------|:---------------|
| **Accuracy** | ~70% | 95%+ |
| **Context Overflows** | Occasional | Rare |
| **Chunk Efficiency** | 60-70% | 80-90% |
| **Provider-Specific** | No | Yes |

### How It Works

```python
# Behind the scenes:
# 1. Load provider-specific tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# 2. Count tokens accurately
tokens = tokenizer.encode(text)
token_count = len(tokens)

# 3. Apply safety margin (20%)
safe_limit = int(max_tokens * 0.8)

# 4. Chunk based on actual token count
if token_count > safe_limit:
    # Split into multiple chunks
```

---

## Safety Margins

### Why Safety Margins?

Even with real tokenizers, we apply a **20% safety margin**:

```python
# Example: Model with 8192 token context
max_tokens = 8192

# Effective limit with 20% safety margin
safe_limit = int(max_tokens * 0.8)  # 6553 tokens

# Why?
# - Schema takes tokens (~500-2000)
# - System prompts take tokens (~200-500)
# - Response buffer needed (~500-1000)
# - Edge cases and variations
```

**Safety Margin Breakdown:**

| Component | Token Usage | Example (8K context) |
|:----------|:------------|:---------------------|
| **Document chunk** | 80% | 6553 tokens |
| **Schema** | 10-15% | 819-1228 tokens |
| **System prompt** | 3-5% | 245-409 tokens |
| **Response buffer** | 5-10% | 409-819 tokens |

### Configuring Safety Margins

```python
# Default: 20% safety margin (recommended)
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=4096  # Effective: ~3276 tokens per chunk
)

# For aggressive batching (not recommended):
# Modify ChunkBatcher.batch_chunks merge_threshold
# But this increases risk of context overflows
```

---

## Token Management

### Token Counting with Statistics

```python
# Get detailed token statistics
chunks, stats = chunker.chunk_document_with_stats(document)

print(f"Total chunks: {stats['total_chunks']}")
print(f"Average tokens: {stats['avg_tokens']:.0f}")
print(f"Max tokens: {stats['max_tokens_in_chunk']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Safety margin: {(1 - stats['max_tokens_in_chunk']/max_tokens)*100:.1f}%")
```

**Output:**
```
Total chunks: 5
Average tokens: 3200
Max tokens: 3950
Total tokens: 16000
Safety margin: 3.5%
```

!!! warning "Monitor Safety Margins"
    If `max_tokens_in_chunk` is > 95% of `max_tokens`, consider:
    
    - Reducing `max_tokens` parameter
    - Increasing schema efficiency
    - Splitting large tables

---

## Schema-Aware Chunking

### Dynamic Adjustment Based on Schema

Chunk size automatically adjusts based on schema complexity:

```python
import json

from my_templates import ComplexTemplate

# Schema-aware chunking
chunker = DocumentChunker(
    provider="mistral",
    schema_json=json.dumps(ComplexTemplate.model_json_schema())
)

# Behind the scenes:
# 1. Build prompt skeleton with schema JSON and empty content
# 2. Count exact tokens for system + user prompt
# 3. max_tokens = context_limit - static_overhead - reserved_output - safety_margin
# 4. Chunk with the adjusted limit
chunks = chunker.chunk_document(document)
```

**Schema Size Impact:**

Chunk size is computed from exact prompt token counts, so larger schemas
reduce available content tokens deterministically without heuristic ratios.

### Update Schema Configuration

```python
# Update schema JSON after initialization
chunker = DocumentChunker(provider="mistral")

# Later, update for different template
from my_templates import LargeTemplate

import json

chunker.update_schema_config(
    schema_json=json.dumps(LargeTemplate.model_json_schema())
)

# Chunker now uses adjusted limits
chunks = chunker.chunk_document(document)
```

!!! tip "Schema Optimization"
    To maximize chunk size:
    
    - Keep schemas focused and minimal
    - Use field descriptions sparingly
    - Avoid deeply nested structures
    - Consider splitting large schemas

---

## Merge Peers Option

### What is Merge Peers?

**Merge peers** combines sibling sections when they fit together:

```python
# Enable merge peers (default)
chunker = DocumentChunker(
    provider="mistral",
    merge_peers=True  # Combine related sections
)
```

### Example

**Without merge_peers:**
```python
chunks = [
    "## Section 1\nContent 1",
    "## Section 2\nContent 2",
    "## Section 3\nContent 3"
]
```

**With merge_peers:**
```python
chunks = [
    "## Section 1\nContent 1\n\n## Section 2\nContent 2",
    "## Section 3\nContent 3"
]
```

**Benefit:** Fewer chunks, better context.

---

## Integration with Pipeline

### Automatic Chunking

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    use_chunking=True  # Automatic chunking (default)
)

run_pipeline(config)
```

### Disable Chunking

```python
config = PipelineConfig(
    source="small_document.pdf",
    template="templates.BillingDocument",
    use_chunking=False  # Process full document
)
```

---

## Complete Examples

### 📍 Basic Chunking

```python
from docling_graph.core.extractors import DocumentChunker, DocumentProcessor

# Convert document
processor = DocumentProcessor(docling_config="ocr")
document = processor.convert_to_docling_doc("document.pdf")

# Chunk with Mistral settings
chunker = DocumentChunker(provider="mistral")
chunks = chunker.chunk_document(document)

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {len(chunk)} characters")
```

### 📍 With Statistics

```python
from docling_graph.core.extractors import DocumentChunker, DocumentProcessor

# Convert and chunk
processor = DocumentProcessor(docling_config="ocr")
document = processor.convert_to_docling_doc("large_document.pdf")

# Get detailed statistics
chunker = DocumentChunker(provider="openai", max_tokens=8000)
chunks, stats = chunker.chunk_document_with_stats(document)

print(f"Chunking Statistics:")
print(f"  Total chunks: {stats['total_chunks']}")
print(f"  Average tokens: {stats['avg_tokens']:.0f}")
print(f"  Max tokens: {stats['max_tokens_in_chunk']}")
print(f"  Total tokens: {stats['total_tokens']}")

# Check if any chunk exceeds limit
if stats['max_tokens_in_chunk'] > 8000:
    print("Warning: Some chunks exceed token limit!")
```

### 📍 Custom Configuration

```python
from docling_graph.core.extractors import DocumentChunker, DocumentProcessor

# Custom chunker for specific use case
chunker = DocumentChunker(
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
    max_tokens=6000,  # Conservative limit
    merge_peers=True,
    schema_json=json.dumps(ContractTemplate.model_json_schema()),
)

processor = DocumentProcessor(docling_config="ocr")
document = processor.convert_to_docling_doc("contract.pdf")

chunks = chunker.chunk_document(document)
print(f"Created {len(chunks)} optimized chunks")
```

### 📍 Fallback Text Chunking

```python
from docling_graph.core.extractors import DocumentChunker

# For raw text (when DoclingDocument unavailable)
chunker = DocumentChunker(provider="mistral")

raw_text = """
Long text content that needs to be chunked...
"""

chunks = chunker.chunk_text_fallback(raw_text)
print(f"Created {len(chunks)} text chunks")
```

---

## Provider-Specific Optimization

### Mistral AI

```python
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=4096  # Optimized for Mistral Large
)
```

**Context limit:** 32K tokens
**Recommended chunk size:** 4096 tokens (with 20% safety margin)
**Effective chunk size:** ~3276 tokens
**Tokenizer:** Mistral-7B-Instruct-v0.2 (real tokenizer)

---

### OpenAI

```python
chunker = DocumentChunker(
    provider="openai",
    max_tokens=8000  # Optimized for GPT-4
)
```

**Context limit:** 128K tokens
**Recommended chunk size:** 8000 tokens (with 20% safety margin)
**Effective chunk size:** ~6400 tokens
**Tokenizer:** tiktoken (GPT-4) (real tokenizer)

---

### Ollama (Local)

```python
chunker = DocumentChunker(
    provider="ollama",
    max_tokens=3500  # Conservative for 8K context
)
```

**Context limit:** 8K tokens (typical)
**Recommended chunk size:** 3500 tokens (with 20% safety margin)
**Effective chunk size:** ~2800 tokens
**Tokenizer:** Model-specific (real tokenizer when available)

!!! note "Ollama Tokenizer Fallback"
    If model-specific tokenizer is unavailable, falls back to character heuristic with extra safety margin (75% instead of 80%).

---

### IBM watsonx

```python
chunker = DocumentChunker(
    provider="watsonx",
    max_tokens=3500  # Optimized for Granite
)
```

**Context limit:** 8K tokens
**Recommended chunk size:** 3500 tokens (with 20% safety margin)
**Effective chunk size:** ~2800 tokens
**Tokenizer:** Granite-specific (real tokenizer)

---

### Google Gemini

```python
chunker = DocumentChunker(
    provider="google",
    max_tokens=6000  # Optimized for Gemini
)
```

**Context limit:** 32K-128K tokens (model-dependent)
**Recommended chunk size:** 6000 tokens (with 20% safety margin)
**Effective chunk size:** ~4800 tokens
**Tokenizer:** Gemini-specific (real tokenizer)

---

## Performance Tuning

### Chunk Size vs Accuracy

| Chunk Size | Accuracy | Speed | Memory |
|:-----------|:---------|:------|:-------|
| **Small (2K)** | Lower | Fast | Low |
| **Medium (4K)** | Good | Medium | Medium |
| **Large (8K)** | Best | Slow | High |

### Recommendations

```python
# ✅ Good - Balance accuracy and speed
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=4096  # Sweet spot
)
```

---

## Troubleshooting

### 🐛 Chunks Too Large

**Solution:**
```python
# Reduce max_tokens
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=3000  # Smaller chunks
)
```

### 🐛 Too Many Chunks

**Solution:**
```python
# Increase max_tokens and enable merge_peers
chunker = DocumentChunker(
    provider="openai",
    max_tokens=8000,  # Larger chunks
    merge_peers=True  # Combine sections
)
```

### 🐛 Tables Split Across Chunks

**Solution:**
```python
# This shouldn't happen with HybridChunker
# If it does, increase max_tokens
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=6000  # Larger to fit tables
)
```

### 🐛 Out of Memory

**Solution:**
```python
# Use smaller chunks
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=2000,  # Smaller chunks
    merge_peers=False  # Don't combine
)
```

---

## Best Practices

### 👍 Match Provider

```python
# ✅ Good - Match chunker to LLM provider
if using_mistral:
    chunker = DocumentChunker(provider="mistral")
elif using_openai:
    chunker = DocumentChunker(provider="openai")
```

### 👍 Enable Merge Peers

```python
# ✅ Good - Better context
chunker = DocumentChunker(
    provider="mistral",
    merge_peers=True  # Recommended
)
```

### 👍 Monitor Statistics

```python
# ✅ Good - Check chunk distribution
chunks, stats = chunker.chunk_document_with_stats(document)

if stats['max_tokens_in_chunk'] > max_tokens * 0.95:
    print("Warning: Chunks near limit")
```

### 👍 Adjust for Schema Complexity

```python
# ✅ Good - Account for schema JSON
import json

schema_json = json.dumps(template.model_json_schema())

chunker = DocumentChunker(
    provider="mistral",
    schema_json=schema_json  # Dynamic adjustment
)
```

---

## Advanced Features

### Custom Tokenizer

```python
from transformers import AutoTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

# Load custom tokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("custom/model")
custom_tokenizer = HuggingFaceTokenizer(
    tokenizer=hf_tokenizer,
    max_tokens=4096
)

# Use with HybridChunker
from docling.chunking import HybridChunker

chunker = HybridChunker(
    tokenizer=custom_tokenizer,
    merge_peers=True
)
```

### Recommended Chunk Size Calculation

```python
from docling_graph.core.extractors import DocumentChunker

# Calculate recommended size
recommended = DocumentChunker.calculate_recommended_max_tokens(
    context_limit=32000,  # Mistral Large
    system_prompt_tokens=500,
    response_buffer_tokens=500
)

print(f"Recommended max_tokens: {recommended}")
# Output: Recommended max_tokens: 24800
```

---

## Performance Impact

### Real Tokenizer vs Heuristic

**Benchmark Results** (100-page document):

| Method | Chunks Created | Context Overflows | Processing Time | API Calls |
|:-------|:---------------|:------------------|:----------------|:----------|
| **Character Heuristic** | 45 | 3 (6.7%) | 180s | 48 (3 retries) |
| **Real Tokenizer** | 38 | 0 (0%) | 152s | 38 (no retries) |

**Improvements:**

- ✅ 15% fewer chunks (better packing)
- ✅ Zero context overflows (vs 6.7%)
- ✅ 15% faster processing (no retries)
- ✅ 21% fewer API calls (no retries)

### Safety Margin Impact

| Safety Margin | Chunk Efficiency | Context Overflows | Recommended For |
|:--------------|:-----------------|:------------------|:----------------|
| **10%** | 90% | Occasional | Aggressive batching |
| **20%** (default) | 80% | Rare | General use |
| **30%** | 70% | Very rare | Complex schemas |

---

## Next Steps

Now that you understand chunking:

1. **[Staged Extraction →](staged-extraction.md)** - Multi-pass extraction for complex templates
2. **[Extraction Backends →](extraction-backends.md)** - Learn about LLM and VLM backends
3. **[Batch Processing →](batch-processing.md)** - Optimize chunk processing
4. **[Model Merging →](model-merging.md)** - Consolidate chunk extractions
5. **[Performance Tuning →](../../usage/advanced/performance-tuning.md)** - Advanced optimization