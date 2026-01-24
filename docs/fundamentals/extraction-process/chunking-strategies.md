# Chunking Strategies


## Overview

**Chunking** is the process of intelligently splitting documents into optimal pieces for LLM processing. Docling Graph uses **structure-aware chunking** that preserves document semantics, tables, and hierarchies.

**In this guide:**
- Why chunking matters
- Structure-aware vs naive chunking
- Token management
- Provider-specific optimization
- Performance tuning

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
from docling_graph.core.extractors import DocumentChunker, DocumentProcessor

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
# Invoice INV-001
## Line Items
### Product Details

Product A costs $50.
```

**Why it matters:** LLM understands context better.

---

## Token Management

### Token Counting

```python
# Get token statistics
chunks, stats = chunker.chunk_document_with_stats(document)

print(f"Total chunks: {stats['total_chunks']}")
print(f"Average tokens: {stats['avg_tokens']:.0f}")
print(f"Max tokens: {stats['max_tokens_in_chunk']}")
print(f"Total tokens: {stats['total_tokens']}")
```

**Output:**
```
Total chunks: 5
Average tokens: 3200
Max tokens: 3950
Total tokens: 16000
```

---

### Dynamic Chunk Sizing

Chunk size adapts to:
1. **Provider context limit**
2. **Schema complexity** (larger schemas = smaller chunks)
3. **Document structure**

```python
# Automatic adjustment based on schema
from my_templates import ComplexTemplate

chunker = DocumentChunker(
    provider="mistral",
    schema_size=len(ComplexTemplate.model_json_schema())
)
```

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
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    use_chunking=True  # Automatic chunking (default)
)

config.run()
```

### Disable Chunking

```python
config = PipelineConfig(
    source="small_document.pdf",
    template="my_templates.Invoice",
    use_chunking=False  # Process full document
)
```

---

## Complete Examples

### Example 1: Basic Chunking

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

### Example 2: With Statistics

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

### Example 3: Custom Configuration

```python
from docling_graph.core.extractors import DocumentChunker, DocumentProcessor

# Custom chunker for specific use case
chunker = DocumentChunker(
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
    max_tokens=6000,  # Conservative limit
    merge_peers=True,
    schema_size=5000  # Large schema
)

processor = DocumentProcessor(docling_config="ocr")
document = processor.convert_to_docling_doc("contract.pdf")

chunks = chunker.chunk_document(document)
print(f"Created {len(chunks)} optimized chunks")
```

### Example 4: Fallback Text Chunking

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
**Recommended chunk size:** 4096 tokens  
**Tokenizer:** Mistral-7B-Instruct-v0.2

---

### OpenAI

```python
chunker = DocumentChunker(
    provider="openai",
    max_tokens=8000  # Optimized for GPT-4
)
```

**Context limit:** 128K tokens  
**Recommended chunk size:** 8000 tokens  
**Tokenizer:** tiktoken (GPT-4)

---

### Ollama (Local)

```python
chunker = DocumentChunker(
    provider="ollama",
    max_tokens=3500  # Conservative for 8K context
)
```

**Context limit:** 8K tokens (typical)  
**Recommended chunk size:** 3500 tokens  
**Tokenizer:** Model-specific

---

### IBM watsonx

```python
chunker = DocumentChunker(
    provider="watsonx",
    max_tokens=3500  # Optimized for Granite
)
```

**Context limit:** 8K tokens  
**Recommended chunk size:** 3500 tokens  
**Tokenizer:** Granite-specific

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

### Issue: Chunks Too Large

**Solution:**
```python
# Reduce max_tokens
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=3000  # Smaller chunks
)
```

### Issue: Too Many Chunks

**Solution:**
```python
# Increase max_tokens and enable merge_peers
chunker = DocumentChunker(
    provider="openai",
    max_tokens=8000,  # Larger chunks
    merge_peers=True  # Combine sections
)
```

### Issue: Tables Split Across Chunks

**Solution:**
```python
# This shouldn't happen with HybridChunker
# If it does, increase max_tokens
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=6000  # Larger to fit tables
)
```

### Issue: Out of Memory

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

### 1. Match Provider

```python
# ✅ Good - Match chunker to LLM provider
if using_mistral:
    chunker = DocumentChunker(provider="mistral")
elif using_openai:
    chunker = DocumentChunker(provider="openai")
```

### 2. Enable Merge Peers

```python
# ✅ Good - Better context
chunker = DocumentChunker(
    provider="mistral",
    merge_peers=True  # Recommended
)
```

### 3. Monitor Statistics

```python
# ✅ Good - Check chunk distribution
chunks, stats = chunker.chunk_document_with_stats(document)

if stats['max_tokens_in_chunk'] > max_tokens * 0.95:
    print("Warning: Chunks near limit")
```

### 4. Adjust for Schema Complexity

```python
# ✅ Good - Account for schema size
schema_size = len(template.model_json_schema())

chunker = DocumentChunker(
    provider="mistral",
    schema_size=schema_size  # Dynamic adjustment
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

## Next Steps

Now that you understand chunking:

1. **[Extraction Backends →](extraction-backends.md)** - Learn about LLM and VLM backends
2. **[Batch Processing →](batch-processing.md)** - Optimize chunk processing
3. **[Model Merging →](model-merging.md)** - Consolidate chunk extractions

---

## Quick Reference

### Basic Chunking

```python
from docling_graph.core.extractors import DocumentChunker

chunker = DocumentChunker(provider="mistral")
chunks = chunker.chunk_document(document)
```

### With Statistics

```python
chunks, stats = chunker.chunk_document_with_stats(document)
print(f"Created {stats['total_chunks']} chunks")
```

### Custom Configuration

```python
chunker = DocumentChunker(
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
    max_tokens=4096,
    merge_peers=True,
    schema_size=5000
)
```

### Fallback Text Chunking

```python
chunks = chunker.chunk_text_fallback(raw_text)
```