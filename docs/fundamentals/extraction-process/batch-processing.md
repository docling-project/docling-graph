# Batch Processing


## Overview

**Batch processing** optimizes extraction by grouping multiple chunks into single LLM calls, reducing API overhead while maximizing context window utilization.

**In this guide:**
- Why batching matters
- Adaptive batching algorithm
- Context window optimization
- Performance tuning
- Best practices

---

## Why Batching Matters

### The API Call Problem

Without batching:

```python
# 10 chunks = 10 API calls
for chunk in chunks:  # 10 iterations
    model = llm.extract(chunk)  # 10 API calls
    models.append(model)

# Cost: 10 × API_COST
# Time: 10 × LATENCY
```

With batching:

```python
# 10 chunks = 3 batches = 3 API calls
batches = batcher.batch_chunks(chunks)  # Group into 3 batches
for batch in batches:  # 3 iterations
    model = llm.extract(batch.combined_text)  # 3 API calls
    models.append(model)

# Cost: 3 × API_COST (70% savings)
# Time: 3 × LATENCY (70% faster)
```

---

## ChunkBatcher

### What is ChunkBatcher?

**ChunkBatcher** intelligently groups chunks to fit within context windows, minimizing API calls while preserving semantic boundaries.

### Architecture

--8<-- "docs/assets/flowcharts/chunk_batcher.md"

---

## Basic Usage

### Initialize Batcher

```python
from docling_graph.core.extractors import ChunkBatcher

# Create batcher with context constraints
batcher = ChunkBatcher(
    context_limit=8000,          # Total context window
    system_prompt_tokens=500,    # System prompt overhead
    response_buffer_tokens=500,  # Response space
    merge_threshold=0.85         # Merge if <85% utilized
)

# Available for content: 8000 - 500 - 500 = 7000 tokens
```

### Batch Chunks

```python
# Batch your chunks
batches = batcher.batch_chunks(chunks)

print(f"Reduced {len(chunks)} chunks to {len(batches)} batches")

# Process batches
for batch in batches:
    print(f"Batch {batch.batch_id}: {batch.chunk_count} chunks")
    print(f"  Tokens: {batch.total_tokens}")
    print(f"  Utilization: {batch.total_tokens / 7000 * 100:.1f}%")
```

---

## Batching Algorithm

### Phase 1: Greedy Packing

**Strategy:** Pack chunks sequentially until context limit reached

```python
current_batch = []
current_tokens = 0

for chunk in chunks:
    chunk_tokens = estimate_tokens(chunk) + OVERHEAD
    
    if current_tokens + chunk_tokens > available_tokens:
        # Start new batch
        batches.append(current_batch)
        current_batch = [chunk]
        current_tokens = chunk_tokens
    else:
        # Add to current batch
        current_batch.append(chunk)
        current_tokens += chunk_tokens
```

**Result:** Candidate batches that fit context window

---

### Phase 2: Merge Undersized

**Strategy:** Combine small batches to improve utilization

```python
threshold = available_tokens * merge_threshold  # e.g., 85%

for batch in batches:
    if batch.total_tokens < threshold:
        # Try to merge with next batch
        if can_merge_with_next(batch):
            merge_batches(batch, next_batch)
```

**Result:** Optimized batches with high utilization

---

## ChunkBatch Object

### Structure

```python
@dataclass
class ChunkBatch:
    batch_id: int              # Batch sequence number
    chunks: List[str]          # Chunk texts
    total_tokens: int          # Estimated tokens
    chunk_indices: List[int]   # Original indices
    
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
    
    @property
    def combined_text(self) -> str:
        # Chunks with separators
        return "\n\n---CHUNK BOUNDARY---\n\n".join(chunks)
```

### Usage

```python
batch = batches[0]

print(f"Batch ID: {batch.batch_id}")
print(f"Chunks: {batch.chunk_count}")
print(f"Tokens: {batch.total_tokens}")
print(f"Indices: {batch.chunk_indices}")

# Get combined text for LLM
combined = batch.combined_text
```

---

## Token Estimation

### Estimation Methods

#### 1. Heuristic (Default)

```python
# Fast but approximate
tokens = len(text) // 4  # ~4 chars per token
```

**Pros:** Fast, no dependencies  
**Cons:** Approximate (±20% error)

#### 2. Custom Tokenizer

```python
from transformers import AutoTokenizer

# Accurate token counting
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Use with batcher
batches = batcher.batch_chunks(chunks, tokenizer_fn=count_tokens)
```

**Pros:** Accurate  
**Cons:** Slower, requires tokenizer

---

## Configuration Parameters

### Context Limit

**Definition:** Total context window size

```python
batcher = ChunkBatcher(
    context_limit=8000  # Mistral: 32K, GPT-4: 128K, Llama: 8K
)
```

**How to choose:**
- Use model's actual context limit
- Be conservative (leave buffer)
- Account for prompt overhead

---

### System Prompt Tokens

**Definition:** Tokens used by system prompt

```python
batcher = ChunkBatcher(
    context_limit=8000,
    system_prompt_tokens=500  # Typical: 300-700
)
```

**Includes:**
- Extraction instructions
- Schema definition
- Example format

---

### Response Buffer Tokens

**Definition:** Space reserved for LLM response

```python
batcher = ChunkBatcher(
    context_limit=8000,
    response_buffer_tokens=500  # Typical: 500-1000
)
```

**Depends on:**
- Schema complexity
- Expected output size
- Safety margin

---

### Merge Threshold

**Definition:** Minimum utilization before merging

```python
batcher = ChunkBatcher(
    context_limit=8000,
    merge_threshold=0.85  # Merge if <85% utilized
)
```

**Effects:**
- **Higher (0.9):** More batches, better fit
- **Lower (0.7):** Fewer batches, more merging

---

## Complete Examples

### 📍 Basic Batching

```python
from docling_graph.core.extractors import ChunkBatcher, DocumentChunker, DocumentProcessor

# Convert and chunk document
processor = DocumentProcessor(docling_config="ocr")
document = processor.convert_to_docling_doc("large_document.pdf")

chunker = DocumentChunker(provider="mistral")
chunks = chunker.chunk_document(document)

print(f"Created {len(chunks)} chunks")

# Batch chunks
batcher = ChunkBatcher(
    context_limit=32000,  # Mistral Large
    system_prompt_tokens=500,
    response_buffer_tokens=500,
    merge_threshold=0.85
)

batches = batcher.batch_chunks(chunks)

print(f"Reduced to {len(batches)} batches")
print(f"API call reduction: {(1 - len(batches)/len(chunks)) * 100:.0f}%")
```

### 📍 With Custom Tokenizer

```python
from docling_graph.core.extractors import ChunkBatcher
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Batch with accurate token counting
batcher = ChunkBatcher(context_limit=8000)
batches = batcher.batch_chunks(chunks, tokenizer_fn=count_tokens)

print(f"Accurate batching: {len(batches)} batches")
```

### 📍 Integration with Extraction

```python
from docling_graph.core.extractors.backends import LlmBackend
from docling_graph.llm_clients import MistralClient

# Initialize backend
client = MistralClient(model="mistral-large-latest")
backend = LlmBackend(llm_client=client)

# Batch chunks
batcher = ChunkBatcher(context_limit=32000)
batches = batcher.batch_chunks(chunks)

# Extract from batches
models = []
for batch in batches:
    print(f"Processing batch {batch.batch_id} ({batch.chunk_count} chunks)")
    
    model = backend.extract_from_markdown(
        markdown=batch.combined_text,
        template=InvoiceTemplate,
        context=f"batch {batch.batch_id}",
        is_partial=True
    )
    
    if model:
        models.append(model)

print(f"Extracted {len(models)} models from {len(batches)} batches")
```

### 📍 Automatic Batching in Pipeline

```python
from docling_graph import run_pipeline, PipelineConfig

# Batching happens automatically
config = PipelineConfig(
    source="large_document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="remote",
    use_chunking=True  # Enables automatic batching
)

run_pipeline(config)

# ChunkBatcher is used internally to optimize API calls
```

---

## Performance Optimization

### Batch Size vs API Calls

| Chunks | No Batching | With Batching | Reduction |
|:-------|:------------|:--------------|:----------|
| 10 | 10 calls | 3 calls | 70% |
| 20 | 20 calls | 5 calls | 75% |
| 50 | 50 calls | 12 calls | 76% |
| 100 | 100 calls | 23 calls | 77% |

### Cost Savings

```python
# Example: Mistral Large API
API_COST_PER_CALL = $0.002  # Input tokens

# Without batching: 50 chunks
cost_without = 50 * API_COST_PER_CALL = $0.10

# With batching: 12 batches
cost_with = 12 * API_COST_PER_CALL = $0.024

# Savings: 76%
```

---

## Context Utilization

### Measuring Utilization

```python
batches = batcher.batch_chunks(chunks)

for batch in batches:
    utilization = batch.total_tokens / batcher.available_tokens * 100
    print(f"Batch {batch.batch_id}: {utilization:.1f}% utilized")

# Average utilization
avg_util = sum(b.total_tokens for b in batches) / (len(batches) * batcher.available_tokens) * 100
print(f"Average utilization: {avg_util:.1f}%")
```

### Optimization Tips

```python
# ✅ Good - High utilization (85-95%)
batcher = ChunkBatcher(
    context_limit=8000,
    merge_threshold=0.85  # Merge undersized batches
)

# ❌ Bad - Low utilization (<70%)
batcher = ChunkBatcher(
    context_limit=8000,
    merge_threshold=0.5  # Too aggressive merging
)
```

---

## Best Practices

### 👍 Match Context Limit to Model

```python
# ✅ Good - Use actual model limits
if model == "mistral-large":
    context_limit = 32000
elif model == "gpt-4-turbo":
    context_limit = 128000
elif model == "llama3.1:8b":
    context_limit = 8000

batcher = ChunkBatcher(context_limit=context_limit)
```

### 👍 Leave Adequate Buffer

```python
# ✅ Good - Conservative buffers
batcher = ChunkBatcher(
    context_limit=8000,
    system_prompt_tokens=500,  # Adequate
    response_buffer_tokens=500  # Safe margin
)

# ❌ Bad - Insufficient buffer
batcher = ChunkBatcher(
    context_limit=8000,
    system_prompt_tokens=100,  # Too small
    response_buffer_tokens=100  # Risky
)
```

### 👍 Use Merge Threshold Wisely

```python
# ✅ Good - Balance efficiency and fit
batcher = ChunkBatcher(
    merge_threshold=0.85  # 85% - good balance
)

# For many small chunks
batcher = ChunkBatcher(
    merge_threshold=0.80  # More aggressive merging
)

# For few large chunks
batcher = ChunkBatcher(
    merge_threshold=0.90  # Less merging
)
```

### 👍 Monitor Batch Statistics

```python
# ✅ Good - Check batching effectiveness
batches = batcher.batch_chunks(chunks)

reduction = (1 - len(batches) / len(chunks)) * 100
print(f"API call reduction: {reduction:.0f}%")

if reduction < 50:
    print("Warning: Low batching efficiency")
```

---

## Troubleshooting

### 🐛 Too Many Batches

Batching not reducing API calls enough

**Solution:**
```python
# Increase merge threshold
batcher = ChunkBatcher(
    context_limit=8000,
    merge_threshold=0.80  # More aggressive (was 0.85)
)

# Or increase context limit if model supports it
batcher = ChunkBatcher(
    context_limit=16000  # Use larger context
)
```

### 🐛 Batches Too Large

Batches exceeding context limit

**Solution:**
```python
# Increase buffer sizes
batcher = ChunkBatcher(
    context_limit=8000,
    system_prompt_tokens=700,  # More buffer (was 500)
    response_buffer_tokens=700  # More buffer (was 500)
)
```

### 🐛 Low Utilization

Batches not filling context window

**Solution:**
```python
# Lower merge threshold
batcher = ChunkBatcher(
    context_limit=8000,
    merge_threshold=0.90  # Less merging (was 0.85)
)

# Or use smaller chunks
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=3000  # Smaller chunks (was 4096)
)
```

---

## Advanced Techniques

### Custom Batch Processing

```python
from docling_graph.core.extractors import ChunkBatcher

def process_with_retry(batches, backend, template):
    """Process batches with retry logic."""
    models = []
    
    for batch in batches:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                model = backend.extract_from_markdown(
                    markdown=batch.combined_text,
                    template=template,
                    context=f"batch {batch.batch_id}",
                    is_partial=True
                )
                
                if model:
                    models.append(model)
                    break
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed batch {batch.batch_id} after {max_retries} attempts")
                else:
                    print(f"Retry {attempt + 1} for batch {batch.batch_id}")
    
    return models
```

---

## Next Steps

Now that you understand batch processing:

1. **[Graph Management →](../graph-management/index.md)** - Work with knowledge graphs
2. **[Export Formats →](../graph-management/export-formats.md)** - Export graphs
3. **[Visualization →](../graph-management/visualization.md)** - Visualize graphs