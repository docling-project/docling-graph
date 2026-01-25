# Performance Tuning


## Overview

Optimize docling-graph pipeline performance for speed, memory efficiency, and resource utilization.

**Prerequisites:**
- Understanding of [Pipeline Configuration](../../fundamentals/pipeline-configuration/index.md)
- Familiarity with [Extraction Process](../../fundamentals/extraction-process/index.md)
- Basic knowledge of system resources

!!! tip "New Performance Features"
    Recent improvements include:
    
    - **Provider-Specific Batching**: Optimized merge thresholds per provider
    - **Real Tokenizer Integration**: Accurate token counting with safety margins
    - **Enhanced GPU Cleanup**: Better memory management for VLM backends
    - **Model Capability Detection**: Automatic prompt adaptation based on model size

---

## Performance Factors

### Key Metrics

1. **Throughput**: Documents processed per hour
2. **Latency**: Time per document
3. **Memory Usage**: RAM and VRAM consumption
4. **Cost**: API costs for remote inference

---

## Model Selection

### Local vs Remote

```python
# ✅ Fast - Local inference (no network latency)
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    backend="llm",
    inference="local",  # Faster for small documents
    model_override="ibm-granite/granite-4.0-1b"  # Smaller = faster
)

# ⚠️ Slower - Remote inference (network overhead)
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    backend="llm",
    inference="remote",  # Better for complex documents
    model_override="gpt-4-turbo"  # More accurate but slower
)
```

### Model Size Trade-offs

| Model Size | Speed | Accuracy | Memory | Use Case |
|------------|-------|----------|--------|----------|
| 1B params | ⚡ Very Fast | 🟡 Moderate Accuracy | 2-4 GB | Simple forms, fast processing |
| 7-8B params | ⚡ Fast | 🟢 Acceptable Accuracy | 8-16 GB | General documents |
| 13B+ params | 🐢 Slow | 💎 High Accuracy | 16-32 GB | Complex documents |

**Recommendation:**

```python
# Simple documents (forms, invoices)
model_override="ibm-granite/granite-4.0-1b"  # Fast

# General documents
model_override="llama-3.1-8b"  # Balanced

# Complex documents (research papers, legal)
model_override="mistral-small-latest"  # Accurate (remote)
```

---

## Model Capability Tiers

Docling Graph automatically detects model capabilities and optimizes performance:

```python
from docling_graph import run_pipeline, PipelineConfig

# Small model (1B-7B) - SIMPLE tier
# - Minimal prompts (fewer tokens)
# - Basic consolidation (faster)
config = PipelineConfig(
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.2:3b"  # Optimized for speed
)

# Medium model (7B-13B) - STANDARD tier
# - Balanced prompts
# - Standard consolidation
config = PipelineConfig(
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.1:8b"  # Balanced performance
)

# Large model (13B+) - ADVANCED tier
# - Detailed prompts (more tokens but better quality)
# - Chain of Density consolidation (multi-turn)
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo"  # Optimized for quality
)
```

**Performance Impact:**

| Tier | Prompt Tokens | Consolidation | Speed | Quality |
|:-----|:--------------|:--------------|:------|:--------|
| **SIMPLE** | ~200-300 | Single-turn | ⚡ Fast | 🟡 Good |
| **STANDARD** | ~400-500 | Single-turn | ⚡ Fast | 🟢 Better |
| **ADVANCED** | ~600-800 | Multi-turn | 🐢 Slower | 💎 Best |

See [Model Capabilities](../../fundamentals/extraction-process/model-capabilities.md) for details.

---

## Batch Processing

### Provider-Specific Batching

Different providers have different optimal batching strategies:

```python
from docling_graph import run_pipeline, PipelineConfig

# OpenAI - Aggressive batching (90% merge threshold)
# Best for: High-volume processing with reliable API
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo",
    use_chunking=True  # Automatically uses threshold
)

# Ollama/Local - Conservative batching (75% threshold)
# Best for: Variable performance local models
config = PipelineConfig(
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.1:8b",
    use_chunking=True  # Automatically uses threshold
)
```

**Why Different Thresholds?**

| Provider | Threshold | Reason |
|:---------|:----------|:-------|
| **OpenAI** | 90% | Robust to near-limit contexts |
| **Google** | 88% | Good context handling |
| **Anthropic** | 85% | More conservative approach |
| **Ollama/Local** | 75% | Variable performance, safer margin |

**Performance Impact:**
- Higher threshold = Fewer API calls = Faster processing
- Lower threshold = More safety margin = Better reliability

### Optimal Batch Sizes

```python
# ✅ Good - Appropriate batch size
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    use_chunking=True,
    max_batch_size=5  # Process 5 chunks at a time
)

# ❌ Avoid - Too large (memory issues)
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    use_chunking=True,
    max_batch_size=50  # May run out of memory
)

# ❌ Avoid - Too small (slow)
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    use_chunking=True,
    max_batch_size=1  # Underutilizes resources
)
```

### Batch Size Guidelines

**For Local Inference:**

```python
# GPU with 8GB VRAM
max_batch_size = 3

# GPU with 16GB VRAM
max_batch_size = 5

# GPU with 24GB+ VRAM
max_batch_size = 10

# CPU only
max_batch_size = 1  # Parallel processing not beneficial
```

**For Remote APIs:**

```python
# Most APIs handle batching internally
max_batch_size = 1  # Send one request at a time

# For APIs with batch endpoints
max_batch_size = 10  # Check API documentation
```

---

## Memory Management

### Monitor Memory Usage

```python
"""Monitor memory during processing."""

import psutil
import GPUtil

def log_memory_usage():
    """Log current memory usage."""
    # RAM
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.percent}% ({ram.used / 1e9:.1f}GB / {ram.total / 1e9:.1f}GB)")
    
    # GPU
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.memoryUtil*100:.1f}% ({gpu.memoryUsed}MB / {gpu.memoryTotal}MB)")
    except:
        print("No GPU detected")

# Use during pipeline
from docling_graph import run_pipeline, PipelineConfig

log_memory_usage()  # Before
config = PipelineConfig(...)
run_pipeline(config)
log_memory_usage()  # After
```

### Reduce Memory Usage

```python
# ✅ Good - Process in smaller chunks
config = PipelineConfig(
    source="large_document.pdf",
    template="templates.MyTemplate",
    use_chunking=True,  # Enable chunking
    processing_mode="one-to-one"  # Process page by page
)

# ❌ Avoid - Load entire document
config = PipelineConfig(
    source="large_document.pdf",
    template="templates.MyTemplate",
    use_chunking=False,  # Load all at once
    processing_mode="many-to-one"
)
```

### Clean Up Resources

```python
"""Properly clean up after processing."""

from docling_graph import run_pipeline, PipelineConfig
import gc
import torch

def process_with_cleanup(source: str):
    """Process document with proper cleanup."""
    config = PipelineConfig(
        source=source,
        template="templates.MyTemplate"
    )
    
    try:
        run_pipeline(config)
    finally:
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if using PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Process multiple documents
for doc in documents:
    process_with_cleanup(doc)
    # Memory is freed between documents
```

!!! tip "Enhanced GPU Cleanup for VLM"
    VLM backends now include enhanced GPU memory management:
    
    ```python
    from docling_graph.core.extractors.backends import VlmBackend
    
    backend = VlmBackend(model_name="numind/NuExtract-2.0-8B")
    try:
        models = backend.extract_from_document(source, template)
    finally:
        backend.cleanup()  # Enhanced cleanup:
        # 1. Moves model to CPU before deletion
        # 2. Explicitly clears CUDA cache
        # 3. Logs memory usage before/after
        # 4. Handles multiple GPU devices
    ```
    
    **Memory Savings**: Up to 8GB VRAM freed per cleanup cycle

---

## GPU Utilization

### Enable GPU Acceleration

```bash
# Install with GPU support
uv sync --extra local

# Verify GPU is available
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Optimize GPU Usage

```python
# ✅ Good - Use GPU for local inference
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    backend="llm",
    inference="local",  # Will use GPU if available
    provider_override="vllm"  # Optimized for GPU
)

# Monitor GPU utilization
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

### Multi-GPU Support

```python
"""Use multiple GPUs for parallel processing."""

import os
from pathlib import Path
from docling_graph import run_pipeline, PipelineConfig

def process_on_gpu(source: str, gpu_id: int):
    """Process document on specific GPU."""
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    config = PipelineConfig(
        source=source,
        template="templates.MyTemplate",
        output_dir=f"outputs/gpu_{gpu_id}"
    )
    run_pipeline(config)

# Process documents in parallel on different GPUs
from concurrent.futures import ThreadPoolExecutor

documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf", "doc4.pdf"]
with ThreadPoolExecutor(max_workers=2) as executor:
    # GPU 0 processes doc1 and doc3
    # GPU 1 processes doc2 and doc4
    futures = [
        executor.submit(process_on_gpu, doc, i % 2)
        for i, doc in enumerate(documents)
    ]
    
    for future in futures:
        future.result()
```

---

## Real Tokenizer Integration

### Accurate Token Counting

Docling Graph now uses real tokenizers for accurate token counting:

```python
from docling_graph import run_pipeline, PipelineConfig

# ✅ Good - Real tokenizer with safety margin
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.1:8b",
    use_chunking=True  # Uses real tokenizer + 20% safety margin
)
```

**Benefits:**

1. **Prevents Context Overflows**: Accurate token counting prevents exceeding context limits
2. **Better Chunk Packing**: More efficient use of context window
3. **Reduced API Calls**: Optimal chunk sizes reduce number of requests
4. **Cost Savings**: Fewer API calls = lower costs

**Performance Comparison:**

| Method | Accuracy | Context Overflows | Chunk Efficiency |
|:-------|:---------|:------------------|:-----------------|
| **Character Heuristic** | ~70% | Occasional | 60-70% |
| **Real Tokenizer** | 95%+ | Rare | 80-90% |

### Safety Margins

```python
# Default: 20% safety margin
# If model has 8192 token context:
# - Effective limit: 6553 tokens (80% of 8192)
# - Prevents edge cases and ensures reliability

# For aggressive batching (not recommended):
# Modify ChunkBatcher.batch_chunks merge_threshold
# But this may cause context overflows
```

---

## Chunking Strategies

### Disable Chunking for Small Documents

```python
# ✅ Good - No chunking for small docs (< 5 pages)
config = PipelineConfig(
    source="short_document.pdf",
    template="templates.MyTemplate",
    use_chunking=False  # Faster for small docs
)

# ✅ Good - Enable chunking for large docs (> 5 pages)
config = PipelineConfig(
    source="long_document.pdf",
    template="templates.MyTemplate",
    use_chunking=True  # Necessary for large docs
)
```

### Optimize Chunk Size

```python
"""Configure chunking for optimal performance."""

from docling_graph import run_pipeline, PipelineConfig

# For fast processing (may sacrifice accuracy)
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    use_chunking=True,
    # Larger chunks = fewer API calls but more memory
)

# For accurate processing (slower)
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    use_chunking=True,
    # Smaller chunks = more API calls but better accuracy
)
```

---

## Consolidation Strategies

### Programmatic vs LLM Consolidation

```python
# ✅ Fast - Programmatic merge (no LLM call)
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    processing_mode="many-to-one",
    llm_consolidation=False  # Fast merge
)

# ⚠️ Slow - LLM consolidation (extra API call)
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    processing_mode="many-to-one",
    llm_consolidation=True  # More accurate but slower
)
```

**When to Use Each:**

| Strategy | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| Programmatic | ⚡ Very Fast | 🟡 Moderate Accuracy | Simple merging, lists |
| LLM (Standard) | 🐢 Slow | 🟢 High Accuracy | Complex conflicts |
| LLM (Chain of Density) | 🐌 Very Slow | 💎 Highest Accuracy | Critical documents |

### Chain of Density Consolidation

For ADVANCED tier models (13B+), consolidation uses a multi-turn approach:

```python
# Automatically enabled for large models
config = PipelineConfig(
    source="complex_document.pdf",
    template="templates.Contract",
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo",  # ADVANCED tier
    processing_mode="many-to-one",
    llm_consolidation=True  # Uses Chain of Density
)
```

**Process:**
1. **Initial Merge** (Turn 1): Create first consolidated version
2. **Refinement** (Turn 2): Identify and resolve conflicts
3. **Final Polish** (Turn 3): Ensure completeness and accuracy

**Performance Impact:**
- **Token Usage**: 3x more tokens than standard consolidation
- **Time**: 3x longer processing time
- **Quality**: Significantly better for complex documents
- **Cost**: 3x API costs

**When to Use:**

- ✅ Critical documents requiring highest accuracy
- ✅ Complex contracts or legal documents
- ✅ Documents with many conflicts
- ❌ Simple forms or invoices (overkill)
- ❌ High-volume batch processing (too slow)


---

## Profiling

### Profile Pipeline Execution

```python
"""Profile pipeline to identify bottlenecks."""

import time
from docling_graph import run_pipeline, PipelineConfig

def profile_pipeline(source: str):
    """Profile pipeline execution."""
    stages = {}
    
    # Overall timing
    start = time.time()
    
    # Would need to instrument pipeline stages
    # This is a simplified example
    
    config = PipelineConfig(
        source=source,
        template="templates.MyTemplate"
    )
    
    run_pipeline(config)
    
    total_time = time.time() - start
    
    print(f"\nProfiling Results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {1/total_time:.2f} docs/sec")

# Profile
profile_pipeline("document.pdf")
```

### Use Python Profiler

```bash
# Profile with cProfile
uv run python -m cProfile -o profile.stats my_script.py

# Analyze results
uv run python -m pstats profile.stats
# Then: sort cumtime, stats 20
```

---

## Optimization Checklist

### Before Processing

- [ ] Choose appropriate model size for task
- [ ] Enable GPU if available
- [ ] Set optimal batch size for hardware
- [ ] Disable chunking for small documents
- [ ] Use programmatic merge when possible

### During Processing

- [ ] Monitor memory usage
- [ ] Watch for GPU utilization
- [ ] Check for bottlenecks
- [ ] Log processing times

### After Processing

- [ ] Clean up GPU memory
- [ ] Force garbage collection
- [ ] Review performance metrics
- [ ] Identify optimization opportunities

---

## Performance Benchmarks

### Typical Processing Times

**Small Document (1-5 pages):**
- VLM Local: 5-15 seconds
- LLM Local: 10-30 seconds
- LLM Remote: 15-45 seconds

**Medium Document (10-20 pages):**
- VLM Local: 30-60 seconds
- LLM Local: 1-3 minutes
- LLM Remote: 2-5 minutes

**Large Document (50+ pages):**
- VLM Local: 2-5 minutes
- LLM Local: 5-15 minutes
- LLM Remote: 10-30 minutes

*Times vary based on hardware, model, and document complexity*

---

## Cost Optimization

### Reduce API Costs

```python
# ✅ Good - Use local inference when possible
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    backend="llm",
    inference="local"  # No API costs
)

# ✅ Good - Use smaller remote models
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    backend="llm",
    inference="remote",
    model_override="mistral-small-latest"  # Cheaper than large models
)

# ❌ Avoid - Unnecessary LLM consolidation
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    llm_consolidation=True  # Extra API call = extra cost
)
```

### Estimate Costs

```python
"""Estimate API costs before processing."""

def estimate_cost(num_pages: int, model: str = "mistral-small-latest"):
    """Estimate processing cost."""
    # Rough estimates (check provider pricing)
    costs_per_page = {
        "mistral-small-latest": 0.01,
        "gpt-4-turbo": 0.05,
        "gemini-2.5-flash": 0.005
    }
    
    cost_per_page = costs_per_page.get(model, 0.02)
    total_cost = num_pages * cost_per_page
    
    print(f"Estimated cost: ${total_cost:.2f}")
    print(f"Model: {model}")
    print(f"Pages: {num_pages}")
    
    return total_cost

# Estimate before processing
estimate_cost(num_pages=100, model="mistral-small-latest")
```

---

## Troubleshooting

### 🐛 Slow Processing

**Solutions:**
1. Use smaller model
2. Enable GPU acceleration
3. Disable chunking for small docs
4. Use local inference
5. Increase batch size

### 🐛 Out of Memory

**Solutions:**
1. Reduce batch size
2. Enable chunking
3. Use smaller model
4. Process one-to-one instead of many-to-one
5. Clean up between documents

### 🐛 GPU Not Utilized

**Solutions:**
1. Verify GPU installation: `torch.cuda.is_available()`
2. Install GPU dependencies: `uv sync --extra local`
3. Check CUDA version compatibility
4. Use vLLM provider for GPU optimization

---

## Performance Optimization Summary

### Quick Wins

1. **Use Provider-Specific Batching**: Automatic optimization per provider
2. **Enable Real Tokenizers**: Accurate token counting prevents overflows
3. **Choose Right Model Tier**: Match model size to task complexity
4. **Clean Up GPU Memory**: Use enhanced cleanup for VLM backends
5. **Disable Chunking for Small Docs**: Faster processing for < 5 pages

### Advanced Optimizations

1. **Multi-GPU Processing**: Parallel document processing
2. **Adaptive Consolidation**: Chain of Density for critical documents
3. **Memory Profiling**: Monitor and optimize resource usage
4. **Batch Size Tuning**: Optimize for your hardware

---

## Next Steps

1. **[Model Capabilities →](../../fundamentals/extraction-process/model-capabilities.md)** - Learn about adaptive prompting
2. **[Error Handling →](error-handling.md)** - Handle errors gracefully
3. **[Testing →](testing.md)** - Test performance optimizations
4. **[GPU Setup →](../../fundamentals/installation/gpu-setup.md)** - Configure GPU