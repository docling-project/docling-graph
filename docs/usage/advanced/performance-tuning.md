# Performance Tuning


## Overview

Optimize docling-graph pipeline performance for speed, memory efficiency, and resource utilization.

**What You'll Learn:**
- Model selection strategies
- Batch size optimization
- Memory management
- GPU utilization
- Caching strategies
- Profiling techniques

**Prerequisites:**
- Understanding of [Pipeline Configuration](../../fundamentals/pipeline-configuration/index.md)
- Familiarity with [Extraction Process](../../fundamentals/extraction-process/index.md)
- Basic knowledge of system resources

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

## Batch Processing

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
from docling_graph import PipelineConfig

log_memory_usage()  # Before
config = PipelineConfig(...)
config.run()
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

from docling_graph import PipelineConfig
import gc
import torch

def process_with_cleanup(source: str):
    """Process document with proper cleanup."""
    config = PipelineConfig(
        source=source,
        template="templates.MyTemplate"
    )
    
    try:
        config.run()
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
from docling_graph import PipelineConfig

def process_on_gpu(source: str, gpu_id: int):
    """Process document on specific GPU."""
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    config = PipelineConfig(
        source=source,
        template="templates.MyTemplate",
        output_dir=f"outputs/gpu_{gpu_id}"
    )
    config.run()

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

from docling_graph import PipelineConfig

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
| LLM | 🐢 Slow | 💎 High Accuracy | Complex conflicts, narratives |


---

## Profiling

### Profile Pipeline Execution

```python
"""Profile pipeline to identify bottlenecks."""

import time
from docling_graph import PipelineConfig

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
    
    config.run()
    
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

### Issue: Slow Processing

**Solutions:**
1. Use smaller model
2. Enable GPU acceleration
3. Disable chunking for small docs
4. Use local inference
5. Increase batch size

### Issue: Out of Memory

**Solutions:**
1. Reduce batch size
2. Enable chunking
3. Use smaller model
4. Process one-to-one instead of many-to-one
5. Clean up between documents

### Issue: GPU Not Utilized

**Solutions:**
1. Verify GPU installation: `torch.cuda.is_available()`
2. Install GPU dependencies: `uv sync --extra local`
3. Check CUDA version compatibility
4. Use vLLM provider for GPU optimization

---

## Next Steps

1. **[Error Handling →](error-handling.md)** - Handle errors gracefully
2. **[Testing →](testing.md)** - Test performance optimizations
3. **[GPU Setup →](../../fundamentals/installation/gpu-setup.md)** - Configure GPU

---

## Related Documentation

- **[Pipeline Configuration](../../fundamentals/pipeline-configuration/index.md)** - Configuration options
- **[Extraction Process](../../fundamentals/extraction-process/index.md)** - How extraction works
- **[GPU Setup](../../fundamentals/installation/gpu-setup.md)** - GPU configuration