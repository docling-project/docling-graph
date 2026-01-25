# Trace Data Debugging

## Overview

Trace data capture provides visibility into the extraction pipeline's intermediate stages, enabling debugging, performance analysis, and quality assurance. This feature is particularly useful when diagnosing extraction issues or optimizing pipeline performance.

**What's Captured:**
- **Pages**: Raw page content and metadata
- **Chunks**: Text chunks with page mappings (many-to-one mode)
- **Extractions**: Extraction results with timing and errors
- **Intermediate Graphs**: Per-chunk/per-page graphs before consolidation
- **Performance Metrics**: Total pipeline processing time

---

## Mode Behavior

### CLI Mode (Trace Enabled by Default)

```bash
# Trace data automatically captured and exported
uv run docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --output-dir "outputs/invoice"

# Output includes trace/ directory with intermediate artifacts
```

### API Mode (Memory Efficient)

```python
from docling_graph import run_pipeline, PipelineConfig

# Default: No trace data (memory efficient)
config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice"
)

context = run_pipeline(config)
# context.trace_data is None
```

---

## Quick Start Examples

### Example 1: Debug Extraction Errors

Enable trace data to diagnose extraction problems:

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="problematic_document.pdf",
    template="templates.ComplexTemplate",
    include_trace=True,  # Enable trace capture
    dump_to_disk=True,   # Export to files
    output_dir="debug_output"
)

context = run_pipeline(config)

# Check for extraction errors
if context.trace_data:
    for extraction in context.trace_data.extractions:
        if extraction.error:
            print(f"❌ Extraction {extraction.extraction_id} failed:")
            print(f"   Source: {extraction.source_type} {extraction.source_id}")
            print(f"   Error: {extraction.error}")
```

### Example 2: Analyze Page Content

Access raw page content for analysis:

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.Research",
    include_trace=True,
    dump_to_disk=False  # Keep in memory only
)

context = run_pipeline(config)

# Analyze page content
if context.trace_data:
    for page in context.trace_data.pages:
        print(f"Page {page.page_number}:")
        print(f"  Content length: {len(page.text_content)} chars")
        print(f"  Has tables: {page.metadata.get('has_tables', False)}")
```

### Example 3: Profile Extraction Performance

Profile extraction performance:

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
    include_trace=True,
    dump_to_disk=False
)

context = run_pipeline(config)

# Calculate statistics
if context.trace_data:
    extraction_times = [e.extraction_time for e in context.trace_data.extractions]
    
    print("Extraction Performance:")
    print(f"  Total extractions: {len(extraction_times)}")
    print(f"  Average time: {sum(extraction_times) / len(extraction_times):.2f}s")
    print(f"  Min time: {min(extraction_times):.2f}s")
    print(f"  Max time: {max(extraction_times):.2f}s")
```

---

## Trace Data Structure

### Accessing Trace Data

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
    include_trace=True,
    dump_to_disk=False
)

context = run_pipeline(config)
trace = context.trace_data

if trace:
    # Pages
    for page in trace.pages:
        page_number: int = page.page_number
        content: str = page.text_content
        metadata: dict = page.metadata
    
    # Chunks (many-to-one mode only)
    if trace.chunks:
        for chunk in trace.chunks:
            chunk_id: int = chunk.chunk_id
            pages: list[int] = chunk.page_numbers
            tokens: int = chunk.token_count
            content: str = chunk.text_content
    
    # Extractions
    for extraction in trace.extractions:
        extraction_id: int = extraction.extraction_id
        source_type: str = extraction.source_type  # "page" or "chunk"
        source_id: int = extraction.source_id
        model = extraction.parsed_model  # Pydantic model or None
        time: float = extraction.extraction_time
        error: str | None = extraction.error
    
    # Intermediate graphs
    for graph_data in trace.intermediate_graphs:
        graph_id: int = graph_data.graph_id
        source_type: str = graph_data.source_type
        source_id: int = graph_data.source_id
        graph = graph_data.graph  # NetworkX DiGraph
        model = graph_data.pydantic_model
        nodes: int = graph_data.node_count
        edges: int = graph_data.edge_count
```

---

## Output Structure

When `dump_to_disk=True` and `include_trace=True`, debug data is exported to:

```
outputs/document_name_timestamp/
└── debug/
    ├── pages/                    # Per-page data
    │   ├── page_001.json
    │   └── page_002.json
    │
    ├── chunks/                   # Chunk data
    │   ├── chunk_000.md
    │   └── metadata.json
    │
    ├── parsed_models/            # Extraction results
    │   ├── extraction_000.json
    │   └── extraction_001.json
    │
    └── intermediate_graphs/      # Per-chunk graphs (many-to-one mode)
        ├── chunk_000/
        │   ├── graph.json
        │   └── model.json
    │   └── chunk_1/
    │       └── ...
    │
    ├── per_page/                 # Per-page data (one-to-one mode)
    │   ├── page_1/
    │   │   ├── extraction.json
    │   │   ├── nodes.csv
    │   │   ├── edges.csv
    │   │   └── graph.json
    │   └── page_2/
    │       └── ...
    │
    └── chunks_metadata.json      # Chunk metadata (many-to-one mode)
```

---

## Common Debugging Patterns

### Pattern 1: Find Failed Extractions

```python
if context.trace_data:
    failed = [e for e in context.trace_data.extractions if e.error]
    
    if failed:
        print(f"Found {len(failed)} failed extractions:")
        for extraction in failed:
            print(f"  {extraction.source_type} {extraction.source_id}: {extraction.error}")
```

### Pattern 2: Identify Slow Extractions

```python
if context.trace_data:
    slow = [e for e in context.trace_data.extractions if e.extraction_time > 5.0]
    
    if slow:
        print(f"Found {len(slow)} slow extractions (>5s):")
        for extraction in sorted(slow, key=lambda e: e.extraction_time, reverse=True):
            print(f"  {extraction.source_type} {extraction.source_id}: {extraction.extraction_time:.2f}s")
```

### Pattern 3: Analyze Chunk Boundaries

```python
if context.trace_data and context.trace_data.chunks:
    print(f"Document split into {len(context.trace_data.chunks)} chunks:")
    
    for chunk in context.trace_data.chunks:
        print(f"  Chunk {chunk.chunk_id}: pages {chunk.page_numbers}, {chunk.token_count} tokens")
```

### Pattern 4: Compare Intermediate Graphs

```python
if context.trace_data:
    for graph_data in context.trace_data.intermediate_graphs:
        print(f"{graph_data.source_type.capitalize()} {graph_data.source_id}:")
        print(f"  Nodes: {graph_data.node_count}, Edges: {graph_data.edge_count}")
```

---

## Best Practices

### Memory Management

```python
# ✅ Good: Disable trace in production
config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
    include_trace=False  # Memory efficient
)

# ✅ Good: Enable trace for debugging with file export
config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
    include_trace=True,
    dump_to_disk=True  # Export to files
)

# ⚠️ Caution: Trace in memory for large documents
config = PipelineConfig(
    source="large_document.pdf",
    template="templates.Report",
    include_trace=True,
    dump_to_disk=False  # May use significant memory
)
```

### Conditional Trace Capture

```python
import os

# Enable trace only in development
debug_mode = os.getenv("DEBUG", "false").lower() == "true"

config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
    include_trace=debug_mode,
    dump_to_disk=debug_mode
)
```

### Testing with Trace Data

```python
def test_extraction_quality():
    config = PipelineConfig(
        source="test_document.pdf",
        template="templates.TestTemplate",
        include_trace=True,
        dump_to_disk=False
    )
    
    context = run_pipeline(config)
    
    # Validate no extraction errors
    assert context.trace_data is not None
    errors = [e for e in context.trace_data.extractions if e.error]
    assert len(errors) == 0, f"Found {len(errors)} extraction errors"
    
    # Validate extraction times
    slow = [e for e in context.trace_data.extractions if e.extraction_time > 10.0]
    assert len(slow) == 0, f"Found {len(slow)} slow extractions"
```

---

## Related Documentation

- **[Configuration API](../../reference/config.md#trace-data-capture)** - `include_trace` field
- **[Pipeline API](../../reference/pipeline.md#trace-data-structure)** - Trace data structure
- **[Error Handling](../advanced/error-handling.md)** - Debugging strategies
- **[Performance Tuning](../advanced/performance-tuning.md)** - Optimization tips

---

## See Also

- **[Output Directory Structure](../../reference/pipeline.md#output-structure)** - File organization
- **[Testing Guide](../advanced/testing.md)** - Using trace data in tests
- **[Batch Processing](../api/batch-processing.md)** - Process multiple documents