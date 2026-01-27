# Pipeline API


## Overview

The Pipeline API provides the main entry point for document extraction and graph conversion.

**Module:** `docling_graph.pipeline`

---

## Functions

### run_pipeline()

```python
def run_pipeline(config: Union[PipelineConfig, Dict[str, Any]]) -> None
```

Run the extraction and graph conversion pipeline.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `PipelineConfig` or `dict` | Pipeline configuration |

**Returns:** `None`

**Raises:**

| Exception | When |
|-----------|------|
| `PipelineError` | Pipeline execution fails |
| `ConfigurationError` | Configuration is invalid |
| `ExtractionError` | Document extraction fails |

**Example:**

```python
from docling_graph import run_pipeline

# Using dict
config = {
    "source": "document.pdf",
    "template": "templates.MyTemplate",
    "backend": "llm",
    "inference": "local",
    "output_dir": "outputs"
}
run_pipeline(config)

# Using PipelineConfig
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate"
)
run_pipeline(config)
```

---

## Pipeline Stages

The pipeline executes the following stages in order:

### 1. Template Loading

**Purpose:** Load and validate Pydantic templates

**Actions:**
- Import template module
- Validate template structure
- Check for required fields

**Errors:**
- `ConfigurationError` if template not found
- `ValidationError` if template invalid

### 2. Extraction

**Purpose:** Extract structured data from documents

**Actions:**
- Convert document with Docling
- Extract using backend (VLM or LLM)
- Validate extracted data

**Errors:**
- `ExtractionError` if extraction fails
- `ValidationError` if data invalid

### 3. Docling Export (Optional)

**Purpose:** Export Docling document outputs

**Actions:**
- Export Docling JSON
- Export markdown
- Export per-page markdown

**Controlled by:**
- `export_docling`
- `export_docling_json`
- `export_markdown`
- `export_per_page_markdown`

### 4. Graph Conversion

**Purpose:** Convert extracted data to knowledge graphs

**Actions:**
- Create NetworkX graph
- Generate stable node IDs
- Create edges from relationships

**Errors:**
- `GraphError` if conversion fails

### 5. Export

**Purpose:** Export graphs in multiple formats

**Actions:**
- Export to CSV (nodes.csv, edges.csv)
- Export to Cypher (graph.cypher)
- Export to JSON (graph.json)

**Controlled by:**
- `export_format`

### 6. Visualization

**Purpose:** Generate reports and interactive visualizations

**Actions:**
- Create HTML visualization
- Generate markdown report
- Calculate statistics

**Outputs:**
- `graph_visualization.html`
- `extraction_report.md`

---

## Pipeline Context

Internal context object passed between stages:

```python
@dataclass
class PipelineContext:
    """Shared context for pipeline stages."""
    
    # Configuration
    config: Dict[str, Any]
    
    # Paths
    source: Path
    output_dir: Path
    
    # Pipeline state
    template: Type[BaseModel] | None = None
    docling_doc: Any = None
    extracted_models: List[BaseModel] | None = None
    graph: nx.MultiDiGraph | None = None
    
    # Output management
    output_manager: OutputDirectoryManager | None = None
    output_dir: str | None = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Note:** The old `trace_data` field has been replaced with the new debug system. When `debug=True` in the config, all intermediate artifacts are saved to disk in the `debug/` directory. See [Debug Mode Documentation](../usage/advanced/trace-data-debugging.md) for details.

---

## Debug Artifacts

When `debug=True` in the pipeline configuration, all intermediate extraction artifacts are saved to `outputs/{document}_{timestamp}/debug/`:

- **slots.jsonl** - Slot metadata (chunks/pages with token counts)
- **atoms_all.jsonl** - All atomic facts extracted
- **field_catalog.json** - Global field catalog
- **reducer_report.json** - Reducer decisions and conflicts
- **best_effort_model.json** - Final model output
- **slots_text/** - Full slot text for replay
- **atoms/** - Per-slot extraction attempts
- **field_catalog_selected/** - Per-slot field selections
- **arbitration/** - Conflict resolution requests/responses

See [Debug Mode Documentation](../usage/advanced/trace-data-debugging.md) for complete details on debug artifacts and usage patterns.

---

## Legacy Trace Data (Deprecated)

The old `TraceData` structure with `pages`, `chunks`, `extractions`, and `intermediate_graphs` has been replaced by the new debug system. If you have code using `context.trace_data`, migrate to the new debug artifacts:

**Old approach:**
```python
if context.trace_data:
    for extraction in context.trace_data.extractions:
        print(extraction.error)
```

**New approach:**
```python
import json
from pathlib import Path

# Enable debug mode
config = PipelineConfig(source="doc.pdf", template="templates.MyTemplate", debug=True)
context = run_pipeline(config)

# Analyze debug artifacts
debug_dir = Path(context.output_dir) / "debug"
atoms_dir = debug_dir / "atoms"

# Check for validation errors
for attempt_file in atoms_dir.glob("*_attempt*.json"):
    with open(attempt_file) as f:
        attempt = json.load(f)
        if not attempt['validation_success']:
            print(f"Error in {attempt['slot_id']}: {attempt['error']}")
```

@dataclass
class ChunkData:
    """Data for a single chunk."""
    chunk_id: int
    text_content: str
    page_numbers: List[int]
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ExtractionData

Captures data for each extraction operation:

```python
@dataclass
class ExtractionData:
    """Data for a single extraction."""
    extraction_id: int
    source_type: Literal["page", "chunk"]
    source_id: int
    parsed_model: BaseModel | None
    extraction_time: float
    error: str | None = None
```

### GraphData

Captures intermediate graphs before consolidation:

```python
@dataclass
class GraphData:
    """Data for an intermediate graph."""
    graph_id: int
    source_type: Literal["page", "chunk"]
    source_id: int
    graph: nx.DiGraph
    pydantic_model: BaseModel
    node_count: int
    edge_count: int
```

---

## Configuration Options

### Required

| Option | Type | Description |
|--------|------|-------------|
| `source` | `str` or `Path` | Path to source document |
| `template` | `str` or `Type[BaseModel]` | Pydantic template |

### Backend Selection

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `backend` | `"llm"` or `"vlm"` | `"llm"` | Extraction backend |
| `inference` | `"local"` or `"remote"` | `"local"` | Inference location |

### Processing

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `processing_mode` | `"one-to-one"` or `"many-to-one"` | `"many-to-one"` | Processing strategy |
| `use_chunking` | `bool` | `True` | Enable chunking |
| `llm_consolidation` | `bool` | `False` | Use LLM for merge |

### Export

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `export_format` | `"csv"` or `"cypher"` | `"csv"` | Graph export format |
| `output_dir` | `str` or `Path` | `"outputs"` | Output directory |

See [Configuration API](config.md) for complete options.

---

## Usage Patterns

### Basic Usage

```python
from docling_graph import run_pipeline

run_pipeline({
    "source": "document.pdf",
    "template": "templates.MyTemplate"
})
```

### With Error Handling

```python
from docling_graph import run_pipeline
from docling_graph.exceptions import (
    ConfigurationError,
    ExtractionError,
    PipelineError
)

try:
    run_pipeline({
        "source": "document.pdf",
        "template": "templates.MyTemplate"
    })
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Details: {e.details}")
except ExtractionError as e:
    print(f"Extraction failed: {e.message}")
except PipelineError as e:
    print(f"Pipeline error: {e.message}")
```

### Batch Processing

```python
from pathlib import Path
from docling_graph import run_pipeline

documents = Path("documents").glob("*.pdf")

for doc in documents:
    print(f"Processing {doc.name}...")
    
    run_pipeline({
        "source": str(doc),
        "template": "templates.MyTemplate",
        "output_dir": f"outputs/{doc.stem}"
    })
    
    print(f"✅ {doc.name} complete")
```

### Custom Configuration

```python
from docling_graph import run_pipeline

config = {
    "source": "document.pdf",
    "template": "templates.MyTemplate",
    
    # Backend
    "backend": "llm",
    "inference": "remote",
    "model_override": "mistral-small-latest",
    "provider_override": "mistral",
    
    # Processing
    "processing_mode": "many-to-one",
    "use_chunking": True,
    "llm_consolidation": True,
    
    # Export
    "export_format": "cypher",
    "export_docling_json": True,
    "export_markdown": True,
    
    # Output
    "output_dir": "outputs/custom"
}

run_pipeline(config)
```

---

## Output Structure

After successful execution with `dump_to_disk=True`, the output directory contains:

```
outputs/
└── document_name_timestamp/
    ├── metadata.json                 # Pipeline metadata and performance metrics
    │
    ├── docling/                      # Docling exports
    │   ├── document.json             # Docling JSON (if enabled)
    │   └── document.md               # Markdown export (if enabled)
    │
    ├── docling_graph/                # Docling-graph outputs
    │   ├── graph.json                # Graph JSON
    │   ├── nodes.csv                 # Graph nodes (if CSV export)
    │   ├── edges.csv                 # Graph edges (if CSV export)
    │   ├── graph.cypher              # Cypher script (if Cypher export)
    │   ├── graph.html                # Interactive visualization
    │   └── report.md                 # Extraction report
    │
    └── debug/                        # Debug artifacts (if debug=True)
        ├── slots.jsonl               # Slot metadata (one per line)
        ├── atoms_all.jsonl           # All atomic facts (one per line)
        ├── field_catalog.json        # Global field catalog
        ├── reducer_report.json       # Reducer decisions and conflicts
        ├── best_effort_model.json    # Final model output
        ├── provenance.json           # Document path and config for replay
        │
        ├── slots_text/               # Full slot text for replay
        │   ├── slot_0.txt
        │   ├── slot_1.txt
        │   └── ...
        │
        ├── atoms/                    # Per-slot extraction attempts
        │   ├── slot_0_attempt1.json
        │   ├── slot_0_attempt2.json  # Retry if first failed
        │   ├── slot_1_attempt1.json
        │   └── ...
        │
        ├── field_catalog_selected/   # Per-slot field selections
        │   ├── slot_0.json
        │   ├── slot_1.json
        │   └── ...
        │
        └── arbitration/              # Conflict resolution
            ├── request.json          # Conflicts sent to LLM
            └── response.json         # LLM arbitration decisions
```

### metadata.json Structure

The `metadata.json` file contains pipeline configuration, results, and performance metrics:

```json
{
  "pipeline_version": "1.1.0",
  "timestamp": "2026-01-25T12:30:45.123456",
  "input": {
    "source": "document.pdf",
    "template": "templates.BillingDocument"
  },
  "config": {
    "pipeline": {
      "processing_mode": "many-to-one",
      "debug": true,
      "reverse_edges": false,
      "docling": "ocr"
    },
    "extraction": {
      "backend": "llm",
      "inference": "remote",
      "model": "mistral-small-latest",
      "provider": "mistral",
      "use_chunking": true,
      "llm_consolidation": true,
      "max_batch_size": 1
    }
  },
  "processing_time_seconds": 15.42,
  "results": {
    "nodes": 25,
    "edges": 18,
    "extracted_models": 4
  },
  "trace_summary": {
    "pages": 3,
    "chunks": 4,
    "extractions": 4,
    "intermediate_graphs": {
      "count": 4,
      "details": [
        {
          "graph_id": 0,
          "source_type": "chunk",
          "source_id": 0,
          "nodes": 6,
          "edges": 4
        }
      ]
    }
  }
}
```

### Output Directory Manager

The `OutputDirectoryManager` organizes all outputs into a structured hierarchy:

```python
from docling_graph.core.utils.output_manager import OutputDirectoryManager

# Create manager
manager = OutputDirectoryManager(
    base_output_dir="outputs",
    source_filename="document.pdf"
)

# Get directories
docling_dir = manager.get_docling_dir()
consolidated_dir = manager.get_consolidated_graph_dir()
trace_dir = manager.get_trace_dir()
per_chunk_dir = manager.get_per_chunk_dir(chunk_id=0)
per_page_dir = manager.get_per_page_dir(page_number=1)
```

---

## Performance Considerations

### Memory Usage

```python
# For large documents, use chunking
run_pipeline({
    "source": "large_document.pdf",
    "template": "templates.MyTemplate",
    "use_chunking": True,  # Reduces memory usage
    "processing_mode": "one-to-one"  # Process page by page
})
```

### Speed Optimization

```python
# For faster processing
run_pipeline({
    "source": "document.pdf",
    "template": "templates.MyTemplate",
    "backend": "llm",
    "inference": "local",  # Faster than remote
    "use_chunking": False,  # Skip chunking for small docs
    "llm_consolidation": False  # Skip LLM merge
})
```

---

## Debugging

### Enable Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run pipeline
from docling_graph import run_pipeline

run_pipeline({
    "source": "document.pdf",
    "template": "templates.MyTemplate"
})
```

### Inspect Outputs

```python
from pathlib import Path
import json

# Run pipeline
from docling_graph import run_pipeline

run_pipeline({
    "source": "document.pdf",
    "template": "templates.MyTemplate",
    "output_dir": "outputs"
})

# Inspect graph
graph_path = Path("outputs/graph.json")
with open(graph_path) as f:
    graph_data = json.load(f)
    print(f"Nodes: {len(graph_data['nodes'])}")
    print(f"Edges: {len(graph_data['links'])}")
```

---

## Related APIs

- **[Configuration API](config.md)** - PipelineConfig class
- **[Exceptions](exceptions.md)** - Exception hierarchy
- **[Extractors](extractors.md)** - Extraction strategies

---

## See Also

- **[Python API Guide](../usage/api/run-pipeline.md)** - Usage guide
- **[CLI Reference](../usage/cli/convert-command.md)** - CLI equivalent
- **[Examples](../usage/examples/index.md)** - Example usage