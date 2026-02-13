# Pipeline API


## Overview

The Pipeline API provides the main entry point for document extraction and graph conversion.

**Module:** `docling_graph.pipeline`

---

## Functions

### run_pipeline()

```python
def run_pipeline(config: Union[PipelineConfig, Dict[str, Any]]) -> PipelineContext
```

Run the extraction and graph conversion pipeline.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `PipelineConfig` or `dict` | Pipeline configuration |

**Returns:** `PipelineContext` with `knowledge_graph`, `extracted_models`, `graph_metadata`, and related results.

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

    config: PipelineConfig
    template: type[BaseModel] | None = None
    extractor: BaseExtractor | None = None
    extracted_models: list[BaseModel] | None = None
    docling_document: Any = None
    knowledge_graph: nx.DiGraph | None = None
    graph_metadata: GraphMetadata | None = None
    output_dir: Path | None = None
    node_registry: Any | None = None

    # Input normalization
    normalized_source: Union[str, Path, Any] | None = None
    input_metadata: Dict[str, Any] | None = None
    input_type: Any | None = None

    # Output and debug
    output_manager: OutputDirectoryManager | None = None
    trace_data: EventTrace | None = None  # Populated when config.debug is True
```

When `debug=True`, `trace_data` is populated as a chronological event log (`EventTrace`). When output is written to disk, it is saved as `debug/trace_data.json` in compact form with `summary` (including `runtime_seconds`), and ordered `steps` entries (`name`, `runtime_seconds`, `status`, `artifacts`). See [Debug Mode](../usage/advanced/trace-data-debugging.md) for details.

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
| `extraction_contract` | `"direct"` or `"staged"` | `"direct"` | LLM extraction contract for many-to-one mode |
| `use_chunking` | `bool` | `True` | Enable chunking |

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
    "extraction_contract": "staged",
    "use_chunking": True,
    
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
    └── debug/                        # Debug output (if debug=True)
        └── trace_data.json           # Compact summary + ordered steps with canonical artifacts
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
      "max_batch_size": 1
    }
  },
  "processing_time_seconds": 15.42,
  "results": {
    "nodes": 25,
    "edges": 18,
    "extracted_models": 4
  }
}
```

### Output Directory Manager

The `OutputDirectoryManager` organizes all outputs into a structured hierarchy:

```python
from docling_graph.core.utils.output_manager import OutputDirectoryManager

# Create manager
manager = OutputDirectoryManager(
    base_output_dir=Path("outputs"),
    source_filename="document.pdf"
)

# Main output directories
docling_dir = manager.get_docling_dir()
graph_dir = manager.get_docling_graph_dir()
debug_dir = manager.get_debug_dir()

# Optional debug subdirs (per-page / per-chunk)
per_page_dir = manager.get_per_page_dir()
per_chunk_dir = manager.get_per_chunk_dir()
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
    "use_chunking": False  # Skip chunking for small docs
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

# Inspect graph (path depends on output_dir and document name)
from docling_graph.core.utils.output_manager import OutputDirectoryManager
manager = OutputDirectoryManager(Path("outputs"), "document.pdf")
graph_path = manager.get_docling_graph_dir() / "graph.json"
if graph_path.exists():
    with open(graph_path) as f:
        graph_data = json.load(f)
        print(f"Nodes: {len(graph_data.get('nodes', []))}")
        print(f"Edges: {len(graph_data.get('links', []))}")
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