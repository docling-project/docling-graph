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
    
    # Trace data (optional)
    trace_data: TraceData | None = None
    output_manager: OutputDirectoryManager | None = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## Trace Data Structure

When `include_trace=True`, the pipeline captures intermediate data at each stage:

```python
@dataclass
class TraceData:
    """Container for trace data captured during pipeline execution."""
    
    # Page-level data
    pages: List[PageData] = field(default_factory=list)
    
    # Chunk-level data (only in many-to-one mode)
    chunks: List[ChunkData] | None = None
    
    # Extraction data
    extractions: List[ExtractionData] = field(default_factory=list)
    
    # Intermediate graphs (before consolidation)
    intermediate_graphs: List[GraphData] = field(default_factory=list)
```

### PageData

Captures data for each document page:

```python
@dataclass
class PageData:
    """Data for a single page."""
    page_number: int
    text_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ChunkData

Captures data for each chunk (many-to-one mode only):

```python
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
    ├── docling/                      # Docling exports
    │   ├── document.json             # Docling JSON (if enabled)
    │   └── markdown/                 # Markdown exports (if enabled)
    │       ├── full_document.md
    │       └── pages/
    │           ├── page_1.md
    │           └── page_2.md
    │
    ├── consolidated_graph/           # Final consolidated graph
    │   ├── nodes.csv                 # Graph nodes (if CSV export)
    │   ├── edges.csv                 # Graph edges (if CSV export)
    │   ├── graph.cypher              # Cypher script (if Cypher export)
    │   ├── graph.json                # Graph JSON
    │   ├── graph_visualization.html  # Interactive visualization
    │   └── extraction_report.md      # Extraction report
    │
    └── trace/                        # Trace data (if include_trace=True)
        ├── pages/                    # Per-page data
        │   ├── page_1.json
        │   ├── page_1.md
        │   └── page_2.json
        │
        ├── per_chunk/                # Per-chunk data (many-to-one mode)
        │   ├── chunk_0/
        │   │   ├── extraction.json
        │   │   ├── chunk.md
        │   │   ├── nodes.csv
        │   │   ├── edges.csv
        │   │   └── graph.json
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