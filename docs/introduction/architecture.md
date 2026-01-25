# Architecture

## System Architecture

Docling Graph follows a modular, pipeline-based architecture with clear separation of concerns:

![Architecture](../assets/screenshots/architecture.png)


## Core Components

### Document Processor

Converts documents to structured format using Docling with OCR or Vision pipelines.

**Location:** `docling_graph/core/extractors/document_processor.py`

### Extraction Backends

**VLM Backend:** Direct extraction from images using vision-language models (local only)
**LLM Backend:** Text-based extraction supporting local (vLLM, Ollama) and remote APIs

**Location:** `docling_graph/core/extractors/backends/`

### Processing Strategies

**One-to-One:** Each page produces a separate model (invoice batches, ID cards)
**Many-to-One:** Multiple pages merged into single model (rheology researchs, reports)

**Location:** `docling_graph/core/extractors/strategies/`

### Document Chunker

Splits large documents while preserving semantic coherence and respecting structure.

**Location:** `docling_graph/core/extractors/document_chunker.py`

### Graph Converter

Transforms Pydantic models to NetworkX graphs with stable node IDs and automatic deduplication.

**Location:** `docling_graph/core/converters/graph_converter.py`

### Exporters & Visualizers

Export graphs in CSV, Cypher, JSON formats and generate interactive HTML visualizations.

**Location:** `docling_graph/core/exporters/`, `docling_graph/core/visualizers/`


### Complete Pipeline Flow

--8<-- "docs/assets/flowcharts/pipeline_flow.md"


### Stage-by-Stage Breakdown

#### Stage 1: Template Loading
```python
# Load Pydantic template
template = import_template("module.Template")
# Validate structure
validate_template(template)
```

#### Stage 2: Document Conversion
```python
# Convert using Docling
doc = processor.convert_to_docling_doc(source)
# Extract markdown
markdown = processor.extract_full_markdown(doc)
```

#### Stage 3: Extraction
```python
# Choose backend
if backend == "vlm":
    models = vlm_backend.extract_from_document(source, template)
else:
    models = llm_backend.extract_from_markdown(markdown, template)
```

#### Stage 4: Consolidation (if needed)
```python
if len(models) > 1:
    if llm_consolidation:
        final_model = llm_backend.consolidate(models, template)
    else:
        final_model = programmatic_merge(models)
```

#### Stage 5: Graph Conversion
```python
# Convert to graph
graph, metadata = converter.pydantic_list_to_graph([final_model])
```

#### Stage 6: Export
```python
# Export in multiple formats
csv_exporter.export(graph, output_dir)
cypher_exporter.export(graph, output_dir)
json_exporter.export(graph, output_dir)
```


## Protocol-Based Design

Docling Graph uses Python Protocols for type-safe, flexible interfaces:

```python
class ExtractionBackendProtocol(Protocol):
    """Protocol for extraction backends"""
    def extract_from_document(self, source: str, template: Type[BaseModel]) -> List[BaseModel]: ...
```

**Benefits:** Type safety, easy mocking, clear contracts, flexible implementations


**Location**: `docling_graph/config.py`

**Purpose**: Type-safe configuration using Pydantic

```python
class PipelineConfig(BaseModel):
    """Single source of truth for all defaults"""
    source: str
    template: Union[str, Type[BaseModel]]
    backend: Literal["llm", "vlm"] = "llm"
    inference: Literal["local", "remote"] = "local"
    processing_mode: Literal["one-to-one", "many-to-one"] = "many-to-one"
    use_chunking: bool = True
    llm_consolidation: bool = False
    export_format: Literal["csv", "cypher"] = "csv"
    output_dir: str = "outputs"
    # ... additional settings
```

## Error Handling

**Location**: `docling_graph/exceptions.py`

**Hierarchy**:
```
DoclingGraphError (base)
├── ConfigurationError
├── ClientError
├── ExtractionError
├── ValidationError
├── GraphError
└── PipelineError
```

**Structured Errors**:
```python
try:
    run_pipeline(config)
except ClientError as e:
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
    print(f"Cause: {e.cause}")
```


## Extensibility

Docling Graph is designed for extension:

- **LLM Providers:** Extend `BaseLlmClient`
- **Pipeline Stages:** Implement `PipelineStage`
- **Export Formats:** Extend `BaseExporter`

See [Custom Backends](../usage/advanced/custom-backends.md) for details.


Now that you understand the architecture:

1. **[Installation](../fundamentals/installation/index.md)** - Set up your environment
2. **[Schema Definition](../fundamentals/schema-definition/index.md)** - Create Pydantic templates
3. **[Pipeline Configuration](../fundamentals/pipeline-configuration/index.md)** - Configure the pipeline