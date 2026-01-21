# Architecture

This document provides an overview of Docling Graph's architecture, explaining how components interact to transform documents into knowledge graphs.

## System Overview

Docling Graph follows a modular, pipeline-based architecture with clear separation of concerns.

![Docling Graph architecture](docling_graph_workflow.png)

## Core Components

### 1. Pipeline Orchestrator

**Location**: `docling_graph/pipeline.py`

The main entry point that orchestrates the entire conversion process:

- Loads and validates configuration
- Initializes components based on settings
- Manages resource lifecycle
- Coordinates data flow between components

**Key Function**: `run_pipeline(config: PipelineConfig)`

### 2. Document Processor

**Location**: `docling_graph/core/extractors/document_processor.py`

Handles document conversion using Docling:

- Converts documents to DoclingDocument format
- Extracts full markdown or per-page markdown
- Supports OCR and Vision pipelines
- Caches converted documents to avoid redundant processing

**Protocols**: `DocumentProcessorProtocol`

### 3. Extraction Backends

**Location**: `docling_graph/core/extractors/backends/`

Two backend families for structured extraction:

#### VLM Backend (`vlm_backend.py`)
- Uses Docling's NuExtract models
- Processes documents directly (images or PDFs)
- Ideal for structured documents with key-value pairs
- Local inference only

#### LLM Backend (`llm_backend.py`)
- Uses language models for extraction
- Processes markdown/text input
- Supports local (vLLM, Ollama) and remote (Mistral, OpenAI, Gemini, WatsonX) providers
- Includes chunking and consolidation strategies

**Protocols**: `ExtractionBackendProtocol`, `TextExtractionBackendProtocol`

### 4. LLM Clients

**Location**: `docling_graph/llm_clients/`

Unified interface for multiple LLM providers:

- **Base Client** (`base.py`): Common interface and utilities
- **Local Providers**: vLLM, Ollama
- **Remote Providers**: Mistral, OpenAI, Gemini, WatsonX

All clients implement `LLMClientProtocol` with:
- `get_json_response()`: Execute LLM calls with JSON schema
- `context_limit`: Token limit management

### 5. Processing Strategies

**Location**: `docling_graph/core/extractors/strategies/`

Two strategies for handling multi-page documents:

#### One-to-One (`one_to_one.py`)
- Processes each page independently
- Returns N Pydantic models (one per page)
- Useful for page-specific analysis

#### Many-to-One (`many_to_one.py`)
- Combines all pages into single extraction
- Returns 1 merged Pydantic model
- Supports chunking for large documents
- Optional LLM-based consolidation

**Protocol**: `ExtractorProtocol`

### 6. Document Chunker

**Location**: `docling_graph/core/extractors/document_chunker.py`

Hybrid chunking strategy:

- Leverages Docling's document segmentation
- Applies semantic chunking with LLM context limits
- Preserves document structure (sections, tables, lists)
- Optimizes for token limits while maintaining coherence

### 7. Graph Converter

**Location**: `docling_graph/core/converters/graph_converter.py`

Transforms validated Pydantic models into NetworkX graphs:

- Creates nodes from Pydantic models
- Generates stable node IDs using `graph_id_fields`
- Establishes edges from `edge()` field definitions
- Supports reverse edge generation
- Validates graph structure

**Key Features**:
- **Node ID Registry**: Ensures deterministic, stable node IDs
- **Entity vs Component**: Handles different deduplication strategies
- **Rich Metadata**: Preserves all model data in node attributes

### 8. Exporters

**Location**: `docling_graph/core/exporters/`

Multiple export formats for different use cases:

- **CSV Exporter** (`csv_exporter.py`): Neo4j-compatible nodes/edges CSV
- **Cypher Exporter** (`cypher_exporter.py`): Bulk import scripts
- **JSON Exporter** (`json_exporter.py`): General-purpose graph data
- **Docling Exporter** (`docling_exporter.py`): Original document + markdown

### 9. Visualizers

**Location**: `docling_graph/core/visualizers/`

Generate human-readable outputs:

- **Interactive Visualizer** (`interactive_visualizer.py`): Cytoscape.js HTML graphs
- **Report Generator** (`report_generator.py`): Detailed markdown reports

## Data Flow

### Typical Execution Flow

1. **Configuration Loading**
   - Load `PipelineConfig` from dict or YAML
   - Validate settings and resolve model configurations
   - Initialize output directories

2. **Template Loading**
   - Import Pydantic template class
   - Validate template structure
   - Extract graph metadata from model configs

3. **Document Processing**
   - Convert source document using Docling
   - Extract markdown (full or per-page)
   - Cache DoclingDocument for exports

4. **Extraction**
   - Initialize appropriate backend (VLM or LLM)
   - Apply processing strategy (one-to-one or many-to-one)
   - For LLM: chunk if needed, batch process, consolidate
   - Validate extracted data against Pydantic schema

5. **Graph Construction**
   - Convert validated models to NetworkX DiGraph
   - Generate stable node IDs
   - Create edges from relationship definitions
   - Validate graph structure

6. **Export**
   - Export Docling document and markdown (if configured)
   - Export graph to selected format (CSV/Cypher/JSON)
   - Generate visualizations (HTML, markdown reports)

7. **Cleanup**
   - Release backend resources
   - Clear GPU memory (if applicable)
   - Close connections

## Protocol-Based Design

Docling Graph uses Python Protocols for type-safe, duck-typed interfaces:

```python
# Example: Backend Protocol
class ExtractionBackendProtocol(Protocol):
    def extract_from_document(
        self, source: str, template: Type[BaseModel]
    ) -> List[BaseModel]:
        ...
    
    def cleanup(self) -> None:
        ...
```

**Benefits**:
- Type safety without rigid inheritance
- Easy mocking for tests
- Clear interface contracts
- Flexible implementations

## Configuration System

**Location**: `docling_graph/config.py`

Type-safe configuration using Pydantic:

```python
class PipelineConfig(BaseModel):
    source: str
    template: Union[str, Type[BaseModel]]
    backend: Literal["llm", "vlm"]
    inference: Literal["local", "remote"]
    processing_mode: Literal["one-to-one", "many-to-one"]
    # ... additional settings
```

**Key Features**:
- Single source of truth for defaults
- Validation at configuration time
- Easy programmatic and CLI usage
- YAML export for persistence

## Extensibility Points

### Adding New LLM Providers

1. Create client class in `llm_clients/`
2. Implement `LLMClientProtocol`
3. Register in `llm_clients/__init__.py`
4. Update configuration templates

### Adding New Export Formats

1. Create exporter in `core/exporters/`
2. Inherit from `BaseExporter`
3. Implement `export()` method
4. Register in pipeline

### Custom Processing Strategies

1. Create strategy in `core/extractors/strategies/`
2. Implement `ExtractorProtocol`
3. Register in `ExtractorFactory`

## Performance Considerations

### Memory Management

- **GPU Memory**: Automatic cleanup after extraction
- **Document Caching**: Reuse converted documents
- **Batch Processing**: Configurable batch sizes for chunking

### Optimization Strategies

- **Chunking**: Reduces memory footprint for large documents
- **Lazy Loading**: Import modules only when needed
- **Resource Pooling**: Reuse LLM clients across batches

## Error Handling

The pipeline implements comprehensive error handling:

- **Validation Errors**: Caught at Pydantic model level
- **Extraction Failures**: Graceful degradation with logging
- **Resource Cleanup**: Guaranteed via try-finally blocks
- **User Feedback**: Rich console output with progress indicators

## Next Steps

- Learn about [Extraction Backends](extraction-backends.md)
- Understand [Processing Strategies](processing-strategies.md)
- Explore [Graph Construction](graph-construction.md)