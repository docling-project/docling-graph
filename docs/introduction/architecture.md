# Architecture

**Pipeline Stage**: 1 - Introduction & Concepts

**Prerequisites**: 
- [Introduction](index.md)
- [Key Concepts](key-concepts.md)

This page provides a high-level overview of Docling Graph's architecture and how components interact.

## System Architecture

Docling Graph follows a modular, pipeline-based architecture with clear separation of concerns:

![Architecture](../assets/screenshots/architecture.png)

## Core Components

### 1. Document Processor

**Location**: `docling_graph/core/extractors/document_processor.py`

**Purpose**: Converts documents to structured format using Docling

**Key Features**:
- Supports OCR and Vision pipelines
- Extracts full markdown or per-page markdown
- Preserves document structure (sections, tables, lists)
- Stateless operation for scalability

**Pipeline Options**:
- **OCR Pipeline**: Classic OCR (most accurate for standard documents)
- **Vision Pipeline**: VLM-based (best for complex layouts)

### 2. Extraction Backends

#### VLM Backend

**Location**: `docling_graph/core/extractors/backends/vlm_backend.py`

**Purpose**: Direct extraction from document images using vision-language models

**Characteristics**:
- Processes documents directly (no markdown needed)
- Uses Docling's NuExtract models
- Local inference only
- Ideal for structured forms

**Flow**:
```
Document → VLM Model → Pydantic Validation → Validated Models
```

#### LLM Backend

**Location**: `docling_graph/core/extractors/backends/llm_backend.py`

**Purpose**: Extraction from markdown using language models

**Characteristics**:
- Requires markdown conversion first
- Supports local (vLLM, Ollama) and remote (OpenAI, Mistral, Gemini, WatsonX)
- Includes chunking for large documents
- Better for complex narratives

**Flow**:
```
Document → Markdown → Chunking → LLM Extraction → Consolidation → Validated Models
```

### 3. LLM Clients

**Location**: `docling_graph/llm_clients/`

**Purpose**: Unified interface for multiple LLM providers

**Architecture**:

--8<-- "docs/assets/flowcharts/llm_clients.md"

**Key Features**:
- Template method pattern for consistency
- Centralized JSON parsing
- YAML-based model configuration
- Easy to add new providers

### 4. Processing Strategies

**Location**: `docling_graph/core/extractors/strategies/`

**Purpose**: Handle multi-page documents differently

#### One-to-One Strategy

```
Page 1 → Extract → Model 1
Page 2 → Extract → Model 2
Page 3 → Extract → Model 3

Result: [Model 1, Model 2, Model 3]
```

**Use Case**: Independent pages (invoice batches, ID card scans)

#### Many-to-One Strategy

```
Page 1 ┐
Page 2 ├→ Extract → Merge → Single Model
Page 3 ┘

Result: [Merged Model]
```

**Use Case**: Related content across pages (research papers, reports)

### 5. Document Chunker

**Location**: `docling_graph/core/extractors/document_chunker.py`

**Purpose**: Split large documents while preserving semantic coherence

**Hybrid Chunking Strategy**:

--8<-- "docs/assets/flowcharts/doc_chunker.md"

**Features**:
- Respects document structure (sections, tables)
- Semantic boundary detection
- Token limit awareness
- Context preservation

### 6. Consolidation

**Location**: `docling_graph/core/extractors/backends/llm_backend.py`

**Purpose**: Merge results from multiple chunks

#### Programmatic Merge (Fast)

```python
# Rules:
# - Lists: Concatenate + deduplicate
# - Scalars: First non-null wins
# - Objects: Recursive merge
```

#### LLM Consolidation (Intelligent)

```python
# LLM receives:
# - All partial models
# - Template schema
# - Consolidation prompt
# Returns: Intelligently merged model
```

### 7. Graph Converter

**Location**: `docling_graph/core/converters/graph_converter.py`

**Purpose**: Transform Pydantic models to NetworkX graphs

**Process**:

--8<-- "docs/assets/flowcharts/graph_converter.md"

**Key Features**:
- Stable, deterministic node IDs
- Entity vs component handling
- Automatic deduplication
- Rich metadata preservation

### 8. Node ID Registry

**Location**: `docling_graph/core/converters/node_id_registry.py`

**Purpose**: Ensure stable, unique node identifiers

**Features**:
- Deterministic ID generation
- Collision detection
- Cross-batch consistency
- Type tracking

**ID Generation**:
```python
# For entities (with graph_id_fields)
Person(name="John", dob="1990-01-15")
→ "Person_John_1990-01-15"

# For components (content-based)
Address(street="123 Main", city="Boston")
→ "Address_{content_hash}"
```

### 9. Exporters

**Location**: `docling_graph/core/exporters/`

**Purpose**: Export graphs in multiple formats

--8<-- "docs/assets/flowcharts/exporters.md"

**Exporters**:
- **CSV**: Neo4j admin import
- **Cypher**: Neo4j script execution
- **JSON**: General-purpose data
- **Docling**: Original document preservation

### 10. Visualizers

**Location**: `docling_graph/core/visualizers/`

**Purpose**: Generate human-readable outputs

**Components**:
- **Interactive Visualizer**: Cytoscape.js HTML graphs
- **Report Generator**: Detailed markdown reports

## Data Flow

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
    def cleanup(self) -> None: ...

class LLMClientProtocol(Protocol):
    """Protocol for LLM clients"""
    @property
    def context_limit(self) -> int: ...
    def get_json_response(self, prompt: str, schema_json: str) -> Dict[str, Any]: ...
```

**Benefits**:
- Type safety without rigid inheritance
- Easy mocking for tests
- Clear interface contracts
- Flexible implementations

## Configuration System

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

## Performance Considerations

### Memory Management
- **GPU Memory**: Automatic cleanup after extraction
- **Stateless Design**: No internal caching
- **Batch Processing**: Configurable batch sizes

### Optimization Strategies
- **Chunking**: Reduces memory footprint
- **Lazy Loading**: Import modules on-demand
- **Resource Pooling**: Reuse LLM clients
- **Parallel Processing**: Future optimization for one-to-one mode

## Extensibility Points

### Adding New LLM Providers

```python
from docling_graph.llm_clients.base import BaseLlmClient

class MyClient(BaseLlmClient):
    def _provider_id(self) -> str:
        return "my_provider"
    
    def _setup_client(self, **kwargs):
        self.api_key = self._get_required_env("MY_API_KEY")
        self.client = MyAPI(api_key=self.api_key)
    
    def _call_api(self, messages, **params):
        return self.client.call(messages)
```

### Adding Custom Pipeline Stages

```python
from docling_graph.pipeline import PipelineStage, PipelineContext

class ValidationStage(PipelineStage):
    def name(self) -> str:
        return "Validation"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        # Custom validation logic
        return context
```

### Adding New Export Formats

```python
from docling_graph.core.exporters.base import BaseExporter

class MyExporter(BaseExporter):
    def export(self, graph: nx.DiGraph, output_dir: str) -> None:
        # Custom export logic
        pass
```

## Next Steps

Now that you understand the architecture:

1. **[Installation](../fundamentals/installation/index.md)** - Set up your environment
2. **[Schema Definition](../fundamentals/schema-definition/index.md)** - Create Pydantic templates
3. **[Pipeline Configuration](../fundamentals/pipeline-configuration/index.md)** - Configure the pipeline

## Related Documentation

- **[Key Concepts](key-concepts.md)**: Core terminology
- **[Use Cases](use-cases.md)**: Domain-specific examples
- **[API Reference](../reference/index.md)**: Detailed API documentation

---

**Ready to dive deeper?** Start with [installation](../fundamentals/installation/index.md) or explore [examples](../usage/examples/index.md)!