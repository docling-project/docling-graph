# Pipeline Flowchart

This page provides a visual representation of the complete Docling Graph pipeline, showing how documents flow through the system from input to knowledge graph output.

## Interactive Flowchart

The following Mermaid diagram illustrates the complete pipeline architecture and data flow:

![Docling Graph architecture](docling_graph_workflow.png)

## Pipeline Stages Explained

### Stage 1: Input & Configuration

**Components**:
- **Source Document**: PDF, image, or other supported format
- **Config**: Pipeline configuration (backend, inference, processing mode)
- **Pydantic Template**: Schema defining extraction structure and graph relationships

**Purpose**: Define what to extract and how to process it.

### Stage 2: Document Processing

**Components**:
- **Docling Pipeline**: Converts documents to structured format
  - **OCR**: Traditional OCR pipeline for standard documents
  - **Vision**: Vision-Language Model pipeline for complex layouts
- **Markdown Processor**: Extracts markdown representation

**Purpose**: Convert unstructured documents into processable text/markdown.

### Stage 3: Extraction

**Components**:
- **Extraction Factory**: Selects appropriate backend and strategy
- **Extraction Backend**: 
  - **VLM**: Direct visual extraction from documents
  - **LLM**: Text-based extraction from markdown
- **Prompt**: Schema-guided extraction instructions

**Purpose**: Extract structured data using AI models.

### Stage 4: Processing Strategy

**Components**:
- **Conversion Strategy**: Determines how to handle multi-page documents
  - **One-to-One**: Each page → separate model
  - **Many-to-One**: All pages → single merged model
- **Smart Template Merger**: Consolidates extracted data (for Many-to-One)

**Purpose**: Organize extracted data according to processing strategy.

### Stage 5: Validation

**Components**:
- **Populated Pydantic Model(s)**: Validated, structured data instances

**Purpose**: Ensure data quality through schema validation.

### Stage 6: Graph Construction

**Components**:
- **Graph Converter**: Transforms Pydantic models into NetworkX graph
- **Knowledge Graph**: Directed graph with nodes and edges

**Purpose**: Create semantic knowledge graph from validated data.

### Stage 7: Export & Visualization

**Components**:
- **Exporter**: Generates multiple output formats
  - **CSV**: Neo4j-compatible nodes/edges
  - **Cypher**: Bulk import scripts
  - **JSON**: General-purpose graph data
- **Visualizer**: Creates human-readable outputs
  - **HTML**: Interactive Cytoscape.js visualization
  - **Markdown**: Detailed reports
  - **Images**: Static graph visualizations
- **Batch Loader**: Prepares data for database ingestion
- **Knowledge Base**: Final graph database

**Purpose**: Make graph data accessible in various formats.

## Data Flow Examples

### Example 1: VLM One-to-One

```
ID Card (3 pages)
    ↓
Docling Vision Pipeline
    ↓
VLM Backend (NuExtract)
    ↓
One-to-One Strategy
    ↓
[IDCard_1, IDCard_2, IDCard_3]
    ↓
Graph Converter
    ↓
Knowledge Graph (3 person nodes)
    ↓
CSV Export → Neo4j
```

### Example 2: LLM Many-to-One with Chunking

```
Research Paper (20 pages)
    ↓
Docling OCR Pipeline
    ↓
Markdown Processor
    ↓
Document Chunker (4 chunks)
    ↓
LLM Backend (Mistral)
    ↓
Many-to-One Strategy
    ↓
Smart Template Merger
    ↓
[ResearchPaper_merged]
    ↓
Graph Converter
    ↓
Knowledge Graph (paper + authors + sections)
    ↓
Interactive HTML Visualization
```

## Key Decision Points

### 1. Backend Selection

```
Document Type?
├─ Structured Form → VLM Backend
└─ Complex Narrative → LLM Backend
```

### 2. Processing Strategy

```
Pages Independent?
├─ Yes → One-to-One
└─ No → Many-to-One
    └─ Document Size?
        ├─ Small (< 5 pages) → No Chunking
        └─ Large (≥ 5 pages) → With Chunking
```

### 3. Consolidation Method

```
Many-to-One + Chunking?
├─ Simple Merging → Programmatic (llm_consolidation=False)
└─ Complex Merging → LLM-Based (llm_consolidation=True)
```

## Component Interactions

### Extraction Factory

The Extraction Factory is the central orchestrator:

```python
# Pseudo-code showing factory logic
if backend == "vlm":
    backend_instance = VLMBackend(model)
    if processing_mode == "one-to-one":
        strategy = OneToOneExtractor(backend_instance)
    else:
        strategy = ManyToOneExtractor(backend_instance)
        
elif backend == "llm":
    llm_client = get_client(provider, model)
    backend_instance = LLMBackend(llm_client)
    if processing_mode == "one-to-one":
        strategy = OneToOneExtractor(backend_instance)
    else:
        strategy = ManyToOneExtractor(
            backend_instance,
            use_chunking=use_chunking,
            llm_consolidation=llm_consolidation
        )
```

### Graph Converter

The Graph Converter transforms models to graphs:

```python
# Pseudo-code showing conversion logic
for model in pydantic_models:
    # Create node from model
    node_id = generate_stable_id(model)
    graph.add_node(node_id, **model.dict())
    
    # Create edges from relationships
    for field_name, field_value in model:
        if is_edge_field(field_name):
            edge_label = get_edge_label(field_name)
            target_id = generate_stable_id(field_value)
            graph.add_edge(node_id, target_id, label=edge_label)
```

## Performance Characteristics

### Pipeline Stages by Time

```
Document Processing:    ████░░░░░░ 20-30%
Extraction:            ████████░░ 40-50%
Graph Conversion:      ██░░░░░░░░ 10-15%
Export/Visualization:  ██░░░░░░░░ 10-15%
```

### Bottlenecks

1. **Extraction**: Most time-consuming (especially for LLM)
2. **Document Processing**: Can be slow for large PDFs
3. **Consolidation**: Adds overhead for Many-to-One with LLM

### Optimization Strategies

- **Caching**: Reuse converted documents
- **Batching**: Process multiple chunks in parallel (future)
- **Model Selection**: Choose faster models for speed-critical applications

## Error Handling Flow

```
Pipeline Start
    ↓
Try: Load Config
    ├─ Success → Continue
    └─ Failure → Exit with error
    ↓
Try: Load Template
    ├─ Success → Continue
    └─ Failure → Exit with error
    ↓
Try: Extract Data
    ├─ Success → Continue
    └─ Failure → Log warning, return empty
    ↓
Try: Validate Models
    ├─ Success → Continue
    └─ Failure → Log error, skip invalid
    ↓
Try: Build Graph
    ├─ Success → Continue
    └─ Failure → Exit with error
    ↓
Try: Export
    ├─ Success → Complete
    └─ Failure → Log error, partial output
    ↓
Finally: Cleanup Resources
```

## Next Steps

- Understand [Architecture](architecture.md) in detail
- Learn about [Extraction Backends](extraction-backends.md)
- Explore [Processing Strategies](processing-strategies.md)
- Deep dive into [Graph Construction](graph-construction.md)