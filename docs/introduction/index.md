# Introduction to Docling Graph

**Pipeline Stage**: 1 - Introduction & Concepts

Welcome to Docling Graph! This section introduces you to the core concepts of knowledge graph extraction from documents.

## What is Docling Graph?

Docling Graph is a powerful toolkit that transforms unstructured documents into validated **knowledge graphs** with precise semantic relationships. Unlike traditional document processing that converts text to vectors or embeddings (losing exact relationships), Docling Graph preserves the explicit connections between entities.

### The Problem with Traditional Approaches

Traditional document processing methods:
- Convert text to embeddings/vectors
- Lose precise semantic relationships
- Cannot answer "who issued what to whom"
- Lack explainability and audit trails

### The Docling Graph Solution

Docling Graph addresses these limitations by:

1. **Extracting Structured Data**: Uses Pydantic schemas to guide extraction
2. **Validating Information**: Ensures data quality through type checking and validation
3. **Building Knowledge Graphs**: Creates explicit entity-relationship graphs
4. **Preserving Relationships**: Maintains exact connections (e.g., "Document A was issued by Organization B to Person C")
5. **Enabling Explainability**: Provides clear audit trails showing how information connects

## Why Knowledge Graphs?

Knowledge graphs are essential for complex domains where understanding exact entity connections is critical:

### Chemistry & Materials Science
- Track chemical compounds and their reactions
- Link materials to their properties and measurements
- Understand synthesis processes and conditions

### Finance & Legal
- Map financial instruments and their dependencies
- Track contractual relationships and obligations
- Maintain compliance and regulatory connections

### Research & Academia
- Connect authors, papers, and citations
- Link methodologies to results
- Track experimental conditions and outcomes

### Healthcare
- Relate patients, treatments, and outcomes
- Connect diagnoses to medications
- Track clinical trial relationships

## How It Works: The Pipeline

Docling Graph follows a clear pipeline flow:

--8<-- "docs/assets/flowcharts/pipeline_stages.md"

### Pipeline Stages

1. **[Installation](../fundamentals/installation/index.md)**: Set up your environment with `uv`
2. **[Schema Definition](../fundamentals/schema-definition/index.md)**: Create Pydantic templates defining what to extract
3. **[Pipeline Configuration](../fundamentals/pipeline-configuration/index.md)**: Configure extraction backend and processing mode
4. **[Extraction Process](../fundamentals/extraction-process/index.md)**: Run document conversion and data extraction
5. **[Graph Management](../fundamentals/graph-management/index.md)**: Export graphs and create visualizations

## Key Features

### 🧠 Flexible Extraction
- **VLM Backend**: Local vision-language models for structured documents
- **LLM Backend**: Local (vLLM, Ollama) or remote (OpenAI, Mistral, Gemini, WatsonX) for complex documents
- **Hybrid Chunking**: Smart document segmentation for large files
- **Processing Modes**: Page-wise or whole-document strategies

### 🔨 Graph Construction
- **Validated Data**: Pydantic ensures type safety and data quality
- **Stable Node IDs**: Deterministic identifiers for consistent graphs
- **Rich Relationships**: Explicit edge labels with semantic meaning
- **Smart Deduplication**: Automatic entity and component merging

### 📦 Multiple Export Formats
- **CSV**: Neo4j-compatible bulk import
- **Cypher**: Script generation for graph databases
- **JSON**: General-purpose data exchange
- **Docling**: Original document preservation
- **HTML**: Interactive visualization

### 📊 Visualization
- **Interactive Graphs**: Cytoscape.js-powered exploration
- **Detailed Reports**: Markdown documentation with statistics
- **Node Inspection**: Click to see entity properties
- **Relationship Tracking**: Hover to view edge labels

## Core Concepts

Before diving into the pipeline, familiarize yourself with these key concepts:

- **[Key Concepts](key-concepts.md)**: Entities, components, nodes, edges, and graphs
- **[Use Cases](use-cases.md)**: Domain-specific examples and patterns
- **[Architecture](architecture.md)**: System design and component interaction

## Quick Example

Here's a minimal example to give you a taste:

```python
from docling_graph import run_pipeline, PipelineConfig
from pydantic import BaseModel, Field

# 1. Define what to extract
class Person(BaseModel):
    model_config = {'is_entity': True, 'graph_id_fields': ['name']}
    name: str = Field(description="Person's full name")
    email: str = Field(description="Email address")

# 2. Configure pipeline
config = PipelineConfig(
    source="document.pdf",
    template=Person,
    backend="llm",
    inference="remote",
    output_dir="outputs"
)

# 3. Run extraction
run_pipeline(config)
```

Result: A knowledge graph with Person nodes, exported as CSV, Cypher, JSON, and interactive HTML!

## Next Steps

Ready to get started? Follow the pipeline stages:

1. **[Installation](../fundamentals/installation/index.md)** - Set up your environment
2. **[Key Concepts](key-concepts.md)** - Understand the fundamentals
3. **[Use Cases](use-cases.md)** - See domain-specific examples

## Related Documentation

- **[Examples](../usage/examples/index.md)**: Working code examples
- **[CLI Reference](../usage/cli/index.md)**: Command-line usage
- **[Python API](../usage/api/index.md)**: Programmatic usage

---

**Ready to transform your documents into knowledge graphs?** Let's begin with [installation](../fundamentals/installation/index.md)!