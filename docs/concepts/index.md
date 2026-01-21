# Concepts

Welcome to the Docling Graph concepts documentation. This section provides in-depth explanations of the core concepts, architecture, and design principles behind Docling Graph.

## Overview

Docling Graph transforms unstructured documents into validated knowledge graphs through a flexible, config-driven pipeline. Understanding these core concepts will help you effectively use and extend the toolkit.

## Core Concepts

### [Architecture](architecture.md)
Learn about the overall system architecture, component interactions, and the pipeline flow from document to knowledge graph.

### [Extraction Backends](extraction-backends.md)
Understand the two extraction families (VLM and LLM), their differences, and when to use each approach.

### [Processing Strategies](processing-strategies.md)
Explore the one-to-one and many-to-one processing modes and how they affect document handling.

### [Pydantic Templates](pydantic-templates.md)
Deep dive into how Pydantic models serve as extraction schemas and define graph structure.

### [Graph Construction](graph-construction.md)
Learn how validated Pydantic objects are converted into NetworkX directed graphs with semantic relationships.

### [Export Formats](export-formats.md)
Understand the various export formats (CSV, Cypher, JSON) and their use cases.

### [Configuration System](configuration.md)
Explore the configuration system and how to customize pipeline behavior.

## Why Knowledge Graphs?

Traditional document processing approaches convert text to vectors or embeddings, which lose precise semantic relationships between entities. Docling Graph addresses this by:

- **Preserving Relationships**: Maintains explicit connections between entities (e.g., "who issued what to whom")
- **Enabling Explainability**: Provides audit trails showing how information connects
- **Supporting Complex Queries**: Enables traversal of exact relationships for domain-specific questions
- **Ensuring Validation**: Validates data against schemas before graph construction

This is critical for complex domains like chemistry, finance, physics, and legal documents where understanding exact entity connections is essential for production systems, regulatory compliance, and scientific accuracy.

## Getting Started

If you're new to Docling Graph, we recommend reading the concepts in this order:

1. [Architecture](architecture.md) - Understand the big picture
2. [Extraction Backends](extraction-backends.md) - Choose your extraction approach
3. [Pydantic Templates](pydantic-templates.md) - Learn to define extraction schemas
4. [Processing Strategies](processing-strategies.md) - Select the right processing mode
5. [Graph Construction](graph-construction.md) - Understand graph generation
6. [Export Formats](export-formats.md) - Choose your output format

## Visual Overview

For a visual representation of the complete pipeline, see the [Pipeline Flowchart](pipeline-flowchart.md).