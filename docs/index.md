# Docling Graph Documentation

<p align="center">
  <img src="assets/logo.png" alt="Docling Graph" width="280"/>
</p>

## What is Docling Graph?

Docling-Graph turns documents into validated **Pydantic** objects, then builds a **directed knowledge graph** with explicit semantic relationships.

This transformation enables high-precision use cases in **chemistry, finance, and legal** domains, where AI must capture exact entity connections (compounds and reactions, instruments and dependencies, properties and measurements) **rather than rely on approximate text embeddings**.

This toolkit supports two extraction paths: **local VLM extraction** via Docling, and **LLM-based extraction** using either local runtimes (vLLM, Ollama) or API providers (Mistral, OpenAI, Gemini, IBM WatsonX), all orchestrated through a flexible, config-driven pipeline.

---

### Key Features

- **✍🏻 Multi-Format Input**: Ingest PDFs, images, URLs, raw text, Markdown and more.
- **🧠 Flexible Extraction:** VLM or LLM-based (vLLM, Ollama, Mistral, Gemini, WatsonX, etc.)
- **🔨 Smart Graphs:** Convert Pydantic models to NetworkX graphs with stable node IDs
- **📦 Multiple Export:** CSV (Neo4j-compatible), Cypher scripts, JSON, Markdown
- **📊 Rich Visualizations:** Interactive HTML and detailed Markdown reports
- **⚙️ Type-Safe Configuration:** Pydantic-based validation

---

## Quick Navigation

### 🚀 Getting Started

<div class="grid cards" markdown>

- **[Installation →](fundamentals/installation/index.md)**

    Set up your environment with uv package manager

- **[Quick Start →](introduction/quickstart.md)**

    Run your first extraction in 5 minutes

- **[Architecture →](introduction/architecture.md)**

    Understand the pipeline stages and components

- **[Key Concepts →](introduction/key-concepts.md)**

    Learn how documents flow through the system

</div>

### 📚 Core Documentation

<div class="grid cards" markdown>

- **[Introduction](introduction/index.md)**

    Overview, architecture, and core concepts

- **[Fundamentals](fundamentals/index.md)**

    Installation, schema definition, pipeline configuration, extraction, and more

- **[Usage](usage/index.md)**

    CLI reference, Python API, examples, and advanced topics

- **[Reference](reference/index.md)**

    Detailed API documentation

- **[Community](community/index.md)**

    Contributing and development guide

</div>

---

## Resources

### Documentation
- **[GitHub Repository](https://github.com/IBM/docling-graph)** - Source code and issues
- **[PyPI Package](https://pypi.org/project/docling-graph/)** - Install via pip/uv
- **[Contributing Guidelines](https://github.com/IBM/docling-graph/blob/main/.github/CONTRIBUTING.md)** - How to contribute

### Community
- **[GitHub Issues](https://github.com/IBM/docling-graph/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/IBM/docling-graph/discussions)** - Ask questions and share ideas

### Related Projects
- **[Docling](https://github.com/docling-project/docling)** - Document processing engine
- **[Pydantic](https://pydantic.dev)** - Data validation library
- **[NetworkX](https://networkx.org/)** - Graph library

---

## Next Steps

1. **[Install Docling Graph →](fundamentals/installation/index.md)**
2. **[Follow the Quick Start →](introduction/quickstart.md)**
3. **[Create Your First Template →](fundamentals/schema-definition/index.md)**
4. **[Explore Examples →](usage/examples/index.md)**

---

## Need Help?

- **Installation Issues**: See [Installation Guide](fundamentals/installation/index.md)
- **Template Questions**: See [Schema Definition](fundamentals/schema-definition/index.md)
- **Configuration Help**: See [Pipeline Configuration](fundamentals/pipeline-configuration/index.md)
- **Error Messages**: See [Error Handling](usage/advanced/error-handling.md)