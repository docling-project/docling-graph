# Docling Graph

<p align="center">
  <img src="assets/logo.png" alt="Docling Graph" width="280"/>
</p>

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://ibm.github.io/docling-graph)
[![Docling](https://img.shields.io/badge/Docling-VLM-red)](https://github.com/docling-project/docling)
[![PyPI version](https://img.shields.io/pypi/v/docling-graph)](https://pypi.org/project/docling-graph/)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-red)](https://networkx.org/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![Typer](https://img.shields.io/badge/Typer-CLI-purple)](https://typer.tiangolo.com/)
[![Rich](https://img.shields.io/badge/Rich-terminal-purple)](https://github.com/Textualize/rich)
[![vLLM](https://img.shields.io/badge/vLLM-compatible-brightgreen)](https://vllm.ai/)
[![Ollama](https://img.shields.io/badge/Ollama-compatible-brightgreen)](https://ollama.ai/)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/11598/badge)](https://www.bestpractices.dev/projects/11598)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-003778?logo=linuxfoundation&logoColor=fff&color=0094ff&labelColor=003778)](https://lfaidata.foundation/projects/)
[![License MIT](https://img.shields.io/github/license/IBM/docling-graph)](https://opensource.org/licenses/MIT)

**Docling Graph** converts documents into validated **Pydantic** objects and then into a **directed knowledge graph**, with exports to CSV or Cypher and both static and interactive visualizations.

## Overview

This transformation of unstructured documents into validated knowledge graphs with precise semantic relationships is essential for complex domains like **chemistry, finance, and physics** where AI systems must understand exact entity connections (e.g., chemical compounds and their reactions, financial instruments and their dependencies, physical properties and their measurements) rather than approximate text vectors, **enabling explainable reasoning over technical document collections**.

The toolkit supports two extraction families: **local VLM** via Docling and **LLM-based extraction** via local (vLLM, Ollama) or API providers (Mistral, OpenAI, Gemini, IBM WatsonX), all orchestrated by a flexible, config-driven pipeline.

## Key Capabilities

### 🧠 Extraction
- **Local VLM** - Docling's information extraction pipeline (ideal for small documents with key-value focus)
- **LLM** - Local via vLLM/Ollama or remote via Mistral/OpenAI/Gemini/IBM WatsonX API
- **Hybrid Chunking** - Leveraging Docling's segmentation with semantic LLM chunking
- **Flexible Strategies** - Page-wise or whole-document conversion

### 🔨 Graph Construction
- **Markdown to Graph** - Convert validated Pydantic instances to NetworkX DiGraph
- **Smart Merge** - Combine multi-page documents into unified processing
- **Type Safety** - Enhanced with Pydantic validation and configuration

### 📦 Export
- **Docling Document** - JSON format with full document structure
- **Markdown** - Full document and per-page options
- **CSV** - Compatible with Neo4j admin import
- **Cypher** - Script generation for bulk ingestion
- **JSON** - General-purpose graph data

### 📊 Visualization
- **Interactive HTML** - Full-page browser view with node/edge exploration
- **Markdown Reports** - Detailed graph nodes content and edges

## Quick Start

```bash
# Install
pip install docling-graph

# Or with all features
pip install docling-graph[all]
```

```python
from docling_graph import run_pipeline, PipelineConfig
from your_templates import YourTemplate

config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="llm",
    output_dir="outputs"
)

run_pipeline(config)
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed installation instructions
- [Quick Start](getting-started/quickstart.md) - Get up and running quickly
- [Pydantic Templates](guides/create_pydantic_templates_for_kg_extraction.md) - Create extraction templates
- [Examples](examples/README.md) - Explore example use cases

## Community

- **GitHub**: [IBM/docling-graph](https://github.com/IBM/docling-graph)
- **Issues**: [Report bugs or request features](https://github.com/IBM/docling-graph/issues)
- **Contributing**: See our [Contributing Guide](https://github.com/IBM/docling-graph/blob/main/CONTRIBUTING.md)

## License

MIT License - see [LICENSE](https://github.com/IBM/docling-graph/blob/main/LICENSE) for details.

---

**IBM ❤️ Open Source AI**

Docling Graph has been brought to you by IBM Research.