# API Reference


## Overview

Complete API reference for docling-graph modules, classes, and functions.

**What's Included:**

- Pipeline API
- Configuration classes
- Protocol definitions
- Exception hierarchy
- Converter classes
- Extractor classes
- Exporter classes
- LLM client interfaces

---

## Quick Links

### Core APIs

**[Pipeline API](pipeline.md)**  
Main entry point for document processing.

- `run_pipeline()` - Execute the pipeline
- Pipeline stages and orchestration

**[Configuration API](config.md)**  
Type-safe configuration classes.

- `PipelineConfig` - Main configuration class
- `ModelConfig` - Model configuration
- `LLMConfig` / `VLMConfig` - Backend configs

**[Protocols](protocols.md)**  
Protocol definitions for type-safe interfaces.

- `ExtractionBackendProtocol` - VLM backends
- `TextExtractionBackendProtocol` - LLM backends
- `LLMClientProtocol` - LLM clients
- `ExtractorProtocol` - Extraction strategies

**[Exceptions](exceptions.md)**  
Exception hierarchy and error handling.

- `DoclingGraphError` - Base exception
- `ConfigurationError` - Config errors
- `ClientError` - API errors
- `ExtractionError` - Extraction failures
- `ValidationError` - Data validation
- `GraphError` - Graph operations
- `PipelineError` - Pipeline execution

---

### Processing APIs

**[Converters](converters.md)**  
Graph conversion from Pydantic models.

- `GraphConverter` - Convert models to graphs
- `NodeIDRegistry` - Stable node IDs
- Graph construction utilities

**[Extractors](extractors.md)**  
Document extraction strategies.

- `OneToOne` - Per-page extraction
- `ManyToOne` - Consolidated extraction
- Backend implementations
- Chunking and batching

**[Exporters](exporters.md)**  
Graph export formats.

- `CSVExporter` - Neo4j-compatible CSV
- `CypherExporter` - Cypher scripts
- `JSONExporter` - JSON format
- `DoclingExporter` - Docling documents
- `graph_to_dict()` / `load_graph_from_dict()` - File-free graph round trip

**[Provenance](provenance.md)**  
Deterministic node-to-source grounding.

- `ProvenanceLedger`, `NodeProvenance`, `ChunkRecord` - Ledger models
- `bind_provenance()` - Binds `__provenance__` onto graph nodes
- `locate_values()` - Verbatim identifier locator

**[LLM Clients](llm-clients.md)**  
LiteLLM-backed client for all LLM calls.

- `LiteLLMClient` - Provider-agnostic client

---

## Module Structure

```
docling_graph/
├── __init__.py              # Public API exports
├── pipeline/                # run_pipeline(), stages, orchestrator
├── config.py                # PipelineConfig
├── protocols.py             # Protocol definitions
├── exceptions.py            # Exception hierarchy
│
├── core/                    # Core processing
│   ├── converters/          # Graph conversion
│   ├── extractors/          # Extraction strategies
│   ├── exporters/           # Export formats
│   ├── importers/           # Load exported graphs back into NetworkX
│   ├── provenance/          # Data grounding (deterministic node-to-source)
│   └── visualizers/         # Visualization
│
├── llm_clients/             # LLM integrations
│   ├── base.py
│   ├── ollama.py
│   ├── mistral.py
│   ├── openai.py
│   ├── gemini.py
│   └── vllm.py
│
└── pipeline/                # Pipeline orchestration
    ├── context.py
    ├── stages.py
    └── orchestrator.py
```

---

## Import Patterns

### Basic Imports

```python
# Main API
from docling_graph import run_pipeline, PipelineConfig

# Configuration classes
from docling_graph import (
    LLMConfig,
    VLMConfig,
    ModelConfig,
    ModelsConfig
)
```

### Advanced Imports

```python
# Protocols
from docling_graph.protocols import (
    ExtractionBackendProtocol,
    TextExtractionBackendProtocol,
    LLMClientProtocol
)

# Exceptions
from docling_graph.exceptions import (
    DoclingGraphError,
    ConfigurationError,
    ClientError,
    ExtractionError,
    ValidationError,
    GraphError,
    PipelineError
)

# Converters
from docling_graph.core.converters import GraphConverter

# Extractors
from docling_graph.core.extractors import OneToOne, ManyToOne

# Exporters
from docling_graph.core.exporters import (
    CSVExporter,
    CypherExporter,
    JSONExporter,
    graph_to_dict
)

# Importers (file-free round trip)
from docling_graph.core.importers import load_graph_from_dict
```

---

## Type Hints

### Common Types

```python
from typing import Any, Dict, List, Type, Union
from pathlib import Path
from pydantic import BaseModel
import networkx as nx

# Configuration
config: PipelineConfig
config_dict: Dict[str, Any]

# Templates
template: Type[BaseModel]
model_instance: BaseModel
models: List[BaseModel]

# Graphs
graph: nx.MultiDiGraph

# Paths
source: Union[str, Path]
output_dir: Path
```

---

## Version Information

```python
import docling_graph

# Get version
print(docling_graph.__version__)  # e.g., "v1.2.0"

# Check available exports
print(docling_graph.__all__)
# ['run_pipeline', 'PipelineConfig', 'LLMConfig', ...]
```

---

## API Stability

### 🟢 Stable APIs

These APIs are stable and safe to use:

- `run_pipeline()`
- `PipelineConfig`
- All configuration classes
- Exception hierarchy
- Public protocols

### 🟣 Internal APIs

These are internal and may change:

- `pipeline.orchestrator` internals
- `core.extractors.backends` internals
- `core.utils` modules

### 🟡 Experimental

These are experimental:

- Custom stage APIs
- Advanced pipeline customization

---

## Deprecation Policy

Deprecated features will:

1. Be marked with `@deprecated` decorator
2. Emit `DeprecationWarning`
3. Be documented in CHANGELOG
4. Be removed after 2 minor versions

Example:

```python
import warnings

@deprecated("Use PipelineConfig instead")
def old_function():
    warnings.warn(
        "old_function is deprecated, use PipelineConfig",
        DeprecationWarning,
        stacklevel=2
    )
```

---

## API Design Principles

### 1. Type Safety

All public APIs use type hints:

```python
def run_pipeline(config: Union[PipelineConfig, Dict[str, Any]]) -> PipelineContext:
    """Type-safe function signature; returns pipeline context with graph and results."""
    pass
```

### 2. Pydantic Validation

Configuration uses Pydantic for validation:

```python
config = PipelineConfig(
    source="doc.pdf",
    template="templates.MyTemplate",
    backend="llm"  # Validated at runtime
)
```

### 3. Protocol-Based

Extensibility through protocols:

```python
class MyBackend(TextExtractionBackendProtocol):
    """Custom backend implementing protocol."""
    pass
```

### 4. Structured Exceptions

Clear error hierarchy:

```python
try:
    run_pipeline(config)
except ConfigurationError as e:
    print(f"Config error: {e.message}")
    print(f"Details: {e.details}")
```

---

## Usage Examples

### Basic Usage

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    backend="llm",
    inference="local"
)

run_pipeline(config)
```

### Advanced Usage

```python
from docling_graph import run_pipeline
from docling_graph.exceptions import ExtractionError

config = {
    "source": "document.pdf",
    "template": "templates.MyTemplate",
    "backend": "llm",
    "inference": "remote",
    "model_override": "mistral-small-latest",
    "use_chunking": True,
    "export_format": "cypher"
}

try:
    run_pipeline(config)
except ExtractionError as e:
    print(f"Extraction failed: {e}")
```

---

## API Documentation Sections

1. **[Pipeline API](pipeline.md)** - Main entry point
2. **[Configuration API](config.md)** - Configuration classes
3. **[Protocols](protocols.md)** - Protocol definitions
4. **[Exceptions](exceptions.md)** - Exception hierarchy
5. **[Converters](converters.md)** - Graph conversion
6. **[Extractors](extractors.md)** - Extraction strategies
7. **[Exporters](exporters.md)** - Export formats
8. **[Provenance](provenance.md)** - Data grounding
9. **[LLM Clients](llm-clients.md)** - LLM integrations

---

## Contributing

See [Development Guide](../community/index.md) for:

- Adding new APIs
- API design guidelines
- Documentation standards
- Testing requirements