# Extractors API


## Overview

Document extraction strategies and backends.

**Module:** `docling_graph.core.extractors`

!!! tip "Recent Improvements"
    - **Model Capability Detection**: Automatic tier detection and adaptive prompting
    - **Chain of Density**: Multi-turn consolidation for ADVANCED tier models
    - **Zero Data Loss**: Returns partial models instead of empty results on failures
    - **Real Tokenizers**: Accurate token counting with 20% safety margins
    - **Enhanced GPU Cleanup**: Better memory management for VLM backends

---

## Extraction Strategies

### OneToOne

Per-page extraction strategy.

```python
class OneToOne(ExtractorProtocol):
    """Extract data from each page separately."""
    
    def __init__(self, backend: Backend):
        """Initialize with backend."""
        self.backend = backend
    
    def extract(
        self,
        source: str,
        template: Type[BaseModel]
    ) -> List[BaseModel]:
        """
        Extract from each page.
        
        Returns:
            List of models (one per page)
        """
```

**Use Cases:**
- Multi-page documents with independent content
- Page-level analysis
- Parallel processing

**Example:**

```python
from docling_graph.core.extractors import OneToOne
from docling_graph.core.extractors.backends import LLMBackend

backend = LLMBackend(model="llama-3.1-8b")
extractor = OneToOne(backend=backend)

results = extractor.extract("document.pdf", MyTemplate)
print(f"Extracted {len(results)} pages")
```

---

### ManyToOne

Consolidated extraction strategy with zero data loss.

```python
class ManyToOne(ExtractorProtocol):
    """Extract and consolidate data from entire document."""
    
    def __init__(
        self,
        backend: Backend,
        use_chunking: bool = True,
        llm_consolidation: bool = False
    ):
        """Initialize with backend and options."""
        self.backend = backend
        self.use_chunking = use_chunking
        self.llm_consolidation = llm_consolidation
    
    def extract(
        self,
        source: str,
        template: Type[BaseModel]
    ) -> List[BaseModel]:
        """
        Extract and consolidate.
        
        Returns:
            List with single consolidated model (success)
            or multiple partial models (merge failure - zero data loss)
        """
```

**Use Cases:**
- Single entity across document
- Consolidated information
- Summary extraction

**Features:**
- **Zero Data Loss**: Returns partial models if consolidation fails
- **Adaptive Consolidation**: Uses Chain of Density for ADVANCED tier models
- **Schema-Aware Chunking**: Dynamically adjusts chunk size based on schema

**Example:**

```python
from docling_graph.core.extractors import ManyToOne
from docling_graph.core.extractors.backends import LLMBackend

backend = LLMBackend(model="llama-3.1-8b")
extractor = ManyToOne(
    backend=backend,
    use_chunking=True,
    llm_consolidation=True
)

results = extractor.extract("document.pdf", MyTemplate)

# Check if consolidation succeeded
if len(results) == 1:
    print(f"✅ Consolidated model: {results[0]}")
else:
    print(f"⚠ Got {len(results)} partial models (data preserved)")
```

---

## Backends

### LLMBackend

LLM-based extraction backend with adaptive prompting.

```python
class LLMBackend(TextExtractionBackendProtocol):
    """LLM backend for text extraction."""
    
    def __init__(
        self,
        client: LLMClientProtocol,
        model: str,
        provider: str
    ):
        """Initialize LLM backend."""
        self.client = client
        self.model_capability = self._detect_capability()  # Auto-detect tier
```

**Methods:**

- `extract_from_markdown(markdown, template, context, is_partial)` - Extract from markdown with adaptive prompting
- `consolidate_from_pydantic_models(raw_models, programmatic_model, template)` - Consolidate models (uses Chain of Density for ADVANCED tier)
- `cleanup()` - Clean up resources

**Model Capability Tiers:**

| Tier | Model Size | Prompt Style | Consolidation |
|:-----|:-----------|:-------------|:--------------|
| **SIMPLE** | 1B-7B | Minimal | Single-turn |
| **STANDARD** | 7B-13B | Balanced | Single-turn |
| **ADVANCED** | 13B+ | Detailed | Chain of Density (3 turns) |

**Example:**

```python
from docling_graph.core.extractors.backends import LLMBackend
from docling_graph.llm_clients import get_client
from docling_graph.llm_clients.config import resolve_effective_model_config

# STANDARD tier model (7B-13B)
effective = resolve_effective_model_config("ollama", "llama3.1:8b")
client = get_client("ollama")(model_config=effective)
backend = LLMBackend(llm_client=client)

# Automatically uses STANDARD tier prompts
model = backend.extract_from_markdown(
    markdown=markdown,
    template=MyTemplate,
    context="full document",
    is_partial=False
)
```

---

### VLMBackend

Vision-Language Model backend with enhanced GPU cleanup.

```python
class VLMBackend(ExtractionBackendProtocol):
    """VLM backend for document extraction."""
    
    def __init__(self, model: str):
        """Initialize VLM backend."""
        self.model_name = model
        self.model = None  # Loaded on first use
```

**Methods:**

- `extract_from_document(source, template)` - Extract from document
- `cleanup()` - Enhanced GPU memory cleanup

**Enhanced GPU Cleanup:**

The `cleanup()` method now includes:
- Model-to-CPU transfer before deletion
- Explicit CUDA cache clearing
- Memory usage tracking and logging
- Multi-GPU device support

**Example:**

```python
from docling_graph.core.extractors.backends import VLMBackend

backend = VLMBackend(model_name="numind/NuExtract-2.0-8B")

try:
    models = backend.extract_from_document("document.pdf", MyTemplate)
finally:
    backend.cleanup()  # Properly releases GPU memory
```

---

## Document Processing

### DocumentProcessor

Handles document conversion and markdown extraction.

```python
class DocumentProcessor(DocumentProcessorProtocol):
    """Process documents with Docling."""
    
    def convert_to_docling_doc(self, source: str) -> Any:
        """Convert to Docling document."""
    
    def extract_full_markdown(self, document: Any) -> str:
        """Extract full markdown."""
    
    def extract_page_markdowns(self, document: Any) -> List[str]:
        """Extract per-page markdown."""
```

---

## Chunking

### DocumentChunker

Handles document chunking with real tokenizers and schema-aware sizing.

```python
class DocumentChunker:
    """Chunk documents for processing."""
    
    def __init__(
        self,
        provider: str,
        max_tokens: int = None,
        tokenizer_name: str = None,
        schema_json: str | None = None
    ):
        """
        Initialize chunker.
        
        Args:
            provider: LLM provider (for tokenizer selection)
            max_tokens: Maximum tokens per chunk
            tokenizer_name: Specific tokenizer to use
            schema_json: Schema JSON string for dynamic adjustment
        """
    
    def chunk_markdown(
        self,
        markdown: str,
        max_tokens: int
    ) -> List[str]:
        """
        Chunk markdown by tokens using real tokenizer.
        
        Args:
            markdown: Markdown content
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of markdown chunks
        """
    
    def update_schema_config(self, schema_json: str):
        """
        Update schema configuration dynamically.
        
        Args:
            schema_json: New schema JSON string
        """
```

**Features:**

- **Real Tokenizers**: Uses provider-specific tokenizers for accurate token counting
- **Safety Margins**: Reserves a fixed 100-token buffer for protocol overhead
- **Schema-Aware**: Dynamically adjusts chunk size based on exact prompt tokens
- **Provider-Specific**: Optimized for each LLM provider

**Example:**

```python
import json

from docling_graph.core.extractors import DocumentChunker

# Create chunker with real tokenizer
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=4096,
    schema_json=json.dumps(MyTemplate.model_json_schema())
)

# Chunk with accurate token counting
chunks = chunker.chunk_markdown(markdown, max_tokens=4096)

# Update for different schema
chunker.update_schema_config(schema_json=json.dumps(OtherTemplate.model_json_schema()))
```

---

## Factory

### create_extractor()

Factory function for creating extractors.

```python
def create_extractor(
    strategy: Literal["one-to-one", "many-to-one"],
    backend: Backend,
    **kwargs
) -> ExtractorProtocol:
    """
    Create extractor with strategy.
    
    Args:
        strategy: Extraction strategy
        backend: Backend instance
        **kwargs: Additional options
        
    Returns:
        Extractor instance
    """
```

**Example:**

```python
from docling_graph.core.extractors import create_extractor

extractor = create_extractor(
    strategy="many-to-one",
    backend=my_backend,
    use_chunking=True
)
```

---

## New Features Summary

### Model Capability Detection

Automatic detection of model capabilities based on parameter count:

```python
# Automatically detected
backend = LLMBackend(llm_client=client)
# backend.model_capability = ModelCapability.STANDARD (for 8B model)
```

### Chain of Density Consolidation

Multi-turn consolidation for ADVANCED tier models (13B+):

```python
# Automatically enabled for large models
backend = LLMBackend(llm_client=openai_client)  # GPT-4
final = backend.consolidate_from_pydantic_models(
    raw_models=models,
    programmatic_model=draft,
    template=MyTemplate
)
# Uses 3-turn Chain of Density process
```

### Zero Data Loss

Returns partial models instead of empty results:

```python
results = extractor.extract("document.pdf", MyTemplate)

if len(results) == 1:
    # Success: merged model
    model = results[0]
else:
    # Partial: multiple models (data preserved!)
    for model in results:
        process_partial(model)
```

### Real Tokenizer Integration

Accurate token counting with safety margins:

```python
chunker = DocumentChunker(
    provider="mistral",
    max_tokens=4096  # Uses real Mistral tokenizer
)
# Applies 20% safety margin automatically
```

---

## Related APIs

- **[Model Capabilities](../fundamentals/extraction-process/model-capabilities.md)** - Capability tiers
- **[Extraction Process](../fundamentals/extraction-process/index.md)** - Usage guide
- **[Model Merging](../fundamentals/extraction-process/model-merging.md)** - Zero data loss
- **[Protocols](protocols.md)** - Backend protocols
- **[Custom Backends](../usage/advanced/custom-backends.md)** - Create backends