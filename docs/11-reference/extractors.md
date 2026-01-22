# Extractors API

**Navigation:** [← Converters](converters.md) | [Next: Exporters →](exporters.md)

---

## Overview

Document extraction strategies and backends.

**Module:** `docling_graph.core.extractors`

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

Consolidated extraction strategy.

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
            List with single consolidated model
        """
```

**Use Cases:**
- Single entity across document
- Consolidated information
- Summary extraction

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
print(f"Consolidated model: {results[0]}")
```

---

## Backends

### LLMBackend

LLM-based extraction backend.

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
```

**Methods:**
- `extract_from_markdown()` - Extract from markdown
- `consolidate_from_pydantic_models()` - Consolidate models
- `cleanup()` - Clean up resources

---

### VLMBackend

Vision-Language Model backend.

```python
class VLMBackend(ExtractionBackendProtocol):
    """VLM backend for document extraction."""
    
    def __init__(self, model: str):
        """Initialize VLM backend."""
```

**Methods:**
- `extract_from_document()` - Extract from document
- `cleanup()` - Clean up resources

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

Handles document chunking for large documents.

```python
class DocumentChunker:
    """Chunk documents for processing."""
    
    def chunk_markdown(
        self,
        markdown: str,
        max_tokens: int
    ) -> List[str]:
        """
        Chunk markdown by tokens.
        
        Args:
            markdown: Markdown content
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of markdown chunks
        """
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

## Related APIs

- **[Extraction Process](../05-extraction-process/index.md)** - Usage guide
- **[Protocols](protocols.md)** - Backend protocols
- **[Custom Backends](../10-advanced/custom-backends.md)** - Create backends

---

**Navigation:** [← Converters](converters.md) | [Next: Exporters →](exporters.md)