# Extractors API


## Overview

Document extraction strategies and backends.

**Module:** `docling_graph.core.extractors`

!!! tip "Recent Improvements"
    - **Zero Data Loss**: Many-to-one returns partial models instead of empty results when merge fails
    - **Enhanced GPU Cleanup**: Model-to-CPU transfer, CUDA cache clearing, and multi-GPU support for VLM backends

---

## Extraction Strategies

### OneToOneStrategy

Per-page extraction strategy. Each page is processed independently.

```python
class OneToOneStrategy(BaseExtractor):
    """Extracts one model per page/item using Protocol-based type checking."""

    def __init__(self, backend: Backend, docling_config: str = "default") -> None:
        """
        Args:
            backend: VlmBackend or LlmBackend instance.
            docling_config: Docling pipeline configuration ('ocr' or 'vision').
        """

    def extract(
        self, source: str, template: Type[BaseModel]
    ) -> Tuple[List[BaseModel], DoclingDocument | None]:
        """
        Returns:
            Tuple of (models, docling_document):
            - models: one Pydantic model per page.
            - docling_document: the converted document, or None on failure.
        """
```

**Use Cases:**

- Multi-page documents with independent content
- Page-level analysis
- Parallel processing

**Example:**

```python
from docling_graph.core.extractors import OneToOneStrategy
from docling_graph.core.extractors.backends.llm_backend import LlmBackend
from docling_graph.llm_clients import get_client
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config("ollama", "llama3.1:8b")
client = get_client("ollama")(model_config=effective)
backend = LlmBackend(llm_client=client)

extractor = OneToOneStrategy(backend=backend, docling_config="ocr")
results, docling_document = extractor.extract("document.pdf", MyTemplate)
print(f"Extracted {len(results)} pages")
```

---

### ManyToOneStrategy

Extracts one consolidated model from an entire document.

```python
class ManyToOneStrategy(BaseExtractor):
    """Extracts one consolidated model from an entire document."""

    def __init__(
        self,
        backend: Backend,
        docling_config: str = "ocr",
        extraction_contract: str = "direct",
        use_chunking: bool = True,
        chunk_max_tokens: int | None = None,
    ) -> None:
        """
        Args:
            backend: VlmBackend or LlmBackend instance.
            docling_config: Docling pipeline configuration ('ocr' or 'vision').
            extraction_contract: 'direct' or 'dense' (LLM backend only).
            use_chunking: Enable document chunking (required for 'dense').
            chunk_max_tokens: Max tokens per chunk when chunking is used.
        """

    def extract(
        self, source: str, template: Type[BaseModel]
    ) -> Tuple[List[BaseModel], DoclingDocument | None]:
        """
        Returns:
            Tuple of (models, docling_document):
            - models: single-element list with the merged model on success;
              multiple partial models if consolidation fails (zero data loss);
              empty list on total failure.
            - docling_document: the converted document, or None on failure.
        """
```

**Use Cases:**

- Single entity across document
- Consolidated information
- Summary extraction

**Features:**

- **Zero Data Loss**: Returns all partial models if consolidation fails, instead of discarding data
- **Contract-driven**: `extraction_contract="direct"` (single call) or `"dense"` (two-phase skeleton-then-fill; see [Dense Extraction](../fundamentals/extraction-process/dense-extraction.md))
- **Chunking**: `use_chunking=True` splits large documents via `DocumentChunker` before extraction

**Example:**

```python
from docling_graph.core.extractors import ManyToOneStrategy
from docling_graph.core.extractors.backends.llm_backend import LlmBackend
from docling_graph.llm_clients import get_client
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config("ollama", "llama3.1:8b")
client = get_client("ollama")(model_config=effective)
backend = LlmBackend(llm_client=client)

extractor = ManyToOneStrategy(backend=backend, use_chunking=True)
results, docling_document = extractor.extract("document.pdf", MyTemplate)

# Check if consolidation succeeded
if len(results) == 1:
    print(f"✅ Consolidated model: {results[0]}")
else:
    print(f"⚠ Got {len(results)} partial models (data preserved)")
```

---

## Backends

### LlmBackend

LLM-based extraction backend. Performs direct full-document extraction in a single call, or contract-driven skeleton-then-fill extraction when `extraction_contract="dense"`.

```python
class LlmBackend:
    """Backend for LLM-based extraction."""

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        extraction_contract: Literal["direct", "dense"] = "direct",
        dense_config: dict[str, Any] | None = None,
        structured_output: bool = True,
        structured_sparse_check: bool = True,
    ) -> None:
        """
        Args:
            llm_client: LLM client instance implementing LLMClientProtocol.
            extraction_contract: 'direct' (single call) or 'dense' (skeleton-then-fill).
            dense_config: Dense-contract tuning (see Dense Extraction docs).
            structured_output: Use API schema-enforced output when supported.
            structured_sparse_check: Retry with legacy prompt mode if structured output looks sparse.
        """
```

**Methods:**

- `extract_from_markdown(markdown, template, context="document", is_partial=False) -> BaseModel | None` — direct, single-call extraction
- `extract_from_chunk_batches(*, chunks, chunk_metadata, template, context="document") -> BaseModel | None` — dense contract entry point (skeleton + fill across pre-chunked content)
- `generate(system_prompt, user_prompt, max_tokens=None)` — free-form generation, used internally for gleaning passes
- `cleanup()` — release the underlying LLM client

**Example:**

```python
from docling_graph.core.extractors.backends.llm_backend import LlmBackend
from docling_graph.llm_clients import get_client
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config("ollama", "llama3.1:8b")
client = get_client("ollama")(model_config=effective)
backend = LlmBackend(llm_client=client)

model = backend.extract_from_markdown(
    markdown=markdown,
    template=MyTemplate,
    context="full document",
    is_partial=False,
)
```

---

### VlmBackend

Vision-Language Model backend (local inference only), with enhanced GPU cleanup.

```python
class VlmBackend:
    """Backend for VLM-based extraction (local only)."""

    def __init__(self, model_name: str) -> None:
        """
        Args:
            model_name: HuggingFace model repository ID (e.g. 'numind/NuExtract-2.0-8B').
        """
```

**Methods:**

- `extract_from_document(source, template) -> List[BaseModel]` — extract directly from the document image/PDF (one model per page/item)
- `cleanup()` — enhanced GPU memory cleanup
- `cleanup_all_gpus()` — clear CUDA cache across every visible device (multi-GPU setups)

**Enhanced GPU Cleanup:**

The `cleanup()` method includes:
- Model-to-CPU transfer before deletion
- Explicit CUDA cache clearing and synchronization
- Memory usage tracking and logging

**Example:**

```python
from docling_graph.core.extractors.backends.vlm_backend import VlmBackend

backend = VlmBackend(model_name="numind/NuExtract-2.0-8B")

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
class DocumentProcessor:
    """Process documents with Docling. Structurally satisfies DocumentProcessorProtocol."""
    
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

Structure-preserving document chunker built on Docling's `HybridChunker`. Keeps tables and lists intact, respects section hierarchy, and guarantees no chunk exceeds `chunk_max_tokens`.

```python
class DocumentChunker:
    """Structure-preserving document chunker using Docling's HybridChunker."""

    def __init__(
        self,
        tokenizer_name: str | None = None,
        chunk_max_tokens: int = 512,
        merge_peers: bool = True,
    ) -> None:
        """
        Args:
            tokenizer_name: Tokenizer to use for counting (default:
                sentence-transformers/all-MiniLM-L6-v2; pass "tiktoken" for
                OpenAI-style counting).
            chunk_max_tokens: Maximum tokens per chunk (hard cap).
            merge_peers: Merge peer sections during chunking.
        """

    def chunk_document(self, document: DoclingDocument) -> List[str]:
        """Chunk a DoclingDocument into structure-aware text chunks."""

    def chunk_document_with_stats(self, document: DoclingDocument) -> tuple[List[str], dict]:
        """Chunk and return stats: total_chunks, avg_tokens, max_tokens_in_chunk, ..."""

    def chunk_text_fallback(self, text: str) -> List[str]:
        """Sentence-aware fallback splitter for raw text without a DoclingDocument."""

    def get_config_summary(self) -> dict:
        """Return the chunker's current configuration."""
```

**Features:**

- **Structure-preserving**: Tables, lists, and section hierarchy are kept intact via Docling's `HybridChunker`
- **Hard cap**: Any chunk that would exceed `chunk_max_tokens` is re-split by `chunk_text_fallback` (sentence, then word, then character boundaries) — chunks never silently exceed the limit
- **Single sizing knob**: `chunk_max_tokens` — no coupling to model context limits or output budgets

**Example:**

```python
from docling_graph.core.extractors.document_chunker import DocumentChunker

chunker = DocumentChunker(chunk_max_tokens=1024, merge_peers=True)

chunks = chunker.chunk_document(docling_document)
print(f"{len(chunks)} chunks, config: {chunker.get_config_summary()}")
```

---

## Factory

### ExtractorFactory.create_extractor()

Creates an extractor from pipeline configuration. Used internally by the pipeline; for programmatic use, import from `docling_graph.core`.

```python
from docling_graph.core import ExtractorFactory

extractor = ExtractorFactory.create_extractor(
    processing_mode="many-to-one",
    backend_name="llm",
    extraction_contract="direct",  # or "dense" (LLM + many-to-one only)
    dense_config=None,             # optional: dense_skeleton_batch_tokens, dense_fill_nodes_cap, dense_fill_context, parallel_workers
    llm_client=client,
    docling_config="ocr",
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `processing_mode` | `"one-to-one"` \| `"many-to-one"` | Extraction strategy |
| `backend_name` | `"llm"` \| `"vlm"` | Backend type |
| `extraction_contract` | `"direct"` \| `"dense"` | LLM contract; `dense` applies to many-to-one |
| `dense_config` | `dict` \| `None` | Optional dense tuning (skeleton batch tokens, fill cap, parallel workers, etc.) |
| `model_name` | `str` \| `None` | Required for VLM |
| `llm_client` | `LLMClientProtocol` \| `None` | Required for LLM |
| `docling_config` | `str` | `"ocr"` or `"vision"` |

**Returns:** `BaseExtractor` instance.

---

## Features

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

### Structure-Preserving Chunking

Chunks respect table, list, and section boundaries and never exceed the configured token cap:

```python
from docling_graph.core.extractors.document_chunker import DocumentChunker

chunker = DocumentChunker(chunk_max_tokens=1024)
chunks = chunker.chunk_document(docling_document)
# Every chunk is <= 1024 tokens; oversized structural chunks are re-split, never dropped.
```

---

## Related APIs

- **[Dense Extraction](../fundamentals/extraction-process/dense-extraction.md)** - Two-phase skeleton-then-fill extraction
- **[Extraction Process](../fundamentals/extraction-process/index.md)** - Usage guide
- **[Model Merging](../fundamentals/extraction-process/model-merging.md)** - Zero data loss
- **[Protocols](protocols.md)** - Backend protocols
- **[Custom Backends](../usage/advanced/custom-backends.md)** - Create backends