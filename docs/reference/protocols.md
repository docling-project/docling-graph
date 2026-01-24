# Protocols


## Overview

Protocol definitions for type-safe interfaces in docling-graph.

**Module:** `docling_graph.protocols`

Protocols define expected interfaces without requiring inheritance, enabling duck typing with type safety.

---

## Backend Protocols

### ExtractionBackendProtocol

Protocol for VLM backends that process documents directly.

```python
@runtime_checkable
class ExtractionBackendProtocol(Protocol):
    """Protocol for extraction backends that process entire documents."""
    
    def extract_from_document(
        self, 
        source: str, 
        template: Type[BaseModel]
    ) -> List[BaseModel]:
        """Extract structured data from a document."""
        ...
    
    def cleanup(self) -> None:
        """Clean up backend resources."""
        ...
```

**Methods:**

#### extract_from_document()

```python
def extract_from_document(
    source: str,
    template: Type[BaseModel]
) -> List[BaseModel]
```

Extract structured data from a document.

**Parameters:**
- `source` (`str`): Path to source document
- `template` (`Type[BaseModel]`): Pydantic model template

**Returns:** List of extracted Pydantic model instances

**Example:**

```python
class MyVLMBackend(ExtractionBackendProtocol):
    def extract_from_document(self, source, template):
        # Process document directly
        result = self.model.process(source)
        return [template.model_validate(result)]
    
    def cleanup(self):
        del self.model
```

---

### TextExtractionBackendProtocol

Protocol for LLM backends that process markdown/text.

```python
@runtime_checkable
class TextExtractionBackendProtocol(Protocol):
    """Protocol for extraction backends that process markdown/text."""
    
    client: Any  # LLM client instance
    
    def extract_from_markdown(
        self,
        markdown: str,
        template: Type[BaseModel],
        context: str = "document",
        is_partial: bool = False
    ) -> BaseModel | None:
        """Extract structured data from markdown."""
        ...
    
    def consolidate_from_pydantic_models(
        self,
        raw_models: List[BaseModel],
        programmatic_model: BaseModel,
        template: Type[BaseModel]
    ) -> BaseModel | None:
        """Consolidate multiple models using LLM."""
        ...
    
    def cleanup(self) -> None:
        """Clean up backend resources."""
        ...
```

**Attributes:**
- `client` (`Any`): LLM client instance

**Methods:**

#### extract_from_markdown()

```python
def extract_from_markdown(
    markdown: str,
    template: Type[BaseModel],
    context: str = "document",
    is_partial: bool = False
) -> BaseModel | None
```

Extract structured data from markdown content.

**Parameters:**
- `markdown` (`str`): Markdown content
- `template` (`Type[BaseModel]`): Pydantic model template
- `context` (`str`): Context description (e.g., "page 1")
- `is_partial` (`bool`): Whether this is partial extraction

**Returns:** Extracted model instance or None

#### consolidate_from_pydantic_models()

```python
def consolidate_from_pydantic_models(
    raw_models: List[BaseModel],
    programmatic_model: BaseModel,
    template: Type[BaseModel]
) -> BaseModel | None
```

Consolidate multiple models using LLM.

**Parameters:**
- `raw_models` (`List[BaseModel]`): List of extracted models
- `programmatic_model` (`BaseModel`): Programmatically merged model
- `template` (`Type[BaseModel]`): Target template

**Returns:** Consolidated model instance or None

**Example:**

```python
class MyLLMBackend(TextExtractionBackendProtocol):
    def __init__(self, client):
        self.client = client
    
    def extract_from_markdown(self, markdown, template, context="", is_partial=False):
        prompt = f"Extract from: {markdown}"
        response = self.client.get_json_response(prompt, template.model_json_schema())
        return template.model_validate(response)
    
    def consolidate_from_pydantic_models(self, raw_models, programmatic_model, template):
        # Use LLM to merge models
        return programmatic_model
    
    def cleanup(self):
        self.client.close()
```

---

## LLM Client Protocol

### LLMClientProtocol

Protocol for LLM clients (Ollama, Mistral, OpenAI, etc.).

```python
@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM clients."""
    
    @property
    def context_limit(self) -> int:
        """Return effective context limit in tokens."""
        ...
    
    def get_json_response(
        self,
        prompt: str | Mapping[str, str],
        schema_json: str
    ) -> Dict[str, Any]:
        """Execute LLM call and return parsed JSON."""
        ...
```

**Properties:**

#### context_limit

```python
@property
def context_limit(self) -> int
```

Return the effective context limit in tokens.

**Returns:** Conservative token limit

**Methods:**

#### get_json_response()

```python
def get_json_response(
    prompt: str | Mapping[str, str],
    schema_json: str
) -> Dict[str, Any]
```

Execute LLM call and return parsed JSON.

**Parameters:**
- `prompt` (`str` or `Mapping[str, str]`): Prompt (legacy string or dict with 'system' and 'user')
- `schema_json` (`str`): Pydantic schema as JSON string

**Returns:** Parsed JSON dictionary

**Example:**

```python
class MyLLMClient(LLMClientProtocol):
    @property
    def context_limit(self) -> int:
        return 8000  # Conservative limit
    
    def get_json_response(self, prompt, schema_json):
        # Handle both formats
        if isinstance(prompt, dict):
            system = prompt.get("system", "")
            user = prompt.get("user", "")
        else:
            system = ""
            user = prompt
        
        # Call LLM API
        response = self.api.chat(system=system, user=user)
        return json.loads(response)
```

---

## Extractor Protocol

### ExtractorProtocol

Protocol for extraction strategies.

```python
@runtime_checkable
class ExtractorProtocol(Protocol):
    """Protocol for extraction strategies."""
    
    backend: Any  # Backend instance
    
    def extract(
        self,
        source: str,
        template: Type[BaseModel]
    ) -> List[BaseModel]:
        """Extract structured data from source document."""
        ...
```

**Attributes:**
- `backend` (`Any`): Backend instance (VLM or LLM)

**Methods:**

#### extract()

```python
def extract(
    source: str,
    template: Type[BaseModel]
) -> List[BaseModel]
```

Extract structured data from source document.

**Parameters:**
- `source` (`str`): Path to source document
- `template` (`Type[BaseModel]`): Pydantic model template

**Returns:** List of extracted Pydantic model instances
- OneToOne: May contain N models (one per page)
- ManyToOne: Contains 1 merged model

**Example:**

```python
class MyExtractor(ExtractorProtocol):
    def __init__(self, backend):
        self.backend = backend
    
    def extract(self, source, template):
        # Use backend to extract
        return self.backend.extract_from_document(source, template)
```

---

## Document Processor Protocol

### DocumentProcessorProtocol

Protocol for document processing and conversion.

```python
@runtime_checkable
class DocumentProcessorProtocol(Protocol):
    """Protocol for document processing and conversion."""
    
    def convert_to_docling_doc(self, source: str) -> Any:
        """Convert document to Docling document object."""
        ...
    
    def extract_full_markdown(self, document: Any) -> str:
        """Extract complete markdown from document."""
        ...
    
    def extract_page_markdowns(self, document: Any) -> List[str]:
        """Extract markdown for each page separately."""
        ...
```

**Methods:**

#### convert_to_docling_doc()

```python
def convert_to_docling_doc(source: str) -> Any
```

Convert document to Docling document object.

**Parameters:**
- `source` (`str`): Path to source document

**Returns:** Docling document object

#### extract_full_markdown()

```python
def extract_full_markdown(document: Any) -> str
```

Extract complete markdown from document.

**Parameters:**
- `document` (`Any`): Docling document object

**Returns:** Full markdown content as string

#### extract_page_markdowns()

```python
def extract_page_markdowns(document: Any) -> List[str]
```

Extract markdown for each page separately.

**Parameters:**
- `document` (`Any`): Docling document object

**Returns:** List of markdown strings, one per page

---

## Type Checking Utilities

### is_vlm_backend()

```python
def is_vlm_backend(backend: Any) -> TypeGuard[ExtractionBackendProtocol]
```

Check if backend behaves like a VLM backend.

**Parameters:**
- `backend` (`Any`): Backend instance to check

**Returns:** True if backend provides document-level extraction

**Example:**

```python
from docling_graph.protocols import is_vlm_backend

if is_vlm_backend(my_backend):
    # Use VLM-specific features
    result = my_backend.extract_from_document(source, template)
```

### is_llm_backend()

```python
def is_llm_backend(backend: Any) -> TypeGuard[TextExtractionBackendProtocol]
```

Check if backend behaves like an LLM backend.

**Parameters:**
- `backend` (`Any`): Backend instance to check

**Returns:** True if backend provides markdown/text extraction

**Example:**

```python
from docling_graph.protocols import is_llm_backend

if is_llm_backend(my_backend):
    # Use LLM-specific features
    result = my_backend.extract_from_markdown(markdown, template)
```

### get_backend_type()

```python
def get_backend_type(backend: Any) -> str
```

Get the backend type as a string.

**Parameters:**
- `backend` (`Any`): Backend instance

**Returns:** "vlm", "llm", or "unknown"

**Example:**

```python
from docling_graph.protocols import get_backend_type

backend_type = get_backend_type(my_backend)
print(f"Backend type: {backend_type}")
```

---

## Type Aliases

Convenient type aliases for clarity:

```python
# Backend can be either VLM or LLM
Backend = ExtractionBackendProtocol | TextExtractionBackendProtocol

# Extractor strategies
Extractor = ExtractorProtocol

# LLM client
LLMClient = LLMClientProtocol

# Document processor
DocumentProcessor = DocumentProcessorProtocol
```

**Usage:**

```python
from docling_graph.protocols import Backend, LLMClient

def process_with_backend(backend: Backend):
    """Process with any backend type."""
    pass

def create_client() -> LLMClient:
    """Create an LLM client."""
    pass
```

---

## Implementation Examples

### Custom VLM Backend

```python
from docling_graph.protocols import ExtractionBackendProtocol
from typing import List, Type
from pydantic import BaseModel

class CustomVLMBackend(ExtractionBackendProtocol):
    """Custom VLM backend implementation."""
    
    def __init__(self, model_name: str):
        self.model = load_model(model_name)
    
    def extract_from_document(
        self,
        source: str,
        template: Type[BaseModel]
    ) -> List[BaseModel]:
        """Extract from document."""
        result = self.model.process(source)
        return [template.model_validate(result)]
    
    def cleanup(self) -> None:
        """Clean up resources."""
        del self.model
```

### Custom LLM Backend

```python
from docling_graph.protocols import TextExtractionBackendProtocol
from typing import List, Type
from pydantic import BaseModel

class CustomLLMBackend(TextExtractionBackendProtocol):
    """Custom LLM backend implementation."""
    
    def __init__(self, client):
        self.client = client
    
    def extract_from_markdown(
        self,
        markdown: str,
        template: Type[BaseModel],
        context: str = "document",
        is_partial: bool = False
    ) -> BaseModel | None:
        """Extract from markdown."""
        schema = template.model_json_schema()
        response = self.client.get_json_response(markdown, str(schema))
        return template.model_validate(response)
    
    def consolidate_from_pydantic_models(
        self,
        raw_models: List[BaseModel],
        programmatic_model: BaseModel,
        template: Type[BaseModel]
    ) -> BaseModel | None:
        """Consolidate models."""
        return programmatic_model
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.client.close()
```

---

## Runtime Checking

All protocols are decorated with `@runtime_checkable`:

```python
from docling_graph.protocols import ExtractionBackendProtocol

# Check at runtime
if isinstance(my_backend, ExtractionBackendProtocol):
    print("Backend implements VLM protocol")
```

---

## Related APIs

- **[Custom Backends](../usage/advanced/custom-backends.md)** - Implementation guide
- **[Extractors](extractors.md)** - Extractor implementations
- **[LLM Clients](llm-clients.md)** - Client implementations

---

## See Also

- **[Python Protocols](https://peps.python.org/pep-0544/)** - PEP 544
- **[Type Hints](https://docs.python.org/3/library/typing.html)** - Python typing