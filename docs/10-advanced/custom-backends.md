# Custom Backends

**Navigation:** [← Advanced Topics](index.md) | [Next: Custom Exporters →](custom-exporters.md)

---

## Overview

Create custom extraction backends to integrate specialized models, APIs, or processing logic into the docling-graph pipeline.

**What You'll Learn:**
- Backend protocol implementation
- VLM backend example
- LLM backend example
- Integration with pipeline
- Testing strategies

**Prerequisites:**
- Understanding of [Extraction Process](../05-extraction-process/index.md)
- Familiarity with [Python API](../08-api/index.md)
- Knowledge of Pydantic models

---

## Backend Types

### VLM Backend (Vision-Language Model)

Processes documents directly without markdown conversion.

**Protocol:** `ExtractionBackendProtocol`

```python
from docling_graph.protocols import ExtractionBackendProtocol

class MyVLMBackend(ExtractionBackendProtocol):
    def extract_from_document(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        """Extract from document directly."""
        pass
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
```

### LLM Backend (Language Model)

Processes markdown/text content.

**Protocol:** `TextExtractionBackendProtocol`

```python
from docling_graph.protocols import TextExtractionBackendProtocol

class MyLLMBackend(TextExtractionBackendProtocol):
    client: Any  # LLM client instance
    
    def extract_from_markdown(
        self, 
        markdown: str, 
        template: Type[BaseModel],
        context: str = "document",
        is_partial: bool = False
    ) -> BaseModel | None:
        """Extract from markdown."""
        pass
    
    def consolidate_from_pydantic_models(
        self,
        raw_models: List[BaseModel],
        programmatic_model: BaseModel,
        template: Type[BaseModel]
    ) -> BaseModel | None:
        """Consolidate multiple models."""
        pass
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
```

---

## Complete VLM Backend Example

### Implementation

```python
"""
Custom VLM backend using a hypothetical vision model.
"""

from typing import Any, List, Type
from pathlib import Path
from pydantic import BaseModel
from docling_graph.protocols import ExtractionBackendProtocol
from docling_graph.exceptions import ExtractionError, ClientError

class CustomVLMBackend(ExtractionBackendProtocol):
    """
    Custom VLM backend for specialized vision model.
    
    Args:
        model_name: Name of the vision model
        api_key: API key for the service
        base_url: Base URL for API (optional)
    """
    
    def __init__(
        self,
        model_name: str = "vision-model-v1",
        api_key: str | None = None,
        base_url: str | None = None
    ):
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key()
        self.base_url = base_url or "https://api.example.com/v1"
        
        # Initialize client
        self.client = self._initialize_client()
    
    def _get_api_key(self) -> str:
        """Get API key from environment."""
        import os
        api_key = os.getenv("CUSTOM_VLM_API_KEY")
        if not api_key:
            raise ClientError(
                "API key not found",
                details={"env_var": "CUSTOM_VLM_API_KEY"}
            )
        return api_key
    
    def _initialize_client(self) -> Any:
        """Initialize the vision model client."""
        try:
            # Your client initialization here
            from my_vision_sdk import VisionClient
            return VisionClient(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name
            )
        except Exception as e:
            raise ClientError(
                "Failed to initialize client",
                details={"model": self.model_name},
                cause=e
            )
    
    def extract_from_document(
        self, 
        source: str, 
        template: Type[BaseModel]
    ) -> List[BaseModel]:
        """
        Extract structured data from document.
        
        Args:
            source: Path to document (image or PDF)
            template: Pydantic model template
            
        Returns:
            List of extracted model instances
            
        Raises:
            ExtractionError: If extraction fails
        """
        try:
            # Validate source
            source_path = Path(source)
            if not source_path.exists():
                raise ExtractionError(
                    "Source file not found",
                    details={"source": source}
                )
            
            # Get schema
            schema = template.model_json_schema()
            
            # Call vision model
            response = self.client.extract(
                image_path=str(source_path),
                schema=schema
            )
            
            # Parse response
            extracted_data = response.get("data", {})
            
            # Validate with Pydantic
            model_instance = template.model_validate(extracted_data)
            
            return [model_instance]
            
        except Exception as e:
            raise ExtractionError(
                "Document extraction failed",
                details={
                    "source": source,
                    "template": template.__name__
                },
                cause=e
            )
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
            except Exception:
                pass  # Best effort cleanup
```

### Usage

```python
"""Use custom VLM backend."""

from docling_graph import PipelineConfig
from my_backends import CustomVLMBackend

# Create backend instance
backend = CustomVLMBackend(
    model_name="vision-model-v1",
    api_key="your_api_key"
)

# Note: Direct backend integration requires custom pipeline code
# For now, use with extraction strategies directly
from docling_graph.core.extractors.strategies import OneToOne

extractor = OneToOne(backend=backend)
results = extractor.extract(
    source="document.pdf",
    template=MyTemplate
)
```

---

## Complete LLM Backend Example

### Implementation

```python
"""
Custom LLM backend using a hypothetical language model.
"""

from typing import Any, Dict, List, Type
from pydantic import BaseModel
from docling_graph.protocols import TextExtractionBackendProtocol, LLMClientProtocol
from docling_graph.exceptions import ExtractionError, ClientError

class CustomLLMClient(LLMClientProtocol):
    """Custom LLM client implementation."""
    
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self._context_limit = 8000  # Token limit
    
    @property
    def context_limit(self) -> int:
        """Return context limit in tokens."""
        return self._context_limit
    
    def get_json_response(
        self, 
        prompt: str | Dict[str, str], 
        schema_json: str
    ) -> Dict[str, Any]:
        """
        Execute LLM call and return parsed JSON.
        
        Args:
            prompt: System/user prompt or legacy string
            schema_json: Pydantic schema as JSON string
            
        Returns:
            Parsed JSON dictionary
        """
        try:
            # Handle both formats
            if isinstance(prompt, dict):
                system_prompt = prompt.get("system", "")
                user_prompt = prompt.get("user", "")
            else:
                system_prompt = ""
                user_prompt = prompt
            
            # Call your LLM API
            from my_llm_sdk import LLMClient
            client = LLMClient(api_key=self.api_key)
            
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                schema=schema_json
            )
            
            # Parse JSON response
            import json
            return json.loads(response.content)
            
        except Exception as e:
            raise ClientError(
                "LLM call failed",
                details={"model": self.model},
                cause=e
            )


class CustomLLMBackend(TextExtractionBackendProtocol):
    """
    Custom LLM backend for text extraction.
    
    Args:
        model: Model name
        api_key: API key
    """
    
    def __init__(self, model: str = "custom-llm-v1", api_key: str | None = None):
        import os
        self.model = model
        self.api_key = api_key or os.getenv("CUSTOM_LLM_API_KEY")
        
        if not self.api_key:
            raise ClientError(
                "API key not found",
                details={"env_var": "CUSTOM_LLM_API_KEY"}
            )
        
        # Initialize client
        self.client = CustomLLMClient(
            model=self.model,
            api_key=self.api_key
        )
    
    def extract_from_markdown(
        self,
        markdown: str,
        template: Type[BaseModel],
        context: str = "document",
        is_partial: bool = False
    ) -> BaseModel | None:
        """
        Extract structured data from markdown.
        
        Args:
            markdown: Markdown content
            template: Pydantic model template
            context: Context description
            is_partial: Whether this is a partial extraction
            
        Returns:
            Extracted model instance or None
        """
        try:
            # Build prompt
            schema_json = template.model_json_schema()
            
            system_prompt = (
                "You are a data extraction expert. "
                "Extract structured information from the provided text "
                "according to the given schema."
            )
            
            user_prompt = f"""
Extract information from this {context}:

{markdown}

Return a JSON object matching this schema:
{schema_json}
"""
            
            # Call LLM
            response = self.client.get_json_response(
                prompt={"system": system_prompt, "user": user_prompt},
                schema_json=str(schema_json)
            )
            
            # Validate with Pydantic
            model_instance = template.model_validate(response)
            return model_instance
            
        except Exception as e:
            raise ExtractionError(
                "Markdown extraction failed",
                details={
                    "context": context,
                    "template": template.__name__
                },
                cause=e
            )
    
    def consolidate_from_pydantic_models(
        self,
        raw_models: List[BaseModel],
        programmatic_model: BaseModel,
        template: Type[BaseModel]
    ) -> BaseModel | None:
        """
        Consolidate multiple models using LLM.
        
        Args:
            raw_models: List of extracted models
            programmatic_model: Programmatically merged model
            template: Target template
            
        Returns:
            Consolidated model instance
        """
        try:
            # Convert models to JSON
            models_json = [m.model_dump() for m in raw_models]
            programmatic_json = programmatic_model.model_dump()
            
            system_prompt = (
                "You are a data consolidation expert. "
                "Merge multiple extractions into a single coherent result."
            )
            
            user_prompt = f"""
Consolidate these extractions:

Raw extractions:
{models_json}

Programmatic merge:
{programmatic_json}

Return the best consolidated result as JSON.
"""
            
            schema_json = template.model_json_schema()
            
            response = self.client.get_json_response(
                prompt={"system": system_prompt, "user": user_prompt},
                schema_json=str(schema_json)
            )
            
            return template.model_validate(response)
            
        except Exception as e:
            raise ExtractionError(
                "Consolidation failed",
                details={"num_models": len(raw_models)},
                cause=e
            )
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Close any open connections
        pass
```

### Usage

```python
"""Use custom LLM backend."""

from my_backends import CustomLLMBackend
from docling_graph.core.extractors.strategies import ManyToOne

# Create backend
backend = CustomLLMBackend(
    model="custom-llm-v1",
    api_key="your_api_key"
)

# Use with extractor
extractor = ManyToOne(backend=backend)
results = extractor.extract(
    source="document.pdf",
    template=MyTemplate
)

# Clean up
backend.cleanup()
```

---

## Testing Custom Backends

### Unit Tests

```python
"""Test custom backend."""

import pytest
from pydantic import BaseModel, Field
from my_backends import CustomLLMBackend

class TestTemplate(BaseModel):
    """Simple test template."""
    name: str = Field(..., description="Name")
    value: int = Field(..., description="Value")

def test_backend_initialization():
    """Test backend can be initialized."""
    backend = CustomLLMBackend(
        model="test-model",
        api_key="test-key"
    )
    assert backend.model == "test-model"
    assert backend.client is not None

def test_extract_from_markdown():
    """Test markdown extraction."""
    backend = CustomLLMBackend(
        model="test-model",
        api_key="test-key"
    )
    
    markdown = "Name: John, Value: 42"
    result = backend.extract_from_markdown(
        markdown=markdown,
        template=TestTemplate
    )
    
    assert result is not None
    assert isinstance(result, TestTemplate)
    assert result.name == "John"
    assert result.value == 42

def test_cleanup():
    """Test cleanup doesn't raise errors."""
    backend = CustomLLMBackend(
        model="test-model",
        api_key="test-key"
    )
    backend.cleanup()  # Should not raise
```

### Integration Tests

```python
"""Integration test with pipeline."""

from docling_graph.core.extractors.strategies import ManyToOne
from my_backends import CustomLLMBackend

def test_backend_with_extractor():
    """Test backend works with extractor."""
    backend = CustomLLMBackend(
        model="test-model",
        api_key="test-key"
    )
    
    extractor = ManyToOne(backend=backend)
    
    results = extractor.extract(
        source="test_document.pdf",
        template=TestTemplate
    )
    
    assert len(results) > 0
    assert all(isinstance(r, TestTemplate) for r in results)
    
    backend.cleanup()
```

---

## Best Practices

### 1. Implement All Protocol Methods

```python
# ✅ Good - Complete implementation
class MyBackend(TextExtractionBackendProtocol):
    client: Any
    
    def extract_from_markdown(self, ...): ...
    def consolidate_from_pydantic_models(self, ...): ...
    def cleanup(self): ...

# ❌ Avoid - Missing methods
class MyBackend:
    def extract_from_markdown(self, ...): ...
    # Missing other methods!
```

### 2. Use Structured Exceptions

```python
# ✅ Good - Structured errors
from docling_graph.exceptions import ExtractionError, ClientError

def extract(self, ...):
    try:
        result = self._process()
        return result
    except APIError as e:
        raise ClientError("API call failed", cause=e)
    except ValidationError as e:
        raise ExtractionError("Validation failed", cause=e)

# ❌ Avoid - Generic exceptions
def extract(self, ...):
    raise Exception("Something went wrong")
```

### 3. Clean Up Resources

```python
# ✅ Good - Proper cleanup
class MyBackend:
    def __init__(self):
        self.client = initialize_client()
        self.model = load_model()
    
    def cleanup(self):
        if hasattr(self, 'client'):
            self.client.close()
        if hasattr(self, 'model'):
            del self.model
            import gc
            gc.collect()

# ❌ Avoid - No cleanup
class MyBackend:
    def cleanup(self):
        pass  # Resources leak!
```

### 4. Validate Inputs

```python
# ✅ Good - Input validation
def extract_from_markdown(self, markdown: str, template, ...):
    if not markdown or not markdown.strip():
        raise ValueError("Markdown cannot be empty")
    if not template:
        raise ValueError("Template is required")
    # Process...

# ❌ Avoid - No validation
def extract_from_markdown(self, markdown, template, ...):
    # Process without checks
    pass
```

---

## Troubleshooting

### Issue: Protocol Not Recognized

**Problem:** Backend not recognized by pipeline

**Solution:**
```python
# Ensure you implement the correct protocol
from docling_graph.protocols import TextExtractionBackendProtocol

class MyBackend(TextExtractionBackendProtocol):
    # Must have 'client' attribute for LLM backends
    client: Any
    
    # Must implement all required methods
    def extract_from_markdown(self, ...): ...
    def consolidate_from_pydantic_models(self, ...): ...
    def cleanup(self): ...
```

### Issue: Memory Leaks

**Problem:** Memory usage grows over time

**Solution:**
```python
# Implement proper cleanup
def cleanup(self):
    # Close connections
    if hasattr(self, 'client'):
        self.client.close()
    
    # Delete large objects
    if hasattr(self, 'model'):
        del self.model
    
    # Force garbage collection
    import gc
    gc.collect()
```

### Issue: API Rate Limits

**Problem:** API calls fail due to rate limits

**Solution:**
```python
import time
from docling_graph.exceptions import ClientError

def _call_api_with_retry(self, *args, **kwargs):
    """Call API with exponential backoff."""
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            return self.client.call(*args, **kwargs)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise ClientError("Rate limit exceeded", cause=e)
            
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
```

---

## Next Steps

1. **[Custom Exporters →](custom-exporters.md)** - Create custom output formats
2. **[Testing →](testing.md)** - Test your backend
3. **[Error Handling →](error-handling.md)** - Handle errors gracefully

---

## Related Documentation

- **[Protocols Reference](../11-reference/protocols.md)** - Protocol definitions
- **[Exceptions Reference](../11-reference/exceptions.md)** - Exception hierarchy
- **[Extraction Process](../05-extraction-process/index.md)** - How extraction works

---

**Navigation:** [← Advanced Topics](index.md) | [Next: Custom Exporters →](custom-exporters.md)