# Advanced Topics

**Navigation:** [← Examples](../09-examples/index.md) | [Custom Backends →](custom-backends.md)

---

## Overview

This section covers advanced topics for extending and optimizing docling-graph. These guides are for users who need to:

- Create custom extraction backends
- Build custom exporters
- Add pipeline stages
- Optimize performance
- Handle errors gracefully
- Test templates and pipelines

---

## Topics

### 🔧 Extensibility

**[Custom Backends](custom-backends.md)**  
Create custom extraction backends for specialized models or APIs.

- Implement backend protocols
- VLM backend example
- LLM backend example
- Integration with pipeline

**[Custom Exporters](custom-exporters.md)**  
Build custom exporters for specialized output formats.

- Implement exporter protocol
- Graph data access
- Custom format generation
- Registration and usage

**[Custom Stages](custom-stages.md)**  
Add custom stages to the pipeline for specialized processing.

- Pipeline stage protocol
- Stage implementation
- Context management
- Error handling

---

### + Optimization

**[Performance Tuning](performance-tuning.md)**  
Optimize extraction speed and resource usage.

- Model selection strategies
- Batch size optimization
- Memory management
- GPU utilization
- Caching strategies

---

### 🛡️ Reliability

**[Error Handling](error-handling.md)**  
Handle errors gracefully and implement retry logic.

- Exception hierarchy
- Error recovery strategies
- Logging and debugging
- Retry mechanisms

**[Testing](testing.md)**  
Test templates, backends, and pipelines.

- Template validation
- Mock backends
- Integration testing
- CI/CD integration

---

## Prerequisites

Before diving into advanced topics, ensure you understand:

1. **[Schema Definition](../03-schema-definition/index.md)** - Pydantic templates
2. **[Pipeline Configuration](../04-pipeline-configuration/index.md)** - Configuration options
3. **[Extraction Process](../05-extraction-process/index.md)** - How extraction works
4. **[Python API](../08-api/index.md)** - Programmatic usage

---

## When to Use Advanced Features

### Custom Backends

Use when:
<br>✅ You have a specialized model not supported by default
<br>✅ You need to integrate with a proprietary API
<br>✅ You want to implement custom preprocessing
<br>✅ You need fine-grained control over extraction

Don't use when:
<br>❌ Default backends meet your needs
<br>❌ You're just starting with docling-graph
<br>❌ You don't need custom logic

### Custom Exporters

Use when:
<br>✅ You need a specialized output format
<br>✅ You're integrating with a specific database
<br>✅ You need custom data transformations
<br>✅ Default formats don't meet requirements

Don't use when:
<br>❌ CSV, Cypher, or JSON formats work
<br>❌ You can post-process existing exports
<br>❌ You're prototyping

### Custom Stages

Use when:
<br>✅ You need custom preprocessing
<br>✅ You want to add validation steps
<br>✅ You need custom post-processing
<br>✅ You're building a specialized pipeline

Don't use when:
<br>❌ Default pipeline stages suffice
<br>❌ You can achieve goals with configuration
<br>❌ You're learning the system

---

## Architecture

### Extension Points

![Extension Points](../assets/flowcharts/img/extension_points.png){width=300px}

**Extension Points:**
- **Custom Backends** (blue): Replace extraction logic
- **Custom Exporters** (blue): Replace export logic
- **Custom Stages** (yellow): Add processing steps

---

## Code Organization

### Project Structure for Extensions

```
my_project/
├── templates/              # Pydantic templates
│   └── my_template.py
├── backends/               # Custom backends
│   ├── __init__.py
│   └── my_backend.py
├── exporters/              # Custom exporters
│   ├── __init__.py
│   └── my_exporter.py
├── stages/                 # Custom stages
│   ├── __init__.py
│   └── my_stage.py
├── tests/                  # Tests
│   ├── test_backend.py
│   ├── test_exporter.py
│   └── test_stage.py
└── main.py                 # Entry point
```

---

## Development Workflow

### 1. Design

```python
# Define interface
from docling_graph.protocols import TextExtractionBackendProtocol

class MyBackend(TextExtractionBackendProtocol):
    """Custom backend implementation."""
    pass
```

### 2. Implement

```python
# Implement methods
def extract_from_markdown(self, markdown: str, template, context="", is_partial=False):
    """Extract structured data."""
    # Your logic here
    pass
```

### 3. Test

```python
# Write tests
def test_my_backend():
    backend = MyBackend()
    result = backend.extract_from_markdown("test", MyTemplate)
    assert result is not None
```

### 4. Integrate

```python
# Use in pipeline
from docling_graph import PipelineConfig
from my_backends import MyBackend

config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    # Custom backend integration
)
```

---

## Best Practices

### 1. Follow Protocols

```python
# ✅ Good - Implement protocol
from docling_graph.protocols import TextExtractionBackendProtocol

class MyBackend(TextExtractionBackendProtocol):
    def extract_from_markdown(self, ...): ...
    def consolidate_from_pydantic_models(self, ...): ...
    def cleanup(self): ...

# ❌ Avoid - Custom interface
class MyBackend:
    def my_custom_method(self, ...): ...
```

### 2. Handle Errors

```python
# ✅ Good - Use docling-graph exceptions
from docling_graph.exceptions import ExtractionError

def extract(self, ...):
    try:
        result = self._process()
        return result
    except Exception as e:
        raise ExtractionError(
            "Extraction failed",
            details={"source": source},
            cause=e
        )

# ❌ Avoid - Generic exceptions
def extract(self, ...):
    raise Exception("Something went wrong")
```

### 3. Write Tests

```python
# ✅ Good - Comprehensive tests
def test_backend_success():
    """Test successful extraction."""
    pass

def test_backend_failure():
    """Test error handling."""
    pass

def test_backend_cleanup():
    """Test resource cleanup."""
    pass

# ❌ Avoid - No tests
# (No tests written)
```

### 4. Document Code

```python
# ✅ Good - Clear documentation
class MyBackend:
    """
    Custom backend for specialized extraction.
    
    This backend uses a proprietary model to extract
    structured data from documents.
    
    Args:
        api_key: API key for the service
        model: Model name to use
        
    Example:
        >>> backend = MyBackend(api_key="key", model="model-v1")
        >>> result = backend.extract_from_markdown(text, Template)
    """
    pass

# ❌ Avoid - No documentation
class MyBackend:
    pass
```

---

## Performance Considerations

### Memory Management

```python
# ✅ Good - Clean up resources
class MyBackend:
    def cleanup(self):
        """Release resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'client'):
            self.client.close()

# ❌ Avoid - Memory leaks
class MyBackend:
    def cleanup(self):
        pass  # Resources not released
```

### Batch Processing

```python
# ✅ Good - Process in batches
def process_documents(docs):
    batch_size = 10
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        process_batch(batch)

# ❌ Avoid - Process all at once
def process_documents(docs):
    process_all(docs)  # May run out of memory
```

---

## Security Considerations

### API Keys

```python
# ✅ Good - Use environment variables
import os

api_key = os.getenv("MY_API_KEY")
if not api_key:
    raise ValueError("MY_API_KEY not set")

# ❌ Avoid - Hardcoded keys
api_key = "sk-1234567890"  # Never do this!
```

### Input Validation

```python
# ✅ Good - Validate inputs
def extract(self, markdown: str, template):
    if not markdown:
        raise ValueError("Markdown cannot be empty")
    if not template:
        raise ValueError("Template is required")
    # Process...

# ❌ Avoid - No validation
def extract(self, markdown, template):
    # Process without checks
    pass
```

---

## Next Steps

Choose a topic based on your needs:

1. **[Custom Backends →](custom-backends.md)** - Extend extraction capabilities
2. **[Custom Exporters →](custom-exporters.md)** - Create custom output formats
3. **[Custom Stages →](custom-stages.md)** - Add pipeline stages
4. **[Performance Tuning →](performance-tuning.md)** - Optimize performance
5. **[Error Handling →](error-handling.md)** - Handle errors gracefully
6. **[Testing →](testing.md)** - Test your extensions

---

## Related Documentation

- **[Protocols](../11-reference/protocols.md)** - Protocol definitions
- **[Exceptions](../11-reference/exceptions.md)** - Exception hierarchy
- **[Pipeline Architecture](../01-introduction/architecture.md)** - System design

---

**Navigation:** [← Examples](../09-examples/index.md) | [Custom Backends →](custom-backends.md)