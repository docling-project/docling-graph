# Exceptions

**Navigation:** [← Protocols](protocols.md) | [Next: Converters →](converters.md)

---

## Overview

Unified exception hierarchy for structured error handling in docling-graph.

**Module:** `docling_graph.exceptions`

All exceptions inherit from `DoclingGraphError` and provide structured error information including message, details, and cause.

---

## Exception Hierarchy

```
DoclingGraphError (base)
├── ConfigurationError      # Invalid configuration
├── ClientError            # LLM/API client errors
├── ExtractionError        # Document extraction failures
├── ValidationError        # Data validation failures
├── GraphError            # Graph operation failures
└── PipelineError         # Pipeline execution failures
```

---

## Base Exception

### DoclingGraphError

Base exception for all docling-graph errors.

```python
class DoclingGraphError(Exception):
    """Base exception for all docling-graph errors."""
    
    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ) -> None:
        """Initialize exception with structured information."""
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error description |
| `details` | `dict[str, Any]` | Additional context dictionary |
| `cause` | `Exception` or `None` | Underlying exception that caused this error |

**Methods:**

#### \_\_str\_\_()

```python
def __str__(self) -> str
```

Format exception with all available information.

**Returns:** Formatted error message with details and cause

#### \_\_repr\_\_()

```python
def __repr__(self) -> str
```

Detailed representation for debugging.

**Returns:** Detailed string representation

**Example:**

```python
from docling_graph.exceptions import DoclingGraphError

try:
    # Some operation
    raise DoclingGraphError(
        "Operation failed",
        details={"file": "doc.pdf", "stage": "extraction"},
        cause=ValueError("Invalid input")
    )
except DoclingGraphError as e:
    print(e.message)  # "Operation failed"
    print(e.details)  # {"file": "doc.pdf", "stage": "extraction"}
    print(e.cause)    # ValueError("Invalid input")
```

---

## Specific Exceptions

### ConfigurationError

Raised when configuration is invalid or missing.

```python
class ConfigurationError(DoclingGraphError):
    """Raised when configuration is invalid or missing."""
```

**Common Causes:**
- Missing required environment variables
- Invalid configuration file
- Unsupported model or provider
- Missing required parameters
- Invalid parameter combinations

**Example:**

```python
from docling_graph.exceptions import ConfigurationError

raise ConfigurationError(
    "API key not found",
    details={"env_var": "MISTRAL_API_KEY"}
)
```

**Usage:**

```python
from docling_graph import PipelineConfig
from docling_graph.exceptions import ConfigurationError

try:
    config = PipelineConfig(
        source="doc.pdf",
        template="templates.MyTemplate",
        backend="vlm",
        inference="remote"  # VLM doesn't support remote!
    )
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Details: {e.details}")
```

---

### ClientError

Raised when LLM client operation fails.

```python
class ClientError(DoclingGraphError):
    """Raised when LLM client operation fails."""
```

**Common Causes:**
- API authentication failure
- Network timeout
- Invalid API response
- Rate limit exceeded
- Model not available

**Example:**

```python
from docling_graph.exceptions import ClientError

raise ClientError(
    "API call failed",
    details={
        "provider": "mistral",
        "model": "mistral-small-latest",
        "status_code": 429
    },
    cause=requests.exceptions.HTTPError("Rate limit exceeded")
)
```

**Usage:**

```python
from docling_graph import run_pipeline
from docling_graph.exceptions import ClientError
import time

def process_with_retry(config, max_retries=3):
    for attempt in range(max_retries):
        try:
            run_pipeline(config)
            return
        except ClientError as e:
            if "rate limit" in str(e).lower():
                wait_time = 2 ** attempt
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

---

### ExtractionError

Raised when document extraction fails.

```python
class ExtractionError(DoclingGraphError):
    """Raised when document extraction fails."""
```

**Common Causes:**
- Document parsing failure
- Empty extraction result
- Invalid document format
- Extraction timeout
- Model inference failure

**Example:**

```python
from docling_graph.exceptions import ExtractionError

raise ExtractionError(
    "Failed to extract data from document",
    details={
        "source": "document.pdf",
        "template": "MyTemplate",
        "backend": "llm"
    },
    cause=ValueError("Empty response from model")
)
```

**Usage:**

```python
from docling_graph import run_pipeline
from docling_graph.exceptions import ExtractionError

try:
    run_pipeline({
        "source": "document.pdf",
        "template": "templates.MyTemplate"
    })
except ExtractionError as e:
    print(f"Extraction failed: {e.message}")
    
    # Try fallback strategy
    print("Trying with different backend...")
    run_pipeline({
        "source": "document.pdf",
        "template": "templates.MyTemplate",
        "backend": "vlm"  # Fallback to VLM
    })
```

---

### ValidationError

Raised when data validation fails.

```python
class ValidationError(DoclingGraphError):
    """Raised when data validation fails."""
```

**Common Causes:**
- Pydantic validation error
- Schema mismatch
- Invalid data structure
- Missing required fields
- Type mismatch

**Example:**

```python
from docling_graph.exceptions import ValidationError
from pydantic import ValidationError as PydanticValidationError

try:
    model = MyTemplate.model_validate(data)
except PydanticValidationError as e:
    raise ValidationError(
        "Data validation failed",
        details={"errors": e.errors()},
        cause=e
    )
```

**Usage:**

```python
from docling_graph import run_pipeline
from docling_graph.exceptions import ValidationError

try:
    run_pipeline({
        "source": "document.pdf",
        "template": "templates.StrictTemplate"
    })
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    
    # Check Pydantic errors
    if e.cause:
        for error in e.cause.errors():
            field = error['loc'][0]
            msg = error['msg']
            print(f"  - {field}: {msg}")
```

---

### GraphError

Raised when graph operation fails.

```python
class GraphError(DoclingGraphError):
    """Raised when graph operation fails."""
```

**Common Causes:**
- Invalid graph structure
- Node/edge creation failure
- Graph validation error
- Export failure
- Circular dependencies

**Example:**

```python
from docling_graph.exceptions import GraphError

raise GraphError(
    "Failed to create graph edge",
    details={
        "source_node": "node_1",
        "target_node": "node_2",
        "edge_type": "RELATES_TO"
    }
)
```

**Usage:**

```python
from docling_graph import run_pipeline
from docling_graph.exceptions import GraphError

try:
    run_pipeline({
        "source": "document.pdf",
        "template": "templates.MyTemplate",
        "export_format": "cypher"
    })
except GraphError as e:
    print(f"Graph error: {e.message}")
    
    # Try alternative export format
    run_pipeline({
        "source": "document.pdf",
        "template": "templates.MyTemplate",
        "export_format": "csv"  # Fallback to CSV
    })
```

---

### PipelineError

Raised when pipeline execution fails.

```python
class PipelineError(DoclingGraphError):
    """Raised when pipeline execution fails."""
```

**Common Causes:**
- Stage execution failure
- Resource initialization error
- Cleanup failure
- Unexpected pipeline state

**Example:**

```python
from docling_graph.exceptions import PipelineError

raise PipelineError(
    "Pipeline stage failed",
    details={
        "stage": "ExtractionStage",
        "source": "document.pdf"
    },
    cause=RuntimeError("Backend initialization failed")
)
```

**Usage:**

```python
from docling_graph import run_pipeline
from docling_graph.exceptions import PipelineError

try:
    run_pipeline({
        "source": "document.pdf",
        "template": "templates.MyTemplate"
    })
except PipelineError as e:
    print(f"Pipeline failed: {e.message}")
    print(f"Stage: {e.details.get('stage', 'unknown')}")
    
    if e.cause:
        print(f"Caused by: {e.cause}")
```

---

## Error Handling Patterns

### Catch Specific Exceptions

```python
from docling_graph import run_pipeline
from docling_graph.exceptions import (
    ConfigurationError,
    ClientError,
    ExtractionError
)

try:
    run_pipeline(config)
    
except ConfigurationError as e:
    print(f"Fix your configuration: {e.message}")
    
except ClientError as e:
    print(f"API error: {e.message}")
    # Maybe retry
    
except ExtractionError as e:
    print(f"Extraction failed: {e.message}")
    # Maybe try different backend
```

### Catch Base Exception

```python
from docling_graph import run_pipeline
from docling_graph.exceptions import DoclingGraphError

try:
    run_pipeline(config)
    
except DoclingGraphError as e:
    print(f"Error: {e.message}")
    print(f"Type: {type(e).__name__}")
    print(f"Details: {e.details}")
    
    if e.cause:
        print(f"Caused by: {e.cause}")
```

### Access Error Details

```python
from docling_graph.exceptions import ExtractionError

try:
    # Some operation
    pass
except ExtractionError as e:
    # Access structured information
    message = e.message
    details = e.details
    cause = e.cause
    
    # Log details
    print(f"Error: {message}")
    for key, value in details.items():
        print(f"  {key}: {value}")
```

### Re-raise with Context

```python
from docling_graph.exceptions import ExtractionError

def my_function():
    try:
        # Some operation
        result = extract_data()
    except ValueError as e:
        # Wrap in docling-graph exception
        raise ExtractionError(
            "Data extraction failed",
            details={"function": "my_function"},
            cause=e
        )
```

---

## Best Practices

### 1. Use Specific Exceptions

```python
# ✅ Good - Specific exception
from docling_graph.exceptions import ConfigurationError

if not api_key:
    raise ConfigurationError(
        "API key not found",
        details={"env_var": "MISTRAL_API_KEY"}
    )

# ❌ Avoid - Generic exception
if not api_key:
    raise Exception("API key not found")
```

### 2. Provide Details

```python
# ✅ Good - Detailed error
from docling_graph.exceptions import ExtractionError

raise ExtractionError(
    "Extraction failed",
    details={
        "source": source,
        "template": template.__name__,
        "backend": "llm",
        "stage": "markdown_extraction"
    }
)

# ❌ Avoid - No details
raise ExtractionError("Extraction failed")
```

### 3. Chain Exceptions

```python
# ✅ Good - Chain exceptions
from docling_graph.exceptions import ClientError

try:
    response = api.call()
except requests.HTTPError as e:
    raise ClientError(
        "API call failed",
        details={"status": e.response.status_code},
        cause=e  # Preserve original exception
    )

# ❌ Avoid - Lose original exception
try:
    response = api.call()
except requests.HTTPError:
    raise ClientError("API call failed")
```

### 4. Log Before Raising

```python
# ✅ Good - Log then raise
import logging
from docling_graph.exceptions import ExtractionError

logger = logging.getLogger(__name__)

try:
    result = extract()
except Exception as e:
    logger.error(f"Extraction failed: {e}", exc_info=True)
    raise ExtractionError(
        "Extraction failed",
        cause=e
    )
```

---

## Related APIs

- **[Error Handling Guide](../10-advanced/error-handling.md)** - Error handling patterns
- **[Pipeline API](pipeline.md)** - Pipeline exceptions
- **[Configuration API](config.md)** - Configuration validation

---

## See Also

- **[Python Exceptions](https://docs.python.org/3/tutorial/errors.html)** - Python error handling
- **[Pydantic Validation](https://docs.pydantic.dev/latest/concepts/validators/)** - Pydantic errors

---

**Navigation:** [← Protocols](protocols.md) | [Next: Converters →](converters.md)