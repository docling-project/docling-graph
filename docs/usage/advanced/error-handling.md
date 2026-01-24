# Error Handling


## Overview

Handle errors gracefully in docling-graph pipelines with structured exception handling, retry logic, and debugging strategies.

**What You'll Learn:**
- Exception hierarchy
- Error recovery strategies
- Retry mechanisms
- Logging and debugging
- Validation errors
- Best practices

**Prerequisites:**
- Understanding of [Pipeline Architecture](../../introduction/architecture.md)
- Familiarity with [Python API](../api/index.md)
- Basic Python exception handling

---

## Exception Hierarchy

Docling-graph uses a structured exception hierarchy:

```python
DoclingGraphError (base)
├── ConfigurationError      # Invalid configuration
├── ClientError            # LLM/API client errors
├── ExtractionError        # Document extraction failures
├── ValidationError        # Data validation failures
├── GraphError            # Graph operation failures
└── PipelineError         # Pipeline execution failures
```

### Import Exceptions

```python
from docling_graph.exceptions import (
    DoclingGraphError,
    ConfigurationError,
    ClientError,
    ExtractionError,
    ValidationError,
    GraphError,
    PipelineError
)
```

---

## Common Error Scenarios

### 1. Configuration Errors

```python
"""Handle configuration errors."""

from docling_graph import PipelineConfig
from docling_graph.exceptions import ConfigurationError

try:
    config = PipelineConfig(
        source="document.pdf",
        template="templates.MyTemplate",
        backend="vlm",
        inference="remote"  # VLM doesn't support remote!
    )
    config.run()
    
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Details: {e.details}")
    # Fix: Use local inference with VLM
    config = PipelineConfig(
        source="document.pdf",
        template="templates.MyTemplate",
        backend="vlm",
        inference="local"  # Corrected
    )
    config.run()
```

### 2. Client Errors (API)

```python
"""Handle API client errors."""

from docling_graph import PipelineConfig
from docling_graph.exceptions import ClientError
import time

def process_with_retry(source: str, max_retries: int = 3):
    """Process with retry on client errors."""
    
    for attempt in range(max_retries):
        try:
            config = PipelineConfig(
                source=source,
                template="templates.MyTemplate",
                backend="llm",
                inference="remote"
            )
            config.run()
            print("✓ Processing successful")
            return
            
        except ClientError as e:
            print(f"Attempt {attempt + 1} failed: {e.message}")
            
            if "rate limit" in str(e).lower():
                # Rate limit - wait and retry
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                
            elif "authentication" in str(e).lower():
                # Auth error - don't retry
                print("Authentication failed. Check API key.")
                raise
                
            elif attempt == max_retries - 1:
                # Last attempt failed
                print("Max retries reached")
                raise
            else:
                # Other error - retry
                time.sleep(1)

# Usage
process_with_retry("document.pdf")
```

### 3. Extraction Errors

```python
"""Handle extraction errors."""

from docling_graph import PipelineConfig
from docling_graph.exceptions import ExtractionError

def process_with_fallback(source: str):
    """Process with fallback strategy."""
    
    # Try VLM first (faster)
    try:
        print("Trying VLM extraction...")
        config = PipelineConfig(
            source=source,
            template="templates.MyTemplate",
            backend="vlm",
            inference="local"
        )
        config.run()
        print("✓ VLM extraction successful")
        return
        
    except ExtractionError as e:
        print(f"VLM failed: {e.message}")
        print("Falling back to LLM...")
    
    # Fallback to LLM
    try:
        config = PipelineConfig(
            source=source,
            template="templates.MyTemplate",
            backend="llm",
            inference="local"
        )
        config.run()
        print("✓ LLM extraction successful")
        
    except ExtractionError as e:
        print(f"Both methods failed: {e.message}")
        print(f"Details: {e.details}")
        raise

# Usage
process_with_fallback("document.pdf")
```

### 4. Validation Errors

```python
"""Handle validation errors."""

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from docling_graph import PipelineConfig
from docling_graph.exceptions import ValidationError

class StrictTemplate(BaseModel):
    """Template with strict validation."""
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=150)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')

def process_with_validation_handling(source: str):
    """Process with validation error handling."""
    
    try:
        config = PipelineConfig(
            source=source,
            template="templates.StrictTemplate"
        )
        config.run()
        
    except ValidationError as e:
        print(f"Validation failed: {e.message}")
        
        # Check if it's a Pydantic validation error
        if e.cause and isinstance(e.cause, PydanticValidationError):
            print("\nValidation errors:")
            for error in e.cause.errors():
                field = error['loc'][0]
                msg = error['msg']
                print(f"  - {field}: {msg}")
        
        # Option 1: Use more lenient template
        print("\nRetrying with lenient template...")
        config = PipelineConfig(
            source=source,
            template="templates.LenientTemplate"
        )
        config.run()

# Usage
process_with_validation_handling("document.pdf")
```

### 5. Graph Errors

```python
"""Handle graph construction errors."""

from docling_graph import PipelineConfig
from docling_graph.exceptions import GraphError

def process_with_graph_validation(source: str):
    """Process with graph validation."""
    
    try:
        config = PipelineConfig(
            source=source,
            template="templates.MyTemplate",
            export_format="cypher"
        )
        config.run()
        
    except GraphError as e:
        print(f"Graph error: {e.message}")
        print(f"Details: {e.details}")
        
        # Try alternative export format
        print("Trying CSV export instead...")
        config = PipelineConfig(
            source=source,
            template="templates.MyTemplate",
            export_format="csv"  # Fallback format
        )
        config.run()

# Usage
process_with_graph_validation("document.pdf")
```

---

## Retry Strategies

### Exponential Backoff

```python
"""Implement exponential backoff for retries."""

import time
from typing import Callable, Any
from docling_graph.exceptions import ClientError

def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
) -> Any:
    """
    Retry function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay
        
    Returns:
        Function result
        
    Raises:
        Exception from last attempt
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
            
        except ClientError as e:
            last_exception = e
            
            if attempt == max_retries - 1:
                # Last attempt
                break
            
            # Calculate delay
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            
            print(f"Attempt {attempt + 1} failed. Retrying in {delay:.1f}s...")
            time.sleep(delay)
    
    # All retries failed
    raise last_exception

# Usage
def process_document():
    config = PipelineConfig(
        source="document.pdf",
        template="templates.MyTemplate",
        backend="llm",
        inference="remote"
    )
    config.run()

retry_with_backoff(process_document, max_retries=3)
```

### Conditional Retry

```python
"""Retry only for specific errors."""

from docling_graph.exceptions import ClientError, ConfigurationError

def should_retry(exception: Exception) -> bool:
    """Determine if error is retryable."""
    
    # Don't retry configuration errors
    if isinstance(exception, ConfigurationError):
        return False
    
    # Retry client errors
    if isinstance(exception, ClientError):
        error_msg = str(exception).lower()
        
        # Don't retry auth errors
        if "authentication" in error_msg or "unauthorized" in error_msg:
            return False
        
        # Retry rate limits and timeouts
        if "rate limit" in error_msg or "timeout" in error_msg:
            return True
    
    # Default: don't retry
    return False

def process_with_conditional_retry(source: str, max_retries: int = 3):
    """Process with conditional retry."""
    
    for attempt in range(max_retries):
        try:
            config = PipelineConfig(
                source=source,
                template="templates.MyTemplate"
            )
            config.run()
            return
            
        except Exception as e:
            if not should_retry(e) or attempt == max_retries - 1:
                raise
            
            print(f"Retryable error. Attempt {attempt + 2}...")
            time.sleep(2 ** attempt)
```

---

## Logging and Debugging

### Enable Detailed Logging

```python
"""Configure logging for debugging."""

import logging
from docling_graph import PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('docling_graph')

# Run pipeline with logging
try:
    config = PipelineConfig(
        source="document.pdf",
        template="templates.MyTemplate"
    )
    config.run()
    
except Exception as e:
    logger.error(f"Pipeline failed: {e}", exc_info=True)
    raise
```

### Debug Mode

```python
"""Run pipeline in debug mode."""

from docling_graph import PipelineConfig
from docling_graph.exceptions import DoclingGraphError

def debug_pipeline(source: str):
    """Run pipeline with detailed error information."""
    
    try:
        config = PipelineConfig(
            source=source,
            template="templates.MyTemplate"
        )
        config.run()
        
    except DoclingGraphError as e:
        print("\n" + "="*60)
        print("ERROR DETAILS")
        print("="*60)
        print(f"Type: {type(e).__name__}")
        print(f"Message: {e.message}")
        
        if e.details:
            print("\nDetails:")
            for key, value in e.details.items():
                print(f"  {key}: {value}")
        
        if e.cause:
            print(f"\nCaused by: {type(e.cause).__name__}")
            print(f"  {e.cause}")
        
        print("="*60)
        raise

# Usage
debug_pipeline("document.pdf")
```

---

## Error Recovery Patterns

### Graceful Degradation

```python
"""Degrade gracefully on errors."""

from docling_graph import PipelineConfig
from docling_graph.exceptions import ExtractionError

def process_with_degradation(source: str):
    """Process with graceful degradation."""
    
    results = {
        "success": False,
        "method": None,
        "output_dir": None
    }
    
    # Try best method first
    methods = [
        ("VLM Local", {"backend": "vlm", "inference": "local"}),
        ("LLM Local", {"backend": "llm", "inference": "local"}),
        ("LLM Remote", {"backend": "llm", "inference": "remote"})
    ]
    
    for method_name, config_overrides in methods:
        try:
            print(f"Trying {method_name}...")
            
            config = PipelineConfig(
                source=source,
                template="templates.MyTemplate",
                **config_overrides
            )
            config.run()
            
            results["success"] = True
            results["method"] = method_name
            results["output_dir"] = config.output_dir
            
            print(f"✓ Success with {method_name}")
            break
            
        except ExtractionError as e:
            print(f"✗ {method_name} failed: {e.message}")
            continue
    
    if not results["success"]:
        print("❌ All methods failed")
    
    return results
```

### Partial Success Handling

```python
"""Handle partial extraction success."""

from pathlib import Path
import json
from docling_graph import PipelineConfig

def process_with_partial_success(source: str):
    """Process and handle partial results."""
    
    try:
        config = PipelineConfig(
            source=source,
            template="templates.MyTemplate",
            output_dir="outputs"
        )
        config.run()
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        
        # Check if partial results exist
        output_dir = Path("outputs")
        if output_dir.exists():
            # Check for extracted data
            nodes_file = output_dir / "nodes.csv"
            if nodes_file.exists():
                print("✓ Partial results available")
                print(f"  Nodes: {nodes_file}")
                
                # Use partial results
                return {"status": "partial", "output_dir": output_dir}
        
        return {"status": "failed", "output_dir": None}
```

---

## Validation Strategies

### Pre-Validation

```python
"""Validate before processing."""

from pathlib import Path
from docling_graph import PipelineConfig
from docling_graph.exceptions import ConfigurationError

def validate_and_process(source: str, template: str):
    """Validate configuration before processing."""
    
    # Validate source
    source_path = Path(source)
    if not source_path.exists():
        raise ConfigurationError(
            "Source file not found",
            details={"source": source}
        )
    
    # Validate template
    try:
        # Try to import template
        module_path, class_name = template.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        template_class = getattr(module, class_name)
    except Exception as e:
        raise ConfigurationError(
            "Invalid template",
            details={"template": template},
            cause=e
        )
    
    # Validate file size
    size_mb = source_path.stat().st_size / (1024 * 1024)
    if size_mb > 100:
        print(f"⚠️  Large file: {size_mb:.1f}MB")
    
    # Process
    config = PipelineConfig(
        source=source,
        template=template
    )
    config.run()
```

---

## Best Practices

### 1. Use Specific Exceptions

```python
# ✅ Good - Catch specific exceptions
try:
    config.run()
except ClientError as e:
    # Handle API errors
    pass
except ExtractionError as e:
    # Handle extraction errors
    pass

# ❌ Avoid - Catch all exceptions
try:
    config.run()
except Exception:
    pass  # What went wrong?
```

### 2. Provide Context

```python
# ✅ Good - Detailed error context
from docling_graph.exceptions import ExtractionError

try:
    result = extract_data(source)
except Exception as e:
    raise ExtractionError(
        "Failed to extract data",
        details={
            "source": source,
            "template": template.__name__,
            "stage": "extraction"
        },
        cause=e
    )

# ❌ Avoid - Generic errors
try:
    result = extract_data(source)
except Exception as e:
    raise Exception("Extraction failed")
```

### 3. Log Before Raising

```python
# ✅ Good - Log then raise
import logging
logger = logging.getLogger(__name__)

try:
    config.run()
except ExtractionError as e:
    logger.error(f"Extraction failed: {e}", exc_info=True)
    raise

# ❌ Avoid - Silent failures
try:
    config.run()
except ExtractionError:
    pass  # Error lost!
```

### 4. Clean Up Resources

```python
# ✅ Good - Always clean up
try:
    config.run()
finally:
    # Clean up even if error occurs
    cleanup_resources()

# ❌ Avoid - No cleanup on error
try:
    config.run()
    cleanup_resources()  # Not called if error!
except:
    pass
```

---

## Troubleshooting Guide

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ConfigurationError: VLM backend only supports local inference` | VLM with remote | Use `inference="local"` |
| `ClientError: API key not found` | Missing API key | Set environment variable |
| `ExtractionError: Empty extraction result` | Poor template | Improve field descriptions |
| `ValidationError: Field required` | Missing data | Make field optional |
| `GraphError: Invalid graph structure` | Bad relationships | Check edge definitions |

---

## Next Steps

1. **[Testing →](testing.md)** - Test error handling
2. **[Exceptions Reference →](../../reference/exceptions.md)** - Full exception API
3. **[Extraction Process →](../../fundamentals/extraction-process/index.md)** - Extraction guide

---

## Related Documentation

- **[Exceptions API](../../reference/exceptions.md)** - Exception reference
- **[Pipeline Architecture](../../introduction/architecture.md)** - System design
- **[Extraction Process](../../fundamentals/extraction-process/index.md)** - Extraction guide