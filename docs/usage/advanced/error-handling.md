# Error Handling


## Overview

Handle errors gracefully in docling-graph pipelines with structured exception handling, retry logic, and debugging strategies.

**Prerequisites:**
- Understanding of [Pipeline Architecture](../../introduction/architecture.md)
- Familiarity with [Python API](../api/index.md)
- Basic Python exception handling

!!! tip "New: Zero Data Loss"
    Docling Graph now implements zero data loss - extraction failures return partial models instead of empty results, ensuring you never lose successfully extracted data.

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

from docling_graph import run_pipeline, PipelineConfig
from docling_graph.exceptions import ConfigurationError

try:
    config = PipelineConfig(
        source="document.pdf",
        template="templates.MyTemplate",
        backend="vlm",
        inference="remote"  # VLM doesn't support remote!
    )
    run_pipeline(config)
    
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
    run_pipeline(config)
```

### 2. Client Errors (API)

```python
"""Handle API client errors."""

from docling_graph import run_pipeline, PipelineConfig
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
            run_pipeline(config)
            print("✅ Processing successful")
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

from docling_graph import run_pipeline, PipelineConfig
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
        run_pipeline(config)
        print("✅ VLM extraction successful")
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
        run_pipeline(config)
        print("✅ LLM extraction successful")
        
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
from docling_graph import run_pipeline, PipelineConfig
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
        run_pipeline(config)
        
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
        run_pipeline(config)

# Usage
process_with_validation_handling("document.pdf")
```

### 5. Graph Errors

```python
"""Handle graph construction errors."""

from docling_graph import run_pipeline, PipelineConfig
from docling_graph.exceptions import GraphError

def process_with_graph_validation(source: str):
    """Process with graph validation."""
    
    try:
        config = PipelineConfig(
            source=source,
            template="templates.MyTemplate",
            export_format="cypher"
        )
        run_pipeline(config)
        
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
        run_pipeline(config)

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
    run_pipeline(config)

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
            run_pipeline(config)
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
from docling_graph import run_pipeline, PipelineConfig

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
    run_pipeline(config)
    
except Exception as e:
    logger.error(f"Pipeline failed: {e}", exc_info=True)
    raise
```

### Debug Mode

```python
"""Run pipeline in debug mode."""

from docling_graph import run_pipeline, PipelineConfig
from docling_graph.exceptions import DoclingGraphError

def debug_pipeline(source: str):
    """Run pipeline with detailed error information."""
    
    try:
        config = PipelineConfig(
            source=source,
            template="templates.MyTemplate"
        )
        run_pipeline(config)
        
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

### Trace Data for Debugging

**Trace data** provides visibility into pipeline internals for debugging extraction issues:

```python
"""Use trace data to debug extraction failures."""

from docling_graph import run_pipeline, PipelineConfig

def debug_with_trace(source: str):
    """Debug extraction using trace data."""
    
    config = PipelineConfig(
        source=source,
        template="templates.ComplexTemplate",
        include_trace=True,  # Enable trace capture
        dump_to_disk=True,   # Export for analysis
        output_dir="debug_output"
    )
    
    context = run_pipeline(config)
    
    # Analyze trace data
    if context.trace_data:
        # Check for extraction errors
        errors = [e for e in context.trace_data.extractions if e.error]
        
        if errors:
            print(f"❌ Found {len(errors)} extraction errors:")
            for extraction in errors:
                print(f"\n  Extraction {extraction.extraction_id}:")
                print(f"    Source: {extraction.source_type} {extraction.source_id}")
                print(f"    Error: {extraction.error}")
                print(f"    Time: {extraction.extraction_time:.2f}s")
        
        # Check for slow extractions
        slow = [e for e in context.trace_data.extractions if e.extraction_time > 5.0]
        
        if slow:
            print(f"\n⚠️  Found {len(slow)} slow extractions (>5s):")
            for extraction in sorted(slow, key=lambda e: e.extraction_time, reverse=True):
                print(f"    {extraction.source_type} {extraction.source_id}: {extraction.extraction_time:.2f}s")
        
        # Analyze chunk boundaries (many-to-one mode)
        if context.trace_data.chunks:
            print(f"\n📊 Document split into {len(context.trace_data.chunks)} chunks:")
            for chunk in context.trace_data.chunks:
                print(f"    Chunk {chunk.chunk_id}: pages {chunk.page_numbers}, {chunk.token_count} tokens")
    
    return context

# Usage
context = debug_with_trace("problematic_document.pdf")
```

**See Also:** [Trace Data Debugging Guide](../advanced/trace-data-debugging.md) for comprehensive examples.

```

### Automatic Cleanup on Failure

When `dump_to_disk=True`, the pipeline automatically cleans up empty output directories if processing fails:

```python
"""Automatic cleanup of empty directories on failure."""

from docling_graph import run_pipeline, PipelineConfig
from docling_graph.exceptions import PipelineError

config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    dump_to_disk=True,
    output_dir="outputs"
)

try:
    context = run_pipeline(config)
except PipelineError as e:
    # If the pipeline fails before writing any files,
    # the empty output directory is automatically removed
    print(f"Pipeline failed: {e.message}")
    # No empty artifact directories left in outputs/
```

**Cleanup Behavior:**

- **Empty directories are removed** - If the pipeline fails before writing any files, the output directory is automatically deleted
- **Partial results are preserved** - If any files were written before the failure, the directory is kept
- **Only when dump_to_disk=True** - Cleanup only runs when file exports are enabled
- **Logged for transparency** - Cleanup actions are logged for visibility

**Example Scenarios:**

```python
# Scenario 1: Failure during template loading (before any files)
# → Output directory is removed (empty)

# Scenario 2: Failure during extraction (after docling conversion)
# → Output directory is kept (contains docling/ files)

# Scenario 3: dump_to_disk=False
# → No cleanup needed (no directory created)
```

This ensures your `outputs/` folder stays clean without manual intervention.


---

## Error Recovery Patterns

### Zero Data Loss

**Zero data loss** ensures extraction failures never result in completely empty results:

```python
"""Handle extraction with zero data loss."""

from docling_graph import run_pipeline, PipelineConfig
from pathlib import Path
import json

def process_with_zero_data_loss(source: str):
    """Process document with zero data loss guarantee."""
    
    config = PipelineConfig(
        source=source,
        template="templates.BillingDocument",
        processing_mode="many-to-one",
        output_dir="outputs"
    )
    
    try:
        results = run_pipeline(config)
        
        # Check result type
        if len(results) == 1:
            print("✅ Successfully merged into single model")
            return {"status": "complete", "models": results}
        else:
            print(f"⚠ Got {len(results)} partial models (merge failed)")
            print("  But data is preserved!")
            return {"status": "partial", "models": results}
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        
        # Even on failure, check for partial results
        output_dir = Path("outputs")
        if output_dir.exists():
            # Look for partial model files
            model_files = list(output_dir.glob("*.json"))
            if model_files:
                print(f"✅ Found {len(model_files)} partial model files")
                
                # Load partial models
                partial_models = []
                for file in model_files:
                    with open(file) as f:
                        partial_models.append(json.load(f))
                
                return {"status": "recovered", "models": partial_models}
        
        return {"status": "failed", "models": []}

# Usage
result = process_with_zero_data_loss("invoice.pdf")

if result["status"] == "complete":
    # Use merged model
    model = result["models"][0]
    print(f"Invoice: {model.get('document_no')}")
    
elif result["status"] == "partial":
    # Use partial models
    print("Working with partial models:")
    for i, model in enumerate(result["models"], 1):
        print(f"  Model {i}: {model.get('document_no', 'N/A')}")
    
elif result["status"] == "recovered":
    # Recovered from files
    print("Recovered partial data from files")
    
else:
    print("No data available")
```

### Partial Model Handling

```python
"""Work with partial models when merging fails."""

from docling_graph import run_pipeline, PipelineConfig
from typing import List, Dict, Any

def extract_with_partial_handling(source: str) -> Dict[str, Any]:
    """Extract and handle partial models intelligently."""
    
    config = PipelineConfig(
        source=source,
        template="templates.BillingDocument",
        processing_mode="many-to-one",
        llm_consolidation=True  # Try LLM consolidation
    )
    
    results = run_pipeline(config)
    
    if len(results) == 1:
        # Success: single merged model
        return {
            "status": "merged",
            "document_no": results[0].document_no,
            "total": results[0].total,
            "line_items": len(results[0].line_items or []),
            "completeness": 100
        }
    else:
        # Partial: multiple models
        print(f"⚠ Merge failed, got {len(results)} partial models")
        
        # Combine data from partial models
        combined = {
            "status": "partial",
            "document_no": None,
            "total": None,
            "line_items": 0,
            "completeness": 0
        }
        
        # Extract what we can
        for model in results:
            if model.document_no and not combined["document_no"]:
                combined["document_no"] = model.document_no
            if model.total and not combined["total"]:
                combined["total"] = model.total
            if model.line_items:
                combined["line_items"] += len(model.line_items)
        
        # Calculate completeness
        fields_found = sum([
            bool(combined["document_no"]),
            bool(combined["total"]),
            bool(combined["line_items"] > 0)
        ])
        combined["completeness"] = int((fields_found / 3) * 100)
        
        return combined

# Usage
result = extract_with_partial_handling("invoice.pdf")

print(f"Status: {result['status']}")
print(f"Invoice: {result['document_no'] or 'N/A'}")
print(f"Total: ${result['total'] or 0}")
print(f"Line items: {result['line_items']}")
print(f"Completeness: {result['completeness']}%")

if result['completeness'] < 100:
    print("⚠ Incomplete extraction - consider manual review")
```

### Graceful Degradation

```python
"""Degrade gracefully on errors."""

from docling_graph import run_pipeline, PipelineConfig
from docling_graph.exceptions import ExtractionError

def process_with_degradation(source: str):
    """Process with graceful degradation."""
    
    results = {
        "success": False,
        "method": None,
        "output_dir": None,
        "models": []
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
            models = run_pipeline(config)
            
            results["success"] = True
            results["method"] = method_name
            results["output_dir"] = config.output_dir
            results["models"] = models
            
            print(f"✅ Success with {method_name}")
            print(f"  Extracted {len(models)} model(s)")
            break
            
        except ExtractionError as e:
            print(f"❌ {method_name} failed: {e.message}")
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
from docling_graph import run_pipeline, PipelineConfig

def process_with_partial_success(source: str):
    """Process and handle partial results."""
    
    try:
        config = PipelineConfig(
            source=source,
            template="templates.MyTemplate",
            output_dir="outputs"
        )
        models = run_pipeline(config)
        
        # Check completeness
        if len(models) == 1:
            return {
                "status": "complete",
                "models": models,
                "output_dir": config.output_dir
            }
        else:
            return {
                "status": "partial",
                "models": models,
                "output_dir": config.output_dir,
                "warning": f"Got {len(models)} partial models instead of 1"
            }
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        
        # Check if partial results exist
        output_dir = Path("outputs")
        if output_dir.exists():
            # Check for extracted data
            nodes_file = output_dir / "nodes.csv"
            if nodes_file.exists():
                print("✅ Partial results available in files")
                print(f"  Nodes: {nodes_file}")
                
                # Use partial results
                return {
                    "status": "recovered",
                    "models": [],
                    "output_dir": output_dir
                }
        
        return {"status": "failed", "models": [], "output_dir": None}
```

---

## Validation Strategies

### Pre-Validation

```python
"""Validate before processing."""

from pathlib import Path
from docling_graph import run_pipeline, PipelineConfig
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
    run_pipeline(config)
```

---

## Best Practices

### 👍 Use Specific Exceptions

```python
# ✅ Good - Catch specific exceptions
try:
    run_pipeline(config)
except ClientError as e:
    # Handle API errors
    pass
except ExtractionError as e:
    # Handle extraction errors
    pass

# ❌ Avoid - Catch all exceptions
try:
    run_pipeline(config)
except Exception:
    pass  # What went wrong?
```

### 👍 Provide Context

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

### 👍 Log Before Raising

```python
# ✅ Good - Log then raise
import logging
logger = logging.getLogger(__name__)

try:
    run_pipeline(config)
except ExtractionError as e:
    logger.error(f"Extraction failed: {e}", exc_info=True)
    raise

# ❌ Avoid - Silent failures
try:
    run_pipeline(config)
except ExtractionError:
    pass  # Error lost!
```

### 👍 Clean Up Resources

```python
# ✅ Good - Always clean up
try:
    run_pipeline(config)
finally:
    # Clean up even if error occurs
    cleanup_resources()

# ❌ Avoid - No cleanup on error
try:
    run_pipeline(config)
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

## Zero Data Loss Best Practices

### 1. Always Check Result Count

```python
# ✅ Good - Check if merge succeeded
results = run_pipeline(config)

if len(results) == 1:
    # Merged successfully
    process_merged_model(results[0])
else:
    # Got partial models
    process_partial_models(results)
```

### 2. Handle Partial Models Gracefully

```python
# ✅ Good - Extract what you can from partial models
def get_document_no(models: List) -> str:
    """Get invoice number from any model that has it."""
    for model in models:
        if model.document_no:
            return model.document_no
    return "N/A"
```

### 3. Log Partial Results

```python
# ✅ Good - Log when you get partial results
import logging
logger = logging.getLogger(__name__)

results = run_pipeline(config)
if len(results) > 1:
    logger.warning(f"Got {len(results)} partial models instead of 1")
    logger.info("Data preserved despite merge failure")
```

### 4. Provide User Feedback

```python
# ✅ Good - Inform users about partial results
results = run_pipeline(config)

if len(results) == 1:
    print("✅ Extraction complete")
else:
    print(f"⚠ Extraction partially complete ({len(results)} fragments)")
    print("  All data preserved - manual review recommended")
```

---

## Next Steps

1. **[Model Merging →](../../fundamentals/extraction-process/model-merging.md)** - Learn about zero data loss
2. **[Testing →](testing.md)** - Test error handling
3. **[Exceptions Reference →](../../reference/exceptions.md)** - Full exception API
4. **[Extraction Process →](../../fundamentals/extraction-process/index.md)** - Extraction guide