# run_pipeline()

**Navigation:** [← Python API Overview](index.md) | [Next: PipelineConfig →](pipeline-config.md)

---

## Overview

The `run_pipeline()` function is the **main entry point** for executing the document-to-graph pipeline programmatically.

**Function Signature:**
```python
def run_pipeline(config: Union[PipelineConfig, Dict[str, Any]]) -> None
```

---

## Basic Usage

### With Dictionary

```python
from docling_graph import run_pipeline

run_pipeline({
    "source": "document.pdf",
    "template": "my_templates.Invoice",
    "backend": "llm",
    "inference": "remote",
    "output_dir": "outputs"
})
```

### With PipelineConfig

```python
from docling_graph import PipelineConfig, run_pipeline

config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    backend="llm",
    inference="remote"
)

run_pipeline(config)
```

---

## Parameters

### config

**Type:** `PipelineConfig | Dict[str, Any]`

**Required:** Yes

**Description:** Pipeline configuration as either:
- `PipelineConfig` object (recommended)
- Dictionary with configuration keys

---

## Configuration Keys

### Required Keys

| Key | Type | Description |
|-----|------|-------------|
| `source` | `str` | Path to source document |
| `template` | `str | Type[BaseModel]` | Pydantic template (dotted path or class) |

### Optional Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backend` | `str` | `"llm"` | Backend type: `"llm"` or `"vlm"` |
| `inference` | `str` | `"local"` | Inference mode: `"local"` or `"remote"` |
| `processing_mode` | `str` | `"many-to-one"` | Processing strategy |
| `docling_config` | `str` | `"ocr"` | Docling pipeline: `"ocr"` or `"vision"` |
| `use_chunking` | `bool` | `True` | Enable document chunking |
| `llm_consolidation` | `bool` | `False` | Enable LLM consolidation |
| `export_format` | `str` | `"csv"` | Export format: `"csv"` or `"cypher"` |
| `output_dir` | `str` | `"outputs"` | Output directory path |
| `model_override` | `str` | `None` | Override model name |
| `provider_override` | `str` | `None` | Override provider name |

**See [PipelineConfig](pipeline-config.md) for complete list.**

---

## Return Value

**Type:** `None`

The function doesn't return a value. Results are written to the output directory.

---

## Exceptions

### ConfigurationError

Raised when configuration is invalid.

```python
from docling_graph import run_pipeline
from docling_graph.exceptions import ConfigurationError

try:
    run_pipeline({
        "source": "document.pdf",
        "template": "templates.Invoice",
        "backend": "invalid"  # Invalid backend
    })
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Details: {e.details}")
```

### ExtractionError

Raised when document extraction fails.

```python
from docling_graph.exceptions import ExtractionError

try:
    run_pipeline({
        "source": "document.pdf",
        "template": "templates.Missing"  # Template not found
    })
except ExtractionError as e:
    print(f"Extraction failed: {e.message}")
```

### PipelineError

Raised when pipeline execution fails.

```python
from docling_graph.exceptions import PipelineError

try:
    run_pipeline({
        "source": "document.pdf",
        "template": "templates.Invoice"
    })
except PipelineError as e:
    print(f"Pipeline error: {e.message}")
```

---

## Complete Examples

### Example 1: Minimal Configuration

```python
from docling_graph import run_pipeline

# Minimal required configuration
run_pipeline({
    "source": "invoice.pdf",
    "template": "templates.Invoice"
})

# Output: outputs/
```

### Example 2: Remote LLM

```python
import os
from docling_graph import run_pipeline

# Set API key
os.environ["MISTRAL_API_KEY"] = "your-key"

# Configure for remote inference
run_pipeline({
    "source": "research.pdf",
    "template": "templates.Research",
    "backend": "llm",
    "inference": "remote",
    "provider_override": "mistral",
    "model_override": "mistral-large-latest",
    "processing_mode": "many-to-one",
    "use_chunking": True,
    "llm_consolidation": True,
    "output_dir": "outputs/research"
})
```

### Example 3: Local VLM

```python
from docling_graph import run_pipeline

# VLM for form extraction
run_pipeline({
    "source": "form.jpg",
    "template": "templates.IDCard",
    "backend": "vlm",
    "inference": "local",
    "processing_mode": "one-to-one",
    "docling_config": "vision",
    "output_dir": "outputs/form"
})
```

### Example 4: With Error Handling

```python
from docling_graph import run_pipeline
from docling_graph.exceptions import (
    ConfigurationError,
    ExtractionError,
    PipelineError
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_document(source: str, template: str) -> bool:
    """Process document with comprehensive error handling."""
    try:
        run_pipeline({
            "source": source,
            "template": template,
            "backend": "llm",
            "inference": "remote",
            "output_dir": f"outputs/{Path(source).stem}"
        })
        logger.info(f"✓ Successfully processed: {source}")
        return True
        
    except ConfigurationError as e:
        logger.error(f"Configuration error for {source}: {e.message}")
        if e.details:
            logger.error(f"Details: {e.details}")
        return False
        
    except ExtractionError as e:
        logger.error(f"Extraction failed for {source}: {e.message}")
        return False
        
    except PipelineError as e:
        logger.error(f"Pipeline error for {source}: {e.message}")
        return False
        
    except Exception as e:
        logger.exception(f"Unexpected error for {source}: {e}")
        return False

# Use the function
success = process_document("invoice.pdf", "templates.Invoice")
```

### Example 5: Batch Processing

```python
from pathlib import Path
from docling_graph import run_pipeline

def batch_process(input_dir: str, template: str):
    """Process all PDFs in a directory."""
    documents = Path(input_dir).glob("*.pdf")
    results = {"success": [], "failed": []}
    
    for doc in documents:
        try:
            run_pipeline({
                "source": str(doc),
                "template": template,
                "output_dir": f"outputs/{doc.stem}"
            })
            results["success"].append(doc.name)
            print(f"✓ {doc.name}")
            
        except Exception as e:
            results["failed"].append((doc.name, str(e)))
            print(f"✗ {doc.name}: {e}")
    
    # Summary
    print(f"\nProcessed: {len(results['success'])} succeeded, {len(results['failed'])} failed")
    return results

# Run batch processing
results = batch_process("documents/", "templates.Invoice")
```

---

## Advanced Usage

### Custom Models Configuration

```python
from docling_graph import run_pipeline

# Override models from config
run_pipeline({
    "source": "document.pdf",
    "template": "templates.Invoice",
    "backend": "llm",
    "inference": "remote",
    "models": {
        "llm": {
            "remote": {
                "default_model": "gpt-4-turbo",
                "provider": "openai"
            }
        }
    }
})
```

### Multiple Export Formats

```python
from docling_graph import run_pipeline

# Export as Cypher for Neo4j
run_pipeline({
    "source": "document.pdf",
    "template": "templates.Invoice",
    "export_format": "cypher",
    "output_dir": "outputs/neo4j"
})

# Then import to Neo4j
import subprocess
subprocess.run([
    "cypher-shell",
    "-f", "outputs/neo4j/graph.cypher"
])
```

### Conditional Processing

```python
from pathlib import Path
from docling_graph import run_pipeline

def smart_process(source: str):
    """Choose configuration based on document type."""
    path = Path(source)
    
    # Determine template and config
    if "invoice" in path.name.lower():
        template = "templates.Invoice"
        backend = "vlm"
        processing = "one-to-one"
    elif "research" in path.name.lower():
        template = "templates.Research"
        backend = "llm"
        processing = "many-to-one"
    else:
        raise ValueError(f"Unknown document type: {path.name}")
    
    # Process with appropriate config
    run_pipeline({
        "source": source,
        "template": template,
        "backend": backend,
        "processing_mode": processing,
        "output_dir": f"outputs/{path.stem}"
    })

# Use smart processing
smart_process("invoice_001.pdf")
smart_process("research_paper.pdf")
```

---

## Integration Patterns

### Flask API

```python
from flask import Flask, request, jsonify
from docling_graph import run_pipeline
from pathlib import Path
import uuid

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_endpoint():
    """API endpoint for document processing."""
    file = request.files.get('document')
    template = request.form.get('template', 'templates.Invoice')
    
    if not file:
        return jsonify({"error": "No file provided"}), 400
    
    # Save temporarily
    temp_id = str(uuid.uuid4())
    temp_path = f"temp/{temp_id}_{file.filename}"
    Path("temp").mkdir(exist_ok=True)
    file.save(temp_path)
    
    try:
        # Process
        output_dir = f"outputs/{temp_id}"
        run_pipeline({
            "source": temp_path,
            "template": template,
            "output_dir": output_dir
        })
        
        return jsonify({
            "status": "success",
            "output_dir": output_dir,
            "id": temp_id
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
        
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

if __name__ == '__main__':
    app.run(debug=True)
```

### Celery Task

```python
from celery import Celery
from docling_graph import run_pipeline
from pathlib import Path

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def process_document_task(source: str, template: str, output_dir: str):
    """Async document processing task."""
    try:
        run_pipeline({
            "source": source,
            "template": template,
            "output_dir": output_dir
        })
        return {"status": "success", "output_dir": output_dir}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Usage
result = process_document_task.delay(
    "document.pdf",
    "templates.Invoice",
    "outputs/task_001"
)
```

### Airflow Operator

```python
from airflow.operators.python import PythonOperator
from docling_graph import run_pipeline

def process_document(**context):
    """Airflow task for document processing."""
    params = context['params']
    
    run_pipeline({
        "source": params['source'],
        "template": params['template'],
        "output_dir": f"outputs/{context['ds']}"
    })

# In DAG definition
process_task = PythonOperator(
    task_id='process_document',
    python_callable=process_document,
    params={
        'source': 'documents/daily.pdf',
        'template': 'templates.Invoice'
    }
)
```

---

## Best Practices

### 1. Use PipelineConfig for Type Safety

```python
# ✅ Good - Type-safe with validation
from docling_graph import PipelineConfig, run_pipeline

config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
    backend="llm"  # Validated at creation
)
run_pipeline(config)

# ❌ Avoid - No validation until runtime
run_pipeline({
    "source": "document.pdf",
    "template": "templates.Invoice",
    "backend": "invalid"  # Error at runtime
})
```

### 2. Handle Errors Explicitly

```python
# ✅ Good - Specific error handling
from docling_graph.exceptions import ExtractionError

try:
    run_pipeline(config)
except ExtractionError as e:
    logger.error(f"Extraction failed: {e.message}")
    # Implement retry or fallback

# ❌ Avoid - Silent failures
try:
    run_pipeline(config)
except:
    pass
```

### 3. Organize Outputs

```python
# ✅ Good - Unique output directories
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_pipeline({
    "source": "document.pdf",
    "template": "templates.Invoice",
    "output_dir": f"outputs/{timestamp}"
})

# ❌ Avoid - Overwriting outputs
run_pipeline({
    "source": "document.pdf",
    "template": "templates.Invoice",
    "output_dir": "outputs"  # Same for all
})
```

---

## Troubleshooting

### Issue: Template Not Found

**Error:**
```
ModuleNotFoundError: No module named 'templates'
```

**Solution:**
```python
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

# Now import works
from docling_graph import run_pipeline

run_pipeline({
    "source": "document.pdf",
    "template": "templates.Invoice"
})
```

### Issue: API Key Not Found

**Error:**
```
ConfigurationError: API key not found for provider: mistral
```

**Solution:**
```python
import os

# Set API key before running
os.environ["MISTRAL_API_KEY"] = "your-key"

from docling_graph import run_pipeline

run_pipeline({
    "source": "document.pdf",
    "template": "templates.Invoice",
    "inference": "remote"
})
```

---

## Next Steps

1. **[PipelineConfig →](pipeline-config.md)** - Configuration class
2. **[Programmatic Examples →](programmatic-examples.md)** - More examples
3. **[Batch Processing →](batch-processing.md)** - Batch patterns

---

## Quick Reference

### Basic Call

```python
from docling_graph import run_pipeline

run_pipeline({
    "source": "document.pdf",
    "template": "templates.Invoice"
})
```

### With Options

```python
run_pipeline({
    "source": "document.pdf",
    "template": "templates.Invoice",
    "backend": "llm",
    "inference": "remote",
    "output_dir": "outputs"
})
```

### Error Handling

```python
from docling_graph.exceptions import PipelineError

try:
    run_pipeline(config)
except PipelineError as e:
    print(f"Error: {e.message}")
```

---

**Navigation:** [← Python API Overview](index.md) | [Next: PipelineConfig →](pipeline-config.md)