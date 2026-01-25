# run_pipeline()


## Overview

The `run_pipeline()` function is the **main entry point** for executing the document-to-graph pipeline programmatically.

**Function Signature:**
```python
def run_pipeline(config: Union[PipelineConfig, Dict[str, Any]]) -> PipelineContext
```

**Returns:** `PipelineContext` object containing the knowledge graph, Pydantic model, and other pipeline results.

---

## Basic Usage

### Default Behavior (No File Exports)

```python
from docling_graph import run_pipeline

# Returns data directly - no file exports by default
context = run_pipeline({
    "source": "document.pdf",
    "template": "templates.BillingDocument",
    "backend": "llm",
    "inference": "remote"
})

# Access results
graph = context.knowledge_graph
model = context.pydantic_model
print(f"Extracted {graph.number_of_nodes()} nodes")
```

### With File Exports

```python
from docling_graph import run_pipeline

# Enable file exports with dump_to_disk
context = run_pipeline({
    "source": "document.pdf",
    "template": "templates.BillingDocument",
    "backend": "llm",
    "inference": "remote",
    "dump_to_disk": True,
    "output_dir": "outputs"
})

# Results available both in memory and on disk
graph = context.knowledge_graph
# Files also written to outputs/
```

### With PipelineConfig

```python
from docling_graph import PipelineConfig, run_pipeline

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="remote"
)

# Returns PipelineContext
context = run_pipeline(config)
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
| `dump_to_disk` | `bool` or `None` | `None` | Control file exports (None=auto: CLI=True, API=False) |
| `export_format` | `str` | `"csv"` | Export format: `"csv"` or `"cypher"` |
| `output_dir` | `str` | `"outputs"` | Output directory path |
| `model_override` | `str` | `None` | Override model name |
| `provider_override` | `str` | `None` | Override provider name |

**See [PipelineConfig](pipeline-config.md) for complete list.**

---

## Return Value

**Type:** `PipelineContext`

Returns a `PipelineContext` object containing:

| Attribute | Type | Description |
|-----------|------|-------------|
| `knowledge_graph` | `nx.DiGraph` | NetworkX directed graph with extracted entities and relationships |
| `pydantic_model` | `BaseModel` | Validated Pydantic model instance |
| `docling_document` | `DoclingDocument` | Original Docling document object |
| `chunks` | `list` | Document chunks (if chunking enabled) |
| `extraction_results` | `list` | Raw extraction results from backend |

**Example:**
```python
context = run_pipeline(config)

# Access the knowledge graph
graph = context.knowledge_graph
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")

# Access the Pydantic model
model = context.pydantic_model
print(f"Model type: {type(model).__name__}")

# Access document chunks
if context.chunks:
    print(f"Chunks: {len(context.chunks)}")
```

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
        "template": "templates.BillingDocument",
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
        "template": "templates.BillingDocument"
    })
except PipelineError as e:
    print(f"Pipeline error: {e.message}")
```

---

## Complete Examples

### 📍 Minimal Configuration (API Mode)

```python
from docling_graph import run_pipeline

# Minimal required configuration - returns data, no file exports
context = run_pipeline({
    "source": "invoice.pdf",
    "template": "templates.BillingDocument"
})

# Access results in memory
graph = context.knowledge_graph
invoice = context.pydantic_model
print(f"Extracted invoice with {graph.number_of_nodes()} entities")
```

### 📍 With File Exports

```python
from docling_graph import run_pipeline

# Enable file exports
context = run_pipeline({
    "source": "invoice.pdf",
    "template": "templates.BillingDocument",
    "dump_to_disk": True,
    "output_dir": "outputs/invoice"
})

# Results available both in memory and on disk
graph = context.knowledge_graph
# Files also written to outputs/invoice/
```

### 📍 Remote LLM

```python
import os
from docling_graph import run_pipeline

# Set API key
os.environ["MISTRAL_API_KEY"] = "your-key"

# Configure for remote inference
context = run_pipeline({
    "source": "research.pdf",
    "template": "templates.Research",
    "backend": "llm",
    "inference": "remote",
    "provider_override": "mistral",
    "model_override": "mistral-large-latest",
    "processing_mode": "many-to-one",
    "use_chunking": True,
    "llm_consolidation": True
})

# Access the knowledge graph
graph = context.knowledge_graph
print(f"Rheology research: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

### 📍 Local VLM

```python
from docling_graph import run_pipeline

# VLM for form extraction
context = run_pipeline({
    "source": "form.jpg",
    "template": "templates.IDCard",
    "backend": "vlm",
    "inference": "local",
    "processing_mode": "one-to-one",
    "docling_config": "vision"
})

# Access extracted data
id_card = context.pydantic_model
print(f"Name: {id_card.first_name} {id_card.last_name}")
```

### 📍 With Error Handling

```python
from docling_graph import run_pipeline, PipelineContext
from docling_graph.exceptions import (
    ConfigurationError,
    ExtractionError,
    PipelineError
)
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_document(source: str, template: str) -> PipelineContext | None:
    """Process document with comprehensive error handling."""
    try:
        context = run_pipeline({
            "source": source,
            "template": template,
            "backend": "llm",
            "inference": "remote"
        })
        logger.info(f"✅ Successfully processed: {source}")
        logger.info(f"  Nodes: {context.knowledge_graph.number_of_nodes()}")
        return context
        
    except ConfigurationError as e:
        logger.error(f"Configuration error for {source}: {e.message}")
        if e.details:
            logger.error(f"Details: {e.details}")
        return None
        
    except ExtractionError as e:
        logger.error(f"Extraction failed for {source}: {e.message}")
        return None
        
    except PipelineError as e:
        logger.error(f"Pipeline error for {source}: {e.message}")
        return None
        
    except Exception as e:
        logger.exception(f"Unexpected error for {source}: {e}")
        return None

# Use the function
context = process_document("invoice.pdf", "templates.BillingDocument")
if context:
    print(f"Graph has {context.knowledge_graph.number_of_nodes()} nodes")
```

### 📍 Batch Processing (Memory-Efficient)

```python
from pathlib import Path
from docling_graph import run_pipeline

def batch_process(input_dir: str, template: str):
    """Process all PDFs in a directory without disk writes."""
    documents = Path(input_dir).glob("*.pdf")
    results = {"success": [], "failed": []}
    all_graphs = []
    
    for doc in documents:
        try:
            # Process without file exports
            context = run_pipeline({
                "source": str(doc),
                "template": template
            })
            
            # Store graph in memory
            all_graphs.append({
                "filename": doc.name,
                "graph": context.knowledge_graph,
                "model": context.pydantic_model
            })
            
            results["success"].append(doc.name)
            print(f"✅ {doc.name}: {context.knowledge_graph.number_of_nodes()} nodes")
            
        except Exception as e:
            results["failed"].append((doc.name, str(e)))
            print(f"❌ {doc.name}: {e}")
    
    # Summary
    print(f"\nProcessed: {len(results['success'])} succeeded, {len(results['failed'])} failed")
    return results, all_graphs

# Run batch processing
results, graphs = batch_process("documents/", "templates.BillingDocument")

# Optionally export combined results
if graphs:
    print(f"\nTotal entities across all documents: {sum(g['graph'].number_of_nodes() for g in graphs)}")
```

---

## Advanced Usage

### dump_to_disk Behavior

The `dump_to_disk` parameter controls file exports:

```python
from docling_graph import run_pipeline

# Default: No file exports (API mode)
context = run_pipeline({
    "source": "document.pdf",
    "template": "templates.BillingDocument"
})
# Returns data in memory only

# Explicit: Enable file exports
context = run_pipeline({
    "source": "document.pdf",
    "template": "templates.BillingDocument",
    "dump_to_disk": True,
    "output_dir": "outputs"
})
# Returns data AND writes files

# Explicit: Disable file exports
context = run_pipeline({
    "source": "document.pdf",
    "template": "templates.BillingDocument",
    "dump_to_disk": False
})
# Returns data only, even if output_dir is set
```

**Note:** CLI mode defaults to `dump_to_disk=True`, API mode defaults to `dump_to_disk=False`.

### Custom Models Configuration

```python
from docling_graph import run_pipeline

# Override models from config
context = run_pipeline({
    "source": "document.pdf",
    "template": "templates.BillingDocument",
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

# Access results
graph = context.knowledge_graph
```

### Multiple Export Formats

```python
from docling_graph import run_pipeline

# Export as Cypher for Neo4j
context = run_pipeline({
    "source": "document.pdf",
    "template": "templates.BillingDocument",
    "dump_to_disk": True,
    "export_format": "cypher",
    "output_dir": "outputs/neo4j"
})

# Access graph in memory
graph = context.knowledge_graph

# Also import files to Neo4j
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
        template = "templates.BillingDocument"
        backend = "vlm"
        processing = "one-to-one"
    elif "research" in path.name.lower():
        template = "templates.Research"
        backend = "llm"
        processing = "many-to-one"
    else:
        raise ValueError(f"Unknown document type: {path.name}")
    
    # Process and return results
    context = run_pipeline({
        "source": source,
        "template": template,
        "backend": backend,
        "processing_mode": processing
    })
    
    return context

# Use smart processing
invoice_context = smart_process("invoice_001.pdf")
research_context = smart_process("research_paper.pdf")

print(f"Invoice nodes: {invoice_context.knowledge_graph.number_of_nodes()}")
print(f"Research nodes: {research_context.knowledge_graph.number_of_nodes()}")
```

---

## Integration Patterns

### Flask API (Memory-Efficient)

```python
from flask import Flask, request, jsonify
from docling_graph import run_pipeline
from pathlib import Path
import uuid

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_endpoint():
    """API endpoint for document processing - returns data without disk writes."""
    file = request.files.get('document')
    template = request.form.get('template', 'templates.BillingDocument')
    
    if not file:
        return jsonify({"error": "No file provided"}), 400
    
    # Save temporarily
    temp_id = str(uuid.uuid4())
    temp_path = f"temp/{temp_id}_{file.filename}"
    Path("temp").mkdir(exist_ok=True)
    file.save(temp_path)
    
    try:
        # Process without file exports (memory-efficient)
        context = run_pipeline({
            "source": temp_path,
            "template": template
        })
        
        # Extract data from context
        graph = context.knowledge_graph
        model = context.pydantic_model
        
        return jsonify({
            "status": "success",
            "id": temp_id,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "data": model.model_dump() if hasattr(model, 'model_dump') else model.dict()
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

### Celery Task (With Return Data)

```python
from celery import Celery
from docling_graph import run_pipeline
from pathlib import Path

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def process_document_task(source: str, template: str):
    """Async document processing task - returns graph statistics."""
    try:
        context = run_pipeline({
            "source": source,
            "template": template
        })
        
        graph = context.knowledge_graph
        return {
            "status": "success",
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "node_types": list(set(data.get('type') for _, data in graph.nodes(data=True)))
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Usage
result = process_document_task.delay(
    "document.pdf",
    "templates.BillingDocument"
)
# Get result
data = result.get(timeout=300)
print(f"Processed: {data['nodes']} nodes, {data['edges']} edges")
```

### Airflow Operator (With XCom)

```python
from airflow.operators.python import PythonOperator
from docling_graph import run_pipeline

def process_document(**context):
    """Airflow task for document processing - pushes results to XCom."""
    params = context['params']
    
    # Process and get results
    pipeline_context = run_pipeline({
        "source": params['source'],
        "template": params['template']
    })
    
    # Push graph statistics to XCom
    graph = pipeline_context.knowledge_graph
    context['task_instance'].xcom_push(
        key='graph_stats',
        value={
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges()
        }
    )
    
    return "Processing complete"

# In DAG definition
process_task = PythonOperator(
    task_id='process_document',
    python_callable=process_document,
    params={
        'source': 'documents/daily.pdf',
        'template': 'templates.BillingDocument'
    }
)
```

---

## Best Practices

### 👍 Use PipelineConfig for Type Safety

```python
# ✅ Good - Type-safe with validation
from docling_graph import PipelineConfig, run_pipeline

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm"  # Validated at creation
)
run_pipeline(config)

# ❌ Avoid - No validation until runtime
run_pipeline({
    "source": "document.pdf",
    "template": "templates.BillingDocument",
    "backend": "invalid"  # Error at runtime
})
```

### 👍 Handle Errors Explicitly

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

### 👍 Organize Outputs

```python
# ✅ Good - Unique output directories
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_pipeline({
    "source": "document.pdf",
    "template": "templates.BillingDocument",
    "output_dir": f"outputs/{timestamp}"
})

# ❌ Avoid - Overwriting outputs
run_pipeline({
    "source": "document.pdf",
    "template": "templates.BillingDocument",
    "output_dir": "outputs"  # Same for all
})
```

---

## Troubleshooting

### 🐛 Template Not Found

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
    "template": "templates.BillingDocument"
})
```

### 🐛 API Key Not Found

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
    "template": "templates.BillingDocument",
    "inference": "remote"
})
```

---

## Next Steps

1. **[PipelineConfig →](pipeline-config.md)** - Configuration class
2. **[Programmatic Examples →](programmatic-examples.md)** - More examples
3. **[Batch Processing →](batch-processing.md)** - Batch patterns