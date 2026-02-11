# Python API


## Overview

The **docling-graph Python API** provides programmatic access to the document-to-graph pipeline, enabling integration into Python applications, notebooks, and workflows.

**Key Components:**
- `run_pipeline()` - Main pipeline function
- `PipelineConfig` - Type-safe configuration
- Direct module imports for advanced usage

---

## Quick Start

### Basic Usage (API Mode - No File Exports)

```python
from docling_graph import run_pipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="remote"
)

# Run pipeline - returns data directly
context = run_pipeline(config)

# Access results in memory
graph = context.knowledge_graph
model = context.pydantic_model
print(f"Extracted {graph.number_of_nodes()} nodes")
```

---

## Installation

```bash
# Install with all features
uv sync

# Or specific features
uv sync  # Remote APIs
uv sync  # Local inference
```

---

## API Components

### 1. PipelineConfig

Type-safe configuration class with validation.

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="remote"
)
```

**Learn more:** [PipelineConfig →](pipeline-config.md)

---

### 2. run_pipeline()

Main pipeline execution function.

```python
from docling_graph import run_pipeline

run_pipeline({
    "source": "document.pdf",
    "template": "templates.BillingDocument"
})
```

**Learn more:** [run_pipeline() →](run-pipeline.md)

---

### 3. Direct Module Access

For advanced usage, import modules directly.

```python
from docling_graph.core.converters import GraphConverter
from docling_graph.core.exporters import CSVExporter
from docling_graph.core.visualizers import InteractiveVisualizer
```

**Learn more:** [API Reference →](../../reference/index.md)

---

## Common Patterns

### Pattern 1: Simple Conversion (Memory-Efficient)

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="invoice.pdf",
    template="templates.BillingDocument"
)

# Returns data directly - no file exports
context = run_pipeline(config)
graph = context.knowledge_graph
invoice = context.pydantic_model
```

---

### Pattern 2: Custom Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="research.pdf",
    template="templates.ScholarlyRheologyPaper",
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-large-latest",
    processing_mode="many-to-one",
    use_chunking=True,
)

# Access results in memory
context = run_pipeline(config)
graph = context.knowledge_graph
print(f"Research: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

---

### Pattern 3: Batch Processing (Memory-Efficient)

```python
from pathlib import Path
from docling_graph import run_pipeline, PipelineConfig

documents = Path("documents").glob("*.pdf")
all_graphs = []

for doc in documents:
    config = PipelineConfig(
        source=str(doc),
        template="templates.BillingDocument"
    )
    
    try:
        # Process without file exports
        context = run_pipeline(config)
        all_graphs.append({
            "filename": doc.name,
            "graph": context.knowledge_graph,
            "model": context.pydantic_model
        })
        print(f"✅ Processed: {doc.name}")
    except Exception as e:
        print(f"❌ Failed: {doc.name} - {e}")

# Aggregate results
total_nodes = sum(g["graph"].number_of_nodes() for g in all_graphs)
print(f"\nTotal entities: {total_nodes}")
```

---

### Pattern 4: Error Handling

```python
from docling_graph import PipelineConfig
from docling_graph.exceptions import (
    ConfigurationError,
    ExtractionError,
    PipelineError
)

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument"
)

try:
    run_pipeline(config)
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Details: {e.details}")
except ExtractionError as e:
    print(f"Extraction failed: {e.message}")
except PipelineError as e:
    print(f"Pipeline error: {e.message}")
```

---

## Comparison: CLI vs Python API

| Feature | CLI | Python API |
|---------|-----|------------|
| **Ease of Use** | Simple commands | Requires Python code |
| **Flexibility** | Limited to options | Full programmatic control |
| **Integration** | Shell scripts | Python applications |
| **File Exports** | Always exports files | No exports by default (memory-efficient) |
| **Return Values** | N/A | Returns `PipelineContext` with graph and model |
| **Batch Processing** | Shell loops | Python loops with error handling |
| **Configuration** | YAML + flags | PipelineConfig objects |
| **Best For** | Quick tasks, scripts | Applications, notebooks, workflows |

!!! note "Python API export behavior"
    Python API defaults to dump_to_disk=False for memory efficiency. Set dump_to_disk=True to enable file exports.

---

## Environment Setup

### API Keys

```python
import os

# Set API keys programmatically
os.environ["MISTRAL_API_KEY"] = "your-key"
os.environ["OPENAI_API_KEY"] = "your-key"

# Or use python-dotenv
from dotenv import load_dotenv
load_dotenv()

from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="remote"
)
run_pipeline(config)
```

### Python Path

```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Now you can import templates
from templates.billing_document import BillingDocument
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template=Invoice  # Pass class directly
)
run_pipeline(config)
```

---

## Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify
from docling_graph import PipelineConfig
from pathlib import Path
import uuid

app = Flask(__name__)

@app.route('/convert', methods=['POST'])
def convert_document():
    # Get uploaded file
    file = request.files['document']
    template = request.form.get('template', 'templates.BillingDocument')
    
    # Save temporarily
    temp_id = str(uuid.uuid4())
    temp_path = f"temp/{temp_id}_{file.filename}"
    file.save(temp_path)
    
    # Process
    try:
        config = PipelineConfig(
            source=temp_path,
            template=template,
        )
        context = run_pipeline(config)

        return jsonify({
            "status": "success",
            "model": context.pydantic_model.model_dump()
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

### Jupyter Notebook

```python
# Cell 1: Setup
from docling_graph import PipelineConfig
import pandas as pd
import matplotlib.pyplot as plt

# Cell 2: Process document
config = PipelineConfig(
    source="research.pdf",
    template="templates.ScholarlyRheologyPaper"
)
context = run_pipeline(config)

# Cell 3: Analyze results
graph = context.knowledge_graph

print(f"Total nodes: {graph.number_of_nodes()}")
print(f"Total edges: {graph.number_of_edges()}")

# Cell 4: Visualize
node_types = nodes['type'].value_counts()
node_types.plot(kind='bar', title='Node Types')
plt.show()
```

### Airflow DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from docling_graph import PipelineConfig

def process_document(**context):
    config = PipelineConfig(
        source=context['params']['source'],
        template=context['params']['template']
    )
    run_pipeline(config)

with DAG(
    'document_processing',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily'
) as dag:
    
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

### 👍 Use Type-Safe Configuration

```python
# ✅ Good - Type-safe with validation
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm"  # Validated
)

# ❌ Avoid - Dictionary without validation
config = {
    "source": "document.pdf",
    "template": "templates.BillingDocument",
    "backend": "invalid"  # No validation
}
```

### 👍 Handle Errors Gracefully

```python
# ✅ Good - Specific error handling
from docling_graph import PipelineConfig
from docling_graph.exceptions import ExtractionError

try:
    run_pipeline(config)
except ExtractionError as e:
    logger.error(f"Extraction failed: {e.message}")
    # Implement retry logic or fallback

# ❌ Avoid - Catching all exceptions
try:
    run_pipeline(config)
except Exception:
    pass  # Silent failure
```

## Next Steps

Explore the Python API in detail:

1. **[run_pipeline() →](run-pipeline.md)** - Pipeline function
2. **[PipelineConfig →](pipeline-config.md)** - Configuration class
3. **[Programmatic Examples →](programmatic-examples.md)** - Code examples
4. **[Batch Processing →](batch-processing.md)** - Batch patterns

Or continue to:
- **[Examples →](../examples/index.md)** - Real-world examples
- **[API Reference →](../../reference/index.md)** - Complete API docs