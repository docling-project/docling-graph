# Python API

**Navigation:** [← CLI Recipes](../07-cli/cli-recipes.md) | [Next: Examples →](../09-examples/index.md)

---

## Overview

The **docling-graph Python API** provides programmatic access to the document-to-graph pipeline, enabling integration into Python applications, notebooks, and workflows.

**Key Components:**
- `run_pipeline()` - Main pipeline function
- `PipelineConfig` - Type-safe configuration
- Direct module imports for advanced usage

---

## Quick Start

### Basic Usage

```python
from docling_graph import PipelineConfig

# Configure pipeline
config = PipelineConfig(
    source="document.pdf",
    template="my_templates.Invoice",
    backend="llm",
    inference="remote",
    output_dir="outputs"
)

# Run pipeline
config.run()
```

### Using run_pipeline()

```python
from docling_graph import run_pipeline

# Run with dictionary config
run_pipeline({
    "source": "document.pdf",
    "template": "my_templates.Invoice",
    "backend": "llm",
    "inference": "remote",
    "output_dir": "outputs"
})
```

---

## Installation

```bash
# Install with all features
uv sync --extra all

# Or specific features
uv sync --extra remote  # Remote APIs
uv sync --extra local   # Local inference
```

---

## API Components

### 1. PipelineConfig

Type-safe configuration class with validation.

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
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
    "template": "templates.Invoice"
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

**Learn more:** [API Reference →](../11-reference/index.md)

---

## Common Patterns

### Pattern 1: Simple Conversion

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="invoice.pdf",
    template="templates.Invoice"
)

config.run()
```

---

### Pattern 2: Custom Configuration

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="research.pdf",
    template="templates.Research",
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-large-latest",
    processing_mode="many-to-one",
    use_chunking=True,
    llm_consolidation=True,
    output_dir="outputs/research"
)

config.run()
```

---

### Pattern 3: Batch Processing

```python
from pathlib import Path
from docling_graph import PipelineConfig

documents = Path("documents").glob("*.pdf")

for doc in documents:
    config = PipelineConfig(
        source=str(doc),
        template="templates.Invoice",
        output_dir=f"outputs/{doc.stem}"
    )
    
    try:
        config.run()
        print(f"✓ Processed: {doc.name}")
    except Exception as e:
        print(f"✗ Failed: {doc.name} - {e}")
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
    template="templates.Invoice"
)

try:
    config.run()
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
| **Batch Processing** | Shell loops | Python loops with error handling |
| **Configuration** | YAML + flags | PipelineConfig objects |
| **Best For** | Quick tasks, scripts | Applications, notebooks, workflows |

---

## When to Use Python API

### ✅ Use Python API for:

- **Application Integration**
  ```python
  # Integrate into web app
  from flask import Flask, request
  from docling_graph import PipelineConfig
  
  @app.route('/process', methods=['POST'])
  def process_document():
      file = request.files['document']
      config = PipelineConfig(
          source=file.filename,
          template="templates.Invoice"
      )
      config.run()
      return {"status": "success"}
  ```

- **Jupyter Notebooks**
  ```python
  # Interactive analysis
  from docling_graph import PipelineConfig
  import pandas as pd
  
  config = PipelineConfig(
      source="data.pdf",
      template="templates.Research"
  )
  config.run()
  
  # Analyze results
  nodes = pd.read_csv("outputs/nodes.csv")
  nodes.head()
  ```

- **Complex Workflows**
  ```python
  # Multi-step processing
  from docling_graph import PipelineConfig
  
  # Step 1: Extract
  config = PipelineConfig(
      source="document.pdf",
      template="templates.Invoice"
  )
  config.run()
  
  # Step 2: Post-process
  import pandas as pd
  nodes = pd.read_csv("outputs/nodes.csv")
  filtered = nodes[nodes['type'] == 'Organization']
  filtered.to_csv("outputs/organizations.csv")
  ```

- **Batch Processing with Logic**
  ```python
  # Conditional processing
  from pathlib import Path
  from docling_graph import PipelineConfig
  
  for doc in Path("documents").glob("*.pdf"):
      # Choose template based on filename
      if "invoice" in doc.name.lower():
          template = "templates.Invoice"
      elif "research" in doc.name.lower():
          template = "templates.Research"
      else:
          continue
      
      config = PipelineConfig(
          source=str(doc),
          template=template
      )
      config.run()
  ```

### ❌ Use CLI for:

- Quick one-off conversions
- Shell script automation
- Simple batch processing
- Manual testing

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
    template="templates.Invoice",
    inference="remote"
)
config.run()
```

### Python Path

```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Now you can import templates
from templates.invoice import Invoice
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template=Invoice  # Pass class directly
)
config.run()
```

---

## Output Handling

### Access Output Files

```python
from pathlib import Path
import json
import pandas as pd

from docling_graph import PipelineConfig

# Run pipeline
config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
    output_dir="outputs"
)
config.run()

# Read outputs
nodes = pd.read_csv("outputs/nodes.csv")
edges = pd.read_csv("outputs/edges.csv")

with open("outputs/graph.json") as f:
    graph = json.load(f)

with open("outputs/graph_stats.json") as f:
    stats = json.load(f)

print(f"Nodes: {stats['node_count']}")
print(f"Edges: {stats['edge_count']}")
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
    template = request.form.get('template', 'templates.Invoice')
    
    # Save temporarily
    temp_id = str(uuid.uuid4())
    temp_path = f"temp/{temp_id}_{file.filename}"
    file.save(temp_path)
    
    # Process
    try:
        config = PipelineConfig(
            source=temp_path,
            template=template,
            output_dir=f"outputs/{temp_id}"
        )
        config.run()
        
        return jsonify({
            "status": "success",
            "output_dir": f"outputs/{temp_id}"
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
    template="templates.Research",
    output_dir="outputs/research"
)
config.run()

# Cell 3: Analyze results
nodes = pd.read_csv("outputs/research/nodes.csv")
edges = pd.read_csv("outputs/research/edges.csv")

print(f"Total nodes: {len(nodes)}")
print(f"Total edges: {len(edges)}")

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
        template=context['params']['template'],
        output_dir=f"outputs/{context['ds']}"
    )
    config.run()

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
            'template': 'templates.Invoice'
        }
    )
```

---

## Best Practices

### 1. Use Type-Safe Configuration

```python
# ✅ Good - Type-safe with validation
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
    backend="llm"  # Validated
)

# ❌ Avoid - Dictionary without validation
config = {
    "source": "document.pdf",
    "template": "templates.Invoice",
    "backend": "invalid"  # No validation
}
```

### 2. Handle Errors Gracefully

```python
# ✅ Good - Specific error handling
from docling_graph import PipelineConfig
from docling_graph.exceptions import ExtractionError

try:
    config.run()
except ExtractionError as e:
    logger.error(f"Extraction failed: {e.message}")
    # Implement retry logic or fallback

# ❌ Avoid - Catching all exceptions
try:
    config.run()
except Exception:
    pass  # Silent failure
```

### 3. Organize Outputs

```python
# ✅ Good - Organized structure
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
    output_dir=f"outputs/invoices/{timestamp}"
)

# ❌ Avoid - Overwriting outputs
config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
    output_dir="outputs"  # Same for all
)
```

---

## Next Steps

Explore the Python API in detail:

1. **[run_pipeline() →](run-pipeline.md)** - Pipeline function
2. **[PipelineConfig →](pipeline-config.md)** - Configuration class
3. **[Programmatic Examples →](programmatic-examples.md)** - Code examples
4. **[Batch Processing →](batch-processing.md)** - Batch patterns

Or continue to:
- **[Examples →](../09-examples/index.md)** - Real-world examples
- **[API Reference →](../11-reference/index.md)** - Complete API docs

---

## Quick Reference

### Basic Usage

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice"
)
config.run()
```

### With Options

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.Invoice",
    backend="llm",
    inference="remote",
    provider_override="mistral",
    output_dir="outputs"
)
config.run()
```

### Error Handling

```python
from docling_graph.exceptions import PipelineError

try:
    config.run()
except PipelineError as e:
    print(f"Error: {e.message}")
```

---

**Navigation:** [← CLI Recipes](../07-cli/cli-recipes.md) | [Next: Examples →](../09-examples/index.md)