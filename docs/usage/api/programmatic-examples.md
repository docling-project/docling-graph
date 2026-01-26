# Programmatic Examples


## Overview

This guide provides **complete, ready-to-run Python examples** for common document processing scenarios using the docling-graph API.

All examples use `uv run python` for execution.

---

## Quick Reference

| Example | Use Case | Backend |
|---------|----------|---------|
| [Simple Invoice](#example-1-simple-invoice-extraction) | Basic extraction | LLM (Remote) |
| [Local Processing](#example-2-local-processing-with-ollama) | Offline processing | LLM (Local) |
| [VLM Form Extraction](#example-3-vlm-form-extraction) | Image forms | VLM (Local) |
| [Rheology Research](#example-4-rheology-research-with-consolidation) | Complex documents | LLM (Remote) |
| [Batch Processing](#example-5-batch-processing) | Multiple documents | Any |
| [Error Handling](#example-6-robust-error-handling) | Production code | Any |
| [Flask Integration](#example-7-flask-api-integration) | Web application | Any |
| [Jupyter Notebook](#example-8-jupyter-notebook-analysis) | Interactive analysis | Any |

---

## Example 1: Simple Invoice Extraction

**Use Case:** Extract structured data from an invoice using remote LLM.

**File:** `examples/simple_billing_document.py`

```python
"""
Simple invoice extraction using remote LLM.
"""

import os
from pathlib import Path
from docling_graph import run_pipeline, PipelineConfig

# Set API key
os.environ["MISTRAL_API_KEY"] = "your-api-key"

# Configure pipeline
config = PipelineConfig(
    source="documents/invoice.pdf",
    template="templates.billing_document.BillingDocument",
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-small-latest",
    output_dir="outputs/invoice"
)

# Run pipeline
print("Processing invoice...")
run_pipeline(config)
print(f"✅ Complete! Results in: {config.output_dir}")

# Read results
import pandas as pd
nodes = pd.read_csv(f"{config.output_dir}/nodes.csv")
print(f"\nExtracted {len(nodes)} nodes")
print(nodes.head())
```

**Run:**
```bash
uv run python examples/simple_billing_document.py
```

---

## Example 2: Local Processing with Ollama

**Use Case:** Process documents locally without API costs.

**File:** `examples/local_ollama.py`

```python
"""
Local document processing using Ollama.
"""

from docling_graph import run_pipeline, PipelineConfig

# Ensure Ollama is running:
# ollama serve
# ollama pull llama3:8b

config = PipelineConfig(
    source="documents/research.pdf",
    template="templates.research.Research",
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3:8b",
    processing_mode="many-to-one",
    use_chunking=True,
    llm_consolidation=False,  # Faster
    output_dir="outputs/research"
)

print("Processing with Ollama...")
try:
    run_pipeline(config)
    print("✅ Complete!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Hint: Is Ollama running? (ollama serve)")
```

**Run:**
```bash
# Start Ollama first
ollama serve

# In another terminal
uv run python examples/local_ollama.py
```

---

## Example 3: VLM Form Extraction

**Use Case:** Extract data from image forms using vision model.

**File:** `examples/vlm_form.py`

```python
"""
VLM extraction from image forms.
"""

from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="documents/id_card.jpg",
    template="templates.id_card.IDCard",
    backend="vlm",
    inference="local",  # VLM only supports local
    processing_mode="one-to-one",
    docling_config="vision",
    output_dir="outputs/id_card"
)

print("Extracting from image...")
run_pipeline(config)
print("✅ Complete!")

# Display results
import json
with open("outputs/id_card/graph.json") as f:
    graph = json.load(f)
    print(f"\nExtracted {len(graph['nodes'])} nodes")
    for node in graph['nodes'][:5]:
        print(f"  - {node['label']}: {node.get('properties', {})}")
```

**Run:**
```bash
uv run python examples/vlm_form.py
```

---

## Example 4: Rheology Research with Consolidation

**Use Case:** High-accuracy extraction from complex documents.

**File:** `examples/research_consolidation.py`

```python
"""
Rheology research extraction with LLM consolidation.
"""

import os
from docling_graph import run_pipeline, PipelineConfig

os.environ["MISTRAL_API_KEY"] = "your-api-key"

config = PipelineConfig(
    source="documents/research_paper.pdf",
    template="templates.research.Research",
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-large-latest",
    processing_mode="many-to-one",
    use_chunking=True,
    llm_consolidation=True,  # Higher accuracy
    docling_config="vision",  # Better for complex layouts
    output_dir="outputs/research"
)

print("Processing rheology research (this may take a few minutes)...")
run_pipeline(config)
print("✅ Complete!")

# Analyze results
import json
with open("outputs/research/graph_stats.json") as f:
    stats = json.load(f)
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {stats['node_count']}")
    print(f"  Edges: {stats['edge_count']}")
    print(f"  Density: {stats['density']:.3f}")
```

**Run:**
```bash
uv run python examples/research_consolidation.py
```

---

## Example 5: Batch Processing

**Use Case:** Process multiple documents with progress tracking.

**File:** `examples/batch_process.py`

```python
"""
Batch process multiple documents.
"""

from pathlib import Path
from docling_graph import run_pipeline, PipelineConfig
from tqdm import tqdm

def process_batch(input_dir: str, template: str, output_base: str):
    """Process all PDFs in a directory."""
    documents = list(Path(input_dir).glob("*.pdf"))
    results = {"success": [], "failed": []}
    
    print(f"Processing {len(documents)} documents...")
    
    for doc in tqdm(documents, desc="Processing"):
        try:
            config = PipelineConfig(
                source=str(doc),
                template=template,
                output_dir=f"{output_base}/{doc.stem}"
            )
            run_pipeline(config)
            results["success"].append(doc.name)
            
        except Exception as e:
            results["failed"].append((doc.name, str(e)))
            tqdm.write(f"❌ {doc.name}: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Completed: {len(results['success'])} succeeded")
    print(f"Failed: {len(results['failed'])}")
    
    if results["failed"]:
        print("\nFailed documents:")
        for name, error in results["failed"]:
            print(f"  - {name}: {error}")
    
    return results

if __name__ == "__main__":
    results = process_batch(
        input_dir="documents/invoices",
        template="templates.billing_document.BillingDocument",
        output_base="outputs/batch"
    )
```

**Run:**
```bash
uv run python examples/batch_process.py
```

---

## Example 6: Robust Error Handling

**Use Case:** Production-ready code with comprehensive error handling.

**File:** `examples/robust_processing.py`

```python
"""
Production-ready document processing with error handling.
"""

import logging
from pathlib import Path
from typing import Optional
from docling_graph import run_pipeline, PipelineConfig
from docling_graph.exceptions import (
    ConfigurationError,
    ExtractionError,
    PipelineError,
    DoclingGraphError
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_document(
    source: str,
    template: str,
    output_dir: Optional[str] = None,
    max_retries: int = 3
) -> bool:
    """
    Process document with retry logic and error handling.
    
    Args:
        source: Path to source document
        template: Pydantic template path
        output_dir: Output directory (auto-generated if None)
        max_retries: Maximum retry attempts
    
    Returns:
        True if successful, False otherwise
    """
    if output_dir is None:
        output_dir = f"outputs/{Path(source).stem}"
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Processing {source} (attempt {attempt}/{max_retries})")
            
            config = PipelineConfig(
                source=source,
                template=template,
                output_dir=output_dir
            )
            
            run_pipeline(config)
            logger.info(f"✅ Successfully processed: {source}")
            return True
            
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e.message}")
            if e.details:
                logger.error(f"Details: {e.details}")
            return False  # Don't retry configuration errors
            
        except ExtractionError as e:
            logger.error(f"Extraction failed: {e.message}")
            if attempt < max_retries:
                logger.info(f"Retrying... ({attempt}/{max_retries})")
                continue
            return False
            
        except PipelineError as e:
            logger.error(f"Pipeline error: {e.message}")
            if attempt < max_retries:
                logger.info(f"Retrying... ({attempt}/{max_retries})")
                continue
            return False
            
        except DoclingGraphError as e:
            logger.error(f"Docling-graph error: {e.message}")
            return False
            
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return False
    
    return False

if __name__ == "__main__":
    # Process single document
    success = process_document(
        source="documents/invoice.pdf",
        template="templates.billing_document.BillingDocument"
    )
    
    if success:
        print("Processing completed successfully")
    else:
        print("Processing failed")
        exit(1)
```

**Run:**
```bash
uv run python examples/robust_processing.py
```

---

## Example 7: Flask API Integration

**Use Case:** Web API for document processing.

**File:** `examples/flask_api.py`

```python
"""
Flask API for document processing.
"""

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid
import os

from docling_graph import run_pipeline, PipelineConfig
from docling_graph.exceptions import DoclingGraphError

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route('/process', methods=['POST'])
def process_document():
    """Process uploaded document."""
    # Validate request
    if 'document' not in request.files:
        return jsonify({"error": "No document provided"}), 400
    
    file = request.files['document']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    template = request.form.get('template', 'templates.billing_document.BillingDocument')
    
    # Save file
    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    temp_path = Path(app.config['UPLOAD_FOLDER']) / f"{job_id}_{filename}"
    file.save(temp_path)
    
    try:
        # Process document
        output_dir = Path(app.config['OUTPUT_FOLDER']) / job_id
        
        config = PipelineConfig(
            source=str(temp_path),
            template=template,
            output_dir=str(output_dir)
        )
        
        run_pipeline(config)
        
        return jsonify({
            "status": "success",
            "job_id": job_id,
            "output_dir": str(output_dir),
            "files": {
                "nodes": f"/download/{job_id}/nodes.csv",
                "edges": f"/download/{job_id}/edges.csv",
                "graph": f"/download/{job_id}/graph.json",
                "visualization": f"/download/{job_id}/graph_visualization.html"
            }
        })
        
    except DoclingGraphError as e:
        return jsonify({
            "status": "error",
            "message": e.message,
            "details": e.details
        }), 500
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
        
    finally:
        # Cleanup temp file
        temp_path.unlink(missing_ok=True)

@app.route('/download/<job_id>/<filename>', methods=['GET'])
def download_file(job_id, filename):
    """Download processed file."""
    file_path = Path(app.config['OUTPUT_FOLDER']) / job_id / filename
    
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**Run:**
```bash
uv run python examples/flask_api.py
```

**Test:**
```bash
# Upload and process document
curl -X POST http://localhost:5000/process \
    -F "document=@invoice.pdf" \
    -F "template=templates.billing_document.BillingDocument"

# Download results
curl -O http://localhost:5000/download/{job_id}/nodes.csv
```

---

## Example 8: Jupyter Notebook Analysis

**Use Case:** Interactive document analysis in Jupyter.

**File:** `examples/notebook_analysis.ipynb`

```python
# Cell 1: Setup
from docling_graph import run_pipeline, PipelineConfig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Cell 2: Process Document
config = PipelineConfig(
    source="documents/research.pdf",
    template="templates.research.Research",
    output_dir="outputs/research"
)

print("Processing document...")
run_pipeline(config)
print("✅ Complete!")

# Cell 3: Load Results
nodes = pd.read_csv("outputs/research/nodes.csv")
edges = pd.read_csv("outputs/research/edges.csv")

print(f"Nodes: {len(nodes)}")
print(f"Edges: {len(edges)}")

# Cell 4: Analyze Node Types
node_counts = nodes['type'].value_counts()
print("\nNode Type Distribution:")
print(node_counts)

# Visualize
plt.figure(figsize=(10, 6))
node_counts.plot(kind='bar')
plt.title('Node Types Distribution')
plt.xlabel('Node Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cell 5: Analyze Relationships
edge_counts = edges['type'].value_counts()
print("\nRelationship Distribution:")
print(edge_counts)

# Visualize
plt.figure(figsize=(10, 6))
edge_counts.plot(kind='bar', color='coral')
plt.title('Relationship Types Distribution')
plt.xlabel('Relationship Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cell 6: Network Analysis
import networkx as nx

# Create graph
G = nx.DiGraph()
for _, edge in edges.iterrows():
    G.add_edge(edge['source'], edge['target'], type=edge['type'])

print(f"\nNetwork Statistics:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Density: {nx.density(G):.3f}")
print(f"  Is connected: {nx.is_weakly_connected(G)}")

# Cell 7: Visualize Network
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5, iterations=50)
nx.draw(G, pos, 
        node_color='lightblue',
        node_size=500,
        with_labels=True,
        font_size=8,
        arrows=True,
        edge_color='gray',
        alpha=0.7)
plt.title('Knowledge Graph Visualization')
plt.tight_layout()
plt.show()
```

**Run:**
```bash
jupyter notebook examples/notebook_analysis.ipynb
```

---

## Best Practices

### 👍 Use Environment Variables for Secrets

```python
# ✅ Good - Environment variables
import os
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

# ❌ Avoid - Hardcoded secrets
os.environ["MISTRAL_API_KEY"] = "sk-1234..."  # Don't commit!
```

### 👍 Handle Errors Gracefully

```python
# ✅ Good - Specific error handling
from docling_graph.exceptions import ExtractionError

try:
    run_pipeline(config)
except ExtractionError as e:
    logger.error(f"Extraction failed: {e.message}")
    # Implement fallback

# ❌ Avoid - Silent failures
try:
    run_pipeline(config)
except:
    pass
```

### 👍 Organize Outputs

```python
# ✅ Good - Organized structure
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    output_dir=f"outputs/invoices/{timestamp}"
)

# ❌ Avoid - Overwriting
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    output_dir="outputs"  # Same for all
)
```

---

## Next Steps

1. **[Batch Processing →](batch-processing.md)** - Advanced batch patterns
2. **[Examples →](../examples/index.md)** - Real-world examples
3. **[Advanced Topics →](../advanced/index.md)** - Custom backends