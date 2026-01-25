# Batch Processing


## Overview

**Batch processing** enables efficient processing of multiple documents with progress tracking, error handling, and result aggregation.

**Key Features:**
- Memory-efficient processing (no file exports by default)
- Parallel processing
- Progress tracking
- Error recovery
- Result aggregation
- Resource management

---

## Basic Batch Processing

### Simple Loop (Memory-Efficient)

```python
from pathlib import Path
from docling_graph import run_pipeline, PipelineConfig

documents = Path("documents").glob("*.pdf")
results = []

for doc in documents:
    config = PipelineConfig(
        source=str(doc),
        template="templates.BillingDocument"
    )
    
    try:
        # Process without file exports - memory efficient
        context = run_pipeline(config)
        results.append({
            "filename": doc.name,
            "nodes": context.knowledge_graph.number_of_nodes(),
            "edges": context.knowledge_graph.number_of_edges()
        })
        print(f"✅ {doc.name}: {results[-1]['nodes']} nodes")
    except Exception as e:
        print(f"❌ {doc.name}: {e}")

# Summary
print(f"\nTotal documents: {len(results)}")
print(f"Total nodes: {sum(r['nodes'] for r in results)}")
```

### With File Exports

```python
from pathlib import Path
from docling_graph import run_pipeline, PipelineConfig

documents = Path("documents").glob("*.pdf")

for doc in documents:
    config = PipelineConfig(
        source=str(doc),
        template="templates.BillingDocument",
        dump_to_disk=True,  # Enable file exports
        output_dir=f"outputs/{doc.stem}"
    )
    
    try:
        context = run_pipeline(config)
        print(f"✅ {doc.name}")
    except Exception as e:
        print(f"❌ {doc.name}: {e}")
```

---

## Progress Tracking

### Using tqdm

```python
from pathlib import Path
from docling_graph import PipelineConfig
from tqdm import tqdm

documents = list(Path("documents").glob("*.pdf"))

for doc in tqdm(documents, desc="Processing"):
    config = PipelineConfig(
        source=str(doc),
        template="templates.BillingDocument",
        output_dir=f"outputs/{doc.stem}"
    )
    
    try:
        run_pipeline(config)
    except Exception as e:
        tqdm.write(f"❌ {doc.name}: {e}")
```

**Install tqdm:**
```bash
uv add tqdm
```

---

## Error Handling

### Comprehensive Error Tracking (Memory-Efficient)

```python
from pathlib import Path
from docling_graph import run_pipeline, PipelineConfig
from docling_graph.exceptions import DoclingGraphError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def batch_process(input_dir: str, template: str):
    """Process documents with error tracking - no file exports."""
    documents = list(Path(input_dir).glob("*.pdf"))
    results = {
        "success": [],
        "failed": [],
        "graphs": []
    }
    
    for doc in documents:
        try:
            config = PipelineConfig(
                source=str(doc),
                template=template
            )
            
            # Process without file exports
            context = run_pipeline(config)
            
            # Store results in memory
            results["success"].append(doc.name)
            results["graphs"].append({
                "filename": doc.name,
                "graph": context.knowledge_graph,
                "model": context.pydantic_model
            })
            logger.info(f"✅ Success: {doc.name} ({context.knowledge_graph.number_of_nodes()} nodes)")
            
        except DoclingGraphError as e:
            results["failed"].append({
                "document": doc.name,
                "error": e.message,
                "details": e.details
            })
            logger.error(f"❌ Failed: {doc.name} - {e.message}")
            
        except Exception as e:
            results["failed"].append({
                "document": doc.name,
                "error": str(e),
                "details": None
            })
            logger.exception(f"❌ Unexpected error: {doc.name}")
    
    # Summary
    total = len(documents)
    logger.info(f"\n{'='*50}")
    logger.info(f"Total: {total}")
    logger.info(f"Success: {len(results['success'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    logger.info(f"Total nodes: {sum(g['graph'].number_of_nodes() for g in results['graphs'])}")
    
    return results

# Run batch processing
results = batch_process(
    input_dir="documents/invoices",
    template="templates.billing_document.BillingDocument"
)
```

---

## Parallel Processing

### Using ThreadPoolExecutor (Memory-Efficient)

```python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from docling_graph import run_pipeline, PipelineConfig
from tqdm import tqdm

def process_document(doc_path: Path, template: str):
    """Process single document without file exports."""
    try:
        config = PipelineConfig(
            source=str(doc_path),
            template=template
        )
        context = run_pipeline(config)
        return {
            "status": "success",
            "document": doc_path.name,
            "nodes": context.knowledge_graph.number_of_nodes(),
            "edges": context.knowledge_graph.number_of_edges(),
            "graph": context.knowledge_graph,
            "model": context.pydantic_model
        }
    except Exception as e:
        return {"status": "error", "document": doc_path.name, "error": str(e)}

def parallel_batch_process(
    input_dir: str,
    template: str,
    max_workers: int = 4
):
    """Process documents in parallel - memory efficient."""
    documents = list(Path(input_dir).glob("*.pdf"))
    results = {"success": [], "failed": [], "graphs": []}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_document, doc, template): doc
            for doc in documents
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(documents), desc="Processing"):
            result = future.result()
            
            if result["status"] == "success":
                results["success"].append(result["document"])
                results["graphs"].append({
                    "filename": result["document"],
                    "graph": result["graph"],
                    "model": result["model"],
                    "nodes": result["nodes"],
                    "edges": result["edges"]
                })
            else:
                results["failed"].append({
                    "document": result["document"],
                    "error": result["error"]
                })
    
    # Summary
    total_nodes = sum(g["nodes"] for g in results["graphs"])
    total_edges = sum(g["edges"] for g in results["graphs"])
    print(f"\nCompleted: {len(results['success'])} succeeded, {len(results['failed'])} failed")
    print(f"Total entities: {total_nodes} nodes, {total_edges} edges")
    return results

# Run parallel processing
results = parallel_batch_process(
    input_dir="documents/invoices",
    template="templates.billing_document.BillingDocument",
    max_workers=4
)
```

---

## Result Aggregation

### Collecting Statistics

```python
from pathlib import Path
import json
import pandas as pd
from docling_graph import PipelineConfig

def batch_with_stats(input_dir: str, template: str, output_base: str):
    """Process documents and collect statistics."""
    documents = list(Path(input_dir).glob("*.pdf"))
    all_stats = []
    
    for doc in documents:
        output_dir = Path(output_base) / doc.stem
        
        try:
            # Process document
            config = PipelineConfig(
                source=str(doc),
                template=template,
                output_dir=str(output_dir)
            )
            run_pipeline(config)
            
            # Load statistics
            stats_file = output_dir / "graph_stats.json"
            with open(stats_file) as f:
                stats = json.load(f)
                stats["document"] = doc.name
                stats["status"] = "success"
                all_stats.append(stats)
                
        except Exception as e:
            all_stats.append({
                "document": doc.name,
                "status": "error",
                "error": str(e)
            })
    
    # Create summary DataFrame
    df = pd.DataFrame(all_stats)
    
    # Save summary
    summary_file = Path(output_base) / "batch_summary.csv"
    df.to_csv(summary_file, index=False)
    
    # Print statistics
    print("\n=== Batch Statistics ===")
    print(f"Total documents: {len(df)}")
    print(f"Successful: {(df['status'] == 'success').sum()}")
    print(f"Failed: {(df['status'] == 'error').sum()}")
    
    if 'node_count' in df.columns:
        successful = df[df['status'] == 'success']
        print(f"\nAverage nodes: {successful['node_count'].mean():.1f}")
        print(f"Average edges: {successful['edge_count'].mean():.1f}")
        print(f"Average density: {successful['density'].mean():.3f}")
    
    return df

# Run with statistics
df = batch_with_stats(
    input_dir="documents/invoices",
    template="templates.billing_document.BillingDocument",
    output_base="outputs/batch_stats"
)

# Analyze results
print("\nTop 5 documents by node count:")
print(df.nlargest(5, 'node_count')[['document', 'node_count', 'edge_count']])
```

---

## Advanced Patterns

### Pattern 1: Conditional Processing

```python
from pathlib import Path
from docling_graph import PipelineConfig

def smart_batch_process(input_dir: str, output_base: str):
    """Process documents with template selection."""
    documents = Path(input_dir).glob("*")
    
    for doc in documents:
        # Determine template based on filename
        if "invoice" in doc.name.lower():
            template = "templates.billing_document.BillingDocument"
            backend = "vlm"
        elif "research" in doc.name.lower():
            template = "templates.research.Research"
            backend = "llm"
        else:
            print(f"⊘ Skipped (unknown type): {doc.name}")
            continue
        
        # Process with appropriate config
        config = PipelineConfig(
            source=str(doc),
            template=template,
            backend=backend,
            output_dir=f"{output_base}/{doc.stem}"
        )
        
        try:
            run_pipeline(config)
            print(f"✅ {doc.name}")
        except Exception as e:
            print(f"❌ {doc.name}: {e}")

smart_batch_process("documents/mixed", "outputs/smart")
```

---

### Pattern 2: Retry Logic

```python
from pathlib import Path
from docling_graph import PipelineConfig
import time

def process_with_retry(
    doc_path: Path,
    template: str,
    output_dir: str,
    max_retries: int = 3,
    delay: int = 5
):
    """Process document with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            config = PipelineConfig(
                source=str(doc_path),
                template=template,
                output_dir=output_dir
            )
            run_pipeline(config)
            return {"status": "success", "attempts": attempt}
            
        except Exception as e:
            if attempt < max_retries:
                print(f"Attempt {attempt} failed, retrying in {delay}s...")
                time.sleep(delay)
            else:
                return {
                    "status": "error",
                    "attempts": attempt,
                    "error": str(e)
                }

def batch_with_retry(input_dir: str, template: str, output_base: str):
    """Batch process with retry logic."""
    documents = list(Path(input_dir).glob("*.pdf"))
    results = []
    
    for doc in documents:
        result = process_with_retry(
            doc_path=doc,
            template=template,
            output_dir=f"{output_base}/{doc.stem}",
            max_retries=3
        )
        result["document"] = doc.name
        results.append(result)
        
        status = "✅" if result["status"] == "success" else "❌"
        print(f"{status} {doc.name} (attempts: {result['attempts']})")
    
    return results

results = batch_with_retry(
    input_dir="documents/invoices",
    template="templates.billing_document.BillingDocument",
    output_base="outputs/retry"
)
```

---

### Pattern 3: Checkpoint and Resume

```python
from pathlib import Path
import json
from docling_graph import PipelineConfig

def batch_with_checkpoint(
    input_dir: str,
    template: str,
    output_base: str,
    checkpoint_file: str = "checkpoint.json"
):
    """Batch process with checkpoint support."""
    documents = list(Path(input_dir).glob("*.pdf"))
    checkpoint_path = Path(output_base) / checkpoint_file
    
    # Load checkpoint
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        processed = set(checkpoint.get("processed", []))
        print(f"Resuming from checkpoint: {len(processed)} already processed")
    else:
        processed = set()
        checkpoint = {"processed": [], "failed": []}
    
    # Process remaining documents
    for doc in documents:
        if doc.name in processed:
            print(f"⊘ Skipped (already processed): {doc.name}")
            continue
        
        try:
            config = PipelineConfig(
                source=str(doc),
                template=template,
                output_dir=f"{output_base}/{doc.stem}"
            )
            run_pipeline(config)
            
            # Update checkpoint
            checkpoint["processed"].append(doc.name)
            print(f"✅ {doc.name}")
            
        except Exception as e:
            checkpoint["failed"].append({
                "document": doc.name,
                "error": str(e)
            })
            print(f"❌ {doc.name}: {e}")
        
        # Save checkpoint after each document
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    print(f"\nProcessed: {len(checkpoint['processed'])}")
    print(f"Failed: {len(checkpoint['failed'])}")
    
    return checkpoint

# Run with checkpoint
checkpoint = batch_with_checkpoint(
    input_dir="documents/invoices",
    template="templates.billing_document.BillingDocument",
    output_base="outputs/checkpoint"
)
```

---

## Resource Management

### Memory Management

```python
from pathlib import Path
from docling_graph import PipelineConfig
import gc

def batch_with_memory_management(
    input_dir: str,
    template: str,
    output_base: str,
    cleanup_interval: int = 10
):
    """Batch process with memory cleanup."""
    documents = list(Path(input_dir).glob("*.pdf"))
    
    for i, doc in enumerate(documents, 1):
        config = PipelineConfig(
            source=str(doc),
            template=template,
            output_dir=f"{output_base}/{doc.stem}"
        )
        
        try:
            run_pipeline(config)
            print(f"✅ {doc.name}")
        except Exception as e:
            print(f"❌ {doc.name}: {e}")
        
        # Periodic cleanup
        if i % cleanup_interval == 0:
            gc.collect()
            print(f"[Cleanup after {i} documents]")

batch_with_memory_management(
    input_dir="documents/large_batch",
    template="templates.billing_document.BillingDocument",
    output_base="outputs/memory_managed",
    cleanup_interval=10
)
```

---

## Complete Example

### Production-Ready Batch Processor

```python
"""
Production-ready batch processor with all features.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from docling_graph import PipelineConfig
from docling_graph.exceptions import DoclingGraphError
import json
import logging
from datetime import datetime
from tqdm import tqdm
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Production-ready batch document processor."""
    
    def __init__(
        self,
        input_dir: str,
        template: str,
        output_base: str,
        max_workers: int = 4,
        max_retries: int = 3
    ):
        self.input_dir = Path(input_dir)
        self.template = template
        self.output_base = Path(output_base)
        self.max_workers = max_workers
        self.max_retries = max_retries
        
        # Create output directory
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint
        self.checkpoint_file = self.output_base / "checkpoint.json"
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load processing checkpoint."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                self.checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint: {len(self.checkpoint['processed'])} processed")
        else:
            self.checkpoint = {
                "processed": [],
                "failed": [],
                "started_at": datetime.now().isoformat()
            }
    
    def save_checkpoint(self):
        """Save processing checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def process_document(self, doc_path: Path):
        """Process single document with retry logic."""
        # Skip if already processed
        if doc_path.name in self.checkpoint["processed"]:
            return {"status": "skipped", "document": doc_path.name}
        
        # Retry loop
        for attempt in range(1, self.max_retries + 1):
            try:
                config = PipelineConfig(
                    source=str(doc_path),
                    template=self.template,
                    output_dir=str(self.output_base / doc_path.stem)
                )
                
                run_pipeline(config)
                
                # Load statistics
                stats_file = self.output_base / doc_path.stem / "graph_stats.json"
                with open(stats_file) as f:
                    stats = json.load(f)
                
                return {
                    "status": "success",
                    "document": doc_path.name,
                    "attempts": attempt,
                    **stats
                }
                
            except DoclingGraphError as e:
                if attempt < self.max_retries:
                    logger.warning(f"Attempt {attempt} failed for {doc_path.name}, retrying...")
                    continue
                else:
                    return {
                        "status": "error",
                        "document": doc_path.name,
                        "attempts": attempt,
                        "error": e.message
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "document": doc_path.name,
                    "attempts": attempt,
                    "error": str(e)
                }
    
    def process_batch(self):
        """Process all documents in batch."""
        documents = list(self.input_dir.glob("*.pdf"))
        logger.info(f"Found {len(documents)} documents to process")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_document, doc): doc
                for doc in documents
            }
            
            for future in tqdm(as_completed(futures), total=len(documents), desc="Processing"):
                result = future.result()
                results.append(result)
                
                # Update checkpoint
                if result["status"] == "success":
                    self.checkpoint["processed"].append(result["document"])
                elif result["status"] == "error":
                    self.checkpoint["failed"].append({
                        "document": result["document"],
                        "error": result["error"]
                    })
                
                self.save_checkpoint()
        
        # Generate summary
        self.generate_summary(results)
        
        return results
    
    def generate_summary(self, results):
        """Generate processing summary."""
        df = pd.DataFrame(results)
        
        # Save detailed results
        summary_file = self.output_base / "batch_results.csv"
        df.to_csv(summary_file, index=False)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("="*50)
        logger.info(f"Total documents: {len(df)}")
        logger.info(f"Successful: {(df['status'] == 'success').sum()}")
        logger.info(f"Failed: {(df['status'] == 'error').sum()}")
        logger.info(f"Skipped: {(df['status'] == 'skipped').sum()}")
        
        if 'node_count' in df.columns:
            successful = df[df['status'] == 'success']
            if len(successful) > 0:
                logger.info(f"\nAverage nodes: {successful['node_count'].mean():.1f}")
                logger.info(f"Average edges: {successful['edge_count'].mean():.1f}")
                logger.info(f"Average density: {successful['density'].mean():.3f}")
        
        logger.info(f"\nResults saved to: {summary_file}")

# Usage
if __name__ == "__main__":
    processor = BatchProcessor(
        input_dir="documents/invoices",
        template="templates.billing_document.BillingDocument",
        output_base="outputs/production_batch",
        max_workers=4,
        max_retries=3
    )
    
    results = processor.process_batch()
```

**Run:**
```bash
uv run python batch_processor.py
```

---

## Best Practices

### 👍 Use Progress Tracking

```python
# ✅ Good - Visual progress
from tqdm import tqdm

for doc in tqdm(documents, desc="Processing"):
    run_pipeline(config)

# ❌ Avoid - No feedback
for doc in documents:
    run_pipeline(config)
```

### 👍 Implement Error Recovery

```python
# ✅ Good - Checkpoint and resume
checkpoint = load_checkpoint()
for doc in documents:
    if doc.name not in checkpoint["processed"]:
        process(doc)
        checkpoint["processed"].append(doc.name)
        save_checkpoint(checkpoint)

# ❌ Avoid - Start from scratch on failure
for doc in documents:
    process(doc)
```

### 👍 Aggregate Results

```python
# ✅ Good - Collect statistics
results = []
for doc in documents:
    result = process(doc)
    results.append(result)

df = pd.DataFrame(results)
df.to_csv("summary.csv")

# ❌ Avoid - No summary
for doc in documents:
    process(doc)
```

---

## Next Steps

1. **[Examples →](../examples/index.md)** - Real-world examples
2. **[Advanced Topics →](../advanced/index.md)** - Custom backends
3. **[API Reference →](../../reference/index.md)** - Complete API docs