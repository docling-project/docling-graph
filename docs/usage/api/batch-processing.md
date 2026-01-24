# Batch Processing


## Overview

**Batch processing** enables efficient processing of multiple documents with progress tracking, error handling, and result aggregation.

**Key Features:**
- Parallel processing
- Progress tracking
- Error recovery
- Result aggregation
- Resource management

---

## Basic Batch Processing

### Simple Loop

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
        print(f"✓ {doc.name}")
    except Exception as e:
        print(f"✗ {doc.name}: {e}")
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
        template="templates.Invoice",
        output_dir=f"outputs/{doc.stem}"
    )
    
    try:
        config.run()
    except Exception as e:
        tqdm.write(f"✗ {doc.name}: {e}")
```

**Install tqdm:**
```bash
uv add tqdm
```

---

## Error Handling

### Comprehensive Error Tracking

```python
from pathlib import Path
from docling_graph import PipelineConfig
from docling_graph.exceptions import DoclingGraphError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def batch_process(input_dir: str, template: str, output_base: str):
    """Process documents with error tracking."""
    documents = list(Path(input_dir).glob("*.pdf"))
    results = {
        "success": [],
        "failed": [],
        "skipped": []
    }
    
    for doc in documents:
        # Skip if already processed
        output_dir = Path(output_base) / doc.stem
        if output_dir.exists():
            results["skipped"].append(doc.name)
            logger.info(f"⊘ Skipped (already processed): {doc.name}")
            continue
        
        try:
            config = PipelineConfig(
                source=str(doc),
                template=template,
                output_dir=str(output_dir)
            )
            
            config.run()
            results["success"].append(doc.name)
            logger.info(f"✓ Success: {doc.name}")
            
        except DoclingGraphError as e:
            results["failed"].append({
                "document": doc.name,
                "error": e.message,
                "details": e.details
            })
            logger.error(f"✗ Failed: {doc.name} - {e.message}")
            
        except Exception as e:
            results["failed"].append({
                "document": doc.name,
                "error": str(e),
                "details": None
            })
            logger.exception(f"✗ Unexpected error: {doc.name}")
    
    # Summary
    total = len(documents)
    logger.info(f"\n{'='*50}")
    logger.info(f"Total: {total}")
    logger.info(f"Success: {len(results['success'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    logger.info(f"Skipped: {len(results['skipped'])}")
    
    return results

# Run batch processing
results = batch_process(
    input_dir="documents/invoices",
    template="templates.invoice.Invoice",
    output_base="outputs/batch"
)
```

---

## Parallel Processing

### Using ThreadPoolExecutor

```python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from docling_graph import PipelineConfig
from tqdm import tqdm

def process_document(doc_path: Path, template: str, output_base: str):
    """Process single document."""
    try:
        config = PipelineConfig(
            source=str(doc_path),
            template=template,
            output_dir=f"{output_base}/{doc_path.stem}"
        )
        config.run()
        return {"status": "success", "document": doc_path.name}
    except Exception as e:
        return {"status": "error", "document": doc_path.name, "error": str(e)}

def parallel_batch_process(
    input_dir: str,
    template: str,
    output_base: str,
    max_workers: int = 4
):
    """Process documents in parallel."""
    documents = list(Path(input_dir).glob("*.pdf"))
    results = {"success": [], "failed": []}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_document, doc, template, output_base): doc
            for doc in documents
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(documents), desc="Processing"):
            result = future.result()
            
            if result["status"] == "success":
                results["success"].append(result["document"])
            else:
                results["failed"].append({
                    "document": result["document"],
                    "error": result["error"]
                })
    
    # Summary
    print(f"\nCompleted: {len(results['success'])} succeeded, {len(results['failed'])} failed")
    return results

# Run parallel processing
results = parallel_batch_process(
    input_dir="documents/invoices",
    template="templates.invoice.Invoice",
    output_base="outputs/parallel",
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
            config.run()
            
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
    template="templates.invoice.Invoice",
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
            template = "templates.invoice.Invoice"
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
            config.run()
            print(f"✓ {doc.name}")
        except Exception as e:
            print(f"✗ {doc.name}: {e}")

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
            config.run()
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
        
        status = "✓" if result["status"] == "success" else "✗"
        print(f"{status} {doc.name} (attempts: {result['attempts']})")
    
    return results

results = batch_with_retry(
    input_dir="documents/invoices",
    template="templates.invoice.Invoice",
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
            config.run()
            
            # Update checkpoint
            checkpoint["processed"].append(doc.name)
            print(f"✓ {doc.name}")
            
        except Exception as e:
            checkpoint["failed"].append({
                "document": doc.name,
                "error": str(e)
            })
            print(f"✗ {doc.name}: {e}")
        
        # Save checkpoint after each document
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    print(f"\nProcessed: {len(checkpoint['processed'])}")
    print(f"Failed: {len(checkpoint['failed'])}")
    
    return checkpoint

# Run with checkpoint
checkpoint = batch_with_checkpoint(
    input_dir="documents/invoices",
    template="templates.invoice.Invoice",
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
            config.run()
            print(f"✓ {doc.name}")
        except Exception as e:
            print(f"✗ {doc.name}: {e}")
        
        # Periodic cleanup
        if i % cleanup_interval == 0:
            gc.collect()
            print(f"[Cleanup after {i} documents]")

batch_with_memory_management(
    input_dir="documents/large_batch",
    template="templates.invoice.Invoice",
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
                
                config.run()
                
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
        template="templates.invoice.Invoice",
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

### 1. Use Progress Tracking

```python
# ✅ Good - Visual progress
from tqdm import tqdm

for doc in tqdm(documents, desc="Processing"):
    config.run()

# ❌ Avoid - No feedback
for doc in documents:
    config.run()
```

### 2. Implement Error Recovery

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

### 3. Aggregate Results

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

---

## Quick Reference

### Basic Batch

```python
for doc in Path("documents").glob("*.pdf"):
    config = PipelineConfig(
        source=str(doc),
        template="templates.Invoice",
        output_dir=f"outputs/{doc.stem}"
    )
    config.run()
```

### With Progress

```python
from tqdm import tqdm

for doc in tqdm(documents, desc="Processing"):
    config.run()
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process, doc) for doc in documents]
    for future in as_completed(futures):
        result = future.result()
```