# Complete Configuration Examples


## Overview

This guide provides **complete, real-world configuration examples** for common use cases. Each example includes full configuration, expected outputs, and integration patterns.

**In this guide:**
- Production-ready configurations
- Common use case patterns
- Integration examples
- Troubleshooting scenarios
- Best practices

---

## Quick Navigation

| Use Case | Backend | Inference | Processing |
|:---------|:--------|:----------|:-----------|
| [Local Development](#example-1-local-development) | LLM | Local | Many-to-one |
| [Production API](#example-2-production-api) | LLM | Remote | Many-to-one |
| [High Accuracy](#example-3-high-accuracy-extraction) | VLM | Local | One-to-one |
| [Batch Processing](#example-4-batch-processing) | LLM | Remote | Many-to-one |
| [Rheology Researchs](#example-5-rheology-researchs) | LLM | Remote | Many-to-one |
| [Invoices](#example-6-billing-document-processing) | LLM | Local | Many-to-one |
| [Forms](#example-7-form-extraction) | VLM | Local | One-to-one |
| [Multi-language](#example-8-multi-language-documents) | LLM | Remote | Many-to-one |

---

## Example 1: Local Development

### Use Case
Fast iteration during template development using local models.

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    # Source and template
    source="test_document.pdf",
    template="templates.BillingDocument",
    
    # Local LLM for fast iteration
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.1:8b",
    
    # Fast processing
    processing_mode="many-to-one",
    docling_config="ocr",
    use_chunking=True,
    
    # CSV for easy inspection
    export_format="csv",
    export_markdown=True,
    
    # Development output
    output_dir="dev_outputs"
)

run_pipeline(config)
```

### Prerequisites

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3.1:8b

# Verify
ollama list
```

### Expected Output

```
dev_outputs/
├── nodes.csv              # Easy to inspect in Excel
├── edges.csv
├── document.md            # Check markdown conversion
├── graph_stats.json       # Quick metrics
└── visualization.html     # Visual inspection
```

### When to Use

✅ **Use for:**
- Template development
- Quick testing
- Debugging extraction
- Offline development

---

## Example 2: Production API

### Use Case
Production deployment using remote API for reliability and scale.

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig
import os

config = PipelineConfig(
    # Source and template
    source="production_document.pdf",
    template="templates.BillingDocument",
    
    # Remote API for reliability
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-large-latest",
    
    # Robust processing
    processing_mode="many-to-one",
    docling_config="ocr",
    use_chunking=True,
    llm_consolidation=True,  # Extra accuracy
    
    # Cypher for Neo4j
    export_format="cypher",
    export_docling=False,  # Minimal exports
    export_markdown=False,
    
    # Production output
    output_dir=f"production/{os.getenv('DOCUMENT_ID')}"
)

# Set API key
os.environ["MISTRAL_API_KEY"] = "your_api_key"

run_pipeline(config)
```

### Environment Setup

```bash
# .env file
MISTRAL_API_KEY=your_api_key_here
DOCUMENT_ID=doc_12345

# Load environment
uv run python -c "from dotenv import load_dotenv; load_dotenv()"
```

### Expected Output

```
production/doc_12345/
├── graph.cypher           # Ready for Neo4j import
├── graph_data.json        # Backup
├── graph_stats.json       # Metrics
└── visualization.html     # QA check
```

### Integration

```python
# Import to Neo4j
import subprocess

cypher_file = f"production/{os.getenv('DOCUMENT_ID')}/graph.cypher"
subprocess.run([
    "cypher-shell",
    "-u", "neo4j",
    "-p", os.getenv("NEO4J_PASSWORD"),
    "-f", cypher_file
])
```

### When to Use

✅ **Use for:**
- Production deployments
- High reliability needs
- Scalable processing
- API-based workflows

---

## Example 3: High Accuracy Extraction

### Use Case
Maximum accuracy for complex documents using VLM.

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    # Source and template
    source="complex_document.pdf",
    template="templates.ResearchPaper",
    
    # VLM for visual understanding
    backend="vlm",
    inference="local",
    model_override="numind/NuExtract-2.0-8B",
    
    # Page-by-page for accuracy
    processing_mode="one-to-one",
    docling_config="vision",  # Vision pipeline
    
    # No chunking for VLM
    use_chunking=False,
    
    # CSV for analysis
    export_format="csv",
    export_per_page_markdown=True,  # Debug per page
    
    output_dir="high_accuracy_outputs"
)

run_pipeline(config)
```

### Prerequisites

```bash
# GPU required
nvidia-smi

# Install with GPU support
uv pip install "docling-graph[gpu]"
```

### Expected Output

```
high_accuracy_outputs/
├── nodes.csv
├── edges.csv
├── pages/
│   ├── page_001.md       # Per-page markdown
│   ├── page_002.md
│   └── ...
└── visualization.html
```

### When to Use

✅ **Use for:**
- Complex layouts
- Visual elements
- High accuracy needs
- Rheology researchs

---

## Example 4: Batch Processing

### Use Case
Process multiple documents efficiently.

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_batch(input_dir: str, template: str):
    """Process all PDFs in a directory."""
    
    input_path = Path(input_dir)
    results = []
    
    for pdf_file in input_path.glob("*.pdf"):
        logger.info(f"Processing {pdf_file.name}")
        
        try:
            config = PipelineConfig(
                source=str(pdf_file),
                template=template,
                
                # Remote for reliability
                backend="llm",
                inference="remote",
                provider_override="mistral",
                
                # Efficient processing
                processing_mode="many-to-one",
                use_chunking=True,
                
                # Organized outputs
                output_dir=f"batch_outputs/{pdf_file.stem}",
                export_format="csv"
            )
            
            run_pipeline(config)
            results.append({"file": pdf_file.name, "status": "success"})
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")
            results.append({"file": pdf_file.name, "status": "failed", "error": str(e)})
    
    return results

# Process batch
results = process_batch("documents/invoices", "templates.BillingDocument")

# Summary
success_count = sum(1 for r in results if r["status"] == "success")
print(f"Processed {success_count}/{len(results)} documents successfully")
```

### Expected Output

```
batch_outputs/
├── invoice_001/
│   ├── nodes.csv
│   └── edges.csv
├── invoice_002/
│   ├── nodes.csv
│   └── edges.csv
└── ...
```

### When to Use

✅ **Use for:**
- Multiple documents
- Automated workflows
- Scheduled processing
- Data pipelines

---

## Example 5: Rheology Researchs

### Use Case
Extract structured data from academic papers.

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    # Rheology research
    source="research_paper.pdf",
    template="templates.ResearchPaper",
    
    # Remote LLM for understanding
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo",
    
    # Full document context
    processing_mode="many-to-one",
    docling_config="ocr",
    use_chunking=True,
    llm_consolidation=True,  # Merge findings
    
    # CSV for analysis
    export_format="csv",
    export_markdown=True,
    
    output_dir="research_outputs"
)

run_pipeline(config)
```

### Template Example

```python
from pydantic import BaseModel, Field
from typing import List

class Author(BaseModel):
    """Rheology research author."""
    name: str
    affiliation: str | None = None
    email: str | None = None

class Citation(BaseModel):
    """Paper citation."""
    title: str
    authors: str
    year: int | None = None

class ResearchPaper(BaseModel):
    """Rheology research extraction template."""
    
    title: str = Field(description="Paper title")
    authors: List[Author] = Field(description="Paper authors")
    abstract: str = Field(description="Paper abstract")
    
    keywords: List[str] = Field(description="Paper keywords")
    methodology: str = Field(description="Research methodology")
    findings: List[str] = Field(description="Key findings")
    
    citations: List[Citation] = Field(description="Referenced papers")
```

### When to Use

✅ **Use for:**
- Academic papers
- Literature reviews
- Citation extraction
- Research analysis

---

## Example 6: Billing Document Processing

### Use Case
Extract invoice data for accounting systems.

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    # BillingDocument document
    source="invoice.pdf",
    template="templates.BillingDocument",
    
    # Local for cost efficiency
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.1:8b",
    
    # Fast processing
    processing_mode="many-to-one",
    docling_config="ocr",
    use_chunking=True,
    
    # CSV for accounting software
    export_format="csv",
    
    output_dir="invoices/processed"
)

run_pipeline(config)
```

### Template Example

```python
from pydantic import BaseModel, Field
from typing import List
from datetime import date

class LineItem(BaseModel):
    """Invoice line item."""
    description: str
    quantity: float
    unit_price: float
    total: float

class Address(BaseModel):
    """Address information."""
    street: str
    city: str
    postal_code: str
    country: str

class Organization(BaseModel):
    """Organization details."""
    name: str
    address: Address
    tax_id: str | None = None

class BillingDocument(BaseModel):
    """BillingDocument extraction template."""
    
    document_no: str = Field(description="Invoice number")
    invoice_date: date = Field(description="Invoice date")
    due_date: date | None = Field(description="Payment due date")
    
    issued_by: Organization = Field(description="Issuing organization")
    sent_to: Organization = Field(description="Recipient organization")
    
    line_items: List[LineItem] = Field(description="Invoice line items")
    
    subtotal: float = Field(description="Subtotal amount")
    tax: float = Field(description="Tax amount")
    total: float = Field(description="Total amount")
```

### Integration with Accounting

```python
import pandas as pd

# Load extracted data
nodes = pd.read_csv("invoices/processed/nodes.csv")

# Filter invoices
invoices = nodes[nodes['node_type'] == 'BillingDocument']

# Export to accounting system
invoices.to_csv("accounting_import.csv", index=False)
```

### When to Use

✅ **Use for:**
- Invoice processing
- Accounting automation
- Financial data extraction
- ERP integration

---

## Example 7: Form Extraction

### Use Case
Extract data from structured forms using VLM.

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    # Form document
    source="application_form.pdf",
    template="templates.ApplicationForm",
    
    # VLM for form structure
    backend="vlm",
    inference="local",
    
    # Page-by-page for forms
    processing_mode="one-to-one",
    docling_config="vision",
    use_chunking=False,
    
    # CSV for database import
    export_format="csv",
    
    output_dir="forms/processed"
)

run_pipeline(config)
```

### Template Example

```python
from pydantic import BaseModel, Field, EmailStr
from datetime import date

class Applicant(BaseModel):
    """Applicant information."""
    first_name: str
    last_name: str
    email: EmailStr
    phone: str
    date_of_birth: date

class Address(BaseModel):
    """Address information."""
    street: str
    city: str
    state: str
    zip_code: str

class ApplicationForm(BaseModel):
    """Application form template."""
    
    application_id: str = Field(description="Application ID")
    submission_date: date = Field(description="Submission date")
    
    applicant: Applicant = Field(description="Applicant details")
    address: Address = Field(description="Applicant address")
    
    employment_status: str = Field(description="Employment status")
    annual_income: float | None = Field(description="Annual income")
```

### When to Use

✅ **Use for:**
- Application forms
- Registration forms
- Survey responses
- Structured documents

---

## Example 8: Multi-language Documents

### Use Case
Process documents in multiple languages.

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    # Multi-language document
    source="multilingual_document.pdf",
    template="templates.Contract",
    
    # Remote LLM with multi-language support
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo",  # Good multi-language support
    
    # Full document context
    processing_mode="many-to-one",
    docling_config="ocr",
    use_chunking=True,
    
    # CSV export
    export_format="csv",
    
    output_dir="multilingual_outputs"
)

run_pipeline(config)
```

### When to Use

✅ **Use for:**
- International documents
- Multi-language contracts
- Global operations
- Translation workflows

---

## Advanced Patterns

### Pattern 1: Error Handling

```python
from docling_graph import run_pipeline, PipelineConfig
from docling_graph.exceptions import PipelineError, ExtractionError
import logging

logger = logging.getLogger(__name__)

def safe_process(source: str, template: str) -> bool:
    """Process document with error handling."""
    
    try:
        config = PipelineConfig(
            source=source,
            template=template,
            backend="llm",
            inference="remote"
        )
        
        run_pipeline(config)
        logger.info(f"Successfully processed {source}")
        return True
        
    except ExtractionError as e:
        logger.error(f"Extraction failed for {source}: {e}")
        return False
        
    except PipelineError as e:
        logger.error(f"Pipeline error for {source}: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error for {source}: {e}")
        return False
```

### Pattern 2: Configuration Validation

```python
from docling_graph import run_pipeline, PipelineConfig
from pydantic import ValidationError

def validate_config(config_dict: dict) -> bool:
    """Validate configuration before running."""
    
    try:
        config = PipelineConfig(**config_dict)
        print("✅ Configuration valid")
        return True
        
    except ValidationError as e:
        print(f"❌ Configuration invalid:")
        for error in e.errors():
            print(f"  - {error['loc']}: {error['msg']}")
        return False

# Test configuration
config_dict = {
    "source": "document.pdf",
    "template": "templates.BillingDocument",
    "backend": "llm",
    "inference": "remote"
}

if validate_config(config_dict):
    config = PipelineConfig(**config_dict)
    run_pipeline(config)
```

### Pattern 3: Dynamic Configuration

```python
from docling_graph import run_pipeline, PipelineConfig
import os

def get_config_for_document(doc_path: str) -> PipelineConfig:
    """Generate configuration based on document type."""
    
    # Determine document type
    if "invoice" in doc_path.lower():
        template = "templates.BillingDocument"
        processing_mode = "many-to-one"
        
    elif "form" in doc_path.lower():
        template = "templates.Form"
        processing_mode = "one-to-one"
        
    else:
        template = "templates.Generic"
        processing_mode = "many-to-one"
    
    # Choose backend based on environment
    if os.getenv("USE_GPU") == "true":
        backend = "vlm"
        inference = "local"
    else:
        backend = "llm"
        inference = "remote"
    
    return PipelineConfig(
        source=doc_path,
        template=template,
        backend=backend,
        inference=inference,
        processing_mode=processing_mode
    )

# Use dynamic configuration
config = get_config_for_document("documents/invoice_001.pdf")
run_pipeline(config)
```

---

## Best Practices Summary

### 👍 Choose the Right Backend

```python
# ✅ Good - Match backend to use case
if document_has_complex_layout:
    backend = "vlm"
elif need_fast_iteration:
    backend = "llm"
    inference = "local"
else:
    backend = "llm"
    inference = "remote"
```

### 👍 Organize Outputs

```python
# ✅ Good - Structured output directories
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"outputs/{document_type}/{timestamp}"
```

### 👍 Handle Errors Gracefully

```python
# ✅ Good - Comprehensive error handling
try:
    run_pipeline(config)
except ExtractionError:
    # Handle extraction failures
    pass
except PipelineError:
    # Handle pipeline failures
    pass
```

### 👍 Validate Before Running

```python
# ✅ Good - Validate configuration
try:
    config = PipelineConfig(**config_dict)
except ValidationError as e:
    print(f"Invalid configuration: {e}")
    exit(1)
```

---

## Troubleshooting

### 🐛 Configuration Validation Fails

**Solution:**
```python
from pydantic import ValidationError

try:
    config = PipelineConfig(**config_dict)
except ValidationError as e:
    print("Configuration errors:")
    for error in e.errors():
        print(f"  {error['loc']}: {error['msg']}")
```

### 🐛 Extraction Produces No Results

**Solution:**
```python
# Check extraction output
import json

with open("outputs/graph_stats.json") as f:
    stats = json.load(f)
    
if stats["node_count"] == 0:
    print("No nodes extracted - check template and document")
```

### 🐛 Out of Memory

**Solution:**
```python
# Use chunking and smaller batch sizes
config = PipelineConfig(
    source="large_document.pdf",
    template="templates.BillingDocument",
    use_chunking=True,  # Enable chunking
    max_batch_size=1,   # Smaller batches
    backend="llm",
    inference="remote"  # Use remote to save memory
)
```

---

## Next Steps

Now that you understand complete configurations:

1. **[Extraction Process →](../extraction-process/index.md)** - Learn how extraction works
2. **[Graph Management](../graph-management/index.md)** - Work with extracted graphs
3. **[CLI Guide](../../usage/cli/index.md)** - Use command-line interface