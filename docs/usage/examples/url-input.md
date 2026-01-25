# URL Input Example


## Overview

This example demonstrates how to process documents directly from URLs, showcasing Docling Graph's ability to download and extract data from remote documents without manual file management.

**Time:** 10 minutes

---

## Use Case: Rheology Research Analysis

Extract structured information from a scientific paper hosted on arXiv, including authors, abstract, methodology, and key findings.

### Document Source

**URL:** `https://arxiv.org/pdf/2207.02720`

**Type:** PDF (Rheology Research on Rheology)

**Content:** Scientific paper with complex structure including authors, abstract, methodology, results, and references.

---

## Template Definition

We'll use a rheology research template that captures the essential structure of scientific documents.

```python
from pydantic import BaseModel, Field
from docling_graph.utils import edge

class Author(BaseModel):
    """Author entity."""
    model_config = {
        'is_entity': True,
        'graph_id_fields': ['name']
    }
    
    name: str = Field(description="Author's full name")
    affiliation: str | None = Field(
        default=None,
        description="Author's institutional affiliation"
    )

class Methodology(BaseModel):
    """Research methodology component."""
    model_config = {'is_entity': False}
    
    approach: str = Field(description="Research approach or method used")
    materials: list[str] = Field(
        default_factory=list,
        description="Materials or tools used"
    )
    procedure: str = Field(description="Experimental or analytical procedure")

class Finding(BaseModel):
    """Key research finding."""
    model_config = {'is_entity': False}
    
    description: str = Field(description="Description of the finding")
    significance: str = Field(description="Significance or implication")

class Research(BaseModel):
    """Complete rheology research structure."""
    model_config = {'is_entity': True}
    
    title: str = Field(description="Paper title")
    abstract: str = Field(description="Paper abstract")
    authors: list[Author] = edge(
        "AUTHORED_BY",
        description="Paper authors"
    )
    methodology: Methodology = Field(description="Research methodology")
    key_findings: list[Finding] = Field(
        default_factory=list,
        description="Key research findings"
    )
    conclusion: str = Field(description="Paper conclusion")
```

**Save as:** `templates/research.py`

---

## Processing with CLI

### Basic URL Processing

```bash
# Process rheology research from URL
uv run docling-graph convert "https://arxiv.org/pdf/2207.02720" \
    --template "templates.research.Research" \
    --processing-mode "many-to-one" \
    --backend llm \
    --inference remote
```

### With Custom Output

```bash
# Process with custom output directory
uv run docling-graph convert "https://arxiv.org/pdf/2207.02720" \
    --template "templates.research.Research" \
    --processing-mode "many-to-one" \
    --output-dir "outputs/research_paper" \
    --export-format json
```

### With Specific Model

```bash
# Use specific LLM model
uv run docling-graph convert "https://arxiv.org/pdf/2207.02720" \
    --template "templates.research.Research" \
    --processing-mode "many-to-one" \
    --backend llm \
    --inference remote \
    --provider openai \
    --model gpt-4-turbo
```

---

## Processing with Python API

### Basic Usage

```python
from docling_graph import run_pipeline, PipelineConfig
from templates.research import Research

# Configure pipeline for URL input
config = PipelineConfig(
    source="https://arxiv.org/pdf/2207.02720",
    template=Research,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    output_dir="outputs/research"
)

# Run pipeline
run_pipeline(config)
```

### With Custom Settings

```python
from docling_graph import run_pipeline, PipelineConfig
from templates.research import Research

# Advanced configuration
config = PipelineConfig(
    source="https://arxiv.org/pdf/2207.02720",
    template=Research,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    provider_override="mistral",
    model_override="mistral-large-latest",
    use_chunking=True,
    llm_consolidation=True,
    output_dir="outputs/research",
    export_format="json"
)

# Run pipeline
run_pipeline(config)
```

### Error Handling

```python
from docling_graph import run_pipeline, PipelineConfig
from docling_graph.exceptions import ValidationError, ExtractionError
from templates.research import Research

try:
    config = PipelineConfig(
        source="https://arxiv.org/pdf/2207.02720",
        template=Research,
        backend="llm",
        inference="remote",
        processing_mode="many-to-one"
    )
    run_pipeline(config)
    
except ValidationError as e:
    print(f"URL validation failed: {e.message}")
    if e.details:
        print(f"Details: {e.details}")
        
except ExtractionError as e:
    print(f"Extraction failed: {e.message}")
    # Handle extraction errors (e.g., retry with different model)
```

---

## Expected Output

### Graph Structure

```
Research (root node)
├── AUTHORED_BY → Author (John Doe)
├── AUTHORED_BY → Author (Jane Smith)
├── methodology (embedded)
│   ├── approach: "Experimental rheology"
│   ├── materials: ["Polymer samples", "Rheometer"]
│   └── procedure: "..."
├── key_findings (list)
│   ├── Finding 1: "..."
│   └── Finding 2: "..."
└── conclusion: "..."
```

### CSV Export

**nodes.csv:**
```csv
node_id,node_type,title,abstract,conclusion
research_1,Research,"Paper Title","Abstract text...","Conclusion text..."

node_id,node_type,name,affiliation
author_john_doe,Author,"John Doe","University X"
author_jane_smith,Author,"Jane Smith","Institute Y"
```

**edges.csv:**
```csv
source_id,target_id,edge_type
research_1,author_john_doe,AUTHORED_BY
research_1,author_jane_smith,AUTHORED_BY
```

### JSON Export

```json
{
  "nodes": [
    {
      "id": "research_1",
      "type": "Research",
      "properties": {
        "title": "Paper Title",
        "abstract": "Abstract text...",
        "conclusion": "Conclusion text..."
      }
    },
    {
      "id": "author_john_doe",
      "type": "Author",
      "properties": {
        "name": "John Doe",
        "affiliation": "University X"
      }
    }
  ],
  "edges": [
    {
      "source": "research_1",
      "target": "author_john_doe",
      "type": "AUTHORED_BY"
    }
  ]
}
```

---

## URL Processing Features

### Automatic Download

The pipeline automatically:
1. Downloads the PDF from the URL
2. Saves to temporary location
3. Detects content type (PDF)
4. Routes to appropriate processing pipeline
5. Cleans up temporary files

### Content Type Detection

Supported URL content types:
- **PDF documents** → Full document pipeline
- **Images** (PNG, JPG) → Full document pipeline
- **Text files** → Text-only pipeline (LLM backend required)
- **Markdown files** → Text-only pipeline (LLM backend required)

### Configuration Options

```python
from docling_graph.core.input.handlers import URLInputHandler

# Custom URL handler settings
handler = URLInputHandler(
    timeout=60,      # Download timeout in seconds
    max_size_mb=100  # Maximum file size in MB
)
```

---

## Troubleshooting

### 🐛 URL Download Timeout

**Error:**
```
ValidationError: URL download timeout after 30 seconds
```

**Solution:**
```python
# Increase timeout for large files
from docling_graph.core.input.handlers import URLInputHandler

handler = URLInputHandler(timeout=120)  # 2 minutes
```

### 🐛 File Too Large

**Error:**
```
ValidationError: File size (150MB) exceeds maximum size (100MB)
```

**Solution:**
```python
# Increase size limit or download manually
handler = URLInputHandler(max_size_mb=200)

# Or download manually first
import requests
response = requests.get(url)
with open("document.pdf", "wb") as f:
    f.write(response.content)

# Then process local file
config = PipelineConfig(source="document.pdf", ...)
```

### 🐛 Unsupported URL Scheme

**Error:**
```
ValidationError: URL must use http or https scheme
```

**Solution:**
```bash
# Only HTTP/HTTPS URLs are supported
# For FTP or other protocols, download manually first
wget ftp://example.com/file.pdf
uv run docling-graph convert file.pdf --template "..."
```

---

## Best Practices

### 👍 Use HTTPS When Available

```python
# ✅ Good - Secure connection
source = "https://arxiv.org/pdf/2207.02720"

# ⚠️ Avoid - Insecure connection
source = "http://example.com/document.pdf"
```

### 👍 Handle Network Errors

```python
from docling_graph.exceptions import ValidationError

try:
    run_pipeline(config)
except ValidationError as e:
    if "timeout" in str(e).lower():
        print("Network timeout - retrying with longer timeout")
        # Retry logic
    elif "failed to download" in str(e).lower():
        print("Download failed - check URL and network connection")
```

### 👍 Verify URL Before Processing

```python
import requests

def verify_url(url: str) -> bool:
    """Verify URL is accessible before processing."""
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except:
        return False

if verify_url(url):
    config = PipelineConfig(source=url, ...)
    run_pipeline(config)
else:
    print(f"URL not accessible: {url}")
```

### 👍 Cache Downloaded Files

```python
from pathlib import Path
import hashlib

def get_cache_path(url: str) -> Path:
    """Generate cache path for URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return Path(f"cache/{url_hash}.pdf")

cache_path = get_cache_path(url)
if cache_path.exists():
    # Use cached file
    config = PipelineConfig(source=str(cache_path), ...)
else:
    # Download from URL
    config = PipelineConfig(source=url, ...)
```

---

## Next Steps

- **[Markdown Input →](markdown-input.md)** - Process markdown documents
- **[DoclingDocument Input →](docling-document-input.md)** - Use pre-processed documents
- **[Input Formats Guide](../../fundamentals/pipeline-configuration/input-formats.md)** - Complete input format reference