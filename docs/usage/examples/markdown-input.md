# Markdown Input Example


## Overview

This example demonstrates how to process Markdown documents directly, extracting structured data from formatted text without requiring OCR or visual processing.

**What You'll Learn:**
- Processing Markdown files
- Text-only extraction workflow
- LLM backend requirements
- Structured text parsing

**Time:** 10 minutes

---

## Use Case: Documentation Analysis

Extract structured information from project documentation, including sections, code examples, and metadata.

### Document Source

**File:** `README.md` or `DOCUMENTATION.md`

**Type:** Markdown

**Content:** Project documentation with sections, code blocks, and structured information.

---

## Template Definition

We'll create a template for documentation that captures sections, code examples, and metadata.

```python
from pydantic import BaseModel, Field
from docling_graph.utils import edge

class CodeExample(BaseModel):
    """Code example component."""
    model_config = {'is_entity': False}
    
    language: str = Field(description="Programming language")
    code: str = Field(description="Code snippet")
    description: str = Field(description="What the code does")

class Section(BaseModel):
    """Documentation section entity."""
    model_config = {
        'is_entity': True,
        'graph_id_fields': ['title']
    }
    
    title: str = Field(description="Section title")
    content: str = Field(description="Section content")
    subsections: list[str] = Field(
        default_factory=list,
        description="Subsection titles"
    )

class Documentation(BaseModel):
    """Complete documentation structure."""
    model_config = {'is_entity': True}
    
    title: str = Field(description="Document title")
    description: str = Field(description="Project description")
    version: str | None = Field(
        default=None,
        description="Documentation version"
    )
    sections: list[Section] = edge(
        "HAS_SECTION",
        description="Documentation sections"
    )
    code_examples: list[CodeExample] = Field(
        default_factory=list,
        description="Code examples"
    )
    requirements: list[str] = Field(
        default_factory=list,
        description="Project requirements"
    )
```

**Save as:** `templates/documentation.py`

---

## Processing with CLI

### Basic Markdown Processing

```bash
# Process README.md
uv run docling-graph convert README.md \
    --template "templates.documentation.Documentation" \
    --backend llm \
    --inference remote
```

**Important:** Markdown files require LLM backend (VLM doesn't support text-only inputs).

### With Local LLM

```bash
# Use local Ollama
uv run docling-graph convert DOCUMENTATION.md \
    --template "templates.documentation.Documentation" \
    --backend llm \
    --inference local \
    --provider ollama \
    --model llama3.1:8b
```

### With Chunking

```bash
# Process large markdown with chunking
uv run docling-graph convert LARGE_DOC.md \
    --template "templates.documentation.Documentation" \
    --backend llm \
    --inference remote \
    --use-chunking \
    --llm-consolidation
```

---

## Processing with Python API

### Basic Usage

```python
from docling_graph import PipelineConfig
from templates.documentation import Documentation

# Configure pipeline for Markdown input
config = PipelineConfig(
    source="README.md",
    template=Documentation,
    backend="llm",  # Required for text inputs
    inference="remote",
    processing_mode="many-to-one",
    output_dir="outputs/docs"
)

# Run pipeline
config.run()
```

### Processing Multiple Markdown Files

```python
from pathlib import Path
from docling_graph import PipelineConfig
from templates.documentation import Documentation

# Process all markdown files in a directory
docs_dir = Path("docs")
markdown_files = docs_dir.glob("**/*.md")

for md_file in markdown_files:
    print(f"Processing: {md_file}")
    
    config = PipelineConfig(
        source=str(md_file),
        template=Documentation,
        backend="llm",
        inference="remote",
        processing_mode="many-to-one",
        output_dir=f"outputs/{md_file.stem}"
    )
    
    try:
        config.run()
        print(f"✓ Completed: {md_file}")
    except Exception as e:
        print(f"✗ Failed: {md_file} - {e}")
```

### With Custom Provider

```python
from docling_graph import PipelineConfig
from templates.documentation import Documentation

# Use specific LLM provider
config = PipelineConfig(
    source="API_DOCS.md",
    template=Documentation,
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo",
    use_chunking=True,
    output_dir="outputs/api_docs"
)

config.run()
```

---

## Expected Output

### Graph Structure

```
Documentation (root node)
├── HAS_SECTION → Section (Installation)
│   ├── title: "Installation"
│   ├── content: "..."
│   └── subsections: ["Requirements", "Setup"]
├── HAS_SECTION → Section (Usage)
│   ├── title: "Usage"
│   └── content: "..."
├── code_examples (list)
│   ├── CodeExample 1: Python
│   └── CodeExample 2: Bash
└── requirements: ["Python 3.10+", "uv"]
```

### CSV Export

**nodes.csv:**
```csv
node_id,node_type,title,description,version
doc_1,Documentation,"Project Name","Description...","1.0.0"

node_id,node_type,title,content
section_installation,Section,"Installation","Installation instructions..."
section_usage,Section,"Usage","Usage guide..."
```

**edges.csv:**
```csv
source_id,target_id,edge_type
doc_1,section_installation,HAS_SECTION
doc_1,section_usage,HAS_SECTION
```

---

## Markdown Processing Features

### What Gets Processed

The pipeline extracts:
- **Headers** → Section titles
- **Paragraphs** → Content
- **Code blocks** → Code examples
- **Lists** → Requirements, features
- **Links** → References
- **Tables** → Structured data

### Markdown Preservation

The original Markdown formatting is preserved in the extracted content, allowing you to:
- Maintain code block syntax
- Preserve link references
- Keep list structures
- Retain emphasis and formatting

### Text-Only Pipeline

Markdown files skip:
<br>❌ OCR (no visual processing needed)
<br>❌ Page segmentation (single text stream)
<br>✅ Direct LLM extraction
<br>✅ Semantic chunking (if enabled)

---

## Troubleshooting

### Issue: VLM Backend Error

**Error:**
```
ExtractionError: VLM backend does not support text-only inputs
```

**Solution:**
```bash
# Always use LLM backend for Markdown
uv run docling-graph convert README.md \
    --template "templates.documentation.Documentation" \
    --backend llm  # Required
```

### Issue: Empty File

**Error:**
```
ValidationError: Text input is empty
```

**Solution:**
```bash
# Ensure file has content
cat README.md  # Check file content
file README.md  # Verify file type

# If file is empty, add content first
echo "# Documentation" > README.md
```

### Issue: Encoding Problems

**Error:**
```
ValidationError: Failed to read text file: encoding error
```

**Solution:**
```python
# Convert file to UTF-8 first
with open("README.md", "r", encoding="latin-1") as f:
    content = f.read()

with open("README_utf8.md", "w", encoding="utf-8") as f:
    f.write(content)

# Then process
config = PipelineConfig(source="README_utf8.md", ...)
```

---

## Best Practices

### 1. Use Descriptive Section Headers

```markdown
✅ Good - Clear hierarchy
# Installation Guide
## Requirements
## Setup Steps

❌ Bad - Unclear structure
# Stuff
## Things
```

### 2. Include Code Language Tags

```markdown
✅ Good - Language specified
```python
def hello():
    print("Hello")
```

❌ Bad - No language
```
def hello():
    print("Hello")
```
```

### 3. Structure Content Logically

```markdown
✅ Good - Logical flow
# Overview
# Installation
# Usage
# Examples
# Troubleshooting

❌ Bad - Random order
# Examples
# Overview
# Troubleshooting
# Installation
```

### 4. Use Consistent Formatting

```markdown
✅ Good - Consistent style
- Item 1
- Item 2
- Item 3

❌ Bad - Mixed styles
- Item 1
* Item 2
+ Item 3
```

---

## Advanced Usage

### Processing Markdown from String

```python
from docling_graph import PipelineConfig, run_pipeline
from templates.documentation import Documentation

# Markdown content as string
markdown_content = """
# My Project

## Overview
This is a sample project.

## Features
- Feature 1
- Feature 2
"""

# Process directly (API mode only)
config = PipelineConfig(
    source=markdown_content,
    template=Documentation,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one"
)

run_pipeline(config, mode="api")  # mode="api" required for string input
```

### Combining Multiple Markdown Files

```python
from pathlib import Path

# Combine multiple markdown files
md_files = ["intro.md", "guide.md", "reference.md"]
combined_content = "\n\n---\n\n".join(
    Path(f).read_text() for f in md_files
)

# Save combined file
Path("combined.md").write_text(combined_content)

# Process combined file
config = PipelineConfig(
    source="combined.md",
    template=Documentation,
    backend="llm",
    inference="remote"
)
config.run()
```

### Extracting Specific Sections

```python
from pydantic import BaseModel, Field

class QuickStart(BaseModel):
    """Extract only quickstart section."""
    model_config = {'is_entity': True}
    
    installation: str = Field(description="Installation instructions")
    basic_usage: str = Field(description="Basic usage example")
    next_steps: list[str] = Field(description="Next steps")

# Process with focused template
config = PipelineConfig(
    source="README.md",
    template=QuickStart,
    backend="llm",
    inference="remote"
)
```

---

## Comparison: Markdown vs PDF

| Feature | Markdown | PDF |
|---------|----------|-----|
| **OCR Required** | ❌ No | ✅ Yes |
| **Processing Speed** | ⚡ Fast | 🐢 Slower |
| **Backend Support** | LLM only | LLM + VLM |
| **Structure Preservation** | ✅ Excellent | ⚠️ Variable |
| **Code Blocks** | ✅ Native | ⚠️ Extracted |
| **Best For** | Documentation, Notes | Scanned docs, Forms |

---

## Next Steps

- **[DoclingDocument Input →](docling-document-input.md)** - Use pre-processed documents
- **[Input Formats Guide](../../fundamentals/pipeline-configuration/input-formats.md)** - Complete input format reference
- **[LLM Backend Configuration](../../fundamentals/pipeline-configuration/backend-selection.md)** - Configure LLM settings