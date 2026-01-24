# Input Formats

Docling Graph supports multiple input formats, allowing you to process various types of documents and data sources through the same pipeline.

## Input Normalization Process

The pipeline automatically detects and validates input types, routing them through the appropriate processing stages:

--8<-- "docs/assets/flowcharts/input_normalization.md"

**Key Features**:
- **Automatic Type Detection**: Identifies input format from file extension, URL, or content
- **Validation**: Ensures input meets requirements (non-empty, correct format, etc.)
- **Smart Routing**: Skips unnecessary stages based on input type
  - Text/Markdown inputs skip OCR
  - DoclingDocument inputs skip extraction and go directly to graph conversion
  - URLs are downloaded and processed based on their content type

## Supported Input Formats

### 1. PDF Documents

**Description**: Standard PDF files with text, images, and complex layouts.

**File Extensions**: `.pdf`

**Processing**: Full pipeline with OCR/VLM, segmentation, and extraction.

**CLI Example**:
```bash
docling-graph convert document.pdf -t templates.invoice.Invoice
```

**Python API Example**:
```python
from docling_graph import PipelineConfig, run_pipeline

config = PipelineConfig(
    source="document.pdf",
    template="templates.invoice.Invoice",
    backend="llm",
    inference="local",
    processing_mode="many-to-one",
    docling_config="ocr",
    output_dir="outputs",
    export_format="csv"
)

run_pipeline(config)
```

---

### 2. Image Files

**Description**: Image files containing document content (scanned documents, photos of documents, etc.).

**File Extensions**: `.png`, `.jpg`, `.jpeg`

**Processing**: Full pipeline with OCR/VLM, segmentation, and extraction.

**CLI Example**:
```bash
docling-graph convert scanned_invoice.png -t templates.invoice.Invoice
```

**Python API Example**:
```python
config = PipelineConfig(
    source="scanned_invoice.jpg",
    template="templates.invoice.Invoice",
    backend="vlm",  # VLM works well with images
    inference="local",
    processing_mode="one-to-one",
    docling_config="vision",
    output_dir="outputs",
    export_format="json"
)

run_pipeline(config)
```

---

### 3. Plain Text Files

**Description**: Simple text files containing unstructured content.

**File Extensions**: `.txt`

**Processing**: Skips OCR and visual processing. Goes directly to LLM extraction.

**Requirements**:
- Must use LLM backend (VLM requires visual content)
- File must not be empty or contain only whitespace

**CLI Example**:
```bash
docling-graph convert notes.txt -t templates.report.Report --backend llm
```

**Python API Example**:
```python
config = PipelineConfig(
    source="meeting_notes.txt",
    template="templates.report.Report",
    backend="llm",  # Required for text inputs
    inference="remote",
    processing_mode="many-to-one",
    docling_config="ocr",  # Ignored for text inputs
    output_dir="outputs",
    export_format="csv"
)

run_pipeline(config)
```

---

### 4. Markdown Files

**Description**: Markdown-formatted text files with structure and formatting.

**File Extensions**: `.md`

**Processing**: Skips OCR and visual processing. Markdown structure is preserved during extraction.

**Requirements**:
- Must use LLM backend
- File must not be empty

**CLI Example**:
```bash
docling-graph convert README.md -t templates.documentation.Documentation --backend llm
```

**Python API Example**:
```python
config = PipelineConfig(
    source="documentation.md",
    template="templates.documentation.Documentation",
    backend="llm",
    inference="local",
    processing_mode="many-to-one",
    output_dir="outputs",
    export_format="json"
)

run_pipeline(config)
```

---

### 5. URLs

**Description**: Download and process documents from URLs.

**Format**: `http://` or `https://` URLs

**Processing**: 
1. Downloads content to temporary location
2. Detects content type (PDF, image, text, markdown)
3. Routes to appropriate processing pipeline

**Supported URL Content Types**:
- PDF documents
- Image files (PNG, JPG, JPEG)
- Plain text files
- Markdown files

**Requirements**:
- Valid HTTP/HTTPS URL
- Accessible without authentication (for now)
- File size under limit (default: 100MB)

**CLI Example**:
```bash
# PDF from URL
docling-graph convert https://example.com/invoice.pdf -t templates.invoice.Invoice

# Image from URL
docling-graph convert https://example.com/scan.jpg -t templates.form.Form

# Text from URL
docling-graph convert https://example.com/notes.txt -t templates.report.Report --backend llm
```

**Python API Example**:
```python
config = PipelineConfig(
    source="https://example.com/document.pdf",
    template="templates.invoice.Invoice",
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    output_dir="outputs",
    export_format="csv"
)

run_pipeline(config)
```

**URL Configuration**:
```python
from docling_graph.core.input.handlers import URLInputHandler

# Custom timeout and size limit
handler = URLInputHandler(
    timeout=30,      # seconds
    max_size_mb=50   # megabytes
)
```

---

### 6. Plain Text Strings (Python API Only)

**Description**: Raw text strings passed directly to the pipeline.

**Format**: Python string

**Processing**: Skips OCR and visual processing. Direct LLM extraction.

**Requirements**:
- Only available via Python API (not CLI)
- Must use LLM backend
- String must not be empty or whitespace-only

**Python API Example**:
```python
# Direct text input
text_content = """
Invoice #12345
Date: 2024-01-15
Amount: $1,234.56
Customer: Acme Corp
"""

config = PipelineConfig(
    source=text_content,  # Pass string directly
    template="templates.invoice.Invoice",
    backend="llm",
    inference="local",
    processing_mode="many-to-one",
    output_dir="outputs",
    export_format="json"
)

run_pipeline(config, mode="api")  # mode="api" required
```

**Note**: CLI does not accept plain text strings to avoid ambiguity with file paths.

---

### 7. DoclingDocument JSON (Advanced)

**Description**: Pre-processed DoclingDocument JSON files.

**File Extensions**: `.json` (with DoclingDocument schema)

**Processing**: Skips document conversion. Uses pre-existing structure.

**Use Cases**:
- Reprocessing previously converted documents
- Custom document preprocessing pipelines
- Integration with external Docling workflows

**Requirements**:
- Valid DoclingDocument JSON schema
- Must include `schema_name: "DoclingDocument"`
- Must include `version` field

**CLI Example**:
```bash
docling-graph convert processed_document.json -t templates.custom.Custom
```

**Python API Example**:
```python
config = PipelineConfig(
    source="preprocessed.json",
    template="templates.custom.Custom",
    backend="llm",
    inference="local",
    processing_mode="many-to-one",
    output_dir="outputs",
    export_format="csv"
)

run_pipeline(config)
```

**DoclingDocument JSON Structure**:
```json
{
  "schema_name": "DoclingDocument",
  "version": "1.0.0",
  "name": "document_name",
  "pages": {
    "0": {
      "page_no": 0,
      "size": {"width": 612, "height": 792}
    }
  },
  "body": {
    "self_ref": "#/body",
    "children": []
  },
  "furniture": {}
}
```

---

## Input Format Detection

The pipeline automatically detects input format based on:

1. **File Extension**: `.pdf`, `.png`, `.jpg`, `.txt`, `.md`, `.json`
2. **URL Scheme**: `http://` or `https://`
3. **Content Analysis**: For JSON files, checks for DoclingDocument schema
4. **Fallback**: Plain text for unrecognized formats (API mode only)

**Detection Examples**:

```python
from docling_graph.core.input.types import InputTypeDetector

# Detect from file path
input_type = InputTypeDetector.detect("document.pdf", mode="cli")
# Returns: InputType.PDF

# Detect from URL
input_type = InputTypeDetector.detect("https://example.com/file.txt", mode="cli")
# Returns: InputType.URL

# Detect from text (API mode only)
input_type = InputTypeDetector.detect("Plain text content", mode="api")
# Returns: InputType.TEXT
```

---

## Processing Pipeline by Input Type

### Visual Documents (PDF, Images)
```
Input → Document Conversion (OCR/VLM) → Segmentation → 
Extraction → Graph Construction → Export
```

### Text Documents (.txt, .md, plain text)
```
Input → Text Normalization → Extraction (LLM only) → 
Graph Construction → Export
```

### URLs
```
URL → Download → Type Detection → Route to appropriate pipeline
```

### DoclingDocument JSON
```
Input → Validation → Graph Construction → Export
(Skips conversion and extraction)
```

---

## Backend Compatibility

| Input Format | LLM Backend | VLM Backend |
|--------------|-------------|-------------|
| PDF | ✅ Yes | ✅ Yes |
| Images | ✅ Yes | ✅ Yes |
| Text Files | ✅ Yes | ❌ No |
| Markdown | ✅ Yes | ❌ No |
| URLs (PDF/Image) | ✅ Yes | ✅ Yes |
| URLs (Text/MD) | ✅ Yes | ❌ No |
| Plain Text | ✅ Yes | ❌ No |
| DoclingDocument | ✅ Yes | ✅ Yes |

**Note**: VLM (Vision Language Model) backend requires visual content. Use LLM backend for text-only inputs.

---

## Error Handling

### Empty Files
```bash
$ docling-graph convert empty.txt -t templates.Report
Error: Text input is empty
```

### Unsupported Backend
```bash
$ docling-graph convert notes.txt -t templates.Report --backend vlm
Error: VLM backend does not support text-only inputs. Use LLM backend instead.
```

### Invalid URL
```bash
$ docling-graph convert ftp://example.com/file.pdf -t templates.Invoice
Error: URL must use http or https scheme
```

### File Not Found
```bash
$ docling-graph convert missing.pdf -t templates.Invoice
Error: File not found: missing.pdf
```

---

## Best Practices

### 1. Choose the Right Backend

- **PDFs and Images**: Use VLM for complex layouts, LLM for text-heavy documents
- **Text Files**: Always use LLM backend
- **Mixed Workflows**: Use LLM backend for maximum compatibility

### 2. Validate Input Files

```python
from pathlib import Path

source_path = Path("document.txt")
if not source_path.exists():
    raise FileNotFoundError(f"Input file not found: {source_path}")

if source_path.stat().st_size == 0:
    raise ValueError("Input file is empty")
```

### 3. Handle URLs Safely

```python
from docling_graph.core.input.validators import URLValidator

validator = URLValidator()
try:
    validator.validate(url)
except ValidationError as e:
    print(f"Invalid URL: {e.message}")
```

### 4. Use Appropriate Processing Modes

- **one-to-one**: Best for multi-page PDFs where each page is independent
- **many-to-one**: Best for text files and single-entity documents

---

## Troubleshooting

### Issue: "Plain text input is only supported via Python API"

**Cause**: Trying to pass plain text string via CLI

**Solution**: Use Python API or save text to a `.txt` file first

```python
# Option 1: Use Python API
run_pipeline(config, mode="api")

# Option 2: Save to file
Path("temp.txt").write_text(text_content)
config.source = "temp.txt"
run_pipeline(config, mode="cli")
```

### Issue: "VLM backend does not support text-only inputs"

**Cause**: Using VLM backend with text files

**Solution**: Switch to LLM backend

```bash
docling-graph convert notes.txt -t templates.Report --backend llm
```

### Issue: URL download timeout

**Cause**: Slow network or large file

**Solution**: Increase timeout or download manually

```python
from docling_graph.core.input.handlers import URLInputHandler

handler = URLInputHandler(timeout=60)  # 60 seconds
temp_path = handler.load(url)
```

---

## Next Steps

- [Backend Selection](backend-selection.md) - Choose the right backend for your input
- [Processing Modes](processing-modes.md) - Understand one-to-one vs many-to-one
- [Configuration Examples](configuration-examples.md) - See complete configuration examples