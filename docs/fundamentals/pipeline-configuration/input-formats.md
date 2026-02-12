# Input Formats

Docling Graph uses a unified ingestion path: all inputs go through Docling except DoclingDocument JSON (which skips conversion), using a unified path: all inputs are converted through Docling; only DoclingDocument JSON skips conversion. See [Docling supported formats](https://docling-project.github.io/docling/usage/supported_formats/) for what Docling accepts.

## Input Normalization Process

The pipeline automatically detects and validates input types, routing them through the appropriate processing stages:

--8<-- "docs/assets/flowcharts/input_normalization.md"

**Key behavior**:
- **DoclingDocument JSON**: Loaded directly; conversion is skipped.
- **All other inputs**: Normalized (e.g. URL download, text to temp .md), then sent to Docling. Docling validates format; unsupported types raise Docling errors.
- **URLs**: Downloaded to a temp file; path is passed to Docling.

## Supported Input Formats

Docling Graph does not whitelist extensions. Any file or URL is sent to Docling; [Docling supported formats](https://docling-project.github.io/docling/usage/supported_formats/) include PDF, Office (DOCX, XLSX, PPTX), images, HTML, Markdown, LaTeX, AsciiDoc, CSV. Unsupported formats produce a Docling conversion error (e.g. `ExtractionError: Conversion failed in Docling: ...`).

---

### Document inputs (files, raw text)

Any Docling-supported file, or raw text (API only). Text and .txt are normalized to markdown, then sent to Docling.

**CLI**: `docling-graph convert document.pdf -t templates.billing_document.BillingDocument`  
**API**: Same; for raw text use `source="text content"` and `run_pipeline(config, mode="api")`.

---

### URLs

**Description**: Download and process documents from HTTP/HTTPS URLs.

**Processing**: Content is downloaded to a temporary file; the path is passed to Docling. Supported formats are those Docling supports.

**Requirements**: Valid http/https URL; file size under limit (default: 100MB).

**CLI Example**:
```bash
# PDF from URL
docling-graph convert https://example.com/invoice.pdf -t templates.billing_document.BillingDocument

# Image from URL
docling-graph convert https://example.com/scan.jpg -t templates.form.Form

# Text from URL
docling-graph convert https://example.com/notes.txt -t templates.report.Report --backend llm
```

**Python API Example**:
```python
config = PipelineConfig(
    source="https://example.com/document.pdf",
    template="templates.billing_document.BillingDocument",
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

### Plain text strings (Python API only)

Raw text: pass a string as `source` and call `run_pipeline(config, mode="api")`. It is normalized to a temporary markdown file and sent to Docling. CLI does not accept plain text (file path or URL only).

---

### DoclingDocument JSON (skip conversion)

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

- **URL**: String starting with `http://` or `https://`.
- **DoclingDocument**: `.json` file with DoclingDocument schema (e.g. `schema_name`, `version`, `pages`).
- **Document**: Everything else (any file path or, in API mode, raw text). Passed to Docling; no extension whitelist in docling-graph.

---

## Processing Pipeline by Input Type

### All inputs except DoclingDocument
```
Input → Normalize (e.g. URL download, text → .md) → Docling conversion →
DoclingDocument → Chunking → Extraction → Graph → Export
```

### DoclingDocument JSON
```
Input → Load DoclingDocument → Chunking / Extraction → Graph → Export
(Conversion skipped)
```

---

## Backend Compatibility

| Input type | LLM Backend | VLM Backend |
|------------|-------------|-------------|
| Documents (files, URLs) | Yes | Yes (PDF/images at Docling level) |
| DoclingDocument JSON | Yes | Yes |
| Plain text (API) | Yes | Converted via Docling |

VLM backend only supports certain inputs at the Docling level (e.g. PDF, images). Other formats may raise Docling or backend errors.

---

## Error Handling

### Unsupported format (from Docling)
When the file type is not supported by Docling:
```
ExtractionError: Conversion failed in Docling: ...
Details: source=/path/to/file.xyz
```
Use a [Docling-supported format](https://docling-project.github.io/docling/usage/supported_formats/) or convert the file first.

### Empty text
`ValidationError: Text input is empty` — ensure content is non-empty.

### File not found (CLI)
`ConfigurationError: File not found` — use a valid file path or URL.

### Invalid URL
`ValidationError: URL must use http or https scheme`

---

## Best Practices

### 👍 Choose the Right Backend

- **PDFs and Images**: Use VLM for complex layouts, LLM for text-heavy documents
- **Text Files**: Always use LLM backend
- **Mixed Workflows**: Use LLM backend for maximum compatibility

### 👍 Validate Input Files

```python
from pathlib import Path

source_path = Path("document.txt")
if not source_path.exists():
    raise FileNotFoundError(f"Input file not found: {source_path}")

if source_path.stat().st_size == 0:
    raise ValueError("Input file is empty")
```

### 👍 Handle URLs Safely

```python
from docling_graph.core.input.validators import URLValidator

validator = URLValidator()
try:
    validator.validate(url)
except ValidationError as e:
    print(f"Invalid URL: {e.message}")
```

### 👍 Use Appropriate Processing Modes

- **one-to-one**: Best for multi-page PDFs where each page is independent
- **many-to-one**: Best for text files and single-entity documents

---

## Troubleshooting

### 🐛 Plain text input is only supported via Python API

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

### 🐛 VLM backend does not support text-only inputs

**Cause**: Using VLM backend with text files

**Solution**: Switch to LLM backend

```bash
docling-graph convert notes.txt -t templates.Report --backend llm
```

### 🐛 URL download timeout

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