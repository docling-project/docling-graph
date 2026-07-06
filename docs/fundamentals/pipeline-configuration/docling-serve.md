# Docling-Serve Integration


## Overview

[docling-serve](https://github.com/docling-project/docling-serve) exposes the Docling conversion pipeline as a REST API. Docling Graph can delegate the **document conversion step** to such an instance instead of converting locally: the server performs OCR/layout analysis and returns the DoclingDocument, and the rest of the pipeline (chunking, LLM extraction, graph conversion, export) runs unchanged.

**Why use it:**

- **No local conversion models** — the client machine doesn't download or load Docling's OCR/layout/VLM model stack.
- **Shared infrastructure** — a GPU-backed docling-serve cluster can serve many lightweight docling-graph clients.
- **Consistent conversions** — every pipeline run converts documents with the same centrally managed service.

**Scope:** only conversion is remote. LLM extraction still uses whatever `backend`/`inference` you configured. The VLM extraction backend (`backend="vlm"`) processes source documents directly and locally, so docling-serve has no effect there.

---

## Configuration

### Python API

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",

    # Convert on a remote docling-serve instance
    docling_serve_url="http://localhost:5001",
    docling_serve_api_key="my-key",   # only if the server requires one
    docling_serve_timeout=300,        # seconds; raise for large documents
)
config.run()
```

### CLI

```bash
uv run docling-graph convert document.pdf \
    --template "templates.BillingDocument" \
    --docling-serve-url http://localhost:5001
```

### config.yaml

```yaml
docling:
  pipeline: ocr
  serve:
    url: http://localhost:5001
    timeout: 300
```

### Environment Variables

When no URL is set explicitly, Docling Graph falls back to environment variables — convenient for cluster deployments where every client should use the same instance:

```bash
export DOCLING_SERVE_URL="http://docling-serve.internal:5001"
export DOCLING_SERVE_API_KEY="my-key"   # optional
```

**Precedence:** CLI flag > `config.yaml` / `PipelineConfig` > environment variable. The API key is never written to `metadata.json`.

---

## How It Works

1. Local files are uploaded to `POST /v1/convert/file`; URL sources are passed to `POST /v1/convert/source` so the server fetches them itself.
2. The client requests DoclingDocument JSON output (`to_formats: ["json"]`).
3. The response is parsed back into a `DoclingDocument`, and the pipeline continues exactly as with local conversion (including `chunks.json`, DocLang export, and provenance).

The `docling_config` pipeline selection still applies and maps to the server-side pipeline:

| `docling_config` | docling-serve pipeline |
|:-----------------|:-----------------------|
| `ocr` (default)  | `standard` (server defaults: OCR + table structure) |
| `vision`         | `vlm` |

---

## Timeouts and Large Documents

The synchronous docling-serve API holds the HTTP connection while converting, so the timeout must cover the whole conversion:

```python
config = PipelineConfig(
    source="500_page_report.pdf",
    template="templates.Report",
    docling_serve_url="http://localhost:5001",
    docling_serve_timeout=1800,  # 30 minutes
)
```

---

## Trying It Locally

Run a local instance with the official container image:

```bash
docker run -p 5001:5001 quay.io/docling-project/docling-serve
```

Then point Docling Graph at it:

```bash
export DOCLING_SERVE_URL="http://localhost:5001"
uv run docling-graph convert document.pdf --template "templates.BillingDocument"
```

---

## Troubleshooting

### 🐛 `Failed to reach docling-serve`

The instance is unreachable. Check the URL, network access, and that the service is running (`curl <url>/health`).

### 🐛 `docling-serve returned HTTP 401/403`

The server has authentication enabled. Provide the key via `DOCLING_SERVE_API_KEY` (or `docling_serve_api_key`); it is sent as the `X-Api-Key` header.

### 🐛 `docling-serve request timed out`

Conversion took longer than `docling_serve_timeout`. Raise the timeout, or check the server's queue/load.

### 🐛 `response contains no DoclingDocument JSON`

The server did not honor the `json` output format — make sure the docling-serve version is recent enough to support `to_formats: ["json"]`.

---

## Next Steps

1. **[Docling Settings](docling-settings.md)** - OCR vs Vision pipeline selection
2. **[Configuration Basics](configuration-basics.md)** - All configuration options
3. **[Input Formats](input-formats.md)** - Reusing pre-converted DoclingDocument JSON
