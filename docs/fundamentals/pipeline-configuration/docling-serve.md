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
    docling_serve_timeout=300,        # per-document job deadline in seconds
    # For deployments behind an auth proxy (e.g. bearer tokens) instead of
    # X-Api-Key — sent on every request:
    # docling_serve_headers={"Authorization": "Bearer <token>"},
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
export DOCLING_SERVE_API_KEY="my-key"                              # optional
export DOCLING_SERVE_HEADERS='{"Authorization": "Bearer <token>"}' # optional, JSON object
```

**Precedence:** CLI flag > `config.yaml` / `PipelineConfig` > environment variable. The API key and custom headers are never written to `metadata.json`.

---

## How It Works

Docling Graph uses the official Docling service client (`docling.service_client`) and the server's **asynchronous task API** (requires docling-serve >= 1.0.0):

1. Local files are uploaded (and URL sources submitted for the server to fetch itself) as an async conversion task.
2. The client polls the task status over HTTP until the job finishes — no long-held connection, so load-balancer idle timeouts and the server's synchronous wait cap don't abort large documents. Transient errors (HTTP 500/502, and 429/503 with a Retry-After header) are retried automatically.
3. The DoclingDocument JSON result (`to_formats: ["json"]`) is fetched and parsed, and the pipeline continues exactly as with local conversion (including `chunks.json`, DocLang export, and provenance).

The `docling_config` pipeline selection still applies and maps to the server-side pipeline:

| `docling_config` | docling-serve pipeline |
|:-----------------|:-----------------------|
| `ocr` (default)  | `standard` (server defaults: OCR + table structure) |
| `vision`         | `vlm` |

---

## Timeouts and Large Documents

`docling_serve_timeout` is the approximate deadline for one document's conversion job, from submission to terminal status. Conversion runs asynchronously on the server and the client polls until this deadline — server queue time counts toward it, so on a busy shared instance raise it accordingly. On timeout, the job may still be running server-side.

```python
config = PipelineConfig(
    source="500_page_report.pdf",
    template="templates.Report",
    docling_serve_url="http://localhost:5001",
    docling_serve_timeout=1800,  # 30 minutes
)
```

Connect/read timeouts and transient-error retries are handled automatically and bounded separately (an unreachable server fails after ~10 seconds, not the full deadline).

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

The instance is unreachable. Check the URL, network access, and that the service is running (`curl <url>/health`). For custom/corporate CAs, set `SSL_CERT_FILE` — the client's HTTP stack (httpx) does not read `REQUESTS_CA_BUNDLE` (Docling Graph bridges it automatically when only `REQUESTS_CA_BUNDLE` is set).

### 🐛 `docling-serve returned HTTP 401/403`

The server has authentication enabled. Provide the key via `DOCLING_SERVE_API_KEY` (or `docling_serve_api_key`); it is sent as the `X-Api-Key` header. Deployments behind an auth proxy (e.g. bearer tokens) can send arbitrary headers via `DOCLING_SERVE_HEADERS` (or `docling_serve_headers`) instead.

### 🐛 `docling-serve returned HTTP 402 (usage limit exceeded)`

The service's usage quota is exhausted (managed/SaaS deployments). The error details carry the current usage and limit.

### 🐛 `docling-serve returned HTTP 404`

If the URL is correct, the server may predate the v1 async task API — remote conversion requires docling-serve >= 1.0.0 (`curl <url>/version`).

### 🐛 `conversion did not finish within ...s (job deadline)`

Conversion (including server queue time) took longer than `docling_serve_timeout`. Raise the timeout, or check the server's queue/load — the job may still be running server-side.

### 🐛 `Failed to parse DoclingDocument response`

The client and server versions may disagree on the response schema — upgrade the older side (client: the `docling` package; server: docling-serve).

### 🐛 `docling-serve returned HTTP 422`

Docling Graph always requests in-body results (no presigned artifact storage). If a deployment mandates presigned/artifact-storage results and rejects in-body targets, conversion fails with a 422 — check the server's target configuration.

---

## Next Steps

1. **[Docling Settings](docling-settings.md)** - OCR vs Vision pipeline selection
2. **[Configuration Basics](configuration-basics.md)** - All configuration options
3. **[Input Formats](input-formats.md)** - Reusing pre-converted DoclingDocument JSON
