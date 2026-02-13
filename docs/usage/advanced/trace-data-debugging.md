# Trace Data Debugging

When `debug=True`, Docling Graph writes `debug/trace_data.json` as a compact, step-oriented payload.

## Trace Shape

Top-level structure:

```json
{
  "summary": {
    "runtime_seconds": 1.237,
    "page_count": 1,
    "extraction_success": true,
    "fallback_used": false,
    "node_count": 7,
    "edge_count": 6
  },
  "steps": [
    {
      "name": "pipeline",
      "runtime_seconds": 1.237,
      "status": "success",
      "artifacts": { "...": "..." }
    },
    {
      "name": "docling_conversion",
      "runtime_seconds": 0.312,
      "status": "success",
      "artifacts": { "...": "..." }
    }
  ]
}
```

Each step object includes only:
- `name`
- `runtime_seconds` (duration in seconds, 4 decimal places)
- `status`
- `artifacts` (single canonical payload; no mirrored `events`)

## Typical Step Artifacts

- `docling_conversion.artifacts.pages`
- `data_extraction.artifacts.extractions`
- `data_extraction.artifacts.fallbacks`
- `data_extraction.artifacts.staged_traces`
- `graph_mapping.artifacts.graph`
- `pipeline.artifacts.start|finish|failure`

## Structured Output Diagnostics

Look at `data_extraction` step artifacts for structured-output behavior.

Example payload keys:
- `structured_attempted`
- `structured_failed`
- `fallback_used`
- `fallback_error_class`
- `structured_primary_attempt_parsed_json`
- `structured_primary_attempt_raw`

## Quick Inspection

### jq

```bash
jq '.steps[] | {name, runtime_seconds, status}' debug/trace_data.json
```

```bash
jq '.steps[] | select(.name == "data_extraction") | .artifacts.fallbacks' debug/trace_data.json
```

### Python

```python
import json
from pathlib import Path

data = json.loads(Path("debug/trace_data.json").read_text())
print(data["summary"])
for step in data["steps"]:
    print(step["name"], step["runtime_seconds"], step["status"])
```

## Notes

- Large strings are truncated during JSON export to keep debug files manageable.
- `trace_data.json` intentionally exports compact step artifacts only (no mirrored raw events).
- `trace_data.json` no longer uses legacy buckets like `extractions`, `intermediate_graphs`, or `consolidation`.
