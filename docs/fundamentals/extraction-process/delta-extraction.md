# Delta Extraction

## Overview

**Delta extraction** is an LLM extraction contract for **many-to-one** processing that turns document chunks into a **flat graph IR** (nodes and relationships), then normalizes, merges, and projects the result into your Pydantic template. It is designed for long documents and chunk-based workflows.

Set `extraction_contract="delta"` in your config or use `--extraction-contract delta` on the CLI. Chunking must be enabled (`use_chunking=True`, which is the default for many-to-one).

**When to use:**

- Long documents where you want **token-bounded batching** (multiple chunks per LLM call, then merge by identity).
- You prefer a **graph-first** representation: entities as nodes with `path`, `ids`, and `parent`, then projected to the template.
- You want optional **post-merge resolvers** (fuzzy/semantic) to merge near-duplicate entities.

**When to use direct (default) or staged:**

- **Direct**: Flat or simple templates; single-pass extraction and programmatic merge.
- **Staged**: Complex nested templates; ID pass → fill pass → merge (no chunk batching).

---

## How It Works

Delta extraction runs these steps:

1. **Chunking** — Done outside delta (document processor or strategy). Produces chunks and optional chunk metadata (e.g. token counts, page numbers).

2. **Batch planning** — Chunks are packed into token-bounded batches (`llm_batch_token_size`). Each batch is sent in one LLM call.

3. **Per-batch LLM** — For each batch, the LLM receives the batch document plus a **path catalog** and **semantic guide** from your template. It returns a **DeltaGraph**: `nodes` (path, ids, parent, properties) and optional `relationships`. Output is validated with retries on failure.

4. **IR normalization** — Batch results are normalized: paths canonicalized to catalog paths, IDs normalized and optionally inferred from path indices, parent references repaired, nested properties stripped, provenance attached. Unknown paths can be dropped if `delta_normalizer_validate_paths` is true.

5. **Graph merge** — Normalized graphs are merged with deduplication by (path, identity). Node properties are merged (e.g. prefer longer string on conflict). Relationships are deduplicated by edge and endpoints.

6. **Resolvers** (optional) — If `delta_resolvers_enabled` is true, a post-merge pass can merge near-duplicate nodes by fuzzy or semantic similarity (`delta_resolvers_mode`: `fuzzy`, `semantic`, or `chain`).

7. **Projection** — The merged graph is projected back into a template-shaped root dict by attaching filled nodes to parents via (path, ids) and building lists/scalars per catalog.

8. **Quality gate** — Checks (e.g. root instance present, minimum instances, parent lookup misses) determine pass/fail. On fail, the pipeline can return `None` and emit a trace with reasons; optionally the strategy may fall back to direct extraction.

---

## Schema Requirements

Delta uses a **catalog** derived from your Pydantic template (same idea as staged):

- **Paths** — Root `""`, then nested paths like `line_items[]`, `line_items[].item`. The LLM must use only these catalog paths.
- **Identity** — Entities with `graph_id_fields` get stable keys for dedup and parent linkage; list items often use a field like `line_number` or `index`.
- **Flat properties** — Node and relationship properties must be flat (scalars or lists of scalars). Nested objects are stripped by the normalizer.

For identity and linkage best practices, see [Schema design for staged extraction](../schema-definition/staged-extraction-schema.md) (same concepts apply to delta).

---

## Configuration and options

All options can be set in Python via `PipelineConfig` or a config dict passed to `run_pipeline()`. CLI flags (when available) override config-file defaults.

### Batching and parallelism

| Python (`PipelineConfig` / config dict) | CLI flag | Default | Description |
|----------------------------------------|----------|---------|-------------|
| `extraction_contract` | `--extraction-contract` | `"direct"` | Set to `"delta"` to enable delta extraction. |
| `use_chunking` | `--use-chunking` / `--no-use-chunking` | `True` | Must be enabled for delta (chunk → batch flow). |
| `llm_batch_token_size` | `--llm-batch-token-size` | `2048` | Max input tokens per LLM batch; a new call is started when a batch would exceed this. |
| `parallel_workers` | `--parallel-workers` | `1` (or preset) | Number of parallel workers for delta batch LLM calls. |
| `staged_pass_retries` | `--staged-retries` | `1` | Retries per batch when the LLM returns invalid JSON (used as `max_pass_retries` for delta). |

### Quality gate

These control whether the merged result passes the quality gate. If the gate fails, delta can return `None` (and the strategy may fall back to direct extraction).

| Python (config dict) | Default | Description |
|----------------------|---------|-------------|
| `delta_quality_require_root` | `True` | Require at least one root instance (`path=""`). |
| `delta_quality_min_instances` | `1` | Minimum total node count. |
| `delta_quality_max_parent_lookup_miss` | `4` | Max allowed parent lookup misses before fail. |
| `delta_quality_adaptive_parent_lookup` | `True` | When root exists, allow a higher effective miss tolerance. |
| `delta_quality_require_relationships` | `False` | Require at least one relationship in the graph. |
| `delta_quality_require_structural_attachments` | `False` | Require list/scalar attachments to parents. |
| `quality_max_unknown_path_drops` | `-1` | Max unknown-path drops before fail; `-1` disables. |
| `quality_max_id_mismatch` | `-1` | Max ID key mismatches before fail; `-1` disables. |
| `quality_max_nested_property_drops` | `-1` | Max nested property drops before fail; `-1` disables. |

Quality gate options are not exposed as CLI flags; set them in a config file (e.g. `config_template.yaml` or your `defaults`) or in a config dict in Python.

### IR normalizer

| Python (config dict) | CLI flag | Default | Description |
|----------------------|----------|---------|-------------|
| `delta_normalizer_validate_paths` | `--delta-normalizer-validate-paths` / `--no-delta-normalizer-validate-paths` | `True` | Drop or repair nodes with unknown catalog paths. |
| `delta_normalizer_canonicalize_ids` | `--delta-normalizer-canonicalize-ids` / `--no-delta-normalizer-canonicalize-ids` | `True` | Canonicalize ID values before merge. |
| `delta_normalizer_strip_nested_properties` | `--delta-normalizer-strip-nested-properties` / `--no-delta-normalizer-strip-nested-properties` | `True` | Drop nested dict/list-of-dict properties from nodes and relationships. |
| `delta_normalizer_attach_provenance` | *(config only)* | `True` | Attach batch/chunk provenance to normalized nodes and relationships. |

### Resolvers (post-merge dedup)

Optional pass to merge near-duplicate entities after the graph merge.

| Python (config dict) | CLI flag | Default | Description |
|----------------------|----------|---------|-------------|
| `delta_resolvers_enabled` | `--delta-resolvers-enabled` / `--no-delta-resolvers-enabled` | `True` | Enable the resolver pass. |
| `delta_resolvers_mode` | `--delta-resolvers-mode` | `"semantic"` | One of `off`, `fuzzy`, `semantic`, `chain`. |
| `delta_resolver_fuzzy_threshold` | `--delta-resolver-fuzzy-threshold` | `0.9` | Similarity threshold for fuzzy matching. |
| `delta_resolver_semantic_threshold` | `--delta-resolver-semantic-threshold` | `0.92` | Similarity threshold for semantic matching. |
| `delta_resolver_properties` | *(config only)* | `None` | List of property names used for matching; default uses catalog fallback fields. |
| `delta_resolver_paths` | *(config only)* | `None` | Restrict resolver to these catalog paths; empty means all. |

---

## Usage

### Python API

Pass options via `PipelineConfig` or a dict to `run_pipeline()`:

```python
from docling_graph import PipelineConfig, run_pipeline

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    processing_mode="many-to-one",
    extraction_contract="delta",
    use_chunking=True,
    # Batching and parallelism
    llm_batch_token_size=2048,
    parallel_workers=2,
    staged_pass_retries=1,
    # Quality gate (optional overrides)
    delta_quality_require_root=True,
    delta_quality_min_instances=1,
    delta_quality_max_parent_lookup_miss=4,
    delta_quality_adaptive_parent_lookup=True,
    # IR normalizer
    delta_normalizer_validate_paths=True,
    delta_normalizer_canonicalize_ids=True,
    delta_normalizer_strip_nested_properties=True,
    delta_normalizer_attach_provenance=True,
    # Resolvers (optional)
    delta_resolvers_enabled=True,
    delta_resolvers_mode="semantic",
    delta_resolver_fuzzy_threshold=0.9,
    delta_resolver_semantic_threshold=0.92,
)
context = run_pipeline(config)
```

The options `delta_quality_require_relationships` and `delta_quality_require_structural_attachments` are not fields on `PipelineConfig`; set them in a config file (e.g. `defaults` in your YAML) or in a config dict: `run_pipeline({..., "delta_quality_require_relationships": False})`.

### CLI

All delta-related flags (when using `--extraction-contract delta`):

```bash
# Required for delta
uv run docling-graph convert document.pdf \
  --template "templates.BillingDocument" \
  --extraction-contract delta

# Batching and parallelism
uv run docling-graph convert document.pdf \
  --template "templates.BillingDocument" \
  --extraction-contract delta \
  --use-chunking \
  --llm-batch-token-size 2048 \
  --parallel-workers 2 \
  --staged-retries 1

# IR normalizer (toggles)
uv run docling-graph convert document.pdf \
  --extraction-contract delta \
  --template "templates.BillingDocument" \
  --delta-normalizer-validate-paths \
  --delta-normalizer-canonicalize-ids \
  --no-delta-normalizer-strip-nested-properties

# Resolvers
uv run docling-graph convert document.pdf \
  --extraction-contract delta \
  --template "templates.BillingDocument" \
  --delta-resolvers-enabled \
  --delta-resolvers-mode fuzzy \
  --delta-resolver-fuzzy-threshold 0.9 \
  --delta-resolver-semantic-threshold 0.92
```

Quality gate and resolver list options (`delta_resolver_properties`, `delta_resolver_paths`, `delta_quality_*`, `quality_max_*`) are not CLI flags; use a config file (e.g. `defaults` in `config_template.yaml` or your project config) to set them.

---

## Trace and debugging

When delta runs, the pipeline emits a **trace** (e.g. via `trace_data` or debug artifacts) containing:

- `contract: "delta"`
- `chunk_count`, `batch_count`, `batch_timings`, `batch_errors`
- `path_counts`, `normalizer_stats`, `merge_stats`, `resolver` (if enabled)
- `quality_gate`: `{ ok, reasons }`
- `diagnostics`: e.g. top missing-id paths, unknown path examples, parent lookup miss examples

With `debug=True`, artifacts like `delta_trace.json`, `delta_merged_graph.json`, and `delta_merged_output.json` can be written to the debug directory.

---

## Related

- [Staged Extraction](staged-extraction.md) — Multi-pass ID → fill → merge (no chunk batching)
- [Extraction Backends](extraction-backends.md) — LLM vs VLM and extraction contracts
- [Configuration reference](../../reference/config.md) — Full config API
- [convert command](../../usage/cli/convert-command.md) — CLI flags for delta
