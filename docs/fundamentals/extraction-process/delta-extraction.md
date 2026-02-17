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

--8<-- "docs/assets/flowcharts/delta_extraction.md"

1. **Chunking** — Done outside delta (document processor or strategy). Produces chunks and optional chunk metadata (e.g. token counts, page numbers).

2. **Batch planning** — Chunks are packed into token-bounded batches (`llm_batch_token_size`). Each batch is sent in one LLM call.

3. **Per-batch LLM** — For each batch, the LLM receives the batch document plus a **path catalog** and **semantic guide** from your template. It returns a **DeltaGraph**: `nodes` (path, ids, parent, properties) and optional `relationships`. Output is validated with retries on failure.

4. **IR normalization** — Batch results are normalized: paths canonicalized to catalog paths, IDs normalized and optionally inferred from path indices, parent references repaired, nested properties stripped, provenance attached. Unknown paths can be dropped if `delta_normalizer_validate_paths` is true.

5. **Graph merge** — Normalized graphs are merged with deduplication by (path, identity). Node properties are merged (e.g. prefer longer string on conflict). Relationships are deduplicated by edge and endpoints.

6. **Resolvers** (optional) — If `delta_resolvers_enabled` is true, a post-merge pass can merge near-duplicate nodes by fuzzy or semantic similarity (`delta_resolvers_mode`: `fuzzy`, `semantic`, or `chain`).

7. **Identity filter** (optional) — If `delta_identity_filter_enabled` is true, entity nodes whose identity looks like a section/chapter title are dropped. With `delta_identity_filter_strict` true, only identities in the schema allowlist are kept.

8. **Projection** — The merged graph is projected into a template-shaped root dict: nodes are attached to parents via (path, ids). When a parent is missing (e.g. dropped by the identity filter), a **best-effort** attachment to the first available parent of the same path is attempted so more nodes stay in the tree.

9. **Quality gate** — The gate uses **attached node count** (nodes that made it into the root tree), not raw graph size. If `attached_node_count` is below `delta_quality_min_instances` (default 20) or parent lookup misses exceed the allowed tolerance, the gate fails. On fail, delta returns `None` and the many-to-one strategy **falls back to direct extraction** (full-document, single LLM call), which usually yields a richer graph for sparse delta runs.

---

## Schema Requirements

Delta uses a **catalog** derived from your Pydantic template (same idea as staged):

- **Paths** — Root `""`, then nested paths like `line_items[]`, `line_items[].item`. The LLM must use only these catalog paths.
- **Identity** — Entities with `graph_id_fields` get stable keys for dedup and parent linkage; list items often use a field like `line_number` or `index`.
- **Flat properties** — Node and relationship properties must be flat (scalars or lists of scalars). Nested objects are stripped by the normalizer.
- **Root required fields** — Required root-level fields (e.g. `reference_document`, `title`) should be documented in the template so the LLM can fill them; the catalog hints the root path to include required root-level fields when present in the document.

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

The gate uses **attached node count** (nodes successfully attached into the root tree during projection). If the gate fails, delta returns `None` and the strategy **falls back to direct extraction**.

| Python (config dict) | Default | Description |
|----------------------|---------|-------------|
| `delta_quality_require_root` | `True` | Require at least one root instance (`path=""`). |
| `delta_quality_min_instances` | `20` | Minimum attached nodes; below this, gate fails and direct extraction is used. |
| `delta_quality_max_parent_lookup_miss` | `4` | Max allowed parent lookup misses before fail. |
| `delta_quality_adaptive_parent_lookup` | `True` | When root exists, allow higher effective miss tolerance (e.g. ~25% of instances, cap 150). |
| `delta_quality_require_relationships` | `False` | Require at least one relationship in the graph. |

### Identity filter

| Python (config dict) | Default | Description |
|----------------------|---------|-------------|
| `delta_identity_filter_enabled` | `True` | Drop entity nodes whose identity looks like a section/chapter title. |
| `delta_identity_filter_strict` | `False` | If true, drop any entity whose identity is not in the schema allowlist (for paths with `identity_example_values`). If false, only section-title heuristic is applied. |

Other gate options (e.g. `delta_quality_require_structural_attachments`, `quality_max_unknown_path_drops`, `quality_max_id_mismatch`, `quality_max_nested_property_drops`) are documented in the config reference. Quality gate and identity filter options are not CLI flags; set them in a config file or config dict.

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

### Gleaning (direct and delta)

Optional second-pass extraction ("what did you miss?") to improve recall. Applies to **direct** and **delta** contracts only (not staged). Disabled by default.

| Python (`PipelineConfig` / config dict) | CLI flag | Default | Description |
|----------------------------------------|----------|---------|-------------|
| `gleaning_enabled` | `--gleaning-enabled` / `--no-gleaning-enabled` | `False` | Run one extra extraction pass and merge additional entities/relations. |
| `gleaning_max_passes` | `--gleaning-max-passes` | `1` | Max number of gleaning passes when gleaning is enabled. |

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
    # Gleaning (optional second-pass recall; also applies to direct)
    gleaning_enabled=False,
    gleaning_max_passes=1,
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
