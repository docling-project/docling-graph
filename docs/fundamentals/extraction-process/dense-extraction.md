# Dense Extraction

## Overview

**Dense extraction** is an LLM extraction contract for **many-to-one** processing that combines granular document structure with rich per-node data via a **two-phase "Skeleton-then-Flesh"** flow. It is fully autonomous (no dependency on the delta or staged contracts).

Set `extraction_contract="dense"` in your config or use `--extraction-contract dense` on the CLI. Chunking must be enabled (`use_chunking=True`, which is the default for many-to-one).

**Structured output:** Dense uses **legacy prompt-schema mode only** (no API-level structured output), same as staged. This avoids provider-specific failures, empty/null JSON on skeleton batches, and repeated structured-output failures on fill. The global `structured_output` setting does not apply to dense.

**When to use:**

- You want both **fine-grained structure** (many distinct entities, e.g. every figure/experiment/curve) and **dense attribute data** (protocols, derived quantities, model fits) in one run.
- Delta gives you the structure but sparse properties ("ghost graph"); staged gives you rich data but fewer entities ("lumping"). Dense aims for both by separating structure discovery (Phase 1) from data filling (Phase 2).

**When to use direct, staged, or delta:**

- **Direct**: Flat or simple templates; single-pass extraction.
- **Staged**: Complex nested templates; ID pass → fill pass on full document (no chunk batching).
- **Delta**: Long documents; chunk-based graph IR with merge and projection (structure-focused).

---

## How It Works

1. **Phase 1 (Skeleton)** — Chunks are packed into token-bounded batches. For each batch, the LLM is asked to **identify every distinct entity** per catalog path: output only `path`, `ids`, `parent`, and **ancestry** (no property values). Phase 1 includes a **scope boundary**: extract only entities that are the primary subject, direct output, or original creation of the document; do not extract external references, cited works, or third-party entities mentioned for context. Phase 1 uses a **skeleton-only** semantic guide (paths and identity fields only), so the LLM cannot see or emit other template properties. The LLM **must** provide an ancestry array for every non-root node (full lineage from root to immediate parent). Results are normalized and merged across batches with deduplication by (path, identity). Output: a merged skeleton graph (nodes with correct hierarchy and identities; properties empty).

2. **Quality gate** — After Phase 1, the pipeline checks for at least one root instance and a minimum number of skeleton nodes. If the gate fails, dense returns `None` and the strategy can fall back to direct extraction.

3. **Phase 2 (Flesh)** — The skeleton is converted to per-path **descriptors** (path, ids, parent). Paths are processed in **bottom-up** order. For each path, descriptors are batched (up to `dense_fill_nodes_cap` per call). The LLM receives the **full document** and the list of instance identifiers; it returns fully filled objects according to the projected schema for that path. Filled objects are merged into a template-shaped root by attaching each object to its parent via descriptor linkage.

4. **Output** — A single template-shaped dict (validated to your Pydantic model as with other contracts).

---

## Schema Requirements

Dense builds its **own catalog** from your Pydantic template (same concepts as delta/staged):

- **Paths** — Root `""`, then nested paths like `studies[]`, `studies[].experiments[]`. Phase 1 and Phase 2 use only these paths.
- **Identity** — Entities with `graph_id_fields` get stable keys for skeleton dedup and parent linkage.
- **Projected fill schema** — For Phase 2, the schema for each path is the model schema for that path **minus** nested child-path fields (so the LLM fills one node type at a time).

Best practices for identity and linkage (e.g. from [Schema design for staged extraction](../schema-definition/staged-extraction-schema.md)) apply to dense as well.

### Schema tips for dense

When your documents have **series of values** (e.g. one batch with solid loading φ = 0.10 … 0.29) or **conditional values** (e.g. different pre-shear rates for different concentrations), define **SeriesDefinition** and **ConditionalValue** in your template (or reuse the pattern from the [rheology template](../../examples/templates/rheology_research.py)):

- **Series-aware:** Add `SeriesDefinition | None` to entities that can represent a series (Batch, Sample, Condition). The fill phase can then populate `variable_name` and `variable_values` (or `range_min`/`range_max`) so one node holds the full variation instead of forcing the skeleton to emit many nodes. See [Advanced patterns: Series and conditional values](../schema-definition/advanced-patterns.md#series-and-conditional-values).
- **Conditional logic:** Replace scalar fields that can vary by context (e.g. `pre_shear_rate: float`) with `List[ConditionalValue]` so the fill phase can return multiple `{value, unit, condition}` entries when the document gives different values under different conditions.

The **Phase 1 (skeleton) prompt** instructs the LLM to extract both **localized** entities (per-identifier instances such as specific figures or tables) and **global/singleton** entities (shared configs, protocols, methodologies) so that root-level or shared nodes are not dropped when they lack a figure- or table-style label. It also includes a granularity heuristic: container nodes (e.g. Dataset with child Curves) should be split into separate instances when child identifiers differ (e.g. Figure 2a vs Figure 7c) or when data comes from distinct sources (different Figures/Tables). No schema change is required for this; the prompt is generic.

---

## Configuration and options

All options can be set in Python via `PipelineConfig` or a config dict. Dense-specific CLI flags include `--dense-prune-barren-branches`; see config file or API for the full set.

| Python (config dict) | Default | Description |
|----------------------|--------|-------------|
| `extraction_contract` | `"direct"` | Set to `"dense"` to enable dense extraction. |
| `use_chunking` | `True` | Must be enabled for dense. |
| `dense_skeleton_batch_tokens` | `1024` | Max tokens per skeleton batch (Phase 1). Capped at 4096. Recommended 1024–2048 for long documents so multiple batches (and parallel workers) are used. |
| `chunk_max_tokens` | e.g. `512` | Max tokens per chunk. **Required for batching:** if the document is one giant chunk, you get one batch regardless of `dense_skeleton_batch_tokens`. Use 512–1024 per chunk so long documents split into many chunks, then into multiple skeleton batches. |
| `dense_fill_nodes_cap` | `5` | Max node instances per LLM call in the fill pass (Phase 2). |
| `parallel_workers` | `1` | When > 1, Phase 1 skeleton batches and Phase 2 fill batches run in parallel to reduce wall-clock time. |
| `max_pass_retries` / `staged_pass_retries` | `1` | Retries per LLM call when output validation fails. |
| `dense_quality_require_root` | `True` | Require at least one root instance after Phase 1. |
| `dense_quality_min_instances` | `1` | Minimum skeleton node count after Phase 1; below this, gate fails. |
| `dense_prune_barren_branches` | `False` | If `True`, remove branch nodes that have no filled children and no scalar data (barren branches) after Phase 2. Domain-agnostic; controlled via config or `--dense-prune-barren-branches` on the CLI. |

With `debug=True`, artifacts such as `dense_skeleton_graph.json` are written to the debug directory.

### Performance

For long documents, set `parallel_workers` (e.g. 2–4) so that Phase 1 skeleton batches and Phase 2 fill batches run in parallel; this reduces wall-clock time without changing merge logic or output quality. **Chunk and batch sizing:** Both **chunk size** (`chunk_max_tokens`, e.g. 512–1024 per chunk) and **dense_skeleton_batch_tokens** (e.g. 1024–2048) determine how many Phase 1 batches you get. If the chunker produces one huge chunk for the whole document, you will get only one batch no matter how low you set `dense_skeleton_batch_tokens`. Set a strict max tokens per chunk so the document splits into many chunks; then the batch token limit groups those chunks into multiple batches. The options `dense_fill_nodes_cap` and `dense_skeleton_batch_tokens` trade off the number of LLM calls vs tokens per call (fewer, larger batches mean fewer calls but more tokens per request).

---

## Usage

### Python API

```python
from docling_graph import PipelineConfig, run_pipeline

config = PipelineConfig(
    source="document.pdf",
    template="templates.MyNestedTemplate",
    backend="llm",
    processing_mode="many-to-one",
    extraction_contract="dense",
    use_chunking=True,
    dense_skeleton_batch_tokens=1024,
    dense_fill_nodes_cap=5,
)
context = run_pipeline(config)
```

### CLI

```bash
uv run docling-graph convert document.pdf \
  --template "templates.MyNestedTemplate" \
  --processing-mode many-to-one \
  --extraction-contract dense
```

---

## Trace and debugging

When dense runs, the pipeline can emit trace data (e.g. via `trace_data`) containing:

- `contract: "dense"`
- `phase1_elapsed`, `phase2_elapsed` (seconds)
- `skeleton_nodes`, `path_counts`

With `debug=True`, the debug directory may contain `dense_skeleton_graph.json` (merged Phase 1 nodes).

---

## Related

- [Delta Extraction](delta-extraction.md) — Chunk-based graph IR extraction
- [Staged Extraction](staged-extraction.md) — ID pass → fill pass (full document)
- [Extraction Backends](extraction-backends.md) — LLM vs VLM and contracts
- [Configuration reference](../../reference/config.md)
