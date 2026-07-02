# Dense Extraction

## Overview

**Dense extraction** is an LLM extraction contract for **many-to-one** processing that combines granular document structure with rich per-node data via a **two-phase "Skeleton-then-Fill"** flow. It is fully autonomous and designed for chunked, structure-aware extraction.

Set `extraction_contract="dense"` in your config or use `--extraction-contract dense` on the CLI. Chunking must be enabled (`use_chunking=True`, which is the default for many-to-one).

**Structured output:** Dense currently uses **legacy prompt-schema mode only** (no API-level structured output). This avoids provider-specific failures, empty/null JSON on skeleton batches, and repeated structured-output failures on fill. The global `structured_output` setting does not apply to dense.

**When to use:**

- You want both **fine-grained structure** (many distinct entities, e.g. every figure/experiment/curve) and **dense attribute data** (protocols, derived quantities, model fits) in one run.
- You need chunk-aware many-to-one extraction that first discovers entity instances and then fills them with complete per-node data.

**When to use direct or dense:**

- **Direct**: Flat or simple templates; single-pass extraction.
- **Dense**: Complex or long documents where you need strong structure discovery plus rich per-entity data.

> **Large documents:** a direct extraction returns the whole result in one LLM response, so a document much larger than the model's output budget will truncate no matter how the prompt is tuned. When this is detected on a direct run, the backend prints a one-time hint suggesting `extraction_contract="dense"`, which splits the work across many calls.

---

## How It Works

1. **Phase 1 (Skeleton)** — Chunks are packed into token-bounded batches. For each batch, the LLM is asked to **identify every distinct entity** per catalog path using a compact **handle contract**: each node is `{"i": <handle>, "path": ..., "ids": {...}, "p": <parent handle>}`, where `i` is a batch-local integer and `p` references the parent node's handle in the same response (no property values, no repeated parent objects). Copying a one-digit integer is far more reliable for small models than re-writing parent identifier strings, and it removes the repeated ancestry objects that used to dominate output tokens (and cause truncation). Identifier values must be **short labels copied verbatim from the document** — never sentences. Phase 1 includes a **scope boundary**: extract only entities that are the primary subject, direct output, or original creation of the document; do not extract external references, cited works, or third-party entities mentioned for context. Phase 1 uses a **skeleton-only** semantic guide (paths and identity fields only), so the LLM cannot see or emit other template properties. Handles are resolved to (path, ids) parent references immediately after parsing; results are then merged across batches with deduplication by (path, identity). Output: a merged skeleton graph (nodes with correct hierarchy and identities; properties empty).

   **Truncation resilience** — If a skeleton batch is too entity-dense to fit one response within the model's output budget (common on small local models with a low output cap), the model's output is truncated and nodes are lost. When this is detected, one retry with a larger output budget is attempted — bounded by the model's true context window, which is probed automatically from LiteLLM metadata or the local server's `/models` endpoint (e.g. vLLM's `max_model_len`), never configured by the user. If the larger budget also truncates, escalation is disabled for the rest of the run and the batch is **split in half and each half retried** (recursively, down to single chunks), so the dropped nodes are recovered. This is fully domain-agnostic and needs no per-model tuning; it is the primary safeguard that keeps large or deeply-nested documents from silently losing entities or failing the quality gate.

   **Small-model robustness** — Skeleton responses are validated **per node**: malformed entries (e.g. echoed schema fragments from very small models) are skipped and counted, while valid nodes in the same batch are kept. Model-emitted paths that drop the `[]` list markers are canonicalized against the catalog instead of being discarded. The root (path `""`) is a singleton by definition, so paraphrased root identifiers across batches are collapsed into a single root node.

   **Skeleton dedupe (`dense_dedupe`)** — After batch merge, duplicate skeleton nodes are collapsed. `off`: exact canonical-id dedup only. `standard` (default): adds one cheap id-space LLM call (no document content) that reviews the per-path instance lists and proposes alias groups — instances that refer to the same real-world entity at different granularities (e.g. a generic "LFP slurry batch" alongside the specific "LFP_20vol_5wtPVDF_4wtCB"). Aliases are merged into the most specific instance; the prompt explicitly forbids merging instances that differ by any parameter, quantity, condition, date, or index, and invalid responses are ignored, so this pass can only merge, never lose nodes. `aggressive`: also fuzzy-merges near-identical same-path identifier strings (OCR noise such as dropped accents or spacing); the similarity threshold is internal, and identifiers that differ in any digit run never merge.

2. **Quality gate** — After Phase 1, the pipeline requires a non-empty skeleton with at least one root instance. This is an invariant, not an option. If the gate fails, dense returns `None` and the strategy falls back to a direct single-call extraction of the full document. A sparse-but-valid dense result is always kept (there is no sparsity veto on dense output; dense relies on this gate instead).

3. **Phase 2 (Flesh)** — The skeleton is converted to per-path **descriptors** (path, ids, parent). Paths are processed in **bottom-up** order. For each path, descriptors are batched (up to `dense_fill_nodes_cap` per call). The LLM receives a **scoped document context** (by default, the skeleton batches where the instances were observed plus the document head; the root instance always gets the full document — set `dense_fill_context="full"` to always send everything) and the complete list of instance identifiers; it returns one filled object per instance, in order, according to the projected schema for that path. Short responses are padded so skeleton instances are never silently dropped, and identity values already captured by the skeleton are restored whenever the fill response omits them or returns something unusable (null, empty string, nested object). Filled objects are merged into a template-shaped root by attaching each object to its parent via descriptor linkage. Because LLMs drift on parent identifiers, the merge applies a **rescue ladder** per instance: exact id match → unique parent instance → unique fuzzy (canonical containment) id match → id-only placeholder parent materialized up the chain. Instances whose parent ids are entirely empty are rescued into a shared id-less bucket parent rather than dropped. After merge, branch nodes with no filled children and no scalar data (barren branches) are pruned — an invariant, not an option. Recoveries and drops are counted, logged, and written to `dense_merge_stats.json` in debug mode.

4. **Output** — A single template-shaped dict (validated to your Pydantic model as with other contracts).

---

## Schema Requirements

Dense builds its **own catalog** from your Pydantic template:

- **Paths** — Root `""`, then nested paths like `studies[]`, `studies[].experiments[]`. Phase 1 and Phase 2 use only these paths.
- **Identity** — Entities with `graph_id_fields` get stable keys for skeleton dedup and parent linkage.
- **Projected fill schema** — For Phase 2, the schema for each path is the model schema for that path **minus** nested child-path fields (so the LLM fills one node type at a time).

Best practices for identity and linkage from the schema-definition guides apply to dense as well.

### Schema tips for dense

When your documents have **series of values** (e.g. one batch with solid loading φ = 0.10 … 0.29) or **conditional values** (e.g. different pre-shear rates for different concentrations), define **SeriesDefinition** and **ConditionalValue** in your template (or reuse the pattern from the [rheology template](../../examples/templates/rheology_research.py)):

- **Series-aware:** Add `SeriesDefinition | None` to entities that can represent a series (Batch, Sample, Condition). The fill phase can then populate `variable_name` and `variable_values` (or `range_min`/`range_max`) so one node holds the full variation instead of forcing the skeleton to emit many nodes. See [Advanced patterns: Series and conditional values](../schema-definition/advanced-patterns.md#series-and-conditional-values).
- **Conditional logic:** Replace scalar fields that can vary by context (e.g. `pre_shear_rate: float`) with `List[ConditionalValue]` so the fill phase can return multiple `{value, unit, condition}` entries when the document gives different values under different conditions.

The **Phase 1 (skeleton) prompt** instructs the LLM to extract both **localized** entities (per-identifier instances such as specific figures or tables) and **global/singleton** entities (shared configs, protocols, methodologies) so that root-level or shared nodes are not dropped when they lack a figure- or table-style label. It also includes a granularity heuristic: container nodes (e.g. Dataset with child Curves) should be split into separate instances when child identifiers differ (e.g. Figure 2a vs Figure 7c) or when data comes from distinct sources (different Figures/Tables). No schema change is required for this; the prompt is generic.

---

## Configuration and options

All options can be set in Python via `PipelineConfig` or a config dict; each has a 1:1 CLI flag (e.g. `--dense-dedupe`, `--dense-fill-context`).

| Python (config dict) | Default | Description |
|----------------------|--------|-------------|
| `extraction_contract` | `"direct"` | Set to `"dense"` to enable dense extraction. |
| `use_chunking` | `True` | Must be enabled for dense. |
| `dense_skeleton_batch_tokens` | `1024` | Max tokens per skeleton batch (Phase 1). Capped at 4096. Recommended 1024–2048 for long documents so multiple batches (and parallel workers) are used. |
| `chunk_max_tokens` | e.g. `512` | Max tokens per chunk. **Required for batching:** if the document is one giant chunk, you get one batch regardless of `dense_skeleton_batch_tokens`. Use 512–1024 per chunk so long documents split into many chunks, then into multiple skeleton batches. |
| `dense_fill_nodes_cap` | `5` | Max node instances per LLM call in the fill pass (Phase 2). |
| `dense_fill_context` | `"scoped"` | Document context per fill call: `"scoped"` sends only the skeleton batches where the node was observed (plus the document head); `"full"` always sends the whole document. Scoped keeps Phase 2 cost proportional to the entities being filled. |
| `dense_dedupe` | `"standard"` | Skeleton dedupe intensity: `off` (exact dedup only), `standard` (adds the id-space reconciliation LLM call), `aggressive` (also fuzzy-merges OCR-noise id variants; internal thresholds, numeric differences never merge). |
| `parallel_workers` | `1` | When > 1, Phase 1 skeleton batches and Phase 2 fill batches run in parallel to reduce wall-clock time. |

Mandatory cleanup steps are **invariants** with no config surface: root singleton collapse, the root-required quality gate, barren-branch pruning, identity restoration from skeleton ids, and root-id semantic validation always run. The last clears a root identifier whose field name promises a number (e.g. `document_number`) but which holds multi-word, digit-free prose — a sparse-document mis-capture where the model grabbed a brand or title for lack of a real number — so Phase 2 can leave it empty instead of locking in the wrong value.

Every dense run writes its health counters to `metadata.json` (`results.dense`) and the markdown report (**Dense Extraction Statistics**): skeleton nodes discovered, truncated responses, batch splits, reconciled aliases, recovered/dropped parent links, and skeleton retention %. Regressions in these failure modes are therefore visible per run without debugging. With `debug=True`, artifacts such as `dense_skeleton_graph.json`, `dense_merge_stats.json` and `dense_run_stats.json` are written to the debug directory.

### Performance

For long documents, set `parallel_workers` (e.g. 2–4) so that Phase 1 skeleton batches and Phase 2 fill batches run in parallel; this reduces wall-clock time without changing merge logic or output quality. **Chunk and batch sizing:** Both **chunk size** (`chunk_max_tokens`, e.g. 512–1024 per chunk) and **dense_skeleton_batch_tokens** (e.g. 1024–2048) determine how many Phase 1 batches you get. If the chunker produces one huge chunk for the whole document, you will get only one batch no matter how low you set `dense_skeleton_batch_tokens`. Set a strict max tokens per chunk so the document splits into many chunks; then the batch token limit groups those chunks into multiple batches. The options `dense_fill_nodes_cap` and `dense_skeleton_batch_tokens` trade off the number of LLM calls vs tokens per call (fewer, larger batches mean fewer calls but more tokens per request). **Fill context:** with the default `dense_fill_context="scoped"`, each Phase 2 call sends only the document regions where the node was observed instead of the whole document; on long documents this cuts Phase 2 token volume by an order of magnitude. Use `dense_fill_context="full"` if your entities are described far away from where they are first mentioned.

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

- [Extraction Backends](extraction-backends.md) — LLM vs VLM and contracts
- [Pipeline Configuration](../pipeline-configuration/configuration-basics.md) — Core runtime settings
- [Schema Definition](../schema-definition/index.md) — Template design guidance
- [Configuration reference](../../reference/config.md)
