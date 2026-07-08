# Dense Extraction

## Overview

**Dense extraction** is an LLM extraction contract for **many-to-one** processing that combines granular document structure with rich per-node data via a **two-phase "skeleton-then-flesh"** flow. It is fully autonomous and designed for chunked, structure-aware extraction.

Set `extraction_contract="dense"` in your config or use `--extraction-contract dense` on the CLI. Chunking must be enabled (`use_chunking=True`, which is the default for many-to-one).

**Structured output:** Dense currently uses **legacy prompt-schema mode only** (no API-level structured output). This avoids provider-specific failures, empty/null JSON on skeleton batches, and repeated structured-output failures on fill. The global `structured_output` setting does not apply to dense.

**When to use:**

- You want both **fine-grained structure** (many distinct entities, e.g. every figure/experiment/curve) and **dense attribute data** (protocols, derived quantities, model fits) in one run.
- You need chunk-aware many-to-one extraction that first discovers entity instances and then fills them with complete per-node data.

**When to use direct, dense, or auto:**

- **Direct**: Flat or simple templates; single-pass extraction.
- **Dense**: Complex or long documents where you need strong structure discovery plus rich per-entity data.
- **Auto**: Let the pipeline decide per document. After conversion, when the real document size is known, `extraction_contract="auto"` resolves to direct only when a single full-document call fits both the model's context window and its output-token budget, and to dense otherwise. The decision and the numbers behind it are logged (`[AutoContract] Resolved contract=...`). Recommended when document sizes vary or when running local models with modest context windows.

> **Large documents:** a direct extraction returns the whole result in one LLM response, so a document much larger than the model's output budget will truncate no matter how the prompt is tuned. When this is detected on a direct run, the backend prints a one-time hint suggesting `extraction_contract="dense"` (or `"auto"`), which splits the work across many calls. Full-document calls whose input arithmetically cannot fit the model's context window are refused up front with an actionable error instead of failing after doomed round-trips to the provider.

---

## How It Works

### Architecture

--8<-- "docs/assets/flowcharts/dense_extraction.md"

1. **Phase 1 (Skeleton)** — Chunks are packed into token-bounded batches. For each batch, the LLM is asked to **identify every distinct entity** per catalog path using a compact **handle contract**: each node is

    - **Cross-batch parent references (reference handles)** — In sequential mode, entities found by earlier batches are advertised to later batches as a sliding window of **negative reference handles** (`{"i": -3, "path": …, "ids": …}`); a new child whose parent was already extracted simply sets `"p": -3` instead of re-emitting the parent. This closes the dominant dense recall gap observed in the field: models reliably obey a "do not re-output" instruction but ignore its re-emit-the-parent exception, so children of cross-batch parents used to arrive with no parent reference at all and were dropped at merge (retention as low as 38–44% on multi-page documents). Re-emitting the parent with identical ids still works as a fallback; resolved handle references are counted as `parents_from_already_found` in the run stats.

    - **Coverage second pass** — After the batch merge, chunks that produced **no** skeleton node are re-examined once when they collectively hold at least 10% of the document's tokens: they are repacked into fresh batches and retried with the full reference-handle list (so recovered children attach to already-discovered parents) and a prompt that explicitly licenses an empty response. Some zero-yield chunks are legitimately empty boilerplate; this pass recovers the ones that were outshone by entity-dense neighbors in their original batch, at a bounded cost of one extra batch round. Recovered nodes are counted as `coverage_pass_recovered`.

    - **Truncation resilience (split before escalate)** — If a skeleton batch is too entity-dense to fit one response within the model's output budget (common on small local models with a low output cap), the model's output is truncated and nodes are lost. A multi-chunk batch is **split in half and each half retried first** (recursively, down to single chunks) — splitting attacks the actual cause (too much content per call), while escalating the output budget first would just spend tokens chasing a repetition loop. Only once a batch is down to a **single chunk that still can't be split further** is one retry with a larger output budget attempted, bounded by the model's true context window (probed automatically from LiteLLM metadata or the local server's `/models` endpoint, e.g. vLLM's `max_model_len` — never configured by the user); if that also truncates, escalation is disabled for the rest of the run. If a single chunk is still unrecoverable after that, it is retried once more with a **minimal root-and-direct-children projection** (a much smaller output ask) so a degraded but chunk-grounded node survives where possible. Only if that also fails is the chunk's content recorded as dropped — surfaced as `skeleton_batches_failed` / `dropped_chunk_ids` in the run stats (see below), never silently lost. This is fully domain-agnostic and needs no per-model tuning; it is the primary safeguard that keeps large or deeply nested documents from silently losing entities or failing the quality gate.

    - **Small-model robustness** — Skeleton responses are validated **per node**: malformed entries (e.g. echoed schema fragments from very small models) are skipped and counted, while valid nodes in the same batch are kept. Model-emitted paths that drop the `[]` list markers are canonicalized against the catalog instead of being discarded. The root (path `""`) is a singleton by definition, so paraphrased root identifiers across batches are collapsed into a single root node.

    - **Skeleton dedupe (`dense_dedupe`)** — After batch merge, duplicate skeleton nodes are collapsed.

        - `off`: exact canonical-id dedup only (case/diacritic-insensitive; see [identity rules](../schema-definition/best-practices.md#identity-and-resolution-primitives)).
        - `standard` (default): adds one id-space LLM reconciliation call (no document content) on top of exact dedup. Deterministic **containment matches** (a same-path id that is a superset of another, e.g. `"nanoscaled LiFePO4 (LFP)"` containing `"LiFePO4 (LFP)"`, guarded by matching digit signatures and a unique-base requirement) are **proposed as candidates** in that call — never auto-applied, because containment cannot tell a same-entity refinement from a distinct product tier (`"CONFORT PLUS"` must survive next to `"CONFORT"`; the prompt carries an explicit tier guard). The call also reviews the full per-path instance lists for granularity aliases (a generic `"LFP slurry batch"` alongside the specific `"LFP_20vol_5wtPVDF_4wtCB"`), is forbidden from merging instances that differ by any parameter, quantity, condition, date, or index, and invalid responses are ignored — the pass can only merge, never lose nodes. A deterministic **co-occurrence veto** backs the prompt guard: instances of the same path first emitted from the *same chunk* are never merged even when the LLM says so — a document does not name one entity twice side by side, so same-chunk neighbors (two columns of a table header, two bullets of a list) are distinct entities by construction. Every applied merge and every veto is logged at INFO and, in debug mode, written to `dense_reconciliation.json` (instance lists, containment candidates, the LLM's answer, and the applied/vetoed events).
        - `aggressive`: also fuzzy-merges near-identical same-path identifier strings (OCR noise such as dropped accents or spacing) deterministically. The similarity threshold is internal, and identifiers that differ in any digit run never merge.

2. **Quality gate** — After Phase 1, the pipeline requires a non-empty skeleton with at least one root instance. This is an invariant, not an option. If the gate fails, `dense` returns `None` and the strategy falls back to a direct single-call extraction of the full document. A sparse-but-valid dense result is always kept (there is no sparsity veto on dense output; dense relies on this gate instead).

    Two root-identity invariants protect the graph anchor. During Phase 1, a root id value that merely **echoes the template class name** (e.g. `reference_document="AssuranceMRH"`) is cleared — schema echo is never document data, and it would make the root look filled while being unmatchable. After extraction (both contracts), if **every** root identity field is still empty, the first one falls back to the **source document's stem** (e.g. `insurance_terms`): the root is a singleton, so a synthetic identity is safe *for the root only*, and it keeps every root-anchored edge matchable across runs. Non-root nodes never receive synthetic identities; an exported node whose declared identity fields are all empty is reported as an integrity warning by the graph converter instead.

3. **Phase 2 (Flesh)** — The skeleton is converted to per-path **descriptors** (path, ids, parent). Paths are processed in **bottom-up** order. For each path, descriptors are batched (up to `dense_fill_nodes_cap` per call; paths whose fill schema carries id-only **reference lists** are filled one instance per call — batching sibling parents into one call was the observed cause of one parent absorbing every membership row of a summary table while its siblings stayed empty, and the fill prompt enforces "memberships must be stated for THIS instance" only when that is unambiguous). The LLM receives a **scoped document context** (by default, the skeleton batches where the instances were observed plus the document head; the root instance always gets the full document — set `dense_fill_context="full"` to always send everything) and the complete list of instance identifiers; it returns one filled object per instance, in order, according to the projected schema for that path. Short responses are padded so skeleton instances are never silently dropped, and identity values already captured by the skeleton are restored whenever the fill response omits them or returns something unusable (null, empty string, nested object). Filled objects are merged into a template-shaped root by attaching each object to its parent via descriptor linkage. Because LLMs drift on parent identifiers, the merge applies a **rescue ladder** per instance: exact id match → unique parent instance → unique fuzzy (canonical containment) id match → unique **co-located** parent (the single parent-path instance observed in the same source chunk, falling back to batch granularity) → id-only placeholder parent materialized up the chain → shared id-less bucket parent. Placeholder and bucket parents are materialized **only when the parent model does not require the missing identity fields**: a rescue parent violating its own required identity would later be deleted by template validation (taking every rescued child with it) or blanked by salvage into a phantom hub, so such orphans are dropped and counted instead. After merge, branch nodes with no filled children and no scalar data (barren branches) are pruned — an invariant, not an option. Recoveries, bucket attachments, and drops are counted, logged, and written to `dense_merge_stats.json` in debug mode.

4. **Output** — A single template-shaped dict (validated to your Pydantic model as with other contracts).y

---

## Schema Requirements

Dense builds its **own catalog** from your Pydantic template:

- **Paths** — Root `""`, then nested **entity** paths like `studies[]`, `studies[].experiments[]`. Phase 1 and Phase 2 use only these paths.
- **Components stay inline** — models with `is_entity=False` never become catalog paths (edge-labeled or not). They are value objects without identity, which Phase 1 cannot reliably discover; their fields remain in the parent's projected fill schema and are filled together with the parent.
- **Reference fields get no path** — fields marked `edge(..., reference=True)` (see [Relationships — Reference edges](../schema-definition/relationships.md#reference-edges-referencetrue)) are re-included in the parent's fill schema as **id-only projections**: the parent's own fill call emits the membership list, so per-parent links survive and there is no separately-discovered child whose parent reference could drift.
- **Identity** — Entities with `graph_id_fields` get stable keys for skeleton dedup and parent linkage.
- **Projected fill schema** — For Phase 2, the schema for each path is the model schema for that path **minus** nested entity child-path fields (so the LLM fills one node type at a time), **plus** inline components and id-only reference projections.
- **Phase 1 semantic guide** — each path line carries the class docstring's first ~240 characters and identity-field examples; front-load the discriminating sentence of every entity docstring.

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
| `dense_skeleton_batch_tokens` | `2048` | Max tokens per skeleton batch (Phase 1). Capped at 4096. Recommended 1024–2048 for long documents so multiple batches (and parallel workers) are used. |
| `chunk_max_tokens` | e.g. `512` | Max tokens per chunk. **Required for batching:** if the document is one giant chunk, you get one batch regardless of `dense_skeleton_batch_tokens`. Use 512–1024 per chunk so long documents split into many chunks, then into multiple skeleton batches. |
| `dense_fill_nodes_cap` | `5` | Max node instances per LLM call in the fill pass (Phase 2). |
| `dense_fill_context` | `"scoped"` | Document context per fill call: `"scoped"` sends only the skeleton batches where the node was observed (plus the document head); `"full"` always sends the whole document. Scoped keeps Phase 2 cost proportional to the entities being filled. |
| `dense_dedupe` | `"standard"` | Skeleton dedupe intensity: `off` (exact dedup only), `standard` (adds the id-space reconciliation LLM call), `aggressive` (also fuzzy-merges OCR-noise id variants; internal thresholds, numeric differences never merge). |
| `parallel_workers` | `1` | When > 1, Phase 1 skeleton batches and Phase 2 fill batches run in parallel to reduce wall-clock time. |

Mandatory cleanup steps are **invariants** with no config surface: root singleton collapse, the root-required quality gate, barren-branch pruning, identity restoration from skeleton ids, and root-id semantic validation always run. The last clears a root identifier whose field name promises a number (e.g. `document_number`) but which holds multi-word, digit-free prose — a sparse-document mis-capture where the model grabbed a brand or title for lack of a real number — so Phase 2 can leave it empty instead of locking in the wrong value.

Every dense run writes its health counters to `metadata.json` (`results.dense`) and the markdown report (**Dense Extraction Statistics**): skeleton nodes discovered, effective parallel worker count, Phase 1/2 wall-clock, truncated responses, batch splits, reconciled aliases, cross-batch parents resolved via reference handles (`parents_from_already_found`), nodes recovered by the coverage second pass (`coverage_pass_recovered`), recovered/dropped parent links, and **merge retention %** (the fraction of skeleton nodes that survived the fill→graph merge). Two additional counters make source coverage — not just merge health — visible: `skeleton_batches_failed` and `dropped_chunk_ids` count chunks that produced **no** skeleton node even after the truncation-recovery ladder above, and `chunk_coverage_pct` is the fraction of chunks that contributed at least one node. Merge retention can read 100% even when whole chunks were lost upstream of the merge step, so the report renders a prominent warning whenever `skeleton_batches_failed > 0` rather than letting a lossy run hide behind a clean-looking retention figure. Regressions in these failure modes are therefore visible per run without debugging. With `debug=True`, artifacts such as `dense_skeleton_graph.json`, `dense_merge_stats.json` and `dense_run_stats.json` are written to the debug directory.

### Provenance

Dense is the contract with the richest [grounding](../graph-management/provenance.md): the skeleton phase's per-batch bookkeeping lets every extracted node be traced back to the chunk(s) it was observed in, and — when its final identifier appears verbatim in the document — to an exact chunk and page. This is fully deterministic and adds no prompt or output-schema overhead; it is controlled by the separate top-level `provenance` setting (default `"standard"`), not by any dense-specific option. See [Data Grounding & Provenance](../graph-management/provenance.md) for the full picture, including how it differs for the direct contract.

### Performance

For long documents, set `parallel_workers` (e.g. 2–4) so that Phase 1 skeleton batches and Phase 2 fill batches run in parallel; this reduces wall-clock time without changing merge logic or output quality. This matters most at scale: on a 124-page report the Phase 2 fill dominates wall-clock (several hundred to >1000 s sequential) and each fill call is independent, so it parallelizes almost linearly. The default stays `1` (conservative, fully deterministic ordering); raise it explicitly for large documents. The parent-linkage recovery (text-anchor / adjacency / reconciliation) is order-independent, so parallel workers do not change which parents children attach to. **Chunk and batch sizing:** Both **chunk size** (`chunk_max_tokens`, e.g. 512–1024 per chunk) and **dense_skeleton_batch_tokens** (e.g. 1024–2048) determine how many Phase 1 batches you get. If the chunker produces one huge chunk for the whole document, you will get only one batch no matter how low you set `dense_skeleton_batch_tokens`. Set a strict max tokens per chunk so the document splits into many chunks; then the batch token limit groups those chunks into multiple batches. The options `dense_fill_nodes_cap` and `dense_skeleton_batch_tokens` trade off the number of LLM calls vs tokens per call (fewer, larger batches mean fewer calls but more tokens per request). **Fill context:** with the default `dense_fill_context="scoped"`, each Phase 2 call sends only the document regions where the node was observed instead of the whole document; on long documents this cuts Phase 2 token volume by an order of magnitude. Use `dense_fill_context="full"` if your entities are described far away from where they are first mentioned.

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
    dense_skeleton_batch_tokens=2048,
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

When dense runs, the pipeline emits a `dense_trace_emitted` trace event (see [Trace Data Debugging](../../usage/advanced/trace-data-debugging.md)) containing:

- `contract: "dense"`
- `phase1_elapsed`, `phase2_elapsed` (seconds)
- `skeleton_nodes`, `path_counts`
- `merge_stats`, `run_stats` (the same counters written to `metadata.json` and the report)

With `debug=True`, the debug directory (`outputs/{document}_{timestamp}/debug/`) contains:

| File | Contents |
|------|----------|
| `dense_skeleton_graph.json` | Merged Phase 1 skeleton nodes (paths + identities, no property values) |
| `dense_merge_stats.json` | Phase 2 merge-into-root stats: attached/recovered/dropped instance counts |
| `dense_run_stats.json` | The same run-level counters surfaced in `metadata.json` `results.dense` |
| `dense_provenance.json` | The full provenance ledger for the run (present when `provenance` is not `"off"`; see [Data Grounding & Provenance](../graph-management/provenance.md)) |
| `trace_data.json` | The full step-by-step pipeline trace |

---

## Related

- [Extraction Backends](extraction-backends.md) — LLM vs VLM and contracts
- [Pipeline Configuration](../pipeline-configuration/configuration-basics.md) — Core runtime settings
- [Schema Definition](../schema-definition/index.md) — Template design guidance
- [Data Grounding & Provenance](../graph-management/provenance.md) — tracing nodes back to source chunks and pages
- [Configuration reference](../../reference/config.md)
