# Staged Extraction - EXPERIMENTAL

## Overview

**Staged extraction** is a multi-pass extraction mode for the LLM backend when using **many-to-one** processing. It is useful for complex nested templates and for models that benefit from smaller, focused tasks.

Set `extraction_contract="staged"` in your config or use `--extraction-contract staged` on the CLI. Staged currently uses **legacy prompt-schema mode** only (no API-level structured output) to avoid provider-specific failures; the global `structured_output` setting does not apply to staged.

!!! warning "Experimental feature - Not production-ready"
    **Staged extraction** is still in an **experimental** phase.
    
    Expect ongoing quality improvements, but also be aware that **clean breaks** may happen and **backward compatibility is not guaranteed** yet.


**When to use:**

- Nested Pydantic templates with lists and sub-objects (e.g. offers with included guarantees)
- You want stable identity-first extraction (IDs from the document, then fill)
- Direct single-pass extraction struggles with consistency

**When to use direct (default):**

- Flat or simple templates
- You prefer a single extraction pass and programmatic merge

---

## How It Works

Staged extraction runs three conceptual phases:

1. **Catalog** — Built from your Pydantic template. Derives all extractable node types and paths (e.g. root, `offres[]`, `offres[].garanties_incluses[]`) and their `graph_id_fields` and parent rules.

2. **ID pass** — The LLM discovers **node instances** per path with only the identifiers (from `graph_id_fields`) and parent linkage. Output is a **skeleton**: path, ids, parent. No full content yet. By default only paths that have identity fields are sent (reducing prompt size and truncation). ID pass can be auto-sharded when the catalog is large (root and top-level paths first); shards run in parallel when `parallel_workers` > 1.

3. **Fill pass** — For each path, the LLM fills full schema content for the skeleton instances. Paths are processed in **bottom-up** order (leaf paths first). Fill calls can run in parallel. Each path gets a **projected** schema (no nested child paths in the same call), so root and children stay consistent. Results are merged into the root model by parent linkage.

4. **Quality gate** — After merge, a quick check runs (e.g. root instance present, minimum instances). If it fails, the pipeline can **fall back to direct extraction** so you still get a result; the trace will indicate why (e.g. `fallback_reason: "quality_gate_failed"`).

**List paths and many-to-many** — For paths that are lists under another list (e.g. `offres[].garanties_incluses[]`), the same child entity can belong to multiple parents. The pipeline keeps **one descriptor per (parent, child) pair** in the ID pass and merge, and fills each **unique child once** in the fill pass, then reuses that filled object for every parent. That preserves many-to-many relationships in the graph (e.g. one guarantee linked to several offers) without duplicate fill calls.

![Staged Extraction](../../assets/screenshots/staged_extraction.png)

---

## Schema requirements

Staged extraction succeeds when the **ID pass** can discover node instances (root and nested entities) and the **quality gate** passes. Your Pydantic template should be designed with that in mind:

- **Root model** must have `graph_id_fields` so at least one root instance can be discovered.
- **Entities** that should appear in the ID pass must have `graph_id_fields`; use required, short, extractable fields and add schema examples.
- **Components** (`is_entity=False`) are not identity paths by default; use `edge()` with `edge_label` when they must appear in the catalog.
- Keep **nesting depth** and catalog size reasonable to avoid truncation and excessive sharding.

For a domain-agnostic checklist, identity best practices, and troubleshooting (e.g. mapping `missing_root_instance` or `insufficient_id_instances` to schema fixes), see [Schema design for staged extraction](../schema-definition/staged-extraction-schema.md).

### Staged-friendly template guidelines

Templates with **many nested entities** (e.g. `list[Exclusion]` with `Exclusion` containing `list[Bien]`) produce a large catalog and more ID-pass shards. That can lead to slower runs, more truncation, and retries. To keep the ID pass fast and reliable:

- Prefer **`list[str]`** (or other shallow types) for high-cardinality nested concepts when you only need labels or short values (e.g. exclusion titles, bien names). The catalog does not create node paths for `list[str]`, so you get fewer paths and smaller ID responses.
- Reserve **nested Pydantic entities** (`list[SomeModel]`) for when you need full structure and identity in the graph (e.g. deduplication by id, edges to other entities).

Example: the MRH insurance template has a full variant ([`docs/examples/templates/cgv_mrh.py`](../../examples/templates/cgv_mrh.py)) with `Garantie.exclusions_specifiques: list[Exclusion]` and `Exclusion.biens_exclus: list[Bien]`, which yields many catalog paths. The staged-optimized variant ([`docs/examples/templates/cgv_mrh_staged.py`](../../examples/templates/cgv_mrh_staged.py)) uses `exclusions_specifiques: list[str]` and `biens_couverts: list[str]` on `Garantie`/`Option`, reducing catalog size and ID pass time while still filling the root structure. Use the staged variant when you want faster extraction and do not need full entity nodes for every exclusion or bien.

---

## Configuration and options

All options can be set in Python via `PipelineConfig` or a config dict passed to `run_pipeline()`. CLI flags (when available) override config-file defaults.

### Preset and overrides

The **preset** (`standard` or `advanced`) sets default values for retries, workers, fill cap, and ID shard size. Overrides apply when provided.

| Python (`PipelineConfig` / config dict) | CLI flag | Default | Description |
|----------------------------------------|----------|---------|-------------|
| `extraction_contract` | `--extraction-contract` | `"direct"` | Set to `"staged"` to enable staged extraction. |
| `staged_tuning_preset` | `--staged-tuning` | `"standard"` | Preset: `"standard"` or `"advanced"` (advanced = larger ID shards, larger fill batches). |
| `staged_pass_retries` | `--staged-retries` | preset (`standard`: 2) | Retries per staged pass when the LLM returns invalid JSON. |
| `parallel_workers` | `--parallel-workers` | preset (`standard`: 1) | Parallel workers for the fill pass and for the ID pass shards; also used for delta. |
| `staged_nodes_fill_cap` | `--staged-nodes-fill-cap` | preset (`standard`: 5) | Max node instances per LLM call in the fill pass. |
| `staged_id_shard_size` | `--staged-id-shard-size` | preset (`standard`: 0) | Paths per ID-pass call; `0` = no sharding or auto-shard when catalog is large. |

### ID pass

| Python (config dict) | CLI flag | Default | Description |
|----------------------|----------|---------|-------------|
| `staged_id_identity_only` | *(config only)* | `True` | Use only paths with identity fields in the ID pass (smaller prompts). |
| `staged_id_compact_prompt` | *(config only)* | `True` | Use compact ID prompt and omit full schema in user message. |
| `staged_id_auto_shard_threshold` | *(config only)* | `10` | If catalog paths exceed this and shard size is 0, auto-enable sharding. |
| `staged_id_shard_min_size` | *(config only)* | `2` | Minimum paths per shard when auto-sharding. |
| `staged_id_max_tokens` | `--staged-id-max-tokens` | `16384` | Max tokens for ID pass responses; avoids truncation on large catalogs. Set to `None` to use client default. |
| `staged_fill_max_tokens` | `--staged-fill-max-tokens` | `None` | Max tokens for fill pass responses; `None` = client default. |

### Quality gate

When the **quality gate** fails (e.g. no root instance, too few instances), the pipeline returns **direct extraction** instead of the staged result. Check the trace for `quality_gate` and `fallback_reason`.

| Python (config dict) | Default | Description |
|----------------------|---------|-------------|
| `staged_quality_require_root` | `True` | Require at least one root instance; if not met, gate fails. |
| `staged_quality_min_instances` | `1` | Minimum total skeleton instances for gate. |
| `staged_quality_max_parent_lookup_miss` | `0` | Max allowed parent lookup misses before gate fails. |

Quality gate options are not CLI flags; set them in a config file or config dict.

---

## Usage

### Python API

Pass options via `PipelineConfig` or a dict to `run_pipeline()`:

```python
from docling_graph import PipelineConfig, run_pipeline

config = PipelineConfig(
    source="document.pdf",
    template="templates.MyNestedTemplate",
    backend="llm",
    processing_mode="many-to-one",
    extraction_contract="staged",
    staged_tuning_preset="standard",  # or "advanced"
    # Optional overrides (preset defaults applied when not set):
    # staged_pass_retries=2,
    # parallel_workers=2,
    # staged_nodes_fill_cap=5,
    # staged_id_shard_size=0,
    # staged_id_max_tokens=16384,  # default; set None for client default
    # staged_fill_max_tokens=None,
    # staged_quality_require_root=True,
    # staged_quality_min_instances=1,
    # staged_quality_max_parent_lookup_miss=0,
)
context = run_pipeline(config)
```

### CLI

Staged-related flags (when using `--extraction-contract staged`):

```bash
# Enable staged
uv run docling-graph convert document.pdf \
  --template "templates.MyNestedTemplate" \
  --processing-mode many-to-one \
  --extraction-contract staged

# Preset and overrides
uv run docling-graph convert document.pdf \
  --template "templates.MyNestedTemplate" \
  --extraction-contract staged \
  --staged-tuning standard \
  --staged-retries 2 \
  --parallel-workers 2 \
  --staged-nodes-fill-cap 5 \
  --staged-id-shard-size 0

# Token limits (e.g. to avoid truncation)
uv run docling-graph convert document.pdf \
  --template "templates.MyNestedTemplate" \
  --extraction-contract staged \
  --staged-id-max-tokens 8192 \
  --staged-fill-max-tokens 8192
```

Options such as `staged_id_identity_only`, `staged_id_compact_prompt`, `staged_id_auto_shard_threshold`, and `staged_quality_*` have no CLI flags; set them in a config file or in a config dict when using the Python API.

See [Configuration reference](../../reference/config.md) and [convert command](../../usage/cli/convert-command.md#staged-extraction-tuning) for the full list.

**When to adjust:**

- **Structured output**: Staged uses legacy prompt-schema mode only (no API structured output); the global `structured_output` setting does not apply to staged.
- **Truncation or invalid ID output**: Default `staged_id_max_tokens=16384` reduces ID-pass truncation; increase or set `staged_fill_max_tokens` if fill responses are cut off.
- **Slow ID pass or “Response Truncated”**: For large catalogs or long documents, the ID pass can hit the response token limit, causing truncation, validation errors, retries, and shard splits—and much longer runtimes. Set `staged_id_max_tokens` explicitly (e.g. `16384` or `32768`) via CLI `--staged-id-max-tokens` or config so ID responses are less likely to truncate; this reduces retries and speeds up the ID pass.
- **Staged fallback to direct**: If the trace shows `fallback_reason: "quality_gate_failed"`, check `quality_gate.reasons` (e.g. missing root instance). Relax `staged_quality_require_root` or `staged_quality_min_instances` only if your template legitimately has no root or very few instances.
- **Large catalogs**: Defaults use identity-only paths and auto-sharding; tune `staged_id_auto_shard_threshold` or `staged_id_shard_size` via config if the ID pass is still too heavy. For very large catalogs, also increase `staged_id_max_tokens` to avoid truncation and extra retries.

---

## Next Steps

- [Schema design for staged extraction](../schema-definition/staged-extraction-schema.md) — Identity fields, linkage, and schema checklist for staged mode
- [Extraction Backends](extraction-backends.md) — LLM vs VLM and extraction contracts
- [Model Merging](model-merging.md) — How chunk results are merged
- [Configuration reference](../../reference/config.md) — Full config and staged fields
