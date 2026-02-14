# Staged Extraction - EXPERIMENTAL

## Overview

**Staged extraction** is a multi-pass extraction mode for the LLM backend when using **many-to-one** processing. It is useful for complex nested templates and for models that benefit from smaller, focused tasks.

Set `extraction_contract="staged"` in your config or use `--extraction-contract staged` on the CLI.

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

2. **ID pass** — The LLM discovers **node instances** per path with only the identifiers (from `graph_id_fields`) and parent linkage. Output is a **skeleton**: path, ids, parent. No full content yet. By default only paths that have identity fields are sent (reducing prompt size and truncation). ID pass runs **sequentially** and can be auto-sharded when the catalog is large (root and top-level paths first).

3. **Fill pass** — For each path, the LLM fills full schema content for the skeleton instances. Paths are processed in **bottom-up** order (leaf paths first). Fill calls can run in parallel. Each path gets a **projected** schema (no nested child paths in the same call), so root and children stay consistent. Results are merged into the root model by parent linkage.

4. **Quality gate** — After merge, a quick check runs (e.g. root instance present, minimum instances). If it fails, the pipeline can **fall back to direct extraction** so you still get a result; the trace will indicate why (e.g. `fallback_reason: "quality_gate_failed"`).

![Staged Extraction](../../assets/screenshots/staged_extraction.png)

---

## Schema requirements

Staged extraction succeeds when the **ID pass** can discover node instances (root and nested entities) and the **quality gate** passes. Your Pydantic template should be designed with that in mind:

- **Root model** must have `graph_id_fields` so at least one root instance can be discovered.
- **Entities** that should appear in the ID pass must have `graph_id_fields`; use required, short, extractable fields and add schema examples.
- **Components** (`is_entity=False`) are not identity paths by default; use `edge()` with `edge_label` when they must appear in the catalog.
- Keep **nesting depth** and catalog size reasonable to avoid truncation and excessive sharding.

For a domain-agnostic checklist, identity best practices, and troubleshooting (e.g. mapping `missing_root_instance` or `insufficient_id_instances` to schema fixes), see [Schema design for staged extraction](../schema-definition/staged-extraction-schema.md).

---

## Tuning

Staged behavior is controlled by a **preset** and optional overrides.

| Option | Meaning | Default (standard preset) |
|--------|---------|---------------------------|
| `staged_tuning_preset` | `"standard"` or `"advanced"` (larger ID shards, larger fill batches) | `"standard"` |
| `staged_pass_retries` | Retries per staged pass when the LLM returns invalid JSON | preset |
| `parallel_workers` | Parallel workers for the fill pass (ID pass is sequential); also used for delta batch parallelism | preset |
| `staged_nodes_fill_cap` | Max node instances per LLM call in the fill pass | preset |
| `staged_id_shard_size` | Max catalog paths per ID-pass call; `0` = no sharding or auto-shard when catalog is large | preset |
| `staged_id_identity_only` | Use only paths with identity fields in the ID pass (smaller prompts, fewer truncations) | `True` |
| `staged_id_compact_prompt` | Use compact ID prompt and omit full schema in user message | `True` |
| `staged_id_auto_shard_threshold` | If catalog paths exceed this and shard size is 0, auto-enable sharding | `12` |
| `staged_quality_require_root` | Require at least one root instance; if not met, quality gate fails | `True` |
| `staged_quality_min_instances` | Minimum total skeleton instances for quality gate | `1` |
| `staged_id_max_tokens` / `staged_fill_max_tokens` | Optional token limits for ID and fill calls (e.g. to avoid truncation) | unset (use client default) |

When the **quality gate** fails (e.g. no root instance, too few instances), the pipeline returns **direct extraction** instead of the staged result so you still get output. Check the pipeline trace for `quality_gate` and `fallback_reason` to see why fallback occurred.

**Python:**

```python
from docling_graph import PipelineConfig, run_pipeline

config = PipelineConfig(
    source="document.pdf",
    template="templates.MyNestedTemplate",
    backend="llm",
    processing_mode="many-to-one",
    extraction_contract="staged",
    staged_tuning_preset="standard",  # or "advanced"
    # Optional overrides:
    # parallel_workers=2,
    # staged_nodes_fill_cap=10,
    # staged_id_shard_size=0,
    # staged_id_identity_only=True,
    # staged_id_compact_prompt=True,
    # staged_id_max_tokens=4096, staged_fill_max_tokens=8192,
)
context = run_pipeline(config)
```

**CLI:**

```bash
uv run docling-graph convert document.pdf \
  --template "templates.MyNestedTemplate" \
  --processing-mode many-to-one \
  --extraction-contract staged \
  --staged-tuning standard
```

See [Configuration reference](../../reference/config.md) and [convert command](../../usage/cli/convert-command.md#staged-extraction-tuning) for all options.

**When to adjust:**

- **Truncation or invalid ID output**: Set `staged_id_max_tokens` and/or `staged_fill_max_tokens` so ID and fill calls have enough headroom; the backend may retry once with higher tokens when truncation is detected.
- **Staged fallback to direct**: If the trace shows `fallback_reason: "quality_gate_failed"`, check `quality_gate.reasons` (e.g. missing root instance). You can relax `staged_quality_require_root` or `staged_quality_min_instances` if your template legitimately has no root or very few instances; otherwise improve template or document so the ID pass finds the expected structure.
- **Large catalogs**: Defaults use identity-only paths and auto-sharding; you can tune `staged_id_auto_shard_threshold` or `staged_id_shard_size` if ID pass is still too heavy.

---

## Next Steps

- [Schema design for staged extraction](../schema-definition/staged-extraction-schema.md) — Identity fields, linkage, and schema checklist for staged mode
- [Extraction Backends](extraction-backends.md) — LLM vs VLM and extraction contracts
- [Model Merging](model-merging.md) — How chunk results are merged
- [Configuration reference](../../reference/config.md) — Full config and staged fields
