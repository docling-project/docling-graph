# merge Command


## Overview

The `merge` command combines multiple exported knowledge graphs into a single coherent graph — **deterministically, with no LLM calls**.

Because docling-graph node IDs are content hashes derived from each entity's identity fields, the same real-world entity extracted from two different documents already carries the same node ID. Merging therefore reduces to key equality: same-ID nodes are folded together (attributes enriched, descriptions combined, conflicts recorded), edges are unioned, and provenance is preserved per source document.

**Key properties:**

- **Deterministic** — the same inputs always produce byte-identical output
- **Idempotent** — merging a graph with itself changes nothing
- **No LLM required** — ambiguous alias candidates are *proposed* into the merge report, never auto-fused; a human-edited decisions file confirms them
- **Fully audited** — every suppressed value and folded node is recorded with its source document

---

## Basic Usage

```bash
uv run docling-graph merge INPUTS... [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `INPUTS...` | Two or more (one is allowed for re-cleaning) run directories or `graph.json` exports, in precedence order — the first graph is the base |

### Examples

```bash
# Merge two convert runs
uv run docling-graph merge outputs/report_A/ outputs/report_B/ -o merged/

# Preview the merge plan without writing the merged graph
uv run docling-graph merge outputs/run_*/ --dry-run

# Merge with a template for re-keying and alias proposal (older v1 exports)
uv run docling-graph merge a/ b/ --template templates.invoices.InvoiceDocument
```

---

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | `outputs/merged_<timestamp>/` | Output directory |
| `--template`, `-t` | — | Dotted path (`module.Class`); enables re-keying and alias proposal for v1 exports (v2 exports are self-describing) |
| `--precedence` | `input-order` | Duplicate-group fold order: `input-order` or `richest` |
| `--conflicts` | `keep-first` | Scalar conflict policy: `keep-first` (record dropped values in the report), `keep-all` (also keep them on the node under `__conflicts__`), or `variants` (also reify them as `<Class>Variant` sub-nodes — see below) |
| `--combine-fields` | `description,summary` | Text fields merged with sentence-level dedup instead of first-wins |
| `--rekey / --no-rekey` | auto | Recompute node IDs from identity attributes before folding (guards against normalizer drift across docling-graph versions) |
| `--alias-decisions` | — | JSON file of confirmed alias candidates (edit the stubs from a previous `merge_report.json`) |
| `--dry-run` | off | Compute the full merge plan, write only `merge_report.json` |
| `--export-format` | — | Extra export beside `graph.json`: `csv` or `cypher` |
| `--strict-template-check / --no-strict-template-check` | strict | Fail (vs warn) when inputs were extracted with different template schemas, or when a passed `--template` doesn't match the inputs |
| `--report / --no-report` | on | Write `merge_report.json` + `report.md` |
| `--open / --no-open` | off | Open the interactive graph visualization after merging |

---

## What Gets Merged, and How

| Situation | Behavior |
|-----------|----------|
| Same node ID in two graphs | Nodes fold: empty fields filled, `description`/`summary` sentence-merged, scalar lists unioned |
| Conflicting scalar values | First graph wins; the suppressed value + its source document are recorded (and kept on the node with `--conflicts keep-all`). Strings that differ only in whitespace, case, or Unicode form (`"059/987 65 40"` vs `"059/9876540"` — OCR formatting noise) are treated as equal, not as conflicts |
| Same node ID from **unrelated documents** with conflicting values | **Split, not folded.** Some identities are only locally unique (line numbers, step indexes); when the two occurrences hang under no common root *and* folding would overwrite conflicting values, the later node is kept separate (`<id>__doc_<document_id>`) and recorded under `cross_document_splits`. One proven conflict extends to the whole (document pair, class) group: once `LineItem` "1" conflicts between two documents, their `LineItem` "2" is split too even when its values agree (`reason: same-class-conflict` in the report), because line-number identities from unrelated documents describe different instances. Re-extractions of the same document (shared root) still fold |
| Same edge, same label | Attributes fold |
| Same edge, different labels | First label wins; losing labels kept in `also_labels` and reported |
| Similar-but-not-identical names ("IBM" vs "International Business Machines") | **Never auto-merged.** Proposed as alias candidate stubs in the report with advisory similarity scores |
| Two different documents sharing a filename stem | Colliding filename-derived root IDs are split (skolemized) when provenance proves distinct documents |
| Duplicate inputs (same document converted twice) | Second copy absorbed with a warning |

### Conflict variants (`--conflicts variants`)

When two sources describe the *same* node but disagree on a value (a JPG and a PDF re-extraction of one invoice reading a total differently), `keep-first` keeps the winner and the losing value survives only in the report. With `--conflicts variants` the losing values stay **in the graph**: the canonical node is exactly what `keep-first` produces (consensus fields plus the precedence winner), and each conflicting source contributes one sub-node holding only its suppressed values:

```text
LineItem_1f9e8d… {quantity: 28.0, …}                     ← canonical (winner)
   └─ HAS_CONFLICT_VARIANT {fields: [quantity]} →
      LineItem_1f9e8d…__var_doc-bbbb {quantity: 26.0}    ← LineItemVariant, type: variant
```

Variant nodes carry `variant_of`, `variant_document_id`, and the source's own `__provenance__`, so "which document said 26.0?" is a graph query instead of a report lookup. They are `type: variant` with label `<Class>Variant` — easy to filter out downstream — and their ids are derived from (base id, document id), so re-merging a constituent input converges onto the existing variant and re-keying moves them in lockstep with their base. Splits are unaffected: nodes from unrelated documents stay whole, they never become canonical-plus-variant.

### Confirming alias candidates (human-in-the-loop, still no LLM)

1. Run a merge; open `merge_report.json` and find `alias_candidates`.
2. Flip `"confirm": true` on the pairs you vouch for, save as `decisions.json`. Each stub lists the `field_conflicts` a confirmed merge would contradict — many conflicts (e.g. different `currency`, `total_amount`, `issue_date`) usually mean two distinct instances sharing an identifier, not aliases.
3. Re-run with `--alias-decisions decisions.json`.

Confirmed pairs replay through the exact same guarded reconciliation pass the extraction pipeline uses — unproposed pairs are ignored, and the sibling-veto still applies. Confirmations the guards veto are surfaced under `ignored_alias_decisions` in `merge_report.json` with the veto reason, so nothing disappears silently.

---

## Output

```
merged_<timestamp>/
└── docling_graph/
    ├── graph.json           # the merged graph (self-describing v2 format)
    ├── merge_report.json    # sources, folds, conflicts, alias candidates
    ├── report.md            # human-readable summary
    ├── graph.html           # interactive visualization
    └── provenance/          # each source ledger verbatim + manifest.json
```

The layout mirrors a `convert` run, so `docling-graph inspect merged/` works unchanged, and merged outputs can themselves be re-merged.

---

## Python API

```python
from docling_graph import merge_graphs, MergePolicy

merged_graph, report = merge_graphs(
    inputs=["outputs/run_A/", "outputs/run_B/docling_graph/graph.json"],
    template=None,  # optional: class or dotted path
    policy=MergePolicy(precedence="input-order", conflicts="keep-first"),
)
```

`merge_graphs` also accepts in-memory `networkx.DiGraph` objects, so it composes with `run_pipeline` results.
