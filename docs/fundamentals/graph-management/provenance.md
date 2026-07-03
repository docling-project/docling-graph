# Data Grounding & Provenance

## Overview

**Provenance** maps every graph node back to *where in the source document* its data came from: the chunk(s), the page(s), and (optionally) the exact character span of the identifier that was found. It is fully **deterministic** — grounding never involves an LLM call, never changes a prompt, and never changes what gets extracted. It runs as bookkeeping alongside extraction and is bound to the graph after conversion.

Enabled by default (`provenance="standard"`), it works with **both** extraction contracts ([direct](../extraction-process/extraction-backends.md) and [dense](../extraction-process/dense-extraction.md)) and needs no changes to your Pydantic templates.

**What you get:**

- A `__provenance__` attribute on every entity node in the graph, with the resolved location.
- A standalone `provenance.json` ledger next to `graph.json`, containing the full chunk index (including chunk **text**) and per-node lineage — self-contained, so you can go from a node straight to the source snippet and page without re-running the pipeline.

---

## Quick example

```python
from docling_graph import PipelineConfig, run_pipeline

config = PipelineConfig(
    source="research_paper.pdf",
    template="templates.ScholarlyRheologyPaper",
    extraction_contract="dense",
    provenance="standard",  # default — can be omitted
)
context = run_pipeline(config)

graph = context.knowledge_graph
for node_id, data in graph.nodes(data=True):
    prov = data.get("__provenance__", {})
    print(node_id, "->", prov.get("match"), prov.get("pages"))
```

```
SlurryComponent_a1b2c3 -> verbatim [1, 2, 3]
SlurryRheologyStudy_9f8e -> observed [1, 2]
ScholarlyRheologyPaper_44de -> None   # scope: document (see below)
```

---

## The `__provenance__` node attribute

Every entity node (see [Entities vs Components](../schema-definition/entities-vs-components.md)) gets exactly one `__provenance__` view, in one of four shapes. The name is a dunder (`__provenance__`) precisely so it can never collide with a template field — Pydantic forbids dunder field names.

### 1. Verbatim — exact location

The node's identifier was found literally in the document text via a deterministic string scan:

```json
{
  "document_id": "9f1a2b3c4d5e6f70",
  "match": "verbatim",
  "chunks": [10],
  "pages": [4]
}
```

### 2. Approximate — batch-level (dense only)

No verbatim match, but the dense skeleton phase saw the node while reading a known set of chunks:

```json
{
  "document_id": "9f1a2b3c4d5e6f70",
  "match": "observed",
  "chunks": [3, 4],
  "pages": [2],
  "approximate": true
}
```

`approximate: true` is the honesty signal — treat this as "somewhere in these chunks," not an exact cite.

### 3. Document scope

The whole document is the best available answer — used for the root node (which the dense fill phase always reads against the full document) and, for the **direct** contract, any node whose identifier can't be located verbatim:

```json
{
  "document_id": "9f1a2b3c4d5e6f70",
  "scope": "document"
}
```

### 4. Unresolved (dense only)

No verbatim match and no skeleton observation exists for this node. This is the deliberate failure mode — **an absent or wrong location is worse than a vague one, so the pipeline never guesses**:

```json
{"status": "unresolved"}
```

!!! note "Fail-empty, never fail-wrong"
    The binder resolution ladder is: **verbatim (exact) → observed (approximate) → document scope (direct only) → unresolved (dense only)**. Nothing ever fabricates a location; a node's view only gets *more* precise across a pipeline run, never less honest.

### Capping and detail

`chunks` is capped at 8 entries by default; when more chunks match, a `chunks_omitted` count is added. The full, uncapped anchor list always lives in `provenance.json` — the node attribute is intentionally small so it doesn't bloat the graph export.

With `provenance="detailed"`, verbatim views also carry `spans` (character offsets, capped at 4):

```json
{
  "document_id": "9f1a2b3c4d5e6f70",
  "match": "verbatim",
  "chunks": [10],
  "pages": [4],
  "spans": [{"chunk": 10, "start": 128, "end": 138}]
}
```

---

## `provenance.json` — the full ledger

Written next to `graph.json` (i.e. `docling_graph/provenance.json`) whenever `provenance` is not `"off"`. It is the source of truth the node attribute is a compact view of, and it is **self-contained**: every referenced chunk's text is stored inline, so you never need to re-run extraction to trace a node back to its source.

```json
{
  "version": 1,
  "document": {
    "document_id": "9f1a2b3c4d5e6f70",
    "source": "research_paper.pdf",
    "input_type": "document",
    "converted_at": "2026-07-03T15:49:20",
    "page_count": 8,
    "template_name": "ScholarlyRheologyPaper",
    "template_schema_hash": "3a7c..."
  },
  "resolution": "span",
  "node_level": true,
  "chunks": {
    "10": {
      "chunk_id": 10,
      "batch_index": 4,
      "page_numbers": [4],
      "doc_item_refs": ["#/texts/57"],
      "headings": ["3. Results"],
      "token_count": 118,
      "text_hash": "1c2d3e4f5a6b7c8d",
      "char_length": 612,
      "text": "3. Results\nUsing the coarse NMC powder, the slurries exhibit ...",
      "resplit_of": null
    }
  },
  "nodes": {
    "studies[].experiments[].slurry_batch.formulation.components[]|material_name=lifepo4": {
      "identity_key": "studies[].experiments[].slurry_batch.formulation.components[]|material_name=lifepo4",
      "catalog_path": "studies[].experiments[].slurry_batch.formulation.components[]",
      "node_type": "SlurryComponent",
      "ids": {"material_name": "LiFePO4"},
      "anchors": [
        {"document_id": "", "chunk_id": 0, "kind": "verbatim", "span": [42, 49]},
        {"document_id": "", "chunk_id": 10, "kind": "observed", "span": null}
      ],
      "merged_from": [],
      "synthetic": false,
      "dropped": false,
      "fill_batches": [0],
      "notes": []
    }
  },
  "bind_stats": {
    "nodes_seen": 16,
    "bound_verbatim": 7,
    "bound_observed": 8,
    "bound_document": 1,
    "unresolved": 0
  }
}
```

### Field reference

| Field | Meaning |
|-------|---------|
| `document` | Source identity: a content hash (`document_id`, stable across runs for the same input bytes), path/URL, detected input type, page count, template name and a hash of its JSON schema (for reproducibility). |
| `resolution` | The precision this run actually achieved: `"document"`, `"chunk"`, or `"span"` (upgraded to `"span"` once at least one node is verbatim-located). |
| `node_level` | `true` for dense (per-node skeleton entries exist), `false` for direct (chunk index only, no per-node entries — see [Dense vs. direct grounding](#dense-vs-direct-grounding)). |
| `chunks` | Every chunk the document was split into, keyed by `chunk_id`, each with its **text**, page numbers, docling item refs (`doc_item_refs`, reach bounding boxes via the exported `docling/document.json`), heading trail, and a content hash (`text_hash`) for drift detection. |
| `nodes` | One entry per grounded identity, keyed by a canonical `identity_key` (`"{catalog_path}|{field}={canonical_value},..."`). Holds the anchor list, dedup/reconciliation lineage (`merged_from`), and audit flags (`synthetic`, `dropped`). |
| `bind_stats` | Coverage counters from the last binding pass — how many nodes landed in each resolution tier. |

### Anchor kinds

Each entry in `anchors` carries a `kind`, ordered by evidence strength (strongest first):

| Kind | Meaning | Produced by |
|------|---------|-------------|
| `verbatim` | The identifier value was found literally in this chunk's text (`span` set). | Binder-side deterministic scan (`anchor_scan.locate_values`) |
| `observed` | The dense skeleton phase asserted this node while reading this chunk. | Phase 1 skeleton bookkeeping |
| `reconciled` | Inherited when the skeleton reconciliation LLM call merged an alias into this node; the absorbed identity stays visible via `merged_from`. | Skeleton dedupe reconciliation |
| `derived` | A synthetic parent (rescue-ladder placeholder/bucket) inheriting its children's anchors. | `merge_filled_into_root` orphan rescue |

---

## Dense vs. direct grounding

Both contracts use the **same binder and the same deterministic verbatim scan**; they differ only in what raw material feeds it.

**Dense** builds a *node-level* ledger (`node_level: true`): the Phase 1 skeleton records which chunk batch a node was observed in, and the fill phase's per-node identifiers are the ones the binder scans for verbatim matches. Because Phase 2 often refines a rough skeleton placeholder into the real value (e.g. skeleton `"SlurryComponent"` → filled `"LiFePO4"`), the ledger entry is **re-keyed to the final filled identifier** before binding, so grounding survives that refinement. A node that is neither verbatim-locatable nor skeleton-observed is `unresolved`.

**Direct** (single-call, whole-document extraction) has no skeleton, so it builds a *chunk index only* (`node_level: false`) — the document is chunked purely to give the binder something to scan (this does not change what gets extracted; the LLM call is unchanged). Every extracted node is verbatim-located the same way; a node whose identifier can't be found falls back to document scope instead of `unresolved`, since a direct call has no per-node signal to fall back to.

| | Dense | Direct |
|---|---|---|
| `node_level` | `true` | `false` |
| Approximate fallback | `observed` (skeleton-scoped) | — |
| No-match fallback | `unresolved` | `scope: "document"` |
| Chunking | Required for extraction | Added only to build the provenance index |

---

## Merge safety

Provenance survives every graph-building step without affecting node identity:

- **Skeleton dedup / reconciliation** (dense): when two skeleton entries collapse into one, their anchors union and the absorbed identity is recorded in `merged_from` — never silently dropped.
- **Orphan rescue ladder** (dense): a synthetic placeholder/bucket parent inherits its rescued children's anchors as `derived` evidence.
- **[Graph cleanup](graph-conversion.md#automatic-cleanup)**: when `GraphCleaner` merges two content-identical nodes, their `__provenance__` views are unioned rather than one being discarded — and `__provenance__` is excluded from the content hash that drives dedup, so two entities with identical data but different anchors still merge correctly.

Provenance never contributes to node identity (`graph_id_fields`) or content hashing — attaching or unioning a `__provenance__` view can never fork or merge a node that wouldn't otherwise fork or merge.

---

## Configuration

One field on `PipelineConfig`, top-level (not nested under dense-specific settings, since it applies to both contracts):

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    provenance="standard",  # "off" | "standard" | "detailed"  (default: "standard")
)
```

| Value | Behavior |
|-------|----------|
| `"off"` | No ledger, no `__provenance__` attribute, no `provenance.json`. Output is byte-identical to grounding never having existed. |
| `"standard"` (default) | `__provenance__` on every entity node + `provenance.json`, with verbatim locations where possible and approximate/document-scope fallbacks otherwise. |
| `"detailed"` | Everything in `standard`, plus character `spans` embedded in the node attribute (spans are always in `provenance.json` regardless of this setting). |

CLI flag: `--provenance {off|standard|detailed}`. See [convert command](../../usage/cli/convert-command.md#provenance).

**Token cost:** zero. Grounding never touches a prompt, a schema, or an LLM response — it is pure post-processing over data the pipeline already produced, so smaller models are completely unaffected.

---

## Reading provenance programmatically

```python
import json
from pathlib import Path

output_dir = Path("outputs/research_paper_pdf_20260703_154920/docling_graph")

# Node-level view (already on the graph)
graph_data = json.loads((output_dir / "graph.json").read_text())
for node in graph_data["nodes"]:
    prov = node.get("__provenance__", {})
    if prov.get("match") == "verbatim":
        print(f"{node['id']}: pages {prov['pages']}")

# Full ledger — resolve a node's grounding down to source text
ledger = json.loads((output_dir / "provenance.json").read_text())
for entry in ledger["nodes"].values():
    if entry["node_type"] != "SlurryComponent":
        continue
    for anchor in entry["anchors"]:
        if anchor["kind"] != "verbatim":
            continue
        chunk = ledger["chunks"][str(anchor["chunk_id"])]
        start, end = anchor["span"]
        print(entry["ids"], "->", chunk["text"][start:end], f"(page {chunk['page_numbers']})")
```

See [`15_provenance_grounding.py`](../../examples/scripts/15_provenance_grounding.py) for a complete runnable script.

### In exported formats

- **JSON** (`graph.json`): `__provenance__` is a native nested object on each node.
- **CSV** (`nodes.csv`): `__provenance__` is serialized as a JSON string column — parse it with `json.loads(row["__provenance__"])`.
- **Cypher** (`graph.cypher`): `__provenance__` is a string node property containing escaped JSON.

See [Export Formats](export-formats.md#provenance-in-exports) for details.

---

## Related

- [Dense Extraction](../extraction-process/dense-extraction.md) — the skeleton-then-fill contract that produces node-level grounding
- [Graph Conversion](graph-conversion.md) — where the provenance binder runs relative to node/edge creation and cleanup
- [Export Formats](export-formats.md) — how `__provenance__` is serialized per format
- [Configuration reference](../../reference/config.md) — the `provenance` field
- [Provenance API reference](../../reference/provenance.md) — `ProvenanceLedger`, `NodeProvenance`, `ChunkRecord` and related types
