# Provenance API

## Overview

Deterministic data grounding: maps graph nodes back to their source document chunks and pages. No LLM involvement — the ledger is built from pipeline bookkeeping and bound to graph nodes after conversion.

**Module:** `docling_graph.core.provenance`

See the [Data Grounding & Provenance guide](../fundamentals/graph-management/provenance.md) for the conceptual overview and JSON examples; this page documents the underlying types and functions.

---

## Models

All models are plain structural Pydantic (`docling_graph.core.provenance.models`); none of them appear inside user templates.

### ProvenanceLedger

The complete grounding record for one pipeline run. Persisted as `provenance.json`.

```python
class ProvenanceLedger(BaseModel):
    version: int = 1
    document: DocumentOrigin | None = None
    resolution: Literal["document", "page", "batch", "chunk", "span"] = "chunk"
    node_level: bool = False
    chunks: dict[int, ChunkRecord] = {}
    nodes: dict[str, NodeProvenance] = {}
    bind_stats: dict[str, int] = {}
```

| Field | Description |
|-------|-------------|
| `document` | Source identity (`DocumentOrigin`); finalized once extraction completes, so it may be `None` mid-run. |
| `resolution` | The best precision this run achieved: `"document"` (whole-document fallback), `"chunk"` (batch-level), or `"span"` (at least one verbatim char-offset anchor exists). |
| `node_level` | `True` for the dense contract (per-node skeleton entries in `nodes`); `False` for direct (chunk index only). |
| `chunks` | `chunk_id -> ChunkRecord`, including the enriched chunk **text** — the ledger is self-contained. |
| `nodes` | `identity_key -> NodeProvenance`, one entry per grounded identity. |
| `bind_stats` | Populated after binding: `nodes_seen`, `bound_verbatim`, `bound_observed`, `bound_document`, `unresolved`. |

```python
def pages_for_entry(self, entry: NodeProvenance) -> tuple[int, ...]:
    """Sorted union of page numbers covered by an entry's anchors."""
```

### DocumentOrigin

```python
class DocumentOrigin(BaseModel):
    document_id: str            # content hash of the normalized source bytes
    source: str                 # path or URL as provided
    input_type: str = "document"
    converted_at: datetime
    page_count: int | None = None
    template_name: str = ""
    template_schema_hash: str = ""
```

### ChunkRecord

One chunker output unit — the atomic grounding unit.

```python
class ChunkRecord(BaseModel):
    chunk_id: int
    batch_index: int
    page_numbers: tuple[int, ...] = ()
    doc_item_refs: tuple[str, ...] = ()   # docling self_refs, e.g. "#/texts/42"
    headings: tuple[str, ...] = ()
    token_count: int = 0
    text_hash: str = ""
    char_length: int = 0
    text: str = ""                        # the enriched chunk text
    resplit_of: int | None = None         # parent chunk when produced by chunk_text_fallback
```

### SourceAnchor

One deterministic link from a node to a location in the source.

```python
class SourceAnchor(BaseModel):
    document_id: str = ""
    chunk_id: int
    kind: Literal["observed", "verbatim", "derived", "reconciled"] = "observed"
    span: tuple[int, int] | None = None   # char offsets into the chunk's enriched text
```

### NodeProvenance

Full lineage for one grounded identity.

```python
class NodeProvenance(BaseModel):
    identity_key: str
    catalog_path: str = ""
    node_type: str = ""
    ids: dict[str, str] = {}
    anchors: list[SourceAnchor] = []
    merged_from: list[str] = []   # identity keys absorbed via dedup/reconciliation
    synthetic: bool = False       # rescue-ladder placeholder/bucket parent
    dropped: bool = False         # instance dropped downstream; kept for audit
    fill_batches: list[int] = []
    notes: list[str] = []         # e.g. "scope:document", "identity:unkeyed"
```

---

## Ledger builders

```python
from docling_graph.core.provenance import document_level_ledger, chunk_index_ledger
```

### document_level_ledger()

```python
def document_level_ledger(text: str, page_count: int | None = None) -> ProvenanceLedger
```

Last-resort ledger with no chunk index — used only when a chunker is unavailable. Every node grounds to `{"scope": "document"}`.

### chunk_index_ledger()

```python
def chunk_index_ledger(
    chunks: list[str],
    metadata: list[dict],
    resolution: Literal["document", "page", "batch", "chunk", "span"] = "chunk",
) -> ProvenanceLedger
```

Builds a `node_level=False` ledger from a chunk list and its metadata (as returned by `DocumentProcessor.extract_chunks_with_metadata`). This is what the **direct** contract uses so its nodes can still be verbatim-located.

---

## Identity & views

```python
from docling_graph.core.provenance import (
    identity_key, identity_pairs, canonical_id_text,
    compact_view, merge_compact_views, PROVENANCE_NODE_ATTR,
)
```

### identity_key()

```python
def identity_key(path: str, ids: Mapping[str, Any], id_fields: Sequence[str]) -> str | None
```

Canonical, order-independent identity string for `(catalog path, ids)` — `"{path}|{field}={canonical_value},..."`, or `None` when no id value survives canonicalization. This is the single source of truth for identity: both the dense orchestrator's skeleton dedup and the binder derive their keys from this function (via `identity_pairs`), so recording and resolution can never disagree.

### compact_view()

```python
def compact_view(
    entry: NodeProvenance,
    ledger: ProvenanceLedger,
    max_anchors: int = 8,
    include_spans: bool = False,
) -> dict[str, Any]
```

Builds the small dict stored as a graph node's `__provenance__` attribute from a full ledger entry — verbatim anchors take precedence over approximate ones; see [anchor kinds](../fundamentals/graph-management/provenance.md#anchor-kinds).

### merge_compact_views()

```python
def merge_compact_views(a: dict | None, b: dict | None) -> dict | None
```

Unions two compact views (used by `GraphCleaner` when merging duplicate nodes). Never widens a claim: `unresolved` yields to any resolved view, and `approximate` is only kept when *both* sides lack a precise location.

### PROVENANCE_NODE_ATTR

```python
PROVENANCE_NODE_ATTR = "__provenance__"
```

The graph node attribute key. Import this constant rather than hardcoding the string.

---

## Binding

```python
from docling_graph.core.provenance.binder import bind_provenance
```

### bind_provenance()

```python
def bind_provenance(
    *,
    graph: nx.DiGraph,
    models: list[BaseModel],
    ledger: ProvenanceLedger,
    registry: NodeIDRegistry,
    template: type[BaseModel],
    include_spans: bool = False,
) -> dict[str, int]
```

Walks the validated model tree along the same catalog paths the extraction contract used, resolves each entity's node ID through the **same** `NodeIDRegistry` instance the graph converter used (so binding can never disagree on IDs), and annotates each node with a `compact_view`. Runs the deterministic verbatim locator against each node's *final* identifier values — this is what lets grounding survive the dense fill phase refining a skeleton placeholder into a real value. When the identity values don't locate (e.g. a direct-mode synthesized id that never appears verbatim), the binder falls back to the node's other short, distinctive `str` fields — a matched description or name grounds the node exactly instead of falling to a coarser tier. The same fallback applies to the root/`scope:document` entry, so a root with a distinctive attribute (an insurer name, a paper title) is pinned to its chunk instead of staying whole-document by default.

Returns bind stats (`nodes_seen`, `bound_verbatim`, `bound_observed`, `bound_document`, `unresolved`) and also writes them to `ledger.bind_stats`.

Not called directly in normal usage — `GraphConverter.pydantic_list_to_graph()` invokes it through an injected `provenance_binder` closure so the converter itself stays agnostic of the provenance module. See [Graph Conversion](../fundamentals/graph-management/graph-conversion.md).

---

## Anchor scan

```python
from docling_graph.core.provenance.anchor_scan import locate_identifier, locate_values, refine_ledger_spans
```

### locate_identifier() / locate_values()

```python
def locate_identifier(value: str, chunk_texts: Mapping[int, str]) -> list[tuple[int, tuple[int, int]]]
def locate_values(values: list[str], chunk_texts: Mapping[int, str]) -> list[tuple[int, tuple[int, int]]]
```

Pure string search: returns `(chunk_id, span)` pairs where a distinctive identifier appears verbatim. Guards against false precision — short and short-numeric values are skipped, and a value that matches too many chunks is treated as non-distinctive and skipped entirely rather than diluting the node's grounding. `locate_values` is the union across all of a node's identifier fields, and is what `bind_provenance` calls with each node's real values.

### refine_ledger_spans()

```python
def refine_ledger_spans(ledger: ProvenanceLedger, chunk_texts: Mapping[int, str]) -> int
```

Bulk variant that scans every non-synthetic ledger entry's *recorded* ids (rather than a specific model's final ids) and appends verbatim anchors in place. Returns the count added. Available for direct ledger post-processing; `bind_provenance` is the primary caller of the underlying locator in normal pipeline usage.

---

## Import patterns

`docling_graph.core.provenance` re-exports the models, ledger builders, and identity/view helpers. `binder` and `anchor_scan` are separate submodules (imported directly) to keep the package import-light — the dense orchestrator imports the top-level package at module load.

```python
# Guide-level usage — reading an existing ledger
from docling_graph.core.provenance import ProvenanceLedger

ledger = ProvenanceLedger.model_validate_json(
    (output_dir / "provenance.json").read_text()
)

# Building a ledger for a custom pipeline stage
from docling_graph.core.provenance import (
    ChunkRecord, DocumentOrigin, NodeProvenance,
    ProvenanceLedger, SourceAnchor,
)

# Binding and the locator are separate submodules
from docling_graph.core.provenance.binder import bind_provenance
from docling_graph.core.provenance.anchor_scan import locate_values
```

---

## Related APIs

- **[Data Grounding & Provenance guide](../fundamentals/graph-management/provenance.md)** — concepts, JSON shapes, configuration
- **[Converters](converters.md)** — `GraphConverter.pydantic_list_to_graph()`, `NodeIDRegistry`
- **[Config](config.md)** — the `provenance` field on `PipelineConfig`
- **[Dense Extraction](../fundamentals/extraction-process/dense-extraction.md)** — the skeleton phase that produces node-level observed anchors
