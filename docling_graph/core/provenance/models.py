"""Structural Pydantic models for the deterministic provenance ledger.

These models never appear inside user templates: the ledger is a sidecar
artifact built from pipeline bookkeeping (never from LLM output) and bound to
graph nodes after conversion. See specs/data-grounding/architecture_spec.md.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AnchorKind = Literal["observed", "verbatim", "derived", "reconciled"]

ResolutionLevel = Literal["document", "page", "batch", "chunk", "span"]

# Evidence strength ordering for anchor kinds (higher = stronger claim).
KIND_STRENGTH: dict[str, int] = {
    "verbatim": 3,
    "observed": 2,
    "reconciled": 1,
    "derived": 0,
}


def text_hash(text: str) -> str:
    """Stable short hash of a text unit (chunk) for drift detection."""
    return hashlib.blake2b((text or "").encode("utf-8"), digest_size=8).hexdigest()


def content_hash(data: bytes) -> str:
    """Stable document-identity hash of raw source bytes."""
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def template_schema_hash(template: type[BaseModel]) -> str:
    """Stable hash of a template's JSON schema.

    Single source of truth for the value stored in
    ``DocumentOrigin.template_schema_hash`` and in format-v2 graph exports
    (``graph.graph["template_schema_hash"]``) — both must agree so merge-time
    compatibility gates can compare them.
    """
    return content_hash(json.dumps(template.model_json_schema(), sort_keys=True).encode("utf-8"))


class DocumentOrigin(BaseModel):
    """Identity of the source document for one pipeline run."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(..., description="Stable content hash of the normalized source")
    source: str = Field(..., description="Source path or URL as provided")
    input_type: str = Field(default="document", description="Detected input type")
    converted_at: datetime = Field(default_factory=datetime.now)
    page_count: int | None = Field(
        default=None, description="Page count when the format exposes pages"
    )
    template_name: str = Field(default="", description="Pydantic template class name")
    template_schema_hash: str = Field(
        default="", description="Hash of the template JSON schema (reproducibility)"
    )


DOCLANG_RESOLUTION = 512


def dclg_location_from_bbox(
    left: float, top: float, right: float, bottom: float, page_width: float, page_height: float
) -> tuple[int, int, int, int]:
    """Quantize a top-left-origin page bbox to DocLang's ``<location>`` grid.

    Replicates docling-core's DocLang serializer exactly on the default 512-grid
    (``clamp(round(512 * coord / page_dim), 0, 511)`` with corner normalization),
    so the result equals the values written in ``document.dclg``. Computed from
    the *original float* coordinates — quantizing after rounding the bbox to whole
    pixels could shift a value by one grid step.
    """

    def _quant(value: float, dim: float) -> int:
        return max(0, min(round(DOCLANG_RESOLUTION * value / dim), DOCLANG_RESOLUTION - 1))

    x0, x1 = min(left, right), max(left, right)
    y0, y1 = min(top, bottom), max(top, bottom)
    return (
        _quant(x0, page_width),
        _quant(y0, page_height),
        _quant(x1, page_width),
        _quant(y1, page_height),
    )


class ItemGeometry(BaseModel):
    """Page + bounding box of one Docling item backing a chunk.

    Derived deterministically from the ``DoclingDocument`` item's provenance
    (``prov[].page_no`` + ``bbox``), keyed by its ``self_ref`` so a chunk anchor
    can be located visually on the source page.

    ``bbox`` is ``(l, t, r, b)`` in whole page pixels, always **top-left origin**
    (BOTTOMLEFT sources are converted at capture; boxes that can't be normalized
    are dropped). ``dclg_location`` is the same box quantized to DocLang's 512
    grid — the exact ``<location>`` values in ``document.dclg`` — and is computed
    from the pre-rounding coordinates so it stays exact.
    """

    model_config = ConfigDict(frozen=True)

    ref: str = Field(..., description="Docling item self_ref, e.g. '#/texts/42'")
    page_no: int
    bbox: tuple[int, int, int, int] = Field(
        ..., description="(l, t, r, b) in whole page pixels, top-left origin"
    )
    page_width: int | None = Field(default=None, description="Page width in pixels")
    page_height: int | None = Field(default=None, description="Page height in pixels")
    dclg_location: tuple[int, int, int, int] | None = Field(
        default=None,
        description="bbox on DocLang's 512 grid — the <location> values in document.dclg",
    )


class ChunkRecord(BaseModel):
    """One chunker output unit; the atomic grounding unit.

    ``text`` is the enriched chunk text (what the LLM read). It is stored so the
    provenance ledger is self-contained: a consumer can map a node's
    ``chunks: [10]`` straight to the source text and page numbers without
    re-running the pipeline.
    """

    chunk_id: int
    batch_index: int = Field(..., description="Skeleton batch this chunk was grouped into")
    page_numbers: tuple[int, ...] = ()
    doc_item_refs: tuple[str, ...] = Field(
        default=(), description="Docling item self_refs, e.g. '#/texts/42'"
    )
    item_geometry: tuple[ItemGeometry, ...] = Field(
        default=(),
        description="Page + bbox for each backing Docling item (empty when the format has no geometry)",
    )
    headings: tuple[str, ...] = Field(
        default=(), description="Heading trail from chunk meta (context, not location)"
    )
    token_count: int = 0
    text_hash: str = Field(default="", description="Hash of the enriched chunk text")
    char_length: int = 0
    text: str = Field(default="", description="The enriched chunk text (source-of-truth snippet)")
    resplit_of: int | None = Field(
        default=None, description="Ordinal of the oversized parent chunk when re-split"
    )


class SourceAnchor(BaseModel):
    """One deterministic link from a node to a location in the source.

    ``document_id`` is empty for anchors belonging to the ledger's own
    document (the common single-document case); it is only set when anchors
    from multiple documents are ever merged.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str = ""
    chunk_id: int
    kind: AnchorKind = "observed"
    span: tuple[int, int] | None = Field(
        default=None, description="Char offsets in the enriched chunk text (verbatim only)"
    )


class NodeProvenance(BaseModel):
    """Full lineage for one skeleton identity."""

    identity_key: str
    catalog_path: str = ""
    node_type: str = ""
    ids: dict[str, str] = Field(
        default_factory=dict, description="Raw skeleton identity values (audit + fuzzy binding)"
    )
    anchors: list[SourceAnchor] = Field(default_factory=list)
    merged_from: list[str] = Field(
        default_factory=list, description="Identity keys absorbed via dedup / reconciliation"
    )
    synthetic: bool = Field(
        default=False, description="Rescue-ladder placeholder/bucket parent (no direct observation)"
    )
    dropped: bool = Field(
        default=False, description="Instance dropped downstream; entry kept for audit"
    )
    fill_batches: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ProvenanceLedger(BaseModel):
    """The complete grounding record for one pipeline run.

    ``document`` is finalized at the pipeline-stage level (the extraction
    internals know chunks, not source identity), so it is optional until then.
    """

    version: int = 2
    document: DocumentOrigin | None = None
    resolution: ResolutionLevel = "chunk"
    node_level: bool = Field(
        default=False,
        description="True when per-node skeleton entries exist (dense); False for direct",
    )
    chunks: dict[int, ChunkRecord] = Field(default_factory=dict)
    nodes: dict[str, NodeProvenance] = Field(default_factory=dict)
    bind_stats: dict[str, int] = Field(
        default_factory=dict, description="Graph binding coverage (set after bind)"
    )

    def pages_for_entry(self, entry: NodeProvenance) -> tuple[int, ...]:
        """Derive the sorted page set for a node from its anchors' chunks."""
        pages: set[int] = set()
        for anchor in entry.anchors:
            record = self.chunks.get(anchor.chunk_id)
            if record is not None:
                pages.update(record.page_numbers)
        return tuple(sorted(pages))

    def geometry_for_entry(self, entry: NodeProvenance) -> tuple[ItemGeometry, ...]:
        """Collect the item geometry of every chunk a node is anchored to.

        Empty when the source format exposes no geometry (e.g. markdown input).
        Deduplicated by ``self_ref`` while preserving first-seen order.
        """
        seen: set[str] = set()
        out: list[ItemGeometry] = []
        for anchor in entry.anchors:
            record = self.chunks.get(anchor.chunk_id)
            if record is None:
                continue
            for geom in record.item_geometry:
                if geom.ref not in seen:
                    seen.add(geom.ref)
                    out.append(geom)
        return tuple(out)


def document_level_ledger(text: str, page_count: int | None = None) -> ProvenanceLedger:
    """Last-resort degenerate ledger for the direct contract (spec §5).

    Used only when the direct path has no chunk index (e.g. raw text/markdown
    input with no DoclingDocument). The binder stamps every entity node with a
    document-scope view instead of marking it unresolved. When a chunk index is
    available, ``chunk_index_ledger`` is preferred so nodes can be located
    precisely.
    """
    return ProvenanceLedger(
        resolution="document",
        node_level=False,
        chunks={
            0: ChunkRecord(
                chunk_id=0,
                batch_index=0,
                page_numbers=tuple(range(1, page_count + 1)) if page_count else (),
                token_count=0,
                text_hash=text_hash(text),
                char_length=len(text),
                text=text,
            )
        },
        nodes={},
    )


def chunk_index_ledger(
    chunks: list[str], metadata: list[dict], resolution: ResolutionLevel = "chunk"
) -> ProvenanceLedger:
    """Chunk-index ledger with no per-node entries (direct contract).

    Carries the full chunk text + page numbers so the binder can verbatim-locate
    each extracted node's identifier deterministically, and so the exported
    ledger is self-contained.
    """
    records: dict[int, ChunkRecord] = {}
    for text, meta in zip(chunks, metadata, strict=False):
        chunk_id = int(meta.get("chunk_id", len(records)))
        records[chunk_id] = ChunkRecord(
            chunk_id=chunk_id,
            batch_index=0,
            page_numbers=tuple(p for p in (meta.get("page_numbers") or []) if isinstance(p, int)),
            doc_item_refs=tuple(str(r) for r in (meta.get("doc_item_refs") or [])),
            item_geometry=geometry_from_meta(meta),
            headings=tuple(str(h) for h in (meta.get("headings") or [])),
            token_count=int(meta.get("token_count") or 0),
            text_hash=meta.get("text_hash") or text_hash(text),
            char_length=int(meta.get("char_length") or len(text)),
            text=text,
        )
    return ProvenanceLedger(resolution=resolution, node_level=False, chunks=records, nodes={})


def geometry_from_meta(meta: dict) -> tuple[ItemGeometry, ...]:
    """Build ItemGeometry tuple from a chunk metadata dict's ``item_geometry`` key.

    Each entry is ``{"ref", "page_no", "bbox": [l,t,r,b], "page_width",
    "page_height", "dclg_location"}`` (all coordinates whole pixels, top-left
    origin). Malformed entries are skipped so a geometry glitch never breaks the
    ledger.
    """
    out: list[ItemGeometry] = []
    for raw in meta.get("item_geometry") or []:
        try:
            bbox = raw["bbox"]
            page_width = raw.get("page_width")
            page_height = raw.get("page_height")
            dclg = raw.get("dclg_location")
            out.append(
                ItemGeometry(
                    ref=str(raw["ref"]),
                    page_no=int(raw["page_no"]),
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    page_width=int(page_width) if page_width is not None else None,
                    page_height=int(page_height) if page_height is not None else None,
                    dclg_location=(
                        (int(dclg[0]), int(dclg[1]), int(dclg[2]), int(dclg[3]))
                        if dclg is not None
                        else None
                    ),
                )
            )
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return tuple(out)
