"""Canonical identity computation shared by extraction recording and graph binding.

This is the single source of truth for the identity chain that links a dense
skeleton descriptor to its eventual graph node:

    skeleton (path, ids) -> canonical pairs -> identity_key
    validated model graph_id_fields values -> same canonical pairs -> same key

The dense orchestrator's ``_skeleton_identity_key`` and the provenance binder
both delegate here, so recording and resolution can never disagree.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..utils.entity_name_normalizer import canonicalize_identity_for_dedup
from .models import KIND_STRENGTH, NodeProvenance, ProvenanceLedger

# Cap on chunk ids / spans serialized into the compact node-attribute view.
# Full anchor detail always lives in provenance.json.
DEFAULT_MAX_ANCHORS = 8
_MAX_SPANS = 4

PROVENANCE_NODE_ATTR = "__provenance__"


def identity_pairs(ids: Mapping[str, Any], id_fields: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Canonical, order-independent (field, canonical_value) pairs for an id dict.

    Prefers the schema-declared id_fields when any are present; otherwise falls
    back to all provided ids (same semantics the dense skeleton dedup has
    always used). Returns an empty tuple when no usable ids exist.
    """
    if id_fields:
        ordered = tuple(
            (f, canonicalize_identity_for_dedup(f, ids.get(f)))
            for f in id_fields
            if ids.get(f) is not None
        )
        if ordered:
            return tuple(sorted(ordered, key=lambda x: x[0]))
    return tuple(
        sorted(
            (str(k), canonicalize_identity_for_dedup(k, v)) for k, v in ids.items() if v is not None
        )
    )


def identity_key(path: str, ids: Mapping[str, Any], id_fields: Sequence[str]) -> str | None:
    """Serialized canonical identity for (catalog path, ids); None when unkeyable.

    A key is unkeyable when no id value survives canonicalization — callers
    assign positional fallbacks (which by design never bind exactly).
    """
    pairs = identity_pairs(ids, id_fields)
    if not pairs or all(not value for _, value in pairs):
        return None
    return f"{path}|" + ",".join(f"{field}={value}" for field, value in pairs)


def canonical_id_text(ids: Mapping[str, Any]) -> str:
    """Single canonical string for an id dict, used for fuzzy containment matching."""
    parts = [canonicalize_identity_for_dedup(k, v) for k, v in sorted(ids.items()) if v is not None]
    return " ".join(p for p in parts if p)


def _pages_for_chunks(ledger: ProvenanceLedger, chunk_ids: Sequence[int]) -> list[int]:
    """Sorted page set covering the given chunk ids."""
    pages: set[int] = set()
    for chunk_id in chunk_ids:
        record = ledger.chunks.get(chunk_id)
        if record is not None:
            pages.update(record.page_numbers)
    return sorted(pages)


def compact_view(
    entry: NodeProvenance,
    ledger: ProvenanceLedger,
    max_anchors: int = DEFAULT_MAX_ANCHORS,
    include_spans: bool = False,
) -> dict[str, Any]:
    """Compact node-attribute view of a ledger entry.

    Leads with the exact location when the node's identifier was found verbatim
    (``match: "verbatim"`` → the reported chunks/pages are precise). When no
    verbatim match exists, falls back to the observed set, flagged
    ``approximate: true`` because the observed anchor is only batch-level.
    ``include_spans`` surfaces char spans in the attribute (``detailed`` mode);
    the full anchor list always lives in provenance.json regardless.
    """
    document_id = ledger.document.document_id if ledger.document is not None else ""
    if "scope:document" in entry.notes:
        view: dict[str, Any] = {"document_id": document_id, "scope": "document"}
        if entry.synthetic:
            view["synthetic"] = True
        return view

    verbatim = sorted(
        {(a.chunk_id, a.span) for a in entry.anchors if a.kind == "verbatim" and a.span is not None}
    )
    if verbatim:
        chunk_ids = sorted({chunk_id for chunk_id, _ in verbatim})
        shown = chunk_ids[:max_anchors]
        view = {
            "document_id": document_id,
            "match": "verbatim",
            "chunks": shown,
            "pages": _pages_for_chunks(ledger, shown),
        }
        if len(chunk_ids) > len(shown):
            view["chunks_omitted"] = len(chunk_ids) - len(shown)
        if include_spans:
            spans = [
                {"chunk": chunk_id, "start": span[0], "end": span[1]} for chunk_id, span in verbatim
            ][:_MAX_SPANS]
            if spans:
                view["spans"] = spans
        if entry.synthetic:
            view["synthetic"] = True
        return view

    # No verbatim locator — approximate (batch-level) observed fallback.
    kinds = {a.kind for a in entry.anchors}
    strongest = max(kinds, key=lambda k: KIND_STRENGTH.get(k, 0)) if kinds else "derived"
    chunk_ids = sorted({a.chunk_id for a in entry.anchors})
    shown = chunk_ids[:max_anchors]
    view = {
        "document_id": document_id,
        "match": strongest,
        "chunks": shown,
        "pages": _pages_for_chunks(ledger, shown),
        "approximate": True,
    }
    if len(chunk_ids) > len(shown):
        view["chunks_omitted"] = len(chunk_ids) - len(shown)
    if entry.synthetic:
        view["synthetic"] = True
    return view


def merge_compact_views(
    a: dict[str, Any] | None, b: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Union two compact views (graph-cleaner dedup merge). Never widens claims.

    An ``unresolved`` view yields to any resolved one; ``scope: document``
    absorbs chunk-level detail (the broader claim wins on merge).
    """
    if not a:
        return dict(b) if b else None
    if not b:
        return dict(a)
    if a.get("status") == "unresolved":
        return dict(b)
    if b.get("status") == "unresolved":
        return dict(a)
    if a.get("scope") == "document" or b.get("scope") == "document":
        merged = {"document_id": a.get("document_id") or b.get("document_id"), "scope": "document"}
        if a.get("synthetic") and b.get("synthetic"):
            merged["synthetic"] = True
        return merged
    merged = {
        "document_id": a.get("document_id") or b.get("document_id"),
        "match": max(
            (a.get("match", "derived"), b.get("match", "derived")),
            key=lambda k: KIND_STRENGTH.get(k, 0),
        ),
        "chunks": sorted({*(a.get("chunks") or []), *(b.get("chunks") or [])}),
        "pages": sorted({*(a.get("pages") or []), *(b.get("pages") or [])}),
    }
    # Approximate only when neither side carried a precise (verbatim) location.
    if a.get("approximate") and b.get("approximate"):
        merged["approximate"] = True
    spans = [*(a.get("spans") or []), *(b.get("spans") or [])]
    if spans:
        seen: set[tuple[Any, ...]] = set()
        unique_spans = []
        for s in spans:
            sig = (s.get("chunk"), s.get("start"), s.get("end"))
            if sig not in seen:
                seen.add(sig)
                unique_spans.append(s)
        merged["spans"] = unique_spans[:_MAX_SPANS]
    omitted = (a.get("chunks_omitted") or 0) + (b.get("chunks_omitted") or 0)
    if omitted:
        merged["chunks_omitted"] = omitted
    if a.get("synthetic") and b.get("synthetic"):
        merged["synthetic"] = True
    return merged
