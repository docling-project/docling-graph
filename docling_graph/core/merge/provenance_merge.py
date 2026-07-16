"""Cross-document provenance for graph merging (design §5.5).

Chunk ids are ledger-local integers, so views from different documents are
never blended: they are kept side by side in the wrapped form
``{"multi_document": True, "sources": [...]}`` that ``iter_provenance_views``
and ``merge_compact_views`` already understand. Ledgers themselves are never
fused — each input's ledger is written verbatim as a sidecar plus a manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...logging_utils import get_component_logger
from ..provenance.identity import merge_compact_views
from ..provenance.models import ProvenanceLedger

logger = get_component_logger("ProvenanceMerge", __name__)

# Cap on per-document source views kept inside one node's wrapped view; the
# full anchor detail always survives in the per-document ledger sidecars.
MAX_VIEW_SOURCES = 8


def merge_node_views(
    a: dict[str, Any] | None,
    b: dict[str, Any] | None,
    max_sources: int = MAX_VIEW_SOURCES,
) -> dict[str, Any] | None:
    """Union two node provenance views, wrapping when documents differ.

    Same ``document_id`` (or one side missing it) delegates to
    ``merge_compact_views`` unchanged. Different documents produce (or extend)
    the wrapped multi-document form, keeping each per-document view resolvable
    against its own ledger. Wrapped views are capped at ``max_sources``
    entries (overflow recorded under ``sources_omitted``).
    """
    if not a:
        return dict(b) if b else None
    if not b:
        return dict(a)
    if a.get("status") == "unresolved":
        return dict(b)
    if b.get("status") == "unresolved":
        return dict(a)
    doc_a = str(a.get("document_id") or "")
    doc_b = str(b.get("document_id") or "")
    wrapped = bool(a.get("multi_document") or b.get("multi_document"))
    if not wrapped and (not doc_a or not doc_b or doc_a == doc_b):
        return merge_compact_views(a, b)
    left = a if a.get("multi_document") else {"multi_document": True, "sources": [dict(a)]}
    merged = merge_compact_views(left, b)
    if merged is not None and merged.get("multi_document"):
        # merge_compact_views carries prior overflow accounting itself; only
        # the source cap is applied here.
        sources = merged.get("sources") or []
        if len(sources) > max_sources:
            merged["sources"] = sources[:max_sources]
            merged["sources_omitted"] = int(merged.get("sources_omitted") or 0) + (
                len(sources) - max_sources
            )
    return merged


def write_ledger_sidecars(
    entries: list[tuple[dict[str, Any], ProvenanceLedger | None]],
    provenance_dir: Path,
) -> Path:
    """Write each input's ledger verbatim plus a manifest (design §5.5).

    ``entries`` pairs a manifest record (document_id, source, template info,
    graph path, ...) with the input's ledger (or None). Ledgers land at
    ``<provenance_dir>/<document_id>.json``; inputs without a ledger get a
    manifest record with ``"ledger": null``. Returns the manifest path.
    """
    provenance_dir.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, Any]] = []
    for record, ledger in entries:
        entry = dict(record)
        if ledger is not None:
            name = str(entry.get("document_id") or "") or f"input_{entry.get('index', 0)}"
            ledger_path = provenance_dir / f"{name}.json"
            ledger_path.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")
            entry["ledger"] = ledger_path.name
        else:
            entry["ledger"] = None
        documents.append(entry)
    manifest_path = provenance_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"documents": documents}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info(
        "Wrote %s provenance sidecar(s) and manifest to %s",
        sum(1 for _, ledger in entries if ledger is not None),
        provenance_dir,
    )
    return manifest_path
