"""Deterministic data grounding & provenance (sidecar ledger).

The ledger is built from pipeline bookkeeping during extraction (never from
LLM output), threaded to the pipeline context, and bound to graph nodes after
conversion via the shared canonical-identity chain.

Note: ``binder`` and ``anchor_scan`` are imported lazily by their callers (not
re-exported here) to keep this package import-light and cycle-free — the dense
orchestrator imports this package at module load.
"""

from .identity import (
    PROVENANCE_NODE_ATTR,
    canonical_id_text,
    compact_view,
    identity_key,
    identity_pairs,
    merge_compact_views,
)
from .models import (
    ChunkRecord,
    DocumentOrigin,
    NodeProvenance,
    ProvenanceLedger,
    SourceAnchor,
    chunk_index_ledger,
    content_hash,
    document_level_ledger,
    text_hash,
)

__all__ = [
    "PROVENANCE_NODE_ATTR",
    "ChunkRecord",
    "DocumentOrigin",
    "NodeProvenance",
    "ProvenanceLedger",
    "SourceAnchor",
    "canonical_id_text",
    "chunk_index_ledger",
    "compact_view",
    "content_hash",
    "document_level_ledger",
    "identity_key",
    "identity_pairs",
    "merge_compact_views",
    "text_hash",
]
