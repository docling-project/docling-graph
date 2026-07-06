"""
Dense extraction orchestrator: Phase 1 (skeleton) and Phase 2 (fill).

Fully autonomous: no imports from other contracts.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, cast

from pydantic import BaseModel

from docling_graph.core.provenance import (
    ChunkRecord,
    NodeProvenance,
    ProvenanceLedger,
    SourceAnchor,
    canonical_id_text as _canonical_id_text,
    geometry_from_meta,
    identity_key as _provenance_identity_key,
    identity_pairs,
    text_hash as _chunk_text_hash,
)
from docling_graph.core.utils.entity_name_normalizer import canonicalize_identity_for_dedup
from docling_graph.core.utils.root_identity import is_class_name_echo
from docling_graph.logging_utils import ProgressTracker, batch_tag, get_component_logger

from .catalog import (
    NodeCatalog,
    NodeSpec,
    bottom_up_path_order,
    build_node_catalog,
    build_projected_fill_schema,
    build_skeleton_semantic_guide,
    get_model_for_path,
    path_has_reference_fields,
    skeleton_output_schema,
)
from .models import DenseSkeletonNode
from .prompts import (
    build_skeleton_catalog_block,
    format_batch_markdown,
    get_fill_batch_prompt,
    get_skeleton_batch_prompt,
    get_skeleton_reconciliation_prompt,
    reconciliation_output_schema,
)
from .resolvers import propose_containment_groups, resolve_skeleton_nodes

logger = get_component_logger("DenseExtraction", __name__)

# Max times a truncated skeleton batch may be halved before keeping the partial result.
# Batches are already small (a few chunks), so a depth of 4 fully isolates pathological chunks.
_MAX_SKELETON_SPLIT_DEPTH = 4

# How many already-extracted entities the sequential skeleton prompt advertises
# as negative reference handles. Sliding window: the most recent entities are
# the likeliest cross-batch parents (documents introduce parents shortly before
# their children), and an unbounded list would crowd out the batch text.
_ALREADY_FOUND_WINDOW = 50

# Fraction of the document's tokens that must sit in zero-yield chunks before
# the Phase 1 coverage second pass spends an extra batch round re-examining
# them. Below this, uncovered chunks are treated as legitimately empty
# boilerplate (references, footers) not worth another set of LLM calls.
_COVERAGE_PASS_MIN_TOKEN_SHARE = 0.10

# Id fields whose name promises a numeric/code identifier (document_number, ref_no,
# invoice number...). On sparse documents a small model, lacking a real number,
# often grabs a prominent brand/title string for these — a mis-capture the fill
# phase then locks in via id restoration. Cleared as an invariant so fill can
# leave the field empty instead.
_NUMERIC_ID_FIELD = re.compile(r"(^|_)(number|no|num|ref|reference)(_|$)", re.IGNORECASE)


def strip_mislabeled_root_ids(
    nodes: list[dict[str, Any]],
    template_class_name: str | None = None,
) -> list[dict[str, Any]]:
    """Clear root id values that contradict their field-name semantics.

    A field named like a number that holds multi-word, digit-free prose (e.g.
    ``document_number = "Zylker PC Builds"``) is almost certainly a mis-capture,
    not an identifier. Clearing it in place lets Phase 2 re-derive the value or
    leave it empty rather than propagating a wrong id from the root singleton.
    Conservative by design: only the root (path "") is touched, and only fields
    whose name promises a number are checked, so alphabetic codes and named
    entities elsewhere are never disturbed.

    When ``template_class_name`` is given, ANY root id value that merely echoes
    the template class name (e.g. ``reference_document = "AssuranceMRH"``) is
    also cleared — that is schema echo, never document data, and it would make
    the root un-matchable across runs while looking filled.
    """
    for node in nodes:
        if (node.get("path") or "") != "":
            continue
        ids = node.get("ids")
        if not isinstance(ids, dict):
            continue
        for field_name, value in list(ids.items()):
            if not isinstance(value, str):
                continue
            if template_class_name and is_class_name_echo(value, template_class_name):
                ids.pop(field_name, None)
                continue
            if not _NUMERIC_ID_FIELD.search(field_name):
                continue
            text = value.strip()
            if text and not any(ch.isdigit() for ch in text) and len(text.split()) >= 2:
                ids.pop(field_name, None)
    return nodes


def _reference_handle_prompt(
    entries: list[dict[str, Any]],
) -> tuple[str | None, dict[int, dict[str, Any]] | None]:
    """Build the ALREADY EXTRACTED prompt block + negative handle map.

    Keeps the most recent ``_ALREADY_FOUND_WINDOW`` entries; handle -1 is the
    most recent entity (the likeliest cross-batch parent). Returns (None, None)
    when there is nothing to advertise.
    """
    window = entries[-_ALREADY_FOUND_WINDOW:]
    known_handles = {-(pos + 1): entry for pos, entry in enumerate(reversed(window))}
    if not known_handles:
        return None, None
    already_str = "\n".join(
        json.dumps(
            {"i": handle, "path": entry["path"], "ids": entry["ids"]},
            ensure_ascii=False,
            default=str,
        )
        for handle, entry in known_handles.items()
    )
    return already_str, known_handles


def _skeleton_identity_key(
    node: dict[str, Any],
    spec_by_path: dict[str, NodeSpec],
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Dedup key for a skeleton node; delegates canonicalization to core.provenance.

    Nodes without any usable id keep a process-unique fallback so id-less
    siblings are never merged with each other.
    """
    path = str(node.get("path") or "").strip()
    spec = spec_by_path.get(path)
    ids = node.get("ids") or {}
    if not isinstance(ids, dict):
        ids = {}
    pairs = identity_pairs(ids, spec.id_fields if spec else [])
    return (path, pairs if pairs else (("__key", str(id(node))),))


def _skeleton_ledger_key(
    node: dict[str, Any],
    spec_by_path: dict[str, NodeSpec],
) -> str | None:
    """Serialized ledger identity for a skeleton node; None when unkeyable."""
    path = str(node.get("path") or "").strip()
    spec = spec_by_path.get(path)
    ids = node.get("ids") or {}
    if not isinstance(ids, dict):
        ids = {}
    return _provenance_identity_key(path, ids, spec.id_fields if spec else [])


def chunk_batches_by_token_limit(
    chunks: list[str],
    token_counts: list[int] | None,
    *,
    max_batch_tokens: int,
) -> list[list[tuple[int, str, int]]]:
    if max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be > 0")
    if token_counts is None or len(token_counts) != len(chunks):
        token_counts = [max(1, len(c.split())) for c in chunks]
    batches: list[list[tuple[int, str, int]]] = []
    current: list[tuple[int, str, int]] = []
    current_tokens = 0
    for idx, chunk in enumerate(chunks):
        tcount = token_counts[idx] if idx < len(token_counts) else max(1, len(chunk.split()))
        if current and current_tokens + tcount > max_batch_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append((idx, chunk, tcount))
        current_tokens += tcount
    if current:
        batches.append(current)
    return batches


def _canonical_catalog_path(path: str, allowed_paths: set[str]) -> str | None:
    """Map a model-emitted path onto a catalog path, tolerating missing [] suffixes.

    Small models frequently drop the list markers (e.g. "studies.experiments"
    instead of "studies[].experiments[]"); such drift must not discard the node.
    """
    p = path.strip()
    if p in allowed_paths:
        return p
    stripped = p.replace("[]", "")
    for candidate in allowed_paths:
        if candidate.replace("[]", "") == stripped:
            return candidate
    return None


def normalize_skeleton_batch(
    nodes: list[DenseSkeletonNode],
    allowed_paths: set[str],
    *,
    source_batch_index: int | None = None,
    source_chunk_ids: list[int] | None = None,
    known_handles: dict[int, dict[str, Any]] | None = None,
    stats_out: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Resolve batch-local integer handles into (path, ids) parent references.

    Two passes: first canonicalize paths and index nodes by their handle ``i``;
    then resolve each node's parent handle ``p`` to the referenced node's
    (path, ids). ``known_handles`` maps the negative handles advertised in the
    prompt's ALREADY EXTRACTED list to entities from earlier batches, so a
    child can reference a cross-batch parent without the model re-emitting it
    (handles in the current response always win; the key spaces are disjoint —
    local handles are positive, known handles negative). An explicit parent
    object is accepted as fallback when a model emits one instead of a handle.
    Nodes with unknown paths are dropped. ``stats_out`` (when given) receives a
    ``parents_from_already_found`` count.
    """
    prepared: list[dict[str, Any]] = []
    by_handle: dict[int, dict[str, Any]] = {}
    for node in nodes:
        path = _canonical_catalog_path(node.path or "", allowed_paths)
        if path is None:
            continue
        ids = {str(k): str(v) for k, v in (node.ids or {}).items() if v is not None}
        entry: dict[str, Any] = {"path": path, "ids": ids, "p": node.p, "parent_ref": node.parent}
        prepared.append(entry)
        if node.i is not None and node.i not in by_handle:
            by_handle[node.i] = entry

    out: list[dict[str, Any]] = []
    for entry in prepared:
        parent: dict[str, Any] | None = None
        handle = entry["p"]
        if handle is not None and handle in by_handle and by_handle[handle] is not entry:
            referenced = by_handle[handle]
            parent = {"path": referenced["path"], "ids": dict(referenced["ids"])}
        elif handle is not None and known_handles and handle in known_handles:
            ref = known_handles[handle]
            ref_ids = ref.get("ids") or {}
            parent = {
                "path": str(ref.get("path") or ""),
                "ids": {str(k): str(v) for k, v in ref_ids.items() if v is not None},
            }
            if stats_out is not None:
                stats_out["parents_from_already_found"] = (
                    stats_out.get("parents_from_already_found", 0) + 1
                )
        elif entry["parent_ref"] is not None:
            ref = entry["parent_ref"]
            ref_path = _canonical_catalog_path(ref.path or "", allowed_paths)
            parent = {
                "path": ref_path if ref_path is not None else (ref.path or ""),
                "ids": {str(k): str(v) for k, v in (ref.ids or {}).items() if v is not None},
            }
        result: dict[str, Any] = {"path": entry["path"], "ids": entry["ids"], "parent": parent}
        if source_batch_index is not None:
            result["_source_batch_index"] = source_batch_index
        if source_chunk_ids is not None:
            # Chunk-level provenance: the exact chunks the LLM read when it
            # asserted this node. Truncation-split sub-batches pass a narrower
            # list than the full batch, so granularity improves under splits.
            result["_source_chunk_ids"] = list(source_chunk_ids)
        out.append(result)
    return out


def merge_skeleton_batches(
    batch_results: list[list[dict[str, Any]]],
    catalog: NodeCatalog,
) -> list[dict[str, Any]]:
    """Dedupe skeleton nodes across batches, accumulating every source batch index.

    The union of source *batches* is kept so Phase 2 can scope its fill context
    to the document regions where the node was observed. The observed *chunk*
    set, by contrast, keeps only the FIRST batch that emitted the node — its
    genuine reading. Sequential mode re-emits every already-found node in every
    later batch (the ``already_found`` prompt echo), so accumulating chunks
    across re-emissions would smear a node across the whole document. The
    verbatim scan restores any true multi-location grounding deterministically.

    The root (path "") is a singleton by definition: batches routinely emit it
    with paraphrased identifier values (e.g. title variants), so all root nodes
    are collapsed into the first one instead of trusting id-based dedup.
    """
    spec_by_path = {s.path: s for s in catalog.nodes}
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for batch in batch_results:
        for node in batch:
            key = _skeleton_identity_key(node, spec_by_path)
            source_idx = node.get("_source_batch_index")
            source_chunks = node.get("_source_chunk_ids") or []
            merged = by_key.get(key)
            if merged is None:
                merged = dict(node)
                merged.pop("_source_batch_index", None)
                merged.pop("_source_chunk_ids", None)
                merged["_source_batch_indexes"] = []
                # First-emission chunks only (see docstring); not accumulated.
                merged["_source_chunk_ids"] = [c for c in source_chunks if isinstance(c, int)]
                by_key[key] = merged
            if isinstance(source_idx, int) and source_idx not in merged["_source_batch_indexes"]:
                merged["_source_batch_indexes"].append(source_idx)
    merged_nodes = list(by_key.values())

    roots = [n for n in merged_nodes if (n.get("path") or "") == ""]
    if len(roots) > 1:
        primary = roots[0]
        for extra in roots[1:]:
            for idx in extra.get("_source_batch_indexes", []):
                if idx not in primary["_source_batch_indexes"]:
                    primary["_source_batch_indexes"].append(idx)
        logger.info(
            "Phase 1 (skeleton): collapsed %s duplicate root instances into one", len(roots) - 1
        )
        merged_nodes = [n for n in merged_nodes if (n.get("path") or "") != ""]
        merged_nodes.insert(0, primary)
    return merged_nodes


def skeleton_to_descriptors(
    skeleton_nodes: list[dict[str, Any]],
    catalog: NodeCatalog,
) -> dict[str, list[dict[str, Any]]]:
    path_descriptors: dict[str, list[dict[str, Any]]] = {}
    for node in skeleton_nodes:
        path = node.get("path") or ""
        ids = node.get("ids") or {}
        parent = node.get("parent")
        desc = {"path": path, "ids": dict(ids), "parent": parent}
        source_indexes = node.get("_source_batch_indexes")
        if isinstance(source_indexes, list) and source_indexes:
            desc["_source_batch_indexes"] = list(source_indexes)
        source_chunks = node.get("_source_chunk_ids")
        if isinstance(source_chunks, list) and source_chunks:
            # Kept for locality-based parent adoption during the merge: a child
            # with a dangling parent handle usually belongs to the parent
            # discovered in the same chunk.
            desc["_source_chunk_ids"] = list(source_chunks)
        path_descriptors.setdefault(path, []).append(desc)
    return path_descriptors


def _canonical_lookup_key(path: str, spec: NodeSpec, ids: dict[str, Any]) -> tuple[Any, ...]:
    """Parent-attachment identity key. Mirrors _skeleton_identity_key so that
    dedup and attachment agree on what counts as the same entity.

    Uses the declared id_fields when at least one is present. When NONE are
    present — e.g. a small model emitted the identifier value as the key
    (``{"ESSENTIELLE": "essentielle"}`` instead of ``{"nom": "ESSENTIELLE"}``) —
    it falls back to the node's raw ids. Without the fallback, every sibling
    that lacks its declared id fields collapses onto a single key
    (``(path, (("nom", ""),))``), so the lookup keeps only the last one and every
    child attaches to the wrong sibling while the merge reports a false exact hit.
    """
    ids = ids or {}
    if spec.id_fields:
        ordered = tuple(
            (f, canonicalize_identity_for_dedup(f, ids.get(f)))
            for f in spec.id_fields
            if ids.get(f) is not None
        )
        if ordered:
            return (path, tuple(sorted(ordered, key=lambda x: x[0])))
        fallback = tuple(
            sorted(
                (str(k), canonicalize_identity_for_dedup(k, v))
                for k, v in ids.items()
                if v is not None
            )
        )
        if fallback:
            return (path, fallback)
    return (path, ())


def apply_skeleton_reconciliation(
    skeleton_nodes: list[dict[str, Any]],
    merge_groups: list[Any],
    spec_by_path: dict[str, NodeSpec],
    events_out: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Apply validated alias merge groups to the merged skeleton.

    Group indices refer to the per-path instance order of ``skeleton_nodes``.
    The kept node absorbs the merged nodes' source batches, and parent
    references to a merged node are remapped to the kept node's ids. Invalid
    entries (unknown path, out-of-range or self indices) are skipped silently —
    a bad reconciliation response must never damage the skeleton.

    Co-occurrence veto: two instances of the same path first emitted from the
    SAME chunk are almost never aliases — a document does not name one entity
    twice side by side (two columns of a table header, two bullets of a list
    are distinct entities). Any LLM-confirmed merge whose members share a
    first-emission chunk is rejected; genuine table-label-vs-section-title
    aliases come from different chunks by construction. Each applied merge and
    each veto is logged and appended to ``events_out`` when given.
    """
    by_path: dict[str, list[dict[str, Any]]] = {}
    for node in skeleton_nodes:
        by_path.setdefault(node.get("path") or "", []).append(node)

    removed: set[int] = set()
    id_remap: dict[Any, dict[str, Any]] = {}
    merged_count = 0
    for group in merge_groups:
        if not isinstance(group, dict):
            continue
        path = group.get("path")
        keep_idx = group.get("keep")
        merge_idxs = group.get("merge")
        instances = by_path.get(path) if isinstance(path, str) else None
        if instances is None or not isinstance(merge_idxs, list):
            continue
        if not isinstance(keep_idx, int) or not (0 <= keep_idx < len(instances)):
            continue
        keep_node = instances[keep_idx]
        if id(keep_node) in removed:
            continue
        for merge_idx in merge_idxs:
            if not isinstance(merge_idx, int) or not (0 <= merge_idx < len(instances)):
                continue
            node = instances[merge_idx]
            if node is keep_node or id(node) in removed:
                continue
            keep_chunks = {
                c for c in (keep_node.get("_source_chunk_ids") or []) if isinstance(c, int)
            }
            node_chunks = {c for c in (node.get("_source_chunk_ids") or []) if isinstance(c, int)}
            shared_chunks = keep_chunks & node_chunks
            if shared_chunks:
                logger.info(
                    "Reconciliation VETO (co-occurrence): %s %s ~ %s first emitted from "
                    "the same chunk(s) %s — same-chunk neighbors are distinct entities",
                    path,
                    dict(node.get("ids") or {}),
                    dict(keep_node.get("ids") or {}),
                    sorted(shared_chunks),
                )
                if events_out is not None:
                    events_out.append(
                        {
                            "action": "vetoed_cooccurrence",
                            "path": path,
                            "keep_ids": dict(keep_node.get("ids") or {}),
                            "merge_ids": dict(node.get("ids") or {}),
                            "shared_chunks": sorted(shared_chunks),
                        }
                    )
                continue
            removed.add(id(node))
            merged_count += 1
            logger.info(
                "Reconciliation merge: %s %s absorbed into %s",
                path,
                dict(node.get("ids") or {}),
                dict(keep_node.get("ids") or {}),
            )
            if events_out is not None:
                events_out.append(
                    {
                        "action": "merged",
                        "path": path,
                        "keep_ids": dict(keep_node.get("ids") or {}),
                        "merge_ids": dict(node.get("ids") or {}),
                    }
                )
            id_remap[_skeleton_identity_key(node, spec_by_path)] = dict(keep_node.get("ids") or {})
            keep_sources = keep_node.setdefault("_source_batch_indexes", [])
            for idx in node.get("_source_batch_indexes", []) or []:
                if idx not in keep_sources:
                    keep_sources.append(idx)
            # Provenance lineage: chunks acquired through an LLM alias merge are
            # kept separate from directly-observed ones (anchor kind "reconciled"),
            # and the absorbed identity is recorded so the merge stays auditable.
            keep_reconciled = keep_node.setdefault("_reconciled_chunk_ids", [])
            own_chunks = keep_node.get("_source_chunk_ids") or []
            for chunk_id in node.get("_source_chunk_ids", []) or []:
                if chunk_id not in keep_reconciled and chunk_id not in own_chunks:
                    keep_reconciled.append(chunk_id)
            absorbed_key = _skeleton_ledger_key(node, spec_by_path)
            if absorbed_key is not None:
                merged_from = keep_node.setdefault("_merged_from", [])
                if absorbed_key not in merged_from:
                    merged_from.append(absorbed_key)

    kept_nodes = [n for n in skeleton_nodes if id(n) not in removed]
    if id_remap:
        for node in kept_nodes:
            parent = node.get("parent")
            if not isinstance(parent, dict):
                continue
            parent_key = _skeleton_identity_key(
                {"path": parent.get("path"), "ids": parent.get("ids") or {}}, spec_by_path
            )
            if parent_key in id_remap:
                node["parent"] = {"path": parent.get("path"), "ids": id_remap[parent_key]}
    return kept_nodes, merged_count


def _unique_fuzzy_parent_match(
    parent_path: str,
    parent_ids: dict[str, Any],
    path_filled: dict[str, list[dict[str, Any]]],
    path_descriptors: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Return the single parent instance whose canonical id text contains (or is
    contained by) the referenced ids, or None when the match is absent/ambiguous."""
    ref_text = _canonical_id_text(parent_ids)
    if not ref_text or len(ref_text) < 3:
        return None
    parent_descs = path_descriptors.get(parent_path, [])
    matches: list[dict[str, Any]] = []
    for i, cand in enumerate(path_filled.get(parent_path, [])):
        cand_ids = (parent_descs[i].get("ids") if i < len(parent_descs) else None) or {}
        cand_text = _canonical_id_text(cand_ids)
        if cand_text and (ref_text in cand_text or cand_text in ref_text):
            matches.append(cand)
    return matches[0] if len(matches) == 1 else None


def _unique_local_parent(
    parent_path: str,
    child_desc: dict[str, Any],
    path_filled: dict[str, list[dict[str, Any]]],
    path_descriptors: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Adopt the single parent-path instance observed in the same chunk(s) as the
    child (falling back to batch granularity when chunk ids are absent).

    Small models routinely drop or dangle parent handles; the parent entity
    described next to the child in the source text is almost always the right
    one. Uniqueness keeps this conservative: any ambiguity at chunk level stays
    ambiguous at the coarser batch level, so the first level with matches
    decides.
    """
    parent_descs = path_descriptors.get(parent_path, [])
    filled = path_filled.get(parent_path, [])
    for key in ("_source_chunk_ids", "_source_batch_indexes"):
        child_set = {v for v in (child_desc.get(key) or []) if isinstance(v, int)}
        if not child_set:
            continue
        matches = [
            filled[i]
            for i, parent_desc in enumerate(parent_descs)
            if i < len(filled)
            and isinstance(filled[i], dict)
            and child_set & {v for v in (parent_desc.get(key) or []) if isinstance(v, int)}
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
    return None


def _missing_required_id_fields(
    template: type[BaseModel] | None,
    path: str,
    spec: NodeSpec,
    placeholder: dict[str, Any],
) -> list[str]:
    """Identity fields absent from ``placeholder`` that the model requires.

    Materializing a rescue parent without them produces an instance that
    template validation later deletes wholesale (taking every rescued child
    with it) or that salvage blanks into a phantom hub — both worse than an
    explicit drop here. Returns [] when no template is available (legacy
    permissive behavior for direct callers)."""
    if template is None:
        return []
    model = get_model_for_path(template, path)
    if model is None:
        return []
    missing: list[str] = []
    for field_name in spec.id_fields:
        if placeholder.get(field_name) not in (None, ""):
            continue
        info = model.model_fields.get(field_name)
        if info is not None and info.is_required():
            missing.append(field_name)
    return missing


class _ParentResolver:
    """Resolution ladder for drifted parent references (see merge_filled_into_root).

    Ladder per lookup: exact id match -> unique parent instance -> unique fuzzy
    (canonical containment) id match -> unique co-located parent (same source
    chunk/batch) -> id-only placeholder parent created up the chain -> shared
    id-less bucket parent. Placeholder/bucket materialization is gated on the
    parent model not REQUIRING the missing identity fields.
    """

    def __init__(
        self,
        *,
        root: dict[str, Any],
        lookup: dict[tuple[Any, ...], dict[str, Any]],
        spec_by_path: dict[str, NodeSpec],
        path_filled: dict[str, list[dict[str, Any]]],
        path_descriptors: dict[str, list[dict[str, Any]]],
        rescue_parents: dict[int, str],
        events_out: list[dict[str, Any]] | None,
        template: type[BaseModel] | None,
        attach: Callable[[dict[str, Any], NodeSpec, dict[str, Any]], None],
    ) -> None:
        self._root = root
        self._lookup = lookup
        self._spec_by_path = spec_by_path
        self._path_filled = path_filled
        self._path_descriptors = path_descriptors
        self._rescue_parents = rescue_parents
        self._events_out = events_out
        self._template = template
        self._attach = attach

    def resolve(
        self,
        parent_path: str,
        parent_ids: dict[str, Any],
        depth: int = 0,
        child_desc: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """Return (parent_obj, how) for a child's parent reference; None if unresolvable."""
        parent_spec = self._spec_by_path.get(parent_path)
        if parent_spec is None:
            return None, ""
        if parent_path == "":
            return self._root, "exact"
        key = _canonical_lookup_key(parent_path, parent_spec, parent_ids)
        obj = self._lookup.get(key)
        if obj is not None:
            return obj, "exact"
        instances = [o for o in self._path_filled.get(parent_path, []) if isinstance(o, dict)]
        if len(instances) == 1:
            self._lookup[key] = instances[0]
            return instances[0], "single"
        fuzzy_match = _unique_fuzzy_parent_match(
            parent_path, parent_ids, self._path_filled, self._path_descriptors
        )
        if fuzzy_match is not None:
            self._lookup[key] = fuzzy_match
            return fuzzy_match, "fuzzy"
        if child_desc is not None:
            local_match = _unique_local_parent(
                parent_path, child_desc, self._path_filled, self._path_descriptors
            )
            if local_match is not None:
                # Deliberately NOT registered in `lookup`: an empty parent
                # reference shares its key with every other parentless sibling,
                # and locality is a per-child decision.
                return local_match, "locality"
        # Last resort: materialize a parent so the subtree survives. With usable
        # ids this is an id-only placeholder; without any (dangling handle,
        # wrong-level ids) it degrades to a shared id-less bucket for the path,
        # which is reused by every sibling orphan via the lookup registration.
        # The recursion strictly walks the finite catalog parent chain, so it
        # always terminates at the root; the depth guard is pure paranoia.
        if depth >= 8:
            return None, ""
        placeholder = {
            f: parent_ids[f] for f in parent_spec.id_fields if parent_ids.get(f) not in (None, "")
        }
        if _missing_required_id_fields(self._template, parent_path, parent_spec, placeholder):
            # A rescue parent violating its own required identity would be
            # deleted by template validation (with all rescued children) or
            # blanked by salvage into a phantom hub. Dropping the child here is
            # the honest failure; the caller logs and counts it.
            return None, ""
        grand_obj, _how = self.resolve(parent_spec.parent_path, {}, depth + 1)
        if grand_obj is None:
            return None, ""
        self._attach(grand_obj, parent_spec, placeholder)
        self._lookup[key] = placeholder
        self._rescue_parents[id(placeholder)] = "placeholder" if placeholder else "bucket"
        if self._events_out is not None:
            # Ledger event: a parent was materialized without direct observation.
            self._events_out.append(
                {"event": "synthetic", "path": parent_path, "ids": dict(placeholder)}
            )
        return placeholder, ("placeholder" if placeholder else "bucket")


def merge_filled_into_root(
    path_filled: dict[str, list[dict[str, Any]]],
    path_descriptors: dict[str, list[dict[str, Any]]],
    catalog: NodeCatalog,
    stats_out: dict[str, int] | None = None,
    events_out: list[dict[str, Any]] | None = None,
    template: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Attach filled instances to their parents, rescuing drifted parent references.

    LLMs (especially small ones) drift on parent identifiers, so a strict
    (path, ids) lookup silently drops entire subtrees. Resolution ladder per
    instance: exact id match -> unique parent instance -> unique fuzzy
    (canonical containment) id match -> unique co-located parent (same source
    chunk/batch) -> id-only placeholder parent created up the chain -> shared
    id-less bucket parent. Placeholders and buckets are only materialized when
    the parent model does not REQUIRE the missing identity fields (template
    provided): an identity-less rescue parent would later be deleted by
    template validation (taking its children along) or blanked by salvage into
    a phantom hub. Instances that survive nothing are dropped, and every
    recovery/drop is counted in stats_out.
    """
    root: dict[str, Any] = {}
    spec_by_path = {s.path: s for s in catalog.nodes}
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    # id(obj) -> "placeholder" | "bucket" for rescue parents, so attachments to
    # a bucket are reported honestly instead of hiding behind the lookup hit.
    rescue_parents: dict[int, str] = {}
    stats = {
        "attached_exact": 0,
        "attached_to_bucket": 0,
        "recovered_single_parent": 0,
        "recovered_fuzzy": 0,
        "recovered_locality": 0,
        "recovered_placeholder": 0,
        "recovered_bucket": 0,
        "dropped": 0,
    }
    for spec in catalog.nodes:
        path = spec.path
        filled_list = path_filled.get(path, [])
        descriptors = path_descriptors.get(path, [])
        for i, obj in enumerate(filled_list):
            if isinstance(obj, dict):
                desc = descriptors[i] if i < len(descriptors) else {}
                ids = desc.get("ids") or {}
                key = _canonical_lookup_key(path, spec, ids)
                lookup[key] = obj

    def _attach(parent_obj: dict[str, Any], spec: NodeSpec, obj: dict[str, Any]) -> None:
        if spec.is_list:
            parent_obj.setdefault(spec.field_name, []).append(obj)
        else:
            parent_obj[spec.field_name] = obj

    resolver = _ParentResolver(
        root=root,
        lookup=lookup,
        spec_by_path=spec_by_path,
        path_filled=path_filled,
        path_descriptors=path_descriptors,
        rescue_parents=rescue_parents,
        events_out=events_out,
        template=template,
        attach=_attach,
    )
    _resolve_parent = resolver.resolve

    _how_to_stat = {
        "single": "recovered_single_parent",
        "fuzzy": "recovered_fuzzy",
        "locality": "recovered_locality",
        "placeholder": "recovered_placeholder",
        "bucket": "recovered_bucket",
    }

    def _attach_instance(
        spec: NodeSpec,
        path: str,
        parent_path: str,
        obj: dict[str, Any],
        desc: dict[str, Any],
    ) -> None:
        parent = desc.get("parent")
        parent_ids = (parent.get("ids") or {}) if isinstance(parent, dict) else {}
        parent_obj, how = _resolve_parent(parent_path, parent_ids, child_desc=desc)
        if parent_obj is None:
            stats["dropped"] += 1
            if events_out is not None:
                events_out.append(
                    {"event": "dropped", "path": path, "ids": dict(desc.get("ids") or {})}
                )
            logger.warning(
                "Merge: dropped %s instance ids=%s (unresolvable parent %s ids=%s)",
                path,
                json.dumps(desc.get("ids") or {}, ensure_ascii=False, default=str)[:120],
                parent_path,
                json.dumps(parent_ids, ensure_ascii=False, default=str)[:120],
            )
            return
        _attach(parent_obj, spec, obj)
        stats[_how_to_stat.get(how, "attached_exact")] += 1
        if rescue_parents.get(id(parent_obj)) == "bucket" and how != "bucket":
            # Lookup hits on an existing bucket count as exact above;
            # surface them so a swelling bucket cannot hide.
            stats["attached_to_bucket"] += 1
        if events_out is not None and how in ("placeholder", "bucket"):
            events_out.append(
                {
                    "event": "rescued",
                    "how": how,
                    "path": path,
                    "ids": dict(desc.get("ids") or {}),
                    "parent_path": parent_path,
                    "parent_ids": dict(parent_ids),
                }
            )

    for spec in catalog.nodes:
        path = spec.path
        filled_list = path_filled.get(path, [])
        descriptors = path_descriptors.get(path, [])
        if not filled_list:
            continue
        if path == "":
            if filled_list and isinstance(filled_list[0], dict):
                root.update(filled_list[0])
            continue
        parent_path = spec.parent_path
        field_name = spec.field_name
        is_list = spec.is_list
        if not field_name:
            continue
        if parent_path == "":
            existing = root.get(field_name)
            if is_list and isinstance(existing, list):
                # A placeholder created for a deeper orphan may already live here.
                existing.extend(o for o in filled_list if o not in existing)
            else:
                root[field_name] = (
                    filled_list if is_list else (filled_list[0] if filled_list else None)
                )
            stats["attached_exact"] += len(filled_list)
            continue
        if parent_path not in spec_by_path:
            continue
        for i, obj in enumerate(filled_list):
            desc = descriptors[i] if i < len(descriptors) else {}
            _attach_instance(spec, path, parent_path, obj, desc)

    _log_merge_summary(stats)
    if stats_out is not None:
        stats_out.update(stats)
    return root


def _log_merge_summary(stats: dict[str, int]) -> None:
    """One consolidated warning covering recoveries, buckets, and drops."""
    recovered = (
        stats["recovered_single_parent"]
        + stats["recovered_fuzzy"]
        + stats["recovered_locality"]
        + stats["recovered_placeholder"]
    )
    if recovered or stats["dropped"] or stats["recovered_bucket"]:
        logger.warning(
            "Merge: %s drifted parent link(s) recovered "
            "(%s unique-parent, %s fuzzy, %s locality, %s placeholder), "
            "%s bucket parent(s) with %s total attachment(s), %s instance(s) dropped",
            recovered,
            stats["recovered_single_parent"],
            stats["recovered_fuzzy"],
            stats["recovered_locality"],
            stats["recovered_placeholder"],
            stats["recovered_bucket"],
            stats["recovered_bucket"] + stats["attached_to_bucket"],
            stats["dropped"],
        )


def _compute_branch_paths(catalog: NodeCatalog) -> set[str]:
    """Paths that have at least one other path extending them (container/branch nodes)."""
    all_paths = set(catalog.paths())
    return {p for p in all_paths if any(q != p and q.startswith(p) for q in all_paths)}


def prune_barren_branches(
    root: dict[str, Any],
    catalog: NodeCatalog,
    events_out: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Remove branch nodes that are childless and have no non-identity scalar data.
    Domain-agnostic: uses only catalog path structure (branch vs leaf) and graph topology.
    """
    branch_paths = _compute_branch_paths(catalog)
    spec_by_path = {s.path: s for s in catalog.nodes}
    children_by_parent: dict[str, list[NodeSpec]] = {}
    for s in catalog.nodes:
        children_by_parent.setdefault(s.parent_path, []).append(s)

    def is_barren(node_obj: dict[str, Any], path: str) -> bool:
        spec = spec_by_path.get(path)
        if not spec:
            return False
        id_fields = set(spec.id_fields or [])
        for k, v in node_obj.items():
            if k in id_fields:
                continue
            if v is None:
                continue
            if v == "":
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            if isinstance(v, dict) and not v:
                continue
            return False
        return True

    def has_children(node_obj: dict[str, Any], path: str) -> bool:
        for cs in children_by_parent.get(path, []):
            val = node_obj.get(cs.field_name)
            if cs.is_list and isinstance(val, list) and len(val) > 0:
                return True
            if not cs.is_list and val is not None:
                return True
        return False

    def prune_in_place(obj: dict[str, Any], current_path: str) -> None:
        for cs in children_by_parent.get(current_path, []):
            fn = cs.field_name
            val = obj.get(fn)
            if cs.is_list and isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        prune_in_place(item, cs.path)
                if cs.path in branch_paths:
                    kept = []
                    for x in val:
                        if (
                            isinstance(x, dict)
                            and not has_children(x, cs.path)
                            and is_barren(x, cs.path)
                        ):
                            if events_out is not None:
                                child_spec = spec_by_path.get(cs.path)
                                pruned_ids = {
                                    f: x[f]
                                    for f in (child_spec.id_fields if child_spec else [])
                                    if x.get(f) not in (None, "")
                                }
                                events_out.append(
                                    {"event": "pruned", "path": cs.path, "ids": pruned_ids}
                                )
                        else:
                            kept.append(x)
                    obj[fn] = kept
            elif not cs.is_list and isinstance(val, dict):
                prune_in_place(val, cs.path)

    prune_in_place(root, "")
    return root


def _is_usable_id_value(value: Any) -> bool:
    """A filled identity value is usable when it is a non-empty scalar."""
    if isinstance(value, bool):
        return True
    if isinstance(value, int | float):
        return True
    return isinstance(value, str) and bool(value.strip())


def _sanitize_filled(
    items: list[Any],
    descriptors: list[dict[str, Any]],
    spec: NodeSpec,
    model: type[BaseModel] | None,
) -> list[dict[str, Any]]:
    allowed = set(spec.id_fields)
    if model is not None:
        allowed |= set(model.model_fields.keys())
    out: list[dict[str, Any]] = []
    for i, obj in enumerate(items):
        src = obj if isinstance(obj, dict) else {}
        clean = {k: v for k, v in src.items() if k in allowed}
        desc = descriptors[i] if i < len(descriptors) else {}
        ids = desc.get("ids") or {}
        # Identity values were already captured during the skeleton phase; when
        # the fill response omits one or returns something unusable (null,
        # empty string, a nested object), restore the known value instead of
        # letting downstream salvage synthesize an empty placeholder.
        for f in spec.id_fields:
            if f in ids and not _is_usable_id_value(clean.get(f)):
                clean[f] = ids[f]
        out.append(clean)
    return out


@dataclass
class DenseOrchestratorConfig:
    """Runtime settings for dense extraction.

    Deliberately small: sizing knobs (batch tokens, fill cap, workers), the
    fill-context mode, and one intent-driven dedupe mode. Cleanup steps that
    are mandatory for a sound graph (root singleton, barren-branch pruning,
    the quality gate) are pipeline invariants, not options.
    """

    max_pass_retries: int = 1
    skeleton_batch_tokens: int = 1024
    fill_nodes_cap: int = 5
    parallel_workers: int = 1
    # "scoped": fill prompts only include the skeleton batches where the node was
    # observed (plus the document head); "full": always send the whole document.
    fill_context_mode: str = "scoped"
    # "off": exact canonical-id dedup only.
    # "standard": + one id-space LLM reconciliation call that collapses
    #             same-entity aliases found at different granularities.
    # "aggressive": + fuzzy string merge of near-identical same-path ids
    #               (OCR noise, casing); threshold handled internally.
    dedupe_mode: str = "standard"
    # "off": no provenance ledger. "standard": chunk-level ledger.
    # "detailed": + deterministic verbatim anchor scan (char spans).
    provenance_mode: str = "standard"
    # Serialization the LLM sees: "markdown" (default), "doclang", "doclang-geo".
    # DocLang formats add a one-line orientation to skeleton/fill prompts.
    input_format: str = "markdown"

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> DenseOrchestratorConfig:
        c = config or {}
        raw_skeleton_tokens = int(c.get("dense_skeleton_batch_tokens", 2048) or 2048)
        skeleton_batch_tokens = (
            min(raw_skeleton_tokens, 4096) if raw_skeleton_tokens > 4096 else raw_skeleton_tokens
        )
        fill_context_mode = str(c.get("dense_fill_context", "scoped") or "scoped").lower()
        if fill_context_mode not in ("scoped", "full"):
            fill_context_mode = "scoped"
        dedupe_mode = str(c.get("dense_dedupe", "standard") or "standard").lower()
        if dedupe_mode not in ("off", "standard", "aggressive"):
            dedupe_mode = "standard"
        provenance_mode = str(c.get("provenance", "standard") or "standard").lower()
        if provenance_mode not in ("off", "standard", "detailed"):
            provenance_mode = "standard"
        input_format = str(c.get("llm_input_format", "markdown") or "markdown").lower()
        if input_format not in ("markdown", "doclang", "doclang-geo"):
            input_format = "markdown"
        return cls(
            skeleton_batch_tokens=skeleton_batch_tokens,
            fill_nodes_cap=int(c.get("dense_fill_nodes_cap", 5) or 5),
            parallel_workers=max(1, int(c.get("parallel_workers", 1) or 1)),
            fill_context_mode=fill_context_mode,
            dedupe_mode=dedupe_mode,
            provenance_mode=provenance_mode,
            input_format=input_format,
        )


class DenseOrchestrator:
    def __init__(
        self,
        *,
        llm_call_fn: Callable[..., dict | list | None],
        template: type[BaseModel],
        config: DenseOrchestratorConfig,
        debug_dir: str | None = None,
        debug_suffix: str = "",
        on_trace: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._llm = llm_call_fn
        self._template = template
        self._config = config
        self._debug_dir = debug_dir or ""
        # Per-attempt namespace (e.g. "_attempt2") so a dense retry within the
        # same output dir never overwrites the previous attempt's artifacts.
        self._debug_suffix = debug_suffix
        self._on_trace = on_trace
        self._catalog = build_node_catalog(template)
        # Per-run observability (exposed after run() as last_run_stats).
        self.last_run_stats: dict[str, Any] = {}
        # Per-run provenance ledger (exposed after run(); None when disabled or
        # when the run produced no usable skeleton).
        self.last_provenance: ProvenanceLedger | None = None
        self._counters: dict[str, int] = {}
        self._counter_lock = threading.Lock()
        # Chunk ids whose content produced no skeleton node after all recovery
        # (truncation splits + terminal fallback exhausted). Distinct from a
        # legitimately empty chunk; surfaced in run stats so a lossy run cannot
        # hide behind a merge-only retention figure.
        self._dropped_chunk_ids: list[int] = []
        self._effective_workers = 1
        self._phase1_elapsed = 0.0
        self._phase2_elapsed = 0.0
        # Lazily-built shallow (root + direct children) skeleton artifacts used
        # only as the terminal fallback for a single chunk that keeps truncating.
        self._shallow_built = False
        self._shallow_artifacts: tuple[str, str | None, set[str], str, list[str]] | None = None

    def _bump(self, counter: str, count: int = 1) -> None:
        with self._counter_lock:
            self._counters[counter] = self._counters.get(counter, 0) + count

    def _record_dropped_chunks(self, chunk_ids: list[int]) -> None:
        """Mark chunk ids as producing no skeleton node (unrecoverable truncation)."""
        with self._counter_lock:
            self._counters["failed_batch_count"] = self._counters.get("failed_batch_count", 0) + 1
            for chunk_id in chunk_ids:
                if isinstance(chunk_id, int) and chunk_id not in self._dropped_chunk_ids:
                    self._dropped_chunk_ids.append(chunk_id)

    @staticmethod
    def _covered_chunk_count(merged_skeleton: list[dict[str, Any]]) -> int:
        """Distinct chunk ids that contributed at least one skeleton node."""
        covered: set[int] = set()
        for node in merged_skeleton:
            for chunk_id in node.get("_source_chunk_ids") or []:
                if isinstance(chunk_id, int):
                    covered.add(chunk_id)
        return len(covered)

    def _write_debug(self, name: str, data: Any) -> None:
        if not self._debug_dir:
            return
        os.makedirs(self._debug_dir, exist_ok=True)
        stem, ext = os.path.splitext(name)
        path = os.path.join(self._debug_dir, f"{stem}{self._debug_suffix}{ext}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _call_skeleton_batch(
        self,
        batch_idx: int,
        batch: list[tuple[int, str, int]],
        total_batches: int,
        catalog_block: str,
        allowed_paths: set[str],
        global_context: str | None,
        semantic_guide: str | None,
        schema_json: str,
        context: str,
        already_found_str: str | None,
        *,
        allow_escalation: bool = True,
        prompt_paths: list[str] | None = None,
        known_handles: dict[int, dict[str, Any]] | None = None,
        coverage_retry: bool = False,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Run the LLM for one skeleton batch.

        Returns (normalized_nodes, truncated) where truncated is True if the model
        hit its output limit (so the caller may split the batch and retry).

        ``allow_escalation=False`` tells the backend not to escalate max_tokens
        in-call, so a multi-chunk batch is split (attacking the real cause) before
        any escalation is spent. ``prompt_paths`` narrows the paths advertised to
        the model (the shallow terminal-fallback projection). ``known_handles``
        maps the negative handles listed in ``already_found_str`` back to their
        entities so cross-batch parent references resolve without re-emission.
        """
        batch_md = format_batch_markdown(batch)
        prompt = get_skeleton_batch_prompt(
            batch_markdown=batch_md,
            catalog_block=catalog_block,
            batch_index=batch_idx,
            total_batches=total_batches,
            allowed_paths=prompt_paths if prompt_paths is not None else self._catalog.paths(),
            global_context=global_context,
            already_found=already_found_str,
            semantic_guide=semantic_guide,
            input_format=self._config.input_format,
            coverage_retry=coverage_retry,
        )
        truncated = False
        for attempt in range(self._config.max_pass_retries + 1):
            diag: dict[str, Any] = {}
            out = self._llm(
                prompt=prompt,
                schema_json=schema_json,
                context=f"{context}_dense_skeleton_{batch_idx}",
                response_top_level="object",
                response_schema_name="dense_skeleton",
                allow_truncation_retry=allow_escalation,
                _diagnostics_out=diag,
            )
            if diag.get("truncated"):
                truncated = True
                self._bump("truncation_count")
            if isinstance(out, dict) and isinstance(out.get("nodes"), list):
                raw_nodes = out["nodes"]
                validated_nodes: list[DenseSkeletonNode] = []
                invalid = 0
                # Per-node validation: small models often mix malformed entries
                # (e.g. echoed schema fragments) with valid ones; keep the good
                # nodes instead of discarding the whole batch.
                for element in raw_nodes:
                    try:
                        validated_nodes.append(DenseSkeletonNode.model_validate(element))
                    except Exception:
                        invalid += 1
                if invalid:
                    logger.warning(
                        "%s Skipped %s invalid node entr%s, kept %s",
                        batch_tag(batch_idx, total_batches),
                        invalid,
                        "y" if invalid == 1 else "ies",
                        len(validated_nodes),
                    )
                handle_stats: dict[str, int] = {}
                normalized_batch = normalize_skeleton_batch(
                    validated_nodes,
                    allowed_paths,
                    source_batch_index=batch_idx,
                    source_chunk_ids=[chunk_idx for chunk_idx, _, _ in batch],
                    known_handles=known_handles,
                    stats_out=handle_stats,
                )
                resolved_refs = handle_stats.get("parents_from_already_found", 0)
                if resolved_refs:
                    self._bump("parents_from_already_found", resolved_refs)
                if normalized_batch or not raw_nodes:
                    return normalized_batch, truncated
                # Every entry was invalid: treat as a failed pass and retry.
            if attempt == self._config.max_pass_retries:
                return [], truncated
        return [], truncated

    def _run_one_skeleton_batch(
        self,
        batch_idx: int,
        batch: list[tuple[int, str, int]],
        total_batches: int,
        catalog_block: str,
        allowed_paths: set[str],
        global_context: str | None,
        semantic_guide: str | None,
        schema_json: str,
        context: str,
        spec_by_path: dict[str, NodeSpec],
        already_found_str: str | None,
        known_handles: dict[int, dict[str, Any]] | None = None,
        coverage_retry: bool = False,
        _depth: int = 0,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Run one skeleton batch; returns (batch_idx, normalized_batch_list).

        If the model truncates its output and the batch holds more than one chunk,
        the batch is split in half and each half is retried (domain-agnostic recovery
        for documents too entity-dense to fit one skeleton response in the model's
        output budget). Sub-batches keep the same batch_idx so fill-context provenance
        still points at the original batch text. Uses already_found_str=None in
        parallel mode.

        Escalation order (split before escalate): a multi-chunk batch is split
        *before* max_tokens is escalated — splitting attacks the real cause,
        while escalating first just spends tokens against a repetition loop.
        Escalation is only allowed once a single chunk cannot be split further;
        a single chunk still unrecoverable falls to a shallow ids-only projection,
        and only if that also fails is the chunk recorded as dropped (never
        silently lost — surfaced in run stats for V3).
        """
        allow_escalation = len(batch) == 1
        nodes, truncated = self._call_skeleton_batch(
            batch_idx,
            batch,
            total_batches,
            catalog_block,
            allowed_paths,
            global_context,
            semantic_guide,
            schema_json,
            context,
            already_found_str,
            allow_escalation=allow_escalation,
            known_handles=known_handles,
            coverage_retry=coverage_retry,
        )
        if truncated and len(batch) > 1 and _depth < _MAX_SKELETON_SPLIT_DEPTH:
            self._bump("split_count")
            mid = len(batch) // 2
            logger.warning(
                "%s Truncated (%s chunks) -> retrying with split (%s + %s chunks)",
                batch_tag(batch_idx, total_batches),
                len(batch),
                mid,
                len(batch) - mid,
            )
            merged: list[dict[str, Any]] = []
            for sub in (batch[:mid], batch[mid:]):
                _, sub_nodes = self._run_one_skeleton_batch(
                    batch_idx,
                    sub,
                    total_batches,
                    catalog_block,
                    allowed_paths,
                    global_context,
                    semantic_guide,
                    schema_json,
                    context,
                    spec_by_path,
                    already_found_str,
                    known_handles,
                    coverage_retry,
                    _depth + 1,
                )
                merged.extend(sub_nodes)
            return (batch_idx, merged)
        if truncated and not nodes:
            # Cannot split further (single chunk, or split depth exhausted) and
            # the model produced nothing usable. Try a minimal root + direct
            # children projection before conceding the content.
            fallback = self._shallow_skeleton_retry(batch_idx, batch, context)
            if fallback:
                logger.warning(
                    "%s Recovered %s node(s) via shallow fallback after truncation",
                    batch_tag(batch_idx, total_batches),
                    len(fallback),
                )
                return (batch_idx, fallback)
            self._record_dropped_chunks([cid for cid, _, _ in batch])
            logger.warning(
                "%s %s chunk(s) unrecoverable after truncation; "
                "content may be missing from the graph",
                batch_tag(batch_idx, total_batches),
                len(batch),
            )
            return (batch_idx, [])
        return (batch_idx, nodes)

    def _shallow_skeleton_artifacts(
        self,
    ) -> tuple[str, str | None, set[str], str, list[str]] | None:
        """Build (and cache) the root + direct-children skeleton projection.

        Returns None when the catalog has no root spec (nothing to fall back to).
        """
        if self._shallow_built:
            return self._shallow_artifacts
        self._shallow_built = True
        shallow_specs = [s for s in self._catalog.nodes if s.path == "" or s.parent_path == ""]
        if not any(s.path == "" for s in shallow_specs):
            self._shallow_artifacts = None
            return None
        shallow_catalog = NodeCatalog(
            nodes=shallow_specs, field_aliases=dict(self._catalog.field_aliases)
        )
        paths = [s.path for s in shallow_specs]
        self._shallow_artifacts = (
            build_skeleton_catalog_block(shallow_catalog),
            build_skeleton_semantic_guide(shallow_catalog),
            set(paths),
            json.dumps(skeleton_output_schema(paths)),
            paths,
        )
        return self._shallow_artifacts

    def _shallow_skeleton_retry(
        self, batch_idx: int, batch: list[tuple[int, str, int]], context: str
    ) -> list[dict[str, Any]]:
        """Terminal fallback: re-run a single truncating chunk with only the root
        and its direct entity children, so a degraded but chunk-grounded node
        survives instead of the content being silently dropped."""
        artifacts = self._shallow_skeleton_artifacts()
        if artifacts is None:
            return []
        catalog_block, semantic_guide, allowed_paths, schema_json, prompt_paths = artifacts
        nodes, _ = self._call_skeleton_batch(
            batch_idx,
            batch,
            1,
            catalog_block,
            allowed_paths,
            None,
            semantic_guide,
            schema_json,
            context,
            None,
            allow_escalation=True,
            prompt_paths=prompt_paths,
        )
        return nodes

    def _run_skeleton_reconciliation(
        self,
        merged_skeleton: list[dict[str, Any]],
        spec_by_path: dict[str, NodeSpec],
        context: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """One id-space LLM call that collapses same-entity aliases across batches.

        Parallel batches (and granularity drift within a document) produce the
        same entity at several specificity levels (e.g. "LFP slurry batch"
        alongside "LFP_20vol_5wtPVDF_4wtCB"). Pure string similarity cannot
        judge granularity, so a single cheap call over the identifier lists
        (no document content) decides alias groups; anything invalid in the
        response is ignored, so this pass can only merge, never lose nodes.

        Deterministic containment matches are passed along as CANDIDATES the
        model must confirm — never auto-applied, because containment cannot
        tell a same-entity refinement from a distinct tier ("CONFORT PLUS"
        must survive next to "CONFORT").
        """
        instances_by_path: dict[str, list[dict[str, Any]]] = {}
        for node in merged_skeleton:
            path = node.get("path") or ""
            if path == "":
                continue  # the root is already collapsed to a singleton
            instances_by_path.setdefault(path, []).append(node.get("ids") or {})
        instances_by_path = {
            path: ids_list for path, ids_list in instances_by_path.items() if len(ids_list) >= 2
        }
        if not instances_by_path:
            return merged_skeleton, 0
        candidates = propose_containment_groups(instances_by_path)
        if candidates and self._on_trace:
            self._on_trace({"contract": "dense", "phase1_containment_proposals": len(candidates)})
        prompt = get_skeleton_reconciliation_prompt(instances_by_path, candidate_groups=candidates)
        out = self._llm(
            prompt=prompt,
            schema_json=json.dumps(reconciliation_output_schema()),
            context=f"{context}_dense_reconcile",
            response_top_level="object",
            response_schema_name="dense_reconcile",
        )
        merges = out.get("merges") if isinstance(out, dict) else None
        if not isinstance(merges, list) or not merges:
            return merged_skeleton, 0
        events: list[dict[str, Any]] = []
        reconciled, merged_count = apply_skeleton_reconciliation(
            merged_skeleton, merges, spec_by_path, events_out=events
        )
        # Forensics artifact: without it, merge decisions only survive as
        # _merged_from breadcrumbs inside the skeleton dump.
        self._write_debug(
            "dense_reconciliation.json",
            {
                "instances_by_path": instances_by_path,
                "containment_candidates": candidates,
                "llm_merges": merges,
                "events": events,
            },
        )
        if merged_count:
            logger.info(
                "Reconciliation: merged %s alias instance(s) into more specific ones",
                merged_count,
            )
        return reconciled, merged_count

    def _dedupe_skeleton(
        self,
        merged_skeleton: list[dict[str, Any]],
        spec_by_path: dict[str, NodeSpec],
        context: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """Collapse duplicate skeleton instances (fuzzy -> LLM-confirmed).

        Runs in standard+ (never in "off"). The aggressive-only fuzzy string
        merge handles OCR noise deterministically; containment matches
        (a superset id like "nanoscaled LiFePO4" alongside its base "LiFePO4")
        are only PROPOSED to the reconciliation LLM call, which confirms or
        rejects them — auto-applying containment destroyed tiered names
        ("CONFORT PLUS" was merged into "CONFORT").
        """

        def key_fn(n: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
            return _skeleton_identity_key(n, spec_by_path)

        if self._config.dedupe_mode == "aggressive":
            merged_skeleton, resolver_stats = resolve_skeleton_nodes(merged_skeleton, key_fn)
            if self._on_trace and resolver_stats.get("merged_count", 0) > 0:
                self._on_trace({"contract": "dense", "phase1_resolvers": resolver_stats})
        reconciliation_merged = 0
        if self._config.dedupe_mode != "off" and len(merged_skeleton) > 1:
            merged_skeleton, reconciliation_merged = self._run_skeleton_reconciliation(
                merged_skeleton, spec_by_path, context
            )
        return merged_skeleton, reconciliation_merged

    def _finalize_run_stats(
        self,
        skeleton_nodes: int,
        reconciliation_merged: int,
        merge_stats: dict[str, int],
        *,
        total_chunks: int = 0,
        covered_chunks: int = 0,
        gate_failure: str | None = None,
    ) -> None:
        """Publish per-run observability counters as last_run_stats.

        ``retention_pct`` is deliberately narrow — it is merge-stage retention
        (skeleton nodes that survived the fill->graph merge), NOT source
        coverage. Batches lost to truncation *before* merge never enter
        ``skeleton_nodes``, so they are reported separately as
        ``skeleton_batches_failed`` / ``dropped_chunk_ids`` and folded into the
        honest ``chunk_coverage_pct`` (chunks that produced at least one node).
        """
        dropped = merge_stats.get("dropped", 0)
        dropped_chunk_ids = sorted({c for c in self._dropped_chunk_ids if isinstance(c, int)})
        stats: dict[str, Any] = {
            "skeleton_nodes": skeleton_nodes,
            "parallel_workers": self._effective_workers,
            "phase1_seconds": round(self._phase1_elapsed, 2),
            "phase2_seconds": round(self._phase2_elapsed, 2),
            "truncation_count": self._counters.get("truncation_count", 0),
            "split_count": self._counters.get("split_count", 0),
            "skeleton_batches_failed": self._counters.get("failed_batch_count", 0),
            "parents_from_already_found": self._counters.get("parents_from_already_found", 0),
            "coverage_pass_recovered": self._counters.get("coverage_pass_recovered", 0),
            "dropped_chunk_ids": dropped_chunk_ids,
            "chunk_coverage_pct": (
                round(100.0 * covered_chunks / total_chunks, 1) if total_chunks else 0.0
            ),
            "reconciliation_merged": reconciliation_merged,
            "merge_orphans_dropped": dropped,
            "merge_recovered": (
                merge_stats.get("recovered_single_parent", 0)
                + merge_stats.get("recovered_fuzzy", 0)
                + merge_stats.get("recovered_locality", 0)
                + merge_stats.get("recovered_placeholder", 0)
            ),
            "merge_bucket_parents": merge_stats.get("recovered_bucket", 0),
            "merge_bucket_attachments": (
                merge_stats.get("recovered_bucket", 0) + merge_stats.get("attached_to_bucket", 0)
            ),
            "retention_pct": (
                round(100.0 * (1 - dropped / skeleton_nodes), 1) if skeleton_nodes else 0.0
            ),
        }
        if gate_failure:
            stats["quality_gate_failure"] = gate_failure
        self.last_run_stats = stats

    def _build_chunk_records(
        self,
        batches: list[list[tuple[int, str, int]]],
        chunk_metadata: list[dict[str, Any]] | None,
        total_chunks: int,
    ) -> dict[int, ChunkRecord]:
        """Chunk index for the provenance ledger (H2).

        The batch tuples are authoritative for chunk_id -> (text, batch) since
        they are exactly what Phase 1 sends to the LLM; per-chunk location
        metadata is joined in when the metadata list aligns with the chunk list.
        """
        aligned = bool(chunk_metadata) and len(chunk_metadata or []) == total_chunks
        records: dict[int, ChunkRecord] = {}
        for batch_index, batch in enumerate(batches):
            for chunk_id, chunk_text, token_count in batch:
                meta = (chunk_metadata[chunk_id] if aligned else {}) or {}  # type: ignore[index]
                records[chunk_id] = ChunkRecord(
                    chunk_id=chunk_id,
                    batch_index=batch_index,
                    page_numbers=tuple(
                        p for p in (meta.get("page_numbers") or []) if isinstance(p, int)
                    ),
                    doc_item_refs=tuple(str(r) for r in (meta.get("doc_item_refs") or [])),
                    item_geometry=geometry_from_meta(meta),
                    headings=tuple(str(h) for h in (meta.get("headings") or [])),
                    token_count=token_count,
                    text_hash=_chunk_text_hash(chunk_text),
                    char_length=len(chunk_text),
                    text=chunk_text,
                    resplit_of=meta.get("resplit_of")
                    if isinstance(meta.get("resplit_of"), int)
                    else None,
                )
        return records

    def _freeze_ledger(
        self,
        merged_skeleton: list[dict[str, Any]],
        chunk_records: dict[int, ChunkRecord],
        spec_by_path: dict[str, NodeSpec],
    ) -> ProvenanceLedger:
        """Ledger freeze (H5): one NodeProvenance per surviving skeleton node.

        Runs after reconciliation and root-id stripping, before Phase 2 — fill
        cannot change identity (_sanitize_filled restores id values), so keys
        frozen here remain valid through the rest of the run. Coordinator
        thread only.
        """
        nodes: dict[str, NodeProvenance] = {}
        unkeyed_ordinal = 0
        for node in merged_skeleton:
            path = str(node.get("path") or "")
            spec = spec_by_path.get(path)
            ids = node.get("ids") or {}
            if not isinstance(ids, dict):
                ids = {}
            key = _skeleton_ledger_key(node, spec_by_path)
            notes: list[str] = []
            if key is None:
                # Id-less nodes get a positional key that by design never binds
                # exactly — fail-empty, never fail-wrong.
                key = f"{path}#unkeyed{unkeyed_ordinal}"
                unkeyed_ordinal += 1
                notes.append("identity:unkeyed")
            entry = nodes.get(key)
            if entry is None:
                entry = NodeProvenance(
                    identity_key=key,
                    catalog_path=path,
                    node_type=spec.node_type if spec else "",
                    ids={str(k): str(v) for k, v in ids.items() if v is not None},
                    notes=notes,
                )
                nodes[key] = entry
            observed = [c for c in (node.get("_source_chunk_ids") or []) if isinstance(c, int)]
            if not observed:
                # Defensive batch-level fallback when chunk ids are missing.
                batch_indexes = {
                    b for b in (node.get("_source_batch_indexes") or []) if isinstance(b, int)
                }
                observed = [
                    c for c, rec in chunk_records.items() if rec.batch_index in batch_indexes
                ]
            reconciled = [
                c for c in (node.get("_reconciled_chunk_ids") or []) if isinstance(c, int)
            ]
            existing = {(a.chunk_id, a.kind) for a in entry.anchors}
            for chunk_id in sorted(set(observed)):
                if chunk_id in chunk_records and (chunk_id, "observed") not in existing:
                    entry.anchors.append(SourceAnchor(chunk_id=chunk_id, kind="observed"))
            for chunk_id in sorted(set(reconciled)):
                if (
                    chunk_id in chunk_records
                    and (chunk_id, "observed") not in existing
                    and (chunk_id, "reconciled") not in existing
                    and not any(
                        a.chunk_id == chunk_id and a.kind == "observed" for a in entry.anchors
                    )
                ):
                    entry.anchors.append(SourceAnchor(chunk_id=chunk_id, kind="reconciled"))
            for absorbed in node.get("_merged_from") or []:
                if absorbed not in entry.merged_from:
                    entry.merged_from.append(absorbed)
            if path == "" and "scope:document" not in entry.notes:
                # The root is always filled against the full document.
                entry.notes.append("scope:document")
        return ProvenanceLedger(node_level=True, chunks=chunk_records, nodes=nodes)

    def _apply_merge_events(
        self,
        ledger: ProvenanceLedger,
        events: list[dict[str, Any]],
        spec_by_path: dict[str, NodeSpec],
    ) -> None:
        """Fold merge_filled_into_root / prune events into the ledger (H6)."""

        def _key_for(path: str, ids: dict[str, Any]) -> str | None:
            spec = spec_by_path.get(path)
            return _provenance_identity_key(path, ids, spec.id_fields if spec else [])

        for event in events:
            kind = event.get("event")
            path = str(event.get("path") or "")
            ids = event.get("ids") or {}
            if kind in ("dropped", "pruned"):
                key = _key_for(path, ids)
                entry = ledger.nodes.get(key) if key else None
                if entry is not None:
                    entry.dropped = True
                    entry.notes.append(f"merge:{kind}")
            elif kind == "synthetic":
                key = _key_for(path, ids) or f"{path}#bucket"
                entry = ledger.nodes.get(key)
                if entry is None:
                    spec = spec_by_path.get(path)
                    ledger.nodes[key] = NodeProvenance(
                        identity_key=key,
                        catalog_path=path,
                        node_type=spec.node_type if spec else "",
                        ids={str(k): str(v) for k, v in ids.items() if v is not None},
                        synthetic=True,
                        notes=["merge:placeholder" if ids else "merge:bucket"],
                    )
                elif not entry.anchors:
                    entry.notes.append("merge:placeholder-reused")
            elif kind == "rescued":
                # Synthetic parents inherit their rescued children's chunks as
                # weaker "derived" evidence.
                parent_path = str(event.get("parent_path") or "")
                parent_spec = spec_by_path.get(parent_path)
                parent_ids = event.get("parent_ids") or {}
                placeholder_ids = {
                    f: parent_ids[f]
                    for f in (parent_spec.id_fields if parent_spec else [])
                    if parent_ids.get(f) not in (None, "")
                }
                parent_key = _key_for(parent_path, placeholder_ids) or f"{parent_path}#bucket"
                parent_entry = ledger.nodes.get(parent_key)
                child_key = _key_for(path, ids)
                child_entry = ledger.nodes.get(child_key) if child_key else None
                if parent_entry is not None and parent_entry.synthetic and child_entry is not None:
                    existing = {(a.chunk_id, a.kind) for a in parent_entry.anchors}
                    for anchor in child_entry.anchors:
                        if (anchor.chunk_id, "derived") not in existing and anchor.kind in (
                            "observed",
                            "verbatim",
                        ):
                            parent_entry.anchors.append(
                                SourceAnchor(chunk_id=anchor.chunk_id, kind="derived")
                            )
                            existing.add((anchor.chunk_id, "derived"))

    def _finalize_provenance(
        self,
        ledger: ProvenanceLedger,
        merge_events: list[dict[str, Any]],
        spec_by_path: dict[str, NodeSpec],
        path_descriptors: dict[str, list[dict[str, Any]]],
        path_filled: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Fold merge events into the ledger, re-key to final ids, and publish it.

        The verbatim locator is NOT run here: the skeleton identifiers are often
        generic placeholders that the fill phase refines to the real value
        (e.g. "SlurryComponent" -> "LiFePO4"). Only the real value appears in
        the document, so the binder runs the locator with the final model ids.
        This ledger carries the observed (skeleton) anchors and the full chunk
        text; the binder adds the exact verbatim anchors.
        """
        self._apply_merge_events(ledger, merge_events, spec_by_path)
        self._rekey_ledger_to_filled(ledger, path_descriptors, path_filled, spec_by_path)
        self.last_provenance = ledger

    def _rekey_ledger_to_filled(
        self,
        ledger: ProvenanceLedger,
        path_descriptors: dict[str, list[dict[str, Any]]],
        path_filled: dict[str, list[dict[str, Any]]],
        spec_by_path: dict[str, NodeSpec],
    ) -> None:
        """Re-key ledger entries from skeleton ids to the FINAL filled ids.

        Phase 2 often replaces a rough skeleton identifier with the real one
        (skeleton "Battery Slurry Batch" -> fill "BATCH-LFP-001"). The graph
        node carries the filled id, so an entry still keyed on the skeleton id
        would never bind. Descriptors and filled objects are index-aligned per
        path (fill pads/truncates each batch to its descriptor count), so the
        correspondence is exact. When two skeleton entries collapse onto the
        same final id, their anchors/lineage merge.
        """
        for path, descriptors in path_descriptors.items():
            spec = spec_by_path.get(path)
            if spec is None or not spec.id_fields:
                continue
            filled = path_filled.get(path, [])
            for i, desc in enumerate(descriptors):
                if i >= len(filled) or not isinstance(filled[i], dict):
                    continue
                skel_ids = desc.get("ids") or {}
                skel_key = _provenance_identity_key(path, skel_ids, spec.id_fields)
                if skel_key is None or skel_key not in ledger.nodes:
                    continue
                obj = filled[i]
                final_ids = {f: obj[f] for f in spec.id_fields if obj.get(f) not in (None, "")}
                final_key = _provenance_identity_key(path, final_ids, spec.id_fields)
                if final_key is None or final_key == skel_key:
                    continue
                entry = ledger.nodes.pop(skel_key)
                target = ledger.nodes.get(final_key)
                if target is not None:
                    seen = {(a.chunk_id, a.kind, a.span) for a in target.anchors}
                    for anchor in entry.anchors:
                        if (anchor.chunk_id, anchor.kind, anchor.span) not in seen:
                            target.anchors.append(anchor)
                    for src in entry.merged_from:
                        if src not in target.merged_from:
                            target.merged_from.append(src)
                else:
                    entry.identity_key = final_key
                    entry.ids = {str(k): str(v) for k, v in final_ids.items()}
                    ledger.nodes[final_key] = entry

    def _build_fill_context(
        self,
        path: str,
        batch_descriptors: list[dict[str, Any]],
        batch_texts: list[str],
        full_markdown: str,
        global_head: str,
    ) -> str:
        """Markdown context for one fill batch.

        In "scoped" mode, only the skeleton batches where the instances were
        observed are included (plus the document head for shared context),
        which keeps Phase 2 token cost proportional to the entities being
        filled instead of resending the whole document for every batch.
        The root instance and nodes without provenance always get the full
        document, as does any scoped context that would not actually shrink it.
        """
        if self._config.fill_context_mode != "scoped" or path == "":
            return full_markdown
        indexes: set[int] = set()
        for desc in batch_descriptors:
            for idx in desc.get("_source_batch_indexes") or []:
                if isinstance(idx, int) and 0 <= idx < len(batch_texts):
                    indexes.add(idx)
        if not indexes:
            return full_markdown
        parts: list[str] = []
        if global_head and 0 not in indexes:
            parts.append(global_head)
        parts.extend(batch_texts[i] for i in sorted(indexes))
        scoped = "\n\n".join(parts)
        if len(scoped) >= len(full_markdown):
            return full_markdown
        return scoped

    def _run_one_fill_batch(
        self,
        path: str,
        spec: NodeSpec,
        batch_descriptors: list[dict[str, Any]],
        batch_index: int,
        sub_schema: str,
        fill_markdown: str,
        context: str,
    ) -> tuple[str, int, list[dict[str, Any]]]:
        """Run one fill batch; returns (path, batch_index, sanitized_list)."""
        prompt = get_fill_batch_prompt(
            markdown=fill_markdown,
            path=path,
            spec=spec,
            descriptors=batch_descriptors,
            projected_schema_json=sub_schema,
            input_format=self._config.input_format,
            has_reference_fields=path_has_reference_fields(self._template, spec),
        )
        # Hoist $defs to the wrapper root: the sub-schema's $ref pointers are
        # root-relative (#/$defs/...), so nesting it unchanged would leave them
        # dangling — invalid JSON schema that grammar-validating providers reject.
        sub_schema_dict = json.loads(sub_schema)
        hoisted_defs = sub_schema_dict.pop("$defs", None)
        wrapped_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": sub_schema_dict}},
            "required": ["items"],
        }
        if hoisted_defs:
            wrapped_schema["$defs"] = hoisted_defs
        fill_diag: dict[str, Any] = {}
        out = self._llm(
            prompt=prompt,
            schema_json=json.dumps(wrapped_schema),
            context=f"{context}_dense_fill_{path}",
            response_top_level="object",
            response_schema_name="dense_fill",
            _diagnostics_out=fill_diag,
        )
        if fill_diag.get("truncated"):
            self._bump("truncation_count")
        if isinstance(out, dict) and "items" in out:
            items = out["items"] if isinstance(out["items"], list) else []
        elif isinstance(out, list):
            items = out
        else:
            items = []
        # Exactly one filled object per requested instance: pad short responses
        # with empty objects (ids are restored from descriptors during sanitize)
        # so skeleton nodes are never silently dropped, and discard extras that
        # have no matching descriptor (they carry no usable parent linkage).
        if len(items) < len(batch_descriptors):
            items = [*items, *([{}] * (len(batch_descriptors) - len(items)))]
        elif len(items) > len(batch_descriptors):
            items = items[: len(batch_descriptors)]
        model = get_model_for_path(self._template, path)
        sanitized = _sanitize_filled(items, batch_descriptors, spec, model)
        return (path, batch_index, sanitized)

    def _run_skeleton_phase(
        self,
        *,
        batches: list[list[tuple[int, str, int]]],
        workers: int,
        catalog_block: str,
        allowed_paths: set[str],
        global_context: str | None,
        semantic_guide: str | None,
        schema_json: str,
        context: str,
        spec_by_path: dict[str, NodeSpec],
    ) -> list[list[dict[str, Any]]]:
        """Execute all Phase 1 skeleton batches, sequentially or in parallel."""
        total_batches = len(batches)
        progress = ProgressTracker(
            logger, total=total_batches, label="Phase 1 (skeleton)", unit="batch"
        )
        skeleton_results: list[list[dict[str, Any]]]
        if workers <= 1 or total_batches <= 1:
            # Sequential: advertise already-extracted entities as stable
            # NEGATIVE reference handles so later batches can attach children
            # to cross-batch parents without re-emitting them (the re-emit
            # instruction alone was observed to be ignored — children arrived
            # with p=null and were dropped at merge).
            already_entries: list[dict[str, Any]] = []
            seen_entry_keys: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
            skeleton_results = []
            for batch_idx, batch in enumerate(batches):
                already_str, known_handles = _reference_handle_prompt(already_entries)
                _, normalized_batch = self._run_one_skeleton_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    total_batches=total_batches,
                    catalog_block=catalog_block,
                    allowed_paths=allowed_paths,
                    global_context=global_context,
                    semantic_guide=semantic_guide,
                    schema_json=schema_json,
                    context=context,
                    spec_by_path=spec_by_path,
                    already_found_str=already_str,
                    known_handles=known_handles or None,
                )
                for node in normalized_batch:
                    ids = node.get("ids") or {}
                    if not ids:
                        # Id-less nodes cannot be referenced by a handle and
                        # only add noise to the already-extracted list.
                        continue
                    key = _skeleton_identity_key(node, spec_by_path)
                    if key in seen_entry_keys:
                        continue
                    seen_entry_keys.add(key)
                    already_entries.append({"path": node["path"], "ids": dict(ids)})
                skeleton_results.append(normalized_batch)
                progress.advance(note=f"+{len(normalized_batch)} node(s)")
            progress.finish()
            return skeleton_results

        # Parallel: no already_found; merge_skeleton_batches and resolvers dedupe
        logger.info(
            "Phase 1 (skeleton): running %s batches with %s workers",
            total_batches,
            workers,
        )
        skeleton_results = [None] * total_batches  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    self._run_one_skeleton_batch,
                    batch_idx=batch_idx,
                    batch=batch,
                    total_batches=total_batches,
                    catalog_block=catalog_block,
                    allowed_paths=allowed_paths,
                    global_context=global_context,
                    semantic_guide=semantic_guide,
                    schema_json=schema_json,
                    context=context,
                    spec_by_path=spec_by_path,
                    already_found_str=None,
                ): batch_idx
                for batch_idx, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    idx, normalized_batch = future.result()
                    skeleton_results[idx] = normalized_batch
                    progress.advance(note=f"+{len(normalized_batch)} node(s)")
                except Exception as e:
                    logger.warning("%s Failed: %s", batch_tag(batch_idx, total_batches), e)
                    skeleton_results[batch_idx] = []
                    progress.advance(note="failed")
        progress.finish()
        return [lst if lst is not None else [] for lst in skeleton_results]

    def _run_coverage_pass(
        self,
        *,
        batches: list[list[tuple[int, str, int]]],
        merged_skeleton: list[dict[str, Any]],
        catalog_block: str,
        allowed_paths: set[str],
        global_context: str | None,
        semantic_guide: str | None,
        schema_json: str,
        context: str,
        spec_by_path: dict[str, NodeSpec],
    ) -> tuple[list[list[tuple[int, str, int]]], list[list[dict[str, Any]]]]:
        """Re-examine chunks that produced no skeleton node on the first pass.

        Chunk coverage in the field sits at 74-91%: some of that is legitimately
        empty boilerplate, some is recall silently left on the table (a chunk
        sharing a batch with entity-dense neighbors gets outshone). One bounded
        extra round — only when the zero-yield chunks hold at least
        ``_COVERAGE_PASS_MIN_TOKEN_SHARE`` of the document's tokens — re-runs
        just those chunks with the full already-found reference-handle list, so
        recovered children attach to existing parents. The prompt explicitly
        licenses an empty response (never pressure-invents instances).

        Returns (coverage_batches, coverage_results) to append to the phase 1
        outputs; both empty when the pass is not warranted.
        """
        if not merged_skeleton:
            return [], []  # nothing to attach to; the quality gate handles this run
        covered: set[int] = set()
        for node in merged_skeleton:
            for chunk_id in node.get("_source_chunk_ids") or []:
                if isinstance(chunk_id, int):
                    covered.add(chunk_id)
        dropped = set(self._dropped_chunk_ids)
        uncovered: list[tuple[int, str, int]] = []
        total_tokens = 0
        for batch in batches:
            for chunk_id, text, tcount in batch:
                total_tokens += tcount
                if chunk_id not in covered and chunk_id not in dropped:
                    uncovered.append((chunk_id, text, tcount))
        uncovered_tokens = sum(t for _, _, t in uncovered)
        if (
            not uncovered
            or total_tokens <= 0
            or uncovered_tokens < _COVERAGE_PASS_MIN_TOKEN_SHARE * total_tokens
        ):
            return [], []

        # Pack uncovered chunks under the skeleton budget, keeping original ids.
        coverage_batches: list[list[tuple[int, str, int]]] = []
        current: list[tuple[int, str, int]] = []
        current_tokens = 0
        for item in uncovered:
            if current and current_tokens + item[2] > self._config.skeleton_batch_tokens:
                coverage_batches.append(current)
                current = []
                current_tokens = 0
            current.append(item)
            current_tokens += item[2]
        if current:
            coverage_batches.append(current)

        already_entries = [
            {"path": node.get("path") or "", "ids": dict(node.get("ids") or {})}
            for node in merged_skeleton
            if node.get("ids")
        ]
        already_str, known_handles = _reference_handle_prompt(already_entries)

        logger.info(
            "Phase 1 coverage pass: %s zero-yield chunk(s) hold %.0f%% of tokens; "
            "re-examining in %s batch(es)",
            len(uncovered),
            100.0 * uncovered_tokens / total_tokens,
            len(coverage_batches),
        )
        base_index = len(batches)
        results: list[list[dict[str, Any]]] = []
        recovered = 0
        for i, coverage_batch in enumerate(coverage_batches):
            _, nodes = self._run_one_skeleton_batch(
                batch_idx=base_index + i,
                batch=coverage_batch,
                total_batches=base_index + len(coverage_batches),
                catalog_block=catalog_block,
                allowed_paths=allowed_paths,
                global_context=global_context,
                semantic_guide=semantic_guide,
                schema_json=schema_json,
                context=f"{context}_coverage",
                spec_by_path=spec_by_path,
                already_found_str=already_str,
                known_handles=known_handles,
                coverage_retry=True,
            )
            results.append(nodes)
            recovered += len(nodes)
        if recovered:
            self._bump("coverage_pass_recovered", recovered)
        logger.info("Phase 1 coverage pass: recovered %s node(s)", recovered)
        return coverage_batches, results

    def run(
        self,
        *,
        chunks: list[str],
        chunk_metadata: list[dict[str, Any]] | None,
        full_markdown: str,
        context: str = "document",
    ) -> dict[str, Any] | None:
        self._counters = {}
        self._dropped_chunk_ids = []
        self._phase1_elapsed = 0.0
        self._phase2_elapsed = 0.0
        self.last_run_stats = {}
        self.last_provenance = None
        provenance_enabled = self._config.provenance_mode != "off"
        token_counts: list[int] | None = None
        if chunk_metadata and len(chunk_metadata) == len(chunks):
            raw_counts = [m.get("token_count") for m in chunk_metadata]
            if not any(t is None for t in raw_counts):
                token_counts = [int(cast(int, t)) for t in raw_counts]
        allowed_paths = set(self._catalog.paths())
        schema_json = json.dumps(skeleton_output_schema(self._catalog.paths()))
        catalog_block = build_skeleton_catalog_block(self._catalog)
        semantic_guide = build_skeleton_semantic_guide(self._catalog)
        global_context = chunks[0][:2000] if chunks and isinstance(chunks[0], str) else None

        batches = chunk_batches_by_token_limit(
            chunks, token_counts, max_batch_tokens=self._config.skeleton_batch_tokens
        )
        logger.info(
            "Phase 1 (skeleton): %s chunks in %s batches (batch budget %s tokens)",
            len(chunks),
            len(batches),
            self._config.skeleton_batch_tokens,
        )
        workers = max(1, self._config.parallel_workers)
        self._effective_workers = workers
        if workers > 1 and len(batches) == 1:
            logger.warning(
                "Phase 1 (skeleton): only one batch; parallel workers will not be used. "
                "Ensure chunk_max_tokens (per chunk) and dense_skeleton_batch_tokens are set "
                "so long documents split into multiple chunks and batches."
            )
        phase1_start = time.perf_counter()
        spec_by_path = {s.path: s for s in self._catalog.nodes}
        skeleton_results = self._run_skeleton_phase(
            batches=batches,
            workers=workers,
            catalog_block=catalog_block,
            allowed_paths=allowed_paths,
            global_context=global_context,
            semantic_guide=semantic_guide,
            schema_json=schema_json,
            context=context,
            spec_by_path=spec_by_path,
        )
        merged_skeleton = merge_skeleton_batches(skeleton_results, self._catalog)
        coverage_batches, coverage_results = self._run_coverage_pass(
            batches=batches,
            merged_skeleton=merged_skeleton,
            catalog_block=catalog_block,
            allowed_paths=allowed_paths,
            global_context=global_context,
            semantic_guide=semantic_guide,
            schema_json=schema_json,
            context=context,
            spec_by_path=spec_by_path,
        )
        if coverage_results:
            # Coverage batches join the batch list so fill-context scoping and
            # provenance chunk records see them like any first-pass batch.
            batches.extend(coverage_batches)
            skeleton_results.extend(coverage_results)
            merged_skeleton = merge_skeleton_batches(skeleton_results, self._catalog)
        phase1_elapsed = time.perf_counter() - phase1_start
        self._phase1_elapsed = phase1_elapsed
        merged_skeleton, reconciliation_merged = self._dedupe_skeleton(
            merged_skeleton, spec_by_path, context
        )
        # Invariant: drop root ids that contradict their field-name semantics
        # (or echo the template class name) so a sparse-document mis-capture is
        # not locked in by Phase 2 id restoration.
        merged_skeleton = strip_mislabeled_root_ids(
            merged_skeleton, template_class_name=self._template.__name__
        )
        if self._debug_dir:
            self._write_debug("dense_skeleton_graph.json", {"nodes": merged_skeleton})

        # Provenance (H2 + H5): chunk index and ledger freeze. Fill cannot
        # change node identity, so the ledger frozen here stays valid.
        ledger: ProvenanceLedger | None = None
        if provenance_enabled:
            chunk_records = self._build_chunk_records(batches, chunk_metadata, len(chunks))
            ledger = self._freeze_ledger(merged_skeleton, chunk_records, spec_by_path)

        path_counts: dict[str, int] = {}
        for n in merged_skeleton:
            p = n.get("path") or ""
            path_counts[p] = path_counts.get(p, 0) + 1
        total = len(merged_skeleton)
        # Quality gate (invariant): a usable skeleton needs a root instance.
        # Without one there is nothing to attach the graph to, and the direct
        # fallback in the strategy is the better path.
        if path_counts.get("", 0) <= 0 or total < 1:
            reason = "missing_root" if total >= 1 else "empty_skeleton"
            logger.warning("Phase 1 (skeleton) quality gate failed: %s", reason)
            self._finalize_run_stats(
                total,
                reconciliation_merged,
                {},
                total_chunks=len(chunks),
                covered_chunks=self._covered_chunk_count(merged_skeleton),
                gate_failure=reason,
            )
            if ledger is not None and self._debug_dir:
                # Partial ledger for audit; last_provenance stays None because
                # the strategy falls back to direct extraction, whose result
                # this ledger does not describe.
                self._write_debug("dense_provenance_partial.json", ledger.model_dump(mode="json"))
            if self._on_trace:
                self._on_trace({"contract": "dense", "phase1_quality": False, "reason": reason})
            return None

        path_descriptors = skeleton_to_descriptors(merged_skeleton, self._catalog)
        path_filled: dict[str, list[dict[str, Any]]] = {}
        fill_paths = [p for p in bottom_up_path_order(self._catalog) if path_descriptors.get(p)]
        phase2_start = time.perf_counter()

        batch_texts = [format_batch_markdown(batch) for batch in batches]
        global_head = chunks[0][:2000] if chunks and isinstance(chunks[0], str) else ""

        # Build flat list of fill jobs:
        # (path, spec, batch_descriptors, batch_index, sub_schema, fill_markdown)
        fill_jobs: list[tuple[str, NodeSpec, list[dict[str, Any]], int, str, str]] = []
        for path in fill_paths:
            descriptors = path_descriptors[path]
            spec = spec_by_path.get(path)
            if not spec:
                continue
            sub_schema = build_projected_fill_schema(self._template, spec, self._catalog)
            # Per-parent fill for reference-carrying paths: batching N sibling
            # parents into one call is the observed cause of first-instance
            # membership dumping (one parent absorbs every row of a summary
            # table, the rest stay empty). One instance per call makes
            # "membership must be stated for THIS instance" enforceable.
            fill_cap = (
                1
                if path_has_reference_fields(self._template, spec)
                else self._config.fill_nodes_cap
            )
            fill_batches = [
                descriptors[i : i + fill_cap] for i in range(0, len(descriptors), fill_cap)
            ]
            for batch_index, batch_descriptors in enumerate(fill_batches):
                fill_markdown = self._build_fill_context(
                    path, batch_descriptors, batch_texts, full_markdown, global_head
                )
                fill_jobs.append(
                    (path, spec, batch_descriptors, batch_index, sub_schema, fill_markdown)
                )
                if ledger is not None:
                    for desc in batch_descriptors:
                        key = _provenance_identity_key(path, desc.get("ids") or {}, spec.id_fields)
                        entry = ledger.nodes.get(key) if key else None
                        if entry is not None and batch_index not in entry.fill_batches:
                            entry.fill_batches.append(batch_index)

        logger.info(
            "Phase 2 (fill): %s jobs across %s paths (%s workers)",
            len(fill_jobs),
            len(fill_paths),
            workers if len(fill_jobs) > 1 else 1,
        )
        fill_progress = ProgressTracker(
            logger, total=len(fill_jobs), label="Phase 2 (fill)", unit="job"
        )
        if workers <= 1 or len(fill_jobs) <= 1:
            for path, spec, batch_descriptors, _bi, sub_schema, fill_markdown in fill_jobs:
                _p, _bi, sanitized = self._run_one_fill_batch(
                    path=path,
                    spec=spec,
                    batch_descriptors=batch_descriptors,
                    batch_index=_bi,
                    sub_schema=sub_schema,
                    fill_markdown=fill_markdown,
                    context=context,
                )
                path_filled.setdefault(path, []).extend(sanitized)
                fill_progress.advance(note=f"path: {path or '<root>'}")
        else:
            results_by_path: dict[str, list[tuple[int, list[dict[str, Any]]]]] = {}
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fill_futures: dict[
                    Future[tuple[str, int, list[dict[str, Any]]]], tuple[str, int]
                ] = {
                    ex.submit(
                        self._run_one_fill_batch,
                        path=path,
                        spec=spec,
                        batch_descriptors=batch_descriptors,
                        batch_index=batch_index,
                        sub_schema=sub_schema,
                        fill_markdown=fill_markdown,
                        context=context,
                    ): (path, batch_index)
                    for path, spec, batch_descriptors, batch_index, sub_schema, fill_markdown in fill_jobs
                }
                for future in as_completed(fill_futures):
                    path, batch_index = fill_futures[future]
                    try:
                        p, bi, sanitized = future.result()
                        results_by_path.setdefault(p, []).append((bi, sanitized))
                        fill_progress.advance(note=f"path: {p or '<root>'}")
                    except Exception as e:
                        logger.warning(
                            "Fill job for path '%s' (batch %s) failed: %s",
                            path,
                            batch_index,
                            e,
                        )
                        fill_progress.advance(note="failed")
            for path, pairs in results_by_path.items():
                pairs.sort(key=lambda x: x[0])
                for _bi, sanitized in pairs:
                    path_filled.setdefault(path, []).extend(sanitized)
        fill_progress.finish()

        phase2_elapsed = time.perf_counter() - phase2_start
        self._phase2_elapsed = phase2_elapsed
        merge_stats: dict[str, int] = {}
        merge_events: list[dict[str, Any]] | None = [] if ledger is not None else None
        root = merge_filled_into_root(
            path_filled,
            path_descriptors,
            self._catalog,
            stats_out=merge_stats,
            events_out=merge_events,
            template=self._template,
        )
        # Invariant: id-only childless branch nodes are skeleton noise that the
        # fill could not substantiate; they are always pruned. Rescue buckets
        # and placeholders keep their children and therefore always survive.
        root = prune_barren_branches(root, self._catalog, events_out=merge_events)
        if ledger is not None and merge_events is not None:
            self._finalize_provenance(
                ledger, merge_events, spec_by_path, path_descriptors, path_filled
            )
        self._finalize_run_stats(
            len(merged_skeleton),
            reconciliation_merged,
            merge_stats,
            total_chunks=len(chunks),
            covered_chunks=self._covered_chunk_count(merged_skeleton),
        )
        if self._debug_dir:
            self._write_debug("dense_merge_stats.json", merge_stats)
            self._write_debug("dense_run_stats.json", self.last_run_stats)
            if ledger is not None:
                self._write_debug("dense_provenance.json", ledger.model_dump(mode="json"))
        if self._on_trace:
            self._on_trace(
                {
                    "contract": "dense",
                    "phase1_elapsed": round(phase1_elapsed, 3),
                    "phase2_elapsed": round(phase2_elapsed, 3),
                    "skeleton_nodes": len(merged_skeleton),
                    "path_counts": path_counts,
                    "merge_stats": merge_stats,
                    "run_stats": self.last_run_stats,
                }
            )
        return root
