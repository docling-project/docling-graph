"""
Deterministic cross-document merge of per-document induction candidates.

The three LLM passes in ``documents.py`` produce one
:class:`DocumentCandidates` per source (already evidence-gated); this module
unions them into a single loose draft dict in ``TemplateSpec`` shape, ready
for ``linter.repair_draft``. Everything here is pure data manipulation — no
LLM, no I/O — so the same candidates always merge to the same draft.

Merge rules (design §4.4):

- Classes union by canonical name, keyed with
  ``canonicalize_identity_for_dedup`` — the same canonicalizer the runtime
  registry uses, so "Line item"/"LineItem" unify identically at generation
  and extraction time. ``kind`` resolves by majority vote; entity wins a tie
  only when an identity candidate survived the verbatim gate in >=1 document.
  An entity-voted class with **no** surviving identity evidence is demoted to
  component here (the design's own demotion — never invent ids) with a
  ``missing_identity`` gap; ``repair_draft`` is the safety net, not the
  primary mechanism.
- Fields union; scalar types promote along the ``int -> float -> str``
  lattice on disagreement (never narrowing); ``is_list = any``; examples
  union deduped canonically, capped at 5, preferring distinct documents
  (round-robin).
- Enums union members; unions wider than ``max_enum_members`` demote the
  field to ``str`` with the top values listed in its description. Enum display
  names are allocated from the same taken-name pool as the class display names
  (mirroring the ontology compilers' ``_taken_names``), so a model ``Status``
  and an enum ``Status`` can never coexist in the draft — the linter's rename
  cascade cannot tell them apart and would corrupt edge targets.
- ``documented_max_count`` takes the max across documents and stays the
  **documented** figure — ``repair_draft`` doubles it exactly once.
- A field seen in only 1 of >=3 documents gets a "Rare: " description prefix
  and a report flag.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.utils.entity_name_normalizer import canonicalize_identity_for_dedup

from ..naming import to_pascal_case
from ..spec import SCALAR_TYPES, SpecGap

__all__ = [
    "MAX_ENUM_MEMBERS",
    "ClassCandidate",
    "DocumentCandidates",
    "FieldCandidate",
    "MergeDecision",
    "MergeReport",
    "canonical_key",
    "merge_documents",
]

MAX_ENUM_MEMBERS = 24
"""Enums wider than this demote to ``str`` (design §4.4 / templategen config)."""

_MAX_MERGED_EXAMPLES = 5
_MAX_MERGED_EVIDENCE = 5
_ENUM_DEMOTION_LISTED = 10
_RARE_FIELD_MIN_DOCS = 3


def canonical_key(name: str) -> str:
    """Merge key for class/field/example names: lowercase alphanumerics only.

    Uses :func:`canonicalize_identity_for_dedup`'s generic branch (field name
    ``"candidate"`` is deliberately not a name-style field), so
    ``"Line item"`` / ``"LineItem"`` / ``"line_item"`` all map to
    ``"lineitem"`` — exactly how the runtime registry would key them.
    """
    return canonicalize_identity_for_dedup("candidate", name)


# ---------------------------------------------------------------------------
# Per-document candidate models (produced by documents.py, consumed here)
# ---------------------------------------------------------------------------


class FieldCandidate(BaseModel):
    """One gated field proposal from one document."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = "str"
    is_list: bool = False
    description: str = ""
    examples: list[str] = Field(default_factory=list)
    role: Literal["identity", "property", "edge"] = "property"
    # edge-only:
    edge_label: str | None = None
    reference: bool = False
    # enum-only:
    enum_name: str | None = None
    enum_members: list[str] = Field(default_factory=list)
    enum_synonyms: dict[str, list[str]] = Field(default_factory=dict)
    unit_varies: bool = False
    evidence: list[str] = Field(default_factory=list)


class ClassCandidate(BaseModel):
    """One gated class proposal from one document."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["entity", "component"] = "component"
    is_root: bool = False
    what_it_is: str = ""
    confusable_with: str = ""
    documented_max_count: int | None = None
    evidence_quotes: list[str] = Field(default_factory=list)
    fields: list[FieldCandidate] = Field(default_factory=list)
    identity_survived: bool = False
    """True when an identity candidate kept >=1 example through the verbatim gate."""


class DocumentCandidates(BaseModel):
    """All gated candidates induced from one source document."""

    model_config = ConfigDict(extra="forbid")

    name: str
    classes: list[ClassCandidate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Merge report
# ---------------------------------------------------------------------------

DecisionKind = Literal[
    "root_election",
    "kind_vote",
    "identity_demotion",
    "type_promotion",
    "enum_demotion",
    "rare_field",
    "overflow_drop",
    "edge_conflict",
    "edge_dropped",
    "unit_varies",
]


class MergeDecision(BaseModel):
    """One deterministic merge decision, for the induction report."""

    model_config = ConfigDict(extra="forbid")

    kind: DecisionKind
    model: str = ""
    field: str | None = None
    message: str


class MergeReport(BaseModel):
    """All decisions of one cross-document merge."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[MergeDecision] = Field(default_factory=list)

    def by_kind(self, kind: str) -> list[MergeDecision]:
        return [d for d in self.decisions if d.kind == kind]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_Occurrences = list[tuple[int, ClassCandidate]]


def _group_classes(
    docs: Sequence[DocumentCandidates],
) -> tuple[dict[str, _Occurrences], list[str]]:
    groups: dict[str, _Occurrences] = {}
    order: list[str] = []
    for doc_index, doc in enumerate(docs):
        seen_in_doc: set[str] = set()
        for cls in doc.classes:
            key = canonical_key(cls.name)
            if not key or key in seen_in_doc:
                continue
            seen_in_doc.add(key)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((doc_index, cls))
    return groups, order


def _elect_root(
    groups: dict[str, _Occurrences],
    order: list[str],
    root_name: str | None,
    decisions: list[MergeDecision],
) -> str:
    if root_name:
        key = canonical_key(root_name)
        if key in groups:
            decisions.append(
                MergeDecision(
                    kind="root_election",
                    model=groups[key][0][1].name,
                    message=f"root fixed by root_name={root_name!r}",
                )
            )
            return key
    votes = {key: sum(1 for _, cls in groups[key] if cls.is_root) for key in order}
    best = max(order, key=lambda k: votes[k])  # first maximal key wins ties
    if votes[best] > 0:
        decisions.append(
            MergeDecision(
                kind="root_election",
                model=groups[best][0][1].name,
                message=f"root elected by is_root vote ({votes[best]} of {len(groups[best])})",
            )
        )
        return best
    # No is_root votes at all: prefer a class that is never an edge target.
    targets = {
        canonical_key(field.type)
        for key in order
        for _, cls in groups[key]
        for field in cls.fields
        if field.role == "edge"
    }
    elected = next((key for key in order if key not in targets), order[0])
    decisions.append(
        MergeDecision(
            kind="root_election",
            model=groups[elected][0][1].name,
            message="no is_root votes; elected the first class that is no edge's target",
        )
    )
    return elected


def _cap_models(
    groups: dict[str, _Occurrences],
    order: list[str],
    root_key: str,
    max_models: int,
    decisions: list[MergeDecision],
) -> list[str]:
    if len(order) <= max_models:
        return list(order)
    ranked = sorted(
        (key for key in order if key != root_key),
        key=lambda k: (-len(groups[k]), order.index(k)),
    )
    kept = {root_key, *ranked[: max_models - 1]}
    for key in order:
        if key not in kept:
            decisions.append(
                MergeDecision(
                    kind="overflow_drop",
                    model=groups[key][0][1].name,
                    message=(
                        f"class dropped: merged model count exceeds max_models={max_models} "
                        f"(seen in {len(groups[key])} document(s))"
                    ),
                )
            )
    return [key for key in order if key in kept]


def _vote_kind(
    occurrences: _Occurrences,
    name: str,
    decisions: list[MergeDecision],
    gaps: list[SpecGap],
) -> str:
    counts = Counter(cls.kind for _, cls in occurrences)
    identity_survived = any(cls.identity_survived for _, cls in occurrences)
    if counts["entity"] > counts["component"]:
        kind = "entity"
    elif counts["component"] > counts["entity"]:
        kind = "component"
    else:
        kind = "entity" if identity_survived else "component"
    if len(counts) > 1:
        decisions.append(
            MergeDecision(
                kind="kind_vote",
                model=name,
                message=(
                    f"kind vote entity={counts['entity']} / component={counts['component']} "
                    f"-> {kind}"
                    + (
                        " (tie broken by surviving identity evidence)"
                        if counts["entity"] == counts["component"]
                        else ""
                    )
                ),
            )
        )
    if kind == "entity" and not identity_survived:
        kind = "component"
        decisions.append(
            MergeDecision(
                kind="identity_demotion",
                model=name,
                message=(
                    "entity with no identity evidence surviving the verbatim gate — "
                    "demoted to component (never invent ids)"
                ),
            )
        )
        gaps.append(
            SpecGap(
                model=name,
                field=None,
                kind="missing_identity",
                note="no identity evidence survived the verbatim gate; demoted to component",
            )
        )
    return kind


def _compose_docstring(name: str, what_it_is: str, confusable_with: str) -> str:
    doc = what_it_is.strip() or f"{name}."
    if not doc.endswith((".", "!", "?")):
        doc += "."
    confusable = confusable_with.strip().rstrip(".")
    if confusable:
        doc += f" Not to be confused with {confusable}."
    return doc


def _resolve_scalar(types: Sequence[str]) -> tuple[str, bool]:
    """Promotion lattice ``int -> float -> str``; returns (type, promoted)."""
    unique = {t if t in SCALAR_TYPES else "str" for t in types}
    if len(unique) == 1:
        return next(iter(unique)), False
    if unique <= {"int", "float"}:
        return "float", True
    return "str", True


def _round_robin_examples(occ: list[tuple[int, FieldCandidate]]) -> list[str]:
    """Union examples, deduped canonically, cap 5, preferring distinct docs."""
    per_doc: dict[int, list[str]] = {}
    for doc_index, field in occ:
        per_doc.setdefault(doc_index, []).extend(field.examples)
    merged: list[str] = []
    seen: set[str] = set()
    queues = [list(examples) for _, examples in sorted(per_doc.items())]
    while len(merged) < _MAX_MERGED_EXAMPLES and any(queues):
        for queue in queues:
            while queue:
                example = queue.pop(0)
                key = canonical_key(example) or " ".join(example.split())
                if key in seen:
                    continue
                seen.add(key)
                merged.append(example)
                break
            if len(merged) >= _MAX_MERGED_EXAMPLES:
                break
    return merged


def _union_evidence(occ: list[tuple[int, FieldCandidate]]) -> list[str]:
    evidence: list[str] = []
    seen: set[str] = set()
    for _, field in occ:
        for quote in field.evidence:
            normalized = " ".join(quote.split())
            if normalized and normalized not in seen:
                seen.add(normalized)
                evidence.append(quote)
            if len(evidence) >= _MAX_MERGED_EVIDENCE:
                return evidence
    return evidence


def _unique_display(base: str, taken: set[str]) -> str:
    """First of ``base``, ``base_2``, ... not in ``taken``; adds it to ``taken``.

    Mirrors the ontology compilers' ``unique_name`` helper (kept local so the
    induce package stays decoupled from ``templategen.ontology``).
    """
    name = base
    counter = 2
    while name in taken:
        name = f"{base}_{counter}"
        counter += 1
    taken.add(name)
    return name


class _EnumRegistry:
    """Accumulates enum unions across classes; demotes over-wide ones at the end.

    ``taken_names`` is the shared class/enum display-name pool: an enum whose
    PascalCase name collides with a kept class (or an earlier enum) gets a
    ``_2`` suffix here, so the draft never carries the collision downstream.
    Entries stay keyed by the *pre-suffix* canonical name, so every field
    referencing the same proposed enum still unions into one entry.
    """

    def __init__(self, max_enum_members: int, taken_names: set[str] | None = None) -> None:
        self.max_enum_members = max_enum_members
        self.taken_names = taken_names if taken_names is not None else set()
        self.entries: dict[str, dict[str, Any]] = {}
        self.field_refs: list[tuple[dict[str, Any], str]] = []

    def register(self, field_dict: dict[str, Any], occ: list[tuple[int, FieldCandidate]]) -> str:
        enum_occ = [f for _, f in occ if f.enum_members or f.enum_name]
        raw_name = next((f.enum_name for f in enum_occ if f.enum_name), None) or field_dict["name"]
        base = to_pascal_case(raw_name)
        key = canonical_key(base) or base
        entry = self.entries.get(key)
        if entry is None:
            display = _unique_display(base, self.taken_names)
            entry = self.entries[key] = {
                "name": display,
                "members": [],
                "member_keys": set(),
                "synonyms": {},
            }
        for candidate in enum_occ:
            for member in candidate.enum_members:
                member_key = canonical_key(member) or member
                if member_key not in entry["member_keys"]:
                    entry["member_keys"].add(member_key)
                    entry["members"].append(member)
            for member, phrases in candidate.enum_synonyms.items():
                bucket = entry["synonyms"].setdefault(member, [])
                for phrase in phrases:
                    if phrase not in bucket:
                        bucket.append(phrase)
        self.field_refs.append((field_dict, key))
        return str(entry["name"])

    def finalize(self, decisions: list[MergeDecision]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for key, entry in self.entries.items():
            members: list[str] = entry["members"]
            if members and len(members) <= self.max_enum_members:
                kept.append(
                    {
                        "name": entry["name"],
                        "members": members,
                        "synonyms": {m: p for m, p in entry["synonyms"].items() if m in members},
                        "include_other": True,
                    }
                )
                continue
            top = ", ".join(members[:_ENUM_DEMOTION_LISTED])
            for field_dict, ref_key in self.field_refs:
                if ref_key != key:
                    continue
                field_dict["type"] = "str"
                if members:
                    suffix = f"One of: {top}" + (
                        ", ..." if len(members) > _ENUM_DEMOTION_LISTED else "."
                    )
                    field_dict["description"] = f"{field_dict['description']} {suffix}".strip()
            if members:
                decisions.append(
                    MergeDecision(
                        kind="enum_demotion",
                        model=entry["name"],
                        message=(
                            f"enum '{entry['name']}' has {len(members)} members "
                            f"(> {self.max_enum_members}) — demoted to str with top values "
                            "in the description"
                        ),
                    )
                )
        return kept


def _merge_field(
    fkey: str,
    occ: list[tuple[int, FieldCandidate]],
    *,
    model_name: str,
    kept_keys: set[str],
    display: dict[str, str],
    enum_registry: _EnumRegistry,
    decisions: list[MergeDecision],
    class_group_count: int,
    group_of: Sequence[int],
) -> dict[str, Any] | None:
    del fkey  # keyed by the caller; the merged field keeps the first-seen name
    name = occ[0][1].name
    roles = {field.role for _, field in occ}
    role = "identity" if "identity" in roles else ("edge" if "edge" in roles else "property")
    description = next((f.description.strip() for _, f in occ if f.description.strip()), "")
    field_dict: dict[str, Any] = {
        "name": name,
        "type": "str",
        "is_list": role != "identity" and any(f.is_list for _, f in occ),
        "description": description,
        "examples": [],
        "role": role,
        "evidence": _union_evidence(occ),
    }

    if role == "edge":
        edge_occ = [f for _, f in occ if f.role == "edge"]
        target_keys = [canonical_key(f.type) for f in edge_occ]
        target_key, _count = Counter(target_keys).most_common(1)[0]
        if len(set(target_keys)) > 1:
            decisions.append(
                MergeDecision(
                    kind="edge_conflict",
                    model=model_name,
                    field=name,
                    message=f"documents disagree on the edge target; majority '{target_key}' wins",
                )
            )
        if target_key not in kept_keys:
            decisions.append(
                MergeDecision(
                    kind="edge_dropped",
                    model=model_name,
                    field=name,
                    message="edge dropped: its target class was not kept in the merge",
                )
            )
            return None
        field_dict["type"] = display[target_key]
        labels = [f.edge_label for f in edge_occ if f.edge_label]
        if labels:
            field_dict["edge_label"] = labels[0]
            if len(set(labels)) > 1:
                decisions.append(
                    MergeDecision(
                        kind="edge_conflict",
                        model=model_name,
                        field=name,
                        message=f"labels disagree {sorted(set(labels))}; '{labels[0]}' wins",
                    )
                )
        field_dict["reference"] = all(f.reference for f in edge_occ)
        return field_dict

    field_dict["examples"] = _round_robin_examples(occ)
    if role == "property" and any(f.enum_members or f.enum_name for _, f in occ):
        field_dict["type"] = enum_registry.register(field_dict, occ)
    else:
        resolved, promoted = _resolve_scalar([f.type for _, f in occ])
        field_dict["type"] = resolved
        if promoted:
            decisions.append(
                MergeDecision(
                    kind="type_promotion",
                    model=model_name,
                    field=name,
                    message=(
                        f"documents disagree on the type "
                        f"({sorted({f.type for _, f in occ})}) — promoted to '{resolved}' "
                        "(int -> float -> str, never narrows)"
                    ),
                )
            )
    if any(f.unit_varies for _, f in occ):
        decisions.append(
            MergeDecision(
                kind="unit_varies",
                model=model_name,
                field=name,
                message=(
                    "unit varies across the document — consider a Measurement component "
                    "(value + unit) for this field"
                ),
            )
        )

    # Document counts are per physical document: windows of one oversized
    # document never make a mid-document field look "rare".
    groups_seen = {group_of[doc_index] for doc_index, _ in occ}
    if class_group_count >= _RARE_FIELD_MIN_DOCS and len(groups_seen) == 1:
        base = field_dict["description"] or (f"Seen in 1 of {class_group_count} sample documents.")
        field_dict["description"] = f"Rare: {base}"
        decisions.append(
            MergeDecision(
                kind="rare_field",
                model=model_name,
                field=name,
                message=(
                    f"field seen in only 1 of {class_group_count} documents — flagged 'Rare:'; "
                    "prune via the SPEC YAML if spurious"
                ),
            )
        )
    return field_dict


def _merge_fields(
    occurrences: _Occurrences,
    *,
    model_name: str,
    kept_keys: set[str],
    display: dict[str, str],
    enum_registry: _EnumRegistry,
    decisions: list[MergeDecision],
    group_of: Sequence[int],
) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[int, FieldCandidate]]] = {}
    order: list[str] = []
    for doc_index, cls in occurrences:
        for field in cls.fields:
            fkey = canonical_key(field.name)
            if not fkey:
                continue
            if fkey not in groups:
                groups[fkey] = []
                order.append(fkey)
            groups[fkey].append((doc_index, field))
    # Identity fields first (stable within each bucket).
    order.sort(key=lambda k: 0 if any(f.role == "identity" for _, f in groups[k]) else 1)
    class_group_count = len({group_of[doc_index] for doc_index, _ in occurrences})
    merged: list[dict[str, Any]] = []
    for fkey in order:
        field_dict = _merge_field(
            fkey,
            groups[fkey],
            model_name=model_name,
            kept_keys=kept_keys,
            display=display,
            enum_registry=enum_registry,
            decisions=decisions,
            class_group_count=class_group_count,
            group_of=group_of,
        )
        if field_dict is not None:
            merged.append(field_dict)
    return merged


def _merge_class(
    key: str,
    occurrences: _Occurrences,
    *,
    is_root: bool,
    display: dict[str, str],
    kept_keys: set[str],
    enum_registry: _EnumRegistry,
    decisions: list[MergeDecision],
    gaps: list[SpecGap],
    group_labels: Mapping[int, str],
    group_of: Sequence[int],
) -> dict[str, Any]:
    name = display[key]
    kind = "root" if is_root else _vote_kind(occurrences, name, decisions, gaps)
    what_it_is = next((c.what_it_is for _, c in occurrences if c.what_it_is.strip()), "")
    confusable = next((c.confusable_with for _, c in occurrences if c.confusable_with.strip()), "")
    fields = _merge_fields(
        occurrences,
        model_name=name,
        kept_keys=kept_keys,
        display=display,
        enum_registry=enum_registry,
        decisions=decisions,
        group_of=group_of,
    )
    if kind == "component":
        for field in fields:
            if field["role"] == "identity":
                field["role"] = "property"
        identity_fields: list[str] = []
    else:
        identity_fields = [f["name"] for f in fields if f["role"] == "identity"]
    model: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "docstring": _compose_docstring(name, what_it_is, confusable),
        "identity_fields": identity_fields,
        "fields": fields,
        "provenance": "induced",
        "source_ref": "induced from: "
        + ", ".join(group_labels[g] for g in sorted({group_of[i] for i, _ in occurrences})),
    }
    if kind != "component":
        documented = max(
            (c.documented_max_count for _, c in occurrences if c.documented_max_count),
            default=None,
        )
        if documented is not None:
            # The DOCUMENTED maximum: repair_draft doubles it exactly once.
            model["max_instances"] = documented
    return model


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def merge_documents(
    docs: Sequence[DocumentCandidates],
    *,
    root_name: str | None = None,
    max_models: int = 30,
    max_enum_members: int = MAX_ENUM_MEMBERS,
    doc_groups: Sequence[int] | None = None,
    group_names: Sequence[str] | None = None,
) -> tuple[dict[str, Any], MergeReport, list[SpecGap]]:
    """Merge per-document candidates into one loose draft dict.

    Returns ``(draft, report, gaps)`` where ``draft`` is a TemplateSpec-shaped
    dict (documented ``max_instances``, possibly missing edge labels) meant
    for ``linter.repair_draft`` — names are NOT pre-sanitized here, because
    ``repair_draft`` cascades renames.

    Args:
        docs: One gated candidate set per induction unit (a document, or one
            window of an oversized document), in source order.
        root_name: Optional root class name; overrides the ``is_root`` vote
            and renames the elected root.
        max_models: Cap on merged models; overflow classes (and edges into
            them) are dropped with loud ``overflow_drop`` decisions.
        max_enum_members: Enum unions wider than this demote to ``str``.
        doc_groups: Physical-document group index per entry of ``docs``
            (windows of one oversized document share a group). Document-count
            semantics — the rare-field flag and ``source_ref`` labels — use
            distinct groups, so one long document never counts as many.
            Defaults to every unit being its own document. Example variety
            (round-robin) deliberately still treats windows as distinct
            sources.
        group_names: Display label per group index (the physical document
            name); defaults to the first unit name of each group.
    """
    if not docs:
        raise ValueError("merge_documents requires at least one DocumentCandidates")
    if doc_groups is None:
        doc_groups = list(range(len(docs)))
    if len(doc_groups) != len(docs):
        raise ValueError("merge_documents: doc_groups must align 1:1 with docs")
    decisions: list[MergeDecision] = []
    gaps: list[SpecGap] = []
    doc_names = [doc.name for doc in docs]
    group_labels: dict[int, str] = {}
    for doc_index, group in enumerate(doc_groups):
        if group not in group_labels:
            named = (
                group_names[group]
                if group_names is not None and 0 <= group < len(group_names)
                else doc_names[doc_index]
            )
            group_labels[group] = named

    groups, order = _group_classes(docs)
    if not groups:
        raise ValueError("merge_documents: no candidate classes to merge")
    root_key = _elect_root(groups, order, root_name, decisions)
    kept_order = _cap_models(groups, order, root_key, max_models, decisions)
    kept_keys = set(kept_order)

    display = {key: to_pascal_case(groups[key][0][1].name) for key in kept_order}
    if root_name:
        new_display = to_pascal_case(root_name)
        if display[root_key] != new_display:
            decisions.append(
                MergeDecision(
                    kind="root_election",
                    model=display[root_key],
                    message=f"root class renamed to '{new_display}' (root_name)",
                )
            )
            display[root_key] = new_display

    # Class display names seed the pool: enum names must not collide with them.
    enum_registry = _EnumRegistry(max_enum_members, taken_names=set(display.values()))
    models = [
        _merge_class(
            key,
            groups[key],
            is_root=key == root_key,
            display=display,
            kept_keys=kept_keys,
            enum_registry=enum_registry,
            decisions=decisions,
            gaps=gaps,
            group_labels=group_labels,
            group_of=doc_groups,
        )
        for key in kept_order
    ]
    enums = enum_registry.finalize(decisions)

    draft: dict[str, Any] = {
        "module_docstring": (
            f"Knowledge-graph template induced from {len(docs)} sample document(s): "
            + ", ".join(doc_names)
            + "."
        ),
        "root": display[root_key],
        "enums": enums,
        "models": models,
        "generator": {
            "tool": "docling-graph templategen",
            "path": "from-docs",
            "sources": doc_names,
        },
    }
    return draft, MergeReport(decisions=decisions), gaps
