"""
Rulebook linter: deterministic check-and-repair for :class:`TemplateSpec` IRs.

Encodes ``docs/fundamentals/schema-definition/`` as executable rules (R1-R22
in the template-generation design). Every rule is a *check* plus a
*deterministic repair*; repairs are monotone (demotions, flag-clearing,
renames, sentence drops), so the fixpoint loop terminates.

Where each rule family is enforced (the design's three nested guarantees):

- **IR — unrepresentable** (``spec.py`` intrinsic validators): the structural
  halves of R1/R2 (component-with-identity, entity with 0 or >2 identity
  fields, list-/enum-/model-typed identity), edge/label coherence (half of
  R9), ``closed_catalog`` without ``reference`` (half of R12), dangling
  type/``canonical_home`` references, duplicate names, and the 5-example cap
  (half of R3). :func:`repair_draft` applies the *pre-validation* repairs
  (R1 demotions / identity trimming / root-identity synthesis, R2 retyping,
  name sanitation, edge-label derivation) that make a loose producer draft
  constructible in the first place.
- **Linter — this module**: R1-R6 (draft halves plus post-validation checks),
  R9-R16, R18-R22.
- **Renderer — by construction** (not checked here): R7 (Optionality Law:
  every non-identity field renders ``Optional``/defaulted), R8
  (``default_factory=list`` on list edges, ``Optional`` on single edges), R17
  (enums render as ``(str, Enum)`` with an OTHER member and a
  ``mode="before"`` normalizer), and the rendering half of R15 (string
  forward references + ``model_rebuild()`` for the self-references recorded
  here).

``max_instances`` contract (R13): producers put the **documented** maximum in
drafts handed to :func:`repair_draft`, which doubles it exactly once ("~2x the
documented maximum", best-practices.md). A validated ``TemplateSpec``
therefore always stores the already-doubled bound; :func:`lint_spec` and the
renderer never double again, and R13's injected docstring sentence derives the
documented figure as ``max_instances // 2``.

Report entries whose ``rule_id`` is ``"IR"`` are pure constructibility repairs
made by :func:`repair_draft` (unresolved types, stray edge markers, duplicate
names, empty docstrings, ...) that have no dedicated R-rule of their own.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter, deque
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from docling_graph.exceptions import DoclingGraphError

from .naming import (
    BANNED_EDGE_LABELS,
    RESERVED_FIELD_RENAMES,
    derive_edge_label,
    is_verb_phrase,
    normalize_edge_label,
    sanitize_class_name,
    sanitize_field_name,
    to_snake_case,
)
from .spec import (
    MAX_FIELD_EXAMPLES,
    MAX_IDENTITY_FIELDS,
    SCALAR_TYPES,
    FieldSpec,
    ModelSpec,
    SpecGap,
    TemplateSpec,
)

__all__ = [
    "DOCSTRING_WINDOW",
    "MAX_LINT_PASSES",
    "MAX_NESTING_DEPTH",
    "LintEntry",
    "LintReport",
    "TemplateLintError",
    "lint_spec",
    "repair_draft",
]

DOCSTRING_WINDOW = 240
"""Characters of a class docstring that dense Phase 1 actually sees (R4)."""

MAX_NESTING_DEPTH = 4
"""Maximum full-nesting depth from the root (best-practices.md: 2-4 levels)."""

MAX_LINT_PASSES = 3
"""Fixpoint bound; repairs are monotone so 2 passes suffice in practice."""

Severity = Literal["error", "warn", "info"]

_ROLE_DATA_FIELD_NAMES = frozenset({"title", "role", "position", "function"})
_IDENTITY_RENAME_CANDIDATES = ("name", "title")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NOT_RE = re.compile(r"\bnot\b", re.IGNORECASE)
_CARDINALITY_PHRASES = ("at most", "maximum", "up to")
_NUMBER_NAME_RE = re.compile(r"(_number$|_no$|^ref_|_ref$)")
_POSITIONAL_ID_RE = re.compile(r"([A-Za-z_][A-Za-z_ ]*?)[ _-]?(\d+)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
_INVENTION_RE = re.compile(
    r"\b(generat(e|es|ed|ing)|assign(s|ed|ing)?|invent(s|ed|ing)?)\b", re.IGNORECASE
)
_COMPUTATION_RE = re.compile(
    r"\b(calculate|compute|sum|convert|round|multiply|derive)\b", re.IGNORECASE
)
_NA_RE = re.compile(r"\bn/a\b", re.IGNORECASE)
_GLOBAL_RULE_PHRASES = ("omit if", "leave empty")

_DOC_REFS: dict[str, str] = {
    "IR": "docling_graph/templategen/spec.py (intrinsic validators)",
    "R1": "field-definitions.md#identity-fields; entities-vs-components.md",
    "R2": "field-definitions.md#identity-fields",
    "R3": "field-definitions.md#identity-fields (2-5 verbatim examples)",
    "R4": "best-practices.md#token-economics-and-compacting (240-char window)",
    "R5": "field-definitions.md#identity-fields (name id fields honestly)",
    "R6": "best-practices.md#identity-and-resolution-primitives (never invented)",
    "R9": "relationships.md#edge-label-conventions",
    "R10": "best-practices.md#reference-edges-vs-nested-full-entities",
    "R11": "relationships.md#reference-edges-referencetrue",
    "R12": "relationships.md#closed-catalog-reference-edges-closed_catalogtrue",
    "R13": "best-practices.md#graph-assembly-mechanics (graph_max_instances)",
    "R14": "best-practices.md (2-4 nesting levels)",
    "R15": "advanced-patterns.md (forward references for recursive models)",
    "R16": "field-definitions.md#optionality-guidance; best-practices.md#deterministic-grounding-readiness",
    "R17": "validation.md (enum members + synonyms)",
    "R18": "best-practices.md#graph-assembly-mechanics (per-role wrapper pattern)",
    "R19": "validation.md (root list deduplication); best-practices.md#multi-llm-resilience",
    "R20": "docling_graph/templategen/naming.py (keyword/builtin renames)",
    "R21": "docling_graph/templategen/naming.py (reserved node-attr keys written by GraphConverter)",
    "R22": "best-practices.md (unmodeled-noise fields)",
    "R23": "relationships.md (every entity needs a discovery path from the root)",
    "R24": "entities-vs-components.md (components embed; they cannot own labeled edges)",
}


# ---------------------------------------------------------------------------
# Report models and the strict-mode exception
# ---------------------------------------------------------------------------


class LintEntry(BaseModel):
    """One finding: a repair performed (or required) or an advisory note."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    severity: Severity
    model: str
    field: str | None = None
    message: str
    repaired: bool = False
    doc_ref: str = ""


class LintReport(BaseModel):
    """All findings of one lint run, plus the gaps raised by repairs."""

    model_config = ConfigDict(extra="forbid")

    entries: list[LintEntry] = Field(default_factory=list)
    gaps: list[SpecGap] = Field(default_factory=list)

    @property
    def repaired_entries(self) -> list[LintEntry]:
        return [e for e in self.entries if e.repaired]

    @property
    def has_repairs(self) -> bool:
        return any(e.repaired for e in self.entries)

    def by_rule(self, rule_id: str) -> list[LintEntry]:
        return [e for e in self.entries if e.rule_id == rule_id]


class TemplateLintError(DoclingGraphError):
    """Strict-mode failure: the spec required repairs.

    The full :class:`LintReport` is attached as ``report`` and the repaired
    spec (when one was produced) as ``spec``, so callers can still print the
    repair log the design mandates ("repairs are printed either way").
    """

    def __init__(
        self,
        message: str,
        *,
        report: LintReport,
        spec: TemplateSpec | None = None,
    ) -> None:
        super().__init__(message, details={"violations": len(report.repaired_entries)})
        self.report = report
        self.spec = spec


class _Ctx:
    """Mutable per-pass collector; ``repair=False`` turns rules report-only."""

    __slots__ = ("entries", "gaps", "repair")

    def __init__(self, *, repair: bool) -> None:
        self.repair = repair
        self.entries: list[LintEntry] = []
        self.gaps: list[SpecGap] = []

    def add(
        self,
        rule_id: str,
        severity: Severity,
        model: str,
        field: str | None,
        message: str,
        *,
        repaired: bool,
    ) -> None:
        self.entries.append(
            LintEntry(
                rule_id=rule_id,
                severity=severity,
                model=model,
                field=field,
                message=message,
                repaired=repaired,
                doc_ref=_DOC_REFS.get(rule_id, ""),
            )
        )

    def gap(self, model: str, field: str | None, kind: str, note: str) -> None:
        self.gaps.append(
            SpecGap.model_validate({"model": model, "field": field, "kind": kind, "note": note})
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    collapsed = " ".join(text.split())
    if not collapsed:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(collapsed) if s.strip()]


def _is_not_sentence(sentence: str) -> bool:
    return bool(_NOT_RE.search(sentence)) or "n'est" in sentence.lower()


def _is_cardinality_sentence(sentence: str) -> bool:
    lower = sentence.lower()
    return any(ch.isdigit() for ch in sentence) or any(p in lower for p in _CARDINALITY_PHRASES)


def _is_forbidden_description_sentence(sentence: str) -> bool:
    lower = sentence.lower()
    return bool(
        _COMPUTATION_RE.search(sentence)
        or _NA_RE.search(sentence)
        or any(p in lower for p in _GLOBAL_RULE_PHRASES)
    )


def _canon_token(text: str) -> str:
    return _NON_ALNUM_RE.sub("", text.lower())


def _entity_like(spec: TemplateSpec) -> list[ModelSpec]:
    return [m for m in spec.models if m.kind in ("entity", "root")]


def _model_map(spec: TemplateSpec) -> dict[str, ModelSpec]:
    return {m.name: m for m in spec.models}


def _edge_items(spec: TemplateSpec) -> list[tuple[ModelSpec, FieldSpec]]:
    return [(m, f) for m in spec.models for f in m.fields if f.role == "edge"]


def _nesting_items(spec: TemplateSpec) -> list[tuple[ModelSpec, FieldSpec]]:
    """Model-typed fields that nest content: edge fields PLUS model-typed
    property fields — ``build_node_catalog`` and the converter walk both, so
    depth/cycle rules (R14/R15) must traverse both."""
    model_names = {m.name for m in spec.models}
    return [
        (m, f)
        for m in spec.models
        for f in m.fields
        if f.role == "edge" or (f.role == "property" and f.type in model_names)
    ]


def _nonref_inbound_counts(spec: TemplateSpec) -> Counter[str]:
    return Counter(f.type for _, f in _nesting_items(spec) if not f.reference)


def _is_identity_only(model: ModelSpec) -> bool:
    """Identity-only shared nodes (the Person pattern) may live on references alone."""
    return bool(model.identity_fields) and all(f.role == "identity" for f in model.fields)


def _edge_depths(spec: TemplateSpec, *, include_reference: bool) -> dict[str, int]:
    """BFS depth of each model from the root over the nesting digraph
    (edge fields plus model-typed property fields)."""
    adjacency: dict[str, set[str]] = {}
    for parent, field in _nesting_items(spec):
        if not include_reference and field.reference:
            continue
        adjacency.setdefault(parent.name, set()).add(field.type)
    depths: dict[str, int] = {spec.root: 0}
    queue: deque[str] = deque([spec.root])
    while queue:
        current = queue.popleft()
        for nxt in sorted(adjacency.get(current, ())):
            if nxt not in depths:
                depths[nxt] = depths[current] + 1
                queue.append(nxt)
    return depths


def _flip_to_reference(field: FieldSpec) -> None:
    """Flip a nesting field to a reference edge (R10/R14/R15 repairs).

    ``reference`` is an edge-only marker in the IR, so a model-typed property
    field is promoted to a proper edge first (label derived from the field
    name, normalizer cleared).
    """
    if field.role != "edge":
        field.role = "edge"
        field.edge_label = derive_edge_label(field.name, field.type)
        field.normalizer = "none"
    field.reference = True


def _unique_name(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    suffix = 2
    while f"{base}_{suffix}" in taken:
        suffix += 1
    return f"{base}_{suffix}"


def _unique_class_name(base: str, taken: set[str]) -> str:
    """Sanitize-stable unique class name.

    The generic ``_unique_name`` underscore suffix is NOT stable for class
    names — ``sanitize_class_name('Foo_2')`` folds it back to ``'Foo2'``,
    which would re-trigger a rename on every lint pass and exhaust the
    fixpoint. Collisions therefore get a bare numeric suffix, re-sanitized
    until the candidate is a fixed point of ``sanitize_class_name``.
    """
    candidate = base
    suffix = 2
    for _ in range(1000):
        stable = sanitize_class_name(candidate)
        if stable != candidate:
            candidate = stable
            continue
        if candidate not in taken:
            return candidate
        candidate = f"{base}{suffix}"
        suffix += 1
    raise TemplateLintError(  # pragma: no cover - sanitize + numeric suffix converges
        f"could not derive a sanitize-stable unique class name from {base!r}",
        report=LintReport(),
    )


def _dedupe_gaps(gaps: list[SpecGap]) -> list[SpecGap]:
    seen: set[tuple[str, str | None, str]] = set()
    unique: list[SpecGap] = []
    for gap in gaps:
        key = (gap.model, gap.field, gap.kind)
        if key not in seen:
            seen.add(key)
            unique.append(gap)
    return unique


# ---------------------------------------------------------------------------
# Rename cascades (R5/R20/R21 repairs must keep every reference resolving)
# ---------------------------------------------------------------------------


def _retype_follows_rename(old: str, role: str, has_edge_label: bool) -> bool:
    """Should a field typed ``old`` follow a class rename ``old -> new``?

    A rename key that shadows a scalar type name ('date', 'int', ...) only
    cascades into fields that are unambiguously edges: a bare 'date'-typed
    property/identity field means the SCALAR (the renderer resolves scalar
    annotations first), and blindly retyping it would corrupt every unrelated
    date field in the spec.
    """
    if old not in SCALAR_TYPES:
        return True
    return role == "edge" or has_edge_label


def _rename_model(spec: TemplateSpec, old: str, new: str) -> None:
    for model in spec.models:
        if model.name == old:
            model.name = new
        for field in model.fields:
            if field.type == old and _retype_follows_rename(
                old, field.role, field.edge_label is not None
            ):
                field.type = new
        if model.canonical_home:
            parent, _, field_name = model.canonical_home.partition(".")
            if parent == old:
                model.canonical_home = f"{new}.{field_name}"
    if spec.root == old:
        spec.root = new


def _rename_enum(spec: TemplateSpec, old: str, new: str) -> None:
    for enum in spec.enums:
        if enum.name == old:
            enum.name = new
    for model in spec.models:
        for field in model.fields:
            if field.type == old and old not in SCALAR_TYPES:
                field.type = new


def _rename_field(spec: TemplateSpec, model: ModelSpec, field: FieldSpec, new: str) -> None:
    old = field.name
    field.name = new
    model.identity_fields = [new if n == old else n for n in model.identity_fields]
    home = f"{model.name}.{old}"
    for other in spec.models:
        if other.canonical_home == home:
            other.canonical_home = f"{model.name}.{new}"
    if model.kind == "root":
        spec.needs_root_list_dedup = [new if n == old else n for n in spec.needs_root_list_dedup]


# ---------------------------------------------------------------------------
# Post-validation rules (each operates on a VALID TemplateSpec)
# ---------------------------------------------------------------------------


def _rule_r20_r21_naming(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R20/R21 — Python-safe, non-shadowing, non-reserved names (naming.py).

    R20: keywords/builtins/template-module names; R21: the reserved node-attr
    keys {id, label, type, __class__} that ``GraphConverter._create_nodes_pass``
    writes on every node. Repairs rename and cascade to ``identity_fields``,
    ``canonical_home``, field types, ``root`` and ``needs_root_list_dedup``.
    """
    taken = {m.name for m in spec.models} | {e.name for e in spec.enums}
    for model in list(spec.models):
        new = sanitize_class_name(model.name)
        if new == model.name:
            continue
        new = _unique_class_name(new, taken - {model.name})
        ctx.add(
            "R20",
            "warn",
            model.name,
            None,
            f"class name '{model.name}' renamed to '{new}' (keyword/builtin/reserved collision)",
            repaired=ctx.repair,
        )
        if ctx.repair:
            taken.discard(model.name)
            taken.add(new)
            _rename_model(spec, model.name, new)
    for enum in spec.enums:
        new = sanitize_class_name(enum.name)
        if new == enum.name:
            continue
        new = _unique_class_name(new, taken - {enum.name})
        ctx.add(
            "R20",
            "warn",
            enum.name,
            None,
            f"enum name '{enum.name}' renamed to '{new}' (keyword/builtin/reserved collision)",
            repaired=ctx.repair,
        )
        if ctx.repair:
            taken.discard(enum.name)
            taken.add(new)
            _rename_enum(spec, enum.name, new)
    for model in spec.models:
        field_names = {f.name for f in model.fields}
        for field in model.fields:
            new = sanitize_field_name(field.name)
            if new == field.name:
                continue
            rule = "R21" if to_snake_case(field.name) in RESERVED_FIELD_RENAMES else "R20"
            new = _unique_name(new, field_names - {field.name})
            reason = (
                "collides with a reserved node-attribute key"
                if rule == "R21"
                else "keyword/builtin/identifier collision"
            )
            ctx.add(
                rule,
                "warn",
                model.name,
                field.name,
                f"field '{field.name}' renamed to '{new}' ({reason})",
                repaired=ctx.repair,
            )
            if ctx.repair:
                field_names.discard(field.name)
                field_names.add(new)
                _rename_field(spec, model, field, new)


def _rule_r24_component_edges(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R24 — components own no labeled edges.

    The converter embeds components into their parent node, so a component
    field's ``edge_label``/``reference`` markers never materialize as the
    declared graph edge — dead metadata that V6 would rightly fail. Worse, an
    ENTITY nested under a component still becomes a node with an
    ancestor-derived edge that can clobber a sibling's declared label on the
    same node pair. Repair, by target kind:

    - entity/root target: the field is severed to a plain ``str`` holding the
      target's identity (the relationship a component cannot own);
    - component target: demoted to a nested property field (pure embedding —
      no nodes or edges involved).

    Common after a kind vote demotes an edge-owning entity to component.
    """
    models = _model_map(spec)
    for model in spec.models:
        if model.kind != "component":
            continue
        for field in model.fields:
            if field.role != "edge":
                continue
            target = models.get(field.type)
            to_entity = target is not None and target.kind != "component"
            ctx.add(
                "R24",
                "warn",
                model.name,
                field.name,
                f"edge on component '{model.name}' — components embed into their parent "
                "node and cannot own graph edges; "
                + (
                    f"field severed to str (the {field.type} identity)"
                    if to_entity
                    else "demoted to a nested property field"
                ),
                repaired=ctx.repair,
            )
            if not ctx.repair:
                continue
            field.role = "property"
            field.edge_label = None
            field.reference = False
            field.closed_catalog = False
            if to_entity:
                field.description = (
                    field.description or f"The identity of the related {field.type}."
                )
                field.type = "str"
                field.normalizer = "none"


def _rule_r9_edge_labels(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R9 — edge labels are ALL_CAPS verb phrases, consistent template-wide.

    Normalization delegates to :func:`naming.normalize_edge_label` (banned
    labels become HAS_<TARGET> and raise a ``missing_edge_label`` gap so
    gap-fill can propose a real verb phrase). Hard rewrites happen ONLY for
    the banned set, case/separator conversion, and label derivation on
    label-less edges — a multi-token label whose first token is not a known
    relationship verb (OWNS_VEHICLE with an unknown verb, MANAGED_BY, ...) is
    KEPT and reported as a warn-severity ADVISORY, never rewritten: user- and
    ontology-chosen verb phrases are the SPEC's escape hatch. Consistency is
    keyed on ``(field name, target model)`` across models — the same-named
    field to the same target must carry one label (first wins). It is
    deliberately NOT keyed on the bare (source, target) pair: multiple
    differently-labelled edges between one pair are the documented multi-role
    pattern (relationships.md "Multiple Edge Types to Same Entity";
    seller/buyer, INCLUDES/EXCLUDES) and must survive.
    """
    for model, field in _edge_items(spec):
        raw = field.edge_label or ""
        try:
            normalized = normalize_edge_label(raw, target=field.type)
        except ValueError:
            normalized = derive_edge_label(field.name, field.type)
        if to_snake_case(raw).upper() in BANNED_EDGE_LABELS:
            ctx.gap(
                model.name,
                field.name,
                "missing_edge_label",
                f"banned label '{raw}' rewritten to '{normalized}'; propose a verb phrase",
            )
        if normalized != raw:
            ctx.add(
                "R9",
                "warn",
                model.name,
                field.name,
                f"edge label '{raw}' normalized to '{normalized}'",
                repaired=ctx.repair,
            )
            if ctx.repair:
                field.edge_label = normalized
        if "_" in normalized and not is_verb_phrase(normalized):
            ctx.add(
                "R9",
                "warn",
                model.name,
                field.name,
                f"edge label '{normalized}' does not start with a known relationship verb — "
                "kept as-is (advisory); verify it reads as a verb phrase",
                repaired=False,
            )
    chosen: dict[tuple[str, str], str] = {}
    for model, field in _edge_items(spec):
        key = (field.name, field.type)
        label = field.edge_label or ""
        first = chosen.setdefault(key, label)
        if label != first:
            ctx.add(
                "R9",
                "warn",
                model.name,
                field.name,
                f"inconsistent label '{label}' for field '{field.name}' -> {field.type}; "
                f"rewritten to '{first}' (first occurrence wins)",
                repaired=ctx.repair,
            )
            if ctx.repair:
                field.edge_label = first


def _rule_r10_canonical_home(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R10 — a rich entity is nested in full at exactly ONE canonical home.

    Entities with >=2 non-reference inbound edges keep the already-marked
    ``canonical_home`` (else the inbound edge whose parent is nearest the root
    by BFS depth) and every other inbound edge flips to ``reference=True``.
    Exception (billing seller/buyer Party shape), keyed purely on SHAPE: when
    EVERY inbound non-reference edge is a single (non-list) edge from the
    SAME parent model, all of them stay full regardless of the target's field
    count — flipping one would silently drop that role's data.
    """
    depths = _edge_depths(spec, include_reference=True)
    model_index = {m.name: i for i, m in enumerate(spec.models)}
    for target in spec.models:
        if target.kind != "entity":
            continue
        inbound = [
            (parent, field, field_index)
            for parent in spec.models
            if parent.name != target.name  # self-references are R15's forward-ref case
            for field_index, field in enumerate(parent.fields)
            if field.role == "edge" and field.type == target.name and not field.reference
        ]
        if len(inbound) < 2:
            continue
        parent_names = {parent.name for parent, _, _ in inbound}
        if len(parent_names) == 1 and all(not field.is_list for _, field, _ in inbound):
            continue
        canonical: tuple[ModelSpec, FieldSpec, int] | None = None
        if target.canonical_home:
            home_parent, _, home_field = target.canonical_home.partition(".")
            canonical = next(
                (t for t in inbound if t[0].name == home_parent and t[1].name == home_field),
                None,
            )
        if canonical is None:
            canonical = min(
                inbound,
                key=lambda t: (
                    depths.get(t[0].name, math.inf),
                    model_index[t[0].name],
                    t[2],
                ),
            )
            if ctx.repair and target.canonical_home is None:
                target.canonical_home = f"{canonical[0].name}.{canonical[1].name}"
        for parent, field, _ in inbound:
            if field is canonical[1]:
                continue
            ctx.add(
                "R10",
                "warn",
                parent.name,
                field.name,
                f"'{target.name}' is nested in full at several paths; canonical home is "
                f"'{canonical[0].name}.{canonical[1].name}' — this edge flips to reference=True",
                repaired=ctx.repair,
            )
            if ctx.repair:
                field.reference = True


def _rule_r12_closed_catalog(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R12 — ``closed_catalog`` only where a non-reference canonical home exists.

    A closed catalog's members must be extractable at the target's canonical
    home; without any non-reference inbound edge the converter would refuse
    class-wide enforcement anyway, so the marker is cleared (reference kept).
    """
    models = _model_map(spec)
    nonref = _nonref_inbound_counts(spec)
    for model, field in _edge_items(spec):
        if not field.closed_catalog:
            continue
        target = models[field.type]
        if target.kind == "root" or nonref.get(target.name, 0) > 0:
            continue
        ctx.add(
            "R12",
            "warn",
            model.name,
            field.name,
            f"closed_catalog on '{field.name}' but '{target.name}' has no non-reference "
            "canonical home elsewhere — marker cleared",
            repaired=ctx.repair,
        )
        if ctx.repair:
            field.closed_catalog = False


def _rule_r11_reference_targets(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R11 — references need identity targets and a canonical home elsewhere.

    A reference to an identity-less target (component) is ignored at runtime
    and is un-referenced/inlined. A reference to a non-root target that has
    non-identity data but NO non-reference inbound edge would leave the node
    with nothing but its name — flipped off. Identity-only shared nodes (the
    Person pattern) are exempt: references on every path are their design.
    """
    models = _model_map(spec)
    nonref = _nonref_inbound_counts(spec)
    for model, field in _edge_items(spec):
        if not field.reference:
            continue
        target = models[field.type]
        if not target.identity_fields:
            ctx.add(
                "R11",
                "warn",
                model.name,
                field.name,
                f"reference to '{target.name}' which declares no graph_id_fields — the marker "
                "is ignored at runtime; edge un-referenced and inlined",
                repaired=ctx.repair,
            )
            if ctx.repair:
                field.reference = False
                field.closed_catalog = False
                nonref[target.name] += 1
            continue
        if target.kind == "root" or _is_identity_only(target):
            continue
        if nonref.get(target.name, 0) == 0:
            ctx.add(
                "R11",
                "warn",
                model.name,
                field.name,
                f"reference on the only full path to '{target.name}' — the node would end up "
                "with nothing but its identity; reference flipped off",
                repaired=ctx.repair,
            )
            if ctx.repair:
                field.reference = False
                field.closed_catalog = False
                nonref[target.name] += 1


def _rule_r23_reachability(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R23 — every entity keeps a discovery path from the root.

    The dense catalog (verification gate V4) discovers entities only through
    non-reference nesting chains from the root, so an entity outside every
    such chain can never be extracted. Induced drafts arrive with islands
    when a model's relationships pass simply never connects a subtree to the
    root (small models under truncation pressure) and the cross-document
    merge then unions each document's clusters verbatim. Repair: attach each
    island's head — an unreachable entity that no other unreachable model
    nests — to the root as a full list edge. One invented connection per
    island, the island's inner structure untouched; a gap is raised so
    gap-fill or the user refines the boilerplate description.
    """
    root = next((m for m in spec.models if m.kind == "root"), None)
    if root is None:
        return
    for _ in range(len(spec.models)):
        reachable = set(_edge_depths(spec, include_reference=False))
        unreachable = [m for m in spec.models if m.kind == "entity" and m.name not in reachable]
        if not unreachable:
            return
        unreachable_names = {m.name for m in unreachable}
        nested_within_island = {
            field.type
            for model in unreachable
            for field in model.fields
            if field.type in unreachable_names
            and not field.reference
            and (field.role == "edge" or field.role == "property")
        }
        heads = [m for m in unreachable if m.name not in nested_within_island] or unreachable[:1]
        for head in heads:
            taken = {f.name for f in root.fields}
            field_name = _unique_name(to_snake_case(head.name), taken)
            ctx.add(
                "R23",
                "warn",
                root.name,
                field_name,
                f"entity '{head.name}' has no discovery path from the root "
                f"(island of {len(unreachable)} unreachable class(es)) — attached to the "
                "root as a full list edge so dense discovery can reach it",
                repaired=ctx.repair,
            )
            if not ctx.repair:
                continue
            root.fields.append(
                FieldSpec(
                    name=field_name,
                    type=head.name,
                    is_list=True,
                    role="edge",
                    edge_label=derive_edge_label(field_name, head.name),
                    description=f"Every {head.name} the document describes.",
                )
            )
            ctx.gap(
                root.name,
                field_name,
                "missing_description",
                f"edge attached by the linter (R23): no document evidence connected "
                f"'{head.name}' to the root — refine the locator description or re-home "
                "the edge via the SPEC YAML",
            )
        if not ctx.repair:
            # Report-only: reachability cannot change, so one pass of findings
            # (the current island heads) is all there is to report.
            return


def _rule_r14_nesting_depth(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R14 — full-nesting depth from the root stays within 2-4 levels.

    A nesting field (edge OR model-typed property — ``build_node_catalog``
    walks both) whose parent already sits at depth >=4 nests its target
    beyond the budget. The field is flipped to reference only when that is
    safe: the target declares identity AND keeps another full home (or is
    identity-only); property fields are promoted to edges in the flip.
    Otherwise the chain is flagged for restructuring — flipping the only full
    path would orphan the target's content. In practice R10 (canonical home)
    resolves most deep multi-path entities before this rule sees them.
    """
    depths = _edge_depths(spec, include_reference=False)
    models = _model_map(spec)
    nonref = _nonref_inbound_counts(spec)
    for parent, field in _nesting_items(spec):
        if field.reference:
            continue
        parent_depth = depths.get(parent.name)
        if parent_depth is None or parent_depth < MAX_NESTING_DEPTH:
            continue
        target = models[field.type]
        can_flip = bool(target.identity_fields) and (
            _is_identity_only(target) or nonref[target.name] >= 2
        )
        if can_flip:
            ctx.add(
                "R14",
                "warn",
                parent.name,
                field.name,
                f"'{target.name}' nested at depth {parent_depth + 1} (> {MAX_NESTING_DEPTH}); "
                "edge flipped to reference=True",
                repaired=ctx.repair,
            )
            if ctx.repair:
                _flip_to_reference(field)
                nonref[target.name] -= 1
        else:
            ctx.add(
                "R14",
                "warn",
                parent.name,
                field.name,
                f"'{target.name}' nested at depth {parent_depth + 1} (> {MAX_NESTING_DEPTH}) "
                "with no safe deterministic repair (only full path or identity-less target) — "
                "restructure or reference a shallower home",
                repaired=False,
            )


def _rule_r15_cycles(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R15 — self-references and mutual cycles.

    Traverses every nesting field — edges AND model-typed property fields,
    exactly what ``build_node_catalog`` walks. Self-loops are recorded for
    the renderer (string forward reference + ``model_rebuild()``); mutual
    A<->B full-nesting cycles get their back edge (from the deeper model to
    the shallower, ties broken by declaration order) flipped to
    ``reference=True`` when that is safe — property back-fields are promoted
    to edges in the flip. Reachable mutual cycles are usually already broken
    by R10.
    """
    for parent, field in _nesting_items(spec):
        if field.type == parent.name:
            ctx.add(
                "R15",
                "info",
                parent.name,
                field.name,
                "self-referencing field: renderer emits a string forward reference and "
                "model_rebuild()",
                repaired=False,
            )
    depths = _edge_depths(spec, include_reference=False)
    nonref = _nonref_inbound_counts(spec)
    model_names = {m.name for m in spec.models}

    def nests(field: FieldSpec, target_name: str) -> bool:
        if field.type != target_name or field.reference:
            return False
        return field.role == "edge" or (field.role == "property" and field.type in model_names)

    for i, model_a in enumerate(spec.models):
        for model_b in spec.models[i + 1 :]:
            a_to_b = [f for f in model_a.fields if nests(f, model_b.name)]
            b_to_a = [f for f in model_b.fields if nests(f, model_a.name)]
            if not a_to_b or not b_to_a:
                continue
            depth_a = depths.get(model_a.name, math.inf)
            depth_b = depths.get(model_b.name, math.inf)
            if depth_a <= depth_b:
                back_model, back_fields, target = model_b, b_to_a, model_a
            else:
                back_model, back_fields, target = model_a, a_to_b, model_b
            for field in back_fields:
                if back_model.kind == "component":
                    # A reference on a component is dead metadata (R24 would
                    # demote it right back to full nesting — a repair
                    # ping-pong). Sever the cycle instead: the back-field
                    # becomes a plain string holding the target's identity.
                    ctx.add(
                        "R15",
                        "warn",
                        back_model.name,
                        field.name,
                        f"mutual nesting cycle {back_model.name} <-> {target.name} closed by "
                        "a component-owned field — components cannot hold reference edges "
                        f"(R24), so the field is retyped to str (the {target.name} identity)",
                        repaired=ctx.repair,
                    )
                    if ctx.repair:
                        field.role = "property"
                        field.type = "str"
                        field.edge_label = None
                        field.reference = False
                        field.closed_catalog = False
                        field.normalizer = "none"
                        field.description = (
                            field.description
                            or f"The identity of the {target.name} this belongs to."
                        )
                    continue
                others = nonref[target.name] - sum(1 for f in back_fields if not f.reference)
                can_flip = bool(target.identity_fields) and (
                    target.kind == "root" or _is_identity_only(target) or others >= 1
                )
                if can_flip:
                    ctx.add(
                        "R15",
                        "warn",
                        back_model.name,
                        field.name,
                        f"mutual nesting cycle {back_model.name} <-> {target.name}; back edge "
                        "flipped to reference=True",
                        repaired=ctx.repair,
                    )
                    if ctx.repair:
                        _flip_to_reference(field)
                        nonref[target.name] -= 1
                else:
                    ctx.add(
                        "R15",
                        "warn",
                        back_model.name,
                        field.name,
                        f"mutual nesting cycle {back_model.name} <-> {target.name} with no safe "
                        "back-edge repair (identity-less or home-less target) — restructure",
                        repaired=False,
                    )


def _rule_r5_digit_honesty(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R5 — digit-honest identity names (*_number/*_no/ref_* must hold digits).

    A number-named identity whose surviving examples are all digit-free is
    renamed to ``name``/``title`` (the pipeline invariant clears prose from
    number-named root ids at extraction time; fix it at generation time).
    The reverse case (digit-only values under ``name``/``title``) is flagged
    only — renaming toward a number name is a semantic judgment.
    """
    for model in _entity_like(spec):
        fields_by_name = {f.name: f for f in model.fields}
        for id_name in list(model.identity_fields):
            field = fields_by_name[id_name]
            if not field.examples:
                continue
            digit_bearing = [ex for ex in field.examples if any(c.isdigit() for c in ex)]
            if _NUMBER_NAME_RE.search(field.name):
                if digit_bearing:
                    continue
                taken = {f.name for f in model.fields}
                new_name = next((c for c in _IDENTITY_RENAME_CANDIDATES if c not in taken), None)
                if new_name is None:
                    ctx.add(
                        "R5",
                        "warn",
                        model.name,
                        field.name,
                        "number-named identity holds digit-free values but 'name'/'title' are "
                        "taken — rename manually",
                        repaired=False,
                    )
                    continue
                ctx.add(
                    "R5",
                    "warn",
                    model.name,
                    field.name,
                    f"number-named identity field holds digit-free values — renamed to "
                    f"'{new_name}'",
                    repaired=ctx.repair,
                )
                if ctx.repair:
                    _rename_field(spec, model, field, new_name)
            elif field.name in _IDENTITY_RENAME_CANDIDATES and len(digit_bearing) == len(
                field.examples
            ):
                ctx.add(
                    "R5",
                    "info",
                    model.name,
                    field.name,
                    "identity named 'name'/'title' holds digit-bearing values only — consider "
                    "a *_number name (not auto-renamed)",
                    repaired=False,
                )


def _rule_r6_invented_ids(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R6 — no invented/positional ids (advisory WARN, never an error gate).

    Identity examples matching ``^\\w+[ _-]?\\d+$`` whose alphabetic prefix
    equals the class name (case/space/underscore-insensitively) are stripped;
    identity descriptions instructing generate/assign/invent lose the
    offending sentence. This is best-effort surface detection — the verbatim
    evidence gate in the documents path is the real anti-invention mechanism.
    """
    for model in _entity_like(spec):
        model_canon = _canon_token(model.name)
        fields_by_name = {f.name: f for f in model.fields}
        for id_name in model.identity_fields:
            field = fields_by_name[id_name]
            kept_examples: list[str] = []
            stripped: list[str] = []
            for example in field.examples:
                match = _POSITIONAL_ID_RE.fullmatch(example.strip())
                if match and _canon_token(match.group(1)) == model_canon:
                    stripped.append(example)
                else:
                    kept_examples.append(example)
            if stripped:
                ctx.add(
                    "R6",
                    "warn",
                    model.name,
                    field.name,
                    f"positional/invented-looking identity examples stripped: {stripped} "
                    "(advisory; the verbatim gate is the real anti-invention mechanism)",
                    repaired=ctx.repair,
                )
                if ctx.repair:
                    field.examples = kept_examples
            sentences = _sentences(field.description)
            kept_sentences = [s for s in sentences if not _INVENTION_RE.search(s)]
            if len(kept_sentences) != len(sentences):
                ctx.add(
                    "R6",
                    "warn",
                    model.name,
                    field.name,
                    "identity description instructs id generation/assignment — sentence deleted",
                    repaired=ctx.repair,
                )
                if ctx.repair:
                    field.description = " ".join(kept_sentences)


def _rule_r13_cardinality(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R13 — ``max_instances`` requires a cardinality sentence in the docstring.

    The spec stores the already-doubled bound (2x documented maximum; the
    doubling happens once in ``repair_draft``), so the injected sentence
    reports ``max_instances // 2`` — the documented figure the model should
    respect at discovery time.
    """
    for model in _entity_like(spec):
        if model.max_instances is None:
            continue
        if any(_is_cardinality_sentence(s) for s in _sentences(model.docstring)):
            continue
        documented = max(1, model.max_instances // 2)
        sentence = f"At most {documented} expected per document."
        ctx.add(
            "R13",
            "warn",
            model.name,
            None,
            f"max_instances={model.max_instances} without a cardinality sentence in the "
            f"docstring — injected: '{sentence}'",
            repaired=ctx.repair,
        )
        if ctx.repair:
            model.docstring = f"{model.docstring.rstrip()} {sentence}"


def _rule_r16_description_scrub(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R16 — descriptions never instruct computation or restate global rules.

    Sentence-level scrub for computation verbs (calculate/compute/sum/convert/
    round/multiply/derive) and global-rule restatements ("N/A", "omit if",
    "leave empty"). A description emptied by the scrub raises a
    ``missing_description`` gap.
    """
    for model in spec.models:
        for field in model.fields:
            sentences = _sentences(field.description)
            if not sentences:
                continue
            kept = [s for s in sentences if not _is_forbidden_description_sentence(s)]
            if len(kept) == len(sentences):
                continue
            removed = [s for s in sentences if _is_forbidden_description_sentence(s)]
            ctx.add(
                "R16",
                "warn",
                model.name,
                field.name,
                f"description instructs computation or restates global prompt rules — "
                f"removed: {removed}",
                repaired=ctx.repair,
            )
            if ctx.repair:
                field.description = " ".join(kept)
                if not kept:
                    ctx.gap(
                        model.name,
                        field.name,
                        "missing_description",
                        "description emptied by the R16 computation/global-rule scrub",
                    )


def _rule_r4_docstring_budget(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R4 — discriminating docstring content fits the 240-char Phase-1 window.

    Docstrings over budget are reordered IS -> IS-NOT -> cardinality
    (heuristics: a sentence containing 'not'/"n'est" is IS-NOT; one containing
    digits or 'at most'/'maximum'/'up to' is cardinality) and the exact
    240-char window Phase 1 sees is reported.
    """
    for model in spec.models:
        collapsed = " ".join(model.docstring.split())
        if len(collapsed) <= DOCSTRING_WINDOW:
            continue
        sentences = _sentences(model.docstring)
        is_bucket = [
            s for s in sentences if not _is_not_sentence(s) and not _is_cardinality_sentence(s)
        ]
        not_bucket = [s for s in sentences if _is_not_sentence(s)]
        cardinality_bucket = [
            s for s in sentences if not _is_not_sentence(s) and _is_cardinality_sentence(s)
        ]
        reordered = " ".join(is_bucket + not_bucket + cardinality_bucket)
        if reordered != collapsed:
            ctx.add(
                "R4",
                "warn",
                model.name,
                None,
                "docstring overruns the 240-char Phase-1 window; sentences reordered "
                "IS -> IS-NOT -> cardinality",
                repaired=ctx.repair,
            )
            if ctx.repair:
                model.docstring = reordered
        window = (reordered if ctx.repair else collapsed)[:DOCSTRING_WINDOW]
        ctx.add(
            "R4",
            "info",
            model.name,
            None,
            f"Phase-1 240-char docstring window: {window!r}",
            repaired=False,
        )


def _rule_r19_root_dedup(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R19 — identity-less root lists get the dedup ``model_validator``.

    Root list fields of scalars, enums, or components (nothing with identity
    to dedup by) are added to ``needs_root_list_dedup`` so the renderer emits
    the normalized-key first-wins dedup validator. Entity lists dedup by
    identity in the registry and are excluded.
    """
    root = next(m for m in spec.models if m.kind == "root")
    models = _model_map(spec)
    enum_names = {e.name for e in spec.enums}
    for field in root.fields:
        if not field.is_list or field.name in spec.needs_root_list_dedup:
            continue
        target = models.get(field.type)
        dedupable = (
            field.type in SCALAR_TYPES
            or field.type in enum_names
            or (target is not None and target.kind == "component")
        )
        if not dedupable:
            continue
        ctx.add(
            "R19",
            "warn",
            root.name,
            field.name,
            "identity-less root list field — dedup model_validator scheduled "
            "(needs_root_list_dedup)",
            repaired=ctx.repair,
        )
        if ctx.repair:
            spec.needs_root_list_dedup.append(field.name)


def _rule_r3_identity_examples(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R3 — identity fields carry 2-5 verbatim examples (the cap is IR-enforced).

    Fewer than 2 surviving examples raises a ``missing_examples`` gap for
    gap-fill / the user; there is no deterministic repair (examples must be
    document-derived, never invented).
    """
    for model in _entity_like(spec):
        fields_by_name = {f.name: f for f in model.fields}
        for id_name in model.identity_fields:
            field = fields_by_name[id_name]
            if len(field.examples) >= 2:
                continue
            ctx.add(
                "R3",
                "warn",
                model.name,
                field.name,
                f"identity field has {len(field.examples)} example(s); 2-5 short verbatim "
                "examples are the only id guidance dense Phase 1 sees",
                repaired=False,
            )
            ctx.gap(
                model.name,
                field.name,
                "missing_examples",
                "identity needs 2-5 verbatim document-derived examples",
            )


def _rule_r18_role_data(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R18 — no role/context data on multi-path shared entities (advisory).

    Duplicate-instance merge is fill-missing-only (first non-empty wins), so
    role fields on a shared entity resolve arbitrarily. The per-role wrapper
    pattern (BoardMember-style) is proposed in the report; applying it is a
    semantic judgment and never automated.
    """
    inbound_all = Counter(f.type for _, f in _edge_items(spec))
    for target in spec.models:
        if target.kind != "entity" or inbound_all[target.name] < 2:
            continue
        role_fields = [
            f.name
            for f in target.fields
            if f.role == "property" and f.name in _ROLE_DATA_FIELD_NAMES
        ]
        for name in role_fields:
            ctx.add(
                "R18",
                "warn",
                target.name,
                name,
                f"role/context field '{name}' on an entity referenced from "
                f"{inbound_all[target.name]} paths — duplicate merges keep first-seen values; "
                "consider the per-role wrapper pattern (BoardMember-style)",
                repaired=False,
            )


def _rule_r22_evidence(spec: TemplateSpec, ctx: _Ctx) -> None:
    """R22 — docs-path fields carry evidence or the "Rare:" flag (report-only).

    Applies to ``provenance="induced"`` models only (ontologies have no
    instance evidence by nature). Verbatim examples count as evidence.
    """
    for model in spec.models:
        if model.provenance != "induced":
            continue
        for field in model.fields:
            if field.evidence or field.examples:
                continue
            if field.description.startswith("Rare:"):
                continue
            ctx.add(
                "R22",
                "info",
                model.name,
                field.name,
                "induced field with no evidence quote, no examples and no 'Rare:' flag — "
                "candidate unmodeled-noise field; prune via the SPEC YAML if spurious",
                repaired=False,
            )


_RULES: tuple[Callable[[TemplateSpec, _Ctx], None], ...] = (
    _rule_r20_r21_naming,
    _rule_r24_component_edges,
    _rule_r9_edge_labels,
    _rule_r10_canonical_home,
    _rule_r12_closed_catalog,
    _rule_r11_reference_targets,
    _rule_r23_reachability,
    _rule_r14_nesting_depth,
    _rule_r15_cycles,
    _rule_r5_digit_honesty,
    _rule_r6_invented_ids,
    _rule_r13_cardinality,
    _rule_r16_description_scrub,
    _rule_r4_docstring_budget,
    _rule_r19_root_dedup,
    _rule_r3_identity_examples,
    _rule_r18_role_data,
    _rule_r22_evidence,
)


def _run_rules(spec: TemplateSpec, ctx: _Ctx) -> None:
    for rule in _RULES:
        rule(spec, ctx)


def _strict_violations(report: LintReport, *, repair: bool) -> list[LintEntry]:
    if repair:
        return report.repaired_entries
    return [e for e in report.entries if e.severity != "info"]


def _raise_strict(
    violations: list[LintEntry], report: LintReport, spec: TemplateSpec | None
) -> None:
    lines = [
        f"[{e.rule_id}] {e.model}" + (f".{e.field}" if e.field else "") + f": {e.message}"
        for e in violations
    ]
    raise TemplateLintError(
        f"Template lint (strict): {len(violations)} violation(s) require repair:\n"
        + "\n".join(lines),
        report=report,
        spec=spec,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def lint_spec(
    spec: TemplateSpec, *, repair: bool = True, strict: bool = False
) -> tuple[TemplateSpec, LintReport]:
    """Run the full rulebook over an already-valid spec.

    The input spec is never mutated. With ``repair=True`` (default) the rules
    run in dependency order to a fixpoint (max :data:`MAX_LINT_PASSES` passes;
    repairs are monotone so this terminates) and the repaired copy is
    returned. With ``repair=False`` the rules are report-only and the input
    spec is returned unchanged (the ``template lint`` command's mode).

    With ``strict=True`` any performed repair (or, in report-only mode, any
    non-info finding) raises :class:`TemplateLintError` listing every
    violation; the report (and repaired spec, when produced) stay attached to
    the exception.
    """
    working = spec.model_copy(deep=True)
    if not repair:
        ctx = _Ctx(repair=False)
        _run_rules(working, ctx)
        report = LintReport(entries=ctx.entries, gaps=_dedupe_gaps(ctx.gaps))
        if strict:
            violations = _strict_violations(report, repair=False)
            if violations:
                _raise_strict(violations, report, None)
        return spec, report

    entries: list[LintEntry] = []
    gaps: list[SpecGap] = []
    for _ in range(MAX_LINT_PASSES):
        ctx = _Ctx(repair=True)
        _run_rules(working, ctx)
        gaps.extend(ctx.gaps)
        repaired = [e for e in ctx.entries if e.repaired]
        if not repaired:
            entries.extend(ctx.entries)
            break
        entries.extend(repaired)
    else:
        # Repairs are monotone, so this indicates a linter bug — but it must
        # surface as the library's own error (with the collected report), not
        # a raw AssertionError traceback.
        raise TemplateLintError(
            f"Template lint failed to reach a fixpoint after {MAX_LINT_PASSES} passes; "
            "a repair is not monotone. This is a docling-graph bug — please report it "
            "with the offending spec.",
            report=LintReport(entries=entries, gaps=_dedupe_gaps(gaps)),
        )
    result = TemplateSpec.model_validate(working.model_dump())
    report = LintReport(entries=entries, gaps=_dedupe_gaps(gaps))
    if strict:
        violations = _strict_violations(report, repair=True)
        if violations:
            _raise_strict(violations, report, result)
    return result, report


def repair_draft(draft: dict[str, Any], *, strict: bool = False) -> tuple[TemplateSpec, LintReport]:
    """Repair a loose TemplateSpec-shaped dict into a valid, linted spec.

    This is the producer entry point (ontology compilers, document induction):
    the IR *rejects* invalid constructions instead of repairing them, so the
    pre-validation repairs here (R1 demotions / identity trimming /
    root-identity synthesis, R2 retyping, name sanitation, edge-label
    derivation, ``max_instances`` doubling) make the draft constructible
    before ``model_validate``; the full post-validation rule set then runs via
    :func:`lint_spec`.

    Contract: ``draft`` carries the *documented* ``max_instances``; it is
    doubled exactly once here (reported as a non-repair info entry). The input
    dict is never mutated. A draft that stays unconstructible after the
    pre-validation repairs raises ``pydantic.ValidationError``.

    With ``strict=True`` any repair (pre- or post-validation) raises
    :class:`TemplateLintError` listing all violations, report attached.
    """
    data = copy.deepcopy(draft)
    ctx = _Ctx(repair=True)
    _predraft_repair(data, ctx)
    spec = TemplateSpec.model_validate(data)
    linted, post_report = lint_spec(spec, repair=True, strict=False)
    report = LintReport(
        entries=ctx.entries + post_report.entries,
        gaps=_dedupe_gaps(ctx.gaps + post_report.gaps),
    )
    if strict:
        violations = _strict_violations(report, repair=True)
        if violations:
            _raise_strict(violations, report, linted)
    return linted, report


# ---------------------------------------------------------------------------
# Pre-validation draft repairs (dict-level; make the draft constructible)
# ---------------------------------------------------------------------------


def _model_dicts(data: dict[str, Any]) -> list[dict[str, Any]]:
    models = data.get("models")
    if not isinstance(models, list):
        return []
    return [m for m in models if isinstance(m, dict)]


def _enum_dicts(data: dict[str, Any]) -> list[dict[str, Any]]:
    enums = data.get("enums")
    if not isinstance(enums, list):
        return []
    return [e for e in enums if isinstance(e, dict)]


def _field_dicts(model: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(model.get("fields"), list):
        model["fields"] = []
    return [f for f in model["fields"] if isinstance(f, dict)]


def _predraft_repair(data: dict[str, Any], ctx: _Ctx) -> None:
    if not _model_dicts(data):
        return  # nothing repairable; let model_validate raise
    _predraft_names(data, ctx)
    _predraft_root(data, ctx)
    _predraft_enums(data, ctx)
    model_names = {m.get("name") for m in _model_dicts(data)}
    enum_names = {e.get("name") for e in _enum_dicts(data)}
    known_types = set(SCALAR_TYPES) | {n for n in model_names | enum_names if isinstance(n, str)}
    for model in _model_dicts(data):
        _predraft_fields(model, known_types, model_names, ctx)
    for model in _model_dicts(data):
        _predraft_identity(model, ctx)
    _predraft_spec_level(data, ctx)


def _predraft_names(data: dict[str, Any], ctx: _Ctx) -> None:
    """Sanitize class/enum/field names (R20/R21) and cascade every reference.

    Scalar-shadowing class names ('date', 'int', ...) are force-suffixed by
    ``sanitize_class_name`` BEFORE the cascade; the type cascade then skips
    non-edge fields for such rename keys — a bare 'date'-typed property field
    means the scalar, and retyping it would corrupt every unrelated date
    field in the draft.
    """
    class_renames: dict[str, str] = {}
    taken: set[str] = set()
    for entry in _model_dicts(data) + _enum_dicts(data):
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        new = _unique_class_name(sanitize_class_name(name), taken)
        if new != name:
            class_renames[name] = new
            entry["name"] = new
            ctx.add(
                "R20",
                "warn",
                name,
                None,
                f"class name '{name}' renamed to '{new}' (sanitized/deduplicated)",
                repaired=True,
            )
        taken.add(new)
    for model in _model_dicts(data):
        for field in _field_dicts(model):
            old_type = field.get("type")
            if (
                isinstance(old_type, str)
                and old_type in class_renames
                and _retype_follows_rename(
                    old_type,
                    str(field.get("role", "property")),
                    isinstance(field.get("edge_label"), str),
                )
            ):
                field["type"] = class_renames[old_type]
        home = model.get("canonical_home")
        if isinstance(home, str) and "." in home:
            parent, _, field_name = home.partition(".")
            if parent in class_renames:
                model["canonical_home"] = f"{class_renames[parent]}.{field_name}"
    if data.get("root") in class_renames:
        data["root"] = class_renames[data["root"]]

    for model in _model_dicts(data):
        model_name = model.get("name", "?")
        field_renames: dict[str, str] = {}
        field_taken: set[str] = set()
        for field in _field_dicts(model):
            name = field.get("name")
            if not isinstance(name, str) or not name:
                continue
            new = _unique_name(sanitize_field_name(name), field_taken)
            if new != name:
                field_renames[name] = new
                field["name"] = new
                rule = "R21" if to_snake_case(name) in RESERVED_FIELD_RENAMES else "R20"
                ctx.add(
                    rule,
                    "warn",
                    model_name,
                    name,
                    f"field '{name}' renamed to '{new}' (sanitized/deduplicated)",
                    repaired=True,
                )
            field_taken.add(new)
        if not field_renames:
            continue
        identity = model.get("identity_fields")
        if isinstance(identity, list):
            model["identity_fields"] = [field_renames.get(n, n) for n in identity]
        for other in _model_dicts(data):
            home = other.get("canonical_home")
            if isinstance(home, str) and "." in home:
                parent, _, field_name = home.partition(".")
                if parent == model_name and field_name in field_renames:
                    other["canonical_home"] = f"{parent}.{field_renames[field_name]}"
        if model.get("name") == data.get("root"):
            dedup = data.get("needs_root_list_dedup")
            if isinstance(dedup, list):
                data["needs_root_list_dedup"] = [field_renames.get(n, n) for n in dedup]


def _predraft_root(data: dict[str, Any], ctx: _Ctx) -> None:
    """Reconcile the root pointer with exactly one kind='root' model."""
    models = _model_dicts(data)
    root_name = data.get("root")
    roots = [m for m in models if m.get("kind") == "root"]
    if not roots and isinstance(root_name, str):
        match = next((m for m in models if m.get("name") == root_name), None)
        if match is not None:
            old_kind = match.get("kind", "unset")
            match["kind"] = "root"
            ctx.add(
                "IR",
                "warn",
                root_name,
                None,
                f"model '{root_name}' named by root but kind was '{old_kind}' — set to 'root'",
                repaired=True,
            )
    elif len(roots) > 1:
        keep = next((m for m in roots if m.get("name") == root_name), roots[0])
        for model in roots:
            if model is keep:
                continue
            model["kind"] = "entity"
            ctx.add(
                "IR",
                "warn",
                str(model.get("name")),
                None,
                "multiple root models; demoted to entity (exactly one root allowed)",
                repaired=True,
            )
        if root_name != keep.get("name"):
            data["root"] = keep.get("name")
    elif len(roots) == 1 and root_name != roots[0].get("name"):
        data["root"] = roots[0].get("name")
        ctx.add(
            "IR",
            "warn",
            str(roots[0].get("name")),
            None,
            f"root pointer corrected to the root model '{roots[0].get('name')}'",
            repaired=True,
        )


def _predraft_enums(data: dict[str, Any], ctx: _Ctx) -> None:
    """Drop empty enums (retyping their fields to str) and unknown synonyms."""
    kept: list[dict[str, Any]] = []
    dropped_names: set[str] = set()
    for enum in _enum_dicts(data):
        members = enum.get("members")
        if not isinstance(members, list) or not members:
            name = str(enum.get("name"))
            dropped_names.add(name)
            ctx.add(
                "R17",
                "warn",
                name,
                None,
                "enum with no members dropped; fields retyped to str",
                repaired=True,
            )
            continue
        synonyms = enum.get("synonyms")
        if isinstance(synonyms, dict):
            unknown = set(synonyms) - set(members)
            if unknown:
                enum["synonyms"] = {k: v for k, v in synonyms.items() if k not in unknown}
                ctx.add(
                    "R17",
                    "warn",
                    str(enum.get("name")),
                    None,
                    f"synonyms for unknown members dropped: {sorted(unknown)}",
                    repaired=True,
                )
        kept.append(enum)
    if dropped_names:
        data["enums"] = kept
        for model in _model_dicts(data):
            for field in _field_dicts(model):
                if field.get("type") in dropped_names:
                    field["type"] = "str"


def _predraft_fields(
    model: dict[str, Any], known_types: set[str], model_names: set[Any], ctx: _Ctx
) -> None:
    """Per-field constructibility: types, examples, edge markers, labels."""
    model_name = str(model.get("name", "?"))
    for field in _field_dicts(model):
        field_name = str(field.get("name", "?"))
        examples = field.get("examples")
        if isinstance(examples, str):
            field["examples"] = [examples]
        elif isinstance(examples, list) and len(examples) > MAX_FIELD_EXAMPLES:
            field["examples"] = examples[:MAX_FIELD_EXAMPLES]
            ctx.add(
                "IR",
                "warn",
                model_name,
                field_name,
                f"examples truncated to {MAX_FIELD_EXAMPLES}",
                repaired=True,
            )
        elif examples is not None and not isinstance(examples, list):
            field["examples"] = []
        field_type = field.get("type", "str")
        if not isinstance(field_type, str) or field_type not in known_types:
            ctx.add(
                "IR",
                "warn",
                model_name,
                field_name,
                f"unresolved type {field_type!r} — retyped to 'str'",
                repaired=True,
            )
            field["type"] = "str"
            field_type = "str"
        role = field.get("role", "property")
        if (
            role != "edge"
            and isinstance(field.get("edge_label"), str)
            and field_type in model_names
        ):
            field["role"] = "edge"
            role = "edge"
            ctx.add(
                "R9",
                "warn",
                model_name,
                field_name,
                "edge_label present on a model-typed field — role corrected to 'edge'",
                repaired=True,
            )
        if role == "edge":
            _predraft_edge_field(model_name, field, field_type, model_names, ctx)
        else:
            _predraft_clear_edge_markers(model_name, field, ctx)
        if field.get("closed_catalog") and not field.get("reference"):
            field["closed_catalog"] = False
            ctx.add(
                "R12",
                "warn",
                model_name,
                field_name,
                "closed_catalog without reference — marker cleared",
                repaired=True,
            )


def _predraft_edge_field(
    model_name: str,
    field: dict[str, Any],
    field_type: str,
    model_names: set[Any],
    ctx: _Ctx,
) -> None:
    field_name = str(field.get("name", "?"))
    if field_type not in model_names:
        ctx.add(
            "IR",
            "warn",
            model_name,
            field_name,
            f"edge targets non-model type '{field_type}' — demoted to property",
            repaired=True,
        )
        field["role"] = "property"
        _predraft_clear_edge_markers(model_name, field, ctx, silent=True)
        return
    label = field.get("edge_label")
    if isinstance(label, str) and label.strip():
        return  # normalization happens post-validation in R9
    derived = derive_edge_label(field_name, field_type)
    field["edge_label"] = derived
    ctx.add(
        "R9",
        "warn",
        model_name,
        field_name,
        f"edge without a label — derived '{derived}' from the field name",
        repaired=True,
    )
    ctx.gap(
        model_name,
        field_name,
        "missing_edge_label",
        f"label '{derived}' was derived; propose an ALL_CAPS verb phrase",
    )


def _predraft_clear_edge_markers(
    model_name: str, field: dict[str, Any], ctx: _Ctx, *, silent: bool = False
) -> None:
    stray = [
        key
        for key in ("edge_label", "reference", "closed_catalog")
        if field.get(key) not in (None, False)
    ]
    if not stray:
        return
    field["edge_label"] = None
    field["reference"] = False
    field["closed_catalog"] = False
    if not silent:
        ctx.add(
            "IR",
            "warn",
            model_name,
            str(field.get("name", "?")),
            f"edge-only markers {stray} on a non-edge field — cleared",
            repaired=True,
        )


def _identity_rank(field: dict[str, Any], position: int) -> tuple[int, float, int]:
    """R1 trim order: digit-bearing examples first, then shortest example."""
    examples = [str(ex) for ex in field.get("examples") or [] if isinstance(ex, str | int | float)]
    has_digit = any(any(ch.isdigit() for ch in ex) for ex in examples)
    shortest = min((len(ex) for ex in examples), default=math.inf)
    return (0 if has_digit else 1, shortest, position)


def _predraft_identity(model: dict[str, Any], ctx: _Ctx) -> None:
    """R1/R2 pre-validation: identity shape, kind demotions, R13 doubling."""
    model_name = str(model.get("name", "?"))
    fields = _field_dicts(model)
    by_name = {str(f.get("name")): f for f in fields}
    kind = model.get("kind")
    if kind not in ("entity", "component", "root"):
        has_identity = bool(model.get("identity_fields")) or any(
            f.get("role") == "identity" for f in fields
        )
        kind = "entity" if has_identity else "component"
        model["kind"] = kind
        ctx.add(
            "IR",
            "warn",
            model_name,
            None,
            f"missing/invalid kind — defaulted to '{kind}'",
            repaired=True,
        )

    raw_identity = model.get("identity_fields")
    identity: list[str] = []
    if isinstance(raw_identity, list):
        for name in raw_identity:
            if name in by_name:
                if name not in identity:
                    identity.append(name)
            else:
                ctx.add(
                    "R1",
                    "warn",
                    model_name,
                    str(name),
                    "identity field not declared in fields — dropped from identity_fields",
                    repaired=True,
                )
    for name in identity:
        by_name[name]["role"] = "identity"
    for field in fields:
        field_name = str(field.get("name"))
        if field.get("role") == "identity" and field_name not in identity:
            if kind in ("entity", "root") and len(identity) < MAX_IDENTITY_FIELDS:
                identity.append(field_name)
                ctx.add(
                    "R1",
                    "warn",
                    model_name,
                    field_name,
                    "field with role='identity' adopted into identity_fields",
                    repaired=True,
                )
            else:
                field["role"] = "property"
                ctx.add(
                    "R1",
                    "warn",
                    model_name,
                    field_name,
                    "stray identity role demoted to property",
                    repaired=True,
                )
    for name in identity:
        _predraft_retype_identity(model_name, by_name[name], ctx)

    if kind == "component":
        if identity:
            for name in identity:
                by_name[name]["role"] = "property"
            identity = []
            ctx.add(
                "R1",
                "warn",
                model_name,
                None,
                "components carry no identity_fields — cleared (roles demoted to property)",
                repaired=True,
            )
        if model.get("max_instances") is not None:
            model["max_instances"] = None
            ctx.add(
                "R1",
                "warn",
                model_name,
                None,
                "components take no max_instances — cleared",
                repaired=True,
            )
    elif kind == "root" and not identity:
        identity = [_predraft_synthesize_root_identity(model, by_name, ctx)]
    elif kind == "entity" and not identity:
        model["kind"] = "component"
        model["max_instances"] = None
        ctx.add(
            "R1",
            "warn",
            model_name,
            None,
            "entity with no identity evidence demoted to component (never invent ids)",
            repaired=True,
        )
    if len(identity) > MAX_IDENTITY_FIELDS:
        ranked = sorted(identity, key=lambda n: _identity_rank(by_name[n], identity.index(n)))
        keep = set(ranked[:MAX_IDENTITY_FIELDS])
        for name in identity:
            if name in keep:
                continue
            by_name[name]["role"] = "property"
            ctx.add(
                "R1",
                "warn",
                model_name,
                name,
                f"more than {MAX_IDENTITY_FIELDS} identity fields — '{name}' demoted to "
                "property (kept the digit-bearing/shortest-example ids)",
                repaired=True,
            )
        identity = [n for n in identity if n in keep]
    model["identity_fields"] = identity

    max_instances = model.get("max_instances")
    if (
        model.get("kind") in ("entity", "root")
        and isinstance(max_instances, int)
        and not isinstance(max_instances, bool)
        and max_instances >= 1
    ):
        model["max_instances"] = max_instances * 2
        ctx.add(
            "R13",
            "info",
            model_name,
            None,
            f"graph_max_instances stored as {max_instances * 2} (2x the documented maximum "
            f"{max_instances}; drafts carry the documented figure, the spec the doubled one)",
            repaired=False,
        )

    docstring = model.get("docstring")
    if not isinstance(docstring, str) or not docstring.strip():
        model["docstring"] = f"{model_name}."
        ctx.add(
            "IR",
            "warn",
            model_name,
            None,
            "empty docstring replaced with a placeholder",
            repaired=True,
        )
        ctx.gap(
            model_name,
            None,
            "missing_docstring",
            "docstring must front-load IS / IS-NOT / cardinality in its first 240 chars",
        )


def _predraft_retype_identity(model_name: str, field: dict[str, Any], ctx: _Ctx) -> None:
    """R2: identity is scalar, non-list, un-normalized, and never an edge."""
    field_name = str(field.get("name", "?"))
    if field.get("is_list"):
        field["is_list"] = False
        ctx.add(
            "R2",
            "warn",
            model_name,
            field_name,
            "identity fields are scalar — is_list cleared",
            repaired=True,
        )
    field_type = field.get("type", "str")
    if field_type not in SCALAR_TYPES:
        field["type"] = "str"
        ctx.add(
            "R2",
            "warn",
            model_name,
            field_name,
            f"identity typed '{field_type}' (enum/model) — retyped to 'str'",
            repaired=True,
        )
    if field.get("normalizer") not in (None, "none"):
        field["normalizer"] = "none"
        ctx.add(
            "R2",
            "warn",
            model_name,
            field_name,
            "identity values are copied verbatim — normalizer cleared",
            repaired=True,
        )
    if field.get("role") == "identity" and (
        field.get("edge_label") is not None or field.get("reference") or field.get("closed_catalog")
    ):
        field["edge_label"] = None
        field["reference"] = False
        field["closed_catalog"] = False
        ctx.add(
            "R2",
            "warn",
            model_name,
            field_name,
            "edge markers on an identity field — cleared",
            repaired=True,
        )
    field["role"] = "identity"


def _predraft_synthesize_root_identity(
    model: dict[str, Any], by_name: dict[str, dict[str, Any]], ctx: _Ctx
) -> str:
    """Identity-less root: synthesize ``document_reference`` (+ gap)."""
    model_name = str(model.get("name", "?"))
    field = by_name.get("document_reference")
    if field is None:
        field = {
            "name": "document_reference",
            "type": "str",
            "role": "identity",
            "description": "Identifier printed on the document, e.g. reference number or title.",
            "examples": [],
        }
        model["fields"] = [field, *model["fields"]]
    else:
        field["role"] = "identity"
        field["type"] = "str"
        field["is_list"] = False
    ctx.add(
        "R1",
        "warn",
        model_name,
        "document_reference",
        "identity-less root — synthesized 'document_reference' identity (a real printed "
        "identity beats the filename fallback)",
        repaired=True,
    )
    ctx.gap(
        model_name,
        "document_reference",
        "missing_identity",
        "root identity synthesized; confirm the document prints a usable reference",
    )
    return "document_reference"


def _predraft_spec_level(data: dict[str, Any], ctx: _Ctx) -> None:
    """Dangling canonical_home / needs_root_list_dedup pruning."""
    models = _model_dicts(data)
    by_name = {str(m.get("name")): m for m in models}
    for model in models:
        home = model.get("canonical_home")
        if home is None:
            continue
        valid = False
        if isinstance(home, str) and "." in home:
            parent_name, _, field_name = home.partition(".")
            parent = by_name.get(parent_name)
            if parent is not None:
                valid = any(f.get("name") == field_name for f in _field_dicts(parent))
        if not valid:
            model["canonical_home"] = None
            ctx.add(
                "IR",
                "warn",
                str(model.get("name")),
                None,
                f"dangling canonical_home {home!r} cleared",
                repaired=True,
            )
    dedup = data.get("needs_root_list_dedup")
    root = by_name.get(str(data.get("root")))
    if isinstance(dedup, list) and root is not None:
        root_fields = {str(f.get("name")): f for f in _field_dicts(root)}
        kept = [n for n in dedup if n in root_fields and root_fields[n].get("is_list")]
        dropped = [n for n in dedup if n not in kept]
        if dropped:
            data["needs_root_list_dedup"] = kept
            ctx.add(
                "R19",
                "warn",
                str(root.get("name")),
                None,
                f"needs_root_list_dedup entries pruned (unknown or non-list): {dropped}",
                repaired=True,
            )
