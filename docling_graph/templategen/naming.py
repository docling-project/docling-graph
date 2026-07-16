"""
Deterministic identifier utilities for template generation.

Everything here is pure string manipulation — no LLM, no I/O — so the same
input SPEC always renders the same Python identifiers and edge labels.

Encodes three rulebooks:

- ``relationships.md`` edge-label conventions: ALL_CAPS verb phrases; the ban
  list {HAS, LINK, RELATED, RELATED_TO, IS, OF}; bare single-token nouns get a
  ``HAS_`` prefix; multi-token labels are kept as-is (the linter's R9 advisory
  flags unknown first tokens instead of rewriting user-chosen verb phrases).
  The verb vocabulary below is data lifted from that document's "Common Edge
  Labels by Category" section, expanded with present-tense forms.
- Python identifier hygiene: keywords and builtins are renamed instead of
  shadowed (linter rule R20).
- Reserved node-attribute keys: ``GraphConverter._create_nodes_pass`` writes
  ``id``/``label``/``type``/``__class__`` on every node, so template fields
  must never collide with them (linter rule R21).
- Template-module collisions: class names must not shadow names every
  generated template imports; field names must not shadow the module-level
  names (``date``, ``datetime``, ``edge``, ``logger``) that the class body
  would otherwise rebind, breaking later annotations and edge() calls.
"""

from __future__ import annotations

import builtins
import keyword
import re

from .spec import SCALAR_TYPES

RESERVED_NODE_ATTRS: frozenset[str] = frozenset({"id", "label", "type", "__class__"})
"""Node-attribute keys written by ``GraphConverter._create_nodes_pass`` —
a template field with one of these names silently corrupts node attributes."""

RESERVED_FIELD_RENAMES: dict[str, str] = {
    "id": "identifier",
    "label": "name_label",
    "type": "category",
}
"""Deterministic renames for fields colliding with :data:`RESERVED_NODE_ATTRS`."""

BANNED_EDGE_LABELS: frozenset[str] = frozenset({"HAS", "LINK", "RELATED", "RELATED_TO", "IS", "OF"})
"""Vague labels banned outright by relationships.md; rewritten to HAS_<TARGET>."""

_BASE_EDGE_VERBS: frozenset[str] = frozenset(
    {
        # relationships.md — Authorship & Ownership
        "ISSUED",
        "CREATED",
        "AUTHORED",
        "OWNED",
        "PUBLISHED",
        "VERIFIED",
        "APPROVED",
        "SIGNED",
        # relationships.md — Recipients & Targets
        "SENT",
        "ADDRESSED",
        "DELIVERED",
        "BILLED",
        "INSURED",
        "COVERED",
        # relationships.md — Location & Physical Presence
        "LOCATED",
        "LIVES",
        "BASED",
        "MANUFACTURED",
        "OPERATES",
        "SHIPS",
        # relationships.md — Composition & Containment
        "CONTAINS",
        "HAS",
        "INCLUDES",
        "COMPOSED",
        # relationships.md — Membership & Association
        "BELONGS",
        "PART",
        "MEMBER",
        "EMPLOYED",
        "EMPLOYS",
        "AFFILIATED",
        "PARTNERED",
        # relationships.md — Services & Offerings
        "OFFERS",
        "PROVIDES",
        # relationships.md — Research & Scientific
        "USES",
        # best-practices.md / shipped canon (WORKS_AT, IS_PERSON, REFERENCES_ITEM)
        "WORKS",
        "IS",
        "REFERENCES",
        # common ontology property verbs (worksFor-style localnames)
        "COVERS",
        "EXTENDS",
        "EXCLUDES",
        "APPLIES",
        "DESCRIBES",
        "DEFINES",
        "REQUIRES",
        "SUPPORTS",
        "MENTIONS",
        "CITES",
        "GOVERNS",
        "PRODUCES",
    }
)
"""Hand-curated verb tokens (relationships.md vocabulary + shipped canon)."""

_PRESENT_TENSE_EXTRAS: frozenset[str] = frozenset(
    {
        "OWNS",
        "KNOWS",
        "PAYS",
        "MANAGES",
        "SELLS",
        "HOLDS",
        "RECEIVES",
        "REPRESENTS",
        "GRANTS",
        "COVERS",
        "EMPLOYS",
        "OPERATES",
        "PRODUCES",
        "SUPPLIES",
        "REQUIRES",
        "REFERENCES",
        "SENDS",
        "MAKES",
        "RUNS",
        "LEADS",
        "SERVES",
        "SUPPORTS",
    }
)
"""Explicit present-tense verbs common in user/ontology labels (ownsVehicle,
grantsCoverage, ...) that the past-tense derivation below cannot produce."""


def _present_tense_forms(verbs: frozenset[str]) -> frozenset[str]:
    """Best-effort present-tense forms of past-tense verbs (ISSUED -> ISSUES).

    Both ``V[:-2] + 'S'`` and ``V[:-1] + 'S'`` are generated; misses like
    ``ISSUS`` are harmless — they never match a real label token.
    """
    forms: set[str] = set()
    for verb in verbs:
        if len(verb) > 2 and verb.endswith("ED"):
            forms.add(verb[:-2] + "S")
            forms.add(verb[:-1] + "S")
    return frozenset(forms)


EDGE_VERB_PREFIXES: frozenset[str] = (
    _BASE_EDGE_VERBS | _present_tense_forms(_BASE_EDGE_VERBS) | _PRESENT_TENSE_EXTRAS
)
"""First tokens that make a label read as a verb phrase. A single-token label
not in this set is treated as a bare noun and prefixed with ``HAS_``; a
multi-token label with an unknown first token is kept as-is (assumed verb
phrase) and only flagged by the linter's R9 advisory. Bare banned labels
(``HAS`` alone, ``IS`` alone) are caught before this check."""

_BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))

TEMPLATE_RESERVED_NAMES: frozenset[str] = frozenset(
    {
        # names every generated template imports or defines at module level
        "BaseModel",
        "ConfigDict",
        "Field",
        "field_validator",
        "model_validator",
        "AliasChoices",
        "Enum",
        "Any",
        "List",
        "Optional",
        "Union",
        "Self",
        "Type",
        "date",
        "datetime",
        "edge",
        "logger",
        "logging",
        "re",
    }
)
"""Module-level names in generated templates that class names must not shadow."""

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")


def _split_words(name: str) -> list[str]:
    """Split camelCase / PascalCase / kebab-case / snake_case / spaces into words."""
    chunks = [chunk for chunk in _NON_ALNUM.split(name) if chunk]
    words: list[str] = []
    for chunk in chunks:
        words.extend(w for w in _CAMEL_BOUNDARY.split(chunk) if w)
    return words


def _pascal_word(word: str) -> str:
    """Capitalize a word, keeping short all-caps acronyms (USB, MRH) intact."""
    if word.isupper() and 1 < len(word) <= 3:
        return word
    return word[:1].upper() + word[1:].lower()


def to_pascal_case(name: str) -> str:
    """``line_item`` / ``lineItem`` / ``line-item`` / ``Line item`` -> ``LineItem``."""
    return "".join(_pascal_word(w) for w in _split_words(name))


def to_snake_case(name: str) -> str:
    """``LineItem`` / ``lineItem`` / ``line-item`` / ``Line item`` -> ``line_item``."""
    return "_".join(w.lower() for w in _split_words(name))


def _to_upper_snake(name: str) -> str:
    return "_".join(w.upper() for w in _split_words(name))


def is_verb_phrase(label: str) -> bool:
    """Pragmatic verb-phrase check: is the first token a known relationship verb?"""
    upper = _to_upper_snake(label)
    if not upper:
        return False
    first = upper.split("_", 1)[0]
    return first in EDGE_VERB_PREFIXES


def normalize_edge_label(label: str, target: str | None = None) -> str:
    """Normalize an edge label to the relationships.md conventions.

    Exact semantics (the templategen-wide contract):

    1. Banned vague labels ({HAS, LINK, RELATED, RELATED_TO, IS, OF}, exact
       full match after case conversion) -> ``HAS_<TARGET>`` (requires
       ``target``, the edge's target class name).
    2. camelCase / kebab-case / spaces -> UPPER_SNAKE
       (``worksFor`` -> ``WORKS_FOR``).
    3. Single-token label: kept when the token is a known relationship verb
       (``covers`` -> ``COVERS``), else treated as a bare noun and prefixed
       (``address`` -> ``HAS_ADDRESS``).
    4. Multi-token label: KEPT AS-IS after case conversion, whatever its first
       token (``ownsVehicle`` -> ``OWNS_VEHICLE``, ``managedBy`` ->
       ``MANAGED_BY``). User-chosen verb phrases are never rewritten; the
       linter's R9 advisory flags unknown first tokens instead.

    Args:
        label: The raw label (ontology localname, LLM proposal, user input).
        target: Target class name, used to rewrite banned labels.

    Raises:
        ValueError: If ``label`` is empty, or banned with no ``target`` to
            rewrite from.
    """
    upper = _to_upper_snake(label)
    if not upper:
        raise ValueError(f"Edge label {label!r} contains no letters or digits")
    if upper in BANNED_EDGE_LABELS:
        target_upper = _to_upper_snake(target or "")
        if not target_upper:
            raise ValueError(
                f"Edge label {label!r} is banned (relationships.md) and no target "
                "was provided to rewrite it as HAS_<TARGET>"
            )
        return f"HAS_{target_upper}"
    if "_" in upper or is_verb_phrase(upper):
        return upper
    return f"HAS_{upper}"


def derive_edge_label(field_name: str, target: str) -> str:
    """Derive a label for a label-less edge from its field name (linter R9).

    Unlike :func:`normalize_edge_label` (which preserves user-chosen
    multi-token labels), derivation has no user intent to preserve: a field
    name that does not read as a verb phrase gets the ``HAS_`` prefix whether
    it is one token or several (``line_items`` -> ``HAS_LINE_ITEMS``,
    ``owns_vehicle`` -> ``OWNS_VEHICLE``). Banned or empty names fall back to
    ``HAS_<TARGET>``.
    """
    upper = _to_upper_snake(field_name)
    if not upper or upper in BANNED_EDGE_LABELS:
        return f"HAS_{_to_upper_snake(target)}"
    if is_verb_phrase(upper):
        return upper
    return f"HAS_{upper}"


RESERVED_TEMPLATE_FIELD_NAMES: frozenset[str] = frozenset({"date", "datetime", "edge", "logger"})
"""Module-level names of generated templates that a field must not rebind.

A field named ``date``/``datetime`` binds the annotation name in the class
body (templates carry no ``from __future__ import annotations``), breaking
every later ``date``-annotated field; a field named ``edge`` shadows the edge
helper for subsequent edge fields; ``logger`` shadows the module logger."""


def sanitize_field_name(name: str) -> str:
    """Return a safe snake_case field name.

    Applies, in order: snake_casing; the :data:`RESERVED_FIELD_RENAMES` map for
    reserved node-attr collisions (R21); a ``field_`` prefix for leading
    digits; and a ``_field`` suffix for Python keywords/builtins and the
    template-module names in :data:`RESERVED_TEMPLATE_FIELD_NAMES` (R20).
    """
    snake = to_snake_case(name)
    if not snake:
        return "field"
    if snake in RESERVED_FIELD_RENAMES:
        return RESERVED_FIELD_RENAMES[snake]
    if snake[0].isdigit():
        snake = f"field_{snake}"
    if (
        keyword.iskeyword(snake)
        or snake in _BUILTIN_NAMES
        or snake in RESERVED_TEMPLATE_FIELD_NAMES
    ):
        return f"{snake}_field"
    return snake


def sanitize_class_name(name: str) -> str:
    """Return a safe PascalCase class name.

    Applies, in order: PascalCasing; a ``Model`` prefix for leading digits; and
    a ``Model`` suffix when the name is a Python keyword, shadows a builtin
    (``Exception``), shadows a name every generated template imports
    (``Field``, ``Enum``, ...), or is a scalar type name (``date`` ->
    ``DateModel``) — a model whose lowercase name is in ``SCALAR_TYPES`` would
    otherwise make every same-named scalar field ambiguous with the model and
    corrupt rename cascades.
    """
    pascal = to_pascal_case(name)
    if not pascal:
        return "Model"
    if pascal[0].isdigit():
        pascal = f"Model{pascal}"
    if (
        keyword.iskeyword(pascal)
        or pascal in _BUILTIN_NAMES
        or pascal in TEMPLATE_RESERVED_NAMES
        or pascal.lower() in SCALAR_TYPES
    ):
        return f"{pascal}Model"
    return pascal
