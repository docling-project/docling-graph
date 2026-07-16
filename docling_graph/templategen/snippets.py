"""
Verbatim code blocks the template renderer emits.

Every constant in this module is a string of Python source. The renderer never
synthesizes validator or helper code on the fly — it can only emit these fixed
blocks, which is how the rulebook's "validators normalize, never reject" law is
enforced by construction: no snippet below contains a ``raise`` statement
(guarded by tests/unit/templategen/test_snippets.py).

Two kinds of constants:

- **Verbatim blocks** (``EDGE_HELPER``, ``IMPORT_BLOCK``, ...): emitted as-is.
- **Per-field templates** (``*_TEMPLATE``): contain ``str.format`` placeholders
  (``{field}``, ``{enum_name}``, ...); literal braces inside them are doubled.

Sources are cited per constant. The ``edge()`` helper's canonical source is the
shipped production example ``docs/examples/templates/insurance_terms.py`` (the
extended form with ``reference``/``closed_catalog``) — its signature and body
are copied character-for-character; only its docstring is carried over in
English instead of the original French. The metadata keys are exactly what the
runtime reads: ``edge_label`` (graph_converter), ``graph_reference``
(contracts/dense/catalog), ``reference_closed_catalog`` (graph_converter).
"""

# Metadata keys read by the runtime — pinned by tests against the source files
# that consume them.
EDGE_METADATA_KEY_LABEL = "edge_label"
EDGE_METADATA_KEY_REFERENCE = "graph_reference"
EDGE_METADATA_KEY_CLOSED_CATALOG = "reference_closed_catalog"

# --- edge() helper -----------------------------------------------------------
# Source: docs/examples/templates/insurance_terms.py (signature and body
# character-for-character; docstring translated to English). Single edges
# default to None per the Optionality Law (field-definitions.md: required =
# identity, nothing else); list edges pass default_factory=list.
EDGE_HELPER = '''\
def edge(
    label: str,
    default: Any = None,
    *,
    reference: bool = False,
    closed_catalog: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Declare a field as a graph edge for Docling-Graph via json_schema_extra.

    ``reference=True`` marks an identity-ONLY link (graph_reference): the field
    carries id-only references to entities described in full elsewhere in the
    schema. In dense extraction these references are filled by the PARENT's own
    fill call (never discovered separately), which preserves per-parent
    membership lists and avoids phantom parents.

    ``closed_catalog=True`` (reference_closed_catalog) declares that the targets
    form a CLOSED catalog defined elsewhere in the document: a target that only
    exists through this field (never instantiated or referenced otherwise) is a
    hallucinated member — the converter then drops the edge (and the orphaned
    target), unless that would wipe more than half of the target class.
    """
    json_schema_extra = dict(kwargs.pop("json_schema_extra", {}) or {})
    json_schema_extra["edge_label"] = label
    if reference:
        json_schema_extra["graph_reference"] = True
    if closed_catalog:
        json_schema_extra["reference_closed_catalog"] = True

    if "default_factory" in kwargs:
        default_factory = kwargs.pop("default_factory")
        return Field(default_factory=default_factory, json_schema_extra=json_schema_extra, **kwargs)

    return Field(default, json_schema_extra=json_schema_extra, **kwargs)
'''

# --- imports -----------------------------------------------------------------
# Source: docs/fundamentals/schema-definition/template-basics.md, "Required
# Imports" — the mandatory two lines every template starts with.
IMPORT_BLOCK = """\
from typing import Any, List, Optional, Union, Self, Type
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
"""

# Conditional imports, included only when the emitted template uses them
# (template-basics.md "Include based on domain needs" / validation.md logging).
OPTIONAL_IMPORT_DATETIME = "from datetime import date, datetime"
OPTIONAL_IMPORT_ENUM = "from enum import Enum"
OPTIONAL_IMPORT_RE = "import re"
OPTIONAL_IMPORT_LOGGING = "import logging"

# Source: docs/examples/templates/insurance_terms.py module preamble.
LOGGER_SETUP = "logger = logging.getLogger(__name__)"

# --- enum normalization ------------------------------------------------------
# Source: docs/fundamentals/schema-definition/validation.md, "Enum
# Normalization Helper" — adapted to the same document's never-reject law
# ("Graceful Error Handling"): the doc's version re-raises on unmappable
# values; this one coerces to OTHER (or returns the value for Pydantic to
# judge) and logs instead.
NORMALIZE_ENUM_HELPER = '''\
def _normalize_enum(enum_cls: Type[Enum], v: Any) -> Any:
    """
    Accept enum instances, value strings, or member names.
    Handles various formats: 'VALUE', 'value', 'Value', 'VALUE_NAME'.
    Falls back to the OTHER member instead of rejecting (never raises).
    """
    if isinstance(v, enum_cls):
        return v
    if isinstance(v, str):
        key = re.sub(r"[^A-Za-z0-9]+", "", v).lower()
        mapping: dict[str, Any] = {}
        for member in enum_cls:
            mapping[re.sub(r"[^A-Za-z0-9]+", "", member.name).lower()] = member
            mapping[re.sub(r"[^A-Za-z0-9]+", "", str(member.value)).lower()] = member
        if key in mapping:
            return mapping[key]
    if "OTHER" in enum_cls.__members__:
        logger.warning("Unmapped enum value %r for %s; falling back to OTHER", v, enum_cls.__name__)
        return enum_cls.OTHER
    return v
'''

# Per-field delegation to _normalize_enum (validation.md "Usage Example").
# Placeholders: {field} (snake_case field name), {enum_name} (enum class name).
ENUM_FIELD_VALIDATOR_TEMPLATE = '''\
    @field_validator("{field}", mode="before")
    @classmethod
    def _normalize_{field}(cls, v: Any) -> Any:
        """Map free-text values onto {enum_name} members (falls back to OTHER)."""
        return _normalize_enum({enum_name}, v)
'''

# List-of-enum variant: every item is normalized individually so case/synonym
# variants in a list never raise (the same never-reject law as the scalar
# form); non-list input delegates to the scalar path for Pydantic to judge.
# Placeholders: {field}, {enum_name}.
ENUM_LIST_FIELD_VALIDATOR_TEMPLATE = '''\
    @field_validator("{field}", mode="before")
    @classmethod
    def _normalize_{field}(cls, v: Any) -> Any:
        """Map free-text list items onto {enum_name} members (falls back to OTHER)."""
        if isinstance(v, list):
            return [_normalize_enum({enum_name}, item) for item in v]
        return _normalize_enum({enum_name}, v)
'''

# --- coerce-and-log field validators (never raising) --------------------------
# Source: validation.md "Lenient Validator Patterns" (symbol-to-code conversion
# + case normalization). Placeholder: {field}.
CURRENCY_VALIDATOR_TEMPLATE = '''\
    @field_validator("{field}", mode="before")
    @classmethod
    def _normalize_{field}_currency(cls, v: Any) -> Any:
        """Normalize currency symbols and casing to ISO 4217 codes; never rejects."""
        if not v:
            return v
        symbol_map = {{
            "€": "EUR",
            "$": "USD",
            "£": "GBP",
            "¥": "JPY",
        }}
        v_str = str(v).strip()
        if v_str in symbol_map:
            return symbol_map[v_str]
        v_upper = v_str.upper()
        if len(v_upper) == 3 and v_upper.isalpha():
            return v_upper
        logger.warning("Currency %r does not match ISO 4217 format; kept as %r", v, v_upper)
        return v_upper
'''

# Source: validation.md "Use Type Guards" — made lenient per the same
# document's never-reject law: unparseable values are dropped (None) with a
# warning instead of raising. Placeholder: {field}.
NUMERIC_VALIDATOR_TEMPLATE = '''\
    @field_validator("{field}", mode="before")
    @classmethod
    def _coerce_{field}_numeric(cls, v: Any) -> Any:
        """Coerce numeric strings ('1 500,00', '$1,500.00') to float; never rejects."""
        if not isinstance(v, str):
            return v
        cleaned = re.sub(r"[^\\d,.\\-]", "", v)
        if "," in cleaned:
            if re.search(r",\\d\\d?$", cleaned):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            logger.warning("Could not parse numeric value %r; dropping it", v)
            return None
'''

# Source: validation.md "String to List Conversion". Placeholder: {field}.
STRING_LIST_VALIDATOR_TEMPLATE = '''\
    @field_validator("{field}", mode="before")
    @classmethod
    def _coerce_{field}_list(cls, v: Any) -> Any:
        """Ensure {field} is always a list (accepts bare or comma-separated strings)."""
        if isinstance(v, str):
            if "," in v:
                return [part.strip() for part in v.split(",") if part.strip()]
            return [v]
        if v is None:
            return []
        return v
'''

# --- root list dedup ---------------------------------------------------------
# Source: validation.md "Deduplicate root-level list by key" — keeps the first
# occurrence per normalized key (chunked extraction can repeat items).
# Placeholders: {field}; {key_expr} is an expression over ``item`` producing
# the dedup key, e.g. ``str(item).strip().lower()`` for scalar lists or
# ``(getattr(item, "full_name", None) or "").strip().lower()`` for models.
ROOT_LIST_DEDUP_TEMPLATE = '''\
    @model_validator(mode="after")
    def _deduplicate_{field}(self) -> Self:
        """Keep first occurrence per key (removes duplicates from chunked extraction)."""
        if not self.{field}:
            return self
        seen: set[str] = set()
        unique: list[Any] = []
        for item in self.{field}:
            key = {key_expr}
            if key not in seen:
                seen.add(key)
                unique.append(item)
        object.__setattr__(self, "{field}", unique)
        return self
'''

# --- __str__ -----------------------------------------------------------------
# Source: advanced-patterns.md "Pattern 9: String Representations" — join
# non-empty identity + leading property parts, "Unknown" fallback.
# Placeholder: {parts} — comma-separated ``self.<field>`` expressions.
STR_METHOD_TEMPLATE = """\
    def __str__(self) -> str:
        parts = [{parts}]
        return " ".join(str(p) for p in parts if p) or "Unknown"
"""
