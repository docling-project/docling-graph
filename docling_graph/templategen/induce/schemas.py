"""
Provider-safe JSON output schemas for the three induction passes and gap-fill.

Every schema here is deliberately **flat and shallow** (objects nest at most
two levels; shared shapes live in ``$defs`` at the root) because
``normalize_schema_for_response_format`` (llm_clients/schema_utils.py) turns
them into strict grammars that small models must satisfy — the dense
contract's integer-handle lesson: cheap output contracts beat expressive ones.

Strict-grammar conventions applied throughout:

- every object sets ``additionalProperties: false`` and requires every
  property (OpenAI-style strict mode accepts nothing less);
- **null-free by design**: optional semantics use empty-value sentinels
  (``""`` for "no identity candidate", ``0`` for "cardinality not stated",
  ``[]`` for "no examples") instead of nullable unions, the least portable
  strict-grammar construct across providers. The parsers in ``documents.py``
  decode the sentinels.

The gap-fill schema is the structural firewall of the design: it is keyed by
``(model, field, kind)`` and carries **only** ``docstring`` / ``description``
/ ``examples`` value slots — there is no slot through which a class, field,
or edge could enter the SPEC, so gap-fill hallucination cannot change
structure by construction.
"""

from __future__ import annotations

from typing import Any

GAP_FILL_KINDS: tuple[str, ...] = (
    "missing_docstring",
    "missing_examples",
    "missing_identity",
    "ambiguous_kind",
    "missing_description",
    "missing_edge_label",
)
"""Mirror of ``spec.SpecGap.kind`` — the only keys gap-fill output may carry."""

GAP_FILL_VALUE_SLOTS: frozenset[str] = frozenset({"docstring", "description", "examples"})
"""The only value slots a gap-fill entry has. Structure is unrepresentable."""


def class_inventory_schema() -> dict[str, Any]:
    """Pass 1 — class inventory. ``{"classes": [...]}``.

    ``identity_candidate.field == ""`` means "no printed identity"
    (components); ``documented_max_count == 0`` means "the document does not
    state a maximum".
    """
    return {
        "type": "object",
        "properties": {
            "classes": {
                "type": "array",
                "items": {"$ref": "#/$defs/class_candidate"},
            }
        },
        "required": ["classes"],
        "additionalProperties": False,
        "$defs": {
            "identity_candidate": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": (
                            "snake_case name of the ONE field whose value names instances "
                            "of this class; '' when the document names no instances."
                        ),
                    },
                    "why": {
                        "type": "string",
                        "description": "Where the identity value is printed in the document.",
                    },
                    "verbatim_examples": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 5,
                        "description": "2-5 identity values copied verbatim from the text.",
                    },
                },
                "required": ["field", "why", "verbatim_examples"],
                "additionalProperties": False,
            },
            "class_candidate": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "PascalCase concept name."},
                    "kind": {"type": "string", "enum": ["entity", "component"]},
                    "is_root": {
                        "type": "boolean",
                        "description": "true for exactly one class: the document itself.",
                    },
                    "what_it_is": {
                        "type": "string",
                        "description": "One sentence (<=240 chars) saying what the class IS.",
                    },
                    "confusable_with": {
                        "type": "string",
                        "description": (
                            "The dominant confusable concept and how this differs; '' if none."
                        ),
                    },
                    "identity_candidate": {"$ref": "#/$defs/identity_candidate"},
                    "documented_max_count": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Maximum instance count ONLY when the document states it; 0 otherwise."
                        ),
                    },
                    "evidence_quotes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 3,
                        "description": "Short verbatim quotes proving the class exists.",
                    },
                },
                "required": [
                    "name",
                    "kind",
                    "is_root",
                    "what_it_is",
                    "confusable_with",
                    "identity_candidate",
                    "documented_max_count",
                    "evidence_quotes",
                ],
                "additionalProperties": False,
            },
        },
    }


def fields_schema() -> dict[str, Any]:
    """Pass 2 — fields per class (batched <=6 classes per call).

    ``type`` is a scalar name (str/int/float/bool/date/datetime) or
    ``"enum:<Name>"`` for closed value sets; enum synonyms are a flat array of
    ``{member, phrases}`` pairs (a member-keyed mapping is not strict-grammar
    safe).
    """
    return {
        "type": "object",
        "properties": {
            "classes": {
                "type": "array",
                "items": {"$ref": "#/$defs/class_fields"},
            }
        },
        "required": ["classes"],
        "additionalProperties": False,
        "$defs": {
            "enum_synonym": {
                "type": "object",
                "properties": {
                    "member": {"type": "string"},
                    "phrases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Document phrases that map to this member.",
                    },
                },
                "required": ["member", "phrases"],
                "additionalProperties": False,
            },
            "field_candidate": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "snake_case field name."},
                    "type": {
                        "type": "string",
                        "description": (
                            "One of: str, int, float, bool, date, datetime — or 'enum:<Name>' "
                            "when the value set is small and closed."
                        ),
                    },
                    "is_list": {"type": "boolean"},
                    "description": {
                        "type": "string",
                        "description": "LOOK-FOR locator plus at most ONE normalization rule.",
                    },
                    "verbatim_examples": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 5,
                        "description": "Values copied verbatim from the text; [] when none.",
                    },
                    "enum_members": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Printed member values; [] unless type is 'enum:<Name>'.",
                    },
                    "enum_synonyms": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/enum_synonym"},
                    },
                    "unit_varies": {
                        "type": "boolean",
                        "description": "true when the same quantity appears with several units.",
                    },
                },
                "required": [
                    "name",
                    "type",
                    "is_list",
                    "description",
                    "verbatim_examples",
                    "enum_members",
                    "enum_synonyms",
                    "unit_varies",
                ],
                "additionalProperties": False,
            },
            "class_fields": {
                "type": "object",
                "properties": {
                    "class_name": {"type": "string"},
                    "fields": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/field_candidate"},
                    },
                },
                "required": ["class_name", "fields"],
                "additionalProperties": False,
            },
        },
    }


def relationships_schema() -> dict[str, Any]:
    """Pass 3 — relationships. ``{"edges": [...]}``.

    ``label == ""`` means "unsure" — the draft repair derives one from the
    field name and raises a ``missing_edge_label`` gap.
    ``target_described_fully_here == false`` is the raw ``reference=True``
    signal; the linter (R10/R11) makes the final canonical-home call.
    """
    return {
        "type": "object",
        "properties": {
            "edges": {
                "type": "array",
                "items": {"$ref": "#/$defs/edge_candidate"},
            }
        },
        "required": ["edges"],
        "additionalProperties": False,
        "$defs": {
            "edge_candidate": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Class name from the list."},
                    "field_name": {
                        "type": "string",
                        "description": "snake_case field on the source class.",
                    },
                    "target": {"type": "string", "description": "Class name from the list."},
                    "label": {
                        "type": "string",
                        "description": "ALL_CAPS verb phrase; '' if unsure.",
                    },
                    "is_list": {
                        "type": "boolean",
                        "description": "true when the source links to several targets.",
                    },
                    "target_described_fully_here": {
                        "type": "boolean",
                        "description": (
                            "false when this edge only names/links the target and its full "
                            "details live elsewhere (membership list, cross-reference)."
                        ),
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 2,
                        "description": "Verbatim quotes showing the relationship.",
                    },
                },
                "required": [
                    "source",
                    "field_name",
                    "target",
                    "label",
                    "is_list",
                    "target_described_fully_here",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },
    }


def gapfill_schema() -> dict[str, Any]:
    """Gap-fill output: content for declared gaps, structure unrepresentable.

    Entries are keyed by ``(model, field, kind)`` — ``field == ""`` for
    model-level gaps — and carry only the :data:`GAP_FILL_VALUE_SLOTS`
    (``docstring`` / ``description`` / ``examples``). There is no slot for a
    class, field, edge, type, label, or identity, so gap-fill output cannot
    change SPEC structure no matter what the model emits. Slots irrelevant to
    a gap kind are ``""`` / ``[]``.
    """
    return {
        "type": "object",
        "properties": {
            "fills": {
                "type": "array",
                "items": {"$ref": "#/$defs/gap_fill"},
            }
        },
        "required": ["fills"],
        "additionalProperties": False,
        "$defs": {
            "gap_fill": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model name from the gap list."},
                    "field": {
                        "type": "string",
                        "description": "Field name from the gap list; '' for model-level gaps.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": list(GAP_FILL_KINDS),
                        "description": "The gap kind being filled, copied from the gap list.",
                    },
                    "docstring": {
                        "type": "string",
                        "description": "For missing_docstring gaps; '' otherwise.",
                    },
                    "description": {
                        "type": "string",
                        "description": "For missing_description gaps; '' otherwise.",
                    },
                    "examples": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 5,
                        "description": "For missing_examples/missing_identity gaps; [] otherwise.",
                    },
                },
                "required": ["model", "field", "kind", "docstring", "description", "examples"],
                "additionalProperties": False,
            },
        },
    }
