"""
Induction prompts: pure ``{'system', 'user'}`` builders for the three
documents->SPEC passes and for gap-fill.

Mirrors the isolation style of ``core/extractors/contracts/*/prompts.py``:
prompts are module-level string constants (data — easy to review and tune)
assembled by small pure functions. No I/O, no LLM clients, no schema logic.

Every system prompt embeds :data:`CONDENSED_RULEBOOK` — the
``docs/fundamentals/schema-definition/`` rulebook compressed to ~60 lines:
the four entity-vs-component decision questions, the identity laws, the
NAME-instances rule, the edge-label vocabulary and ban list, the LOOK-FOR
description style, and the forbidden-computation rule. Keeping the rulebook
in the prompt is what lets the deterministic gates downstream stay thin.
"""

from __future__ import annotations

from typing import Sequence, TypedDict

from ..spec import SpecGap


class PromptDict(TypedDict):
    """Type definition for prompt dictionaries."""

    system: str
    user: str


# ---------------------------------------------------------------------------
# The condensed rulebook (shared by all passes)
# ---------------------------------------------------------------------------

CONDENSED_RULEBOOK = """\
You are designing a knowledge-graph extraction schema for documents of the kind shown.
Follow this rulebook exactly.

ENTITY vs COMPONENT — the four decision questions:
1. Should this be tracked individually? Yes -> entity. No -> component.
2. If I see this twice with identical values, is it one thing or two?
   One thing -> component. Two things -> entity.
3. Does this represent a unique object or a shared value?
   Unique object -> entity. Shared value -> component.
4. Would I want to query for all instances of this specific thing?
   Yes -> entity. No -> component.
Value objects (addresses, monetary amounts, measurements) are components.

THE NAME-INSTANCES RULE:
Model as an entity only what the document NAMES. If instances carry no verbatim label
the extraction can copy (an unnamed "study", "experiment", "run"), it is NOT an entity:
make it a component. NEVER invent, generate, or assign identifiers ("ITEM-1", "Offer 2");
invented ids never match across extraction batches.

IDENTITY LAWS:
- Identity is ONE scalar field (two at the absolute most): required, short, and copyable
  VERBATIM from the document.
- Never list-valued, enum-typed, model-typed, or long free-text identity fields.
- Prefer ids with distinguishing digits ("INV-2024-0113", "Batch-20vol").
- Name id fields honestly: *_number / *_no / ref_* fields must hold values that contain
  digits; use name / title when the identity is a name.
- Give 2-5 SHORT examples per identity field, copied character-for-character.

FIELD DESCRIPTIONS (1-3 sentences):
- A LOOK-FOR locator (where the value appears in the document) plus at most ONE
  normalization rule (date format, casing, code form).
- FORBIDDEN: instructions to calculate, compute, sum, convert, round, multiply, or derive
  values; id generation; restating global rules ("omit if absent", "never use N/A",
  "copy digits verbatim") — the extraction prompts enforce those pipeline-wide.

EDGE LABELS (relationships):
- ALL_CAPS verb phrases, consistent across the schema.
- Vocabulary by category:
  authorship: ISSUED_BY, CREATED_BY, AUTHORED_BY, OWNED_BY, PUBLISHED_BY, VERIFIED_BY,
    APPROVED_BY, SIGNED_BY
  recipients: SENT_TO, ADDRESSED_TO, DELIVERED_TO, BILLED_TO, INSURED_BY, COVERED_BY
  location: LOCATED_AT, LIVES_AT, BASED_AT, MANUFACTURED_AT, OPERATES_IN, SHIPS_TO
  composition: CONTAINS_LINE, HAS_COMPONENT, INCLUDES_PART, COMPOSED_OF, HAS_SECTION
  membership: BELONGS_TO, PART_OF, MEMBER_OF, EMPLOYED_BY, AFFILIATED_WITH, PARTNERED_WITH
  offerings: HAS_GUARANTEE, OFFERS_PLAN, PROVIDES_COVERAGE, OFFERS_PRODUCT, PROVIDES_SERVICE
  research: HAS_EXPERIMENT, USES_MATERIAL, HAS_MEASUREMENT, HAS_RESULT, USES_METHOD
- BANNED labels (too vague): HAS, LINK, RELATED, RELATED_TO, IS, OF.
  A bare noun becomes HAS_<NOUN> ("address" -> HAS_ADDRESS).

EVIDENCE:
Every example and quote you output MUST be a verbatim substring of the provided document
text (whitespace aside). Non-verbatim output is discarded by a deterministic gate.
Never paraphrase quotes. Text may have been sampled; elided regions are marked
"[... elided ...]" — never invent content for the gaps.
"""


# ---------------------------------------------------------------------------
# Pass-specific instruction blocks
# ---------------------------------------------------------------------------

_PASS1_INSTRUCTIONS = """\
TASK — CLASS INVENTORY:
Inventory the classes (entities and components) needed to model documents of this kind,
most important first, at most {max_models} classes.
Per class:
- name: PascalCase concept name. kind: "entity" or "component" (apply the four questions).
- is_root: true for exactly ONE class — the document itself.
- what_it_is: one sentence (<= 240 chars) saying what the class IS. It becomes the class
  docstring, so make it discriminate against the dominant confusable.
- confusable_with: the concept it is most easily confused with and how it differs
  ("" if none).
- identity_candidate: for entities, the ONE field whose value names instances:
  {{"field", "why", "verbatim_examples"}}. Set field to "" when the document does not
  name instances (components, unnamed structures).
- documented_max_count: the maximum instance count ONLY if the document itself states it
  (e.g. "the 3 coverage tiers"); 0 otherwise.
- evidence_quotes: 1-3 short verbatim quotes proving the class appears in the document.
OUTPUT BUDGET (hard): list each class exactly once — never repeat an entry; keep every
quote under 15 words. Stop after the last class.
"""

_PASS2_INSTRUCTIONS = """\
TASK — FIELDS:
For each listed class, propose its data fields as observed in the document.
Per field:
- name: snake_case. type: one of str, int, float, bool, date, datetime — or "enum:<Name>"
  when the value set is small and closed (a document type, a status, a currency).
- is_list: true when the document repeats the value for one instance.
- description: LOOK-FOR locator plus at most ONE normalization rule.
- verbatim_examples: 0-5 values copied verbatim from the text.
- enum_members: the printed member values (only for "enum:<Name>" types); enum_synonyms:
  document phrases that map to a member.
- unit_varies: true when the same quantity appears with different units.
Do NOT include relationship fields pointing at other classes here — a later pass
handles relationships. Do not invent fields the document does not show.
OUTPUT BUDGET (hard): at most 12 fields per class — the most informative ones. One short
sentence per description, at most 3 verbatim_examples per field, each under 12 words.
List each class exactly once and each field name at most once — never repeat yourself.
Stop after the last listed class.
"""

_PASS3_INSTRUCTIONS = """\
TASK — RELATIONSHIPS:
Propose the relationships between the listed classes as edges.
Per edge:
- source / target: class names copied exactly from the list below.
- field_name: snake_case field name on the source class holding the link.
- label: an ALL_CAPS verb phrase following the vocabulary above ("" if unsure — one will
  be derived from the field name).
- is_list: true when one source instance links to several targets.
- target_described_fully_here: false when this edge only names or links the target and the
  target's full details live at another path (membership lists, cross-references);
  true when this is where the target's details appear.
- evidence: 1-2 verbatim quotes showing the relationship.
Only propose edges between classes in the list. One edge per (source, field_name).
OUTPUT BUDGET (hard): at most 40 edges; never repeat an edge; keep quotes under 15 words.
Stop after the last edge.
"""

_ONESHOT_INSTRUCTIONS = """\
TASK — FULL ONTOLOGY IN ONE ANSWER:
Design the complete extraction ontology for documents of this kind in a single JSON
object: every class, its data fields, and its relationships, most important first,
at most {max_models} classes.
Per class:
- name (PascalCase), kind ("entity" or "component" — apply the four questions),
  is_root (true for exactly ONE class: the document itself),
  what_it_is (one sentence, <= 240 chars), confusable_with ("" if none).
- identity_candidate: for entities, {{"field", "why", "verbatim_examples"}} — the ONE
  field whose value names instances; field "" when the document names no instances.
- fields: the class's data fields. Per field: name (snake_case), type (str, int, float,
  bool, date, datetime — or "enum:<Name>" for a small closed value set), is_list,
  description (LOOK-FOR locator + at most ONE normalization rule),
  verbatim_examples (0-3 values copied verbatim), enum_members ([] unless enum type).
  Do NOT list relationship fields here.
- edges: the class's relationships. Per edge: field_name (snake_case), target (a class
  name from this same answer), label (ALL_CAPS verb phrase, "" if unsure), is_list,
  target_described_fully_here (false when the target's details live elsewhere),
  evidence (0-1 short verbatim quote).
OUTPUT BUDGET (hard): at most 12 fields and 6 edges per class, one short sentence per
description, every example under 12 words. List each class exactly once, each field name
at most once — never repeat yourself. Stop after the last class.
"""

_ONESHOT_SHAPE = """\
Return JSON with exactly this shape:
{"classes": [
  {"name": "Invoice", "kind": "entity", "is_root": true,
   "what_it_is": "A commercial invoice billing a buyer.", "confusable_with": "",
   "identity_candidate": {"field": "invoice_number", "why": "printed in the header",
                          "verbatim_examples": ["INV-2024-0113"]},
   "documented_max_count": 0,
   "evidence_quotes": ["Invoice No. INV-2024-0113"],
   "fields": [{"name": "total_amount", "type": "float", "is_list": false,
               "description": "LOOK FOR the Total line.", "verbatim_examples": ["138.90"],
               "enum_members": []}],
   "edges": [{"field_name": "issued_by", "target": "Party", "label": "ISSUED_BY",
              "is_list": false, "target_described_fully_here": true,
              "evidence": ["Issued by: Acme GmbH"]}]}
]}"""


_GAPFILL_INSTRUCTIONS = """\
TASK — FILL DOCUMENTATION GAPS:
An existing schema has declared documentation gaps. You may ONLY provide docstrings,
field descriptions, and example values for the gaps listed — copy model, field, and kind
exactly from the gap list into each fill entry. You cannot add, rename, retype, or remove
classes, fields, or relationships; output for anything not in the gap list is discarded.
Per fill entry:
- missing_docstring: set "docstring" (one IS sentence, <= 240 chars, plus the dominant
  IS-NOT discriminator when useful). Leave "description" empty and "examples" [].
- missing_description: set "description" (LOOK-FOR locator + at most ONE normalization
  rule). Leave "docstring" empty and "examples" [].
- missing_examples / missing_identity: set "examples" to 2-5 short, plausible values in
  the exact printed form such documents use. Leave "docstring" and "description" empty.
- Other gap kinds cannot be filled here; skip them.
"""


# ---------------------------------------------------------------------------
# User-prompt templates
# ---------------------------------------------------------------------------

_DOC_BLOCK = (
    "Document: {doc_name}\n\n=== DOCUMENT TEXT ===\n{document_text}\n=== END DOCUMENT TEXT ===\n\n"
)

_JSON_ONLY = "Return ONLY a JSON object matching the enforced schema."


def _system(instructions: str) -> str:
    return f"{CONDENSED_RULEBOOK}\n{instructions}\nImportant: Your response MUST be valid JSON."


# ---------------------------------------------------------------------------
# Public API: prompt builders
# ---------------------------------------------------------------------------


def get_oneshot_prompt(
    document_text: str,
    *,
    doc_name: str,
    max_models: int,
    root_hint: str | None = None,
) -> PromptDict:
    """One-shot prompt: the complete ontology (classes + fields + edges) in one call.

    The programmatic twin of the proven manual workflow — example documents
    plus the condensed schema-definition rulebook into one strong-model call,
    ontology JSON out; the deterministic merge/linter/renderer replace the
    "now turn it into Pydantic" second call. One LLM call per induction unit,
    and (by default) no grammar-constrained decoding: the JSON shape is shown
    as an example, and the response parsers tolerate missing keys.
    """
    hint = (
        f"The root class (the document itself) should be named '{root_hint}'.\n"
        if root_hint
        else ""
    )
    user = (
        _DOC_BLOCK.format(doc_name=doc_name, document_text=document_text)
        + hint
        + f"Design the full ontology (at most {max_models} classes) for this document.\n\n"
        + _ONESHOT_SHAPE
    )
    return {
        "system": _system(_ONESHOT_INSTRUCTIONS.format(max_models=max_models)),
        "user": user,
    }


def get_class_inventory_prompt(
    document_text: str,
    *,
    doc_name: str,
    max_models: int,
    root_hint: str | None = None,
) -> PromptDict:
    """Pass 1 prompt: inventory candidate classes for one document."""
    hint = (
        f"The root class (the document itself) should be named '{root_hint}'.\n"
        if root_hint
        else ""
    )
    user = (
        _DOC_BLOCK.format(doc_name=doc_name, document_text=document_text)
        + hint
        + f"Inventory at most {max_models} classes for this document. {_JSON_ONLY}"
    )
    return {
        "system": _system(_PASS1_INSTRUCTIONS.format(max_models=max_models)),
        "user": user,
    }


def get_fields_prompt(
    document_text: str,
    *,
    doc_name: str,
    classes: Sequence[tuple[str, str]],
) -> PromptDict:
    """Pass 2 prompt: propose fields for a batch of accepted classes.

    Args:
        document_text: The (possibly sampled) document text.
        doc_name: Display name of the source document.
        classes: ``(class_name, what_it_is)`` pairs for this batch (<=6).
    """
    class_lines = "\n".join(
        f"- {name}: {what_it_is}" if what_it_is else f"- {name}" for name, what_it_is in classes
    )
    user = (
        _DOC_BLOCK.format(doc_name=doc_name, document_text=document_text)
        + "Propose fields for exactly these classes:\n"
        + f"{class_lines}\n\n{_JSON_ONLY}"
    )
    return {"system": _system(_PASS2_INSTRUCTIONS), "user": user}


def get_relationships_prompt(
    document_text: str,
    *,
    doc_name: str,
    classes: Sequence[str],
) -> PromptDict:
    """Pass 3 prompt: propose edges between the accepted classes."""
    user = (
        _DOC_BLOCK.format(doc_name=doc_name, document_text=document_text)
        + "Classes: "
        + ", ".join(classes)
        + f"\n\nPropose the relationships between these classes. {_JSON_ONLY}"
    )
    return {"system": _system(_PASS3_INSTRUCTIONS), "user": user}


def get_gapfill_prompt(spec_summary: str, gaps: Sequence[SpecGap]) -> PromptDict:
    """Gap-fill prompt: fill only the declared documentation gaps."""
    gap_lines = "\n".join(
        f"- model={gap.model} field={gap.field or ''} kind={gap.kind}"
        + (f": {gap.note}" if gap.note else "")
        for gap in gaps
    )
    user = (
        "=== CURRENT SCHEMA ===\n"
        f"{spec_summary}\n"
        "=== END SCHEMA ===\n\n"
        "Declared gaps (fill ONLY these):\n"
        f"{gap_lines}\n\n{_JSON_ONLY}"
    )
    return {"system": _system(_GAPFILL_INSTRUCTIONS), "user": user}
