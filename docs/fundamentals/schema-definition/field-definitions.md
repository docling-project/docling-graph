# Field Definitions

Field definitions are the main quality lever for extraction consistency.

## Required patterns

- Use `Field(..., description=..., examples=[...])` for core fields.
- Keep examples format-aligned with description rules.
- Normalize value formats in descriptions (dates, units, codes, casing).
- Use `default_factory=list` for list fields.
- Avoid nested object payloads for Delta-critical scalar properties.

## Identity fields

- Identity fields in `graph_id_fields` must be **required, scalar, and concise** — one field ideal, two maximum.
- Never use list-valued fields or enums as identity: their surface forms drift between extraction batches and break parent linkage.
- Avoid long free text as identity (`description`, `resume`, paragraph fields) — long ids inflate skeleton output (truncation pressure) and cannot be reconciled as aliases.
- Ids must be **copyable verbatim** from the document; never instruct fallback generation (`'ITEM-1'`) — invented ids never match across batches.
- Put distinguishing **digits in the id** (`Batch-20vol`): ids differing in any digit run are protected from fuzzy merging.
- Name id fields honestly: `*_number` / `*_no` / `ref_*` fields must hold values that contain digits (a pipeline invariant clears prose from them at the root); use `name` / `title` when the identity is a name.
- Include 2-5 short, document-derived examples for each identity field — in dense extraction these examples are the only id guidance Phase 1 sees.
- For local identities (for example line indexes), add context fields in schema to disambiguate.

## Optionality guidance

- **Required = identity, nothing else.** The fill phase pads short responses and restores ids from the skeleton, but a required non-identity field makes every partial instance fail validation — an all-or-nothing loss that small models hit constantly.
- Optional (`| None`, `default_factory=list`) or defaulted for every property field.
- Never make identity fields optional: children referencing id-less parents lose sibling attribution.
- Don't restate global prompt rules per field ("omit if absent", "never use N/A", "copy digits verbatim") — the extraction prompts enforce these pipeline-wide, and field descriptions are paid for on every fill call.

## Description style

- Mention where the value appears in the document.
- Mention normalization/canonicalization rules.
- Mention ambiguity resolution when relevant.

Example:

```python
currency_code: str = Field(
    ...,
    description="ISO 4217 currency code from totals section. Normalize to uppercase 3-letter code.",
    examples=["EUR", "USD", "GBP"],
)
```

## Where-to-look hints

For fields that are often missed (e.g. protocol parameters, axis labels), add short “where to look” hints so the LLM searches the right part of the document:

- **Methods section:** e.g. *“Look in Methods for ‘pre-shear’, ‘equilibration’, ‘gap’; extract values even if they appear mid-paragraph.”*
- **Figure captions / axis labels:** e.g. *“Look in figure captions and axis labels for the quantity name.”*

These hints improve extraction of explicit numbers and reduce empty shells when the schema has many optional fields.

## Enum synonyms and mapping

When an enum is used (e.g. geometry type, test mode), document synonyms and discourage overuse of “Other”:

- In the **Field description**, list common document phrases that map to each value, e.g. *“Map ‘parallel plate’, ‘parallel disk’, or ‘plate-plate’ to ‘Plate-Plate’. Do not use Other when the text matches a known type.”*
- Optionally add a `mode="before"` field validator that maps frequent phrases (e.g. string containing “parallel” and “plate”) to the correct enum member before calling a generic enum normalizer.

This reduces “Other” when the document clearly states a known type in different wording.

---

**See also:** [Best practices](best-practices.md) (identity and descriptive IDs, optionality, deduplication), [Validation](validation.md) (semantic sanity validators, enum mapping, list deduplication).
