# Field Definitions

Field definitions are the main quality lever for extraction consistency.

## Required patterns

- Use `Field(..., description=..., examples=[...])` for core fields.
- Keep examples format-aligned with description rules.
- Normalize value formats in descriptions (dates, units, codes, casing).
- Use `default_factory=list` for list fields.
- Avoid nested object payloads for Delta-critical scalar properties.

## Identity fields

- Identity fields in `graph_id_fields` should be required and concise.
- Avoid long free text as identity (`description`, `resume`, paragraph fields).
- Include 2-5 examples for each identity field.
- For local identities (for example line indexes), add context fields in schema to disambiguate.

## Optionality guidance

- Required for identity and structural anchor fields.
- Optional for sparse enrichments that may not exist in source documents.
- Avoid optional identity fields in dense extraction, where stable catalog identity is important for merge and linkage.

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
