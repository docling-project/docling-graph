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
- Avoid optional identity fields in staged and delta extraction (both use catalog identity for merge and linkage).

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
