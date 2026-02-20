# Best Practices

## Template checklist

- Define root identity with `graph_id_fields`.
- Keep entity IDs stable, short, and required.
- Use components for value objects (`is_entity=False`).
- Use explicit `edge(label=...)` where relationship materialization matters.
- Limit nesting depth (2-4 recommended).
- Use consistent naming and canonical examples.

## Deterministic extraction guidance

- Prefer schema fields that appear explicitly in source documents.
- Add canonicalization hints for dates, units, and codes.
- Avoid identity fields that require invention by the model.
- Use lenient validators that normalize values instead of rejecting entire output.

## Dense extraction: series and conditional values

- When one node can represent a **series** of values (e.g. one batch with φ = 0.10 … 0.29), add a `SeriesDefinition | None` type (define it in your template; see the [rheology template](../../examples/templates/rheology_research.py) for the pattern) so the fill phase can capture the full variation in one node.
- When a field has **different values under different conditions** (e.g. pre-shear rate for low vs high concentration), use `List[ConditionalValue]` (again, define in your template) instead of a scalar so the fill phase can extract all branches with their conditions.
- See [Advanced patterns: Series and conditional values](advanced-patterns.md#series-and-conditional-values) for examples.

## Delta-specific quality guidance

- Keep node properties flat (primitives or list of primitives).
- Use path-consistent relationship structures.
- Design local entities with parent context available in schema.
- Avoid ambiguous fallback identities by exposing meaningful discriminator fields.

## Identity and descriptive IDs

- Prefer **descriptive, human-readable** identifiers (e.g. `STUDY-TEMPERATURE-DEPENDENCE`, `Phenomenological fitting`) over bare section labels.
- **Avoid** using only a section letter or Roman numeral (e.g. `C`, `V`, `VI`) as an identity field when the document uses them as headings; in the Field description, add: *“Avoid using only a section letter or Roman numeral; prefer a descriptive label or combine with topic.”*
- Give 2–5 **concrete examples** that show the desired style (e.g. `FIG-4`, `STUDY-BINDER-MW`); avoid examples that look like raw headings (e.g. `3.1`, `Section-3`) if you want the model to prefer descriptive IDs.

## Optionality for often-missing fields

- Keep **identity fields** required when possible; the pipeline uses them for merge and linkage.
- For **non-identity** fields that the LLM often omits (e.g. objective, protocol details, axis labels), consider making them `Optional` with default `None` so validation passes without salvage and the graph does not store synthetic placeholders.
- Use optionality when the value is genuinely sparse in the source (e.g. “if not in Methods, leave empty”) rather than making every field optional.

## Deduplicating root-level lists (chunked extraction)

In **delta** (and other chunked) extraction, root-level list fields without identity (e.g. `authors`) can receive the same item from multiple batches, so the merged list may contain duplicates.

- Add a **model_validator** (`mode="after"`) on the root model that deduplicates the list by a stable key (e.g. `full_name` for authors: keep first occurrence per normalized name).
- This keeps the pipeline domain-agnostic; dedup logic lives in the template and applies after validation.

## Common failure causes

- Optional identity fields.
- Over-nested schemas with weak parent identifiers.
- Vague descriptions with no extraction hints.
- Inconsistent examples across equivalent fields.
- Edge labels omitted on relationship-bearing fields.
- Identity field descriptions that encourage raw section/figure labels (e.g. “section number”) without discouraging single-letter or Roman-numeral-only IDs.
