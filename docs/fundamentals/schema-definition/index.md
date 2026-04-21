# Schema Definition

Pydantic templates are the schema contract for the supported extraction modes (`direct`, `dense`).

## Core rules

- Use explicit entities (`graph_id_fields`) and components (`is_entity=False`).
- Keep identity fields short, stable, and required when possible; prefer descriptive IDs over raw section/figure labels.
- Prefer 2-4 nesting levels; flatten deeply recursive structures.
- Use `edge(label=...)` consistently for relationship-bearing fields.
- Write extraction-oriented descriptions and realistic examples for every important field.
- Use validators to correct semantic errors (e.g. wrong unit in amount) and to deduplicate root-level lists when using chunked extraction.

## Extraction-focused design

- **Direct:** optimize semantic clarity and validation tolerance.
- **Dense:** optimize identity discovery, parent linkage, and chunk-aware per-entity filling.

## Recommended reading order

1. `template-basics.md`
2. `entities-vs-components.md`
3. `field-definitions.md`
4. `relationships.md`
5. `best-practices.md`
6. `validation.md`
8. `advanced-patterns.md`
