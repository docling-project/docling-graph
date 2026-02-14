# Schema Definition

Pydantic templates are the schema contract for all extraction modes (`direct`, `staged`, `delta`).

## Core rules

- Use explicit entities (`graph_id_fields`) and components (`is_entity=False`).
- Keep identity fields short, stable, and required when possible.
- Prefer 2-4 nesting levels; flatten deeply recursive structures.
- Use `edge(label=...)` consistently for relationship-bearing fields.
- Write extraction-oriented descriptions and realistic examples for every important field.

## Extraction-focused design

- **Direct:** optimize semantic clarity and validation tolerance.
- **Staged:** optimize ID discovery and parent linkage determinism.
- **Delta:** optimize path fidelity, flat properties, canonicalized values, and merge-safe identities.

## Recommended reading order

1. `template-basics.md`
2. `entities-vs-components.md`
3. `field-definitions.md`
4. `relationships.md`
5. `best-practices.md`
6. `staged-extraction-schema.md`
7. `validation.md`
8. `advanced-patterns.md`
