# Schema Definition

Pydantic templates are the schema contract for the supported extraction modes (`direct`, `dense`). The pipeline is domain-agnostic; the template is where all domain knowledge lives.

## Core rules

- Use explicit entities (`graph_id_fields`) and components (`is_entity=False`).
- Identity fields: **required, scalar, short, copied verbatim** from the document — never invented, list-valued, or enum-typed. Give 2-5 document-derived examples per id field.
- Non-identity fields: **optional or defaulted**, so partial output from smaller models degrades gracefully instead of failing validation.
- Prefer 2-4 nesting levels; never nest the same rich entity model at several paths — give it one root-level home and reference it by name elsewhere.
- Keep entities referenced from several paths identity-minimal; context-specific data (a role, a title) belongs on per-context entities linking to them — duplicate-instance merge fills missing values only, first non-empty wins.
- Use `edge(label=...)` consistently for relationship-bearing fields; edges optional by default.
- Keep field descriptions to a locator plus one normalization rule; never instruct computation or unit conversion (the pipeline grounds numbers digit-for-digit).
- Use validators to normalize what models actually emit (scalars, strings, stringified lists) and to deduplicate identity-less root lists — never to reject whole payloads.

## Extraction-focused design

- **Direct:** optimize semantic clarity and validation tolerance; keep templates flat (single-response output budget).
- **Dense:** optimize identity discovery, parent linkage, and chunk-aware per-entity filling; identity examples are the only id guidance Phase 1 sees.

## Recommended reading order

1. `template-basics.md`
2. `entities-vs-components.md`
3. `field-definitions.md`
4. `relationships.md`
5. `best-practices.md`
6. `validation.md`
7. `advanced-patterns.md`
