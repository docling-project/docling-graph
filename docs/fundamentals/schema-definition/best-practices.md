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

## Delta-specific quality guidance

- Keep node properties flat (primitives or list of primitives).
- Use path-consistent relationship structures.
- Design local entities with parent context available in schema.
- Avoid ambiguous fallback identities by exposing meaningful discriminator fields.

## Common failure causes

- Optional identity fields.
- Over-nested schemas with weak parent identifiers.
- Vague descriptions with no extraction hints.
- Inconsistent examples across equivalent fields.
- Edge labels omitted on relationship-bearing fields.
