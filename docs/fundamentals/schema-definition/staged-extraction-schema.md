# Schema Design for Staged and Delta Extraction

This guide defines schema constraints for multi-pass extraction (staged and delta contracts use the same catalog and identity concepts).

## Identity requirements

- Root must expose reliable `graph_id_fields`.
- Entity models should define required, extractable identity fields.
- Identity examples are mandatory for robust ID discovery prompts.
- Avoid long free-text identifiers.

**Identity examples** for list-entities can be provided in either (or both) of these ways; the catalog collects from both:
- **Parent field**: list-of-dict examples on the field that contains the list (e.g. `studies = Field(examples=[{"study_id": "3.1", "objective": "..."}])`).
- **Child model's identity fields**: scalar `examples` on the entity's `graph_id_fields` (e.g. `study_id = Field(examples=["3.1", "STUDY-BINDER-MW"])`). These are included in the catalog so the LLM sees concrete valid-ID examples.
Prefer short, **document-derived** ID examples (section numbers, figure/table labels, named items). Do **not** use section or chapter titles as entity identities.

## Parent linkage requirements

- Child paths must have resolvable parent paths.
- Parent identity must be discoverable before child attachment.
- Local child IDs should be disambiguated by parent context when needed.

## Components and edges

- Components use `is_entity=False`.
- Components participating in graph relationships should be attached via `edge(label=...)`.
- Keep component payloads concise and extraction-friendly.

## Complexity limits

- Prefer 2-4 depth levels.
- Reduce fan-out where possible.
- Split very broad domains into focused templates.

## Delta-specific additions

- Keep extracted properties flat.
- Enforce exact catalog path usage.
- Favor fields that stabilize cross-batch identity and reconciliation.
- Provide canonicalization instructions in field descriptions.

## Quality-readiness checklist

- Root identity present and stable.
- Entity IDs required and exemplified.
- Parent-child paths deterministic.
- No critical fields hidden only in deeply nested branches.
- Relationship-bearing fields have explicit edge labels.
