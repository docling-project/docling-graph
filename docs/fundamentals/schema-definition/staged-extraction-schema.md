# Schema Design for Dense Extraction

Dense extraction runs in two phases — Phase 1 discovers a skeleton of entity instances (paths + identities only), Phase 2 fills each instance — and the schema is compiled into a **node catalog** that drives both. This guide covers what the catalog extracts from your template and what that implies for design. For the full rationale see [Best practices](best-practices.md).

## What the catalog takes from your template

For every entity model reachable from the root, the catalog records: the path (e.g. `studies[].experiments[]`), the entity's `graph_id_fields`, up to **three `examples` per identity field** (each truncated to 50 characters), and the model **docstring** (capped at 400 characters). Phase 1 sees *only* this — no property names, types, or descriptions.

- Identity `examples` are mandatory in practice: they are the only concrete id guidance the discovery prompt gets. Use short, **document-derived** examples (codes, figure/table labels, proper names).
- Entity docstrings: one or two sentences on what the entity is and how it's identified.
- Identity examples can live on the child model's id fields (preferred) or as list-of-dict examples on the parent field; the catalog collects both.

## Identity requirements

- Root and every entity expose `graph_id_fields`; identity fields are **required**.
- One scalar id field is ideal, two is the maximum. Never list-valued fields or enums as identity — their surface forms drift between batches and break parent linkage.
- Ids must be **verbatim-copyable** labels (codes, names, figure labels), never invented or positional (`ITEM-1`, `Offer 2`).
- Encode distinguishing parameters as **digits in the id** (`Batch-20vol`): ids that differ in any digit run are guard-protected from fuzzy merging, while prose distinctions are not.
- Do not use section or chapter titles, or bare section letters/Roman numerals, as identities; prefer descriptive labels (`STUDY-BINDER-MW`, `FIG-4`).
- Name id fields honestly: `*_number` / `*_no` / `ref_*` fields are expected to contain digits — a root invariant clears prose values from them. Use `title` / `name` when the identity is a name.

## Parent linkage

Phase 1 links each node to its parent by an integer handle within the same response; after merge, filled instances re-attach to parents by identity. When identity drift occurs, a rescue ladder recovers the link (unique parent → fuzzy containment → placeholder → shared bucket). To make the early, precise rungs succeed:

- Keep sibling parents' ids short and mutually distinct early in the string.
- Parents that are natural singletons (one protocol, one setup per document) tolerate drift best — don't split them into per-mention instances.
- Children of parents with empty ids survive (bucket rung) but lose sibling attribution — another reason identity fields are required.

## Components and edges

- Components (`is_entity=False`) are filled inline with their parent and add **no catalog path** — use them for value objects (amounts, measurements, addresses).
- Attach relationship-bearing fields with `edge(label=...)`; edges are optional by default.
- Never nest the same rich entity model at several paths — give it one root-level home with full detail and make other occurrences name-only references (graph assembly merges them by identity).

## Complexity limits

- Prefer 2–4 depth levels; deep chains multiply fill passes and parent-linkage hops.
- Reduce fan-out; split very broad domains into focused templates.
- Every entity path costs catalog lines in every Phase 1 batch and a fill pass per discovered instance.

## Quality-readiness checklist

- Root identity present, stable, honestly named.
- Entity ids required, scalar, exemplified with document-derived labels.
- Non-identity fields optional or defaulted (small-model resilience).
- Parent-child paths deterministic; no rich model duplicated across paths.
- Relationship-bearing fields carry explicit edge labels.
- After a run, check `results.dense` in `metadata.json` (retention %, drops) to verify the schema behaves.
