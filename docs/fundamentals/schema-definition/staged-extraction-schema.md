# Schema Design for Staged Extraction - EXPERIMENTAL

## Overview

When using **staged extraction** (`extraction_contract="staged"`), the pipeline runs an **ID pass** that discovers node instances using only identity fields, then a **fill pass** that fills full content. Schema design directly affects whether the ID pass succeeds and the quality gate passes.

This guide is **domain-agnostic**: the same rules apply whether you are extracting invoices, insurance terms, research data, or any nested structure.

**In this guide:**

- Required schema properties for staged mode
- Identity fields: what works and what fails
- Components vs entities in the catalog
- Nested structure and catalog complexity
- Checklist and troubleshooting

See also: [Staged Extraction](../extraction-process/staged-extraction.md) for tuning and behavior.

---

## Required Schema Properties for Staged Mode

### Root model must have graph_id_fields

The quality gate requires at least one **root instance** (path `""`). The root model must define `graph_id_fields` so the ID pass can discover it.

```python
# Required for staged: root has identity
class Document(BaseModel):
    """Root document."""
    model_config = ConfigDict(graph_id_fields=["document_id"])

    document_id: str = Field(
        ...,
        description="Unique document identifier. Look for 'Doc No', 'Reference', or header.",
        examples=["DOC-2024-001", "INV-12345"]
    )
```

If the root has no `graph_id_fields`, or the chosen field is never present in the document, the ID pass will not produce a root instance and the quality gate will fail (fallback to direct extraction).

### Every entity in the ID pass should have graph_id_fields

By default (`staged_id_identity_only=True`), only **paths with non-empty `graph_id_fields`** are sent to the ID pass. Entities without `graph_id_fields` are skipped in identity-only mode.

- **Entities** (models you want as distinct graph nodes with stable IDs) must set `graph_id_fields`.
- **Components** (`is_entity=False`) have no identity in the ID pass; they are filled as part of their parent path or via edge-labeled relationships.

### Identity fields must be required and extractable

For staged extraction, ID fields are validated strictly: null or empty values cause validation errors and retries.

- Prefer **required** fields for `graph_id_fields` (use `...` or a non-None default).
- Avoid **optional** ID fields unless the document always supplies them; otherwise the LLM often omits them and validation fails.
- Choose fields that are **explicitly present** in typical documents (e.g. "Document number: X", "Offer name: Y") rather than implied or derived.

### Components and edge_label

Components are included in the staged catalog only when the relationship uses `edge()` with an `edge_label`. If you need a component to appear as a separate path for fill/merge, give it an edge label. Otherwise it is inlined under the parent path.

---

## Identity Fields: Good vs Fragile Choices

### Prefer short, stable, explicit identifiers

| Good | Fragile | Reason |
|------|---------|--------|
| `document_id`, `invoice_number` | Long free-text `reference_document` | Short IDs are less likely to be truncated or misparsed. |
| `name`, `title`, `code` | Long `description` or `resume` as ID | IDs should be labels or codes, not paragraphs. |
| Single field `["name"]` or compact composite | Many fields or long strings | Fewer, shorter ID fields reduce prompt size and errors. |

### Domain-agnostic examples

```python
# Good: root ID is short and explicit
class Contract(BaseModel):
    model_config = ConfigDict(graph_id_fields=["contract_code"])
    contract_code: str = Field(
        ...,
        description="Contract reference code (e.g. from header or footer).",
        examples=["CT-2024-01", "MRH-HAB-001"]
    )

# Good: entity ID is a natural key
class Offer(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str = Field(
        ...,
        description="Offer or plan name as shown in the document.",
        examples=["ESSENTIELLE", "CONFORT", "STANDARD"]
    )

# Fragile: optional ID field
class Item(BaseModel):
    model_config = ConfigDict(graph_id_fields=["nom"])
    nom: Optional[str] = Field(None, ...)  # LLM may omit → validation fails

# Fragile: long text as ID
class Exclusion(BaseModel):
    model_config = ConfigDict(graph_id_fields=["resume"])  # long text
    resume: str = Field(...)  # Prefer short "code" or "title" if possible
```

### Provide examples for ID fields

The ID-pass prompt uses schema hints (including `examples`) for each path. Adding 2–5 concise examples per ID field improves discovery.

```python
reference_document: str = Field(
    ...,
    description="Document reference. Look for 'Reference', 'Doc ref', or version.",
    examples=["HABITATION 2023-10", "CGV-MRH-2024", "v2.1"]
)
```

---

## Components vs Entities in Staged Extraction

### Entities

- Must have `graph_id_fields` to participate in the identity-only ID pass.
- Each entity path produces skeleton nodes with `path`, `ids`, and `parent`.
- Parent linkage uses these IDs; ensure parent paths also have identity so references resolve.

### Components

- Use `is_entity=False`.
- No `graph_id_fields`; they do not appear as separate identity paths when `staged_id_identity_only=True`.
- To have a component as a separate catalog path (for fill/merge), the field must use `edge(label="...")` with an `edge_label`.
- Components without an edge label are not separate catalog nodes; they are filled as part of the parent.

### Parent linkage

Merge attaches filled objects by `(path, id_tuple)` and parent `(parent_path, parent_id_tuple)`. For this to work:

- Parent paths that have identity should have **stable, extractable** `graph_id_fields`.
- Avoid deep chains where intermediate nodes have no identity (ambiguous parent refs).

---

## Nested Structure and Catalog Complexity

### Keep depth and fanout manageable

- **Depth:** Prefer 2–4 levels of nesting. Deeper structures increase catalog size and ID-pass prompt length, which can cause truncation or timeouts.
- **Fanout:** Many sibling paths under one parent (e.g. dozens of list types) increase the number of ID shards and retries.

### Catalog size

- The catalog is derived from your template: one path per root and per nested list/entity path (with the rules above).
- Large catalogs trigger **auto-sharding** when path count exceeds `staged_id_auto_shard_threshold`.
- Simplifying the template (fewer nested lists, fewer entity types) reduces catalog size and improves ID-pass reliability.

---

## Staged Schema Checklist

Use this when authoring or revising a template for staged extraction:

### Identity and root

- [ ] **Root model** has `graph_id_fields` and at least one required, extractable field.
- [ ] **Every entity** that should appear in the ID pass has `graph_id_fields`.
- [ ] **ID fields** are required (or reliably present) and have concise descriptions and examples.
- [ ] **ID values** are short and stable (e.g. codes, names, IDs), not long free text.

### Components and edges

- [ ] **Components** use `is_entity=False`.
- [ ] **Relationships** that must appear in the catalog use `edge(label="...")` with a clear `edge_label`.
- [ ] **Parent linkage** is unambiguous: parent paths that need to be referenced have identity.

### Complexity

- [ ] **Nesting depth** is reasonable (e.g. 2–4 levels).
- [ ] **Catalog size** is acceptable (fewer paths → smaller prompts and fewer shards).

---

## Troubleshooting: Failure Symptom

| Symptom (trace / logs) | Likely schema cause | Action |
|------------------------|---------------------|--------|
| `missing_root_instance` | Root has no `graph_id_fields` or root ID field is optional/missing in document | Add or fix root `graph_id_fields`; make ID required; ensure it appears in docs. |
| `insufficient_id_instances` | Too few paths with identity, or ID fields often empty | Add/fix `graph_id_fields` on entities; make ID fields required and add examples. |
| `empty_merged_output` | No valid skeleton after ID pass (validation failed on all shards) | Fix ID field types and examples; avoid optional ID fields; shorten IDs. |
| `missing id field(s): ['x']` in validation | LLM not returning required ID `x` | Make `x` required; add description and examples; prefer short, explicit identifiers. |
| `missing or invalid path` | LLM returning paths not in catalog | Ensure allowed paths match catalog (identity-only vs full); check entity/component and edge_label. |
| `unexpected id field(s): ['y']` | LLM returning a key not in `graph_id_fields` | Align prompt with schema; avoid generic names (e.g. `id_field`) that the model may invent. |
| Response truncated (ID pass) | Catalog or prompt too large | Reduce nesting; use identity-only; add `staged_id_max_tokens`; consider smaller shards. |
| Parent lookup misses | Parent path has no identity or wrong parent refs | Give parent entities `graph_id_fields`; ensure parent is discovered before children. |

---

## Validation Workflow

A repeatable way to check that your schema is staged-ready:

1. **Run with staged + debug**
   - Use `extraction_contract="staged"` and `debug=True` (or `--debug`).
   - Run on a representative document.

2. **Inspect trace**
   - Check for `quality_gate` and `fallback_reason` in the pipeline trace.
   - If fallback: read `quality_gate.reasons` (e.g. `missing_root_instance`, `insufficient_id_instances`, `empty_merged_output`).

3. **Map to schema**
   - Use the [Troubleshooting](#troubleshooting-failure-symptom) table to link reasons to schema (root ID, entity IDs, optional fields, depth).

4. **Adjust schema**
   - Apply the [Staged Schema Checklist](#staged-schema-checklist): root and entity IDs, required + examples, components/edges, depth.

5. **Re-run**
   - Run again with the same document; confirm ID pass produces instances and quality gate passes (or relax quality config only if your domain legitimately has no root / few instances).

---

## Next Steps

- [Staged Extraction](../extraction-process/staged-extraction.md) — Tuning and behavior
- [Entities vs Components](entities-vs-components.md) — Entity vs component and `graph_id_fields`
- [Best Practices](best-practices.md) — General template checklist including staged
