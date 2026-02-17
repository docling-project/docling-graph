# EdgeQuake-Inspired Improvements: Implementation Detail (All Contracts)

This document details each recommended improvement, what code is shared vs contract-specific, and in what order to implement. It extends the main plan (EdgeQuake vs Docling Graph IE).

---

## 1. Architecture: Shared vs Contract-Specific Code

### 1.1 Current Contract Boundaries

- **Direct**: Single LLM call (full doc or one chunk). Output: one JSON → one Pydantic model. Merge only when VLM returns multiple page models → `merge_pydantic_models` (uses `dict_merger.deep_merge_dicts`).
- **Staged**: 3-pass (catalog → ID → fill → edges). Fill pass in batches; merge via `merge_and_dedupe_flat_nodes` in `docling_graph/core/extractors/contracts/staged/catalog.py` (dedupe by `(path, sorted(ids))`). Output: one template-shaped dict.
- **Delta**: Chunk → token batches → LLM per batch → `normalize_delta_ir_batch_results` → `merge_delta_graphs` in `docling_graph/core/extractors/contracts/delta/helpers.py` (node key = `node_identity_key(path, ids, dedup_policy)`; property merge via `_preferred_property_value`) → optional `resolve_post_merge_graph` → `project_graph_to_template_root`. Output: one template-shaped dict.

### 1.2 What Should Be Shared (All Three Contracts)

| Shared component | Location | Used by |
|------------------|----------|---------|
| **Entity name normalizer** | `core/utils/entity_name_normalizer.py` | dict_merger (direct/VLM merge), delta (resolvers / identity key for name-based paths), staged (identity when path has name field), node_id_registry (optional name in fingerprint) |
| **Description merger** | `core/utils/description_merger.py` | dict_merger (direct, staged, delta when merging "description" fields), delta `_preferred_property_value` or dedicated description merge in `merge_delta_graphs` |
| **Append-only provenance helper** | `core/utils/provenance.py` (new) or extend dict_merger | All: when merging list fields that represent sources/chunk_ids; delta already does this in `__property_provenance` |
| **Relationship cleanup** | `core/utils/graph_cleaner.py` (extend) | Delta (after merge), staged (when assembling edges), graph_converter / export |
| **Retry on truncation** | `core/extractors/backends/llm_backend.py` | All contracts (direct, staged, delta all go through this backend) |
| **Gleaning interface** | `core/extractors/gleaning.py` (new) | Contract-specific runners; shared config and "merge gleaned into result" helpers |

### 1.3 What Stays Contract-Specific

- **Direct**: No chunking; gleaning = one optional second full-doc call; merge = `merge_pydantic_models` only (VLM multi-page). Integrates shared: dict_merger (already used), description merge for "description" fields, entity normalizer if we add name-based dedup in dict_merger.
- **Staged**: Catalog build, ID pass, fill batches, edge assembly. Integrates shared: in `merge_and_dedupe_flat_nodes` use entity normalizer when comparing identity for paths that have a "name" field; use description_merger when merging fill results for the same (path, ids); relationship cleanup when building edges.
- **Delta**: Normalizer, `merge_delta_graphs`, resolvers, projection. Integrates shared: entity normalizer in resolvers or when building `node_identity_key` for name-based paths; description_merger in `merge_delta_graphs` for "description" property; relationship cleanup after merge; append-only provenance already in place.

---

## 2. Detailed Implementation Steps (Pragmatic Order)

### Phase 1 – Shared foundations (no contract-specific behavior yet)

**1.1 Entity name normalizer**

- **File**: `docling_graph/core/utils/entity_name_normalizer.py` (new).
- **API**: `normalize_entity_name(raw: str) -> str`: trim → strip prefixes "The ", "A ", "An " (case-insensitive) → split words → strip possessive `'s` → join with `_` → uppercase. Handle empty/whitespace-only → `""`. Optional: Unicode NFKD normalize for accents.
- **Tests**: Unit tests for "John Doe" → "JOHN_DOE", "The Company" → "COMPANY", "  Sarah  Chen  " → "SARAH_CHEN", empty string, single word.
- **No callers yet**: Just land the module.

**1.2 Description merger**

- **File**: `docling_graph/core/utils/description_merger.py` (new).
- **API**:
  - `merge_descriptions(existing: str, new: str, max_length: int = 4096) -> str`: sentence-split (`.!?`), add only sentences from `new` not in `existing`, join, then truncate at sentence boundary.
  - `truncate_at_sentence_boundary(text: str, max_length: int) -> str`: truncate at last `.?!` before `max_length`.
- **Tests**: Empty existing, empty new, duplicate sentence not re-added, truncation at sentence.
- **No callers yet**: Land the module.

**1.3 Extend dict_merger for description fields**

- **File**: `docling_graph/core/utils/dict_merger.py`.
- **Change**: Add optional parameter `description_merge_fields: set[str] | None = None` (e.g. `{"description", "summary"}`). When merging two scalar strings for a key in this set, call `merge_descriptions(existing, new, max_length)` instead of overwriting. Default `None` = current behavior. Caller can pass from config.
- **Benefit**: Direct (VLM merge) and any path that uses `merge_pydantic_models`/`deep_merge_dicts` with this option get description merge.

**1.4 Relationship cleanup in graph_cleaner**

- **File**: `docling_graph/core/utils/graph_cleaner.py`.
- **Add**: `drop_self_edges(graph: nx.DiGraph) -> nx.DiGraph` (or in-place): remove edges where source == target. `cap_edge_keywords(graph, edge_attr: str = "keywords", max_keywords: int = 5)`: if edges have a list/tuple `keywords`, truncate to first `max_keywords`. Integrate into `clean_graph()` as optional steps.
- **Tests**: Unit test for self-edge removal and keyword cap.

**Order**: 1.1 → 1.2 → 1.3 → 1.4. Phase 1 is done when all four are merged and tests pass.

---

### Phase 2 – Retry on truncation (all contracts)

**2.1 Unify truncation detection and retry in LlmBackend**

- **File**: `docling_graph/core/extractors/backends/llm_backend.py`.
- **Current**: Staged already has some truncation retry; direct and delta batch calls do not consistently retry on truncation/parse failure.
- **Change**: After any `get_json_response`: if parse fails with EOF / "unexpected end" / "unclosed" (or provider returns `finish_reason=length` when available), retry once with higher `max_tokens` (e.g. 2x, cap at 32k). Apply to direct extraction call, delta batch calls, and staged fill/id calls. Config: `retry_on_truncation: bool = True`, `truncation_retry_max_tokens_multiplier: float = 2.0`.
- **Tests**: Mock LLM that returns truncated JSON on first call and valid on second; assert one retry and success.

**Order**: After Phase 1. Single file change; all contracts benefit.

---

### Phase 3 – Wire shared utils into each contract

**3.1 Delta**

- **Entity normalizer**: In `delta/helpers.py` or `delta/resolvers.py`, when comparing two nodes for dedup (e.g. in resolvers when building a fallback key from "name" field), normalize name-like fields with `entity_name_normalizer.normalize_entity_name` before comparison. Option: in `node_identity_key` when dedup_policy says a path uses a "name" identity field, normalize that field's value.
- **Description merge**: In `merge_delta_graphs`, when merging node properties, if `prop_key` is "description" (or in a small configurable list), call `merge_descriptions(existing_value, incoming_value, max_length)` instead of `_preferred_property_value`. Keep `_preferred_property_value` for other keys. Add config `description_merge_max_length` (default 4096).
- **Relationship cleanup**: After `merge_delta_graphs` (and resolvers if any), before projection: drop relationships where source_key == target_key; if relationship has a "keywords" list, cap at 5. Either in helpers or a small function in delta/orchestrator called after merge.
- **Provenance**: Delta already appends in `__property_provenance`. Document as append-only; no code change required.

**3.2 Direct**

- **Description merge**: When calling `merge_pydantic_models`, pass `description_merge_fields={"description", "summary"}` (or from config) into `deep_merge_dicts` so that VLM multi-page merge uses description_merger. Requires many_to_one to pass this through; dict_merger to accept and use it (Phase 1.3).
- **Entity normalizer**: Optional: if we add name-based list dedup in dict_merger for a path keyed by normalized name, use `normalize_entity_name` when computing identity. Can be Phase 4 or later.

**3.3 Staged**

- **Entity normalizer**: In `staged/catalog.py` `merge_and_dedupe_flat_nodes`, when path has an identity field that is "name" (or in an allowlist), normalize `ids.get("name")` before building key so "John Doe" and "john doe" dedupe to one node.
- **Description merge**: When merging two nodes with same (path, ids) in staged, use `merge_descriptions` for description-like fields if we add that merge path. Otherwise defer.
- **Relationship cleanup**: When building edges in staged, drop self-edges and cap keywords if edges have source/target node refs.

**Order**: 3.1 (delta) first, then 3.2 (direct), then 3.3 (staged).

---

### Phase 4 – Append-only provenance (documentation + small helper)

**4.1 Document and optional helper**

- **Doc**: In `dict_merger.py` or `docs/merge_behavior.md`, document that list fields used for provenance (e.g. `source_chunk_ids`, `provenance`, `__property_provenance`) must be appended to, never replaced. Delta already does this.
- **Helper**: Optional `provenance_append(existing_list, new_items, max_len=None)` in `core/utils/provenance.py`; call from dict_merger when merging a key in `provenance_append_only_fields` (configurable set). If not needed for direct/staged, skip helper and only document.

**Order**: After Phase 3. Low risk.

---

### Phase 5 – Gleaning (optional second pass)

**5.1 Shared gleaning interface**

- **File**: `docling_graph/core/extractors/gleaning.py` (new).
- **API**:
  - `GleaningConfig(max_passes: int = 1, prompt_builder: Callable | None = None)`.
  - `run_gleaning_pass(chunks, existing_result, template, llm_call, contract)`: build "already extracted: …" summary from `existing_result`, call LLM with "what did you miss?", parse response, return extra result in contract-specific shape.
  - Contract-specific merge: `merge_gleaned_direct(existing, extra, template)` using dict_merger; `merge_gleaned_delta(existing_graph, extra_batches, catalog, dedup_policy)` using normalize + merge_delta_graphs.
- **Integration**: LlmBackend or ManyToOneStrategy checks config `gleaning_enabled` and `gleaning_max_passes`; after first extraction, if enabled, calls `run_gleaning_pass` then merges. Direct: one extra full-doc call. Delta: one extra batch loop with "already found" in prompt. Staged: optional after fill pass (later).

**Order**: After Phase 4. Implement direct first, then delta; staged if needed.

---

### Phase 6 – Optional LLM summarization for descriptions

**6.1 Optional summarizer in description merge**

- **File**: `docling_graph/core/utils/description_merger.py`.
- **Change**: Add optional `summarizer: Callable[[str, list[str]], str] | None = None`. When merging more than N descriptions or total length > threshold, call summarizer; else use sentence-dedup + truncate. Pipeline can inject LLM callable from config.
- **Config**: `use_llm_summarization_for_descriptions: bool = False`, `llm_summarizer_max_inputs: int = 4`. Fallback to non-LLM merge.

**Order**: Last; depends on description_merger and contract wiring being stable.

---

## 3. Implementation Order Summary

| Phase | What | Shared code | Contract touch points |
|-------|-----|-------------|------------------------|
| 1 | Entity normalizer, description merger, dict_merger extension, graph_cleaner | New utils, extend dict_merger, graph_cleaner | None |
| 2 | Retry on truncation | llm_backend | All (single backend) |
| 3 | Wire utils into contracts | — | Delta (helpers/resolvers, merge_delta_graphs), Direct (dict_merger options), Staged (merge_and_dedupe_flat_nodes, edges) |
| 4 | Append-only provenance | Doc, optional provenance helper | dict_merger if helper used |
| 5 | Gleaning | gleaning.py | Backend / many_to_one; direct then delta |
| 6 | LLM summarization for descriptions | description_merger | Config + optional callable in merge path |

**Recommended sequence**: 1 → 2 → 3 → 5 (skip 4 if not needed) → 4 → 6. Phase 1 and 2 give immediate value with minimal risk; Phase 3 makes all three contracts benefit from the same foundations; Phase 5 adds recall; Phase 4 and 6 polish.
