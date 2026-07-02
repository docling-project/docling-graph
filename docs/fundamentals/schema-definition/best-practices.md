# Best Practices

How to design Pydantic templates that work *with* the extraction pipeline instead of against it. Every rule below maps to a concrete, domain-agnostic pipeline mechanism; the schema is where domain knowledge lives, and these are the contract points between the two.

## Quick checklist

- Root and every entity define `graph_id_fields`; identity fields are **required, scalar, short, and copyable verbatim** from the document.
- Non-identity fields are **optional** (`| None` or `default_factory=list`) — required-ness belongs to identity only.
- Components (`is_entity=False`) for value objects; entities only where discovery and deduplication matter.
- `edge(label=...)` on relationship-bearing fields; edges optional by default.
- Field descriptions: 1–3 sentences (where to look + normalization rule); don't restate pipeline-level rules.
- Validators normalize; they never reject the whole payload.
- 2–4 nesting levels; never nest the same rich entity model at several paths.

---

## Token economics and compacting

Dense extraction discovers entities in Phase 1 (skeleton) and fills them in Phase 2. Both phases have strict token budgets, and the schema directly controls how much of that budget is spent.

**What the skeleton phase actually sees.** Phase 1 receives only the catalog: one line per entity path with its `graph_id_fields` names, plus up to three `examples` per identity field (each truncated to 50 chars) and the entity docstring (capped at 400 chars). Property fields, their types, and their descriptions are *invisible* in Phase 1. Consequences:

- Identity field **`examples` are your main Phase 1 lever** — they are the only concrete guidance the model gets for what an id should look like. Give 2–5 short, document-derived examples.
- Entity **docstrings** should say in one or two sentences what the entity is and how it is identified. Anything longer is truncated.
- Don't move load-bearing extraction hints into property descriptions expecting them to shape entity *discovery*; they only shape Phase 2 *filling*.

**Skeleton output is identity-only and handle-linked.** Each discovered node is emitted as `{"i": <int>, "path", "ids", "p": <parent handle>}` — the parent is a one-digit integer reference, not a repeated ancestry object. The remaining output cost is therefore the **ids themselves**, once per node. Short ids (`"FIG-4"`, `"Batch-1"`, `"CONFORT"`) keep entity-dense batches inside the model's output budget. Long ids — titles, sentences, descriptions-as-ids — inflate every node and push batches into truncation, which triggers the recovery cascade: one escalated retry, then recursive batch splitting. Recovery is lossless but multiplies LLM calls and wall-clock time. **The cheapest truncation fix is a shorter identity field.**

**Field descriptions are paid for on every fill call.** Phase 2 sends each entity's projected JSON schema — descriptions included — with every fill batch. Keep descriptions to a "LOOK FOR" locator plus one normalization rule. Do **not** repeat rules the fill prompt already enforces globally:

- omit absent values (never `"N/A"` / `"Not specified"` placeholders),
- copy numbers digit-for-digit (never compute, round, or aggregate),
- summary-row totals belong to document-level fields, not row instances.

Restating these per field costs tokens on every call and adds nothing.

**Entities cost catalog space; components don't.** Every entity model at every path adds a catalog line to Phase 1 and a fill pass to Phase 2. A component (`is_entity=False`) is filled inline with its parent. Model measurement values, addresses, amounts, and other value objects as components; reserve entity status for things that need discovery, identity, and graph-level deduplication.

**Never nest the same rich entity model at several paths.** If `Garantie` (with its full subtree of conditions, exclusions, ceilings) appears under `offres[].garanties_incluses[]`, `offres[].garanties_optionnelles[]` *and* `options[].etend_garanties[]`, the catalog triples and the same real-world entity gets discovered and filled several times. Instead: give the entity **one root-level home** where full detail is extracted once, and make the other occurrences **name-only references** (same model, same identity field, description says "reference by name only"). Graph assembly deduplicates entities by identity, so the shallow references merge into the fully-filled node.

## Identity and resolution primitives

Identity fields feed three deterministic mechanisms: cross-batch deduplication, the parent **rescue ladder** in graph merge, and the **skeleton reconciliation** pass. Design ids for these mechanisms and orphan loss disappears.

**The canonical form is forgiving about surface noise, strict about words and digits.** Dedup keys are case-insensitive and strip punctuation (`Run-1` ≡ `run_1` ≡ `RUN1`; names additionally get entity-name normalization). So you never need validators to fix casing. What canonicalization can *not* absorb: different words, and different digits.

**Rules for `graph_id_fields`:**

1. **Required, always.** An optional identity field invites empty ids, and children referencing an id-less parent are only saved by the last rescue rung (a shared bucket parent) — reliable, but lossy about which sibling the child belonged to.
2. **One scalar field; two at most.** Every extra id field is another value the model must reproduce exactly when a child references its parent. Never use list-valued fields (they stringify inconsistently across batches) or enums (surface forms vary: `"binder"` vs `"Polymer Binder"`) as identity.
3. **Verbatim-copyable, never invented.** The id must be a label the model can copy from the page: a code, a proper name, a figure/table label. Never instruct fallback generation (`"ITEM-1"`, `"Offer 2"`): invented ids differ between batches, so nothing matches, and positional labels defeat the skeleton's preference for the most specific proper name.
4. **Put distinguishing digits in the id.** Aggressive dedupe (`dense_dedupe="aggressive"`) fuzzy-merges near-identical same-path ids to absorb OCR noise, but ids that differ in **any digit run never merge** (`LFP_20vol` vs `LFP_30vol` stay distinct by design). Encode parameter variants numerically (`Batch-20vol`) and they are guard-protected; encode them in prose ("the higher loading batch") and they are one OCR hiccup away from a false merge.
5. **Name id fields honestly.** A pipeline invariant clears root identity values whose *field name* promises a number (`document_number`, `ref_no`, …) but whose value is multi-word, digit-free prose — that pattern is a mis-capture (a brand or title grabbed for lack of a real number). If the natural identity of your document is a name, call the field `title` or `name`; reserve `*_number` fields for values that actually contain digits.

**How the rescue ladder consumes ids.** When a filled child's parent reference doesn't match any parent instance exactly, the merge tries: unique parent instance (only one candidate exists) → unique fuzzy containment match on canonical ids → materialized placeholder parent → shared bucket parent. Schema implications: parents that commonly have a single instance per document (a protocol, a setup) tolerate id drift well; parents distinguished only by long ids drift more, so keep sibling parents' ids **short and mutually distinct early in the string**.

**How reconciliation consumes ids.** One id-space LLM pass merges instances that are the *same entity at different granularity* — a generic alias (`"LFP slurry batch"`) into its specific form (`"LFP_20vol_5wtPVDF"`). It is forbidden from merging instances that differ by any parameter, quantity, date, or index. This recovers generic/specific duplication automatically — but only when ids are label-like. Sentence-long ids look like distinct descriptions and stay unmerged.

## Deterministic grounding readiness

The pipeline grounds values in the source deterministically; the schema must not fight it.

- **Never ask for computation in a description.** The fill contract forbids computing, rounding, unit conversion, or aggregation. A description like *"'median of 130 nm' → 0.13 µm"* instructs the model to violate that contract and produces silently wrong numbers. Extract verbatim (`130`, `"nm"`); convert downstream in your own code or validators.
- **Give totals a home.** Summary-row values are steered to document-level fields. If your root has `subtotal` / `tax_total` / `total_amount` fields, the total lands correctly; if not, it lands in a row instance or is dropped.
- **Identity values are self-healing; properties are not.** Phase 2 restores identity values from the Phase 1 skeleton whenever the fill response omits or mangles them. Data you cannot afford to lose about *which* instance this is belongs in the id, not in a property.
- **Scoped provenance.** Each fill call receives only the document regions where the instance was observed (plus the document head). Entities whose attributes are described near their mentions fill accurately by default. If your domain scatters attributes far from mentions (e.g. a glossary defining items listed elsewhere), set `dense_fill_context="full"` rather than distorting the schema.
- **Verify with run stats.** Every dense run writes `results.dense` to `metadata.json` (skeleton nodes, truncations, splits, reconciled aliases, recovered/dropped links, retention %). After a schema change, compare retention and drops — schema regressions show up here before you notice them in the graph.

## Multi-LLM resilience

The same template must degrade gracefully on a small local model and scale on a large API model. The pipeline salvages partial output at every stage — per-node skeleton validation skips malformed entries, short fill responses are padded, invalid fields are pruned before final validation — but salvage can only save what the schema lets survive.

1. **Required = identity, nothing else.** Phase 2 pads short fill responses with empty objects so instances are never silently dropped; ids are restored from the skeleton. If a non-identity field is required, every padded or partial instance fails validation and gets pruned — an all-or-nothing loss a smaller model will hit constantly. Make every property optional (`| None`, `default_factory=list`) or give it a safe default.
2. **Validators normalize; they never reject.** Use `mode="before"` validators to coerce what small models actually emit — scalars where objects are expected, strings for numbers (`"1 500 €"` → `1500.0`), stringified lists (`"['a','b']"`), comma-joined names. Rejecting these shapes discards data the model correctly found.
3. **Enums: synonyms in the description, `OTHER` as the safety net, default where sensible.** List the document phrasings that map to each member ("map 'parallel plate' or 'parallel disk' to 'Plate-Plate'"), normalize with a lenient validator, and fall back to `OTHER` instead of failing. A defaulted enum (`role: Role = Role.OTHER`) survives omission entirely.
4. **Edges optional by default.** Audit your `edge()` helper: an edge that silently becomes a required field (e.g. treating `default=None` as "no default") makes one missing relationship kill an otherwise valid document.
5. **Deduplicate identity-less root lists in a validator.** Chunked extraction can deliver the same item (an author, a keyword) from several batches. For list fields without entity identity, add a root `model_validator(mode="after")` that keeps the first occurrence per normalized key. Dedup logic lives in the template — the pipeline stays domain-agnostic.
6. **Keep direct-contract templates flat.** The direct contract returns the whole result in one response; documents much larger than the model's output budget will truncate regardless of schema (the backend warns and suggests dense). If your template is deep and instance-rich, it is a dense template — design its ids accordingly.

## Common failure causes

- Optional, list-valued, enum, or invented identity fields.
- Required non-identity fields (all-or-nothing validation loss on small models).
- Descriptions instructing computation, unit conversion, or id generation.
- The same rich entity model nested at multiple paths.
- Long free-text identifiers (truncation pressure, unreconcilable duplicates).
- Positional labels (`"Offer 1"`) where the document gives proper names.
- Number-named fields (`*_number`, `ref_*`) holding prose values.
- Boilerplate rules repeated per field that the prompts already enforce globally.
