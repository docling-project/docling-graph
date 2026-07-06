# Best Practices

How to design Pydantic templates that work *with* the extraction pipeline instead of against it. Every rule below maps to a concrete, domain-agnostic pipeline mechanism; the schema is where domain knowledge lives, and these are the contract points between the two.

## Quick checklist

- Root and every entity define `graph_id_fields`; identity fields are **required, scalar, short, and copyable verbatim** from the document.
- Non-identity fields are **optional** (`| None` or `default_factory=list`) — required-ness belongs to identity only.
- Components (`is_entity=False`) for value objects; entities only where discovery and deduplication matter.
- `edge(label=...)` on relationship-bearing fields; edges optional by default.
- `edge(..., reference=True)` on every **identity-only link** — membership lists, cross-references, links to shared identity nodes. Full detail lives at the target's one canonical home.
- Field descriptions: 1–3 sentences (where to look + normalization rule); don't restate pipeline-level rules.
- Validators normalize; they never reject the whole payload.
- 2–4 nesting levels; never nest the same rich entity model at several paths.
- An entity referenced from several paths stays **identity-minimal**; context-specific data (a role, a title) lives on per-context entities linking to it — duplicate-instance merge is fill-missing-only, first non-empty value wins.
- Class docstrings **front-load the discriminating sentence** (what this entity is and what it is NOT) — the first ~240 characters reach Phase 1, where classification happens.

---

## Token economics and compacting

Dense extraction discovers entities in Phase 1 (skeleton) and fills them in Phase 2. Both phases have strict token budgets, and the schema directly controls how much of that budget is spent.

**What the skeleton phase actually sees.** Phase 1 receives one guide line per entity path: its `graph_id_fields` names, the **first ~240 characters of the class docstring**, and up to three `examples` per identity field (each truncated to 50 chars). Property fields, their types, and their descriptions are *invisible* in Phase 1. Consequences:

- The **docstring's opening sentence decides classification.** Phase 1 places instances at paths using these descriptions ("an Option is not an Offre", "one clause per Exclusion, never one per keyword", "categories, not enumerated elements"). Front-load what the entity is and what it is NOT; anything past ~240 chars never reaches Phase 1.
- Identity field **`examples` are your other Phase 1 lever** — the only concrete guidance for what an id should look like. Give 2–5 short, document-derived examples.
- Don't move load-bearing extraction hints into property descriptions expecting them to shape entity *discovery*; they only shape Phase 2 *filling*.

**Skeleton output is identity-only and handle-linked.** Each discovered node is emitted as `{"i": <int>, "path", "ids", "p": <parent handle>}` — the parent is a one-digit integer reference, not a repeated ancestry object. The remaining output cost is therefore the **ids themselves**, once per node. Short ids (`"FIG-4"`, `"Batch-1"`, `"CONFORT"`) keep entity-dense batches inside the model's output budget. Long ids — titles, sentences, descriptions-as-ids — inflate every node and push batches into truncation, which triggers the recovery cascade: one escalated retry, then recursive batch splitting. Recovery is lossless but multiplies LLM calls and wall-clock time. **The cheapest truncation fix is a shorter identity field.**

**Field descriptions are paid for on every fill call.** Phase 2 sends each entity's projected JSON schema — descriptions included — with every fill batch. Keep descriptions to a "LOOK FOR" locator plus one normalization rule. Do **not** repeat rules the fill prompt already enforces globally:

- omit absent values (never `"N/A"` / `"Not specified"` placeholders),
- copy numbers digit-for-digit (never compute, round, or aggregate),
- summary-row totals belong to document-level fields, not row instances.

Restating these per field costs tokens on every call and adds nothing.

**Entities cost catalog space; components don't.** Every entity model at every path adds a catalog line to Phase 1 and a fill pass to Phase 2. A component (`is_entity=False`) is filled inline with its parent. Model measurement values, addresses, amounts, and other value objects as components; reserve entity status for things that need discovery, identity, and graph-level deduplication.

**Never nest the same rich entity model at several paths — declare references.** If `Garantie` (with its full subtree of conditions, exclusions, ceilings) appears under `offres[].garanties_incluses[]`, `offres[].garanties_optionnelles[]` *and* `options[].etend_garanties[]`, the catalog multiplies and the same real-world entity gets discovered and filled several times. Give the entity **one canonical home** where full detail is extracted once, and mark every other occurrence `edge(..., reference=True)` (sets `json_schema_extra={"graph_reference": True}`). A reference field gets **no catalog path at all**: the *parent's own fill call* emits it as an id-only list (`[{"nom": "Bris de vitre"}]`), and graph assembly resolves the ids onto the canonical node. See "Reference edges vs nested full entities" below for when to use which.

## Reference edges vs nested full entities

`edge(..., reference=True)` declares that a field carries **identity-only links** to entities, not the entities themselves. It is the single most important structural decision for graph quality in dense extraction, because it changes how the pipeline treats the field:

| | Nested full entity (default) | `reference=True` |
|---|---|---|
| Catalog / skeleton | Own discovery path (and sub-paths) | No path — invisible to Phase 1 |
| Filled by | Its own fill calls | The **parent's** fill call, projected to `graph_id_fields` only |
| Parent linkage | Skeleton parent handles (can drift) | None needed — the refs live inside the parent |
| Graph node | Full node (merged by identity) | Id-only instance resolved onto the canonical node |

**Use `reference=True` when:**

- the target's full detail is extracted at another, canonical path (offer→guarantee membership, acquisition→segment assignment), or
- the target is **identity-only by design** — a shared identity node like `Person`, or
- the field expresses membership/cross-reference semantics (a table column of names, a "see also" list).

**Keep a nested full entity when** this field is the *only* place the target's attributes can be extracted — a reference would leave the node with nothing but its name.

**Why this matters (observed failure modes it removes):**

1. **Per-parent membership collapse.** Skeleton dedup keys instances by `(path, ids)` — without parents. Four offers each including "Dégâts des eaux" collapse into ONE discovered instance with one parent; three memberships vanish. Reference fields ride inside each parent's fill output, so every parent keeps its own list.
2. **Parent drift → phantom hubs.** Separately-discovered reference children must name their parent; small models drift or dangle those references, and the rescued orphans historically pooled under id-less bucket parents that validation then deleted (taking all membership edges along) or salvage blanked into `""`-named hub nodes. Reference fields have no parent reference to drift.
3. **Catalog and cost explosion.** Each nested occurrence of a rich entity multiplies discovery paths and fill calls (an MRH insurance schema dropped from 22 paths to 6 by declaring its reference fields).

The marker is honored only when the target model declares `graph_id_fields`; on identity-less targets it is ignored. Direct (single-call) extraction is unaffected — reference fields keep their normal schema there, and graph assembly already merges id-only instances by identity in both contracts.

## Identity and resolution primitives

Identity fields feed three deterministic mechanisms: cross-batch deduplication, the parent **rescue ladder** in graph merge, and the **skeleton reconciliation** pass. Design ids for these mechanisms and orphan loss disappears.

**The canonical form is forgiving about surface noise, strict about words and digits.** Dedup keys are case-insensitive and strip punctuation (`Run-1` ≡ `run_1` ≡ `RUN1`; names additionally get entity-name normalization). So you never need validators to fix casing. What canonicalization can *not* absorb: different words, and different digits.

**Rules for `graph_id_fields`:**

1. **Required, always.** An optional identity field invites empty ids, and children referencing an id-less parent are only saved by the last rescue rung (a shared bucket parent) — reliable, but lossy about which sibling the child belonged to.
2. **One scalar field; two at most.** Every extra id field is another value the model must reproduce exactly when a child references its parent. Never use list-valued fields (they stringify inconsistently across batches) or enums (surface forms vary: `"binder"` vs `"Polymer Binder"`) as identity.
3. **Verbatim-copyable, never invented.** The id must be a label the model can copy from the page: a code, a proper name, a figure/table label. Never instruct fallback generation (`"ITEM-1"`, `"Offer 2"`): invented ids differ between batches, so nothing matches, and positional labels defeat the skeleton's preference for the most specific proper name.
4. **Put distinguishing digits in the id.** Aggressive dedupe (`dense_dedupe="aggressive"`) fuzzy-merges near-identical same-path ids to absorb OCR noise, but ids that differ in **any digit run never merge** (`LFP_20vol` vs `LFP_30vol` stay distinct by design). Encode parameter variants numerically (`Batch-20vol`) and they are guard-protected; encode them in prose ("the higher loading batch") and they are one OCR hiccup away from a false merge.
5. **Name id fields honestly.** A pipeline invariant clears root identity values whose *field name* promises a number (`document_number`, `ref_no`, …) but whose value is multi-word, digit-free prose — that pattern is a mis-capture (a brand or title grabbed for lack of a real number). If the natural identity of your document is a name, call the field `title` or `name`; reserve `*_number` fields for values that actually contain digits.
6. **Canonical name for identity, descriptors as properties.** When an entity is identified by a descriptive name (a material, a product), steer the model — through the field description and `examples` — to the *canonical* short form (an abbreviation or formula) and keep descriptive qualifiers (size, morphology, grade) in separate property fields. Otherwise the same real thing arrives under several surface forms (`LiFePO4 (LFP)`, `nanoscaled LiFePO4 (LFP)`) and becomes several nodes. Containment pairs (a superset id alongside its base) are *proposed* deterministically and merged only when the reconciliation LLM confirms they denote the same entity — tier names like `CONFORT` vs `CONFORT PLUS` are explicitly protected. The reliable fix is still to never put the qualifier in the id in the first place.

**How the rescue ladder consumes ids.** When a filled child's parent reference doesn't match any parent instance exactly, the merge tries: unique parent instance (only one candidate exists) → unique fuzzy containment match on canonical ids → unique co-located parent (discovered in the same source chunk/batch as the child) → materialized placeholder parent → shared bucket parent. Placeholders and buckets are only materialized when the parent model does **not require** the missing identity fields; with a required identity the orphan is dropped and counted instead — an id-less rescue parent would be deleted by validation (taking every rescued child with it) or salvaged into a phantom `""`-identity hub. Schema implications: parents that commonly have a single instance per document (a protocol, a setup) tolerate id drift well; parents distinguished only by long ids drift more, so keep sibling parents' ids **short and mutually distinct early in the string** — and prefer `reference=True` fields, which bypass parent references entirely.

**How reconciliation consumes ids.** One id-space LLM pass merges instances that are the *same entity at different granularity* — a generic alias (`"LFP slurry batch"`) into its specific form (`"LFP_20vol_5wtPVDF"`). It is forbidden from merging instances that differ by any parameter, quantity, date, or index. This recovers generic/specific duplication automatically — but only when ids are label-like. Sentence-long ids look like distinct descriptions and stay unmerged.

## Graph assembly mechanics

Four converter behaviors the schema should be designed around:

1. **Duplicate instances merge fill-missing-only.** When the same entity is discovered at several paths (e.g. one person in both the board list and the executive list), all instances collapse into one node: later instances fill attributes the first left empty, but **conflicting non-empty values keep the first-seen value**. So when one real-world thing plays several roles with role-specific data, keep the shared entity **identity-minimal** (its id field and nothing else) and model each role as its own entity — identified by the same name, holding that role's fields, linking to the shared node via an edge:

    ```python
    class Person(BaseModel):
        """Identity-only shared node — safe to reference from anywhere."""
        model_config = ConfigDict(graph_id_fields=["full_name"])
        full_name: str = Field(...)

    class BoardMember(BaseModel):
        """Per-role record: role data lives here, never on the shared Person."""
        model_config = ConfigDict(graph_id_fields=["full_name"])
        full_name: str = Field(...)
        title: str | None = Field(None)
        person: Person | None = edge(label="IS_PERSON", reference=True)
    ```

    Role fields on the shared node would resolve arbitrarily (whichever list was processed first wins); per-role entities never conflict, and dense extraction discovers each role as its own catalog instance. The `reference=True` on the shared-identity link means the Person ref is emitted by the role's own fill call — an identity-only node needs no discovery of its own.

2. **Alias reconciliation merges table labels into section titles — with an LLM veto.** After cleanup, same-class nodes whose canonical identities are containment pairs (`Attentat` inside `Attentat et actes de terrorisme`) are proposed for merging and applied only when an id-space LLM call confirms they denote the same entity; the attribute-richer node survives and records the absorbed identity under `merged_aliases`. Tier names (`CONFORT` vs `CONFORT PLUS`) are protected by both the prompt guard and the requirement that only deterministically-proposed pairs may merge. Schema leverage: reference fields SHOULD use the name as printed where they occur (a table's short label) — reconciliation reunites it with the section-titled node.

3. **Entities nested inside components attach to the nearest entity ancestor.** Component subtrees are traversed: an entity below a component still becomes a node, and its edge comes from the closest enclosing *entity* (the component itself never gets a node — its scalar fields embed in that ancestor, with nested-entity fields nulled). This mirrors how the dense catalog parents such entities. It works, but prefer nesting entities directly under entities: the wrapper component adds a level of indirection without adding a graph node.

4. **Verify the graph shape, not just validation.** `Model(**sample)` proves the schema validates; it says nothing about what the graph looks like. Convert a hand-built instance and inspect nodes and edges before running extraction:

    ```python
    from docling_graph.core.converters.graph_converter import GraphConverter

    graph, _ = GraphConverter().pydantic_list_to_graph([sample_instance])
    for node_id, data in graph.nodes(data=True):
        print(data["label"], node_id)
    for u, v, data in graph.edges(data=True):
        print(u, f"--{data['label']}-->", v)
    ```

    This catches misclassified entities/components, missing edges, and one-node-instead-of-two identity collisions in seconds, using exactly the code path the pipeline runs.

## Deterministic grounding readiness

The pipeline grounds values in the source deterministically; the schema must not fight it.

- **Never ask for computation in a description.** The fill contract forbids computing, rounding, unit conversion, or aggregation. A description like *"'median of 130 nm' → 0.13 µm"* instructs the model to violate that contract and produces silently wrong numbers. Extract verbatim (`130`, `"nm"`); convert downstream in your own code or validators.
- **Give totals a home.** Summary-row values are steered to document-level fields. If your root has `subtotal` / `tax_total` / `total_amount` fields, the total lands correctly; if not, it lands in a row instance or is dropped.
- **Per-line vs document tax.** A tax attached to a line item is *that line's* tax; a tax in the totals block is document-level. Instruct a line-level tax field to hold the line's own tax and to be **omitted when the line does not state one** — otherwise the model copies the single whole-document VAT total onto every line. Always give the document its own `tax_total` so the whole-invoice figure has a home and never has to borrow a line field.
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
- Role/context data on an entity referenced from several paths (conflicting duplicate values resolve first-seen-wins; see [Graph assembly mechanics](#graph-assembly-mechanics)).
