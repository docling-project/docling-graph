# template Command


## Overview

The `template` command group generates, lints, and empirically evaluates extraction templates — the Pydantic models that drive every conversion.

Generation is built on a two-stage architecture: first a compact, machine-validated **SPEC** (entities, components, fields, relationships, identity, cardinalities, enums) is produced — by an LLM from example documents, or by a deterministic compiler from an ontology — then a deterministic renderer turns the SPEC into a Python template module. **No LLM ever writes code.** Every rule from the [schema definition guide](../../fundamentals/schema-definition/index.md) is enforced by the SPEC's validators, a rulebook linter with automatic repairs, or the renderer itself, and each generated file is verified against the actual runtime (structured-output schema, dense catalog walk, graph conversion smoke test) before it is written.

| Subcommand | Purpose | LLM? |
|------------|---------|------|
| `from-docs` | Induce a template from example documents | Yes |
| `from-ontology` | Compile a template from OWL/RDFS/SKOS, LinkML, or JSON Schema | **No** (optional gap-fill) |
| `from-spec` | Re-render a template from a (hand-edited) SPEC YAML | No |
| `lint` | Check an existing template module against the rulebook | No |
| `evaluate` | Run real extractions and report empirical template-quality signals | Uses your configured backend |

---

## from-docs — from example documents

```bash
uv run docling-graph template from-docs invoice1.pdf invoice2.pdf \
    --output templates/invoices.py \
    --name InvoiceDocument \
    --provider mistral --model mistral-medium-latest \
    --trial-run
```

Sources can be file paths or `http(s)://` URLs — URLs are fetched and converted by Docling like any other document:

```bash
uv run docling-graph template from-docs invoice1.pdf https://example.com/samples/invoice2.pdf
```

The documents are converted with Docling, then the LLM proposes classes, fields, and relationships. Deterministic gates filter hallucinations — most importantly the **verbatim gate**: every identity example must appear verbatim in the source document, or it is dropped. Candidates from multiple documents merge deterministically (type promotion `int → float → str`, majority votes, example union), and a deterministic renderer emits the module — no LLM ever writes code.

**Strategies** (`--strategy`, default `templategen.strategy` = `one-shot`):

- `one-shot` — **one plain-JSON LLM call per document/window** carrying the condensed schema-definition rulebook, returning the full ontology (classes + fields + edges) at once. This is the programmatic form of the proven manual workflow (documents + the schema guides into a strong model, ontology out); the deterministic merge/linter/renderer replace the "now write the Pydantic" step. No grammar-constrained decoding — it works with small local models that degenerate under guided JSON grammars, and costs ~1/8 of the calls.
- `three-pass` — the focused inventory → fields → relationships passes under strict structured output. More per-field evidence for models that handle guided decoding well.

Both strategies run the exact same evidence gates, cross-document merge, and rulebook repair.

If a pass's output overflows the model's `max_tokens` budget, the call is retried once with a doubled budget (capped at half the context window); a fields pass that still overflows is split into smaller class batches automatically. The fields pass is also **pre-sized** against the model's output budget (~1500 output tokens per class), so small-budget models make more, smaller calls instead of truncating. Persistent truncation warnings mean the model's output budget is too small for your documents — raise `llm_overrides.max_output_tokens` in `config.yaml` or switch to a larger model.

Induction LLM calls run **concurrently** (`--workers`, default `templategen.workers` = 4): documents in parallel, and fields batches in parallel within each document. Results are deterministic regardless of the worker count — candidates merge in source order and batch payloads apply in batch order. Use `--workers 1` for strictly sequential calls (e.g. against rate-limited APIs).

**Scale** is handled by two mechanisms sharing one "induction unit" abstraction:

- **Oversized documents** (text beyond `input_budget_chars`, ~24k chars) split into up to `templategen.max_windows_per_doc` (6) evenly spread, line-aligned **windows** — each inducted as its own unit and merged back as one document. A 200-page report contributes evidence from its whole body instead of one head-biased sample, and each window's verbatim gate checks against text the model actually saw.
- **Large corpora** (more than 10 units) process in a deterministic name-hash order (decorrelating vendor/date filename grouping) with a **saturation stop**: once 6 consecutive units add no new classes and essentially no new fields, the rest is skipped and reported — the schema has converged, and documents 20–100 would only repeat it. `templategen.max_units` (24) is the hard cap on top.

Pass `--exhaustive` to disable both the saturation stop and the unit cap. Skipped units are always listed in the induction report (`skipped_saturated` / `skipped_capped`), never dropped silently.

| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output module path (default `templates/<snake_name>.py`) |
| `--name`, `-n` | Root model class name |
| `--provider` / `--model`, `-m` | LLM settings (default: `config.yaml` `models.llm`) |
| `--spec-out` | Where to write the editable SPEC YAML (default `<output>.spec.yaml`) |
| *(always written)* | `<output stem>.report.md` — the full generation detail (induction stats, merge decisions, repair log, gaps, verification, semantic guide); the console shows one metric line per concern |
| `--llm-gap-fill / --no-llm-gap-fill` | One extra call to fill missing docstrings/examples — it structurally cannot add classes, fields, or edges |
| `--trial-run` | After generation, run a real extraction on the first document and print an advisory quality report |
| `--strict / --no-strict` | Fail instead of auto-repairing rulebook violations |
| `--force` | Overwrite without confirmation |

## from-ontology — deterministic compilation, zero LLM

```bash
uv run docling-graph template from-ontology schema.ttl \
    --root ex:InsurancePolicy \
    --output templates/policy.py
```

Supported inputs (`--format auto` sniffs): **OWL/RDFS/SKOS** (`owl:hasKey`/`InverseFunctionalProperty` → identity fields, object properties → edges, `maxCardinality` → instance bounds, `owl:oneOf`/SKOS schemes → enums, subclass hierarchies flattened), **LinkML** (`identifier: true` → identity, `inlined: false` → reference edges, `permissible_values` → enums), and **JSON Schema**. Classes without identity evidence are demoted to components — ids are never invented. Requires the optional extra:

```bash
pip install 'docling-graph[templategen]'
```

| Option | Description |
|--------|-------------|
| `--root`, `-r` | Root class (IRI, CURIE, or unique local name) |
| `--format`, `-f` | `owl`, `linkml`, `jsonschema`, or `auto` |
| `--depth` | Max traversal depth from the root (default 4) |
| `--include` / `--exclude` | Globs over class local names to prune large ontologies |
| `--llm-gap-fill` | Optional single LLM call for missing docstrings/examples (off by default — the path is otherwise 100% deterministic) |

## from-spec — the escape hatch

Every generator writes an editable SPEC YAML next to the template. Rename an edge label or flip an entity to a component with a one-line YAML edit, then re-render:

```bash
uv run docling-graph template from-spec templates/invoices.spec.yaml -o templates/invoices.py
```

## lint — check any existing template

```bash
uv run docling-graph template lint templates.invoices.InvoiceDocument
```

Reconstructs a SPEC from the live classes and reports every rulebook finding — including the reserved-attribute collision (a field named `label`, `type`, or `id` silently corrupts graph node attributes), missing `default_factory` on list edges, raising validators, and edge-label violations. Also prints the exact 240-character docstring window dense extraction will see and the semantic guide the LLM receives. Report-only by default (exit 0); `--strict` exits 1 on findings. Module imports are gated by an allowlist; pass `--no-import-check` for trusted local files with extra imports.

## evaluate — empirical quality signals

```bash
uv run docling-graph template evaluate templates.invoices.InvoiceDocument doc1.pdf doc2.pdf
```

Runs real extractions with your configured backend and reports what static linting cannot see: graph audit signals translated to the rulebook clause they violate (e.g. empty identity nodes → "the document does not NAME these instances — demote to component"), per-class field fill-rates, grounding precision against the provenance ledger, and synthetic root-id flags. Advisory only.

---

## Configuration

Template commands read `config.yaml` only for LLM settings and an optional block (no `config.yaml` is needed for `from-ontology`/`from-spec`/`lint`):

```yaml
templategen:
  input_budget_chars: 24000   # per-unit markdown budget (also bounded by the model's context window)
  max_models: 30
  max_enum_members: 24
  ontology_depth: 4
  llm_gap_fill: false
  strict: false
  strategy: one-shot          # one-shot (1 plain-JSON call/unit) | three-pass (strict grammars)
  workers: 4                  # concurrent induction LLM calls
  max_units: 24               # hard cap on induction units (documents/windows); 0 = unlimited
  max_windows_per_doc: 6      # windows an oversized document may split into
  saturation_stop: true       # stop once the schema stops changing (corpora > 10 units)
```

---

## Python API

```python
from docling_graph.templategen import (
    spec_draft_from_ontology, repair_draft, render_template,
    verify_template_source, induce_spec_from_documents,
    spec_from_dotted_path, evaluate_template, TemplateSpec,
)

draft, gaps = spec_draft_from_ontology("schema.ttl", root="ex:Policy")
spec, lint_report = repair_draft(draft)
source = render_template(spec)
report = verify_template_source(source, root_class=spec.root, spec=spec)
```

Or the one-shot convenience, wrapping draft → repair → render → verify (plus an atomic write when `output` is given — no overwrite prompt, so pass a fresh path):

```python
from docling_graph.templategen import generate_template

result = generate_template("schema.ttl", kind="ontology", root="ex:Policy",
                           output="templates/policy.py")
result.verification.passed  # the file is only written when this is True
result.spec, result.lint_report, result.gaps, result.source_code, result.written_path
```

`kind="spec"` re-renders a SPEC YAML; `kind="docs"` (one or many documents) requires an injected `llm_call_fn`.

### Building an `llm_call_fn`

Induction does not create LLM clients itself — it calls an injected `llm_call_fn`, so you stay in control of the provider, model, and credentials. `build_llm_call_fn` binds one to a LiteLLM client for you:

```python
from docling_graph.templategen import build_llm_call_fn

llm_call_fn = build_llm_call_fn("mistral", "mistral-small-latest")
```

The returned callable follows the induction contract (`llm_call_fn(*, prompt, schema_json, context)`) and is **thread-safe**: induction fans calls out across `--workers`, so each thread gets its own client.

It also owns truncation handling — if a response comes back truncated, it retries once with `max_tokens` doubled (capped at half the model's context window, and never more than 2x the configured budget). A model stuck in a repetition loop truncates at any budget, so the ceiling stops a runaway from turning junk into minutes of generation time; you get a warning instead.

Optional arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `llm_overrides` | `LlmRuntimeOverrides` \| `dict` \| `None` | `None` | Runtime overrides (e.g. `{"max_output_tokens": 8192}`), same shape as the `llm_overrides` block in `config.yaml` |
| `structured_output` | `bool` | `True` | `True` uses a schema grammar (pair with `strategy="three-pass"`); `False` requests plain `json_object` decoding (pair with `strategy="one-shot"`) |

!!! warning "Pair `structured_output` with your `strategy`"

    The two settings live on different functions — `structured_output` on `build_llm_call_fn`, `strategy` on `induce_spec_from_documents` — and nothing cross-checks them. Mismatching them silently applies grammar-constrained decoding to the `one-shot` strategy, the exact combination `one-shot` exists to avoid for small local models.

    | Strategy | `structured_output` |
    |:---------|:--------------------|
    | `three-pass` | `True` |
    | `one-shot` | `False` |

    The defaults line up (`build_llm_call_fn` → `True`, `induce_spec_from_documents` → `strategy="three-pass"`), so leaving both alone is safe. Note the **CLI** defaults to `one-shot` instead and wires `structured_output` for you; when you switch a programmatic call to `one-shot`, you must flip `structured_output` yourself:

    ```python
    llm_call_fn = build_llm_call_fn(
        "ollama",
        "gemma3:12b",
        llm_overrides={"max_output_tokens": 8192},
        structured_output=False,          # ← must match ...
    )

    spec, report = induce_spec_from_documents(
        ["invoice1.pdf"],
        llm_call_fn,
        strategy="one-shot",              # ← ... this
    )
    ```

Construction is offline: it fails fast if `litellm` is missing, but it does **not** validate credentials or reject unknown providers (those fall back to generic defaults with a warning). A bad API key first surfaces as a `ClientError` from the returned callable's first real call.

### Inducing a SPEC

Document sources for `kind="docs"` / `induce_spec_from_documents` can be file paths, `http(s)://` URLs, or — when you already hold the text — `DocumentContent` objects passed directly, no file needed:

```python
from docling_graph.templategen import (
    DocumentContent, build_llm_call_fn, induce_spec_from_documents,
)

llm_call_fn = build_llm_call_fn("mistral", "mistral-small-latest")

spec, report = induce_spec_from_documents(
    [
        "invoice1.pdf",                                          # file
        "https://example.com/samples/invoice2.pdf",              # URL (converted by Docling)
        DocumentContent(name="invoice3", text=markdown_string),  # direct content
    ],
    llm_call_fn,
)
```
