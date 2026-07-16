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

The documents are converted with Docling, then three focused LLM passes propose classes, fields, and relationships as structured data. Deterministic gates filter hallucinations — most importantly the **verbatim gate**: every identity example must appear verbatim in the source document, or it is dropped. Candidates from multiple documents merge deterministically (type promotion `int → float → str`, majority votes, example union).

| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output module path (default `templates/<snake_name>.py`) |
| `--name`, `-n` | Root model class name |
| `--provider` / `--model`, `-m` | LLM settings (default: `config.yaml` `models.llm`) |
| `--spec-out` | Where to write the editable SPEC YAML (default `<output>.spec.yaml`) |
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
  input_budget_chars: 24000   # per-document markdown budget (also bounded by the model's context window)
  max_models: 30
  max_enum_members: 24
  ontology_depth: 4
  llm_gap_fill: false
  strict: false
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
