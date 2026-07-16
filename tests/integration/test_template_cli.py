"""
End-to-end integration tests for the `docling-graph template` CLI sub-app.

Zero network: the ontology path is deterministic (zero LLM by design), the
documents path runs against a scripted LLM client (the canned pass payloads
from tests/unit/templategen/test_induce.py), and `evaluate`/`--trial-run` run
against a patched `run_pipeline` returning a synthetic PipelineContext.
"""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import networkx as nx
import pytest
import yaml
from typer.testing import CliRunner

from docling_graph.cli.commands.template import _rename_root_model
from docling_graph.cli.main import app
from docling_graph.core.converters.graph_converter import GraphConverter
from docling_graph.core.extractors.contracts.dense.catalog import build_node_catalog
from docling_graph.pipeline.stages import TemplateLoadingStage
from docling_graph.templategen import SpecGap, TemplateSpec, synthesize_sample
from tests.unit.templategen.test_induce import ScriptedLLM, invoice_script

REPO_ROOT = Path(__file__).parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "templategen"
ONTOLOGIES = FIXTURES / "ontologies"
DOCS = FIXTURES / "docs"


@pytest.fixture
def cli_runner():
    return CliRunner()


def _load_written_template(dotted: str, spec_path: Path) -> tuple[type, TemplateSpec]:
    """Load a generated template exactly like the pipeline, plus its SPEC."""
    template_cls = TemplateLoadingStage._load_from_string(dotted)
    spec = TemplateSpec.from_yaml(spec_path.read_text(encoding="utf-8"))
    return template_cls, spec


def _graph_smoke(template_cls: type, spec: TemplateSpec) -> nx.DiGraph:
    """synthesize_sample -> real GraphConverter, returning the graph."""
    namespace = vars(sys.modules[template_cls.__module__])
    sample = synthesize_sample(spec, namespace)
    graph, _metadata = GraphConverter().pydantic_list_to_graph([sample])
    return graph


class ScriptedClient:
    """LiteLLMClient stand-in driving get_json_response from a ScriptedLLM.

    The CLI's llm_call_fn sanitizes the context tag into a provider-safe
    ``response_schema_name`` (``^[a-zA-Z0-9_-]+$``, truncated to 40 chars) —
    the sanitized name still carries the pass tag
    (``templategen_pass1_classes_...``) the script routes on. Every schema
    name received is recorded for the compliance assertion.
    """

    def __init__(self, script: ScriptedLLM) -> None:
        self._script = script
        self.schema_names: list[str] = []
        self.last_call_diagnostics: dict[str, Any] = {}

    def get_json_response(
        self,
        prompt: dict[str, str] | str,
        schema_json: str,
        structured_output: bool = True,
        response_top_level: str = "object",
        response_schema_name: str = "extraction_result",
    ) -> Any:
        self.schema_names.append(response_schema_name)
        self.last_call_diagnostics = {"truncated": False}
        return self._script(
            prompt=dict(prompt) if isinstance(prompt, dict) else {"user": str(prompt)},
            schema_json=schema_json,
            context=response_schema_name,
        )


def _stub_effective_config(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(context_limit=32_000, max_output_tokens=4_096, model_id="stub")


def _stub_pipeline_context() -> SimpleNamespace:
    graph = nx.DiGraph()
    graph.add_node("Invoice_1", __class__="Invoice", invoice_number="INV-2024-0113")
    return SimpleNamespace(
        knowledge_graph=graph, provenance=None, extracted_models=[], extractor=None
    )


@pytest.mark.integration
class TestTemplateHelp:
    def test_template_help_lists_subcommands(self, cli_runner):
        result = cli_runner.invoke(app, ["template", "--help"])
        assert result.exit_code == 0
        for subcommand in ("from-docs", "from-ontology", "from-spec", "lint", "evaluate"):
            assert subcommand in result.output


@pytest.mark.integration
class TestFromOntology:
    def test_end_to_end_zero_llm(self, cli_runner, tmp_path, monkeypatch):
        """.ttl -> written template -> pipeline loader -> catalog -> graph smoke."""
        monkeypatch.chdir(tmp_path)
        with patch("docling_graph.llm_clients.get_client") as mock_get_client:
            result = cli_runner.invoke(
                app,
                [
                    "template",
                    "from-ontology",
                    str(ONTOLOGIES / "policy_basic.ttl"),
                    "-o",
                    "gen_onto/policy.py",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_get_client.assert_not_called()  # zero LLM, no provider configured

        output = tmp_path / "gen_onto" / "policy.py"
        spec_path = tmp_path / "gen_onto" / "policy.spec.yaml"
        assert output.is_file()
        assert spec_path.is_file()  # --spec-out defaults to <output>.spec.yaml
        assert "Template written to" in result.output
        assert "V6" in result.output  # verification summary printed

        template_cls, spec = _load_written_template("gen_onto.policy.Policy", spec_path)
        assert spec.root == "Policy"

        catalog = build_node_catalog(template_cls)
        assert catalog.nodes
        assert catalog.nodes[0].node_type == "Policy"

        graph = _graph_smoke(template_cls, spec)
        node_classes = {data.get("__class__") for _, data in graph.nodes(data=True)}
        assert {"Policy", "Guarantee"} <= node_classes
        assert not graph.graph.get("empty_identity_nodes")

    def test_strict_fails_on_repairs(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(
            app,
            [
                "template",
                "from-ontology",
                str(ONTOLOGIES / "policy_basic.ttl"),
                "-o",
                "strict/policy.py",
                "--strict",
            ],
        )
        assert result.exit_code == 1
        assert "Violations (--strict)" in result.output
        assert not (tmp_path / "strict" / "policy.py").exists()  # nothing written

    def test_skos_only_suggests_from_docs_and_writes_enums(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(
            app,
            [
                "template",
                "from-ontology",
                str(ONTOLOGIES / "skos_only.ttl"),
                "--spec-out",
                "skos_enums.yaml",
            ],
        )
        assert result.exit_code == 1
        assert "from-docs" in result.output
        enums = yaml.safe_load((tmp_path / "skos_enums.yaml").read_text(encoding="utf-8"))
        assert enums["enums"][0]["members"] == ["France", "Germany"]

    def test_missing_extra_names_the_install_command(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setitem(sys.modules, "rdflib", None)  # block the import
        result = cli_runner.invoke(
            app,
            ["template", "from-ontology", str(ONTOLOGIES / "policy_basic.ttl")],
        )
        assert result.exit_code == 1
        assert "docling-graph[templategen]" in result.output

    def test_unsniffable_format_asks_for_explicit_flag(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mystery = tmp_path / "mystery.xyz"
        mystery.write_text("::: definitely not an ontology :::", encoding="utf-8")
        result = cli_runner.invoke(app, ["template", "from-ontology", str(mystery)])
        assert result.exit_code == 1
        assert "--format" in result.output

    def test_invalid_format_rejected(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(
            app,
            [
                "template",
                "from-ontology",
                str(ONTOLOGIES / "policy_basic.ttl"),
                "--format",
                "bogus",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid ontology format" in result.output

    def test_overwrite_prompt_and_force(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        args = [
            "template",
            "from-ontology",
            str(ONTOLOGIES / "policy_basic.ttl"),
            "-o",
            "gen_ow/policy.py",
        ]
        result = cli_runner.invoke(app, args)
        assert result.exit_code == 0, result.output

        # Mark the file, decline the prompt: the file must remain untouched.
        output = tmp_path / "gen_ow" / "policy.py"
        marked = output.read_text(encoding="utf-8") + "\n# MARKER\n"
        output.write_text(marked, encoding="utf-8")
        result = cli_runner.invoke(app, args, input="n\n")
        assert result.exit_code == 0
        assert "already exists" in result.output
        assert "Generation cancelled" in result.output
        assert output.read_text(encoding="utf-8") == marked

        # --force regenerates without prompting.
        result = cli_runner.invoke(app, [*args, "--force"])
        assert result.exit_code == 0, result.output
        assert "# MARKER" not in output.read_text(encoding="utf-8")


@pytest.mark.integration
class TestRootRename:
    def test_rename_root_model_readdresses_gaps(self):
        """--name rewrites gap addresses too: gap-fill keys on (model, field,
        kind), so a stale root name would silently make root gaps unfillable."""
        draft = {"root": "Policy", "models": [{"name": "Policy", "fields": []}]}
        gaps = [
            SpecGap(model="Policy", field="policy_number", kind="missing_examples"),
            SpecGap(model="Guarantee", field=None, kind="missing_identity"),
        ]
        _rename_root_model(draft, "PolicyRecord", gaps)
        assert draft["root"] == "PolicyRecord"
        assert gaps[0].model == "PolicyRecord"
        assert gaps[1].model == "Guarantee"  # non-root gaps untouched

    def test_from_ontology_name_flow_readdresses_root_gaps(self, cli_runner, tmp_path, monkeypatch):
        """from-ontology --name: printed gaps address the renamed root, never
        the stale ontology name (jsonschema path: stdlib, no extra needed)."""
        monkeypatch.chdir(tmp_path)
        schema = {
            "title": "Policy",
            "type": "object",
            "required": ["policy_id"],
            "properties": {
                "policy_id": {"type": "string"},
                "premium": {"type": "number"},
            },
        }
        schema_path = tmp_path / "policy.schema.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        result = cli_runner.invoke(
            app,
            [
                "template",
                "from-ontology",
                str(schema_path),
                "--format",
                "jsonschema",
                "--name",
                "PolicyRecord",
                "-o",
                "gen_nm/policy.py",
            ],
        )
        assert result.exit_code == 0, result.output
        gap_section = result.output.split("Open gaps")[1]
        # The identity field has no examples -> a root-addressed gap exists,
        # and it is addressed to the renamed root.
        assert "PolicyRecord.policy_id" in gap_section
        assert "Policy.policy_id" not in gap_section


@pytest.mark.integration
class TestFromSpec:
    def test_round_trip_kind_flip_renders(self, cli_runner, tmp_path, monkeypatch):
        """from-ontology --spec-out -> flip a kind in the YAML -> from-spec."""
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(
            app,
            [
                "template",
                "from-ontology",
                str(ONTOLOGIES / "policy_basic.ttl"),
                "-o",
                "gen_rt/policy.py",
                "--spec-out",
                "gen_rt/policy.spec.yaml",
            ],
        )
        assert result.exit_code == 0, result.output

        spec_path = tmp_path / "gen_rt" / "policy.spec.yaml"
        generated = (tmp_path / "gen_rt" / "policy.py").read_text(encoding="utf-8")
        assert 'graph_id_fields=["guarantee_name"]' in generated  # entity before the flip

        data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        guarantee = next(m for m in data["models"] if m["name"] == "Guarantee")
        guarantee["kind"] = "component"
        guarantee["identity_fields"] = []
        guarantee["max_instances"] = None  # components take no cardinality bound
        for field in guarantee["fields"]:
            if field.get("role") == "identity":
                field["role"] = "property"
        spec_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

        result = cli_runner.invoke(
            app,
            ["template", "from-spec", str(spec_path), "-o", "gen_rt2/policy.py"],
        )
        assert result.exit_code == 0, result.output
        rerendered = (tmp_path / "gen_rt2" / "policy.py").read_text(encoding="utf-8")
        guarantee_block = rerendered.split("class Guarantee")[1].split("class ")[0]
        assert "is_entity=False" in guarantee_block  # the flip rendered
        assert "graph_id_fields" not in guarantee_block

    def test_invalid_spec_yaml_exits_1(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bad = tmp_path / "bad.spec.yaml"
        bad.write_text("root: Missing\nmodels: []\n", encoding="utf-8")
        result = cli_runner.invoke(app, ["template", "from-spec", str(bad)])
        assert result.exit_code == 1
        assert "Invalid template spec" in result.output


@pytest.mark.integration
class TestFromDocs:
    def test_end_to_end_with_scripted_llm(self, cli_runner, tmp_path, monkeypatch):
        """Scripted 3-pass payloads over the markdown fixtures -> template -> graph.

        Full-pipeline extraction mocking is deliberately not exercised here (the
        direct-contract client surface is large); the pipeline boundary is
        covered by --trial-run below and the evaluate test, and the graph loop
        closes through the real GraphConverter.
        """
        monkeypatch.chdir(tmp_path)
        script = invoice_script()
        client = ScriptedClient(script)
        monkeypatch.setattr(
            "docling_graph.llm_clients.get_client", lambda provider: lambda effective: client
        )
        monkeypatch.setattr(
            "docling_graph.llm_clients.config.resolve_effective_model_config",
            _stub_effective_config,
        )

        result = cli_runner.invoke(
            app,
            [
                "template",
                "from-docs",
                str(DOCS / "invoice_1.md"),
                str(DOCS / "invoice_2.md"),
                "--provider",
                "mistral",
                "--model",
                "mistral-small-latest",
                "-o",
                "gen_docs/invoice.py",
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(script.calls) == 6  # 3 passes x 2 documents, nothing else
        assert "Induction report" in result.output
        assert "Template written to" in result.output

        # Every response_format schema name sent to the provider is compliant
        # (OpenAI-family: ^[a-zA-Z0-9_-]+$, <=64 chars) — the raw context tag
        # carries ':' and the source filename's '.' and must never leak.
        assert client.schema_names
        assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", n) for n in client.schema_names)
        assert client.schema_names[0].startswith("templategen_pass1_classes_invoice_1_md")

        output = tmp_path / "gen_docs" / "invoice.py"
        spec_path = tmp_path / "gen_docs" / "invoice.spec.yaml"
        assert output.is_file()
        assert spec_path.is_file()

        template_cls, spec = _load_written_template("gen_docs.invoice.Invoice", spec_path)
        assert spec.root == "Invoice"
        graph = _graph_smoke(template_cls, spec)
        node_classes = {data.get("__class__") for _, data in graph.nodes(data=True)}
        assert {"Invoice", "Party", "LineItem"} <= node_classes

    def test_trial_run_is_advisory(self, cli_runner, tmp_path, monkeypatch):
        """--trial-run drives evaluate_template through a patched run_pipeline."""
        monkeypatch.chdir(tmp_path)
        client = ScriptedClient(invoice_script())
        monkeypatch.setattr(
            "docling_graph.llm_clients.get_client", lambda provider: lambda effective: client
        )
        monkeypatch.setattr(
            "docling_graph.llm_clients.config.resolve_effective_model_config",
            _stub_effective_config,
        )
        with patch(
            "docling_graph.pipeline.run_pipeline", return_value=_stub_pipeline_context()
        ) as mock_run:
            result = cli_runner.invoke(
                app,
                [
                    "template",
                    "from-docs",
                    str(DOCS / "invoice_1.md"),
                    str(DOCS / "invoice_2.md"),
                    "--provider",
                    "mistral",
                    "--model",
                    "mistral-small-latest",
                    "-o",
                    "gen_trial/invoice.py",
                    "--trial-run",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Trial run (advisory)" in result.output
        assert "# Template evaluation" in result.output
        mock_run.assert_called_once()
        config = mock_run.call_args[0][0]
        assert config.provider_override == "mistral"
        assert config.backend == "llm"
        assert config.dump_to_disk is False

    def test_missing_provider_names_init_and_flags(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # no config.yaml here
        result = cli_runner.invoke(app, ["template", "from-docs", str(DOCS / "invoice_1.md")])
        assert result.exit_code == 1
        assert "docling-graph init" in result.output
        assert "--provider/--model" in result.output

    def test_trial_run_consumes_cached_document_json(self, cli_runner, tmp_path, monkeypatch):
        """A converted (non-text) source re-enters --trial-run through the
        cached <stem>.document.json (DOCLING_DOCUMENT input: no re-OCR)."""
        monkeypatch.chdir(tmp_path)
        source = tmp_path / "invoice_scan.pdf"
        source.write_bytes(b"%PDF-1.4 stub")
        markdown = (DOCS / "invoice_1.md").read_text(encoding="utf-8")

        class StubDoclingDocument:
            def export_to_dict(self) -> dict:
                return {"schema_name": "DoclingDocument"}

        class StubProcessor:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            def convert_to_docling_doc(self, _src: str) -> "StubDoclingDocument":
                return StubDoclingDocument()

            def extract_full_markdown(self, _document: Any) -> str:
                return markdown

        monkeypatch.setattr(
            "docling_graph.core.extractors.document_processor.DocumentProcessor",
            StubProcessor,
        )
        client = ScriptedClient(invoice_script())
        monkeypatch.setattr(
            "docling_graph.llm_clients.get_client", lambda provider: lambda effective: client
        )
        monkeypatch.setattr(
            "docling_graph.llm_clients.config.resolve_effective_model_config",
            _stub_effective_config,
        )

        trial_sources: list[tuple[str, bool]] = []

        def record_run(config: Any, mode: str = "api") -> Any:
            source_str = str(config.source)
            trial_sources.append((source_str, Path(source_str).is_file()))
            return _stub_pipeline_context()

        with patch("docling_graph.pipeline.run_pipeline", side_effect=record_run):
            result = cli_runner.invoke(
                app,
                [
                    "template",
                    "from-docs",
                    str(source),
                    "--provider",
                    "mistral",
                    "--model",
                    "mistral-small-latest",
                    "-o",
                    "gen_cache/invoice.py",
                    "--trial-run",
                ],
            )
        assert result.exit_code == 0, result.output
        assert len(trial_sources) == 1
        trial_source, existed_during_run = trial_sources[0]
        assert trial_source.endswith("invoice_scan.document.json")
        assert existed_during_run  # the cache was alive when the pipeline consumed it

    def test_decline_derived_output_keeps_spec(self, cli_runner, tmp_path, monkeypatch):
        """Declining the derived-path prompt (which fires AFTER all LLM spend)
        must keep the SPEC on disk and say how to re-render without re-paying."""
        monkeypatch.chdir(tmp_path)
        client = ScriptedClient(invoice_script())
        monkeypatch.setattr(
            "docling_graph.llm_clients.get_client", lambda provider: lambda effective: client
        )
        monkeypatch.setattr(
            "docling_graph.llm_clients.config.resolve_effective_model_config",
            _stub_effective_config,
        )
        derived = tmp_path / "templates" / "invoice.py"
        derived.parent.mkdir(parents=True)
        derived.write_text("# existing template\n", encoding="utf-8")

        result = cli_runner.invoke(
            app,
            [
                "template",
                "from-docs",
                str(DOCS / "invoice_1.md"),
                str(DOCS / "invoice_2.md"),
                "--provider",
                "mistral",
                "--model",
                "mistral-small-latest",
            ],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "Generation cancelled" in result.output
        assert derived.read_text(encoding="utf-8") == "# existing template\n"
        # The paid induction survives as the SPEC, and the decline message
        # points at it plus the from-spec re-render command.
        spec_path = tmp_path / "templates" / "invoice.spec.yaml"
        assert spec_path.is_file()
        assert TemplateSpec.from_yaml(spec_path.read_text(encoding="utf-8")).root == "Invoice"
        assert "from-spec" in result.output
        assert "invoice.spec.yaml" in result.output

    def test_decline_with_colliding_spec_out_writes_rescue(self, cli_runner, tmp_path, monkeypatch):
        """When the derived spec path itself collides (unconfirmed), the SPEC
        is rescued to <output>.new.spec.yaml and the collision stays untouched."""
        monkeypatch.chdir(tmp_path)
        client = ScriptedClient(invoice_script())
        monkeypatch.setattr(
            "docling_graph.llm_clients.get_client", lambda provider: lambda effective: client
        )
        monkeypatch.setattr(
            "docling_graph.llm_clients.config.resolve_effective_model_config",
            _stub_effective_config,
        )
        derived = tmp_path / "templates" / "invoice.py"
        derived.parent.mkdir(parents=True)
        derived.write_text("# existing template\n", encoding="utf-8")
        spec_existing = tmp_path / "templates" / "invoice.spec.yaml"
        spec_existing.write_text("# user spec, do not clobber\n", encoding="utf-8")

        result = cli_runner.invoke(
            app,
            [
                "template",
                "from-docs",
                str(DOCS / "invoice_1.md"),
                str(DOCS / "invoice_2.md"),
                "--provider",
                "mistral",
                "--model",
                "mistral-small-latest",
            ],
            input="n\n",
        )
        assert result.exit_code == 0
        assert spec_existing.read_text(encoding="utf-8") == "# user spec, do not clobber\n"
        rescue = tmp_path / "templates" / "invoice.new.spec.yaml"
        assert rescue.is_file()
        assert TemplateSpec.from_yaml(rescue.read_text(encoding="utf-8")).root == "Invoice"
        assert "invoice.new.spec.yaml" in result.output

    def test_tiny_context_window_exits_before_any_llm_call(self, cli_runner, tmp_path, monkeypatch):
        """A context window with no input room fails clearly (never silently
        falling back to the full settings cap) and spends zero tokens."""
        monkeypatch.chdir(tmp_path)
        script = invoice_script()
        client = ScriptedClient(script)
        monkeypatch.setattr(
            "docling_graph.llm_clients.get_client", lambda provider: lambda effective: client
        )
        monkeypatch.setattr(
            "docling_graph.llm_clients.config.resolve_effective_model_config",
            lambda *_a, **_k: SimpleNamespace(
                context_limit=8192, max_output_tokens=8192, model_id="stub"
            ),
        )
        result = cli_runner.invoke(
            app,
            [
                "template",
                "from-docs",
                str(DOCS / "invoice_1.md"),
                "--provider",
                "mistral",
                "--model",
                "mistral-small-latest",
            ],
        )
        assert result.exit_code == 1
        assert "context window too small" in result.output
        assert script.calls == []  # exits before a single token is spent


@pytest.mark.integration
class TestLint:
    def test_billing_document_reports_canon_findings(self, cli_runner, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        result = cli_runner.invoke(
            app,
            ["template", "lint", "docs.examples.templates.billing_document.BillingDocument"],
        )
        assert result.exit_code == 0, result.output  # report-only mode exits 0
        assert "Findings" in result.output
        assert "[R16]" in result.output  # pinned canon finding (computation scrub)
        assert "Docstring windows" in result.output
        assert "BillingDocument" in result.output
        assert "semantic guide" in result.output

    def test_strict_exits_1_on_non_info_findings(self, cli_runner, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        result = cli_runner.invoke(
            app,
            [
                "template",
                "lint",
                "docs.examples.templates.billing_document.BillingDocument",
                "--strict",
            ],
        )
        assert result.exit_code == 1
        assert "--strict" in result.output

    def test_import_allowlist_gate_and_escape_hatch(self, cli_runner, monkeypatch):
        """insurance_terms imports `ast` (outside the allowlist): rejected by
        default, loadable for trusted local files via --no-import-check."""
        monkeypatch.chdir(REPO_ROOT)
        dotted = "docs.examples.templates.insurance_terms.AssuranceMRH"
        result = cli_runner.invoke(app, ["template", "lint", dotted])
        assert result.exit_code == 1
        assert "import allowlist" in result.output

        result = cli_runner.invoke(app, ["template", "lint", dotted, "--no-import-check"])
        assert result.exit_code == 0, result.output
        assert "Findings" in result.output

    def test_unloadable_path_exits_1(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(app, ["template", "lint", "no.such.module.Template"])
        assert result.exit_code == 1


def _write_mini_template(tmp_path: Path) -> str:
    """A loadable one-class template module; returns its dotted path."""
    module_dir = tmp_path / "stub_tmpl"
    module_dir.mkdir()
    (module_dir / "mini.py").write_text(
        "from pydantic import BaseModel, ConfigDict, Field\n"
        "\n"
        "\n"
        "class MiniDoc(BaseModel):\n"
        '    """A minimal document."""\n'
        "\n"
        '    model_config = ConfigDict(graph_id_fields=["doc_id"])\n'
        "\n"
        "    doc_id: str = Field(...)\n",
        encoding="utf-8",
    )
    return "stub_tmpl.mini.MiniDoc"


@pytest.mark.integration
class TestEvaluate:
    def test_evaluate_with_patched_pipeline(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_mini_template(tmp_path)
        graph = nx.DiGraph()
        graph.add_node("MiniDoc_1", __class__="MiniDoc", doc_id="D-1")
        context = SimpleNamespace(
            knowledge_graph=graph, provenance=None, extracted_models=[], extractor=None
        )
        with patch("docling_graph.pipeline.run_pipeline", return_value=context) as mock_run:
            result = cli_runner.invoke(
                app,
                [
                    "template",
                    "evaluate",
                    "stub_tmpl.mini.MiniDoc",
                    "some_doc.md",
                    "--provider",
                    "mistral",
                    "--model",
                    "mistral-small-latest",
                    "-o",
                    "report.md",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "# Template evaluation" in result.output
        assert (tmp_path / "report.md").is_file()
        mock_run.assert_called_once()
        config = mock_run.call_args[0][0]
        assert config.backend == "llm"
        assert config.provider_override == "mistral"
        assert config.model_override == "mistral-small-latest"
        assert str(config.source) == "some_doc.md"

    def test_evaluate_forwards_docling_config(self, cli_runner, tmp_path, monkeypatch):
        """config.yaml's docling block (serve URL/api_key/timeout/headers +
        pipeline) reaches the PipelineConfig — evaluate/--trial-run must
        convert exactly like convert and the induction sampling step do."""
        monkeypatch.chdir(tmp_path)
        for var in ("DOCLING_SERVE_URL", "DOCLING_SERVE_API_KEY", "DOCLING_SERVE_HEADERS"):
            monkeypatch.delenv(var, raising=False)
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "docling": {
                        "pipeline": "vision",
                        "serve": {
                            "url": "http://serve.example:5001",
                            "api_key": "sk-test",
                            "timeout": 42,
                            "headers": {"X-Team": "graph"},
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        dotted = _write_mini_template(tmp_path)
        with patch(
            "docling_graph.pipeline.run_pipeline", return_value=_stub_pipeline_context()
        ) as mock_run:
            result = cli_runner.invoke(
                app,
                [
                    "template",
                    "evaluate",
                    dotted,
                    "some_doc.md",
                    "--provider",
                    "mistral",
                    "--model",
                    "mistral-small-latest",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        config = mock_run.call_args[0][0]
        assert config.docling_config == "vision"
        assert config.docling_serve_url == "http://serve.example:5001"
        assert config.docling_serve_api_key == "sk-test"
        assert config.docling_serve_timeout == 42
        assert config.docling_serve_headers == {"X-Team": "graph"}

    def test_evaluate_missing_provider(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = cli_runner.invoke(app, ["template", "evaluate", "stub.Template", "doc.md"])
        assert result.exit_code == 1
        assert "--provider/--model" in result.output


@pytest.mark.integration
class TestGenerateTemplateApi:
    """The design §2.2 one-shot convenience: draft -> repair -> render ->
    verify -> optional atomic write, with zero CLI coupling."""

    def test_ontology_path_end_to_end(self, tmp_path):
        from docling_graph.templategen import generate_template

        out = tmp_path / "api_gen" / "policy.py"
        result = generate_template(ONTOLOGIES / "policy_basic.ttl", kind="ontology", output=out)
        assert result.verification.passed
        assert result.written_path == out
        assert out.read_text(encoding="utf-8") == result.source_code
        assert result.spec.root == "Policy"
        assert "class Policy" in result.source_code
        assert result.lint_report is not None

    def test_spec_path_round_trip(self, tmp_path):
        from docling_graph.templategen import generate_template

        first = generate_template(ONTOLOGIES / "policy_basic.ttl", kind="ontology")
        assert first.written_path is None  # no output requested
        spec_file = tmp_path / "policy.spec.yaml"
        spec_file.write_text(first.spec.to_yaml(), encoding="utf-8")

        result = generate_template(spec_file, kind="spec")
        assert result.verification.passed
        assert result.written_path is None
        assert result.spec.root == "Policy"

    def test_docs_path_requires_llm_call_fn(self):
        from docling_graph.templategen import generate_template

        with pytest.raises(ValueError, match="llm_call_fn"):
            generate_template(DOCS / "invoice_1.md", kind="docs")

    def test_docs_path_with_injected_llm(self, tmp_path):
        from docling_graph.templategen import generate_template

        out = tmp_path / "api_docs" / "invoice.py"
        result = generate_template(
            [DOCS / "invoice_1.md", DOCS / "invoice_2.md"],
            kind="docs",
            output=out,
            llm_call_fn=invoice_script(),
        )
        assert result.verification.passed
        assert result.spec.root == "Invoice"
        assert result.written_path == out
        assert out.is_file()

    def test_verify_failure_returns_result_without_writing(self, tmp_path, monkeypatch):
        from docling_graph.templategen import generate, generate_template
        from docling_graph.templategen.verify import GateResult, VerificationReport

        failed = VerificationReport(
            root_class="Policy",
            gates=[GateResult(gate="V1", name="ast parse", passed=False, detail="forced")],
        )
        monkeypatch.setattr(generate, "verify_template_source", lambda *_a, **_k: failed)
        out = tmp_path / "never.py"
        result = generate_template(ONTOLOGIES / "policy_basic.ttl", kind="ontology", output=out)
        assert not result.verification.passed
        assert result.written_path is None
        assert not out.exists()  # the requested path is never touched
