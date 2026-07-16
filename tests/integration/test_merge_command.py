"""
End-to-end integration tests for the merge CLI command.
"""

import json
from pathlib import Path

import networkx as nx
import pytest
from typer.testing import CliRunner

from docling_graph.cli.main import app
from docling_graph.core.converters.graph_converter import GraphConverter
from docling_graph.core.exporters.json_exporter import JSONExporter
from docling_graph.core.importers.graph_json import load_graph_json
from tests.fixtures.sample_templates.test_template import SampleCompany, SamplePerson


def _convert_run(base_dir: Path, name: str, company: SampleCompany) -> Path:
    """Convert models and export into a run-dir layout, like ExportStage does."""
    graph, _meta = GraphConverter().pydantic_list_to_graph([company])
    run_dir = base_dir / name
    JSONExporter().export(graph, run_dir / "docling_graph" / "graph.json")
    return run_dir


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def two_runs(tmp_path):
    """Two convert runs sharing one SampleCompany, with disjoint SamplePersons."""
    run_a = _convert_run(
        tmp_path,
        "run_a",
        SampleCompany(
            company_name="Acme",
            industry="",
            founded_year=1999,
            employees=[
                SamplePerson(first_name="Ada", last_name="Byron", email="ada@acme.com"),
                SamplePerson(first_name="Alan", last_name="Turing", email="alan@acme.com"),
            ],
        ),
    )
    run_b = _convert_run(
        tmp_path,
        "run_b",
        SampleCompany(
            company_name="ACME",  # same canonical identity, different surface form
            industry="Technology",
            founded_year=1999,
            employees=[
                SamplePerson(first_name="Grace", last_name="Hopper", email="grace@acme.com"),
            ],
        ),
    )
    return run_a, run_b


@pytest.mark.integration
class TestCLIMergeCommand:
    def test_merge_command_help(self, cli_runner):
        result = cli_runner.invoke(app, ["merge", "--help"])
        assert result.exit_code == 0
        assert "merge" in result.stdout.lower()

    def test_merge_folds_shared_company_and_adds_disjoint_persons(
        self, cli_runner, tmp_path, two_runs
    ):
        run_a, run_b = two_runs
        out_dir = tmp_path / "merged"
        result = cli_runner.invoke(app, ["merge", str(run_a), str(run_b), "-o", str(out_dir)])
        assert result.exit_code == 0, result.output

        graph_path = out_dir / "docling_graph" / "graph.json"
        merged = load_graph_json(graph_path)  # output is re-loadable through the loader

        companies = [n for n, d in merged.nodes(data=True) if d.get("__class__") == "SampleCompany"]
        persons = [n for n, d in merged.nodes(data=True) if d.get("__class__") == "SamplePerson"]
        assert len(companies) == 1  # shared company folded into one node
        assert len(persons) == 3  # person counts add
        company = merged.nodes[companies[0]]
        assert company["industry"] == "Technology"  # fill-empty union across inputs
        assert {e["source"] for e in company["merged_from"]} == {str(run_a), str(run_b)}
        # All employee edges landed on the single company node.
        assert all(merged.has_edge(companies[0], p) for p in persons)

        report_path = out_dir / "docling_graph" / "merge_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert len(report["sources"]) == 2
        assert report["nodes_folded"] == 1
        assert report["identity_source"] == "v2_export"
        assert (out_dir / "docling_graph" / "report.md").is_file()
        assert (out_dir / "docling_graph" / "graph.html").is_file()
        assert (out_dir / "docling_graph" / "provenance" / "manifest.json").is_file()

    def test_inspect_succeeds_on_merged_output(self, cli_runner, tmp_path, two_runs):
        run_a, run_b = two_runs
        out_dir = tmp_path / "merged"
        result = cli_runner.invoke(app, ["merge", str(run_a), str(run_b), "-o", str(out_dir)])
        assert result.exit_code == 0, result.output

        viz_path = tmp_path / "viz.html"
        result = cli_runner.invoke(
            app,
            [
                "inspect",
                str(out_dir / "docling_graph" / "graph.json"),
                "--format",
                "json",
                "--no-open",
                "--output",
                str(viz_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert viz_path.is_file()

    def test_golden_determinism_byte_identical_graph_json(self, cli_runner, tmp_path, two_runs):
        run_a, run_b = two_runs
        out_1, out_2 = tmp_path / "merged_1", tmp_path / "merged_2"
        for out_dir in (out_1, out_2):
            result = cli_runner.invoke(app, ["merge", str(run_a), str(run_b), "-o", str(out_dir)])
            assert result.exit_code == 0, result.output
        graph_1 = (out_1 / "docling_graph" / "graph.json").read_bytes()
        graph_2 = (out_2 / "docling_graph" / "graph.json").read_bytes()
        assert graph_1 == graph_2

    def test_rekey_restores_fold_after_id_drift(self, cli_runner, tmp_path, two_runs):
        run_a, run_b = two_runs
        # Simulate normalizer drift: mutate every node id in run_b's export.
        graph_json = run_b / "docling_graph" / "graph.json"
        drifted = graph_json.read_text(encoding="utf-8").replace(
            "SampleCompany_", "SampleCompany_drift"
        )
        graph_json.write_text(drifted, encoding="utf-8")

        out_dir = tmp_path / "merged"
        result = cli_runner.invoke(
            app, ["merge", str(run_a), str(run_b), "-o", str(out_dir), "--rekey"]
        )
        assert result.exit_code == 0, result.output
        merged = load_graph_json(out_dir / "docling_graph" / "graph.json")
        companies = [n for n, d in merged.nodes(data=True) if d.get("__class__") == "SampleCompany"]
        assert len(companies) == 1  # re-keying restored the fold
        report = json.loads(
            (out_dir / "docling_graph" / "merge_report.json").read_text(encoding="utf-8")
        )
        assert report["rekeyed"] is True
        assert report["rekeyed_changed"] >= 1

    def test_duplicate_input_absorbed(self, cli_runner, tmp_path, two_runs):
        run_a, _run_b = two_runs
        out_dir = tmp_path / "merged"
        result = cli_runner.invoke(app, ["merge", str(run_a), str(run_a), "-o", str(out_dir)])
        assert result.exit_code == 0, result.output
        merged = load_graph_json(out_dir / "docling_graph" / "graph.json")
        original = load_graph_json(run_a / "docling_graph" / "graph.json")
        assert merged.number_of_nodes() == original.number_of_nodes()
        assert merged.number_of_edges() == original.number_of_edges()
        report = json.loads(
            (out_dir / "docling_graph" / "merge_report.json").read_text(encoding="utf-8")
        )
        assert len(report["duplicates_absorbed"]) == 1

    def test_dry_run_writes_only_merge_report(self, cli_runner, tmp_path, two_runs):
        run_a, run_b = two_runs
        out_dir = tmp_path / "merged"
        result = cli_runner.invoke(
            app, ["merge", str(run_a), str(run_b), "-o", str(out_dir), "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        written = [p.relative_to(out_dir) for p in out_dir.rglob("*") if p.is_file()]
        assert written == [Path("docling_graph") / "merge_report.json"]
        report = json.loads(
            (out_dir / "docling_graph" / "merge_report.json").read_text(encoding="utf-8")
        )
        assert report["dry_run"] is True
        assert report["node_count"] == 4

    def test_export_format_csv_adds_csv_files(self, cli_runner, tmp_path, two_runs):
        run_a, run_b = two_runs
        out_dir = tmp_path / "merged"
        result = cli_runner.invoke(
            app,
            ["merge", str(run_a), str(run_b), "-o", str(out_dir), "--export-format", "csv"],
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "docling_graph" / "nodes.csv").is_file()
        assert (out_dir / "docling_graph" / "edges.csv").is_file()

    def test_vetoed_alias_decisions_are_reported_and_tip_suppressed(self, cli_runner, tmp_path):
        """A human-confirmed pair the sibling veto rejects must be surfaced in
        the report, and the 're-run with --alias-decisions' tip (which would
        just repeat the veto) must be suppressed."""

        def _offer_run(name: str, offer: str, offer_id: str) -> Path:
            g = nx.DiGraph()
            g.add_node(
                "Root_r", id="Root_r", label="Root", type="entity", __class__="Root", rid="R"
            )
            g.add_node(
                offer_id, id=offer_id, label="Offre", type="entity", __class__="Offre", nom=offer
            )
            g.add_edge("Root_r", offer_id, label="AOFFRE")
            g.graph["id_fields_map"] = {"Root": ["rid"], "Offre": ["nom"]}
            run_dir = tmp_path / name
            JSONExporter().export(g, run_dir / "docling_graph" / "graph.json")
            return run_dir

        run_a = _offer_run("offer_a", "CONFORT", "O_c")
        run_b = _offer_run("offer_b", "CONFORT PLUS", "O_cp")
        out_1 = tmp_path / "merged_1"
        result = cli_runner.invoke(app, ["merge", str(run_a), str(run_b), "-o", str(out_1)])
        assert result.exit_code == 0, result.output
        report = json.loads(
            (out_1 / "docling_graph" / "merge_report.json").read_text(encoding="utf-8")
        )
        candidates = report["alias_candidates"]
        assert len(candidates) == 1
        for stub in candidates:
            stub["confirm"] = True
        decisions = tmp_path / "decisions.json"
        decisions.write_text(json.dumps(candidates), encoding="utf-8")

        out_2 = tmp_path / "merged_2"
        result = cli_runner.invoke(
            app,
            [
                "merge",
                str(run_a),
                str(run_b),
                "-o",
                str(out_2),
                "--alias-decisions",
                str(decisions),
            ],
        )
        assert result.exit_code == 0, result.output
        flat = " ".join(result.output.split())  # undo rich line wrapping
        assert "vetoed by reconciliation guards" in flat
        assert "re-run with --alias-decisions" not in flat
        report = json.loads(
            (out_2 / "docling_graph" / "merge_report.json").read_text(encoding="utf-8")
        )
        assert report["alias_stats"]["vetoed_sibling"] == 1
        assert report["alias_stats"]["merged"] == 0
        assert report["ignored_alias_decisions"] == [
            {
                "class": "Offre",
                "keep_id": candidates[0]["keep_id"],
                "merge_ids": candidates[0]["merge_ids"],
                "reason": "vetoed by reconciliation guards (e.g. sibling co-occurrence)",
            }
        ]

    def test_invalid_input_exits_with_error(self, cli_runner, tmp_path):
        bogus = tmp_path / "bogus.json"
        bogus.write_text(json.dumps({"foo": 1}), encoding="utf-8")
        result = cli_runner.invoke(app, ["merge", str(bogus), "-o", str(tmp_path / "out")])
        assert result.exit_code == 1
        assert not (tmp_path / "out").exists()  # no partial output dir

    def test_invalid_precedence_rejected(self, cli_runner, tmp_path, two_runs):
        run_a, run_b = two_runs
        result = cli_runner.invoke(app, ["merge", str(run_a), str(run_b), "--precedence", "wrong"])
        assert result.exit_code == 1
