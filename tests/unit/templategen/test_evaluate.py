"""Tests for the ``template evaluate`` empirical harness (design §7.3).

The pipeline is never run: a stub ``run_pipeline_fn`` returns synthetic
``PipelineContext``-shaped results (a real ``networkx`` graph carrying the
converter's audit keys, plus a duck-typed provenance ledger), which exercises
the full harvesting path — audit-key translation, fill rates, grounding
precision with the root-stem exclusion, dense stats, and markdown rendering.
One guard test pins every ``AUDIT_KEY_RULEBOOK`` key to the codebase line that
actually writes it, so the table can never drift from the converter.
"""

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import networkx as nx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from docling_graph.templategen.evaluate import (
    AUDIT_KEY_RULEBOOK,
    EvaluationReport,
    evaluate_template,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# A small template + synthetic pipeline results
# ---------------------------------------------------------------------------


def edge(label: str, **kwargs: Any) -> Any:
    if "default" not in kwargs and "default_factory" not in kwargs:
        kwargs["default"] = None
    return Field(json_schema_extra={"edge_label": label}, **kwargs)


class Tax(BaseModel):
    """Tax bracket component, embedded on its parent."""

    model_config = ConfigDict(is_entity=False)

    rate_percent: float | None = Field(None)


class Party(BaseModel):
    """A party named on the invoice."""

    model_config = ConfigDict(graph_id_fields=["name"])

    name: str = Field(...)
    city: str | None = Field(None)
    email: str | None = Field(None)


class Invoice(BaseModel):
    """The invoice document root."""

    model_config = ConfigDict(graph_id_fields=["document_number"])

    document_number: str = Field(...)
    currency: str | None = Field(None)
    total_amount: float | None = Field(None)
    tax: Tax | None = edge(label="HAS_TAX")
    seller: Party | None = edge(label="ISSUED_BY")
    buyer: Party | None = edge(label="BILLED_TO")


def make_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(
        "Invoice_1",
        __class__="Invoice",
        document_number="INV-2024-001",
        currency="EUR",
        total_amount=None,  # unfilled declared attr
        tax={"rate_percent": 20.0},  # embedded component counts as filled
    )
    graph.add_node(
        "Party_1",
        __class__="Party",
        name="Acme GmbH",
        city="Berlin",
        email=None,
    )
    graph.add_node(
        "Party_2",
        __class__="Party",
        name="Hallucinated Corp",
        city=None,
        email=None,
    )
    graph.add_edge("Invoice_1", "Party_1", label="ISSUED_BY")
    graph.add_edge("Invoice_1", "Party_2", label="BILLED_TO")
    return graph


CHUNK_TEXT = "Invoice INV-2024-001 issued by Acme   GmbH, Berlin. All amounts in EUR."


def make_ledger() -> SimpleNamespace:
    # Duck-typed ProvenanceLedger: chunks mapping of records exposing .text.
    return SimpleNamespace(
        chunks={
            0: SimpleNamespace(text=CHUNK_TEXT),
            1: SimpleNamespace(text=""),
        }
    )


def make_context(graph: nx.DiGraph, ledger: Any = None, dense: dict | None = None) -> Any:
    backend = SimpleNamespace(last_dense_stats=dense or {})
    return SimpleNamespace(
        knowledge_graph=graph,
        provenance=ledger,
        extracted_models=[object()],
        extractor=SimpleNamespace(backend=backend),
    )


class StubRunner:
    """Records the config dicts it was called with; returns queued contexts."""

    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, config: dict[str, Any]) -> Any:
        self.calls.append(config)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


# ---------------------------------------------------------------------------
# Audit-key translation
# ---------------------------------------------------------------------------


class TestAuditTranslation:
    def make_report(self, graph: nx.DiGraph) -> EvaluationReport:
        runner = StubRunner([make_context(graph)])
        return evaluate_template(Invoice, ["doc.pdf"], run_pipeline_fn=runner)

    def test_every_audit_key_translates_through_the_table(self):
        graph = make_graph()
        graph.graph["empty_identity_nodes"] = ["Party_9", "Party_10"]
        graph.graph["demoted_nodes"] = [{"id": "Garantie_9", "reason": "cardinality_bound"}]
        graph.graph["closed_catalog_drops"] = {"REFERENCES_ITEM": 3, "COVERS": 1}
        graph.graph["dropped_relationships"] = [
            {"source": "Invoice_1", "label": "HAS_TAX", "target": "Tax_9"}
        ]
        graph.graph["alias_reconciliation"] = {
            "candidates": 4,
            "confirmed": 2,
            "merged": 2,
            "vetoed_sibling": 0,
        }
        report = self.make_report(graph)
        findings = {f.key: f for f in report.per_document[0].audit_findings}
        assert set(findings) == set(AUDIT_KEY_RULEBOOK)
        # Counts reflect each payload's shape.
        assert findings["empty_identity_nodes"].count == 2
        assert findings["demoted_nodes"].count == 1
        assert findings["closed_catalog_drops"].count == 4  # summed drop counts
        assert findings["dropped_relationships"].count == 1
        assert findings["alias_reconciliation"].count == 2  # merged pairs
        # Every finding carries its rulebook clause verbatim.
        for key, finding in findings.items():
            assert finding.rulebook == AUDIT_KEY_RULEBOOK[key]

    def test_clean_graph_yields_no_findings(self):
        report = self.make_report(make_graph())
        assert report.per_document[0].audit_findings == []
        assert report.summary.audit_totals == {}

    def test_alias_reconciliation_falls_back_to_candidates(self):
        graph = make_graph()
        graph.graph["alias_reconciliation"] = {"candidates": 3, "confirmed": 0, "merged": 0}
        report = self.make_report(graph)
        findings = {f.key: f for f in report.per_document[0].audit_findings}
        assert findings["alias_reconciliation"].count == 3

    def test_summary_aggregates_audit_totals_across_documents(self):
        graph_a = make_graph()
        graph_a.graph["empty_identity_nodes"] = ["a"]
        graph_b = make_graph()
        graph_b.graph["empty_identity_nodes"] = ["b", "c"]
        runner = StubRunner([make_context(graph_a), make_context(graph_b)])
        report = evaluate_template(Invoice, ["a.pdf", "b.pdf"], run_pipeline_fn=runner)
        assert report.summary.audit_totals == {"empty_identity_nodes": 3}


class TestAuditKeysExistInCodebase:
    """The rulebook table must name real graph.graph keys — no drift allowed."""

    AUDIT_KEY_WRITERS = [
        REPO_ROOT / "docling_graph" / "core" / "converters" / "graph_converter.py",
        REPO_ROOT / "docling_graph" / "core" / "utils" / "graph_cleaner.py",
    ]

    def test_every_rulebook_key_is_written_by_the_converter_stack(self):
        source = "\n".join(path.read_text(encoding="utf-8") for path in self.AUDIT_KEY_WRITERS)
        for key in AUDIT_KEY_RULEBOOK:
            assert f'graph.graph["{key}"]' in source, (
                f"AUDIT_KEY_RULEBOOK key '{key}' is not written anywhere in "
                f"{[str(p) for p in self.AUDIT_KEY_WRITERS]}"
            )

    def test_every_clause_cites_a_schema_definition_doc(self):
        docs_dir = REPO_ROOT / "docs" / "fundamentals" / "schema-definition"
        doc_names = {path.name for path in docs_dir.glob("*.md")}
        for key, clause in AUDIT_KEY_RULEBOOK.items():
            cited = re.findall(r"([a-z-]+\.md)", clause)
            assert cited, f"clause for '{key}' cites no doc"
            for name in cited:
                assert name in doc_names, f"clause for '{key}' cites unknown doc '{name}'"


# ---------------------------------------------------------------------------
# Fill rates
# ---------------------------------------------------------------------------


class TestFillRates:
    def test_declared_fields_exclude_entity_edges_include_components(self):
        runner = StubRunner([make_context(make_graph())])
        report = evaluate_template(Invoice, ["doc.pdf"], run_pipeline_fn=runner)
        rates = {r.model: r for r in report.per_document[0].fill_rates}
        # Components never become nodes; entity-target edges are graph edges.
        assert set(rates) == {"Invoice", "Party"}
        assert rates["Invoice"].declared_fields == [
            "document_number",
            "currency",
            "total_amount",
            "tax",
        ]
        assert rates["Party"].declared_fields == ["name", "city", "email"]

    def test_fill_rate_is_mean_fraction_of_filled_attrs(self):
        runner = StubRunner([make_context(make_graph())])
        report = evaluate_template(Invoice, ["doc.pdf"], run_pipeline_fn=runner)
        rates = {r.model: r for r in report.per_document[0].fill_rates}
        # Invoice_1 fills 3 of 4 declared attrs (total_amount is None).
        assert rates["Invoice"].node_count == 1
        assert rates["Invoice"].fill_rate == pytest.approx(3 / 4)
        assert rates["Invoice"].filled_counts == {
            "document_number": 1,
            "currency": 1,
            "total_amount": 0,
            "tax": 1,
        }
        # Party_1 fills 2/3, Party_2 fills 1/3 -> mean 1/2.
        assert rates["Party"].node_count == 2
        assert rates["Party"].fill_rate == pytest.approx(0.5)
        assert rates["Party"].filled_counts == {"name": 2, "city": 1, "email": 0}
        assert report.summary.mean_fill_rate == pytest.approx((3 / 4 + 1 / 2) / 2)


# ---------------------------------------------------------------------------
# Grounding precision
# ---------------------------------------------------------------------------


class TestGrounding:
    def run_one(self, *, source: str = "doc.pdf", graph: nx.DiGraph | None = None) -> Any:
        runner = StubRunner([make_context(graph or make_graph(), ledger=make_ledger())])
        report = evaluate_template(Invoice, [source], run_pipeline_fn=runner)
        return report.per_document[0].grounding, report

    def test_whitespace_normalized_substring_matching(self):
        # "Acme GmbH" appears in the chunk as "Acme   GmbH": normalization
        # must bridge the whitespace difference.
        grounding, _ = self.run_one()
        assert grounding is not None
        # Checked strings: INV-2024-001, EUR, Acme GmbH, Berlin, Hallucinated Corp.
        assert grounding.checked == 5
        assert grounding.grounded == 4
        assert grounding.precision == pytest.approx(4 / 5)
        assert grounding.ungrounded_samples == ["Party.name='Hallucinated Corp'"]
        assert grounding.root_id_synthetic is False

    def test_root_stem_id_is_excluded_and_flagged(self):
        graph = make_graph()
        # repair_root_identity's fallback: root id == source file stem.
        graph.nodes["Invoice_1"]["document_number"] = "scanned_invoice"
        grounding, report = self.run_one(source="inbox/scanned_invoice.pdf", graph=graph)
        assert grounding is not None
        assert grounding.root_id_synthetic is True
        # The stem value is excluded from the checked population entirely:
        # neither grounded nor counted as a failure.
        assert grounding.checked == 4
        assert not any("scanned_invoice" in s for s in grounding.ungrounded_samples)
        assert report.summary.root_id_synthetic_documents == 1

    def test_same_value_on_non_root_class_is_still_checked(self):
        graph = make_graph()
        graph.nodes["Party_2"]["name"] = "scanned_invoice"  # not the root id
        grounding, _ = self.run_one(source="scanned_invoice.pdf", graph=graph)
        assert grounding is not None
        assert grounding.root_id_synthetic is False
        assert "Party.name='scanned_invoice'" in grounding.ungrounded_samples

    def test_no_ledger_means_no_grounding_section(self):
        runner = StubRunner([make_context(make_graph(), ledger=None)])
        report = evaluate_template(Invoice, ["doc.pdf"], run_pipeline_fn=runner)
        assert report.per_document[0].grounding is None
        assert report.summary.grounding_precision is None

    def test_empty_chunk_texts_mean_no_grounding_section(self):
        ledger = SimpleNamespace(chunks={0: SimpleNamespace(text="   ")})
        runner = StubRunner([make_context(make_graph(), ledger=ledger)])
        report = evaluate_template(Invoice, ["doc.pdf"], run_pipeline_fn=runner)
        assert report.per_document[0].grounding is None


# ---------------------------------------------------------------------------
# Harness mechanics: config building, injection, failures, dense stats
# ---------------------------------------------------------------------------


class TestHarness:
    def test_config_dict_carries_template_source_and_overrides(self):
        runner = StubRunner([make_context(make_graph()), make_context(make_graph())])
        evaluate_template(
            Invoice,
            ["a.pdf", Path("b.pdf")],
            run_pipeline_fn=runner,
            config_overrides={"backend": "llm", "inference": "remote"},
        )
        assert [c["source"] for c in runner.calls] == ["a.pdf", "b.pdf"]
        for config in runner.calls:
            assert config["template"] is Invoice
            assert config["dump_to_disk"] is False  # harness default
            assert config["backend"] == "llm"
            assert config["inference"] == "remote"

    def test_overrides_beat_the_harness_defaults(self):
        runner = StubRunner([make_context(make_graph())])
        evaluate_template(
            Invoice, ["a.pdf"], run_pipeline_fn=runner, config_overrides={"dump_to_disk": True}
        )
        assert runner.calls[0]["dump_to_disk"] is True

    def test_one_failing_document_does_not_sink_the_batch(self):
        runner = StubRunner([RuntimeError("model exploded"), make_context(make_graph())])
        report = evaluate_template(Invoice, ["bad.pdf", "good.pdf"], run_pipeline_fn=runner)
        failed, succeeded = report.per_document
        assert failed.succeeded is False
        assert failed.error == "RuntimeError: model exploded"
        assert succeeded.succeeded is True
        assert report.summary.documents == 2
        assert report.summary.failed == 1

    def test_missing_graph_is_an_extraction_failure(self):
        runner = StubRunner([SimpleNamespace(knowledge_graph=None)])
        report = evaluate_template(Invoice, ["doc.pdf"], run_pipeline_fn=runner)
        document = report.per_document[0]
        assert document.succeeded is False
        assert "no knowledge graph" in (document.error or "")

    def test_dense_stats_surface_when_present(self):
        dense = {"truncations": 2, "retention_rate": 0.9}
        runner = StubRunner([make_context(make_graph(), dense=dense)])
        report = evaluate_template(Invoice, ["doc.pdf"], run_pipeline_fn=runner)
        assert report.per_document[0].dense_stats == dense

    def test_absent_dense_stats_stay_none(self):
        runner = StubRunner([make_context(make_graph())])
        report = evaluate_template(Invoice, ["doc.pdf"], run_pipeline_fn=runner)
        assert report.per_document[0].dense_stats is None

    def test_empty_sources_rejected(self):
        with pytest.raises(ValueError, match="at least one source"):
            evaluate_template(Invoice, [], run_pipeline_fn=StubRunner([]))

    def test_non_template_rejected(self):
        with pytest.raises(TypeError, match="BaseModel subclass or dotted path"):
            evaluate_template(42, ["doc.pdf"], run_pipeline_fn=StubRunner([]))  # type: ignore[arg-type]

    def test_dotted_path_template_loads_like_the_pipeline(self):
        # Same loader as `--template` (TemplateLoadingStage._load_from_string).
        runner = StubRunner([make_context(make_graph())])
        dotted = "tests.fixtures.sample_templates.test_template.SampleInvoice"
        report = evaluate_template(dotted, ["doc.pdf"], run_pipeline_fn=runner)
        assert issubclass(runner.calls[0]["template"], BaseModel)
        assert runner.calls[0]["template"].__name__ == "SampleInvoice"
        assert report.template == dotted

    def test_report_uses_dotted_name_for_live_classes(self):
        runner = StubRunner([make_context(make_graph())])
        report = evaluate_template(Invoice, ["doc.pdf"], run_pipeline_fn=runner)
        assert report.template.endswith(".Invoice")


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


class TestMarkdown:
    def full_report(self) -> EvaluationReport:
        graph = make_graph()
        graph.graph["empty_identity_nodes"] = ["Party_9"]
        graph.nodes["Invoice_1"]["document_number"] = "doc"
        runner = StubRunner(
            [
                make_context(graph, ledger=make_ledger(), dense={"truncations": 2}),
                RuntimeError("boom"),
            ]
        )
        return evaluate_template(Invoice, ["doc.pdf", "bad.pdf"], run_pipeline_fn=runner)

    def test_markdown_carries_every_section(self):
        markdown = self.full_report().render_markdown()
        assert markdown.startswith("# Template evaluation")
        assert "## Summary" in markdown
        assert "### Rulebook violations (graph audit keys)" in markdown
        assert "`empty_identity_nodes`" in markdown
        assert AUDIT_KEY_RULEBOOK["empty_identity_nodes"] in markdown
        assert "## doc.pdf" in markdown
        assert "## bad.pdf" in markdown
        assert "**Extraction failed:** RuntimeError: boom" in markdown
        assert "| Party | 2 |" in markdown  # fill-rate table row
        assert "Root id equals the source file stem" in markdown
        assert "truncations: 2" in markdown

    def test_markdown_is_timestamp_free_and_deterministic(self):
        report = self.full_report()
        markdown = report.render_markdown()
        assert markdown == report.render_markdown()
        # No dates/times anywhere (report must diff cleanly across runs).
        assert not re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", markdown)

    def test_model_dump_is_timestamp_free(self):
        dumped = self.full_report().model_dump()
        assert "timestamp" not in str(dumped).lower()
