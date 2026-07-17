"""Unit tests for the documents->SPEC induction path (templategen.induce).

A scripted ``llm_call_fn`` (canned pass payloads, per the tests/mocks pattern)
drives end-to-end induction; the deterministic pieces — evidence gates, the
budget sampler, the cross-document merge, gap-fill — are also exercised
directly. No network, no real DocumentProcessor.
"""

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from docling_graph.exceptions import ClientError, ExtractionError
from docling_graph.templategen.induce.documents import (
    ELISION_MARKER,
    DocumentContent,
    InductionReport,
    induce_spec_from_documents,
    prepare_document_text,
    prepare_document_windows,
    source_display_name,
)
from docling_graph.templategen.induce.gapfill import fill_gaps
from docling_graph.templategen.induce.merge import (
    ClassCandidate,
    DocumentCandidates,
    FieldCandidate,
    merge_documents,
)
from docling_graph.templategen.induce.schemas import (
    class_inventory_schema,
    fields_schema,
    gapfill_schema,
    relationships_schema,
)
from docling_graph.templategen.linter import LintReport
from docling_graph.templategen.spec import FieldSpec, ModelSpec, SpecGap, TemplateSpec

FIXTURE_DOCS = Path(__file__).parents[2] / "fixtures" / "templategen" / "docs"


# ---------------------------------------------------------------------------
# Scripted llm_call_fn (per the tests/mocks pattern)
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Routes calls by substring of the context tag; pops payloads FIFO."""

    def __init__(self, script: dict[str, list[Any]]) -> None:
        self.script = {key: list(value) for key, value in script.items()}
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, prompt: dict[str, str], schema_json: str, context: str) -> Any:
        self.calls.append({"prompt": prompt, "schema_json": schema_json, "context": context})
        for key, queue in self.script.items():
            if key in context:
                assert queue, f"no scripted payload left for context {context!r}"
                return copy.deepcopy(queue.pop(0))
        raise AssertionError(f"unexpected context: {context!r}")

    def calls_for(self, key: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if key in call["context"]]


# ---------------------------------------------------------------------------
# Payload factories (the canned pass shapes from design §4.2)
# ---------------------------------------------------------------------------


def p1(
    name,
    kind="entity",
    *,
    is_root=False,
    what="",
    confus="",
    ident=("", "", ()),
    max_count=0,
    quotes=(),
):
    return {
        "name": name,
        "kind": kind,
        "is_root": is_root,
        "what_it_is": what or f"A {name}.",
        "confusable_with": confus,
        "identity_candidate": {
            "field": ident[0],
            "why": ident[1],
            "verbatim_examples": list(ident[2]),
        },
        "documented_max_count": max_count,
        "evidence_quotes": list(quotes),
    }


def p2f(
    name,
    ftype="str",
    *,
    is_list=False,
    description="",
    examples=(),
    enum_members=(),
    enum_synonyms=(),
    unit_varies=False,
):
    return {
        "name": name,
        "type": ftype,
        "is_list": is_list,
        "description": description,
        "verbatim_examples": list(examples),
        "enum_members": list(enum_members),
        "enum_synonyms": list(enum_synonyms),
        "unit_varies": unit_varies,
    }


def p3e(source, field_name, target, *, label="", is_list=False, fully=True, evidence=()):
    return {
        "source": source,
        "field_name": field_name,
        "target": target,
        "label": label,
        "is_list": is_list,
        "target_described_fully_here": fully,
        "evidence": list(evidence),
    }


def get_model(spec: TemplateSpec, name: str) -> ModelSpec:
    return next(m for m in spec.models if m.name == name)


def get_field(spec: TemplateSpec, model: str, field: str) -> FieldSpec:
    return next(f for f in get_model(spec, model).fields if f.name == field)


# ---------------------------------------------------------------------------
# Candidate factories (direct merge tests)
# ---------------------------------------------------------------------------


def fc(name, **kw: Any) -> FieldCandidate:
    return FieldCandidate(name=name, **kw)


def cc(name, kind="entity", *, is_root=False, identity=None, fields=(), **kw: Any):
    all_fields = list(fields)
    identity_survived = False
    if identity is not None:
        id_name, id_examples = identity
        all_fields.insert(0, fc(id_name, role="identity", examples=list(id_examples)))
        identity_survived = True
    return ClassCandidate(
        name=name,
        kind=kind,
        is_root=is_root,
        what_it_is=f"A {name}.",
        fields=all_fields,
        identity_survived=identity_survived,
        **kw,
    )


def doc(name, *classes: ClassCandidate) -> DocumentCandidates:
    return DocumentCandidates(name=name, classes=list(classes))


def root_cls(doc_index: int = 1) -> ClassCandidate:
    return cc("Doc", is_root=True, identity=("doc_id", [f"D-{doc_index}"]))


def draft_model(draft: dict, name: str) -> dict:
    return next(m for m in draft["models"] if m["name"] == name)


def draft_field(model: dict, name: str) -> dict:
    return next(f for f in model["fields"] if f["name"] == name)


# ---------------------------------------------------------------------------
# End-to-end induction over the two invoice fixtures
# ---------------------------------------------------------------------------


def invoice_script() -> ScriptedLLM:
    doc1_pass1 = {
        "classes": [
            p1(
                "Invoice",
                is_root=True,
                what="A commercial invoice billing a buyer for delivered items.",
                confus="a purchase order, which requests rather than bills",
                ident=("invoice_number", "printed in the header", ["INV-2024-0113"]),
            ),
            p1(
                "Party",
                what="A company appearing on the invoice.",
                ident=("name", "after Issued by / Billed to", ["Acme GmbH", "Beta SARL"]),
            ),
            p1(
                "LineItem",
                what="One row of the billing table.",
                confus="Item, the product itself",
                ident=("line_number", "first column of the table", ["1", "2"]),
            ),
            p1("Address", "component", what="A postal address block."),
        ]
    }
    doc1_pass2 = {
        "classes": [
            {
                "class_name": "Invoice",
                "fields": [
                    p2f(
                        "total_amount",
                        "float",
                        description="LOOK FOR the Total line at the bottom.",
                        examples=["138.90"],
                    ),
                    p2f(
                        "currency",
                        "enum:Currency",
                        examples=["EUR"],
                        enum_members=["EUR"],
                        enum_synonyms=[{"member": "EUR", "phrases": ["euros"]}],
                    ),
                    p2f("purchase_order", "str", examples=["NOT-IN-DOC-XYZ"]),
                ],
            },
            {
                "class_name": "Party",
                "fields": [
                    p2f(
                        "vat_number",
                        "str",
                        description="LOOK FOR the VAT line.",
                        examples=["DE-812-940-113"],
                    )
                ],
            },
            {"class_name": "LineItem", "fields": [p2f("quantity", "int", examples=["2"])]},
            {"class_name": "Address", "fields": [p2f("street", examples=["Hauptstrasse 12"])]},
        ]
    }
    doc1_pass3 = {
        "edges": [
            p3e(
                "Invoice",
                "issued_by",
                "Party",
                label="ISSUED_BY",
                evidence=["Issued by: Acme GmbH"],
            ),
            p3e("Invoice", "billed_to", "Party", label="BILLED_TO"),
            p3e("Invoice", "line_items", "LineItem", label="CONTAINS_LINE", is_list=True),
            p3e("Party", "address", "Address"),
            p3e("Invoice", "ghost", "Warehouse", label="STORED_AT"),
        ]
    }
    doc2_pass1 = {
        "classes": [
            p1(
                "Invoice",
                is_root=True,
                what="A commercial invoice billing a buyer.",
                ident=("invoice_number", "printed in the header", ["INV-2024-0207"]),
            ),
            p1(
                "Party",
                what="A company on the invoice.",
                ident=("name", "after Issued by / Billed to", ["Gamma Corp", "Delta LLC"]),
            ),
            # Name variant: canonical merge must unify "Line item" with "LineItem".
            p1(
                "Line item",
                what="One billing table row.",
                ident=("line_number", "first column", ["1", "2"]),
            ),
            p1("Address", "component", what="A postal address block."),
        ]
    }
    doc2_pass2 = {
        "classes": [
            {
                "class_name": "Invoice",
                "fields": [
                    p2f("total_amount", "float", examples=["69.50"]),
                    p2f(
                        "currency",
                        "enum:Currency",
                        examples=["EUR"],
                        enum_members=["EUR", "USD"],
                    ),
                ],
            },
            {"class_name": "Party", "fields": [p2f("vat_number", examples=["IE-6388047V"])]},
            {"class_name": "Line item", "fields": [p2f("quantity", "float", examples=["1.5"])]},
            {"class_name": "Address", "fields": [p2f("street", examples=["1 Market Square"])]},
        ]
    }
    doc2_pass3 = {
        "edges": [
            p3e("Invoice", "issued_by", "Party", label="ISSUED_BY"),
            p3e("Invoice", "billed_to", "Party", label="BILLED_TO"),
            p3e("Invoice", "line_items", "Line item", label="CONTAINS_LINE", is_list=True),
            p3e("Party", "address", "Address"),
        ]
    }
    return ScriptedLLM(
        {
            "pass1": [doc1_pass1, doc2_pass1],
            "pass2": [doc1_pass2, doc2_pass2],
            "pass3": [doc1_pass3, doc2_pass3],
        }
    )


class TestEndToEndInduction:
    @pytest.fixture()
    def result(self) -> tuple[TemplateSpec, InductionReport, ScriptedLLM]:
        llm = invoice_script()
        spec, report = induce_spec_from_documents(
            [FIXTURE_DOCS / "invoice_1.md", FIXTURE_DOCS / "invoice_2.md"], llm
        )
        return spec, report, llm

    def test_produces_valid_spec_with_expected_shape(self, result):
        spec, _, _ = result
        assert isinstance(spec, TemplateSpec)
        assert spec.root == "Invoice"
        assert get_model(spec, "Invoice").kind == "root"
        assert get_model(spec, "Party").kind == "entity"
        assert get_model(spec, "LineItem").kind == "entity"
        assert get_model(spec, "Address").kind == "component"
        assert all(m.provenance == "induced" for m in spec.models)

    def test_identity_examples_merge_across_documents(self, result):
        spec, _, _ = result
        assert get_model(spec, "Invoice").identity_fields == ["invoice_number"]
        assert get_field(spec, "Invoice", "invoice_number").examples == [
            "INV-2024-0113",
            "INV-2024-0207",
        ]
        # Round-robin across documents, deduped canonically.
        assert get_field(spec, "Party", "name").examples == [
            "Acme GmbH",
            "Gamma Corp",
            "Beta SARL",
            "Delta LLC",
        ]

    def test_type_promotion_across_documents(self, result):
        spec, report, _ = result
        assert get_field(spec, "LineItem", "quantity").type == "float"
        promotions = report.merge.by_kind("type_promotion")
        assert any(d.field == "quantity" for d in promotions)

    def test_edges_and_labels(self, result):
        spec, report, _ = result
        assert get_field(spec, "Invoice", "issued_by").edge_label == "ISSUED_BY"
        assert get_field(spec, "Invoice", "billed_to").edge_label == "BILLED_TO"
        line_items = get_field(spec, "Invoice", "line_items")
        assert line_items.edge_label == "CONTAINS_LINE"
        assert line_items.is_list
        # Label-less edge: derived + missing_edge_label gap.
        assert get_field(spec, "Party", "address").edge_label == "HAS_ADDRESS"
        assert any(
            g.kind == "missing_edge_label" and g.model == "Party" and g.field == "address"
            for g in report.gaps
        )
        # Edge to an unknown class is dropped and reported.
        assert not any(f.name == "ghost" for f in get_model(spec, "Invoice").fields)
        assert report.documents[0].edges_dropped == ["Invoice -> Warehouse"]

    def test_enum_union(self, result):
        spec, _, _ = result
        enum = next(e for e in spec.enums if e.name == "Currency")
        assert enum.members == ["EUR", "USD"]
        assert enum.synonyms == {"EUR": ["euros"]}
        assert enum.include_other
        assert get_field(spec, "Invoice", "currency").type == "Currency"

    def test_verbatim_gate_drops_hallucinated_example(self, result):
        spec, report, _ = result
        assert get_field(spec, "Invoice", "purchase_order").examples == []
        assert report.documents[0].examples_dropped_by_gate >= 1

    def test_report_contents(self, result):
        _, report, _ = result
        assert isinstance(report, InductionReport)
        assert [d.name for d in report.documents] == ["invoice_1.md", "invoice_2.md"]
        assert all(d.classes_proposed == 4 and d.classes_kept == 4 for d in report.documents)
        assert not report.documents[0].sampled
        assert isinstance(report.lint, LintReport)
        assert report.skipped_sources == []

    def test_llm_call_contract(self, result):
        _, _, llm = result
        first = llm.calls[0]
        assert set(first["prompt"]) == {"system", "user"}
        assert first["context"].startswith("templategen_pass1_classes:")
        assert isinstance(first["schema_json"], str)
        assert "BANNED labels" in first["prompt"]["system"]
        # Call order per document: pass1 -> pass2 -> pass3.
        contexts = [c["context"] for c in llm.calls]
        assert [c.split(":")[0] for c in contexts] == [
            "templategen_pass1_classes",
            "templategen_pass2_fields",
            "templategen_pass3_edges",
        ] * 2


def test_root_name_renames_elected_root(tmp_path):
    source = tmp_path / "doc.md"
    source.write_text(
        "# INVOICE INV-1\n\nIssued by Acme GmbH for services rendered in March.\n"
        "Payment is due in thirty days from the date printed above.\n"
        "Reference INV-1 must appear on the bank transfer statement.\n"
        "Late payments accrue interest at the statutory annual rate.\n",
        encoding="utf-8",
    )
    llm = ScriptedLLM(
        {
            "pass1": [
                {"classes": [p1("Invoice", is_root=True, ident=("invoice_number", "", ["INV-1"]))]}
            ],
            "pass2": [{"classes": []}],
            "pass3": [{"edges": []}],
        }
    )
    spec, _ = induce_spec_from_documents([source], llm, root_name="InvoiceDocument")
    assert spec.root == "InvoiceDocument"
    assert get_model(spec, "InvoiceDocument").kind == "root"


# ---------------------------------------------------------------------------
# Evidence gates
# ---------------------------------------------------------------------------


GATE_DOC = (
    "# CONTRACT REGISTER\n\n"
    "Contract: Alpha Beta signed with the northern division office.\n"
    "Contract: Gamma Delta signed with the southern division office.\n"
    "This policy covers up to 6 guarantees for every insured site.\n"
    "Guarantee: Fire Damage\n"
    "Guarantee: Water Damage\n"
    "Option: Legal Assistance\n"
    "Optional coverages may be added at renewal time by the holder.\n"
    "Identifier GOOD-1 appears once in this register for the audit.\n"
)


def gate_source(tmp_path) -> Path:
    source = tmp_path / "register.md"
    source.write_text(GATE_DOC, encoding="utf-8")
    return source


def test_verbatim_gate_and_identity_drop(tmp_path):
    llm = ScriptedLLM(
        {
            "pass1": [
                {
                    "classes": [
                        p1(
                            "Register", is_root=True, ident=("register_id", "", ["GOOD-1", "BAD-9"])
                        ),
                        # Every identity example hallucinated -> candidate dropped,
                        # class demoted to component by the merge.
                        p1("Site", ident=("site_code", "", ["ZZZ-404", "ZZZ-405"])),
                    ]
                }
            ],
            "pass2": [{"classes": []}],
            "pass3": [{"edges": []}],
        }
    )
    spec, report = induce_spec_from_documents([gate_source(tmp_path)], llm)
    assert get_field(spec, "Register", "register_id").examples == ["GOOD-1"]
    stats = report.documents[0]
    assert stats.examples_dropped_by_gate == 3  # BAD-9 + both ZZZ examples
    assert stats.identity_candidates_dropped == ["Site.site_code"]
    assert get_model(spec, "Site").kind == "component"
    assert any(g.kind == "missing_identity" and g.model == "Site" for g in report.gaps)
    # 1 surviving example on the root identity -> R3 missing_examples gap.
    assert any(
        g.kind == "missing_examples" and g.model == "Register" and g.field == "register_id"
        for g in report.gaps
    )


def test_digit_honesty_rename(tmp_path):
    llm = ScriptedLLM(
        {
            "pass1": [
                {
                    "classes": [
                        p1("Register", is_root=True, ident=("register_id", "", ["GOOD-1"])),
                        p1(
                            "Contract",
                            ident=("contract_number", "", ["Alpha Beta", "Gamma Delta"]),
                        ),
                    ]
                }
            ],
            "pass2": [{"classes": []}],
            "pass3": [{"edges": []}],
        }
    )
    spec, report = induce_spec_from_documents([gate_source(tmp_path)], llm)
    contract = get_model(spec, "Contract")
    assert contract.identity_fields == ["name"]
    assert get_field(spec, "Contract", "name").examples == ["Alpha Beta", "Gamma Delta"]
    assert report.documents[0].digit_honesty_renames == ["Contract.contract_number->name"]


def test_cardinality_gate(tmp_path):
    llm = ScriptedLLM(
        {
            "pass1": [
                {
                    "classes": [
                        p1("Register", is_root=True, ident=("register_id", "", ["GOOD-1"])),
                        p1(
                            "Guarantee",
                            ident=("name", "", ["Fire Damage", "Water Damage"]),
                            max_count=6,
                            quotes=["This policy covers up to 6 guarantees"],
                        ),
                        # Quote holds no digit/number word -> bound rejected.
                        p1(
                            "Option",
                            ident=("name", "", ["Legal Assistance"]),
                            max_count=3,
                            quotes=["Optional coverages may be added"],
                        ),
                    ]
                }
            ],
            "pass2": [{"classes": []}],
            "pass3": [{"edges": []}],
        }
    )
    spec, report = induce_spec_from_documents([gate_source(tmp_path)], llm)
    # Draft carried the DOCUMENTED max (6); repair_draft doubled exactly once.
    assert get_model(spec, "Guarantee").max_instances == 12
    assert get_model(spec, "Option").max_instances is None
    assert report.documents[0].cardinality_bounds_dropped == ["Option"]


# ---------------------------------------------------------------------------
# Pass 2 batching (<=6 classes per call)
# ---------------------------------------------------------------------------


def test_pass2_batches_at_most_six_classes(tmp_path):
    source = tmp_path / "big.md"
    source.write_text("Plain prose sentence for the batching test. " * 10, encoding="utf-8")
    names = [f"Klass{i}" for i in range(1, 9)]
    llm = ScriptedLLM(
        {
            "pass1": [
                {"classes": [p1(name, "component", is_root=(name == "Klass1")) for name in names]}
            ],
            "pass2": [{"classes": []}, {"classes": []}],
            "pass3": [{"edges": []}],
        }
    )
    spec, _ = induce_spec_from_documents([source], llm)
    pass2_calls = llm.calls_for("pass2")
    assert len(pass2_calls) == 2
    assert pass2_calls[0]["context"].endswith(":batch0")
    assert pass2_calls[1]["context"].endswith(":batch1")

    def batch_size(call: dict[str, Any]) -> int:
        listing = call["prompt"]["user"].split("Propose fields for exactly these classes:")[1]
        return sum(1 for line in listing.splitlines() if line.startswith("- "))

    assert batch_size(pass2_calls[0]) == 6
    assert batch_size(pass2_calls[1]) == 2
    assert len(spec.models) == 8


# ---------------------------------------------------------------------------
# Retry policy and the no-progress guard
# ---------------------------------------------------------------------------


def test_retry_recovers_from_one_invalid_payload(tmp_path):
    llm = ScriptedLLM(
        {
            "pass1": [
                {"unexpected": True},
                {"classes": [p1("Register", is_root=True, ident=("register_id", "", ["GOOD-1"]))]},
            ],
            "pass2": [{"classes": []}],
            "pass3": [{"edges": []}],
        }
    )
    spec, report = induce_spec_from_documents([gate_source(tmp_path)], llm)
    assert spec.root == "Register"
    assert report.documents[0].retries == 1


def test_no_progress_guard_stops_on_identical_retry_payload(tmp_path):
    llm = ScriptedLLM({"pass1": [{"unexpected": True}, {"unexpected": True}]})
    with pytest.raises(ExtractionError, match="no progress"):
        induce_spec_from_documents([gate_source(tmp_path)], llm)


def test_distinct_invalid_payloads_fail_after_one_retry(tmp_path):
    llm = ScriptedLLM({"pass1": [{"unexpected": 1}, {"unexpected": 2}]})
    with pytest.raises(ExtractionError, match="failed after one retry"):
        induce_spec_from_documents([gate_source(tmp_path)], llm)


# ---------------------------------------------------------------------------
# Pass 2 truncation splitting
# ---------------------------------------------------------------------------


class _Pass2SizeLimitedLLM(ScriptedLLM):
    """Raises for pass-2 batches above ``max_batch``, scripted otherwise."""

    def __init__(self, script: dict, *, max_batch: int, truncated: bool) -> None:
        super().__init__(script)
        self._max_batch = max_batch
        self._truncated = truncated

    @staticmethod
    def _batch_size(prompt: dict[str, str]) -> int:
        listing = prompt["user"].split("Propose fields for exactly these classes:")[1]
        return sum(1 for line in listing.splitlines() if line.startswith("- "))

    def __call__(self, *, prompt: dict[str, str], schema_json: str, context: str) -> Any:
        if "pass2" in context and self._batch_size(prompt) > self._max_batch:
            self.calls.append({"prompt": prompt, "schema_json": schema_json, "context": context})
            details = {"truncated": True} if self._truncated else {}
            raise ClientError("Invalid JSON response", details=details)
        return super().__call__(prompt=prompt, schema_json=schema_json, context=context)


def _four_class_source(tmp_path) -> Path:
    source = tmp_path / "big.md"
    source.write_text("Plain prose sentence for the batching test. " * 10, encoding="utf-8")
    return source


def _four_class_script(pass2_payloads: list[Any]) -> dict:
    names = [f"Klass{i}" for i in range(1, 5)]
    return {
        "pass1": [
            {"classes": [p1(name, "component", is_root=(name == "Klass1")) for name in names]}
        ],
        "pass2": pass2_payloads,
        "pass3": [{"edges": []}],
    }


def test_pass2_truncated_batch_splits_and_recovers(tmp_path):
    llm = _Pass2SizeLimitedLLM(
        _four_class_script([{"classes": []}, {"classes": []}]), max_batch=2, truncated=True
    )
    spec, report = induce_spec_from_documents([_four_class_source(tmp_path)], llm)
    assert len(spec.models) == 4
    stats = report.documents[0]
    assert stats.pass2_splits == 1
    # The 4-class batch failed twice (call + retry), then each half succeeded.
    assert len(llm.calls_for("pass2")) == 4
    assert stats.retries == 1


def test_pass2_non_truncation_failure_never_splits(tmp_path):
    llm = _Pass2SizeLimitedLLM(_four_class_script([]), max_batch=2, truncated=False)
    with pytest.raises(ExtractionError, match="failed after one retry"):
        induce_spec_from_documents([_four_class_source(tmp_path)], llm)


# ---------------------------------------------------------------------------
# Pass 2 batch sizing and concurrent induction
# ---------------------------------------------------------------------------


def test_pass2_batch_size_parameter(tmp_path):
    llm = ScriptedLLM(_four_class_script([{"classes": []}, {"classes": []}]))
    spec, _ = induce_spec_from_documents([_four_class_source(tmp_path)], llm, pass2_batch_size=2)
    pass2_calls = llm.calls_for("pass2")
    assert len(pass2_calls) == 2
    assert pass2_calls[0]["context"].endswith(":batch0")
    assert pass2_calls[1]["context"].endswith(":batch1")
    assert len(spec.models) == 4


class _RoutedLLM:
    """Thread-safe scripted callable keyed by the exact context tag.

    Context tags are deterministic under any worker count (documents by name,
    pass-2 batches by position), so routing by tag makes the parity test
    independent of call-completion order.
    """

    def __init__(self, table: dict[str, Any]) -> None:
        self.table = dict(table)
        self._lock = __import__("threading").Lock()
        self.contexts: list[str] = []

    def __call__(self, *, prompt: dict[str, str], schema_json: str, context: str) -> Any:
        with self._lock:
            self.contexts.append(context)
            assert context in self.table, f"unexpected context: {context!r}"
            return copy.deepcopy(self.table[context])


def _routed_invoice_llm() -> _RoutedLLM:
    script = invoice_script().script
    return _RoutedLLM(
        {
            "templategen_pass1_classes:invoice_1.md": script["pass1"][0],
            "templategen_pass2_fields:invoice_1.md:batch0": script["pass2"][0],
            "templategen_pass3_edges:invoice_1.md": script["pass3"][0],
            "templategen_pass1_classes:invoice_2.md": script["pass1"][1],
            "templategen_pass2_fields:invoice_2.md:batch0": script["pass2"][1],
            "templategen_pass3_edges:invoice_2.md": script["pass3"][1],
        }
    )


def test_parallel_induction_matches_sequential():
    """workers > 1 changes wall time, never the induced spec or the report."""
    sources = [FIXTURE_DOCS / "invoice_1.md", FIXTURE_DOCS / "invoice_2.md"]
    spec_seq, report_seq = induce_spec_from_documents(sources, _routed_invoice_llm())
    spec_par, report_par = induce_spec_from_documents(sources, _routed_invoice_llm(), workers=4)
    assert spec_par.model_dump() == spec_seq.model_dump()
    assert [d.model_dump() for d in report_par.documents] == [
        d.model_dump() for d in report_seq.documents
    ]


# ---------------------------------------------------------------------------
# prepare_document_text: budget sampler
# ---------------------------------------------------------------------------


def sampler_text() -> str:
    return (
        "HEAD-MARKER\n"
        + "prose line\n" * 60
        + "x prose\n" * 40
        + "## STRUCT-HEADING\n"
        + "| col_a | col_b |\n"
        + "more prose\n" * 60
        + "closing prose\n" * 20
        + "TAIL-MARKER"
    )


def test_sampler_markers_and_content_survival(tmp_path):
    source = tmp_path / "long.md"
    text = sampler_text()
    source.write_text(text, encoding="utf-8")
    prepared = prepare_document_text(source, budget_chars=800)
    assert prepared.sampled
    header = prepared.markdown.splitlines()[0]
    assert re.fullmatch(rf"\[docling-graph\] sampled \d+ of {len(text)} chars .*", header)
    assert prepared.markdown.count(ELISION_MARKER) == 2
    # Head and tail content survive; structure-dense middle lines survive.
    assert "HEAD-MARKER" in prepared.markdown
    assert "TAIL-MARKER" in prepared.markdown
    assert "## STRUCT-HEADING" in prepared.markdown
    assert "| col_a | col_b |" in prepared.markdown
    # Non-structural middle prose is elided.
    assert "more prose" not in prepared.markdown
    # Budget respected up to the constant header/marker overhead.
    assert len(prepared.markdown) <= 800 + 250


def test_sampler_is_deterministic(tmp_path):
    source = tmp_path / "long.md"
    source.write_text(sampler_text(), encoding="utf-8")
    first = prepare_document_text(source, budget_chars=800)
    second = prepare_document_text(source, budget_chars=800)
    assert first == second


def test_under_budget_text_is_untouched(tmp_path):
    source = tmp_path / "short.md"
    source.write_text(GATE_DOC, encoding="utf-8")
    prepared = prepare_document_text(source, budget_chars=24_000)
    assert not prepared.sampled
    assert prepared.markdown == GATE_DOC
    assert prepared.name == "short.md"


class StubProcessor:
    """Duck-typed DocumentProcessor stand-in."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.converted: list[str] = []

    def convert_to_docling_doc(self, source: str) -> dict:
        self.converted.append(source)
        return {"source": source}

    def extract_full_markdown(self, document: dict) -> str:
        assert document["source"] in self.converted
        return self.text


def test_non_text_input_uses_injected_processor():
    stub = StubProcessor("# Converted document\nBody text.")
    prepared = prepare_document_text("scan.pdf", doc_processor=stub)
    assert stub.converted == ["scan.pdf"]
    assert prepared.markdown == "# Converted document\nBody text."
    assert prepared.name == "scan.pdf"


def test_non_text_input_without_processor_raises():
    with pytest.raises(ValueError, match="DocumentProcessor"):
        prepare_document_text("scan.pdf")


# ---------------------------------------------------------------------------
# prepare_document_text: DoclingDocument cache (design §4.1/§7.2)
# ---------------------------------------------------------------------------


class StubDoclingDocument:
    """Duck-typed DoclingDocument: only what DoclingExporter's JSON export reads."""

    def __init__(self, source: str) -> None:
        self.source = source

    def export_to_dict(self) -> dict:
        return {"schema_name": "DoclingDocument", "origin": self.source}


class CachingStubProcessor:
    """Duck-typed DocumentProcessor whose conversion yields a cacheable document."""

    def __init__(self, text: str) -> None:
        self.text = text

    def convert_to_docling_doc(self, source: str) -> StubDoclingDocument:
        return StubDoclingDocument(source)

    def extract_full_markdown(self, document: StubDoclingDocument) -> str:
        return self.text


def test_prepare_document_text_caches_converted_document(tmp_path):
    cache_dir = tmp_path / "cache"
    stub = CachingStubProcessor("# Converted document\nBody text.")
    prepared = prepare_document_text("scan.pdf", doc_processor=stub, cache_dir=cache_dir)
    assert prepared.cache_path == cache_dir / "scan.document.json"
    data = json.loads(prepared.cache_path.read_text(encoding="utf-8"))
    assert data["schema_name"] == "DoclingDocument"
    # Only the document JSON lands in the cache — no markdown/DocLang exports.
    assert sorted(p.name for p in cache_dir.iterdir()) == ["scan.document.json"]


def test_cache_skipped_for_text_sources_and_without_cache_dir(tmp_path):
    source = tmp_path / "short.md"
    source.write_text(GATE_DOC, encoding="utf-8")
    # Text sources are never converted, so there is nothing to cache.
    prepared = prepare_document_text(source, cache_dir=tmp_path / "cache")
    assert prepared.cache_path is None
    assert not (tmp_path / "cache").exists()
    # Converted sources without a cache_dir keep the old no-cache behavior.
    stub = CachingStubProcessor("# Converted\nBody.")
    assert prepare_document_text("scan.pdf", doc_processor=stub).cache_path is None


def test_induction_report_carries_cache_paths(tmp_path):
    cache_dir = tmp_path / "cache"
    stub = CachingStubProcessor(GATE_DOC)
    llm = ScriptedLLM(
        {
            "pass1": [
                {"classes": [p1("Register", is_root=True, ident=("register_id", "", ["GOOD-1"]))]}
            ],
            "pass2": [{"classes": []}],
            "pass3": [{"edges": []}],
        }
    )
    _, report = induce_spec_from_documents(
        ["register.pdf"], llm, doc_processor=stub, cache_dir=cache_dir
    )
    assert report.documents[0].cache_path == cache_dir / "register.document.json"
    assert report.documents[0].cache_path.is_file()


# ---------------------------------------------------------------------------
# prepare_document_text: URL sources and direct DocumentContent
# ---------------------------------------------------------------------------


def test_source_display_name_shapes():
    assert source_display_name(Path("/tmp/scan.pdf")) == "scan.pdf"
    assert source_display_name("https://example.com/docs/spec-sheet.pdf") == "spec-sheet.pdf"
    assert source_display_name("https://example.com/") == "example.com"
    assert source_display_name(DocumentContent(name="inline doc", text="x")) == "inline doc"


def test_url_source_uses_injected_processor():
    stub = StubProcessor("# Converted page\nBody text.")
    prepared = prepare_document_text("https://example.com/docs/report.pdf", doc_processor=stub)
    # The URL reaches the processor verbatim — never mangled through Path.
    assert stub.converted == ["https://example.com/docs/report.pdf"]
    assert prepared.markdown == "# Converted page\nBody text."
    assert prepared.name == "report.pdf"


def test_url_source_without_processor_raises():
    with pytest.raises(ValueError, match="DocumentProcessor"):
        prepare_document_text("https://example.com/report.md")


def test_url_source_cache_stem_is_sanitized(tmp_path):
    cache_dir = tmp_path / "cache"
    stub = CachingStubProcessor("# Converted\nBody.")
    prepared = prepare_document_text(
        "https://example.com/docs/spec%20sheet.pdf", doc_processor=stub, cache_dir=cache_dir
    )
    assert prepared.cache_path == cache_dir / "spec_20sheet.document.json"
    assert prepared.cache_path.is_file()


def test_document_content_is_read_directly(tmp_path):
    content = DocumentContent(name="inline register", text=GATE_DOC)
    prepared = prepare_document_text(content, cache_dir=tmp_path / "cache")
    assert prepared.name == "inline register"
    assert prepared.markdown == GATE_DOC
    # Nothing was converted, so nothing is cached.
    assert prepared.cache_path is None
    assert not (tmp_path / "cache").exists()


def test_document_content_is_budget_sampled():
    prepared = prepare_document_text(
        DocumentContent(name="long doc", text=sampler_text()), budget_chars=800
    )
    assert prepared.sampled
    assert ELISION_MARKER in prepared.markdown


def test_induction_accepts_document_content_directly():
    llm = ScriptedLLM(
        {
            "pass1": [
                {"classes": [p1("Register", is_root=True, ident=("register_id", "", ["GOOD-1"]))]}
            ],
            "pass2": [{"classes": []}],
            "pass3": [{"edges": []}],
        }
    )
    spec, report = induce_spec_from_documents(
        [DocumentContent(name="register.md", text=GATE_DOC)], llm
    )
    assert spec.root == "Register"
    assert report.documents[0].name == "register.md"


def test_near_empty_sources_are_skipped_and_all_empty_fails(tmp_path):
    source = tmp_path / "empty.md"
    source.write_text("too short", encoding="utf-8")
    llm = ScriptedLLM({})
    with pytest.raises(ExtractionError, match="no usable content"):
        induce_spec_from_documents([source], llm)
    assert llm.calls == []


# ---------------------------------------------------------------------------
# Cross-document merge (deterministic)
# ---------------------------------------------------------------------------


class TestMerge:
    def test_type_promotion_lattice(self):
        draft, report, _ = merge_documents(
            [
                doc(
                    "d1",
                    cc(
                        "Doc",
                        is_root=True,
                        identity=("doc_id", ["D-1"]),
                        fields=[fc("amount", type="int"), fc("code", type="int")],
                    ),
                ),
                doc(
                    "d2",
                    cc(
                        "Doc",
                        is_root=True,
                        identity=("doc_id", ["D-2"]),
                        fields=[fc("amount", type="float"), fc("code", type="str")],
                    ),
                ),
            ]
        )
        model = draft_model(draft, "Doc")
        assert draft_field(model, "amount")["type"] == "float"
        assert draft_field(model, "code")["type"] == "str"
        assert len(report.by_kind("type_promotion")) == 2

    def test_kind_majority_vote(self):
        draft, report, _ = merge_documents(
            [
                doc("d1", root_cls(1), cc("Party", "entity", identity=("name", ["Acme GmbH"]))),
                doc("d2", root_cls(2), cc("Party", "entity", identity=("name", ["Beta SARL"]))),
                doc("d3", root_cls(3), cc("Party", "component")),
            ]
        )
        assert draft_model(draft, "Party")["kind"] == "entity"
        assert any("2" in d.message for d in report.by_kind("kind_vote"))

    def test_kind_tie_breaks_on_surviving_identity(self):
        draft, _, _ = merge_documents(
            [
                doc("d1", root_cls(1), cc("Party", "entity", identity=("name", ["Acme GmbH"]))),
                doc("d2", root_cls(2), cc("Party", "component")),
            ]
        )
        assert draft_model(draft, "Party")["kind"] == "entity"

    def test_kind_tie_without_identity_resolves_component(self):
        draft, _, gaps = merge_documents(
            [
                doc("d1", root_cls(1), cc("Party", "entity")),  # no identity survived
                doc("d2", root_cls(2), cc("Party", "component")),
            ]
        )
        assert draft_model(draft, "Party")["kind"] == "component"
        assert gaps == []  # tie resolution, not a demotion

    def test_entity_without_identity_evidence_demoted_with_gap(self):
        draft, report, gaps = merge_documents(
            [
                doc("d1", root_cls(1), cc("Party", "entity")),
                doc("d2", root_cls(2), cc("Party", "entity")),
            ]
        )
        party = draft_model(draft, "Party")
        assert party["kind"] == "component"
        assert party["identity_fields"] == []
        assert report.by_kind("identity_demotion")
        assert [(g.model, g.kind) for g in gaps] == [("Party", "missing_identity")]

    def test_enum_union_over_cap_demotes_to_str(self):
        members = [f"M{i}" for i in range(25)]
        draft, report, _ = merge_documents(
            [
                doc(
                    "d1",
                    cc(
                        "Doc",
                        is_root=True,
                        identity=("doc_id", ["D-1"]),
                        fields=[fc("category", enum_name="Category", enum_members=members)],
                    ),
                )
            ]
        )
        assert draft["enums"] == []
        field = draft_field(draft_model(draft, "Doc"), "category")
        assert field["type"] == "str"
        assert "One of: M0" in field["description"]
        assert report.by_kind("enum_demotion")

    def test_examples_prefer_distinct_documents(self):
        draft, _, _ = merge_documents(
            [
                doc(
                    "d1",
                    cc(
                        "Doc",
                        is_root=True,
                        identity=("doc_id", ["D-1"]),
                        fields=[fc("color", examples=["red", "blue", "green", "black"])],
                    ),
                ),
                doc(
                    "d2",
                    cc(
                        "Doc",
                        is_root=True,
                        identity=("doc_id", ["D-2"]),
                        fields=[fc("color", examples=["white", "grey"])],
                    ),
                ),
            ]
        )
        field = draft_field(draft_model(draft, "Doc"), "color")
        assert field["examples"] == ["red", "white", "blue", "grey", "green"]

    def test_rare_field_flag(self):
        draft, report, _ = merge_documents(
            [
                doc("d1", root_cls(1)),
                doc(
                    "d2",
                    cc(
                        "Doc",
                        is_root=True,
                        identity=("doc_id", ["D-2"]),
                        fields=[fc("notes", description="Free-text notes.")],
                    ),
                ),
                doc("d3", root_cls(3)),
            ]
        )
        field = draft_field(draft_model(draft, "Doc"), "notes")
        assert field["description"] == "Rare: Free-text notes."
        assert report.by_kind("rare_field")
        # The identity field is in all three documents: never flagged.
        assert not draft_field(draft_model(draft, "Doc"), "doc_id")["description"].startswith(
            "Rare:"
        )

    def test_documented_max_stays_documented_in_draft(self):
        draft, _, _ = merge_documents(
            [
                doc(
                    "d1",
                    cc("Doc", is_root=True, identity=("doc_id", ["D-1"]), documented_max_count=4),
                ),
                doc(
                    "d2",
                    cc("Doc", is_root=True, identity=("doc_id", ["D-2"]), documented_max_count=6),
                ),
            ]
        )
        # Max across docs, NOT doubled here — repair_draft doubles exactly once.
        assert draft_model(draft, "Doc")["max_instances"] == 6

    def test_max_models_overflow_is_reported(self):
        classes = [root_cls(1)] + [cc(f"Extra{i}", "component") for i in range(1, 5)]
        draft, report, _ = merge_documents([doc("d1", *classes)], max_models=3)
        assert len(draft["models"]) == 3
        assert len(report.by_kind("overflow_drop")) == 2

    def test_enum_display_name_avoids_class_collision(self):
        """Class 'Status' + enum 'Status': the enum draws from the shared
        taken-name pool, so the draft never carries the collision the linter's
        single rename cascade would resolve by corrupting edge targets."""
        draft, _, _ = merge_documents(
            [
                doc(
                    "d1",
                    cc(
                        "Doc",
                        is_root=True,
                        identity=("doc_id", ["D-1"]),
                        fields=[
                            fc("state", enum_name="Status", enum_members=["open", "closed"]),
                            fc("previous_state", enum_name="Status", enum_members=["reopened"]),
                            fc("tracked_status", type="Status", role="edge"),
                        ],
                    ),
                    cc("Status", identity=("name", ["Open"])),
                )
            ]
        )
        # One enum entry (both fields reference the same proposed enum), suffixed.
        assert [e["name"] for e in draft["enums"]] == ["Status_2"]
        assert draft["enums"][0]["members"] == ["open", "closed", "reopened"]
        model = draft_model(draft, "Doc")
        assert draft_field(model, "state")["type"] == "Status_2"
        assert draft_field(model, "previous_state")["type"] == "Status_2"
        # The edge still targets the CLASS, untouched by the enum rename.
        assert draft_field(model, "tracked_status")["type"] == "Status"
        assert draft_model(draft, "Status")["name"] == "Status"


TICKET_DOC = (
    "# TICKET T-1\n\n"
    "Status: Open Review is the current workflow state of this ticket.\n"
    "Priority: high is printed in the corner of the ticket header sheet.\n"
    "The board tracks each ticket status change with a timestamp entry.\n"
    "Escalations move the ticket into the queue of the on-call reviewer.\n"
)


def test_enum_class_collision_survives_full_induction(tmp_path):
    """End-to-end: class 'Status' + enum_name 'Status' get distinct final names
    and every reference points at the right entity post-repair."""
    source = tmp_path / "ticket.md"
    source.write_text(TICKET_DOC, encoding="utf-8")
    llm = ScriptedLLM(
        {
            "pass1": [
                {
                    "classes": [
                        p1("Ticket", is_root=True, ident=("ticket_id", "", ["T-1"])),
                        p1("Status", ident=("name", "", ["Open Review"])),
                    ]
                }
            ],
            "pass2": [
                {
                    "classes": [
                        {
                            "class_name": "Ticket",
                            "fields": [
                                p2f(
                                    "priority",
                                    "enum:Status",
                                    examples=["high"],
                                    enum_members=["high", "low"],
                                )
                            ],
                        }
                    ]
                }
            ],
            "pass3": [
                {
                    "edges": [
                        p3e(
                            "Ticket",
                            "status",
                            "Status",
                            label="HAS_STATUS",
                            evidence=["ticket status change"],
                        )
                    ]
                }
            ],
        }
    )
    spec, _report = induce_spec_from_documents([source], llm)
    model_names = {m.name for m in spec.models}
    assert {"Ticket", "Status"} <= model_names
    # The enum kept a distinct display name (allocated from the shared pool;
    # repair_draft may PascalCase the suffix, e.g. Status_2 -> Status2).
    assert len(spec.enums) == 1
    enum_name = spec.enums[0].name
    assert enum_name != "Status"
    assert enum_name not in model_names
    edge = get_field(spec, "Ticket", "status")
    assert edge.role == "edge"
    assert edge.type == "Status"  # the model, not the renamed enum
    prop = get_field(spec, "Ticket", "priority")
    assert prop.role == "property"
    assert prop.type == enum_name  # the enum, not the model


# ---------------------------------------------------------------------------
# Gap-fill
# ---------------------------------------------------------------------------


def gap_spec() -> TemplateSpec:
    return TemplateSpec(
        module_docstring="Test.",
        root="Doc",
        models=[
            ModelSpec(
                name="Doc",
                kind="root",
                docstring="Doc.",
                identity_fields=["doc_id"],
                fields=[
                    FieldSpec(name="doc_id", role="identity"),
                    FieldSpec(name="notes"),
                ],
            ),
            ModelSpec(
                name="Item",
                kind="component",
                docstring="An item.",
                fields=[FieldSpec(name="label_text")],
            ),
        ],
    )


def gap_list() -> list[SpecGap]:
    return [
        SpecGap(model="Doc", field="doc_id", kind="missing_examples"),
        SpecGap(model="Doc", field=None, kind="missing_docstring"),
        SpecGap(model="Doc", field="notes", kind="missing_description"),
        SpecGap(model="Item", field=None, kind="ambiguous_kind"),
    ]


def gf(model, field, kind, *, docstring="", description="", examples=()):
    return {
        "model": model,
        "field": field,
        "kind": kind,
        "docstring": docstring,
        "description": description,
        "examples": list(examples),
    }


class TestGapFill:
    def test_fills_only_declared_gaps_and_marks_provenance(self):
        spec = gap_spec()
        llm = ScriptedLLM(
            {
                "gapfill": [
                    {
                        "fills": [
                            gf("Doc", "doc_id", "missing_examples", examples=["DOC-1", "DOC-2"]),
                            gf("Doc", "", "missing_docstring", docstring="A document record."),
                            gf(
                                "Doc",
                                "notes",
                                "missing_description",
                                description="LOOK FOR the notes block.",
                            ),
                            # Undeclared fill: silently discarded.
                            gf("Item", "", "missing_docstring", docstring="INJECTED"),
                            # Declared but structurally unfillable: stays open.
                            gf("Item", "", "ambiguous_kind", docstring="also ignored"),
                        ]
                    }
                ]
            }
        )
        filled, remaining = fill_gaps(spec, gap_list(), llm)
        doc_model = next(m for m in filled.models if m.name == "Doc")
        item_model = next(m for m in filled.models if m.name == "Item")
        assert doc_model.docstring == "A document record."
        assert next(f for f in doc_model.fields if f.name == "doc_id").examples == [
            "DOC-1",
            "DOC-2",
        ]
        assert next(f for f in doc_model.fields if f.name == "notes").description == (
            "LOOK FOR the notes block."
        )
        assert doc_model.provenance == "gapfill"
        # Untouched model: content and provenance intact.
        assert item_model.docstring == "An item."
        assert item_model.provenance == "induced"
        assert [(g.model, g.kind) for g in remaining] == [("Item", "ambiguous_kind")]
        # The input spec was never mutated.
        assert spec.models[0].docstring == "Doc."
        assert spec.models[0].provenance == "induced"

    def test_structure_cannot_change(self):
        spec = gap_spec()
        llm = ScriptedLLM(
            {
                "gapfill": [
                    {
                        "fills": [
                            gf("Doc", "", "missing_docstring", docstring="A document record."),
                        ]
                    }
                ]
            }
        )
        filled, _ = fill_gaps(spec, gap_list(), llm)
        assert [m.name for m in filled.models] == [m.name for m in spec.models]
        assert [len(m.fields) for m in filled.models] == [len(m.fields) for m in spec.models]
        assert [e.name for e in filled.enums] == [e.name for e in spec.enums]

    def test_failed_call_leaves_gaps_open(self):
        spec = gap_spec()

        def exploding(**_kwargs: Any) -> dict:
            raise RuntimeError("provider down")

        filled, remaining = fill_gaps(spec, gap_list(), exploding)
        assert filled is spec
        assert remaining == gap_list()

    def test_no_gaps_no_call(self):
        spec = gap_spec()
        llm = ScriptedLLM({})
        filled, remaining = fill_gaps(spec, [], llm)
        assert filled is spec
        assert remaining == []
        assert llm.calls == []


# ---------------------------------------------------------------------------
# Schema shapes (provider safety + the gap-fill structural firewall)
# ---------------------------------------------------------------------------


def _walk_objects(node: Any, found: list[dict]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            _walk_objects(value, found)
    elif isinstance(node, list):
        for value in node:
            _walk_objects(value, found)


@pytest.mark.parametrize(
    "schema_fn",
    [class_inventory_schema, fields_schema, relationships_schema, gapfill_schema],
)
def test_schemas_are_strict_and_shallow(schema_fn):
    schema = schema_fn()
    objects: list[dict] = []
    _walk_objects(schema, objects)
    for obj in objects:
        assert obj.get("additionalProperties") is False
        assert sorted(obj.get("required", [])) == sorted(obj.get("properties", {}))
    # $defs live at the root only.
    assert all("$defs" not in obj or obj is schema for obj in objects)


def test_gapfill_schema_is_structurally_incapable_of_injection():
    schema = gapfill_schema()
    objects: list[dict] = []
    _walk_objects(schema, objects)
    property_names: set[str] = set()
    for obj in objects:
        property_names.update(obj.get("properties", {}))
    # Keyed by (model, field, kind); value slots ONLY docstring/description/examples.
    assert property_names == {
        "fills",
        "model",
        "field",
        "kind",
        "docstring",
        "description",
        "examples",
    }
    # No slot through which structure could enter the SPEC.
    forbidden = {
        "models",
        "classes",
        "fields",
        "edges",
        "edge_label",
        "identity_fields",
        "enum_members",
        "is_list",
        "reference",
    }
    assert not (property_names & forbidden)


# ---------------------------------------------------------------------------
# prepare_document_windows: oversized documents split into spread windows
# ---------------------------------------------------------------------------


def _lined_text(lines: int, tag: str) -> str:
    return "".join(f"{tag} content line number {i:04d} with prose.\n" for i in range(lines))


def test_small_document_is_a_single_full_unit(tmp_path):
    source = tmp_path / "small.md"
    source.write_text(GATE_DOC, encoding="utf-8")
    units = prepare_document_windows(source, budget_chars=24_000)
    assert len(units) == 1
    assert units[0].markdown == GATE_DOC
    assert units[0].window_index is None
    assert units[0].name == "small.md"


def test_oversized_document_tiles_into_full_coverage_windows(tmp_path):
    text = _lined_text(60, "BODY")  # ~2.7k chars
    source = tmp_path / "big.md"
    source.write_text(text, encoding="utf-8")
    units = prepare_document_windows(source, budget_chars=1_000)
    assert [u.name for u in units] == [f"big.md [{i + 1}/{len(units)}]" for i in range(len(units))]
    assert len(units) == 3  # ceil(2.7k / 1k)
    assert all(u.window_count == len(units) for u in units)
    # Full coverage: every source line appears in some window.
    joined = "\n".join(u.markdown for u in units)
    for i in range(60):
        assert f"number {i:04d}" in joined
    assert not units[0].sampled  # tiling regime: nothing elided
    # Windows are line-aligned and self-describing.
    for u in units:
        body = u.markdown.split("\n", 1)[1]
        assert body.startswith("BODY ")
        assert u.markdown.startswith("[docling-graph] window ")


def test_giant_document_caps_windows_and_marks_sampling(tmp_path):
    text = _lined_text(2_000, "HUGE")  # ~90k chars >> 6 x 1k cap
    source = tmp_path / "huge.md"
    source.write_text(text, encoding="utf-8")
    units = prepare_document_windows(source, budget_chars=1_000, max_windows=6)
    assert len(units) == 6
    assert all(u.sampled for u in units)  # gaps between windows
    # Evenly spread: first window starts at the head, last covers the tail.
    assert "number 0000" in units[0].markdown
    assert "number 1999" in units[-1].markdown


def test_window_cache_rides_on_first_window_only(tmp_path):
    stub = CachingStubProcessor(_lined_text(60, "SCAN"))
    units = prepare_document_windows(
        "scan.pdf", doc_processor=stub, budget_chars=1_000, cache_dir=tmp_path / "cache"
    )
    assert len(units) > 1
    assert units[0].cache_path is not None and units[0].cache_path.is_file()
    assert all(u.cache_path is None for u in units[1:])


def test_windowed_document_flows_through_induction():
    """Two windows of one document contribute classes only their own text
    holds — proof the whole body (not a head-biased sample) reaches the LLM."""
    part_a = "Alpha section line with marker ALPHA-77 in the text.\n" * 12
    part_b = "Beta section line with marker BETA-88 in the text.\n" * 12
    content = DocumentContent(name="big.md", text=part_a + part_b)

    def pass1(cls_name: str, marker: str, is_root: bool) -> dict:
        return {"classes": [p1(cls_name, is_root=is_root, ident=("name", "", [marker]))]}

    llm = _RoutedLLM(
        {
            "templategen_pass1_classes:big.md [1/2]": pass1("AlphaThing", "ALPHA-77", True),
            "templategen_pass2_fields:big.md [1/2]:batch0": {"classes": []},
            "templategen_pass3_edges:big.md [1/2]": {"edges": []},
            "templategen_pass1_classes:big.md [2/2]": pass1("BetaThing", "BETA-88", False),
            "templategen_pass2_fields:big.md [2/2]:batch0": {"classes": []},
            "templategen_pass3_edges:big.md [2/2]": {"edges": []},
        }
    )
    spec, report = induce_spec_from_documents([content], llm, budget_chars=700)
    names = {m.name for m in spec.models}
    assert {"AlphaThing", "BetaThing"} <= names
    # The verbatim gate ran against each window's own text.
    assert get_field(spec, "AlphaThing", "name").examples == ["ALPHA-77"]
    assert get_field(spec, "BetaThing", "name").examples == ["BETA-88"]
    assert [s.name for s in report.documents] == ["big.md [1/2]", "big.md [2/2]"]
    assert report.units_total == 2


def test_rare_field_counts_physical_documents_not_windows():
    def unit(name: str, extra_fields: list) -> DocumentCandidates:
        return doc(
            name,
            root_cls(),
            cc("Item", identity=("name", ["Widget"]), fields=extra_fields),
        )

    units = [
        unit("big.md [1/3]", [fc("color", examples=["red"])]),
        unit("big.md [2/3]", []),
        unit("big.md [3/3]", []),
    ]
    # Three windows of ONE document: a mid-document field is never "rare".
    draft, report, _ = merge_documents(units, doc_groups=[0, 0, 0], group_names=["big.md"])
    color = draft_field(draft_model(draft, "Item"), "color")
    assert not color["description"].startswith("Rare:")
    assert not report.by_kind("rare_field")
    assert draft_model(draft, "Item")["source_ref"] == "induced from: big.md"

    # The same shape across three PHYSICAL documents stays flagged.
    draft, report, _ = merge_documents(units, doc_groups=[0, 1, 2])
    color = draft_field(draft_model(draft, "Item"), "color")
    assert color["description"].startswith("Rare:")
    assert report.by_kind("rare_field")


# ---------------------------------------------------------------------------
# Saturation stop and the unit cap (large corpora)
# ---------------------------------------------------------------------------


class _ConstantLLM:
    """Thread-safe callable returning the same payload per pass, any unit."""

    def __init__(self, by_pass: dict[str, Any]) -> None:
        self.by_pass = dict(by_pass)
        self.contexts: list[str] = []
        self._lock = __import__("threading").Lock()

    def __call__(self, *, prompt: dict[str, str], schema_json: str, context: str) -> Any:
        with self._lock:
            self.contexts.append(context)
        for key, payload in self.by_pass.items():
            if key in context:
                return copy.deepcopy(payload)
        raise AssertionError(f"unexpected context: {context!r}")


def _register_corpus(count: int) -> list[DocumentContent]:
    return [DocumentContent(name=f"doc{i:02d}.md", text=GATE_DOC) for i in range(count)]


def _register_llm() -> _ConstantLLM:
    return _ConstantLLM(
        {
            "pass1": {
                "classes": [p1("Register", is_root=True, ident=("register_id", "", ["GOOD-1"]))]
            },
            "pass2": {"classes": []},
            "pass3": {"edges": []},
        }
    )


def test_saturation_stops_a_homogeneous_corpus():
    llm = _register_llm()
    spec, report = induce_spec_from_documents(_register_corpus(12), llm)
    assert spec.root == "Register"
    # Unit 1 is novel; units 2-7 are quiet -> streak of 6 stops the run.
    assert len(report.documents) == 7
    assert len(report.skipped_saturated) == 5
    assert report.units_total == 12
    assert len(llm.contexts) == 7 * 3  # three passes per induced unit


def test_saturation_disabled_induces_everything():
    llm = _register_llm()
    _spec, report = induce_spec_from_documents(_register_corpus(12), llm, saturation=False)
    assert len(report.documents) == 12
    assert report.skipped_saturated == []


def test_saturation_never_engages_below_the_floor():
    llm = _register_llm()
    _spec, report = induce_spec_from_documents(_register_corpus(8), llm)
    assert len(report.documents) == 8
    assert report.skipped_saturated == []


def test_max_units_cap_skips_and_reports():
    llm = _register_llm()
    _spec, report = induce_spec_from_documents(
        _register_corpus(12), llm, max_units=4, saturation=False
    )
    assert len(report.documents) == 4
    # Without saturation the processing order is the source order.
    assert [s.name for s in report.documents] == [f"doc{i:02d}.md" for i in range(4)]
    assert report.skipped_capped == [f"doc{i:02d}.md" for i in range(4, 12)]


def test_saturation_order_is_deterministic():
    first = induce_spec_from_documents(_register_corpus(12), _register_llm())
    second = induce_spec_from_documents(_register_corpus(12), _register_llm())
    assert first[0].model_dump() == second[0].model_dump()
    assert [d.name for d in first[1].documents] == [d.name for d in second[1].documents]
    assert first[1].skipped_saturated == second[1].skipped_saturated


# ---------------------------------------------------------------------------
# Pass schemas: every array is bounded (degenerate-repetition wall)
# ---------------------------------------------------------------------------


def _walk_arrays(node: Any, path: str = "$") -> list[str]:
    unbounded: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "array" and "maxItems" not in node:
            unbounded.append(path)
        for key, value in node.items():
            unbounded.extend(_walk_arrays(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            unbounded.extend(_walk_arrays(value, f"{path}[{i}]"))
    return unbounded


@pytest.mark.parametrize(
    "schema_fn",
    [class_inventory_schema, fields_schema, relationships_schema, gapfill_schema],
)
def test_every_pass_schema_array_is_bounded(schema_fn):
    """Guided decoding constrains shape, not repetition: an unbounded array is
    where a looping model pours tokens until any max_tokens budget truncates."""
    assert _walk_arrays(schema_fn()) == []


# ---------------------------------------------------------------------------
# One-shot strategy: full ontology per call, same gates
# ---------------------------------------------------------------------------


def _combine_oneshot(p1_payload: dict, p2_payload: dict, p3_payload: dict) -> dict:
    """Nest the pass-2 fields and pass-3 edges inside each pass-1 class."""
    fields_by_class = {e["class_name"]: e["fields"] for e in p2_payload["classes"]}
    edges_by_source: dict[str, list[dict]] = {}
    for edge in p3_payload["edges"]:
        payload = {k: v for k, v in edge.items() if k != "source"}
        edges_by_source.setdefault(edge["source"], []).append(payload)
    return {
        "classes": [
            {
                **entry,
                "fields": fields_by_class.get(entry["name"], []),
                "edges": edges_by_source.get(entry["name"], []),
            }
            for entry in p1_payload["classes"]
        ]
    }


def _oneshot_invoice_llm() -> _RoutedLLM:
    script = invoice_script().script
    return _RoutedLLM(
        {
            "templategen_oneshot:invoice_1.md": _combine_oneshot(
                script["pass1"][0], script["pass2"][0], script["pass3"][0]
            ),
            "templategen_oneshot:invoice_2.md": _combine_oneshot(
                script["pass1"][1], script["pass2"][1], script["pass3"][1]
            ),
        }
    )


def test_oneshot_strategy_single_call_per_unit_with_gates():
    """One LLM call per document; the composed parser runs the exact same
    evidence gates (verbatim examples, unknown edge targets) as three-pass."""
    llm = _oneshot_invoice_llm()
    spec, report = induce_spec_from_documents(
        [FIXTURE_DOCS / "invoice_1.md", FIXTURE_DOCS / "invoice_2.md"],
        llm,
        strategy="one-shot",
    )
    assert len(llm.contexts) == 2  # exactly one call per document
    assert spec.root == "Invoice"
    names = {m.name for m in spec.models}
    assert {"Invoice", "Party", "LineItem", "Address"} <= names
    # Pass-2 content arrived: fields with verbatim-gated examples.
    assert get_field(spec, "Party", "vat_number").examples == [
        "DE-812-940-113",
        "IE-6388047V",
    ]
    # The hallucinated example never survives the verbatim gate.
    assert "NOT-IN-DOC-XYZ" not in get_field(spec, "Invoice", "purchase_order").examples
    # Pass-3 content arrived: labeled edges; the unknown 'Warehouse' target dropped.
    assert get_field(spec, "Invoice", "issued_by").edge_label == "ISSUED_BY"
    assert get_field(spec, "Invoice", "line_items").is_list is True
    assert all(f.name != "ghost" for f in get_model(spec, "Invoice").fields)
    assert any(s.edges_dropped for s in report.documents)


def test_oneshot_matches_three_pass_on_the_same_content():
    """Same candidates in, same spec out — the strategy only changes call shape."""
    sources = [FIXTURE_DOCS / "invoice_1.md", FIXTURE_DOCS / "invoice_2.md"]
    spec_oneshot, _ = induce_spec_from_documents(
        sources, _oneshot_invoice_llm(), strategy="one-shot"
    )
    spec_passes, _ = induce_spec_from_documents(sources, _routed_invoice_llm())
    assert spec_oneshot.model_dump() == spec_passes.model_dump()


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="strategy"):
        induce_spec_from_documents(
            [DocumentContent(name="x.md", text=GATE_DOC)], _register_llm(), strategy="magic"
        )
