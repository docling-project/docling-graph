"""Root-identity resolution micro-pass (Phase 2 follow-up invariant).

The pass runs only when every declared root id field is empty after fill+merge,
excerpts the serialized document's head/tail/page furniture, and accepts a
returned value only when it is verbatim-anchored in that excerpt — fail-empty,
never fail-wrong.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from docling_graph.core.extractors.contracts.dense.catalog import NodeSpec
from docling_graph.core.extractors.contracts.dense.orchestrator import (
    DenseOrchestrator,
    _page_furniture_lines,
    _squash_anchor,
)


class _Doc:
    """Template stand-in (only __name__ is read by the pass)."""


def _spec_by_path(id_fields: list[str]) -> dict[str, NodeSpec]:
    return {
        "": NodeSpec(
            path="",
            node_type="Doc",
            id_fields=id_fields,
            parent_path="",
            field_name="",
            is_list=False,
        )
    }


def _stub(llm_response: Any) -> SimpleNamespace:
    calls: list[dict[str, Any]] = []

    def _llm(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if isinstance(llm_response, Exception):
            raise llm_response
        return llm_response

    return SimpleNamespace(_template=_Doc, _llm=_llm, calls=calls)


def test_page_furniture_lines_extracts_deduped_header_footer_texts() -> None:
    text = (
        "<doc>\n"
        '<page_header><location value="1"/> REF_0725 </page_header>\n'
        "<paragraph>body</paragraph>\n"
        '<page_header><location value="2"/> REF_0725 </page_header>\n'
        "<page_footer>Page 2</page_footer>\n"
        "</doc>"
    )
    furniture = _page_furniture_lines(text)
    assert furniture.splitlines() == ["REF_0725", "Page 2"]


def test_page_furniture_lines_empty_for_markdown() -> None:
    assert _page_furniture_lines("# Title\n\nplain markdown body") == ""


def test_resolve_accepts_furniture_anchored_value() -> None:
    """The footer code lives ONLY in mid-document page furniture (never in
    chunks, head, or tail) — the exact contract-domain failure shape."""
    body = "<paragraph>filler</paragraph>\n" * 200
    full = (
        "<doc>cover page text\n"
        + body
        + "<page_header>HABITATION_0725</page_header>\n"
        + body
        + "closing text</doc>"
    )
    stub = _stub({"reference_document": "HABITATION_0725"})
    root: dict[str, Any] = {"reference_document": "", "assureur": "AXA"}
    resolved = DenseOrchestrator._resolve_root_identity(
        stub, root, _spec_by_path(["reference_document"]), full, "t"
    )
    assert resolved == {"reference_document": "HABITATION_0725"}
    assert root["reference_document"] == "HABITATION_0725"
    assert len(stub.calls) == 1


def test_resolve_refuses_unanchored_value() -> None:
    """A value not printed in the excerpt is refused — the field stays empty
    for the stem fallback rather than locking in an invention."""
    stub = _stub({"reference_document": "TOTALLY-INVENTED-CODE-99"})
    root: dict[str, Any] = {"reference_document": ""}
    resolved = DenseOrchestrator._resolve_root_identity(
        stub, root, _spec_by_path(["reference_document"]), "cover text only", "t"
    )
    assert resolved == {}
    assert root["reference_document"] == ""


def test_resolve_skips_when_any_id_field_is_filled() -> None:
    stub = _stub({"reference_document": "whatever"})
    root: dict[str, Any] = {"reference_document": "REF-1"}
    resolved = DenseOrchestrator._resolve_root_identity(
        stub, root, _spec_by_path(["reference_document"]), "REF-1 cover", "t"
    )
    assert resolved == {}
    assert not stub.calls  # never even calls the LLM
    assert root["reference_document"] == "REF-1"


def test_resolve_refuses_overlong_and_empty_values() -> None:
    long_value = "x" * 200
    stub = _stub({"reference_document": long_value, "code": ""})
    root: dict[str, Any] = {"reference_document": "", "code": ""}
    resolved = DenseOrchestrator._resolve_root_identity(
        stub, root, _spec_by_path(["reference_document", "code"]), long_value, "t"
    )
    assert resolved == {}


def test_resolve_survives_llm_exception() -> None:
    stub = _stub(RuntimeError("boom"))
    root: dict[str, Any] = {"reference_document": ""}
    resolved = DenseOrchestrator._resolve_root_identity(
        stub, root, _spec_by_path(["reference_document"]), "cover text", "t"
    )
    assert resolved == {}
    assert root["reference_document"] == ""


def test_squash_anchor_folds_ocr_and_pua_glyphs() -> None:
    assert _squash_anchor("HABITATION_0725") == "habitation0725"
    assert _squash_anchor("steady state fl ow") == _squash_anchor("steady state flow")
