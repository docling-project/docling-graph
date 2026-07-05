"""Tests for the DocLang control-character sanitizer."""

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel

from docling_graph.core.utils.doclang_sanitizer import (
    _has_control_chars,
    sanitize_for_doclang,
)


def _doc_with_text(text: str) -> DoclingDocument:
    doc = DoclingDocument(name="sample")
    doc.add_page(page_no=1, size={"width": 612, "height": 792})
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox(l=10, t=10, r=100, b=30, coord_origin=CoordOrigin.TOPLEFT),
        charspan=(0, len(text)),
    )
    doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)
    return doc


def test_strips_nul_and_control_chars() -> None:
    doc = _doc_with_text("Total\x00 due\x0b: 50\x1f")
    clean = sanitize_for_doclang(doc)
    assert clean.texts[0].text == "Total due: 50"
    assert not _has_control_chars(clean)


def test_preserves_tab_newline_carriage_return() -> None:
    doc = _doc_with_text("line1\nline2\tcol\rend")
    clean = sanitize_for_doclang(doc)
    assert clean.texts[0].text == "line1\nline2\tcol\rend"


def test_clean_document_returned_unchanged_fast_path() -> None:
    doc = _doc_with_text("perfectly clean text")
    clean = sanitize_for_doclang(doc)
    # Fast path: no copy is made when nothing needs scrubbing.
    assert clean is doc


def test_original_document_never_mutated() -> None:
    doc = _doc_with_text("dirty\x00value")
    clean = sanitize_for_doclang(doc)
    assert doc.texts[0].text == "dirty\x00value"
    assert clean is not doc
    assert clean.texts[0].text == "dirtyvalue"


def test_sanitized_document_exports_to_doclang() -> None:
    doc = _doc_with_text("Amplitude\x00 sweeps")
    clean = sanitize_for_doclang(doc)
    # Would raise ValueError on the un-sanitized document.
    out = clean.export_to_doclang()
    assert "Amplitude sweeps" in out
