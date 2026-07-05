"""Unit tests for first-class DocLang input (detection, validation, handling)."""

import tempfile
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel

from docling_graph.core.input import (
    DoclangInputHandler,
    DoclangValidator,
    InputType,
    InputTypeDetector,
)
from docling_graph.exceptions import ConfigurationError, ValidationError


def _build_doc() -> DoclingDocument:
    doc = DoclingDocument(name="sample")
    doc.add_page(page_no=1, size={"width": 612, "height": 792})
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox(l=10, t=10, r=100, b=30, coord_origin=CoordOrigin.TOPLEFT),
        charspan=(0, 14),
    )
    doc.add_heading(text="Invoice #12345", prov=prov)
    return doc


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dclg_file(temp_dir):
    path = temp_dir / "sample.dclg"
    _build_doc().save_as_doclang(path)
    return path


class TestDoclangDetection:
    def test_detect_dclg(self, dclg_file):
        assert InputTypeDetector.detect(str(dclg_file), mode="cli") == InputType.DOCLANG
        assert InputTypeDetector.detect(str(dclg_file), mode="api") == InputType.DOCLANG

    def test_detect_dclg_uppercase(self, temp_dir):
        path = temp_dir / "SAMPLE.DCLG"
        _build_doc().save_as_doclang(path)
        assert InputTypeDetector.detect(str(path), mode="cli") == InputType.DOCLANG

    def test_detect_dclg_xml_double_extension(self, temp_dir, dclg_file):
        path = temp_dir / "sample.dclg.xml"
        path.write_text(dclg_file.read_text(encoding="utf-8"), encoding="utf-8")
        assert InputTypeDetector.detect(str(path), mode="cli") == InputType.DOCLANG

    def test_detect_bare_xml_with_doclang_root(self, temp_dir, dclg_file):
        path = temp_dir / "sample.xml"
        path.write_text(dclg_file.read_text(encoding="utf-8"), encoding="utf-8")
        assert InputTypeDetector.detect(str(path), mode="cli") == InputType.DOCLANG

    def test_bare_xml_without_doclang_root_is_document(self, temp_dir):
        path = temp_dir / "other.xml"
        path.write_text("<html><body>nope</body></html>", encoding="utf-8")
        assert InputTypeDetector.detect(str(path), mode="cli") == InputType.DOCUMENT

    def test_detect_dclx_archive(self, temp_dir):
        # Only the extension is checked at detection time; content is validated later.
        path = temp_dir / "sample.dclx"
        path.write_bytes(b"PK\x03\x04 fake zip")
        assert InputTypeDetector.detect(str(path), mode="cli") == InputType.DOCLANG


class TestDoclangValidator:
    def test_accepts_valid_dclg(self, dclg_file):
        DoclangValidator().validate(dclg_file)  # no raise

    def test_rejects_missing_file(self, temp_dir):
        with pytest.raises(ConfigurationError):
            DoclangValidator().validate(temp_dir / "missing.dclg")

    def test_rejects_empty_file(self, temp_dir):
        path = temp_dir / "empty.dclg"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValidationError):
            DoclangValidator().validate(path)

    def test_rejects_non_doclang_xml(self, temp_dir):
        path = temp_dir / "nope.dclg"
        path.write_text("<html></html>", encoding="utf-8")
        with pytest.raises(ValidationError):
            DoclangValidator().validate(path)

    def test_rejects_non_zip_dclx(self, temp_dir):
        path = temp_dir / "bad.dclx"
        path.write_bytes(b"not a zip archive")
        with pytest.raises(ValidationError):
            DoclangValidator().validate(path)


class TestDoclangInputHandler:
    def test_loads_dclg_into_document(self, dclg_file):
        loaded = DoclangInputHandler().load(dclg_file)
        assert isinstance(loaded, DoclingDocument)
        assert [t.text for t in loaded.texts] == ["Invoice #12345"]

    def test_missing_file_raises(self, temp_dir):
        with pytest.raises(ValidationError):
            DoclangInputHandler().load(temp_dir / "missing.dclg")

    def test_unparseable_dclg_raises_validation_error(self, temp_dir):
        path = temp_dir / "broken.dclg"
        path.write_text('<doclang version="0.7"><not-closed>', encoding="utf-8")
        with pytest.raises(ValidationError):
            DoclangInputHandler().load(path)
