"""
Tests for Docling document exporter.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel

from docling_graph.core.exporters.docling_exporter import DoclingExporter


@pytest.fixture
def mock_docling_document():
    """Create a mock Docling document."""
    doc = MagicMock()
    doc.pages = {1: MagicMock(), 2: MagicMock()}
    doc.export_to_markdown.return_value = "# Document\n\nContent here"
    doc.export_to_dict.return_value = {"pages": [{"page_number": 1}], "metadata": {}}
    return doc


def _real_document(text: str = "Invoice #12345") -> DoclingDocument:
    """A minimal real DoclingDocument that can serialize to DocLang."""
    doc = DoclingDocument(name="sample")
    doc.add_page(page_no=1, size={"width": 612, "height": 792})
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox(l=10, t=10, r=100, b=30, coord_origin=CoordOrigin.TOPLEFT),
        charspan=(0, len(text)),
    )
    doc.add_heading(text=text, prov=prov)
    return doc


class TestDoclingExporterInitialization:
    """Test DoclingExporter initialization."""

    def test_initialization_default(self):
        """Should initialize with default output directory."""
        exporter = DoclingExporter()
        assert exporter.output_dir == Path("outputs")

    def test_initialization_custom_directory(self, tmp_path):
        """Should accept custom output directory."""
        exporter = DoclingExporter(output_dir=tmp_path)
        assert exporter.output_dir == tmp_path


class TestDoclingExporterExportDocument:
    """Test document export."""

    def test_export_document_creates_directory(self, mock_docling_document, tmp_path):
        """Should create output directory."""
        exporter = DoclingExporter(output_dir=tmp_path)

        exporter.export_document(mock_docling_document, "test_doc")

        assert tmp_path.exists()

    def test_export_document_returns_dict(self, mock_docling_document, tmp_path):
        """Should return dictionary of exported file paths."""
        exporter = DoclingExporter(output_dir=tmp_path)

        result = exporter.export_document(mock_docling_document, "test_doc")

        assert isinstance(result, dict)

    def test_export_document_with_json_included(self, mock_docling_document, tmp_path):
        """Should export document as JSON when enabled."""
        exporter = DoclingExporter(output_dir=tmp_path)

        result = exporter.export_document(mock_docling_document, "test_doc", include_json=True)

        assert "document_json" in result

    def test_export_document_with_markdown_included(self, mock_docling_document, tmp_path):
        """Should export markdown when enabled."""
        exporter = DoclingExporter(output_dir=tmp_path)

        result = exporter.export_document(mock_docling_document, "test_doc", include_markdown=True)

        assert "markdown" in result

    def test_export_document_without_json(self, mock_docling_document, tmp_path):
        """Should not export JSON when disabled."""
        exporter = DoclingExporter(output_dir=tmp_path)

        result = exporter.export_document(mock_docling_document, "test_doc", include_json=False)

        assert "document_json" not in result

    def test_export_document_without_markdown(self, mock_docling_document, tmp_path):
        """Should not export markdown when disabled."""
        exporter = DoclingExporter(output_dir=tmp_path)

        result = exporter.export_document(mock_docling_document, "test_doc", include_markdown=False)

        assert "markdown" not in result

    def test_export_document_per_page_markup(self, mock_docling_document, tmp_path):
        """Should export per-page markdown when enabled."""
        exporter = DoclingExporter(output_dir=tmp_path)

        result = exporter.export_document(mock_docling_document, "test_doc", per_page=True)

        assert "page_markdowns" in result

    def test_export_document_filename_format(self, mock_docling_document, tmp_path):
        """Exported files should follow naming convention."""
        exporter = DoclingExporter(output_dir=tmp_path)

        exporter.export_document(mock_docling_document, "test_doc")

        # Check for expected filename patterns
        files = list(tmp_path.glob("*"))
        filenames = [f.name for f in files]
        assert any("test_doc" in name for name in filenames)


class TestDoclingExporterExportDoclang:
    """Test DocLang (.dclg) export."""

    def test_export_document_writes_dclg_by_default(self, tmp_path):
        """A real document should produce a .dclg artifact by default."""
        exporter = DoclingExporter(output_dir=tmp_path)
        result = exporter.export_document(_real_document(), "document")
        assert "doclang" in result
        dclg = tmp_path / "document.dclg"
        assert dclg.exists()
        content = dclg.read_text(encoding="utf-8")
        assert "<doclang" in content
        assert "Invoice #12345" in content

    def test_export_document_without_doclang(self, tmp_path):
        """include_doclang=False suppresses the .dclg artifact."""
        exporter = DoclingExporter(output_dir=tmp_path)
        result = exporter.export_document(_real_document(), "document", include_doclang=False)
        assert "doclang" not in result
        assert not (tmp_path / "document.dclg").exists()

    def test_export_sanitizes_control_chars(self, tmp_path):
        """A document with NUL still exports; the NUL is stripped."""
        exporter = DoclingExporter(output_dir=tmp_path)
        result = exporter.export_document(_real_document("Invoice\x00 #99"), "document")
        assert "doclang" in result
        content = (tmp_path / "document.dclg").read_text(encoding="utf-8")
        assert "\x00" not in content
        assert "Invoice #99" in content

    def test_doclang_export_failure_is_non_fatal(self, tmp_path):
        """A serializer failure must not abort export; other artifacts survive."""
        exporter = DoclingExporter(output_dir=tmp_path)
        with patch.object(DoclingDocument, "export_to_doclang", side_effect=ValueError("boom")):
            result = exporter.export_document(_real_document(), "document")
        assert "doclang" not in result
        assert "document_json" in result
        assert "markdown" in result
        assert not (tmp_path / "document.dclg").exists()

    def test_doclang_skipped_when_unavailable(self, tmp_path):
        """If docling-core lacks export_to_doclang, skip gracefully."""
        doc = _real_document()
        exporter = DoclingExporter(output_dir=tmp_path)
        # Simulate an older docling-core by hiding the attribute.
        with patch("docling_graph.core.exporters.docling_exporter.hasattr", return_value=False):
            dclg_path = exporter._export_doclang(doc, tmp_path / "document.dclg")
        assert dclg_path is None


class TestDoclingExporterExportDocumentJSON:
    """Test document JSON export."""

    @patch("builtins.open", new_callable=mock_open)
    def test_export_document_json_creates_file(self, mock_file, mock_docling_document, tmp_path):
        """Should create JSON file."""
        exporter = DoclingExporter(output_dir=tmp_path)
        json_path = tmp_path / "test.json"

        exporter._export_document_json(mock_docling_document, json_path)

        mock_docling_document.export_to_dict.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    def test_export_document_json_uses_dict_export(
        self, mock_file, mock_docling_document, tmp_path
    ):
        """Should use document's export_to_dict method."""
        exporter = DoclingExporter(output_dir=tmp_path)
        json_path = tmp_path / "test.json"

        exporter._export_document_json(mock_docling_document, json_path)

        mock_docling_document.export_to_dict.assert_called_once()


class TestDoclingExporterSaveText:
    """Test text saving."""

    def test_save_text_creates_file(self, tmp_path):
        """Should create file with text content."""
        exporter = DoclingExporter(output_dir=tmp_path)
        text_path = tmp_path / "test.md"
        content = "# Test\n\nContent here"

        exporter._save_text(content, text_path)

        assert text_path.exists()
        assert text_path.read_text() == content

    def test_save_text_uses_utf8_encoding(self, tmp_path):
        """Should use UTF-8 encoding."""
        exporter = DoclingExporter(output_dir=tmp_path)
        text_path = tmp_path / "test.md"
        content = "# Tëst with üñíçödé"

        exporter._save_text(content, text_path)

        assert text_path.read_text(encoding="utf-8") == content

    def test_save_text_overwrites_existing(self, tmp_path):
        """Should overwrite existing file."""
        exporter = DoclingExporter(output_dir=tmp_path)
        text_path = tmp_path / "test.md"

        text_path.write_text("old content")
        exporter._save_text("new content", text_path)

        assert text_path.read_text() == "new content"
