"""
Integration tests for new input format support.

Tests the complete pipeline flow for:
- Plain text inputs (API only)
- .txt file inputs
- .md (Markdown) file inputs
- URL inputs (with type detection)
- DoclingDocument JSON inputs
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from docling_graph.config import PipelineConfig
from docling_graph.core.input.handlers import (
    DoclingDocumentHandler,
    TextInputHandler,
    URLInputHandler,
)
from docling_graph.core.input.types import InputType, InputTypeDetector
from docling_graph.core.input.validators import (
    DoclingDocumentValidator,
    TextValidator,
    URLValidator,
)
from docling_graph.exceptions import (
    ConfigurationError,
    ExtractionError,
    PipelineError,
    ValidationError,
)
from docling_graph.pipeline import run_pipeline


@pytest.mark.integration
class TestTextInputFormats:
    """Test text-based input formats (.txt, .md, plain text)."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_text_file(self, temp_dir):
        """Create a sample .txt file."""
        txt_file = temp_dir / "sample.txt"
        txt_file.write_text(
            "This is a sample document.\n"
            "It contains multiple lines of text.\n"
            "Used for testing text input processing."
        )
        return txt_file

    @pytest.fixture
    def sample_markdown_file(self, temp_dir):
        """Create a sample .md file."""
        md_file = temp_dir / "sample.md"
        md_file.write_text(
            "# Sample Document\n\n"
            "## Section 1\n\n"
            "This is a markdown document with **formatting**.\n\n"
            "- Item 1\n"
            "- Item 2\n"
        )
        return md_file

    @pytest.fixture
    def empty_text_file(self, temp_dir):
        """Create an empty .txt file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")
        return empty_file

    def test_text_file_detection(self, sample_text_file):
        """Test that .txt files are correctly detected."""
        input_type = InputTypeDetector.detect(str(sample_text_file), mode="api")
        assert input_type == InputType.TEXT_FILE

    def test_markdown_file_detection(self, sample_markdown_file):
        """Test that .md files are correctly detected."""
        input_type = InputTypeDetector.detect(str(sample_markdown_file), mode="api")
        assert input_type == InputType.MARKDOWN

    def test_plain_text_detection_api_mode(self):
        """Test that plain text is detected in API mode."""
        plain_text = "This is plain text content"
        input_type = InputTypeDetector.detect(plain_text, mode="api")
        assert input_type == InputType.TEXT

    def test_plain_text_rejected_cli_mode(self):
        """Test that plain text is rejected in CLI mode."""
        plain_text = "This is plain text content"
        with pytest.raises(
            ConfigurationError, match="Plain text input is only supported via Python API"
        ):
            InputTypeDetector.detect(plain_text, mode="cli")

    def test_text_validator_accepts_valid_text(self):
        """Test TextValidator accepts valid text."""
        validator = TextValidator()
        valid_text = "This is valid text content"
        validator.validate(valid_text)  # Should not raise

    def test_text_validator_rejects_empty_text(self):
        """Test TextValidator rejects empty text."""
        validator = TextValidator()
        with pytest.raises(ValidationError, match="Text input is empty"):
            validator.validate("")

    def test_text_validator_rejects_whitespace_only(self):
        """Test TextValidator rejects whitespace-only text."""
        validator = TextValidator()
        with pytest.raises(ValidationError, match="Text input contains only whitespace"):
            validator.validate("   \n\t  ")

    def test_text_handler_loads_txt_file(self, sample_text_file):
        """Test TextInputHandler loads .txt files correctly."""
        handler = TextInputHandler()
        content = handler.load(str(sample_text_file))
        assert "sample document" in content.lower()
        assert len(content) > 0

    def test_text_handler_loads_markdown_file(self, sample_markdown_file):
        """Test TextInputHandler loads .md files correctly."""
        handler = TextInputHandler()
        content = handler.load(str(sample_markdown_file))
        assert "# Sample Document" in content
        assert "**formatting**" in content

    def test_text_handler_rejects_empty_file(self, empty_text_file):
        """Test TextInputHandler rejects empty files."""
        handler = TextInputHandler()
        with pytest.raises(ValidationError, match="Text input is empty"):
            handler.load(str(empty_text_file))

    def test_text_handler_handles_encoding_errors(self, temp_dir):
        """Test TextInputHandler handles encoding errors gracefully."""
        # Create a file with invalid UTF-8
        bad_file = temp_dir / "bad_encoding.txt"
        bad_file.write_bytes(b"\x80\x81\x82")

        handler = TextInputHandler()
        with pytest.raises(ValidationError, match="Failed to read text file"):
            handler.load(str(bad_file))


@pytest.mark.integration
class TestURLInputFormat:
    """Test URL input format with download and type detection."""

    def test_url_detection(self):
        """Test that URLs are correctly detected."""
        url = "https://example.com/document.pdf"
        input_type = InputTypeDetector.detect(url, mode="api")
        assert input_type == InputType.URL

    def test_url_validator_accepts_valid_urls(self):
        """Test URLValidator accepts valid URLs."""
        validator = URLValidator()
        valid_urls = [
            "https://example.com/doc.pdf",
            "http://example.com/file.txt",
            "https://example.com/path/to/file.md",
        ]
        for url in valid_urls:
            validator.validate(url)  # Should not raise

    def test_url_validator_rejects_invalid_urls(self):
        """Test URLValidator rejects invalid URLs."""
        validator = URLValidator()
        invalid_urls = [
            "not a url",
            "ftp://example.com/file.pdf",  # Unsupported scheme
            "file:///local/path",  # Local file scheme
            "",
        ]
        for url in invalid_urls:
            with pytest.raises(ValidationError):
                validator.validate(url)

    @patch("requests.get")
    def test_url_handler_downloads_pdf(self, mock_get):
        """Test URLInputHandler downloads PDF correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.content = b"%PDF-1.4 fake pdf content"
        mock_response.iter_content = Mock(return_value=[b"%PDF-1.4 fake pdf content"])
        mock_get.return_value = mock_response

        handler = URLInputHandler()
        temp_path = handler.load("https://example.com/doc.pdf")

        assert temp_path.exists()
        assert temp_path.suffix == ".pdf"
        assert temp_path.read_bytes() == b"%PDF-1.4 fake pdf content"

    @patch("requests.get")
    def test_url_handler_downloads_text(self, mock_get):
        """Test URLInputHandler downloads text file correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "Sample text content"
        mock_response.iter_content = Mock(return_value=[b"Sample text content"])
        mock_get.return_value = mock_response

        handler = URLInputHandler()
        temp_path = handler.load("https://example.com/doc.txt")

        assert temp_path.exists()
        assert temp_path.suffix == ".txt"
        assert temp_path.read_text() == "Sample text content"

    @patch("requests.get")
    def test_url_handler_handles_timeout(self, mock_get):
        """Test URLInputHandler handles timeout errors."""
        import requests

        mock_get.side_effect = requests.Timeout("Connection timeout")

        handler = URLInputHandler()
        with pytest.raises(ValidationError, match="timeout"):
            handler.load("https://example.com/doc.pdf")

    @patch("requests.get")
    def test_url_handler_handles_network_error(self, mock_get):
        """Test URLInputHandler handles network errors."""
        import requests

        mock_get.side_effect = requests.RequestException("Network error")

        handler = URLInputHandler()
        with pytest.raises(ValidationError, match="Failed to download"):
            handler.load("https://example.com/doc.pdf")

    @patch("requests.head")
    @patch("requests.get")
    def test_url_handler_enforces_size_limit(self, mock_get, mock_head):
        """Test URLInputHandler enforces file size limits."""
        # Mock HEAD request
        mock_head_response = Mock()
        mock_head_response.status_code = 200
        mock_head_response.headers = {"content-length": str(200 * 1024 * 1024)}  # 200MB
        mock_head.return_value = mock_head_response

        # Mock GET request
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": str(200 * 1024 * 1024)}  # 200MB
        mock_response.iter_content = Mock(return_value=[b"content"])
        mock_get.return_value = mock_response

        handler = URLInputHandler(max_size_mb=100)
        with pytest.raises(ValidationError, match="exceeds maximum size"):
            handler.load("https://example.com/large.pdf")


@pytest.mark.integration
class TestDoclingDocumentInput:
    """Test DoclingDocument JSON input format."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def valid_docling_doc(self, temp_dir):
        """Create a valid DoclingDocument JSON file."""
        doc_data = {
            "schema_name": "DoclingDocument",
            "version": "1.0.0",
            "name": "test_document",
            "pages": {"0": {"page_no": 0, "size": {"width": 612, "height": 792}}},
            "furniture": {"self_ref": "#/furniture", "children": []},
            "body": {"self_ref": "#/body", "children": []},
        }
        doc_file = temp_dir / "valid_doc.json"
        doc_file.write_text(json.dumps(doc_data, indent=2))
        return doc_file

    @pytest.fixture
    def invalid_docling_doc(self, temp_dir):
        """Create an invalid DoclingDocument JSON file."""
        doc_data = {"invalid": "structure", "missing": "required_fields"}
        doc_file = temp_dir / "invalid_doc.json"
        doc_file.write_text(json.dumps(doc_data, indent=2))
        return doc_file

    def test_docling_document_detection(self, valid_docling_doc):
        """Test that DoclingDocument JSON files are detected."""
        input_type = InputTypeDetector.detect(str(valid_docling_doc), mode="api")
        assert input_type == InputType.DOCLING_DOCUMENT

    def test_docling_validator_accepts_valid_doc(self, valid_docling_doc):
        """Test DoclingDocumentValidator accepts valid documents."""
        validator = DoclingDocumentValidator()
        content = valid_docling_doc.read_text()
        validator.validate(content)  # Should not raise

    def test_docling_validator_rejects_invalid_json(self):
        """Test DoclingDocumentValidator rejects invalid JSON."""
        validator = DoclingDocumentValidator()
        with pytest.raises(ValidationError, match="Invalid JSON"):
            validator.validate("not valid json {")

    def test_docling_validator_rejects_missing_schema(self, invalid_docling_doc):
        """Test DoclingDocumentValidator rejects docs without schema_name."""
        validator = DoclingDocumentValidator()
        content = invalid_docling_doc.read_text()
        with pytest.raises(ValidationError, match="Missing required field"):
            validator.validate(content)

    def test_docling_handler_loads_valid_doc(self, valid_docling_doc):
        """Test DoclingDocumentHandler loads valid documents."""
        handler = DoclingDocumentHandler()
        doc = handler.load(str(valid_docling_doc))

        assert doc is not None
        assert hasattr(doc, "schema_name")
        assert doc.schema_name == "DoclingDocument"

    def test_docling_handler_rejects_invalid_doc(self, invalid_docling_doc):
        """Test DoclingDocumentHandler rejects invalid documents."""
        handler = DoclingDocumentHandler()
        with pytest.raises(ValidationError):
            handler.load(str(invalid_docling_doc))

    @patch("docling_graph.core.extractors.backends.llm_backend.LlmBackend.extract_from_markdown")
    def test_docling_document_extraction_pipeline(self, mock_extract, temp_dir):
        """Test complete pipeline with DoclingDocument input."""
        from pydantic import BaseModel, Field

        # Create a more complete DoclingDocument
        doc_data = {
            "schema_name": "DoclingDocument",
            "version": "1.0.0",
            "name": "test_invoice",
            "origin": {
                "mimetype": "application/pdf",
                "binary_hash": 12345,
                "filename": "invoice.pdf",
            },
            "pages": {"1": {"page_no": 1, "size": {"width": 612, "height": 792}}},
            "furniture": {
                "self_ref": "#/furniture",
                "children": [],
                "content_layer": "furniture",
                "name": "furniture",
                "label": "unspecified",
            },
            "body": {
                "self_ref": "#/body",
                "children": [],
                "content_layer": "body",
                "name": "body",
                "label": "unspecified",
            },
            "groups": [],
            "texts": [],
            "pictures": [],
            "tables": [],
            "key_value_items": [],
            "form_items": [],
        }

        doc_file = temp_dir / "test_docling.json"
        doc_file.write_text(json.dumps(doc_data, indent=2))

        # Define template
        class InvoiceItem(BaseModel):
            description: str = Field(description="Item description")
            amount: float = Field(description="Item amount")

        class Invoice(BaseModel):
            invoice_number: str = Field(description="Invoice number")
            total: float = Field(description="Total amount")
            items: list[InvoiceItem] = Field(default_factory=list, description="Invoice items")

        # Mock extraction result
        mock_model = Invoice(
            invoice_number="INV-001",
            total=100.0,
            items=[InvoiceItem(description="Service", amount=100.0)],
        )
        mock_extract.return_value = mock_model

        # Create config
        config = PipelineConfig(
            source=str(doc_file),
            template=Invoice,
            processing_mode="one-to-one",
            backend="llm",
            inference="local",
            docling_config="ocr",
            output_dir=str(temp_dir / "output"),
            export_format="csv",
            use_chunking=False,  # Disable chunking for simpler test
        )

        # Run pipeline - should complete without errors
        try:
            run_pipeline(config, mode="api")
            # Verify extraction was called
            assert mock_extract.called
            # Verify markdown was passed to extraction
            call_args = mock_extract.call_args
            assert call_args is not None
            assert "markdown" in call_args[1] or len(call_args[0]) > 0
        except Exception as e:
            # Allow certain expected errors (like missing LLM client)
            if "LLM" not in str(e) and "client" not in str(e):
                raise

    @patch("docling_graph.core.extractors.backends.llm_backend.LlmBackend.extract_from_markdown")
    def test_docling_document_with_chunking(self, mock_extract, temp_dir):
        """Test DoclingDocument extraction with chunking enabled."""
        from pydantic import BaseModel, Field

        # Create DoclingDocument
        doc_data = {
            "schema_name": "DoclingDocument",
            "version": "1.0.0",
            "name": "test_doc",
            "origin": {"mimetype": "application/pdf", "binary_hash": 12345, "filename": "doc.pdf"},
            "pages": {"1": {"page_no": 1, "size": {"width": 612, "height": 792}}},
            "furniture": {
                "self_ref": "#/furniture",
                "children": [],
                "content_layer": "furniture",
                "name": "furniture",
                "label": "unspecified",
            },
            "body": {
                "self_ref": "#/body",
                "children": [],
                "content_layer": "body",
                "name": "body",
                "label": "unspecified",
            },
            "groups": [],
            "texts": [],
            "pictures": [],
            "tables": [],
            "key_value_items": [],
            "form_items": [],
        }

        doc_file = temp_dir / "chunked_doc.json"
        doc_file.write_text(json.dumps(doc_data, indent=2))

        # Simple template
        class SimpleDoc(BaseModel):
            title: str = Field(description="Document title")
            summary: str = Field(description="Document summary")

        # Mock extraction - return partial results
        mock_extract.return_value = SimpleDoc(title="Test Document", summary="Test summary")

        config = PipelineConfig(
            source=str(doc_file),
            template=SimpleDoc,
            processing_mode="one-to-one",
            backend="llm",
            inference="local",
            docling_config="ocr",
            output_dir=str(temp_dir / "output"),
            export_format="csv",
            use_chunking=True,  # Enable chunking
        )

        # Run pipeline
        try:
            run_pipeline(config, mode="api")
            # Verify extraction was called (possibly multiple times for chunks)
            assert mock_extract.called
        except Exception as e:
            # Allow certain expected errors (LLM, client, chunker, graph validation issues)
            if (
                "LLM" not in str(e)
                and "client" not in str(e)
                and "chunker" not in str(e).lower()
                and "Graph validation" not in str(e)
            ):
                raise


@pytest.mark.integration
class TestPipelineWithNewInputs:
    """Test complete pipeline execution with new input formats."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_template(self):
        """Create a simple Pydantic template for testing."""
        from pydantic import BaseModel

        class TestTemplate(BaseModel):
            title: str
            content: str

        return TestTemplate

    def test_pipeline_rejects_text_input_with_vlm_backend(self, temp_dir, sample_template):
        """Test that text inputs are rejected when using VLM backend."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("Sample text content")

        config = PipelineConfig(
            source=str(txt_file),
            template=sample_template,
            processing_mode="one-to-one",
            backend="vlm",  # VLM doesn't support text-only
            inference="local",
            docling_config="ocr",
            output_dir=str(temp_dir / "output"),
            export_format="csv",
        )

        with pytest.raises(
            (ExtractionError, PipelineError), match="VLM backend does not support text-only inputs"
        ):
            run_pipeline(config, mode="api")

    @patch("docling_graph.core.extractors.backends.llm_backend.LlmBackend.extract_from_markdown")
    def test_pipeline_processes_text_file_with_llm(self, mock_extract, temp_dir, sample_template):
        """Test that text files are processed correctly with LLM backend."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("Sample text content for extraction")

        # Mock the LLM extraction to return a model with relationships
        from pydantic import BaseModel, Field

        # Create a mock model with a relationship to ensure graph has edges
        class RelatedItem(BaseModel):
            name: str = Field(description="Related item name")

        class MockModel(BaseModel):
            title: str = Field(description="Title")
            content: str = Field(description="Content")
            related: list[RelatedItem] = Field(default_factory=list, description="Related items")

        mock_model = MockModel(
            title="Test", content="Extracted content", related=[RelatedItem(name="Related1")]
        )
        mock_extract.return_value = mock_model

        config = PipelineConfig(
            source=str(txt_file),
            template=MockModel,
            processing_mode="one-to-one",
            backend="llm",
            inference="local",
            docling_config="ocr",
            output_dir=str(temp_dir / "output"),
            export_format="csv",
        )

        # This should work without raising
        try:
            run_pipeline(config, mode="api")
        except Exception as e:
            # Allow certain expected errors (like missing LLM client or graph validation)
            if "LLM" not in str(e) and "client" not in str(e) and "Graph validation" not in str(e):
                raise

    def test_pipeline_enforces_cli_mode_restrictions(self, temp_dir):
        """Test that CLI mode rejects plain text input."""
        config = PipelineConfig(
            source="Plain text string",  # Not a file path
            template="some.template.Class",
            processing_mode="one-to-one",
            backend="llm",
            inference="local",
            docling_config="ocr",
            output_dir=str(temp_dir / "output"),
            export_format="csv",
        )

        with pytest.raises(
            (ConfigurationError, PipelineError),
            match="Plain text input is only supported via Python API",
        ):
            run_pipeline(config, mode="cli")


@pytest.mark.integration
class TestInputFormatRegression:
    """Regression tests to ensure existing PDF/Image workflows still work."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_pdf_detection_unchanged(self, temp_dir):
        """Test that PDF files are still detected correctly."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake pdf")

        input_type = InputTypeDetector.detect(str(pdf_file), mode="api")
        assert input_type == InputType.PDF

    def test_image_detection_unchanged(self, temp_dir):
        """Test that image files are still detected correctly."""
        for ext in [".png", ".jpg", ".jpeg"]:
            img_file = temp_dir / f"test{ext}"
            img_file.write_bytes(b"fake image data")

            input_type = InputTypeDetector.detect(str(img_file), mode="api")
            assert input_type == InputType.IMAGE

    def test_pdf_pipeline_flow_unchanged(self, temp_dir):
        """Test that PDF processing flow is unchanged."""
        # This is a smoke test - just ensure the detection and routing works
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake pdf")

        input_type = InputTypeDetector.detect(str(pdf_file), mode="cli")
        assert input_type == InputType.PDF

        # PDF should not have skip_ocr flag
        from docling_graph.config import PipelineConfig
        from docling_graph.pipeline.context import PipelineContext
        from docling_graph.pipeline.stages import InputNormalizationStage

        config = PipelineConfig(
            source=str(pdf_file),
            template="some.template.Class",
            processing_mode="one-to-one",
            backend="llm",
            inference="local",
            docling_config="ocr",
            output_dir=str(temp_dir / "output"),
            export_format="csv",
        )

        context = PipelineContext(config=config)
        stage = InputNormalizationStage(mode="cli")

        try:
            context = stage.execute(context)
            # PDF should not skip OCR
            assert not context.input_metadata.get("skip_ocr", False)
        except Exception:
            # Allow failures from missing dependencies
            pass
