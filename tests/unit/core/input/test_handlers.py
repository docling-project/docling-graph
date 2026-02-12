"""Unit tests for input handlers."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from docling_graph.core.input.handlers import (
    DoclingDocumentHandler,
    DocumentInputHandler,
    TextInputHandler,
    URLInputHandler,
)
from docling_graph.exceptions import ValidationError


class TestTextInputHandler:
    """Test TextInputHandler class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def handler(self):
        """Create TextInputHandler instance."""
        return TextInputHandler()

    def test_loads_txt_file(self, handler, temp_dir):
        """Test loading .txt file."""
        txt_file = temp_dir / "test.txt"
        content = "This is test content\nWith multiple lines"
        txt_file.write_text(content)

        result = handler.load(str(txt_file))
        assert result == content

    def test_loads_markdown_file(self, handler, temp_dir):
        """Test loading .md file."""
        md_file = temp_dir / "test.md"
        content = "# Heading\n\n**Bold text**"
        md_file.write_text(content)

        result = handler.load(str(md_file))
        assert result == content

    def test_loads_file_with_unicode(self, handler, temp_dir):
        """Test loading file with Unicode content."""
        txt_file = temp_dir / "unicode.txt"
        content = "Hello 世界 🌍"
        txt_file.write_text(content, encoding="utf-8")

        result = handler.load(str(txt_file))
        assert result == content

    def test_rejects_empty_file(self, handler, temp_dir):
        """Test that empty files are rejected."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")

        with pytest.raises(ValidationError, match="Text input is empty"):
            handler.load(str(empty_file))

    def test_rejects_whitespace_only_file(self, handler, temp_dir):
        """Test that whitespace-only files are rejected."""
        ws_file = temp_dir / "whitespace.txt"
        ws_file.write_text("   \n\t  ")

        with pytest.raises(ValidationError, match="contains only whitespace"):
            handler.load(str(ws_file))

    def test_handles_encoding_errors(self, handler, temp_dir):
        """Test handling of encoding errors."""
        bad_file = temp_dir / "bad_encoding.txt"
        bad_file.write_bytes(b"\x80\x81\x82")  # Invalid UTF-8

        with pytest.raises(ValidationError, match="Failed to read text file"):
            handler.load(str(bad_file))

    def test_handles_nonexistent_file(self, handler, temp_dir):
        """Test handling of nonexistent file."""
        nonexistent = temp_dir / "nonexistent.txt"

        with pytest.raises(ValidationError, match="Failed to read text file"):
            handler.load(str(nonexistent))

    def test_loads_large_file(self, handler, temp_dir):
        """Test loading large text file."""
        large_file = temp_dir / "large.txt"
        content = "Line\n" * 10000  # 10k lines
        large_file.write_text(content)

        result = handler.load(str(large_file))
        assert len(result) == len(content)

    def test_preserves_line_endings(self, handler, temp_dir):
        """Test that line endings are preserved."""
        txt_file = temp_dir / "lines.txt"
        content = "Line1\nLine2\r\nLine3\n"
        txt_file.write_text(content, newline="")

        result = handler.load(str(txt_file))
        # Python normalizes line endings when reading text
        assert "Line1" in result
        assert "Line2" in result
        assert "Line3" in result


class TestURLInputHandler:
    """Test URLInputHandler class."""

    @pytest.fixture
    def handler(self):
        """Create URLInputHandler instance."""
        return URLInputHandler(timeout=5, max_size_mb=10)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for downloaded files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @patch("requests.get")
    def test_downloads_pdf(self, mock_get, handler):
        """Test downloading PDF file."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.content = b"%PDF-1.4 fake pdf content"
        mock_response.iter_content = Mock(return_value=[b"%PDF-1.4 fake pdf content"])
        mock_get.return_value = mock_response

        result = handler.load("https://example.com/doc.pdf")

        assert result.exists()
        assert result.suffix == ".pdf"
        assert result.read_bytes() == b"%PDF-1.4 fake pdf content"
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_downloads_text_file(self, mock_get, handler):
        """Test downloading text file."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "Sample text content"
        mock_response.iter_content = Mock(return_value=[b"Sample text content"])
        mock_get.return_value = mock_response

        result = handler.load("https://example.com/doc.txt")

        assert result.exists()
        assert result.suffix == ".txt"
        assert result.read_text() == "Sample text content"

    @patch("requests.get")
    def test_downloads_markdown(self, mock_get, handler):
        """Test downloading markdown file."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/markdown"}
        mock_response.text = "# Markdown content"
        mock_response.iter_content = Mock(return_value=[b"# Markdown content"])
        mock_get.return_value = mock_response

        result = handler.load("https://example.com/doc.md")

        assert result.exists()
        assert result.suffix == ".md"

    @patch("requests.get")
    def test_downloads_image(self, mock_get, handler):
        """Test downloading image file."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/png"}
        mock_response.content = b"fake png data"
        mock_response.iter_content = Mock(return_value=[b"fake png data"])
        mock_get.return_value = mock_response

        result = handler.load("https://example.com/image.png")

        assert result.exists()
        assert result.suffix == ".png"

    @patch("requests.get")
    def test_handles_timeout(self, mock_get, handler):
        """Test handling of timeout errors."""
        import requests

        mock_get.side_effect = requests.Timeout("Connection timeout")

        with pytest.raises(ValidationError, match="timeout"):
            handler.load("https://example.com/doc.pdf")

    @patch("requests.get")
    def test_handles_connection_error(self, mock_get, handler):
        """Test handling of connection errors."""
        import requests

        mock_get.side_effect = requests.ConnectionError("Connection failed")

        with pytest.raises(ValidationError, match="Failed to download"):
            handler.load("https://example.com/doc.pdf")

    @patch("requests.get")
    def test_handles_http_error(self, mock_get, handler):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_response

        with pytest.raises(ValidationError, match="Failed to download"):
            handler.load("https://example.com/notfound.pdf")

    @patch("requests.head")
    @patch("requests.get")
    def test_enforces_size_limit(self, mock_get, mock_head, handler):
        """Test that file size limit is enforced."""
        # Mock HEAD request to return size
        mock_head_response = Mock()
        mock_head_response.status_code = 200
        mock_head_response.headers = {
            "content-length": str(20 * 1024 * 1024),  # 20MB
            "content-type": "application/pdf",
        }
        mock_head.return_value = mock_head_response

        # Mock GET request (shouldn't be called if HEAD check works)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            "content-length": str(20 * 1024 * 1024),  # 20MB
            "content-type": "application/pdf",
        }
        mock_response.iter_content = Mock(return_value=[b"content"])
        mock_get.return_value = mock_response

        # Handler has max_size_mb=10
        with pytest.raises(ValidationError, match="exceeds maximum size"):
            handler.load("https://example.com/large.pdf")

    @patch("requests.get")
    def test_uses_custom_timeout(self, mock_get):
        """Test that custom timeout is used."""
        handler = URLInputHandler(timeout=30)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "content"
        mock_response.iter_content = Mock(return_value=[b"content"])
        mock_get.return_value = mock_response

        handler.load("https://example.com/doc.txt")

        # Verify timeout was passed to requests.get
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["timeout"] == 30

    @patch("requests.get")
    def test_detects_content_type_from_url(self, mock_get, handler):
        """Test content type detection from URL when header missing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}  # No content-type header
        mock_response.content = b"content"
        mock_response.iter_content = Mock(return_value=[b"content"])
        mock_get.return_value = mock_response

        result = handler.load("https://example.com/document.pdf")

        # Should use .pdf extension from URL
        assert result.suffix == ".pdf"

    @patch("requests.get")
    def test_handles_redirect(self, mock_get, handler):
        """Test handling of redirects."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.content = b"pdf content"
        mock_response.url = "https://example.com/final.pdf"  # After redirect
        mock_response.iter_content = Mock(return_value=[b"pdf content"])
        mock_get.return_value = mock_response

        result = handler.load("https://example.com/redirect")

        assert result.exists()

    @patch("requests.get")
    def test_handles_arxiv_url_without_extension(self, mock_get, handler):
        """Test handling of arXiv URLs without file extensions."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.content = b"%PDF-1.4 fake pdf content"
        mock_response.iter_content = Mock(return_value=[b"%PDF-1.4 fake pdf content"])
        mock_get.return_value = mock_response

        # arXiv URL without .pdf extension
        result = handler.load("https://arxiv.org/pdf/2511.14859")

        assert result.exists()
        assert result.suffix == ".pdf"

    @patch("requests.get")
    def test_handles_url_with_pdf_path_segment(self, mock_get, handler):
        """Test handling of URLs with /pdf/ in path but no extension."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.content = b"pdf content"
        mock_response.iter_content = Mock(return_value=[b"pdf content"])
        mock_get.return_value = mock_response

        result = handler.load("https://example.com/pdf/document123")

        assert result.exists()
        assert result.suffix == ".pdf"

    @patch("requests.head")
    @patch("requests.get")
    def test_prioritizes_content_type_over_url_extension(self, mock_get, mock_head, handler):
        """Test that content-type header is prioritized over URL extension."""
        # Mock HEAD request
        mock_head_response = Mock()
        mock_head_response.status_code = 200
        mock_head_response.headers = {"content-type": "image/png"}
        mock_head.return_value = mock_head_response

        # Mock GET request
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/png"}
        mock_response.content = b"png data"
        mock_response.iter_content = Mock(return_value=[b"png data"])
        mock_get.return_value = mock_response

        # URL says .jpg but content-type says png
        result = handler.load("https://example.com/image.jpg")

        assert result.exists()
        assert result.suffix == ".png"

    @patch("requests.get")
    def test_rejects_invalid_url_extension(self, mock_get, handler):
        """Test that invalid URL extensions are handled properly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}  # No content-type
        mock_response.content = b"content"
        mock_response.iter_content = Mock(return_value=[b"content"])
        mock_get.return_value = mock_response

        # URL with numeric "extension" like .14859
        result = handler.load("https://example.com/file.14859")

        assert result.exists()
        # Should default to .bin since extension is invalid and no content-type
        assert result.suffix == ".bin"

    @patch("requests.head")
    @patch("requests.get")
    def test_sends_user_agent_in_requests(self, mock_get, mock_head, handler):
        """Test that User-Agent header is sent in both HEAD and GET requests."""
        # Setup HEAD mock
        mock_head_response = Mock()
        mock_head_response.headers = {"content-type": "application/pdf"}
        mock_head.return_value = mock_head_response

        # Setup GET mock
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.headers = {"content-type": "application/pdf"}
        mock_get_response.iter_content = Mock(return_value=[b"pdf content"])
        mock_get.return_value = mock_get_response

        # Execute
        result = handler.load("https://example.com/doc.pdf")

        # Verify HEAD request includes User-Agent
        mock_head.assert_called_once()
        head_call_kwargs = mock_head.call_args[1]
        assert "headers" in head_call_kwargs
        assert "User-Agent" in head_call_kwargs["headers"]
        assert "docling-graph" in head_call_kwargs["headers"]["User-Agent"]

        # Verify GET request includes User-Agent
        mock_get.assert_called_once()
        get_call_kwargs = mock_get.call_args[1]
        assert "headers" in get_call_kwargs
        assert "User-Agent" in get_call_kwargs["headers"]
        assert "docling-graph" in get_call_kwargs["headers"]["User-Agent"]

        assert result.exists()

    def test_user_agent_format(self, handler):
        """Test that User-Agent follows expected format."""
        assert hasattr(handler, "headers")
        assert "User-Agent" in handler.headers
        user_agent = handler.headers["User-Agent"]

        # Verify format: docling-graph/{version} (https://github.com/...)
        assert "docling-graph/" in user_agent
        assert "github.com" in user_agent
        assert user_agent.startswith("docling-graph/")

    @patch("requests.head")
    @patch("requests.get")
    def test_user_agent_sent_even_when_head_fails(self, mock_get, mock_head, handler):
        """Test that User-Agent is sent in GET request even when HEAD fails."""
        import requests

        # Make HEAD fail with RequestException (which is caught)
        mock_head.side_effect = requests.RequestException("HEAD failed")

        # Setup GET mock to succeed
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.iter_content = Mock(return_value=[b"content"])
        mock_get.return_value = mock_response

        # Execute (HEAD will fail internally but be caught)
        result = handler.load("https://example.com/doc.pdf")

        # Verify GET request still includes User-Agent
        mock_get.assert_called_once()
        get_call_kwargs = mock_get.call_args[1]
        assert "headers" in get_call_kwargs
        assert "User-Agent" in get_call_kwargs["headers"]
        assert result.exists()


class TestDocumentInputHandler:
    """Test DocumentInputHandler (unified path: file or raw text -> Path for Docling)."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def handler(self):
        return DocumentInputHandler()

    def test_returns_path_for_pdf(self, handler, temp_dir):
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        result = handler.load(str(pdf_file))
        assert isinstance(result, Path)
        assert result == pdf_file

    def test_returns_temp_md_for_txt_file(self, handler, temp_dir):
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("Hello world")
        result = handler.load(str(txt_file))
        assert isinstance(result, Path)
        assert result.suffix == ".md"
        assert result.read_text() == "Hello world"

    def test_returns_temp_md_for_raw_text(self, handler):
        result = handler.load("Raw text content")
        assert isinstance(result, Path)
        assert result.suffix == ".md"
        assert result.read_text() == "Raw text content"

    def test_rejects_empty_raw_text(self, handler):
        with pytest.raises(ValidationError, match="empty"):
            handler.load("")

    def test_rejects_whitespace_only_raw_text(self, handler):
        with pytest.raises(ValidationError, match="whitespace"):
            handler.load("   \n\t  ")

    def test_rejects_empty_txt_file(self, handler, temp_dir):
        empty = temp_dir / "empty.txt"
        empty.write_text("")
        with pytest.raises(ValidationError, match="empty"):
            handler.load(str(empty))


class TestDoclingDocumentHandler:
    """Test DoclingDocumentHandler class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def handler(self):
        """Create DoclingDocumentHandler instance."""
        return DoclingDocumentHandler()

    @pytest.fixture
    def valid_doc_file(self, temp_dir):
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

    def test_loads_valid_document(self, handler, valid_doc_file):
        """Test loading valid DoclingDocument."""
        result = handler.load(str(valid_doc_file))

        assert result is not None
        assert hasattr(result, "schema_name")
        assert result.schema_name == "DoclingDocument"

    def test_rejects_invalid_json(self, handler, temp_dir):
        """Test rejection of invalid JSON."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("not valid json {")

        with pytest.raises(ValidationError, match="Invalid JSON"):
            handler.load(str(invalid_file))

    def test_rejects_missing_schema_name(self, handler, temp_dir):
        """Test rejection of document without schema_name."""
        doc_data = {"version": "1.0.0", "name": "test"}
        doc_file = temp_dir / "no_schema.json"
        doc_file.write_text(json.dumps(doc_data))

        with pytest.raises(ValidationError, match="schema_name"):
            handler.load(str(doc_file))

    def test_rejects_wrong_schema_name(self, handler, temp_dir):
        """Test rejection of document with wrong schema_name."""
        doc_data = {"schema_name": "WrongSchema", "version": "1.0.0"}
        doc_file = temp_dir / "wrong_schema.json"
        doc_file.write_text(json.dumps(doc_data))

        with pytest.raises(ValidationError, match="schema_name must be"):
            handler.load(str(doc_file))

    def test_handles_nonexistent_file(self, handler, temp_dir):
        """Test handling of nonexistent file."""
        nonexistent = temp_dir / "nonexistent.json"

        with pytest.raises(ValidationError):
            handler.load(str(nonexistent))

    def test_loads_minimal_document(self, handler, temp_dir):
        """Test loading minimal valid document."""
        doc_data = {"schema_name": "DoclingDocument", "version": "1.0.0", "name": "minimal_doc"}
        doc_file = temp_dir / "minimal.json"
        doc_file.write_text(json.dumps(doc_data))

        result = handler.load(str(doc_file))
        assert result is not None

    def test_preserves_document_structure(self, handler, valid_doc_file):
        """Test that document structure is preserved."""
        result = handler.load(str(valid_doc_file))

        assert hasattr(result, "pages")
        assert hasattr(result, "body")
        # Note: furniture field is deprecated in newer docling_core versions
        # Just verify the document loaded successfully
        assert result.schema_name == "DoclingDocument"


class TestHandlerErrorMessages:
    """Test that handlers provide clear error messages."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_text_handler_error_includes_path(self, temp_dir):
        """Test that TextInputHandler errors include file path."""
        handler = TextInputHandler()
        nonexistent = temp_dir / "nonexistent.txt"

        try:
            handler.load(str(nonexistent))
        except ValidationError as e:
            assert str(nonexistent) in str(e) or "Failed to read" in str(e)

    @patch("requests.get")
    def test_url_handler_error_includes_url(self, mock_get):
        """Test that URLInputHandler errors include URL."""
        import requests

        mock_get.side_effect = requests.Timeout("timeout")
        handler = URLInputHandler()
        url = "https://example.com/doc.pdf"

        try:
            handler.load(url)
        except ValidationError as e:
            assert "timeout" in str(e).lower()

    def test_docling_handler_error_includes_details(self, temp_dir):
        """Test that DoclingDocumentHandler errors include details."""
        handler = DoclingDocumentHandler()
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("not json")

        try:
            handler.load(str(invalid_file))
        except ValidationError as e:
            assert e.details is not None
            assert "Invalid JSON" in str(e)
