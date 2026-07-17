"""Unit tests for input validators."""

import json
import socket
import tempfile
from pathlib import Path

import pytest

from docling_graph.core.input.validators import (
    DoclingDocumentValidator,
    TextValidator,
    URLValidator,
)
from docling_graph.exceptions import ValidationError


def _addrinfo(*ips: str) -> list:
    """Build a socket.getaddrinfo-style result list for the given IP addresses."""
    return [
        (socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))
        for ip in ips
    ]


class TestTextValidator:
    """Test TextValidator class."""

    def test_accepts_valid_text(self):
        """Test that valid text passes validation."""
        validator = TextValidator()
        valid_texts = [
            "Simple text",
            "Multi\nline\ntext",
            "Text with special chars: !@#$%",
            "   Text with leading/trailing spaces   ",
            "A" * 10000,  # Long text
        ]
        for text in valid_texts:
            validator.validate(text)  # Should not raise

    def test_rejects_empty_string(self):
        """Test that empty string is rejected."""
        validator = TextValidator()
        with pytest.raises(ValidationError, match="Text input is empty"):
            validator.validate("")

    def test_rejects_whitespace_only(self):
        """Test that whitespace-only text is rejected."""
        validator = TextValidator()
        whitespace_texts = [
            " ",
            "   ",
            "\n",
            "\t",
            "\n\n\n",
            "  \n\t  ",
        ]
        for text in whitespace_texts:
            with pytest.raises(ValidationError, match="contains only whitespace"):
                validator.validate(text)

    def test_rejects_none(self):
        """Test that None is rejected."""
        validator = TextValidator()
        with pytest.raises(ValidationError):
            validator.validate(None)

    def test_accepts_text_with_unicode(self):
        """Test that Unicode text is accepted."""
        validator = TextValidator()
        unicode_texts = [
            "Hello 世界",
            "Привет мир",
            "مرحبا بالعالم",
            "🎉 Emoji text 🚀",
        ]
        for text in unicode_texts:
            validator.validate(text)  # Should not raise


class TestURLValidator:
    """Test URLValidator class."""

    def test_accepts_valid_https_urls(self):
        """Test that valid HTTPS URLs pass validation."""
        from unittest.mock import patch

        validator = URLValidator()
        valid_urls = [
            "https://example.com",
            "https://example.com/path",
            "https://example.com/path/to/file.pdf",
            "https://example.com:8080/file",
            "https://sub.example.com/file",
            "https://example.com/file?param=value",
            "https://example.com/file?p1=v1&p2=v2",
            "https://example.com/file#anchor",
        ]

        # Mock DNS resolution to return a public IP address
        with patch(
            "socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")
        ):  # example.com's actual IP
            for url in valid_urls:
                validator.validate(url)  # Should not raise

    def test_accepts_valid_http_urls(self):
        """Test that valid HTTP URLs pass validation."""
        from unittest.mock import patch

        validator = URLValidator()
        valid_urls = [
            "http://example.com",
            "http://example.com/file.txt",
            "http://public-server.com/document",
        ]

        # Mock DNS resolution to return a public IP address
        with patch(
            "socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")
        ):  # example.com's actual IP
            for url in valid_urls:
                validator.validate(url)  # Should not raise

    def test_rejects_unsupported_schemes(self):
        """Test that unsupported URL schemes are rejected."""
        validator = URLValidator()
        invalid_urls = [
            "ftp://example.com/file",
            "file:///local/path",
            "ssh://example.com",
            "mailto:user@example.com",
        ]
        for url in invalid_urls:
            with pytest.raises(ValidationError, match="must use http or https"):
                validator.validate(url)

    def test_rejects_invalid_url_format(self):
        """Test that invalid URL formats are rejected."""
        validator = URLValidator()
        invalid_urls = [
            "not a url",
            "example.com",  # Missing scheme
            "://example.com",  # Missing scheme name
            "https://",  # Missing domain
            "",
            "   ",
        ]
        for url in invalid_urls:
            with pytest.raises(ValidationError):
                validator.validate(url)

    def test_rejects_none(self):
        """Test that None is rejected."""
        validator = URLValidator()
        with pytest.raises(ValidationError):
            validator.validate(None)

    def test_url_with_authentication(self):
        """Test URLs with authentication info."""
        validator = URLValidator()
        # URLs with auth should be accepted
        url = "https://user:pass@example.com/file"
        validator.validate(url)  # Should not raise


class TestDoclingDocumentValidator:
    """Test DoclingDocumentValidator class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_accepts_valid_docling_document(self):
        """Test that valid DoclingDocument JSON passes validation."""
        validator = DoclingDocumentValidator()
        valid_docs = [
            {
                "schema_name": "DoclingDocument",
                "version": "1.0.0",
                "name": "test",
            },
            {
                "schema_name": "DoclingDocument",
                "version": "2.0.0",
                "name": "test",
                "pages": {},
                "body": {},
            },
        ]
        for doc in valid_docs:
            json_str = json.dumps(doc)
            validator.validate(json_str)  # Should not raise

    def test_rejects_invalid_json(self):
        """Test that invalid JSON is rejected."""
        validator = DoclingDocumentValidator()
        invalid_jsons = [
            "not json",
            "{invalid json",
            '{"key": value}',  # Unquoted value
            "",
        ]
        for invalid in invalid_jsons:
            with pytest.raises(ValidationError, match="Invalid JSON"):
                validator.validate(invalid)

    def test_rejects_missing_schema_name(self):
        """Test that documents without schema_name are rejected."""
        validator = DoclingDocumentValidator()
        doc = {
            "version": "1.0.0",
            "name": "test",
        }
        json_str = json.dumps(doc)
        with pytest.raises(ValidationError, match="Missing required field: schema_name"):
            validator.validate(json_str)

    def test_rejects_wrong_schema_name(self):
        """Test that documents with wrong schema_name are rejected."""
        validator = DoclingDocumentValidator()
        doc = {
            "schema_name": "WrongSchema",
            "version": "1.0.0",
        }
        json_str = json.dumps(doc)
        with pytest.raises(ValidationError, match="schema_name must be 'DoclingDocument'"):
            validator.validate(json_str)

    def test_rejects_missing_version(self):
        """Test that documents without version are rejected."""
        validator = DoclingDocumentValidator()
        doc = {
            "schema_name": "DoclingDocument",
            "name": "test",
        }
        json_str = json.dumps(doc)
        with pytest.raises(ValidationError, match="Missing required field: version"):
            validator.validate(json_str)

    def test_accepts_minimal_valid_document(self):
        """Test that minimal valid document is accepted."""
        validator = DoclingDocumentValidator()
        doc = {
            "schema_name": "DoclingDocument",
            "version": "1.0.0",
        }
        json_str = json.dumps(doc)
        validator.validate(json_str)  # Should not raise

    def test_rejects_none(self):
        """Test that None is rejected."""
        validator = DoclingDocumentValidator()
        with pytest.raises(ValidationError):
            validator.validate(None)

    def test_accepts_document_with_extra_fields(self):
        """Test that documents with extra fields are accepted."""
        validator = DoclingDocumentValidator()
        doc = {
            "schema_name": "DoclingDocument",
            "version": "1.0.0",
            "name": "test",
            "pages": {"0": {"page_no": 0}},
            "body": {"children": []},
            "furniture": {},
            "extra_field": "extra_value",
        }
        json_str = json.dumps(doc)
        validator.validate(json_str)  # Should not raise

    def test_rejects_empty_string(self):
        """Test that empty string is rejected."""
        validator = DoclingDocumentValidator()
        with pytest.raises(ValidationError, match="Invalid JSON"):
            validator.validate("")

    def test_rejects_whitespace_only(self):
        """Test that whitespace-only string is rejected."""
        validator = DoclingDocumentValidator()
        with pytest.raises(ValidationError, match="Invalid JSON"):
            validator.validate("   \n\t  ")


class TestValidatorErrorMessages:
    """Test that validators provide clear error messages."""

    def test_text_validator_error_message_empty(self):
        """Test TextValidator error message for empty input."""
        validator = TextValidator()
        try:
            validator.validate("")
        except ValidationError as e:
            assert "empty" in str(e).lower()
            assert e.details is not None

    def test_text_validator_error_message_whitespace(self):
        """Test TextValidator error message for whitespace input."""
        validator = TextValidator()
        try:
            validator.validate("   ")
        except ValidationError as e:
            assert "whitespace" in str(e).lower()

    def test_url_validator_error_message_scheme(self):
        """Test URLValidator error message for wrong scheme."""
        validator = URLValidator()
        try:
            validator.validate("ftp://example.com")
        except ValidationError as e:
            assert "http" in str(e).lower()
            assert "https" in str(e).lower()

    def test_docling_validator_error_message_schema(self):
        """Test DoclingDocumentValidator error message for missing schema."""
        validator = DoclingDocumentValidator()
        doc = {"version": "1.0.0"}
        try:
            validator.validate(json.dumps(doc))
        except ValidationError as e:
            assert "schema_name" in str(e).lower()
            assert e.details is not None
