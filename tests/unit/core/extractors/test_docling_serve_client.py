"""Tests for the docling-serve remote conversion client."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
from docling_core.types.doc import DoclingDocument

from docling_graph.core.extractors.docling_serve_client import DoclingServeClient
from docling_graph.exceptions import ConfigurationError, ExtractionError

POST_TARGET = "docling_graph.core.extractors.docling_serve_client.requests.post"


def _doc_dict() -> dict[str, Any]:
    """A minimal valid DoclingDocument payload, as docling-serve returns it."""
    return DoclingDocument(name="test").export_to_dict()


def _payload(
    status: str = "success",
    json_content: dict[str, Any] | None = None,
    errors: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "document": {"json_content": json_content},
        "errors": errors or [],
    }


def _response(
    payload: Any = None, status_code: int = 200, text: str = "", json_error: bool = False
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_error:
        resp.json.side_effect = ValueError("not json")
    else:
        resp.json.return_value = payload
    return resp


class TestInit:
    def test_strips_trailing_slash(self) -> None:
        client = DoclingServeClient(base_url="http://serve:5001/")
        assert client.base_url == "http://serve:5001"

    def test_rejects_missing_scheme(self) -> None:
        with pytest.raises(ConfigurationError, match="http"):
            DoclingServeClient(base_url="serve:5001")

    def test_rejects_empty_url(self) -> None:
        with pytest.raises(ConfigurationError):
            DoclingServeClient(base_url="")


class TestConvertFile:
    @patch(POST_TARGET)
    def test_uploads_file_and_parses_document(self, mock_post: MagicMock, tmp_path: Path) -> None:
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4 fake")
        mock_post.return_value = _response(_payload(json_content=_doc_dict()))

        client = DoclingServeClient(base_url="http://serve:5001", api_key="secret")
        document = client.convert_to_docling_doc(str(source))

        assert isinstance(document, DoclingDocument)
        args, kwargs = mock_post.call_args
        assert args[0] == "http://serve:5001/v1/convert/file"
        assert kwargs["headers"] == {"X-Api-Key": "secret"}
        assert kwargs["data"]["to_formats"] == ["json"]
        assert "pipeline" not in kwargs["data"]  # ocr -> server default (standard)
        assert kwargs["files"]["files"][0] == "doc.pdf"
        assert kwargs["files"]["files"][2] == "application/pdf"
        assert kwargs["timeout"] == 300.0

    @patch(POST_TARGET)
    def test_vision_config_selects_vlm_pipeline(self, mock_post: MagicMock, tmp_path: Path) -> None:
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4 fake")
        mock_post.return_value = _response(_payload(json_content=_doc_dict()))

        client = DoclingServeClient(base_url="http://serve:5001", docling_config="vision")
        client.convert_to_docling_doc(str(source))

        assert mock_post.call_args.kwargs["data"]["pipeline"] == "vlm"

    @patch(POST_TARGET)
    def test_no_api_key_sends_no_auth_header(self, mock_post: MagicMock, tmp_path: Path) -> None:
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4 fake")
        mock_post.return_value = _response(_payload(json_content=_doc_dict()))

        DoclingServeClient(base_url="http://serve:5001").convert_to_docling_doc(str(source))

        assert mock_post.call_args.kwargs["headers"] == {}

    def test_missing_file_raises(self) -> None:
        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="File not found"):
            client.convert_to_docling_doc("/nonexistent/doc.pdf")


class TestConvertUrl:
    @patch(POST_TARGET)
    def test_posts_http_source_as_json(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _response(_payload(json_content=_doc_dict()))

        client = DoclingServeClient(base_url="http://serve:5001")
        document = client.convert_to_docling_doc("https://example.com/paper.pdf")

        assert isinstance(document, DoclingDocument)
        args, kwargs = mock_post.call_args
        assert args[0] == "http://serve:5001/v1/convert/source"
        body = kwargs["json"]
        assert body["http_sources"] == [{"url": "https://example.com/paper.pdf"}]
        assert body["options"]["to_formats"] == ["json"]


class TestResponseHandling:
    @patch(POST_TARGET)
    def test_failure_status_raises(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _response(_payload(status="failure", errors=["boom"]))

        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="status: failure"):
            client.convert_to_docling_doc("https://example.com/doc.pdf")

    @patch(POST_TARGET)
    def test_partial_success_still_parses(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _response(
            _payload(status="partial_success", json_content=_doc_dict(), errors=["page 3 skipped"])
        )

        client = DoclingServeClient(base_url="http://serve:5001")
        document = client.convert_to_docling_doc("https://example.com/doc.pdf")

        assert isinstance(document, DoclingDocument)

    @patch(POST_TARGET)
    def test_missing_json_content_raises(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _response(_payload(json_content=None))

        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="no DoclingDocument JSON"):
            client.convert_to_docling_doc("https://example.com/doc.pdf")

    @patch(POST_TARGET)
    def test_invalid_document_payload_raises(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _response(_payload(json_content={"schema_name": "bogus"}))

        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="Failed to parse DoclingDocument"):
            client.convert_to_docling_doc("https://example.com/doc.pdf")

    @patch(POST_TARGET)
    def test_http_error_raises(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _response(status_code=401, text="unauthorized")

        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="HTTP 401"):
            client.convert_to_docling_doc("https://example.com/doc.pdf")

    @patch(POST_TARGET)
    def test_timeout_raises_with_hint(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = requests.Timeout("too slow")

        client = DoclingServeClient(base_url="http://serve:5001", timeout=1.0)
        with pytest.raises(ExtractionError, match="timed out"):
            client.convert_to_docling_doc("https://example.com/doc.pdf")

    @patch(POST_TARGET)
    def test_connection_error_raises(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = requests.ConnectionError("refused")

        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="Failed to reach docling-serve"):
            client.convert_to_docling_doc("https://example.com/doc.pdf")

    @patch(POST_TARGET)
    def test_non_json_response_raises(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _response(json_error=True)

        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="non-JSON"):
            client.convert_to_docling_doc("https://example.com/doc.pdf")
