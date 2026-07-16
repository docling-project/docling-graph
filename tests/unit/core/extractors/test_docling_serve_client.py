"""Tests for the docling-serve remote conversion client facade.

The facade delegates transport to the official docling.service_client SDK;
these tests mock the SDK class (as aliased in the facade module) and pin the
facade's contract: constructor pass-through, options mapping, in-body target
submission, result handling, and the exception-translation table.
"""

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from docling.datamodel.base_models import ConversionStatus, OutputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import ProcessingPipeline
from docling.datamodel.service.responses import (
    FailureCategory,
    FailurePhase,
    PublicFailureInfo,
)
from docling.datamodel.service.targets import InBodyTarget
from docling.service_client import (
    DoclingServiceClientError,
    ResponseSchemaMismatchError,
    ResultExpiredError,
    ResultNotReadyError,
    ServiceError,
    ServiceUnavailableError,
    StatusWatcherKind,
    TaskExecutionError,
    TaskNotFoundError,
    TaskTimeoutError,
    UsageLimitExceededError,
)
from docling_core.types.doc import DoclingDocument, DocumentOrigin
from pydantic import ValidationError

from docling_graph.core.extractors.docling_serve_client import DoclingServeClient
from docling_graph.exceptions import ConfigurationError, ExtractionError

CLIENT_TARGET = "docling_graph.core.extractors.docling_serve_client._UpstreamClient"


def _document() -> DoclingDocument:
    """A converted-looking document (origin set, as real conversions produce)."""
    return DoclingDocument(
        name="test",
        origin=DocumentOrigin(mimetype="application/pdf", binary_hash=0, filename="doc.pdf"),
    )


def _result(
    status: ConversionStatus = ConversionStatus.SUCCESS,
    errors: list[Any] | None = None,
    document: DoclingDocument | None = None,
) -> MagicMock:
    """A stub upstream ConversionResult."""
    result = MagicMock(spec=ConversionResult)
    result.status = status
    result.document = document if document is not None else _document()
    result.errors = errors or []
    return result


@pytest.fixture
def upstream() -> Any:
    """Patch the SDK client class; yields the MagicMock class object."""
    with patch(CLIENT_TARGET) as mock_cls:
        mock_cls.return_value.submit.return_value.result.return_value = _result()
        yield mock_cls


def _convert_calls(upstream: MagicMock) -> tuple[MagicMock, MagicMock]:
    """Return (submit call, job.result call) mocks."""
    return (
        upstream.return_value.submit,
        upstream.return_value.submit.return_value.result,
    )


class TestInit:
    def test_strips_trailing_slash(self, upstream: MagicMock) -> None:
        client = DoclingServeClient(base_url="http://serve:5001/")
        assert client.base_url == "http://serve:5001"

    def test_rejects_missing_scheme(self) -> None:
        with pytest.raises(ConfigurationError, match="http"):
            DoclingServeClient(base_url="serve:5001")

    def test_rejects_empty_url(self) -> None:
        with pytest.raises(ConfigurationError):
            DoclingServeClient(base_url="")

    def test_upstream_url_validation_maps_to_configuration_error(self, upstream: MagicMock) -> None:
        upstream.side_effect = ValueError("query strings are not allowed")
        with pytest.raises(ConfigurationError, match="URL is invalid"):
            DoclingServeClient(base_url="http://serve:5001?bad=1")

    def test_constructor_passthrough(self, upstream: MagicMock) -> None:
        DoclingServeClient(base_url="http://serve:5001", api_key="secret", timeout=600)

        kwargs = upstream.call_args.kwargs
        assert kwargs["url"] == "http://serve:5001"
        assert kwargs["api_key"] == "secret"
        assert kwargs["job_timeout"] == 600.0
        assert kwargs["http_read_timeout"] == 600.0  # max(60, timeout)
        assert kwargs["status_watcher"] is StatusWatcherKind.POLLING
        assert kwargs["options"].to_formats == [OutputFormat.JSON]
        # ocr (default) -> server default pipeline, no override on the wire
        assert "pipeline" not in kwargs["options"].model_dump(exclude_defaults=True)

    def test_read_timeout_never_below_sdk_default(self, upstream: MagicMock) -> None:
        DoclingServeClient(base_url="http://serve:5001", timeout=5)

        kwargs = upstream.call_args.kwargs
        assert kwargs["job_timeout"] == 5.0
        assert kwargs["http_read_timeout"] == 60.0

    def test_vision_config_selects_vlm_pipeline(self, upstream: MagicMock) -> None:
        DoclingServeClient(base_url="http://serve:5001", docling_config="vision")

        assert upstream.call_args.kwargs["options"].pipeline is ProcessingPipeline.VLM

    def test_no_api_key_passes_empty_string(self, upstream: MagicMock) -> None:
        # The SDK only emits the X-Api-Key header for a truthy key, so the
        # facade hands it an empty string when no key is configured.
        DoclingServeClient(base_url="http://serve:5001")

        assert upstream.call_args.kwargs["api_key"] == ""


class TestHeaders:
    def test_headers_injected_as_client_defaults(self, upstream: MagicMock) -> None:
        # Poll/result requests only carry the httpx client's default headers,
        # so custom headers must be injected there (not per-call).
        DoclingServeClient(
            base_url="http://serve:5001",
            headers={"Authorization": "Bearer token"},
        )

        upstream.return_value._http_client.headers.update.assert_called_once_with(
            {"Authorization": "Bearer token"}
        )

    def test_no_headers_no_injection(self, upstream: MagicMock) -> None:
        DoclingServeClient(base_url="http://serve:5001")

        upstream.return_value._http_client.headers.update.assert_not_called()

    def test_missing_http_client_attribute_raises(self, upstream: MagicMock) -> None:
        # If a future SDK version drops the private attribute, fail loudly
        # instead of silently sending unauthenticated requests.
        instance = MagicMock(spec=[])  # no attributes at all
        upstream.return_value = instance
        with pytest.raises(ConfigurationError, match="headers"):
            DoclingServeClient(base_url="http://serve:5001", headers={"Authorization": "Bearer x"})


class TestConvertSources:
    def test_local_file_submitted_in_body(self, upstream: MagicMock, tmp_path: Path) -> None:
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4 fake")

        client = DoclingServeClient(base_url="http://serve:5001")
        document = client.convert_to_docling_doc(str(source))

        assert isinstance(document, DoclingDocument)
        submit, job_result = _convert_calls(upstream)
        assert submit.call_args.kwargs["source"] == source
        # Explicit in-body target: never the SDK's presigned-first auto-target
        # (double uploads, SSRF-guarded artifact downloads without our headers).
        assert isinstance(submit.call_args.kwargs["target"], InBodyTarget)
        job_result.assert_called_once_with(timeout=300.0)

    def test_url_passed_through_as_string(self, upstream: MagicMock) -> None:
        client = DoclingServeClient(base_url="http://serve:5001")
        document = client.convert_to_docling_doc("https://example.com/paper.pdf")

        assert isinstance(document, DoclingDocument)
        submit, _ = _convert_calls(upstream)
        assert submit.call_args.kwargs["source"] == "https://example.com/paper.pdf"
        assert isinstance(submit.call_args.kwargs["target"], InBodyTarget)

    def test_missing_file_raises(self, upstream: MagicMock) -> None:
        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="File not found"):
            client.convert_to_docling_doc("/nonexistent/doc.pdf")
        upstream.return_value.submit.assert_not_called()

    @pytest.mark.parametrize("bad_url", ["http://", "https://exa mple.com/doc.pdf"])
    def test_malformed_source_url_raises_before_submission(
        self, upstream: MagicMock, bad_url: str
    ) -> None:
        # The SDK coerces http(s) strings its URL validation rejects into a
        # Path (-> FileNotFoundError); the facade must pre-validate instead.
        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="source URL is invalid"):
            client.convert_to_docling_doc(bad_url)
        upstream.return_value.submit.assert_not_called()


class TestResultHandling:
    def test_failure_status_raises_with_error_text(self, upstream: MagicMock) -> None:
        error = MagicMock()
        error.error_message = "boom"
        upstream.return_value.submit.return_value.result.return_value = _result(
            status=ConversionStatus.FAILURE, errors=[error]
        )

        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="status: failure") as exc_info:
            client.convert_to_docling_doc("https://example.com/doc.pdf")
        assert "boom" in str(exc_info.value)

    def test_partial_success_still_parses(self, upstream: MagicMock) -> None:
        upstream.return_value.submit.return_value.result.return_value = _result(
            status=ConversionStatus.PARTIAL_SUCCESS, errors=["page 3 skipped"]
        )

        client = DoclingServeClient(base_url="http://serve:5001")
        document = client.convert_to_docling_doc("https://example.com/doc.pdf")

        assert isinstance(document, DoclingDocument)

    def test_empty_substituted_document_raises(self, upstream: MagicMock) -> None:
        # The SDK substitutes an empty DoclingDocument (no origin, no content)
        # when a success result carries no json_content — surface that as the
        # explicit protocol error instead of an empty extraction downstream.
        upstream.return_value.submit.return_value.result.return_value = _result(
            document=DoclingDocument(name="doc")
        )

        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="no DoclingDocument JSON"):
            client.convert_to_docling_doc("https://example.com/doc.pdf")

    def test_unexpected_result_type_raises(self, upstream: MagicMock) -> None:
        upstream.return_value.submit.return_value.result.return_value = {"raw": True}

        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="unexpected result type"):
            client.convert_to_docling_doc("https://example.com/doc.pdf")


class TestExceptionTranslation:
    """Every SDK exception maps to ExtractionError with the pinned message."""

    @pytest.mark.parametrize(
        ("exception", "match", "detail_keys"),
        [
            (
                UsageLimitExceededError("quota", status_code=402, current_usage=10, limit=10),
                "HTTP 402",
                {"current_usage", "limit"},
            ),
            (
                ResponseSchemaMismatchError("bad payload", status_code=200),
                "Failed to parse DoclingDocument",
                {"error"},
            ),
            (
                ServiceUnavailableError("server error", status_code=503),
                "HTTP 503",
                {"error", "status_code"},
            ),
            (
                ServiceUnavailableError("connection refused"),
                "Failed to reach docling-serve",
                {"error"},
            ),
            (
                ServiceError("Task submission failed.", status_code=401),
                "HTTP 401",
                {"response"},
            ),
            (
                TaskTimeoutError("too slow"),
                "did not finish within",
                {"hint"},
            ),
            (
                TaskNotFoundError("unknown task"),
                "no longer knows the conversion task",
                {"error"},
            ),
            (
                ResultNotReadyError("still running"),
                "result unavailable",
                {"error"},
            ),
            (
                ResultExpiredError("gone"),
                "result unavailable",
                {"error"},
            ),
            (
                TaskExecutionError(
                    "conversion failed",
                    failure=PublicFailureInfo(
                        category=FailureCategory.BACKEND_FAILURE,
                        message="OCR crashed",
                        retryable=False,
                        phase=FailurePhase.EXECUTION,
                    ),
                ),
                "status: failure",
                {"errors", "retryable"},
            ),
            (
                DoclingServiceClientError("something odd"),
                "request failed",
                {"error"},
            ),
        ],
    )
    def test_sdk_exception_maps_to_extraction_error(
        self,
        upstream: MagicMock,
        exception: DoclingServiceClientError,
        match: str,
        detail_keys: set[str],
    ) -> None:
        upstream.return_value.submit.side_effect = exception

        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match=match) as exc_info:
            client.convert_to_docling_doc("https://example.com/doc.pdf")

        err = exc_info.value
        assert err.cause is exception
        assert err.details["source"] == "https://example.com/doc.pdf"
        assert detail_keys <= set(err.details)
        assert err.__cause__ is exception  # never leaks the SDK exception

    def test_result_phase_exception_translated(self, upstream: MagicMock) -> None:
        # Task-phase failures surface from job.result(), not submit().
        upstream.return_value.submit.return_value.result.side_effect = TaskTimeoutError("deadline")

        client = DoclingServeClient(base_url="http://serve:5001", timeout=1.0)
        with pytest.raises(ExtractionError, match=r"1\.0s") as exc_info:
            client.convert_to_docling_doc("https://example.com/doc.pdf")
        assert "docling_serve_timeout" in exc_info.value.details["hint"]

    def test_task_execution_failure_message_in_error(self, upstream: MagicMock) -> None:
        upstream.return_value.submit.return_value.result.side_effect = TaskExecutionError(
            "conversion failed",
            failure=PublicFailureInfo(
                category=FailureCategory.BACKEND_FAILURE,
                message="OCR crashed",
                retryable=True,
                phase=FailurePhase.EXECUTION,
            ),
        )

        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="OCR crashed") as exc_info:
            client.convert_to_docling_doc("https://example.com/doc.pdf")
        assert exc_info.value.details["retryable"] is True

    def test_submit_404_hints_at_old_server(self, upstream: MagicMock) -> None:
        upstream.return_value.submit.side_effect = ServiceError(
            "Not found", status_code=404, detail="Not Found"
        )

        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="HTTP 404") as exc_info:
            client.convert_to_docling_doc("https://example.com/doc.pdf")
        assert "docling-serve >= 1.0.0" in exc_info.value.details["hint"]

    def test_sdk_value_error_translated(self, upstream: MagicMock) -> None:
        upstream.return_value.submit.side_effect = ValueError("bad option")

        client = DoclingServeClient(base_url="http://serve:5001")
        with pytest.raises(ExtractionError, match="rejected the conversion request"):
            client.convert_to_docling_doc("https://example.com/doc.pdf")


class TestLifecycle:
    def test_close_delegates_to_sdk(self, upstream: MagicMock) -> None:
        client = DoclingServeClient(base_url="http://serve:5001")
        client.close()
        client.close()  # safe to call twice

        assert upstream.return_value.close.call_count == 2

    def test_context_manager_closes(self, upstream: MagicMock) -> None:
        with DoclingServeClient(base_url="http://serve:5001") as client:
            assert client.base_url == "http://serve:5001"

        upstream.return_value.close.assert_called_once()


class TestTlsShim:
    def test_ca_bundle_visible_during_sdk_construction_then_restored(
        self, upstream: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # httpx builds its SSL context inside httpx.Client.__init__ (called by
        # the SDK ctor), so the shim only needs to cover construction; leaving
        # SSL_CERT_FILE set process-wide would repoint TLS trust for every
        # later httpx client (e.g. LLM calls).
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/corp-ca.pem")
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)

        seen: dict[str, str | None] = {}

        def capture(**kwargs: Any) -> MagicMock:
            seen["during_ctor"] = os.environ.get("SSL_CERT_FILE")
            return MagicMock()

        upstream.side_effect = capture
        DoclingServeClient(base_url="https://serve:5001")

        assert seen["during_ctor"] == "/etc/corp-ca.pem"
        assert "SSL_CERT_FILE" not in os.environ  # restored after construction

    def test_existing_ssl_cert_file_untouched(
        self, upstream: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/corp-ca.pem")
        monkeypatch.setenv("SSL_CERT_FILE", "/etc/other-ca.pem")

        DoclingServeClient(base_url="https://serve:5001")

        assert os.environ["SSL_CERT_FILE"] == "/etc/other-ca.pem"

    def test_restored_even_when_ctor_raises(
        self, upstream: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/corp-ca.pem")
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        upstream.side_effect = ValueError("bad url")

        with pytest.raises(ConfigurationError):
            DoclingServeClient(base_url="https://serve:5001?x=1")

        assert "SSL_CERT_FILE" not in os.environ
