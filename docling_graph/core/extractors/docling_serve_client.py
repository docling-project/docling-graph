"""
Client for converting documents through a remote docling-serve instance.

docling-serve (https://github.com/docling-project/docling-serve) exposes the
Docling conversion pipeline as a REST API. This facade delegates the wire
protocol to the official client SDK (docling.service_client) — async task
submission, HTTP-polled status, retries, typed responses — and translates
its results and errors into docling-graph's pipeline contract, so the rest
of the pipeline (chunking, extraction, graph conversion) runs unchanged
without loading any local conversion models.

Only conversion is remote: extraction (LLM/VLM) still runs wherever the
pipeline is configured to run it.
"""

import os
from pathlib import Path
from types import TracebackType
from typing import Any

from docling.datamodel.base_models import ConversionStatus, OutputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import ProcessingPipeline
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.datamodel.service.targets import InBodyTarget
from docling.service_client import (
    ArtifactDownloadError,
    DoclingServiceClient as _UpstreamClient,
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
from docling_core.types.doc import DoclingDocument
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from ...exceptions import ConfigurationError, ExtractionError
from ...logging_utils import get_component_logger

logger = get_component_logger("DoclingServeClient", __name__)

# Conversion results the server reports as usable.
_OK_STATUSES = {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}


class DoclingServeClient:
    """Converts documents to DoclingDocument via a docling-serve REST API.

    Thin adapter over the official client SDK: keeps docling-graph's
    configuration surface and error taxonomy while the SDK handles
    transport (async task submission, status polling, retries).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        docling_config: str = "ocr",
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the docling-serve client.

        Args:
            base_url: Base URL of the docling-serve instance
                (e.g., "http://localhost:5001").
            api_key: Optional API key, sent as the ``X-Api-Key`` header
                (docling-serve's ``DOCLING_SERVE_API_KEY`` authentication).
            timeout: Approximate deadline in seconds for one document's
                conversion job, from submission to terminal status. Server
                queue time counts toward it; connect/read timeouts and
                transient-error retries are bounded separately.
            docling_config: Docling pipeline selection, mirroring local
                conversion: "ocr" maps to the standard pipeline, "vision" to
                the VLM pipeline.
            headers: Extra HTTP headers sent on every request to the
                docling-serve API (submit, poll, result), e.g.
                ``{"Authorization": "Bearer <token>"}`` for deployments
                fronted by an auth proxy instead of X-Api-Key.
        """
        base_url = (base_url or "").strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ConfigurationError(
                "docling-serve URL must start with http:// or https://",
                details={"base_url": base_url},
            )
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = float(timeout)
        self.docling_config = docling_config
        self.headers = dict(headers or {})

        # TLS parity shim: the previous requests-based client honored
        # REQUESTS_CA_BUNDLE; the SDK's httpx transport reads SSL_CERT_FILE.
        # httpx builds its SSL context eagerly inside httpx.Client.__init__
        # (which the SDK constructs below), so the variable only needs to be
        # set during construction — leaving it set process-wide would repoint
        # TLS trust for every later httpx client (e.g. LLM calls via litellm)
        # at the corp-only bundle.
        shimmed = False
        if os.environ.get("REQUESTS_CA_BUNDLE") and not os.environ.get("SSL_CERT_FILE"):
            os.environ["SSL_CERT_FILE"] = os.environ["REQUESTS_CA_BUNDLE"]
            shimmed = True
        try:
            self._client = _UpstreamClient(
                url=self.base_url,
                # The SDK only emits the X-Api-Key header when the key is
                # truthy, so an unset key sends no auth header at all.
                api_key=self.api_key or "",
                options=self._options(),
                job_timeout=self.timeout,
                # The SDK's fixed 60s read/write timeout would abort large
                # multipart uploads the old client (one timeout for the whole
                # request) tolerated; never go below the SDK default.
                http_read_timeout=max(60.0, self.timeout),
                # Always poll over HTTP: the WebSocket watcher cannot carry
                # custom headers, puts the api key in the WS URL query (which
                # lands in proxy access logs), and pays a multi-attempt
                # fallback penalty behind WS-hostile proxies. Batch pipelines
                # gain nothing from push latency over the 5s server long-poll.
                status_watcher=StatusWatcherKind.POLLING,
            )
        except ValueError as e:
            raise ConfigurationError(
                f"docling-serve URL is invalid: {e}",
                details={"base_url": base_url},
            ) from e
        finally:
            if shimmed:
                os.environ.pop("SSL_CERT_FILE", None)

        if self.headers:
            # The SDK has no public hook for extra headers, and per-call
            # headers are dropped by its poll/result requests; the underlying
            # httpx client's default headers are inherited by every request.
            http_client = getattr(self._client, "_http_client", None)
            if http_client is None:
                raise ConfigurationError(
                    "Custom docling-serve headers are not supported by the "
                    "installed docling version (no _http_client attribute on "
                    "DoclingServiceClient)",
                    details={"headers": sorted(self.headers)},
                )
            http_client.headers.update(self.headers)

        logger.info(
            "Initialized for %s (%s pipeline)",
            self.base_url,
            "vlm" if docling_config == "vision" else "standard",
        )

    def _options(self) -> ConvertDocumentsOptions:
        """Conversion options requesting DoclingDocument JSON output.

        Kept minimal on purpose: server-side defaults (do_ocr, table
        structure, image placeholders, ...) match docling-graph's local
        "ocr" pipeline, and fewer fields means fewer compatibility breaks
        across docling-serve versions.
        """
        options = ConvertDocumentsOptions(to_formats=[OutputFormat.JSON])
        if self.docling_config == "vision":
            options.pipeline = ProcessingPipeline.VLM
        return options

    def convert_to_docling_doc(self, source: str) -> DoclingDocument:
        """
        Convert a document remotely and return the parsed DoclingDocument.

        Blocking call: the document is submitted to the server, its status is
        polled until the job finishes (bounded by ``timeout``), and the
        result is fetched and parsed.

        Args:
            source: Local file path, or an http(s) URL the server should
                fetch itself.

        Raises:
            ExtractionError: On connection failures, HTTP errors, a failed
                conversion status, job timeouts, or an unparsable response
                document.
        """
        upstream_source: Path | str
        if source.startswith(("http://", "https://")):
            # Pre-validate: the SDK silently coerces http(s) strings its
            # AnyHttpUrl validation rejects into a Path, which then fails
            # with an untranslated FileNotFoundError.
            try:
                TypeAdapter(AnyHttpUrl).validate_python(source)
            except ValidationError as e:
                raise ExtractionError(
                    f"docling-serve source URL is invalid: {e}",
                    details={"source": source},
                ) from e
            upstream_source = source  # the server fetches the URL itself
        else:
            path = Path(source)
            if not path.is_file():
                raise ExtractionError(
                    f"File not found for docling-serve conversion: {path}",
                    details={"source": str(path)},
                )
            upstream_source = path  # uploaded as multipart

        logger.info("Converting via docling-serve: %s", source)
        try:
            # Explicit in-body target: the SDK's default auto-target tries a
            # presigned-URL result first, which double-submits every document
            # against servers without artifact storage, is refused by the
            # SDK's SSRF guard for private-network artifact URLs, and fetches
            # artifacts with a bare client that carries neither the api key
            # nor custom headers. In-body results avoid all three and match
            # the previous client's behavior.
            job = self._client.submit(source=upstream_source, target=InBodyTarget())
            result = job.result(timeout=self.timeout)
        except DoclingServiceClientError as e:
            raise self._translate(e, source) from e
        except ValueError as e:
            # Defensive: translate residual SDK-side pydantic validation
            # errors instead of leaking them to the strategies' generic
            # exception handler.
            raise ExtractionError(
                f"docling-serve rejected the conversion request: {e}",
                details={"source": source},
            ) from e

        if not isinstance(result, ConversionResult):
            raise ExtractionError(
                "docling-serve returned an unexpected result type",
                details={"source": source, "type": type(result).__name__},
            )
        if result.status not in _OK_STATUSES:
            errors = [self._error_text(err) for err in result.errors]
            error_text = "; ".join(errors)[:500]
            raise ExtractionError(
                f"docling-serve conversion failed (status: {result.status.value})"
                + (f": {error_text}" if error_text else ""),
                details={"source": source, "errors": errors},
            )
        if result.status == ConversionStatus.PARTIAL_SUCCESS:
            logger.warning(
                "docling-serve reported partial success for %s: %s",
                source,
                [self._error_text(err) for err in result.errors],
            )
        document = result.document
        # The SDK substitutes an empty DoclingDocument when a success result
        # carries no json_content; a genuinely converted document always has
        # an origin, so this fingerprint only matches the substitution.
        if document.origin is None and not document.texts and not document.tables:
            raise ExtractionError(
                "docling-serve response contains no DoclingDocument JSON",
                details={
                    "source": source,
                    "status": result.status.value,
                    "hint": (
                        "The server did not honor the 'json' output format "
                        "(to_formats) — check the docling-serve version and "
                        "any proxy rewriting responses."
                    ),
                },
            )
        return document

    @staticmethod
    def _error_text(error: object) -> str:
        return str(getattr(error, "error_message", None) or error)

    def _translate(self, e: DoclingServiceClientError, source: str) -> ExtractionError:
        """Map SDK exceptions onto docling-graph's error taxonomy.

        Most-derived classes first; every branch keeps the message substrings
        that docs and callers rely on ("Failed to reach docling-serve",
        "HTTP <code>", "status: failure", "Failed to parse DoclingDocument").
        """
        details: dict[str, Any]
        if isinstance(e, UsageLimitExceededError):
            return ExtractionError(
                f"docling-serve returned HTTP {e.status_code or 402} (usage limit exceeded)",
                details={
                    "source": source,
                    "current_usage": e.current_usage,
                    "limit": e.limit,
                },
                cause=e,
            )
        if isinstance(e, ResponseSchemaMismatchError):
            return ExtractionError(
                "Failed to parse DoclingDocument response from docling-serve "
                "(client and server versions may differ)",
                details={"source": source, "error": str(e)},
                cause=e,
            )
        if isinstance(e, ServiceUnavailableError):
            if e.status_code is not None:
                return ExtractionError(
                    f"docling-serve returned HTTP {e.status_code}",
                    details={"source": source, "error": str(e), "status_code": e.status_code},
                    cause=e,
                )
            details = {"source": source, "error": str(e)}
            if "certificate" in str(e).lower() or "ssl" in str(e).lower():
                details["hint"] = (
                    "TLS verification failed. For custom CAs set SSL_CERT_FILE "
                    "(httpx does not read REQUESTS_CA_BUNDLE)."
                )
            return ExtractionError(
                f"Failed to reach docling-serve at {self.base_url}",
                details=details,
                cause=e,
            )
        if isinstance(e, ServiceError):
            details = {"source": source, "response": str(e.detail or e)[:500]}
            if e.status_code == 404 or (e.status_code == 422 and "target" in str(e.detail or "")):
                details["hint"] = (
                    "The server may not expose the v1 async task API "
                    "(requires docling-serve >= 1.0.0) — check GET /version."
                )
            return ExtractionError(
                f"docling-serve returned HTTP {e.status_code}"
                if e.status_code is not None
                else f"docling-serve request failed: {e}",
                details=details,
                cause=e,
            )
        if isinstance(e, TaskTimeoutError):
            return ExtractionError(
                f"docling-serve conversion did not finish within {self.timeout}s (job deadline)",
                details={
                    "source": source,
                    "hint": (
                        "Raise docling_serve_timeout, or check server load/queue — "
                        "the job may still be running server-side."
                    ),
                },
                cause=e,
            )
        if isinstance(e, TaskNotFoundError):
            return ExtractionError(
                "docling-serve no longer knows the conversion task (expired or restarted server)",
                details={"source": source, "error": str(e)},
                cause=e,
            )
        if isinstance(e, ResultNotReadyError | ResultExpiredError):
            return ExtractionError(
                f"docling-serve result unavailable: {e}",
                details={"source": source, "error": str(e)},
                cause=e,
            )
        if isinstance(e, TaskExecutionError):
            failure = e.failure
            message = "docling-serve conversion failed (status: failure)"
            details = {"source": source}
            if failure is not None:
                message += f": {failure.message}"
                details["errors"] = [
                    f"{failure.category.value}/{failure.phase.value}: {failure.message}"
                ]
                details["retryable"] = failure.retryable
            else:
                details["errors"] = [str(e)]
            return ExtractionError(message, details=details, cause=e)
        if isinstance(e, ArtifactDownloadError):
            return ExtractionError(
                "Failed to download the docling-serve conversion result",
                details={"source": source, "error": str(e)},
                cause=e,
            )
        return ExtractionError(
            f"docling-serve request failed: {e}",
            details={"source": source, "error": str(e)},
            cause=e,
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "DoclingServeClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
