"""
Client for converting documents through a remote docling-serve instance.

docling-serve (https://github.com/docling-project/docling-serve) exposes the
Docling conversion pipeline as a REST API. This client offloads the document
conversion step to such an instance and parses the returned DoclingDocument
JSON, so the rest of the docling-graph pipeline (chunking, extraction, graph
conversion) runs unchanged — without loading any local conversion models.

Only conversion is remote: extraction (LLM/VLM) still runs wherever the
pipeline is configured to run it.
"""

import mimetypes
from pathlib import Path
from typing import Any

import requests
from docling_core.types.doc import DoclingDocument

from ...exceptions import ConfigurationError, ExtractionError
from ...logging_utils import get_component_logger

logger = get_component_logger("DoclingServeClient", __name__)

# Conversion results the server reports as usable.
_OK_STATUSES = {"success", "partial_success"}


class DoclingServeClient:
    """Converts documents to DoclingDocument via a docling-serve REST API."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        docling_config: str = "ocr",
    ) -> None:
        """
        Initialize the docling-serve client.

        Args:
            base_url: Base URL of the docling-serve instance
                (e.g., "http://localhost:5001").
            api_key: Optional API key, sent as the ``X-Api-Key`` header
                (docling-serve's ``DOCLING_SERVE_API_KEY`` authentication).
            timeout: Request timeout in seconds. Synchronous conversion holds
                the connection for the whole conversion, so large documents
                need generous timeouts.
            docling_config: Docling pipeline selection, mirroring local
                conversion: "ocr" maps to the standard pipeline, "vision" to
                the VLM pipeline.
        """
        base_url = (base_url or "").strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ConfigurationError(
                "docling-serve URL must start with http:// or https://",
                details={"base_url": base_url},
            )
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.docling_config = docling_config
        logger.info(
            "Initialized for %s (%s pipeline)",
            self.base_url,
            "vlm" if docling_config == "vision" else "standard",
        )

    def _headers(self) -> dict[str, str]:
        return {"X-Api-Key": self.api_key} if self.api_key else {}

    def _options(self) -> dict[str, Any]:
        """Conversion options requesting DoclingDocument JSON output.

        Kept minimal on purpose: server-side defaults (do_ocr, table
        structure, ...) match docling-graph's local "ocr" pipeline, and
        fewer fields means fewer compatibility breaks across docling-serve
        versions.
        """
        options: dict[str, Any] = {
            "to_formats": ["json"],
            "image_export_mode": "placeholder",
        }
        if self.docling_config == "vision":
            options["pipeline"] = "vlm"
        return options

    def convert_to_docling_doc(self, source: str) -> DoclingDocument:
        """
        Convert a document remotely and return the parsed DoclingDocument.

        Args:
            source: Local file path, or an http(s) URL the server should
                fetch itself.

        Raises:
            ExtractionError: On connection failures, HTTP errors, a failed
                conversion status, or an unparsable response document.
        """
        if source.startswith(("http://", "https://")):
            payload = self._convert_url(source)
        else:
            payload = self._convert_file(Path(source))
        return self._parse_response(payload, source)

    def _convert_url(self, url: str) -> dict[str, Any]:
        """Ask the server to fetch and convert a URL (POST /v1/convert/source)."""
        endpoint = f"{self.base_url}/v1/convert/source"
        body = {"options": self._options(), "http_sources": [{"url": url}]}
        logger.info("Converting URL via docling-serve: %s", url)
        return self._post_json(endpoint, source=url, json=body)

    def _convert_file(self, path: Path) -> dict[str, Any]:
        """Upload a local file for conversion (POST /v1/convert/file).

        Options are flattened to multipart form fields; ``requests`` encodes
        list values as repeated fields, which is what FastAPI expects.
        """
        endpoint = f"{self.base_url}/v1/convert/file"
        if not path.is_file():
            raise ExtractionError(
                f"File not found for docling-serve conversion: {path}",
                details={"source": str(path)},
            )
        mimetype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        logger.info("Uploading to docling-serve: %s", path)
        with open(path, "rb") as fh:
            return self._post_json(
                endpoint,
                source=str(path),
                data=self._options(),
                files={"files": (path.name, fh, mimetype)},
            )

    def _post_json(self, endpoint: str, *, source: str, **kwargs: Any) -> dict[str, Any]:
        """POST to the server and return the decoded JSON response."""
        try:
            response = requests.post(
                endpoint, headers=self._headers(), timeout=self.timeout, **kwargs
            )
        except requests.Timeout as e:
            raise ExtractionError(
                f"docling-serve request timed out after {self.timeout}s",
                details={
                    "endpoint": endpoint,
                    "source": source,
                    "hint": "Raise docling_serve_timeout for large documents.",
                },
                cause=e,
            ) from e
        except requests.RequestException as e:
            raise ExtractionError(
                f"Failed to reach docling-serve at {self.base_url}",
                details={"endpoint": endpoint, "source": source, "error": str(e)},
                cause=e,
            ) from e

        if response.status_code >= 400:
            raise ExtractionError(
                f"docling-serve returned HTTP {response.status_code}",
                details={
                    "endpoint": endpoint,
                    "source": source,
                    "response": response.text[:500],
                },
            )

        try:
            payload = response.json()
        except ValueError as e:
            raise ExtractionError(
                "docling-serve returned a non-JSON response",
                details={"endpoint": endpoint, "source": source},
                cause=e,
            ) from e
        if not isinstance(payload, dict):
            raise ExtractionError(
                "docling-serve returned an unexpected response shape",
                details={"endpoint": endpoint, "source": source, "type": type(payload).__name__},
            )
        return payload

    def _parse_response(self, payload: dict[str, Any], source: str) -> DoclingDocument:
        """Validate conversion status and parse document.json_content."""
        status = payload.get("status")
        if status not in _OK_STATUSES:
            raise ExtractionError(
                f"docling-serve conversion failed (status: {status})",
                details={"source": source, "errors": payload.get("errors") or []},
            )
        if status == "partial_success":
            logger.warning(
                "docling-serve reported partial success for %s: %s",
                source,
                payload.get("errors") or [],
            )

        document = payload.get("document") or {}
        json_content = document.get("json_content")
        if not json_content:
            raise ExtractionError(
                "docling-serve response contains no DoclingDocument JSON",
                details={
                    "source": source,
                    "hint": "The server must support the 'json' output format (to_formats).",
                },
            )

        try:
            return DoclingDocument.model_validate(json_content)
        except Exception as e:
            raise ExtractionError(
                "Failed to parse DoclingDocument returned by docling-serve",
                details={"source": source, "error": str(e)},
                cause=e,
            ) from e
