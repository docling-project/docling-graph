"""
Input handlers for loading and normalizing different input types.

This module provides handlers that load various input formats and
normalize them into a consistent internal representation.
"""

import json
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import requests
from docling_core.types.doc import DoclingDocument

from ... import __version__
from ...exceptions import ConfigurationError, ValidationError
from .types import InputType, InputTypeDetector


class InputHandler(ABC):
    """Base class for input handlers."""

    @abstractmethod
    def load(self, source: Any) -> Union[str, Path, DoclingDocument]:
        """
        Load and normalize input.

        Args:
            source: Input source to load

        Returns:
            Normalized content (text string, file path, or DoclingDocument)

        Raises:
            ValidationError: If validation fails
            ConfigurationError: If loading fails
        """


class DocumentInputHandler(InputHandler):
    """
    Handles any input that goes to Docling for conversion.

    - File path (any extension): returns path as-is; Docling validates format.
    - .txt file: reads content and writes to a temporary .md file (Docling accepts markdown).
    - Raw text string (API): writes to a temporary .md file.
    """

    def load(self, source: Union[str, Path]) -> Path:
        """
        Normalize document input to a path Docling can consume.

        Returns:
            Path to the source file, or to a temporary .md file for text/.txt content.
        """
        if isinstance(source, Path):
            path = source
        else:
            path = Path(source)

        # Existing file path
        if path.exists() and path.is_file():
            # Normalize .txt to temp .md so Docling gets markdown
            if path.suffix.lower() in {".txt"}:
                content = path.read_text(encoding="utf-8")
                if not content.strip():
                    raise ValidationError(
                        "Text input is empty",
                        details={"file": str(path)},
                    )
                return self._write_temp_md(content)
            return path

        # Raw text string (API mode): write to temp .md
        if isinstance(source, str):
            if not source:
                raise ValidationError(
                    "Text input is empty",
                    details={"hint": "Provide non-empty text content"},
                )
            if not source.strip():
                raise ValidationError(
                    "Text input contains only whitespace",
                    details={"hint": "Provide text with actual content"},
                )
            return self._write_temp_md(source)

        raise ValidationError(
            "Document input must be a file path or a string",
            details={"type": type(source).__name__},
        )

    @staticmethod
    def _write_temp_md(content: str) -> Path:
        """Write content to a temporary .md file and return its path."""
        fd = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
        try:
            fd.write(content)
            return Path(fd.name)
        finally:
            fd.close()


class TextInputHandler(InputHandler):
    """Handles plain text, .txt, and .md files."""

    def load(self, source: Union[str, Path]) -> str:
        """
        Load text input.

        Args:
            source: Text string or path to text file

        Returns:
            Text content as string

        Raises:
            ValidationError: If validation fails (empty, whitespace-only, encoding errors)
            ConfigurationError: If file reading fails
        """
        # Determine if it's a file or raw text
        if isinstance(source, Path):
            # It's a Path object - must be a file
            file_path = source
        elif isinstance(source, str):
            # Check if it looks like a file path
            source_path = Path(source)
            # Only treat as file if it exists or looks like a path (has directory separators)
            if source_path.exists() or ("/" in source or "\\" in source):
                file_path = source_path
            else:
                # Treat as raw text
                content = source
                if not content:
                    raise ValidationError(
                        "Text input is empty",
                        details={"hint": "Provide non-empty text content"},
                    )
                if not content.strip():
                    raise ValidationError(
                        "Text input contains only whitespace",
                        details={"hint": "Provide text with actual content"},
                    )
                return content

        # If we get here, we're treating it as a file
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError as e:
            raise ValidationError(
                f"Failed to read text file: {file_path}",
                details={
                    "file": str(file_path),
                    "error": "Invalid UTF-8 encoding",
                    "hint": "File must be UTF-8 encoded",
                },
            ) from e
        except FileNotFoundError as e:
            raise ValidationError(
                f"Failed to read text file: {file_path}",
                details={"file": str(file_path), "error": "File not found"},
            ) from e
        except Exception as e:
            raise ValidationError(
                f"Failed to read text file: {file_path}",
                details={"file": str(file_path), "error": str(e)},
            ) from e

        # Validate content
        if not content:
            raise ValidationError(
                "Text input is empty",
                details={"file": str(file_path)},
            )

        if not content.strip():
            raise ValidationError(
                "Text input contains only whitespace",
                details={"file": str(file_path)},
            )

        return content


class URLInputHandler(InputHandler):
    """Handles URL inputs with download and type detection."""

    def __init__(self, timeout: int = 30, max_size_mb: int = 100) -> None:
        """
        Initialize URL handler.

        Args:
            timeout: Timeout for downloads in seconds
            max_size_mb: Maximum download size in MB
        """
        self.timeout = timeout
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        # Set User-Agent header to avoid 403 errors from sites like Wikimedia
        self.headers = {
            "User-Agent": f"docling-graph/{__version__} (https://github.com/docling-project/docling-graph)"
        }

    def load(self, source: str) -> Path:
        """
        Download URL content and detect type.

        Process:
        1. Download to temp location with size/timeout limits
        2. Detect content type from headers or extension
        3. Return path to downloaded file

        Args:
            source: URL string

        Returns:
            Path to downloaded file

        Raises:
            ValidationError: If download fails or validation fails
            ConfigurationError: If unexpected error occurs
        """
        try:
            # Make HEAD request first to check size and content type
            try:
                head_response = requests.head(
                    source, timeout=self.timeout, allow_redirects=True, headers=self.headers
                )
                content_length = head_response.headers.get("content-length")
                content_type = head_response.headers.get("content-type", "").lower()

                # Check size if available
                if content_length:
                    size_bytes = int(content_length)
                    if size_bytes > self.max_size_bytes:
                        raise ValidationError(
                            f"URL content exceeds maximum size of {self.max_size_mb}MB",
                            details={
                                "url": source,
                                "size_mb": size_bytes / (1024 * 1024),
                                "max_size_mb": self.max_size_mb,
                            },
                        )
            except requests.RequestException:
                # HEAD request failed, continue with GET
                content_type = None

            # Download content
            response = requests.get(source, timeout=self.timeout, stream=True, headers=self.headers)
            response.raise_for_status()

            # Get content type from response if not from HEAD
            if not content_type:
                content_type = response.headers.get("content-type", "").lower()

            # Determine file extension from content type or URL
            extension = self._determine_extension(source, content_type)

            # Create temp file with appropriate extension
            temp_file = tempfile.NamedTemporaryFile(mode="wb", suffix=extension, delete=False)
            temp_path = Path(temp_file.name)

            # Download with size limit
            downloaded_size = 0
            try:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded_size += len(chunk)
                        if downloaded_size > self.max_size_bytes:
                            temp_file.close()
                            temp_path.unlink()  # Delete temp file
                            raise ValidationError(
                                f"URL content exceeds maximum size of {self.max_size_mb}MB during download",
                                details={
                                    "url": source,
                                    "max_size_mb": self.max_size_mb,
                                },
                            )
                        temp_file.write(chunk)
            finally:
                temp_file.close()

            return temp_path

        except requests.Timeout as e:
            raise ValidationError(
                f"URL request timed out after {self.timeout}s",
                details={"url": source, "timeout": self.timeout},
            ) from e
        except requests.HTTPError as e:
            raise ValidationError(
                f"Failed to download URL (HTTP {e.response.status_code if e.response else 'error'})",
                details={
                    "url": source,
                    "status_code": e.response.status_code if e.response else None,
                    "error": str(e),
                },
            ) from e
        except requests.ConnectionError as e:
            raise ValidationError(
                "Failed to download URL: Connection error",
                details={"url": source, "error": str(e)},
            ) from e
        except requests.RequestException as e:
            raise ValidationError(
                "Failed to download URL",
                details={"url": source, "error": str(e)},
            ) from e
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Wrap unexpected errors as ValidationError for consistency
            raise ValidationError(
                "Failed to download URL",
                details={"url": source, "error": str(e), "type": type(e).__name__},
            ) from e

    def _determine_extension(self, url: str, content_type: str) -> str:
        """
        Determine file extension from URL or content type.

        Args:
            url: URL string
            content_type: Content-Type header value

        Returns:
            File extension with leading dot
        """
        # First, check for known URL patterns (e.g., arXiv)
        url_lower = url.lower()
        if "/pdf/" in url_lower or url_lower.endswith("/pdf"):
            # URLs like https://arxiv.org/pdf/2511.14859 or similar
            return ".pdf"

        # Try to determine from content type first (more reliable)
        content_type_map = {
            "application/pdf": ".pdf",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/tiff": ".tiff",
            "image/tif": ".tif",
            "image/bmp": ".bmp",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "application/json": ".json",
        }

        if content_type:
            for ct_prefix, ext in content_type_map.items():
                if content_type.startswith(ct_prefix):
                    return ext

        # Try to get extension from URL path
        url_path = Path(url.split("?")[0])  # Remove query params
        suffix = url_path.suffix.lower()

        # Only use URL suffix if it looks like a valid extension (2-5 chars)
        if suffix and 2 <= len(suffix) <= 5 and suffix[1:].isalpha():
            return suffix

        # Default to .bin if unknown
        return ".bin"


class DoclingDocumentHandler(InputHandler):
    """Handles serialized DoclingDocument JSON files."""

    def load(self, source: Union[str, Path]) -> DoclingDocument:
        """
        Load DoclingDocument from JSON.

        Args:
            source: Path to DoclingDocument JSON file

        Returns:
            DoclingDocument object

        Raises:
            ValidationError: If validation fails
            ConfigurationError: If loading fails
        """
        file_path = Path(source)

        # Check if file exists
        if not file_path.exists():
            raise ValidationError(
                f"DoclingDocument file not found: {file_path}",
                details={"file": str(file_path)},
            )

        if not file_path.is_file():
            raise ValidationError(
                f"Not a file: {file_path}",
                details={"file": str(file_path)},
            )

        try:
            # Load JSON
            with open(file_path, encoding="utf-8") as f:
                doc_dict = json.load(f)

            # Validate required fields
            if not isinstance(doc_dict, dict):
                raise ValidationError(
                    "Invalid JSON in DoclingDocument file",
                    details={
                        "file": str(file_path),
                        "error": "JSON must be an object",
                    },
                )

            # Check for schema_name
            if "schema_name" not in doc_dict:
                raise ValidationError(
                    "Missing required field in DoclingDocument",
                    details={
                        "file": str(file_path),
                        "missing_field": "schema_name",
                    },
                )

            # Validate schema_name
            if doc_dict.get("schema_name") != "DoclingDocument":
                raise ValidationError(
                    "Invalid schema_name in DoclingDocument",
                    details={
                        "file": str(file_path),
                        "expected": "DoclingDocument",
                        "actual": doc_dict.get("schema_name"),
                        "hint": "schema_name must be 'DoclingDocument'",
                    },
                )

            # Reconstruct DoclingDocument
            # DoclingDocument has a from_dict method
            try:
                document = DoclingDocument.from_dict(doc_dict)  # type: ignore[attr-defined]
            except AttributeError:
                # If from_dict doesn't exist, try direct construction
                # This is a fallback - actual implementation may vary
                document = DoclingDocument(**doc_dict)

            return document  # type: ignore[no-any-return]

        except json.JSONDecodeError as e:
            raise ValidationError(
                "Invalid JSON in DoclingDocument file",
                details={
                    "file": str(file_path),
                    "error": str(e),
                },
            ) from e
        except ValidationError:
            # Re-raise validation errors
            raise
        except FileNotFoundError as e:
            raise ValidationError(
                f"DoclingDocument file not found: {file_path}",
                details={"file": str(file_path), "error": str(e)},
            ) from e
        except Exception as e:
            raise ValidationError(
                "Error loading DoclingDocument",
                details={"file": str(file_path), "error": str(e), "type": type(e).__name__},
            ) from e
