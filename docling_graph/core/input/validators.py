"""
Input validators for different input types.

This module provides validation logic for various input formats
to ensure they meet requirements before processing.
"""

import ipaddress
import json
import socket
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union
from urllib.parse import urlparse

from ...exceptions import ConfigurationError, ValidationError


class InputValidator(ABC):
    """Base class for input validators."""

    @abstractmethod
    def validate(self, source: Any) -> None:
        """
        Validate input source.

        Args:
            source: Input source to validate

        Raises:
            ValidationError: If validation fails
            ConfigurationError: If configuration is invalid
        """


class TextValidator(InputValidator):
    """Validates text inputs (plain text, .txt, .md)."""

    def validate(self, source: Union[str, Path]) -> None:
        """
        Validate text input.

        Checks:
        - Not empty
        - Not only whitespace
        - Readable encoding (for files)

        Args:
            source: Text string or path to text file

        Raises:
            ValidationError: If validation fails
            ConfigurationError: If file not found
        """
        # Handle None input
        if source is None:
            raise ValidationError(
                "Text input cannot be None",
                details={"hint": "Provide valid text content or file path"},
            )

        # If it's a Path object, validate as file
        if isinstance(source, Path):
            self._validate_file(source)
            return

        # For strings, check if it's a file path that exists
        source_str = str(source)

        # Try to check if it's a file, but handle cases where the string
        # is too long or invalid as a file path
        try:
            source_path = Path(source_str)
            # Only treat as file if it actually exists and is a file
            # This prevents treating empty strings or "." as file paths
            if source_path.exists() and source_path.is_file():
                self._validate_file(source_path)
            else:
                # Validate as raw text string
                self._validate_string(source_str)
        except (OSError, ValueError):
            # If path checking fails (e.g., filename too long), treat as text
            self._validate_string(source_str)

    def _validate_string(self, text: str) -> None:
        """Validate raw text string."""
        if not text:
            raise ValidationError(
                "Text input is empty",
                details={"hint": "Provide non-empty text content"},
            )

        if not text.strip():
            raise ValidationError(
                "Text input contains only whitespace",
                details={"hint": "Provide text with actual content"},
            )

    def _validate_file(self, file_path: Path) -> None:
        """Validate text file."""
        if not file_path.exists():
            raise ConfigurationError(
                f"Text file not found: {file_path}",
                details={"file": str(file_path)},
            )

        if not file_path.is_file():
            raise ConfigurationError(
                f"Not a file: {file_path}",
                details={"file": str(file_path)},
            )

        # Check if file is empty
        if file_path.stat().st_size == 0:
            raise ValidationError(
                f"Text file is empty: {file_path}",
                details={"file": str(file_path)},
            )

        # Try to read file to check encoding
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Check if content is only whitespace
            if not content.strip():
                raise ValidationError(
                    f"Text file contains only whitespace: {file_path}",
                    details={"file": str(file_path)},
                )

        except UnicodeDecodeError as e:
            raise ValidationError(
                f"Text file has invalid encoding: {file_path}",
                details={
                    "file": str(file_path),
                    "error": str(e),
                    "hint": "File must be UTF-8 encoded",
                },
            ) from e
        except Exception as e:
            raise ValidationError(
                f"Error reading text file: {file_path}",
                details={"file": str(file_path), "error": str(e)},
            ) from e


class URLValidator(InputValidator):
    """Validates URL inputs."""

    def __init__(self, timeout: int = 30, max_size_mb: int = 100) -> None:
        """
        Initialize URL validator.

        Args:
            timeout: Timeout for URL checks in seconds
            max_size_mb: Maximum allowed download size in MB
        """
        self.timeout = timeout
        self.max_size_mb = max_size_mb

    def validate(self, source: str) -> None:
        """
        Validate URL input.

        Checks:
        - Valid URL format
        - Supported scheme (http/https)
        - Safe IP address (blocks private, loopback, link-local, and cloud metadata endpoints)

        Note: Reachability and size checks are done during download
        to avoid duplicate requests.

        Args:
            source: URL string

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(source, str):
            raise ValidationError(
                "URL must be a string",
                details={"type": type(source).__name__},
            )

        # Parse URL
        try:
            parsed = urlparse(source)
        except Exception as e:
            raise ValidationError(
                f"Invalid URL format: {source}",
                details={"url": source, "error": str(e)},
            ) from e

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            raise ValidationError(
                "URL must use http or https scheme",
                details={
                    "url": source,
                    "scheme": parsed.scheme or "(none)",
                    "supported_schemes": ["http", "https"],
                },
            )

        # Check if URL has a netloc (domain)
        if not parsed.netloc:
            raise ValidationError(
                f"Invalid URL (missing domain): {source}",
                details={"url": source},
            )

        # Extract hostname (remove port if present)
        hostname = parsed.hostname
        if not hostname:
            raise ValidationError(
                f"Invalid URL (cannot extract hostname): {source}",
                details={"url": source},
            )

        # Perform DNS resolution and IP validation to prevent SSRF attacks
        try:
            # Resolve hostname to IP address
            ip_str = socket.gethostbyname(hostname)
            ip_addr = ipaddress.ip_address(ip_str)

            # Explicitly block cloud metadata endpoint (169.254.169.254) FIRST
            # This is critical for preventing access to cloud instance metadata
            if ip_str == "169.254.169.254":
                raise ValidationError(
                    "Access to cloud metadata endpoint is not allowed: 169.254.169.254",
                    details={
                        "url": source,
                        "hostname": hostname,
                        "resolved_ip": ip_str,
                        "reason": "Cloud metadata endpoint (AWS, Azure, GCP)",
                    },
                )

            # Block loopback addresses (127.0.0.0/8, ::1)
            if ip_addr.is_loopback:
                raise ValidationError(
                    f"Access to loopback addresses is not allowed: {ip_str}",
                    details={
                        "url": source,
                        "hostname": hostname,
                        "resolved_ip": ip_str,
                        "reason": "Loopback address (127.0.0.0/8, ::1)",
                    },
                )

            # Block link-local addresses (169.254.0.0/16, fe80::/10)
            if ip_addr.is_link_local:
                raise ValidationError(
                    f"Access to link-local addresses is not allowed: {ip_str}",
                    details={
                        "url": source,
                        "hostname": hostname,
                        "resolved_ip": ip_str,
                        "reason": "Link-local address (169.254.0.0/16, fe80::/10)",
                    },
                )

            # Block multicast addresses
            if ip_addr.is_multicast:
                raise ValidationError(
                    f"Access to multicast addresses is not allowed: {ip_str}",
                    details={
                        "url": source,
                        "hostname": hostname,
                        "resolved_ip": ip_str,
                        "reason": "Multicast address",
                    },
                )

            # Block reserved addresses
            if ip_addr.is_reserved:
                raise ValidationError(
                    f"Access to reserved IP addresses is not allowed: {ip_str}",
                    details={
                        "url": source,
                        "hostname": hostname,
                        "resolved_ip": ip_str,
                        "reason": "Reserved IP address",
                    },
                )

            # Block private IP addresses (RFC 1918)
            # Check this LAST because is_private also returns True for loopback and link-local
            if ip_addr.is_private:
                raise ValidationError(
                    f"Access to private IP addresses is not allowed: {ip_str}",
                    details={
                        "url": source,
                        "hostname": hostname,
                        "resolved_ip": ip_str,
                        "reason": "Private IP address (RFC 1918: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)",
                    },
                )

        except socket.gaierror as e:
            # DNS resolution failed
            raise ValidationError(
                f"Failed to resolve hostname: {hostname}",
                details={
                    "url": source,
                    "hostname": hostname,
                    "error": str(e),
                    "hint": "Hostname could not be resolved to an IP address",
                },
            ) from e
        except ValidationError:
            # Re-raise validation errors from IP checks
            raise
        except Exception as e:
            # Handle unexpected errors during IP validation
            raise ValidationError(
                f"Error validating URL safety: {source}",
                details={
                    "url": source,
                    "hostname": hostname,
                    "error": str(e),
                    "type": type(e).__name__,
                },
            ) from e


class DoclangValidator(InputValidator):
    """Validates DocLang inputs (.dclg / .dclg.xml / .dclx).

    Only cheap structural checks are done here (existence, non-empty, root tag /
    ZIP magic); the Docling DocLang backend performs full schema validation when
    the file is parsed.
    """

    _ZIP_MAGIC = b"PK\x03\x04"

    def validate(self, source: Union[str, Path]) -> None:
        if source is None:
            raise ValidationError(
                "DocLang source cannot be None",
                details={"hint": "Provide a path to a .dclg, .dclg.xml or .dclx file"},
            )

        path = Path(source)
        if not path.exists():
            raise ConfigurationError(
                f"DocLang file not found: {path}",
                details={"file": str(path)},
            )
        if not path.is_file():
            raise ConfigurationError(
                f"Not a file: {path}",
                details={"file": str(path)},
            )
        if path.stat().st_size == 0:
            raise ValidationError(
                f"DocLang file is empty: {path}",
                details={"file": str(path)},
            )

        name = path.name.lower()
        if name.endswith(".dclx"):
            self._validate_archive(path)
        else:
            self._validate_xml(path)

    def _validate_archive(self, path: Path) -> None:
        """A .dclx archive must be a ZIP (OPC container)."""
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
        except OSError as e:
            raise ValidationError(
                f"Error reading DocLang archive: {path}",
                details={"file": str(path), "error": str(e)},
            ) from e
        if magic != self._ZIP_MAGIC:
            raise ValidationError(
                f"DocLang archive is not a valid ZIP container: {path}",
                details={"file": str(path), "hint": ".dclx must be an OPC ZIP archive"},
            )

    def _validate_xml(self, path: Path) -> None:
        """A .dclg / .dclg.xml document must open a <doclang> root element."""
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                head = f.read(2048)
        except OSError as e:
            raise ValidationError(
                f"Error reading DocLang file: {path}",
                details={"file": str(path), "error": str(e)},
            ) from e
        if "<doclang" not in head:
            raise ValidationError(
                f"File does not appear to be DocLang (no <doclang> root): {path}",
                details={"file": str(path)},
            )


class DoclingDocumentValidator(InputValidator):
    """Validates DoclingDocument JSON files."""

    def validate(self, source: Union[str, Path]) -> None:
        """
        Validate DoclingDocument JSON.

        Checks:
        - Valid JSON format
        - Contains required DoclingDocument fields
        - Schema compatibility

        Args:
            source: Path to DoclingDocument JSON file or JSON string

        Raises:
            ValidationError: If validation fails
            ConfigurationError: If file not found
        """
        # Handle None input
        if source is None:
            raise ValidationError(
                "DoclingDocument source cannot be None",
                details={"hint": "Provide a file path or JSON string"},
            )

        # Handle both file paths and JSON strings
        if isinstance(source, Path):
            # It's a Path object - must be a file
            source_path = source
            if not source_path.exists():
                raise ConfigurationError(
                    f"DoclingDocument file not found: {source_path}",
                    details={"file": str(source_path)},
                )
            if not source_path.is_file():
                raise ConfigurationError(
                    f"Not a file: {source_path}",
                    details={"file": str(source_path)},
                )
            # Load JSON from file
            if source_path.stat().st_size == 0:
                raise ValidationError(
                    f"DoclingDocument file is empty: {source_path}",
                    details={"file": str(source_path)},
                )
            try:
                with open(source_path, encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValidationError(
                    "Invalid JSON in DoclingDocument file",
                    details={
                        "file": str(source_path),
                        "error": str(e),
                        "line": e.lineno if hasattr(e, "lineno") else None,
                    },
                ) from e
        elif isinstance(source, str):
            # For strings, check if it looks like a file path
            source_path = Path(source)

            # Try to check if it's a file, but handle cases where the string
            # is too long or invalid as a file path
            try:
                # If it exists as a file, load from file
                if source_path.exists() and source_path.is_file():
                    if source_path.stat().st_size == 0:
                        raise ValidationError(
                            f"DoclingDocument file is empty: {source_path}",
                            details={"file": str(source_path)},
                        )
                    try:
                        with open(source_path, encoding="utf-8") as f:
                            data = json.load(f)
                    except json.JSONDecodeError as e:
                        raise ValidationError(
                            "Invalid JSON in DoclingDocument file",
                            details={
                                "file": str(source_path),
                                "error": str(e),
                                "line": e.lineno if hasattr(e, "lineno") else None,
                            },
                        ) from e
                else:
                    # Treat as JSON string
                    try:
                        data = json.loads(source)
                    except json.JSONDecodeError as e:
                        raise ValidationError(
                            "Invalid JSON in DoclingDocument",
                            details={"error": str(e)},
                        ) from e
            except (OSError, ValueError):
                # If path checking fails (e.g., string too long), treat as JSON string
                try:
                    data = json.loads(source)
                except json.JSONDecodeError as e:
                    raise ValidationError(
                        "Invalid JSON in DoclingDocument",
                        details={"error": str(e)},
                    ) from e
        else:
            raise ValidationError(
                "DoclingDocument source must be a file path or JSON string",
                details={"type": type(source).__name__},
            )

        # Validate JSON structure
        if not isinstance(data, dict):
            raise ValidationError(
                "DoclingDocument must be a JSON object",
                details={"type": type(data).__name__},
            )

        # Check for required fields
        if "schema_name" not in data:
            raise ValidationError(
                "Missing required field: schema_name",
                details={"hint": "DoclingDocument must have 'schema_name' field"},
            )

        if data.get("schema_name") != "DoclingDocument":
            raise ValidationError(
                "schema_name must be 'DoclingDocument'",
                details={
                    "expected": "DoclingDocument",
                    "actual": data.get("schema_name"),
                },
            )

        if "version" not in data:
            raise ValidationError(
                "Missing required field: version",
                details={"hint": "DoclingDocument must have 'version' field"},
            )

        # Validate pages structure if present
        if "pages" in data:
            if not isinstance(data["pages"], dict):
                raise ValidationError(
                    "Invalid 'pages' field in DoclingDocument",
                    details={"error": "'pages' must be a dictionary"},
                )
