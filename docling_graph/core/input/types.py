"""
Input type detection and classification.

This module provides the InputType enum and InputTypeDetector class
for identifying and classifying different input formats.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Literal, Union

from ...exceptions import ConfigurationError


class InputType(Enum):
    """Supported input types for the pipeline."""

    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"  # Kept for backward compatibility; detection uses DOCUMENT
    TEXT_FILE = "text_file"
    MARKDOWN = "markdown"
    URL = "url"
    DOCLING_DOCUMENT = "docling_document"
    # Catch-all: any file or raw text sent to Docling for conversion (validated by Docling)
    DOCUMENT = "document"


class InputTypeDetector:
    """Detects input type from source with mode awareness."""

    # File extension mappings
    PDF_EXTENSIONS = {".pdf"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"}
    TEXT_EXTENSIONS = {".txt"}
    MARKDOWN_EXTENSIONS = {".md", ".markdown"}
    JSON_EXTENSIONS = {".json"}

    @classmethod
    def detect(cls, source: Union[str, Path], mode: Literal["cli", "api"] = "api") -> InputType:
        """
        Detect input type from source with mode awareness.

        Args:
            source: Input source (file path, URL, or text for API mode)
            mode: Execution mode - "cli" or "api"
                  CLI mode: Only accepts file paths and URLs
                  API mode: Also accepts raw text strings

        Returns:
            InputType enum value

        Raises:
            ConfigurationError: If input type cannot be determined or is invalid for mode
            ValueError: If mode is invalid
        """
        # Validate mode parameter
        if mode not in ("cli", "api"):
            raise ValueError(f"mode must be 'cli' or 'api', got: {mode}")

        source_str = str(source)

        # Check if it's a URL
        if cls._is_url(source_str):
            return InputType.URL

        # In API mode, empty strings or whitespace-only are invalid (handler will reject)
        if mode == "api" and not source_str.strip():
            return InputType.DOCUMENT

        # Try to interpret as a file path
        source_path = Path(source_str)

        # CLI mode: must be an existing file
        if mode == "cli":
            if not source_path.exists():
                raise ConfigurationError(
                    f"File not found: {source_str}",
                    details={
                        "source": source_str,
                        "mode": mode,
                        "hint": "Provide a valid file path or URL. Plain text input is only supported via Python API.",
                    },
                )
            if not source_path.is_file():
                raise ConfigurationError(
                    f"Not a file: {source_str}",
                    details={"source": source_str, "mode": mode},
                )
            return cls._detect_from_file(source_path)

        # API mode: file exists -> detect from file; else treat as raw text (DOCUMENT, normalized to .md for Docling)
        try:
            if source_path.exists():
                if source_path.is_file():
                    return cls._detect_from_file(source_path)
                else:
                    # Path exists but is not a file (e.g., directory) -> DOCUMENT
                    return InputType.DOCUMENT
            else:
                # Not a file and not a URL -> raw text, sent to Docling after normalizing to .md
                return InputType.DOCUMENT
        except (OSError, ValueError):
            return InputType.DOCUMENT

    @classmethod
    def _is_url(cls, source: str) -> bool:
        """
        Check if source is a URL.

        Args:
            source: Source string to check

        Returns:
            True if source is a URL, False otherwise
        """
        return source.startswith(("http://", "https://"))

    @classmethod
    def _detect_from_file(cls, file_path: Path) -> InputType:
        """
        Detect input type from file extension and content.

        Only DoclingDocument JSON is special (skip conversion). All other files
        are DOCUMENT and are sent to Docling for conversion; Docling validates
        supported formats.
        """
        extension = file_path.suffix.lower()

        # JSON: peek inside to see if it's a DoclingDocument (skip conversion)
        if extension in cls.JSON_EXTENSIONS:
            return cls._detect_json_type(file_path)

        # Everything else (PDF, images, .txt, .md, Office, HTML, etc.) -> Docling
        return InputType.DOCUMENT

    @classmethod
    def _detect_json_type(cls, file_path: Path) -> InputType:
        """
        Detect if JSON file is a DoclingDocument.

        Args:
            file_path: Path to JSON file

        Returns:
            InputType.DOCLING_DOCUMENT if it's a DoclingDocument, InputType.TEXT otherwise
        """
        if cls._is_docling_document(file_path):
            return InputType.DOCLING_DOCUMENT

        # Not a DoclingDocument - send to Docling like any other file
        return InputType.DOCUMENT

    @classmethod
    def _is_docling_document(cls, file_path: Path) -> bool:
        """
        Check if a file is a DoclingDocument.

        Args:
            file_path: Path to file to check

        Returns:
            True if file is a DoclingDocument, False otherwise
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Check for DoclingDocument markers
            if isinstance(data, dict):
                # Check for common DoclingDocument fields
                has_schema = "schema_name" in data or "version" in data
                has_pages = "pages" in data
                has_main_text = "main_text" in data

                if has_schema or (has_pages and has_main_text):
                    return True

            return False

        except (json.JSONDecodeError, Exception):
            return False
