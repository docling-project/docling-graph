"""
Input handling module for Docling Graph.

This module provides input type detection, validation, and normalization
for various input formats including PDFs, images, text files, URLs, and
pre-processed DoclingDocument JSON files.
"""

from .handlers import (
    DoclangInputHandler,
    DoclingDocumentHandler,
    DocumentInputHandler,
    InputHandler,
    TextInputHandler,
    URLInputHandler,
)
from .types import InputType, InputTypeDetector
from .validators import (
    DoclangValidator,
    DoclingDocumentValidator,
    InputValidator,
    TextValidator,
    URLValidator,
)

__all__ = [
    "DoclangInputHandler",
    "DoclangValidator",
    "DoclingDocumentHandler",
    "DoclingDocumentValidator",
    "DocumentInputHandler",
    # Handlers
    "InputHandler",
    # Types
    "InputType",
    "InputTypeDetector",
    # Validators
    "InputValidator",
    "TextInputHandler",
    "TextValidator",
    "URLInputHandler",
    "URLValidator",
]
