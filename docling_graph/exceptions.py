"""
Unified exception hierarchy for docling-graph.

This module provides a consistent exception structure across the entire codebase,
replacing the previous mixed patterns of returning None, empty dicts, or raising
generic exceptions.
"""

from typing import Any


class DoclingGraphError(Exception):
    """
    Base exception for all docling-graph errors.

    Provides structured error information including:
    - Clear error message
    - Optional details dictionary for debugging
    - Optional cause exception for error chaining
    """

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ) -> None:
        """
        Initialize exception with structured information.

        Args:
            message: Human-readable error description
            details: Optional dictionary with additional context
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        """Format exception with all available information."""
        parts = [self.message]

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")

        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {self.cause}")

        return "\n".join(parts)

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"details={self.details!r}, "
            f"cause={self.cause!r})"
        )


class ConfigurationError(DoclingGraphError):
    """
    Raised when configuration is invalid or missing.

    Examples:
        - Missing required environment variables
        - Invalid configuration file
        - Unsupported model or provider
        - Missing required parameters
    """


class ClientError(DoclingGraphError):
    """
    Raised when LLM client operation fails.

    Examples:
        - API authentication failure
        - Network timeout
        - Invalid API response
        - Rate limit exceeded
        - Model not available
    """


class ExtractionError(DoclingGraphError):
    """
    Raised when document extraction fails.

    Examples:
        - Document parsing failure
        - Empty extraction result
        - Invalid document format
        - Extraction timeout
    """


class ValidationError(DoclingGraphError):
    """
    Raised when data validation fails.

    Examples:
        - Pydantic validation error
        - Schema mismatch
        - Invalid data structure
        - Missing required fields
    """


class GraphError(DoclingGraphError):
    """
    Raised when graph operation fails.

    Examples:
        - Invalid graph structure
        - Node/edge creation failure
        - Graph validation error
        - Export failure
    """


class PipelineError(DoclingGraphError):
    """
    Raised when pipeline execution fails.

    Examples:
        - Stage execution failure
        - Resource initialization error
        - Cleanup failure
    """

