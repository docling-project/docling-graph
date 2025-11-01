"""
Protocol definitions for type-safe interfaces.

This module defines Protocol classes that specify the expected interfaces
for backends, extractors, and clients. Using Protocols instead of abstract
base classes provides better type checking and duck typing support.
"""

from typing import Any, Dict, List, Optional, Protocol, Type, runtime_checkable

from pydantic import BaseModel

# =============================================================================
# Backend Protocols
# =============================================================================


@runtime_checkable
class ExtractionBackendProtocol(Protocol):
    """Protocol for extraction backends that process entire documents.

    This protocol is implemented by VLM backends that can process
    documents directly without requiring markdown conversion.
    """

    def extract_from_document(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        """Extract structured data from a document.

        Args:
            source: Path to the source document.
            template: Pydantic model template to extract into.

        Returns:
            List of extracted and validated Pydantic model instances.
        """
        ...

    def cleanup(self) -> None:
        """Clean up backend resources (GPU memory, connections, etc.)."""
        ...


@runtime_checkable
class TextExtractionBackendProtocol(Protocol):
    """Protocol for extraction backends that process markdown/text.

    This protocol is implemented by LLM backends that require
    pre-processed text input.
    """

    client: Any  # LLM client instance

    def extract_from_markdown(
        self, markdown: str, template: Type[BaseModel], context: str = "document"
    ) -> Optional[BaseModel]:
        """Extract structured data from markdown content.

        Args:
            markdown: Markdown content to extract from.
            template: Pydantic model template.
            context: Context description (e.g., "page 1", "full document").

        Returns:
            Extracted and validated model instance, or None if extraction failed.
        """
        ...

    def cleanup(self) -> None:
        """Clean up backend resources."""
        ...


# =============================================================================
# LLM Client Protocol
# =============================================================================


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM clients (Ollama, Mistral, OpenAI, etc.).

    All LLM client implementations must provide these methods.
    """

    @property
    def context_limit(self) -> int:
        """Return the effective context limit in tokens.

        This should be a conservative number, leaving room for prompts.
        """
        ...

    def get_json_response(self, prompt: str, schema_json: str) -> Dict[str, Any]:
        """Execute LLM call and return parsed JSON.

        Args:
            prompt: The full prompt to send to the model.
            schema_json: The Pydantic schema in JSON format.

        Returns:
            Parsed JSON dictionary from the LLM response.
        """
        ...


# =============================================================================
# Extractor Protocol
# =============================================================================


@runtime_checkable
class ExtractorProtocol(Protocol):
    """Protocol for extraction strategies.

    Extraction strategies (OneToOne, ManyToOne) must implement this protocol.
    """

    backend: Any  # Backend instance (VLM or LLM)

    def extract(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        """Extract structured data from a source document.

        Args:
            source: Path to the source document.
            template: Pydantic model template to extract into.

        Returns:
            List of extracted Pydantic model instances.
            - For OneToOne: May contain N models (one per page).
            - For ManyToOne: Contains 1 merged model.
        """
        ...


# =============================================================================
# Document Processor Protocol
# =============================================================================


@runtime_checkable
class DocumentProcessorProtocol(Protocol):
    """Protocol for document processing and conversion."""

    def convert_to_markdown(self, source: str) -> Any:
        """Convert document to Docling document object.

        Args:
            source: Path to source document.

        Returns:
            Docling document object.
        """
        ...

    def extract_full_markdown(self, document: Any) -> str:
        """Extract complete markdown from document.

        Args:
            document: Docling document object.

        Returns:
            Full markdown content as string.
        """
        ...

    def extract_page_markdowns(self, document: Any) -> List[str]:
        """Extract markdown for each page separately.

        Args:
            document: Docling document object.

        Returns:
            List of markdown strings, one per page.
        """
        ...


# =============================================================================
# Type Checking Utilities
# =============================================================================


def is_vlm_backend(backend: Any) -> bool:
    """Check if backend implements VLM extraction protocol.

    Args:
        backend: Backend instance to check.

    Returns:
        True if backend can extract from documents directly.
    """
    return isinstance(backend, ExtractionBackendProtocol)


def is_llm_backend(backend: Any) -> bool:
    """Check if backend implements LLM extraction protocol.

    Args:
        backend: Backend instance to check.

    Returns:
        True if backend requires markdown input.
    """
    return isinstance(backend, TextExtractionBackendProtocol)


def get_backend_type(backend: Any) -> str:
    """Get the backend type as a string.

    Args:
        backend: Backend instance.

    Returns:
        "vlm" if VLM backend, "llm" if LLM backend, "unknown" otherwise.
    """
    if is_vlm_backend(backend):
        return "vlm"
    elif is_llm_backend(backend):
        return "llm"
    else:
        return "unknown"


# =============================================================================
# Type Aliases for Clarity
# =============================================================================

# Backend can be either VLM or LLM
Backend = ExtractionBackendProtocol | TextExtractionBackendProtocol

# Extractor strategies
Extractor = ExtractorProtocol

# LLM client
LLMClient = LLMClientProtocol

# Document processor
DocumentProcessor = DocumentProcessorProtocol
