"""
Base extractor interface for all extraction strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Type

from docling_core.types.doc import DoclingDocument
from pydantic import BaseModel


class BaseExtractor(ABC):
    """Abstract base class for all extraction strategies."""
    
    def __init__(self) -> None:
        """Initialize base extractor with optional trace data."""
        self.trace_data: Any = None  # Will be set by pipeline if trace is enabled

    @abstractmethod
    def extract(
        self, source: str, template: Type[BaseModel]
    ) -> Tuple[List[BaseModel], DoclingDocument | None]:
        """
        Extracts structured data from a source document based on a Pydantic template.

        Args:
            source (str): The file path to the document.
            template (Type[BaseModel]): The Pydantic model to extract into.

        Returns:
            Tuple[List[BaseModel], Optional[DoclingDocument]]: A tuple containing:
                - List of Pydantic model instances:
                    - For "One-to-One": Contains N models (one per page).
                    - For "Many-to-One":
                        * Success: Contains 1 consolidated model
                        * Partial failure: May contain multiple models (batch/page results)
                          when merge/consolidation fails (zero data loss strategy)
                        * Complete failure: Empty list
                - The DoclingDocument object used during extraction (or None if extraction failed).

        Note:
            The Many-to-One strategy implements a zero data loss policy. When consolidation
            or merging fails, it returns all successfully extracted partial models rather
            than discarding data. Consumers should handle cases where len(models) > 1 for
            Many-to-One extractions by either:
                1. Using the first model as the primary result
                2. Implementing custom merge logic
                3. Processing all partial models independently
        """
