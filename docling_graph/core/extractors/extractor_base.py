"""
Base extractor interface for all extraction strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Type

from pydantic import BaseModel


class BaseExtractor(ABC):
    """Abstract base class for all extraction strategies."""

    @abstractmethod
    def extract(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        """
        Extracts structured data from a source document based on a Pydantic template.

        Args:
            source (str): The file path to the document.
            template (Type[BaseModel]): The Pydantic model to extract into.

        Returns:
            List[BaseModel]: A list of Pydantic model instances.
                - For "One-to-One", this list may contain N models (one per page).
                - For "Many-to-One", this list will contain 1 model.
        """
