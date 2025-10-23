from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Type, List

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
            A list of Pydantic model instances.
            - For "One-to-One", this list may contain N models.
            - For "Many-to-One", this list will contain 1 model.
        """
        pass

