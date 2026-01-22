"""
Enhanced base class for all LLM clients with template method pattern.

This refactored base class eliminates code duplication by providing:
- Shared JSON response handling via ResponseHandler
- Common message preparation logic
- Unified error handling
- Consistent configuration loading

Each client only needs to implement provider-specific API calls.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..exceptions import ConfigurationError
from .response_handler import ResponseHandler

logger = logging.getLogger(__name__)


class BaseLlmClient(ABC):
    """
    Enhanced base class for all LLM clients.

    Uses template method pattern to eliminate duplication while allowing
    provider-specific customization where needed.

    Subclasses must implement:
    - _setup_client(): Provider-specific initialization
    - _call_api(): Provider-specific API call
    - _provider_id(): Return provider identifier for config lookup
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        """
        Initialize LLM client.

        Args:
            model: Model identifier
            **kwargs: Provider-specific parameters
        """
        self.model = model
        self._context_limit: int = 8192  # Default, will be overridden

        # Provider-specific setup
        self._setup_client(**kwargs)

        # Load model configuration
        self._load_model_config()

        logger.info(f"{self.__class__.__name__} initialized for model: {model}")

    @abstractmethod
    def _setup_client(self, **kwargs: Any) -> None:
        """
        Provider-specific client initialization.

        This method should:
        - Load API credentials
        - Initialize provider SDK
        - Validate configuration

        Args:
            **kwargs: Provider-specific parameters

        Raises:
            ConfigurationError: If setup fails
        """

    @abstractmethod
    def _call_api(self, messages: list[Dict[str, str]], **params: Any) -> str:
        """
        Provider-specific API call.

        This method should call the provider's API and return the raw
        response as a string. All JSON parsing and validation is handled
        by the base class.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **params: Additional parameters (e.g., schema_json)

        Returns:
            Raw response string from the API

        Raises:
            ClientError: If API call fails
        """

    @abstractmethod
    def _provider_id(self) -> str:
        """
        Return provider identifier for configuration lookup.

        Returns:
            Provider ID (e.g., "openai", "mistral", "watsonx")
        """

    def get_json_response(
        self, prompt: str | dict[str, str], schema_json: str
    ) -> Dict[str, Any] | list[Any]:
        """
        Execute LLM call and return parsed JSON response.

        This method is the same for all clients - it handles:
        - Message preparation
        - API call
        - Response parsing and validation

        Args:
            prompt: Either a string or dict with 'system' and 'user' keys
            schema_json: Pydantic schema as JSON string

        Returns:
            Parsed and validated JSON (dictionary or list)

        Raises:
            ClientError: If API call or parsing fails
        """
        # Prepare messages
        messages = self._prepare_messages(prompt)

        # Call provider API
        raw_response = self._call_api(messages, schema_json=schema_json)

        # Parse using shared handler
        return ResponseHandler.parse_json_response(
            raw_response,
            self.__class__.__name__,
            aggressive_clean=self._needs_aggressive_cleaning(),
        )

    def _prepare_messages(self, prompt: str | dict) -> list[Dict[str, str]]:
        """
        Convert prompt to standardized message format.

        Args:
            prompt: String or dict with 'system' and 'user' keys

        Returns:
            List of message dictionaries
        """
        if isinstance(prompt, dict):
            messages = []
            if prompt.get("system"):
                messages.append({"role": "system", "content": prompt["system"]})
            if "user" in prompt:
                messages.append({"role": "user", "content": prompt["user"]})
            return messages
        else:
            return [{"role": "user", "content": prompt}]

    def _needs_aggressive_cleaning(self) -> bool:
        """
        Override this to enable aggressive response cleaning.

        Returns:
            True if provider needs extra cleaning (e.g., WatsonX)
        """
        return False

    def _load_model_config(self) -> None:
        """Load model configuration from centralized registry."""
        try:
            from .config import get_model_config

            config = get_model_config(self._provider_id(), self.model)
            if config:
                self._context_limit = config.context_limit
                if hasattr(config, "max_new_tokens"):
                    self._max_new_tokens = config.max_new_tokens
                logger.info(
                    f"{self.__class__.__name__}: "
                    f"context={self._context_limit}, "
                    f"max_tokens={getattr(self, '_max_new_tokens', 'N/A')}"
                )
            else:
                logger.warning(
                    f"{self.__class__.__name__}: Model '{self.model}' not in config, using defaults"
                )
        except Exception as e:
            logger.warning(f"Failed to load model config: {e}")

    @staticmethod
    def _get_required_env(key: str) -> str:
        """
        Get required environment variable.

        Args:
            key: Environment variable name

        Returns:
            Environment variable value

        Raises:
            ConfigurationError: If variable is not set
        """
        value = os.getenv(key)
        if not value:
            raise ConfigurationError(
                f"Required environment variable not set: {key}", details={"variable": key}
            )
        return value

    @property
    def context_limit(self) -> int:
        """Return the context window size in tokens."""
        return self._context_limit
