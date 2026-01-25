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

    def __init__(
        self, model: str, max_tokens: int | None = None, timeout: int | None = None, **kwargs: Any
    ) -> None:
        """
        Initialize LLM client.

        Args:
            model: Model identifier
            max_tokens: Maximum tokens to generate (overrides config)
            timeout: Request timeout in seconds (overrides config)
            **kwargs: Provider-specific parameters
        """
        self.model = model
        self._context_limit: int = 8192  # Default, will be overridden
        self._max_tokens: int | None = max_tokens  # User override
        self._timeout: int | None = timeout  # User override

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
    def _call_api(
        self, messages: list[Dict[str, str]], **params: Any
    ) -> tuple[str, Dict[str, Any]]:
        """
        Provider-specific API call.

        This method should call the provider's API and return both the raw
        response and metadata about the generation (e.g., finish_reason).

        Args:
            messages: List of message dicts with 'role' and 'content'
            **params: Additional parameters (e.g., schema_json)

        Returns:
            Tuple of (raw_response, metadata) where metadata contains:
            - finish_reason: Why generation stopped (if available)
            - usage: Token usage info (if available)
            - other provider-specific data

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
        - Truncation detection
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

        # Call provider API (returns response + metadata)
        raw_response, metadata = self._call_api(messages, schema_json=schema_json)

        # Check for truncation
        truncated = self._check_truncation(metadata)

        # Parse using shared handler with truncation awareness
        return ResponseHandler.parse_json_response(
            raw_response,
            self.__class__.__name__,
            aggressive_clean=self._needs_aggressive_cleaning(),
            truncated=truncated,
            max_tokens=self.max_tokens,
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

    def _check_truncation(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if response was truncated due to max_tokens limit.

        Args:
            metadata: Response metadata from _call_api()

        Returns:
            True if response was truncated, False otherwise
        """
        # Check for finish_reason (OpenAI-compatible APIs)
        finish_reason = metadata.get("finish_reason")
        if finish_reason:
            return bool(finish_reason == "length")

        # For providers without finish_reason, use heuristics
        # (Conservative: don't assume truncation without evidence)
        return False

    def _load_model_config(self) -> None:
        """Load model configuration from centralized registry."""
        try:
            from .config import get_model_config, get_provider_config

            provider_id = self._provider_id()
            config = get_model_config(provider_id, self.model)
            provider_config = get_provider_config(provider_id)

            if config:
                self._context_limit = config.context_limit
                if hasattr(config, "max_new_tokens"):
                    self._max_new_tokens = config.max_new_tokens

                # Load max_tokens (use user override if provided)
                if self._max_tokens is None:
                    if config.max_tokens is not None:
                        self._max_tokens = config.max_tokens
                    elif provider_config:
                        self._max_tokens = provider_config.default_max_tokens
                    else:
                        self._max_tokens = 8192

                # Load timeout (use user override if provided)
                if self._timeout is None:
                    if config.timeout is not None:
                        self._timeout = config.timeout
                    elif provider_config:
                        self._timeout = provider_config.timeout_seconds
                    else:
                        self._timeout = 300

                logger.info(
                    f"{self.__class__.__name__}: "
                    f"context={self._context_limit}, "
                    f"max_tokens={self._max_tokens}, "
                    f"timeout={self._timeout}s"
                )
            else:
                # Fallback defaults
                self._max_tokens = self._max_tokens or 8192
                self._timeout = self._timeout or 300
                logger.warning(
                    f"{self.__class__.__name__}: Model '{self.model}' not in config, using defaults"
                )
        except Exception as e:
            logger.warning(f"Failed to load model config: {e}")
            self._max_tokens = self._max_tokens or 8192
            self._timeout = self._timeout or 300

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
    def provider(self) -> str:
        """Return the provider identifier for this client."""
        return self._provider_id()

    @property
    def context_limit(self) -> int:
        """Return the context window size in tokens."""
        return self._context_limit

    @property
    def max_tokens(self) -> int:
        """Return the maximum tokens to generate."""
        return self._max_tokens if self._max_tokens is not None else 8192

    @property
    def timeout(self) -> int:
        """Return the request timeout in seconds."""
        return self._timeout if self._timeout is not None else 300
