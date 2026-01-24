"""
vLLM (local LLM) client implementation.
Uses OpenAI-compatible API server from vLLM.
Cross-platform (Linux/Windows) via vLLM server mode.
"""

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Dict, NoReturn

from ..exceptions import ClientError, ConfigurationError
from .base import BaseLlmClient

logger = logging.getLogger(__name__)

_OpenAI: Any | None = None
try:
    from openai import OpenAI as OpenAI_module

    _OpenAI = OpenAI_module
except ImportError:
    logger.warning(
        "openai package not found. Please run `pip install openai` to use the vLLM client."
    )
    _OpenAI = None

OpenAI: Any = _OpenAI

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


class VllmClient(BaseLlmClient):
    """vLLM client implementation using OpenAI-compatible API."""

    def _provider_id(self) -> str:
        """Return provider ID for config lookup."""
        return "vllm"

    def _setup_client(self, **kwargs: Any) -> None:
        """Initialize vLLM-specific client."""
        if OpenAI is None:
            raise ConfigurationError(
                "openai package not installed",
                details={"package": "openai", "install": "pip install openai"},
            )

        self.base_url = kwargs.get("base_url", "http://localhost:8000/v1")
        self.api_key = kwargs.get("api_key", "EMPTY")

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        try:
            logger.info(f"Connecting to vLLM server at: {self.base_url}")
            self.client.models.list()
            logger.info("vLLM client connected successfully")
            logger.info(f"Using model: {self.model}")
        except Exception as e:
            raise ConfigurationError(
                f"vLLM connection failed: {e}",
                details={
                    "base_url": self.base_url,
                    "model": self.model,
                    "error": str(e),
                    "instructions": [
                        "1. Start vLLM server in a separate terminal:",
                        f"   vllm serve {self.model}",
                        "2. Wait for server to load (may take 1-2 minutes)",
                        f"3. Ensure server is accessible at: {self.base_url}",
                        "",
                        "On Windows: Run vLLM server in WSL2 or Docker",
                    ],
                },
            ) from e

    def _call_api(self, messages: list[Dict[str, str]], **params: Any) -> str:
        """
        Call vLLM API via OpenAI-compatible interface.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **params: Additional parameters (schema_json, etc.)

        Returns:
            Raw response string from vLLM

        Raises:
            ClientError: If API call fails or times out
        """
        import signal
        from contextlib import contextmanager

        @contextmanager
        def timeout_handler(seconds: int) -> Generator[None, None, None]:
            """Context manager for request timeout."""

            def _timeout_handler(signum: int, frame: Any) -> NoReturn:
                raise TimeoutError(f"vLLM request exceeded {seconds}s timeout")

            # Set alarm (Unix only, will be no-op on Windows)
            try:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            except AttributeError:
                # Windows doesn't have SIGALRM, just yield without timeout
                logger.warning("Timeout not supported on this platform (Windows)")
                yield

        try:
            # Get max_tokens and timeout from instance (configured via base class)
            max_tokens = getattr(self, "_max_tokens", 8192)
            timeout_seconds = getattr(self, "_timeout", 300)

            logger.info(f"vLLM request: max_tokens={max_tokens}, timeout={timeout_seconds}s")

            with timeout_handler(timeout_seconds):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                )

            raw_json = response.choices[0].message.content

            if not raw_json:
                raise ClientError(
                    "vLLM returned empty content",
                    details={"model": self.model, "base_url": self.base_url},
                )

            return str(raw_json)

        except TimeoutError as e:
            raise ClientError(
                f"vLLM request timeout after {getattr(self, '_timeout', 300)}s",
                details={
                    "model": self.model,
                    "base_url": self.base_url,
                    "timeout": getattr(self, "_timeout", 300),
                    "max_tokens": getattr(self, "_max_tokens", 8192),
                },
            ) from e
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(
                f"vLLM API call failed: {type(e).__name__}",
                details={"model": self.model, "base_url": self.base_url, "error": str(e)},
            ) from e
