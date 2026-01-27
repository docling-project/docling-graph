"""
Utility for managing optional dependencies with helpful error messages.
"""

import importlib.util
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional

# Module-level cache for dependency checks
_DEPENDENCY_CACHE: Dict[str, bool] = {}
_CACHE_ENABLED = True


def clear_dependency_cache() -> None:
    """Clear the dependency cache. Useful for testing."""
    global _DEPENDENCY_CACHE
    _DEPENDENCY_CACHE.clear()


def disable_dependency_cache() -> None:
    """Disable dependency caching. Useful for testing."""
    global _CACHE_ENABLED
    _CACHE_ENABLED = False


def enable_dependency_cache() -> None:
    """Enable dependency caching (default behavior)."""
    global _CACHE_ENABLED
    _CACHE_ENABLED = True


class DependencyStatus(Enum):
    """Status of an optional dependency."""

    INSTALLED = "installed"
    NOT_INSTALLED = "not_installed"
    UNKNOWN = "unknown"


class OptionalDependency:
    """Represents an optional dependency with metadata."""

    def __init__(
        self,
        name: str,
        package: str,
        extra: str | None = None,
        description: str | None = None,
        inference_type: str | None = None,
    ) -> None:
        self.name = name
        self.package = package
        self.extra = extra or "all"
        self.description = description or f"{name} provider"
        self.inference_type = inference_type  # "local" or "remote"
        self._status: DependencyStatus | None = None

    @property
    def is_installed(self) -> bool:
        """Check if the dependency is installed (with caching)."""
        # If caching is disabled, always check fresh
        if not _CACHE_ENABLED:
            return self._check_status() == DependencyStatus.INSTALLED

        # Check cache first
        if self.package in _DEPENDENCY_CACHE:
            return _DEPENDENCY_CACHE[self.package]

        # Check and cache result
        is_installed = self._check_status() == DependencyStatus.INSTALLED
        _DEPENDENCY_CACHE[self.package] = is_installed
        return is_installed

    def _check_status(self) -> DependencyStatus:
        """Check installation status of the package."""
        try:
            spec = importlib.util.find_spec(self.package)
            return DependencyStatus.INSTALLED if spec else DependencyStatus.NOT_INSTALLED
        except (ImportError, ModuleNotFoundError, ValueError):
            return DependencyStatus.NOT_INSTALLED
        except Exception:
            return DependencyStatus.UNKNOWN

    def get_install_command(self) -> str:
        """Get the pip install command for this dependency."""
        return f"pip install 'docling-graph[{self.extra}]'"

    def get_direct_install_command(self) -> str:
        """Get the direct pip install command for this dependency."""
        return f"pip install {self.package}"

    def __repr__(self) -> str:
        return f"OptionalDependency({self.name}, installed={self.is_installed})"


# Registry of all optional dependencies
OPTIONAL_DEPS: Dict[str, OptionalDependency] = {
    # Local inference providers
    "ollama": OptionalDependency(
        name="ollama",
        package="ollama",
        extra="ollama",
        description="Ollama local inference provider",
        inference_type="local",
    ),
    "vllm": OptionalDependency(
        name="vllm",
        package="vllm",
        extra="vllm",
        description="vLLM local inference provider",
        inference_type="local",
    ),
    # Remote/API providers
    "mistral": OptionalDependency(
        name="mistral",
        package="mistralai",
        extra="mistral",
        description="Mistral API provider",
        inference_type="remote",
    ),
    "openai": OptionalDependency(
        name="openai",
        package="openai",
        extra="openai",
        description="OpenAI API provider",
        inference_type="remote",
    ),
    "gemini": OptionalDependency(
        name="gemini",
        package="google-generativeai",
        extra="gemini",
        description="Gemini API provider",
        inference_type="remote",
    ),
    "ibm-watsonx-ai": OptionalDependency(
        name="ibm-watsonx-ai",
        package="ibm-watsonx-ai",
        extra="ibm-watsonx-ai",
        description="IBM watsonx.ai API provider",
        inference_type="remote",
    ),
    # Shared LiteLLM client library used by multiple providers
    "litellm": OptionalDependency(
        name="litellm",
        package="litellm",
        description="LiteLLM client library",
        inference_type="remote",
    ),
}

# Mapping of inference types to their providers
INFERENCE_PROVIDERS: Dict[str, List[str]] = {
    "local": ["ollama", "vllm"],
    "remote": ["mistral", "openai", "gemini", "ibm-watsonx-ai"],
}


def check_dependency(name: str) -> bool:
    """
    Check if an optional dependency is installed.

    Args:
        name: Name of the dependency (e.g., 'vllm', 'ollama')

    Returns:
        True if installed, False otherwise
    """
    dep = OPTIONAL_DEPS.get(name)
    if not dep:
        return True  # Unknown dependency, assume it's fine
    return dep.is_installed


def require_dependency(name: str) -> None:
    """
    Raise an error if a required dependency is not installed.

    Args:
        name: Name of the dependency

    Raises:
        ImportError: If the dependency is not installed
    """
    dep = OPTIONAL_DEPS.get(name)
    if not dep:
        return  # Unknown dependency

    if not dep.is_installed:
        raise ImportError(
            f"\n{dep.description} requires '{dep.package}' package.\n"
            f"Install it with: {dep.get_install_command()}\n"
            f"Or directly: {dep.get_direct_install_command()}"
        )


def get_missing_dependencies(provider_names: List[str]) -> List[OptionalDependency]:
    """
    Get a list of missing dependencies for the given providers.

    Args:
        provider_names: List of provider names to check

    Returns:
        List of missing OptionalDependency objects
    """
    missing = []
    for name in provider_names:
        dep = OPTIONAL_DEPS.get(name)
        if dep and not dep.is_installed:
            missing.append(dep)
    return missing


def get_missing_for_inference_type(inference_type: str) -> List[OptionalDependency]:
    """
    Get all missing dependencies for a specific inference type.

    Args:
        inference_type: Either "local" or "remote"

    Returns:
        List of missing OptionalDependency objects for that inference type
    """
    providers = INFERENCE_PROVIDERS.get(inference_type, [])
    return get_missing_dependencies(providers)


def check_inference_type_available(
    inference_type: str, selected_provider: str | None = None
) -> bool:
    """
    Check if an inference type has required dependencies installed.

    Args:
        inference_type: Either "local" or "remote"
        selected_provider: Specific provider to check, or None to check all for that type

    Returns:
        True if dependencies are available, False otherwise
    """
    if selected_provider:
        return check_dependency(selected_provider)

    providers = INFERENCE_PROVIDERS.get(inference_type, [])
    return all(check_dependency(p) for p in providers)


def get_all_missing_dependencies() -> Dict[str, List[OptionalDependency]]:
    """
    Get all missing dependencies grouped by inference type.

    Returns:
        Dictionary with "local" and "remote" keys mapping to lists of missing deps
    """
    return {
        "local": get_missing_for_inference_type("local"),
        "remote": get_missing_for_inference_type("remote"),
    }
