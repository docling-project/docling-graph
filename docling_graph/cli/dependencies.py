"""
Utility for managing optional dependencies with helpful error messages.
"""

import importlib.util
from enum import Enum
from typing import Any, Dict, List, Optional


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
        extra: Optional[str] = None,
        description: Optional[str] = None,
        inference_type: Optional[str] = None,
    ) -> None:
        self.name = name
        self.package = package
        self.extra = extra or "all"
        self.description = description or f"{name} provider"
        self.inference_type = inference_type  # "local" or "remote"
        self._status: Optional[DependencyStatus] = None

    @property
    def is_installed(self) -> bool:
        """Check if the dependency is installed."""
        if self._status is None:
            self._status = self._check_status()
        return self._status == DependencyStatus.INSTALLED

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
    # Local providers
    "vllm": OptionalDependency(
        name="vllm",
        package="openai",
        extra="openai",
        description="vLLM for running local LLMs via OpenAI Local API",
        inference_type="local",
    ),
    "ollama": OptionalDependency(
        name="ollama",
        package="ollama",
        extra="ollama",
        description="Ollama for running local LLMs",
        inference_type="local",
    ),
    # Remote providers
    "openai": OptionalDependency(
        name="openai",
        package="openai",
        extra="openai",
        description="OpenAI API client",
        inference_type="remote",
    ),
    "mistral": OptionalDependency(
        name="mistral",
        package="mistralai",
        extra="mistral",
        description="Mistral AI API client",
        inference_type="remote",
    ),
    "gemini": OptionalDependency(
        name="gemini",
        package="google.genai",
        extra="gemini",
        description="Google Gemini API client",
        inference_type="remote",
    ),
}

# Mapping of inference types to their providers
INFERENCE_PROVIDERS: Dict[str, List[str]] = {
    "local": ["vllm", "ollama"],
    "remote": ["mistral", "openai", "gemini"],
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
    inference_type: str, selected_provider: Optional[str] = None
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
