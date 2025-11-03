"""
Input validation functions for CLI commands.
"""

from typing import Literal, Optional, Tuple

import typer
from rich import print as rich_print

from ..deps import (
    INFERENCE_PROVIDERS,
    OPTIONAL_DEPS,
    get_missing_dependencies,
    get_missing_for_inference_type,
)
from .constants import (
    BACKENDS,
    DOCLING_PIPELINES,
    EXPORT_FORMATS,
    INFERENCE_LOCATIONS,
    PROCESSING_MODES,
)

# --- Configuration Validators ---


def validate_processing_mode(mode: str) -> str:
    """Validate processing mode.

    Args:
        mode: Processing mode to validate.

    Returns:
        Lowercase validated mode.

    Raises:
        typer.Exit: If mode is invalid.
    """
    mode = mode.lower()
    if mode not in PROCESSING_MODES:
        rich_print(f"[red]Error:[/red] Invalid processing mode '{mode}'.")
        rich_print(f"Must be one of: {', '.join(PROCESSING_MODES)}")
        raise typer.Exit(code=1)
    return mode


def validate_backend_type(backend: str) -> str:
    """Validate backend type.

    Args:
        backend: Backend type to validate.

    Returns:
        Lowercase validated backend.

    Raises:
        typer.Exit: If backend is invalid.
    """
    backend = backend.lower()
    if backend not in BACKENDS:
        rich_print(f"[red]Error:[/red] Invalid backend type '{backend}'.")
        rich_print(f"Must be one of: {', '.join(BACKENDS)}")
        raise typer.Exit(code=1)
    return backend


def validate_inference(inference: str) -> str:
    """Validate inference location.

    Args:
        inference: Inference location to validate.

    Returns:
        Lowercase validated inference.

    Raises:
        typer.Exit: If inference is invalid.
    """
    inference = inference.lower()
    if inference not in INFERENCE_LOCATIONS:
        rich_print(f"[red]Error:[/red] Invalid inference location '{inference}'.")
        rich_print(f"Must be one of: {', '.join(INFERENCE_LOCATIONS)}")
        raise typer.Exit(code=1)
    return inference


def validate_docling_config(config: str) -> str:
    """Validate Docling pipeline configuration.

    Args:
        config: Docling config to validate.

    Returns:
        Lowercase validated config.

    Raises:
        typer.Exit: If config is invalid.
    """
    config = config.lower()
    if config not in DOCLING_PIPELINES:
        rich_print(f"[red]Error:[/red] Invalid docling config '{config}'.")
        rich_print(f"Must be one of: {', '.join(DOCLING_PIPELINES)}")
        raise typer.Exit(code=1)
    return config


def validate_export_format(export_format: str) -> str:
    """Validate export format.

    Args:
        export_format: Export format to validate.

    Returns:
        Lowercase validated format.

    Raises:
        typer.Exit: If format is invalid.
    """
    export_format = export_format.lower()
    if export_format not in EXPORT_FORMATS:
        rich_print(f"[red]Error:[/red] Invalid export format '{export_format}'.")
        rich_print(f"Must be one of: {', '.join(EXPORT_FORMATS)}")
        raise typer.Exit(code=1)
    return export_format


def validate_vlm_constraints(backend: str, inference: str) -> None:
    """Validate VLM-specific constraints.

    Args:
        backend: Backend type.
        inference: Inference location.

    Raises:
        typer.Exit: If VLM constraints are violated.
    """
    if backend == "vlm" and inference == "remote":
        rich_print(
            "[red]Error:[/red] VLM (Vision-Language Model) is currently only supported with local inference."
        )
        rich_print("Please use '--inference local' or switch to '--backend llm' for API inference.")
        raise typer.Exit(code=1)


def validate_provider(provider: str, inference: str) -> str:
    """Validate provider choice."""
    from .constants import API_PROVIDERS, LOCAL_PROVIDERS

    valid_providers = LOCAL_PROVIDERS if inference == "local" else API_PROVIDERS

    if provider not in valid_providers:
        raise ValueError(
            f"Invalid provider '{provider}' for inference='{inference}'. "
            f"Valid options: {', '.join(valid_providers)}"
        )
    return provider


# --- Dependencies Validators ---


def check_provider_installed(provider: str) -> bool:
    """Check if a provider's package is installed."""
    dep = OPTIONAL_DEPS.get(provider)
    if not dep:
        return True  # Unknown provider
    return dep.is_installed


def validate_config_dependencies(config_dict: dict) -> Tuple[bool, str]:
    """
    Validate that required dependencies for the config are available.

    Args:
        config_dict: The configuration dictionary

    Returns:
        Tuple of (is_valid, inference_type) where:
        - is_valid: True if all dependencies are available
        - inference_type: The inference type from config ("local" or "remote")
    """
    # Extract inference type from config
    defaults = config_dict.get("defaults", {})
    inference_type: str = defaults.get("inference", "remote")

    # Extract provider from config
    models = config_dict.get("models", {})

    if inference_type == "local":
        llm_config = models.get("llm", {})
        local_config = llm_config.get("local", {})
        provider = local_config.get("provider")
    else:  # remote
        llm_config = models.get("llm", {})
        remote_config = llm_config.get("remote", {})
        provider = remote_config.get("provider")

    # Check if provider is installed
    if provider and not check_provider_installed(provider):
        return False, inference_type

    return True, inference_type


def validate_and_warn_dependencies(config_dict: dict, interactive: bool = True) -> bool:
    """
    Validate dependencies and show helpful warnings if missing.

    Shows warnings but doesn't block configuration creation.

    Args:
        config_dict: The configuration dictionary
        interactive: If True, show interactive prompts to install

    Returns:
        True if all dependencies are available, False if some are missing
    """
    is_valid, inference_type = validate_config_dependencies(config_dict)

    if not is_valid:
        missing = get_missing_for_inference_type(inference_type)

        rich_print("\n[yellow]Warning: Required dependencies not installed[/yellow]")
        rich_print(f"\nYour configuration uses [bold]{inference_type}[/bold] inference.")
        rich_print("\nThe following provider dependencies are missing:")

        for dep in missing:
            rich_print(f"  • [bold]{dep.name}[/bold] - {dep.description}")

        rich_print("\n[blue]Install them with:[/blue]")

        if inference_type == "local":
            rich_print("  pip install 'docling-graph[local]'")
            rich_print("\n  Or install specific providers:")
            for dep in missing:
                rich_print(f"  {dep.get_install_command()}")
        else:  # remote
            rich_print("  pip install 'docling-graph[remote]'")
            rich_print("\n  Or install specific providers:")
            for dep in missing:
                rich_print(f"  {dep.get_install_command()}")

        rich_print()
        return False

    return True


def print_dependency_setup_guide(inference_type: str) -> None:
    """
    Print setup guide for the selected inference type.

    Args:
        inference_type: Either "local" or "remote"
    """
    providers = INFERENCE_PROVIDERS.get(inference_type, [])
    missing = get_missing_dependencies(providers)

    if not missing:
        rich_print(f"\n[green]All {inference_type} inference dependencies are installed![/green]")
        return

    rich_print(f"\n[yellow]Setup required for {inference_type} inference[/yellow]")
    rich_print(f"\nYou selected [bold]{inference_type}[/bold] inference.")
    rich_print("\n[blue]Available providers and their dependencies:[/blue]")

    for provider in providers:
        dep = OPTIONAL_DEPS.get(provider)
        if dep:
            status = "[green]+[/green]" if dep.is_installed else "[red]-[/red]"
            rich_print(f"  {status} {dep.description}")

    if missing:
        rich_print("\n[blue]Run the following comamnd to install missing dependencies:[/blue]")

        if inference_type == "local":
            rich_print("  uv sync --extra local")
        else:  # remote
            rich_print("  uv sync --extra remote")

    rich_print()


def print_next_steps_with_deps(config_dict: dict) -> None:
    """
    Print setup guide for the configured inference type.

    Args:
        config_dict: The configuration dictionary
    """
    defaults = config_dict.get("defaults", {})
    inference_type = defaults.get("inference", "remote")
    print_dependency_setup_guide(inference_type)
