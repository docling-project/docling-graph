"""
Input validation functions for CLI commands.
"""

from typing import Callable, Optional, Tuple

import typer
from rich import print as rich_print

from .constants import (
    API_PROVIDERS,
    BACKENDS,
    DOCLING_PIPELINES,
    EXTRACTION_CONTRACTS,
    EXPORT_FORMATS,
    INFERENCE_LOCATIONS,
    LOCAL_PROVIDERS,
    PROCESSING_MODES,
)
from .dependencies import INFERENCE_PROVIDERS, OPTIONAL_DEPS, get_missing_dependencies

# Validation option sets mapped to names
VALIDATION_SETS = {
    "processing_mode": (PROCESSING_MODES, "processing mode"),
    "backend": (BACKENDS, "backend type"),
    "inference": (INFERENCE_LOCATIONS, "inference location"),
    "docling_config": (DOCLING_PIPELINES, "docling config"),
    "extraction_contract": (EXTRACTION_CONTRACTS, "extraction contract"),
    "export_format": (EXPORT_FORMATS, "export format"),
}


def validate_option(value: str, valid_options: set[str], option_name: str) -> str:
    """Generic validator for CLI options."""
    value = value.lower()
    if value not in valid_options:
        rich_print(f"[red]Error:[/red] Invalid {option_name} '{value}'.")
        rich_print(f"Must be one of: {', '.join(sorted(valid_options))}")
        raise typer.Exit(code=1)
    return value


# Generate validators dynamically
def _make_validator(key: str) -> Callable[[str], str]:
    """Factory for creating validators."""
    options, name = VALIDATION_SETS[key]
    return lambda value: validate_option(value, set(options), name)


# Create validators
validate_processing_mode = _make_validator("processing_mode")
validate_backend_type = _make_validator("backend")
validate_inference = _make_validator("inference")
validate_docling_config = _make_validator("docling_config")
validate_extraction_contract = _make_validator("extraction_contract")
validate_export_format = _make_validator("export_format")


def validate_vlm_constraints(backend: str, inference: str) -> None:
    """Validate VLM-specific constraints."""
    if backend == "vlm" and inference == "remote":
        rich_print("[red]Error:[/red] VLM (Vision-Language Model) only supports local inference.")
        rich_print("Please use '--inference local' or switch to '--backend llm'.")
        raise typer.Exit(code=1)


def validate_provider(provider: str, inference: str) -> str:
    """Validate provider choice.

    Ensures that the provider is valid for the selected inference type.
    Raises ValueError for invalid combinations so that CLI commands and tests
    can fail fast with clear feedback.
    """
    valid_providers = LOCAL_PROVIDERS if inference == "local" else API_PROVIDERS
    if provider not in valid_providers:
        raise ValueError(
            f"Invalid provider '{provider}' for inference='{inference}'. "
            f"Expected one of: {', '.join(sorted(valid_providers))}."
        )
    return provider


def check_provider_installed(provider: str) -> bool:
    """Check if a provider's package is installed."""
    dep = OPTIONAL_DEPS.get("litellm")
    return dep.is_installed if dep else True


def get_provider_from_config(config_dict: dict) -> Tuple[str, str]:
    """Extract provider and inference type from config."""
    defaults = config_dict.get("defaults", {})
    inference_type = defaults.get("inference", "remote")
    models = config_dict.get("models", {})
    llm_config = models.get("llm", {})

    inference_key = "local" if inference_type == "local" else "remote"
    config = llm_config.get(inference_key, {})
    provider = config.get("provider", "")

    return provider, inference_type


def validate_config_dependencies(config_dict: dict) -> Tuple[bool, str]:
    """Validate that required dependencies are available."""
    provider, inference_type = get_provider_from_config(config_dict)

    if provider and not check_provider_installed(provider):
        return False, inference_type

    return True, inference_type


def _print_provider_status(provider: str) -> None:
    """Print dependency status for a provider."""
    dep = OPTIONAL_DEPS.get(provider)
    if not dep:
        rich_print(f"[yellow]Warning:[/yellow] Unknown provider '{provider}'")
        return

    status = "[green]+[/green]" if dep.is_installed else "[red]-[/red]"
    rich_print(f" {status} {dep.description}")


def _get_install_command(provider: str) -> str:
    """Get the installation command for a provider."""
    return "uv sync"


def print_dependency_setup_guide(inference_type: str, provider: str | None = None) -> None:
    """Print setup guide for the selected inference type."""
    # Determine which providers to check
    if provider is not None:
        provider_names = [provider]
    else:
        provider_names = INFERENCE_PROVIDERS.get(inference_type, [])

    missing = get_missing_dependencies(provider_names)

    if not missing:
        msg = (
            f"The {provider} provider"
            if provider
            else f"All {inference_type} inference dependencies"
        )
        rich_print(f"\n[green]{msg} are installed![/green]")
        return

    # Show missing dependencies
    rich_print(f"\n[yellow]Note: Setup required for {inference_type} inference[/yellow]")
    rich_print("\n[blue]Dependency status:[/blue]")
    for dep in missing:
        _print_provider_status(dep.name)

    rich_print("\n[blue]Run the following to install missing dependencies:[/blue]")
    for dep in missing:
        rich_print(f" {dep.get_install_command()}")
    rich_print()


def validate_and_warn_dependencies(config_dict: dict, interactive: bool = True) -> bool:
    """Validate dependencies and show warnings if missing (optimized)."""
    provider, inference_type = get_provider_from_config(config_dict)

    # If no provider is configured, assume dependencies are fine
    if not provider:
        return True

    is_available = check_provider_installed(provider)

    if is_available:
        if interactive:
            rich_print("\n[green]- Dependencies are installed[/green]")
        return True

    if interactive:
        print_dependency_setup_guide(inference_type, provider)
    return False


def print_next_steps_with_deps(config_dict: dict, existing_steps: str) -> None:
    """Print next steps with dependency installation as first step if needed."""
    _provider, _inference_type = get_provider_from_config(config_dict)

    # Check if dependencies are missing
    needs_install = False
    install_cmd = None

    if not check_provider_installed("litellm"):
        needs_install = True
        install_cmd = _get_install_command("litellm")

    if needs_install and install_cmd:
        # Extract only the content after "Next steps:" to avoid duplication
        lines = existing_steps.split("\n")
        header_printed = False

        rich_print("\n[bold yellow]Next steps:[/bold yellow]")
        rich_print(f"   0. Install missing dependencies: [bold yellow]{install_cmd}[/bold yellow]")

        # Print remaining steps, skipping the original header
        for _i, line in enumerate(lines):
            if not header_printed and "Next steps:" in line:
                header_printed = True
                continue
            if line.strip():
                # Increment step numbers from the original
                if line.startswith(("1.", "2.")):
                    # Increment step number
                    step_num = int(line[0]) + 1
                    rich_print(line.replace(f"{line[0]}.", f"{step_num}."))
                else:
                    rich_print(line)
    else:
        # No installation needed, print as-is
        rich_print(existing_steps)
