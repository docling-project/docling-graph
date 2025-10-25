"""
Input validation functions for CLI commands.
"""

from typing import Literal
from rich import print
import typer

from .constants import (
    PROCESSING_MODES,
    BACKEND_TYPES,
    INFERENCE_LOCATIONS,
    DOCLING_PIPELINES,
    EXPORT_FORMATS
)


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
        print(f"[red]Error:[/red] Invalid processing mode '{mode}'.")
        print(f"Must be one of: {', '.join(PROCESSING_MODES)}")
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
    if backend not in BACKEND_TYPES:
        print(f"[red]Error:[/red] Invalid backend type '{backend}'.")
        print(f"Must be one of: {', '.join(BACKEND_TYPES)}")
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
        print(f"[red]Error:[/red] Invalid inference location '{inference}'.")
        print(f"Must be one of: {', '.join(INFERENCE_LOCATIONS)}")
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
        print(f"[red]Error:[/red] Invalid docling config '{config}'.")
        print(f"Must be one of: {', '.join(DOCLING_PIPELINES)}")
        raise typer.Exit(code=1)
    return config


def validate_export_format(format: str) -> str:
    """Validate export format.

    Args:
        format: Export format to validate.

    Returns:
        Lowercase validated format.

    Raises:
        typer.Exit: If format is invalid.
    """
    format = format.lower()
    if format not in EXPORT_FORMATS:
        print(f"[red]Error:[/red] Invalid export format '{format}'.")
        print(f"Must be one of: {', '.join(EXPORT_FORMATS)}")
        raise typer.Exit(code=1)
    return format


def validate_vlm_constraints(backend_type: str, inference: str) -> None:
    """Validate VLM-specific constraints.

    Args:
        backend_type: Backend type.
        inference: Inference location.

    Raises:
        typer.Exit: If VLM constraints are violated.
    """
    if backend_type == "vlm" and inference == "remote":
        print(f"[red]Error:[/red] VLM (Vision-Language Model) is currently only supported with local inference.")
        print("Please use '--inference local' or switch to '--backend_type llm' for API inference.")
        raise typer.Exit(code=1)
