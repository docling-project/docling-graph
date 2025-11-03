"""
Configuration builder for interactive config creation.
"""

from typing import Any, Dict

import click
import typer
from rich import print as rich_print

from .constants import (
    API_PROVIDERS,
    BACKENDS,
    DOCLING_PIPELINES,
    EXPORT_FORMATS,
    INFERENCE_LOCATIONS,
    LOCAL_PROVIDER_DEFAULTS,
    LOCAL_PROVIDERS,
    PROCESSING_MODES,
    PROVIDER_DEFAULT_MODELS,
    VLM_DEFAULT_MODEL,
)



def build_config_interactive() -> Dict[str, Any]:
    """Build configuration through interactive prompts.

    Returns:
        Dictionary containing complete configuration.
    """
    rich_print("[bold blue]Welcome to Docling-Graph Setup![/bold blue]")
    rich_print("Let's configure your knowledge graph pipeline.\n")

    # Get all configuration sections
    defaults = _prompt_defaults()
    docling = _prompt_docling()
    models = _prompt_models(defaults["backend"], defaults["inference"])
    output = _prompt_output()

    # Build complete config
    config_dict = {
        "defaults": defaults,
        "docling": docling,
        "models": models,
        "output": output,
    }

    return config_dict


def _prompt_defaults() -> Dict[str, str]:
    """Prompt for default settings."""
    rich_print("── [bold]Default Settings[/bold] ──\n")

    # Processing mode
    rich_print("1. Processing Mode")
    rich_print(" How should documents be processed?")
    rich_print(" • one-to-one: Creates a separate Pydantic instance for each page.")
    rich_print(" • many-to-one: Combines the entire document into a single Pydantic instance.")
    processing_mode = typer.prompt(
        f"Select processing mode ({', '.join(PROCESSING_MODES)})",
        default="many-to-one",
        type=click.Choice(PROCESSING_MODES, case_sensitive=False),
    )

    # Backend
    rich_print("\n2. Backend Type")
    rich_print(" Which AI backend should be used?")
    rich_print(" • llm: Language Model (text-based)")
    rich_print(" • vlm: Vision-Language Model (image-based)")
    backend = typer.prompt(
        f"Select backend type ({', '.join(BACKENDS)})",
        default="llm",
        type=click.Choice(BACKENDS, case_sensitive=False),
    )

    # Inference
    rich_print("\n3. Inference Location")
    if backend == "vlm":
        rich_print(" Note: VLM only supports local inference")
        inference = "local"
    else:
        rich_print(" • local: Run on your machine")
        rich_print(" • remote: Use cloud APIs")
        inference = typer.prompt(
            f"Select inference location ({', '.join(INFERENCE_LOCATIONS)})",
            default="remote",
            type=click.Choice(INFERENCE_LOCATIONS, case_sensitive=False),
        )

    # Export format
    rich_print("\n4. Export Format")
    rich_print(" • csv: CSV files (nodes.csv, edges.csv)")
    rich_print(" • cypher: Cypher script for Neo4j")
    export_format = typer.prompt(
        f"Select export format ({', '.join(EXPORT_FORMATS)})",
        default="csv",
        type=click.Choice(EXPORT_FORMATS, case_sensitive=False),
    )

    return {
        "processing_mode": processing_mode,
        "backend": backend,
        "inference": inference,
        "export_format": export_format,
    }


def _prompt_docling() -> Dict[str, Any]:
    """Prompt for Docling settings."""
    rich_print("\n── [bold]Docling Pipeline[/bold] ──\n")

    # Pipeline type
    rich_print("5. Document Processing Pipeline")
    rich_print("  • ocr: OCR pipeline (standard documents)")
    rich_print("  • vision: VLM pipeline (complex layouts)")
    pipeline = typer.prompt(
        f"Select docling pipeline ({', '.join(DOCLING_PIPELINES)})",
        default="ocr",
        type=click.Choice(DOCLING_PIPELINES, case_sensitive=False),
    )

    # Export options
    rich_print("\n6. Docling Export Options")
    rich_print("  Choose what to export from document processing:")
    docling_json = typer.confirm("Export Docling document structure (JSON)?", default=True)
    markdown = typer.confirm("Export full document markdown?", default=True)
    per_page = typer.confirm("Export per-page markdown files?", default=False)

    return {
        "pipeline": pipeline,
        "export": {
            "docling_json": docling_json,
            "markdown": markdown,
            "per_page_markdown": per_page,
        },
    }


def _prompt_models(backend: str, inference: str) -> Dict[str, Any]:
    """Prompt for model configuration."""
    rich_print("\n── [bold]Model Configuration[/bold] ──\n")

    if backend == "vlm":
        return _prompt_vlm_models()
    elif inference == "local":
        return _prompt_llm_local_models()
    else:  # remote
        return _prompt_llm_remote_models()


def _prompt_vlm_models() -> Dict[str, Any]:
    """Prompt for VLM model."""
    rich_print("6. VLM Local Model")
    rich_print(f" • {VLM_DEFAULT_MODEL}: Default")
    model = typer.prompt(
        "Select VLM model",
        default=VLM_DEFAULT_MODEL,
    )

    return {
        "vlm": {
            "local": {
                "default_model": model,
                "provider": "docling",
            }
        },
        "llm": {
            "local": {
                "default_model": LOCAL_PROVIDER_DEFAULTS["vllm"],
                "provider": "vllm",
            },
            "remote": {
                "default_model": PROVIDER_DEFAULT_MODELS["mistral"],
                "provider": "mistral",
            },
        },
    }


def _prompt_llm_local_models() -> Dict[str, Any]:
    """Prompt for local LLM configuration."""
    rich_print("6. Local LLM Provider")
    rich_print(f" • {', '.join(LOCAL_PROVIDERS)}")

    provider = typer.prompt(
        f"Select local provider ({', '.join(LOCAL_PROVIDERS)})",
        default="vllm",
        type=click.Choice(LOCAL_PROVIDERS, case_sensitive=False),
    )

    default_model = LOCAL_PROVIDER_DEFAULTS.get(provider, LOCAL_PROVIDER_DEFAULTS["vllm"])

    rich_print(f"\n7. LLM Model for {provider}")
    rich_print(f" • {default_model}: Default")
    model = typer.prompt(
        "Select LLM model",
        default=default_model,
    )

    return {
        "llm": {
            "local": {
                "default_model": model,
                "provider": provider,
            },
            "remote": {
                "default_model": PROVIDER_DEFAULT_MODELS["mistral"],
                "provider": "mistral",
            },
        },
        "vlm": {
            "local": {
                "default_model": VLM_DEFAULT_MODEL,
                "provider": "docling",
            }
        },
    }


def _prompt_llm_remote_models() -> Dict[str, Any]:
    """Prompt for remote LLM configuration."""
    rich_print("6. API Provider")
    rich_print(f" • {', '.join(API_PROVIDERS)}")

    provider = typer.prompt(
        f"Select API provider ({', '.join(API_PROVIDERS)})",
        default="mistral",
        type=click.Choice(API_PROVIDERS, case_sensitive=False),
    )

    default_model = PROVIDER_DEFAULT_MODELS.get(provider, PROVIDER_DEFAULT_MODELS["mistral"])

    rich_print(f"\n7. Model for {provider}")
    rich_print(f" • {default_model}: Default")
    model = typer.prompt(
        f"Model for {provider}",
        default=default_model,
    )

    return {
        "llm": {
            "local": {
                "default_model": LOCAL_PROVIDER_DEFAULTS["vllm"],
                "provider": "vllm",
            },
            "remote": {
                "default_model": model,
                "provider": provider,
            },
        },
        "vlm": {
            "local": {
                "default_model": VLM_DEFAULT_MODEL,
                "provider": "docling",
            }
        },
    }


def _prompt_output() -> Dict[str, str]:
    """Prompt for output settings."""
    rich_print("\n── [bold]Output[/bold] ──\n")
    directory = typer.prompt(
        "Output directory",
        default="outputs",
    )
    return {"directory": directory}


def print_next_steps(config: Dict[str, Any]) -> None:
    """Print next steps after configuration."""
    rich_print("\n[bold green]Configuration created successfully![/bold green]")
    rich_print("\n[bold]Next steps:[/bold]")
    rich_print("1. Customize your Pydantic template in templates/")
    rich_print("2. Run: [cyan]docling-graph convert --source <doc> --template <template_path>[/cyan]")
