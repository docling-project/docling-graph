"""
Configuration builder for interactive config creation.
"""

from typing import Any, Dict, Tuple

import click
import typer
from rich import print

from ..constants import (
    API_PROVIDERS,
    BACKEND_TYPES,
    DEFAULT_MODELS,
    DOCLING_PIPELINES,
    EXPORT_FORMATS,
    INFERENCE_LOCATIONS,
    PROCESSING_MODES,
)


def build_config_interactive() -> Dict[str, Any]:
    """Build configuration through interactive prompts.

    Returns:
        Dictionary containing complete configuration.
    """
    print("[bold blue]Welcome to Docling-Graph Setup![/bold blue]")
    print("Let's configure your knowledge graph pipeline.\n")

    # Get all configuration sections
    defaults = _prompt_defaults()
    docling = _prompt_docling()
    models = _prompt_models(defaults["backend_type"], defaults["inference"])
    output = _prompt_output()

    # Build complete config
    config_dict = {"defaults": defaults, "docling": docling, "models": models, "output": output}

    return config_dict


def _prompt_defaults() -> Dict[str, str]:
    """Prompt for default settings."""
    print("[bold cyan]── Default Settings ──[/bold cyan]")

    # Processing Mode
    print("\n[bold]1. Processing Mode[/bold]")
    print(" [dim]How should documents be processed?[/dim]")
    print(" • [cyan]one-to-one[/cyan]: Process each page individually")
    print(" • [cyan]many-to-one[/cyan]: Process entire document as one unit")
    processing_mode = typer.prompt(
        "Select processing mode",
        type=click.Choice(PROCESSING_MODES, case_sensitive=False),
        default="one-to-one",
    )

    # Backend Type
    print("\n[bold]2. Backend Type[/bold]")
    print(" [dim]Which AI backend should be used?[/dim]")
    print(" • [cyan]llm[/cyan]: Language Model (text-based)")
    print(" • [cyan]vlm[/cyan]: Vision-Language Model (image-based)")
    backend_type = typer.prompt(
        "Select backend type", type=click.Choice(BACKEND_TYPES, case_sensitive=False), default="llm"
    )

    # Inference Location
    print("\n[bold]3. Inference Location[/bold]")
    if backend_type == "vlm":
        print(" [yellow]Note: VLM only supports local inference[/yellow]")
        inference = "local"
    else:
        print(" • [cyan]local[/cyan]: Run on your machine")
        print(" • [cyan]remote[/cyan]: Use cloud APIs")
        inference = typer.prompt(
            "Select inference location",
            type=click.Choice(INFERENCE_LOCATIONS, case_sensitive=False),
            default="remote",
        )

    # Export Format
    print("\n[bold]4. Export Format[/bold]")
    print(" • [cyan]csv[/cyan]: CSV files (nodes.csv, edges.csv)")
    print(" • [cyan]cypher[/cyan]: Cypher script for Neo4j")
    export_format = typer.prompt(
        "Select export format",
        type=click.Choice(EXPORT_FORMATS, case_sensitive=False),
        default="csv",
    )

    return {
        "processing_mode": processing_mode,
        "backend_type": backend_type,
        "inference": inference,
        "export_format": export_format,
    }


def _prompt_docling() -> Dict[str, Any]:
    """Prompt for Docling configuration."""
    print("\n[bold cyan]── Docling Pipeline ──[/bold cyan]")

    # Pipeline selection
    print("\n[bold]5. Document Processing Pipeline[/bold]")
    print("  • [cyan]ocr[/cyan]: OCR pipeline (standard documents)")
    print("  • [cyan]vision[/cyan]: VLM pipeline (complex layouts)")

    pipeline = typer.prompt(
        "Select docling pipeline",
        type=click.Choice(DOCLING_PIPELINES, case_sensitive=False),
        default="ocr",
    )

    # Export options
    print("\n[bold]6. Docling Export Options[/bold]")
    print("  [dim]Choose what to export from document processing:[/dim]")

    export_docling_json = typer.confirm("  Export Docling document structure (JSON)?", default=True)

    export_markdown = typer.confirm("  Export full document markdown?", default=True)

    export_per_page = typer.confirm("  Export per-page markdown files?", default=False)

    return {
        "pipeline": pipeline,
        "export": {
            "docling_json": export_docling_json,
            "markdown": export_markdown,
            "per_page_markdown": export_per_page,
        },
    }


def _prompt_models(backend_type: str, inference: str) -> Dict[str, Any]:
    """Prompt for model configuration."""
    print("\n[bold cyan]── Model Configuration ──[/bold cyan]")

    if backend_type == "vlm":
        vlm, llm_local, llm_remote, remote_provider, _ = _prompt_vlm_models()
        local_provider = "docling"  # VLM uses docling
    elif inference == "local":
        vlm, llm_local, llm_remote, remote_provider, local_provider = _prompt_llm_local_models()
    else:
        vlm, llm_local, llm_remote, remote_provider, local_provider = _prompt_llm_remote_models()

    return {
        "vlm": {"local": {"default_model": vlm, "provider": "docling"}},
        "llm": {
            "local": {"default_model": llm_local, "provider": local_provider},
            "remote": {"default_model": llm_remote, "provider": remote_provider},
            "providers": {
                "ollama": {"default_model": "llama3:8b-instruct"},
                "vllm": {"default_model": "llama-3.1-8b"},
                "mistral": {"default_model": DEFAULT_MODELS["llm_remote"]["mistral"]},
                "openai": {"default_model": DEFAULT_MODELS["llm_remote"]["openai"]},
                "gemini": {"default_model": DEFAULT_MODELS["llm_remote"]["gemini"]},
            },
        },
    }


def _prompt_vlm_models() -> Tuple[str, str, str, str, str]:
    """Prompt for VLM model."""
    print("[bold]6. VLM Local Model[/bold]")
    print(f" • [cyan]{DEFAULT_MODELS['vlm']}[/cyan]: Default")
    vlm = typer.prompt("Select VLM model", default=DEFAULT_MODELS["vlm"])
    if vlm == "custom":
        vlm = typer.prompt("Enter custom model path")
    return (
        vlm,
        DEFAULT_MODELS["llm_local"],
        DEFAULT_MODELS["llm_remote"]["mistral"],
        "mistral",
        "docling",
    )


def _prompt_llm_local_models() -> Tuple[str, str, str, str]:
    """Prompt for local LLM model."""
    print("\n[bold]6. Local LLM Provider[/bold]")
    print(" • [cyan]ollama[/cyan]: Ollama (requires ollama serve)")
    print(" • [cyan]vllm[/cyan]: vLLM (direct Python API, GPU required)")

    local_provider = typer.prompt(
        "Select local provider",
        type=click.Choice(["ollama", "vllm"], case_sensitive=False),
        default="vllm",
    )

    print(f"\n[bold]7. LLM Model for {local_provider}[/bold]")
    print(f" • [cyan]{DEFAULT_MODELS['llm_local']}[/cyan]: Default")

    llm = typer.prompt("Select LLM model", default=DEFAULT_MODELS)
    if llm == "custom":
        llm = typer.prompt(f"Enter {local_provider} model name")

    return (
        DEFAULT_MODELS["vlm"],
        llm,
        DEFAULT_MODELS["llm_remote"]["mistral"],
        "mistral",
        local_provider,
    )


def _prompt_llm_remote_models() -> Tuple[str, str, str, str, str]:
    """Prompt for remote LLM model."""
    print("[bold]6. API Provider[/bold]")
    print(" • [cyan]mistral[/cyan], [cyan]openai[/cyan], [cyan]gemini[/cyan]")
    provider = typer.prompt(
        "Select API provider",
        type=click.Choice(API_PROVIDERS, case_sensitive=False),
        default="mistral",
    )
    model = typer.prompt(f"Model for {provider}", default=DEFAULT_MODELS["llm_remote"][provider])
    return (DEFAULT_MODELS["vlm"], DEFAULT_MODELS["llm_local"], model, provider, "vllm")


def _prompt_output() -> Dict[str, Any]:
    """Prompt for output settings."""
    print("\n[bold cyan]── Output Settings ──[/bold cyan]")

    print("\n[bold]7. Output Directory[/bold]")
    directory = typer.prompt("Default output directory", default="outputs")

    print("\n[bold]8. Visualization Options[/bold]")
    visualizations = typer.confirm("Create interactive visualizations?", default=True)
    markdown = typer.confirm("Create markdown report?", default=True)

    return {
        "default_directory": directory,
        "create_visualizations": visualizations,
        "create_markdown": markdown,
    }


def print_next_steps(config: Dict[str, Any]) -> None:
    """Print next steps after config creation."""
    inference = config["defaults"]["inference"]
    backend = config["defaults"]["backend_type"]

    print("\n[bold]Next steps:[/bold]")
    if inference == "local" and backend == "llm":
        provider = config["models"]["llm"]["local"]["provider"]
        model = config["models"]["llm"]["local"]["default_model"]

        if provider == "ollama":
            print(" 1. Start Ollama: [cyan]ollama serve[/cyan]")
            print(f" 2. Pull model: [cyan]ollama pull {model}[/cyan]")
        elif provider == "vllm":
            print(f" 1. Start vLLM: [cyan]vllm serve {model} --host 0.0.0.0[/cyan]")
            print(" 2. Ensure GPU available with sufficient memory")
    elif inference == "remote":
        provider = config["models"]["llm"]["remote"]["provider"].upper()
        print(f" 1. Set API key: [cyan]export {provider}_API_KEY='...'[/cyan]")

    print(" 3. Convert: [cyan]docling-graph convert doc.pdf -t templates.Invoice[/cyan]")
    print("\n[dim]Edit config.yaml anytime to adjust settings.[/dim]")
