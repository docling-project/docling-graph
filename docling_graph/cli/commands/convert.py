"""
Convert command - converts documents to knowledge graphs.
"""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich import print as rich_print
from typing_extensions import Annotated

from docling_graph.config import PipelineConfig
from docling_graph.exceptions import (
    ConfigurationError,
    DoclingGraphError,
    ExtractionError,
    PipelineError,
)
from docling_graph.pipeline import run_pipeline

from ..config_utils import load_config
from ..validators import (
    validate_backend_type,
    validate_docling_config,
    validate_export_format,
    validate_inference,
    validate_processing_mode,
    validate_vlm_constraints,
)

logger = logging.getLogger(__name__)


def convert_command(
    source: Annotated[
        str,
        typer.Argument(
            help="Path to source document (PDF, JPG, PNG, TXT, MD), URL, or DoclingDocument JSON file.",
        ),
    ],
    template: Annotated[
        str,
        typer.Option(
            "--template",
            "-t",
            help="Dotted path to Pydantic template (e.g., 'templates.billing_document.BillingDocument').",
        ),
    ],
    processing_mode: Annotated[
        str | None,
        typer.Option(
            "--processing-mode", "-p", help="Processing strategy: 'one-to-one' or 'many-to-one'."
        ),
    ] = None,
    backend: Annotated[
        str | None, typer.Option("--backend", "-b", help="Backend: 'llm' or 'vlm'.")
    ] = None,
    inference: Annotated[
        str | None, typer.Option("--inference", "-i", help="Inference: 'local' or 'remote'.")
    ] = None,
    docling_pipeline: Annotated[
        str | None,
        typer.Option("--docling-pipeline", "-d", help="Docling pipeline: 'ocr' or 'vision'."),
    ] = None,
    # Extraction options
    llm_consolidation: Annotated[
        bool | None,
        typer.Option(
            "--llm-consolidation/--no-llm-consolidation",
            help="Enable/disable final LLM consolidation step.",
        ),
    ] = None,
    use_chunking: Annotated[
        bool | None,
        typer.Option(
            "--use-chunking/--no-use-chunking",
            help="Enable/disable document chunking.",
        ),
    ] = None,
    # Docling export options
    export_docling_json: Annotated[
        bool,
        typer.Option(
            "--export-docling-json/--no-docling-json", help="Export Docling document as JSON."
        ),
    ] = True,
    export_markdown: Annotated[
        bool, typer.Option("--export-markdown/--no-markdown", help="Export full document markdown.")
    ] = True,
    export_per_page: Annotated[
        bool,
        typer.Option("--export-per-page/--no-per-page", help="Export per-page markdown files."),
    ] = False,
    # Output options
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o", help="Output directory.", file_okay=False, writable=True
        ),
    ] = Path("outputs"),
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override model name.")] = None,
    provider: Annotated[str | None, typer.Option("--provider", help="Override provider.")] = None,
    export_format: Annotated[
        str | None,
        typer.Option("--export-format", "-e", help="Export format: 'csv' or 'cypher'."),
    ] = None,
    reverse_edges: Annotated[
        bool, typer.Option("--reverse-edges", "-r", help="Create bidirectional edges.")
    ] = False,
) -> None:
    """Convert a document to a knowledge graph."""
    logger.debug("Starting convert command")
    logger.debug(f"Source: {source}, Template: {template}")
    logger.debug(
        f"CLI args - Backend: {backend}, Inference: {inference}, Processing: {processing_mode}"
    )

    rich_print("[green]--- Starting Docling-Graph Conversion ---[/green]")

    # Load YAML configuration (flat)
    logger.debug("Loading configuration from config.yaml")
    config_data = load_config()
    logger.debug(f"Loaded config keys: {list(config_data.keys())}")
    defaults = config_data.get("defaults", {})
    docling_cfg = config_data.get("docling", {})
    models_from_yaml = config_data.get("models", {})  # flat models only

    # Resolve configuration (CLI args override config file)
    processing_mode_val = processing_mode or defaults.get("processing_mode", "many-to-one")
    backend_val = backend or defaults.get("backend", "llm")
    inference_val = inference or defaults.get("inference", "local")
    export_format_val = export_format or defaults.get("export_format", "csv")

    # Docling settings
    docling_pipeline_val = docling_pipeline or docling_cfg.get("pipeline", "ocr")

    # Resolve extraction settings
    final_llm_consolidation = (
        llm_consolidation
        if llm_consolidation is not None
        else defaults.get("llm_consolidation", True)
    )
    final_use_chunking = (
        use_chunking if use_chunking is not None else defaults.get("use_chunking", True)
    )

    # Docling export settings - use config file as fallback
    docling_export_settings = docling_cfg.get("export", {})
    final_export_docling_json = (
        export_docling_json
        if export_docling_json is not None
        else docling_export_settings.get("docling_json", True)
    )
    final_export_markdown = (
        export_markdown
        if export_markdown is not None
        else docling_export_settings.get("markdown", True)
    )
    final_export_per_page = (
        export_per_page
        if export_per_page is not None
        else docling_export_settings.get("per_page_markdown", False)
    )

    # Validate all inputs
    processing_mode_val = validate_processing_mode(processing_mode_val)
    backend_val = validate_backend_type(backend_val)
    inference_val = validate_inference(inference_val)
    docling_pipeline_val = validate_docling_config(docling_pipeline_val)
    export_format_val = validate_export_format(export_format_val)
    validate_vlm_constraints(backend_val, inference_val)

    logger.debug(f"Validated configuration - Backend: {backend_val}, Inference: {inference_val}")
    logger.debug(f"Processing mode: {processing_mode_val}, Export format: {export_format_val}")

    # Detect and display input type
    from docling_graph.core.input.types import InputTypeDetector

    try:
        detected_type = InputTypeDetector.detect(source, mode="cli")
        input_type_display = detected_type.value.replace("_", " ").title()
    except Exception:
        input_type_display = "Unknown"

    # Display configuration
    rich_print("[yellow][PipelineConfiguration][/yellow]")
    rich_print(f" • Source: [cyan]{source}[/cyan]")
    rich_print(f" • Input Type: [cyan]{input_type_display}[/cyan]")
    rich_print(f" • Template: [cyan]{template}[/cyan]")
    rich_print(f" • Docling Pipeline: [cyan]{docling_pipeline_val}[/cyan]")
    rich_print(f" • Processing: [cyan]{processing_mode_val}[/cyan]")
    rich_print(f" • Backend: [cyan]{backend_val}[/cyan]")
    rich_print(f" • Inference: [cyan]{inference_val}[/cyan]")
    rich_print(f" • Export: [cyan]{export_format_val}[/cyan]")
    rich_print(f" • Reverse edges: [cyan]{reverse_edges}[/cyan]")

    # Display Extraction settings
    rich_print("[yellow][ExtractionSettings][/yellow]")
    rich_print(f" • LLM Consolidation: [cyan]{final_llm_consolidation}[/cyan]")
    rich_print(f" • Use Chunking: [cyan]{final_use_chunking}[/cyan]")

    # Display Docling export settings
    rich_print("[yellow][DoclingExport][/yellow]")
    rich_print(f" • Document JSON: [cyan]{final_export_docling_json}[/cyan]")
    rich_print(f" • Markdown: [cyan]{final_export_markdown}[/cyan]")
    rich_print(f" • Per-page MD: [cyan]{final_export_per_page}[/cyan]")

    # Build typed config
    logger.debug("Building PipelineConfig object")
    cfg = PipelineConfig(
        source=str(source),
        template=template,
        backend=backend_val,
        inference=inference_val,
        processing_mode=processing_mode_val,
        docling_config=docling_pipeline_val,
        model_override=model,
        provider_override=provider,
        models=models_from_yaml,
        llm_consolidation=final_llm_consolidation,
        use_chunking=final_use_chunking,
        export_format=export_format_val,
        export_docling=True,
        export_docling_json=final_export_docling_json,
        export_markdown=final_export_markdown,
        export_per_page_markdown=final_export_per_page,
        reverse_edges=reverse_edges,
        output_dir=str(output_dir),
    )

    logger.debug(f"PipelineConfig created: backend={cfg.backend}, inference={cfg.inference}")
    logger.debug(f"Output directory: {cfg.output_dir}")

    # Run pipeline with normalized/validated config
    logger.info("Starting pipeline execution")
    try:
        logger.debug("Calling run_pipeline() in CLI mode")
        run_pipeline(cfg, mode="cli")
        logger.info("--- Pipeline execution Completed Successfully ---")
        rich_print("[green]--- Docling-Graph Conversion Successfull ---[/green]")
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e.message}", exc_info=True)
        rich_print(f"\n[red]Configuration Error:[/red] {e.message}")
        if e.details:
            rich_print("[yellow]Details:[/yellow]")
            for key, value in e.details.items():
                rich_print(f"  • {key}: {value}")
        raise typer.Exit(code=1) from e
    except ExtractionError as e:
        logger.error(f"Extraction error: {e.message}", exc_info=True)
        rich_print(f"\n[red]Extraction Error:[/red] {e.message}")
        if e.details:
            rich_print("[yellow]Details:[/yellow]")
            for key, value in e.details.items():
                rich_print(f"  • {key}: {value}")
        raise typer.Exit(code=1) from e
    except PipelineError as e:
        logger.error(f"Pipeline error: {e.message}", exc_info=True)
        rich_print(f"\n[red]Pipeline Error:[/red] {e.message}")
        if e.details:
            rich_print("[yellow]Details:[/yellow]")
            for key, value in e.details.items():
                rich_print(f"  • {key}: {value}")
        raise typer.Exit(code=1) from e
    except DoclingGraphError as e:
        logger.error(f"Docling-graph error: {e.message}", exc_info=True)
        rich_print(f"\n[red]Error:[/red] {e.message}")
        if e.details:
            rich_print("[yellow]Details:[/yellow]")
            for key, value in e.details.items():
                rich_print(f"  • {key}: {value}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}: {e}")
        rich_print(f"\n[red]Unexpected error:[/red] {type(e).__name__}: {e}")
        raise typer.Exit(code=1) from e
