"""
Convert command - converts documents to knowledge graphs.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rich_print
from typing_extensions import Annotated

sys.path.append(str(Path.cwd()))

from docling_graph.config import PipelineConfig
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


def convert_command(
    source: Annotated[
        Path,
        typer.Argument(
            help="Path to the source document (PDF, JPG, PNG).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    template: Annotated[
        str,
        typer.Option(
            "--template",
            "-t",
            help="Dotted path to Pydantic template (e.g., 'templates.invoice.Invoice').",
        ),
    ],
    processing_mode: Annotated[
        Optional[str],
        typer.Option(
            "--processing-mode", "-p", help="Processing strategy: 'one-to-one' or 'many-to-one'."
        ),
    ] = None,
    backend: Annotated[
        Optional[str], typer.Option("--backend", "-b", help="Backend: 'llm' or 'vlm'.")
    ] = None,
    inference: Annotated[
        Optional[str], typer.Option("--inference", "-i", help="Inference: 'local' or 'remote'.")
    ] = None,
    docling_pipeline: Annotated[
        Optional[str],
        typer.Option("--docling-pipeline", "-d", help="Docling pipeline: 'ocr' or 'vision'."),
    ] = None,
    # Extraction options
    llm_consolidation: Annotated[
        Optional[bool],
        typer.Option(
            "--llm-consolidation/--no-llm-consolidation",
            help="Enable/disable final LLM consolidation step.",
        ),
    ] = None,
    use_chunking: Annotated[
        Optional[bool],
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
    model: Annotated[
        Optional[str], typer.Option("--model", "-m", help="Override model name.")
    ] = None,
    provider: Annotated[
        Optional[str], typer.Option("--provider", help="Override provider.")
    ] = None,
    export_format: Annotated[
        Optional[str],
        typer.Option("--export-format", "-e", help="Export format: 'csv' or 'cypher'."),
    ] = None,
    reverse_edges: Annotated[
        bool, typer.Option("--reverse-edges", "-r", help="Create bidirectional edges.")
    ] = False,
) -> None:
    """Convert a document to a knowledge graph."""
    rich_print("--- [blue]Docling-Graph Conversion[/blue] ---")

    # Load YAML configuration (flat)
    config_data = load_config()
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

    # Display configuration
    rich_print("\n[bold]Configuration:[/bold]")
    rich_print(f" • Source: [cyan]{source}[/cyan]")
    rich_print(f" • Template: [cyan]{template}[/cyan]")
    rich_print(f" • Docling Pipeline: [cyan]{docling_pipeline_val}[/cyan]")
    rich_print(f" • Processing: [cyan]{processing_mode_val}[/cyan]")
    rich_print(f" • Backend: [cyan]{backend_val}[/cyan]")
    rich_print(f" • Inference: [cyan]{inference_val}[/cyan]")
    rich_print(f" • Export: [cyan]{export_format_val}[/cyan]")
    rich_print(f" • Reverse edges: [cyan]{reverse_edges}[/cyan]")

    # Display Extraction settings
    rich_print("\n[bold]Extraction Settings:[/bold]")
    rich_print(f" • LLM Consolidation: [cyan]{final_llm_consolidation}[/cyan]")
    rich_print(f" • Use Chunking: [cyan]{final_use_chunking}[/cyan]")

    # Display Docling export settings
    rich_print("\n[bold]Docling Export:[/bold]")
    rich_print(f" • Document JSON: [cyan]{final_export_docling_json}[/cyan]")
    rich_print(f" • Markdown: [cyan]{final_export_markdown}[/cyan]")
    rich_print(f" • Per-page MD: [cyan]{final_export_per_page}[/cyan]")

    # Build typed config
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

    # Run pipeline with normalized/validated config
    try:
        run_pipeline(cfg)
    except Exception as e:
        raise ValueError(str(e)) from e
