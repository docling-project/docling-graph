"""
Convert command - converts documents to knowledge graphs.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print
from typing_extensions import Annotated

sys.path.append(str(Path.cwd()))

from docling_graph.pipeline import run_pipeline

from ..config_utils import get_config_value, load_config
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
    # Configuration options
    processing_mode: Annotated[
        Optional[str],
        typer.Option(
            "--processing-mode", "-p", help="Processing strategy: 'one-to-one' or 'many-to-one'."
        ),
    ] = None,
    backend_type: Annotated[
        Optional[str], typer.Option("--backend-type", "-b", help="Backend: 'llm' or 'vlm'.")
    ] = None,
    inference: Annotated[
        Optional[str], typer.Option("--inference", "-i", help="Inference: 'local' or 'remote'.")
    ] = None,
    docling_pipeline: Annotated[
        Optional[str],
        typer.Option("--docling-pipeline", "-d", help="Docling pipeline: 'ocr' or 'vision'."),
    ] = None,
    # Docling export options Experiment
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
):
    """Convert a document to a knowledge graph."""
    print("--- [blue]Docling-Graph Conversion[/blue] ---")

    # Load configuration
    config_data = load_config()
    defaults = config_data.get("defaults", {})
    docling_config = config_data.get("docling", {})

    # Resolve configuration (CLI args override config file)
    processing_mode = processing_mode or defaults.get("processing_mode", "many-to-one")
    backend_type = backend_type or defaults.get("backend_type", "llm")
    inference = inference or defaults.get("inference", "local")
    export_format = export_format or defaults.get("export_format", "csv")

    # Docling settings
    docling_pipeline = docling_pipeline or docling_config.get("pipeline", "ocr")

    # Docling export settings - use config file as fallback
    docling_export_settings = docling_config.get("export", {})
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
    processing_mode = validate_processing_mode(processing_mode)
    backend_type = validate_backend_type(backend_type)
    inference = validate_inference(inference)
    docling_pipeline = validate_docling_config(docling_pipeline)
    export_format = validate_export_format(export_format)

    # Validate VLM constraints
    validate_vlm_constraints(backend_type, inference)

    # Display configuration
    print("\n[bold]Configuration:[/bold]")
    print(f"  Source: [cyan]{source}[/cyan]")
    print(f"  Template: [cyan]{template}[/cyan]")
    print(f"  Docling Pipeline: [cyan]{docling_pipeline}[/cyan]")
    print(f"  Processing: [cyan]{processing_mode}[/cyan]")
    print(f"  Backend: [cyan]{backend_type}[/cyan]")
    print(f"  Inference: [cyan]{inference}[/cyan]")
    print(f"  Export: [cyan]{export_format}[/cyan]")
    print(f"  Reverse edges: [cyan]{reverse_edges}[/cyan]")

    # Display Docling export settings
    print("\n[bold]Docling Export:[/bold]")
    print(f"  Document JSON: [cyan]{final_export_docling_json}[/cyan]")
    print(f"  Markdown: [cyan]{final_export_markdown}[/cyan]")
    print(f"  Per-page MD: [cyan]{final_export_per_page}[/cyan]")

    # Build run configuration
    run_config = {
        "source": str(source),
        "template": template,
        "output_dir": str(output_dir),
        "docling_config": docling_pipeline,
        "processing_mode": processing_mode,
        "backend_type": backend_type,
        "inference": inference,
        "reverse_edges": reverse_edges,
        "export_format": export_format,
        # Docling export settings Experiment
        "export_docling": True,  # Master switch
        "export_docling_json": final_export_docling_json,
        "export_markdown": final_export_markdown,
        "export_per_page_markdown": final_export_per_page,
        "config": config_data,
        "model_override": model,
        "provider_override": provider,
    }

    # Run pipeline
    try:
        run_pipeline(run_config)
    except Exception as e:
        print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)
