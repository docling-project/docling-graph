"""
Convert command - converts documents to knowledge graphs.
"""

from typing_extensions import Annotated
from typing import Optional
from pathlib import Path

from rich import print
import typer
import sys

sys.path.append(str(Path.cwd()))

from ..config_utils import load_config, get_config_value
from ..validators import (
    validate_processing_mode,
    validate_backend_type,
    validate_inference,
    validate_docling_config,
    validate_export_format,
    validate_vlm_constraints
)


# Import pipeline (needs to be relative to original structure)
try:
    from ...pipeline import run_pipeline
except ImportError:
    # Fallback for development
    from pipeline import run_pipeline


def convert_command(
    source: Annotated[Path, typer.Argument(
        help="Path to the source document (PDF, JPG, PNG).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    )],
    template: Annotated[str, typer.Option(
        "--template", "-t",
        help="Dotted path to Pydantic template (e.g., 'templates.invoice.Invoice')."
    )],
    # Configuration options
    processing_mode: Annotated[Optional[str], typer.Option(
        "--processing-mode", "-p",
        help="Processing strategy: 'one-to-one' or 'many-to-one'."
    )] = None,
    backend_type: Annotated[Optional[str], typer.Option(
        "--backend-type", "-b",
        help="Backend: 'llm' or 'vlm'."
    )] = None,
    inference: Annotated[Optional[str], typer.Option(
        "--inference", "-i",
        help="Inference: 'local' or 'remote'."
    )] = None,
    docling_config: Annotated[Optional[str], typer.Option(
        "--docling-config", "-d",
        help="Docling pipeline: 'ocr' or 'vision'."
    )] = None,
    # Output options
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Output directory.",
        file_okay=False,
        writable=True
    )] = Path("outputs"),
    model: Annotated[Optional[str], typer.Option(
        "--model",
        help="Override model name."
    )] = None,
    provider: Annotated[Optional[str], typer.Option(
        "--provider",
        help="Override provider."
    )] = None,
    export_format: Annotated[Optional[str], typer.Option(
        "--export-format", "-e",
        help="Export format: 'csv' or 'cypher'."
    )] = None,
    reverse_edges: Annotated[bool, typer.Option(
        "--reverse-edges", "-r",
        help="Create bidirectional edges."
    )] = False
):
    """Convert a document to a knowledge graph."""

    print("[blue]━━━ Docling-Graph Conversion ━━━[/blue]")

    # Load configuration
    config_data = load_config()
    defaults = config_data.get('defaults', {})

    # Resolve configuration (CLI args override config file)
    processing_mode = processing_mode or defaults.get('processing_mode', 'many-to-one')
    backend_type = backend_type or defaults.get('backend_type', 'llm')
    inference = inference or defaults.get('inference', 'local')
    export_format = export_format or defaults.get('export_format', 'csv')
    docling_config = docling_config or get_config_value(config_data, 'docling', 'pipeline', default='ocr')

    # Validate all inputs
    processing_mode = validate_processing_mode(processing_mode)
    backend_type = validate_backend_type(backend_type)
    inference = validate_inference(inference)
    docling_config = validate_docling_config(docling_config)
    export_format = validate_export_format(export_format)

    # Validate VLM constraints
    validate_vlm_constraints(backend_type, inference)

    # Display configuration
    print(f"\n[bold]Configuration:[/bold]")
    print(f"  Source: [cyan]{source}[/cyan]")
    print(f"  Template: [cyan]{template}[/cyan]")
    print(f"  Docling: [cyan]{docling_config}[/cyan]")
    print(f"  Processing: [cyan]{processing_mode}[/cyan]")
    print(f"  Backend: [cyan]{backend_type}[/cyan]")
    print(f"  Inference: [cyan]{inference}[/cyan]")
    print(f"  Export: [cyan]{export_format}[/cyan]")
    print(f"  Reverse edges: [cyan]{reverse_edges}[/cyan]")

    # Build run configuration
    run_config = {
        "source": str(source),
        "template": template,
        "output_dir": str(output_dir),
        "docling_config": docling_config,
        "processing_mode": processing_mode,
        "backend_type": backend_type,
        "inference": inference,
        "reverse_edges": reverse_edges,
        "export_format": export_format,
        "config": config_data,
        "model_override": model,
        "provider_override": provider
    }

    # Run pipeline
    try:
        run_pipeline(run_config)
        print("\n[green]✓ Conversion completed successfully![/green]")
    except Exception as e:
        print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)
