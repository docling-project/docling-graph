import typer
from typing_extensions import Annotated
from pathlib import Path
from rich import print
import sys
import yaml

sys.path.append(str(Path.cwd()))

from .pipeline import run_pipeline

app = typer.Typer(
    name="docling-graph",
    help="A tool to convert documents into knowledge graphs using configurable pipelines.",
    add_completion=False
)

CONFIG_FILE_NAME = "config.yaml"

@app.command(
    name="init",
    help=f"Create a default {CONFIG_FILE_NAME} in the current directory."
)
def init_command():
    """Creates a default configuration file in the current directory."""
    from shutil import copy
    config_template_path = Path(__file__).parent / "config_template.yaml"
    output_path = Path.cwd() / CONFIG_FILE_NAME

    if output_path.exists():
        print(f"[yellow]'{CONFIG_FILE_NAME}' already exists in this directory.[/yellow]")
        overwrite = typer.confirm("Do you want to overwrite it?")
        if not overwrite:
            print("Initialization cancelled.")
            raise typer.Abort()

    try:
        copy(config_template_path, output_path)
        print(f"[green]Successfully created '{output_path}'[/green]")
        print("You can now edit this file to customize your configuration.")
    except Exception as e:
        print(f"[red]Error creating config file:[/red] {e}")
        raise typer.Exit(code=1)


def _load_config() -> dict:
    """Loads the config file from the current directory."""
    config_path = Path.cwd() / CONFIG_FILE_NAME
    if not config_path.exists():
        print(f"[red]Error:[/red] Configuration file '{CONFIG_FILE_NAME}' not found.")
        print(f"Please run [cyan]docling-graph init[/cyan] first.")
        raise typer.Exit(code=1)

    with open(config_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"[red]Error parsing '{CONFIG_FILE_NAME}':[/red] {e}")
            raise typer.Exit(code=1)


@app.command(
    name="convert",
    help="Convert a document to a knowledge graph."
)
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
        help="Dotted path to the Pydantic template class (e.g., 'templates.invoice.Invoice')."
    )],
    
    # --- Three Independent Configuration Dimensions ---
    processing_mode: Annotated[str, typer.Option(
        "--processing-mode", "-p",
        help="Processing strategy: 'one-to-one' (per page) or 'many-to-one' (entire document)."
    )] = "many-to-one",
    
    model_type: Annotated[str, typer.Option(
        "--model-type", "-m",
        help="Model type: 'llm' (Language Model) or 'vlm' (Vision-Language Model)."
    )] = "llm",
    
    inference: Annotated[str, typer.Option(
        "--inference", "-i",
        help="Inference location: 'local' or 'api'."
    )] = "local",

    docling_config: Annotated[str, typer.Option(
        "--docling-config", "-d",
        help="Docling pipeline configuration: 'default' (OCR) or 'vlm' (Vision-Language Model)."
    )] = None,
    
    # --- Optional Overrides ---
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Directory to save the output files.",
        file_okay=False,
        writable=True
    )] = Path("outputs"),
    
    model: Annotated[str, typer.Option(
        "--model",
        help="Override specific model name (e.g., 'mistral-large-latest', 'llama3:8b')."
    )] = None,
    
    provider: Annotated[str, typer.Option(
        "--provider",
        help="Override provider (e.g., 'mistral', 'openai', 'ollama')."
    )] = None,
    
    export_format: Annotated[str, typer.Option(
        "--export-format", "-e",
        help="Format to export the graph data (csv or cypher)."
    )] = "csv"
):
    """
    Main CLI command to convert a document to a knowledge graph.
    """
    print("--- [blue]Initiating Docling-Graph Conversion[/blue] ---")

    # Load config
    config_data = _load_config()

    # Update the config loading section
    defaults = config_data.get('defaults', {})

    # Use CLI values if provided, otherwise fall back to config defaults
    processing_mode = (processing_mode or defaults.get('processing_mode', 'many-to-one')).lower()
    model_type = (model_type or defaults.get('model_type', 'llm')).lower()
    inference = (inference or defaults.get('inference', 'local')).lower()
    export_format = (export_format or defaults.get('export_format', 'csv')).lower()
    docling_config = (docling_config or config_data.get('docling', {}).get('pipeline', 'default')).lower()
    
    # Arguments validation
    if processing_mode not in ["one-to-one", "many-to-one"]:
        print(f"[red]Error:[/red] Invalid processing mode '{processing_mode}'. Must be 'one-to-one' or 'many-to-one'.")
        raise typer.Exit(code=1)
    
    if model_type not in ["llm", "vlm"]:
        print(f"[red]Error:[/red] Invalid model type '{model_type}'. Must be 'llm' or 'vlm'.")
        raise typer.Exit(code=1)
    
    if inference not in ["local", "api"]:
        print(f"[red]Error:[/red] Invalid inference location '{inference}'. Must be 'local' or 'api'.")
        raise typer.Exit(code=1)
    
    if docling_config not in ["default", "vlm"]:
        print(f"[red]Error:[/red] Invalid docling config '{docling_config}'. Must be 'default' or 'vlm'.")
        raise typer.Exit(code=1)
        
    if export_format not in ["csv", "cypher"]:
        print(f"[red]Error:[/red] Invalid export format '{export_format}'. Must be 'csv' or 'cypher'.")
        raise typer.Exit(code=1)
    
    # Validate VLM constraint (VLM only works locally for now)
    if model_type == "vlm" and inference == "api":
        print(f"[red]Error:[/red] VLM (Vision-Language Model) is currently only supported with local inference.")
        print("Please use '--inference local' or switch to '--model-type llm' for API inference.")
        raise typer.Exit(code=1)
    
    # Display configuration
    print(f"Configuration:")
    print(f"  Processing mode: [cyan]{processing_mode}[/cyan]")
    print(f"  Model type:      [cyan]{model_type}[/cyan]")
    print(f"  Inference:       [cyan]{inference}[/cyan]")
    print(f"  Docling config: [cyan]{docling_config}[/cyan]")
    print(f"  Export format:   [cyan]{export_format}[/cyan]")
    
    # Bundle settings for the pipeline
    run_config = {
        "source": str(source),
        "template": template,
        "output_dir": str(output_dir),
        "processing_mode": processing_mode,
        "model_type": model_type,
        "inference": inference,
        "export_format": export_format,
        "docling_config": docling_config,
        "config": config_data,
        "model_override": model,
        "provider_override": provider
    }

    try:
        run_pipeline(run_config)
    except Exception as e:
        print(f"\n[bold red]An unexpected error occurred:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


def main():
    app()
