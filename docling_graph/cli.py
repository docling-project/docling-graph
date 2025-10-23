import typer
from typing_extensions import Annotated
from pathlib import Path
from rich import print
import sys
import os
import shutil
import yaml

# Add current working directory to path to find `templates`
sys.path.append(str(Path.cwd()))

from .pipeline import run_pipeline

app = typer.Typer(
    name="docling-graph",
    help="A tool to convert documents into knowledge graphs using configurable pipelines.",
    add_completion=False
)

CONFIG_FILE_NAME = "docling_graph_config.yaml"

@app.command(
    name="init",
    help=f"Create a default {CONFIG_FILE_NAME} in the current directory."
)
def init_command():
    """
    Creates a default configuration file in the current directory.
    """
    config_template_path = Path(__file__).parent / "config_template.yaml"
    output_path = Path.cwd() / CONFIG_FILE_NAME

    if output_path.exists():
        print(f"[yellow]'{CONFIG_FILE_NAME}' already exists in this directory.[/yellow]")
        overwrite = typer.confirm("Do you want to overwrite it?")
        if not overwrite:
            print("Initialization cancelled.")
            raise typer.Abort()

    try:
        shutil.copy(config_template_path, output_path)
        print(f"[green]Successfully created '{output_path}'[/green]")
        print("You can now edit this file to customize your pipelines.")
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
    help="Convert a document to a knowledge graph using a defined pipeline."
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
    
    pipeline: Annotated[str, typer.Option(
        "--pipeline", "-p",
        help="Name of the pipeline to use from your config file (e.g., 'one_to_one_local')."
    )],
    
    output_dir: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Directory to save the output graph image and JSON.",
        file_okay=False,
        writable=True
    )] = Path("outputs"),

    # --- Optional Overrides ---
    model: Annotated[str, typer.Option(
        "--model",
        help="Override the pipeline's default model (e.g., 'numind/NuExtract-2.0-8B')."
    )] = None,
    
    provider: Annotated[str, typer.Option(
        "--provider",
        help="[api method] Override the pipeline's default provider (e.g., 'openai')."
    )] = None
):
    """
    Main CLI command to convert a document.
    """
    print(f"[bold]Starting Docling-Graph Conversion[/bold]")
    
    # 1. Load the main config file
    config_data = _load_config()
    
    # 2. Find the requested pipeline
    if pipeline not in config_data.get('pipelines', {}):
        print(f"[red]Error:[/red] Pipeline '{pipeline}' not found in '{CONFIG_FILE_NAME}'.")
        print("Available pipelines:")
        for name in config_data.get('pipelines', {}):
            print(f"  - {name}")
        raise typer.Exit(code=1)
        
    pipeline_config = config_data['pipelines'][pipeline]
    print(f"Using pipeline: [blue]{pipeline}[/blue] ({pipeline_config.get('description', 'No description')})")

    # 3. Bundle settings for the pipeline
    run_config = {
        "source": str(source),
        "template": template,
        "output_dir": str(output_dir),
        
        # Pass the whole pipeline config
        "pipeline": pipeline_config,
        
        # Add overrides
        "model_override": model,
        "provider_override": provider
    }
    
    try:
        run_pipeline(run_config)
    except Exception as e:
        print(f"\n[bold red]An unexpected error occurred:[/bold red] {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for full debugging
        raise typer.Exit(code=1)

def main():
    app()

