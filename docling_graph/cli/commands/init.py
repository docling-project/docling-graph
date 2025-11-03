"""
Init command - creates configuration file interactively.
"""

from pathlib import Path

import typer
import yaml
from rich import print as rich_print

from docling_graph.config import PipelineConfig

from ..config_builder import build_config_interactive, print_next_steps
from ..config_utils import save_config
from ..constants import CONFIG_FILE_NAME
from ..validators import print_next_steps_with_deps, validate_and_warn_dependencies


def init_command() -> None:
    """Create a customized configuration file through interactive prompts."""
    output_path = Path.cwd() / CONFIG_FILE_NAME

    # Check if config already exists
    if output_path.exists():
        rich_print(f"[yellow]'{CONFIG_FILE_NAME}' already exists.[/yellow]")
        if not typer.confirm("Overwrite it?"):
            rich_print("Initialization cancelled.")
            return

    # Build configuration interactively
    try:
        config_dict = build_config_interactive()
    except (EOFError, KeyboardInterrupt, typer.Abort):
        # Handle non-interactive environment: use defaults from PipelineConfig
        rich_print("[yellow]Interactive mode not available. Using default configuration.[/yellow]")
        config_dict = PipelineConfig.generate_yaml_dict()
        rich_print("[blue]Loaded default configuration from PipelineConfig.[/blue]")
    except Exception as err:
        rich_print(f"[red]Error creating config: {err}[/red]")
        raise typer.Exit(code=1) from err

    # Validate dependencies BEFORE saving
    rich_print("\n[bold cyan]Validating dependencies...[/bold cyan]")
    deps_valid = validate_and_warn_dependencies(config_dict, interactive=True)

    # Save configuration
    try:
        save_config(config_dict, output_path)
        rich_print(f"[green]Successfully created {output_path}[/green]")
    except Exception as err:
        rich_print(f"[red]Error saving config: {err}[/red]")
        raise typer.Exit(code=1) from err

    # Print next steps
    print_next_steps(config_dict)
    print_next_steps_with_deps(config_dict)

    if not deps_valid:
        rich_print(
            "[yellow]Note: Install the dependencies above before running conversions.[/yellow]"
        )
