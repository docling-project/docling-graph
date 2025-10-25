"""
Init command - creates configuration file interactively.
"""

from pathlib import Path
from rich import print
import typer

from .config_builder import build_config_interactive, print_next_steps
from ..constants import CONFIG_FILE_NAME
from ..config_utils import save_config


def init_command():
    """Create a customized configuration file through interactive prompts."""
    output_path = Path.cwd() / CONFIG_FILE_NAME

    # Check if config already exists
    if output_path.exists():
        print(f"[yellow]'{CONFIG_FILE_NAME}' already exists.[/yellow]")
        if not typer.confirm("Overwrite it?"):
            print("Initialization cancelled.")
            raise typer.Abort()

    # Build configuration interactively
    try:
        config_dict = build_config_interactive()

        # Save configuration
        save_config(config_dict, output_path)

        print(f"\n[green]✓ Successfully created '{output_path}'[/green]")

        # Print next steps
        print_next_steps(config_dict)

    except Exception as e:
        print(f"[red]Error creating config:[/red] {e}")
        raise typer.Exit(code=1)
