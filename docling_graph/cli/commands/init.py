"""
Init command - creates configuration file interactively.
"""

from pathlib import Path

import typer
from rich import print as rich_print

from docling_graph.config import PipelineConfig

from ..config_builder import build_config_interactive, print_next_steps
from ..config_utils import save_config
from ..constants import CONFIG_FILE_NAME
from ..validators import (
    print_next_steps_with_deps,
    validate_and_warn_dependencies,
)


def init_command() -> None:
    """Create a customized configuration file through interactive prompts."""
    output_path = Path.cwd() / CONFIG_FILE_NAME

    # Check if config already exists
    if output_path.exists():
        rich_print(f"[yellow]A configuration file: '{CONFIG_FILE_NAME}' already exists.[/yellow]")
        if not typer.confirm("Overwrite it?"):
            rich_print("Initialization cancelled.")
            return

    # Build configuration
    config_dict = _build_config_safe()
    if config_dict is None:
        raise typer.Exit(code=1)

    # Validate dependencies
    rich_print("\n[bold cyan]Validating dependencies...[/bold cyan]")
    deps_valid = validate_and_warn_dependencies(config_dict)

    # Save configuration
    if not _save_config_safe(config_dict, output_path):
        raise typer.Exit(code=1)

    # Print next steps (consolidated logic handles dependency installation)
    _print_final_steps(config_dict, deps_valid)


def _build_config_safe() -> dict | None:
    """Safely build configuration with fallback to defaults."""
    try:
        return build_config_interactive()
    except (EOFError, KeyboardInterrupt, typer.Abort):
        rich_print("[yellow]Interactive mode not available. Using default configuration.[/yellow]")
        config = PipelineConfig.generate_yaml_dict()
        rich_print("[blue]Loaded default configuration.[/blue]")
        return config
    except Exception as err:
        rich_print(f"[red]Error creating config: {err}[/red]")
        return None


def _save_config_safe(config_dict: dict, output_path: Path) -> bool:
    """Safely save configuration file."""
    try:
        save_config(config_dict, output_path)
        rich_print(f"[green]Config successfully initiated at: {output_path}[/green]")
        return True
    except Exception as err:
        rich_print(f"[red]Error saving config: {err}[/red]")
        return False


def _print_final_steps(config_dict: dict, deps_valid: bool) -> None:
    """Print final next steps, handling dependency installation if needed."""
    next_steps = print_next_steps(config_dict, return_text=True)

    if deps_valid:
        # Dependencies are already installed, just print steps
        rich_print(next_steps)
    else:
        if next_steps is None:
            next_steps = ""
        # Dependencies missing, use the function that prepends install step
        print_next_steps_with_deps(config_dict, next_steps)
