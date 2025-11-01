"""
Init command - creates configuration file interactively.
"""

from pathlib import Path

import typer
import yaml
from rich import print

from ..config_utils import save_config
from ..constants import CONFIG_FILE_NAME
from .config_builder import build_config_interactive, print_next_steps


def init_command():
    """Create a customized configuration file through interactive prompts."""
    output_path = Path.cwd() / CONFIG_FILE_NAME

    # Check if config already exists
    if output_path.exists():
        print(f"[yellow]'{CONFIG_FILE_NAME}' already exists.[/yellow]")
        if not typer.confirm("Overwrite it?"):
            print("Initialization cancelled.")
            return  # Return normally (exit code 0)

    # Build configuration interactively
    try:
        config_dict = build_config_interactive()

    except (EOFError, KeyboardInterrupt, typer.Abort):
        # Handle non-interactive environment gracefully
        print("[yellow]Interactive mode not available. Using default configuration.[/yellow]")

        # Load default config from template
        template_path = Path(__file__).parent.parent.parent / "config_template.yaml"

        if template_path.exists():
            with open(template_path) as f:
                config_dict = yaml.safe_load(f)
            print("[blue]Loaded default configuration from template.[/blue]")
        else:
            # Minimal fallback config if template not found
            config_dict = {
                "defaults": {
                    "processing_mode": "many-to-one",
                    "backend_type": "llm",
                    "inference": "local",
                    "export_format": "csv",
                },
                "docling": {"pipeline": "default"},
                "models": {
                    "vlm": {
                        "local": {"default_model": "numind/NuExtract-2.0-8B", "provider": "docling"}
                    },
                    "llm": {"local": {"default_model": "llama3:8b-instruct", "provider": "ollama"}},
                },
                "output": {"directory": "./output"},
            }
            print("[blue]Using minimal default configuration.[/blue]")

    except Exception as e:
        print(f"[red]Error creating config:[/red] {e}")
        raise typer.Exit(code=1)

    # Save configuration
    try:
        save_config(config_dict, output_path)
        print(f"\n[green]Successfully created '{output_path}'[/green]")

        # Print next steps
        print_next_steps(config_dict)

    except Exception as e:
        print(f"[red]Error saving config:[/red] {e}")
        raise typer.Exit(code=1)
