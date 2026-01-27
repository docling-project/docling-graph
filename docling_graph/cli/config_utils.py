"""
Configuration loading and validation utilities.
"""

from pathlib import Path
from typing import Any, Dict, cast

import typer
import yaml
from rich import print as rich_print

from .constants import CONFIG_FILE_NAME


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file.

    Returns:
        Dictionary containing configuration data.

    Raises:
        typer.Exit: If config file doesn't exist or has errors.
    """
    config_path = Path.cwd() / CONFIG_FILE_NAME

    if not config_path.exists():
        rich_print(f"[red]Error:[/red] Configuration file '{CONFIG_FILE_NAME}' not found.")
        rich_print("Please run [cyan]docling-graph init[/cyan] first.")
        raise typer.Exit(code=1)

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
            # Allow empty config files: treat as empty dict
            if data is None:
                return {}
            if isinstance(data, dict):
                return cast(Dict[str, Any], data)
            rich_print(
                f"[red]Error:[/red] Configuration file '{CONFIG_FILE_NAME}' must contain a mapping at the top level."
            )
            raise typer.Exit(code=1)
    except yaml.YAMLError as err:
        rich_print(f"[red]Error parsing '{CONFIG_FILE_NAME}':[/red] {err}")
        raise typer.Exit(code=1) from err


def save_config(config_dict: Dict[str, Any], output_path: Path) -> None:
    """Save configuration dictionary to YAML file.

    Args:
        config_dict: Configuration dictionary to save.
        output_path: Path where to save the config file.

    Raises:
        Exception: If writing fails.
    """
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)


def get_config_value(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get nested configuration value.

    Args:
        config: Configuration dictionary.
        *keys: Nested keys to traverse.
        default: Default value if key not found.

    Returns:
        Configuration value or default.

    Example:
        >>> get_config_value(config, "models", "llm", "local", "model")
    """
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        next_val = current.get(key)
        if next_val is None:
            return default
        current = next_val
    return current
