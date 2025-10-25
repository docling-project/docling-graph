
"""
Configuration loading and validation utilities.
"""

from typing import Any, Dict
from pathlib import Path
from rich import print
import typer
import yaml

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
        print(f"[red]Error:[/red] Configuration file '{CONFIG_FILE_NAME}' not found.")
        print(f"Please run [cyan]docling-graph init[/cyan] first.")
        raise typer.Exit(code=1)

    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"[red]Error parsing '{CONFIG_FILE_NAME}':[/red] {e}")
        raise typer.Exit(code=1)


def save_config(config_dict: Dict[str, Any], output_path: Path) -> None:
    """Save configuration dictionary to YAML file.

    Args:
        config_dict: Configuration dictionary to save.
        output_path: Path where to save the config file.

    Raises:
        Exception: If writing fails.
    """
    with open(output_path, 'w') as f:
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
        >>> get_config_value(config, "models", "llm", "local", "default_model")
    """
    current = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current
