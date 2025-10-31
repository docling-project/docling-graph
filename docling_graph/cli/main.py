"""
Main CLI application setup and entry point.
"""

from .commands.convert import convert_command
from .commands.inspect import inspect_command
from .commands.init import init_command

from pathlib import Path
import typer


app = typer.Typer(
    name="docling-graph",
    help="A tool to convert documents into knowledge graphs using configurable pipelines.",
    add_completion=False
)

# Register commands
app.command(
    name="init",
    help="Create a default config.yaml in the current directory with interactive setup."
)(init_command)

app.command(
    name="convert", 
    help="Convert a document to a knowledge graph."
)(convert_command)

app.command(
    name="inspect",
    help="Visualize graph data in the browser."
)(inspect_command)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
