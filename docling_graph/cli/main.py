"""
Main CLI application setup and entry point.
"""

import logging
from pathlib import Path

import typer

from .commands.convert import convert_command
from .commands.init import init_command
from .commands.inspect import inspect_command

logging.basicConfig(level=logging.WARNING)

# Suppress noisy INFO logs from RapidOCR (used by docling OCR pipeline)
for _logger_name in ("RapidOCR", "rapidocr"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        from docling_graph import __version__

        typer.echo(f"docling-graph version: {__version__}")
        raise typer.Exit()


def verbose_callback(ctx: typer.Context, value: bool) -> bool:
    """Configure logging based on verbose flag."""
    if value:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.getLogger("docling_graph").setLevel(logging.DEBUG)
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )
    return value


app = typer.Typer(
    name="docling-graph",
    help="A tool to convert documents (PDF, images, text, markdown, URLs) into knowledge graphs using configurable pipelines.",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def main_callback(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output with detailed logging",
        callback=verbose_callback,
        is_eager=True,
    ),
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """
    Docling-Graph CLI - Convert documents to knowledge graphs.

    Global options can be used with any command.
    """


# Register commands
app.command(
    name="init",
    help="Create a default config.yaml in the current directory with interactive setup.",
)(init_command)

app.command(
    name="convert",
    help="Convert a document (PDF, image, text, markdown, URL) to a knowledge graph.",
)(convert_command)

app.command(name="inspect", help="Visualize graph data in the browser.")(inspect_command)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
