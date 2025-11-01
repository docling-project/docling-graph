"""
Inspect command - visualizes graph data in browser.
"""

from pathlib import Path
from typing import Optional

import typer
from rich import print
from typing_extensions import Annotated

from ...core.visualizers.interactive_visualizer import InteractiveVisualizer


def inspect_command(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to graph data. For CSV: directory with nodes.csv and edges.csv. For JSON: path to .json file.",
            exists=True,
        ),
    ],
    format: Annotated[
        str, typer.Option("--format", "-f", help="Import format: 'csv' or 'json'.")
    ] = "csv",
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o", help="Output HTML file path. If not specified, uses temporary file."
        ),
    ] = None,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Automatically open browser.")
    ] = True,
) -> None:
    """
    Visualize graph data in the browser.

    This command creates an interactive HTML visualization that opens
    in your default web browser. The HTML file is self-contained and
    can be shared or saved for later viewing.

    Examples:
        # Visualize CSV format (default) - opens in browser
        docling-graph inspect ./output_dir

        # Visualize JSON format
        docling-graph inspect graph.json --format json

        # Save to specific location
        docling-graph inspect ./output_dir --output graph_viz.html

        # Create HTML without opening browser
        docling-graph inspect ./output_dir --no-open --output viz.html
    """

    # Validate format
    format = format.lower()
    if format not in ["csv", "json"]:
        print(f"[bold red]Error:[/bold red] Format must be 'csv' or 'json', got '{format}'")
        raise typer.Exit(code=1)

    # Validate path based on format
    if format == "csv":
        if not path.is_dir():
            print(
                "[bold red]Error:[/bold red] For CSV format, path must be a directory containing nodes.csv and edges.csv"
            )
            raise typer.Exit(code=1)

        nodes_path = path / "nodes.csv"
        edges_path = path / "edges.csv"

        if not nodes_path.exists():
            print(f"[bold red]Error:[/bold red] nodes.csv not found in {path}")
            raise typer.Exit(code=1)

        if not edges_path.exists():
            print(f"[bold red]Error:[/bold red] edges.csv not found in {path}")
            raise typer.Exit(code=1)

    elif format == "json":
        if not path.is_file() or path.suffix != ".json":
            print("[bold red]Error:[/bold red] For JSON format, path must be a .json file")
            raise typer.Exit(code=1)

    print("--- [blue]Starting Docling-Graph Inspection[/blue] ---")
    print("\n[bold]Interactive Visualization[/bold]")
    print(f"  Input: [cyan]{path}[/cyan]")
    print(f"  Format: [cyan]{format}[/cyan]")
    if output:
        print(f"  Output: [cyan]{output}[/cyan]")
    else:
        print("  Output: [cyan]temporary file[/cyan]")

    try:
        # Create visualizer
        visualizer = InteractiveVisualizer()

        # Load and visualize
        print("\nLoading graph data...")
        visualizer.display_cytoscape_graph(
            path=path, format=format, output_path=output, open_browser=open_browser
        )

        print("--- [blue]Docling-Graph Inspection Finished Successfully[/blue] ---")

        if not open_browser:
            print(
                "\n[blue]Tip:[/blue] Open the HTML file in your browser to view the visualization"
            )

    except Exception as e:
        print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)
