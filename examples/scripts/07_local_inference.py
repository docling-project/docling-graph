"""
Example 07: Local Inference - Ollama and vLLM

Description:
    Demonstrates local LLM inference using Ollama for privacy-focused,
    offline document processing. No API keys or internet required after setup.

Use Cases:
    - Privacy-sensitive documents
    - Offline processing
    - Cost-free inference
    - Development and testing

Prerequisites:
    - Installation: uv sync
    - Ollama: Install from https://ollama.ai
    - Model: ollama pull llama3:8b
    - Start server: ollama serve

Key Concepts:
    - Local Inference: Runs entirely on your machine
    - Ollama: Easy local LLM deployment
    - No API Costs: Free inference
    - Privacy: Data never leaves your machine

Expected Output:
    - Same outputs as remote inference
    - Processed entirely locally
    - No network calls to external APIs

Related Examples:
    - Example 02: Remote inference comparison
    - Example 10: Multi-provider configurations
    - Documentation: https://ibm.github.io/docling-graph/fundamentals/installation/local-setup/
"""

import sys
from pathlib import Path

from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from examples.templates.rheology_research import ScholarlyRheologyPaper

    from docling_graph import PipelineConfig, run_pipeline
except ImportError:
    rich_print("[red]Error:[/red] Could not import required modules.")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

SOURCE_FILE = "docs/examples/data/research_paper/rheology.pdf"
TEMPLATE_CLASS = ScholarlyRheologyPaper
console = Console()


def main() -> None:
    """Execute local LLM extraction."""
    console.print(
        Panel.fit(
            "[bold blue]Example 07: Local Inference with Ollama[/bold blue]\n"
            "[dim]Process documents locally without API keys or internet[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[yellow]📋 Configuration:[/yellow]")
    console.print(f"  • Source: [cyan]{SOURCE_FILE}[/cyan]")
    console.print(f"  • Template: [cyan]{TEMPLATE_CLASS.__name__}[/cyan]")
    console.print("  • Backend: [cyan]LLM (Local)[/cyan]")
    console.print("  • Provider: [cyan]Ollama[/cyan]")
    console.print("  • Model: [cyan]llama3:8b[/cyan]")

    console.print("\n[yellow]⚠️  Prerequisites:[/yellow]")
    console.print("  1. Install Ollama: [cyan]https://ollama.ai[/cyan]")
    console.print("  2. Pull model: [cyan]ollama pull llama3:8b[/cyan]")
    console.print("  3. Start server: [cyan]ollama serve[/cyan]")
    console.print("  4. Install dependencies: [cyan]uv sync[/cyan]")

    console.print("\n[bold]💡 Benefits of Local Inference:[/bold]")
    console.print("  • ✅ Complete privacy - data never leaves your machine")
    console.print("  • ✅ No API costs - free inference")
    console.print("  • ✅ Offline capable - no internet required")
    console.print("  • ✅ Full control - choose any model")
    console.print("  • ⚠️  Slower than cloud APIs (depends on hardware)")
    console.print("  • ⚠️  Requires GPU for good performance")

    try:
        config = PipelineConfig(
            source=SOURCE_FILE,
            template=TEMPLATE_CLASS,
            backend="llm",
            inference="local",  # Local inference
            provider_override="ollama",  # Use Ollama
            model_override="llama3:8b",  # Llama 3 8B model
            processing_mode="many-to-one",
            use_chunking=True,
            llm_consolidation=False,
        )

        console.print("\n[yellow]⚙️  Processing locally (may take 3-5 minutes)...[/yellow]")
        console.print("  • Converting PDF to markdown")
        console.print("  • Chunking document")
        console.print("  • Processing with local Ollama")
        console.print("  • Building knowledge graph")

        context = run_pipeline(config)

        console.print("\n[green]✓ Success![/green]")
        graph = context.knowledge_graph
        console.print(
            f"\n[bold]Extracted:[/bold] [cyan]{graph.number_of_nodes()} nodes[/cyan] "
            f"and [cyan]{graph.number_of_edges()} edges[/cyan]"
        )

        console.print("\n[bold]🎯 Local vs Remote Comparison:[/bold]")
        console.print("  • Speed: Remote APIs typically faster")
        console.print("  • Cost: Local is free, remote has API costs")
        console.print("  • Privacy: Local keeps all data on your machine")
        console.print("  • Quality: Depends on model size and task")

        console.print("\n[bold]🔧 Other Local Options:[/bold]")
        console.print("  • vLLM: For production deployments")
        console.print("  • llama.cpp: For CPU-only inference")
        console.print("  • See Example 10 for multi-provider setup")

    except Exception as e:
        error_msg = str(e).lower()
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")

        if "ollama" in error_msg or "connection" in error_msg:
            console.print("  • Start Ollama server: [cyan]ollama serve[/cyan]")
            console.print("  • Pull model: [cyan]ollama pull llama3:8b[/cyan]")
            console.print("  • Check status: [cyan]ollama list[/cyan]")
            console.print("  • Verify Ollama is running: [cyan]curl http://localhost:11434[/cyan]")
        else:
            console.print("  • Install dependencies: [cyan]uv sync[/cyan]")
            console.print("  • Check GPU availability for better performance")
            console.print("  • Try smaller model: [cyan]ollama pull llama3:8b[/cyan]")

        sys.exit(1)


if __name__ == "__main__":
    main()
