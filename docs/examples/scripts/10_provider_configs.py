"""
Example 10: Multi-Provider Configurations

Description:
    Demonstrates how to configure and use different LLM providers (OpenAI, Mistral,
    Gemini, watsonx). Shows provider-specific settings and model selection.

Use Cases:
    - Comparing provider performance
    - Switching between providers
    - Cost optimization
    - Provider-specific features

Prerequisites:
    - Installation: uv sync  # LiteLLM is included by default
    - API Keys: Set environment variables for desired providers
      - OPENAI_API_KEY
      - MISTRAL_API_KEY
      - GEMINI_API_KEY
      - WATSONX_API_KEY, WATSONX_PROJECT_ID

Key Concepts:
    - Provider Selection: Choose LLM provider
    - Model Override: Specify exact model
    - API Configuration: Provider-specific settings
    - Cost vs Quality: Trade-offs between providers

Expected Output:
    - Separate outputs per provider
    - Performance comparison
    - Cost estimation

Related Examples:
    - Example 02: Basic remote LLM
    - Example 07: Local inference
    - Documentation: https://ibm.github.io/docling-graph/reference/llm-clients/
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from examples.templates.rheology_research import ScholarlyRheologyPaper

    from docling_graph import PipelineConfig
except ImportError:
    rich_print("[red]Error:[/red] Could not import required modules.")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

SOURCE_FILE = "docs/examples/data/research_paper/rheology.pdf"
TEMPLATE_CLASS = ScholarlyRheologyPaper

console = Console()


def get_provider_configs() -> List[Tuple[str, str, str, str]]:
    """
    Get available provider configurations.

    Returns:
        List of (name, provider, model, env_var) tuples
    """
    return [
        ("OpenAI GPT-4", "openai", "gpt-4-turbo-preview", "OPENAI_API_KEY"),
        ("Mistral Large", "mistral", "mistral-large-latest", "MISTRAL_API_KEY"),
        ("Google Gemini", "gemini", "gemini-1.5-pro", "GEMINI_API_KEY"),
        ("IBM watsonx", "watsonx", "ibm/granite-4-h-small", "WATSONX_API_KEY"),
        # Bedrock uses the AWS credential chain; region lives in AWS_REGION_NAME.
        (
            "Amazon Bedrock",
            "bedrock",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "AWS_ACCESS_KEY_ID",
        ),
    ]


def check_api_key(env_var: str) -> bool:
    """Check if API key is set."""
    return bool(os.getenv(env_var))


def process_with_provider(name: str, provider: str, model: str) -> Tuple[bool, str]:
    """
    Process document with specific provider.

    Returns:
        Tuple of (success, message)
    """
    try:
        config = PipelineConfig(
            source=SOURCE_FILE,
            template=TEMPLATE_CLASS,
            backend="llm",
            inference="remote",
            provider_override=provider,
            model_override=model,
            processing_mode="many-to-one",
            use_chunking=True,
        )

        config.run()
        return True, f"Success with {name}"

    except Exception as e:
        return False, f"Failed with {name}: {e!s}"


def main() -> None:
    """Execute multi-provider demonstration."""
    console.print(
        Panel.fit(
            "[bold blue]Example 10: Multi-Provider Configurations[/bold blue]\n"
            "[dim]Compare different LLM providers and models[/dim]",
            border_style="blue",
        )
    )

    # Get provider configurations
    providers = get_provider_configs()

    console.print("\n[yellow]📋 Available Providers:[/yellow]")
    for name, provider, model, env_var in providers:
        has_key = check_api_key(env_var)
        status = "[green]✓[/green]" if has_key else "[red]✗[/red]"
        console.print(f"  {status} {name} ({provider}/{model})")

    # Check which providers are configured
    configured = [(n, p, m, e) for n, p, m, e in providers if check_api_key(e)]

    if not configured:
        console.print("\n[red]No API keys configured![/red]")
        console.print("\n[yellow]Setup Instructions:[/yellow]")
        console.print("  • OpenAI: [cyan]export OPENAI_API_KEY='sk-...'[/cyan]")
        console.print("  • Mistral: [cyan]export MISTRAL_API_KEY='...'[/cyan]")
        console.print("  • Gemini: [cyan]export GEMINI_API_KEY='...'[/cyan]")
        console.print("  • watsonx: [cyan]export WATSONX_API_KEY='...'[/cyan]")
        console.print("            [cyan]export WATSONX_PROJECT_ID='...'[/cyan]")
        console.print("  • Bedrock: [cyan]export AWS_ACCESS_KEY_ID='...'[/cyan]")
        console.print("            [cyan]export AWS_SECRET_ACCESS_KEY='...'[/cyan]")
        console.print("            [cyan]export AWS_REGION_NAME='us-east-1'[/cyan]")
        sys.exit(1)

    console.print(f"\n[green]Found {len(configured)} configured provider(s)[/green]")
    console.print("\n[yellow]⚙️  Processing with each provider...[/yellow]")

    # Process with each configured provider
    results = []
    for name, provider, model, _ in configured:
        console.print(f"\n[cyan]Processing with {name}...[/cyan]")

        success, message = process_with_provider(name, provider, model)
        results.append((name, provider, model, success, message))

        if success:
            console.print(f"  [green]✓[/green] {message}")
        else:
            console.print(f"  [red]✗[/red] {message}")

    # Provider comparison table
    console.print("\n[bold]📊 Provider Comparison:[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Strengths")
    table.add_column("Pricing")

    table.add_row("OpenAI", "GPT-4 Turbo", "Highest quality, best reasoning", "$$$ (Premium)")
    table.add_row("Mistral", "Mistral Large", "Good balance, fast", "$$ (Moderate)")
    table.add_row("Gemini", "Gemini 1.5 Pro", "Long context, multimodal", "$$ (Moderate)")
    table.add_row("watsonx", "Granite", "Enterprise, compliance", "$ (Enterprise)")
    table.add_row("Bedrock", "Claude 3.5 Sonnet", "AWS-native, IAM auth", "$$ (AWS pricing)")

    console.print(table)

    console.print("\n[bold]💡 Provider Selection Guide:[/bold]")
    console.print("\n[cyan]Choose OpenAI when:[/cyan]")
    console.print("  • Highest quality is required")
    console.print("  • Complex reasoning tasks")
    console.print("  • Budget allows premium pricing")

    console.print("\n[cyan]Choose Mistral when:[/cyan]")
    console.print("  • Good balance of quality and cost")
    console.print("  • Fast inference needed")
    console.print("  • European data residency preferred")

    console.print("\n[cyan]Choose Gemini when:[/cyan]")
    console.print("  • Very long documents (2M tokens)")
    console.print("  • Multimodal capabilities needed")
    console.print("  • Google Cloud integration")

    console.print("\n[cyan]Choose watsonx when:[/cyan]")
    console.print("  • Enterprise compliance required")
    console.print("  • IBM ecosystem integration")
    console.print("  • Custom model fine-tuning")

    console.print("\n[bold]🔧 Model Selection Tips:[/bold]")
    console.print("  • Larger models: Better quality, higher cost")
    console.print("  • Smaller models: Faster, lower cost")
    console.print("  • Test with small model first")
    console.print("  • Use larger model for production")

    console.print("\n[bold]📊 Output Locations:[/bold]")
    for name, provider, _, success, _ in results:
        if success:
            console.print(f"  • {name}: [cyan]outputs/10_provider_configs/{provider}/[/cyan]")

    console.print("\n[bold]🔍 Compare Results:[/bold]")
    console.print("  [cyan]# Compare extraction quality[/cyan]")
    console.print("  [dim]diff outputs/10_provider_configs/openai/docling_graph/nodes.csv \\[/dim]")
    console.print("       [dim]outputs/10_provider_configs/mistral/docling_graph/nodes.csv[/dim]")


if __name__ == "__main__":
    main()
