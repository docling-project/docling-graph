"""
Example 03: Remote LLM (Mistral) on a Multi-Page PDF
"""

import sys
from pathlib import Path
from rich import print as rich_print


# --- 1. Add project root to sys.path ---
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from docling_graph import PipelineConfig
    from examples.templates.battery_research import Research
except ImportError:
    rich_print("[red]Error:[/red] Could not import modules. ")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

# --- 2. Define Paths (relative to project root) ---
SOURCE_FILE = "docs/examples/data/battery_research/bauer2014.pdf"
TEMPLATE_CLASS = Research
OUTPUT_DIR = "outputs/03_llm_remote_api"

rich_print(f"[blue]--- Running Example 03: Remote LLM (Mistral) ---[/blue]")

# --- 3. Configure the Pipeline ---
config = PipelineConfig(
    source=SOURCE_FILE,
    template=TEMPLATE_CLASS,
    output_dir=OUTPUT_DIR,

    # --- Key Settings ---
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-large-latest", # Use a powerful model
    
    # Strategy for multi-page docs
    processing_mode="many-to-one",
    use_chunking=True,         # Split the doc into manageable chunks
    llm_consolidation=False,   # Merge chunk results programmatically (default)
)

# --- 4. Run the pipeline ---
try:
    config.run()
    rich_print(f"\n[green]Success![/green] Graph data saved to: {OUTPUT_DIR}")
    rich_print(f"Explore: {OUTPUT_DIR}/{Path(SOURCE_FILE).stem}_graph.html")
except Exception as e:
    rich_print(f"\n[red]Pipeline Failed:[/red] {e}")
    rich_print("[yellow]Hint:[/yellow] Did you set the 'MISTRAL_API_KEY' env var?")