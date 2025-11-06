"""
Example 05: LLM with Final Consolidation Step
"""

import sys
from pathlib import Path

from rich import print as rich_print

# --- 1. Add project root to sys.path ---
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from examples.templates.battery_research import Research

    from docling_graph import PipelineConfig
except ImportError:
    rich_print("[red]Error:[/red] Could not import modules. ")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

# --- 2. Define Paths (relative to project root) ---
SOURCE_FILE = "docs/examples/data/battery_research/bauer2014.pdf"
TEMPLATE_CLASS = Research
OUTPUT_DIR = "outputs/05_llm_with_consolidation"

rich_print("[blue]--- Running Example 05: LLM Consolidation ---[/blue]")

# --- 3. Configure the Pipeline ---
config = PipelineConfig(
    source=SOURCE_FILE,
    template=TEMPLATE_CLASS,
    output_dir=OUTPUT_DIR,
    # --- Key Settings ---
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-small-latest",
    processing_mode="many-to-one",
    use_chunking=True,
    # This is the key difference from Example 03
    llm_consolidation=True,
)

# --- 4. Run the pipeline ---
try:
    config.run()
    rich_print(f"\n[green]Success![/green] Graph data saved to: {OUTPUT_DIR}")
    rich_print(f"Explore: {OUTPUT_DIR}/{Path(SOURCE_FILE).stem}_graph.html")
except Exception as e:
    rich_print(f"\n[red]Pipeline Failed:[/red] {e}")
