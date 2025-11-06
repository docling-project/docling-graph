"""
Example 07: LLM Extraction Without Chunking
"""

import sys
from pathlib import Path

from rich import print as rich_print

# --- 1. Add project root to sys.path ---
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from examples.templates.insurance import InsuranceTerms

    from docling_graph import PipelineConfig
except ImportError:
    rich_print("[red]Error:[/red] Could not import modules. ")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

# --- 2. Define Paths (relative to project root) ---
SOURCE_FILE = "docs/examples/data/insurance/insurance_terms_sp.pdf"
TEMPLATE_CLASS = InsuranceTerms
OUTPUT_DIR = "outputs/07_llm_no_chunking"

rich_print("[blue]--- Running Example 07: LLM No Chunking ---[/blue]")

# --- 3. Configure the Pipeline ---
config = PipelineConfig(
    source=SOURCE_FILE,
    template=TEMPLATE_CLASS,
    output_dir=OUTPUT_DIR,
    # --- Key Settings ---
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3:8b",
    processing_mode="many-to-one",
    # Process the entire document text at once
    use_chunking=False,
)

# --- 4. Run the pipeline ---
try:
    config.run()
    rich_print(f"\n[green]Success![/green] Graph data saved to: {OUTPUT_DIR}")
    rich_print(f"Explore: {OUTPUT_DIR}/{Path(SOURCE_FILE).stem}_graph.html")
except Exception as e:
    rich_print(f"\n[red]Pipeline Failed:[/red] {e}")
