"""
Example: Using IBM WatsonX for document extraction with docling-graph.

This example demonstrates how to use IBM WatsonX's Granite models
for extracting structured data from documents.

Prerequisites:
    1. Install WatsonX support: uv sync --extra watsonx
    2. Set environment variables:
       - WATSONX_API_KEY: Your IBM Cloud API key
       - WATSONX_PROJECT_ID: Your WatsonX project ID
       - WATSONX_URL: (Optional) Custom endpoint URL
"""

import sys
from pathlib import Path

# Add the workspace root to Python path to enable imports
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from docling_graph import PipelineConfig, run_pipeline
from examples.templates.rheology_research import Research

# Define Source document :
source_doc="docs/examples/data/research_paper/rheology.pdf"

# Example 1: Using WatsonX with default Granite model
config_default: PipelineConfig = PipelineConfig(
    source=source_doc,
    template=Research,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",  # Change to one-to-one if needed
    provider_override="watsonx",
    model_override="ibm/granite-4-h-small",  # Default Granite model
    output_dir="outputs/watsonx_example_default",
)

def main() -> None:
    """Run the WatsonX extraction examples."""
    print("=" * 80)
    print("IBM WatsonX Document Extraction Examples")
    print("=" * 80)

    # Run Example 1: Default Granite model
    print("\n[1/1] Running extraction with Granite 4.0 H Small...")
    try:
        run_pipeline(config_default)
        print(f"✓ Success! Output saved to: {config_default.output_dir}")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()

# Made with Bob, Checked by Guilhaume
