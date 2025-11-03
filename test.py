from docling_graph import run_pipeline, PipelineConfig
from examples.templates.battery_research import Research  # Pydantic model to use as an extraction template

# Create typed config
config = PipelineConfig(
    source="examples/data/battery_research/bauer2014.pdf",
    template=Research,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    provider_override="mistral",
    model_override="mistral-medium-latest",
    output_dir="outputs/battery_research"
)

try:
    run_pipeline(config)
    print(f"\nExtraction complete! Graph data saved to: {config.output_dir}")
except Exception as e:
    print(f"An error occurred: {e}")