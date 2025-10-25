# File: pipeline.py
# Location: docling_graph/pipeline.py
# Description: Main extraction and graph conversion pipeline (REFACTORED)
"""
Main extraction and graph conversion pipeline.

This module orchestrates the complete workflow from document extraction
to graph generation, export, and visualization using the refactored graph module.
"""
from typing import Dict, Any, Optional
from pathlib import Path
from rich import print
import importlib

# Import LLM Clients
from .llm_clients.mistral import MistralClient
from .llm_clients.ollama import OllamaClient
from .llm_clients.llm_base import BaseLlmClient

# Import Extractors
from .extractors.factory import ExtractorFactory

# Import REFACTORED graph module - ALL components
from .graph import (
    GraphConverter,
    GraphConfig,
    VisualizationConfig,
    CSVExporter,
    CypherExporter,
    JSONExporter,
    StaticVisualizer,
    InteractiveVisualizer,
    ReportGenerator,
)


def _load_template_class(template_str: str):
    """Dynamically imports a Pydantic model class from a string.

    Args:
        template_str: Dotted path to Pydantic model class.

    Returns:
        Pydantic model class.

    Raises:
        Exception: If template cannot be loaded.
    """
    print(f"Loading template: [yellow]{template_str}[/yellow]")

    try:
        module_path, class_name = template_str.rsplit('.', 1)
        module = importlib.import_module(module_path)
        TemplateClass = getattr(module, class_name)
        print(f"[green]Successfully loaded Pydantic template: {class_name}[/green]")
        return TemplateClass
    except Exception as e:
        print(f"[red]Failed to load template {template_str}:[/red] {e}")
        raise


def _get_model_config(
    config_data: Dict[str, Any],
    backend_type: str,
    inference: str,
    model_override: Optional[str] = None,
    provider_override: Optional[str] = None
) -> Dict[str, str]:
    """Retrieves the appropriate model configuration based on settings.

    Args:
        config_data: Configuration dictionary.
        backend_type: Backend type ('llm' or 'vlm').
        inference: Inference location ('local' or 'remote').
        model_override: Optional model name override.
        provider_override: Optional provider override.

    Returns:
        Dictionary with 'model' and 'provider' keys.

    Raises:
        ValueError: If configuration is invalid.
    """
    model_config = config_data.get('models', {}).get(backend_type, {}).get(inference, {})

    if not model_config:
        raise ValueError(
            f"No configuration found for backend_type='{backend_type}' "
            f"with inference='{inference}'"
        )

    provider = provider_override or model_config.get(
        'provider',
        'ollama' if inference == 'local' else 'mistral'
    )

    if model_override:
        model = model_override
    elif provider_override and inference == 'remote':
        providers = model_config.get('providers', {})
        model = providers.get(provider_override, {}).get(
            'default_model',
            model_config.get('default_model')
        )
    else:
        model = model_config.get('default_model')

    return {'model': model, 'provider': provider}


def _initialize_llm_client(provider: str, model: str) -> BaseLlmClient:
    """Initializes an LLM client based on provider.

    Args:
        provider: Provider name ('mistral', 'ollama', etc.).
        model: Model name.

    Returns:
        Initialized LLM client.

    Raises:
        ValueError: If provider is unknown.
    """
    if provider == "mistral":
        return MistralClient(model=model)
    elif provider == "ollama":
        return OllamaClient(model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def run_pipeline(config: Dict[str, Any]) -> None:
    """Runs the extraction and graph conversion pipeline.

    Args:
        config: Pipeline configuration dictionary with keys:
            - source: Path to source document
            - template: Dotted path to Pydantic template
            - processing_mode: 'one-to-one' or 'many-to-one'
            - backend_type: 'llm' or 'vlm'
            - inference: 'local' or 'remote'
            - docling_config: Docling pipeline config ('ocr' or 'vision')
            - reverse_edges: Whether to add reverse edges
            - output_dir: Output directory path
            - export_format: Export format ('csv', 'cypher', 'json')
            - config: Nested config with models, etc.
            - model_override: Optional model override
            - provider_override: Optional provider override
    """
    print("--- [blue]Starting Docling-Graph Pipeline (Refactored)[/blue] ---")

    processing_mode = config.get("processing_mode")
    backend_type = config.get("backend_type")
    inference = config.get("inference")
    docling_config = config.get("docling_config", "ocr")
    reverse_edges = config.get("reverse_edges", False)

    # Initialize variables for cleanup
    extractor = None
    llm_client = None

    try:
        # 1. Load Template
        try:
            TemplateClass = _load_template_class(config["template"])
        except Exception:
            return

        # 2. Get model configuration
        try:
            model_config = _get_model_config(
                config.get('config', {}),
                backend_type,
                inference,
                config.get('model_override'),
                config.get('provider_override')
            )
            print(f"Using model: [cyan]{model_config['model']}[/cyan] "
                  f"(provider: {model_config['provider']})")
        except Exception as e:
            print(f"[red]Configuration error:[/red] {e}")
            return

        # 3. Create extractor using factory
        try:
            if backend_type == "vlm":
                extractor = ExtractorFactory.create_extractor(
                    processing_mode=processing_mode,
                    backend_type=backend_type,
                    model_name=model_config['model'],
                    docling_config=docling_config
                )
            elif backend_type == "llm":
                llm_client = _initialize_llm_client(
                    model_config['provider'],
                    model_config['model']
                )
                extractor = ExtractorFactory.create_extractor(
                    processing_mode=processing_mode,
                    backend_type=backend_type,
                    llm_client=llm_client,
                    docling_config=docling_config
                )
            else:
                print(f"[red]Error:[/red] Invalid backend_type: {backend_type}")
                return
        except Exception as e:
            print(f"[red]Failed to create extractor:[/red] {e}")
            return

        # 4. Run Extraction
        extracted_data = extractor.extract(config["source"], TemplateClass)

        if not extracted_data:
            print("[red]Pipeline stopped: Extraction returned no data.[/red]")
            return

        print(f"Successfully extracted {len(extracted_data)} item(s).")

        # 5. Convert to Graph using REFACTORED module
        print("Converting Pydantic model(s) to Knowledge Graph...")

        # Create graph config with custom settings
        graph_config = GraphConfig(add_reverse_edges=reverse_edges)

        # Create converter with config
        converter = GraphConverter(config=graph_config)

        # Convert models to graph (NOW RETURNS TUPLE!)
        knowledge_graph, graph_metadata = converter.pydantic_list_to_graph(extracted_data)

        print(f"Graph created with [blue]{graph_metadata.node_count} nodes[/blue] "
              f"and [blue]{graph_metadata.edge_count} edges[/blue].")

        # Display node type distribution
        if graph_metadata.node_types:
            print("Node types:")
            for node_type, count in graph_metadata.node_types.items():
                print(f"  - {node_type}: {count}")

        # 6. Save outputs
        output_dir = Path(config.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(config["source"]).stem
        output_path = output_dir / f"{base_name}_graph"

        # 7. Export graph using REFACTORED exporters
        export_format = config.get("export_format", "csv")
        print(f"Exporting graph data in [cyan]{export_format.upper()}[/cyan] format...")

        if export_format == "csv":
            exporter = CSVExporter()
            exporter.export(knowledge_graph, output_dir)
            print(f"[green]->[/green] Saved CSV files to [green]{output_dir}[/green]")

        elif export_format == "cypher":
            cypher_path = output_dir / f"{base_name}_graph.cypher"
            exporter = CypherExporter()
            exporter.export(knowledge_graph, cypher_path)
            print(f"[green]->[/green] Saved Cypher script to [green]{cypher_path}[/green]")

        elif export_format == "json":
            json_path = output_dir / f"{base_name}_graph.json"
            exporter = JSONExporter()
            exporter.export(knowledge_graph, json_path)
            print(f"[green]->[/green] Saved JSON to [green]{json_path}[/green]")

        # 8. Generate visualizations using REFACTORED visualizers
        print(f"[green]->[/green] Generating visualizations...")

        # Markdown report using NEW ReportGenerator
        report_generator = ReportGenerator()
        report_generator.visualize(
            knowledge_graph,
            output_path,
            source_model_count=len(extracted_data)
        )
        print(f"[green]->[/green] Generated markdown report")

        # Interactive visualization using NEW InteractiveVisualizer
        interactive_viz = InteractiveVisualizer()
        interactive_viz.visualize(knowledge_graph, output_path)
        print(f"[green]->[/green] Generated interactive HTML visualization")

        # Static visualizations using NEW StaticVisualizer
        static_viz = StaticVisualizer()

        # Generate PNG
        static_viz.visualize(knowledge_graph, output_path, format='png')
        print(f"[green]->[/green] Generated static PNG")

        # Generate SVG (optional, good for presentations)
        static_viz.visualize(
            knowledge_graph,
            Path(str(output_path) + '_vector'),
            format='svg'
        )
        print(f"[green]->[/green] Generated static SVG")

        print(f"[green]->[/green] All outputs saved to [green]{output_dir}[/green]")
        print("--- [blue]Pipeline Finished Successfully![/blue] ---")

    finally:
        # Cleanup resources
        print("Cleaning up resources...")

        # Clean up extractor and its components
        if extractor is not None:
            # Clean up backend (VLM or LLM)
            if hasattr(extractor, 'backend') and hasattr(extractor.backend, 'cleanup'):
                extractor.backend.cleanup()

            # Clean up document processor
            if hasattr(extractor, 'doc_processor') and hasattr(extractor.doc_processor, 'cleanup'):
                extractor.doc_processor.cleanup()

        # Clean up LLM client if used
        if llm_client is not None:
            del llm_client

        # Final garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
