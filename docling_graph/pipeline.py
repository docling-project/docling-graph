"""
Main extraction and graph conversion pipeline.

This module orchestrates the complete workflow from document extraction
to graph generation, export, and visualization using the graph module.
"""

import importlib
from pathlib import Path
from typing import Any, Dict, Optional

from rich import print

# Import core components
from .core import (
    CSVExporter,
    CypherExporter,
    DoclingExporter,
    ExtractorFactory,
    GraphConfig,
    GraphConverter,
    InteractiveVisualizer,
    JSONExporter,
    ReportGenerator,
)

# Import LLM clients
from .llm_clients import BaseLlmClient, MistralClient, OllamaClient, VllmClient


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
        module_path, class_name = template_str.rsplit(".", 1)
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
    provider_override: Optional[str] = None,
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
    model_config = config_data.get("models", {}).get(backend_type, {}).get(inference, {})

    if not model_config:
        raise ValueError(
            f"No configuration found for backend_type='{backend_type}' with inference='{inference}'"
        )

    provider = provider_override or model_config.get(
        "provider", "ollama" if inference == "local" else "mistral"
    )

    if model_override:
        model = model_override
    elif provider_override and inference == "remote":
        providers = model_config.get("providers", {})
        model = providers.get(provider_override, {}).get(
            "default_model", model_config.get("default_model")
        )
    else:
        model = model_config.get("default_model")

    return {"model": model, "provider": provider}


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
    elif provider == "vllm":
        return VllmClient(model=model)
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
            - export_format: Export format ('csv', 'cypher')
            - config: Nested config with models, etc.
            - model_override: Optional model override
            - provider_override: Optional provider override
    """
    print("\n--- [blue]Starting Docling-Graph Pipeline[/blue] ---")

    processing_mode = config.get("processing_mode")
    backend_type = config.get("backend_type")
    inference = config.get("inference")
    docling_config = config.get("docling_config", "ocr")
    reverse_edges = config.get("reverse_edges", False)

    # Initialize variables for cleanup
    extractor = None
    llm_client = None

    # Set outputs directory
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(config["source"]).stem

    try:
        # 1. Load Template
        try:
            TemplateClass = _load_template_class(config["template"])
        except Exception:
            return

        # 2. Get model configuration
        try:
            model_config = _get_model_config(
                config.get("config", {}),
                backend_type,
                inference,
                config.get("model_override"),
                config.get("provider_override"),
            )
            print(
                f"Using model: [cyan]{model_config['model']}[/cyan] "
                f"(provider: {model_config['provider']})"
            )
        except Exception as e:
            print(f"[red]Configuration error:[/red] {e}")
            return

        # 3. Create extractor using factory
        try:
            if backend_type == "vlm":
                extractor = ExtractorFactory.create_extractor(
                    processing_mode=processing_mode,
                    backend_type=backend_type,
                    model_name=model_config["model"],
                    docling_config=docling_config,
                )
            elif backend_type == "llm":
                llm_client = _initialize_llm_client(model_config["provider"], model_config["model"])
                extractor = ExtractorFactory.create_extractor(
                    processing_mode=processing_mode,
                    backend_type=backend_type,
                    llm_client=llm_client,
                    docling_config=docling_config,
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

        # Docling document and markdown export
        if config.get("export_docling", True):
            print("Exporting Docling document and markdown...")
            docling_exporter = DoclingExporter(output_dir=output_dir)

            # Get the document from extractor
            if hasattr(extractor, "doc_processor") and hasattr(
                extractor.doc_processor, "converter"
            ):
                # Re-convert to get document object if needed
                doc_result = extractor.doc_processor.converter.convert(config["source"])
                docling_document = doc_result.document

                docling_exporter.export_document(
                    docling_document,
                    base_name=base_name,
                    include_json=config.get("export_docling_json", True),
                    include_markdown=config.get("export_markdown", True),
                    per_page=config.get("export_per_page_markdown", False),
                )

        # 5. Convert to Graph
        print("Converting Pydantic model(s) to Knowledge Graph...")

        # Create graph config with custom settings
        graph_config = GraphConfig(add_reverse_edges=reverse_edges)

        # Create converter with config
        converter = GraphConverter(config=graph_config)

        # Convert models to graph (NOW RETURNS TUPLE!)
        knowledge_graph, graph_metadata = converter.pydantic_list_to_graph(extracted_data)

        print(
            f"Graph created with [blue]{graph_metadata.node_count} nodes[/blue] "
            f"and [blue]{graph_metadata.edge_count} edges[/blue]."
        )

        # 6. Export graph
        export_format = config.get("export_format", "csv")
        print(f"Exporting graph data in [cyan]{export_format.upper()}[/cyan] format...")

        if export_format == "csv":
            exporter = CSVExporter()
            exporter.export(knowledge_graph, output_dir)
            print(f"[green]→[/green] Saved CSV files to [green]{output_dir}[/green]")

        elif export_format == "cypher":
            cypher_path = output_dir / f"{base_name}_graph.cypher"
            exporter = CypherExporter()
            exporter.export(knowledge_graph, cypher_path)
            print(f"[green]→[/green] Saved Cypher script to [green]{cypher_path}[/green]")

        # Always export to JSON format
        json_path = output_dir / f"{base_name}_graph.json"
        exporter = JSONExporter()
        exporter.export(knowledge_graph, json_path)
        print(f"[green]→[/green] Saved JSON to [green]{json_path}[/green]")

        # 7. Generate reports
        print("Generating outputs...")

        # Markdown report
        report_generator = ReportGenerator()
        report_path = output_dir / f"{base_name}_report"
        report_generator.visualize(
            knowledge_graph, report_path, source_model_count=len(extracted_data)
        )
        print("[green]→[/green] Generated markdown report")

        # Interactive HTML Graph
        html_path = output_dir / f"{base_name}_graph"
        visualizer = InteractiveVisualizer()
        visualizer.save_cytoscape_graph(knowledge_graph, html_path)
        print("[green]→[/green] Generated interactive html graph")

        print("--- [blue]Pipeline Finished Successfully[/blue] ---")

    finally:
        # Cleanup resources
        print("Cleaning up resources...")

        # Clean up extractor and its components
        if extractor is not None:
            # Clean up backend (VLM or LLM)
            if hasattr(extractor, "backend") and hasattr(extractor.backend, "cleanup"):
                extractor.backend.cleanup()

            # Clean up document processor
            if hasattr(extractor, "doc_processor") and hasattr(extractor.doc_processor, "cleanup"):
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
