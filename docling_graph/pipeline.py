"""
Main extraction and graph conversion pipeline.
This module orchestrates the complete workflow from document extraction
to graph generation, export, and visualization using the graph module.
"""

import importlib
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union, cast

from pydantic import BaseModel
from rich import print as rich_print

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
    PipelineConfig,
    ReportGenerator,
)
from .core.converters.node_id_registry import NodeIDRegistry

# Import LLM clients
from .llm_clients import BaseLlmClient, get_client


def _load_template_class(template_str: str) -> type[BaseModel]:
    """Dynamically imports a Pydantic model class from a string."""
    if "." not in template_str:
        raise ValueError("Template path must contain at least one dot (e.g., 'module.Class')")
    try:
        module_path, class_name = template_str.rsplit(".", 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, class_name)
        if not isinstance(obj, type) or not issubclass(obj, BaseModel):
            raise TypeError("Template must be a subclass of pydantic.BaseModel")
        return obj
    except (ModuleNotFoundError, AttributeError):
        raise  # Let caller handle


def _get_model_config(
    models_config: Dict[str, Any],
    backend: str,
    inference: str,
    model_override: Optional[str] = None,
    provider_override: Optional[str] = None,
) -> Dict[str, str]:
    """Retrieves the appropriate model configuration based on settings."""
    model_config = models_config.get(backend, {}).get(inference, {})
    if not model_config:
        raise ValueError(
            f"No configuration found for backend='{backend}' with inference='{inference}'"
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

    if not model:
        raise ValueError("Resolved model is empty; check models configuration")

    return {"model": model, "provider": provider}


def _initialize_llm_client(provider: str, model: str) -> BaseLlmClient:
    """Initializes an LLM client based on provider using lazy-loading."""
    client_class = get_client(provider)
    return client_class(model=model)


def run_pipeline(config: Union[PipelineConfig, Dict[str, Any]]) -> None:
    """Runs the extraction and graph conversion pipeline."""
    # Normalize to typed config up-front
    cfg: PipelineConfig = config if isinstance(config, PipelineConfig) else PipelineConfig(**config)

    # Use normalized flat dict for downstream access
    conf: Dict[str, Any] = cfg.to_dict()

    rich_print("\n--- [blue]Starting Docling-Graph Pipeline[/blue] ---")

    # Create shared registry for deterministic node IDs across batches
    node_registry = NodeIDRegistry()

    # Validate modes
    processing_mode = cast(Literal["one-to-one", "many-to-one"], conf["processing_mode"])
    backend_literal = cast(Literal["vlm", "llm"], conf["backend"])

    inference = cast(str, conf["inference"])
    docling_config = cast(str, conf["docling_config"])
    reverse_edges = cast(bool, conf.get("reverse_edges", False))
    llm_consolidation = cast(bool, conf.get("llm_consolidation", True))
    use_chunking = cast(bool, conf.get("use_chunking", True))

    output_dir = Path(conf.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(conf["source"]).stem

    extractor = None
    llm_client: Optional[BaseLlmClient] = None

    try:
        # 1. Load Template
        template_val = conf["template"]
        if isinstance(template_val, str):
            template_class = _load_template_class(template_val)
        elif isinstance(template_val, type):
            template_class = template_val
        else:
            raise TypeError(
                "Template must be a dotted path string or a Pydantic BaseModel subclass."
            )

        # 2. Get model configuration
        models_config = cast(Dict[str, Any], conf["models"])
        model_config = _get_model_config(
            models_config,
            backend_literal,
            inference,
            conf.get("model_override"),
            conf.get("provider_override"),
        )

        rich_print(
            f"[blue][Pipeline][/blue] Using model: [cyan]{model_config['model']}[/cyan] "
            f"(provider: {model_config['provider']})"
        )

        # 3. Create extractor
        if backend_literal == "vlm":
            extractor = ExtractorFactory.create_extractor(
                processing_mode=processing_mode,
                backend_name=backend_literal,
                model_name=model_config["model"],
                docling_config=docling_config,
            )
        else:
            llm_client = _initialize_llm_client(model_config["provider"], model_config["model"])
            extractor = ExtractorFactory.create_extractor(
                processing_mode=processing_mode,
                backend_name=backend_literal,
                llm_client=llm_client,
                docling_config=docling_config,
                llm_consolidation=llm_consolidation,
                use_chunking=use_chunking,
            )

        # 4. Run Extraction
        rich_print("[blue][Pipeline][/blue] Starting extraction...")
        extracted_models = extractor.extract(conf["source"], template_class)

        if not extracted_models:
            rich_print("[yellow][Pipeline][/yellow] No models extracted.")
            return

        rich_print(
            f"[green][Pipeline][/green] Successfully extracted "
            f"[cyan]{len(extracted_models)}[/cyan] item(s)."
        )

        # 5. Export Docling outputs (if configured)
        if (
            conf.get("export_docling", True)
            or conf.get("export_docling_json", True)
            or conf.get("export_markdown", True)
        ):
            rich_print("[blue][Pipeline][/blue] Exporting Docling document and markdown...")
            docling_exporter = DoclingExporter(output_dir=output_dir)

            # Reuse the already-converted document from the extraction phase
            if hasattr(extractor, "doc_processor") and extractor.doc_processor.last_document:
                docling_document = extractor.doc_processor.last_document
                rich_print(
                    "[blue][Pipeline][/blue] Reusing cached DoclingDocument (avoiding duplicate conversion)"
                )
                docling_exporter.export_document(
                    docling_document,
                    base_name=base_name,
                    include_json=conf.get("export_docling_json", True),
                    include_markdown=conf.get("export_markdown", True),
                    per_page=conf.get("export_per_page_markdown", False),
                )
            else:
                rich_print(
                    "[yellow][Pipeline][/yellow] No cached document available, skipping Docling export"
                )

        # 6. Convert to Graph
        rich_print("[blue][Pipeline][/blue] Converting Pydantic model(s) to Knowledge Graph...")

        converter = GraphConverter(
            add_reverse_edges=reverse_edges,
            validate_graph=True,
            registry=node_registry,
        )

        try:
            knowledge_graph, graph_metadata = converter.pydantic_list_to_graph(extracted_models)
        except ValueError as e:
            rich_print(f"[red][Pipeline] Graph creation failed:[/red] {e}")
            raise

        rich_print(
            f"[green][Pipeline][/green] Graph created with "
            f"[blue]{graph_metadata.node_count} nodes[/blue] "
            f"and [blue]{graph_metadata.edge_count} edges[/blue]."
        )

        # 7. Export graph
        export_format = cast(str, conf.get("export_format", "csv"))
        rich_print(
            f"[blue][Pipeline][/blue] Exporting graph data in [cyan]{export_format.upper()}[/cyan] format..."
        )

        if export_format == "csv":
            CSVExporter().export(knowledge_graph, output_dir)
            rich_print(f"[green]→[/green] Saved CSV files to [green]{output_dir}[/green]")
        elif export_format == "cypher":
            cypher_path = output_dir / f"{base_name}_graph.cypher"
            CypherExporter().export(knowledge_graph, cypher_path)
            rich_print(f"[green]→[/green] Saved Cypher script to [green]{cypher_path}[/green]")
        else:
            raise ValueError(f"Unknown export format: {export_format}")

        # Always export to JSON
        json_path = output_dir / f"{base_name}_graph.json"
        JSONExporter().export(knowledge_graph, json_path)
        rich_print(f"[green]→[/green] Saved JSON to [green]{json_path}[/green]")

        # 8. Reports and visualization
        rich_print("[blue][Pipeline][/blue] Generating reports and visualizations...")
        report_path = output_dir / f"{base_name}_report"
        ReportGenerator().visualize(
            knowledge_graph, report_path, source_model_count=len(extracted_models)
        )
        rich_print(f"[green]→[/green] Generated markdown report at {report_path}")

        html_path = output_dir / f"{base_name}_graph.html"
        InteractiveVisualizer().save_cytoscape_graph(knowledge_graph, html_path)
        rich_print(f"[green]→[/green] Generated interactive HTML graph at {html_path}")

        rich_print("--- [blue]Pipeline Finished Successfully[/blue] ---")

    finally:
        # Cleanup resources
        rich_print("Cleaning up resources...")
        if extractor and hasattr(extractor, "backend") and hasattr(extractor.backend, "cleanup"):
            extractor.backend.cleanup()
        if (
            extractor
            and hasattr(extractor, "doc_processor")
            and hasattr(extractor.doc_processor, "cleanup")
        ):
            extractor.doc_processor.cleanup()
        if llm_client is not None:
            del llm_client

        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
