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

    # Narrow processing_mode to the expected Literal
    raw_processing_mode: str = cast(str, conf["processing_mode"])
    if raw_processing_mode not in ("one-to-one", "many-to-one"):
        raise ValueError("processing_mode must be 'one-to-one' or 'many-to-one'")
    processing_mode = cast(Literal["one-to-one", "many-to-one"], raw_processing_mode)

    backend_name: str = cast(str, conf["backend"])
    if backend_name not in ("vlm", "llm"):
        raise ValueError("backend must be 'vlm' or 'llm'")
    backend_literal = cast(Literal["vlm", "llm"], backend_name)

    inference: str = cast(str, conf["inference"])
    docling_config: str = cast(str, conf["docling_config"])
    reverse_edges = cast(bool, conf.get("reverse_edges", False))

    extractor = None
    llm_client: Optional[BaseLlmClient] = None

    output_dir = Path(conf.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(conf["source"]).stem

    try:
        # 1. Load Template
        template_val = conf["template"]
        if isinstance(template_val, str):
            template_class = _load_template_class(template_val)
        elif isinstance(template_val, type) and issubclass(template_val, BaseModel):
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
            cast(Optional[str], conf.get("model_override")),
            cast(Optional[str], conf.get("provider_override")),
        )

        rich_print(
            f"Using model: [cyan]{model_config['model']}[/cyan] "
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
        elif backend_literal == "llm":
            llm_client = _initialize_llm_client(model_config["provider"], model_config["model"])
            extractor = ExtractorFactory.create_extractor(
                processing_mode=processing_mode,
                backend_name=backend_literal,
                llm_client=llm_client,
                docling_config=docling_config,
            )
        else:
            raise ValueError(f"Invalid backend: {backend_literal}")

        # 4. Run Extraction
        extracted_data = extractor.extract(conf["source"], template_class)
        if not extracted_data:
            rich_print("[red]Pipeline stopped: Extraction returned no data.[/red]")
            return

        rich_print(f"Successfully extracted {len(extracted_data)} item(s).")

        # Docling export
        if conf.get("export_docling", True):
            rich_print("Exporting Docling document and markdown...")
            docling_exporter = DoclingExporter(output_dir=output_dir)

            if hasattr(extractor, "doc_processor") and hasattr(
                extractor.doc_processor, "converter"
            ):
                doc_result = extractor.doc_processor.converter.convert(conf["source"])
                docling_document = doc_result.document
                docling_exporter.export_document(
                    docling_document,
                    base_name=base_name,
                    include_json=conf.get("export_docling_json", True),
                    include_markdown=conf.get("export_markdown", True),
                    per_page=conf.get("export_per_page_markdown", False),
                )

        # 5. Convert to Graph
        rich_print("Converting Pydantic model(s) to Knowledge Graph...")
        graph_config = GraphConfig(add_reverse_edges=reverse_edges)
        converter = GraphConverter(config=graph_config)
        knowledge_graph, graph_metadata = converter.pydantic_list_to_graph(extracted_data)
        rich_print(
            f"Graph created with [blue]{graph_metadata.node_count} nodes[/blue] "
            f"and [blue]{graph_metadata.edge_count} edges[/blue]."
        )

        # 6. Export graph
        export_format = cast(str, conf.get("export_format", "csv"))
        rich_print(f"Exporting graph data in [cyan]{export_format.upper()}[/cyan] format...")
        if export_format == "csv":
            CSVExporter().export(knowledge_graph, output_dir)
            rich_print(f"[green]→[/green] Saved CSV files to [green]{output_dir}[/green]")
        elif export_format == "cypher":
            cypher_path = output_dir / f"{base_name}_graph.cypher"
            CypherExporter().export(knowledge_graph, cypher_path)
            rich_print(f"[green]→[/green] Saved Cypher script to [green]{cypher_path}[/green]")
        else:
            raise ValueError(f"Unknown export format: {export_format}")

        # Always export to JSON format
        json_path = output_dir / f"{base_name}_graph.json"
        JSONExporter().export(knowledge_graph, json_path)
        rich_print(f"[green]→[/green] Saved JSON to [green]{json_path}[/green]")

        # 7. Reports and visualization
        rich_print("Generating outputs...")
        report_path = output_dir / f"{base_name}_report"
        ReportGenerator().visualize(
            knowledge_graph, report_path, source_model_count=len(extracted_data)
        )
        rich_print("[green]→[/green] Generated markdown report")

        html_path = output_dir / f"{base_name}_graph"
        InteractiveVisualizer().save_cytoscape_graph(knowledge_graph, html_path)
        rich_print("[green]→[/green] Generated interactive html graph")

        rich_print("--- [blue]Pipeline Finished Successfully[/blue] ---")

    finally:
        # Cleanup resources
        rich_print("Cleaning up resources...")
        if extractor is not None:
            if hasattr(extractor, "backend") and hasattr(extractor.backend, "cleanup"):
                extractor.backend.cleanup()
            if hasattr(extractor, "doc_processor") and hasattr(extractor.doc_processor, "cleanup"):
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
