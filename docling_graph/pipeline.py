from typing import Dict, Any
from pathlib import Path
from rich import print
import importlib

# Import LLM Clients
from .llm_clients.mistral import MistralClient
from .llm_clients.ollama import OllamaClient
from .llm_clients.base import BaseLlmClient

from .extractors.factory import ExtractorFactory
from .graph_converter import GraphConverter

from .graph_visualizer import create_interactive_graph, create_static_graph, create_markdown_report
from .graph_exporter import to_csv, to_cypher


def _load_template_class(template_str: str):
    """Dynamically imports a Pydantic model class from a string."""
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


def _get_model_config(config_data: Dict, model_type: str, inference: str, 
                      model_override: str = None, provider_override: str = None) -> Dict[str, Any]:
    """Retrieves the appropriate model configuration based on settings."""
    
    model_config = config_data.get('models', {}).get(model_type, {}).get(inference, {})
    
    if not model_config:
        raise ValueError(f"No configuration found for model_type='{model_type}' with inference='{inference}'")
    
    provider = provider_override or model_config.get('provider', 'ollama' if inference == 'local' else 'mistral')
    
    if model_override:
        model = model_override
    elif provider_override and inference == 'api':
        providers = model_config.get('providers', {})
        model = providers.get(provider_override, {}).get('default_model', model_config.get('default_model'))
    else:
        model = model_config.get('default_model')
    
    return {'model': model, 'provider': provider}


def _initialize_llm_client(provider: str, model: str) -> BaseLlmClient:
    """Initializes an LLM client based on provider."""
    if provider == "mistral":
        return MistralClient(model=model)
    elif provider == "ollama":
        return OllamaClient(model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def run_pipeline(config: Dict[str, Any]):
    """
    Runs the extraction and graph conversion pipeline with refactored extractors.
    """
    print("--- [blue]Starting Docling-Graph Pipeline[/blue] ---")

    processing_mode = config.get("processing_mode")
    model_type = config.get("model_type")
    inference = config.get("inference")
    docling_config = config.get("docling_config", "default")
    
    # 1. Load Template
    try:
        TemplateClass = _load_template_class(config["template"])
    except Exception:
        return

    # 2. Get model configuration
    try:
        model_config = _get_model_config(
            config.get('config', {}),
            model_type,
            inference,
            config.get('model_override'),
            config.get('provider_override')
        )
        print(f"Using model: [cyan]{model_config['model']}[/cyan] (provider: {model_config['provider']})")
    except Exception as e:
        print(f"[red]Configuration error:[/red] {e}")
        return

    # 3. Create extractor using factory
    try:
        if model_type == "vlm":
            extractor = ExtractorFactory.create_extractor(
                processing_mode=processing_mode,
                model_type=model_type,
                model_name=model_config['model'],
                docling_config=docling_config
            )
        elif model_type == "llm":
            llm_client = _initialize_llm_client(model_config['provider'], model_config['model'])
            extractor = ExtractorFactory.create_extractor(
                processing_mode=processing_mode,
                model_type=model_type,
                llm_client=llm_client,
                docling_config=docling_config
            )
        else:
            print(f"[red]Error:[/red] Invalid model_type: {model_type}")
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

    # 5. Convert to Graph
    print("Converting Pydantic model(s) to Knowledge Graph...")
    # To enable reverse edges: GraphConverter(add_reverse_edges=True)
    converter = GraphConverter() 
    knowledge_graph = converter.pydantic_list_to_graph(extracted_data)
    print(f"Graph created with [blue]{knowledge_graph.number_of_nodes()} nodes[/blue] and [blue]{knowledge_graph.number_of_edges()} edges[/blue].")

    # 6. Save outputs
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(config["source"]).stem
    output_path = output_dir / f"{base_name}_graph"

    export_format = config.get("export_format", "csv")
    print(f"Exporting graph data in [cyan]{export_format.upper()}[/cyan] format...")
    
    if export_format == "csv":
        to_csv(knowledge_graph, output_dir)
        print(f"[green]->[/green] Saved CSV files to [green]{output_dir}[/green]")
    elif export_format == "cypher":
        cypher_path = output_dir / f"{base_name}_graph.cypher"
        to_cypher(knowledge_graph, cypher_path)
        print(f"[green]->[/green] Saved Cypher script to [green]{cypher_path}[/green]")

    print(f"[green]->[/green] Saved graphs and report to [green]{output_dir}[/green]")

    # Markdown report
    create_markdown_report(knowledge_graph, output_path)

    # Interactive visualizations
    create_interactive_graph(knowledge_graph, output_path)

    # Static exports (PNG, SVG, PDF)
    create_static_graph(knowledge_graph, output_path, format='png')

    print("--- [blue]Pipeline Finished Successfully[/blue] ---")
