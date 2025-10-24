import importlib
import os
from pathlib import Path
from rich import print
from typing import Dict, Any

# --- Import Extractors ---
from .extractors.one_to_one_local import OneToOneLocalExtractor
# Import the refactored OneToOneApiExtractor
from .extractors.one_to_one_api import OneToOneApiExtractor
from .extractors.many_to_one import ManyToOneExtractor
from .graph_exporter import to_csv, to_cypher

# --- Import LLM Clients ---
from .llm_clients.base import BaseLlmClient
from .llm_clients.mistral import MistralClient
from .llm_clients.ollama import OllamaClient
# (Add future clients here, e.g., OpenAiClient)

from .graph_converter import GraphConverter
from .graph_visualizer import create_static_graph, create_interactive_graph, create_markdown_report

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

def _initialize_llm_client(config: Dict[str, Any]) -> BaseLlmClient:
    """Initializes an LLM client from configuration."""
    client_type = config.get("client")
    model = config.get("model")
    
    if client_type == "mistral":
        return MistralClient(model=model)
    elif client_type == "ollama":
        return OllamaClient(model=model)
    # (Add other clients here)
    else:
        raise ValueError(f"Unknown LLM client type: {client_type}")

def run_pipeline(config: Dict[str, Any]):
    """
    Runs the full extraction and graph conversion pipeline based on a config dict.
    """
    print("--- [blue]Starting Docling-Graph Pipeline[/blue] ---")
    
    # 1. Load Template
    try:
        TemplateClass = _load_template_class(config["template"])
    except Exception:
        return # Error is printed in the helper

    # 2. Initialize Extractor
    mode = config.get("pipeline_mode", "one_to_one")
    etype = config.get("extractor_type", "local_vlm")
    export_format = config.get("export_format", "csv")
    
    extractor = None
    try:
        if mode == "one_to_one" and etype == "local_vlm":
            print("Using pipeline: [cyan]One-to-One (Local VLM)[/cyan]")
            model_repo = config.get("model_repo_id", "numind/NuExtract-2.0-2B")
            extractor = OneToOneLocalExtractor(model_name=model_repo) 
        
        elif mode == "one_to_one" and etype == "api_docling":
            print("Using pipeline: [cyan]One-to-One (Docling API)[/cyan]")
            extractor = OneToOneApiExtractor() # Assumes API key is in env
        
        elif mode == "many_to_one" and etype == "local_llm":
            print("Using pipeline: [cyan]Many-to-One (Local LLM)[/cyan]")
            llm_config = config.get("llm")
            if not llm_config:
                print("[red]Error:[/red] 'llm' configuration is required for 'many_to_one' pipeline.")
                return
            llm_client = _initialize_llm_client(llm_config)
            extractor = ManyToOneExtractor(llm_client=llm_client)
        
        else:
            print(f"[red]Error:[/red] Invalid pipeline configuration: mode='{mode}', extractor='{etype}'")
            return
            
    except Exception as e:
        print(f"[red]Failed to initialize extractor:[/red] {e}")
        return

    # 4. Run Extraction
    extracted_data = extractor.extract(config["source"], TemplateClass)

    extracted_models = []
    if extracted_data:
        if isinstance(extracted_data, list):
            extracted_models = extracted_data
        else:
            extracted_models = [extracted_data] # Wrap single object in a list

    if not extracted_models:
        print("[red]Pipeline stopped: Extraction returned no data.[/red]")
        return
        
    print(f"Successfully extracted {len(extracted_models)} item(s).")

    # 5. Convert to Graph (Uses pydantic_list_to_graph)
    print("Converting Pydantic model(s) to Knowledge Graph...")
    converter = GraphConverter()
    knowledge_graph = converter.pydantic_list_to_graph(extracted_models)
    
    print(f"Graph created with [blue]{knowledge_graph.number_of_nodes()} nodes[/blue] and [blue]{knowledge_graph.number_of_edges()} edges[/blue].")

    # 6. Save outputs
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(config["source"]).stem
    output_path = output_dir / f"{base_name}_graph"

    # Export graph data
    export_format = config.get("export_format", "csv")
    print(f"Exporting graph data in [cyan]{export_format.upper()}[/cyan] format...")
    
    if export_format == "csv":
        to_csv(knowledge_graph, output_dir)
        print(f"Saved CSV files to [green]{output_dir}[/green]")
    elif export_format == "cypher":
        cypher_path = output_dir / f"{base_name}_graph.cypher"
        to_cypher(knowledge_graph, cypher_path)
        print(f"Saved Cypher script to [green]{cypher_path}[/green]")

    # Create visualizations
    print(f"Creating visualizations in [green]{output_dir}[/green]")
    create_markdown_report(knowledge_graph, output_path)
    create_interactive_graph(knowledge_graph, output_path)
    create_static_graph(knowledge_graph, output_path)

    print("--- [blue]Pipeline Finished Successfully[/blue] ---")
