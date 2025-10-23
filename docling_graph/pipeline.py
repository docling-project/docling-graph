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

# --- Import LLM Clients ---
from .llm_clients.base import BaseLlmClient
from .llm_clients.mistral import MistralClient
from .llm_clients.ollama import OllamaClient
# (Add future clients here, e.g., OpenAiClient)

from .graph_converter import GraphConverter
from .graph_visualizer import create_static_graph

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

def _get_llm_client(provider: str, model: str) -> BaseLlmClient:
    """Factory function to create an LLM client."""
    if provider == "mistral":
        return MistralClient(model=model)
    elif provider == "ollama":
        return OllamaClient(model=model)
    # elif provider == "openai":
    #     return OpenAiClient(model=model) 
    else:
        raise NotImplementedError(f"Provider '{provider}' not supported.")

def run_pipeline(config: Dict[str, Any]):
    """
    Runs the complete extraction and graph conversion pipeline based on config.
    """
    
    pipeline_config = config["pipeline"]
    
    # 1. Get final model and provider, applying overrides
    model = config.get("model_override") or pipeline_config.get("default_model")
    provider = config.get("provider_override") or pipeline_config.get("provider")

    if not model and pipeline_config.get("extractor_type") != "local_vlm":
        print(f"[red]Error:[/red] No model specified for this pipeline.")
        print("Set 'default_model' in the config or use --model.")
        return
    elif not model and pipeline_config.get("extractor_type") == "local_vlm":
        model = "numind/NuExtract-2.0-2B" # Default for local_vlm
        print(f"[yellow]No model specified, using default for local_vlm: {model}[/yellow]")


    # 2. Load Pydantic Template
    TemplateClass = _load_template_class(config["template"])

    # 3. Select and Initialize Extractor based on config
    extractor = None
    mode = pipeline_config.get("processing_mode")
    etype = pipeline_config.get("extractor_type")

    try:
        if mode == "one_to_one" and etype == "local_vlm":
            extractor = OneToOneLocalExtractor(model_repo_id=model)
        
        elif mode == "one_to_one" and etype == "api":
            if not provider:
                raise ValueError("'provider' is required for 'one_to_one_api' pipeline")
            # --- This logic is now refactored ---
            client = _get_llm_client(provider=provider, model=model)
            extractor = OneToOneApiExtractor(client=client)
        
        elif mode == "many_to_one":
            client: BaseLlmClient = None
            if etype == "api":
                if not provider:
                    raise ValueError("'provider' is required for 'many_to_one_api' pipeline")
                client = _get_llm_client(provider=provider, model=model)
            
            elif etype == "local_llm":
                # For many_to_one local, the "provider" is the client type
                client = _get_llm_client(provider="ollama", model=model) 
            
            else:
                raise ValueError(f"Invalid extractor_type '{etype}' for 'many_to_one' mode.")
            
            extractor = ManyToOneExtractor(client=client)

        else:
            print(f"[red]Error:[/red] Invalid pipeline configuration: mode='{mode}', extractor='{etype}'")
            return
            
    except Exception as e:
        print(f"[red]Failed to initialize extractor:[/red] {e}")
        return

    # 4. Run Extraction (This now *always* returns a list)
    extracted_models = extractor.extract(source=config["source"], template=TemplateClass)

    if not extracted_models:
        print("[red]Pipeline stopped: Extraction returned no data.[/red]")
        return
        
    print(f"Successfully extracted {len(extracted_models)} item(s).")

    # 5. Convert to Graph (Uses pydantic_list_to_graph)
    print("Converting Pydantic model(s) to Knowledge Graph...")
    converter = GraphConverter()
    knowledge_graph = converter.pydantic_list_to_graph(extracted_models)
    
    print(f"Graph created with [blue]{knowledge_graph.number_of_nodes()} nodes[/blue] and [blue]{knowledge_graph.number_of_edges()} edges[/blue].")

    # 6. Save and Visualize
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(config["source"]).stem
    output_filename = f"{base_name}_graph.png"
    output_path = output_dir / output_filename

    print(f"Saving static graph visualization to [green]{output_path}[/green]...")
    create_static_graph(knowledge_graph, filename=str(output_path))
    
    print("[bold green]Pipeline finished successfully.[/bold green]")