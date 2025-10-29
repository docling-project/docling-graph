"""
LLM (Language Model) extraction backend.
Handles document extraction using LLM models (local or API).
"""
import json
import gc

from pydantic import BaseModel, ValidationError
from typing import Type, Optional
from rich import print

from ....llm_clients.llm_base import BaseLlmClient
from ....llm_clients.prompts import get_prompt

class LlmBackend:
    """Backend for LLM-based extraction (local or API)."""

    def __init__(self, llm_client: BaseLlmClient):
        """
        Initialize LLM backend with a client.
        
        Args:
            llm_client (BaseLlmClient): LLM client instance (Mistral, Ollama, etc.)
        """
        self.client = llm_client
        print(f"[blue][LlmBackend][/blue] Initialized with client: [cyan]{self.client.__class__.__name__}[/cyan]")

    def extract_from_markdown(self, markdown: str, template: Type[BaseModel],
                            context: str = "document") -> Optional[BaseModel]:
        """
        Extract structured data from markdown content using LLM.
        
        Args:
            markdown (str): Markdown content to extract from.
            template (Type[BaseModel]): Pydantic model template.
            context (str): Context description for the extraction (e.g., "page 1", "full document").
        
        Returns:
            Optional[BaseModel]: Extracted and validated Pydantic model instance, or None if failed.
        """
        print(f"[blue][LlmBackend][/blue] Extracting from {context} ([cyan]{len(markdown)}[/cyan] chars)")
        
        # Validation for empty markdown
        if not markdown or len(markdown.strip()) == 0:
            print(f"[red]Error:[/red] Extracted markdown is empty for {context}. Cannot proceed with LLM extraction.")
            return None
        
        try:
            # Get the Pydantic schema as JSON
            schema_json = json.dumps(template.model_json_schema(), indent=2)
            
            # Generate prompt using the correct signature
            prompt = get_prompt(markdown_content=markdown, schema_json=schema_json, is_partial=False)
            
            # Call LLM with correct method name
            parsed_json = self.client.get_json_response(prompt=prompt, schema_json=schema_json)
            
            if not parsed_json:
                print(f"[yellow]Warning:[/yellow] No valid JSON returned from LLM for {context}")
                return None
            
            # Use model_validate for proper Pydantic validation
            try:
                validated_model = template.model_validate(parsed_json)
                print(f"[blue][LlmBackend][/blue] Successfully extracted data from {context}")
                return validated_model
            
            except ValidationError as e:
                # Detailed error reporting
                print(f"[blue][LlmBackend][/blue] [yellow]Validation Error for {context}:[/yellow]")
                print(f"  The data extracted by the LLM does not match your Pydantic template.")
                print("[red]Details:[/red]")
                for error in e.errors():
                    loc = " -> ".join(map(str, error['loc']))
                    print(f"  - [bold magenta]{loc}[/bold magenta]: [red]{error['msg']}[/red]")
                print(f"\n[yellow]Extracted Data (raw):[/yellow]\n{parsed_json}\n")
                return None
        
        except Exception as e:
            print(f"[red]Error during LLM extraction for {context}:[/red] {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """Clean up LLM client resources."""
        try:
            # Release the client reference
            if hasattr(self, 'client'):
                # If the client has its own cleanup method, call it
                if hasattr(self.client, 'cleanup'):
                    self.client.cleanup()
                del self.client
            
            # Force garbage collection
            gc.collect()
            
            print("[blue][LlmBackend][/blue] [green]Cleaned up resources[/green]")
            
        except Exception as e:
            print(f"[blue][LlmBackend][/blue] [yellow]Warning during cleanup:[/yellow] {e}")
