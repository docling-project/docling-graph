import json
from typing import Dict, Any
from rich import print
from .base import BaseLlmClient

# Requires `pip install ollama`
try:
    import ollama
except ImportError:
    print("[red]Error:[/red] `ollama` package not found. Please run `pip install ollama` to use local LLMs.")
    ollama = None

class OllamaClient(BaseLlmClient):
    """Ollama (local LLM) implementation of the LLM Client."""
    
    def __init__(self, model: str):
        if ollama is None:
            raise ImportError("Ollama client could not be imported.")
            
        self.model = model
        
        # --- Context Limits (Approximate & Conservative) ---
        # This is a good candidate for a new config.yaml setting.
        self._context_limit = 8000 # Default for Llama 3 8B
        
        try:
            print(f"[OllamaClient] Checking Ollama connection and model '{self.model}'...")
            ollama.show(self.model) 
            print(f"[OllamaClient] Initialized with Ollama model: [blue]{self.model}[/blue]")
        except Exception as e:
            print(f"[red]Ollama connection error:[/red] {e}")
            print(f"Please ensure Ollama is running and has the model '{self.model}' available via `ollama pull {self.model}`.")
            raise

    def get_json_response(self, prompt: str, schema_json: str) -> Dict[str, Any]:
        """Executes the Ollama chat call."""
        res = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        raw_json = res['message']['content']
        return json.loads(raw_json)

    @property
    def context_limit(self) -> int:
        return self._context_limit
