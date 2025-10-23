import os
import json
from typing import Dict, Any
from mistralai.client import Mistral
from rich import print
from .base import BaseLlmClient

class MistralClient(BaseLlmClient):
    """Mistral API implementation of the LLM Client."""
    
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("[MistralClient] [red]Error:[/red] MISTRAL_API_KEY not set.")
        
        self.client = Mistral(api_key=self.api_key)
        
        # --- Context Limits (Approximate) ---
        model_context_limits = {
            "mistral-large-latest": 32000,
            "mistral-small-latest": 32000,
        }
        self._context_limit = model_context_limits.get(model, 32000)
        
        print(f"[MistralClient] Initialized for [blue]{self.model}[/blue]")

    def get_json_response(self, prompt: str, schema_json: str) -> Dict[str, Any]:
        """Executes the Mistral chat.complete call."""
        res = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(res.choices[0].message.content)

    @property
    def context_limit(self) -> int:
        return self._context_limit
