import json
from pydantic import BaseModel, ValidationError
from typing import Type, List
from rich import print

from .base import BaseExtractor
from docling.document_converter import DocumentConverter

# Import our new client interface and prompt generator
from ..llm_clients.base import BaseLlmClient
from ..llm_clients.prompts import get_prompt

# Import shared utilities
from .utils import _deep_merge_dicts, TOKEN_CHAR_RATIO


class ManyToOneExtractor(BaseExtractor):
    """
    Unified extractor for "Many-to-One" logic.
    It accepts any 'BaseLlmClient' and handles the full
    orchestration: fast-path, fallback, and JSON merge.
    """
    
    def __init__(self, client: BaseLlmClient):
        self.client = client
        # Set a conservative limit, leaving room for the prompt and response
        self.context_limit_tokens = self.client.context_limit - 2000
        self.converter = DocumentConverter()
        print(f"[ManyToOneExtractor] Initialized with client: [blue]{client.__class__.__name__}[/blue]")
        print(f"  > Effective Context Limit: {self.context_limit_tokens} tokens")

    def extract(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        """
        Orchestrates the extraction, trying the "fast path" first and
        falling back to the page-by-page merge if needed.
        """
        print(f"[ManyToOneExtractor] Converting document [green]{source}[/green] to full Markdown...")
        doc = self.converter.convert(source=source)
        if not doc.pages:
            print("[ManyToOneExtractor] [yellow]Warning:[/yellow] Document processing returned no pages.")
            return []
            
        full_markdown = doc.export_to_markdown()
        schema_json = json.dumps(template.model_json_schema(), indent=2)

        # --- Check Token Limit ---
        token_estimate = (len(full_markdown) + len(schema_json)) / TOKEN_CHAR_RATIO
        
        attempt_fast_path = True
        if token_estimate > self.context_limit_tokens:
            print(f"[yellow]Warning:[/yellow] Document estimate ({int(token_estimate)} tokens) exceeds context limit ({self.context_limit_tokens}).")
            print("Forcing page-by-page extraction.")
            attempt_fast_path = False

        # --- 1. Attempt "Fast Path" (Single API Call) ---
        if attempt_fast_path:
            print(f"Document fits in context ({int(token_estimate)} tokens). Attempting single call...")
            try:
                prompt = get_prompt(full_markdown, schema_json, is_partial=False)
                json_data = self.client.get_json_response(prompt, schema_json)
                
                model = template.model_validate(json_data)
                print("[ManyToOneExtractor] [green]Fast path successful.[/green] Found 1 item.")
                return [model]
            except Exception as e:
                print(f"[ManyToOneExtractor] [yellow]Fast path failed:[/yellow] {e}")
                print("Falling back to page-by-page extraction.")
        
        # --- 2. "Fallback Path" (Page-by-Page + Merge) ---
        print(f"Starting page-by-page extraction ({len(doc.pages)} calls)...")
        all_partial_dicts = []
        
        for i, page in enumerate(doc.pages):
            print(f"  Processing page {i+1}/{len(doc.pages)}...")
            try:
                page_md = page.export_to_markdown()
                if not page_md.strip():
                    print(f"  Skipping empty page {i+1}.")
                    continue
                
                prompt = get_prompt(page_md, schema_json, is_partial=True)
                page_json_data = self.client.get_json_response(prompt, schema_json)
                
                all_partial_dicts.append(page_json_data)
            except Exception as e:
                print(f"  [red]Failed to extract page {i+1}:[/red] {e}")

        if not all_partial_dicts:
            print("[ManyToOneExtractor] [red]Error:[/red] Page-by-page extraction yielded no data.")
            return []

        # --- 3. The Merge Step ---
        print(f"Merging JSON data from {len(all_partial_dicts)} pages...")
        merged_dict = {}
        for partial_dict in all_partial_dicts:
            _deep_merge_dicts(merged_dict, partial_dict)

        print("Validating merged JSON against Pydantic schema...")
        try:
            final_model = template.model_validate(merged_dict)
            print("[ManyToOneExtractor] [green]Fallback extraction successful.[/green] Found 1 item.")
            return [final_model]
        except ValidationError as e:
            print(f"[ManyToOneExtractor] [bold red]Failed to validate merged JSON:[/bold red]")
            print(e)
            print("\n--- Merged JSON (for debugging) ---")
            print(json.dumps(merged_dict, indent=2))
            print("--- End Merged JSON ---")
            return []
        except Exception as e:
            print(f"[ManyToOneExtractor] [bold red]An unexpected error occurred during final validation:[/bold red] {e}")
            return []
