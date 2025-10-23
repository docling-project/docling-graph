import json
from pydantic import BaseModel, ValidationError
from typing import Type, List
from rich import print

from .base import BaseExtractor
from docling.document_converter import DocumentConverter

# Import our new client interface and prompt generator
from ..llm_clients.base import BaseLlmClient
from ..llm_clients.prompts import get_prompt


class OneToOneApiExtractor(BaseExtractor):
    """
    Extractor for "One-to-One" logic using a remote API.
    It accepts any 'BaseLlmClient'.
    
    This extractor converts a document to Markdown, then loops
    through each page, performing one *full* extraction per page.
    """
    
    def __init__(self, client: BaseLlmClient):
        self.client = client
        self.converter = DocumentConverter()
        print(f"[OneToOneApiExtractor] Initialized with client: [blue]{client.__class__.__name__}[/blue]")

    def extract(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        """
        Orchestrates the extraction, running one extraction per page.
        """
        print(f"[OneToOneApiExtractor] Converting document [green]{source}[/green] to Markdown...")
        doc = self.converter.convert(source=source)
        if not doc.pages:
            print("[OneToOneApiExtractor] [yellow]Warning:[/yellow] Document processing returned no pages.")
            return []

        schema_json = json.dumps(template.model_json_schema(), indent=2)
        extracted_models = []

        print(f"Starting page-by-page extraction ({len(doc.pages)} calls)...")
        
        for i, page in enumerate(doc.pages):
            print(f"  Processing page {i+1}/{len(doc.pages)}...")
            try:
                page_md = page.export_to_markdown()
                if not page_md.strip():
                    print(f"  Skipping empty page {i+1}.")
                    continue
                
                # We use 'is_partial=False' because we expect one
                # *complete* document/model per page.
                prompt = get_prompt(page_md, schema_json, is_partial=False)
                
                json_data = self.client.get_json_response(prompt, schema_json)
                
                model = template.model_validate(json_data)
                extracted_models.append(model)
                
            except ValidationError as e:
                print(f"  [red]Failed to validate page {i+1}:[/red] {e}")
            except Exception as e:
                print(f"  [red]Failed to extract page {i+1}:[/red] {e}")

        print(f"[OneToOneApiExtractor] [green]Extraction successful.[/green] Found {len(extracted_models)} item(s).")
        return extracted_models

