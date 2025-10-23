
from pydantic import BaseModel
from typing import Type, List
from docling.document_extractor import DocumentExtractor
from docling.datamodel.base_models import InputFormat
from rich import print
from .base import BaseExtractor

class OneToOneLocalExtractor(BaseExtractor):
    """
    Extractor for "One-to-One" logic using the local Docling VLM pipeline (NuExtract).
    Assumes one document item per page.
    """
    def __init__(self, model_repo_id: str):
        print(f"[OneToOneLocalExtractor] Initializing with model: [blue]{model_repo_id}[/blue]")
        try:
            pipeline_options = DocumentExtractor.get_default_options()[InputFormat.PDF].pipeline_options
            pipeline_options.vlm_options.repo_id = model_repo_id
            
            self.extractor = DocumentExtractor(
                allowed_formats=[InputFormat.PDF, InputFormat.IMAGE]
            )
            self.pipeline_options = pipeline_options
            print("[OneToOneLocalExtractor] Initialization complete.")
        except Exception as e:
            print(f"[OneToOneLocalExtractor] [red]Error during initialization:[/red] {e}")
            raise

    def extract(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        print(f"[OneToOneLocalExtractor] Starting extraction from [green]{source}[/green]...")
        result = self.extractor.extract(
            source=source, 
            template=template,
            default_pipeline_options=self.pipeline_options
        )
        
        extracted_models = []
        if not result.pages:
            print("[OneToOneLocalExtractor] [yellow]Warning:[/yellow] Document processing returned no pages.")
            return []
            
        for i, page in enumerate(result.pages):
            if page.extracted_data:
                extracted_models.append(page.extracted_data)
            else:
                print(f"[OneToOneLocalExtractor] No data extracted from page {i+1}")
                
        print(f"[OneToOneLocalExtractor] [green]Extraction successful.[/green] Found {len(extracted_models)} items.")
        return extracted_models
