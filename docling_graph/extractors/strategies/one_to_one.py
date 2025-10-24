"""
One-to-one extraction strategy.
Processes each page independently and returns multiple models.
"""

from pydantic import BaseModel
from typing import Type, List
from rich import print

from ..base import BaseExtractor
from ..document_processor import DocumentProcessor


class OneToOneStrategy(BaseExtractor):
    """
    One-to-one extraction strategy.
    Extracts one model per page/item.
    """
    
    def __init__(self, backend, docling_config: str = "default"):
        """
        Initialize with a backend (VlmBackend or LlmBackend).
        
        Args:
            backend: Extraction backend instance (VlmBackend or LlmBackend).
        """
        self.backend = backend
        self.doc_processor = DocumentProcessor(docling_config=docling_config)
        print(f"[OneToOneStrategy] Initialized with backend: [cyan]{backend.__class__.__name__}[/cyan]")
    
    def extract(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        """
        Extract data using one-to-one strategy.
        
        For VLM: Uses direct VLM extraction (already page-based).
        For LLM: Converts to markdown and processes each page separately.
        """
        backend_name = self.backend.__class__.__name__
        
        if backend_name == "VlmBackend":
            # VLM backend handles page-based extraction natively
            return self.backend.extract_from_document(source, template)
        
        elif backend_name == "LlmBackend":
            # LLM backend: convert document and process each page
            document = self.doc_processor.convert_to_markdown(source)
            page_markdowns = self.doc_processor.extract_page_markdowns(document)
            
            extracted_models = []
            for page_num, page_md in enumerate(page_markdowns, 1):
                print(f"[OneToOneStrategy] Processing page {page_num}/{len(page_markdowns)}")
                
                model = self.backend.extract_from_markdown(
                    markdown=page_md,
                    template=template,
                    context=f"page {page_num}"
                )
                
                if model:
                    extracted_models.append(model)
            
            print(f"[OneToOneStrategy] Extracted {len(extracted_models)} model(s)")
            return extracted_models
        
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")
