"""
Shared document processing utilities.
"""

from typing import List
from rich import print
import gc

from docling.datamodel.base_models import InputFormat
from docling.datamodel import vlm_model_specs
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    VlmPipelineOptions,
)

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


class DocumentProcessor:
    """Handles document conversion to Markdown format."""
    
    def __init__(self, docling_config: str = "ocr"):
        """
        Initialize document processor with specified pipeline.
        
        Args:
            docling_config (str): Either "vision" or "ocr" by default.
                vision: Uses VLM pipeline for complex layouts.
                ocr: Uses classic OCR pipeline for standard documents.
        """
        self.docling_config = docling_config
        
        if docling_config == "vision":
            # VLM Pipeline - Best for complex layouts and images
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                    ),
                    InputFormat.IMAGE: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                    )
                }
            )
            print("[blue][DocumentProcessor][/blue] Initialized with [magenta]VLM pipeline[/magenta]")
        else:
            # Default Pipeline - Most accurate with OCR for standard documents
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            pipeline_options.ocr_options.lang = ["en", "fr"]
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=4,
                device=AcceleratorDevice.AUTO
            )
            
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
            print("[blue][DocumentProcessor][/blue] Initialized with [green]Classic OCR pipeline[/green] (French)")
    
    def convert_to_markdown(self, source: str):
        """
        Converts a document to Docling's Document format.
        
        Args:
            source (str): Path to the source document.
        
        Returns:
            Document: Docling document object.
        """
        print(f"[blue][DocumentProcessor][/blue] Converting document: [yellow]{source}[/yellow]")
        result = self.converter.convert(source)
        print(f"[blue][DocumentProcessor][/blue] Converted [cyan]{result.document.num_pages()}[/cyan] pages")
        return result.document
    
    def extract_page_markdowns(self, document) -> List[str]:
        """
        Extracts Markdown content for each page.
        
        Args:
            document (Document): Docling document object.
        
        Returns:
            List[str]: List of Markdown strings, one per page.
        """
        page_markdowns = []
        # Pages are indexed in the document.pages dict
        # They may start at 0 or 1 depending on the pipeline
        for page_no in sorted(document.pages.keys()):
            md = document.export_to_markdown(page_no=page_no)
            page_markdowns.append(md)
        
        print(f"[blue][DocumentProcessor][/blue] Extracted Markdown for [cyan]{len(page_markdowns)}[/cyan] pages")
        return page_markdowns
    
    def extract_full_markdown(self, document) -> str:
        """
        Extracts the full document as a single Markdown string.
        
        Args:
            document (Document): Docling document object.
        
        Returns:
            str: Complete document in Markdown format.
        """
        md = document.export_to_markdown()
        print(f"[blue][DocumentProcessor][/blue] Extracted full document Markdown ([cyan]{len(md)}[/cyan] chars)")
        return md
    
    def cleanup(self):
        """Clean up document converter resources."""
        try:
            if hasattr(self, 'converter'):
                del self.converter
            gc.collect()
            print("[blue][DocumentProcessor][/blue] [green]Cleaned up resources[/green]")
        except Exception as e:
            print(f"[blue][DocumentProcessor][/blue] [yellow]Warning during cleanup:[/yellow] {e}")
