"""
VLM (Vision-Language Model) extraction backend.
Handles document extraction using local VLM models via Docling.
"""

import gc
from typing import List, Type

import torch
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_extractor import DocumentExtractor, ExtractionFormatOption
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline
from pydantic import BaseModel, ValidationError
from rich import print as rich_print


class VlmBackend:
    """Backend for VLM-based extraction (local only)."""

    def __init__(self, model_name: str) -> None:
        """
        Initialize VLM backend with specified model.

        Args:
            model_name (str): HuggingFace model repository ID (e.g., 'numind/NuExtract-2.0-2B')
        """
        self.model_name = model_name
        self._initialize_extractor()

    def _initialize_extractor(self) -> None:
        """Initialize Docling's VLM extractor with custom settings."""
        try:
            # Get default VLM pipeline options
            pipeline_options = ExtractionVlmPipeline.get_default_options()
            # Use getattr guards to avoid static typing issues with mypy
            vlm_opts = getattr(pipeline_options, "vlm_options", None)
            if vlm_opts is not None and hasattr(vlm_opts, "repo_id"):
                vlm_opts.repo_id = self.model_name

            # Define custom format options - MUST include backend parameter
            custom_format_options = {
                InputFormat.PDF: ExtractionFormatOption(
                    pipeline_cls=ExtractionVlmPipeline,
                    backend=PyPdfiumDocumentBackend,
                    pipeline_options=pipeline_options,
                ),
                InputFormat.IMAGE: ExtractionFormatOption(
                    pipeline_cls=ExtractionVlmPipeline,
                    backend=PyPdfiumDocumentBackend,
                    pipeline_options=pipeline_options,
                ),
            }

            # Create extractor
            self.doc_extractor = DocumentExtractor(
                allowed_formats=[InputFormat.IMAGE, InputFormat.PDF],
                extraction_format_options=custom_format_options,
            )

            rich_print(
                f"[blue][VlmBackend][/blue] Initialized with model: [cyan]{self.model_name}[/cyan]"
            )

        except Exception as e:
            rich_print(f"[red]Error initializing VLM backend:[/red] {e}")
            raise

    def extract_from_document(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        """
        Extract structured data from entire document using VLM.

        Args:
            source (str): Path to source document.
            template (Type[BaseModel]): Pydantic model template.

        Returns:
            List[BaseModel]: List of extracted model instances (one per page/item).
        """
        rich_print(f"[blue][VlmBackend][/blue] Extracting from: [yellow]{source}[/yellow]")

        try:
            # Extract using VLM
            extraction_result = self.doc_extractor.extract(source=source, template=template)

            extracted_objects = []

            # Process each page's extracted data
            if extraction_result.pages:
                for page_num, page in enumerate(extraction_result.pages, 1):
                    if page.extracted_data:
                        try:
                            # Use model_validate for proper Pydantic validation
                            validated_model = template.model_validate(page.extracted_data)
                            extracted_objects.append(validated_model)
                        except ValidationError as e:
                            # Detailed error reporting like your original code
                            rich_print(
                                f"[blue][VlmBackend][/blue] [yellow]Validation Error on page {page_num}:[/yellow]"
                            )
                            rich_print(
                                "  The data extracted by the VLM does not match your Pydantic template."
                            )
                            rich_print("[red]Details:[/red]")
                            for error in e.errors():
                                loc = " -> ".join(map(str, error["loc"]))
                                rich_print(
                                    f"  - [bold magenta]{loc}[/bold magenta]: [red]{error['msg']}[/red]"
                                )
                            rich_print(
                                f"\n[yellow]Extracted Data (raw):[/yellow]\n{page.extracted_data}\n"
                            )
                            continue

            if extracted_objects:
                rich_print(
                    f"[blue][VlmBackend][/blue] Extracted [green]{len(extracted_objects)}[/green] valid items"
                )
            else:
                rich_print(
                    "[blue][VlmBackend][/blue] [yellow]Warning:[/yellow] No valid data extracted"
                )

            return extracted_objects

        except Exception as e:
            rich_print(f"[red]Error during VLM extraction:[/red] {e}")
            import traceback

            traceback.print_exc()
            return []

    def cleanup(self) -> None:
        """Clean up GPU memory and release resources."""
        try:
            # Delete the extractor to release document objects
            if hasattr(self, "doc_extractor"):
                del self.doc_extractor

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            rich_print(
                "[blue][VlmBackend][/blue] [green]Cleaned up resources and GPU memory[/green]"
            )

        except Exception as e:
            rich_print(f"[blue][VlmBackend][/blue] [yellow]Warning during cleanup:[/yellow] {e}")
