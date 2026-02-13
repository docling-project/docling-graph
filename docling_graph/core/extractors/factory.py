"""
Factory for creating extractors based on configuration.
"""

from typing import Literal

from rich import print as rich_print

from ...protocols import Backend, LLMClientProtocol
from .backends.llm_backend import LlmBackend
from .backends.vlm_backend import VlmBackend
from .extractor_base import BaseExtractor
from .strategies.many_to_one import ManyToOneStrategy
from .strategies.one_to_one import OneToOneStrategy


class ExtractorFactory:
    """Factory for creating the right extractor combination."""

    @staticmethod
    def create_extractor(
        processing_mode: Literal["one-to-one", "many-to-one"],
        backend_name: Literal["vlm", "llm"],
        extraction_contract: Literal["direct", "staged"] = "direct",
        structured_output: bool = True,
        structured_sparse_check: bool = True,
        staged_config: dict | None = None,
        model_name: str | None = None,
        llm_client: LLMClientProtocol | None = None,
        docling_config: str = "ocr",
    ) -> BaseExtractor:
        """
        Create an extractor based on configuration.

        Args:
            processing_mode (str): 'one-to-one' or 'many-to-one'
            backend_name (str): 'vlm' or 'llm'
            extraction_contract (str): 'direct' or 'staged' (LLM only)
            staged_config (dict): Optional staged extraction tuning config
            model_name (str): Model name for VLM (optional)
            llm_client (LLMClientProtocol): LLM client instance (optional)
            docling_config (str): Docling pipeline configuration ('ocr' or 'vision')

        Returns:
            BaseExtractor: Configured extractor instance.
        """

        # Create backend instance
        backend_obj: Backend
        if backend_name == "vlm":
            if not model_name:
                raise ValueError("VLM requires model_name parameter")
            backend_obj = VlmBackend(model_name=model_name)
        elif backend_name == "llm":
            if not llm_client:
                raise ValueError("LLM requires llm_client parameter")
            effective_contract = extraction_contract
            if processing_mode != "many-to-one" and extraction_contract == "staged":
                rich_print(
                    "[yellow][ExtractorFactory][/yellow] "
                    "Staged contract currently applies only to many-to-one; using direct."
                )
                effective_contract = "direct"
            backend_obj = LlmBackend(
                llm_client=llm_client,
                extraction_contract=effective_contract,
                staged_config=staged_config,
                structured_output=structured_output,
                structured_sparse_check=structured_sparse_check,
            )
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

        # Create strategy with docling_config
        extractor: BaseExtractor

        if processing_mode == "one-to-one":
            extractor = OneToOneStrategy(
                backend=backend_obj,
                docling_config=docling_config,
            )
        elif processing_mode == "many-to-one":
            extractor = ManyToOneStrategy(
                backend=backend_obj,
                docling_config=docling_config,
            )
        else:
            raise ValueError(f"Unknown processing_mode: {processing_mode}")

        rich_print(
            f"[blue][ExtractorFactory][/blue] Created [green]{extractor.__class__.__name__}[/green]"
        )
        return extractor
