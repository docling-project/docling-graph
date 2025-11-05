"""
Factory for creating extractors based on configuration.
"""

from typing import Literal

from rich import print as rich_print

from ...llm_clients.base import BaseLlmClient
from ...protocols import Backend
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
        model_name: str | None = None,
        llm_client: BaseLlmClient | None = None,
        docling_config: str = "ocr",
        use_chunking: bool = True,
        llm_consolidation: bool = False,
    ) -> BaseExtractor:
        """
        Create an extractor based on configuration.

        Args:
            processing_mode (str): 'one-to-one' or 'many-to-one'
            backend_name (str): 'vlm' or 'llm'
            model_name (str): Model name for VLM (optional)
            llm_client (BaseLlmClient): LLM client instance (optional)
            docling_config (str): Docling pipeline configuration ('default' or 'vlm')
            llm_consolidation (bool): Whether to use LLM consolidation.
            use_chunking (bool): Whether to use chunking.

        Returns:
            BaseExtractor: Configured extractor instance.
        """
        rich_print("[blue][ExtractorFactory][/blue] Creating extractor:")
        rich_print(f" • Mode: [cyan]{processing_mode}[/cyan]")
        rich_print(f" • Type: [cyan]{backend_name}[/cyan]")
        rich_print(f" • Docling: [cyan]{docling_config}[/cyan]")
        # --- ADDED PRINT STATEMENTS START ---
        if backend_name == "llm":
            rich_print(f" • Consolidation: [cyan]{llm_consolidation}[/cyan]")
        rich_print(f" • Chunking: [cyan]{use_chunking}[/cyan]")
        # --- ADDED PRINT STATEMENTS END ---

        # Create backend instance
        backend_obj: Backend
        if backend_name == "vlm":
            if not model_name:
                raise ValueError("VLM requires model_name parameter")
            backend_obj = VlmBackend(model_name=model_name)
        elif backend_name == "llm":
            if not llm_client:
                raise ValueError("LLM requires llm_client parameter")
            backend_obj = LlmBackend(llm_client=llm_client)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

        # Create strategy with docling_config
        extractor: BaseExtractor

        strategy_args = {
            "backend": backend_obj,
            "docling_config": docling_config,
            "use_chunking": use_chunking,
        }

        # Only pass llm_consolidation if backend is llm
        if backend_name == "llm":
            strategy_args["llm_consolidation"] = llm_consolidation

        if processing_mode == "one-to-one":
            extractor = OneToOneStrategy(**strategy_args)
        elif processing_mode == "many-to-one":
            extractor = ManyToOneStrategy(**strategy_args)
        else:
            raise ValueError(f"Unknown processing_mode: {processing_mode}")

        rich_print(
            f"[blue][ExtractorFactory][/blue] Created [green]{extractor.__class__.__name__}[/green]"
        )
        return extractor
