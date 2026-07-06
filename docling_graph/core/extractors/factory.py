"""
Factory for creating extractors based on configuration.
"""

from typing import Any, Literal, cast

from ...logging_utils import get_component_logger
from ...protocols import Backend, LLMClientProtocol
from .backends.llm_backend import LlmBackend
from .backends.vlm_backend import VlmBackend
from .extractor_base import BaseExtractor
from .strategies.many_to_one import ManyToOneStrategy
from .strategies.one_to_one import OneToOneStrategy

logger = get_component_logger("ExtractorFactory", __name__)


class ExtractorFactory:
    """Factory for creating the right extractor combination."""

    @staticmethod
    def create_extractor(
        processing_mode: Literal["one-to-one", "many-to-one"],
        backend_name: Literal["vlm", "llm"],
        extraction_contract: Literal["direct", "dense", "auto"] = "direct",
        structured_output: bool = True,
        structured_sparse_check: bool = True,
        model_name: str | None = None,
        llm_client: LLMClientProtocol | None = None,
        docling_config: str = "ocr",
        use_chunking: bool = True,
        chunk_max_tokens: int | None = None,
        dense_config: dict[str, Any] | None = None,
        llm_input_format: str = "markdown",
        docling_serve_config: dict[str, Any] | None = None,
    ) -> BaseExtractor:
        """
        Create an extractor based on configuration.

        Args:
            processing_mode (str): 'one-to-one' or 'many-to-one'
            backend_name (str): 'vlm' or 'llm'
            extraction_contract (str): 'direct', 'dense', or 'auto' (LLM only);
                'auto' picks direct vs dense per document once its size is known
            model_name (str): Model name for VLM (optional)
            llm_client (LLMClientProtocol): LLM client instance (optional)
            docling_config (str): Docling pipeline configuration ('ocr' or 'vision')
            dense_config (dict): Dense/gleaning runtime settings forwarded to LlmBackend
                (batch tokens, fill cap, workers, resolvers, debug_dir, ...)
            docling_serve_config (dict): Remote docling-serve conversion settings
                (base_url, api_key, timeout); document conversion runs remotely
                when set. Applies to the LLM backend's conversion step only.

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
            if processing_mode != "many-to-one" and extraction_contract in ("dense", "auto"):
                logger.warning(
                    "The '%s' contract applies only to many-to-one "
                    "(one-to-one already extracts page by page); using direct.",
                    extraction_contract,
                )
                effective_contract = "direct"
            backend_obj = cast(
                Backend,
                LlmBackend(
                    llm_client=llm_client,
                    extraction_contract=effective_contract,
                    dense_config=dense_config,
                    structured_output=structured_output,
                    structured_sparse_check=structured_sparse_check,
                ),
            )
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

        # Create strategy with docling_config
        extractor: BaseExtractor

        if processing_mode == "one-to-one":
            extractor = OneToOneStrategy(
                backend=backend_obj,
                docling_config=docling_config,
                llm_input_format=llm_input_format,
                docling_serve_config=docling_serve_config,
            )
        elif processing_mode == "many-to-one":
            extractor = ManyToOneStrategy(
                backend=backend_obj,
                docling_config=docling_config,
                extraction_contract=effective_contract if backend_name == "llm" else "direct",
                use_chunking=use_chunking,
                chunk_max_tokens=chunk_max_tokens,
                llm_input_format=llm_input_format,
                docling_serve_config=docling_serve_config,
            )
        else:
            raise ValueError(f"Unknown processing_mode: {processing_mode}")

        logger.info("Created %s", extractor.__class__.__name__)
        return extractor
