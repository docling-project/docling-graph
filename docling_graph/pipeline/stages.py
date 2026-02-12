"""
Pipeline stages for modular execution.

This module defines individual pipeline stages that can be composed
to create flexible processing pipelines. Each stage is independent,
testable, and follows the single responsibility principle.
"""

import importlib
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, cast

from pydantic import BaseModel

from ..core import (
    CSVExporter,
    CypherExporter,
    DoclingExporter,
    ExtractorFactory,
    GraphConverter,
    InteractiveVisualizer,
    JSONExporter,
    ReportGenerator,
)
from ..core.input import (
    DoclingDocumentHandler,
    DoclingDocumentValidator,
    DocumentInputHandler,
    InputType,
    InputTypeDetector,
    URLInputHandler,
    URLValidator,
)
from ..exceptions import ConfigurationError, ExtractionError, PipelineError
from ..llm_clients import get_client
from ..protocols import LLMClientProtocol
from .context import PipelineContext

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """
    Base class for pipeline stages.

    Each stage implements a single step in the pipeline, receiving
    a context object, performing its work, and returning the updated
    context for the next stage.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the name of this stage for logging."""
        ...

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute this stage and return updated context.

        Args:
            context: Current pipeline context

        Returns:
            Updated pipeline context

        Raises:
            PipelineError: If stage execution fails
        """
        ...


class InputNormalizationStage(PipelineStage):
    """
    Normalize and validate input before processing.

    This stage:
    1. Detects input type (respecting CLI vs API mode)
    2. Validates input
    3. Loads and normalizes content
    4. Sets processing flags in context
    """

    def __init__(self, mode: Literal["cli", "api"] = "api") -> None:
        """
        Initialize stage with execution mode.

        Args:
            mode: "cli" for CLI invocations, "api" for Python API
        """
        self.mode = mode

    def name(self) -> str:
        return "Input Normalization"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Normalize input and set processing flags.

        Updates context with:
        - normalized_source: Processed input ready for extraction
        - input_metadata: Processing hints (skip_ocr, etc.)
        - input_type: Detected input type
        """
        logger.info(f"[{self.name()}] Detecting input type (mode: {self.mode})...")

        # Detect input type with mode awareness
        input_type = InputTypeDetector.detect(context.config.source, mode=self.mode)
        logger.info(f"[{self.name()}] Detected: {input_type.value}")

        # Get appropriate validator and handler
        validator = self._get_validator(input_type)
        handler = self._get_handler(input_type)

        # Validate input
        logger.info(f"[{self.name()}] Validating input...")
        validator.validate(context.config.source)

        # Load and normalize
        logger.info(f"[{self.name()}] Loading and normalizing input...")
        normalized_content = handler.load(context.config.source)

        # Build metadata based on input type
        metadata = self._build_metadata(input_type, context.config.source, normalized_content)

        # Update context
        # Special handling for DoclingDocument: store in docling_document field
        if input_type == InputType.DOCLING_DOCUMENT:
            from docling_core.types import DoclingDocument

            if isinstance(normalized_content, DoclingDocument):
                context.docling_document = normalized_content
                context.normalized_source = None  # Not needed for DoclingDocument
                logger.info(f"[{self.name()}] Loaded DoclingDocument into context")
            else:
                raise ConfigurationError(
                    "DoclingDocument handler did not return a DoclingDocument object",
                    details={"returned_type": type(normalized_content).__name__},
                )
        else:
            context.normalized_source = normalized_content

        context.input_metadata = metadata
        context.input_type = input_type

        logger.info(f"[{self.name()}] Normalized successfully")
        logger.info(
            f"[{self.name()}] Processing flags: skip_ocr={metadata.get('skip_ocr', False)}, "
            f"skip_segmentation={metadata.get('skip_segmentation', False)}"
        )

        return context

    def _build_metadata(
        self, input_type: InputType, source: Any, normalized_content: Any
    ) -> Dict[str, Any]:
        """Build metadata dictionary based on input type."""
        from pathlib import Path

        metadata: Dict[str, Any] = {}

        if input_type == InputType.URL:
            if isinstance(normalized_content, Path):
                detected_type = InputTypeDetector._detect_from_file(normalized_content)
                metadata = {
                    "input_type": "url",
                    "downloaded_path": str(normalized_content),
                    "original_url": str(source),
                    "detected_type": detected_type.value,
                    "is_temporary": True,
                }
        elif input_type == InputType.DOCLING_DOCUMENT:
            metadata = {
                "input_type": "docling_document",
                "skip_ocr": True,
                "skip_segmentation": True,
                "skip_document_conversion": True,
                "original_source": str(source),
                "is_file": True,
            }
        else:
            # DOCUMENT (all inputs sent to Docling for conversion)
            metadata = {
                "input_type": "document",
                "skip_ocr": False,
                "skip_segmentation": False,
                "original_source": str(source),
            }

        return metadata

    def _get_validator(self, input_type: InputType) -> Any:
        """Get appropriate validator for input type."""
        if input_type == InputType.URL:
            return URLValidator()
        if input_type == InputType.DOCLING_DOCUMENT:
            return DoclingDocumentValidator()
        if input_type == InputType.DOCUMENT:
            return _NoOpValidator()
        raise ConfigurationError(
            f"No validator available for input type: {input_type.value}",
            details={"input_type": input_type.value},
        )

    def _get_handler(self, input_type: InputType) -> Any:
        """Get appropriate handler for input type."""
        if input_type == InputType.URL:
            return URLInputHandler()
        if input_type == InputType.DOCLING_DOCUMENT:
            return DoclingDocumentHandler()
        if input_type == InputType.DOCUMENT:
            return DocumentInputHandler()
        raise ConfigurationError(
            f"No handler available for input type: {input_type.value}",
            details={"input_type": input_type.value},
        )


class _NoOpValidator:
    """No-op validator for types that don't need validation."""

    def validate(self, source: Any) -> None:
        pass


class _PassThroughHandler:
    """Pass-through handler for types handled by existing pipeline."""

    def load(self, source: Any) -> Any:
        # For PDF/Image, just return the source path as-is
        # The existing document processor will handle it
        # Metadata is built separately by _build_metadata()
        return source


class TemplateLoadingStage(PipelineStage):
    """Load and validate Pydantic template."""

    def name(self) -> str:
        return "Template Loading"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Load template from config."""
        logger.info(f"[{self.name()}] Loading template...")

        template_val = context.config.template
        if isinstance(template_val, str):
            context.template = self._load_from_string(template_val)
        elif isinstance(template_val, type):
            context.template = template_val
        else:
            raise ConfigurationError(
                "Invalid template type", details={"type": type(template_val).__name__}
            )

        logger.info(f"[{self.name()}] Loaded: {context.template.__name__}")
        return context

    @staticmethod
    def _load_from_string(template_str: str) -> type[BaseModel]:
        """
        Load template from dotted path.

        Args:
            template_str: Dotted path to template class

        Returns:
            Template class

        Raises:
            ConfigurationError: If template cannot be loaded
        """
        if "." not in template_str:
            raise ConfigurationError(
                "Template path must contain at least one dot",
                details={"template": template_str, "example": "module.Class"},
            )

        try:
            module_path, class_name = template_str.rsplit(".", 1)

            # Try importing as-is first
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError:
                # If that fails, try adding current directory to path temporarily
                import sys
                from pathlib import Path

                cwd = str(Path.cwd())
                if cwd not in sys.path:
                    sys.path.insert(0, cwd)
                    try:
                        module = importlib.import_module(module_path)
                    finally:
                        # Clean up: remove cwd from path
                        if cwd in sys.path:
                            sys.path.remove(cwd)
                else:
                    # cwd already in path, just try import
                    module = importlib.import_module(module_path)

            obj = getattr(module, class_name)

            if not isinstance(obj, type) or not issubclass(obj, BaseModel):
                raise ConfigurationError(
                    "Template must be a Pydantic BaseModel subclass",
                    details={"template": template_str, "type": type(obj).__name__},
                )

            return obj
        except (ModuleNotFoundError, AttributeError) as e:
            raise ConfigurationError(
                f"Failed to load template: {e}", details={"template": template_str}
            ) from e


class ExtractionStage(PipelineStage):
    """Execute document extraction."""

    def name(self) -> str:
        return "Extraction"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Run extraction on source document."""
        # Ensure template is not None before extraction
        if context.template is None:
            raise ExtractionError(
                "Template is required for extraction",
                details={"source": str(context.config.source)},
            )

        # Check if we have pre-normalized input
        if context.input_metadata:
            input_type = context.input_metadata.get("input_type")

            # Handle DoclingDocument input (skip conversion)
            if input_type == "docling_document":
                logger.info(f"[{self.name()}] Using pre-loaded DoclingDocument")
                context.extracted_models = self._extract_from_docling_document(context)
                logger.info(f"[{self.name()}] Extracted {len(context.extracted_models)} items")
                return context

        # All other inputs: Docling conversion path (file, URL download, text normalized to .md)
        logger.info(f"[{self.name()}] Creating extractor...")
        context.extractor = self._create_extractor(context)
        if context.trace_data and hasattr(context.extractor, "trace_data"):
            context.extractor.trace_data = context.trace_data

        # Use normalized path when available (URL download or DOCUMENT handler output)
        source_for_extract = (
            context.normalized_source
            if isinstance(context.normalized_source, Path)
            else context.config.source
        )
        logger.info(f"[{self.name()}] Extracting from: {source_for_extract}")
        context.extracted_models, context.docling_document = context.extractor.extract(
            str(source_for_extract), context.template
        )

        if not context.extracted_models:
            raise ExtractionError(
                "No models extracted from document", details={"source": context.config.source}
            )

        logger.info(f"[{self.name()}] Extracted {len(context.extracted_models)} items")

        return context

    def _create_extractor(self, context: PipelineContext) -> Any:
        """
        Create extractor from config.

        Args:
            context: Pipeline context with config

        Returns:
            Configured extractor instance
        """
        conf = context.config.to_dict()

        processing_mode = cast(Literal["one-to-one", "many-to-one"], conf["processing_mode"])
        extraction_contract = cast(
            Literal["direct", "staged"], conf.get("extraction_contract", "direct")
        )
        staged_config = {
            "max_pass_retries": conf.get("staged_pass_retries", 1),
            "catalog_max_nodes_per_call": conf.get("staged_nodes_fill_cap", 5),
            "parallel_workers": conf.get("staged_workers", 1),
            "id_shard_size": conf.get("staged_id_shard_size", 0),
            "id_identity_only": conf.get("staged_id_identity_only", True),
            "id_compact_prompt": conf.get("staged_id_compact_prompt", True),
            "id_auto_shard_threshold": conf.get("staged_id_auto_shard_threshold", 10),
            "id_shard_min_size": conf.get("staged_id_shard_min_size", 2),
            "quality_require_root": conf.get("staged_quality_require_root", True),
            "quality_min_instances": conf.get("staged_quality_min_instances", 1),
            "quality_max_parent_lookup_miss": conf.get("staged_quality_max_parent_lookup_miss", 0),
            "id_max_tokens": conf.get("staged_id_max_tokens"),
            "fill_max_tokens": conf.get("staged_fill_max_tokens"),
        }
        if conf.get("debug"):
            if context.output_manager is not None:
                staged_config["debug_dir"] = str(context.output_manager.get_debug_dir())
            elif conf.get("output_dir"):
                from pathlib import Path

                staged_config["debug_dir"] = str(Path(conf["output_dir"]) / "debug")
        backend = cast(Literal["vlm", "llm"], conf["backend"])
        inference = cast(str, conf["inference"])

        model_config = self._get_model_config(
            conf["models"],
            backend,
            inference,
            conf.get("model_override"),
            conf.get("provider_override"),
        )

        logger.info(f"Using model: {model_config['model']} (provider: {model_config['provider']})")

        if backend == "vlm":
            return ExtractorFactory.create_extractor(
                processing_mode=processing_mode,
                backend_name="vlm",
                extraction_contract=extraction_contract,
                staged_config=staged_config,
                model_name=model_config["model"],
                docling_config=conf["docling_config"],
            )
        else:
            if context.config.llm_client is not None:
                llm_client = context.config.llm_client
            else:
                llm_client = self._initialize_llm_client(
                    model_config["provider"],
                    model_config["model"],
                    context.config.llm_overrides,
                )
            return ExtractorFactory.create_extractor(
                processing_mode=processing_mode,
                backend_name="llm",
                extraction_contract=extraction_contract,
                staged_config=staged_config,
                llm_client=llm_client,
                docling_config=conf["docling_config"],
            )

    @staticmethod
    def _get_model_config(
        models_config: Dict[str, Any],
        backend: str,
        inference: str,
        model_override: str | None = None,
        provider_override: str | None = None,
    ) -> Dict[str, str]:
        """Retrieve model configuration based on settings."""
        model_config = models_config.get(backend, {}).get(inference, {})
        if not model_config:
            raise ConfigurationError(
                f"No configuration found for backend='{backend}' with inference='{inference}'",
                details={"backend": backend, "inference": inference},
            )

        provider = provider_override or model_config.get("provider")
        model = model_override or model_config.get("model")

        if not model:
            raise ConfigurationError(
                "Resolved model is empty", details={"backend": backend, "inference": inference}
            )

        return {"model": model, "provider": provider}

    @staticmethod
    def _initialize_llm_client(
        provider: str, model: str, overrides: Any | None = None
    ) -> LLMClientProtocol:
        """Initialize LLM client based on provider."""
        from docling_graph.llm_clients.config import (
            LlmRuntimeOverrides,
            resolve_effective_model_config,
        )

        client_class = get_client(provider)
        effective_config = resolve_effective_model_config(
            provider,
            model,
            overrides=overrides if isinstance(overrides, LlmRuntimeOverrides) else None,
        )
        return client_class(model_config=effective_config)

    def _extract_from_text(self, context: PipelineContext) -> List[Any]:
        """
        Extract from text-based inputs (plain text, .txt, .md).

        Uses a single LLM call for direct extraction.

        Args:
            context: Pipeline context with normalized text

        Returns:
            List of extracted Pydantic models

        Raises:
            ExtractionError: If extraction fails
        """
        if not context.normalized_source:
            input_type = (
                context.input_metadata.get("input_type") if context.input_metadata else "unknown"
            )
            raise ExtractionError(
                "No normalized text content available",
                details={"input_type": input_type},
            )

        # Only LLM backend supports text extraction
        conf = context.config.to_dict()
        backend = cast(Literal["vlm", "llm"], conf["backend"])

        if backend == "vlm":
            input_type = (
                context.input_metadata.get("input_type") if context.input_metadata else "unknown"
            )
            raise ExtractionError(
                "VLM backend does not support text-only inputs. Use LLM backend instead.",
                details={
                    "backend": backend,
                    "input_type": input_type,
                },
            )

        # Type assertions for mypy
        if not isinstance(context.normalized_source, str):
            raise ExtractionError(
                "Normalized source must be a string for text extraction",
                details={"type": type(context.normalized_source).__name__},
            )
        if context.template is None:
            raise ExtractionError(
                "Template is required for extraction",
                details={"template": None},
            )

        logger.info(f"[{self.name()}] Extracting from text using LLM backend (direct extraction)")

        # Initialize LLM client
        inference = cast(str, conf["inference"])

        model_config = self._get_model_config(
            conf["models"],
            backend,
            inference,
            conf.get("model_override"),
            conf.get("provider_override"),
        )

        if context.config.llm_client is not None:
            llm_client = context.config.llm_client
        else:
            llm_client = self._initialize_llm_client(
                model_config["provider"],
                model_config["model"],
                context.config.llm_overrides,
            )

        # Import LlmBackend here to avoid circular imports
        from ..core.extractors.backends.llm_backend import LlmBackend
        from .trace import ExtractionData

        extraction_contract = (
            context.config.extraction_contract
            if context.config.processing_mode == "many-to-one"
            else "direct"
        )
        staged_config = {
            "max_pass_retries": conf.get("staged_pass_retries", 1),
            "catalog_max_nodes_per_call": conf.get("staged_nodes_fill_cap", 5),
            "parallel_workers": conf.get("staged_workers", 1),
            "id_shard_size": conf.get("staged_id_shard_size", 0),
            "id_identity_only": conf.get("staged_id_identity_only", True),
            "id_compact_prompt": conf.get("staged_id_compact_prompt", True),
            "id_auto_shard_threshold": conf.get("staged_id_auto_shard_threshold", 12),
            "id_shard_min_size": conf.get("staged_id_shard_min_size", 2),
            "quality_require_root": conf.get("staged_quality_require_root", True),
            "quality_min_instances": conf.get("staged_quality_min_instances", 1),
            "quality_max_parent_lookup_miss": conf.get("staged_quality_max_parent_lookup_miss", 0),
            "id_max_tokens": conf.get("staged_id_max_tokens"),
            "fill_max_tokens": conf.get("staged_fill_max_tokens"),
        }
        if conf.get("debug"):
            if context.output_manager is not None:
                staged_config["debug_dir"] = str(context.output_manager.get_debug_dir())
            elif conf.get("output_dir"):
                from pathlib import Path

                staged_config["debug_dir"] = str(Path(conf["output_dir"]) / "debug")
        llm_backend = LlmBackend(
            llm_client,
            extraction_contract=extraction_contract,
            staged_config=staged_config,
        )
        if context.trace_data is not None:
            llm_backend.trace_data = context.trace_data

        start_time = time.time()
        extracted_model = llm_backend.extract_from_markdown(
            markdown=context.normalized_source,
            template=context.template,
            context="text input",
            is_partial=False,
        )
        extraction_time = time.time() - start_time

        if context.trace_data is not None:
            context.trace_data.extractions.append(
                ExtractionData(
                    extraction_id=0,
                    source_type="chunk",
                    source_id=0,
                    parsed_model=extracted_model,
                    extraction_time=extraction_time,
                    error=None,
                )
            )

        if not extracted_model:
            raise ExtractionError(
                "Failed to extract data from text input",
                details={"text_length": len(context.normalized_source)},
            )

        return [extracted_model]

    def _extract_from_docling_document(self, context: PipelineContext) -> List[Any]:
        """
        Extract from pre-loaded DoclingDocument.

        For DoclingDocument inputs, we use the extractor's internal methods
        to process the already-parsed document. This allows reprocessing of
        DoclingDocuments with different templates.

        Args:
            context: Pipeline context with DoclingDocument

        Returns:
            List of extracted Pydantic models

        Raises:
            ExtractionError: If DoclingDocument is not available or extraction fails
        """
        if not context.docling_document:
            raise ExtractionError(
                "No DoclingDocument available in context",
                details={"input_type": "docling_document"},
            )

        logger.info(f"[{self.name()}] Extracting from pre-loaded DoclingDocument")

        # Create extractor if not already created
        if not context.extractor:
            logger.info(f"[{self.name()}] Creating extractor for DoclingDocument...")
            context.extractor = self._create_extractor(context)
        if context.trace_data and hasattr(context.extractor, "trace_data"):
            context.extractor.trace_data = context.trace_data

        # Get the document processor and backend from the extractor
        doc_processor = getattr(context.extractor, "doc_processor", None)
        backend = getattr(context.extractor, "backend", None)

        if not doc_processor:
            raise ExtractionError(
                "Extractor does not have a document processor",
                details={"extractor_type": type(context.extractor).__name__},
            )

        if not backend:
            raise ExtractionError(
                "Extractor does not have a backend",
                details={"extractor_type": type(context.extractor).__name__},
            )

        try:
            # Convert entire document to markdown and extract in a single call
            logger.info(f"[{self.name()}] Converting DoclingDocument to markdown")
            markdown_text = context.docling_document.export_to_markdown()

            extracted_model = backend.extract_from_markdown(
                markdown=markdown_text,
                template=context.template,
                context="DoclingDocument",
                is_partial=False,
            )

            if not extracted_model:
                raise ExtractionError(
                    "Failed to extract data from DoclingDocument",
                    details={"markdown_length": len(markdown_text)},
                )

            extracted_models = [extracted_model]

            if not extracted_models:
                raise ExtractionError(
                    "No models extracted from DoclingDocument",
                    details={"input_type": "docling_document"},
                )

            logger.info(
                f"[{self.name()}] Extracted {len(extracted_models)} items from DoclingDocument"
            )
            return extracted_models

        except Exception as e:
            logger.error(f"[{self.name()}] Error extracting from DoclingDocument: {e}")
            raise ExtractionError(
                f"Failed to extract from DoclingDocument: {e!s}",
                details={"input_type": "docling_document", "error": str(e)},
            ) from e


class DoclingExportStage(PipelineStage):
    """Export Docling document outputs."""

    def name(self) -> str:
        return "Docling Export"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Export Docling document if configured."""
        conf = context.config.to_dict()

        if not (
            conf.get("export_docling", True)
            or conf.get("export_docling_json", True)
            or conf.get("export_markdown", True)
        ):
            logger.info(f"[{self.name()}] Skipped (not configured)")
            return context

        if not context.docling_document:
            logger.warning(f"[{self.name()}] No document available for export")
            return context

        if not context.output_manager:
            logger.warning(f"[{self.name()}] No output manager available")
            return context

        logger.info(f"[{self.name()}] Exporting Docling document...")

        docling_dir = context.output_manager.get_docling_dir()

        exporter = DoclingExporter(output_dir=docling_dir)
        exporter.export_document(
            context.docling_document,
            base_name="document",  # Use fixed name
            include_json=conf.get("export_docling_json", True),
            include_markdown=conf.get("export_markdown", True),
            per_page=conf.get("export_per_page_markdown", False),
        )

        logger.info(f"[{self.name()}] Exported to {docling_dir}")
        return context


class GraphConversionStage(PipelineStage):
    """Convert models to knowledge graph."""

    def name(self) -> str:
        return "Graph Conversion"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Convert extracted models to graph."""
        logger.info(f"[{self.name()}] Converting to graph...")

        converter = GraphConverter(
            add_reverse_edges=context.config.reverse_edges,
            validate_graph=True,
            registry=context.node_registry,
        )

        # Ensure extracted_models is not None
        if context.extracted_models is None:
            raise PipelineError(
                "No extracted models available for graph conversion", details={"stage": self.name()}
            )
        # Capture intermediate graphs if trace is enabled and in many-to-one mode
        if context.trace_data and context.config.processing_mode == "many-to-one":
            from ..pipeline.trace import GraphData

            # Use a converter without cleanup/validation for trace to avoid duplicate logs
            trace_converter = GraphConverter(
                add_reverse_edges=context.config.reverse_edges,
                validate_graph=False,
                auto_cleanup=False,
                registry=context.node_registry,
            )
            for i, model in enumerate(context.extracted_models):
                # Create individual graph for this model (no cleanup/validation logs)
                temp_graph, temp_metadata = trace_converter.pydantic_list_to_graph([model])

                graph_data = GraphData(
                    graph_id=i,
                    source_type="chunk",
                    source_id=i,
                    graph=temp_graph,
                    pydantic_model=model,
                    node_count=temp_metadata.node_count,
                    edge_count=temp_metadata.edge_count,
                )
                context.trace_data.intermediate_graphs.append(graph_data)

            logger.info(
                f"[{self.name()}] Captured {len(context.trace_data.intermediate_graphs)} intermediate graphs"
            )

        context.knowledge_graph, context.graph_metadata = converter.pydantic_list_to_graph(
            context.extracted_models
        )

        logger.info(
            f"[{self.name()}] Created graph: "
            f"{context.graph_metadata.node_count} nodes, "
            f"{context.graph_metadata.edge_count} edges"
        )
        return context


class ExportStage(PipelineStage):
    """Export graph in multiple formats."""

    def name(self) -> str:
        return "Export"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Export graph to configured formats."""
        if not context.output_manager:
            logger.warning(f"[{self.name()}] No output manager available")
            return context

        logger.info(f"[{self.name()}] Exporting graph...")

        # Export to docling_graph directory
        graph_dir = context.output_manager.get_docling_graph_dir()

        conf = context.config.to_dict()
        export_format = conf.get("export_format", "csv")

        if export_format == "csv":
            CSVExporter().export(context.knowledge_graph, graph_dir)
            logger.info(f"Saved CSV files to {graph_dir}")
        elif export_format == "cypher":
            cypher_path = graph_dir / "graph.cypher"
            CypherExporter().export(context.knowledge_graph, cypher_path)
            logger.info(f"Saved Cypher script to {cypher_path}")

        # Also export JSON
        json_path = graph_dir / "graph.json"
        JSONExporter().export(context.knowledge_graph, json_path)
        logger.info(f"Saved JSON to {json_path}")

        logger.info(f"[{self.name()}] Exported to {graph_dir}")
        return context


class VisualizationStage(PipelineStage):
    """Generate visualizations and reports."""

    def name(self) -> str:
        return "Visualization"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Generate visualizations and reports."""
        logger.info(f"[{self.name()}] Generating visualizations...")

        # Get output directory from output_manager or fallback to output_dir
        output_dir = None
        if context.output_manager:
            output_dir = context.output_manager.get_docling_graph_dir()
        elif context.output_dir:
            output_dir = context.output_dir

        # Ensure output_dir and extracted_models are not None
        if output_dir is None:
            raise PipelineError(
                "Output directory is required for visualization", details={"stage": self.name()}
            )
        if context.extracted_models is None:
            raise PipelineError(
                "No extracted models available for visualization", details={"stage": self.name()}
            )

        # Use generic filenames instead of source-based names
        report_path = output_dir / "report"
        extraction_contract = getattr(context.config, "extraction_contract", None)
        staged_passes_count = 0
        if context.trace_data:
            if getattr(context.trace_data, "staged_trace", None):
                staged_passes_count = 3
            elif hasattr(context.trace_data, "staged_passes"):
                staged_passes_count = len(context.trace_data.staged_passes)
        ReportGenerator().visualize(
            context.knowledge_graph,
            report_path,
            source_model_count=len(context.extracted_models),
            extraction_contract=extraction_contract,
            staged_passes_count=staged_passes_count,
        )
        logger.info(f"Generated markdown report at {report_path}.md")

        html_path = output_dir / "graph.html"
        InteractiveVisualizer().save_cytoscape_graph(context.knowledge_graph, html_path)
        logger.info(f"Generated interactive HTML graph at {html_path}")

        logger.info(f"[{self.name()}] Generated visualizations")
        return context
