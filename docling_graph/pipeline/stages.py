"""
Pipeline stages for modular execution.

This module defines individual pipeline stages that can be composed
to create flexible processing pipelines. Each stage is independent,
testable, and follows the single responsibility principle.
"""

import importlib
import logging
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
    InputType,
    InputTypeDetector,
    TextInputHandler,
    TextValidator,
    URLInputHandler,
    URLValidator,
)
from ..exceptions import ConfigurationError, ExtractionError, PipelineError
from ..llm_clients import BaseLlmClient, get_client
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

        # CLI mode: reject plain text input
        if self.mode == "cli" and input_type == InputType.TEXT:
            raise ConfigurationError(
                "Plain text input is only supported via Python API",
                details={
                    "source": str(context.config.source),
                    "mode": self.mode,
                    "hint": "Use a file path (.txt, .md) or URL instead",
                },
            )

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

        if input_type == InputType.TEXT:
            metadata = {
                "input_type": "text",
                "skip_ocr": True,
                "skip_segmentation": True,
                "original_source": "<raw_text>",
                "is_file": False,
            }
        elif input_type == InputType.TEXT_FILE:
            metadata = {
                "input_type": "text_file",
                "skip_ocr": True,
                "skip_segmentation": True,
                "original_source": str(source),
                "is_file": True,
            }
        elif input_type == InputType.MARKDOWN:
            metadata = {
                "input_type": "markdown",
                "skip_ocr": True,
                "skip_segmentation": True,
                "original_source": str(source),
                "is_file": True,
            }
        elif input_type == InputType.URL:
            # For URLs, normalized_content is a Path to downloaded file
            if isinstance(normalized_content, Path):
                # Detect the actual type of downloaded file
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
        elif input_type in (InputType.PDF, InputType.IMAGE):
            metadata = {
                "input_type": "pdf_or_image",
                "skip_ocr": False,
                "skip_segmentation": False,
                "original_source": str(source),
            }

        return metadata

    def _get_validator(self, input_type: InputType) -> Any:
        """Get appropriate validator for input type."""
        if input_type in (InputType.TEXT, InputType.TEXT_FILE, InputType.MARKDOWN):
            return TextValidator()
        elif input_type == InputType.URL:
            # Get URL config from context if available
            return URLValidator()
        elif input_type == InputType.DOCLING_DOCUMENT:
            return DoclingDocumentValidator()
        elif input_type in (InputType.PDF, InputType.IMAGE):
            # PDF and images don't need special validation beyond file existence
            # which is already done by InputTypeDetector
            return _NoOpValidator()
        else:
            raise ConfigurationError(
                f"No validator available for input type: {input_type.value}",
                details={"input_type": input_type.value},
            )

    def _get_handler(self, input_type: InputType) -> Any:
        """Get appropriate handler for input type."""
        if input_type in (InputType.TEXT, InputType.TEXT_FILE, InputType.MARKDOWN):
            return TextInputHandler()
        elif input_type == InputType.URL:
            # Get URL config from context if available
            return URLInputHandler()
        elif input_type == InputType.DOCLING_DOCUMENT:
            return DoclingDocumentHandler()
        elif input_type in (InputType.PDF, InputType.IMAGE):
            # PDF and images are handled by existing document processor
            # Return a pass-through handler
            return _PassThroughHandler()
        else:
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

            # Handle DoclingDocument input (already processed)
            if input_type == "docling_document":
                logger.info(f"[{self.name()}] Using pre-loaded DoclingDocument")
                context.extracted_models = self._extract_from_docling_document(context)
                # DoclingDocument is already in context.docling_document
                logger.info(f"[{self.name()}] Extracted {len(context.extracted_models)} items")
                return context

            # Handle text-based inputs (plain text, .txt, .md)
            elif input_type in ["text", "text_file", "markdown"]:
                logger.info(f"[{self.name()}] Processing text input (type: {input_type})")
                context.extracted_models = self._extract_from_text(context)
                # No DoclingDocument for text inputs
                context.docling_document = None
                logger.info(f"[{self.name()}] Extracted {len(context.extracted_models)} items")
                return context

        # Default path: PDF/Image processing (existing behavior)
        logger.info(f"[{self.name()}] Creating extractor...")
        context.extractor = self._create_extractor(context)

        logger.info(f"[{self.name()}] Extracting from: {context.config.source}")
        context.extracted_models, context.docling_document = context.extractor.extract(
            str(context.config.source), context.template
        )

        if not context.extracted_models:
            raise ExtractionError(
                "No models extracted from document", details={"source": context.config.source}
            )

        logger.info(f"[{self.name()}] Extracted {len(context.extracted_models)} items")

        # Capture page data if trace is enabled
        if context.trace_data and context.docling_document:
            from ..pipeline.trace import PageData

            for page_no in sorted(context.docling_document.pages.keys()):
                page_md = context.docling_document.export_to_markdown(page_no=page_no)
                page_data = PageData(
                    page_number=page_no,
                    text_content=page_md,
                    metadata={
                        "page_size": len(page_md),
                        "has_tables": any(
                            item.label == "table"
                            for item in context.docling_document.pages[page_no].items  # type: ignore[union-attr]
                        ),
                    },
                )
                context.trace_data.pages.append(page_data)

            logger.info(f"[{self.name()}] Captured {len(context.trace_data.pages)} pages")

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
                model_name=model_config["model"],
                docling_config=conf["docling_config"],
            )
        else:
            llm_client = self._initialize_llm_client(
                model_config["provider"], model_config["model"]
            )
            return ExtractorFactory.create_extractor(
                processing_mode=processing_mode,
                backend_name="llm",
                llm_client=llm_client,
                docling_config=conf["docling_config"],
                llm_consolidation=conf.get("llm_consolidation", True),
                use_chunking=conf.get("use_chunking", True),
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

        provider = provider_override or model_config.get(
            "provider", "ollama" if inference == "local" else "mistral"
        )

        if model_override:
            model = model_override
        elif provider_override and inference == "remote":
            providers = model_config.get("providers", {})
            model = providers.get(provider_override, {}).get(
                "default_model", model_config.get("default_model")
            )
        else:
            model = model_config.get("default_model")

        if not model:
            raise ConfigurationError(
                "Resolved model is empty", details={"backend": backend, "inference": inference}
            )

        return {"model": model, "provider": provider}

    @staticmethod
    def _initialize_llm_client(provider: str, model: str) -> BaseLlmClient:
        """Initialize LLM client based on provider."""
        client_class = get_client(provider)
        return client_class(model=model)

    def _extract_from_text(self, context: PipelineContext) -> List[Any]:
        """
        Extract from text-based inputs (plain text, .txt, .md).

        Skips document conversion and directly uses LLM extraction.

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

        logger.info(f"[{self.name()}] Extracting from text using LLM backend...")

        # Initialize LLM client
        inference = cast(str, conf["inference"])

        model_config = self._get_model_config(
            conf["models"],
            backend,
            inference,
            conf.get("model_override"),
            conf.get("provider_override"),
        )

        llm_client = self._initialize_llm_client(model_config["provider"], model_config["model"])

        # Import LlmBackend here to avoid circular imports
        from ..core.extractors.backends.llm_backend import LlmBackend

        llm_backend = LlmBackend(llm_client)

        # Extract directly from text
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

        extracted_model = llm_backend.extract_from_markdown(
            markdown=context.normalized_source,
            template=context.template,
            context="text input",
            is_partial=False,
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

        # Check if chunking is enabled
        use_chunking = context.config.to_dict().get("use_chunking", True)

        try:
            if use_chunking and doc_processor.chunker:
                # Use the extractor's chunk-based extraction method
                logger.info(f"[{self.name()}] Using chunk-based extraction")

                # Call the extractor's internal chunk extraction method if available
                if hasattr(context.extractor, "_extract_with_chunks"):
                    extracted_models: list[Any] = context.extractor._extract_with_chunks(
                        backend, context.docling_document, context.template
                    )
                else:
                    # Fallback: extract chunks and process them
                    chunks = doc_processor.extract_chunks(context.docling_document)
                    logger.info(f"[{self.name()}] Processing {len(chunks)} chunks")

                    # Process chunks through backend
                    partial_models = []
                    for i, chunk in enumerate(chunks):
                        logger.info(f"[{self.name()}] Extracting from chunk {i + 1}/{len(chunks)}")
                        model = backend.extract_from_markdown(
                            markdown=chunk,
                            template=context.template,
                            context=f"DoclingDocument chunk {i + 1}/{len(chunks)}",
                            is_partial=True,
                        )
                        if model:
                            partial_models.append(model)

                    # Merge partial models
                    if partial_models:
                        from ..core.utils.dict_merger import merge_pydantic_models

                        if context.template is None:
                            raise ExtractionError(
                                "Template is required for merging partial models",
                                details={"input_type": "docling_document"},
                            )
                        merged_model = merge_pydantic_models(partial_models, context.template)
                        extracted_models = [merged_model] if merged_model else partial_models[:1]
                    else:
                        extracted_models = []
            else:
                # No chunking - convert entire document to markdown
                logger.info(f"[{self.name()}] Converting DoclingDocument to markdown")
                markdown_text = context.docling_document.export_to_markdown()

                # Extract from the full markdown
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

            for i, model in enumerate(context.extracted_models):
                # Create individual graph for this model
                temp_graph, temp_metadata = converter.pydantic_list_to_graph([model])

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

        # Export to consolidated directory
        consolidated_dir = context.output_manager.get_consolidated_graph_dir()

        conf = context.config.to_dict()
        export_format = conf.get("export_format", "csv")

        if export_format == "csv":
            CSVExporter().export(context.knowledge_graph, consolidated_dir)
            logger.info(f"Saved CSV files to {consolidated_dir}")
        elif export_format == "cypher":
            cypher_path = consolidated_dir / "graph.cypher"
            CypherExporter().export(context.knowledge_graph, cypher_path)
            logger.info(f"Saved Cypher script to {cypher_path}")

        # Also export JSON
        json_path = consolidated_dir / "graph.json"
        JSONExporter().export(context.knowledge_graph, json_path)
        logger.info(f"Saved JSON to {json_path}")

        logger.info(f"[{self.name()}] Exported to {consolidated_dir}")
        return context


class TraceExportStage(PipelineStage):
    """Export trace data to disk."""

    def name(self) -> str:
        return "Trace Export"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Export all trace data using TraceExporter."""
        if not context.trace_data or not context.output_manager:
            logger.info(f"[{self.name()}] Skipped (no trace data or output manager)")
            return context

        logger.info(f"[{self.name()}] Exporting trace data...")

        from ..core.utils.trace_exporter import TraceExporter

        exporter = TraceExporter(context.output_manager)

        # Queue all trace data
        for page in context.trace_data.pages:
            exporter.queue_page_export(page.page_number, page)

        if context.trace_data.chunks:
            for chunk in context.trace_data.chunks:
                exporter.queue_chunk_export(chunk.chunk_id, chunk)
            exporter.queue_chunks_metadata(context.trace_data.chunks)

        for extraction in context.trace_data.extractions:
            exporter.queue_extraction_export(extraction)

        for graph in context.trace_data.intermediate_graphs:
            mode = "per_chunk" if graph.source_type == "chunk" else "per_page"
            exporter.queue_graph_export(graph, mode)

        # Flush all writes
        exporter.flush()

        logger.info(f"[{self.name()}] Exported {exporter.get_pending_count()} files")
        return context


class VisualizationStage(PipelineStage):
    """Generate visualizations and reports."""

    def name(self) -> str:
        return "Visualization"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Generate visualizations and reports."""
        logger.info(f"[{self.name()}] Generating visualizations...")

        conf = context.config.to_dict()
        base_name = Path(conf["source"]).stem

        # Ensure output_dir and extracted_models are not None
        if context.output_dir is None:
            raise PipelineError(
                "Output directory is required for visualization", details={"stage": self.name()}
            )
        if context.extracted_models is None:
            raise PipelineError(
                "No extracted models available for visualization", details={"stage": self.name()}
            )

        report_path = context.output_dir / f"{base_name}_report"
        ReportGenerator().visualize(
            context.knowledge_graph, report_path, source_model_count=len(context.extracted_models)
        )
        logger.info(f"Generated markdown report at {report_path}")

        html_path = context.output_dir / f"{base_name}_graph.html"
        InteractiveVisualizer().save_cytoscape_graph(context.knowledge_graph, html_path)
        logger.info(f"Generated interactive HTML graph at {html_path}")

        logger.info(f"[{self.name()}] Generated visualizations")
        return context
