"""
Pipeline stages for modular execution.

This module defines individual pipeline stages that can be composed
to create flexible processing pipelines. Each stage is independent,
testable, and follows the single responsibility principle.
"""

import importlib
import json
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
    DoclangInputHandler,
    DoclangValidator,
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
from ..logging_utils import get_component_logger
from ..protocols import LLMClientProtocol
from .context import PipelineContext


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

    @property
    def log(self) -> logging.LoggerAdapter:
        """Logger tagged with this stage's display name as the component."""
        return get_component_logger(self.name(), __name__)

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
        self.log.info(f"Detecting input type (mode: {self.mode})...")

        # Detect input type with mode awareness
        input_type = InputTypeDetector.detect(context.config.source, mode=self.mode)
        self.log.info(f"Detected: {input_type.value}")

        # Get appropriate validator and handler
        validator = self._get_validator(input_type)
        handler = self._get_handler(input_type)

        # Validate input
        self.log.info("Validating input...")
        validator.validate(context.config.source)

        # Load and normalize
        self.log.info("Loading and normalizing input...")
        normalized_content = handler.load(context.config.source)

        # Build metadata based on input type
        metadata = self._build_metadata(input_type, context.config.source, normalized_content)

        # Update context
        # DoclingDocument and DocLang both yield a pre-parsed document: store it
        # in docling_document and skip the conversion path.
        if input_type in (InputType.DOCLING_DOCUMENT, InputType.DOCLANG):
            from docling_core.types import DoclingDocument

            if isinstance(normalized_content, DoclingDocument):
                context.docling_document = normalized_content
                context.normalized_source = None  # Not needed for a pre-parsed document
                self.log.info(f"Loaded {input_type.value} into context")
            else:
                raise ConfigurationError(
                    f"{input_type.value} handler did not return a DoclingDocument object",
                    details={"returned_type": type(normalized_content).__name__},
                )
        else:
            context.normalized_source = normalized_content

        context.input_metadata = metadata
        context.input_type = input_type

        self.log.info("Normalized successfully")
        self.log.info(
            f"Processing flags: skip_ocr={metadata.get('skip_ocr', False)}, "
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
        elif input_type == InputType.DOCLANG:
            metadata = {
                "input_type": "doclang",
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
        if input_type == InputType.DOCLANG:
            return DoclangValidator()
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
        if input_type == InputType.DOCLANG:
            return DoclangInputHandler()
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


class TemplateLoadingStage(PipelineStage):
    """Load and validate Pydantic template."""

    def name(self) -> str:
        return "Template Loading"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Load template from config."""
        self.log.info("Loading template...")

        template_val = context.config.template
        if isinstance(template_val, str):
            context.template = self._load_from_string(template_val)
        elif isinstance(template_val, type):
            context.template = template_val
        else:
            raise ConfigurationError(
                "Invalid template type", details={"type": type(template_val).__name__}
            )

        self.log.info(f"Loaded: {context.template.__name__}")
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

            # Pre-parsed document input (DoclingDocument JSON or DocLang): skip conversion
            if input_type in ("docling_document", "doclang"):
                self.log.info(f"Using pre-loaded document ({input_type})")
                context.extracted_models = self._extract_from_docling_document(context)
                self.log.info(f"Extracted {len(context.extracted_models)} items")
                self._capture_provenance(context)
                return context

        # All other inputs: Docling conversion path (file, URL download, text normalized to .md)
        self.log.info("Creating extractor...")
        context.extractor = self._create_extractor(context)
        if context.trace_data and hasattr(context.extractor, "trace_data"):
            context.extractor.trace_data = context.trace_data

        # Use normalized path when available (URL download or DOCUMENT handler output)
        source_for_extract = (
            context.normalized_source
            if isinstance(context.normalized_source, Path)
            else context.config.source
        )
        self.log.info(f"Extracting from: {source_for_extract}")
        context.extracted_models, context.docling_document = context.extractor.extract(
            str(source_for_extract), context.template
        )

        if not context.extracted_models:
            raise ExtractionError(
                "No models extracted from document", details={"source": context.config.source}
            )

        self.log.info(f"Extracted {len(context.extracted_models)} items")

        self._capture_provenance(context)
        return context

    def _capture_provenance(self, context: PipelineContext) -> None:
        """Attach the extraction provenance ledger to the context (spec hook H7).

        Finalizes ``DocumentOrigin`` here: extraction internals know chunks,
        the stage knows source identity (path, input type, template).
        """
        if getattr(context.config, "provenance", "standard") == "off":
            return

        from ..core.provenance import DocumentOrigin, ProvenanceLedger, content_hash

        backend = getattr(context.extractor, "backend", None)
        ledger = getattr(backend, "last_provenance", None)
        # Strict type check: mocks and foreign backends must never smuggle a
        # non-ledger value into the context.
        if not isinstance(ledger, ProvenanceLedger):
            return

        source = str(context.config.source or "")
        input_type = "document"
        if context.input_metadata:
            source = str(context.input_metadata.get("original_source") or source)
            input_type = str(context.input_metadata.get("input_type") or input_type)

        document_id = ""
        candidate: Path | None = None
        if isinstance(context.normalized_source, Path):
            candidate = context.normalized_source
        elif context.config.source:
            maybe = Path(str(context.config.source))
            candidate = maybe if maybe.is_file() else None
        try:
            if candidate is not None and candidate.is_file():
                document_id = content_hash(candidate.read_bytes())
        except Exception:
            document_id = ""
        if not document_id:
            document_id = content_hash(source.encode("utf-8"))

        page_count = None
        if context.docling_document is not None:
            try:
                num_pages_fn = getattr(context.docling_document, "num_pages", None)
                if callable(num_pages_fn):
                    page_count = int(num_pages_fn() or 0) or None
            except Exception:
                page_count = None

        template_name = ""
        schema_hash = ""
        if context.template is not None:
            template_name = getattr(context.template, "__name__", "")
            try:
                import json as _json

                schema_hash = content_hash(
                    _json.dumps(context.template.model_json_schema(), sort_keys=True).encode(
                        "utf-8"
                    )
                )
            except Exception:
                schema_hash = ""

        ledger.document = DocumentOrigin(
            document_id=document_id,
            source=source,
            input_type=input_type,
            page_count=page_count,
            template_name=template_name,
            template_schema_hash=schema_hash,
        )
        context.provenance = ledger
        self.log.info(
            f"Captured provenance ledger: "
            f"{len(ledger.nodes)} node entries, {len(ledger.chunks)} chunks, "
            f"resolution={ledger.resolution}"
        )
        if context.trace_data is not None:
            context.trace_data.emit(
                "provenance_captured",
                "extraction",
                {
                    "document_id": document_id,
                    "resolution": ledger.resolution,
                    "node_entries": len(ledger.nodes),
                    "chunk_records": len(ledger.chunks),
                },
            )

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
            Literal["direct", "dense", "auto"], conf.get("extraction_contract", "direct")
        )
        dense_config = {
            "structured_output": bool(conf.get("structured_output", True)),
            "structured_sparse_check": bool(conf.get("structured_sparse_check", True)),
            "parallel_workers": conf.get("parallel_workers", 1),
            "gleaning_enabled": conf.get("gleaning_enabled", True),
            "dense_skeleton_batch_tokens": conf.get("dense_skeleton_batch_tokens", 1024),
            "dense_fill_nodes_cap": conf.get("dense_fill_nodes_cap", 5),
            "dense_fill_context": conf.get("dense_fill_context", "scoped"),
            "dense_dedupe": conf.get("dense_dedupe", "standard"),
            "provenance": conf.get("provenance", "standard"),
            "llm_input_format": conf.get("llm_input_format", "markdown"),
        }
        if conf.get("debug"):
            if context.output_manager is not None:
                dense_config["debug_dir"] = str(context.output_manager.get_debug_dir())
            elif conf.get("output_dir"):
                from pathlib import Path

                dense_config["debug_dir"] = str(Path(conf["output_dir"]) / "debug")
        backend = cast(Literal["vlm", "llm"], conf["backend"])
        inference = cast(str, conf["inference"])

        model_config = self._get_model_config(
            conf["models"],
            backend,
            inference,
            conf.get("model_override"),
            conf.get("provider_override"),
        )

        self.log.info(
            f"Using model: {model_config['model']} (provider: {model_config['provider']})"
        )

        llm_input_format = cast(str, conf.get("llm_input_format", "markdown"))

        if backend == "vlm":
            return ExtractorFactory.create_extractor(
                processing_mode=processing_mode,
                backend_name="vlm",
                extraction_contract=extraction_contract,
                model_name=model_config["model"],
                docling_config=conf["docling_config"],
                structured_output=bool(conf.get("structured_output", True)),
                structured_sparse_check=bool(conf.get("structured_sparse_check", True)),
                use_chunking=bool(conf.get("use_chunking", True)),
                chunk_max_tokens=conf.get("chunk_max_tokens"),
                llm_input_format=llm_input_format,
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
                llm_client=llm_client,
                docling_config=conf["docling_config"],
                structured_output=bool(conf.get("structured_output", True)),
                structured_sparse_check=bool(conf.get("structured_sparse_check", True)),
                use_chunking=bool(conf.get("use_chunking", True)),
                chunk_max_tokens=conf.get("chunk_max_tokens"),
                dense_config=dense_config,
                llm_input_format=llm_input_format,
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

    def _extract_from_docling_document(self, context: PipelineContext) -> List[Any]:
        """
        Extract from pre-loaded DoclingDocument.

        Delegates to the extractor strategy's extract_from_document() so the
        configured processing mode and extraction contract (including dense
        chunked extraction) apply to pre-converted documents as well.

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

        self.log.info("Extracting from pre-loaded DoclingDocument")

        if not context.extractor:
            self.log.info("Creating extractor for DoclingDocument...")
            context.extractor = self._create_extractor(context)
        if context.trace_data and hasattr(context.extractor, "trace_data"):
            context.extractor.trace_data = context.trace_data

        extract_from_document = getattr(context.extractor, "extract_from_document", None)
        if not callable(extract_from_document):
            raise ExtractionError(
                "Extractor does not support pre-converted DoclingDocument input",
                details={"extractor_type": type(context.extractor).__name__},
            )

        try:
            extracted_models, _ = extract_from_document(context.docling_document, context.template)
        except ExtractionError:
            raise
        except Exception as e:
            self.log.error(f"Error extracting from DoclingDocument: {e}")
            raise ExtractionError(
                f"Failed to extract from DoclingDocument: {e!s}",
                details={"input_type": "docling_document", "error": str(e)},
            ) from e

        if not extracted_models:
            raise ExtractionError(
                "No models extracted from DoclingDocument",
                details={"input_type": "docling_document"},
            )

        self.log.info(f"Extracted {len(extracted_models)} items from DoclingDocument")
        return extracted_models  # type: ignore[no-any-return]


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
            or conf.get("export_doclang", True)
        ):
            self.log.info("Skipped (not configured)")
            return context

        if not context.docling_document:
            self.log.warning("No document available for export")
            return context

        if not context.output_manager:
            self.log.warning("No output manager available")
            return context

        self.log.info("Exporting Docling document...")

        docling_dir = context.output_manager.get_docling_dir()

        # When the input itself was DocLang, the source file already is the
        # artifact — don't re-serialize (a lossy round trip).
        input_was_doclang = bool(
            context.input_metadata and context.input_metadata.get("input_type") == "doclang"
        )
        include_doclang = conf.get("export_doclang", True) and not input_was_doclang
        if input_was_doclang and conf.get("export_doclang", True):
            self.log.info("DocLang export skipped (input was DocLang)")

        exporter = DoclingExporter(output_dir=docling_dir)
        exporter.export_document(
            context.docling_document,
            base_name="document",  # Use fixed name
            include_json=conf.get("export_docling_json", True),
            include_markdown=conf.get("export_markdown", True),
            include_doclang=include_doclang,
            per_page=conf.get("export_per_page_markdown", False),
        )
        if context.trace_data is not None:
            context.trace_data.emit(
                "export_written",
                "docling_export",
                {
                    "target": str(docling_dir),
                    "export_docling_json": conf.get("export_docling_json", True),
                    "export_markdown": conf.get("export_markdown", True),
                    "export_doclang": include_doclang,
                    "export_per_page_markdown": conf.get("export_per_page_markdown", False),
                },
            )

        self._export_chunks(context, docling_dir)

        self.log.info(f"Exported to {docling_dir}")
        return context

    def _export_chunks(self, context: PipelineContext, docling_dir: Path) -> None:
        """Write chunks.json next to the Docling export (spec hook H11 sibling).

        The chunker's output is otherwise discarded once extraction finishes,
        which makes it impossible to compare provenance.json's chunk
        references back to the text the LLM actually read. This dumps the
        same ``ChunkRecord`` data captured in the provenance ledger, so it's
        available even when the ledger isn't produced yet (e.g. inspecting
        chunking before extraction runs).
        """
        ledger = context.provenance
        if ledger is None or not ledger.chunks:
            return

        chunks_payload = [
            record.model_dump(mode="json") for _, record in sorted(ledger.chunks.items())
        ]
        chunks_path = docling_dir / "chunks.json"
        chunks_path.write_text(
            json.dumps(chunks_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.log.info(f"Saved {len(chunks_payload)} chunk records to {chunks_path}")
        if context.trace_data is not None:
            context.trace_data.emit(
                "export_written",
                "docling_export",
                {"target": str(chunks_path), "chunk_count": len(chunks_payload)},
            )


class GraphConversionStage(PipelineStage):
    """Convert models to knowledge graph."""

    def name(self) -> str:
        return "Graph Conversion"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Convert extracted models to graph."""
        self.log.info("Converting to graph...")

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

        # Provenance binder (spec hook H9): injected as a closure so the
        # converter stays agnostic of the provenance module. Runs inside the
        # converter, after edge assembly and before cleanup.
        provenance_binder = None
        bind_stats: Dict[str, int] = {}
        if (
            context.provenance is not None
            and context.template is not None
            and getattr(context.config, "provenance", "standard") != "off"
        ):
            from ..core.provenance.binder import bind_provenance

            ledger = context.provenance
            template = context.template
            registry = context.node_registry
            # 'detailed' surfaces char spans in the node attribute; 'standard'
            # keeps them in provenance.json only.
            include_spans = getattr(context.config, "provenance", "standard") == "detailed"

            def provenance_binder(graph: Any, models: Any) -> None:
                bind_stats.update(
                    bind_provenance(
                        graph=graph,
                        models=models,
                        ledger=ledger,
                        registry=registry,
                        template=template,
                        include_spans=include_spans,
                    )
                )

        context.knowledge_graph, context.graph_metadata = converter.pydantic_list_to_graph(
            context.extracted_models,
            provenance_binder=provenance_binder,
        )
        if bind_stats and context.trace_data is not None:
            context.trace_data.emit("provenance_bound", "graph_conversion", dict(bind_stats))
        if context.trace_data is not None:
            context.trace_data.emit(
                "graph_created",
                "graph_conversion",
                {
                    "processing_mode": context.config.processing_mode,
                    "source_model_count": len(context.extracted_models),
                    "node_count": context.graph_metadata.node_count,
                    "edge_count": context.graph_metadata.edge_count,
                },
            )

        self.log.info(
            f"Created graph: "
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
            self.log.warning("No output manager available")
            return context

        self.log.info("Exporting graph...")

        # Export to docling_graph directory
        graph_dir = context.output_manager.get_docling_graph_dir()

        conf = context.config.to_dict()
        export_format = conf.get("export_format", "csv")

        if export_format == "csv":
            CSVExporter().export(context.knowledge_graph, graph_dir)
            self.log.info(f"Saved CSV files to {graph_dir}")
        elif export_format == "cypher":
            cypher_path = graph_dir / "graph.cypher"
            CypherExporter().export(context.knowledge_graph, cypher_path)
            self.log.info(f"Saved Cypher script to {cypher_path}")

        # Also export JSON
        json_path = graph_dir / "graph.json"
        JSONExporter().export(context.knowledge_graph, json_path)
        self.log.info(f"Saved JSON to {json_path}")

        # Persist the full provenance ledger next to the graph (spec hook H11)
        if (
            context.provenance is not None
            and getattr(context.config, "provenance", "standard") != "off"
        ):
            provenance_path = graph_dir / "provenance.json"
            provenance_path.write_text(
                context.provenance.model_dump_json(indent=2), encoding="utf-8"
            )
            self.log.info(f"Saved provenance ledger to {provenance_path}")
        if context.trace_data is not None:
            context.trace_data.emit(
                "export_written",
                "export",
                {
                    "target": str(graph_dir),
                    "format": export_format,
                    "json_path": str(json_path),
                },
            )

        self.log.info(f"Exported to {graph_dir}")
        return context


class VisualizationStage(PipelineStage):
    """Generate visualizations and reports."""

    def name(self) -> str:
        return "Visualization"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Generate visualizations and reports."""
        self.log.info("Generating visualizations...")

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
        llm_diagnostics: dict[str, Any] = {}
        if context.trace_data:
            extraction_events = context.trace_data.find_events("extraction_completed")
            if extraction_events:
                first_payload = extraction_events[0].payload
                first_meta = (
                    first_payload.get("metadata") if isinstance(first_payload, dict) else {}
                )
                if isinstance(first_meta, dict):
                    for key in (
                        "structured_attempted",
                        "structured_failed",
                        "fallback_used",
                        "fallback_error_class",
                    ):
                        if key in first_meta:
                            llm_diagnostics[key] = first_meta[key]
        extraction_backend = getattr(context.extractor, "backend", None)
        dense_stats = getattr(extraction_backend, "last_dense_stats", None)
        ReportGenerator().visualize(
            context.knowledge_graph,
            report_path,
            source_model_count=len(context.extracted_models),
            extraction_contract=extraction_contract,
            llm_diagnostics=llm_diagnostics,
            dense_stats=dense_stats if isinstance(dense_stats, dict) and dense_stats else None,
        )
        self.log.info(f"Generated markdown report at {report_path}.md")

        html_path = output_dir / "graph.html"
        InteractiveVisualizer().save_cytoscape_graph(context.knowledge_graph, html_path)
        self.log.info(f"Generated interactive HTML graph at {html_path}")
        if context.trace_data is not None:
            context.trace_data.emit(
                "export_written",
                "visualization",
                {"report_path": str(report_path) + ".md", "html_path": str(html_path)},
            )

        self.log.info("Generated visualizations")
        return context
