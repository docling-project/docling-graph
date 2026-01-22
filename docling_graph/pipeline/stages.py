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
from typing import Any, Dict, Literal, cast

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

    @abstractmethod
    def name(self) -> str:
        """
        Return stage name for logging.

        Returns:
            Human-readable stage name
        """


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
                "Invalid template type",
                details={"type": type(template_val).__name__}
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
                details={"template": template_str, "example": "module.Class"}
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
                    details={"template": template_str, "type": type(obj).__name__}
                )

            return obj
        except (ModuleNotFoundError, AttributeError) as e:
            raise ConfigurationError(
                f"Failed to load template: {e}",
                details={"template": template_str}
            ) from e


class ExtractionStage(PipelineStage):
    """Execute document extraction."""

    def name(self) -> str:
        return "Extraction"

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Run extraction on source document."""
        logger.info(f"[{self.name()}] Creating extractor...")

        context.extractor = self._create_extractor(context)

        logger.info(f"[{self.name()}] Extracting from: {context.config.source}")
        # Ensure template is not None before extraction
        if context.template is None:
            raise ExtractionError(
                "Template is required for extraction",
                details={"source": str(context.config.source)}
            )
        context.extracted_models, context.docling_document = \
            context.extractor.extract(str(context.config.source), context.template)

        if not context.extracted_models:
            raise ExtractionError(
                "No models extracted from document",
                details={"source": context.config.source}
            )

        logger.info(
            f"[{self.name()}] Extracted {len(context.extracted_models)} items"
        )
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

        processing_mode = cast(
            Literal["one-to-one", "many-to-one"],
            conf["processing_mode"]
        )
        backend = cast(Literal["vlm", "llm"], conf["backend"])
        inference = cast(str, conf["inference"])

        model_config = self._get_model_config(
            conf["models"],
            backend,
            inference,
            conf.get("model_override"),
            conf.get("provider_override"),
        )

        logger.info(
            f"Using model: {model_config['model']} "
            f"(provider: {model_config['provider']})"
        )

        if backend == "vlm":
            return ExtractorFactory.create_extractor(
                processing_mode=processing_mode,
                backend_name="vlm",
                model_name=model_config["model"],
                docling_config=conf["docling_config"],
            )
        else:
            llm_client = self._initialize_llm_client(
                model_config["provider"],
                model_config["model"]
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
                details={"backend": backend, "inference": inference}
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
                "Resolved model is empty",
                details={"backend": backend, "inference": inference}
            )

        return {"model": model, "provider": provider}

    @staticmethod
    def _initialize_llm_client(provider: str, model: str) -> BaseLlmClient:
        """Initialize LLM client based on provider."""
        client_class = get_client(provider)
        return client_class(model=model)


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

        logger.info(f"[{self.name()}] Exporting Docling document...")

        output_dir = Path(conf.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(conf["source"]).stem

        exporter = DoclingExporter(output_dir=output_dir)
        exporter.export_document(
            context.docling_document,
            base_name=base_name,
            include_json=conf.get("export_docling_json", True),
            include_markdown=conf.get("export_markdown", True),
            per_page=conf.get("export_per_page_markdown", False),
        )

        logger.info(f"[{self.name()}] Exported to {output_dir}")
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
                "No extracted models available for graph conversion",
                details={"stage": self.name()}
            )
        context.knowledge_graph, context.graph_metadata = \
            converter.pydantic_list_to_graph(context.extracted_models)

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
        logger.info(f"[{self.name()}] Exporting graph...")

        conf = context.config.to_dict()
        context.output_dir = Path(conf.get("output_dir", "outputs"))
        context.output_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(conf["source"]).stem

        export_format = conf.get("export_format", "csv")

        if export_format == "csv":
            CSVExporter().export(context.knowledge_graph, context.output_dir)
            logger.info(f"Saved CSV files to {context.output_dir}")
        elif export_format == "cypher":
            cypher_path = context.output_dir / f"{base_name}_graph.cypher"
            CypherExporter().export(context.knowledge_graph, cypher_path)
            logger.info(f"Saved Cypher script to {cypher_path}")

        json_path = context.output_dir / f"{base_name}_graph.json"
        JSONExporter().export(context.knowledge_graph, json_path)
        logger.info(f"Saved JSON to {json_path}")

        logger.info(f"[{self.name()}] Exported to {context.output_dir}")
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
                "Output directory is required for visualization",
                details={"stage": self.name()}
            )
        if context.extracted_models is None:
            raise PipelineError(
                "No extracted models available for visualization",
                details={"stage": self.name()}
            )

        report_path = context.output_dir / f"{base_name}_report"
        ReportGenerator().visualize(
            context.knowledge_graph,
            report_path,
            source_model_count=len(context.extracted_models)
        )
        logger.info(f"Generated markdown report at {report_path}")

        html_path = context.output_dir / f"{base_name}_graph.html"
        InteractiveVisualizer().save_cytoscape_graph(
            context.knowledge_graph,
            html_path
        )
        logger.info(f"Generated interactive HTML graph at {html_path}")

        logger.info(f"[{self.name()}] Generated visualizations")
        return context

# Made with Bob
