"""
Convert command - converts documents to knowledge graphs.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import typer
from rich import print as rich_print
from typing_extensions import Annotated

from docling_graph.config import PipelineConfig
from docling_graph.exceptions import (
    ConfigurationError,
    DoclingGraphError,
    ExtractionError,
    PipelineError,
)
from docling_graph.pipeline import run_pipeline

from ..config_utils import load_config
from ..validators import (
    validate_backend_type,
    validate_docling_config,
    validate_export_format,
    validate_extraction_contract,
    validate_inference,
    validate_processing_mode,
    validate_vlm_constraints,
)

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path("outputs")


def _resolve_cli_settings(
    defaults: dict[str, Any],
    docling_cfg: dict[str, Any],
    processing_mode: str | None,
    extraction_contract: str | None,
    backend: str | None,
    inference: str | None,
    export_format: str | None,
    docling_pipeline: str | None,
    llm_input_format: str | None,
    chunk_max_tokens: int | None,
    parallel_workers: int | None,
    dense_skeleton_batch_tokens: int | None,
    dense_fill_nodes_cap: int | None,
    dense_fill_context: str | None,
    dense_dedupe: str | None,
    provenance: str | None,
    schema_enforced_llm: bool | None,
    structured_sparse_check: bool | None,
    gleaning_enabled: bool | None,
    export_docling_json: bool,
    export_markdown: bool,
    export_doclang: bool,
    export_per_page: bool,
) -> dict[str, Any]:
    """Resolve CLI and config-file settings into effective values."""
    processing_mode_val = processing_mode or defaults.get("processing_mode", "many-to-one")
    extraction_contract_val = extraction_contract or defaults.get("extraction_contract", "direct")
    backend_val = backend or defaults.get("backend", "llm")
    inference_val = inference or defaults.get("inference", "local")
    export_format_val = export_format or defaults.get("export_format", "csv")
    docling_pipeline_val = docling_pipeline or docling_cfg.get("pipeline", "ocr")

    final_llm_input_format = str(
        llm_input_format
        if llm_input_format is not None
        else defaults.get("llm_input_format", "markdown")
    ).lower()
    if final_llm_input_format not in ("markdown", "doclang", "doclang-geo"):
        final_llm_input_format = "markdown"

    final_dense_dedupe = str(
        dense_dedupe if dense_dedupe is not None else defaults.get("dense_dedupe", "standard")
    ).lower()
    if final_dense_dedupe not in ("off", "standard", "aggressive"):
        final_dense_dedupe = "standard"

    final_dense_fill_context = str(
        dense_fill_context
        if dense_fill_context is not None
        else defaults.get("dense_fill_context", "scoped")
    ).lower()
    if final_dense_fill_context not in ("scoped", "full"):
        final_dense_fill_context = "scoped"

    final_provenance = str(
        provenance if provenance is not None else defaults.get("provenance", "standard")
    ).lower()
    if final_provenance not in ("off", "standard", "detailed"):
        final_provenance = "standard"

    docling_export_settings = docling_cfg.get("export", {})

    return {
        "processing_mode": processing_mode_val,
        "extraction_contract": extraction_contract_val,
        "backend": backend_val,
        "inference": inference_val,
        "export_format": export_format_val,
        "docling_pipeline": docling_pipeline_val,
        "llm_input_format": final_llm_input_format,
        "chunk_max_tokens": (
            chunk_max_tokens if chunk_max_tokens is not None else defaults.get("chunk_max_tokens")
        ),
        "parallel_workers": (
            parallel_workers if parallel_workers is not None else defaults.get("parallel_workers")
        ),
        "dense_skeleton_batch_tokens": int(
            dense_skeleton_batch_tokens
            if dense_skeleton_batch_tokens is not None
            else defaults.get("dense_skeleton_batch_tokens", 1024) or 1024
        ),
        "dense_fill_nodes_cap": int(
            dense_fill_nodes_cap
            if dense_fill_nodes_cap is not None
            else defaults.get("dense_fill_nodes_cap", 5) or 5
        ),
        "dense_fill_context": final_dense_fill_context,
        "dense_dedupe": final_dense_dedupe,
        "provenance": final_provenance,
        "structured_output": (
            schema_enforced_llm
            if schema_enforced_llm is not None
            else bool(defaults.get("structured_output", True))
        ),
        "structured_sparse_check": (
            structured_sparse_check
            if structured_sparse_check is not None
            else bool(defaults.get("structured_sparse_check", True))
        ),
        "gleaning_enabled": (
            gleaning_enabled
            if gleaning_enabled is not None
            else bool(defaults.get("gleaning_enabled", True))
        ),
        "export_docling_json": (
            export_docling_json
            if export_docling_json is not None
            else docling_export_settings.get("docling_json", True)
        ),
        "export_markdown": (
            export_markdown
            if export_markdown is not None
            else docling_export_settings.get("markdown", True)
        ),
        "export_doclang": (
            export_doclang
            if export_doclang is not None
            else docling_export_settings.get("doclang", True)
        ),
        "export_per_page": (
            export_per_page
            if export_per_page is not None
            else docling_export_settings.get("per_page_markdown", False)
        ),
    }


def _validate_resolved_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Validate effective CLI settings."""
    settings["processing_mode"] = validate_processing_mode(settings["processing_mode"])
    settings["extraction_contract"] = validate_extraction_contract(settings["extraction_contract"])
    settings["backend"] = validate_backend_type(settings["backend"])
    settings["inference"] = validate_inference(settings["inference"])
    settings["docling_pipeline"] = validate_docling_config(settings["docling_pipeline"])
    settings["export_format"] = validate_export_format(settings["export_format"])
    validate_vlm_constraints(settings["backend"], settings["inference"])
    return settings


def convert_command(
    source: Annotated[
        str,
        typer.Argument(
            help="Path to source document (any Docling-supported format), URL, DoclingDocument JSON, or DocLang file (.dclg/.dclx). DoclingDocument and DocLang skip conversion.",
        ),
    ],
    template: Annotated[
        str,
        typer.Option(
            "--template",
            "-t",
            help="Dotted path to Pydantic template (e.g., 'templates.billing_document.BillingDocument').",
        ),
    ],
    processing_mode: Annotated[
        str | None,
        typer.Option(
            "--processing-mode", "-p", help="Processing strategy: 'one-to-one' or 'many-to-one'."
        ),
    ] = None,
    extraction_contract: Annotated[
        str | None,
        typer.Option(
            "--extraction-contract",
            help=(
                "Extraction contract: 'direct' (single full-document call), 'dense' "
                "(skeleton-then-fill over chunks), or 'auto' (picks direct or dense per "
                "document based on its size vs. the model's context window and output budget)."
            ),
        ),
    ] = None,
    backend: Annotated[
        str | None, typer.Option("--backend", "-b", help="Backend: 'llm' or 'vlm'.")
    ] = None,
    inference: Annotated[
        str | None, typer.Option("--inference", "-i", help="Inference: 'local' or 'remote'.")
    ] = None,
    docling_pipeline: Annotated[
        str | None,
        typer.Option("--docling-pipeline", "-d", help="Docling pipeline: 'ocr' or 'vision'."),
    ] = None,
    llm_input_format: Annotated[
        str | None,
        typer.Option(
            "--llm-format",
            help=(
                "Serialization sent to the LLM: markdown (default) | doclang | doclang-geo. "
                "DocLang preserves structure/geometry at a higher token cost."
            ),
        ),
    ] = None,
    # Extraction options
    debug: Annotated[
        bool,
        typer.Option(
            "--debug/--no-debug",
            help="Enable debug artifacts.",
        ),
    ] = False,
    schema_enforced_llm: Annotated[
        bool | None,
        typer.Option(
            "--schema-enforced-llm/--no-schema-enforced-llm",
            help="Use API schema-enforced output for LLM calls (default: enabled).",
        ),
    ] = None,
    structured_sparse_check: Annotated[
        bool | None,
        typer.Option(
            "--structured-sparse-check/--no-structured-sparse-check",
            help="Enable sparse structured-output check and legacy retry (default: enabled).",
        ),
    ] = None,
    chunk_max_tokens: Annotated[
        int | None,
        typer.Option(
            "--chunk-max-tokens",
            help="Max tokens per chunk when chunking is used (default: 512).",
        ),
    ] = None,
    parallel_workers: Annotated[
        int | None,
        typer.Option(
            "--parallel-workers",
            help="Parallel workers for extraction.",
        ),
    ] = None,
    dense_skeleton_batch_tokens: Annotated[
        int | None,
        typer.Option(
            "--dense-skeleton-batch-tokens",
            help="Max tokens per dense Phase 1 (skeleton) chunk batch (default: 1024).",
        ),
    ] = None,
    dense_fill_nodes_cap: Annotated[
        int | None,
        typer.Option(
            "--dense-fill-nodes-cap",
            help="Max node instances per dense Phase 2 (fill) LLM call (default: 5).",
        ),
    ] = None,
    dense_fill_context: Annotated[
        str | None,
        typer.Option(
            "--dense-fill-context",
            help="Document context per dense fill call: scoped | full (default: scoped).",
        ),
    ] = None,
    dense_dedupe: Annotated[
        str | None,
        typer.Option(
            "--dense-dedupe",
            help="Skeleton dedupe intensity: off | standard | aggressive (default: standard).",
        ),
    ] = None,
    provenance: Annotated[
        str | None,
        typer.Option(
            "--provenance",
            help=(
                "Deterministic node-to-source grounding: off | standard | detailed "
                "(default: standard; 'detailed' adds verbatim char spans)."
            ),
        ),
    ] = None,
    # Docling export options
    export_docling_json: Annotated[
        bool,
        typer.Option(
            "--export-docling-json/--no-docling-json", help="Export Docling document as JSON."
        ),
    ] = True,
    export_markdown: Annotated[
        bool, typer.Option("--export-markdown/--no-markdown", help="Export full document markdown.")
    ] = True,
    export_doclang: Annotated[
        bool,
        typer.Option(
            "--export-doclang/--no-doclang",
            help="Export the Docling document as DocLang (.dclg, content+geometry).",
        ),
    ] = True,
    export_per_page: Annotated[
        bool,
        typer.Option("--export-per-page/--no-per-page", help="Export per-page markdown files."),
    ] = False,
    # Output options
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o", help="Output directory.", file_okay=False, writable=True
        ),
    ] = DEFAULT_OUTPUT_DIR,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override model name.")] = None,
    provider: Annotated[str | None, typer.Option("--provider", help="Override provider.")] = None,
    llm_temperature: Annotated[
        float | None, typer.Option("--llm-temperature", help="Override LLM temperature.")
    ] = None,
    llm_max_tokens: Annotated[
        int | None, typer.Option("--llm-max-tokens", help="Override LLM max tokens.")
    ] = None,
    llm_top_p: Annotated[
        float | None, typer.Option("--llm-top-p", help="Override LLM top_p.")
    ] = None,
    llm_timeout: Annotated[
        int | None, typer.Option("--llm-timeout", help="Override LLM timeout in seconds.")
    ] = None,
    llm_retries: Annotated[
        int | None, typer.Option("--llm-retries", help="Override LLM max retries.")
    ] = None,
    llm_base_url: Annotated[
        str | None, typer.Option("--llm-base-url", help="Override LLM base URL.")
    ] = None,
    llm_context_limit: Annotated[
        int | None,
        typer.Option(
            "--llm-context-limit",
            help="Override LLM context limit (total context window size in tokens).",
        ),
    ] = None,
    llm_max_output_tokens: Annotated[
        int | None,
        typer.Option(
            "--llm-max-output-tokens",
            help="Override LLM max output tokens (maximum tokens the model can generate).",
        ),
    ] = None,
    llm_streaming: Annotated[
        bool | None,
        typer.Option(
            "--llm-streaming/--no-llm-streaming",
            help="Enable streaming responses from LLM to avoid timeout issues in constrained infrastructures (default: disabled).",
        ),
    ] = None,
    show_llm_config: Annotated[
        bool,
        typer.Option(
            "--show-llm-config/--no-show-llm-config",
            help="Print resolved LLM config and exit.",
        ),
    ] = False,
    export_format: Annotated[
        str | None,
        typer.Option("--export-format", "-e", help="Export format: 'csv' or 'cypher'."),
    ] = None,
    reverse_edges: Annotated[
        bool, typer.Option("--reverse-edges", "-r", help="Create bidirectional edges.")
    ] = False,
    gleaning_enabled: Annotated[
        bool | None,
        typer.Option(
            "--gleaning/--no-gleaning",
            help="Run a second-pass extraction to improve recall (direct contract). Default: enabled.",
        ),
    ] = None,
) -> None:
    """Convert a document to a knowledge graph."""
    logger.debug("Starting convert command")
    logger.debug(f"Source: {source}, Template: {template}")
    logger.debug(
        f"CLI args - Backend: {backend}, Inference: {inference}, Processing: {processing_mode}"
    )

    # Load YAML configuration (flat)
    logger.debug("Loading configuration from config.yaml")
    config_data = load_config()
    logger.debug(f"Loaded config keys: {list(config_data.keys())}")
    defaults = config_data.get("defaults", {})
    docling_cfg = config_data.get("docling", {})
    models_from_yaml = config_data.get("models", {})
    llm_overrides = config_data.get("llm_overrides", {}) or {}

    settings = _validate_resolved_settings(
        _resolve_cli_settings(
            defaults=defaults,
            docling_cfg=docling_cfg,
            processing_mode=processing_mode,
            extraction_contract=extraction_contract,
            backend=backend,
            inference=inference,
            export_format=export_format,
            docling_pipeline=docling_pipeline,
            llm_input_format=llm_input_format,
            chunk_max_tokens=chunk_max_tokens,
            parallel_workers=parallel_workers,
            dense_skeleton_batch_tokens=dense_skeleton_batch_tokens,
            dense_fill_nodes_cap=dense_fill_nodes_cap,
            dense_fill_context=dense_fill_context,
            dense_dedupe=dense_dedupe,
            provenance=provenance,
            schema_enforced_llm=schema_enforced_llm,
            structured_sparse_check=structured_sparse_check,
            gleaning_enabled=gleaning_enabled,
            export_docling_json=export_docling_json,
            export_markdown=export_markdown,
            export_doclang=export_doclang,
            export_per_page=export_per_page,
        )
    )

    processing_mode_val = settings["processing_mode"]
    extraction_contract_val = settings["extraction_contract"]
    backend_val = settings["backend"]
    inference_val = settings["inference"]
    export_format_val = settings["export_format"]
    docling_pipeline_val = settings["docling_pipeline"]
    final_llm_input_format = settings["llm_input_format"]
    final_chunk_max_tokens = settings["chunk_max_tokens"]
    final_structured_output = settings["structured_output"]
    final_structured_sparse_check = settings["structured_sparse_check"]
    final_gleaning_enabled = settings["gleaning_enabled"]
    final_export_docling_json = settings["export_docling_json"]
    final_export_markdown = settings["export_markdown"]
    final_export_doclang = settings["export_doclang"]
    final_export_per_page = settings["export_per_page"]

    logger.debug(f"Validated configuration - Backend: {backend_val}, Inference: {inference_val}")
    logger.debug(f"Processing mode: {processing_mode_val}, Export format: {export_format_val}")

    # Detect and display input type
    from docling_graph.core.input.types import InputTypeDetector

    try:
        detected_type = InputTypeDetector.detect(source, mode="cli")
        input_type_display = detected_type.value.replace("_", " ").title()
    except Exception:
        input_type_display = "Unknown"

    # Display the resolved configuration, one compact line per section.
    logger.info(
        "source=%s, template=%s, input_type=%s, docling=%s, processing=%s, "
        "inference=%s, export=%s, reverse_edges=%s",
        source,
        template,
        input_type_display,
        docling_pipeline_val,
        processing_mode_val,
        inference_val,
        export_format_val,
        reverse_edges,
        extra={"component": "PipelineConfiguration"},
    )
    logger.info(
        "document_json=%s, markdown=%s, doclang=%s, per_page_md=%s",
        final_export_docling_json,
        final_export_markdown,
        final_export_doclang,
        final_export_per_page,
        extra={"component": "DoclingExport"},
    )
    extraction_summary = (
        f"backend={backend_val}, contract={extraction_contract_val}, "
        f"llm_input_format={final_llm_input_format}, "
        f"structured_output={final_structured_output}, "
        f"structured_sparse_check={final_structured_sparse_check}, "
        f"provenance={settings['provenance']}, debug={debug}"
    )
    if extraction_contract_val in ("direct", "auto"):
        extraction_summary += f", gleaning={final_gleaning_enabled}"
    if final_chunk_max_tokens is not None:
        extraction_summary += f", chunk_max_tokens={final_chunk_max_tokens}"
    logger.info(extraction_summary, extra={"component": "ExtractionSettings"})
    if extraction_contract_val in ("dense", "auto"):
        logger.info(
            "skeleton_batch_tokens=%s, fill_nodes_cap=%s, fill_context=%s, dedupe=%s",
            settings["dense_skeleton_batch_tokens"],
            settings["dense_fill_nodes_cap"],
            settings["dense_fill_context"],
            settings["dense_dedupe"],
            extra={"component": "DenseTuning"},
        )

    # Build typed config
    logger.debug("Building PipelineConfig object")
    if llm_temperature is not None:
        llm_overrides.setdefault("generation", {})["temperature"] = llm_temperature
    if llm_max_tokens is not None:
        llm_overrides.setdefault("generation", {})["max_tokens"] = llm_max_tokens
    if llm_top_p is not None:
        llm_overrides.setdefault("generation", {})["top_p"] = llm_top_p
    if llm_timeout is not None:
        llm_overrides.setdefault("reliability", {})["timeout_s"] = llm_timeout
    if llm_retries is not None:
        llm_overrides.setdefault("reliability", {})["max_retries"] = llm_retries
    if llm_base_url is not None:
        llm_overrides.setdefault("connection", {})["base_url"] = llm_base_url
    if llm_context_limit is not None:
        llm_overrides["context_limit"] = llm_context_limit
    if llm_max_output_tokens is not None:
        llm_overrides["max_output_tokens"] = llm_max_output_tokens
    if llm_streaming is not None:
        llm_overrides["streaming"] = llm_streaming

    cfg = PipelineConfig(
        source=str(source),
        template=template,
        backend=backend_val,
        inference=inference_val,
        processing_mode=processing_mode_val,
        extraction_contract=extraction_contract_val,
        docling_config=docling_pipeline_val,
        model_override=model,
        provider_override=provider,
        models=models_from_yaml,
        llm_overrides=llm_overrides,
        structured_output=final_structured_output,
        structured_sparse_check=final_structured_sparse_check,
        llm_input_format=final_llm_input_format,
        debug=debug,
        chunk_max_tokens=final_chunk_max_tokens,
        parallel_workers=settings["parallel_workers"],
        gleaning_enabled=final_gleaning_enabled,
        dense_skeleton_batch_tokens=settings["dense_skeleton_batch_tokens"],
        dense_fill_nodes_cap=settings["dense_fill_nodes_cap"],
        dense_fill_context=settings["dense_fill_context"],
        dense_dedupe=settings["dense_dedupe"],
        provenance=settings["provenance"],
        export_format=export_format_val,
        export_docling=True,
        export_docling_json=final_export_docling_json,
        export_markdown=final_export_markdown,
        export_doclang=final_export_doclang,
        export_per_page_markdown=final_export_per_page,
        reverse_edges=reverse_edges,
        output_dir=str(output_dir),
    )

    logger.debug(f"PipelineConfig created: backend={cfg.backend}, inference={cfg.inference}")
    logger.debug(f"Output directory: {cfg.output_dir}")

    if show_llm_config and backend_val == "llm":
        from docling_graph.llm_clients.config import resolve_effective_model_config

        selection = cfg.models.llm.local if cfg.inference == "local" else cfg.models.llm.remote
        resolved_provider = provider or selection.provider
        resolved_model = model or selection.model
        effective = resolve_effective_model_config(
            resolved_provider,
            resolved_model,
            overrides=cfg.llm_overrides,
        )
        logger.info(
            "(no capability/registry) %s",
            effective.model_dump(),
            extra={"component": "ResolvedLLMConfig"},
        )
        raise typer.Exit(code=0)

    # Run pipeline with normalized/validated config; the pipeline logs its own
    # start and completion banners, so none are duplicated here.
    try:
        logger.debug("Calling run_pipeline() in CLI mode")
        run_pipeline(cfg, mode="cli")
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e.message}", exc_info=True)
        rich_print(f"\n[red]Configuration Error:[/red] {e.message}")
        if e.details:
            rich_print("[yellow]Details:[/yellow]")
            for key, value in e.details.items():
                rich_print(f"  • {key}: {value}")
        raise typer.Exit(code=1) from e
    except ExtractionError as e:
        logger.error(f"Extraction error: {e.message}", exc_info=True)
        rich_print(f"\n[red]Extraction Error:[/red] {e.message}")
        if e.details:
            rich_print("[yellow]Details:[/yellow]")
            for key, value in e.details.items():
                rich_print(f"  • {key}: {value}")
        raise typer.Exit(code=1) from e
    except PipelineError as e:
        logger.error(f"Pipeline error: {e.message}", exc_info=True)
        rich_print(f"\n[red]Pipeline Error:[/red] {e.message}")
        if e.details:
            rich_print("[yellow]Details:[/yellow]")
            for key, value in e.details.items():
                rich_print(f"  • {key}: {value}")
        raise typer.Exit(code=1) from e
    except DoclingGraphError as e:
        logger.error(f"Docling-graph error: {e.message}", exc_info=True)
        rich_print(f"\n[red]Error:[/red] {e.message}")
        if e.details:
            rich_print("[yellow]Details:[/yellow]")
            for key, value in e.details.items():
                rich_print(f"  • {key}: {value}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}: {e}")
        rich_print(f"\n[red]Unexpected error:[/red] {type(e).__name__}: {e}")
        raise typer.Exit(code=1) from e
