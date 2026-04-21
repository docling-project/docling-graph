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
    chunk_max_tokens: int | None,
    dense_resolvers_enabled: bool | None,
    dense_resolvers_mode: str | None,
    dense_resolvers_fuzzy_threshold: float | None,
    dense_resolvers_semantic_threshold: float | None,
    dense_resolvers_allow_merge_different_ids: bool | None,
    dense_prune_barren_branches: bool | None,
    schema_enforced_llm: bool | None,
    structured_sparse_check: bool | None,
    gleaning_enabled: bool,
    gleaning_max_passes: int | None,
    export_docling_json: bool,
    export_markdown: bool,
    export_per_page: bool,
) -> dict[str, Any]:
    """Resolve CLI and config-file settings into effective values."""
    processing_mode_val = processing_mode or defaults.get("processing_mode", "many-to-one")
    extraction_contract_val = extraction_contract or defaults.get("extraction_contract", "direct")
    backend_val = backend or defaults.get("backend", "llm")
    inference_val = inference or defaults.get("inference", "local")
    export_format_val = export_format or defaults.get("export_format", "csv")
    docling_pipeline_val = docling_pipeline or docling_cfg.get("pipeline", "ocr")

    final_dense_resolvers_mode = (
        dense_resolvers_mode
        if dense_resolvers_mode is not None
        else defaults.get("dense_resolvers_mode", "off")
    )
    if final_dense_resolvers_mode not in ("off", "fuzzy", "semantic", "chain"):
        final_dense_resolvers_mode = "off"

    docling_export_settings = docling_cfg.get("export", {})

    return {
        "processing_mode": processing_mode_val,
        "extraction_contract": extraction_contract_val,
        "backend": backend_val,
        "inference": inference_val,
        "export_format": export_format_val,
        "docling_pipeline": docling_pipeline_val,
        "chunk_max_tokens": (
            chunk_max_tokens if chunk_max_tokens is not None else defaults.get("chunk_max_tokens")
        ),
        "dense_resolvers_enabled": (
            dense_resolvers_enabled
            if dense_resolvers_enabled is not None
            else bool(defaults.get("dense_resolvers_enabled", False))
        ),
        "dense_resolvers_mode": final_dense_resolvers_mode,
        "dense_resolvers_fuzzy_threshold": (
            dense_resolvers_fuzzy_threshold
            if dense_resolvers_fuzzy_threshold is not None
            else float(defaults.get("dense_resolvers_fuzzy_threshold", 0.8))
        ),
        "dense_resolvers_semantic_threshold": (
            dense_resolvers_semantic_threshold
            if dense_resolvers_semantic_threshold is not None
            else float(defaults.get("dense_resolvers_semantic_threshold", 0.8))
        ),
        "dense_resolvers_allow_merge_different_ids": (
            dense_resolvers_allow_merge_different_ids
            if dense_resolvers_allow_merge_different_ids is not None
            else bool(defaults.get("dense_resolvers_allow_merge_different_ids", False))
        ),
        "dense_prune_barren_branches": (
            dense_prune_barren_branches
            if dense_prune_barren_branches is not None
            else bool(defaults.get("dense_prune_barren_branches", False))
        ),
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
        "gleaning_enabled": gleaning_enabled or bool(defaults.get("gleaning_enabled", True)),
        "gleaning_max_passes": (
            gleaning_max_passes
            if gleaning_max_passes is not None
            else int(defaults.get("gleaning_max_passes", 1) or 1)
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
            help="Path to source document (any Docling-supported format), URL, or DoclingDocument JSON file. DoclingDocument skips conversion.",
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
            help="Extraction contract: 'direct' or 'dense'.",
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
    llm_batch_token_size: Annotated[
        int | None,
        typer.Option(
            "--llm-batch-token-size",
            help="Max total chunk tokens per batch. Does not set LLM output limit (default: 1024).",
        ),
    ] = None,
    parallel_workers: Annotated[
        int | None,
        typer.Option(
            "--parallel-workers",
            help="Parallel workers for extraction.",
        ),
    ] = None,
    dense_resolvers_enabled: Annotated[
        bool | None,
        typer.Option(
            "--dense-resolvers-enabled/--no-dense-resolvers-enabled",
            help="Enable optional post-merge dense duplicate resolvers (fuzzy/semantic merge).",
        ),
    ] = None,
    dense_resolvers_mode: Annotated[
        str | None,
        typer.Option(
            "--dense-resolvers-mode",
            help="Dense resolver mode: off | fuzzy | semantic | chain.",
        ),
    ] = None,
    dense_resolvers_fuzzy_threshold: Annotated[
        float | None,
        typer.Option(
            "--dense-resolvers-fuzzy-threshold",
            help="Dense fuzzy resolver similarity threshold.",
        ),
    ] = None,
    dense_resolvers_semantic_threshold: Annotated[
        float | None,
        typer.Option(
            "--dense-resolvers-semantic-threshold",
            help="Dense semantic resolver similarity threshold.",
        ),
    ] = None,
    dense_resolvers_allow_merge_different_ids: Annotated[
        bool | None,
        typer.Option(
            "--dense-resolvers-allow-merge-different-ids/--no-dense-resolvers-allow-merge-different-ids",
            help="Allow dense resolver to merge nodes with different non-empty ids.",
        ),
    ] = None,
    dense_prune_barren_branches: Annotated[
        bool | None,
        typer.Option(
            "--dense-prune-barren-branches/--no-dense-prune-barren-branches",
            help="Remove dense skeleton nodes that have no filled children and no scalar data (barren branches).",
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
        bool,
        typer.Option(
            "--gleaning-enabled/--no-gleaning-enabled",
            help="Run optional second-pass extraction to improve recall (direct contract). Default: enabled.",
        ),
    ] = True,
    gleaning_max_passes: Annotated[
        int | None,
        typer.Option(
            "--gleaning-max-passes",
            help="Max gleaning passes when gleaning is enabled (default: 1).",
        ),
    ] = None,
) -> None:
    """Convert a document to a knowledge graph."""
    logger.debug("Starting convert command")
    logger.debug(f"Source: {source}, Template: {template}")
    logger.debug(
        f"CLI args - Backend: {backend}, Inference: {inference}, Processing: {processing_mode}"
    )

    rich_print("[green]--- Starting Docling-Graph Conversion ---[/green]")

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
            chunk_max_tokens=chunk_max_tokens,
            dense_resolvers_enabled=dense_resolvers_enabled,
            dense_resolvers_mode=dense_resolvers_mode,
            dense_resolvers_fuzzy_threshold=dense_resolvers_fuzzy_threshold,
            dense_resolvers_semantic_threshold=dense_resolvers_semantic_threshold,
            dense_resolvers_allow_merge_different_ids=dense_resolvers_allow_merge_different_ids,
            dense_prune_barren_branches=dense_prune_barren_branches,
            schema_enforced_llm=schema_enforced_llm,
            structured_sparse_check=structured_sparse_check,
            gleaning_enabled=gleaning_enabled,
            gleaning_max_passes=gleaning_max_passes,
            export_docling_json=export_docling_json,
            export_markdown=export_markdown,
            export_per_page=export_per_page,
        )
    )

    processing_mode_val = settings["processing_mode"]
    extraction_contract_val = settings["extraction_contract"]
    backend_val = settings["backend"]
    inference_val = settings["inference"]
    export_format_val = settings["export_format"]
    docling_pipeline_val = settings["docling_pipeline"]
    final_chunk_max_tokens = settings["chunk_max_tokens"]
    final_dense_resolvers_enabled = settings["dense_resolvers_enabled"]
    final_dense_resolvers_mode = settings["dense_resolvers_mode"]
    final_dense_resolvers_fuzzy_threshold = settings["dense_resolvers_fuzzy_threshold"]
    final_dense_resolvers_semantic_threshold = settings["dense_resolvers_semantic_threshold"]
    final_dense_resolvers_allow_merge_different_ids = settings[
        "dense_resolvers_allow_merge_different_ids"
    ]
    final_dense_prune_barren_branches = settings["dense_prune_barren_branches"]
    final_structured_output = settings["structured_output"]
    final_structured_sparse_check = settings["structured_sparse_check"]
    final_gleaning_enabled = settings["gleaning_enabled"]
    final_gleaning_max_passes = settings["gleaning_max_passes"]
    final_export_docling_json = settings["export_docling_json"]
    final_export_markdown = settings["export_markdown"]
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

    # Display configuration
    rich_print("[yellow][PipelineConfiguration][/yellow]")
    rich_print(f"  • Source: [cyan]{source}[/cyan]")
    rich_print(f"  • Template: [cyan]{template}[/cyan]")
    rich_print(f"  • Input Type: [cyan]{input_type_display}[/cyan]")
    rich_print(f"  • Docling: [cyan]{docling_pipeline_val}[/cyan]")
    rich_print(f"  • Processing: [cyan]{processing_mode_val}[/cyan]")
    rich_print(f"  • Inference: [cyan]{inference_val}[/cyan]")
    rich_print(f"  • Export: [cyan]{export_format_val}[/cyan]")
    rich_print(f"  • Reverse edges: [cyan]{reverse_edges}[/cyan]")

    # Display Docling export settings
    rich_print("[yellow][DoclingExport][/yellow]")
    rich_print(f"  • Document JSON: [cyan]{final_export_docling_json}[/cyan]")
    rich_print(f"  • Markdown: [cyan]{final_export_markdown}[/cyan]")
    rich_print(f"  • Per-page MD: [cyan]{final_export_per_page}[/cyan]")

    # Display Extraction settings
    rich_print("[yellow][ExtractionSettings][/yellow]")
    rich_print(f"  • Backend: [cyan]{backend_val}[/cyan]")
    rich_print(f"  • Contract: [cyan]{extraction_contract_val}[/cyan]")
    rich_print(f"  • Structured Output: [cyan]{final_structured_output}[/cyan]")
    rich_print(f"  • Structured Sparse Check: [cyan]{final_structured_sparse_check}[/cyan]")
    rich_print(f"  • Debug: [cyan]{debug}[/cyan]")
    if extraction_contract_val in ("direct", "dense"):
        rich_print(
            f"  • Gleaning: [cyan]enabled={final_gleaning_enabled}, max_passes={final_gleaning_max_passes}[/cyan]"
        )
    if final_chunk_max_tokens is not None:
        rich_print(f"  • Chunk Max Tokens: [cyan]{final_chunk_max_tokens}[/cyan]")
    if extraction_contract_val == "dense":
        rich_print("[yellow][DenseTuning][/yellow]")
        rich_print("  • Skeleton batch tokens / fill nodes cap: from config or defaults")
        rich_print(
            f"  • Resolvers: [cyan]enabled={final_dense_resolvers_enabled}, mode={final_dense_resolvers_mode}[/cyan]"
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
        debug=debug,
        chunk_max_tokens=final_chunk_max_tokens,
        gleaning_enabled=final_gleaning_enabled,
        gleaning_max_passes=final_gleaning_max_passes,
        dense_resolvers_enabled=final_dense_resolvers_enabled,
        dense_resolvers_mode=final_dense_resolvers_mode,
        dense_resolvers_fuzzy_threshold=final_dense_resolvers_fuzzy_threshold,
        dense_resolvers_semantic_threshold=final_dense_resolvers_semantic_threshold,
        dense_resolvers_allow_merge_different_ids=final_dense_resolvers_allow_merge_different_ids,
        dense_prune_barren_branches=final_dense_prune_barren_branches,
        export_format=export_format_val,
        export_docling=True,
        export_docling_json=final_export_docling_json,
        export_markdown=final_export_markdown,
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
        rich_print("[yellow][ResolvedLLMConfig (no capability/registry)][/yellow]")
        rich_print(effective.model_dump())
        raise typer.Exit(code=0)

    # Run pipeline with normalized/validated config
    logger.info("Starting pipeline execution")
    try:
        logger.debug("Calling run_pipeline() in CLI mode")
        run_pipeline(cfg, mode="cli")
        logger.info("--- Pipeline execution Completed Successfully ---")
        rich_print("[green]--- Docling-Graph Conversion Successfull ---[/green]")
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
