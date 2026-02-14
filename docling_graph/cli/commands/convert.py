"""
Convert command - converts documents to knowledge graphs.
"""

import logging
from pathlib import Path
from typing import Optional

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
            "--extraction-contract", help="Extraction contract: 'direct', 'staged', or 'delta'."
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
            help="Delta: max total chunk tokens per batch (chunks are batched until this limit). Does not set LLM output limit (default: 2048).",
        ),
    ] = None,
    staged_tuning_preset: Annotated[
        str | None,
        typer.Option(
            "--staged-tuning",
            help="Staged preset: 'standard' or 'advanced' (larger ID-pass shards, larger fill batches).",
        ),
    ] = None,
    staged_pass_retries: Annotated[
        int | None,
        typer.Option(
            "--staged-retries",
            help="Retries per staged pass when LLM returns invalid JSON (overrides preset).",
        ),
    ] = None,
    parallel_workers: Annotated[
        int | None,
        typer.Option(
            "--parallel-workers",
            help="Parallel workers for extraction (staged fill pass and delta batch calls; overrides preset for staged).",
        ),
    ] = None,
    delta_normalizer_validate_paths: Annotated[
        bool | None,
        typer.Option(
            "--delta-normalizer-validate-paths/--no-delta-normalizer-validate-paths",
            help="Validate and repair/drop unknown delta IR paths before merge.",
        ),
    ] = None,
    delta_normalizer_canonicalize_ids: Annotated[
        bool | None,
        typer.Option(
            "--delta-normalizer-canonicalize-ids/--no-delta-normalizer-canonicalize-ids",
            help="Canonicalize delta IR IDs before merge.",
        ),
    ] = None,
    delta_normalizer_strip_nested_properties: Annotated[
        bool | None,
        typer.Option(
            "--delta-normalizer-strip-nested-properties/--no-delta-normalizer-strip-nested-properties",
            help="Drop nested delta IR properties before merge.",
        ),
    ] = None,
    delta_resolvers_enabled: Annotated[
        bool | None,
        typer.Option(
            "--delta-resolvers-enabled/--no-delta-resolvers-enabled",
            help="Enable optional post-merge delta duplicate resolvers.",
        ),
    ] = None,
    delta_resolvers_mode: Annotated[
        str | None,
        typer.Option(
            "--delta-resolvers-mode", help="Resolver mode: off | fuzzy | semantic | chain."
        ),
    ] = None,
    delta_resolver_fuzzy_threshold: Annotated[
        float | None,
        typer.Option(
            "--delta-resolver-fuzzy-threshold", help="Fuzzy resolver similarity threshold."
        ),
    ] = None,
    delta_resolver_semantic_threshold: Annotated[
        float | None,
        typer.Option(
            "--delta-resolver-semantic-threshold", help="Semantic resolver similarity threshold."
        ),
    ] = None,
    staged_nodes_fill_cap: Annotated[
        int | None,
        typer.Option(
            "--staged-nodes-fill-cap",
            help="Max nodes per LLM call in fill pass (overrides preset).",
        ),
    ] = None,
    staged_id_shard_size: Annotated[
        int | None,
        typer.Option(
            "--staged-id-shard-size",
            help="Paths per ID pass call (0 = no sharding, overrides preset).",
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

    # Resolve configuration (CLI args override config file)
    processing_mode_val = processing_mode or defaults.get("processing_mode", "many-to-one")
    extraction_contract_val = extraction_contract or defaults.get("extraction_contract", "direct")
    backend_val = backend or defaults.get("backend", "llm")
    inference_val = inference or defaults.get("inference", "local")
    export_format_val = export_format or defaults.get("export_format", "csv")

    # Docling settings
    docling_pipeline_val = docling_pipeline or docling_cfg.get("pipeline", "ocr")

    # Chunking: resolve from CLI or config defaults
    final_chunk_max_tokens = (
        chunk_max_tokens if chunk_max_tokens is not None else defaults.get("chunk_max_tokens")
    )
    final_llm_batch_token_size = (
        llm_batch_token_size
        if llm_batch_token_size is not None
        else defaults.get("llm_batch_token_size", 2048)
    )
    final_staged_tuning_preset = (
        staged_tuning_preset
        if staged_tuning_preset is not None
        else defaults.get("staged_tuning_preset", "standard")
    )
    if final_staged_tuning_preset not in ("standard", "advanced"):
        final_staged_tuning_preset = "standard"
    final_staged_pass_retries = (
        staged_pass_retries
        if staged_pass_retries is not None
        else defaults.get("staged_pass_retries")
    )
    final_parallel_workers = (
        parallel_workers if parallel_workers is not None else defaults.get("parallel_workers")
    )
    final_delta_normalizer_validate_paths = (
        delta_normalizer_validate_paths
        if delta_normalizer_validate_paths is not None
        else bool(defaults.get("delta_normalizer_validate_paths", True))
    )
    final_delta_normalizer_canonicalize_ids = (
        delta_normalizer_canonicalize_ids
        if delta_normalizer_canonicalize_ids is not None
        else bool(defaults.get("delta_normalizer_canonicalize_ids", True))
    )
    final_delta_normalizer_strip_nested_properties = (
        delta_normalizer_strip_nested_properties
        if delta_normalizer_strip_nested_properties is not None
        else bool(defaults.get("delta_normalizer_strip_nested_properties", True))
    )
    final_delta_normalizer_attach_provenance = bool(
        defaults.get("delta_normalizer_attach_provenance", True)
    )
    final_delta_resolvers_enabled = (
        delta_resolvers_enabled
        if delta_resolvers_enabled is not None
        else bool(defaults.get("delta_resolvers_enabled", True))
    )
    final_delta_resolvers_mode = (
        delta_resolvers_mode
        if delta_resolvers_mode is not None
        else defaults.get("delta_resolvers_mode", "semantic")
    )
    if final_delta_resolvers_mode not in ("off", "fuzzy", "semantic", "chain"):
        final_delta_resolvers_mode = "off"
    final_delta_resolver_fuzzy_threshold = (
        delta_resolver_fuzzy_threshold
        if delta_resolver_fuzzy_threshold is not None
        else float(defaults.get("delta_resolver_fuzzy_threshold", 0.9))
    )
    final_delta_resolver_semantic_threshold = (
        delta_resolver_semantic_threshold
        if delta_resolver_semantic_threshold is not None
        else float(defaults.get("delta_resolver_semantic_threshold", 0.9))
    )
    final_delta_resolver_properties = defaults.get("delta_resolver_properties")
    final_delta_resolver_paths = defaults.get("delta_resolver_paths")
    final_quality_max_unknown_path_drops = int(defaults.get("quality_max_unknown_path_drops", -1))
    final_quality_max_id_mismatch = int(defaults.get("quality_max_id_mismatch", -1))
    final_quality_max_nested_property_drops = int(
        defaults.get("quality_max_nested_property_drops", -1)
    )
    final_staged_nodes_fill_cap = (
        staged_nodes_fill_cap
        if staged_nodes_fill_cap is not None
        else defaults.get("staged_nodes_fill_cap")
    )
    final_staged_id_shard_size = (
        staged_id_shard_size
        if staged_id_shard_size is not None
        else defaults.get("staged_id_shard_size")
    )
    final_structured_output = (
        schema_enforced_llm
        if schema_enforced_llm is not None
        else bool(defaults.get("structured_output", True))
    )
    final_structured_sparse_check = (
        structured_sparse_check
        if structured_sparse_check is not None
        else bool(defaults.get("structured_sparse_check", True))
    )

    # Docling export settings - use config file as fallback
    docling_export_settings = docling_cfg.get("export", {})
    final_export_docling_json = (
        export_docling_json
        if export_docling_json is not None
        else docling_export_settings.get("docling_json", True)
    )
    final_export_markdown = (
        export_markdown
        if export_markdown is not None
        else docling_export_settings.get("markdown", True)
    )
    final_export_per_page = (
        export_per_page
        if export_per_page is not None
        else docling_export_settings.get("per_page_markdown", False)
    )

    # Validate all inputs
    processing_mode_val = validate_processing_mode(processing_mode_val)
    extraction_contract_val = validate_extraction_contract(extraction_contract_val)
    backend_val = validate_backend_type(backend_val)
    inference_val = validate_inference(inference_val)
    docling_pipeline_val = validate_docling_config(docling_pipeline_val)
    export_format_val = validate_export_format(export_format_val)
    validate_vlm_constraints(backend_val, inference_val)

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
    if final_chunk_max_tokens is not None:
        rich_print(f"  • Chunk Max Tokens: [cyan]{final_chunk_max_tokens}[/cyan]")
    rich_print(f"  • LLM Batch Tokens: [yellow]{final_llm_batch_token_size}[/yellow]")
    if extraction_contract_val == "delta":
        rich_print("[yellow][DeltaTuning][/yellow]")
        rich_print(
            f"  • Parallel Workers: [cyan]{final_parallel_workers if final_parallel_workers is not None else 1}[/cyan]"
        )
        rich_print(
            f"  • IR Normalizer: [cyan]paths={final_delta_normalizer_validate_paths}, ids={final_delta_normalizer_canonicalize_ids}, strip_nested={final_delta_normalizer_strip_nested_properties}[/cyan]"
        )
        rich_print(
            f"  • Resolvers: [cyan]enabled={final_delta_resolvers_enabled}, mode={final_delta_resolvers_mode}[/cyan]"
        )
    if extraction_contract_val == "staged":
        from docling_graph.config import get_effective_staged_tuning

        eff_retries, eff_workers, eff_fill_cap, eff_id_shard_size = get_effective_staged_tuning(
            final_staged_tuning_preset,
            final_staged_pass_retries,
            final_parallel_workers,
            final_staged_nodes_fill_cap,
            final_staged_id_shard_size,
        )
        rich_print("[yellow][StagedTuning][/yellow]")
        rich_print(f"  • Preset: [cyan]{final_staged_tuning_preset}[/cyan]")
        rich_print(f"  • Retries: [cyan]{eff_retries}[/cyan]")
        rich_print(f"  • Workers: [cyan]{eff_workers}[/cyan]")
        rich_print(f"  • Nodes Fill Cap: [cyan]{eff_fill_cap}[/cyan]")
        rich_print(f"  • ID shard size: [cyan]{eff_id_shard_size}[/cyan]")

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
        llm_batch_token_size=final_llm_batch_token_size,
        staged_tuning_preset=final_staged_tuning_preset,
        staged_pass_retries=final_staged_pass_retries,
        parallel_workers=final_parallel_workers,
        delta_normalizer_validate_paths=final_delta_normalizer_validate_paths,
        delta_normalizer_canonicalize_ids=final_delta_normalizer_canonicalize_ids,
        delta_normalizer_strip_nested_properties=final_delta_normalizer_strip_nested_properties,
        delta_normalizer_attach_provenance=final_delta_normalizer_attach_provenance,
        delta_resolvers_enabled=final_delta_resolvers_enabled,
        delta_resolvers_mode=final_delta_resolvers_mode,
        delta_resolver_fuzzy_threshold=final_delta_resolver_fuzzy_threshold,
        delta_resolver_semantic_threshold=final_delta_resolver_semantic_threshold,
        delta_resolver_properties=final_delta_resolver_properties,
        delta_resolver_paths=final_delta_resolver_paths,
        quality_max_unknown_path_drops=final_quality_max_unknown_path_drops,
        quality_max_id_mismatch=final_quality_max_id_mismatch,
        quality_max_nested_property_drops=final_quality_max_nested_property_drops,
        staged_nodes_fill_cap=final_staged_nodes_fill_cap,
        staged_id_shard_size=final_staged_id_shard_size,
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
