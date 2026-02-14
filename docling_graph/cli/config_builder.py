"""
Configuration builder for interactive config creation.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, cast

import click
import typer
from rich import print as rich_print

from .constants import (
    API_PROVIDERS,
    BACKENDS,
    DELTA_RESOLVER_MODES,
    DOCLING_PIPELINES,
    EXPORT_FORMATS,
    EXTRACTION_CONTRACTS,
    INFERENCE_LOCATIONS,
    LOCAL_PROVIDER_DEFAULTS,
    LOCAL_PROVIDERS,
    PROCESSING_MODES,
    PROVIDER_DEFAULT_MODELS,
    VLM_DEFAULT_MODEL,
)


@lru_cache(maxsize=1)
def _get_provider_defaults() -> dict:
    """Cache provider defaults to avoid repeated lookups."""
    return {
        "local": LOCAL_PROVIDER_DEFAULTS,
        "remote": PROVIDER_DEFAULT_MODELS,
    }


@dataclass
class PromptConfig:
    """Configuration for a single prompt."""

    label: str
    description: str
    options: list[str]
    default: str
    option_help: Dict[str, str]
    step_num: int


class ConfigurationBuilder:
    """Orchestrates interactive configuration building."""

    def __init__(self) -> None:
        self.step_counter = 1
        self._use_custom_endpoint = False

    def build_config(self) -> Dict[str, Any]:
        """Build configuration through interactive prompts (optimized)."""
        rich_print("[bold blue]Welcome to Docling-Graph Setup![/bold blue]")
        rich_print("Let's configure your knowledge graph pipeline.\n")

        # Build all sections
        defaults = self._build_defaults()
        models = self._build_models(defaults["backend"], defaults["inference"])
        defaults["export_format"] = self._build_export_format()
        docling = self._build_docling()
        output = self._build_output()

        result = {
            "defaults": defaults,
            "docling": docling,
            "models": models,
            "output": output,
        }

        if self._use_custom_endpoint:
            result["_init_hints"] = {"use_custom_endpoint": True}

        return result

    def _prompt_option(self, config: PromptConfig) -> str:
        """Generic option prompt with reusable pattern."""
        rich_print(f"\n{config.step_num}. {config.label}")
        rich_print(f" {config.description}")

        for option in config.options:
            help_text = config.option_help.get(option, "")
            rich_print(f"  • {option}: {help_text}")

        self.step_counter += 1
        return cast(
            str,
            typer.prompt(
                f"Select {config.label.lower()}",
                default=config.default,
                type=click.Choice(config.options, case_sensitive=False),
            ),
        ).lower()

    def _build_defaults(self) -> Dict[str, Any]:
        """Build default settings section."""
        rich_print("\n── [bold]Default Settings[/bold] ──")
        processing_mode = self._prompt_option(
            PromptConfig(
                label="Processing Mode",
                description="How should documents be processed?",
                options=list(PROCESSING_MODES),
                default="many-to-one",
                step_num=self.step_counter,
                option_help={
                    "one-to-one": "Creates a separate Pydantic instance for each page",
                    "many-to-one": "Combines the entire document into a single Pydantic instance",
                },
            )
        )

        extraction_contract = self._prompt_option(
            PromptConfig(
                label="Extraction Contract",
                description="How should LLM extraction prompts/execution be orchestrated?",
                options=list(EXTRACTION_CONTRACTS),
                default="direct",
                step_num=self.step_counter,
                option_help={
                    "direct": "Single-pass best-effort extraction (fastest)",
                    "staged": "Multi-pass staged extraction for improved small-model quality",
                },
            )
        )
        delta_defaults = self._build_delta_defaults(extraction_contract)

        backend = self._prompt_option(
            PromptConfig(
                label="Backend Type",
                description="Which AI backend should be used?",
                options=list(BACKENDS),
                default="llm",
                step_num=self.step_counter,
                option_help={
                    "llm": "Language Model (text-based)",
                    "vlm": "Vision-Language Model (image-based)",
                },
            )
        )

        # VLM constraint: only local inference
        if backend == "vlm":
            rich_print("[yellow]Note: VLM backend only supports local inference for now.[/yellow]")
            inference = "local"
        else:
            inference = self._prompt_option(
                PromptConfig(
                    label="Inference Location",
                    description="How should models be executed?",
                    options=list(INFERENCE_LOCATIONS),
                    default="remote",
                    step_num=self.step_counter,
                    option_help={
                        "local": "Run on your machine",
                        "remote": "Use cloud APIs",
                    },
                )
            )

        return {
            "processing_mode": processing_mode,
            "extraction_contract": extraction_contract,
            "backend": backend,
            "inference": inference,
            **delta_defaults,
        }

    def _build_delta_defaults(self, extraction_contract: str) -> Dict[str, Any]:
        """Collect optional delta-specific defaults when delta contract is selected."""
        if extraction_contract != "delta":
            return {}

        rich_print("\n── [bold]Delta Extraction Tuning[/bold] ──")
        rich_print(
            f"\n{self.step_counter}. Delta Resolvers"
            "\n Configure optional post-merge resolver deduplication:"
        )
        self.step_counter += 1
        resolvers_enabled = typer.confirm(
            "Enable delta resolvers for post-merge dedup?",
            default=True,
        )
        resolvers_mode = "off"
        if resolvers_enabled:
            resolvers_mode = cast(
                str,
                typer.prompt(
                    "Select delta resolver mode",
                    default="fuzzy",
                    type=click.Choice(DELTA_RESOLVER_MODES, case_sensitive=False),
                ),
            ).lower()

        rich_print(
            f"\n{self.step_counter}. Delta Quality Gates"
            "\n Tune strictness for merge/projection quality checks:"
        )
        self.step_counter += 1
        customize_quality = typer.confirm(
            "Customize delta quality thresholds now?",
            default=True,
        )
        quality_defaults: Dict[str, Any] = {}
        if customize_quality:
            quality_defaults = self._prompt_delta_quality_defaults()

        return {
            "delta_resolvers_enabled": resolvers_enabled,
            "delta_resolvers_mode": resolvers_mode,
            **quality_defaults,
        }

    def _prompt_delta_quality_defaults(self) -> Dict[str, Any]:
        """Prompt for delta quality-related defaults."""
        parent_lookup_miss = typer.prompt(
            "Max parent lookup misses before failing quality gate (-1 disables)",
            default=4,
            type=int,
        )
        adaptive_parent_lookup = typer.confirm(
            "Enable adaptive parent lookup tolerance?",
            default=True,
        )
        require_relationships = typer.confirm(
            "Require at least one relationship in projected graph?",
            default=False,
        )
        require_structural_attachments = typer.confirm(
            "Require structural attachments (avoid root-only outputs)?",
            default=False,
        )
        return {
            "delta_quality_max_parent_lookup_miss": parent_lookup_miss,
            "delta_quality_adaptive_parent_lookup": adaptive_parent_lookup,
            "delta_quality_require_relationships": require_relationships,
            "delta_quality_require_structural_attachments": require_structural_attachments,
        }

    def _build_export_format(self) -> str:
        """Build export format prompt after model selection."""
        return self._prompt_option(
            PromptConfig(
                label="Export Format",
                description="Output format for results",
                options=list(EXPORT_FORMATS),
                default="csv",
                step_num=self.step_counter,
                option_help={
                    "csv": "CSV files (nodes.csv, edges.csv)",
                    "cypher": "Cypher script for Neo4j",
                },
            )
        )

    def _build_docling(self) -> Dict[str, Any]:
        """Build Docling settings section."""
        rich_print("\n── [bold]Docling Pipeline[/bold] ──")
        pipeline = self._prompt_option(
            PromptConfig(
                label="Document Processing Pipeline",
                description="Choose processing strategy",
                options=list(DOCLING_PIPELINES),
                default="ocr",
                step_num=self.step_counter,
                option_help={
                    "ocr": "OCR pipeline (standard documents - faster)",
                    "vision": "VLM pipeline (complex layouts - slower)",
                },
            )
        )

        rich_print(f"\n{self.step_counter}. Docling Export Options")
        rich_print(" Choose what to export from document processing:")
        self.step_counter += 1

        docling_json = typer.confirm("Export Docling document structure (JSON)?", default=True)
        markdown = typer.confirm("Export full document markdown?", default=True)
        per_page = typer.confirm("Export per-page markdown files?", default=False)

        return {
            "pipeline": pipeline,
            "export": {
                "docling_json": docling_json,
                "markdown": markdown,
                "per_page_markdown": per_page,
            },
        }

    def _build_models(self, backend: str, inference: str) -> Dict[str, Any]:
        """Build model configuration based on backend and inference type."""
        if backend == "vlm":
            return self._build_vlm_config()
        elif inference == "local":
            return self._build_local_llm_config()
        else:
            return self._build_remote_llm_config()

    def _build_vlm_config(self) -> Dict[str, Any]:
        """Build VLM configuration."""
        model = typer.prompt(
            "Select VLM model",
            default=VLM_DEFAULT_MODEL,
        )
        return self._build_model_structure(
            vlm_model=model,
            vlm_provider="docling",
            llm_local_model=LOCAL_PROVIDER_DEFAULTS["vllm"],
            llm_local_provider="vllm",
            llm_remote_model=PROVIDER_DEFAULT_MODELS["mistral"],
            llm_remote_provider="mistral",
        )

    def _build_local_llm_config(self) -> Dict[str, Any]:
        """Build local LLM configuration."""
        provider = self._prompt_option(
            PromptConfig(
                label="Local LLM Provider",
                description="Select provider (LiteLLM routing)",
                options=list(LOCAL_PROVIDERS),
                default="vllm",
                step_num=self.step_counter,
                option_help={
                    "vllm": "Use vLLM server (OpenAI-compatible)",
                    "ollama": "Use Ollama server (local inference)",
                    "custom": "Enter a custom LiteLLM provider ID",
                },
            )
        )

        if provider == "custom":
            provider_id = typer.prompt(
                "Enter LiteLLM provider id (e.g., ollama, vllm, openai)",
                default="ollama",
            )
            default_model = LOCAL_PROVIDER_DEFAULTS.get(provider_id, "llama3.1:8b")
            model = typer.prompt(
                f"Enter model for {provider_id}",
                default=default_model,
            )
            provider = provider_id
        else:
            default_model = LOCAL_PROVIDER_DEFAULTS.get(provider, LOCAL_PROVIDER_DEFAULTS["vllm"])
            model = typer.prompt(
                f"Select model for {provider}",
                default=default_model,
            )

        return self._build_model_structure(
            vlm_model=VLM_DEFAULT_MODEL,
            vlm_provider="docling",
            llm_local_model=model,
            llm_local_provider=provider,
            llm_remote_model=PROVIDER_DEFAULT_MODELS["mistral"],
            llm_remote_provider="mistral",
        )

    def _build_remote_llm_config(self) -> Dict[str, Any]:
        """Build remote LLM configuration."""
        provider = self._prompt_option(
            PromptConfig(
                label="API Provider",
                description="Select API provider (LiteLLM routing)",
                options=list(API_PROVIDERS),
                default="mistral",
                step_num=self.step_counter,
                option_help={
                    "mistral": "Use Mistral API",
                    "openai": "Use OpenAI API",
                    "gemini": "Use Gemini API",
                    "watsonx": "Use IBM watsonx",
                    "custom": "Enter a custom LiteLLM provider ID",
                },
            )
        )

        if provider == "custom":
            self._use_custom_endpoint = typer.confirm(
                "Use custom endpoint (URL and API key via environment variables)?",
                default=False,
            )
            provider_id = typer.prompt(
                "Enter LiteLLM provider id (e.g., openai, anthropic)",
                default="openai",
            )
            default_model = PROVIDER_DEFAULT_MODELS.get(provider_id, "gpt-4o")
            model = typer.prompt(
                f"Enter model for {provider_id}",
                default=default_model,
            )
            provider = provider_id
        else:
            default_model = PROVIDER_DEFAULT_MODELS.get(
                provider, PROVIDER_DEFAULT_MODELS["mistral"]
            )
            model = typer.prompt(
                f"Select model for {provider}",
                default=default_model,
            )

        return self._build_model_structure(
            vlm_model=VLM_DEFAULT_MODEL,
            vlm_provider="docling",
            llm_local_model=LOCAL_PROVIDER_DEFAULTS["vllm"],
            llm_local_provider="vllm",
            llm_remote_model=model,
            llm_remote_provider=provider,
        )

    @staticmethod
    def _build_model_structure(
        vlm_model: str,
        vlm_provider: str,
        llm_local_model: str,
        llm_local_provider: str,
        llm_remote_model: str,
        llm_remote_provider: str,
    ) -> Dict[str, Any]:
        """Build the standard model configuration structure."""
        return {
            "vlm": {
                "local": {
                    "model": vlm_model,
                    "provider": vlm_provider,
                }
            },
            "llm": {
                "local": {
                    "model": llm_local_model,
                    "provider": llm_local_provider,
                },
                "remote": {
                    "model": llm_remote_model,
                    "provider": llm_remote_provider,
                },
            },
        }

    def _build_output(self) -> Dict[str, str]:
        """Build output settings section."""
        rich_print("\n── [bold]Output[/bold] ──")
        directory = typer.prompt(
            "Output directory",
            default="outputs",
        )
        return {"directory": directory}


def build_config_interactive() -> Dict[str, Any]:
    """Build configuration through interactive prompts."""
    builder = ConfigurationBuilder()
    return builder.build_config()


def print_next_steps(config_dict: dict, return_text: bool = False) -> str | None:
    """Print or return next steps after configuration creation."""
    steps = [
        "\n[bold green]Next steps:[/bold green]",
        "   1. Create a Pydantic model that matches your extraction needs. See docs/guides for detailed instructions.",
        "       You can also start from one of the templates in docs/examples/templates.",
    ]

    # Custom endpoint env vars (on-prem)
    use_custom = bool((config_dict.get("_init_hints") or {}).get("use_custom_endpoint"))
    if use_custom:
        steps.append("   2. Set endpoint environment variables (e.g. in .env or shell):")
        steps.append("      [cyan]export CUSTOM_LLM_BASE_URL=...[/cyan]")
        steps.append("      [cyan]export CUSTOM_LLM_API_KEY=...[/cyan]")
        steps.append(
            "   3. Run: [bold cyan]docling-graph convert <source> --template <path>[/bold cyan]"
        )
    else:
        steps.append(
            "   2. Run: [bold cyan]docling-graph convert <source> --template <path>[/bold cyan]"
        )

    text = "\n".join(steps)
    if return_text:
        return text
    else:
        rich_print(text)
        return None
