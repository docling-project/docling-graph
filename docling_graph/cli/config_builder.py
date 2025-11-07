"""
Configuration builder for interactive config creation.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, cast

import click
import typer
from rich import print as rich_print

from .constants import (
    API_PROVIDERS,
    BACKENDS,
    DOCLING_PIPELINES,
    EXPORT_FORMATS,
    INFERENCE_LOCATIONS,
    LOCAL_PROVIDER_DEFAULTS,
    LOCAL_PROVIDERS,
    PROCESSING_MODES,
    PROVIDER_DEFAULT_MODELS,
    VLM_DEFAULT_MODEL,
)


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

    def build_config(self) -> Dict[str, Any]:
        """Build configuration through interactive prompts."""
        rich_print("[bold blue]Welcome to Docling-Graph Setup![/bold blue]")
        rich_print("Let's configure your knowledge graph pipeline.")

        defaults = self._prompt_section("Default Settings", self._build_defaults)
        models = self._prompt_section(
            "Model Configuration",
            lambda: self._build_models(defaults["backend"], defaults["inference"]),
        )
        docling = self._prompt_section("Docling Pipeline", self._build_docling)
        output = self._prompt_section("Output", self._build_output)

        return {
            "defaults": defaults,
            "docling": docling,
            "models": models,
            "output": output,
        }

    def _prompt_section(
        self, title: str, builder_func: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        rich_print(f"\n── [bold]{title}[/bold] ──")
        return builder_func()

    def _prompt_option(self, config: PromptConfig) -> str:
        """Generic option prompt with reusable pattern."""
        rich_print(f"\n{config.step_num}. {config.label}")
        rich_print(f" {config.description}")

        for option in config.options:
            help_text = config.option_help.get(option, "")
            rich_print(f" • {option}: {help_text}")

        self.step_counter += 1
        return cast(
            str,
            typer.prompt(
                f"Select {config.label.lower()}",
                default=config.default,
                type=click.Choice(config.options, case_sensitive=False),
            ),
        ).lower()

    def _build_defaults(self) -> Dict[str, str]:
        """Build default settings section."""
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

        export_format = self._prompt_option(
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

        return {
            "processing_mode": processing_mode,
            "backend": backend,
            "inference": inference,
            "export_format": export_format,
        }

    def _build_docling(self) -> Dict[str, Any]:
        """Build Docling settings section."""
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
                description="Select provider",
                options=list(LOCAL_PROVIDERS),
                default="vllm",
                step_num=self.step_counter,
                option_help={p: f"Use {p} for local inference" for p in LOCAL_PROVIDERS},
            )
        )

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
                description="Select API provider",
                options=list(API_PROVIDERS),
                default="mistral",
                step_num=self.step_counter,
                option_help={p: f"Use {p} API" for p in API_PROVIDERS},
            )
        )

        default_model = PROVIDER_DEFAULT_MODELS.get(provider, PROVIDER_DEFAULT_MODELS["mistral"])
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
                    "default_model": vlm_model,
                    "provider": vlm_provider,
                }
            },
            "llm": {
                "local": {
                    "default_model": llm_local_model,
                    "provider": llm_local_provider,
                },
                "remote": {
                    "default_model": llm_remote_model,
                    "provider": llm_remote_provider,
                },
            },
        }

    def _build_output(self) -> Dict[str, str]:
        """Build output settings section."""
        directory = typer.prompt(
            "Output directory",
            default="outputs",
        )
        return {"directory": directory}


def build_config_interactive() -> Dict[str, Any]:
    """Build configuration through interactive prompts."""
    builder = ConfigurationBuilder()
    return builder.build_config()


def print_next_steps(config_dict: dict, return_text: bool = False) -> Optional[str]:
    """Print or return next steps after configuration creation."""
    steps = [
        "\n[bold green]Next steps:[/bold green]",
        "   1. Create a Pydantic model that matches your extraction needs. See docs/guides for detailed instructions.",
        "       You can also start from one of the templates in docs/examples/templates.",
        "   2. Run: [bold cyan]docling-graph convert <source> --template <path>[/bold cyan]",
    ]

    text = "\n".join(steps)
    if return_text:
        return text
    else:
        rich_print(text)
        return None
