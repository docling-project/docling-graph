"""
LLM (Language Model) extraction backend.
Handles document extraction using LLM models (local or API).
"""

import gc
import json
from typing import List, Optional, Type

from pydantic import BaseModel, ValidationError
from rich import print as rich_print

from ....llm_clients.base import BaseLlmClient
from ....llm_clients.prompts import get_extraction_prompt, get_consolidation_prompt


class LlmBackend:
    """Backend for LLM-based extraction (local or API)."""

    def __init__(self, llm_client: BaseLlmClient) -> None:
        """
        Initialize LLM backend with a client.

        Args:
            llm_client (BaseLlmClient): LLM client instance (Mistral, Ollama, etc.)
        """
        self.client = llm_client
        rich_print(
            f"[blue][LlmBackend][/blue] Initialized with client: [cyan]{self.client.__class__.__name__}[/cyan]"
        )

    def extract_from_markdown(
        self,
        markdown: str,
        template: Type[BaseModel],
        context: str = "document",
        is_partial: bool = False,
    ) -> Optional[BaseModel]:
        """
        Extract structured data from markdown content using LLM.

        Args:
            markdown (str): Markdown content to extract from.
            template (Type[BaseModel]): Pydantic model template.
            context (str): Context description for the extraction (e.g., "page 1", "full document").
            is_partial (bool): If True, use the partial/chunk-based prompt.

        Returns:
            Optional[BaseModel]: Extracted and validated Pydantic model instance, or None if failed.
        """
        rich_print(
            f"[blue][LlmBackend][/blue] Extracting from {context} ([cyan]{len(markdown)}[/cyan] chars)"
        )

        # Validation for empty markdown
        if not markdown or len(markdown.strip()) == 0:
            rich_print(
                f"[red]Error:[/red] Extracted markdown is empty for {context}. Cannot proceed with LLM extraction."
            )
            return None

        try:
            # Get the Pydantic schema as JSON
            schema_json = json.dumps(template.model_json_schema(), indent=2)

            # Generate prompt using the correct signature
            prompt = get_extraction_prompt(
                markdown_content=markdown, schema_json=schema_json, is_partial=is_partial
            )

            # Call LLM with correct method name
            parsed_json = self.client.get_json_response(prompt=prompt, schema_json=schema_json)

            if not parsed_json:
                rich_print(
                    f"[yellow]Warning:[/yellow] No valid JSON returned from LLM for {context}"
                )
                return None

            # Use model_validate for proper Pydantic validation
            try:
                validated_model = template.model_validate(parsed_json)
                rich_print(f"[blue][LlmBackend][/blue] Successfully extracted data from {context}")
                return validated_model

            except ValidationError as e:
                # Detailed error reporting
                rich_print(
                    f"[blue][LlmBackend][/blue] [yellow]Validation Error for {context}:[/yellow]"
                )
                rich_print("  The data extracted by the LLM does not match your Pydantic template.")
                rich_print("[red]Details:[/red]")
                for error in e.errors():
                    loc = " -> ".join(map(str, error["loc"]))
                    rich_print(f"  - [bold magenta]{loc}[/bold magenta]: [red]{error['msg']}[/red]")
                rich_print(f"\n[yellow]Extracted Data (raw):[/yellow]\n{parsed_json}\n")
                return None

        except Exception as e:
            rich_print(f"[red]Error during LLM extraction for {context}:[/red] {type(e).__name__}: {e}")
            return None
        
    def consolidate_from_pydantic_models(
        self,
        raw_models: List[BaseModel],
        programmatic_model: BaseModel,
        template: Type[BaseModel],
    ) -> Optional[BaseModel]:
        """
        Uses an LLM to consolidate multiple extracted models into a final one.

        Args:
            raw_models: The list of raw models from each batch/page.
            programmatic_model: The programmatically merged model (as a draft).
            template: The Pydantic model template to validate against.

        Returns:
            A single, validated, LLM-consolidated Pydantic model, or None if failed.
        """
        rich_print(
            f"[blue][LlmBackend][/blue] Consolidating {len(raw_models)} models with LLM..."
        )
        try:
            schema_json = json.dumps(template.model_json_schema(), indent=2)

            prompt = get_consolidation_prompt(
                schema_json=schema_json,
                raw_models=raw_models,
                programmatic_model=programmatic_model,
            )

            # Call LLM
            parsed_json = self.client.get_json_response(prompt=prompt, schema_json=schema_json)

            if not parsed_json:
                rich_print("[yellow]Warning:[/yellow] LLM consolidation returned no valid JSON.")
                return None

            # Use model_validate for proper Pydantic validation
            try:
                validated_model = template.model_validate(parsed_json)
                rich_print("[blue][LlmBackend][/blue] Successfully consolidated and validated model.")
                return validated_model

            except ValidationError as e:
                rich_print(
                    f"[blue][LlmBackend][/blue] [yellow]Validation Error during consolidation:[/yellow]"
                )
                rich_print("  The data consolidated by the LLM does not match your Pydantic template.")
                rich_print("[red]Details:[/red]")

                for error in e.errors():
                    loc = " -> ".join(map(str, error["loc"]))
                    rich_print(f"  - [bold magenta]{loc}[/bold magenta]: [red]{error['msg']}[/red]")
                rich_print(f"\n[yellow]Consolidated Data (raw):[/yellow]\n{parsed_json}\n")
                return None

        except Exception as e:
            rich_print(f"[red]Error during LLM consolidation:[/red] {type(e).__name__}: {e}")
            return None

    def cleanup(self) -> None:
        """Clean up LLM client resources."""
        try:
            # Release the client reference
            if hasattr(self, "client"):
                # If the client has its own cleanup method, call it
                if hasattr(self.client, "cleanup"):
                    self.client.cleanup()
                del self.client

            # Force garbage collection
            gc.collect()

            rich_print("[blue][LlmBackend][/blue] [green]Cleaned up resources[/green]")

        except Exception as e:
            rich_print(f"[blue][LlmBackend][/blue] [yellow]Warning during cleanup:[/yellow] {e}")
