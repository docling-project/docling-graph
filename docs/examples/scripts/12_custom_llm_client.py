"""
Example 12: Custom LLM Client (Bring Your Own)

Description:
    Runs the docling-graph pipeline with a custom LLM client that you configure
    (base URL, model, API key, headers). Docling-graph still builds prompts and
    schemas; your client sends them to your inference endpoint and returns JSON.

Use Cases:
    - Custom inference URL (on-prem, proxy, OpenAI-compatible endpoint)
    - On-prem WatsonX or other enterprise endpoints
    - vLLM / Ollama / LM Studio behind a custom base URL
    - Full control over the HTTP client and auth

Prerequisites:
    - Installation: uv sync
    - Set environment variables for your endpoint:
      - CUSTOM_LLM_BASE_URL (e.g. https://your-api.example.com/v1)
      - CUSTOM_LLM_MODEL (e.g. openai/your-model or hosted_vllm/model-name)
      - CUSTOM_LLM_API_KEY (optional)
    - Your endpoint must accept OpenAI-style /chat/completions and return JSON.

Key Concepts:
    - LLMClientProtocol: implement get_json_response(prompt, schema_json) -> dict | list
    - PipelineConfig(llm_client=...) bypasses provider/model config
    - ResponseHandler.parse_json_response() for consistent JSON parsing

Related Documentation:
    - https://ibm.github.io/docling-graph/reference/llm-clients/#custom-llm-clients
    - https://ibm.github.io/docling-graph/usage/api/pipeline-config/#custom-llm-client
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import litellm
except ImportError:
    litellm = None

try:
    from examples.templates.rheology_research import ScholarlyRheologyPaper

    from docling_graph import PipelineConfig, run_pipeline
    from docling_graph.exceptions import ClientError
    from docling_graph.llm_clients.response_handler import ResponseHandler
except ImportError as e:
    rich_print("[red]Error:[/red] Could not import required modules.")
    rich_print("Please run this script from the project root directory.")
    rich_print(f"Details: {e}")
    sys.exit(1)

SOURCE_FILE = "docs/examples/data/research_paper/rheology.pdf"
TEMPLATE_CLASS = ScholarlyRheologyPaper
console = Console()


# -----------------------------------------------------------------------------
# Custom LLM client (reference implementation — copy into your project as needed)
# -----------------------------------------------------------------------------


class LiteLLMEndpointClient:
    """Custom client: call a single inference endpoint via LiteLLM with your URL/auth."""

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: int = 120,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = headers or {}
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature

    def get_json_response(
        self, prompt: str | Mapping[str, str], schema_json: str
    ) -> Dict[str, Any] | List[Any]:
        messages = self._messages(prompt)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "api_base": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout_s,
            "response_format": {"type": "json_object"},
        }
        if self.api_key:
            request["api_key"] = self.api_key
        if self.headers:
            request["headers"] = dict(self.headers)

        try:
            response = litellm.completion(**request)
        except Exception as e:
            raise ClientError(
                f"LiteLLM call failed: {type(e).__name__}",
                details={"model": self.model, "api_base": self.base_url, "error": str(e)},
                cause=e,
            ) from e

        choices = response.get("choices", [])
        if not choices:
            raise ClientError("No choices in response", details={"model": self.model})
        content = choices[0].get("message", {}).get("content") or ""
        finish_reason = choices[0].get("finish_reason")
        truncated = finish_reason == "length"

        return ResponseHandler.parse_json_response(
            content,
            client_name=self.__class__.__name__,
            aggressive_clean=False,
            truncated=truncated,
            max_tokens=self.max_tokens,
        )

    def _messages(self, prompt: str | Mapping[str, str]) -> list[dict[str, str]]:
        if isinstance(prompt, Mapping):
            out = []
            if prompt.get("system"):
                out.append({"role": "system", "content": prompt["system"]})
            if prompt.get("user"):
                out.append({"role": "user", "content": prompt["user"]})
            return out if out else [{"role": "user", "content": ""}]
        return [{"role": "user", "content": prompt}]


def main() -> None:
    """Run pipeline with custom LLM client."""
    console.print(
        Panel.fit(
            "[bold blue]Example 12: Custom LLM Client[/bold blue]\n"
            "[dim]Use your own inference URL with docling-graph prompts and pipeline[/dim]",
            border_style="blue",
        )
    )

    base_url = os.getenv("CUSTOM_LLM_BASE_URL")
    model = os.getenv("CUSTOM_LLM_MODEL", "openai/gpt-4o-mini")
    api_key = os.getenv("CUSTOM_LLM_API_KEY")

    if not base_url:
        console.print("\n[red]CUSTOM_LLM_BASE_URL is not set.[/red]")
        console.print("\n[yellow]Set your inference endpoint:[/yellow]")
        console.print("  [cyan]export CUSTOM_LLM_BASE_URL='https://your-api.example.com/v1'[/cyan]")
        console.print("  [cyan]export CUSTOM_LLM_MODEL='openai/your-model'[/cyan]")
        console.print("  [cyan]export CUSTOM_LLM_API_KEY='optional-key'[/cyan]")
        sys.exit(1)

    if litellm is None:
        console.print("[red]LiteLLM is not installed. Run: uv sync[/red]")
        sys.exit(1)

    custom_client = LiteLLMEndpointClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_s=180,
        max_tokens=2048,
        temperature=0.1,
    )

    console.print("\n[yellow]Configuration:[/yellow]")
    console.print(f"  • Source: [cyan]{SOURCE_FILE}[/cyan]")
    console.print(f"  • Template: [cyan]{TEMPLATE_CLASS.__name__}[/cyan]")
    console.print(f"  • Base URL: [cyan]{base_url}[/cyan]")
    console.print(f"  • Model: [cyan]{model}[/cyan]")

    try:
        config = PipelineConfig(
            source=SOURCE_FILE,
            template=TEMPLATE_CLASS,
            backend="llm",
            inference="remote",
            llm_client=custom_client,
            processing_mode="many-to-one",
            use_chunking=True,
            dump_to_disk=True,
            output_dir="outputs/12_custom_llm_client",
        )

        console.print("\n[yellow]Running pipeline with custom client...[/yellow]")
        context = run_pipeline(config)

        console.print("\n[green]Success![/green]")
        graph = context.knowledge_graph
        console.print(
            f"  Extracted [cyan]{graph.number_of_nodes()} nodes[/cyan] "
            f"and [cyan]{graph.number_of_edges()} edges[/cyan]"
        )
        console.print(f"  Output: [cyan]{config.output_dir}/[/cyan]")

    except FileNotFoundError:
        console.print(f"\n[red]Source file not found:[/red] {SOURCE_FILE}")
        console.print("  Run from project root and ensure sample data exists.")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
