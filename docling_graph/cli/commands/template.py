"""
Template command - generate, lint, and empirically evaluate extraction templates.

A Typer sub-app (design .claude/design/template-generation.md §2.1) with five
subcommands:

- ``from-docs``    — induce a template from example documents (3 LLM passes);
- ``from-ontology``— compile a template from OWL/LinkML/JSON Schema (zero LLM);
- ``from-spec``    — re-render a hand-edited SPEC YAML (the escape hatch);
- ``lint``         — lint an existing template module against the rulebook;
- ``evaluate``     — run real extractions and report empirical signals.

Every generator ends with the verification gate (V1-V6) and an atomic write:
the requested path is only ever touched after the rendered source passed every
gate, so Ctrl-C or a verification failure never leaves a partial template
behind. Provider/model resolution mirrors ``convert``: CLI flag >
``config.yaml`` ``models.llm.<inference>`` > a clear error. ``config.yaml``
itself is optional for every subcommand (``load_config_optional``).
"""

import logging
import os
import re
import sys
import tempfile
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import typer
import yaml
from pydantic import ValidationError
from rich import print as rich_print
from rich.markup import escape as rich_escape
from rich.table import Table
from typing_extensions import Annotated

from docling_graph.exceptions import ConfigurationError, DoclingGraphError
from docling_graph.llm_clients.schema_utils import build_compact_semantic_guide
from docling_graph.templategen import (
    LintReport,
    SpecGap,
    TemplateGenSettings,
    TemplateLintError,
    TemplateSpec,
    VerificationReport,
    lint_spec,
    load_templategen_settings,
    render_template,
    repair_draft,
    spec_from_dotted_path,
    spec_from_template,
    verify_template_source,
)
from docling_graph.templategen.linter import DOCSTRING_WINDOW
from docling_graph.templategen.naming import to_snake_case

from ..config_utils import get_config_value, load_config_optional
from ..constants import API_PROVIDERS, LOCAL_PROVIDERS
from ..validators import check_provider_installed, validate_inference, validate_template_format

if TYPE_CHECKING:
    from docling_graph.llm_clients.config import EffectiveModelConfig
    from docling_graph.templategen.induce.documents import InductionReport

logger = logging.getLogger(__name__)

template_app = typer.Typer(
    name="template",
    help="Generate, lint, and empirically evaluate extraction templates.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

DEFAULT_OUTPUT_DIR = Path("templates")

# Sources read directly as text by the induction sampler; anything else needs
# a DocumentProcessor (mirrors templategen.induce.documents._TEXT_SUFFIXES).
_TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})

# Input-budget heuristic (design §4.1): chars ~ tokens x 4, reserving the
# model's output budget plus a flat prompt overhead (system prompt + condensed
# rulebook + schema) against its context window.
_CHARS_PER_TOKEN = 4
_PROMPT_OVERHEAD_TOKENS = 2_000

LlmCallFn = Callable[..., Any]


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def _fail(message: str) -> typer.Exit:
    """Print a rich error line and return an exit-1 for the caller to raise."""
    rich_print(f"[bold red]Error:[/bold red] {message}")
    return typer.Exit(code=1)


def _fail_from(error: BaseException) -> typer.Exit:
    """Like :func:`_fail` for exception text, escaped against rich markup.

    Exception messages routinely carry bracketed text (install extras, Python
    lists) that rich would otherwise swallow as style tags.
    """
    message = getattr(error, "message", None)
    return _fail(rich_escape(str(message if message else error)))


def _load_config_and_settings() -> tuple[dict[str, Any] | None, TemplateGenSettings]:
    """Tolerantly read config.yaml and its optional ``templategen:`` block."""
    config = load_config_optional()
    try:
        settings = load_templategen_settings(config)
    except (ValueError, ValidationError) as e:
        raise _fail(f"Invalid 'templategen' block in config.yaml: {rich_escape(str(e))}") from e
    return config, settings


def _resolve_provider_model(
    config: dict[str, Any] | None,
    provider: str | None,
    model: str | None,
    inference: str | None = None,
) -> tuple[str, str, str]:
    """Resolve the LLM provider/model: CLI flag > config.yaml > clear error.

    ``inference`` picks which ``models.llm.<inference>`` block backs the
    fallback (flag > ``defaults.inference`` in config.yaml > ``remote``).
    """
    cfg = config or {}
    inference_val = str(
        inference or get_config_value(cfg, "defaults", "inference") or "remote"
    ).lower()
    inference_val = validate_inference(inference_val)

    provider_val = provider or get_config_value(cfg, "models", "llm", inference_val, "provider")
    model_val = model or get_config_value(cfg, "models", "llm", inference_val, "model")
    if not provider_val or not model_val:
        raise _fail(
            "No LLM provider/model configured. "
            "Run [cyan]docling-graph init[/cyan] or pass --provider/--model."
        )

    known_providers = set(API_PROVIDERS) | set(LOCAL_PROVIDERS)
    if provider_val not in known_providers:
        raise _fail(
            f"Invalid provider '{provider_val}'. "
            f"Must be one of: {', '.join(sorted(known_providers))}"
        )
    if not check_provider_installed(provider_val):
        raise _fail(
            f"The '{provider_val}' provider requires the LLM client dependency. "
            "Install it with: pip install docling-graph"
        )
    return str(provider_val), str(model_val), inference_val


def _build_llm_call(
    provider: str, model: str, config: dict[str, Any] | None
) -> tuple[LlmCallFn, "EffectiveModelConfig"]:
    """Bind an ``llm_call_fn`` to a LiteLLM client (stages.py two-liner).

    The callable follows the ``templategen.induce.documents`` contract and owns
    truncation handling: one retry with escalated ``max_tokens`` (doubled,
    capped at half the model's context window), then the retry result is
    returned either way.
    """
    from docling_graph.llm_clients import get_client
    from docling_graph.llm_clients.config import resolve_effective_model_config

    overrides = get_config_value(config or {}, "llm_overrides")
    effective = resolve_effective_model_config(
        provider, model, overrides=overrides if isinstance(overrides, dict) else None
    )
    client = get_client(provider)(effective)

    def llm_call_fn(*, prompt: dict[str, str], schema_json: str, context: str) -> Any:
        # OpenAI-family providers require response_format schema names to match
        # ^[a-zA-Z0-9_-]+$ (<=64 chars); the context tag carries ':' and the
        # source filename's '.', so sanitize before it reaches the provider.
        schema_name = re.sub(r"[^a-zA-Z0-9_-]", "_", context)[:40]
        out = client.get_json_response(
            prompt,
            schema_json,
            structured_output=True,
            response_top_level="object",
            response_schema_name=schema_name,
        )
        if not client.last_call_diagnostics.get("truncated"):
            return out
        generation = getattr(client, "_generation", None)
        current = int(getattr(client, "max_tokens", 0) or 0)
        context_limit = int(getattr(client, "context_limit", 0) or 0)
        escalated = current * 2
        if context_limit > 0:
            escalated = min(escalated, max(current, context_limit // 2))
        if generation is None or current <= 0 or escalated <= current:
            return out
        logger.warning(
            "Induction call '%s' was truncated; retrying once with max_tokens=%d",
            context,
            escalated,
        )
        original = getattr(generation, "max_tokens", None)
        try:
            generation.max_tokens = escalated
            return client.get_json_response(
                prompt,
                schema_json,
                structured_output=True,
                response_top_level="object",
                response_schema_name=schema_name,
            )
        finally:
            generation.max_tokens = original

    return llm_call_fn, effective


def _effective_budget_chars(settings: TemplateGenSettings, effective: Any) -> int:
    """Per-document sampler budget: min(settings cap, context-derived budget).

    Derivation (documented heuristic, design §4.1): the tokens left after
    reserving the model's output budget and a flat prompt overhead against its
    context window, times ~4 chars/token. A missing (unknown) context window
    falls back to the settings cap; a known-too-small one means there is no
    input room at all, which exits with a clear error instead of pretending
    the full settings cap fits (the design's fail-clearly stance).
    """
    context_limit = int(getattr(effective, "context_limit", 0) or 0)
    if context_limit <= 0:
        return settings.input_budget_chars
    output_tokens = int(getattr(effective, "max_output_tokens", 0) or 0)
    input_tokens = context_limit - output_tokens - _PROMPT_OVERHEAD_TOKENS
    if input_tokens <= 0:
        raise _fail(
            f"model context window too small for template induction "
            f"(context {context_limit} tokens, reserved output {output_tokens} + "
            f"{_PROMPT_OVERHEAD_TOKENS} prompt-overhead tokens leave no input room). "
            "Pick a larger-context model or lower max_tokens via llm_overrides."
        )
    return min(settings.input_budget_chars, input_tokens * _CHARS_PER_TOKEN)


def _build_doc_processor(config: dict[str, Any] | None, sources: Sequence[Path]) -> Any | None:
    """DocumentProcessor for non-text sources, honoring docling config/serve."""
    if all(source.suffix.lower() in _TEXT_SUFFIXES for source in sources):
        return None
    from docling_graph.core.extractors.document_processor import DocumentProcessor

    docling_cfg = get_config_value(config or {}, "docling") or {}
    if not isinstance(docling_cfg, dict):
        docling_cfg = {}
    serve_cfg = docling_cfg.get("serve") or {}
    docling_serve_config: dict[str, Any] | None = None
    if isinstance(serve_cfg, dict) and serve_cfg.get("url"):
        docling_serve_config = {
            "base_url": serve_cfg["url"],
            "api_key": serve_cfg.get("api_key"),
            "timeout": serve_cfg.get("timeout") or 300,
            "headers": serve_cfg.get("headers"),
        }
    return DocumentProcessor(
        docling_config=str(docling_cfg.get("pipeline", "ocr")),
        docling_serve_config=docling_serve_config,
    )


def _pipeline_config_overrides(
    config: dict[str, Any] | None, provider: str, model: str, inference: str
) -> dict[str, Any]:
    """``PipelineConfig`` fields for evaluate/--trial-run runs.

    Besides the LLM selection, config.yaml's ``docling`` block (the same keys
    :func:`_build_doc_processor` reads) is forwarded so evaluate/--trial-run
    convert exactly like ``convert`` and the induction sampling step — never a
    silent local re-convert with default settings.
    """
    overrides: dict[str, Any] = {
        "backend": "llm",
        "inference": inference,
        "provider_override": provider,
        "model_override": model,
    }
    models_cfg = get_config_value(config or {}, "models")
    if isinstance(models_cfg, dict) and models_cfg:
        overrides["models"] = models_cfg
    llm_overrides = get_config_value(config or {}, "llm_overrides")
    if isinstance(llm_overrides, dict) and llm_overrides:
        overrides["llm_overrides"] = llm_overrides
    docling_cfg = get_config_value(config or {}, "docling")
    if isinstance(docling_cfg, dict):
        pipeline = docling_cfg.get("pipeline")
        if pipeline in ("ocr", "vision"):
            overrides["docling_config"] = pipeline
        serve_cfg = docling_cfg.get("serve")
        if isinstance(serve_cfg, dict) and serve_cfg.get("url"):
            overrides["docling_serve_url"] = serve_cfg["url"]
            if serve_cfg.get("api_key"):
                overrides["docling_serve_api_key"] = serve_cfg["api_key"]
            if serve_cfg.get("timeout"):
                overrides["docling_serve_timeout"] = serve_cfg["timeout"]
            if isinstance(serve_cfg.get("headers"), dict):
                overrides["docling_serve_headers"] = serve_cfg["headers"]
    return overrides


# ---------------------------------------------------------------------------
# Output-path handling (atomic writes, overwrite confirmation)
# ---------------------------------------------------------------------------


def _default_output_path(root_name: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{to_snake_case(root_name)}.py"


def _default_spec_out(output: Path) -> Path:
    return output.with_suffix(".spec.yaml")


def _confirm_overwrite(paths: Sequence[Path], force: bool) -> None:
    """Prompt before overwriting existing files (like ``init``); --force skips."""
    if force:
        return
    for path in paths:
        if path.exists():
            rich_print(f"[yellow]'{path}' already exists.[/yellow]")
            if not typer.confirm("Overwrite it?"):
                rich_print("Generation cancelled.")
                raise typer.Exit(code=0)


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + ``os.replace`` so the path is never partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def _print_lint_entries(report: LintReport, *, title: str = "Repair log") -> None:
    rich_print(f"\n[bold]{title}:[/bold]")
    if not report.entries:
        typer.echo("  clean (no findings)")
        return
    for entry in report.entries:
        location = entry.model + (f".{entry.field}" if entry.field else "")
        status = "repaired" if entry.repaired else entry.severity
        typer.echo(f"  [{entry.rule_id}] {status:<8} {location}: {entry.message}")


def _print_gaps(gaps: Sequence[SpecGap]) -> None:
    rich_print(f"\n[bold]Open gaps / TODOs:[/bold] {len(gaps)}")
    for gap in gaps:
        location = gap.model + (f".{gap.field}" if gap.field else "")
        note = f" — {gap.note}" if gap.note else ""
        typer.echo(f"  {gap.kind}: {location}{note}")


def _print_spec_summary(spec: TemplateSpec) -> None:
    entities = [m.name for m in spec.models if m.kind == "entity"]
    components = [m.name for m in spec.models if m.kind == "component"]
    edge_labels: list[str] = []
    for model in spec.models:
        for field in model.fields:
            if field.role == "edge" and field.edge_label and field.edge_label not in edge_labels:
                edge_labels.append(field.edge_label)

    table = Table(title="Template summary", title_justify="left")
    table.add_column("Kind")
    table.add_column("Count", justify="right")
    table.add_column("Names")
    table.add_row("Root", "1", spec.root)
    table.add_row("Entities", str(len(entities)), ", ".join(entities))
    table.add_row("Components", str(len(components)), ", ".join(components))
    table.add_row("Edges", str(len(edge_labels)), ", ".join(edge_labels))
    table.add_row("Enums", str(len(spec.enums)), ", ".join(e.name for e in spec.enums))
    rich_print()
    rich_print(table)


def _print_semantic_guide(preview: str) -> None:
    rich_print("\n[bold]What the LLM will see (compact semantic guide):[/bold]")
    typer.echo(preview)


def _print_induction_report(report: "InductionReport") -> None:
    rich_print("\n[bold]Induction report:[/bold]")
    for stats in report.documents:
        sampled = " (sampled)" if stats.sampled else ""
        typer.echo(
            f"  {stats.name}{sampled}: {stats.classes_kept}/{stats.classes_proposed} "
            f"classes kept, {stats.examples_dropped_by_gate} example(s) dropped by the "
            f"verbatim gate, {stats.retries} retry(ies)"
        )
        if stats.overflow_classes:
            typer.echo(f"    overflow classes (max_models): {', '.join(stats.overflow_classes)}")
        if stats.identity_candidates_dropped:
            typer.echo(
                f"    identity candidates dropped: {', '.join(stats.identity_candidates_dropped)}"
            )
        if stats.digit_honesty_renames:
            typer.echo(f"    digit-honesty renames: {', '.join(stats.digit_honesty_renames)}")
        if stats.cardinality_bounds_dropped:
            typer.echo(
                f"    cardinality bounds dropped: {', '.join(stats.cardinality_bounds_dropped)}"
            )
        if stats.edges_dropped:
            typer.echo(f"    edges dropped: {', '.join(stats.edges_dropped)}")
    for skipped in report.skipped_sources:
        rich_print(f"  [yellow]Skipped {skipped}: near-empty text (check OCR settings)[/yellow]")
    if report.merge.decisions:
        rich_print("\n[bold]Merge decisions:[/bold]")
        for decision in report.merge.decisions:
            location = decision.model + (f".{decision.field}" if decision.field else "")
            typer.echo(f"  [{decision.kind}] {location}: {decision.message}")
    _print_lint_entries(report.lint)


def _handle_strict_failure(error: TemplateLintError) -> typer.Exit:
    """--strict tripped: print the violations (repairs are printed either way)."""
    rich_print(f"[bold red]Error:[/bold red] {rich_escape(str(error))}")
    _print_lint_entries(error.report, title="Violations (--strict)")
    return typer.Exit(code=1)


def _dedupe_gaps(gaps: Sequence[SpecGap]) -> list[SpecGap]:
    seen: set[tuple[str, str | None, str]] = set()
    unique: list[SpecGap] = []
    for gap in gaps:
        key = (gap.model, gap.field, gap.kind)
        if key not in seen:
            seen.add(key)
            unique.append(gap)
    return unique


# ---------------------------------------------------------------------------
# Render -> verify -> write (the shared tail of every generator)
# ---------------------------------------------------------------------------


def _finalize_template(spec: TemplateSpec, output: Path) -> VerificationReport:
    """Render the SPEC, run gates V1-V6, and atomically write on success.

    A gate failure is by definition a templategen bug (design §7.2): the
    source and SPEC are dumped with a ``.failed`` suffix and the requested
    path is never written.
    """
    source = render_template(spec)
    report = verify_template_source(source, root_class=spec.root, spec=spec)
    rich_print("\n[bold]Verification (V1-V6):[/bold]")
    typer.echo(report.summary())
    if not report.passed:
        failed_source = output.with_name(output.name + ".failed")
        failed_spec = output.with_name(output.name + ".failed.spec.yaml")
        _atomic_write(failed_source, source)
        _atomic_write(failed_spec, spec.to_yaml())
        rich_print(
            "\n[bold red]Verification failed[/bold red] — this is a docling-graph "
            "template-generation bug, not a problem with your input."
        )
        for gate in report.failures():
            typer.echo(f"  {gate.gate} {gate.name}: {gate.detail}")
        rich_print(
            f"The draft was dumped to [cyan]{failed_source}[/cyan] and "
            f"[cyan]{failed_spec}[/cyan]; '{output}' was not written. "
            "Please open an issue with these files attached."
        )
        raise typer.Exit(code=1)
    _atomic_write(output, source)
    return report


def _write_spec_yaml(spec: TemplateSpec, spec_out: Path) -> None:
    _atomic_write(spec_out, spec.to_yaml())
    rich_print(f"Spec written to [cyan]{spec_out}[/cyan] (edit + re-render via 'from-spec')")


def _persist_spec_before_prompt(
    spec: TemplateSpec, output: Path, spec_out: Path, early_paths: Sequence[Path], force: bool
) -> Path:
    """Write the SPEC YAML *before* the derived-path overwrite prompt.

    The SPEC is the durable artifact of a (paid) induction (design §1) and the
    derived-path prompt can only fire AFTER every LLM call — declining it must
    never discard what was paid for. The SPEC lands at ``spec_out`` when that
    path is safe to write (fresh, ``--force``, or confirmed before any token
    was spent); a colliding unconfirmed ``spec_out`` stays untouched and the
    SPEC goes to a non-conflicting ``<output>.new.spec.yaml`` rescue path.
    """
    if force or spec_out in early_paths or not spec_out.exists():
        _atomic_write(spec_out, spec.to_yaml())
        return spec_out
    rescue = output.with_name(f"{output.stem}.new.spec.yaml")
    counter = 2
    while rescue.exists():
        rescue = output.with_name(f"{output.stem}.new{counter}.spec.yaml")
        counter += 1
    _atomic_write(rescue, spec.to_yaml())
    return rescue


def _confirm_derived_paths(paths: Sequence[Path], force: bool, *, spec_path: Path) -> None:
    """Overwrite prompt for derived paths; a decline keeps the persisted SPEC.

    On decline, point at the SPEC written by :func:`_persist_spec_before_prompt`
    and the ``from-spec`` command that re-renders it without re-paying.
    """
    try:
        _confirm_overwrite(paths, force)
    except typer.Exit as e:
        if e.exit_code == 0:
            rich_print(
                f"The generated SPEC was saved to [cyan]{spec_path}[/cyan] — re-render it "
                "without re-running the LLM via:"
            )
            rich_print(
                f"  [cyan]docling-graph template from-spec {spec_path} -o <output.py>[/cyan]"
            )
        raise


def _trial_run(
    spec: TemplateSpec,
    output: Path,
    first_source: Path,
    config: dict[str, Any] | None,
    provider: str,
    model: str,
    inference: str,
) -> None:
    """V7 (--trial-run): one real extraction over the first source, advisory.

    The written template is re-executed into a fresh module (the same
    registered-module discipline as verification gate V2) and its live root
    class is handed to ``evaluate_template`` — trial failures warn and never
    change the exit code (the template is already V1-V6-verified).
    """
    rich_print(f"\n[bold]Trial run (advisory):[/bold] extracting from {first_source}")
    module_name = "docling_graph_template_trial"
    module = types.ModuleType(module_name)
    module.__file__ = str(output)
    sys.modules[module_name] = module
    try:
        from docling_graph.templategen import evaluate_template

        # Executing our own just-verified rendered source (same discipline as V2).
        exec(
            compile(output.read_text(encoding="utf-8"), str(output), "exec"),
            module.__dict__,
        )
        root_cls = module.__dict__[spec.root]
        report = evaluate_template(
            root_cls,
            [str(first_source)],
            config_overrides=_pipeline_config_overrides(config, provider, model, inference),
        )
        typer.echo(report.render_markdown())
    except Exception as e:  # advisory by design: never fail the command
        rich_print(
            f"[yellow]Trial run failed (advisory, template already verified): "
            f"{type(e).__name__}: {e}[/yellow]"
        )
    finally:
        sys.modules.pop(module_name, None)


def _rename_root_model(
    draft: dict[str, Any], new_name: str, gaps: Sequence[SpecGap] | None = None
) -> None:
    """Rename the draft's root model (--name), updating every reference.

    ``gaps`` minted against the old root name are re-addressed in place:
    gap-fill keys on ``(model, field, kind)``, so a stale model name would
    silently make every root gap unfillable. Residual limitation: renames
    applied later by ``repair_draft`` (keyword/builtin collisions) are not
    remapped here — the lint report exposes them only as prose messages, not
    as a structured old->new mapping.
    """
    old_name = draft.get("root")
    if not isinstance(old_name, str) or not old_name or old_name == new_name:
        return
    draft["root"] = new_name
    for model in draft.get("models", []):
        if not isinstance(model, dict):
            continue
        if model.get("name") == old_name:
            model["name"] = new_name
        for field in model.get("fields", []) or []:
            if isinstance(field, dict) and field.get("type") == old_name:
                field["type"] = new_name
        home = model.get("canonical_home")
        if isinstance(home, str) and home.startswith(f"{old_name}."):
            model["canonical_home"] = new_name + home[len(old_name) :]
    for gap in gaps or []:
        if gap.model == old_name:
            gap.model = new_name


def _cached_trial_source(report: "InductionReport", first_source: Path) -> Path:
    """The --trial-run source: the cached DoclingDocument JSON when one exists.

    ``prepare_document_text`` exports converted sources as
    ``<stem>.document.json`` (design §4.1/§7.2); the pipeline detects that as
    ``DOCLING_DOCUMENT`` input and skips conversion entirely, so the trial run
    never re-OCRs the source induction just converted. Text sources were never
    converted and pass through unchanged.
    """
    cached = next(
        (stats.cache_path for stats in report.documents if stats.name == first_source.name),
        None,
    )
    return Path(cached) if cached else first_source


# ---------------------------------------------------------------------------
# from-docs
# ---------------------------------------------------------------------------


@template_app.command(name="from-docs")
def from_docs_command(
    sources: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Example documents (PDF, markdown, Office, ...) to induce the template from.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Generated template path. Default: templates/<snake_root>.py.",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Root class name (overrides the induced root vote)."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="LLM provider (default: config.yaml models.llm)."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="LLM model (default: config.yaml models.llm)."),
    ] = None,
    spec_out: Annotated[
        Path | None,
        typer.Option(
            "--spec-out",
            help="Where to write the editable SPEC YAML. Default: <output>.spec.yaml.",
        ),
    ] = None,
    llm_gap_fill: Annotated[
        bool | None,
        typer.Option(
            "--llm-gap-fill/--no-llm-gap-fill",
            help="One extra LLM call filling docstring/example gaps (content only, "
            "never structure). Default: templategen.llm_gap_fill in config.yaml.",
        ),
    ] = None,
    trial_run: Annotated[
        bool,
        typer.Option(
            "--trial-run",
            help="After writing, run one real extraction over the first source "
            "(advisory: failures warn, exit code stays 0).",
        ),
    ] = False,
    strict: Annotated[
        bool | None,
        typer.Option(
            "--strict/--no-strict",
            help="Fail instead of auto-repairing lint violations "
            "(default: templategen.strict in config.yaml).",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing outputs without prompting (CI)."),
    ] = False,
) -> None:
    """Induce a Pydantic template from example documents (LLM structured output).

    Three focused passes per document (class inventory, fields, relationships)
    plus deterministic anti-hallucination gates produce a SPEC; a deterministic
    renderer then emits the template — no LLM ever writes code.
    """
    config, settings = _load_config_and_settings()
    strict_val = settings.strict if strict is None else strict
    gap_fill_val = settings.llm_gap_fill if llm_gap_fill is None else llm_gap_fill
    provider_val, model_val, inference_val = _resolve_provider_model(config, provider, model)
    try:
        llm_call_fn, effective = _build_llm_call(provider_val, model_val, config)
    except (ConfigurationError, ImportError, ValueError) as e:
        raise _fail_from(e) from e
    budget_chars = _effective_budget_chars(settings, effective)

    rich_print("--- [blue]Docling-Graph Template Generation (from-docs)[/blue] ---")
    rich_print(f"  Sources: [cyan]{', '.join(str(s) for s in sources)}[/cyan]")
    rich_print(
        f"  LLM: [cyan]{provider_val}/{model_val}[/cyan] | "
        f"Input budget: [cyan]{budget_chars}[/cyan] chars/document"
    )

    # Confirm explicit output paths before any LLM tokens are spent; paths
    # derived from the induced root name can only be confirmed afterwards.
    early_paths = [p for p in (output, spec_out) if p is not None]
    _confirm_overwrite(early_paths, force)

    doc_processor = _build_doc_processor(config, sources)
    # Converted sources are cached as DoclingDocument JSON (design §4.1/§7.2)
    # so --trial-run re-enters via the DOCLING_DOCUMENT input type instead of
    # re-converting (re-OCRing) the very source induction just converted.
    cache_tmp = (
        tempfile.TemporaryDirectory(prefix="docling-graph-templategen-")
        if doc_processor is not None
        else None
    )
    try:
        from docling_graph.templategen import induce_spec_from_documents

        try:
            spec, report = induce_spec_from_documents(
                [str(source) for source in sources],
                llm_call_fn,
                root_name=name,
                strict=strict_val,
                doc_processor=doc_processor,
                budget_chars=budget_chars,
                max_models=settings.max_models,
                max_enum_members=settings.max_enum_members,
                cache_dir=Path(cache_tmp.name) if cache_tmp is not None else None,
            )
        except TemplateLintError as e:
            raise _handle_strict_failure(e) from e
        except DoclingGraphError as e:
            raise _fail_from(e) from e

        _print_induction_report(report)

        gaps = list(report.gaps)
        if gap_fill_val and gaps:
            from docling_graph.templategen import fill_gaps

            before = len(gaps)
            spec, gaps = fill_gaps(spec, gaps, llm_call_fn)
            rich_print(f"\nGap-fill: closed {before - len(gaps)} of {before} gap(s)")

        output_val = output or _default_output_path(spec.root)
        spec_out_val = spec_out or _default_spec_out(output_val)
        # Snapshot collisions BEFORE the SPEC write below creates spec_out; the
        # SPEC is persisted ahead of the derived-path prompt because every LLM
        # token is already spent — declining must never discard the induction.
        colliding = [p for p in (output_val, spec_out_val) if p not in early_paths and p.exists()]
        spec_path = _persist_spec_before_prompt(spec, output_val, spec_out_val, early_paths, force)
        _confirm_derived_paths(colliding, force, spec_path=spec_path)

        _print_spec_summary(spec)
        verify_report = _finalize_template(spec, output_val)
        _write_spec_yaml(spec, spec_out_val)
        if spec_path != spec_out_val:
            spec_path.unlink(missing_ok=True)  # rescue copy superseded by the confirmed home
        rich_print(f"\nTemplate written to [cyan]{output_val}[/cyan]")
        _print_gaps(gaps)
        _print_semantic_guide(verify_report.semantic_guide_preview)

        if trial_run:
            _trial_run(
                spec,
                output_val,
                _cached_trial_source(report, sources[0]),
                config,
                provider_val,
                model_val,
                inference_val,
            )
    finally:
        if cache_tmp is not None:
            cache_tmp.cleanup()

    rich_print(
        f"\n[blue]Tip:[/blue] convert a document with: docling-graph convert <source> "
        f"--template {_dotted_hint(output_val, spec.root)}"
    )
    rich_print("--- [blue]Docling-Graph Template Generation Finished[/blue] ---")


def _dotted_hint(output: Path, root: str) -> str:
    """Best-effort dotted path for the written template (for the final tip)."""
    parts = [*output.parent.parts, output.stem]
    return ".".join(part for part in parts if part not in ("", ".")) + f".{root}"


# ---------------------------------------------------------------------------
# from-ontology
# ---------------------------------------------------------------------------


@template_app.command(name="from-ontology")
def from_ontology_command(
    source: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Ontology file (OWL/RDFS/SKOS, LinkML YAML, or JSON Schema).",
        ),
    ],
    root: Annotated[
        str | None,
        typer.Option(
            "--root",
            "-r",
            help="Root class (local name, CURIE, or IRI). Default: auto-elected when "
            "unambiguous; ambiguity exits 1 listing the candidates.",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Ontology format: owl | linkml | jsonschema | auto."),
    ] = "auto",
    depth: Annotated[
        int | None,
        typer.Option(
            "--depth",
            help="BFS depth bound from the root class (default: templategen.ontology_depth, 4).",
        ),
    ] = None,
    include: Annotated[
        list[str] | None,
        typer.Option("--include", help="Class glob to keep (repeatable)."),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", help="Class glob to prune (repeatable; wins over --include)."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Rename the root class in the generated template."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Generated template path. Default: templates/<snake_root>.py.",
        ),
    ] = None,
    spec_out: Annotated[
        Path | None,
        typer.Option(
            "--spec-out",
            help="Where to write the editable SPEC YAML. Default: <output>.spec.yaml.",
        ),
    ] = None,
    llm_gap_fill: Annotated[
        bool | None,
        typer.Option(
            "--llm-gap-fill/--no-llm-gap-fill",
            help="One optional LLM call filling docstring/example gaps (content only, "
            "never structure). Without it the path is fully deterministic — zero LLM.",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="LLM provider for --llm-gap-fill only."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="LLM model for --llm-gap-fill only."),
    ] = None,
    strict: Annotated[
        bool | None,
        typer.Option(
            "--strict/--no-strict",
            help="Fail instead of auto-repairing lint violations "
            "(default: templategen.strict in config.yaml).",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing outputs without prompting (CI)."),
    ] = False,
) -> None:
    """Compile a Pydantic template from an ontology — deterministic, zero LLM.

    OWL/RDFS/SKOS (rdflib), LinkML (linkml-runtime), and JSON Schema (stdlib)
    are supported; the first two need the 'templategen' extra. An LLM is only
    ever used under --llm-gap-fill, and can only fill docstrings/examples.
    """
    config, settings = _load_config_and_settings()
    fmt_val = validate_template_format(fmt)
    strict_val = settings.strict if strict is None else strict
    gap_fill_val = settings.llm_gap_fill if llm_gap_fill is None else llm_gap_fill
    depth_val = depth if depth is not None else settings.ontology_depth

    rich_print("--- [blue]Docling-Graph Template Generation (from-ontology)[/blue] ---")
    rich_print(f"  Source: [cyan]{source}[/cyan] | Format: [cyan]{fmt_val}[/cyan]")

    early_paths = [p for p in (output, spec_out) if p is not None]
    _confirm_overwrite(early_paths, force)

    from docling_graph.templategen import spec_draft_from_ontology

    try:
        draft, compiler_gaps = spec_draft_from_ontology(
            source,
            fmt=fmt_val,
            root=root,
            depth=depth_val,
            include=include or None,
            exclude=exclude or None,
            max_models=settings.max_models,
        )
    except ImportError as e:
        raise _fail_from(e) from e
    except ValueError as e:
        skos_enums = getattr(e, "enums", None)  # owl.SkosOnlyOntologyError
        if skos_enums is None:
            raise _fail_from(e) from e
        rich_print(f"[bold red]Error:[/bold red] {rich_escape(str(e))}")
        rich_print(
            "This vocabulary has no class structure to compile. Build the template "
            "from example documents instead: "
            "[cyan]docling-graph template from-docs <doc...>[/cyan]"
        )
        if spec_out is not None:
            _atomic_write(spec_out, yaml.safe_dump({"enums": skos_enums}, sort_keys=False))
            rich_print(
                f"Extracted {len(skos_enums)} enum vocabular(y/ies) written to "
                f"[cyan]{spec_out}[/cyan]"
            )
        else:
            rich_print("Pass [cyan]--spec-out[/cyan] to save the extracted enum vocabularies.")
        raise typer.Exit(code=1) from e

    if name:
        # Before repair/gap-fill dispatch: compiler gaps must be re-addressed to
        # the new root name or gap-fill can never match them (keyed on model).
        _rename_root_model(draft, name, compiler_gaps)

    try:
        spec, lint_report = repair_draft(draft, strict=strict_val)
    except TemplateLintError as e:
        raise _handle_strict_failure(e) from e
    except ValidationError as e:
        raise _fail(f"The compiled ontology draft did not validate: {rich_escape(str(e))}") from e

    _print_lint_entries(lint_report)
    gaps = _dedupe_gaps([*compiler_gaps, *lint_report.gaps])

    if gap_fill_val and gaps:
        provider_val, model_val, _ = _resolve_provider_model(config, provider, model)
        try:
            llm_call_fn, _effective = _build_llm_call(provider_val, model_val, config)
        except (ConfigurationError, ImportError, ValueError) as e:
            raise _fail_from(e) from e
        from docling_graph.templategen import fill_gaps

        before = len(gaps)
        spec, gaps = fill_gaps(spec, gaps, llm_call_fn)
        rich_print(
            f"\nGap-fill ({provider_val}/{model_val}): closed {before - len(gaps)} of {before} gap(s)"
        )

    output_val = output or _default_output_path(spec.root)
    spec_out_val = spec_out or _default_spec_out(output_val)
    # Snapshot collisions BEFORE the SPEC write below creates spec_out; the SPEC
    # is persisted ahead of the derived-path prompt (gap-fill spend, if any,
    # already happened) so declining re-renders via from-spec instead of re-paying.
    colliding = [p for p in (output_val, spec_out_val) if p not in early_paths and p.exists()]
    spec_path = _persist_spec_before_prompt(spec, output_val, spec_out_val, early_paths, force)
    _confirm_derived_paths(colliding, force, spec_path=spec_path)

    _print_spec_summary(spec)
    verify_report = _finalize_template(spec, output_val)
    _write_spec_yaml(spec, spec_out_val)
    if spec_path != spec_out_val:
        spec_path.unlink(missing_ok=True)  # rescue copy superseded by the confirmed home
    rich_print(f"\nTemplate written to [cyan]{output_val}[/cyan]")
    _print_gaps(gaps)
    _print_semantic_guide(verify_report.semantic_guide_preview)
    rich_print("--- [blue]Docling-Graph Template Generation Finished[/blue] ---")


# ---------------------------------------------------------------------------
# from-spec
# ---------------------------------------------------------------------------


@template_app.command(name="from-spec")
def from_spec_command(
    spec_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="SPEC YAML file (e.g. a hand-edited --spec-out).",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Generated template path. Default: templates/<snake_root>.py.",
        ),
    ] = None,
    strict: Annotated[
        bool | None,
        typer.Option(
            "--strict/--no-strict",
            help="Fail instead of auto-repairing lint violations "
            "(default: templategen.strict in config.yaml).",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing outputs without prompting (CI)."),
    ] = False,
) -> None:
    """Re-render a template from a (hand-edited) SPEC YAML — the escape hatch.

    Edit one line of the YAML written by --spec-out (rename an edge label,
    flip an entity to a component) and re-render, instead of hand-editing
    hundreds of lines of generated Python. Note: 'max_instances' in the SPEC
    YAML is the ALREADY-DOUBLED graph bound (2x the documented maximum) and is
    rendered as-is — it is never doubled again.
    """
    _config, settings = _load_config_and_settings()
    strict_val = settings.strict if strict is None else strict

    rich_print("--- [blue]Docling-Graph Template Generation (from-spec)[/blue] ---")
    rich_print(f"  Spec: [cyan]{spec_file}[/cyan]")

    try:
        spec = TemplateSpec.from_yaml(spec_file.read_text(encoding="utf-8"))
    except (ValidationError, ValueError, yaml.YAMLError) as e:
        raise _fail(f"Invalid template spec '{spec_file}': {rich_escape(str(e))}") from e

    try:
        spec, lint_report = lint_spec(spec, repair=True, strict=strict_val)
    except TemplateLintError as e:
        raise _handle_strict_failure(e) from e

    _print_lint_entries(lint_report)

    output_val = output or _default_output_path(spec.root)
    _confirm_overwrite([output_val], force)

    _print_spec_summary(spec)
    verify_report = _finalize_template(spec, output_val)
    rich_print(f"\nTemplate written to [cyan]{output_val}[/cyan]")
    _print_gaps(lint_report.gaps)
    _print_semantic_guide(verify_report.semantic_guide_preview)
    rich_print("--- [blue]Docling-Graph Template Generation Finished[/blue] ---")


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------


@template_app.command(name="lint")
def lint_command(
    template: Annotated[
        str,
        typer.Argument(
            help="Dotted path to the template's root class "
            "(e.g. 'templates.invoices.InvoiceDocument').",
        ),
    ],
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Exit 1 when any non-info finding is reported.",
        ),
    ] = False,
    import_check: Annotated[
        bool,
        typer.Option(
            "--import-check/--no-import-check",
            help="AST import-allowlist pre-check before the module is executed. "
            "Disable ONLY for trusted local files whose imports fall outside "
            "the allowlist (linting imports — i.e. runs — the module).",
        ),
    ] = True,
) -> None:
    """Lint an existing template module against the schema-definition rulebook.

    The live classes are walked back into a SPEC and every rulebook rule is
    reported: what the generator WOULD repair, plus the exact 240-char
    docstring windows and the compact semantic guide the LLM sees at
    extraction time. Exit code: 0 whenever a report is produced (report-only
    mode); 1 when the module cannot be loaded (bad path, import-allowlist
    rejection) or when --strict is set and any non-info finding exists.
    """
    try:
        if import_check:
            spec, report, template_cls = spec_from_dotted_path(template)
        else:
            from docling_graph.pipeline.stages import TemplateLoadingStage

            template_cls = TemplateLoadingStage._load_from_string(template)
            spec, report = spec_from_template(template_cls)
    except TemplateLintError as e:
        rich_print(f"[bold red]Error:[/bold red] {rich_escape(str(e))}")
        raise typer.Exit(code=1) from e
    except ConfigurationError as e:
        raise _fail_from(e) from e

    rich_print(f"--- [blue]Template Lint[/blue] --- [cyan]{template}[/cyan]")
    _print_lint_entries(report, title="Findings")
    if report.gaps:
        _print_gaps(report.gaps)

    rich_print(
        f"\n[bold]Docstring windows (first {DOCSTRING_WINDOW} chars the dense "
        "contract sees):[/bold]"
    )
    for model in spec.models:
        window = " ".join(model.docstring.split())[:DOCSTRING_WINDOW]
        typer.echo(f"  {model.name}: {window}")

    try:
        _print_semantic_guide(build_compact_semantic_guide(template_cls.model_json_schema()))
    except Exception as e:  # preview only — never fail the lint report over it
        rich_print(f"[yellow]Semantic-guide preview unavailable: {e}[/yellow]")

    non_info = [entry for entry in report.entries if entry.severity != "info"]
    rich_print(f"\n[bold]Result:[/bold] {len(report.entries)} finding(s), {len(non_info)} non-info")
    if strict and non_info:
        rich_print("[bold red]--strict:[/bold red] non-info findings present")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@template_app.command(name="evaluate")
def evaluate_command(
    template: Annotated[
        str,
        typer.Argument(
            help="Dotted path to the template's root class "
            "(e.g. 'templates.invoices.InvoiceDocument').",
        ),
    ],
    sources: Annotated[
        list[str],
        typer.Argument(help="Documents to evaluate the template against."),
    ],
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="LLM provider (default: config.yaml models.llm)."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="LLM model (default: config.yaml models.llm)."),
    ] = None,
    inference: Annotated[
        str | None,
        typer.Option(
            "--inference",
            "-i",
            help="Inference location: 'local' or 'remote' "
            "(default: config.yaml defaults.inference, then remote).",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Also write the Markdown report to this path."),
    ] = None,
) -> None:
    """Empirically evaluate a template against real documents (advisory).

    Runs one real extraction per source and reports the converter's own audit
    signals — empty-identity nodes, demotions, closed-catalog drops, field
    fill-rates, grounding precision — each translated to the rulebook clause
    it violates. Advisory report only: nothing is scored, gated, or repaired.
    """
    config, _settings = _load_config_and_settings()
    provider_val, model_val, inference_val = _resolve_provider_model(
        config, provider, model, inference
    )
    rich_print("--- [blue]Docling-Graph Template Evaluation[/blue] ---")
    rich_print(
        f"  Template: [cyan]{template}[/cyan] | "
        f"LLM: [cyan]{provider_val}/{model_val}[/cyan] ({inference_val})"
    )

    from docling_graph.templategen import evaluate_template

    try:
        report = evaluate_template(
            template,
            sources,
            config_overrides=_pipeline_config_overrides(
                config, provider_val, model_val, inference_val
            ),
        )
    except DoclingGraphError as e:
        raise _fail_from(e) from e
    except (TypeError, ValueError) as e:
        raise _fail_from(e) from e

    markdown = report.render_markdown()
    typer.echo(markdown)
    if output is not None:
        _atomic_write(output, markdown)
        rich_print(f"Report written to [cyan]{output}[/cyan]")
