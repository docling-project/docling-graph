"""
Merge command - deterministically merges exported knowledge graphs.
"""

import logging
from pathlib import Path

import typer
from rich import print as rich_print
from typing_extensions import Annotated

from docling_graph.core.exporters.csv_exporter import CSVExporter
from docling_graph.core.exporters.cypher_exporter import CypherExporter
from docling_graph.core.exporters.json_exporter import JSONExporter
from docling_graph.core.merge import GraphMerger, MergePolicy, MergeReport
from docling_graph.core.utils.output_manager import OutputDirectoryManager
from docling_graph.core.visualizers.interactive_visualizer import InteractiveVisualizer
from docling_graph.core.visualizers.report_generator import ReportGenerator
from docling_graph.exceptions import DoclingGraphError

from ..validators import validate_export_format, validate_merge_conflicts, validate_merge_precedence

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path("outputs")


def merge_command(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            help="Run directories or graph.json exports to merge, in precedence "
            "order (the first graph is the base)."
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory. Default: outputs/merged_<timestamp>/.",
        ),
    ] = None,
    template: Annotated[
        str | None,
        typer.Option(
            "--template",
            "-t",
            help="Dotted path to the extraction template (module.Class); "
            "enables re-keying and alias proposal for old (v1) exports.",
        ),
    ] = None,
    precedence: Annotated[
        str,
        typer.Option(
            "--precedence",
            help="Duplicate-group fold order: 'input-order' or 'richest'.",
        ),
    ] = "input-order",
    conflicts: Annotated[
        str,
        typer.Option(
            "--conflicts",
            help="Scalar conflict policy: 'keep-first', 'keep-all' (suppressed "
            "values kept on the node under __conflicts__), or 'variants' "
            "(suppressed values reified as <Class>Variant sub-nodes linked by "
            "HAS_CONFLICT_VARIANT edges).",
        ),
    ] = "keep-first",
    combine_fields: Annotated[
        str,
        typer.Option(
            "--combine-fields",
            help="Comma-separated text fields merged with sentence-level dedup "
            "instead of first-wins.",
        ),
    ] = "description,summary",
    rekey: Annotated[
        bool | None,
        typer.Option(
            "--rekey/--no-rekey",
            help="Recompute node IDs from identity attributes before folding "
            "(default: on when an id-fields source is available).",
        ),
    ] = None,
    alias_decisions: Annotated[
        Path | None,
        typer.Option(
            "--alias-decisions",
            help="JSON file with confirmed alias candidates (edit the "
            "'alias_candidates' stubs from a previous merge_report.json).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Compute the full merge plan but write only merge_report.json.",
        ),
    ] = False,
    export_format: Annotated[
        str | None,
        typer.Option(
            "--export-format",
            help="Extra export beside graph.json: 'csv' or 'cypher'.",
        ),
    ] = None,
    strict_template_check: Annotated[
        bool,
        typer.Option(
            "--strict-template-check/--no-strict-template-check",
            help="Refuse to merge inputs extracted with different template schemas.",
        ),
    ] = True,
    report: Annotated[
        bool,
        typer.Option("--report/--no-report", help="Write merge_report.json and report.md."),
    ] = True,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open the interactive graph in the browser."),
    ] = False,
) -> None:
    """
    Merge multiple exported knowledge graphs into one, deterministically.

    Entities merge when their node IDs are byte-equal (same class, same
    canonicalized identity); everything else is folded with a fill-empty
    policy and fully audited in merge_report.json. No LLM is involved:
    ambiguous alias candidates are proposed into the report, and a
    human-edited decisions file confirms them on a re-run.

    Examples:
        # Merge two convert runs
        docling-graph merge outputs/report_A/ outputs/report_B/ -o merged/

        # Preview without writing the graph
        docling-graph merge a/ b/ --dry-run -o merged/

        # Apply human-confirmed alias merges from a previous report
        docling-graph merge a/ b/ --alias-decisions decisions.json -o merged/
    """
    precedence = validate_merge_precedence(precedence)
    conflicts = validate_merge_conflicts(conflicts)
    if export_format is not None:
        export_format = validate_export_format(export_format)

    policy = MergePolicy.model_validate(
        {
            "precedence": precedence,
            "conflicts": conflicts,
            "combine_fields": {f.strip() for f in combine_fields.split(",") if f.strip()},
            "rekey": rekey,
            "alias_decisions": alias_decisions,
            "export_format": export_format,
            "strict_template_check": strict_template_check,
            "dry_run": dry_run,
        }
    )

    logger.info("Starting Docling-Graph Merge", extra={"component": "Merge"})
    logger.info(
        "inputs=%s, template=%s, precedence=%s, conflicts=%s, rekey=%s, dry_run=%s",
        ", ".join(str(p) for p in inputs),
        template or "none",
        precedence,
        conflicts,
        "auto" if rekey is None else rekey,
        dry_run,
        extra={"component": "MergeConfiguration"},
    )
    if dry_run:
        logger.info(
            "Dry-run mode: only merge_report.json will be written",
            extra={"component": "Merge"},
        )

    # The merge runs fully in memory (validation included) before any output
    # directory exists — a failure never leaves a half-baked run dir behind.
    try:
        merger = GraphMerger(list(inputs), template=template, policy=policy)
        graph, merge_report = merger.merge()
    except DoclingGraphError as e:
        rich_print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e
    except ValueError as e:
        rich_print(f"[bold red]Error:[/bold red] Merged graph failed validation: {e}")
        raise typer.Exit(code=1) from e

    manager: OutputDirectoryManager | None = None
    if output is not None:
        document_dir = output
        document_dir.mkdir(parents=True, exist_ok=True)
    else:
        manager = OutputDirectoryManager(DEFAULT_OUTPUT_DIR, "merged")
        document_dir = manager.get_document_dir()
    graph_dir = document_dir / "docling_graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    try:
        report_path = graph_dir / "merge_report.json"
        if dry_run:
            report_path.write_text(merge_report.model_dump_json(indent=2), encoding="utf-8")
            rich_print(f"\nDry run: merge plan written to [cyan]{report_path}[/cyan]")
            _print_summary(merge_report)
            logger.info("Merge dry run completed", extra={"component": "Merge"})
            return

        json_path = graph_dir / "graph.json"
        JSONExporter().export(graph, json_path)
        if export_format == "csv":
            CSVExporter().export(graph, graph_dir)
        elif export_format == "cypher":
            CypherExporter().export(graph, graph_dir / "graph.cypher")

        merger.write_provenance_sidecars(graph_dir / "provenance")

        if report:
            report_path.write_text(merge_report.model_dump_json(indent=2), encoding="utf-8")
            ReportGenerator().visualize(
                graph,
                graph_dir / "report",
                source_model_count=len(merge_report.sources),
            )

        html_path = graph_dir / "graph.html"
        InteractiveVisualizer().save_cytoscape_graph(graph, html_path, open_browser=open_browser)
    except Exception as e:
        rich_print(f"[bold red]Error:[/bold red] {type(e).__name__}: {e}")
        if manager is not None:
            manager.cleanup_if_empty()
        raise typer.Exit(code=1) from e

    rich_print(f"\nMerged graph written to [cyan]{json_path}[/cyan]")
    _print_summary(merge_report)
    rich_print(f"\n[blue]Tip:[/blue] inspect it with: docling-graph inspect {json_path} -f json")
    logger.info("Merge completed successfully", extra={"component": "Merge"})


def _print_summary(merge_report: MergeReport) -> None:
    """Human-readable one-screen summary of the merge report."""
    rich_print(
        f"  Result: [green]{merge_report.node_count}[/green] nodes, "
        f"[green]{merge_report.edge_count}[/green] edges from "
        f"{len(merge_report.sources)} input(s)"
    )
    rich_print(
        f"  Folded: {merge_report.nodes_folded} node(s) | "
        f"Field conflicts: {len(merge_report.field_conflicts)} | "
        f"Edge label conflicts: {len(merge_report.edge_label_conflicts)}"
    )
    if merge_report.conflict_variants:
        rich_print(
            f"  Conflict variants: {merge_report.conflict_variants} sub-node(s) "
            "preserve suppressed values (HAS_CONFLICT_VARIANT edges)"
        )
    if merge_report.cross_document_splits:
        rich_print(
            f"  [yellow]Cross-document splits: {len(merge_report.cross_document_splits)} "
            "same-ID node(s) kept separate (unrelated documents, conflicting content) — "
            "see 'cross_document_splits' in merge_report.json[/yellow]"
        )
    rich_print(
        f"  Identity source: {merge_report.identity_source}"
        + (
            f" | re-keyed ({merge_report.rekeyed_changed} id(s) changed)"
            if merge_report.rekeyed
            else ""
        )
    )
    vetoed = [
        d
        for d in merge_report.ignored_alias_decisions
        if str(d.get("reason", "")).startswith("vetoed")
    ]
    # Re-running with --alias-decisions cannot change a guard veto, so the tip
    # is misleading when every confirmation was vetoed.
    all_confirmations_vetoed = bool(vetoed) and merge_report.alias_merged == 0
    if merge_report.alias_candidates:
        summary = (
            f"  Alias candidates: [yellow]{len(merge_report.alias_candidates)}[/yellow] "
            f"(confirmed merges applied: {merge_report.alias_merged})"
        )
        if not all_confirmations_vetoed:
            summary += (
                " — review 'alias_candidates' in merge_report.json and "
                "re-run with --alias-decisions"
            )
        rich_print(summary)
    if vetoed:
        rich_print(
            f"  [yellow]{len(vetoed)} confirmed alias decision(s) vetoed by reconciliation "
            "guards — see 'ignored_alias_decisions' in merge_report.json[/yellow]"
        )
    if merge_report.ignored_alias_decisions:
        rich_print(
            f"  [yellow]Ignored alias decisions: {len(merge_report.ignored_alias_decisions)} "
            "(see merge_report.json)[/yellow]"
        )
    for warning in merge_report.warnings:
        rich_print(f"  [yellow]Warning:[/yellow] {warning}")
