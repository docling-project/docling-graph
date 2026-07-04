"""Markdown report generator for graphs."""

from pathlib import Path
from typing import Optional, cast

import networkx as nx

from ..converters.models import GraphMetadata
from ..utils.stats_calculator import calculate_graph_stats
from ..utils.string_formatter import format_property_key, format_property_value


class ReportGenerator:
    """Generate markdown reports describing graph structure."""

    def visualize(
        self,
        graph: nx.DiGraph,
        output_path: Path,
        source_model_count: int = 1,
        include_samples: bool = True,
        extraction_contract: str | None = None,
        llm_diagnostics: dict | None = None,
        dense_stats: dict | None = None,
    ) -> None:
        """Generate markdown report for graph.

        Args:
            graph: NetworkX directed graph to document.
            output_path: Path for output markdown file (extension added if missing).
            source_model_count: Number of source Pydantic models.
            include_samples: Whether to include sample nodes/edges.
            extraction_contract: Extraction contract used (e.g. 'direct', 'dense').
            llm_diagnostics: Structured-output diagnostics from the LLM backend.
            dense_stats: Per-run dense observability counters (skeleton size,
                truncation/split counts, orphan rescue stats, retention).

        Raises:
            ValueError: If graph is empty.
        """
        if not self.validate_graph(graph):
            raise ValueError("Cannot generate report for empty graph")

        # Ensure .md extension
        if not str(output_path).endswith(".md"):
            output_path = Path(str(output_path) + ".md")

        # Calculate statistics
        metadata = calculate_graph_stats(graph, source_model_count)

        # Generate report sections
        report_parts = [
            self._create_header(),
            self._create_overview(metadata),
            self._create_node_type_distribution(metadata),
            self._create_edge_type_distribution(metadata),
        ]

        if extraction_contract is not None:
            report_parts.append(
                self._create_extraction_diagnostics(
                    extraction_contract=extraction_contract,
                    llm_diagnostics=llm_diagnostics,
                )
            )

        if dense_stats:
            report_parts.append(self._create_dense_stats(dense_stats))

        dropped_relationships = graph.graph.get("dropped_relationships")
        if dropped_relationships:
            report_parts.append(self._create_dropped_relationships(dropped_relationships))

        if include_samples:
            report_parts.append(self._create_sample_nodes(graph))
            report_parts.append(self._create_sample_edges(graph))

        # Write report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(report_parts))

    @staticmethod
    def _create_extraction_diagnostics(
        extraction_contract: str | None = None,
        llm_diagnostics: dict | None = None,
    ) -> str:
        """Create extraction diagnostics section."""
        lines = ["## Extraction Diagnostics", ""]
        if extraction_contract:
            lines.append(f"- **Extraction contract**: {extraction_contract}")
        if llm_diagnostics:
            attempted = llm_diagnostics.get("structured_attempted")
            failed = llm_diagnostics.get("structured_failed")
            fallback = llm_diagnostics.get("fallback_used")
            if attempted is not None:
                lines.append(f"- **Structured attempted**: {attempted}")
            if failed is not None:
                lines.append(f"- **Structured failed**: {failed}")
            if fallback is not None:
                lines.append(f"- **Legacy fallback used**: {fallback}")
            error_cls = llm_diagnostics.get("fallback_error_class")
            if error_cls:
                lines.append(f"- **Fallback trigger**: {error_cls}")
        if len(lines) == 2:
            return "\n".join([*lines, "*No extraction diagnostics available.*"])
        return "\n".join(lines)

    @staticmethod
    def _create_dense_stats(dense_stats: dict) -> str:
        """Create dense run-statistics section (Phase 1/2 health per run)."""
        labels = {
            "skeleton_nodes": "Skeleton nodes discovered",
            "parallel_workers": "Parallel workers",
            "phase1_seconds": "Phase 1 (skeleton) wall-clock (s)",
            "phase2_seconds": "Phase 2 (fill) wall-clock (s)",
            "chunk_coverage_pct": "Source chunk coverage (%)",
            "truncation_count": "Truncated LLM responses",
            "split_count": "Skeleton batch splits",
            "skeleton_batches_failed": "Skeleton batches failed (content lost)",
            "dropped_chunk_ids": "Dropped chunk ids (no skeleton)",
            "reconciliation_merged": "Alias instances reconciled",
            "merge_recovered": "Drifted parent links recovered",
            "merge_orphans_dropped": "Instances dropped (unresolvable parent)",
            "retention_pct": "Merge retention (post-fill, %)",
            "quality_gate_failure": "Quality gate failure",
        }
        lines = ["## Dense Extraction Statistics", ""]
        # A lossy run must not hide behind merge-only retention: warn prominently
        # when whole chunks produced no skeleton (content may be missing).
        failed = dense_stats.get("skeleton_batches_failed") or 0
        if isinstance(failed, int) and failed > 0:
            dropped_ids = dense_stats.get("dropped_chunk_ids") or []
            detail = f" (chunk ids: {dropped_ids})" if dropped_ids else ""
            lines.append(
                f"> ⚠️ **{failed} chunk batch(es) produced no skeleton — content may be "
                f"missing**{detail}. `retention_pct` reflects only post-fill merge retention, "
                "not source coverage; see **Source chunk coverage (%)** below."
            )
            lines.append("")
        for key, label in labels.items():
            if key in dense_stats:
                lines.append(f"- **{label}**: {dense_stats[key]}")
        if len(lines) == 2:
            return "\n".join([*lines, "*No dense statistics available.*"])
        return "\n".join(lines)

    @staticmethod
    def _create_dropped_relationships(dropped: list[dict]) -> str:
        """List relationships lost when a dangling (phantom) target was removed.

        A dropped edge is a lost relationship, not a mere count — surfacing the
        (source, label, target) tells the user *what* connection went missing.
        """
        lines = ["## Dropped Relationships", ""]
        lines.append(
            f"{len(dropped)} relationship(s) were removed because their target node "
            "was a phantom (dangling reference the model could not substantiate):"
        )
        lines.append("")
        for rel in dropped[:25]:
            label = rel.get("label") or "REFERENCES"
            lines.append(f"- `{rel.get('source')}` -[{label}]-> `{rel.get('target')}`")
        if len(dropped) > 25:
            lines.append(f"- … and {len(dropped) - 25} more")
        return "\n".join(lines)

    def validate_graph(self, graph: nx.DiGraph) -> bool:
        """Validate that graph is not empty."""
        node_count = cast(int, graph.number_of_nodes())
        return node_count > 0

    @staticmethod
    def _create_header() -> str:
        """Create report header."""
        return "# Knowledge Graph Report\n\nAutomatically generated by docling-graph."

    @staticmethod
    def _create_overview(metadata: GraphMetadata) -> str:
        """Create overview section."""
        return f"""## Overview

- **Total Nodes**: {metadata.node_count}
- **Total Edges**: {metadata.edge_count}
- **Source Models**: {metadata.source_models}
- **Generated**: {metadata.created_at.strftime("%Y-%m-%d %H:%M:%S")}"""

    @staticmethod
    def _create_node_type_distribution(metadata: GraphMetadata) -> str:
        """Create node type distribution section."""
        lines = ["## Node Type Distribution", ""]

        if not metadata.node_types:
            lines.append("*No node type information available.*")
        else:
            lines.append("| Node Type | Count | Percentage |")
            lines.append("|-----------|-------|------------|")

            total = metadata.node_count
            for node_type, count in sorted(
                metadata.node_types.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total * 100) if total > 0 else 0
                lines.append(f"| {node_type} | {count} | {percentage:.1f}% |")

        return "\n".join(lines)

    @staticmethod
    def _create_edge_type_distribution(metadata: GraphMetadata) -> str:
        """Create edge type distribution section."""
        lines = ["## Edge Type Distribution", ""]

        if not metadata.edge_types:
            lines.append("*No edge type information available.*")
        else:
            lines.append("| Edge Type | Count | Percentage |")
            lines.append("|-----------|-------|------------|")

            total = metadata.edge_count
            for edge_type, count in sorted(
                metadata.edge_types.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total * 100) if total > 0 else 0
                lines.append(f"| {edge_type} | {count} | {percentage:.1f}% |")

        return "\n".join(lines)

    @staticmethod
    def _create_sample_nodes(graph: nx.DiGraph, max_samples: int = 5) -> str:
        """Create sample nodes section."""
        lines = ["## Sample Nodes", ""]

        sample_nodes = list(graph.nodes(data=True))[:max_samples]

        for node_id, data in sample_nodes:
            lines.append(f"### Node: {node_id}")
            lines.append("")

            for key, value in data.items():
                formatted_key = format_property_key(key)
                formatted_value = format_property_value(value, max_length=100)
                lines.append(f"- **{formatted_key}**: {formatted_value}")

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _create_sample_edges(graph: nx.DiGraph, max_samples: int = 5) -> str:
        """Create sample edges section."""
        lines = ["## Sample Edges", ""]

        sample_edges = list(graph.edges(data=True))[:max_samples]

        for source, target, data in sample_edges:
            label = data.get("label", "related_to")
            lines.append(f"### {source} → {target}")
            lines.append(f"**Type**: {label}")
            lines.append("")

        return "\n".join(lines)
