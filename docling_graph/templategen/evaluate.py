"""
``template evaluate`` — the empirical template harness (design §7.3).

Static validity is not extraction quality: a lint-clean template whose
descriptions locate nothing still fails in the field. This module runs real
extractions over given documents and reports the empirical signals the
converter already emits for free — no new instrumentation, no LLM judging:

- ``graph.graph`` audit keys, each translated through
  :data:`AUDIT_KEY_RULEBOOK` (a static audit-key -> rulebook-clause table,
  reusable verbatim in any future repair prompt);
- node/edge counts and per-class field fill-rates (non-empty node attributes
  over the fields the template declares, via ``reverse.reverse_draft``);
- dense-contract stats (``last_dense_stats``) when the run produced them;
- grounding precision: extracted string node-attribute values checked as
  whitespace-normalized substrings of the ``ProvenanceLedger`` chunk texts —
  **excluding** file-stem root ids injected by
  ``core/utils/root_identity.repair_root_identity``, which are surfaced as a
  distinct ``root_id_synthetic`` template smell instead ("the root id is not
  findable in the document").

Advisory report only (v1): no auto-repair loop, no scores-as-gates. The
report is timestamp-free so identical runs diff cleanly.

Import discipline: this module never imports the pipeline (or networkx) at
module level — ``run_pipeline`` and ``PipelineConfig`` load lazily inside the
default runner, and ``run_pipeline_fn`` is injectable so tests (and callers
with a pre-built pipeline) never touch the extraction stack at all. The
injected callable receives one plain config dict per document::

    run_pipeline_fn({"source": <str>, "template": <live class>,
                     "dump_to_disk": False, **config_overrides}) -> context

and must return a ``PipelineContext``-shaped object (attributes
``knowledge_graph``, ``provenance``, ``extracted_models``, ``extractor`` —
all read defensively).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .reverse import reverse_draft

__all__ = [
    "AUDIT_KEY_RULEBOOK",
    "AuditFinding",
    "ClassFillRate",
    "DocumentEvaluation",
    "EvaluationReport",
    "EvaluationSummary",
    "GroundingReport",
    "RunPipelineFn",
    "evaluate_template",
]

RunPipelineFn = Callable[[dict[str, Any]], Any]
"""Injected pipeline runner; see the module docstring for the exact contract."""

AUDIT_KEY_RULEBOOK: dict[str, str] = {
    # graph_converter.py: graph.graph["empty_identity_nodes"]
    "empty_identity_nodes": (
        "field-definitions.md#identity-fields — the document does not NAME these "
        "instances (no identity value was extractable); demote the class to a component "
        "(entities-vs-components.md) or fix the identity field's LOOK-FOR description."
    ),
    # graph_converter.py: graph.graph["demoted_nodes"]
    "demoted_nodes": (
        "best-practices.md#graph-assembly-mechanics — more instances were discovered than "
        "graph_max_instances allows; bound only classes whose docstring states "
        "cardinality, at ~2x the documented maximum."
    ),
    # graph_converter.py: graph.graph["closed_catalog_drops"]
    "closed_catalog_drops": (
        "relationships.md#closed-catalog-reference-edges-closed_catalogtrue — "
        "closed-catalog reference edges named targets never anchored at the catalog's "
        "canonical home: hallucinated members of a fixed catalog."
    ),
    # graph_cleaner.py: graph.graph["dropped_relationships"]
    "dropped_relationships": (
        "best-practices.md#deterministic-grounding-readiness — edges pointed at phantom "
        "nodes carrying no meaningful data; the target class's field descriptions "
        "(field-definitions.md LOOK-FOR guidance) locate nothing in the document."
    ),
    # graph_converter.py: graph.graph["alias_reconciliation"]
    "alias_reconciliation": (
        "field-definitions.md#identity-fields — the same entity was extracted under "
        "several identity spellings/granularities and needed alias merging; tighten the "
        "identity field's verbatim examples so one canonical form wins."
    ),
}
"""``graph.graph`` audit key -> the rulebook clause it violates (design §7.3)."""

_UNGROUNDED_SAMPLE_CAP = 10
_MIN_GROUNDABLE_CHARS = 2


# ---------------------------------------------------------------------------
# Report models (all timestamp-free)
# ---------------------------------------------------------------------------


class AuditFinding(BaseModel):
    """One populated ``graph.graph`` audit key, translated to its rulebook clause."""

    model_config = ConfigDict(extra="forbid")

    key: str
    count: int
    rulebook: str = Field(description="The rulebook clause this audit key violates.")


class ClassFillRate(BaseModel):
    """Fill-rate of one template class's declared node attributes."""

    model_config = ConfigDict(extra="forbid")

    model: str
    node_count: int
    declared_fields: list[str] = Field(
        description="Fields that materialize as node attributes (edges to entities excluded)."
    )
    filled_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Per-field count of nodes carrying a non-empty value.",
    )
    fill_rate: float = Field(
        default=0.0,
        description="Mean fraction of declared attributes filled per node (0 when no nodes).",
    )


class GroundingReport(BaseModel):
    """Verbatim-grounding precision of extracted string attributes."""

    model_config = ConfigDict(extra="forbid")

    checked: int = 0
    grounded: int = 0
    precision: float | None = Field(
        default=None, description="grounded / checked; None when nothing was checkable."
    )
    root_id_synthetic: bool = Field(
        default=False,
        description="True when the root id equals the source file stem (injected by "
        "repair_root_identity): the document prints no usable root identity.",
    )
    ungrounded_samples: list[str] = Field(
        default_factory=list,
        description=f"Up to {_UNGROUNDED_SAMPLE_CAP} 'Class.field=value' samples that were "
        "not found in any ledger chunk.",
    )


class DocumentEvaluation(BaseModel):
    """Everything harvested from one source document's pipeline run."""

    model_config = ConfigDict(extra="forbid")

    source: str
    succeeded: bool
    error: str | None = None
    node_count: int = 0
    edge_count: int = 0
    extracted_models: int = 0
    audit_findings: list[AuditFinding] = Field(default_factory=list)
    fill_rates: list[ClassFillRate] = Field(default_factory=list)
    dense_stats: dict[str, Any] | None = None
    grounding: GroundingReport | None = None


class EvaluationSummary(BaseModel):
    """Aggregate view over every evaluated document."""

    model_config = ConfigDict(extra="forbid")

    documents: int = 0
    failed: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    audit_totals: dict[str, int] = Field(default_factory=dict)
    mean_fill_rate: float | None = None
    grounding_precision: float | None = None
    root_id_synthetic_documents: int = 0


class EvaluationReport(BaseModel):
    """The full ``template evaluate`` result: per-document + aggregate."""

    model_config = ConfigDict(extra="forbid")

    template: str
    per_document: list[DocumentEvaluation] = Field(default_factory=list)
    summary: EvaluationSummary = Field(default_factory=EvaluationSummary)

    @classmethod
    def from_documents(
        cls, template: str, per_document: list[DocumentEvaluation]
    ) -> EvaluationReport:
        """Build the report, computing the aggregate summary deterministically."""
        audit_totals: dict[str, int] = {}
        fill_rates: list[float] = []
        checked = 0
        grounded = 0
        synthetic = 0
        for document in per_document:
            for finding in document.audit_findings:
                audit_totals[finding.key] = audit_totals.get(finding.key, 0) + finding.count
            fill_rates.extend(f.fill_rate for f in document.fill_rates if f.node_count > 0)
            if document.grounding is not None:
                checked += document.grounding.checked
                grounded += document.grounding.grounded
                if document.grounding.root_id_synthetic:
                    synthetic += 1
        summary = EvaluationSummary(
            documents=len(per_document),
            failed=sum(1 for d in per_document if not d.succeeded),
            total_nodes=sum(d.node_count for d in per_document),
            total_edges=sum(d.edge_count for d in per_document),
            audit_totals=audit_totals,
            mean_fill_rate=(sum(fill_rates) / len(fill_rates)) if fill_rates else None,
            grounding_precision=(grounded / checked) if checked else None,
            root_id_synthetic_documents=synthetic,
        )
        return cls(template=template, per_document=per_document, summary=summary)

    def render_markdown(self) -> str:
        """Render the report as timestamp-free Markdown (the CLI's output body)."""
        lines: list[str] = [f"# Template evaluation — `{self.template}`", ""]
        lines.extend(self._summary_markdown())
        for document in self.per_document:
            lines.extend(self._document_markdown(document))
        return "\n".join(lines).rstrip() + "\n"

    def _summary_markdown(self) -> list[str]:
        summary = self.summary
        lines = [
            "## Summary",
            "",
            "| documents | failed | nodes | edges | mean fill rate | grounding precision |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| {summary.documents} | {summary.failed} | {summary.total_nodes} | {summary.total_edges} | {_fmt_rate(summary.mean_fill_rate)} | {_fmt_rate(summary.grounding_precision)} |",
            "",
        ]
        if summary.root_id_synthetic_documents:
            lines += [
                f"- Root id not findable in document (file-stem fallback) in "
                f"{summary.root_id_synthetic_documents} document(s) — the template's root "
                "identity field locates nothing the document prints.",
                "",
            ]
        if summary.audit_totals:
            lines += [
                "### Rulebook violations (graph audit keys)",
                "",
                "| audit key | total | rulebook clause |",
                "| --- | --- | --- |",
            ]
            for key in sorted(summary.audit_totals):
                clause = AUDIT_KEY_RULEBOOK.get(key, "")
                lines.append(f"| `{key}` | {summary.audit_totals[key]} | {clause} |")
            lines.append("")
        return lines

    def _document_markdown(self, document: DocumentEvaluation) -> list[str]:
        lines = [f"## {document.source}", ""]
        if not document.succeeded:
            lines += [f"- **Extraction failed:** {document.error}", ""]
            return lines
        lines += [f"- {document.node_count} node(s) / {document.edge_count} edge(s)", ""]
        if document.audit_findings:
            lines += ["| audit key | count | rulebook clause |", "| --- | --- | --- |"]
            for finding in document.audit_findings:
                lines.append(f"| `{finding.key}` | {finding.count} | {finding.rulebook} |")
            lines.append("")
        if document.fill_rates:
            lines += ["| class | nodes | fill rate |", "| --- | --- | --- |"]
            for rate in document.fill_rates:
                lines.append(f"| {rate.model} | {rate.node_count} | {_fmt_rate(rate.fill_rate)} |")
            lines.append("")
        grounding = document.grounding
        if grounding is not None:
            lines.append(
                f"- Grounding: {grounding.grounded}/{grounding.checked} checked values "
                f"found verbatim in the ledger ({_fmt_rate(grounding.precision)})"
            )
            if grounding.root_id_synthetic:
                lines.append(
                    "- Root id equals the source file stem (synthetic fallback) — "
                    "excluded from grounding, flagged as a template smell."
                )
            for sample in grounding.ungrounded_samples:
                lines.append(f"  - ungrounded: {sample}")
            lines.append("")
        if document.dense_stats:
            lines += ["Dense contract stats:", ""]
            for key in sorted(document.dense_stats):
                lines.append(f"- {key}: {document.dense_stats[key]}")
            lines.append("")
        return lines


def _fmt_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


# ---------------------------------------------------------------------------
# Harvesting helpers (all duck-typed: no networkx / pipeline imports)
# ---------------------------------------------------------------------------


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str | list | tuple | set | dict):
        return len(value) == 0
    return False


def _audit_count(key: str, value: Any) -> int:
    """Best-effort magnitude of one audit payload (shape varies per key)."""
    if key == "alias_reconciliation" and isinstance(value, Mapping):
        return int(value.get("merged") or 0) or int(value.get("candidates") or 0)
    if isinstance(value, Mapping):
        try:
            return int(sum(int(v) for v in value.values()))
        except (TypeError, ValueError):
            return len(value)
    if isinstance(value, list | tuple | set):
        return len(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return 1


def _audit_findings(graph: Any) -> list[AuditFinding]:
    meta = getattr(graph, "graph", None)
    if not isinstance(meta, Mapping):
        return []
    findings: list[AuditFinding] = []
    for key, rulebook in AUDIT_KEY_RULEBOOK.items():
        value = meta.get(key)
        if not value:
            continue
        findings.append(AuditFinding(key=key, count=_audit_count(key, value), rulebook=rulebook))
    return findings


def _draft_shapes(draft: dict[str, Any]) -> tuple[dict[str, list[str]], str, list[str]]:
    """Per-class node-attribute fields, plus the root name and its identity fields.

    A field materializes as a node attribute unless it targets a non-component
    model (those become graph edges/nodes instead; components embed inline).
    """
    models = [m for m in draft.get("models", []) if isinstance(m, dict)]
    kinds = {str(m.get("name")): str(m.get("kind")) for m in models}
    attr_fields: dict[str, list[str]] = {}
    for model in models:
        if model.get("kind") == "component":
            continue  # components never become nodes
        names: list[str] = []
        for field in model.get("fields", []):
            target_kind = kinds.get(str(field.get("type")))
            if target_kind is not None and target_kind != "component":
                continue
            names.append(str(field.get("name")))
        attr_fields[str(model.get("name"))] = names
    root = str(draft.get("root"))
    root_identity = [
        str(name)
        for model in models
        if model.get("name") == root
        for name in model.get("identity_fields", [])
    ]
    return attr_fields, root, root_identity


def _fill_rates(graph: Any, attr_fields: dict[str, list[str]]) -> list[ClassFillRate]:
    per_class_nodes: dict[str, list[Mapping[str, Any]]] = {name: [] for name in attr_fields}
    for _, data in graph.nodes(data=True):
        cls = str(data.get("__class__") or "")
        if cls in per_class_nodes:
            per_class_nodes[cls].append(data)

    rates: list[ClassFillRate] = []
    for model, fields in attr_fields.items():
        nodes = per_class_nodes[model]
        filled_counts = dict.fromkeys(fields, 0)
        fractions: list[float] = []
        for data in nodes:
            filled = 0
            for name in fields:
                if not _is_empty_value(data.get(name)):
                    filled_counts[name] += 1
                    filled += 1
            if fields:
                fractions.append(filled / len(fields))
        rates.append(
            ClassFillRate(
                model=model,
                node_count=len(nodes),
                declared_fields=fields,
                filled_counts=filled_counts,
                fill_rate=(sum(fractions) / len(fractions)) if fractions else 0.0,
            )
        )
    return rates


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _grounding(
    graph: Any,
    attr_fields: dict[str, list[str]],
    root: str,
    root_identity: list[str],
    ledger: Any,
    source_stem: str,
) -> GroundingReport | None:
    chunks = getattr(ledger, "chunks", None)
    if not isinstance(chunks, Mapping) or not chunks:
        return None
    corpus = [_normalize_ws(str(getattr(record, "text", "") or "")) for record in chunks.values()]
    corpus = [text for text in corpus if text]
    if not corpus:
        return None

    report = GroundingReport()
    for _, data in graph.nodes(data=True):
        cls = str(data.get("__class__") or "")
        fields = attr_fields.get(cls)
        if fields is None:
            continue
        for name in fields:
            value = data.get(name)
            if not isinstance(value, str):
                continue  # numbers/dates reformat freely; verbatim checks are string-only
            normalized = _normalize_ws(value)
            if len(normalized) < _MIN_GROUNDABLE_CHARS:
                continue
            if cls == root and name in root_identity and value == source_stem:
                # repair_root_identity's file-stem fallback: synthetic, never
                # document text — a template smell, not a grounding failure.
                report.root_id_synthetic = True
                continue
            report.checked += 1
            if any(normalized in text for text in corpus):
                report.grounded += 1
            elif len(report.ungrounded_samples) < _UNGROUNDED_SAMPLE_CAP:
                report.ungrounded_samples.append(f"{cls}.{name}={normalized[:80]!r}")
    report.precision = (report.grounded / report.checked) if report.checked else None
    return report


def _dense_stats(result: Any) -> dict[str, Any] | None:
    backend = getattr(getattr(result, "extractor", None), "backend", None)
    stats = getattr(backend, "last_dense_stats", None)
    if isinstance(stats, dict) and stats:
        return dict(stats)
    return None


def _evaluate_run(result: Any, draft: dict[str, Any], source: str) -> DocumentEvaluation:
    graph = getattr(result, "knowledge_graph", None)
    if graph is None:
        return DocumentEvaluation(
            source=source,
            succeeded=False,
            error="pipeline produced no knowledge graph",
        )
    attr_fields, root, root_identity = _draft_shapes(draft)
    extracted = getattr(result, "extracted_models", None)
    return DocumentEvaluation(
        source=source,
        succeeded=True,
        node_count=int(graph.number_of_nodes()),
        edge_count=int(graph.number_of_edges()),
        extracted_models=len(extracted) if isinstance(extracted, list | tuple) else 0,
        audit_findings=_audit_findings(graph),
        fill_rates=_fill_rates(graph, attr_fields),
        dense_stats=_dense_stats(result),
        grounding=_grounding(
            graph,
            attr_fields,
            root,
            root_identity,
            getattr(result, "provenance", None),
            Path(source).stem,
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _default_run_pipeline(config: dict[str, Any]) -> Any:
    """The real pipeline, imported lazily so evaluate stays import-light."""
    from docling_graph.config import PipelineConfig
    from docling_graph.pipeline import run_pipeline

    return run_pipeline(PipelineConfig(**config), mode="api")


def _resolve_template(template: type[BaseModel] | str) -> type[BaseModel]:
    if isinstance(template, str):
        # Lazy: pipeline.stages pulls the extraction stack.
        from docling_graph.pipeline.stages import TemplateLoadingStage

        return TemplateLoadingStage._load_from_string(template)
    if isinstance(template, type) and issubclass(template, BaseModel):
        return template
    raise TypeError(
        f"evaluate_template expects a BaseModel subclass or dotted path, got {template!r}"
    )


def evaluate_template(
    template: type[BaseModel] | str,
    sources: Sequence[str | Path],
    *,
    config_overrides: dict[str, Any] | None = None,
    run_pipeline_fn: RunPipelineFn | None = None,
) -> EvaluationReport:
    """Empirically evaluate a template against real documents (design §7.3).

    Per source, one pipeline run (``mode="api"``, no disk exports unless the
    overrides say otherwise) is harvested for the converter's own audit
    signals; a failed run becomes a ``succeeded=False`` entry instead of
    aborting the batch. Advisory report only — nothing here scores, gates, or
    repairs.

    Args:
        template: Live root class, or a dotted path loaded exactly like the
            pipeline loads ``--template`` (``TemplateLoadingStage``).
        sources: Document paths to evaluate against (at least one).
        config_overrides: Extra ``PipelineConfig`` fields merged into each
            per-document config dict (e.g. ``{"backend": "llm",
            "inference": "remote", "model_override": ...}``); they override
            the harness defaults (``dump_to_disk=False``).
        run_pipeline_fn: Injectable runner (module docstring has the exact
            contract). Defaults to the real ``run_pipeline`` with a minimal
            ``PipelineConfig``, imported lazily.

    Returns:
        The full :class:`EvaluationReport` (per-document + aggregate summary).

    Raises:
        ValueError: ``sources`` is empty.
        TypeError: ``template`` is neither a BaseModel subclass nor a string.
    """
    if not sources:
        raise ValueError("evaluate_template requires at least one source document")
    template_cls = _resolve_template(template)
    draft, _findings = reverse_draft(template_cls)
    runner = run_pipeline_fn or _default_run_pipeline

    per_document: list[DocumentEvaluation] = []
    for source in sources:
        source_str = str(source)
        config: dict[str, Any] = {
            "source": source_str,
            "template": template_cls,
            "dump_to_disk": False,
            **(config_overrides or {}),
        }
        try:
            result = runner(config)
        except Exception as exc:  # one bad document must not sink the batch
            per_document.append(
                DocumentEvaluation(
                    source=source_str,
                    succeeded=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        per_document.append(_evaluate_run(result, draft, source_str))

    template_name = (
        template if isinstance(template, str) else f"{template.__module__}.{template.__qualname__}"
    )
    return EvaluationReport.from_documents(template_name, per_document)
