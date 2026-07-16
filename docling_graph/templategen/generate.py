"""
One-shot programmatic template generation (the design §2.2 convenience).

:func:`generate_template` wraps the full pipeline every CLI generator runs —
draft -> ``repair_draft``/``lint_spec`` -> ``render_template`` ->
``verify_template_source`` (+ an atomic write when ``output`` is given) —
behind a single call, returning everything the stages produced as a
:class:`GenerationResult`.

Deliberately thin and CLI-free: no prompts, no rich output, no config.yaml
reads. There is **no overwrite prompt** — API callers pass a fresh path (the
write is atomic either way). A failed verification gate never writes the
requested path; the result carries the report and ``written_path=None``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .linter import LintReport, lint_spec, repair_draft
from .renderer import render_template
from .spec import SpecGap, TemplateSpec
from .verify import VerificationReport, verify_template_source

__all__ = ["GenerationResult", "generate_template"]

LlmCallFn = Callable[..., Any]
"""Injected LLM callable (the ``induce.documents`` contract)."""


class GenerationResult(BaseModel):
    """Everything one :func:`generate_template` run produced."""

    model_config = ConfigDict(extra="forbid")

    spec: TemplateSpec
    lint_report: LintReport
    gaps: list[SpecGap] = Field(default_factory=list)
    source_code: str
    verification: VerificationReport
    written_path: Path | None = None
    """The written template path; ``None`` when no ``output`` was given or
    verification failed (the requested path is then never touched)."""


def _single_path(source: str | Path | Sequence[str | Path], kind: str) -> Path:
    if isinstance(source, str | Path):
        return Path(source)
    raise TypeError(f"generate_template(kind={kind!r}) takes a single source path")


def _dedupe_gaps(gaps: Sequence[SpecGap]) -> list[SpecGap]:
    seen: set[tuple[str, str | None, str]] = set()
    unique: list[SpecGap] = []
    for gap in gaps:
        key = (gap.model, gap.field, gap.kind)
        if key not in seen:
            seen.add(key)
            unique.append(gap)
    return unique


def _atomic_write(path: Path, text: str) -> None:
    """Temp file + ``os.replace``: the target path is never partial."""
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


def generate_template(
    source: str | Path | Sequence[str | Path],
    *,
    kind: Literal["ontology", "spec", "docs"] = "ontology",
    output: str | Path | None = None,
    root: str | None = None,
    fmt: str = "auto",
    depth: int = 4,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    strict: bool = False,
    llm_call_fn: LlmCallFn | None = None,
) -> GenerationResult:
    """Generate a verified template module from one source, in one call.

    Args:
        source: The input — an ontology file (``kind="ontology"``), a SPEC
            YAML (``kind="spec"``), or one/many example documents
            (``kind="docs"``, the only kind accepting a sequence).
        kind: Which front-end produces the SPEC.
        output: Optional path for the rendered module; written atomically and
            **only** when verification passes. No overwrite prompt — pass a
            fresh path.
        root: Root class selector: the ontology root (local name/CURIE/IRI)
            for ``kind="ontology"``, the root class name (``--name``) for
            ``kind="docs"``; ignored for ``kind="spec"``.
        fmt: Ontology format (``owl | linkml | jsonschema | auto``);
            ``kind="ontology"`` only.
        depth: BFS depth bound from the root; ``kind="ontology"`` only.
        include: Class globs to keep; ``kind="ontology"`` only.
        exclude: Class globs to prune (wins over ``include``); ontology only.
        strict: Fail (``TemplateLintError``) instead of auto-repairing.
        llm_call_fn: The injected LLM callable (``induce.documents``
            contract); required by — and only used for — ``kind="docs"``.

    Returns:
        A :class:`GenerationResult`; check ``result.verification.passed``
        (and ``result.written_path`` when ``output`` was given).

    Raises:
        ValueError: Unknown ``kind``, ``kind="docs"`` without ``llm_call_fn``,
            or an invalid/unsniffable source.
        TypeError: A sequence source for a single-path ``kind``.
        TemplateLintError: ``strict=True`` and the draft required repairs.
    """
    gaps: list[SpecGap]
    if kind == "ontology":
        from .ontology import spec_draft_from_ontology

        draft, compiler_gaps = spec_draft_from_ontology(
            _single_path(source, kind),
            fmt=fmt,
            root=root,
            depth=depth,
            include=include,
            exclude=exclude,
        )
        spec, lint_report = repair_draft(draft, strict=strict)
        gaps = _dedupe_gaps([*compiler_gaps, *lint_report.gaps])
    elif kind == "spec":
        text = _single_path(source, kind).read_text(encoding="utf-8")
        spec, lint_report = lint_spec(TemplateSpec.from_yaml(text), repair=True, strict=strict)
        gaps = list(lint_report.gaps)
    elif kind == "docs":
        if llm_call_fn is None:
            raise ValueError(
                "generate_template(kind='docs') requires llm_call_fn "
                "(the induce.documents injected-callable contract)"
            )
        from .induce.documents import induce_spec_from_documents

        sources = [source] if isinstance(source, str | Path) else list(source)
        spec, report = induce_spec_from_documents(
            sources, llm_call_fn, root_name=root, strict=strict
        )
        lint_report = report.lint
        gaps = list(report.gaps)
    else:
        raise ValueError(f"Unknown kind {kind!r}: expected 'ontology', 'spec', or 'docs'")

    source_code = render_template(spec)
    verification = verify_template_source(source_code, root_class=spec.root, spec=spec)
    written_path: Path | None = None
    if output is not None and verification.passed:
        written_path = Path(output)
        _atomic_write(written_path, source_code)
    return GenerationResult(
        spec=spec,
        lint_report=lint_report,
        gaps=gaps,
        source_code=source_code,
        verification=verification,
        written_path=written_path,
    )
