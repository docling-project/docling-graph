"""
Template generation for docling-graph (``docling-graph template ...``).

Two stages with a hard boundary: SPEC induction (documents via LLM structured
output, or ontologies via deterministic parsing) produces a machine-validated
:class:`TemplateSpec` IR, and a deterministic renderer turns that IR into a
runnable Pydantic template module. No LLM ever writes code.

Public surface (design §2.2):

- **IR**: ``TemplateSpec``, ``ModelSpec``, ``FieldSpec``, ``EnumSpec``,
  ``SpecGap``, ``ScalarType``;
- **linter**: ``lint_spec``, ``repair_draft``, ``LintReport``,
  ``TemplateLintError``;
- **renderer + verification**: ``render_template``, ``verify_template_source``,
  ``VerificationReport``, ``synthesize_sample``;
- **reverse flow** (``template lint`` on existing templates):
  ``reverse_draft``, ``spec_from_template``, ``spec_from_dotted_path``;
- **front-ends**: ``spec_draft_from_ontology`` (deterministic, zero LLM),
  ``induce_spec_from_documents`` (sources: file paths, http(s) URLs, or
  ``DocumentContent`` objects carrying the text directly), ``fill_gaps``;
- **one-shot convenience**: ``generate_template``, ``GenerationResult``
  (draft -> repair -> render -> verify -> optional atomic write);
- **empirical harness**: ``evaluate_template``, ``EvaluationReport``;
- **config**: ``TemplateGenSettings``, ``load_templategen_settings``.

The ontology/induction front-ends and the evaluate harness are exported
**lazily** (PEP 562): importing this package never touches the optional
``templategen`` extra (rdflib / linkml-runtime) and never pulls the extraction
pipeline — those load on first attribute access only.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .linter import LintReport, TemplateLintError, lint_spec, repair_draft
from .renderer import render_template
from .reverse import reverse_draft, spec_from_dotted_path, spec_from_template
from .settings import TemplateGenSettings, load_templategen_settings
from .spec import EnumSpec, FieldSpec, ModelSpec, ScalarType, SpecGap, TemplateSpec
from .verify import VerificationReport, synthesize_sample, verify_template_source

if TYPE_CHECKING:  # pragma: no cover - static-analysis view of the lazy exports
    from .evaluate import EvaluationReport, evaluate_template
    from .generate import GenerationResult, generate_template
    from .induce.documents import DocumentContent, induce_spec_from_documents
    from .induce.gapfill import fill_gaps
    from .llm_call import build_llm_call_fn
    from .ontology import spec_draft_from_ontology

__all__ = [
    "DocumentContent",
    "EnumSpec",
    "EvaluationReport",
    "FieldSpec",
    "GenerationResult",
    "LintReport",
    "ModelSpec",
    "ScalarType",
    "SpecGap",
    "TemplateGenSettings",
    "TemplateLintError",
    "TemplateSpec",
    "VerificationReport",
    "build_llm_call_fn",
    "evaluate_template",
    "fill_gaps",
    "generate_template",
    "induce_spec_from_documents",
    "lint_spec",
    "load_templategen_settings",
    "render_template",
    "repair_draft",
    "reverse_draft",
    "spec_draft_from_ontology",
    "spec_from_dotted_path",
    "spec_from_template",
    "synthesize_sample",
    "verify_template_source",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DocumentContent": (".induce.documents", "DocumentContent"),
    "EvaluationReport": (".evaluate", "EvaluationReport"),
    "GenerationResult": (".generate", "GenerationResult"),
    "build_llm_call_fn": (".llm_call", "build_llm_call_fn"),
    "evaluate_template": (".evaluate", "evaluate_template"),
    "fill_gaps": (".induce.gapfill", "fill_gaps"),
    "generate_template": (".generate", "generate_template"),
    "induce_spec_from_documents": (".induce.documents", "induce_spec_from_documents"),
    "spec_draft_from_ontology": (".ontology", "spec_draft_from_ontology"),
}


def __getattr__(name: str) -> Any:
    """Resolve the lazy exports on first access (PEP 562)."""
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name, __name__), attr)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
