"""
Documents -> SPEC induction (the ``from-docs`` path) and targeted gap-fill.

Three structured-output passes per document (class inventory -> fields ->
relationships) through an injected ``llm_call_fn``, deterministic evidence
gates, a deterministic cross-document merge, and ``linter.repair_draft`` at
the end. ``gapfill`` closes declared documentation gaps for both the
documents and ontology paths with one content-only LLM call.

This package never imports ``llm_clients`` and never constructs a client:
``llm_call_fn`` is injected (see ``documents.py`` for the exact contract).
"""

from .documents import (
    DEFAULT_BUDGET_CHARS,
    ELISION_MARKER,
    MAX_PASS2_BATCH,
    MAX_WINDOWS_PER_DOC,
    MIN_DOC_CHARS,
    SATURATION_MIN_UNITS,
    SATURATION_STREAK,
    DocumentContent,
    DocumentStats,
    InductionReport,
    LlmCallFn,
    PreparedDoc,
    SourceInput,
    induce_spec_from_documents,
    prepare_document_text,
    prepare_document_windows,
    source_display_name,
)
from .gapfill import fill_gaps
from .merge import (
    MAX_ENUM_MEMBERS,
    ClassCandidate,
    DocumentCandidates,
    FieldCandidate,
    MergeDecision,
    MergeReport,
    merge_documents,
)

__all__ = [
    "DEFAULT_BUDGET_CHARS",
    "ELISION_MARKER",
    "MAX_ENUM_MEMBERS",
    "MAX_PASS2_BATCH",
    "MAX_WINDOWS_PER_DOC",
    "MIN_DOC_CHARS",
    "SATURATION_MIN_UNITS",
    "SATURATION_STREAK",
    "ClassCandidate",
    "DocumentCandidates",
    "DocumentContent",
    "DocumentStats",
    "FieldCandidate",
    "InductionReport",
    "LlmCallFn",
    "MergeDecision",
    "MergeReport",
    "PreparedDoc",
    "SourceInput",
    "fill_gaps",
    "induce_spec_from_documents",
    "merge_documents",
    "prepare_document_text",
    "prepare_document_windows",
    "source_display_name",
]
