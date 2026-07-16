"""Deterministic knowledge-graph fusion (`docling-graph merge`).

Merges exported graphs by node-ID key equality (node IDs are deterministic
content hashes), folds duplicate nodes with the pipeline's own fill-empty
policy, and proposes — never auto-applies — alias merges. No LLM anywhere.
"""

from .merger import GraphMerger, MergeReport, MergeSource, merge_graphs
from .policy import MergePolicy

__all__ = [
    "GraphMerger",
    "MergePolicy",
    "MergeReport",
    "MergeSource",
    "merge_graphs",
]
