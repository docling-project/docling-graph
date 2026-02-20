"""Dense extraction contract: two-phase Skeleton-then-Flesh. Fully autonomous."""

from .backend_ops import run_dense_orchestrator
from .catalog import (
    NodeCatalog,
    NodeSpec,
    build_node_catalog,
    build_projected_fill_schema,
    get_model_for_path,
)
from .models import DenseSkeletonGraph, DenseSkeletonNode
from .orchestrator import DenseOrchestrator, DenseOrchestratorConfig

__all__ = [
    "DenseOrchestrator",
    "DenseOrchestratorConfig",
    "DenseSkeletonGraph",
    "DenseSkeletonNode",
    "NodeCatalog",
    "NodeSpec",
    "build_node_catalog",
    "build_projected_fill_schema",
    "get_model_for_path",
    "run_dense_orchestrator",
]
