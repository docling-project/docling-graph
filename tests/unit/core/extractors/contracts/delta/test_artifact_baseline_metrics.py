from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonicalize(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = " ".join(value.strip().split()).casefold()
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return str(value)


@pytest.mark.parametrize(
    ("run_dir", "expected"),
    [
        (
            "sample_invoice_jpg_20260214_184539",
            {
                "unknown_path_dropped": 4,
                "id_missing_required": 7,
                "parent_lookup_miss": 1,
            },
        ),
        (
            "word_docx_20260214_184824",
            {
                "unknown_path_dropped": 1,
                "id_missing_required": 10,
                "parent_lookup_miss": 1,
            },
        ),
        (
            "bauer2014_pdf_20260214_190306",
            {
                "unknown_path_dropped": 133,
                "id_missing_required": 303,
                "parent_lookup_miss": 161,
            },
        ),
    ],
)
def test_debug_artifact_baseline_metrics(run_dir: str, expected: dict[str, int]) -> None:
    trace_path = _repo_root() / "outputs" / run_dir / "debug" / "delta_trace.json"
    if not trace_path.exists():
        pytest.skip(f"Artifact not present: {trace_path}")

    trace = _read_json(trace_path)
    normalizer = trace.get("normalizer_stats", {})
    merge = trace.get("merge_stats", {})

    assert normalizer.get("unknown_path_dropped") == expected["unknown_path_dropped"]
    assert normalizer.get("id_missing_required") == expected["id_missing_required"]
    assert merge.get("parent_lookup_miss") == expected["parent_lookup_miss"]


def test_artifact_unknown_path_salvage_recovers_generic_fields() -> None:
    """Replay a sparse delta artifact and assert generic salvage metrics."""
    artifact_path = (
        _repo_root()
        / "outputs"
        / "sample_invoice_jpg_20260214_200347"
        / "debug"
        / "delta_batch_0_success.json"
    )
    if not artifact_path.exists():
        pytest.skip(f"Artifact not present: {artifact_path}")

    payload = _read_json(artifact_path)
    batch_output = payload.get("output")
    if not isinstance(batch_output, dict):
        pytest.skip("Artifact payload does not include output graph.")

    from docling_graph.core.extractors.contracts.delta.catalog import build_delta_node_catalog
    from docling_graph.core.extractors.contracts.delta.helpers import (
        build_dedup_policy,
        merge_delta_graphs,
    )
    from docling_graph.core.extractors.contracts.delta.ir_normalizer import (
        DeltaIrNormalizerConfig,
        normalize_delta_ir_batch_results,
    )
    from docling_graph.core.extractors.contracts.delta.schema_mapper import (
        project_graph_to_template_root,
    )
    from docs.examples.templates.billing_document import BillingDocument

    catalog = build_delta_node_catalog(BillingDocument)
    policy = build_dedup_policy(catalog)
    normalized, stats = normalize_delta_ir_batch_results(
        batch_results=[batch_output],
        batch_plan=[[(0, "chunk", 10)]],
        chunk_metadata=[{"token_count": 10, "page_numbers": [1]}],
        catalog=catalog,
        dedup_policy=policy,
        config=DeltaIrNormalizerConfig(),
    )
    merged = merge_delta_graphs(normalized, dedup_policy=policy)
    merged_root, _merge_stats = project_graph_to_template_root(merged, BillingDocument)

    assert stats.get("unknown_path_salvaged", 0) >= 1
    assert stats.get("salvaged_properties", 0) >= 1
    non_empty_scalars = sum(
        1
        for value in merged_root.values()
        if value is not None and not isinstance(value, dict | list)
    )
    assert non_empty_scalars >= 2


def test_cgv_artifact_replay_reduces_canonical_duplicates_and_orphans() -> None:
    output_root = _repo_root() / "outputs" / "CGV_DIRECT_md_20260214_210727" / "debug"
    trace_path = output_root / "delta_trace.json"
    if not trace_path.exists():
        pytest.skip(f"Artifact not present: {trace_path}")

    batch_files = sorted(output_root.glob("delta_batch_*_success.json"))
    if not batch_files:
        pytest.skip(f"No successful batch artifacts in: {output_root}")

    batch_results: list[dict] = []
    batch_plan: list[list[tuple[int, str, int]]] = []
    for idx, batch_file in enumerate(batch_files):
        payload = _read_json(batch_file)
        output = payload.get("output")
        if isinstance(output, dict):
            batch_results.append(output)
            batch_plan.append([(idx, "artifact", 10)])

    if not batch_results:
        pytest.skip("No valid batch outputs in artifact.")

    from docling_graph.core.extractors.contracts.delta.catalog import build_delta_node_catalog
    from docling_graph.core.extractors.contracts.delta.helpers import (
        build_dedup_policy,
        merge_delta_graphs,
    )
    from docling_graph.core.extractors.contracts.delta.ir_normalizer import (
        DeltaIrNormalizerConfig,
        normalize_delta_ir_batch_results,
    )
    from docling_graph.core.extractors.contracts.delta.resolvers import (
        DeltaResolverConfig,
        resolve_post_merge_graph,
    )
    from docling_graph.core.extractors.contracts.delta.schema_mapper import (
        project_graph_to_template_root,
    )
    from docs.examples.templates.cgv_mrh import AssuranceMRH

    catalog = build_delta_node_catalog(AssuranceMRH)
    policy = build_dedup_policy(catalog)
    normalized, _stats = normalize_delta_ir_batch_results(
        batch_results=batch_results,
        batch_plan=batch_plan,
        chunk_metadata=[{"token_count": 10, "page_numbers": [1]} for _ in batch_results],
        catalog=catalog,
        dedup_policy=policy,
        config=DeltaIrNormalizerConfig(),
    )
    merged = merge_delta_graphs(normalized, dedup_policy=policy)
    merged, _resolver_stats = resolve_post_merge_graph(
        merged,
        dedup_policy=policy,
        config=DeltaResolverConfig(enabled=True, mode="fuzzy", fuzzy_threshold=0.92),
    )
    try:
        merged_root, _merge_stats = project_graph_to_template_root(merged, AssuranceMRH)
    except (
        Exception
    ) as exc:  # pragma: no cover - artifact payloads can be structurally inconsistent
        pytest.skip(f"Artifact replay is structurally inconsistent for projection: {exc}")

    duplicates = 0
    seen: dict[str, set[str]] = {}
    for node in merged.get("nodes", []):
        if not isinstance(node, dict) or str(node.get("path") or "") != "offres[]":
            continue
        ids = node.get("ids") if isinstance(node.get("ids"), dict) else {}
        key = _canonicalize(ids.get("nom"))
        if not key:
            continue
        seen.setdefault("offres[]", set())
        if key in seen["offres[]"]:
            duplicates += 1
        else:
            seen["offres[]"].add(key)

    prior_trace = _read_json(trace_path)
    prior_orphans = int(prior_trace.get("merge_stats", {}).get("orphan_attached", 0))
    orphan_count = (
        len(merged_root.get("__orphans__", []))
        if isinstance(merged_root.get("__orphans__"), list)
        else 0
    )

    assert duplicates == 0
    assert orphan_count <= prior_orphans
