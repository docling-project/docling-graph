"""
Example 16: Extraction Evaluation (graph vs. ground truth)

Description:
    Scores an extracted `graph.json` against a template-shaped ground-truth JSON.
    The ground truth is validated through the SAME Pydantic template and converted
    with the SAME GraphConverter as the pipeline output, then the two graphs are
    compared. Fully domain-agnostic: any `(template, ground_truth.json, graph.json)`
    triple works, so it doubles as a regression harness when tuning a template or
    the extraction contracts.

Use Cases:
    - Measuring extraction quality (node/edge P/R/F1) before and after a change
    - Comparing the "direct" vs "dense" contracts on the same document
    - Catching integrity regressions (empty-identity nodes, orphans)

Prerequisites:
    - Installation: uv sync
    - A completed run's `docling_graph/graph.json`
    - A ground-truth JSON shaped like the template (validates against it)
    - Optional: the source markdown, to compute a verbatim-fidelity ratio

Key Concepts:
    - Strict node match: exact canonical `graph_id_fields` equality
    - Relaxed node match: unique same-class containment with equal digit signature
      (tolerates a short table label vs. a full section-title alias)
    - Edge metrics: endpoints matched via node identity, per edge label
    - Attribute completeness: fill-rate on strictly matched nodes, per field

Expected Output:
    - A one-screen console report (nodes, edges, completeness, integrity)
    - Optional `--out eval.json` with the full machine-readable report

Related Examples:
    - Example 02: Basic LLM extraction (produces the graph.json to score)
    - Example 15: Provenance grounding (another way to audit extracted nodes)
    - Documentation: docs/examples/README.md

Usage:
    uv run python docs/examples/scripts/16_extraction_evaluation.py \
        --graph outputs/RUN_DIR/docling_graph/graph.json \
        --truth data/insurance_terms/ground_truth.json \
        --template docs.examples.templates.insurance_terms.AssuranceMRH \
        [--out eval.json] [--source document.md]
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

# Allow the dotted --template path (e.g. docs.examples.templates.<name>.<Class>)
# to resolve when run from anywhere: put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pydantic import BaseModel

from docling_graph.core.converters.graph_converter import GraphConverter
from docling_graph.core.utils.entity_name_normalizer import (
    canonicalize_identity_for_dedup,
)

META_KEYS = {"id", "label", "type", "__class__", "__provenance__", "merged_aliases"}
_DIGIT_RUNS = re.compile(r"\d+")


def load_template(dotted: str) -> type[BaseModel]:
    module_path, _, attr = dotted.rpartition(".")
    if not module_path:
        raise SystemExit(f"--template must be a dotted path, got {dotted!r}")
    module = importlib.import_module(module_path)
    template = getattr(module, attr)
    if not (isinstance(template, type) and issubclass(template, BaseModel)):
        raise SystemExit(f"{dotted!r} is not a Pydantic model")
    return template


def _unwrap_models(annotation: Any) -> list[type[BaseModel]]:
    from typing import get_args

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    out: list[type[BaseModel]] = []
    for arg in get_args(annotation):
        out.extend(_unwrap_models(arg))
    return out


def collect_id_fields(template: type[BaseModel]) -> dict[str, list[str]]:
    """Map every model class reachable from the template to its graph_id_fields."""
    result: dict[str, list[str]] = {}
    stack: list[type[BaseModel]] = [template]
    seen: set[type[BaseModel]] = set()
    while stack:
        model = stack.pop()
        if model in seen:
            continue
        seen.add(model)
        cfg = getattr(model, "model_config", {}) or {}
        raw = cfg.get("graph_id_fields", []) if isinstance(cfg, dict) else []
        result[model.__name__] = [f for f in raw if isinstance(f, str)]
        for field_info in model.model_fields.values():
            stack.extend(_unwrap_models(field_info.annotation))
    return result


def graph_from_truth(
    truth_path: Path, template: type[BaseModel]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate + convert the ground truth with pipeline mechanics."""
    data = json.loads(truth_path.read_text(encoding="utf-8"))
    model = template.model_validate(data)
    converter = GraphConverter(validate_graph=False, auto_cleanup=True)
    graph, _meta = converter.pydantic_list_to_graph([model])
    nodes = [{"id": nid, **attrs} for nid, attrs in graph.nodes(data=True)]
    edges = [
        {"source": u, "target": v, "label": d.get("label", "")}
        for u, v, d in graph.edges(data=True)
    ]
    return nodes, edges


def load_graph_json(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("nodes", []), data.get("edges", [])


def node_identity(
    node: dict[str, Any], id_fields_by_class: dict[str, list[str]]
) -> tuple[str, tuple[str, ...]] | None:
    """(class, canonical id values) or None when the node has no usable identity."""
    cls = str(node.get("__class__") or node.get("label") or "")
    values: list[str] = []
    for field in id_fields_by_class.get(cls, []):
        canon = canonicalize_identity_for_dedup(field, node.get(field))
        values.append(canon)
    if not any(values):
        return None
    return (cls, tuple(values))


def _digit_signature(text: str) -> tuple[str, ...]:
    return tuple(_DIGIT_RUNS.findall(text))


def relaxed_match(
    unmatched_a: list[tuple[str, tuple[str, ...]]],
    unmatched_b: list[tuple[str, tuple[str, ...]]],
) -> list[tuple[tuple[str, tuple[str, ...]], tuple[str, tuple[str, ...]]]]:
    """Pair remaining identities via unique same-class containment (equal digit runs)."""
    pairs = []
    used_b: set[int] = set()
    for key_a in unmatched_a:
        text_a = "".join(key_a[1])
        candidates = []
        for i, key_b in enumerate(unmatched_b):
            if i in used_b or key_b[0] != key_a[0]:
                continue
            text_b = "".join(key_b[1])
            if not text_a or not text_b:
                continue
            if _digit_signature(text_a) != _digit_signature(text_b):
                continue
            if text_a in text_b or text_b in text_a:
                candidates.append(i)
        if len(candidates) == 1:
            used_b.add(candidates[0])
            pairs.append((key_a, unmatched_b[candidates[0]]))
    return pairs


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "p": round(precision, 3),
        "r": round(recall, 3),
        "f1": round(f1, 3),
    }


def is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | dict):
        return bool(value)
    return True


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return re.sub(r"\s+", " ", text).casefold().strip()


def evaluate(
    graph_path: Path,
    truth_path: Path,
    template_dotted: str,
    source_path: Path | None,
) -> dict[str, Any]:
    template = load_template(template_dotted)
    id_fields_by_class = collect_id_fields(template)
    gt_nodes, gt_edges = graph_from_truth(truth_path, template)
    got_nodes, got_edges = load_graph_json(graph_path)

    def identity_index(
        nodes: list[dict[str, Any]],
    ) -> tuple[
        dict[tuple[str, tuple[str, ...]], dict[str, Any]],
        list[dict[str, Any]],
        dict[str, tuple[str, tuple[str, ...]]],
    ]:
        by_key: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        unkeyed: list[dict[str, Any]] = []
        node_id_to_key: dict[str, tuple[str, tuple[str, ...]]] = {}
        for node in nodes:
            key = node_identity(node, id_fields_by_class)
            if key is None:
                unkeyed.append(node)
                continue
            by_key.setdefault(key, node)  # first wins; duplicates are integrity findings
            node_id_to_key[str(node.get("id"))] = key
        return by_key, unkeyed, node_id_to_key

    gt_by_key, gt_unkeyed, gt_id_to_key = identity_index(gt_nodes)
    got_by_key, got_unkeyed, got_id_to_key = identity_index(got_nodes)

    # --- node metrics ---
    strict_matches = set(gt_by_key) & set(got_by_key)
    gt_only = [k for k in gt_by_key if k not in strict_matches]
    got_only = [k for k in got_by_key if k not in strict_matches]
    relaxed_pairs = relaxed_match(gt_only, got_only)
    relaxed_gt = {a for a, _ in relaxed_pairs}
    relaxed_got = {b for _, b in relaxed_pairs}

    classes = sorted({k[0] for k in list(gt_by_key) + list(got_by_key)})
    node_metrics: dict[str, dict[str, Any]] = {}
    for cls in classes:
        gt_cls = {k for k in gt_by_key if k[0] == cls}
        got_cls = {k for k in got_by_key if k[0] == cls}
        tp = len(gt_cls & got_cls)
        node_metrics[cls] = {
            "strict": prf(tp, len(got_cls) - tp, len(gt_cls) - tp),
            "relaxed": prf(
                tp + len([a for a in relaxed_gt if a[0] == cls]),
                len(got_cls) - tp - len([b for b in relaxed_got if b[0] == cls]),
                len(gt_cls) - tp - len([a for a in relaxed_gt if a[0] == cls]),
            ),
        }

    # --- edge metrics ---
    def edge_keys(
        edges: list[dict[str, Any]], id_to_key: dict[str, tuple], relaxed_map: dict
    ) -> set[tuple]:
        keys = set()
        for edge in edges:
            src = id_to_key.get(str(edge.get("source")))
            dst = id_to_key.get(str(edge.get("target")))
            if src is None or dst is None:
                continue
            src = relaxed_map.get(src, src)
            dst = relaxed_map.get(dst, dst)
            keys.add((str(edge.get("label", "")), src, dst))
        return keys

    # relaxed alias map folds the produced-side alias onto the GT identity
    alias_fold = {b: a for a, b in relaxed_pairs}
    gt_edge_keys = edge_keys(gt_edges, gt_id_to_key, {})
    got_edge_keys_strict = edge_keys(got_edges, got_id_to_key, {})
    got_edge_keys_relaxed = edge_keys(got_edges, got_id_to_key, alias_fold)

    labels = sorted({k[0] for k in gt_edge_keys | got_edge_keys_strict})
    edge_metrics: dict[str, dict[str, Any]] = {}
    for label in labels:
        gt_l = {k for k in gt_edge_keys if k[0] == label}
        strict_l = {k for k in got_edge_keys_strict if k[0] == label}
        relaxed_l = {k for k in got_edge_keys_relaxed if k[0] == label}
        edge_metrics[label] = {
            "strict": prf(len(gt_l & strict_l), len(strict_l - gt_l), len(gt_l - strict_l)),
            "relaxed": prf(len(gt_l & relaxed_l), len(relaxed_l - gt_l), len(gt_l - relaxed_l)),
        }

    # --- attribute completeness on strict matches ---
    source_text = _normalize_text(source_path.read_text(encoding="utf-8")) if source_path else None
    completeness: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    verbatim = [0, 0]
    for key in strict_matches:
        gt_node, got_node = gt_by_key[key], got_by_key[key]
        for field, gt_value in gt_node.items():
            if field in META_KEYS or not is_filled(gt_value):
                continue
            slot = completeness[key[0]][field]
            slot[1] += 1
            if is_filled(got_node.get(field)):
                slot[0] += 1
        if source_text:
            for field, value in got_node.items():
                if field in META_KEYS or not isinstance(value, str) or len(value) < 80:
                    continue
                verbatim[1] += 1
                if _normalize_text(value) in source_text:
                    verbatim[0] += 1

    completeness_out = {
        cls: {field: {"filled": c[0], "expected": c[1]} for field, c in fields.items()}
        for cls, fields in completeness.items()
    }

    # --- integrity ---
    empty_identity = []
    for node in got_nodes:
        cls = str(node.get("__class__") or "")
        fields = id_fields_by_class.get(cls, [])
        if fields and not any(is_filled(node.get(f)) for f in fields):
            empty_identity.append(str(node.get("id")))
    referenced = {str(e.get("source")) for e in got_edges} | {
        str(e.get("target")) for e in got_edges
    }
    orphans = [str(n.get("id")) for n in got_nodes if str(n.get("id")) not in referenced]

    return {
        "graph": str(graph_path),
        "truth": str(truth_path),
        "template": template_dotted,
        "nodes": node_metrics,
        "edges": edge_metrics,
        "relaxed_alias_pairs": [
            {"truth": " / ".join(a[1]), "extracted": " / ".join(b[1]), "class": a[0]}
            for a, b in relaxed_pairs
        ],
        "attribute_completeness": completeness_out,
        "verbatim_ratio": (
            round(verbatim[0] / verbatim[1], 3) if source_text and verbatim[1] else None
        ),
        "integrity": {
            "empty_identity_nodes": empty_identity,
            "orphan_nodes": orphans,
            "unkeyed_extracted_nodes": len(got_unkeyed),
            "unkeyed_truth_nodes": len(gt_unkeyed),
        },
    }


def print_report(result: dict[str, Any]) -> None:
    print(f"\n# Extraction eval: {result['graph']}")
    print("\n## Nodes (strict | relaxed)  P / R / F1")
    for cls, m in result["nodes"].items():
        s, r = m["strict"], m["relaxed"]
        print(
            f"  {cls:<16} {s['p']:.2f}/{s['r']:.2f}/{s['f1']:.2f}"
            f"  |  {r['p']:.2f}/{r['r']:.2f}/{r['f1']:.2f}"
            f"   (tp={s['tp']} fp={s['fp']} fn={s['fn']})"
        )
    print("\n## Edges (strict | relaxed)  P / R / F1")
    for label, m in result["edges"].items():
        s, r = m["strict"], m["relaxed"]
        print(
            f"  {label:<22} {s['p']:.2f}/{s['r']:.2f}/{s['f1']:.2f}"
            f"  |  {r['p']:.2f}/{r['r']:.2f}/{r['f1']:.2f}"
            f"   (tp={s['tp']} fp={s['fp']} fn={s['fn']})"
        )
    if result["relaxed_alias_pairs"]:
        print("\n## Alias pairs tolerated in relaxed mode")
        for pair in result["relaxed_alias_pairs"]:
            print(f"  {pair['class']}: {pair['truth']!r} ~ {pair['extracted']!r}")
    print("\n## Attribute completeness (filled/expected on matched nodes)")
    for cls, fields in result["attribute_completeness"].items():
        parts = ", ".join(f"{f}={c['filled']}/{c['expected']}" for f, c in sorted(fields.items()))
        print(f"  {cls}: {parts}")
    if result.get("verbatim_ratio") is not None:
        print(f"\n## Verbatim ratio (long str fields found in source): {result['verbatim_ratio']}")
    integ = result["integrity"]
    print(
        f"\n## Integrity: empty-identity={len(integ['empty_identity_nodes'])}, "
        f"orphans={len(integ['orphan_nodes'])}, "
        f"unkeyed extracted/truth={integ['unkeyed_extracted_nodes']}/{integ['unkeyed_truth_nodes']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate(args.graph, args.truth, args.template, args.source)
    print_report(result)
    if args.out:
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print(f"\nSaved JSON report to {args.out}")


if __name__ == "__main__":
    main()
