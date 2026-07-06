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
    - Scoring ground truths whose identifiers are synthesized (structural alignment)

Prerequisites:
    - Installation: uv sync
    - A completed run's `docling_graph/graph.json`
    - A ground-truth JSON shaped like the template (validates against it)
    - Optional: the source text, to compute a verbatim-fidelity ratio — pass the
      serialization the model was actually served (`document.md`, or
      `document.dclg` for DocLang runs; markup is stripped automatically)

Key Concepts:
    - Strict node match: exact canonical `graph_id_fields` equality
    - Relaxed node match: unique same-class containment with equal digit signature
      (tolerates a short table label vs. a full section-title alias)
    - Structural alignment (id-agnostic): when both sides invent identifier slugs
      (detected via the source text, or forced with --structural-align on),
      unmatched same-class nodes are paired by attribute overlap instead — strict
      identity matching on two models' invented slugs is unwinnable by design
    - Edge metrics: endpoints matched via node identity, per edge label
    - Micro P/R/F1: tp/fp/fn summed across classes (large classes dominate),
      reported separately for nodes and edges — the headline numbers
    - Attribute completeness: fill-rate on strictly matched nodes, per field
    - Edge fan-out: per label, the share of edges hanging off the single busiest
      source node, extracted vs truth — flags "dump everything on one parent"

Expected Output:
    - A one-screen console report (nodes, edges, micro, completeness, integrity)
    - Optional `--out eval.json` with the full machine-readable report

Related Examples:
    - Example 02: Basic LLM extraction (produces the graph.json to score)
    - Example 15: Provenance grounding (another way to audit extracted nodes)
    - Example 17: Benchmark aggregation (consumes eval.json across a matrix)
    - Documentation: docs/examples/README.md

Usage:
    uv run python docs/examples/scripts/16_extraction_evaluation.py \
        --graph outputs/RUN_DIR/docling_graph/graph.json \
        --truth data/insurance_terms/ground_truth.json \
        --template docs.examples.templates.insurance_terms.AssuranceMRH \
        [--out eval.json] [--source document.md] [--structural-align auto|on|off]
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
from docling_graph.core.utils.doclang_format import strip_doclang_markup
from docling_graph.core.utils.entity_name_normalizer import (
    canonicalize_identity_for_dedup,
)

META_KEYS = {"id", "label", "type", "__class__", "__provenance__", "merged_aliases"}
_DIGIT_RUNS = re.compile(r"\d+")
# Bullet/dash markers that follow whitespace (or start a line). Both the source
# and extracted values are normalized with this, so list reconstruction quirks
# ("- -les murs" vs "-les murs") stop failing the verbatim containment check.
_LOOSE_BULLETS = re.compile(r"(?:(?<=\s)|^)[-•*+]+(?=\S)", re.MULTILINE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


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


def load_source_text(path: Path) -> str:
    """Read the source text; DocLang serializations are stripped to content."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".dclg", ".dclx", ".xml"} or raw.lstrip().startswith("<doctag"):
        return strip_doclang_markup(raw)
    return raw


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


def micro_prf(metrics: dict[str, dict[str, Any]], kind: str) -> dict[str, float]:
    """Micro P/R/F1 across all classes/labels of one metrics section."""
    tp = sum(m[kind]["tp"] for m in metrics.values() if kind in m)
    fp = sum(m[kind]["fp"] for m in metrics.values() if kind in m)
    fn = sum(m[kind]["fn"] for m in metrics.values() if kind in m)
    return prf(tp, fp, fn)


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
    text = _LOOSE_BULLETS.sub("", text)
    return re.sub(r"\s+", " ", text).casefold().strip()


def _squash(text: str) -> str:
    """Lowercased [a-z0-9]-only projection for identifier-in-source checks."""
    return _NON_ALNUM.sub("", text.casefold())


# ---------------------------------------------------------------------------
# Structural alignment (id-agnostic matching for synthesized-identifier GTs)
# ---------------------------------------------------------------------------


def detect_synthetic_classes(
    gt_by_key: dict[tuple[str, tuple[str, ...]], dict[str, Any]],
    source_squashed: str,
) -> set[str]:
    """Classes whose ground-truth identifiers mostly do NOT occur in the source.

    A GT id like ``STUDY-LFP-GELATION`` was invented by the GT author; no honest
    extraction can reproduce it, so strict identity matching is unwinnable for
    that class and structural alignment is the fair rung.
    """
    per_class: dict[str, list[bool]] = {}
    for cls, id_values in gt_by_key:
        candidates = [_squash(v) for v in id_values if v and len(_squash(v)) >= 3]
        found = any(c in source_squashed for c in candidates)
        per_class.setdefault(cls, []).append(found)
    return {cls for cls, hits in per_class.items() if hits and sum(hits) / len(hits) < 0.5}


def _pair_similarity(gt_node: dict[str, Any], got_node: dict[str, Any]) -> float:
    """Attribute-overlap score between two same-class nodes (ids excluded implicitly).

    +1 per exact-equal filled scalar field, +0.5 per containment match or
    overlapping list field. Identity fields participate like any other field:
    when slugs are invented they simply never match, which is the point.
    """
    score = 0.0
    for field, gt_value in gt_node.items():
        if field in META_KEYS or not is_filled(gt_value):
            continue
        got_value = got_node.get(field)
        if not is_filled(got_value):
            continue
        if isinstance(gt_value, str | int | float) and isinstance(got_value, str | int | float):
            a, b = _normalize_text(str(gt_value)), _normalize_text(str(got_value))
            if not a or not b:
                continue
            if a == b:
                score += 1.0
            elif len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
                score += 0.5
        elif isinstance(gt_value, list) and isinstance(got_value, list):
            a_set = {_normalize_text(str(x)) for x in gt_value if isinstance(x, str | int | float)}
            b_set = {_normalize_text(str(x)) for x in got_value if isinstance(x, str | int | float)}
            if a_set & b_set:
                score += 0.5
    return score


def structural_align(
    remaining_gt: list[tuple[str, tuple[str, ...]]],
    remaining_got: list[tuple[str, tuple[str, ...]]],
    gt_by_key: dict[tuple[str, tuple[str, ...]], dict[str, Any]],
    got_by_key: dict[tuple[str, tuple[str, ...]], dict[str, Any]],
    classes: set[str],
) -> list[tuple[tuple[str, tuple[str, ...]], tuple[str, tuple[str, ...]], float]]:
    """Greedy one-to-one pairing of unmatched same-class nodes by attribute overlap.

    Only pairs with score >= 1.0 (at least one exactly-matching attribute) are
    accepted, best-first, each node used at most once.
    """
    candidates: list[tuple[float, tuple[str, tuple[str, ...]], tuple[str, tuple[str, ...]]]] = []
    for key_gt in remaining_gt:
        if key_gt[0] not in classes:
            continue
        for key_got in remaining_got:
            if key_got[0] != key_gt[0]:
                continue
            score = _pair_similarity(gt_by_key[key_gt], got_by_key[key_got])
            if score >= 1.0:
                candidates.append((score, key_gt, key_got))
    candidates.sort(key=lambda t: -t[0])
    used_gt: set[tuple[str, tuple[str, ...]]] = set()
    used_got: set[tuple[str, tuple[str, ...]]] = set()
    pairs = []
    for score, key_gt, key_got in candidates:
        if key_gt in used_gt or key_got in used_got:
            continue
        used_gt.add(key_gt)
        used_got.add(key_got)
        pairs.append((key_gt, key_got, score))
    return pairs


def edge_fanout(
    edges: list[dict[str, Any]], id_to_key: dict[str, tuple[str, tuple[str, ...]]]
) -> dict[str, dict[str, Any]]:
    """Per edge label: total edges and the share held by the busiest source node."""
    per_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for edge in edges:
        label = str(edge.get("label", ""))
        src = id_to_key.get(str(edge.get("source"))) or str(edge.get("source"))
        per_label[label][str(src)] += 1
    out: dict[str, dict[str, Any]] = {}
    for label, counts in per_label.items():
        total = sum(counts.values())
        out[label] = {
            "total": total,
            "sources": len(counts),
            "top_source_share": round(max(counts.values()) / total, 3) if total else 0.0,
        }
    return out


def evaluate(
    graph_path: Path,
    truth_path: Path,
    template_dotted: str,
    source_path: Path | None,
    structural_align_mode: str = "auto",
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

    raw_source = load_source_text(source_path) if source_path else None
    source_text = _normalize_text(raw_source) if raw_source else None
    source_squashed = _squash(raw_source) if raw_source else None

    # --- node matching: strict -> relaxed -> structural alignment ---
    strict_matches = set(gt_by_key) & set(got_by_key)
    gt_only = [k for k in gt_by_key if k not in strict_matches]
    got_only = [k for k in got_by_key if k not in strict_matches]
    relaxed_pairs = relaxed_match(gt_only, got_only)
    relaxed_gt = {a for a, _ in relaxed_pairs}
    relaxed_got = {b for _, b in relaxed_pairs}

    synthetic_classes: set[str] = set()
    aligned_pairs: list[tuple[Any, Any, float | str]] = []
    if structural_align_mode == "on":
        align_classes = {k[0] for k in gt_by_key} | {k[0] for k in got_by_key}
    elif structural_align_mode == "auto" and source_squashed:
        synthetic_classes = detect_synthetic_classes(gt_by_key, source_squashed)
        align_classes = synthetic_classes
    else:
        align_classes = set()

    # Root singleton pairing: both graphs hold exactly one root-class node, so
    # its identity is decorative for matching purposes — an extraction whose
    # root fell back to a synthetic identity (source stem) must still have its
    # root-anchored edges scored. Applies whenever alignment is not "off".
    root_cls = template.__name__
    if structural_align_mode != "off":
        gt_root_keys = [k for k in gt_by_key if k[0] == root_cls]
        got_root_keys = [k for k in got_by_key if k[0] == root_cls]
        if (
            len(gt_root_keys) == 1
            and len(got_root_keys) == 1
            and gt_root_keys[0] in gt_only
            and got_root_keys[0] in got_only
            and gt_root_keys[0] not in relaxed_gt
            and got_root_keys[0] not in relaxed_got
        ):
            aligned_pairs.append((gt_root_keys[0], got_root_keys[0], "singleton-root"))

    if align_classes:
        paired_gt = {a for a, _, _ in aligned_pairs}
        paired_got = {b for _, b, _ in aligned_pairs}
        remaining_gt = [k for k in gt_only if k not in relaxed_gt and k not in paired_gt]
        remaining_got = [k for k in got_only if k not in relaxed_got and k not in paired_got]
        aligned_pairs.extend(
            structural_align(remaining_gt, remaining_got, gt_by_key, got_by_key, align_classes)
        )
    aligned_gt = {a for a, _, _ in aligned_pairs}
    aligned_got = {b for _, b, _ in aligned_pairs}
    aligned_active = bool(align_classes) or bool(aligned_pairs)

    classes = sorted({k[0] for k in list(gt_by_key) + list(got_by_key)})
    node_metrics: dict[str, dict[str, Any]] = {}
    for cls in classes:
        gt_cls = {k for k in gt_by_key if k[0] == cls}
        got_cls = {k for k in got_by_key if k[0] == cls}
        tp = len(gt_cls & got_cls)
        relaxed_tp_extra = len([a for a in relaxed_gt if a[0] == cls])
        aligned_tp_extra = relaxed_tp_extra + len([a for a in aligned_gt if a[0] == cls])
        node_metrics[cls] = {
            "strict": prf(tp, len(got_cls) - tp, len(gt_cls) - tp),
            "relaxed": prf(
                tp + relaxed_tp_extra,
                len(got_cls) - tp - len([b for b in relaxed_got if b[0] == cls]),
                len(gt_cls) - tp - relaxed_tp_extra,
            ),
        }
        if aligned_active:
            node_metrics[cls]["aligned"] = prf(
                tp + aligned_tp_extra,
                len(got_cls)
                - tp
                - len([b for b in relaxed_got if b[0] == cls])
                - len([b for b in aligned_got if b[0] == cls]),
                len(gt_cls) - tp - aligned_tp_extra,
            )

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

    # relaxed alias map folds the produced-side alias onto the GT identity;
    # the aligned map extends it with the structural pairs.
    alias_fold = {b: a for a, b in relaxed_pairs}
    aligned_fold = dict(alias_fold)
    aligned_fold.update({b: a for a, b, _ in aligned_pairs})
    gt_edge_keys = edge_keys(gt_edges, gt_id_to_key, {})
    got_edge_keys_strict = edge_keys(got_edges, got_id_to_key, {})
    got_edge_keys_relaxed = edge_keys(got_edges, got_id_to_key, alias_fold)
    got_edge_keys_aligned = edge_keys(got_edges, got_id_to_key, aligned_fold)

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
        if aligned_active:
            aligned_l = {k for k in got_edge_keys_aligned if k[0] == label}
            edge_metrics[label]["aligned"] = prf(
                len(gt_l & aligned_l), len(aligned_l - gt_l), len(gt_l - aligned_l)
            )

    # --- micro summaries (headline numbers; never mix nodes with edges) ---
    micro: dict[str, dict[str, Any]] = {
        "nodes": {kind: micro_prf(node_metrics, kind) for kind in ("strict", "relaxed")},
        "edges": {kind: micro_prf(edge_metrics, kind) for kind in ("strict", "relaxed")},
    }
    if aligned_active:
        micro["nodes"]["aligned"] = micro_prf(node_metrics, "aligned")
        micro["edges"]["aligned"] = micro_prf(edge_metrics, "aligned")

    # --- attribute completeness on strict matches ---
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

    # --- edge fan-out (flags single-parent dumping on reference-style labels) ---
    got_fanout = edge_fanout(got_edges, got_id_to_key)
    gt_fanout = edge_fanout(gt_edges, gt_id_to_key)
    fanout = {
        label: {"extracted": got_fanout.get(label), "truth": gt_fanout.get(label)}
        for label in labels
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
        "micro": micro,
        "nodes": node_metrics,
        "edges": edge_metrics,
        "relaxed_alias_pairs": [
            {"truth": " / ".join(a[1]), "extracted": " / ".join(b[1]), "class": a[0]}
            for a, b in relaxed_pairs
        ],
        "structural_alignment": {
            "mode": structural_align_mode,
            "classes": sorted(align_classes),
            "synthetic_id_classes": sorted(synthetic_classes),
            "pairs": [
                {
                    "truth": " / ".join(a[1]),
                    "extracted": " / ".join(b[1]),
                    "class": a[0],
                    "score": score,
                }
                for a, b, score in aligned_pairs
            ],
        },
        "attribute_completeness": completeness_out,
        "verbatim_ratio": (
            round(verbatim[0] / verbatim[1], 3) if source_text and verbatim[1] else None
        ),
        "verbatim_n": verbatim[1] if source_text else None,
        "edge_fanout": fanout,
        "integrity": {
            "empty_identity_nodes": empty_identity,
            "orphan_nodes": orphans,
            "unkeyed_extracted_nodes": len(got_unkeyed),
            "unkeyed_truth_nodes": len(gt_unkeyed),
        },
    }


def print_report(result: dict[str, Any]) -> None:
    print(f"\n# Extraction eval: {result['graph']}")

    kinds = ["strict", "relaxed"]
    has_aligned = "aligned" in result["micro"]["nodes"]
    if has_aligned:
        kinds.append("aligned")

    def _rung_str(m: dict[str, Any]) -> str:
        parts = [f"{m[k]['p']:.2f}/{m[k]['r']:.2f}/{m[k]['f1']:.2f}" for k in kinds if k in m]
        return "  |  ".join(parts)

    print(f"\n## Micro P / R / F1 ({' | '.join(kinds)})")
    print(f"  nodes: {_rung_str(result['micro']['nodes'])}")
    print(f"  edges: {_rung_str(result['micro']['edges'])}")

    print(f"\n## Nodes per class ({' | '.join(kinds)})  P / R / F1")
    for cls, m in result["nodes"].items():
        s = m["strict"]
        print(f"  {cls:<16} {_rung_str(m)}   (tp={s['tp']} fp={s['fp']} fn={s['fn']})")
    print(f"\n## Edges per label ({' | '.join(kinds)})  P / R / F1")
    for label, m in result["edges"].items():
        s = m["strict"]
        print(f"  {label:<22} {_rung_str(m)}   (tp={s['tp']} fp={s['fp']} fn={s['fn']})")

    if result["relaxed_alias_pairs"]:
        print("\n## Alias pairs tolerated in relaxed mode")
        for pair in result["relaxed_alias_pairs"]:
            print(f"  {pair['class']}: {pair['truth']!r} ~ {pair['extracted']!r}")

    alignment = result.get("structural_alignment", {})
    if alignment.get("synthetic_id_classes"):
        print(
            "\n## Synthetic-id classes (GT identifiers absent from source; "
            "strict matching unwinnable): " + ", ".join(alignment["synthetic_id_classes"])
        )
    if alignment.get("pairs"):
        print("\n## Structurally aligned pairs (attribute overlap)")
        for pair in alignment["pairs"]:
            print(
                f"  {pair['class']}: {pair['truth']!r} ~ {pair['extracted']!r}"
                f" (score {pair['score']})"
            )

    print("\n## Attribute completeness (filled/expected on matched nodes)")
    total_filled = total_expected = 0
    for cls, fields in result["attribute_completeness"].items():
        parts = ", ".join(f"{f}={c['filled']}/{c['expected']}" for f, c in sorted(fields.items()))
        total_filled += sum(c["filled"] for c in fields.values())
        total_expected += sum(c["expected"] for c in fields.values())
        print(f"  {cls}: {parts}")
    if total_expected:
        print(f"  TOTAL: {total_filled}/{total_expected}")

    if result.get("verbatim_ratio") is not None:
        print(
            f"\n## Verbatim ratio (long str fields found in served source): "
            f"{result['verbatim_ratio']} (n={result.get('verbatim_n')})"
        )

    flagged = []
    for label, sides in result.get("edge_fanout", {}).items():
        got_side, gt_side = sides.get("extracted"), sides.get("truth")
        if not got_side or got_side["total"] < 5:
            continue
        gt_share = gt_side["top_source_share"] if gt_side else 0.0
        if got_side["top_source_share"] >= 0.6 and gt_share <= 0.4:
            flagged.append(
                f"  {label}: extracted top-source share {got_side['top_source_share']:.2f} "
                f"({got_side['total']} edges) vs truth {gt_share:.2f} — "
                "membership may be dumped onto one parent"
            )
    if flagged:
        print("\n## Edge fan-out warnings")
        for line in flagged:
            print(line)

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
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Source text the model was served (document.md, or document.dclg for "
            "DocLang runs); enables the verbatim ratio and synthetic-id detection"
        ),
    )
    parser.add_argument(
        "--structural-align",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Id-agnostic node alignment: 'auto' aligns only classes whose GT "
            "identifiers are absent from --source (default; no-op without "
            "--source), 'on' aligns every class, 'off' disables"
        ),
    )
    args = parser.parse_args()

    result = evaluate(args.graph, args.truth, args.template, args.source, args.structural_align)
    print_report(result)
    if args.out:
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print(f"\nSaved JSON report to {args.out}")


if __name__ == "__main__":
    main()
