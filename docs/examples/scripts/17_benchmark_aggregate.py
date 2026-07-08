"""
Example 17: Benchmark Matrix Aggregation

Description:
    Scans a benchmark output tree (documents x extraction contracts x LLM input
    formats, optionally x repeats) and aggregates each run's `eval.json`
    (produced by Example 16), `metadata.json`, and dense debug stats into one
    machine-readable matrix plus a human-readable synthesis report. Micro
    P/R/F1 is computed separately for nodes and edges (strict and relaxed) —
    never averaged together — and dense merge retention / chunk coverage are
    surfaced next to the quality numbers so a lossy run cannot hide behind a
    good-looking F1.

Use Cases:
    - Comparing extraction contracts (direct vs dense) on the same documents
    - Comparing LLM serializations (markdown vs doclang vs doclang-geo)
    - Tracking regressions across benchmark reruns (repeat-aware: mean +/- spread)

Prerequisites:
    - A benchmark tree shaped like `run_benchmark.sh` produces:
        <root>/<doc>/<contract>/<format>[/rep<N>]/<run_dir>/
      where each run_dir holds `metadata.json`, optionally `eval.json`
      (Example 16 output) and `debug/dense_run_stats.json`.

Key Concepts:
    - Micro F1: tp/fp/fn summed across classes before computing P/R/F1, so
      large classes (e.g. 60 exclusions) dominate a 1-instance root
    - Merge retention: % of skeleton instances that survived the dense
      fill->merge (NOT source coverage; see chunk_coverage_pct)
    - Repeats: cells may contain rep1..repN; numeric metrics report the mean,
      and the run count is shown so single-run cells are recognizable

Expected Output:
    - `<root>/_aggregate.json` — one record per run
    - `<root>/synthesis_report.md` — the matrix table (mean over repeats)

Related Examples:
    - Example 16: Extraction evaluation (produces the eval.json scored here)
    - Documentation: docs/examples/README.md

Usage:
    uv run python docs/examples/scripts/17_benchmark_aggregate.py \
        [--root outputs/benchmarks] [--quiet]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

CONTRACTS = ("direct", "dense")


def micro(section: dict[str, Any], kind: str) -> dict[str, Any]:
    """Micro P/R/F1 over all classes of an eval.json nodes/edges section."""
    tp = sum(v[kind]["tp"] for v in section.values())
    fp = sum(v[kind]["fp"] for v in section.values())
    fn = sum(v[kind]["fn"] for v in section.values())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": round(p, 3), "r": round(r, 3), "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


def completeness_rate(comp: dict[str, Any]) -> tuple[float | None, int]:
    """(fill-rate, expected-count) over all matched-node attribute slots."""
    filled = expected = 0
    for fields in comp.values():
        for c in fields.values():
            filled += c["filled"]
            expected += c["expected"]
    return (round(filled / expected, 3) if expected else None), expected


def run_record(run_dir: Path, doc: str, contract: str, fmt: str, rep: int) -> dict[str, Any]:
    """Collect one run directory's metrics into a flat record."""
    row: dict[str, Any] = {
        "doc": doc,
        "contract": contract,
        "format": fmt,
        "rep": rep,
        "run": run_dir.name,
    }
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        row["time_s"] = meta.get("processing_time_seconds")
        results = meta.get("results") or {}
        row["nodes"] = results.get("nodes")
        row["edges"] = results.get("edges")

    eval_path = run_dir / "eval.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        row["node_strict"] = micro(ev["nodes"], "strict")
        row["node_relaxed"] = micro(ev["nodes"], "relaxed")
        row["edge_strict"] = micro(ev["edges"], "strict")
        row["edge_relaxed"] = micro(ev["edges"], "relaxed")
        comp, comp_n = completeness_rate(ev.get("attribute_completeness", {}))
        row["completeness"] = comp
        row["completeness_n"] = comp_n
        row["verbatim"] = ev.get("verbatim_ratio")
        integ = ev.get("integrity", {})
        row["empty_identity"] = len(integ.get("empty_identity_nodes", []))
        row["orphans"] = len(integ.get("orphan_nodes", []))

    stats_path = run_dir / "debug" / "dense_run_stats.json"
    if stats_path.exists():
        st = json.loads(stats_path.read_text(encoding="utf-8"))
        row["retention_pct"] = st.get("retention_pct")
        row["merge_dropped"] = st.get("merge_orphans_dropped")
        row["chunk_coverage_pct"] = st.get("chunk_coverage_pct")
        row["skeleton_nodes"] = st.get("skeleton_nodes")
        row["truncations"] = st.get("truncation_count")
        row["parents_from_already_found"] = st.get("parents_from_already_found")
    return row


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "metadata.json").exists()


def discover_runs(root: Path) -> list[dict[str, Any]]:
    """Walk <root>/<doc>/<contract>/<format>[/rep<N>]/<run_dir> (rep level optional)."""
    records: list[dict[str, Any]] = []
    for doc_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for contract in CONTRACTS:
            contract_dir = doc_dir / contract
            if not contract_dir.is_dir():
                continue
            for fmt_dir in sorted(p for p in contract_dir.iterdir() if p.is_dir()):
                for entry in sorted(fmt_dir.iterdir()):
                    if not entry.is_dir():
                        continue
                    if entry.name.startswith("rep"):
                        try:
                            rep = int(entry.name[3:])
                        except ValueError:
                            continue
                        runs = [p for p in sorted(entry.iterdir()) if is_run_dir(p)]
                        for run in runs:
                            records.append(
                                run_record(run, doc_dir.name, contract, fmt_dir.name, rep)
                            )
                        # A cell that was attempted but produced no scorable run
                        # leaves a status marker; surface it as a failed record so
                        # the synthesis table shows FAILED, not a silent gap.
                        if not runs:
                            failed = _failed_record(
                                entry, doc_dir.name, contract, fmt_dir.name, rep
                            )
                            if failed is not None:
                                records.append(failed)
                    elif is_run_dir(entry):
                        # Legacy layout without a repeat level: rep 1.
                        records.append(run_record(entry, doc_dir.name, contract, fmt_dir.name, 1))
    return records


def _failed_record(
    cell_dir: Path, doc: str, contract: str, fmt: str, rep: int
) -> dict[str, Any] | None:
    """A minimal record for a cell that was attempted but yielded no run.

    Reads the harness's ``cell_status.json`` marker; returns None when the cell
    simply was not run (no marker, no run dir)."""
    marker = cell_dir / "cell_status.json"
    if not marker.exists():
        return None
    try:
        status = json.loads(marker.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        status = {}
    return {
        "doc": doc,
        "contract": contract,
        "format": fmt,
        "rep": rep,
        "run": None,
        "status": "failed",
        "failure_reason": status.get("reason", "unknown"),
    }


def _mean(values: list[Any]) -> float | None:
    nums = [v for v in values if isinstance(v, int | float)]
    return round(mean(nums), 3) if nums else None


def _cell_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one (doc, contract, format) cell across repeats."""

    def f1_mean(key: str) -> float | None:
        return _mean([r.get(key, {}).get("f1") for r in runs if isinstance(r.get(key), dict)])

    return {
        "runs": len(runs),
        "node_f1_strict": f1_mean("node_strict"),
        "node_f1_relaxed": f1_mean("node_relaxed"),
        "edge_f1_strict": f1_mean("edge_strict"),
        "edge_f1_relaxed": f1_mean("edge_relaxed"),
        "completeness": _mean([r.get("completeness") for r in runs]),
        "time_s": _mean([r.get("time_s") for r in runs]),
        "retention_pct": _mean([r.get("retention_pct") for r in runs]),
        "chunk_coverage_pct": _mean([r.get("chunk_coverage_pct") for r in runs]),
        "empty_identity": sum(r.get("empty_identity") or 0 for r in runs),
        "orphans": sum(r.get("orphans") or 0 for r in runs),
    }


def render_markdown(records: list[dict[str, Any]]) -> str:
    """Render the synthesis report (mean over repeats per cell)."""
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in records:
        cells.setdefault((r["doc"], r["contract"], r["format"]), []).append(r)

    def fmt_num(v: Any, suffix: str = "") -> str:
        return f"{v}{suffix}" if v is not None else "—"

    lines = [
        "# Benchmark Synthesis Report",
        "",
        "Aggregated by `docs/examples/scripts/17_benchmark_aggregate.py`; one row per",
        "(document, contract, format) cell, numeric metrics are the mean over repeats.",
        "Micro F1 (strict, relaxed in parens) is reported separately for nodes and",
        "edges. `ret%` = dense merge retention, `cov%` = skeleton chunk coverage —",
        "quality numbers are only comparable between runs with similar retention.",
        "",
        "| Doc | Contract | Format | Runs | Node F1 | Edge F1 | Compl. | Time s | ret% | cov% | Empty-id | Orphans |",
        "| :-- | :-- | :-- | --: | :-- | :-- | --: | --: | --: | --: | --: | --: |",
    ]
    for (doc, contract, fmt), runs in sorted(cells.items()):
        scored = [r for r in runs if r.get("status") != "failed"]
        failed = [r for r in runs if r.get("status") == "failed"]
        if not scored:
            # Every attempt in this cell failed — show it explicitly.
            reason = failed[0].get("failure_reason", "failed") if failed else "failed"
            lines.append(
                f"| {doc} | {contract} | {fmt} | {len(failed)} | FAILED ({reason}) | — "
                f"| — | — | — | — | — | — |"
            )
            continue
        s = _cell_summary(scored)
        node = (
            f"{fmt_num(s['node_f1_strict'])} ({fmt_num(s['node_f1_relaxed'])})"
            if s["node_f1_strict"] is not None
            else "—"
        )
        edge = (
            f"{fmt_num(s['edge_f1_strict'])} ({fmt_num(s['edge_f1_relaxed'])})"
            if s["edge_f1_strict"] is not None
            else "—"
        )
        runs_label = str(s["runs"]) if not failed else f"{s['runs']}+{len(failed)}✗"
        lines.append(
            f"| {doc} | {contract} | {fmt} | {runs_label} | {node} | {edge} "
            f"| {fmt_num(s['completeness'])} | {fmt_num(s['time_s'])} "
            f"| {fmt_num(s['retention_pct'])} | {fmt_num(s['chunk_coverage_pct'])} "
            f"| {s['empty_identity']} | {s['orphans']} |"
        )
    lines += [
        "",
        "Per-run records: `_aggregate.json`; per-run class/edge breakdowns: each run",
        "directory's `eval.txt` / `eval.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--quiet", action="store_true", help="Skip printing the report")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"Benchmark root not found: {args.root}")
    records = discover_runs(args.root)
    (args.root / "_aggregate.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = render_markdown(records)
    (args.root / "synthesis_report.md").write_text(report, encoding="utf-8")
    if not args.quiet:
        print(report)
    print(f"Aggregated {len(records)} run(s) -> {args.root / '_aggregate.json'}")


if __name__ == "__main__":
    main()
