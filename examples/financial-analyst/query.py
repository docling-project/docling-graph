"""Query year-over-year Pfizer revenue categories in the merged graph."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from docling_graph.core.importers.graph_json import load_graph_json

GRAPHS_DIR = Path(__file__).resolve().parent / "data" / "graphs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", help="Only show a category whose name contains this text")
    args = parser.parse_args()
    graph_path = GRAPHS_DIR / "pfizer_revenue_graph.json"
    if not graph_path.is_file():
        raise SystemExit(f"Build the graph first: {graph_path}")
    graph = load_graph_json(graph_path)
    observations = defaultdict(list)
    totals = []
    for report_id, report in graph.nodes(data=True):
        if report.get("label") != "PfizerAnnualReport":
            continue
        year = report.get("fiscal_year")
        if report.get("total_revenue") is not None:
            totals.append(
                (
                    int(year),
                    float(report["total_revenue"]),
                    report.get("total_revenue_unit") or "unit unknown",
                )
            )
        for _, category_id, edge in graph.out_edges(report_id, data=True):
            if edge.get("label") != "HAS_REVENUE_CATEGORY":
                continue
            category = graph.nodes[category_id]
            name = category.get("name")
            amount = category.get("amount")
            if not name or amount is None or category.get("fiscal_year") != year:
                continue
            if args.category and args.category.casefold() not in name.casefold():
                continue
            observations[name.casefold()].append(
                (
                    int(year),
                    name,
                    float(amount),
                    category.get("unit") or "unit unknown",
                    category_id,
                    category.get("evidence"),
                )
            )

    if not observations:
        raise SystemExit(
            "No year-scoped revenue categories found; inspect the yearly graph exports"
        )
    print("Pfizer revenue by reported category")
    print(
        "Amounts are shown in the units printed in each report. "
        "Category names may change across years.\n"
    )
    if totals:
        print("Consolidated total")
        previous_total = None
        for year, amount, unit in sorted(totals):
            change = ""
            if (
                previous_total
                and previous_total[0] == year - 1
                and previous_total[2] == unit
                and previous_total[1] != 0
            ):
                pct = (amount / previous_total[1] - 1) * 100
                change = f"  ({pct:+.1f}% vs {previous_total[0]})"
            print(f"  {year}: {amount:,.2f} {unit}{change}")
            previous_total = (year, amount, unit)
        print()
    for rows in sorted(observations.values(), key=lambda rows: rows[0][1].casefold()):
        previous = None
        print(rows[0][1])
        for year, name, amount, unit, node_id, evidence in sorted(rows):
            change = ""
            if previous and previous[0] == year - 1 and previous[2] == unit and previous[1] != 0:
                change = f"  ({(amount / previous[1] - 1) * 100:+.1f}% vs {previous[0]})"
            print(
                f"  {year}: {amount:,.2f} {unit}{change}  "
                f"[PFIZER_{year}_10K.pdf, node {node_id}]"
            )
            if evidence:
                print(f"    Evidence noted by model: {evidence}")
            previous = (year, amount, unit)


if __name__ == "__main__":
    main()
