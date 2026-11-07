"""Download Pfizer annual reports from the PDF repository linked by FinanceBench."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).resolve().parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
REPO = "patronus-ai/financebench"
API_URL = f"https://api.github.com/repos/{REPO}/contents/pdfs"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/main/pdfs"
FILENAME_RE = re.compile(r"^PFIZER_(\d{4})_10K\.pdf$")


def available_reports() -> dict[int, str]:
    """List Pfizer 10-K PDFs in FinanceBench's linked PDF repository."""
    request = Request(
        API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "docling-graph-financial-analyst",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            entries = json.load(response)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Cannot list FinanceBench PDFs: {exc}") from exc
    if not isinstance(entries, list):
        raise RuntimeError("FinanceBench PDF listing was not a file list")
    reports = {}
    for entry in entries:
        name = entry.get("name", "")
        match = FILENAME_RE.fullmatch(name)
        if match:
            reports[int(match.group(1))] = name
    if not reports:
        raise RuntimeError("No Pfizer 10-K PDFs found in the FinanceBench PDF repository")
    return dict(sorted(reports.items()))


def download_report(name: str) -> Path:
    """Download one report atomically, or reuse an existing PDF."""
    if not FILENAME_RE.fullmatch(name):
        raise ValueError(f"Unexpected Pfizer report filename: {name}")
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    target = PDF_DIR / name
    if target.is_file():
        with target.open("rb") as existing:
            if existing.read(5) == b"%PDF-":
                return target
    temporary = target.with_suffix(".pdf.part")
    request = Request(
        f"{RAW_URL}/{quote(name)}",
        headers={"User-Agent": "docling-graph-financial-analyst"},
    )
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        with temporary.open("rb") as downloaded:
            if downloaded.read(5) != b"%PDF-":
                raise RuntimeError(f"Downloaded file is not a PDF: {name}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--year", type=int, default=2015, help="Fiscal year (default: 2015)")
    selection.add_argument("--all", action="store_true", help="All available Pfizer 10-K years")
    args = parser.parse_args()
    reports = available_reports()
    years = list(reports) if args.all else [args.year]
    for year in years:
        if year not in reports:
            available = ", ".join(map(str, reports))
            raise SystemExit(f"Pfizer {year} 10-K is unavailable; available years: {available}")
        print(download_report(reports[year]))


if __name__ == "__main__":
    main()
