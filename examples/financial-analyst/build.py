"""Convert FinanceBench Pfizer PDFs to DocLang and build a cumulative graph."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

from docling_graph import PipelineConfig, merge_graphs, run_pipeline
from docling_graph.core.exporters.json_exporter import JSONExporter
from docling_graph.core.utils.doclang_sanitizer import sanitize_for_doclang

from download import DATA_DIR, PDF_DIR, available_reports, download_report
from template import PfizerAnnualReport

DCLX_DIR = DATA_DIR / "dclx"
GRAPHS_DIR = DATA_DIR / "graphs"
RUNS_DIR = DATA_DIR / "runs"
CONVERSION_SETTINGS = {"do_ocr": False, "do_table_structure": True}


def has_current_conversion_settings(marker: Path) -> bool:
    """Only reuse outputs created with the current PDF conversion settings."""
    try:
        return json.loads(marker.read_text(encoding="utf-8")) == CONVERSION_SETTINGS
    except (OSError, json.JSONDecodeError):
        return False


def write_conversion_settings(marker: Path) -> None:
    marker.write_text(json.dumps(CONVERSION_SETTINGS, indent=2) + "\n", encoding="utf-8")


def make_pdf_converter():
    """Use PDF text plus layout and table models, with OCR disabled."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def dclx_for(pdf: Path) -> Path:
    """Convert one PDF once; the archive is the extraction input on later runs."""
    DCLX_DIR.mkdir(parents=True, exist_ok=True)
    archive = DCLX_DIR / f"{pdf.stem}.dclx"
    marker = archive.with_suffix(".conversion.json")
    if (
        archive.is_file()
        and zipfile.is_zipfile(archive)
        and has_current_conversion_settings(marker)
    ):
        return archive
    print(f"Converting {pdf.name} to DocLang archive...")
    converter = make_pdf_converter()
    document = converter.convert(str(pdf)).document
    if document is None:
        raise RuntimeError(f"Docling did not produce a document for {pdf}")
    temporary = archive.with_name(f"{archive.stem}.part.dclx")
    try:
        sanitize_for_doclang(document).save_as_doclang_archive(temporary)
        temporary.replace(archive)
        write_conversion_settings(marker)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def extract_year(year: int, archive: Path, model: str) -> Path:
    """Extract a year and retain graph JSON alongside its provenance ledger."""
    destination = GRAPHS_DIR / str(year)
    graph_path = destination / "graph.json"
    marker = destination / "conversion.json"
    if graph_path.is_file() and has_current_conversion_settings(marker):
        print(f"Reusing {graph_path}")
        return graph_path
    print(f"Extracting {year} with Ollama {model}...")
    config = PipelineConfig(
        source=archive,
        template=PfizerAnnualReport,
        backend="llm",
        inference="local",
        provider_override="ollama",
        model_override=model,
        processing_mode="many-to-one",
        extraction_contract="dense",
        llm_input_format="doclang",
        use_chunking=True,
        chunk_max_tokens=1024,
        parallel_workers=1,
        provenance="standard",
        output_dir=RUNS_DIR,
        dump_to_disk=True,
        export_docling=False,
        export_docling_json=False,
        export_markdown=False,
        export_doclang=False,
    )
    context = run_pipeline(config)
    if context.knowledge_graph is None or context.knowledge_graph.number_of_nodes() == 0:
        raise RuntimeError(f"No graph extracted from {archive}")
    reports = [
        data
        for _, data in context.knowledge_graph.nodes(data=True)
        if data.get("label") == "PfizerAnnualReport"
    ]
    if len(reports) != 1 or reports[0].get("fiscal_year") != year:
        raise RuntimeError(f"Expected one Pfizer {year} report node; found {reports}")
    if context.output_manager is None:
        raise RuntimeError("Pipeline did not provide an output directory")
    source_dir = context.output_manager.get_docling_graph_dir()
    source_graph = source_dir / "graph.json"
    if not source_graph.is_file():
        raise RuntimeError(f"Pipeline graph export is missing: {source_graph}")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_graph, graph_path)
    provenance = source_dir / "provenance.json"
    if provenance.is_file():
        shutil.copyfile(provenance, destination / "provenance.json")
    write_conversion_settings(marker)
    print(f"Saved {graph_path}")
    return graph_path


def merge_years() -> Path:
    """Merge all completed years, including years processed in prior invocations."""
    graphs = sorted(GRAPHS_DIR.glob("*/graph.json"), key=lambda path: int(path.parent.name))
    if not graphs:
        raise RuntimeError("No yearly graphs are available to merge")
    merged, report = merge_graphs(graphs, template=PfizerAnnualReport)
    output = GRAPHS_DIR / "pfizer_revenue_graph.json"
    JSONExporter().export(merged, output)
    (GRAPHS_DIR / "merge_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    print(
        f"Merged {len(graphs)} years into {output} "
        f"({merged.number_of_nodes()} nodes, {merged.number_of_edges()} edges)"
    )
    if report.field_conflicts:
        print(f"Review {len(report.field_conflicts)} merge field conflicts in the per-year graphs")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--year", type=int, default=2015, help="Fiscal year (default: 2015)")
    selection.add_argument("--all", action="store_true", help="Add all available Pfizer 10-K years")
    parser.add_argument(
        "--no-download", action="store_true", help="Use PDFs already placed in data/pdfs"
    )
    parser.add_argument("--model", default="qwen3.8:27b-mlx", help="Ollama model tag")
    args = parser.parse_args()

    if args.no_download:
        reports = {}
        for path in PDF_DIR.glob("PFIZER_*_10K.pdf"):
            match = re.fullmatch(r"PFIZER_(\d{4})_10K\.pdf", path.name)
            if match:
                reports[int(match.group(1))] = path.name
        if not reports:
            raise SystemExit(f"No Pfizer PDFs found in {PDF_DIR}")
    else:
        reports = available_reports()
    years = sorted(reports) if args.all else [args.year]
    for year in years:
        if year not in reports:
            available = ", ".join(map(str, sorted(reports)))
            raise SystemExit(f"Pfizer {year} 10-K is unavailable; available years: {available}")
        pdf = PDF_DIR / reports[year] if args.no_download else download_report(reports[year])
        archive = dclx_for(pdf)
        extract_year(year, archive, args.model)
    merge_years()


if __name__ == "__main__":
    main()
