# Pfizer revenue across annual reports

This example builds a year-by-year knowledge graph from Pfizer Form 10-K PDFs in
[FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench). The Hugging Face
dataset holds FinanceBench's questions; its dataset card links to the
[FinanceBench PDF repository](https://github.com/patronus-ai/financebench/tree/main/pdfs),
which the downloader uses. No PDFs or generated data are committed.

The extraction template records a report for each year and separate revenue
category nodes for each year. This matters when merging reports: a category's
2015 revenue must not overwrite its 2016 revenue. The query reports categories
and totals, with percentage changes where adjacent years have the same category
name and unit. Inspect the cited report before treating an extracted number as
authoritative; revenue classifications can change between filings.

## Setup

From the repository root:

```bash
uv sync
ollama pull qwen3.8:27b-mlx
ollama serve
```

Ollama's default endpoint is `http://localhost:11434`. Set `OLLAMA_BASE_URL`
if yours differs. The model is large; conversion and extraction of a full 10-K
may take a while. A working Ollama server and enough memory for the model are
required. Docling conversion also downloads its own layout models on first use.
PDF conversion explicitly disables OCR and keeps Docling's layout and table
structure processing. Existing `.dclx` archives and yearly graphs made before
this setting was recorded are rebuilt on the next run.

## Start with 2015

```bash
uv run python examples/financial-analyst/build.py --year 2015
uv run python examples/financial-analyst/query.py
```

The first command downloads `PFIZER_2015_10K.pdf` into `data/pdfs/`, converts it
to `data/dclx/PFIZER_2015_10K.dclx`, extracts a yearly graph with Ollama, and
creates `data/graphs/pfizer_revenue_graph.json`. The pipeline reads the `.dclx`
archive and serializes DocLang for the LLM. Intermediate per-year graph exports
and provenance ledgers are in `data/graphs/<year>/`; full run output is in
`data/runs/`. The merge audit is `data/graphs/merge_report.json`.

## Extend to every available Pfizer 10-K

```bash
uv run python examples/financial-analyst/build.py --all
uv run python examples/financial-analyst/query.py
uv run python examples/financial-analyst/query.py --category biopharma
```

`--all` discovers the available annual PDFs from the linked FinanceBench PDF
repository. It reuses downloaded PDFs, `.dclx` archives and completed yearly
graphs, extracts missing years, and rebuilds the cumulative graph. To add just
one year, use `--year 2016` (or another available year). The query compares
only consecutive years with identical units; it does not rescale values or
assume differently named categories are the same business.

If PDF download is unavailable, place files named `PFIZER_<year>_10K.pdf` in
`examples/financial-analyst/data/pdfs/`, then run:

```bash
uv run python examples/financial-analyst/build.py --year 2015 --no-download
uv run python examples/financial-analyst/build.py --all --no-download
```

To download PDFs without conversion or extraction, use
`uv run python examples/financial-analyst/download.py --year 2015` or `--all`.
FinanceBench is licensed [CC BY-NC 4.0](https://huggingface.co/datasets/PatronusAI/financebench/blob/main/README.md).
