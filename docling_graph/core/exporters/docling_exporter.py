"""Docling document and markdown exporter."""

import json
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from ...logging_utils import get_component_logger
from ..utils.doclang_sanitizer import sanitize_for_doclang

logger = get_component_logger("DoclingExporter", __name__)


class DoclingExporter:
    """Export Docling documents and markdown to output directory."""

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize Docling exporter.

        Args:
            output_dir: Directory where outputs will be saved.
        """
        self.output_dir = output_dir or Path("outputs")

    def export_document(
        self,
        document: DoclingDocument,
        base_name: str,
        include_json: bool = True,
        include_markdown: bool = True,
        include_doclang: bool = True,
        per_page: bool = False,
    ) -> dict[str, str | list[str]]:
        """Export Docling document, markdown, and DocLang.

        Args:
            document: Docling Document object.
            base_name: Base name for output files (without extension).
            include_json: Whether to export document as JSON (canonical, lossless).
            include_markdown: Whether to export markdown (human-readable view).
            include_doclang: Whether to export DocLang (.dclg, content+geometry interchange).
            per_page: Whether to export per-page markdown files.

        Returns:
            Dictionary with paths to created files.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        exported_files: dict[str, str | list[str]] = {}

        # Export document as JSON
        if include_json:
            json_path = self.output_dir / f"{base_name}.json"
            self._export_document_json(document, json_path)
            exported_files["document_json"] = str(json_path)

        # Export full markdown
        if include_markdown:
            md_path = self.output_dir / f"{base_name}.md"
            full_markdown = document.export_to_markdown()
            self._save_text(full_markdown, md_path)
            exported_files["markdown"] = str(md_path)

        # Export DocLang (best-effort: never fail the pipeline on a serializer issue)
        if include_doclang:
            dclg_path = self._export_doclang(document, self.output_dir / f"{base_name}.dclg")
            if dclg_path is not None:
                exported_files["doclang"] = str(dclg_path)

        # Export per-page markdown
        if per_page:
            page_dir = self.output_dir / f"{base_name}_pages"
            page_dir.mkdir(parents=True, exist_ok=True)

            page_files = []
            for page_no in sorted(document.pages.keys()):
                page_md = document.export_to_markdown(page_no=page_no)
                page_path = page_dir / f"page_{page_no:03d}.md"
                self._save_text(page_md, page_path)
                page_files.append(str(page_path))

            exported_files["page_markdowns"] = page_files
            logger.info("Saved %s page markdown files to %s", len(page_files), page_dir)

        return exported_files

    def _export_doclang(self, document: DoclingDocument, output_path: Path) -> Path | None:
        """Serialize the document to DocLang, sanitizing control chars first.

        Returns the written path, or ``None`` if DocLang export is unavailable or
        fails — the export is best-effort and must never abort the pipeline (the
        JSON and markdown artifacts are unaffected).
        """
        if not hasattr(document, "export_to_doclang"):
            logger.warning(
                "DocLang export skipped: installed docling-core has no export_to_doclang()"
            )
            return None
        try:
            clean = sanitize_for_doclang(document)
            self._save_text(clean.export_to_doclang(), output_path)
            return output_path
        except Exception as e:
            logger.warning("DocLang export skipped: %s", e)
            return None

    def _export_document_json(self, document: DoclingDocument, output_path: Path) -> None:
        """Export Docling document to JSON format.

        Args:
            document: Docling Document object.
            output_path: Path where to save JSON file.
        """
        # Export using Docling's native export method
        doc_dict = document.export_to_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False, default=str)

    def _save_text(self, content: str, output_path: Path) -> None:
        """Save text content to file.

        Args:
            content: Text content to save.
            output_path: Path where to save file.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
