"""Docling document and markdown exporter."""

from typing import Optional
from pathlib import Path
from rich import print
import json


class DoclingExporter:
    """Export Docling documents and markdown to output directory."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize Docling exporter.
        
        Args:
            output_dir: Directory where outputs will be saved.
        """
        self.output_dir = output_dir or Path("outputs")
        
    def export_document(
        self, 
        document, 
        base_name: str,
        include_json: bool = True,
        include_markdown: bool = True,
        per_page: bool = False
    ) -> dict:
        """Export Docling document and markdown.
        
        Args:
            document: Docling Document object.
            base_name: Base name for output files (without extension).
            include_json: Whether to export document as JSON.
            include_markdown: Whether to export markdown.
            per_page: Whether to export per-page markdown files.
            
        Returns:
            Dictionary with paths to created files.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export document as JSON
        if include_json:
            json_path = self.output_dir / f"{base_name}_docling.json"
            self._export_document_json(document, json_path)
            exported_files['document_json'] = str(json_path)
            print(f"[green]→[/green] Saved Docling document to [green]{json_path}[/green]")
        
        # Export full markdown
        if include_markdown:
            md_path = self.output_dir / f"{base_name}_markdown.md"
            full_markdown = document.export_to_markdown()
            self._save_text(full_markdown, md_path)
            exported_files['markdown'] = str(md_path)
            print(f"[green]→[/green] Saved full markdown to [green]{md_path}[/green]")
        
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
            
            exported_files['page_markdowns'] = page_files
            print(f"[green]→[/green] Saved {len(page_files)} page markdown files to [green]{page_dir}[/green]")
        
        return exported_files
    
    def _export_document_json(self, document, output_path: Path) -> None:
        """Export Docling document to JSON format.
        
        Args:
            document: Docling Document object.
            output_path: Path where to save JSON file.
        """
        # Export using Docling's native export method
        doc_dict = document.export_to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_text(self, content: str, output_path: Path) -> None:
        """Save text content to file.
        
        Args:
            content: Text content to save.
            output_path: Path where to save file.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
