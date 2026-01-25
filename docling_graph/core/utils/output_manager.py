"""
Output directory manager for unified directory structure.

This module provides utilities for managing output directories and
sanitizing filenames according to the file organization plan.
"""

import re
from datetime import datetime
from pathlib import Path

# Maximum filename length (safe for Windows 260 char path limit)
MAX_FILENAME_LENGTH = 180


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename with fixed rules.

    Rules:
    - Replace dots with underscores (including extension dot)
    - Replace special chars with underscores: /\\:*?"<>|[](){}
    - Replace spaces with underscores
    - Do NOT remove consecutive underscores (they indicate removed special chars)
    - Truncate to 180 chars (reserve 17 for _YYYYMMDD_HHMMSS)
    - Add timestamp suffix

    Args:
        filename: Original filename to sanitize

    Returns:
        Sanitized filename with timestamp

    Examples:
        >>> sanitize_filename("invoice.pdf")
        "invoice_pdf_20260125_073500"
        >>> sanitize_filename("My Document (2024).pdf")
        "My_Document_2024__pdf_20260125_073500"
    """
    # Replace dots with underscores (including extension)
    safe = filename.replace(".", "_")

    # Replace special characters and spaces
    safe = re.sub(r'[/\\:*?"<>|\[\](){}]', "_", safe)
    safe = safe.replace(" ", "_")

    # Do NOT remove consecutive underscores - they indicate removed special chars

    # Strip leading/trailing underscores
    safe = safe.strip("_")

    # Truncate if needed (reserve 17 chars for timestamp: _YYYYMMDD_HHMMSS)
    max_base = MAX_FILENAME_LENGTH - 17
    if len(safe) > max_base:
        safe = safe[:max_base]

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe}_{timestamp}"


class OutputDirectoryManager:
    """
    Manages output directory structure for pipeline exports.

    This class provides a unified interface for creating and accessing
    output directories according to the file organization plan.
    """

    def __init__(self, base_output_dir: Path, source_filename: str) -> None:
        """
        Initialize output directory manager.

        Args:
            base_output_dir: Base output directory (e.g., "outputs")
            source_filename: Source document filename for directory naming
        """
        self.base_output_dir = Path(base_output_dir)
        self.source_filename = source_filename
        self.document_dir = self._create_document_directory(source_filename)

    def _create_document_directory(self, source_filename: str) -> Path:
        """
        Create sanitized document directory.

        Args:
            source_filename: Source document filename

        Returns:
            Path to created document directory
        """
        sanitized_name = sanitize_filename(source_filename)
        doc_dir = self.base_output_dir / sanitized_name
        doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir

    # Trace directories
    def get_trace_dir(self) -> Path:
        """Get trace/ directory."""
        path = self.document_dir / "trace"
        path.mkdir(exist_ok=True)
        return path

    def get_page_dir(self, page_num: int) -> Path:
        """Get trace/pages/page_NNN/ directory."""
        path = self.get_trace_dir() / "pages" / f"page_{page_num:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_chunks_dir(self) -> Path:
        """Get trace/chunks/ directory."""
        path = self.get_trace_dir() / "chunks"
        path.mkdir(exist_ok=True)
        return path

    def get_extractions_dir(self) -> Path:
        """Get trace/extractions/ directory."""
        path = self.get_trace_dir() / "extractions"
        path.mkdir(exist_ok=True)
        return path

    # Graph directories
    def get_graphs_dir(self) -> Path:
        """Get graphs/ directory."""
        path = self.document_dir / "graphs"
        path.mkdir(exist_ok=True)
        return path

    def get_per_chunk_graph_dir(self, chunk_id: int) -> Path:
        """Get graphs/per_chunk/chunk_NNN/ directory."""
        path = self.get_graphs_dir() / "per_chunk" / f"chunk_{chunk_id:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_per_page_graph_dir(self, page_num: int) -> Path:
        """Get graphs/per_page/page_NNN/ directory."""
        path = self.get_graphs_dir() / "per_page" / f"page_{page_num:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_consolidated_graph_dir(self) -> Path:
        """Get consolidated_graph/ directory (directly under document dir)."""
        path = self.document_dir / "consolidated_graph"
        path.mkdir(exist_ok=True)
        return path

    # Other directories
    def get_visualizations_dir(self) -> Path:
        """Get visualizations/ directory."""
        path = self.document_dir / "visualizations"
        path.mkdir(exist_ok=True)
        return path

    def get_docling_dir(self) -> Path:
        """Get docling/ directory."""
        path = self.document_dir / "docling"
        path.mkdir(exist_ok=True)
        return path

    def save_metadata(self, metadata: dict) -> Path:
        """
        Save metadata.json to document directory.

        Args:
            metadata: Metadata dictionary to save

        Returns:
            Path to saved metadata file
        """
        import json

        metadata_path = self.document_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        return metadata_path

    def get_document_dir(self) -> Path:
        """Get the main document directory."""
        return self.document_dir

    def get_per_page_dir(self) -> Path:
        """Get trace/per_page/ directory (convenience method for tests)."""
        path = self.get_trace_dir() / "per_page"
        path.mkdir(exist_ok=True)
        return path

    def get_per_chunk_dir(self) -> Path:
        """Get trace/per_chunk/ directory (convenience method for tests)."""
        path = self.get_trace_dir() / "per_chunk"
        path.mkdir(exist_ok=True)
        return path

# Made with Bob
