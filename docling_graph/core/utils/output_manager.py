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
        # Generate timestamp once at initialization to ensure consistency
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.document_dir = self._create_document_directory(source_filename)

    def _create_document_directory(self, source_filename: str) -> Path:
        """
        Create sanitized document directory using stored timestamp.

        Args:
            source_filename: Source document filename

        Returns:
            Path to created document directory
        """
        # Sanitize filename without timestamp
        safe = source_filename.replace(".", "_")
        safe = re.sub(r'[/\\:*?"<>|\[\](){}]', "_", safe)
        safe = safe.replace(" ", "_")
        safe = safe.strip("_")

        # Truncate if needed (reserve 17 chars for timestamp)
        max_base = MAX_FILENAME_LENGTH - 17
        if len(safe) > max_base:
            safe = safe[:max_base]

        # Use stored timestamp for consistency
        sanitized_name = f"{safe}_{self.timestamp}"
        doc_dir = self.base_output_dir / sanitized_name
        doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir

    # Main output directories
    def get_docling_dir(self) -> Path:
        """Get docling/ directory for Docling conversion outputs."""
        path = self.document_dir / "docling"
        path.mkdir(exist_ok=True)
        return path

    def get_docling_graph_dir(self) -> Path:
        """Get docling_graph/ directory for graph outputs."""
        path = self.document_dir / "docling_graph"
        path.mkdir(exist_ok=True)
        return path

    def get_debug_dir(self) -> Path:
        """Get debug/ directory for debug artifacts."""
        path = self.document_dir / "debug"
        path.mkdir(exist_ok=True)
        return path

    def get_per_page_dir(self) -> Path:
        """Get debug/per_page/ directory for per-page debug outputs."""
        path = self.get_debug_dir() / "per_page"
        path.mkdir(exist_ok=True)
        return path

    def get_per_chunk_dir(self) -> Path:
        """Get debug/per_chunk/ directory for per-chunk debug outputs."""
        path = self.get_debug_dir() / "per_chunk"
        path.mkdir(exist_ok=True)
        return path

    # Debug subdirectories
    def get_slots_text_dir(self) -> Path:
        """Get debug/slots_text/ directory for slot text files."""
        path = self.get_debug_dir() / "slots_text"
        path.mkdir(exist_ok=True)
        return path

    def get_atoms_dir(self) -> Path:
        """Get debug/atoms/ directory for atom extraction attempts."""
        path = self.get_debug_dir() / "atoms"
        path.mkdir(exist_ok=True)
        return path

    def get_field_catalog_selected_dir(self) -> Path:
        """Get debug/field_catalog_selected/ directory for per-slot field catalogs."""
        path = self.get_debug_dir() / "field_catalog_selected"
        path.mkdir(exist_ok=True)
        return path

    def get_arbitration_dir(self) -> Path:
        """Get debug/arbitration/ directory for arbitration requests/responses."""
        path = self.get_debug_dir() / "arbitration"
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

    def is_directory_empty(self) -> bool:
        """
        Check if the document directory is empty (no files, only empty subdirs allowed).

        Returns:
            True if directory contains no files, False otherwise
        """
        for item in self.document_dir.rglob("*"):
            if item.is_file():
                return False
        return True

    def cleanup_if_empty(self) -> bool:
        """
        Remove the document directory if it's empty.

        Returns:
            True if directory was removed, False if kept (contains files)
        """
        if self.is_directory_empty():
            import shutil

            shutil.rmtree(self.document_dir)
            return True
        return False
