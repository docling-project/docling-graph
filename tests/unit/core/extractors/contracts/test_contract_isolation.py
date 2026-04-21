from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
CONTRACTS_ROOT = REPO_ROOT / "docling_graph" / "core" / "extractors" / "contracts"


def _python_files(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*.py") if p.is_file()]
