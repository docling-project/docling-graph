from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
CONTRACTS_ROOT = REPO_ROOT / "docling_graph" / "core" / "extractors" / "contracts"


def _python_files(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*.py") if p.is_file()]


def test_delta_contract_does_not_import_staged_contract() -> None:
    delta_root = CONTRACTS_ROOT / "delta"
    offenders: list[str] = []
    for py_file in _python_files(delta_root):
        text = py_file.read_text(encoding="utf-8")
        if "contracts.staged" in text or "..staged." in text:
            offenders.append(str(py_file.relative_to(REPO_ROOT)))
    assert offenders == []


def test_staged_contract_does_not_import_delta_contract() -> None:
    staged_root = CONTRACTS_ROOT / "staged"
    offenders: list[str] = []
    for py_file in _python_files(staged_root):
        text = py_file.read_text(encoding="utf-8")
        if "contracts.delta" in text or "..delta." in text:
            offenders.append(str(py_file.relative_to(REPO_ROOT)))
    assert offenders == []
