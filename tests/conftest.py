"""
Shared pytest fixtures and configuration for all tests.

This conftest is at the root of the tests/ directory and provides
fixtures accessible to all test modules.
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ==================== Configuration Fixtures ====================


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "defaults": {
            "processing_mode": "many-to-one",
            "backend": "llm",
            "inference": "local",
            "export_format": "csv",
        },
        "docling": {
            "pipeline": "ocr",
            "export": {
                "docling_json": True,
                "markdown": True,
                "per_page_markdown": False,
            },
        },
        "models": {
            "vlm": {
                "local": {
                    "default_model": "numind/NuExtract-2.0-2B",
                    "provider": "docling",
                }
            },
            "llm": {
                "local": {
                    "default_model": "llama-3.1-8b",
                    "provider": "ollama",
                },
                "remote": {
                    "default_model": "mistral-small-latest",
                    "provider": "mistral",
                },
            },
        },
        "output": {
            "default_directory": "outputs",
            "create_visualizations": True,
            "create_markdown": True,
        },
    }


@pytest.fixture
def sample_config_vlm() -> Dict[str, Any]:
    """Sample VLM configuration."""
    return {
        "defaults": {
            "processing_mode": "one-to-one",
            "backend": "vlm",
            "inference": "local",
            "export_format": "json",
        },
        "docling": {
            "pipeline": "vision",
            "export": {
                "docling_json": True,
                "markdown": False,
                "per_page_markdown": False,
            },
        },
        "models": {
            "vlm": {
                "local": {
                    "default_model": "numind/NuExtract-2.0-2B",
                    "provider": "docling",
                }
            },
        },
        "output": {
            "default_directory": "outputs",
            "create_visualizations": False,
            "create_markdown": False,
        },
    }


@pytest.fixture
def sample_config_remote() -> Dict[str, Any]:
    """Sample remote inference configuration."""
    return {
        "defaults": {
            "processing_mode": "many-to-one",
            "backend": "llm",
            "inference": "remote",
            "export_format": "csv",
        },
        "docling": {
            "pipeline": "ocr",
            "export": {
                "docling_json": True,
                "markdown": True,
                "per_page_markdown": False,
            },
        },
        "models": {
            "llm": {
                "remote": {
                    "default_model": "mistral-small-latest",
                    "provider": "mistral",
                },
            },
        },
        "output": {
            "default_directory": "outputs",
            "create_visualizations": True,
            "create_markdown": True,
        },
    }


@pytest.fixture
def config_file(tmp_path: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a temporary config.yaml file."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path


# ==================== File and Path Fixtures ====================


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal PDF-like test file."""
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\ntest content")
    return pdf_file


@pytest.fixture
def sample_jpg(tmp_path: Path) -> Path:
    """Create a minimal JPEG-like test file."""
    jpg_file = tmp_path / "sample.jpg"
    # Minimal JPEG header
    jpg_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    return jpg_file


@pytest.fixture
def sample_png(tmp_path: Path) -> Path:
    """Create a minimal PNG-like test file."""
    png_file = tmp_path / "sample.png"
    # PNG signature
    png_file.write_bytes(b"\x89PNG\r\n\x1a\n")
    return png_file


# ==================== Graph Data Fixtures ====================


@pytest.fixture
def sample_csv_graph(tmp_path: Path) -> Path:
    """Create sample CSV graph files (nodes and edges)."""
    graph_dir = tmp_path / "graph_output"
    graph_dir.mkdir()

    # Create nodes.csv
    nodes_file = graph_dir / "nodes.csv"
    nodes_file.write_text(
        "id,label,type\n"
        "node_1,Invoice_12345,Invoice\n"
        "node_2,Amount_5000,Amount\n"
        "node_3,Vendor_ABC,Vendor\n"
    )

    # Create edges.csv
    edges_file = graph_dir / "edges.csv"
    edges_file.write_text("source,target,label\nnode_1,node_2,contains\nnode_1,node_3,issued_by\n")

    return graph_dir


@pytest.fixture
def sample_json_graph(tmp_path: Path) -> Path:
    """Create a sample JSON graph file."""
    graph_file = tmp_path / "graph.json"
    graph_content = {
        "nodes": [
            {"id": "node_1", "label": "Invoice_12345", "type": "Invoice"},
            {"id": "node_2", "label": "Amount_5000", "type": "Amount"},
        ],
        "edges": [{"source": "node_1", "target": "node_2", "label": "contains"}],
    }
    import json

    graph_file.write_text(json.dumps(graph_content, indent=2))
    return graph_file


# ==================== Mock Fixtures ====================


@pytest.fixture
def mock_run_pipeline() -> MagicMock:
    """Mock for docling_graph.pipeline.run_pipeline."""
    return MagicMock()


@pytest.fixture
def mock_interactive_visualizer() -> MagicMock:
    """Mock for InteractiveVisualizer class."""
    return MagicMock()


@pytest.fixture
def mock_config_builder() -> MagicMock:
    """Mock for config builder."""
    mock = MagicMock()
    mock.return_value = {
        "defaults": {"backend": "llm", "inference": "local"},
        "docling": {"pipeline": "ocr"},
    }
    return mock


@pytest.fixture
def mock_docling_pipeline() -> MagicMock:
    """Mock for Docling pipeline."""
    mock = MagicMock()
    mock.convert_single.return_value = MagicMock(pages=[MagicMock(text="Sample document text")])
    return mock


# ==================== Temporary Directory Fixtures ====================


@pytest.fixture
def working_dir(tmp_path: Path, monkeypatch) -> Path:
    """
    Create a temporary working directory and change to it.

    This is useful for tests that interact with the filesystem
    and don't want to clutter the actual project directory.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


# ==================== Hook for Test Markers ====================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")


# ==================== Cleanup and Utilities ====================


@pytest.fixture(autouse=True)
def reset_imports():
    """Reset module imports between tests to avoid state pollution."""
    yield
    # Cleanup code here if needed (runs after test)


@pytest.fixture
def mock_rich_print(monkeypatch):
    """Mock rich.print to avoid output during tests."""
    mock = MagicMock()
    monkeypatch.setattr("rich.print", mock)
    return mock


# ==================== Graph Fixtures ====================


@pytest.fixture
def sample_networkx_graph():
    """Create a sample NetworkX directed graph for testing."""
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node("node_1", label="Person", type="entity")
    graph.add_node("node_2", label="Company", type="entity")
    graph.add_node("node_3", label="Date", type="value")

    graph.add_edge("node_1", "node_2", label="works_for")
    graph.add_edge("node_2", "node_3", label="founded")

    return graph


@pytest.fixture
def pydantic_models():
    """Create sample Pydantic models for testing."""
    from typing import Optional

    from pydantic import BaseModel, ConfigDict, Field

    class Address(BaseModel):
        """Address model."""

        street: str
        city: str
        country: str

    class Person(BaseModel):
        """Person model."""

        model_config = ConfigDict(is_entity=True)

        id: str = Field(..., json_schema_extra={"graph_id_fields": ["id"]})
        name: str
        email: Optional[str] = None
        address: Optional[Address] = None

    class Company(BaseModel):
        """Company model."""

        model_config = ConfigDict(is_entity=True)

        id: str = Field(..., json_schema_extra={"graph_id_fields": ["id"]})
        name: str
        industry: str

    return {"Person": Person, "Company": Company, "Address": Address}


# ==================== Protocol/Backend Fixtures ====================


@pytest.fixture
def mock_vlm_backend():
    """Create a mock VLM backend."""
    from unittest.mock import MagicMock

    backend = MagicMock()
    backend.extract_from_document = MagicMock(return_value=[MagicMock()])
    backend.cleanup = MagicMock()
    return backend


@pytest.fixture
def mock_llm_backend():
    """Create a mock LLM backend."""
    from unittest.mock import MagicMock

    backend = MagicMock()
    backend.extract_from_markdown = MagicMock(return_value=MagicMock())
    backend.client = MagicMock()
    backend.cleanup = MagicMock()
    return backend


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.context_limit = 4096
    client.get_json_response = MagicMock(return_value={"result": "success"})
    return client


# ==================== Converter Fixtures ====================


@pytest.fixture
def sample_pydantic_models_for_conversion():
    """Create sample Pydantic models for graph conversion testing."""
    from typing import List, Optional

    from pydantic import BaseModel, ConfigDict, Field

    class AddressModel(BaseModel):
        """Address model."""

        model_config = ConfigDict(is_entity=False)
        street: str
        city: str

    class PersonModel(BaseModel):
        """Person model."""

        model_config = ConfigDict(is_entity=True)
        name: str = Field(..., json_schema_extra={"graph_id_fields": ["name"]})
        email: Optional[str] = None
        address: Optional[AddressModel] = None

    class CompanyModel(BaseModel):
        """Company model."""

        model_config = ConfigDict(is_entity=True)
        name: str = Field(..., json_schema_extra={"graph_id_fields": ["name"]})
        industry: str

    return {"Address": AddressModel, "Person": PersonModel, "Company": CompanyModel}


@pytest.fixture
def graph_for_visualization():
    """Create a graph for visualization testing."""
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node("node_1", label="Person", name="John Doe", type="entity")
    graph.add_node("node_2", label="Company", name="ACME Corp", type="entity")
    graph.add_node("node_3", label="Location", name="New York", type="value")

    graph.add_edge("node_1", "node_2", label="works_for")
    graph.add_edge("node_2", "node_3", label="located_in")

    return graph


# ==================== Extractor Fixtures ====================


@pytest.fixture
def mock_extractor_backend():
    """Create a mock extractor backend."""
    from unittest.mock import MagicMock

    backend = MagicMock()
    backend.extract_from_document = MagicMock(return_value=[MagicMock()])
    backend.extract_from_markdown = MagicMock(return_value=MagicMock())
    backend.cleanup = MagicMock()
    return backend


@pytest.fixture
def mock_document_processor():
    """Create a mock document processor."""
    from unittest.mock import MagicMock

    processor = MagicMock()
    processor.process_document = MagicMock(return_value=["Page 1 content", "Page 2 content"])
    processor.convert_to_markdown = MagicMock(return_value=MagicMock())
    processor.extract_full_markdown = MagicMock(return_value="# Full Document\n\nContent")
    processor.extract_page_markdowns = MagicMock(return_value=["# Page 1", "# Page 2"])
    return processor


@pytest.fixture
def sample_extraction_models():
    """Create sample Pydantic models for extraction testing."""
    from pydantic import BaseModel

    class InvoiceModel(BaseModel):
        """Invoice extraction model."""

        invoice_number: str
        total_amount: float
        date: str

    class DocumentModel(BaseModel):
        """Document extraction model."""

        title: str
        content: str

    return {"Invoice": InvoiceModel, "Document": DocumentModel}


# ==================== LLM Client Fixtures ====================


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.model = "test-model"
    client.context_limit = 4096
    client.get_json_response = MagicMock(return_value={"result": "test"})
    return client


@pytest.fixture
def llm_prompt_examples():
    """Provide example prompts for LLM testing."""
    return {
        "string_prompt": "Extract the data from this document",
        "dict_prompt": {
            "system": "You are a JSON extractor for documents",
            "user": "Extract all information from this invoice",
        },
        "empty_prompt": "",
        "empty_dict_prompt": {"system": "", "user": ""},
    }


@pytest.fixture
def llm_response_examples():
    """Provide example LLM responses."""
    return {
        "valid_json": '{"invoice_number": "INV-001", "amount": 100.00}',
        "invalid_json": "not valid json",
        "empty_json": "{}",
        "null_values": '{"key1": null, "key2": null}',
        "empty_object": "{}",
    }
