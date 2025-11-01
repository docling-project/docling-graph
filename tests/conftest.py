"""
Pytest configuration and shared fixtures.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, Mock

import networkx as nx
import pytest
from pydantic import BaseModel, Field

# ============================================================================
# FIXTURES: Directories and Paths
# ============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures" / "sample_documents"


@pytest.fixture
def test_config_dir() -> Path:
    """Path to test config directory."""
    return Path(__file__).parent / "fixtures" / "sample_configs"


# ============================================================================
# FIXTURES: Sample Pydantic Models
# ============================================================================


class Person(BaseModel):
    """Sample Person model for testing."""

    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    email: str = Field(description="Person's email")

    model_config = {"graph_id_fields": ["email"]}


class Address(BaseModel):
    """Sample Address model for testing."""

    street: str
    city: str
    country: str


class Company(BaseModel):
    """Sample Company model with nested relationships."""

    name: str
    employees: list[Person] = Field(default_factory=list)
    address: Address | None = None

    model_config = {"graph_id_fields": ["name"]}


@pytest.fixture
def sample_person() -> Person:
    """Create a sample Person instance."""
    return Person(name="John Doe", age=30, email="john.doe@example.com")


@pytest.fixture
def sample_person_list() -> list[Person]:
    """Create a list of Person instances."""
    return [
        Person(name="Alice", age=25, email="alice@example.com"),
        Person(name="Bob", age=35, email="bob@example.com"),
        Person(name="Charlie", age=28, email="charlie@example.com"),
    ]


@pytest.fixture
def sample_company() -> Company:
    """Create a sample Company with nested relationships."""
    return Company(
        name="TechCorp",
        employees=[
            Person(name="Alice", age=25, email="alice@example.com"),
            Person(name="Bob", age=35, email="bob@example.com"),
        ],
        address=Address(street="123 Main St", city="San Francisco", country="USA"),
    )


# ============================================================================
# FIXTURES: Configuration
# ============================================================================


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Sample configuration dictionary."""
    return {
        "defaults": {
            "processing_mode": "many-to-one",
            "backend_type": "llm",
            "inference": "local",
            "export_format": "csv",
        },
        "docling": {"pipeline": "ocr"},
        "models": {
            "vlm": {"local": {"default_model": "numind/NuExtract-2.0-8B", "provider": "docling"}},
            "llm": {
                "local": {"default_model": "llama3:8b-instruct", "provider": "ollama"},
                "remote": {"default_model": "mistral-small-latest", "provider": "mistral"},
            },
        },
        "output": {
            "default_directory": "outputs",
            "create_visualizations": True,
            "create_markdown": True,
        },
    }


@pytest.fixture
def config_file(temp_dir: Path, sample_config_dict: Dict[str, Any]) -> Path:
    """Create a temporary config.yaml file."""
    import yaml

    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path


# ============================================================================
# FIXTURES: NetworkX Graphs
# ============================================================================


@pytest.fixture
def simple_graph() -> nx.DiGraph:
    """Create a simple directed graph for testing."""
    graph = nx.DiGraph()
    graph.add_node("person_1", label="Person", name="Alice", age=25)
    graph.add_node("person_2", label="Person", name="Bob", age=30)
    graph.add_node("company_1", label="Company", name="TechCorp")
    graph.add_edge("company_1", "person_1", label="employs")
    graph.add_edge("company_1", "person_2", label="employs")
    return graph


@pytest.fixture
def complex_graph() -> nx.DiGraph:
    """Create a complex graph with multiple node types and relationships."""
    graph = nx.DiGraph()

    # Add persons
    for i in range(5):
        graph.add_node(f"person_{i}", label="Person", name=f"Person{i}", age=25 + i)

    # Add companies
    for i in range(2):
        graph.add_node(f"company_{i}", label="Company", name=f"Company{i}")

    # Add edges
    graph.add_edge("company_0", "person_0", label="employs")
    graph.add_edge("company_0", "person_1", label="employs")
    graph.add_edge("company_1", "person_2", label="employs")
    graph.add_edge("person_0", "person_1", label="knows")

    return graph


# ============================================================================
# FIXTURES: Mock Objects
# ============================================================================


@pytest.fixture
def mock_llm_client():
    """Mock LLM/VLM client with proper return types."""
    mock = Mock()
    mock.client = Mock()
    mock.client.context_limit = 8000

    # For VLM-style extraction (returns list of models)
    mock.extract_from_document = Mock(
        return_value=[Person(name="Test Person", age=30, email="test@example.com")]
    )

    # For LLM-style extraction (returns single model)
    mock.extract_from_markdown = Mock(
        return_value=Person(name="Test Person", age=30, email="test@example.com")
    )

    return mock


@pytest.fixture
def mock_vlm_backend() -> Mock:
    """Mock VLM backend."""
    mock = Mock()
    mock.extract.return_value = [{"field": "value"}]
    return mock


@pytest.fixture
def mock_document_processor() -> Mock:
    """Mock document processor."""
    mock = Mock()
    mock.process_document.return_value = ["page 1 content", "page 2 content"]
    mock.cleanup = Mock()
    return mock


# ============================================================================
# FIXTURES: File Paths
# ============================================================================


@pytest.fixture
def sample_pdf_path(test_data_dir: Path) -> Path:
    """Path to sample PDF (will be created if doesn't exist)."""
    pdf_path = test_data_dir / "sample.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if not pdf_path.exists():
        # Create a minimal PDF for testing
        pdf_path.write_bytes(b"%PDF-1.4\nSample PDF content")
    return pdf_path


@pytest.fixture
def sample_image_path(test_data_dir: Path) -> Path:
    """Path to sample image (will be created if doesn't exist)."""
    img_path = test_data_dir / "sample.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    if not img_path.exists():
        # Create a minimal PNG for testing
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return img_path


# ============================================================================
# FIXTURES: Graph Converter
# ============================================================================


@pytest.fixture
def graph_converter():
    """Create a GraphConverter instance."""
    from docling_graph.core import GraphConfig, GraphConverter

    config = GraphConfig()
    return GraphConverter(config=config)


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "requires_api: mark test as requiring API access")
