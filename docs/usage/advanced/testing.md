# Testing


## Overview

Test Pydantic templates, custom backends, and pipeline configurations to ensure reliable extraction and graph generation.

**What You'll Learn:**
- Template validation testing
- Mock backends for testing
- Integration testing
- CI/CD integration
- Test fixtures
- Best practices

**Prerequisites:**
- Understanding of [Schema Definition](../../fundamentals/schema-definition/index.md)
- Familiarity with [Python API](../api/index.md)
- Basic pytest knowledge

---

## Setup

### Install Test Dependencies

```bash
# Install with test dependencies
uv sync --extra dev

# Or install pytest separately
uv add --dev pytest pytest-cov pytest-mock
```

### Project Structure

```
my_project/
├── templates/
│   └── my_template.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── test_templates.py        # Template tests
│   ├── test_extraction.py       # Extraction tests
│   └── test_integration.py      # End-to-end tests
├── pyproject.toml
└── pytest.ini
```

---

## Template Testing

### Basic Template Validation

```python
"""Test Pydantic template validation."""

import pytest
from pydantic import ValidationError
from templates.my_template import Person, Organization

def test_person_valid():
    """Test valid person creation."""
    person = Person(
        name="John Doe",
        age=30,
        email="john@example.com"
    )
    
    assert person.name == "John Doe"
    assert person.age == 30
    assert person.email == "john@example.com"

def test_person_invalid_age():
    """Test person with invalid age."""
    with pytest.raises(ValidationError) as exc_info:
        Person(
            name="John Doe",
            age=-5,  # Invalid
            email="john@example.com"
        )
    
    errors = exc_info.value.errors()
    assert any(e['loc'] == ('age',) for e in errors)

def test_person_invalid_email():
    """Test person with invalid email."""
    with pytest.raises(ValidationError):
        Person(
            name="John Doe",
            age=30,
            email="not-an-email"  # Invalid
        )

def test_person_optional_fields():
    """Test person with optional fields."""
    person = Person(
        name="John Doe",
        age=30
        # email is optional
    )
    
    assert person.email is None
```

### Test Field Validators

```python
"""Test custom field validators."""

from pydantic import BaseModel, Field, field_validator

class EmailTemplate(BaseModel):
    """Template with email validation."""
    
    email: str = Field(..., description="Email address")
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        """Validate email format."""
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower()

def test_email_validator_valid():
    """Test valid email."""
    template = EmailTemplate(email="John@Example.com")
    assert template.email == "john@example.com"  # Lowercased

def test_email_validator_invalid():
    """Test invalid email."""
    with pytest.raises(ValidationError) as exc_info:
        EmailTemplate(email="not-an-email")
    
    errors = exc_info.value.errors()
    assert "Invalid email format" in str(errors)
```

### Test Relationships

```python
"""Test entity relationships."""

from pydantic import BaseModel, Field, ConfigDict

def edge(label: str, **kwargs):
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

class Address(BaseModel):
    model_config = ConfigDict(is_entity=False)
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address = edge(label="LIVES_AT")

def test_relationship_structure():
    """Test relationship is properly defined."""
    person = Person(
        name="John",
        address=Address(street="123 Main St", city="NYC")
    )
    
    assert person.name == "John"
    assert person.address.street == "123 Main St"
    assert person.address.city == "NYC"

def test_relationship_metadata():
    """Test edge metadata is present."""
    field_info = Person.model_fields["address"]
    assert field_info.json_schema_extra is not None
    assert field_info.json_schema_extra.get("edge_label") == "LIVES_AT"
```

---

## Mock Backends

### Create Mock Backend

```python
"""Mock backend for testing."""

from typing import List, Type
from pydantic import BaseModel

class MockLLMBackend:
    """Mock LLM backend for testing."""
    
    def __init__(self, mock_response: dict | None = None):
        self.mock_response = mock_response or {}
        self.call_count = 0
        self.last_markdown = None
        self.last_template = None
    
    def extract_from_markdown(
        self,
        markdown: str,
        template: Type[BaseModel],
        context: str = "document",
        is_partial: bool = False
    ) -> BaseModel | None:
        """Mock extraction."""
        self.call_count += 1
        self.last_markdown = markdown
        self.last_template = template
        
        # Return mock response
        if self.mock_response:
            return template.model_validate(self.mock_response)
        
        return None
    
    def consolidate_from_pydantic_models(
        self,
        raw_models: List[BaseModel],
        programmatic_model: BaseModel,
        template: Type[BaseModel]
    ) -> BaseModel | None:
        """Mock consolidation."""
        return programmatic_model
    
    def cleanup(self) -> None:
        """Mock cleanup."""
        pass
```

### Use Mock Backend

```python
"""Test extraction with mock backend."""

import pytest
from templates.my_template import Person

def test_extraction_with_mock():
    """Test extraction using mock backend."""
    # Create mock backend
    mock_backend = MockLLMBackend(
        mock_response={
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        }
    )
    
    # Use mock backend
    result = mock_backend.extract_from_markdown(
        markdown="Name: John Doe, Age: 30",
        template=Person
    )
    
    # Verify
    assert result is not None
    assert result.name == "John Doe"
    assert result.age == 30
    assert mock_backend.call_count == 1

def test_extraction_tracks_calls():
    """Test mock tracks method calls."""
    mock_backend = MockLLMBackend()
    
    mock_backend.extract_from_markdown("test", Person)
    mock_backend.extract_from_markdown("test2", Person)
    
    assert mock_backend.call_count == 2
    assert mock_backend.last_markdown == "test2"
```

---

## Integration Testing

### Test Complete Pipeline

```python
"""Integration test for complete pipeline."""

import pytest
from pathlib import Path
from docling_graph import PipelineConfig

@pytest.fixture
def sample_document(tmp_path):
    """Create sample document for testing."""
    doc_path = tmp_path / "test.pdf"
    # Create minimal PDF (or use existing test file)
    doc_path.write_bytes(b"%PDF-1.4\n%Test PDF")
    return doc_path

@pytest.fixture
def output_dir(tmp_path):
    """Create output directory."""
    output = tmp_path / "outputs"
    output.mkdir()
    return output

def test_pipeline_execution(sample_document, output_dir):
    """Test pipeline executes successfully."""
    config = PipelineConfig(
        source=str(sample_document),
        template="templates.my_template.Person",
        output_dir=str(output_dir)
    )
    
    # Should not raise
    config.run()
    
    # Verify outputs exist
    assert (output_dir / "nodes.csv").exists()
    assert (output_dir / "edges.csv").exists()

def test_pipeline_with_invalid_source():
    """Test pipeline handles invalid source."""
    config = PipelineConfig(
        source="nonexistent.pdf",
        template="templates.my_template.Person"
    )
    
    with pytest.raises(Exception):
        config.run()
```

### Test with Real Documents

```python
"""Test with real document samples."""

import pytest
from pathlib import Path
from docling_graph import PipelineConfig

@pytest.fixture
def invoice_pdf():
    """Path to sample invoice."""
    return Path("tests/fixtures/sample_invoice.pdf")

@pytest.fixture
def research_paper_pdf():
    """Path to sample research paper."""
    return Path("tests/fixtures/sample_paper.pdf")

def test_invoice_extraction(invoice_pdf, tmp_path):
    """Test invoice extraction."""
    if not invoice_pdf.exists():
        pytest.skip("Sample invoice not available")
    
    config = PipelineConfig(
        source=str(invoice_pdf),
        template="templates.invoice.Invoice",
        output_dir=str(tmp_path)
    )
    
    config.run()
    
    # Verify invoice-specific outputs
    nodes_file = tmp_path / "nodes.csv"
    assert nodes_file.exists()
    
    # Check for expected node types
    content = nodes_file.read_text()
    assert "Invoice" in content
    assert "LineItem" in content

def test_research_paper_extraction(research_paper_pdf, tmp_path):
    """Test research paper extraction."""
    if not research_paper_pdf.exists():
        pytest.skip("Sample paper not available")
    
    config = PipelineConfig(
        source=str(research_paper_pdf),
        template="templates.research.ResearchPaper",
        output_dir=str(tmp_path),
        use_chunking=True  # Large document
    )
    
    config.run()
    
    # Verify outputs
    assert (tmp_path / "nodes.csv").exists()
    assert (tmp_path / "edges.csv").exists()
```

---

## Test Fixtures

### Shared Fixtures

```python
"""Shared test fixtures in conftest.py."""

import pytest
from pathlib import Path
from pydantic import BaseModel, Field

@pytest.fixture
def sample_template():
    """Sample template for testing."""
    class TestTemplate(BaseModel):
        name: str = Field(..., description="Name")
        value: int = Field(..., description="Value")
    
    return TestTemplate

@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "name": "Test",
        "value": 42
    }

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir

@pytest.fixture
def mock_backend():
    """Mock backend for testing."""
    from tests.mocks import MockLLMBackend
    return MockLLMBackend()
```

### Use Fixtures

```python
"""Use shared fixtures in tests."""

def test_with_fixtures(sample_template, sample_data):
    """Test using fixtures."""
    instance = sample_template.model_validate(sample_data)
    
    assert instance.name == "Test"
    assert instance.value == 42

def test_with_output_dir(temp_output_dir):
    """Test with temporary output directory."""
    test_file = temp_output_dir / "test.txt"
    test_file.write_text("test")
    
    assert test_file.exists()
```

---

## Parametrized Tests

### Test Multiple Inputs

```python
"""Test with multiple parameter sets."""

import pytest
from templates.my_template import Person

@pytest.mark.parametrize("name,age,valid", [
    ("John", 30, True),
    ("Jane", 25, True),
    ("Bob", -5, False),  # Invalid age
    ("", 30, False),     # Empty name
])
def test_person_validation(name, age, valid):
    """Test person validation with various inputs."""
    if valid:
        person = Person(name=name, age=age)
        assert person.name == name
        assert person.age == age
    else:
        with pytest.raises(Exception):
            Person(name=name, age=age)

@pytest.mark.parametrize("backend,inference", [
    ("llm", "local"),
    ("llm", "remote"),
    ("vlm", "local"),
])
def test_pipeline_configurations(backend, inference, tmp_path):
    """Test different pipeline configurations."""
    from docling_graph import PipelineConfig
    
    config = PipelineConfig(
        source="test.pdf",
        template="templates.my_template.Person",
        backend=backend,
        inference=inference,
        output_dir=str(tmp_path)
    )
    
    # Verify configuration
    assert config.backend == backend
    assert config.inference == inference
```

---

## Coverage Testing

### Run with Coverage

```bash
# Run tests with coverage
uv run pytest --cov=templates --cov=my_module --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Coverage Configuration

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Coverage settings
[coverage:run]
source = templates,my_module
omit = tests/*,*/__pycache__/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install uv
      run: curl -LsSf https://astral.sh/uv/install.sh | sh
    
    - name: Install dependencies
      run: uv sync --extra dev
    
    - name: Run tests
      run: uv run pytest --cov --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

---

## Best Practices

### 1. Test Edge Cases

```python
# ✅ Good - Test edge cases
def test_empty_string():
    """Test with empty string."""
    with pytest.raises(ValidationError):
        Person(name="", age=30)

def test_boundary_values():
    """Test boundary values."""
    Person(name="A", age=0)    # Minimum
    Person(name="A"*100, age=150)  # Maximum

# ❌ Avoid - Only happy path
def test_person():
    """Test person."""
    person = Person(name="John", age=30)
    assert person.name == "John"
```

### 2. Use Descriptive Names

```python
# ✅ Good - Descriptive test names
def test_person_validation_rejects_negative_age():
    """Test that negative ages are rejected."""
    pass

def test_invoice_extraction_handles_multiple_line_items():
    """Test extraction of invoices with multiple items."""
    pass

# ❌ Avoid - Vague names
def test_person():
    pass

def test_extraction():
    pass
```

### 3. Keep Tests Independent

```python
# ✅ Good - Independent tests
def test_create_person():
    """Test person creation."""
    person = Person(name="John", age=30)
    assert person.name == "John"

def test_validate_person():
    """Test person validation."""
    with pytest.raises(ValidationError):
        Person(name="", age=30)

# ❌ Avoid - Dependent tests
person = None

def test_create():
    global person
    person = Person(name="John", age=30)

def test_validate():
    # Depends on test_create!
    assert person.name == "John"
```

### 4. Mock External Dependencies

```python
# ✅ Good - Mock external APIs
@pytest.fixture
def mock_api(monkeypatch):
    """Mock external API."""
    def mock_call(*args, **kwargs):
        return {"result": "success"}
    
    monkeypatch.setattr("my_module.api.call", mock_call)

def test_with_mock_api(mock_api):
    """Test using mocked API."""
    result = my_function()
    assert result == "success"

# ❌ Avoid - Real API calls in tests
def test_with_real_api():
    """Test with real API."""
    result = api.call()  # Slow, unreliable, costs money
    assert result
```

---

## Troubleshooting Tests

### Issue: Tests Fail Locally But Pass in CI

**Solution:**
```python
# Use tmp_path fixture for file operations
def test_file_operations(tmp_path):
    """Test file operations."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    assert test_file.exists()

# Don't use hardcoded paths
# ❌ test_file = Path("/tmp/test.txt")
```

### Issue: Slow Tests

**Solution:**
```python
# Mark slow tests
@pytest.mark.slow
def test_large_document():
    """Test with large document."""
    pass

# Run fast tests only
# pytest -m "not slow"
```

### Issue: Flaky Tests

**Solution:**
```python
# Add retries for flaky tests
@pytest.mark.flaky(reruns=3)
def test_api_call():
    """Test API call (may be flaky)."""
    pass
```

---

## Next Steps

1. **[Advanced Topics Index](index.md)** - Back to overview
2. **[Custom Backends →](custom-backends.md)** - Test custom backends
3. **[Error Handling →](error-handling.md)** - Test error scenarios

---

## Related Documentation

- **[Schema Definition](../../fundamentals/schema-definition/index.md)** - Template creation
- **[Python API](../api/index.md)** - API usage
- **[Examples](../examples/index.md)** - Example templates