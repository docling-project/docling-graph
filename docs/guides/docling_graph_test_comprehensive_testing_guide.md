# Docling-Graph Comprehensive Testing Guide

**Version:** 1.0  
**Last Updated:** October 26, 2025  
**Project:** docling-graph


## Table of Contents

1. [Overview](#overview)
2. [Testing Philosophy](#testing-philosophy)
3. [Test Organization](#test-organization)
4. [Naming Conventions](#naming-conventions)
5. [Writing Tests](#writing-tests)
6. [Running Tests](#running-tests)
7. [Test Coverage](#test-coverage)
8. [Continuous Integration](#continuous-integration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)


## Overview

This document defines the testing standards, conventions, and best practices for the docling-graph project. Following these guidelines ensures consistent, maintainable, and reliable tests across the codebase.

### Testing Framework

- **Framework:** pytest >= 7.4.0
- **Coverage Tool:** pytest-cov >= 4.1.0
- **Mocking:** pytest-mock >= 3.11.0
- **Property Testing:** hypothesis >= 6.82.0
- **CLI Testing:** typer.testing.CliRunner

### Test Statistics

- **Total Tests:** 370+
- **Unit Tests:** 285+
- **Integration Tests:** 65+
- **Coverage Target:** 85%+



## Testing Philosophy

### Core Principles

1. **Test Behavior, Not Implementation**
   - Focus on what the code does, not how it does it
   - Tests should survive refactoring

2. **Fast and Isolated**
   - Unit tests should run in milliseconds
   - Each test is independent and can run in any order

3. **Clear and Readable**
   - Test names describe what they test
   - Test code is simple and self-documenting

4. **Comprehensive Coverage**
   - Test happy paths, edge cases, and error conditions
   - Critical paths have multiple tests

5. **Mock External Dependencies**
   - No real API calls, database connections, or file I/O in unit tests
   - Use mock objects for external services



## Test Organization

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── pytest.ini               # Pytest configuration
│
├── unit/                    # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── test_validators.py
│   ├── test_config_utils.py
│   ├── test_converter.py
│   ├── test_exporters.py
│   ├── test_utils.py
│   ├── test_models.py
│   ├── test_graph_stats.py
│   └── test_visualizers.py
│
├── integration/             # Integration tests (slower, end-to-end)
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_cli_commands.py
│   └── test_extraction_flow.py
│
├── fixtures/                # Test data
│   ├── __init__.py
│   ├── sample_documents/
│   ├── sample_configs/
│   └── sample_templates/
│
└── mocks/                   # Mock objects
    ├── __init__.py
    ├── mock_backends.py
    ├── mock_processors.py
    └── mock_clients.py
```

### File Organization Rules

1. **One test file per source file** (for unit tests)
   ```
   docling_graph/cli/validators.py  →  tests/unit/test_validators.py
   docling_graph/core/converter.py  →  tests/unit/test_converter.py
   ```

2. **Logical grouping for integration tests**
   ```
   Full pipeline flow  →  tests/integration/test_pipeline.py
   CLI commands       →  tests/integration/test_cli_commands.py
   ```

3. **Shared fixtures in conftest.py**
   - Common test data
   - Frequently used fixtures
   - Pytest configuration hooks



## Naming Conventions

### Test Files

**Pattern:** `test_<module_name>.py`

```python
# Good
test_validators.py
test_config_utils.py
test_converter.py

# Bad
validators_test.py
test_all_validators.py
testing_validators.py
```

### Test Classes

**Pattern:** `Test<ClassName>` or `Test<FunctionalArea>`

```python
# Good - Testing a class
class TestGraphConverter:
    """Tests for GraphConverter class."""
    pass

# Good - Testing a functional area
class TestValidateProcessingMode:
    """Tests for validate_processing_mode function."""
    pass

# Bad
class GraphConverterTests:    # Wrong suffix
class TestGraphConverterClass:  # Redundant "Class"
class graph_converter_test:   # Not PascalCase
```

### Test Functions

**Pattern:** `test_<what>_<condition>_<expected_result>`

```python
# Good - Clear and descriptive
def test_convert_single_model():
    """Test converting a single Pydantic model."""
    pass

def test_validate_mode_with_invalid_input_raises_error():
    """Test that invalid mode raises validation error."""
    pass

def test_graph_has_correct_node_count():
    """Test that graph has expected number of nodes."""
    pass

# Bad
def test1():                    # No description
def test_converter():           # Too vague
def testConvertModel():         # Wrong case
def test_everything():          # Too broad
```

### Test Function Naming Patterns

| Pattern | Example | Use Case |
|---------|---------|----------|
| `test_<action>_<object>` | `test_create_graph()` | Simple action test |
| `test_<object>_<property>` | `test_graph_is_directed()` | Property verification |
| `test_<condition>_<result>` | `test_empty_input_raises_error()` | Error condition |
| `test_<scenario>` | `test_concurrent_conversions_succeed()` | Complex scenario |

### Fixture Names

**Pattern:** `<descriptive_noun>` (lowercase with underscores)

```python
# Good
@pytest.fixture
def sample_person():
    """Create a sample Person instance."""
    return Person(name="John", age=30, email="john@example.com")

@pytest.fixture
def temp_dir():
    """Provide a temporary directory."""
    pass

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    pass

# Bad
@pytest.fixture
def get_sample_person():        # Don't use verb prefix
    pass

@pytest.fixture
def SamplePerson():             # Wrong case
    pass
```

### Mock Object Names

**Pattern:** `Mock<ClassName>` or `mock_<object_name>`

```python
# Good - Class names
class MockLLMBackend:
    pass

class MockDocumentProcessor:
    pass

# Good - Instance names
mock_llm_client = Mock()
mock_backend = MockLLMBackend()

# Bad
class LLMMock:                  # Wrong suffix
fake_llm = Mock()               # Use 'mock', not 'fake'
test_backend = Mock()           # Use 'mock', not 'test'
```


## Writing Tests

### Test Structure: Arrange-Act-Assert (AAA)

Every test should follow the AAA pattern:

```python
def test_convert_single_model(sample_person):
    """Test converting a single Pydantic model."""

    # Arrange - Set up test data and expectations
    converter = GraphConverter()
    expected_node_count = 1

    # Act - Execute the code under test
    graph, metadata = converter.pydantic_list_to_graph([sample_person])

    # Assert - Verify the results
    assert metadata.node_count == expected_node_count
    assert len(graph.nodes) == 1
```

### Docstrings

Every test function must have a docstring:

```python
def test_validation_with_empty_input():
    """Test that empty input raises ValueError.

    The validator should reject empty strings and raise
    a ValueError with an appropriate error message.
    """
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_input("")
```

### Parametrized Tests

Use `@pytest.mark.parametrize` for multiple similar test cases:

```python
@pytest.mark.parametrize("mode,expected", [
    ("one-to-one", "one-to-one"),
    ("ONE-TO-ONE", "one-to-one"),
    ("many-to-one", "many-to-one"),
])
def test_validate_processing_mode_various_cases(mode, expected):
    """Test processing mode validation with various cases."""
    assert validate_processing_mode(mode) == expected


@pytest.mark.parametrize("invalid_input", [
    "invalid",
    "one-two-one",
    "",
    "batch",
])
def test_validate_processing_mode_rejects_invalid(invalid_input):
    """Test that invalid modes are rejected."""
    with pytest.raises(typer.Exit):
        validate_processing_mode(invalid_input)
```

### Testing Exceptions

```python
# Good - Check exception type and message
def test_empty_list_raises_value_error():
    """Test that empty list raises ValueError."""
    converter = GraphConverter()

    with pytest.raises(ValueError, match="empty model list"):
        converter.pydantic_list_to_graph([])


# Good - Check exception details
def test_invalid_config_raises_with_details():
    """Test that invalid config provides helpful error."""
    with pytest.raises(ValidationError) as exc_info:
        load_config("invalid.yaml")

    assert "invalid syntax" in str(exc_info.value).lower()


# Bad - Too general
def test_error():
    """Test error handling."""
    with pytest.raises(Exception):  # Too broad
        some_function()
```

### Using Fixtures

```python
# Define fixtures in conftest.py
@pytest.fixture
def sample_graph():
    """Create a simple test graph."""
    graph = nx.DiGraph()
    graph.add_edges_from([("a", "b"), ("b", "c")])
    return graph


# Use fixtures in tests
def test_graph_has_nodes(sample_graph):
    """Test that graph has expected nodes."""
    assert len(sample_graph.nodes) == 3
    assert "a" in sample_graph.nodes


# Combine multiple fixtures
def test_export_graph(sample_graph, temp_dir):
    """Test exporting graph to CSV."""
    exporter = CSVExporter()
    exporter.export(sample_graph, temp_dir)

    assert (temp_dir / "nodes.csv").exists()
```

### Mocking

```python
from unittest.mock import Mock, patch, MagicMock

# Good - Mock external dependencies
@patch("docling_graph.pipeline.DocumentProcessor")
def test_pipeline_with_mocked_processor(mock_processor):
    """Test pipeline with mocked document processor."""
    # Configure mock
    mock_instance = Mock()
    mock_instance.process_document.return_value = ["page content"]
    mock_processor.return_value = mock_instance

    # Test code
    result = run_pipeline(config)

    # Verify mock was called
    mock_instance.process_document.assert_called_once()


# Good - Use custom mock objects
def test_extraction_with_mock_backend():
    """Test extraction with mock LLM backend."""
    from tests.mocks.mock_backends import MockLLMBackend

    backend = MockLLMBackend()
    strategy = OneToOneStrategy(backend=backend)

    result = strategy.extract(document, Template)

    assert backend.call_count == 1


# Bad - Testing the mock instead of the code
def test_mock_returns_value():
    """Bad test - only tests the mock."""
    mock = Mock(return_value=42)
    assert mock() == 42  # This only tests the mock!
```

### Test Markers

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_validate_input():
    """Unit test for input validation."""
    pass


@pytest.mark.integration
def test_full_pipeline():
    """Integration test for complete pipeline."""
    pass


@pytest.mark.slow
def test_large_document_processing():
    """Slow test that processes large documents."""
    pass


@pytest.mark.requires_api
def test_with_real_api():
    """Test requiring API access."""
    pass


@pytest.mark.requires_ollama
def test_with_ollama_backend():
    """Test requiring Ollama running locally."""
    pytest.skip("Ollama not available")
```


## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_validators.py

# Run specific test class
pytest tests/unit/test_converter.py::TestGraphConverter

# Run specific test function
pytest tests/unit/test_converter.py::TestGraphConverter::test_convert_single_model

# Run tests matching pattern
pytest -k "validator"
pytest -k "test_convert"
```

### By Category

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run everything except slow tests
pytest -m "not slow"

# Run tests requiring API
pytest -m requires_api

# Combine markers
pytest -m "unit and not slow"
```

### With Coverage

```bash
# Run with coverage
pytest --cov=docling_graph

# Generate HTML coverage report
pytest --cov=docling_graph --cov-report=html

# Generate both terminal and HTML reports
pytest --cov=docling_graph --cov-report=html --cov-report=term-missing

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Parallel Execution

```bash
# Run tests in parallel (faster)
pytest -n auto

# Run with specific number of workers
pytest -n 4

# Combine with coverage
pytest -n auto --cov=docling_graph
```

### Debugging

```bash
# Show print statements
pytest -s

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l

# Stop after first failure
pytest -x

# Show detailed traceback
pytest --tb=long

# Show only failed tests
pytest --lf  # last failed
pytest --ff  # failed first
```


## Test Coverage

### Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| CLI validators | 100% |
| Config utilities | 100% |
| Core converter | 95% |
| Exporters | 90% |
| Extractors | 90% |
| Utilities | 90% |
| **Overall Project** | **85%+** |

### Checking Coverage

```bash
# Generate coverage report
pytest --cov=docling_graph --cov-report=term-missing

# Output shows missing lines
Name                                    Stmts   Miss  Cover   Missing------------------------------------------------------------------
docling_graph/cli/validators.py            45      2    96%   89-90
docling_graph/core/converter.py           150      8    95%   142, 156-162
...
```

### Coverage Configuration

Located in `pytest.ini`:

```ini
[coverage:run]
source = docling_graph
omit = 
    */tests/*
    */test_*
    */__pycache__/*

[coverage:report]
precision = 2
show_missing = True
skip_covered = False
```

### What to Cover

**Always Cover:**
- Happy paths (normal usage)
- Edge cases (boundary conditions)
- Error conditions (exceptions, validation failures)
- Critical business logic

**Can Skip:**
- Trivial getters/setters
- `__repr__` and `__str__` methods
- Simple property accessors
- Debug code


## Continuous Integration

### GitHub Actions Example

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e .
          pip install -r requirements-test.txt

      - name: Run tests
        run: |
          pytest --cov=docling_graph --cov-report=xml --cov-report=term

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
```

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: ['-m', 'unit', '--tb=short']
```


## Best Practices

### DO

1. **Write tests first** (TDD when possible)
2. **Test one thing per test**
3. **Use descriptive test names**
4. **Keep tests simple and readable**
5. **Mock external dependencies**
6. **Use fixtures for common setup**
7. **Test edge cases and errors**
8. **Keep tests fast**
9. **Make tests independent**
10. **Update tests with code changes**

### DON'T

1. **Don't test implementation details**
2. **Don't use production databases/APIs**
3. **Don't write tests that depend on order**
4. **Don't test the framework/library code**
5. **Don't create complex test logic**
6. **Don't skip writing tests**
7. **Don't ignore failing tests**
8. **Don't test private methods directly**
9. **Don't make tests overly specific**
10. **Don't forget to update tests**

### Code Examples

**Good Test:**
```python
def test_converter_creates_correct_node_count():
    """Test that converter creates expected number of nodes."""
    # Arrange
    converter = GraphConverter()
    models = [Person(name="Alice", age=25, email="alice@example.com")]

    # Act
    graph, metadata = converter.pydantic_list_to_graph(models)

    # Assert
    assert metadata.node_count == 1
    assert len(graph.nodes) == 1
```

**Bad Test:**
```python
def test_stuff():  # Vague name
    """Test various things."""  # Vague description
    c = GraphConverter()  # Unclear variable names
    result = c.pydantic_list_to_graph(get_data())  # Hidden dependencies
    assert result[1].node_count > 0  # Magic numbers
```


## Troubleshooting

### Common Issues

#### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'docling_graph'`

**Solution:**
```bash
# Install package in development mode
pip install -e .
```

#### Fixture Not Found

**Problem:** `fixture 'sample_person' not found`

**Solution:**
- Ensure `conftest.py` is in the correct location
- Check that fixture is defined with `@pytest.fixture`
- Verify fixture name spelling

#### Tests Pass Locally But Fail in CI

**Problem:** Tests work on your machine but fail in CI

**Solutions:**
- Check Python version differences
- Verify all dependencies are in `requirements-test.txt`
- Look for hardcoded paths or OS-specific code
- Ensure tests don't depend on local files/services

#### Slow Tests

**Problem:** Test suite takes too long to run

**Solutions:**
```python
# Mark slow tests
@pytest.mark.slow
def test_large_operation():
    pass

# Run without slow tests
pytest -m "not slow"

# Use parallel execution
pytest -n auto
```

#### Flaky Tests

**Problem:** Tests sometimes pass, sometimes fail

**Solutions:**
- Remove dependencies on timing
- Mock random/datetime functions
- Ensure proper test isolation
- Check for shared state between tests

### Getting Help

1. Check this guide first
2. Review existing tests for examples
3. Check pytest documentation: https://docs.pytest.org/
4. Ask team members
5. Open an issue on GitHub


## Quick Reference

### Essential Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=docling_graph --cov-report=html

# Run fast tests only
pytest -m "unit and not slow"

# Debug failing test
pytest tests/unit/test_converter.py::test_name -s --pdb

# Update snapshots (if using snapshot testing)
pytest --snapshot-update
```

### Test Template

```python
"""
Unit tests for <module_name>.
"""
import pytest
from docling_graph.<module_path> import <ClassOrFunction>


class Test<ClassName>:
    """Tests for <ClassName> class."""

    def test_<action>_<condition>(self, fixture_name):
        """Test that <action> <expected_result> when <condition>."""
        # Arrange
        instance = <ClassName>()
        expected = <expected_value>

        # Act
        result = instance.method(input_data)

        # Assert
        assert result == expected

    def test_<error_condition>_raises_exception(self):
        """Test that <error_condition> raises <ExceptionType>."""
        with pytest.raises(<ExceptionType>, match="error message"):
            <code_that_should_raise>()

    @pytest.mark.parametrize("input,expected", [
        (input1, expected1),
        (input2, expected2),
    ])
    def test_<action>_with_various_inputs(self, input, expected):
        """Test <action> with various input values."""
        assert <function>(input) == expected
```


## Summary

Following this testing guide ensures:
- **Consistency** across the codebase
- **Maintainability** of tests over time
- **Confidence** in code changes
- **Documentation** through test names and docstrings
- **Quality** through comprehensive coverage

**Remember:** Good tests are an investment. They save time in the long run by catching bugs early and enabling confident refactoring.