# Development Guide


## Overview

Guide for contributing to docling-graph development.

**What's Included:**
- Contributing guidelines
- Development setup
- Code standards
- Testing requirements
- GitHub workflow
- Release process

---

## Quick Start

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/docling-graph.git
cd docling-graph
```

### 2. Setup Development Environment

```bash
# Install with all dependencies
uv sync --extra dev

# Verify installation
uv run python -c "import docling_graph; print(docling_graph.__version__)"
```

### 3. Create Branch

```bash
# Create feature branch
git checkout -b feature/my-feature

# Or bug fix branch
git checkout -b fix/issue-123
```

### 4. Make Changes

```bash
# Edit code
# Add tests
# Update documentation
```

### 5. Run Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=docling_graph --cov-report=html
```

### 6. Submit Pull Request

```bash
# Commit changes
git add .
git commit -s -m "feat: add new feature"

# Push to your fork
git push origin feature/my-feature

# Create PR on GitHub
```

---

## Development Topics

### 📝 Contributing

**Contributing Guidelines**
Official contribution guidelines for the project.

- Code of conduct
- Contribution workflow
- Issue reporting
- Pull request process
- Legal requirements (DCO)

### 🔧 GitHub Workflow

**[GitHub Workflow](github-workflow.md)**  
Working with GitHub and CI/CD.

- Branch strategy
- Commit conventions
- CI/CD pipeline
- Automated testing
- Code quality checks

### 🚀 Release Process

**[Release Process](release-process.md)**  
How releases are managed.

- Version numbering
- Release checklist
- Changelog management
- Publishing process
- Documentation updates

---

## Development Setup

### System Requirements

- Python 3.10+
- Git
- uv package manager
- (Optional) GPU with CUDA for local inference

### Install Development Dependencies

```bash
# Full development setup
uv sync --extra dev

# This installs:
# - Core dependencies
# - Local inference (vLLM, transformers)
# - Remote API clients
# - Development tools (pytest, ruff, mypy)
# - Documentation tools (mkdocs)
```

### Verify Setup

```bash
# Check Python version
python --version  # Should be 3.10+

# Check uv
uv --version

# Run tests
uv run pytest

# Check code quality
uv run ruff check .
uv run mypy docling_graph
```

---

## Code Standards

### Style Guide

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use type hints for all functions
- Docstrings for all public APIs
- Format with `ruff format`

### Type Checking

All code must pass mypy:

```bash
uv run mypy docling_graph
```

### Linting

Code must pass ruff checks:

```bash
# Check code
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

---

## Testing Requirements

### Test Coverage

- Minimum 80% code coverage
- All new features must have tests
- Bug fixes must include regression tests

### Running Tests

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/unit/test_config.py

# Specific test
uv run pytest tests/unit/test_config.py::test_pipeline_config

# With coverage
uv run pytest --cov=docling_graph --cov-report=html

# Fast tests only (skip slow)
uv run pytest -m "not slow"
```

### Writing Tests

```python
"""Test example."""

import pytest
from docling_graph import PipelineConfig

def test_pipeline_config_creation():
    """Test PipelineConfig can be created."""
    config = PipelineConfig(
        source="test.pdf",
        template="templates.Test"
    )
    assert config.source == "test.pdf"
    assert config.backend == "llm"  # Default

def test_pipeline_config_validation():
    """Test PipelineConfig validates inputs."""
    with pytest.raises(ValueError):
        PipelineConfig(
            source="test.pdf",
            template="templates.Test",
            backend="invalid"  # Should fail
        )
```

---

## Documentation

### Building Documentation

```bash
# Install mkdocs
uv add --dev mkdocs mkdocs-material

# Serve locally
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

### Documentation Standards

- All public APIs must be documented
- Include code examples
- Use clear, concise language
- Cross-reference related docs
- Keep examples up to date

---

## Project Structure

```
docling-graph/
├── docling_graph/           # Source code
│   ├── __init__.py
│   ├── pipeline.py
│   ├── config.py
│   ├── protocols.py
│   ├── exceptions.py
│   ├── core/               # Core modules
│   ├── llm_clients/        # LLM integrations
│   ├── pipeline/           # Pipeline orchestration
│   └── cli/                # CLI commands
│
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── fixtures/          # Test fixtures
│   └── mocks/             # Mock objects
│
├── docs/                   # Documentation
│   ├── 01-introduction/
│   ├── installation/
│   └── ...
│
├── examples/               # Example code
│   ├── scripts/
│   └── templates/
│
├── .github/               # GitHub configuration
│
├── pyproject.toml         # Project configuration
├── README.md              # Project README
├── CHANGELOG.md           # Version history
└── LICENSE                # License file
```

---

## Common Tasks

### Add New Feature

1. Create issue describing feature
2. Create feature branch
3. Implement feature with tests
4. Update documentation
5. Submit pull request

### Fix Bug

1. Create issue describing bug
2. Create fix branch
3. Write failing test
4. Fix bug
5. Verify test passes
6. Submit pull request

### Update Documentation

1. Edit markdown files in `docs/`
2. Test locally with `mkdocs serve`
3. Submit pull request

### Add New LLM Client

1. Implement `LLMClientProtocol`
2. Add to `llm_clients/`
3. Add tests
4. Update documentation
5. Submit pull request

---

## Getting Help

### Resources

- **[GitHub Issues](https://github.com/DS4SD/docling-graph/issues)** - Report bugs, request features
- **[GitHub Discussions](https://github.com/docling-project/docling-graph/discussions)** - Ask questions
- **[GitHub Repository](https://github.com/docling-project/docling-graph)** - Source code and issues

### Communication

- Be respectful and constructive
- Provide clear, detailed information
- Include code examples when relevant
- Follow up on feedback

---

## Next Steps

1. **[GitHub Workflow →](github-workflow.md)** - Understand the workflow
2. **[Release Process →](release-process.md)** - Learn about releases
3. **[GitHub Workflow →](github-workflow.md)** - Development workflow