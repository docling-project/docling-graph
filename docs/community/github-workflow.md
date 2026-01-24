# GitHub Workflow


## Overview

Guide to working with GitHub for docling-graph development.

---

## Branch Strategy

### Main Branches

| Branch | Purpose | Protected |
|--------|---------|-----------|
| `main` | Stable releases | ✅ Yes |
| `develop` | Development integration | ✅ Yes |

### Feature Branches

Create from `develop`:

```bash
# Feature
git checkout -b feature/add-custom-backend

# Bug fix
git checkout -b fix/extraction-error

# Documentation
git checkout -b docs/update-api-reference
```

### Branch Naming

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/<description>` | `feature/add-vlm-support` |
| Bug Fix | `fix/<description>` | `fix/config-validation` |
| Documentation | `docs/<description>` | `docs/update-examples` |
| Refactor | `refactor/<description>` | `refactor/pipeline-stages` |

---

## Workflow Steps

### 1. Create Issue

Before starting work:

```markdown
**Title**: Add custom backend support

**Description**:
Allow users to create custom extraction backends by implementing protocols.

**Acceptance Criteria**:
- [ ] Protocol defined
- [ ] Example implementation
- [ ] Tests added
- [ ] Documentation updated
```

### 2. Create Branch

```bash
# From develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/custom-backends
```

### 3. Develop

```bash
# Make changes
vim docling_graph/protocols.py

# Add tests
vim tests/unit/test_protocols.py

# Test locally
uv run pytest
uv run ruff check .
uv run mypy docling_graph
```

### 4. Commit

```bash
# Stage changes
git add .

# Commit with conventional message
git commit -m "feat(protocols): add custom backend protocol

- Define ExtractionBackendProtocol
- Add example implementation
- Include comprehensive tests"
```

### 5. Push

```bash
# Push to your fork
git push origin feature/custom-backends
```

### 6. Create Pull Request

On GitHub:

1. Click "New Pull Request"
2. Select `develop` as base branch
3. Fill in PR template
4. Link related issue
5. Request review

---

## Pull Request Template

```markdown
## Description
Brief description of changes

Fixes #123

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## How Has This Been Tested?
Describe the tests you ran:
- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where needed
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing unit tests pass locally
- [ ] I have updated CHANGELOG.md

## Screenshots (if applicable)
Add screenshots to help explain your changes
```

---

## CI/CD Pipeline

### Automated Checks

On every push and PR:

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
        run: uv sync --extra all --extra dev
      
      - name: Run tests
        run: uv run pytest --cov --cov-report=xml
      
      - name: Lint
        run: uv run ruff check .
      
      - name: Type check
        run: uv run mypy docling_graph
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Status Checks

Required checks before merge:

<br>✅ Tests pass
<br>✅ Code coverage ≥ 80%
<br>✅ Linting passes
<br>✅ Type checking passes
<br>✅ Documentation builds

---

## Code Review Process

### For Contributors

**After submitting PR:**

1. Wait for automated checks
2. Address any failures
3. Respond to reviewer feedback
4. Make requested changes
5. Re-request review

**Responding to feedback:**

```markdown
> Can you add a test for the error case?

Good point! Added test in commit abc123.

> This could be simplified

Refactored in commit def456. Let me know if this is clearer.
```

### For Reviewers

**Review checklist:**

- [ ] Code follows style guide
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance considerations addressed
- [ ] Security implications considered

**Providing feedback:**

```markdown
# ✅ Good feedback
The logic here is correct, but could be simplified:

\`\`\`python
# Instead of:
if condition:
    return True
else:
    return False

# Consider:
return condition
\`\`\`

# ❌ Avoid
This is wrong. Fix it.
```

---

## Merge Strategy

### Squash and Merge

We use squash merging:

```bash
# Multiple commits:
feat: add feature part 1
feat: add feature part 2
fix: typo
docs: update

# Become single commit:
feat: add custom backend support (#123)
```

### Merge Requirements

Before merging:

<br>✅ All checks pass
<br>✅ At least one approval
<br>✅ No unresolved conversations
<br>✅ Branch is up to date

---

## Issue Management

### Labels

| Label | Purpose |
|-------|---------|
| `bug` | Something isn't working |
| `enhancement` | New feature or request |
| `documentation` | Documentation improvements |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention needed |
| `question` | Further information requested |

### Issue Templates

**Bug Report:**

```markdown
**Describe the bug**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior.

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.12]
- docling-graph: [e.g., 0.3.0]
```

**Feature Request:**

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.
```

---

## Release Workflow

### Version Bumping

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Commit changes
git commit -m "chore: bump version to 0.4.0"

# Tag release
git tag -a v0.4.0 -m "Release v0.4.0"

# Push
git push origin main --tags
```

### Automated Release

GitHub Actions automatically:

1. Runs tests
2. Builds package
3. Publishes to PyPI
4. Creates GitHub release
5. Updates documentation

---

## Best Practices

### Commit Messages

```bash
# ✅ Good
feat(extractors): add support for custom chunking

Allows users to provide custom chunking strategies
via the ChunkerProtocol interface.

Closes #123

# ❌ Avoid
update code
fix stuff
wip
```

### PR Size

- Keep PRs focused and small
- One feature/fix per PR
- Split large changes into multiple PRs

### Communication

- Be responsive to feedback
- Ask questions if unclear
- Update PR description if scope changes
- Close stale PRs

---

## Troubleshooting

### CI Failures

**Tests fail:**

```bash
# Run tests locally
uv run pytest -v

# Check specific failure
uv run pytest tests/unit/test_config.py::test_validation -v
```

**Linting fails:**

```bash
# Check issues
uv run ruff check .

# Auto-fix
uv run ruff check --fix .
```

**Type checking fails:**

```bash
# Check types
uv run mypy docling_graph

# Check specific file
uv run mypy docling_graph/config.py
```

### Merge Conflicts

```bash
# Update your branch
git checkout feature/my-feature
git fetch origin
git rebase origin/develop

# Resolve conflicts
# Edit conflicted files
git add .
git rebase --continue

# Force push
git push origin feature/my-feature --force
```

---

## Next Steps

1. **[Release Process →](release-process.md)** - Learn about releases
2. **[Development Guide](index.md)** - Back to overview
3. **[Testing Guide](../usage/advanced/testing.md)** - Testing practices