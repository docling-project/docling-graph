# Complete GitHub Setup Guide

This guide walks you through setting up Docling Graph with full automation including semantic versioning, automated releases, and GitHub Pages documentation.

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [GitHub Configuration](#github-configuration)
3. [PyPI Configuration](#pypi-configuration)
4. [Local Development](#local-development)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

## Initial Setup

### 1. Clone and Install

```bash
# Clone repository
git clone https://github.com/IBM/docling-graph.git
cd docling-graph

# Install with all dependencies
uv sync --all-extras --dev

# Install pre-commit hooks
uv run pre-commit install
```

### 2. Configure Git

Set up commit signing (required for DCO):

```bash
# Configure globally
git config --global format.signoff true

# Or sign each commit
git commit -s -m "your message"
```

## GitHub Configuration

### 1. Enable GitHub Pages

1. Go to **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: **gh-pages** / **root**
4. Click **Save**

The documentation will be available at: `https://ibm.github.io/docling-graph`

### 2. Configure Branch Protection

Go to **Settings → Branches → Add rule** for `main`:

**Required settings:**
- Require a pull request before merging
- Require approvals: 1
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Require conversation resolution before merging
- Include administrators

**Optional but recommended:**
- Require signed commits
- Require linear history

### 3. Create GitHub Environments

#### PyPI Environment

1. Go to **Settings → Environments**
2. Click **New environment**
3. Name: `pypi`
4. **Deployment protection rules:**
   - Required reviewers (optional, for extra safety)
   - Deployment branches: **Selected branches** → Add `main`
5. Click **Save**

#### TestPyPI Environment

1. Click **New environment**
2. Name: `testpypi`
3. No special protection needed
4. Click **Save**

### 4. Enable Security Features

Go to **Settings → Code security and analysis** and enable the following:

- Dependency graph
- Dependabot alerts
- Dependabot security updates

### 5. Create Repository Labels

Go to **Issues → Labels** and create:

```
documentation (color: #0075ca)
tests (color: #d4c5f9)
ci/cd (color: #28a745)
core (color: #d73a4a)
cli (color: #fbca04)
```

Or use the GitHub CLI:

```bash
gh label create documentation --color 0075ca
gh label create tests --color d4c5f9
gh label create ci/cd --color 28a745
gh label create core --color d73a4a
gh label create cli --color fbca04
```

## PyPI Configuration

### 1. Create PyPI Account

If you don't have one:
1. Go to https://pypi.org/account/register/
2. Verify your email
3. Enable 2FA (required for trusted publishing)

### 2. Configure Trusted Publishing (PyPI)

**No API tokens needed!** Use OIDC authentication:

1. Go to https://pypi.org/manage/account/publishing/
2. Click **Add a new publisher**
3. Fill in:
   - **PyPI Project Name**: `docling-graph`
   - **Owner**: `IBM` (or your GitHub username)
   - **Repository name**: `docling-graph`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`
4. Click **Add**

### 3. Configure Trusted Publishing (TestPyPI)

Same process for TestPyPI:

1. Go to https://test.pypi.org/manage/account/publishing/
2. Click **Add a new publisher**
3. Fill in same details as above
4. **Environment name**: `testpypi`
5. Click **Add**

### 4. Verify Configuration

```bash
# Check if project exists on PyPI
pip index versions docling-graph

# Check TestPyPI
pip index versions -i https://test.pypi.org/simple/ docling-graph
```

## Local Development

### 1. Development Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes
# ... edit files ...

# Run pre-commit checks
uv run pre-commit run --all-files

# Run tests
uv run pytest

# Commit with conventional format and sign-off
git commit -s -m "feat: add new feature"

# Push and create PR
git push -u origin feature/my-feature
```

### 2. Conventional Commits

Use these commit types:

| Type | Description | Version Bump |
|------|-------------|--------------|
| `feat:` | New feature | Minor (0.x.0) |
| `fix:` | Bug fix | Patch (0.0.x) |
| `perf:` | Performance | Patch (0.0.x) |
| `docs:` | Documentation | None |
| `test:` | Tests | None |
| `chore:` | Maintenance | None |

Example:
```bash
git commit -s -m "feat(cli): add batch processing command"
git commit -s -m "fix(core): resolve memory leak"
git commit -s -m "docs: update installation guide"
```

### 3. Testing Locally

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=docling_graph

# Run specific test file
uv run pytest tests/unit/test_pipeline.py

# Run type checking
uv run mypy docling_graph

# Run linting
uv run ruff check .
uv run ruff format --check .
```

### 4. Build Documentation Locally

```bash
# Install docs dependencies
pip install mkdocs-material mkdocstrings[python] pymdown-extensions

# Serve locally
mkdocs serve

# Open http://127.0.0.1:8000
```

## Verification

### 1. Test CI Workflows

Create a test PR:

```bash
git checkout -b test/ci-verification
echo "# Test" >> test.md
git add test.md
git commit -s -m "test: verify CI workflows"
git push -u origin test/ci-verification
```

Create PR and verify:
- Pre-commit checks pass
- Tests pass on Python 3.10 and 3.12
- Type checking passes
- Linting passes
- DCO check passes

### 2. Test Semantic Release

After merging a PR with conventional commits to `main`:

1. Check **Actions → Semantic Release** workflow
2. Verify it creates a new version and tag
3. Check that CHANGELOG.md is updated
4. Verify version in `pyproject.toml` and `__init__.py`

### 3. Test Release Workflow

After semantic release creates a tag:

1. Check **Actions → Release** workflow
2. Verify stages:
   - TestPyPI publish succeeds
   - PyPI publish succeeds
   - GitHub Release created
3. Check package on PyPI: https://pypi.org/project/docling-graph/
4. Test installation: `pip install docling-graph`

### 4. Test Documentation Deployment

After pushing docs changes to `main`:

1. Check **Actions → Documentation** workflow
2. Verify deployment succeeds
3. Visit: https://ibm.github.io/docling-graph
4. Verify content is updated

## Troubleshooting

### Issue: Pre-commit Checks Fail

**Solution:**
```bash
# Run locally to see errors
uv run pre-commit run --all-files

# Fix issues and commit
git add .
git commit -s -m "fix: resolve linting issues"
```

### Issue: DCO Check Fails

**Solution:**
```bash
# Amend last commit to add sign-off
git commit --amend -s --no-edit

# Force push (if already pushed)
git push --force-with-lease

# For multiple commits
git rebase --signoff HEAD~3  # Last 3 commits
git push --force-with-lease
```

### Issue: Semantic Release Doesn't Create Version

**Causes:**
- No conventional commits since last release
- Only non-release commits (docs, chore, etc.)

**Solution:**
```bash
# Check commit history
git log --oneline

# Ensure commits follow conventional format
git commit -s -m "feat: add new feature"  # Will trigger minor bump
git commit -s -m "fix: resolve bug"       # Will trigger patch bump
```

### Issue: PyPI Publishing Fails

**Common causes:**
1. OIDC not configured
2. Version already exists
3. Environment not set up

**Solutions:**
```bash
# Verify OIDC configuration
# Go to PyPI → Account → Publishing

# Check if version exists
pip index versions docling-graph

# Verify GitHub environment exists
# Settings → Environments → pypi
```

### Issue: Documentation Not Deploying

**Solutions:**
```bash
# Check workflow logs
# Actions → Documentation → View logs

# Verify gh-pages branch exists
git branch -r | grep gh-pages

# Manually trigger deployment
# Actions → Documentation → Run workflow
```

### Issue: Tests Fail in CI but Pass Locally

**Common causes:**
- Different Python version
- Missing dependencies
- Environment-specific issues

**Solutions:**
```bash
# Test with specific Python version
pyenv install 3.10
pyenv local 3.10
uv run pytest

# Check dependencies
uv sync --all-extras --dev

# Run in clean environment
uv venv --python 3.10
source .venv/bin/activate
uv sync --all-extras --dev
pytest
```

## Next Steps

After setup is complete:

1. **Read the documentation:**
   - [Release Process](release.md)
   - [Contributing Guide](https://github.com/IBM/docling-graph/blob/main/CONTRIBUTING.md)

2. **Start developing:**
   - Create feature branch
   - Make changes
   - Write tests
   - Submit PR

3. **Monitor automation:**
   - Watch GitHub Actions
   - Review release notes
   - Check documentation updates

## Quick Reference

### Common Commands

```bash
# Development
uv sync --all-extras --dev
uv run pre-commit run --all-files
uv run pytest
uv run mypy docling_graph

# Documentation
mkdocs serve
mkdocs build

# Release (automatic)
git commit -s -m "feat: new feature"
git push
# Semantic release handles the rest

# Manual release (emergency only)
git tag v0.3.0
git push origin v0.3.0
```

### Useful Links

- **Repository**: https://github.com/IBM/docling-graph
- **Documentation**: https://ibm.github.io/docling-graph
- **PyPI**: https://pypi.org/project/docling-graph/
- **Issues**: https://github.com/IBM/docling-graph/issues
- **Actions**: https://github.com/IBM/docling-graph/actions

## Support

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section
2. Search [GitHub Issues](https://github.com/IBM/docling-graph/issues)
3. Create new issue with:
   - Clear description
   - Steps to reproduce
   - Error messages
   - Environment details

## Contributing

See [CONTRIBUTING.md](https://github.com/IBM/docling-graph/blob/main/CONTRIBUTING.md) for detailed contribution guidelines.