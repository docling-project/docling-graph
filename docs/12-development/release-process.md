# Release Process

**Navigation:** [← GitHub Workflow](github-workflow.md) | [Development Guide](index.md)

---

## Overview

Guide to the docling-graph release process.

---

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

Example: 0.3.0
```

### Version Components

| Component | When to Increment |
|-----------|-------------------|
| **MAJOR** | Breaking changes |
| **MINOR** | New features (backward compatible) |
| **PATCH** | Bug fixes (backward compatible) |

### Examples

```
0.2.0 → 0.2.1  # Bug fix
0.2.1 → 0.3.0  # New feature
0.3.0 → 1.0.0  # Breaking change (stable release)
```

---

## Release Types

### Patch Release (0.3.0 → 0.3.1)

**When:**
- Bug fixes
- Documentation updates
- Performance improvements (no API changes)

**Example:**
```bash
# Fix extraction error
git commit -m "fix(extractors): handle empty markdown"

# Release
git tag v0.3.1
```

### Minor Release (0.3.0 → 0.4.0)

**When:**
- New features
- New backends/exporters
- Deprecations (with warnings)

**Example:**
```bash
# Add new feature
git commit -m "feat(exporters): add GraphML exporter"

# Release
git tag v0.4.0
```

### Major Release (0.4.0 → 1.0.0)

**When:**
- Breaking API changes
- Removed deprecated features
- Major refactoring

**Example:**
```bash
# Breaking change
git commit -m "feat(pipeline)!: change run_pipeline signature

BREAKING CHANGE: run_pipeline now requires PipelineConfig"

# Release
git tag v1.0.0
```

---

## Release Checklist

### Pre-Release

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version bumped in `pyproject.toml`
- [ ] No open critical bugs
- [ ] Breaking changes documented

### Release

- [ ] Create release branch
- [ ] Final testing
- [ ] Tag release
- [ ] Push to GitHub
- [ ] Automated build and publish
- [ ] Verify PyPI upload

### Post-Release

- [ ] Create GitHub release notes
- [ ] Announce release
- [ ] Update documentation site
- [ ] Close milestone
- [ ] Merge back to develop

---

## Step-by-Step Process

### 1. Prepare Release

```bash
# Checkout develop
git checkout develop
git pull origin develop

# Create release branch
git checkout -b release/0.4.0
```

### 2. Update Version

**pyproject.toml:**

```toml
[project]
name = "docling-graph"
version = "0.4.0"  # Update version
```

**docling_graph/\_\_init\_\_.py:**

```python
__version__ = "0.4.0"  # Update version
```

### 3. Update CHANGELOG

**CHANGELOG.md:**

```markdown
# Changelog

## [0.4.0] - 2024-01-22

### Added
- GraphML exporter for graph visualization tools
- Support for custom chunking strategies
- New examples for insurance policy extraction

### Changed
- Improved error messages in extraction pipeline
- Updated documentation structure

### Fixed
- Fixed VLM backend memory leak
- Corrected date parsing in templates

### Deprecated
- Old configuration format (use PipelineConfig)

## [0.3.0] - 2024-01-15
...
```

### 4. Commit Changes

```bash
git add pyproject.toml docling_graph/__init__.py CHANGELOG.md
git commit -m "chore: bump version to 0.4.0"
```

### 5. Final Testing

```bash
# Run full test suite
uv run pytest

# Check code quality
uv run ruff check .
uv run mypy docling_graph

# Build documentation
uv run mkdocs build

# Test package build
uv build
```

### 6. Merge to Main

```bash
# Push release branch
git push origin release/0.4.0

# Create PR to main
# Get approval
# Merge PR
```

### 7. Tag Release

```bash
# Checkout main
git checkout main
git pull origin main

# Create tag
git tag -a v0.4.0 -m "Release v0.4.0

- Add GraphML exporter
- Support custom chunking
- Improve error messages
- Fix VLM memory leak"

# Push tag
git push origin v0.4.0
```

### 8. Automated Build

GitHub Actions automatically:

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Build package
        run: uv build
      
      - name: Publish to PyPI
        run: uv publish
        env:
          UV_PUBLISH_TOKEN: ${{ secrets.PYPI_TOKEN }}
      
      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          body_path: CHANGELOG.md
```

### 9. Verify Release

```bash
# Check PyPI
pip install docling-graph==0.4.0

# Verify version
python -c "import docling_graph; print(docling_graph.__version__)"
```

### 10. Create Release Notes

On GitHub:

1. Go to Releases
2. Click "Draft a new release"
3. Select tag v0.4.0
4. Title: "Release 0.4.0"
5. Description from CHANGELOG
6. Publish release

### 11. Announce Release

- GitHub Discussions
- Project README
- Social media (if applicable)

### 12. Merge Back

```bash
# Merge main back to develop
git checkout develop
git merge main
git push origin develop
```

---

## CHANGELOG Format

Follow [Keep a Changelog](https://keepachangelog.com/):

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- New features in development

### Changed
- Changes to existing features

### Deprecated
- Features to be removed

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security fixes

## [0.4.0] - 2024-01-22

### Added
- GraphML exporter (#123)
- Custom chunking support (#124)

### Fixed
- VLM memory leak (#125)

## [0.3.0] - 2024-01-15
...
```

---

## Breaking Changes

### Documentation

Document breaking changes clearly:

```markdown
## [1.0.0] - 2024-02-01

### BREAKING CHANGES

#### run_pipeline signature changed

**Before:**
\`\`\`python
run_pipeline(source, template, backend="llm")
\`\`\`

**After:**
\`\`\`python
config = PipelineConfig(source=source, template=template)
run_pipeline(config)
\`\`\`

**Migration:**
Update all calls to use PipelineConfig.

#### Removed deprecated features

- Removed `old_function()` (deprecated in 0.8.0)
- Use `new_function()` instead
```

### Deprecation Period

1. **Version N**: Add deprecation warning
2. **Version N+1**: Keep with warning
3. **Version N+2**: Remove feature

**Example:**

```python
# Version 0.8.0 - Add warning
def old_function():
    warnings.warn(
        "old_function is deprecated, use new_function",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()

# Version 0.9.0 - Keep warning

# Version 1.0.0 - Remove
# old_function() removed
```

---

## Hotfix Process

For critical bugs in production:

### 1. Create Hotfix Branch

```bash
# From main
git checkout main
git checkout -b hotfix/0.3.1
```

### 2. Fix Bug

```bash
# Fix the bug
vim docling_graph/module.py

# Add test
vim tests/unit/test_module.py

# Commit
git commit -m "fix: critical extraction bug"
```

### 3. Update Version

```bash
# Bump patch version
# Update CHANGELOG

git commit -m "chore: bump version to 0.3.1"
```

### 4. Release

```bash
# Merge to main
git checkout main
git merge hotfix/0.3.1

# Tag
git tag v0.3.1

# Push
git push origin main --tags

# Merge back to develop
git checkout develop
git merge hotfix/0.3.1
git push origin develop
```

---

## Release Schedule

### Regular Releases

- **Minor releases**: Monthly (if features ready)
- **Patch releases**: As needed (bug fixes)
- **Major releases**: When breaking changes accumulated

### Release Windows

- Avoid releases on Fridays
- Avoid holiday periods
- Allow time for testing

---

## Rollback Procedure

If a release has critical issues:

### 1. Identify Issue

```bash
# Check reports
# Verify bug
# Assess severity
```

### 2. Quick Fix or Rollback

**Option A: Quick hotfix**

```bash
# If fix is simple
git checkout -b hotfix/0.4.1
# Fix bug
# Release 0.4.1
```

**Option B: Rollback**

```bash
# If fix is complex
# Yank from PyPI (if possible)
# Announce rollback
# Recommend previous version
```

### 3. Communicate

- Update GitHub release
- Post in Discussions
- Update documentation

---

## Post-Release Tasks

### Documentation

- [ ] Update docs site
- [ ] Update examples
- [ ] Update tutorials

### Communication

- [ ] Announce on GitHub
- [ ] Update README badges
- [ ] Social media posts

### Monitoring

- [ ] Watch for issues
- [ ] Monitor PyPI downloads
- [ ] Check user feedback

---

## Tools

### Version Management

```bash
# Check current version
grep version pyproject.toml

# Update version
sed -i 's/version = "0.3.0"/version = "0.4.0"/' pyproject.toml
```

### Build and Publish

```bash
# Build package
uv build

# Check package
uv run twine check dist/*

# Publish to TestPyPI (testing)
uv publish --repository testpypi

# Publish to PyPI (production)
uv publish
```

---

## Next Steps

1. **[Development Guide](index.md)** - Back to overview
2. **[GitHub Workflow](github-workflow.md)** - Development workflow
3. **[Testing Guide](../10-advanced/testing.md)** - Testing practices

---

**Navigation:** [← GitHub Workflow](github-workflow.md) | [Development Guide](index.md)