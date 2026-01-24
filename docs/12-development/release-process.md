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

Example: 1.0.0
```

### Version Components

| Component | When to Increment |
|-----------|-------------------|
| **MAJOR** | Breaking changes (manual only) |
| **MINOR** | New features (backward compatible) |
| **PATCH** | Bug fixes (backward compatible) |

### Examples

```
1.0.0 → 1.0.1  # Bug fix
1.0.1 → 1.1.0  # New feature
1.1.0 → 2.0.0  # Breaking change (manual tag required)
```

---

## Automated vs Manual Releases

### Automated Releases (Semantic Release)

Our CI/CD automatically handles routine releases:

| Commit Type | Version Bump | Example |
|-------------|--------------|---------|
| `fix:` | Patch (1.0.0 → 1.0.1) | `fix: handle null values` |
| `feat:` | Minor (1.0.0 → 1.1.0) | `feat: add CSV export` |
| `refactor:` | Minor (1.0.0 → 1.1.0) | `refactor: improve parser` |
| `perf:` | Patch (1.0.0 → 1.0.1) | `perf: optimize graph build` |
| `BREAKING CHANGE:` | Minor (1.0.0 → 1.1.0) | **Capped at minor, not major** |

**Note:** BREAKING CHANGE commits will trigger minor releases, not major. This allows you to accumulate breaking changes and release them together as a planned major version.

### Manual Releases (Git Tags)

Major version bumps require manual Git tags:

- **Major releases** (1.x.x → 2.0.0): Manual Git tags only
- Requires deliberate decision and planning
- Includes migration guides and announcements
- Prevents accidental major bumps from commit messages

### Why Manual Major Releases?

Following industry best practices:

1. **Strategic Milestones**: Major versions signal significant changes requiring coordination
2. **Communication**: Need time for announcements, migration guides, and user preparation
3. **Prevent Accidents**: Developers can't accidentally trigger major bumps with commit messages
4. **Stability Perception**: Controlled major releases signal project maturity
5. **Accumulate Changes**: Collect multiple breaking changes for a single coordinated release

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

### Major Release (1.x.x → 2.0.0) - Manual Only

**When:**
- Breaking API changes
- Removed deprecated features
- Major architectural refactoring
- Multiple accumulated breaking changes

**Important:** Major releases require manual Git tags and cannot be triggered by commits.

**Process:**

1. **Prepare breaking changes** on a release branch
2. **Update version manually** in `pyproject.toml` and `__init__.py`
3. **Create comprehensive CHANGELOG** with migration guide
4. **Merge to main** via PR
5. **Create and push Git tag:**

```bash
git checkout main
git pull origin main

# Create annotated tag with detailed message
git tag -a v2.0.0 -m "Major release v2.0.0

Breaking changes:
- Changed run_pipeline signature to require PipelineConfig
- Removed deprecated old_function() (use new_function instead)
- Refactored graph converter API

See CHANGELOG.md for full details and migration guide."

# Push tag to trigger release workflow
git push origin v2.0.0
```

**Note:** Even if you use `BREAKING CHANGE:` in commits, semantic-release will only bump to the next minor version. Major bumps require manual tags.

---
## Creating a Major Release (Detailed Guide)

Major releases are strategic milestones that require careful planning and execution. Follow this comprehensive guide:

### Prerequisites

Before starting a major release:

- [ ] All breaking changes are documented
- [ ] Migration guide is prepared
- [ ] Deprecation warnings were in place in previous versions
- [ ] Team consensus on timing and scope
- [ ] Communication plan is ready (announcements, blog posts, etc.)
- [ ] All tests pass with breaking changes
- [ ] Documentation is updated for new APIs

### Step-by-Step Process

#### 1. Create Release Branch

```bash
# Start from main
git checkout main
git pull origin main

# Create release branch
git checkout -b release/2.0.0
```

#### 2. Implement Breaking Changes

Make all necessary breaking changes on this branch:

```bash
# Make changes
vim docling_graph/pipeline.py

# Commit with clear messages
git commit -m "feat!: change run_pipeline signature

BREAKING CHANGE: run_pipeline now requires PipelineConfig object
instead of individual parameters. This provides better type safety
and makes the API more maintainable.

Migration:
  Before: run_pipeline(source, template, backend='llm')
  After:  config = PipelineConfig(source=source, template=template)
          run_pipeline(config)"
```

#### 3. Update Version Numbers Manually

**pyproject.toml:**
```toml
[project]
name = "docling-graph"
version = "2.0.0"  # Update to new major version
```

**docling_graph/__init__.py:**
```python
__version__ = "2.0.0"  # Update to new major version
```

#### 4. Create Comprehensive CHANGELOG

**CHANGELOG.md:**
```markdown
# Changelog

## [2.0.0] - 2024-XX-XX

### BREAKING CHANGES

#### Changed run_pipeline API

**Before:**
\`\`\`python
from docling_graph import run_pipeline

result = run_pipeline(
    source="document.pdf",
    template=MyTemplate,
    backend="llm"
)
\`\`\`

**After:**
\`\`\`python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template=MyTemplate,
    backend="llm"
)
result = run_pipeline(config)
\`\`\`

**Migration:** Update all `run_pipeline` calls to use `PipelineConfig`.

#### Removed deprecated functions

- Removed `old_function()` (deprecated in v1.5.0)
  - **Migration:** Use `new_function()` instead

### Added
- New graph validation features
- Enhanced error messages

### Fixed
- Various bug fixes
```

#### 5. Update Documentation

Update all documentation to reflect breaking changes:

```bash
# Update examples
vim docs/09-examples/quickstart.md

# Update API documentation
vim docs/08-api/run-pipeline.md

# Create migration guide
vim docs/12-development/migration-v2.md
```

#### 6. Commit Version Updates

```bash
git add pyproject.toml docling_graph/__init__.py CHANGELOG.md docs/
git commit -m "chore: prepare version 2.0.0 release"
```

#### 7. Final Testing

```bash
# Run full test suite
uv run pytest

# Check code quality
uv run ruff check .
uv run mypy docling_graph

# Build and test package
uv build
uv run pip install dist/docling_graph-2.0.0-*.whl
```

#### 8. Create Pull Request

```bash
# Push release branch
git push origin release/2.0.0

# Create PR to main
# Title: "Release v2.0.0"
# Description: Include summary of breaking changes
# Get team approval
```

#### 9. Merge to Main

After PR approval:
```bash
# Merge PR (via GitHub UI or command line)
# DO NOT create a tag yet - this is done manually next
```

#### 10. Create and Push Git Tag

This is the critical step that triggers the major release:

```bash
# Checkout and update main
git checkout main
git pull origin main

# Create annotated tag with comprehensive message
git tag -a v2.0.0 -m "Major release v2.0.0

Breaking changes:
- Changed run_pipeline API to use PipelineConfig
- Removed old_function() (use new_function instead)
- Refactored graph converter for better performance

New features:
- Enhanced graph validation
- Improved error messages

See CHANGELOG.md for complete details and migration guide.
See docs/12-development/migration-v2.md for migration instructions."

# Push tag to trigger release workflow
git push origin v2.0.0
```

#### 11. Monitor Release

Watch the GitHub Actions workflow:
- Build completes successfully
- Tests pass
- Package published to PyPI
- GitHub release created

#### 12. Post-Release Tasks

```bash
# Verify PyPI release
pip install docling-graph==2.0.0

# Update documentation site
# Publish announcement
# Monitor for issues
```

### Communication Checklist

- [ ] GitHub release notes published
- [ ] Migration guide available
- [ ] Announcement in GitHub Discussions
- [ ] Update README badges if needed
- [ ] Social media announcement (if applicable)
- [ ] Email to major users (if applicable)

### Rollback Plan

If critical issues are discovered:

1. **Quick fix:** Release v2.0.1 hotfix
2. **Major issues:** Yank v2.0.0 from PyPI, recommend v1.x.x

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