# Custom Pipeline Stages


## Overview

Add custom stages to the docling-graph pipeline for specialized preprocessing, validation, or post-processing tasks.

**What You'll Learn:**
- Pipeline stage protocol
- Stage implementation
- Context management
- Integration with pipeline
- Error handling

**Prerequisites:**
- Understanding of [Pipeline Architecture](../../introduction/architecture.md)
- Familiarity with [Python API](../api/index.md)
- Knowledge of pipeline context

---

## Pipeline Stage Protocol

Custom stages should follow this pattern:

```python
from docling_graph.pipeline.context import PipelineContext

class CustomStage:
    """Custom pipeline stage."""
    
    def execute(self, context: PipelineContext) -> None:
        """
        Execute the stage.
        
        Args:
            context: Pipeline context with shared state
        """
        raise NotImplementedError
```

---

## Pipeline Context

The `PipelineContext` provides access to pipeline state:

```python
@dataclass
class PipelineContext:
    """Shared context for pipeline stages."""
    
    # Configuration
    config: Dict[str, Any]
    
    # Paths
    source: Path
    output_dir: Path
    
    # Pipeline state
    template: Type[BaseModel] | None = None
    docling_doc: Any = None
    extracted_models: List[BaseModel] | None = None
    graph: nx.MultiDiGraph | None = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## Complete Stage Examples

### 1. Preprocessing Stage

```python
"""
Preprocessing stage to validate and prepare documents.
"""

from pathlib import Path
from docling_graph.pipeline.context import PipelineContext
from docling_graph.exceptions import PipelineError

class DocumentValidationStage:
    """
    Validate document before processing.
    
    Checks:
    - File exists and is readable
    - File size is within limits
    - File format is supported
    """
    
    def __init__(
        self,
        max_size_mb: int = 50,
        allowed_formats: list[str] | None = None
    ):
        self.max_size_mb = max_size_mb
        self.allowed_formats = allowed_formats or ['.pdf', '.png', '.jpg', '.jpeg']
    
    def execute(self, context: PipelineContext) -> None:
        """Execute validation."""
        print("🔍 Validating document...")
        
        # Check file exists
        if not context.source.exists():
            raise PipelineError(
                "Source file not found",
                details={"source": str(context.source)}
            )
        
        # Check file size
        size_mb = context.source.stat().st_size / (1024 * 1024)
        if size_mb > self.max_size_mb:
            raise PipelineError(
                f"File too large: {size_mb:.1f}MB (max: {self.max_size_mb}MB)",
                details={"source": str(context.source), "size_mb": size_mb}
            )
        
        # Check format
        if context.source.suffix.lower() not in self.allowed_formats:
            raise PipelineError(
                f"Unsupported format: {context.source.suffix}",
                details={
                    "source": str(context.source),
                    "allowed": self.allowed_formats
                }
            )
        
        # Store metadata
        context.metadata["validation"] = {
            "size_mb": size_mb,
            "format": context.source.suffix,
            "validated": True
        }
        
        print(f"✅ Document validated ({size_mb:.1f}MB)")
```

### 2. Post-Processing Stage

```python
"""
Post-processing stage to enrich extracted data.
"""

from typing import List
from pydantic import BaseModel
from docling_graph.pipeline.context import PipelineContext
from docling_graph.exceptions import PipelineError

class DataEnrichmentStage:
    """
    Enrich extracted data with additional information.
    
    Examples:
    - Add timestamps
    - Normalize values
    - Add computed fields
    - Validate business rules
    """
    
    def __init__(self, add_timestamps: bool = True):
        self.add_timestamps = add_timestamps
    
    def execute(self, context: PipelineContext) -> None:
        """Execute enrichment."""
        if not context.extracted_models:
            print("⚠️  No models to enrich")
            return
        
        print("🔧 Enriching extracted data...")
        
        enriched_count = 0
        for model in context.extracted_models:
            if self._enrich_model(model):
                enriched_count += 1
        
        context.metadata["enrichment"] = {
            "models_processed": len(context.extracted_models),
            "models_enriched": enriched_count
        }
        
        print(f"✅ Enriched {enriched_count} models")
    
    def _enrich_model(self, model: BaseModel) -> bool:
        """Enrich a single model."""
        enriched = False
        
        # Add timestamp if enabled
        if self.add_timestamps:
            from datetime import datetime
            if hasattr(model, '__dict__'):
                # Add as metadata (not modifying Pydantic model)
                if not hasattr(model, '_metadata'):
                    model._metadata = {}
                model._metadata['processed_at'] = datetime.now().isoformat()
                enriched = True
        
        # Add more enrichment logic here
        
        return enriched
```

### 3. Validation Stage

```python
"""
Validation stage to check extracted data quality.
"""

from typing import List
from pydantic import BaseModel
from docling_graph.pipeline.context import PipelineContext
from docling_graph.exceptions import ValidationError

class QualityCheckStage:
    """
    Validate extracted data quality.
    
    Checks:
    - Required fields are populated
    - Data meets business rules
    - Relationships are valid
    """
    
    def __init__(
        self,
        min_confidence: float = 0.7,
        require_relationships: bool = True
    ):
        self.min_confidence = min_confidence
        self.require_relationships = require_relationships
    
    def execute(self, context: PipelineContext) -> None:
        """Execute quality checks."""
        if not context.extracted_models:
            raise ValidationError("No models to validate")
        
        print("✅ Running quality checks...")
        
        issues = []
        
        for i, model in enumerate(context.extracted_models):
            model_issues = self._check_model(model, i)
            issues.extend(model_issues)
        
        if issues:
            print(f"⚠️  Found {len(issues)} quality issues:")
            for issue in issues[:5]:  # Show first 5
                print(f"  - {issue}")
            if len(issues) > 5:
                print(f"  ... and {len(issues) - 5} more")
        
        context.metadata["quality_check"] = {
            "models_checked": len(context.extracted_models),
            "issues_found": len(issues),
            "passed": len(issues) == 0
        }
        
        if issues and self._is_critical():
            raise ValidationError(
                f"Quality check failed with {len(issues)} issues",
                details={"issues": issues[:10]}
            )
        
        print(f"✅ Quality check complete ({len(issues)} issues)")
    
    def _check_model(self, model: BaseModel, index: int) -> List[str]:
        """Check a single model."""
        issues = []
        
        # Check for empty required fields
        for field_name, field_info in model.model_fields.items():
            if field_info.is_required():
                value = getattr(model, field_name, None)
                if value is None or (isinstance(value, str) and not value.strip()):
                    issues.append(
                        f"Model {index}: Required field '{field_name}' is empty"
                    )
        
        # Check relationships if required
        if self.require_relationships:
            has_relationships = self._has_relationships(model)
            if not has_relationships:
                issues.append(
                    f"Model {index}: No relationships found"
                )
        
        return issues
    
    def _has_relationships(self, model: BaseModel) -> bool:
        """Check if model has any relationships."""
        for field_name, field_info in model.model_fields.items():
            json_schema_extra = field_info.json_schema_extra or {}
            if "edge_label" in json_schema_extra:
                value = getattr(model, field_name, None)
                if value is not None:
                    return True
        return False
    
    def _is_critical(self) -> bool:
        """Determine if issues are critical."""
        # Could be configurable
        return False
```

### 4. Logging Stage

```python
"""
Logging stage to track pipeline execution.
"""

import json
from datetime import datetime
from pathlib import Path
from docling_graph.pipeline.context import PipelineContext

class PipelineLoggingStage:
    """
    Log pipeline execution details.
    
    Creates a log file with:
    - Execution timestamp
    - Configuration used
    - Processing statistics
    - Any errors or warnings
    """
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
        self.start_time = None
    
    def execute(self, context: PipelineContext) -> None:
        """Execute logging."""
        if self.start_time is None:
            self.start_time = datetime.now()
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "source": str(context.source),
            "output_dir": str(context.output_dir),
            "config": self._sanitize_config(context.config),
            "metadata": context.metadata,
            "statistics": self._gather_statistics(context)
        }
        
        # Write log file
        log_path = context.output_dir / "pipeline.log.json"
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        print(f"📝 Log written to {log_path}")
    
    def _sanitize_config(self, config: dict) -> dict:
        """Remove sensitive data from config."""
        sanitized = config.copy()
        # Remove API keys
        for key in list(sanitized.keys()):
            if 'key' in key.lower() or 'token' in key.lower():
                sanitized[key] = "***REDACTED***"
        return sanitized
    
    def _gather_statistics(self, context: PipelineContext) -> dict:
        """Gather processing statistics."""
        stats = {}
        
        if context.extracted_models:
            stats["num_models"] = len(context.extracted_models)
        
        if context.graph:
            stats["num_nodes"] = context.graph.number_of_nodes()
            stats["num_edges"] = context.graph.number_of_edges()
        
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            stats["duration_seconds"] = duration
        
        return stats
```

---

## Integration with Pipeline

### Method 1: Wrapper Function

```python
"""
Wrap pipeline execution with custom stages.
"""

from docling_graph import PipelineConfig
from docling_graph.pipeline.context import PipelineContext
from my_stages import DocumentValidationStage, QualityCheckStage

def run_pipeline_with_stages(config: PipelineConfig):
    """Run pipeline with custom stages."""
    
    # Create context
    context = PipelineContext(
        config=config.to_dict(),
        source=Path(config.source),
        output_dir=Path(config.output_dir)
    )
    
    # Pre-processing stages
    validation_stage = DocumentValidationStage(max_size_mb=100)
    validation_stage.execute(context)
    
    # Run main pipeline
    config.run()
    
    # Post-processing stages
    # (Would need to load results from output_dir)
    quality_stage = QualityCheckStage()
    # quality_stage.execute(context)
    
    print("✅ Pipeline with custom stages complete")

# Usage
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate"
)
run_pipeline_with_stages(config)
```

### Method 2: Custom Orchestrator

```python
"""
Create custom pipeline orchestrator.
"""

from typing import List
from docling_graph.pipeline.context import PipelineContext
from docling_graph.pipeline.stages import (
    TemplateLoadingStage,
    ExtractionStage,
    GraphConversionStage,
    ExportStage
)
from my_stages import DocumentValidationStage, QualityCheckStage

class CustomPipelineOrchestrator:
    """Custom pipeline with additional stages."""
    
    def __init__(self, config: dict):
        self.config = config
        self.stages = self._build_stages()
    
    def _build_stages(self) -> List:
        """Build pipeline stages."""
        return [
            DocumentValidationStage(),      # Custom pre-processing
            TemplateLoadingStage(),         # Built-in
            ExtractionStage(),              # Built-in
            QualityCheckStage(),            # Custom validation
            GraphConversionStage(),         # Built-in
            ExportStage(),                  # Built-in
        ]
    
    def run(self) -> None:
        """Execute pipeline."""
        context = PipelineContext(
            config=self.config,
            source=Path(self.config["source"]),
            output_dir=Path(self.config["output_dir"])
        )
        
        for stage in self.stages:
            stage_name = stage.__class__.__name__
            print(f"\n{'='*60}")
            print(f"Stage: {stage_name}")
            print(f"{'='*60}")
            
            try:
                stage.execute(context)
            except Exception as e:
                print(f"❌ Stage {stage_name} failed: {e}")
                raise
        
        print("\n✅ All stages complete")
```

---

## Testing Custom Stages

```python
"""Test custom pipeline stage."""

import pytest
from pathlib import Path
from docling_graph.pipeline.context import PipelineContext
from my_stages import DocumentValidationStage

@pytest.fixture
def sample_context(tmp_path):
    """Create sample context."""
    # Create a test file
    test_file = tmp_path / "test.pdf"
    test_file.write_bytes(b"PDF content")
    
    return PipelineContext(
        config={},
        source=test_file,
        output_dir=tmp_path / "output"
    )

def test_stage_execution(sample_context):
    """Test stage executes successfully."""
    stage = DocumentValidationStage(max_size_mb=1)
    
    # Should not raise
    stage.execute(sample_context)
    
    # Check metadata was added
    assert "validation" in sample_context.metadata
    assert sample_context.metadata["validation"]["validated"]

def test_stage_file_not_found():
    """Test stage handles missing file."""
    context = PipelineContext(
        config={},
        source=Path("nonexistent.pdf"),
        output_dir=Path("output")
    )
    
    stage = DocumentValidationStage()
    
    with pytest.raises(Exception):
        stage.execute(context)

def test_stage_file_too_large(tmp_path):
    """Test stage rejects large files."""
    # Create large file
    large_file = tmp_path / "large.pdf"
    large_file.write_bytes(b"x" * (100 * 1024 * 1024))  # 100MB
    
    context = PipelineContext(
        config={},
        source=large_file,
        output_dir=tmp_path / "output"
    )
    
    stage = DocumentValidationStage(max_size_mb=50)
    
    with pytest.raises(Exception):
        stage.execute(context)
```

---

## Best Practices

### 1. Keep Stages Focused

```python
# ✅ Good - Single responsibility
class ValidationStage:
    """Validate document format and size."""
    def execute(self, context): ...

class EnrichmentStage:
    """Enrich extracted data."""
    def execute(self, context): ...

# ❌ Avoid - Multiple responsibilities
class ProcessingStage:
    """Validate, enrich, and export."""
    def execute(self, context): ...
```

### 2. Handle Errors Gracefully

```python
# ✅ Good - Structured error handling
from docling_graph.exceptions import PipelineError

def execute(self, context):
    try:
        self._process(context)
    except ValueError as e:
        raise PipelineError("Validation failed", cause=e)
    except Exception as e:
        raise PipelineError("Stage execution failed", cause=e)

# ❌ Avoid - Silent failures
def execute(self, context):
    try:
        self._process(context)
    except:
        pass  # Error ignored!
```

### 3. Update Context Metadata

```python
# ✅ Good - Track stage execution
def execute(self, context):
    start_time = time.time()
    
    # Process...
    
    context.metadata[self.__class__.__name__] = {
        "executed": True,
        "duration": time.time() - start_time,
        "items_processed": count
    }

# ❌ Avoid - No tracking
def execute(self, context):
    # Process without tracking
    pass
```

### 4. Make Stages Configurable

```python
# ✅ Good - Configurable behavior
class MyStage:
    def __init__(self, threshold: float = 0.8, strict: bool = False):
        self.threshold = threshold
        self.strict = strict

# ❌ Avoid - Hardcoded behavior
class MyStage:
    def __init__(self):
        self.threshold = 0.8  # Cannot be changed
```

---

## Next Steps

1. **[Performance Tuning →](performance-tuning.md)** - Optimize pipeline
2. **[Error Handling →](error-handling.md)** - Handle errors
3. **[Testing →](testing.md)** - Test your stages

---

## Related Documentation

- **[Pipeline Architecture](../../introduction/architecture.md)** - System design
- **[Python API](../api/index.md)** - Programmatic usage
- **[Exceptions](../../reference/exceptions.md)** - Exception hierarchy