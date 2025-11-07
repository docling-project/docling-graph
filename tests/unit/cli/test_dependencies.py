"""
Tests for dependency management utilities.
"""

from unittest.mock import MagicMock, patch

import pytest

from docling_graph.cli.dependencies import (
    INFERENCE_PROVIDERS,
    OPTIONAL_DEPS,
    DependencyStatus,
    OptionalDependency,
    check_dependency,
    check_inference_type_available,
    get_all_missing_dependencies,
    get_missing_dependencies,
    get_missing_for_inference_type,
    require_dependency,
)


class TestDependencyStatus:
    """Test DependencyStatus enum."""

    def test_dependency_status_has_valid_values(self):
        """Should have all required status values."""
        assert DependencyStatus.INSTALLED.value == "installed"
        assert DependencyStatus.NOT_INSTALLED.value == "not_installed"
        assert DependencyStatus.UNKNOWN.value == "unknown"


class TestOptionalDependency:
    """Test OptionalDependency class."""

    def test_optional_dependency_initialization(self):
        """Should initialize with required attributes."""
        dep = OptionalDependency(
            name="test_pkg",
            package="test_module",
            extra="test_extra",
            description="Test dependency",
            inference_type="local",
        )
        assert dep.name == "test_pkg"
        assert dep.package == "test_module"
        assert dep.extra == "test_extra"
        assert dep.description == "Test dependency"
        assert dep.inference_type == "local"

    def test_optional_dependency_default_extra(self):
        """Should default extra to 'all' if not provided."""
        dep = OptionalDependency(name="test", package="test_module")
        assert dep.extra == "all"

    def test_optional_dependency_default_description(self):
        """Should generate default description."""
        dep = OptionalDependency(name="test_pkg", package="test_module")
        assert "test_pkg" in dep.description

    def test_optional_dependency_is_installed_caches_result(self):
        """Should cache installation check result."""
        dep = OptionalDependency(name="sys", package="sys")
        # First call
        result1 = dep.is_installed
        # Second call should return cached result
        result2 = dep.is_installed
        assert result1 == result2
        # sys should be installed
        assert result1 is True

    def test_optional_dependency_get_install_command(self):
        """Should generate correct install command."""
        dep = OptionalDependency(name="ollama", package="ollama", extra="ollama")
        cmd = dep.get_install_command()
        assert "pip install" in cmd or "docling-graph[" in cmd
        assert "ollama" in cmd

    def test_optional_dependency_get_direct_install_command(self):
        """Should generate direct install command."""
        dep = OptionalDependency(name="ollama", package="ollama")
        cmd = dep.get_direct_install_command()
        assert "pip install ollama" == cmd

    def test_optional_dependency_repr(self):
        """Should have meaningful repr."""
        dep = OptionalDependency(name="test", package="test")
        repr_str = repr(dep)
        assert "OptionalDependency" in repr_str
        assert "test" in repr_str


class TestOptionalDepsRegistry:
    """Test the OPTIONAL_DEPS registry."""

    def test_optional_deps_has_required_local_providers(self):
        """Should have all local providers."""
        assert "ollama" in OPTIONAL_DEPS
        assert "vllm" in OPTIONAL_DEPS

    def test_optional_deps_has_required_remote_providers(self):
        """Should have all remote providers."""
        assert "mistral" in OPTIONAL_DEPS
        assert "openai" in OPTIONAL_DEPS
        assert "gemini" in OPTIONAL_DEPS

    def test_optional_deps_entries_are_valid(self):
        """All registry entries should be OptionalDependency instances."""
        for name, dep in OPTIONAL_DEPS.items():
            assert isinstance(dep, OptionalDependency)
            assert dep.name == name

    def test_inference_providers_mapping(self):
        """Should have correct inference type mappings."""
        assert "local" in INFERENCE_PROVIDERS
        assert "remote" in INFERENCE_PROVIDERS
        assert set(INFERENCE_PROVIDERS["local"]) == {"ollama", "vllm"}
        assert set(INFERENCE_PROVIDERS["remote"]) == {"mistral", "openai", "gemini"}


class TestCheckDependency:
    """Test check_dependency function."""

    def test_check_dependency_installed(self):
        """Should return True for installed package."""
        # sys is always installed
        result = check_dependency("sys")
        assert result is True

    def test_check_dependency_unknown_provider(self):
        """Should return True for unknown provider."""
        result = check_dependency("nonexistent_provider_xyz")
        assert result is True  # Safe default

    def test_check_dependency_uninstalled(self):
        """Should return False for uninstalled packages."""
        # Create a fake dependency
        with patch.dict(
            OPTIONAL_DEPS,
            {
                "fake_pkg": OptionalDependency(
                    name="fake_pkg", package="this_definitely_does_not_exist_xyz"
                )
            },
        ):
            result = check_dependency("fake_pkg")
            assert result is False


class TestRequireDependency:
    """Test require_dependency function."""

    def test_require_dependency_installed_succeeds(self):
        """Should not raise for installed packages."""
        # sys is always installed
        require_dependency("sys")  # Should not raise

    def test_require_dependency_uninstalled_raises(self):
        """Should raise ImportError for uninstalled packages."""
        with patch.dict(
            OPTIONAL_DEPS,
            {
                "fake_pkg": OptionalDependency(
                    name="fake_pkg", package="this_definitely_does_not_exist_xyz"
                )
            },
        ):
            with pytest.raises(ImportError) as exc_info:
                require_dependency("fake_pkg")
            assert "fake_pkg" in str(exc_info.value)
            assert "pip install" in str(exc_info.value)

    def test_require_dependency_unknown_provider(self):
        """Should not raise for unknown providers."""
        require_dependency("unknown_provider")  # Should not raise


class TestGetMissingDependencies:
    """Test get_missing_dependencies function."""

    def test_get_missing_dependencies_empty_list(self):
        """Should return empty list if all dependencies installed."""
        # sys is always installed
        result = get_missing_dependencies(["sys"])
        assert result == []

    def test_get_missing_dependencies_returns_objects(self):
        """Should return OptionalDependency objects."""
        with patch.dict(
            OPTIONAL_DEPS,
            {"fake_pkg": OptionalDependency(name="fake_pkg", package="nonexistent_xyz")},
        ):
            result = get_missing_dependencies(["fake_pkg"])
            assert len(result) == 1
            assert isinstance(result[0], OptionalDependency)
            assert result[0].name == "fake_pkg"

    def test_get_missing_dependencies_mixed(self):
        """Should handle mix of installed and missing."""
        with patch.dict(
            OPTIONAL_DEPS,
            {"fake_pkg": OptionalDependency(name="fake_pkg", package="nonexistent_xyz")},
        ):
            result = get_missing_dependencies(["sys", "fake_pkg"])
            # Only fake_pkg should be in missing
            assert len(result) == 1
            assert result[0].name == "fake_pkg"


class TestGetMissingForInferenceType:
    """Test get_missing_for_inference_type function."""

    def test_get_missing_for_inference_type_local(self):
        """Should check local inference providers."""
        result = get_missing_for_inference_type("local")
        assert isinstance(result, list)
        # Result should contain OptionalDependency objects
        for dep in result:
            assert isinstance(dep, OptionalDependency)

    def test_get_missing_for_inference_type_remote(self):
        """Should check remote inference providers."""
        result = get_missing_for_inference_type("remote")
        assert isinstance(result, list)
        for dep in result:
            assert isinstance(dep, OptionalDependency)

    def test_get_missing_for_inference_type_unknown(self):
        """Should return empty list for unknown inference type."""
        result = get_missing_for_inference_type("unknown_type")
        assert result == []


class TestCheckInferenceTypeAvailable:
    """Test check_inference_type_available function."""

    def test_check_inference_type_available_with_provider(self):
        """Should check specific provider if provided."""
        result = check_inference_type_available("local", selected_provider="sys")
        assert isinstance(result, bool)

    def test_check_inference_type_available_all_providers(self):
        """Should check all providers for inference type."""
        result = check_inference_type_available("local")
        assert isinstance(result, bool)


class TestGetAllMissingDependencies:
    """Test get_all_missing_dependencies function."""

    def test_get_all_missing_dependencies_structure(self):
        """Should return dict with local and remote keys."""
        result = get_all_missing_dependencies()
        assert isinstance(result, dict)
        assert "local" in result
        assert "remote" in result

    def test_get_all_missing_dependencies_values_are_lists(self):
        """Should have lists of OptionalDependency objects."""
        result = get_all_missing_dependencies()
        for dep_list in result.values():
            assert isinstance(dep_list, list)
            for dep in dep_list:
                assert isinstance(dep, OptionalDependency)
