"""Tests for PipelineConfig docling-serve settings."""

import pytest

from docling_graph.config import PipelineConfig


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from docling-serve env vars on the host."""
    monkeypatch.delenv("DOCLING_SERVE_URL", raising=False)
    monkeypatch.delenv("DOCLING_SERVE_API_KEY", raising=False)
    monkeypatch.delenv("DOCLING_SERVE_HEADERS", raising=False)


class TestDoclingServeFields:
    def test_defaults_to_local_conversion(self) -> None:
        config = PipelineConfig()
        assert config.docling_serve_url is None
        assert config.docling_serve_api_key is None
        assert config.docling_serve_timeout == 300

    def test_url_is_normalized(self) -> None:
        config = PipelineConfig(docling_serve_url="http://serve:5001/ ")
        assert config.docling_serve_url == "http://serve:5001"

    def test_empty_url_becomes_none(self) -> None:
        config = PipelineConfig(docling_serve_url="")
        assert config.docling_serve_url is None

    def test_invalid_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="http"):
            PipelineConfig(docling_serve_url="serve:5001")


class TestDoclingServeEnvFallback:
    def test_url_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCLING_SERVE_URL", "http://cluster:5001/")
        config = PipelineConfig()
        assert config.docling_serve_url == "http://cluster:5001"

    def test_explicit_url_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCLING_SERVE_URL", "http://cluster:5001")
        config = PipelineConfig(docling_serve_url="http://other:5001")
        assert config.docling_serve_url == "http://other:5001"

    def test_api_key_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCLING_SERVE_API_KEY", "secret")
        config = PipelineConfig(docling_serve_url="http://serve:5001")
        assert config.docling_serve_api_key == "secret"

    def test_api_key_env_ignored_without_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCLING_SERVE_API_KEY", "secret")
        config = PipelineConfig()
        assert config.docling_serve_api_key is None

    def test_headers_fall_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCLING_SERVE_HEADERS", '{"Authorization": "Bearer token"}')
        config = PipelineConfig(docling_serve_url="http://serve:5001")
        assert config.docling_serve_headers == {"Authorization": "Bearer token"}

    def test_explicit_headers_win_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCLING_SERVE_HEADERS", '{"X-Env": "1"}')
        config = PipelineConfig(
            docling_serve_url="http://serve:5001",
            docling_serve_headers={"X-Explicit": "1"},
        )
        assert config.docling_serve_headers == {"X-Explicit": "1"}

    def test_headers_env_ignored_without_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCLING_SERVE_HEADERS", '{"Authorization": "Bearer token"}')
        config = PipelineConfig()
        assert config.docling_serve_headers is None

    def test_invalid_headers_env_fails_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCLING_SERVE_HEADERS", "not json")
        with pytest.raises(ValueError, match="DOCLING_SERVE_HEADERS"):
            PipelineConfig(docling_serve_url="http://serve:5001")

    def test_non_object_headers_env_fails_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCLING_SERVE_HEADERS", '["Authorization"]')
        with pytest.raises(ValueError, match="JSON object"):
            PipelineConfig(docling_serve_url="http://serve:5001")


class TestDoclingServeSerialization:
    def test_to_dict_includes_serve_settings(self) -> None:
        config = PipelineConfig(
            docling_serve_url="http://serve:5001",
            docling_serve_api_key="secret",
            docling_serve_timeout=60,
        )
        data = config.to_dict()
        assert data["docling_serve_url"] == "http://serve:5001"
        assert data["docling_serve_api_key"] == "secret"
        assert data["docling_serve_timeout"] == 60

    def test_metadata_dump_never_contains_api_key(self) -> None:
        config = PipelineConfig(
            template="pydantic.BaseModel",
            docling_serve_url="http://serve:5001",
            docling_serve_api_key="secret",
        )
        data = config.to_metadata_config_dict()
        assert "docling_serve_api_key" not in data
        assert "secret" not in str(data)
        assert data["docling_serve_url"] == "http://serve:5001"

    def test_metadata_dump_never_contains_headers(self) -> None:
        config = PipelineConfig(
            template="pydantic.BaseModel",
            docling_serve_url="http://serve:5001",
            docling_serve_headers={"Authorization": "Bearer token"},
        )
        data = config.to_metadata_config_dict()
        assert "docling_serve_headers" not in data
        assert "Bearer token" not in str(data)

    def test_to_dict_includes_headers(self) -> None:
        config = PipelineConfig(
            docling_serve_url="http://serve:5001",
            docling_serve_headers={"X-Test": "1"},
        )
        assert config.to_dict()["docling_serve_headers"] == {"X-Test": "1"}

    def test_generate_yaml_dict_has_serve_section(self) -> None:
        yaml_dict = PipelineConfig.generate_yaml_dict()
        serve = yaml_dict["docling"]["serve"]
        assert serve["url"] is None
        assert serve["timeout"] == 300
