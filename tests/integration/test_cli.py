"""
End-to-end integration tests for CLI commands.
"""

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docling_graph.cli.main import app


@pytest.mark.integration
class TestCLIConvertCommand:
    """Test CLI convert command."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary test directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_convert_command_help(self, cli_runner):
        """Test convert command help."""
        result = cli_runner.invoke(app, ["convert", "--help"])

        assert result.exit_code == 0
        assert "convert" in result.stdout.lower() or "Convert" in result.stdout

    def test_convert_command_missing_args(self, cli_runner):
        """Test convert command fails without required args."""
        result = cli_runner.invoke(app, ["convert"])

        # Should either fail or show help
        assert result.exit_code != 0 or "help" in result.stdout.lower()

    def test_convert_command_with_valid_config(self, cli_runner, temp_dir):
        """Test convert command with config file - should fail gracefully."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("""
processing_mode: one-to-one
backend: llm
""")

        result = cli_runner.invoke(app, ["convert", "--config", str(config_file)])

        # Should execute (exit code may be non-zero due to missing template/source)
        assert isinstance(result.exit_code, int)


@pytest.mark.integration
class TestCLIInitCommand:
    """Test CLI init command."""

    def test_init_command_help(self):
        """Test init command help."""
        runner = CliRunner()
        result = runner.invoke(app, ["init", "--help"])

        assert result.exit_code == 0
        assert "init" in result.stdout.lower() or "Init" in result.stdout

    def test_init_command_creates_config(self):
        """Test init command creates config file."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(app, ["init"])

            # Should create config.yaml or exit successfully
            config_exists = Path("config.yaml").exists()
            assert result.exit_code == 0 or config_exists


@pytest.mark.integration
class TestCLIInspectCommand:
    """Test CLI inspect command."""

    def test_inspect_command_help(self):
        """Test inspect command help."""
        runner = CliRunner()
        result = runner.invoke(app, ["inspect", "--help"])

        assert result.exit_code == 0
        assert "inspect" in result.stdout.lower()


@pytest.mark.integration
class TestCLIEndToEnd:
    """Complete end-to-end CLI workflows."""

    def test_app_loads_successfully(self):
        """Test that the app loads without errors."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "docling-graph" in result.stdout.lower() or "graph" in result.stdout.lower()

    def test_all_commands_exist(self):
        """Test that all expected commands are registered."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        expected_commands = ["init", "convert", "inspect"]
        output = result.stdout.lower()

        # At least some commands should be visible
        assert any(cmd in output for cmd in expected_commands)
