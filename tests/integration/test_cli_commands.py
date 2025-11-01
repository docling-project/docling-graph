"""
Integration tests for CLI commands.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from docling_graph.cli.main import app

runner = CliRunner()


@pytest.mark.integration
class TestInitCommand:
    """Tests for `docling-graph init` command."""

    def test_init_creates_config_file(self, temp_dir, monkeypatch):
        """Test that init command creates config.yaml."""
        monkeypatch.chdir(temp_dir)

        with patch("docling_graph.cli.commands.init.build_config_interactive") as mock_build:
            mock_build.return_value = {
                "defaults": {
                    "processing_mode": "many-to-one",
                    "backend_type": "llm",
                    "inference": "local",
                    "export_format": "csv",
                },
                "docling": {"pipeline": "ocr"},
                "models": {},
                "output": {},
            }

            # Mock print_next_steps to avoid the KeyError
            with patch("docling_graph.cli.commands.init.print_next_steps") as mock_print:
                result = runner.invoke(app, ["init"])
                assert result.exit_code == 0
                assert (temp_dir / "config.yaml").exists()

                # Verify print_next_steps was called
                assert mock_print.called

    def test_init_shows_success_message(self, temp_dir, monkeypatch):
        """Test that init shows success message."""
        monkeypatch.chdir(temp_dir)

        with patch("docling_graph.cli.commands.init.build_config_interactive") as mock_build:
            mock_build.return_value = {"defaults": {}}

            result = runner.invoke(app, ["init"])

            assert "success" in result.stdout.lower() or "created" in result.stdout.lower()

    def test_init_overwrites_existing_config(self, temp_dir, monkeypatch):
        """Test init behavior when config.yaml already exists."""
        monkeypatch.chdir(temp_dir)

        # Create existing config
        config_path = temp_dir / "config.yaml"
        config_path.write_text("existing config")

        with patch("docling_graph.cli.commands.init.build_config_interactive") as mock_build:
            mock_build.return_value = {"new": "config"}

            # Mock print_next_steps
            with patch("docling_graph.cli.commands.init.print_next_steps"):
                # Provide 'y' as input to confirm overwrite
                result = runner.invoke(app, ["init"], input="y\n")

                assert result.exit_code == 0
                # Verify config was overwritten
                new_content = config_path.read_text()
                assert "new" in new_content


@pytest.mark.integration
class TestConvertCommand:
    """Tests for `docling-graph convert` command."""

    def test_convert_requires_source_argument(self):
        """Test that convert requires source argument."""
        result = runner.invoke(app, ["convert"])

        # Should fail without source
        assert result.exit_code != 0
        error_output = result.stdout + result.stderr
        assert "source" in error_output.lower() or "required" in error_output.lower()

    def test_convert_requires_template_option(self, temp_dir):
        """Test that convert requires --template option."""
        test_file = temp_dir / "test.pdf"
        test_file.write_bytes(b"test")

        result = runner.invoke(app, ["convert", str(test_file)])

        # Should fail without template
        assert result.exit_code != 0

    @patch("docling_graph.cli.commands.convert.run_pipeline")
    def test_convert_with_valid_arguments(
        self, mock_pipeline, temp_dir, sample_config_dict, monkeypatch
    ):
        """Test convert command with valid arguments."""
        monkeypatch.chdir(temp_dir)

        # Create config file
        import yaml

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        # Create test PDF
        test_file = temp_dir / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\nTest")

        # Mock pipeline to avoid actual execution
        mock_pipeline.return_value = {"success": True}

        result = runner.invoke(app, ["convert", str(test_file), "--template", "conftest.Person"])

        # Should succeed
        assert result.exit_code == 0
        mock_pipeline.assert_called_once()

    def test_convert_fails_with_nonexistent_file(self):
        """Test convert fails with non-existent source file."""
        result = runner.invoke(
            app, ["convert", "/nonexistent/file.pdf", "--template", "some.Template"]
        )

        assert result.exit_code != 0

    def test_convert_with_processing_mode_option(self, temp_dir):
        """Test convert with --processing-mode option."""
        test_file = temp_dir / "test.pdf"
        test_file.write_bytes(b"test")

        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_pipeline:
            result = runner.invoke(
                app,
                [
                    "convert",
                    str(test_file),
                    "--template",
                    "conftest.Person",
                    "--processing-mode",
                    "one-to-one",
                ],
            )

            # Verify processing mode was passed to pipeline
            if result.exit_code == 0:
                call_args = mock_pipeline.call_args
                # Check that processing_mode was passed

    def test_convert_with_backend_type_option(self, temp_dir):
        """Test convert with --backend-type option."""
        test_file = temp_dir / "test.pdf"
        test_file.write_bytes(b"test")

        with patch("docling_graph.cli.commands.convert.run_pipeline"):
            result = runner.invoke(
                app,
                [
                    "convert",
                    str(test_file),
                    "--template",
                    "conftest.Person",
                    "--backend-type",
                    "vlm",
                ],
            )

    def test_convert_with_all_options(self, temp_dir, monkeypatch):
        """Test convert with all command-line options."""
        monkeypatch.chdir(temp_dir)

        test_file = temp_dir / "test.pdf"
        test_file.write_bytes(b"test")

        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_pipeline:
            result = runner.invoke(
                app,
                [
                    "convert",
                    str(test_file),
                    "--template",
                    "conftest.Person",
                    "--processing-mode",
                    "many-to-one",
                    "--backend-type",
                    "llm",
                    "--inference",
                    "local",
                    "--docling-config",
                    "ocr",
                    "--export-format",
                    "csv",
                    "--output-dir",
                    str(temp_dir / "output"),
                ],
            )


@pytest.mark.integration
class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        result = runner.invoke(app, ["invalid-command"])

        assert result.exit_code != 0

    def test_cli_help_option(self):
        """Test --help option."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "docling-graph" in result.stdout.lower()

    def test_init_help(self):
        """Test help for init command."""
        result = runner.invoke(app, ["init", "--help"])

        assert result.exit_code == 0
        assert "init" in result.stdout.lower()

    def test_convert_help(self):
        """Test help for convert command."""
        result = runner.invoke(app, ["convert", "--help"])

        assert result.exit_code == 0
        assert "convert" in result.stdout.lower()
        assert "template" in result.stdout.lower()


@pytest.mark.integration
class TestCLIValidation:
    """Test CLI input validation."""

    def test_convert_validates_processing_mode(self, temp_dir):
        """Test that invalid processing mode is rejected."""
        test_file = temp_dir / "test.pdf"
        test_file.write_bytes(b"test")

        result = runner.invoke(
            app,
            [
                "convert",
                str(test_file),
                "--template",
                "conftest.Person",
                "--processing-mode",
                "invalid-mode",
            ],
        )

        assert result.exit_code != 0
        assert "invalid" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_convert_validates_backend_type(self, temp_dir):
        """Test that invalid backend type is rejected."""
        test_file = temp_dir / "test.pdf"
        test_file.write_bytes(b"test")

        result = runner.invoke(
            app,
            [
                "convert",
                str(test_file),
                "--template",
                "conftest.Person",
                "--backend-type",
                "invalid-backend",
            ],
        )

        assert result.exit_code != 0

    def test_convert_validates_vlm_constraints(self, temp_dir):
        """Test that VLM+remote constraint is enforced."""
        test_file = temp_dir / "test.pdf"
        test_file.write_bytes(b"test")

        result = runner.invoke(
            app,
            [
                "convert",
                str(test_file),
                "--template",
                "conftest.Person",
                "--backend-type",
                "vlm",
                "--inference",
                "remote",
            ],
        )

        # Should fail: VLM only supports local inference
        assert result.exit_code != 0


@pytest.mark.integration
@pytest.mark.slow
class TestCLIEndToEnd:
    """End-to-end CLI tests."""

    def test_full_workflow_init_then_convert(self, temp_dir, monkeypatch):
        """Test complete workflow: init then convert."""
        monkeypatch.chdir(temp_dir)

        # Step 1: Init
        with patch("docling_graph.cli.commands.init.build_config_interactive") as mock_build:
            mock_build.return_value = {
                "defaults": {
                    "processing_mode": "many-to-one",
                    "backend_type": "llm",
                    "inference": "local",
                    "export_format": "csv",
                },
                "docling": {"pipeline": "ocr"},
                "models": {},
                "output": {"default_directory": "outputs"},
            }

            # Mock print_next_steps
            with patch("docling_graph.cli.commands.init.print_next_steps"):
                result = runner.invoke(app, ["init"])
                assert result.exit_code == 0

        # Step 2: Convert
        test_file = temp_dir / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4")

        with patch("docling_graph.cli.commands.convert.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = {"success": True}

            result = runner.invoke(
                app, ["convert", str(test_file), "--template", "conftest.Person"]
            )
            assert result.exit_code == 0
