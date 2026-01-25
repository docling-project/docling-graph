"""Unit tests for OptimizedFileWriter."""

import asyncio
import json
from pathlib import Path

import pytest

from docling_graph.core.utils.io_optimizer import OptimizedFileWriter


class TestOptimizedFileWriter:
    """Tests for OptimizedFileWriter class."""

    def test_writer_initialization(self):
        """Test OptimizedFileWriter initialization."""
        writer = OptimizedFileWriter()
        assert writer is not None
        assert writer.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_write_json_async(self, tmp_path):
        """Test write_json_async method."""
        writer = OptimizedFileWriter()
        test_file = tmp_path / "test.json"
        test_data = {"key": "value", "number": 42}

        await writer.write_json_async(test_file, test_data)

        assert test_file.exists()
        with open(test_file) as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

    @pytest.mark.asyncio
    async def test_write_text_async(self, tmp_path):
        """Test write_text_async method."""
        writer = OptimizedFileWriter()
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!\nThis is a test."

        await writer.write_text_async(test_file, test_content)

        assert test_file.exists()
        with open(test_file) as f:
            loaded_content = f.read()
        assert loaded_content == test_content

    @pytest.mark.asyncio
    async def test_write_json_creates_parent_dirs(self, tmp_path):
        """Test that write_json_async creates parent directories."""
        writer = OptimizedFileWriter()
        test_file = tmp_path / "subdir" / "nested" / "test.json"
        test_data = {"nested": True}

        await writer.write_json_async(test_file, test_data)

        assert test_file.exists()
        assert test_file.parent.exists()

    @pytest.mark.asyncio
    async def test_write_text_creates_parent_dirs(self, tmp_path):
        """Test that write_text_async creates parent directories."""
        writer = OptimizedFileWriter()
        test_file = tmp_path / "subdir" / "nested" / "test.txt"
        test_content = "Nested content"

        await writer.write_text_async(test_file, test_content)

        assert test_file.exists()
        assert test_file.parent.exists()

    def test_queue_write_json(self, tmp_path):
        """Test queue_write method for JSON."""
        writer = OptimizedFileWriter()
        test_file = tmp_path / "queued.json"
        test_data = {"queued": True}

        writer.queue_write(test_file, test_data, "json")

        assert writer.get_pending_count() == 1
        assert not test_file.exists()  # Not written yet

    def test_queue_write_text(self, tmp_path):
        """Test queue_write method for text."""
        writer = OptimizedFileWriter()
        test_file = tmp_path / "queued.txt"
        test_content = "Queued content"

        writer.queue_write(test_file, test_content, "text")

        assert writer.get_pending_count() == 1
        assert not test_file.exists()  # Not written yet

    def test_queue_multiple_writes(self, tmp_path):
        """Test queuing multiple writes."""
        writer = OptimizedFileWriter()

        for i in range(5):
            test_file = tmp_path / f"file_{i}.json"
            writer.queue_write(test_file, {"id": i}, "json")

        assert writer.get_pending_count() == 5

    def test_flush_writes_all_queued(self, tmp_path):
        """Test that flush writes all queued files."""
        writer = OptimizedFileWriter()

        files = []
        for i in range(3):
            test_file = tmp_path / f"file_{i}.json"
            files.append(test_file)
            writer.queue_write(test_file, {"id": i}, "json")

        assert writer.get_pending_count() == 3

        writer.flush()

        assert writer.get_pending_count() == 0
        for test_file in files:
            assert test_file.exists()

    def test_flush_mixed_file_types(self, tmp_path):
        """Test flushing mixed JSON and text files."""
        writer = OptimizedFileWriter()

        json_file = tmp_path / "data.json"
        text_file = tmp_path / "data.txt"

        writer.queue_write(json_file, {"type": "json"}, "json")
        writer.queue_write(text_file, "text content", "text")

        writer.flush()

        assert json_file.exists()
        assert text_file.exists()

        with open(json_file) as f:
            assert json.load(f) == {"type": "json"}
        with open(text_file) as f:
            assert f.read() == "text content"

    def test_flush_empty_queue(self, tmp_path):
        """Test that flushing empty queue doesn't error."""
        writer = OptimizedFileWriter()

        assert writer.get_pending_count() == 0
        writer.flush()  # Should not raise
        assert writer.get_pending_count() == 0

    def test_flush_clears_queue(self, tmp_path):
        """Test that flush clears the queue."""
        writer = OptimizedFileWriter()

        test_file = tmp_path / "test.json"
        writer.queue_write(test_file, {"data": "test"}, "json")

        assert writer.get_pending_count() == 1
        writer.flush()
        assert writer.get_pending_count() == 0

    def test_multiple_flushes(self, tmp_path):
        """Test multiple flush operations."""
        writer = OptimizedFileWriter()

        # First batch
        file1 = tmp_path / "file1.json"
        writer.queue_write(file1, {"batch": 1}, "json")
        writer.flush()
        assert file1.exists()

        # Second batch
        file2 = tmp_path / "file2.json"
        writer.queue_write(file2, {"batch": 2}, "json")
        writer.flush()
        assert file2.exists()

        # Both files should exist
        assert file1.exists()
        assert file2.exists()

    @pytest.mark.asyncio
    async def test_write_json_with_complex_data(self, tmp_path):
        """Test writing complex JSON data."""
        writer = OptimizedFileWriter()
        test_file = tmp_path / "complex.json"

        complex_data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"key": "value", "list": ["a", "b", "c"]},
        }

        await writer.write_json_async(test_file, complex_data)

        with open(test_file) as f:
            loaded = json.load(f)
        assert loaded == complex_data

    @pytest.mark.asyncio
    async def test_write_text_with_unicode(self, tmp_path):
        """Test writing text with unicode characters."""
        writer = OptimizedFileWriter()
        test_file = tmp_path / "unicode.txt"

        unicode_content = "Hello 世界 🌍 Привет مرحبا"

        await writer.write_text_async(test_file, unicode_content)

        with open(test_file, encoding="utf-8") as f:
            loaded = f.read()
        assert loaded == unicode_content

    def test_get_pending_count_accuracy(self, tmp_path):
        """Test that get_pending_count is accurate."""
        writer = OptimizedFileWriter()

        assert writer.get_pending_count() == 0

        writer.queue_write(tmp_path / "file1.json", {}, "json")
        assert writer.get_pending_count() == 1

        writer.queue_write(tmp_path / "file2.json", {}, "json")
        assert writer.get_pending_count() == 2

        writer.flush()
        assert writer.get_pending_count() == 0

    def test_batch_write_performance(self, tmp_path):
        """Test batch writing multiple files."""
        writer = OptimizedFileWriter()

        # Queue many files
        num_files = 50
        for i in range(num_files):
            test_file = tmp_path / f"batch_{i}.json"
            writer.queue_write(test_file, {"id": i, "data": f"content_{i}"}, "json")

        assert writer.get_pending_count() == num_files

        # Flush all at once
        writer.flush()

        # Verify all files were written
        assert writer.get_pending_count() == 0
        for i in range(num_files):
            test_file = tmp_path / f"batch_{i}.json"
            assert test_file.exists()

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, tmp_path):
        """Test concurrent async writes."""
        writer = OptimizedFileWriter()

        # Create multiple write tasks
        tasks = []
        for i in range(10):
            test_file = tmp_path / f"concurrent_{i}.json"
            task = writer.write_json_async(test_file, {"id": i})
            tasks.append(task)

        # Wait for all writes to complete
        await asyncio.gather(*tasks)

        # Verify all files exist
        for i in range(10):
            test_file = tmp_path / f"concurrent_{i}.json"
            assert test_file.exists()
