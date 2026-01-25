"""
Shared fixtures for integration tests.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_invoice_markdown():
    """Sample invoice document for testing."""
    return """
    # Invoice
    Invoice Number: INV-2024-001
    Date: November 2, 2024
    Customer: ACME Corp

    Items:
    - Product A: $100.00
    - Product B: $200.00

    Total: $300.00
    """


@pytest.fixture(scope="session")
def sample_config_yaml():
    """Sample YAML configuration."""
    return """
    processing_mode: one-to-one
    backend: llm
    inference: local
    docling_config: ocr
    template: docling_graph.templates.billing_document.BillingDocument
    source: sample.pdf
    output_dir: outputs
    export_format: csv
    reverse_edges: false

    config:
    models:
        llm:
        local:
            provider: ollama
            default_model: llama3.1:8b
        """


@pytest.fixture(scope="session")
def sample_invoice_json():
    """Sample invoice data as JSON."""
    return {
        "invoice_number": "INV-2024-001",
        "date": "2024-11-02",
        "customer": "ACME Corp",
        "items": [{"name": "Product A", "price": 100.00}, {"name": "Product B", "price": 200.00}],
        "total": 300.00,
    }
