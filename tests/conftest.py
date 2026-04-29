"""
Shared pytest fixtures for integration tests.

Following TDD principles: no mock data, tests against real functionality.
"""

import os
import sys
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


@pytest.fixture
def test_data_dir(tmp_path):
    """
    Return temporary directory for test data.

    Uses pytest's tmp_path fixture for isolated test directories.
    """
    return tmp_path


@pytest.fixture
def real_data_dir():
    """
    Return path to real downloaded data directory.

    This fixture points to the actual data directory where
    downloaded datasets are stored.
    """
    return Path("./data")


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    from api.main import app
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture
def hf_token():
    """
    Get HuggingFace token from environment.

    Returns None if not set (test will be skipped).
    """
    return os.environ.get("HF_TOKEN")
