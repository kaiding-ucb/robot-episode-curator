"""
Pytest fixtures for backend tests.

Uses real HuggingFace data from datasets specified in CLAUDE.md:
- MicroAGI00: MicroAGI-Labs/MicroAGI00 (MCAP format, RGB+Depth+IMU)
- Egocentric-10K: builddotai/Egocentric-10K (WebDataset, video-only)
- RealOmni: genrobot2025/10Kh-RealOmin-OpenData (MCAP, hierarchical)
"""
import os
import pytest
from pathlib import Path


# HuggingFace token from CLAUDE.md
HF_TOKEN = "REDACTED-HF-TOKEN"


@pytest.fixture(scope="session", autouse=True)
def setup_hf_token():
    """Set up HuggingFace token for API access."""
    os.environ["HF_TOKEN"] = HF_TOKEN
    yield


@pytest.fixture
def microagi00_repo_id():
    """MicroAGI00 dataset - MCAP format with RGB, depth, IMU."""
    return "MicroAGI-Labs/MicroAGI00"


@pytest.fixture
def egocentric10k_repo_id():
    """Egocentric-10K dataset - WebDataset format, video-only."""
    return "builddotai/Egocentric-10K"


@pytest.fixture
def realomni_repo_id():
    """RealOmni dataset - MCAP format, hierarchical with tasks."""
    return "genrobot2025/10Kh-RealOmin-OpenData"


@pytest.fixture
def cache_dir(tmp_path):
    """Temporary cache directory for test downloads."""
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache
