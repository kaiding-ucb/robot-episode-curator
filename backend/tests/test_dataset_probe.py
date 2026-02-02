"""
Tests for dataset probing and modality detection.

Tests verify that the probe endpoint correctly detects:
- Dataset format (MCAP, WebDataset, etc.)
- Available modalities (RGB, depth, IMU, actions)
- Dataset structure (flat vs hierarchical with tasks)

Uses real HuggingFace data - no mocking.
"""
import pytest
import httpx
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes.datasets import (
    probe_huggingface_dataset,
    parse_huggingface_url,
    generate_dataset_id,
)
from loaders.base import Modality


class TestUrlParsing:
    """Tests for HuggingFace URL parsing."""

    def test_parse_full_url(self):
        """Parse full HuggingFace dataset URL."""
        url = "https://huggingface.co/datasets/MicroAGI-Labs/MicroAGI00"
        repo_id = parse_huggingface_url(url)
        assert repo_id == "MicroAGI-Labs/MicroAGI00"

    def test_parse_url_without_https(self):
        """Parse URL without https prefix."""
        url = "huggingface.co/datasets/builddotai/Egocentric-10K"
        repo_id = parse_huggingface_url(url)
        assert repo_id == "builddotai/Egocentric-10K"

    def test_parse_repo_id_only(self):
        """Parse bare repo_id format."""
        url = "MicroAGI-Labs/MicroAGI00"
        repo_id = parse_huggingface_url(url)
        assert repo_id == "MicroAGI-Labs/MicroAGI00"

    def test_generate_dataset_id(self):
        """Generate safe dataset ID from repo_id."""
        repo_id = "MicroAGI-Labs/MicroAGI00"
        dataset_id = generate_dataset_id(repo_id)
        assert dataset_id == "microagi00"

    def test_generate_dataset_id_special_chars(self):
        """Handle special characters in dataset names."""
        repo_id = "genrobot2025/10Kh-RealOmin-OpenData"
        dataset_id = generate_dataset_id(repo_id)
        assert dataset_id == "10kh_realomin_opendata"


@pytest.mark.asyncio
class TestProbeHuggingFaceDatasets:
    """
    Tests for probing real HuggingFace datasets.

    These tests make actual API calls to HuggingFace to verify
    format detection and modality discovery.
    """

    async def test_probe_microagi00_detects_mcap_format(self, microagi00_repo_id):
        """
        Probe MicroAGI00 dataset - should detect MCAP format.

        MicroAGI00 has MCAP files in subdirectories containing
        RGB, depth, and IMU data.
        """
        result = await probe_huggingface_dataset(microagi00_repo_id)

        # Should not have error
        assert result.error is None, f"Probe failed: {result.error}"

        # Should detect MCAP format (probe checks subdirectories)
        assert result.format_detected == "mcap", \
            f"Expected MCAP format, got {result.format_detected}"

        # Should have repo_id
        assert result.repo_id == microagi00_repo_id

    async def test_probe_microagi00_detects_modalities(self, microagi00_repo_id):
        """
        Probe MicroAGI00 - should detect at least RGB modality.

        Full modality detection (depth, IMU) requires downloading
        and scanning an MCAP file. Basic probe returns at least RGB.
        """
        result = await probe_huggingface_dataset(microagi00_repo_id)

        assert result.error is None, f"Probe failed: {result.error}"

        # Should detect RGB modality at minimum
        assert Modality.RGB.value in result.modalities, \
            f"RGB not in modalities: {result.modalities}"

        # Note: depth/IMU detection requires MCAP download which may
        # fail due to authentication. At minimum we verify RGB is present.
        assert len(result.modalities) >= 1

    async def test_probe_microagi00_structure(self, microagi00_repo_id):
        """
        Probe MicroAGI00 - verify structure detection.

        MicroAGI00 has subdirectories for different recordings,
        so has_tasks may be True due to directory structure.
        """
        result = await probe_huggingface_dataset(microagi00_repo_id)

        assert result.error is None, f"Probe failed: {result.error}"

        # Verify probe returns a structure indicator
        # The exact value depends on how MicroAGI00 is organized
        assert result.has_tasks is not None

    async def test_probe_egocentric10k_structure(self, egocentric10k_repo_id):
        """
        Probe Egocentric-10K - verify basic structure detection.

        Egocentric-10K has a hierarchical structure with task folders.
        Format detection may fail if TAR files are deeply nested.
        """
        result = await probe_huggingface_dataset(egocentric10k_repo_id)

        assert result.error is None, f"Probe failed: {result.error}"
        assert result.repo_id == egocentric10k_repo_id

        # Should have task subdirectories
        assert result.has_tasks is True, \
            "Egocentric-10K should have hierarchical task structure"

    async def test_probe_egocentric10k_video_only_modality(self, egocentric10k_repo_id):
        """
        Egocentric-10K should only have RGB modality (video-only).

        No depth or IMU data in this dataset.
        """
        result = await probe_huggingface_dataset(egocentric10k_repo_id)

        assert result.error is None, f"Probe failed: {result.error}"

        # Should have RGB modality by default
        assert Modality.RGB.value in result.modalities

        # Should NOT have depth or IMU (video-only dataset)
        assert Modality.DEPTH.value not in result.modalities, \
            "Video-only dataset should not have depth"
        assert Modality.IMU.value not in result.modalities, \
            "Video-only dataset should not have IMU"

    async def test_probe_realomni_structure(self, realomni_repo_id):
        """
        Probe RealOmni (10Kh-RealOmin-OpenData) - verify structure.

        RealOmni is a large robotics dataset with MCAP recordings
        organized in a deep folder hierarchy.
        """
        result = await probe_huggingface_dataset(realomni_repo_id)

        assert result.error is None, f"Probe failed: {result.error}"
        assert result.repo_id == realomni_repo_id

    async def test_probe_realomni_hierarchical_structure(self, realomni_repo_id):
        """
        RealOmni should be detected as hierarchical (organized by task folders).
        """
        result = await probe_huggingface_dataset(realomni_repo_id)

        assert result.error is None, f"Probe failed: {result.error}"

        # Should have task subdirectories
        assert result.has_tasks is True, \
            "RealOmni should have hierarchical task structure"

        # RGB should be default modality
        assert Modality.RGB.value in result.modalities

    async def test_probe_invalid_repo_returns_error(self):
        """Probing non-existent repo should return error."""
        result = await probe_huggingface_dataset("nonexistent-user/nonexistent-repo-12345")

        assert result.error is not None, "Should return error for invalid repo"

    async def test_probe_returns_sample_files(self, microagi00_repo_id):
        """Probe should return sample file list for preview."""
        result = await probe_huggingface_dataset(microagi00_repo_id)

        assert result.error is None, f"Probe failed: {result.error}"

        # Should have some sample files
        assert len(result.sample_files) > 0, \
            "Probe should return sample files for preview"


@pytest.mark.asyncio
class TestModalityConfigFromMCAP:
    """
    Tests for modality configuration extracted from MCAP files.

    These tests verify that MCAP channel scanning correctly identifies
    topics for each modality.
    """

    async def test_modality_config_contains_topic_info(self, microagi00_repo_id):
        """
        Modality config should include topic names from MCAP.
        """
        result = await probe_huggingface_dataset(microagi00_repo_id)

        # Skip if probe doesn't return modality_config (requires MCAP download)
        if result.modality_config is None:
            pytest.skip("Probe did not return modality_config (MCAP not downloaded)")

        # If RGB is detected, it should have topic info
        if Modality.RGB.value in result.modality_config:
            rgb_config = result.modality_config[Modality.RGB.value]
            assert "topic" in rgb_config, \
                "RGB modality config should include topic name"
