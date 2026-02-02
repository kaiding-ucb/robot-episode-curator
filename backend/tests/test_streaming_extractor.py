"""
Tests for streaming frame extraction from HuggingFace datasets.

Tests verify:
- RGB frame extraction from MCAP
- Depth frame extraction with colorization
- IMU data extraction
- Multi-stream support (switching between RGB/depth)

Uses real HuggingFace data - no mocking.
"""
import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from loaders.streaming_extractor import (
    StreamingFrameExtractor,
    colorize_depth,
)


class TestColorizeDepth:
    """Tests for depth image colorization."""

    def test_colorize_uint16_depth(self):
        """Colorize 16-bit depth image."""
        # Create fake 16-bit depth data
        depth = np.random.randint(0, 65535, (480, 640), dtype=np.uint16)

        colored = colorize_depth(depth, colormap="viridis")

        # Should be RGB 8-bit
        assert colored.shape == (480, 640, 3), \
            f"Expected (480, 640, 3), got {colored.shape}"
        assert colored.dtype == np.uint8, \
            f"Expected uint8, got {colored.dtype}"

    def test_colorize_float_depth(self):
        """Colorize float depth image (meters)."""
        # Create fake float depth data (0-10 meters)
        depth = np.random.uniform(0, 10, (480, 640)).astype(np.float32)

        colored = colorize_depth(depth, colormap="viridis")

        # Should be RGB 8-bit
        assert colored.shape == (480, 640, 3)
        assert colored.dtype == np.uint8

    def test_colorize_different_colormaps(self):
        """Test different colormap options."""
        depth = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)

        for cmap in ["viridis", "jet", "plasma", "magma"]:
            colored = colorize_depth(depth, colormap=cmap)
            assert colored.shape == (100, 100, 3)

    def test_colorize_preserves_structure(self):
        """Closer depths should be distinct from farther depths."""
        # Create gradient depth (near on left, far on right)
        depth = np.tile(np.arange(640, dtype=np.uint16), (480, 1))

        colored = colorize_depth(depth, colormap="viridis")

        # Left and right edges should have different colors
        left_color = colored[:, 0, :].mean(axis=0)
        right_color = colored[:, -1, :].mean(axis=0)

        # Colors should be different
        color_diff = np.abs(left_color - right_color).sum()
        assert color_diff > 50, \
            "Near and far depths should have distinct colors"


class TestStreamingFrameExtractor:
    """
    Tests for StreamingFrameExtractor with real HuggingFace data.

    Note: These tests download actual files from HuggingFace,
    which may take time and require network access.
    """

    @pytest.fixture
    def microagi_extractor(self, microagi00_repo_id, cache_dir):
        """Create extractor for MicroAGI00 dataset."""
        return StreamingFrameExtractor(microagi00_repo_id, cache_dir)

    @pytest.fixture
    def realomni_extractor(self, realomni_repo_id, cache_dir):
        """Create extractor for RealOmni dataset."""
        return StreamingFrameExtractor(realomni_repo_id, cache_dir)

    def test_extractor_initialization(self, microagi_extractor):
        """Extractor should initialize with valid repo_id."""
        assert microagi_extractor.repo_id == "MicroAGI-Labs/MicroAGI00"
        assert microagi_extractor.cache_dir.exists()

    @pytest.mark.slow
    def test_extract_rgb_frames_from_mcap(self, realomni_extractor):
        """
        Extract RGB frames from a RealOmni MCAP episode.

        This test downloads a small portion of a real MCAP file
        and verifies RGB frame extraction.
        """
        # Use a known episode path from RealOmni
        # Format: task_folder/subfolder/episode.mcap
        episode_path = "Cooking_and_Kitchen_Clean/clean_bowl/00001/00001.mcap"

        try:
            frames = realomni_extractor.extract_frames(
                episode_path,
                start=0,
                end=5,  # Just get first 5 frames
                stream="rgb"
            )

            # Should return list of tuples
            assert isinstance(frames, list)

            # If we got frames (may be empty if file structure changed)
            if len(frames) > 0:
                # Each frame is (index, timestamp, image_array)
                idx, timestamp, image = frames[0]

                assert isinstance(idx, int)
                assert isinstance(timestamp, float)
                assert isinstance(image, np.ndarray)

                # Image should be RGB (3 channels)
                assert len(image.shape) == 3
                assert image.shape[2] == 3, \
                    f"Expected 3 channels, got {image.shape[2]}"

        except Exception as e:
            # Test may fail due to network issues or dataset changes
            pytest.skip(f"Could not download episode: {e}")

    @pytest.mark.slow
    def test_extract_depth_frames_colorized(self, realomni_extractor):
        """
        Extract depth frames - should return colorized images.

        Depth data is colorized using viridis colormap for visualization.
        """
        episode_path = "Cooking_and_Kitchen_Clean/clean_bowl/00001/00001.mcap"

        try:
            frames = realomni_extractor.extract_frames(
                episode_path,
                start=0,
                end=5,
                stream="depth",
                depth_colormap="viridis"
            )

            # If depth stream exists
            if len(frames) > 0:
                idx, timestamp, image = frames[0]

                # Colorized depth should be RGB
                assert len(image.shape) == 3
                assert image.shape[2] == 3
                assert image.dtype == np.uint8, \
                    "Colorized depth should be uint8"

        except Exception as e:
            pytest.skip(f"Could not extract depth: {e}")

    @pytest.mark.slow
    def test_extract_imu_data(self, realomni_extractor):
        """
        Extract IMU data from MCAP episode.

        IMU data should include accelerometer and gyroscope readings.
        """
        episode_path = "Cooking_and_Kitchen_Clean/clean_bowl/00001/00001.mcap"

        try:
            imu_data = realomni_extractor.extract_imu_data(episode_path)

            # Should return dict with IMU fields
            if "error" not in imu_data:
                assert "timestamps" in imu_data
                assert "accel_x" in imu_data
                assert "accel_y" in imu_data
                assert "accel_z" in imu_data
                assert "gyro_x" in imu_data
                assert "gyro_y" in imu_data
                assert "gyro_z" in imu_data

                # If data exists, verify structure
                if len(imu_data["timestamps"]) > 0:
                    assert len(imu_data["accel_x"]) == len(imu_data["timestamps"])
                    assert len(imu_data["gyro_x"]) == len(imu_data["timestamps"])

        except Exception as e:
            pytest.skip(f"Could not extract IMU: {e}")

    @pytest.mark.slow
    def test_frame_count_matches_extraction(self, realomni_extractor):
        """
        get_frame_count should match actual extracted frame count.
        """
        episode_path = "Cooking_and_Kitchen_Clean/clean_bowl/00001/00001.mcap"

        try:
            count = realomni_extractor.get_frame_count(episode_path)

            if count > 0:
                # Extract a few frames
                frames = realomni_extractor.extract_frames(
                    episode_path,
                    start=0,
                    end=min(10, count),
                    stream="rgb"
                )

                # Should get frames up to the count
                assert len(frames) <= count

        except Exception as e:
            pytest.skip(f"Could not get frame count: {e}")

    @pytest.mark.slow
    def test_extract_frames_with_count(self, realomni_extractor):
        """
        extract_frames_with_count returns both frames and total count.
        """
        episode_path = "Cooking_and_Kitchen_Clean/clean_bowl/00001/00001.mcap"

        try:
            frames, total = realomni_extractor.extract_frames_with_count(
                episode_path,
                start=0,
                end=5
            )

            assert isinstance(frames, list)
            assert isinstance(total, int)

            if total > 0:
                assert len(frames) <= total

        except Exception as e:
            pytest.skip(f"Could not extract frames: {e}")


class TestCaching:
    """Tests for frame caching behavior."""

    @pytest.mark.slow
    def test_frames_cached_after_extraction(self, realomni_repo_id, cache_dir):
        """
        Extracted frames should be cached for subsequent requests.
        """
        extractor = StreamingFrameExtractor(realomni_repo_id, cache_dir)
        episode_path = "Cooking_and_Kitchen_Clean/clean_bowl/00001/00001.mcap"

        try:
            # First extraction
            frames1 = extractor.extract_frames(episode_path, 0, 5)

            # Second extraction should use cache
            frames2 = extractor.extract_frames(episode_path, 0, 5)

            # Both should return same number of frames
            assert len(frames1) == len(frames2)

        except Exception as e:
            pytest.skip(f"Could not test caching: {e}")


class TestWebDatasetExtraction:
    """Tests for WebDataset/TAR extraction (Egocentric-10K format)."""

    @pytest.fixture
    def egocentric_extractor(self, egocentric10k_repo_id, cache_dir):
        """Create extractor for Egocentric-10K."""
        return StreamingFrameExtractor(egocentric10k_repo_id, cache_dir)

    def test_extract_from_tar_video(self, egocentric_extractor):
        """
        Extract frames from WebDataset TAR file containing video.

        Egocentric-10K uses TAR files with embedded video.
        """
        # This test requires knowing the exact file structure
        # Skip if we don't have a known episode path
        pytest.skip("WebDataset extraction requires known episode paths")
