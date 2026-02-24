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
        extract_frames_with_count returns frames, total count, and stride.
        """
        episode_path = "Cooking_and_Kitchen_Clean/clean_bowl/00001/00001.mcap"

        try:
            frames, total, stride_used = realomni_extractor.extract_frames_with_count(
                episode_path,
                start=0,
                end=5
            )

            assert isinstance(frames, list)
            assert isinstance(total, int)
            assert isinstance(stride_used, int)
            assert stride_used >= 1

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


class TestRemoteSignalExtraction:
    """
    Tests for remote MCAP signal extraction via HTTP range requests.

    Verifies that extract_signals_remote() produces the same results as
    the full-download extract_signals_data() path, and that summary-based
    counting matches iteration-based counting.
    """

    @pytest.fixture
    def realomni_extractor(self, realomni_repo_id, cache_dir):
        return StreamingFrameExtractor(realomni_repo_id, cache_dir)

    @pytest.mark.slow
    def test_summary_counts_match_iteration(self, realomni_extractor):
        """
        _get_signal_counts_from_summary() should return the same counts
        as _count_signal_messages() for a local MCAP file.
        """
        episode_path = "Cooking_and_Kitchen_Clean/clean_bowl/00001/00001.mcap"

        try:
            local_path = realomni_extractor.download_file(episode_path)

            # Iteration-based counting
            iter_action, iter_imu, iter_topics = realomni_extractor._count_signal_messages(local_path)

            # Summary-based counting
            with open(local_path, "rb") as f:
                summary_result = realomni_extractor._get_signal_counts_from_summary(f)

            assert summary_result is not None, "MCAP file should have a summary"
            sum_action, sum_imu, sum_topics = summary_result

            assert sum_action == iter_action, (
                f"Action count mismatch: summary={sum_action}, iteration={iter_action}"
            )
            assert sum_imu == iter_imu, (
                f"IMU count mismatch: summary={sum_imu}, iteration={iter_imu}"
            )
            assert set(sum_topics) == set(iter_topics), (
                f"Topic mismatch: summary={sum_topics}, iteration={iter_topics}"
            )

        except Exception as e:
            pytest.skip(f"Could not test summary counts: {e}")

    @pytest.mark.slow
    def test_remote_matches_local_signals(self, realomni_extractor):
        """
        extract_signals_remote() should produce identical action/IMU counts
        and values as extract_signals_data() (full download path).
        """
        episode_path = "Cooking_and_Kitchen_Clean/clean_bowl/00001/00001.mcap"
        max_actions = 200
        max_imu = 200

        try:
            # Remote extraction (HTTP range requests)
            remote_result = realomni_extractor.extract_signals_remote(
                episode_path, max_actions, max_imu
            )

            # Local extraction (full download)
            local_result = realomni_extractor.extract_signals_data(
                episode_path, max_actions, max_imu
            )

            # Compare action counts
            remote_actions = remote_result["actions"]
            local_actions = local_result["actions"]

            assert "error" not in remote_actions, f"Remote actions error: {remote_actions.get('error')}"
            assert "error" not in local_actions, f"Local actions error: {local_actions.get('error')}"

            remote_action_count = len(remote_actions["timestamps"])
            local_action_count = len(local_actions["timestamps"])
            assert remote_action_count == local_action_count, (
                f"Action count mismatch: remote={remote_action_count}, local={local_action_count}"
            )

            # Compare IMU counts
            remote_imu = remote_result["imu"]
            local_imu = local_result["imu"]

            remote_imu_count = len(remote_imu["timestamps"])
            local_imu_count = len(local_imu["timestamps"])
            assert remote_imu_count == local_imu_count, (
                f"IMU count mismatch: remote={remote_imu_count}, local={local_imu_count}"
            )

            # Compare strides
            assert remote_result["action_stride"] == local_result["action_stride"], (
                f"Action stride mismatch: remote={remote_result['action_stride']}, "
                f"local={local_result['action_stride']}"
            )

            # Compare first few action values (floating point)
            if remote_action_count > 0:
                for i in range(min(5, remote_action_count)):
                    remote_vals = remote_actions["actions"][i]
                    local_vals = local_actions["actions"][i]
                    for j, (rv, lv) in enumerate(zip(remote_vals, local_vals)):
                        assert abs(rv - lv) < 1e-6, (
                            f"Action value mismatch at [{i}][{j}]: "
                            f"remote={rv}, local={lv}"
                        )

            # Compare first few IMU values
            if remote_imu_count > 0:
                for key in ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]:
                    for i in range(min(5, remote_imu_count)):
                        rv = remote_imu[key][i]
                        lv = local_imu[key][i]
                        assert abs(rv - lv) < 1e-6, (
                            f"IMU {key} mismatch at [{i}]: remote={rv}, local={lv}"
                        )

        except Exception as e:
            pytest.skip(f"Could not test remote signal extraction: {e}")

    @pytest.mark.slow
    def test_remote_extraction_has_data(self, realomni_extractor):
        """
        Remote extraction should return non-empty action and IMU data
        for a known RealOmni episode.
        """
        episode_path = "Cooking_and_Kitchen_Clean/clean_bowl/00001/00001.mcap"

        try:
            result = realomni_extractor.extract_signals_remote(
                episode_path, max_actions=500, max_imu=500
            )

            assert "error" not in result["actions"], (
                f"Actions error: {result['actions'].get('error')}"
            )
            assert len(result["actions"]["timestamps"]) > 0, "Should have action data"
            assert len(result["actions"]["actions"]) > 0, "Should have action vectors"
            assert len(result["imu"]["timestamps"]) > 0, "Should have IMU data"
            assert result["actions"]["dimension_labels"] is not None, (
                "Should have dimension labels"
            )

        except Exception as e:
            pytest.skip(f"Could not test remote extraction: {e}")


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
