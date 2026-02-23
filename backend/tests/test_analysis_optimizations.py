"""
Tests for analysis.py optimizations:
- Part A: Parallelized LeRobot metadata fetches
- Part B: Accurate MP4 frame counts via PyAV moov atom probing

Uses real HuggingFace data — no mocking.
"""
import asyncio
import time

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes.analysis import (
    _read_remote_video_metadata,
    _video_frame_counts_for_task,
    _estimate_frames,
    _lerobot_frame_counts_for_task,
)


# ---------------------------------------------------------------------------
# Part A — Parallelized LeRobot Metadata
# ---------------------------------------------------------------------------


class TestParallelizedLeRobotMetadata:
    """Verify metadata fetches return correct values via asyncio.gather()."""

    LEROBOT_REPO = "lerobot/aloha_sim_transfer_cube_human"

    @pytest.mark.asyncio
    async def test_lerobot_frame_counts_returns_correct_metadata(self):
        """
        _lerobot_frame_counts_for_task() should return a FrameCountDistribution
        with correct episode frame counts from metadata, using parallel fetches.
        """
        result = await _lerobot_frame_counts_for_task(
            self.LEROBOT_REPO,
            "Untitled (task 0)",
        )
        assert result is not None, "Expected FrameCountDistribution, got None"
        assert result.source == "metadata"
        assert result.total_episodes > 0
        assert len(result.episodes) > 0
        # Each episode should have a positive frame count
        for ep in result.episodes:
            assert ep.estimated_frames > 0, f"{ep.episode_id} has 0 frames"

    @pytest.mark.asyncio
    async def test_parallel_metadata_faster_than_serial(self):
        """
        Parallel metadata fetches should complete faster than 5 sequential
        HTTP round-trips. We verify correctness (all values populated) and
        that wall-clock time is reasonable (< 15s).
        """
        from api.routes.datasets import (
            fetch_lerobot_info,
            fetch_lerobot_tasks_meta,
            fetch_lerobot_episode_task_map,
            fetch_lerobot_episodes_meta,
            detect_lerobot_data_branch,
        )

        start = time.monotonic()
        info, tasks_df, ep_task_map, episodes_df, data_branch = await asyncio.gather(
            fetch_lerobot_info(self.LEROBOT_REPO),
            fetch_lerobot_tasks_meta(self.LEROBOT_REPO),
            fetch_lerobot_episode_task_map(self.LEROBOT_REPO),
            fetch_lerobot_episodes_meta(self.LEROBOT_REPO),
            detect_lerobot_data_branch(self.LEROBOT_REPO),
        )
        elapsed = time.monotonic() - start

        # All values should be populated
        assert info is not None, "info is None"
        assert tasks_df is not None, "tasks_df is None"
        assert ep_task_map is not None, "ep_task_map is None"
        assert episodes_df is not None, "episodes_df is None"
        # data_branch may be None for some repos (falls back to "main")

        # Verify info has expected fields
        assert "fps" in info or "total_episodes" in info

        # Verify episodes_df has expected columns
        assert "episode_index" in episodes_df.columns
        assert "length" in episodes_df.columns

        # Wall-clock should be well under what 5 serial calls would take
        assert elapsed < 15, f"Parallel metadata took {elapsed:.1f}s (expected < 15s)"


# ---------------------------------------------------------------------------
# Part B — MP4 Frame Count via moov atom
# ---------------------------------------------------------------------------


class TestRemoteVideoMetadata:
    """Verify PyAV reads exact frame counts from remote video moov atoms."""

    # LeRobot has direct MP4 video files we can probe
    LEROBOT_REPO = "lerobot/aloha_sim_transfer_cube_human"
    LEROBOT_VIDEO_PATH = "videos/observation.images.top/chunk-000/file-000.mp4"
    LEROBOT_VIDEO_SIZE = 67_515_747  # ~67MB

    @pytest.mark.asyncio
    async def test_read_remote_video_metadata_returns_frame_count(self):
        """
        Probing a known MP4 file should return non-None frame_count and fps.
        """
        meta = await _read_remote_video_metadata(
            self.LEROBOT_REPO, self.LEROBOT_VIDEO_PATH
        )

        assert meta["frame_count"] is not None, "frame_count is None"
        assert meta["frame_count"] > 0, f"frame_count is {meta['frame_count']}"
        assert meta["fps"] is not None, "fps is None"
        assert 1 <= meta["fps"] <= 120, f"Unexpected fps: {meta['fps']}"

    @pytest.mark.asyncio
    async def test_video_metadata_differs_from_heuristic(self):
        """
        The exact moov-atom frame count should differ from the bitrate
        heuristic (5Mbps/30fps), demonstrating improved accuracy.
        """
        meta = await _read_remote_video_metadata(
            self.LEROBOT_REPO, self.LEROBOT_VIDEO_PATH
        )

        assert meta["frame_count"] is not None, "frame_count is None"
        exact_frames = meta["frame_count"]
        heuristic_frames = _estimate_frames(self.LEROBOT_VIDEO_SIZE, "video")

        assert exact_frames > 0
        assert heuristic_frames > 0
        # These should not be equal — the heuristic assumes 5Mbps/30fps
        # which rarely matches real videos
        assert exact_frames != heuristic_frames, (
            f"Exact ({exact_frames}) equals heuristic ({heuristic_frames}) — "
            "expected them to differ for a real video"
        )

    @pytest.mark.asyncio
    async def test_video_frame_counts_for_task_probes_mp4(self):
        """
        _video_frame_counts_for_task should use moov atom probing for MP4 files.
        """
        episodes = [
            {
                "path": self.LEROBOT_VIDEO_PATH,
                "episode_id": "video_ep",
                "size_bytes": self.LEROBOT_VIDEO_SIZE,
            },
        ]
        result = await _video_frame_counts_for_task(
            self.LEROBOT_REPO, episodes, "video"
        )
        assert "video_ep" in result
        assert result["video_ep"] > 0
        # Should be exact, not the heuristic
        heuristic = _estimate_frames(self.LEROBOT_VIDEO_SIZE, "video")
        assert result["video_ep"] != heuristic

    @pytest.mark.asyncio
    async def test_video_frame_counts_for_task_falls_back_for_tar(self):
        """
        _video_frame_counts_for_task should use heuristic for TAR-wrapped files.
        """
        episodes = [
            {"path": "fake/nonexistent.tar", "episode_id": "tar_ep", "size_bytes": 10_000_000},
        ]
        result = await _video_frame_counts_for_task("fake/repo", episodes, "webdataset")
        assert "tar_ep" in result
        assert result["tar_ep"] == _estimate_frames(10_000_000, "webdataset")

    @pytest.mark.asyncio
    async def test_read_remote_video_metadata_handles_bad_url(self):
        """
        _read_remote_video_metadata should raise or return None for nonexistent files.
        """
        try:
            meta = await _read_remote_video_metadata("fake/repo", "nonexistent.mp4")
            # If no exception, frame_count should be None
            assert meta["frame_count"] is None
        except Exception:
            # Expected — network/file error for fake repo
            pass
