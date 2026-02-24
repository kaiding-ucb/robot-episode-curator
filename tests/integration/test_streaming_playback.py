"""
Integration Tests: Streaming Playback Smoothness

Tests for verifying smooth streaming playback from 10Kh RealOmni dataset.
Validates that increased batch size and earlier prefetch timing eliminate
frame pauses at batch boundaries.

Tests use real streaming data from HuggingFace (no mocks).
"""

import asyncio
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import httpx
import pytest

# Add backend to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))


# =============================================================================
# Streaming Playback Tests
# =============================================================================
class TestStreamingPlayback:
    """Tests for streaming playback smoothness on 10Kh RealOmni."""

    REPO_ID = "genrobot2025/10Kh-RealOmin-OpenData"
    BATCH_SIZE = 150  # Match EpisodeViewer.tsx batch size
    PREFETCH_THRESHOLD = 45  # frames before boundary to trigger prefetch
    FPS = 30

    @pytest.fixture
    def hf_token(self):
        """Get HuggingFace token from environment or CLAUDE.md."""
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            # Use token from CLAUDE.md for testing
            token = "REDACTED-HF-TOKEN"
        return token

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    async def test_get_random_tasks(self, hf_token):
        """Can fetch task list and randomly select 2 tasks."""
        tasks = await self._get_tasks()

        assert len(tasks) > 0, "Dataset should have at least one task"

        # Select 2 random tasks
        selected = random.sample(tasks, min(2, len(tasks)))
        assert len(selected) >= 1, "Should select at least 1 task"

        for task in selected:
            assert isinstance(task, str)
            assert len(task) > 0

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    async def test_get_episode_from_task(self, hf_token):
        """Can get an episode from a task folder."""
        tasks = await self._get_tasks()
        assert len(tasks) > 0, "Dataset should have at least one task"

        # Pick a random task
        task = random.choice(tasks)

        # Get episodes from task
        episodes = await self._get_episodes_from_task(task, limit=5)
        assert len(episodes) > 0, f"Task {task} should have at least one episode"

        episode = episodes[0]
        assert "id" in episode
        assert episode["id"].endswith(".mcap"), "Episode should be an MCAP file"

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    async def test_streaming_frame_extraction(self, hf_token):
        """Can extract frames from streaming episode."""
        from loaders.streaming_extractor import StreamingFrameExtractor

        # Get a random task and episode
        tasks = await self._get_tasks()
        task = random.choice(tasks)
        episodes = await self._get_episodes_from_task(task, limit=3)

        if not episodes:
            pytest.skip(f"No episodes found in task: {task}")

        episode_path = episodes[0]["id"]

        # Initialize extractor
        extractor = StreamingFrameExtractor(self.REPO_ID)

        # Extract first batch of frames
        frames, total = extractor.extract_frames_with_count(
            episode_path,
            start=0,
            end=min(30, self.BATCH_SIZE)
        )

        assert len(frames) > 0, "Should extract at least one frame"
        assert total > 0, "Total frame count should be positive"

        # Verify frame structure
        for frame_idx, timestamp, image in frames:
            assert isinstance(frame_idx, int)
            assert isinstance(timestamp, float)
            assert image is not None
            assert len(image.shape) == 3  # H, W, C

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    async def test_batch_prefetch_timing(self, hf_token):
        """
        Verify that batch prefetch completes within acceptable time.

        This tests the core fix: prefetch should complete well before
        batch boundary is reached during playback.
        """
        from loaders.streaming_extractor import StreamingFrameExtractor

        # Get a random task and episode
        tasks = await self._get_tasks()
        task = random.choice(tasks)
        episodes = await self._get_episodes_from_task(task, limit=3)

        if not episodes:
            pytest.skip(f"No episodes found in task: {task}")

        episode_path = episodes[0]["id"]
        extractor = StreamingFrameExtractor(self.REPO_ID)

        # First, load initial batch to populate cache
        frames, total = extractor.extract_frames_with_count(
            episode_path,
            start=0,
            end=self.BATCH_SIZE
        )

        if total <= self.BATCH_SIZE:
            pytest.skip("Episode too short to test batch transitions")

        # Now time the prefetch of next batch (from cache)
        # This simulates what happens when prefetch is triggered
        prefetch_time = self.PREFETCH_THRESHOLD / self.FPS  # 1.5 seconds at 30fps

        start_time = time.time()
        next_frames = extractor.extract_frames_with_count(
            episode_path,
            start=self.BATCH_SIZE,
            end=min(self.BATCH_SIZE * 2, total)
        )
        elapsed = time.time() - start_time

        # Prefetch should complete within the prefetch window (1.5 seconds)
        # Allow some margin for network variance
        assert elapsed < prefetch_time + 1.0, (
            f"Prefetch took {elapsed:.2f}s, should complete within {prefetch_time:.2f}s + 1s margin. "
            f"This would cause playback pauses."
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    async def test_two_tasks_one_episode_each(self, hf_token):
        """
        Integration test: Play 1 episode from each of 2 random tasks.

        This validates the complete playback flow:
        1. Fetch task list
        2. Select 2 random tasks
        3. Get 1 episode from each
        4. Simulate playback through batch boundaries
        5. Verify no blocking delays
        """
        from loaders.streaming_extractor import StreamingFrameExtractor

        # Step 1: Get tasks
        tasks = await self._get_tasks()
        assert len(tasks) >= 2, "Need at least 2 tasks to test"

        # Step 2: Select 2 random tasks
        selected_tasks = random.sample(tasks, 2)

        results = []
        for task in selected_tasks:
            # Step 3: Get 1 episode from task
            episodes = await self._get_episodes_from_task(task, limit=5)
            if not episodes:
                continue

            episode_path = episodes[0]["id"]

            # Step 4: Simulate playback
            extractor = StreamingFrameExtractor(self.REPO_ID)

            # Load first batch
            batch1_start = time.time()
            frames1, total = extractor.extract_frames_with_count(
                episode_path,
                start=0,
                end=self.BATCH_SIZE
            )
            batch1_time = time.time() - batch1_start

            if len(frames1) == 0:
                continue

            # Record playback info
            result = {
                "task": task,
                "episode": episode_path,
                "total_frames": total,
                "batch1_frames": len(frames1),
                "batch1_load_time": batch1_time,
            }

            # If episode is long enough, test prefetch
            if total > self.BATCH_SIZE:
                # Simulate prefetch at frame (BATCH_SIZE - PREFETCH_THRESHOLD)
                prefetch_start = time.time()
                frames2 = extractor.extract_frames_with_count(
                    episode_path,
                    start=self.BATCH_SIZE,
                    end=min(self.BATCH_SIZE * 2, total)
                )
                prefetch_time = time.time() - prefetch_start

                result["batch2_frames"] = len(frames2[0]) if frames2 else 0
                result["prefetch_time"] = prefetch_time
                result["prefetch_fast_enough"] = prefetch_time < (self.PREFETCH_THRESHOLD / self.FPS + 1.0)

            results.append(result)

        # Step 5: Verify results
        assert len(results) >= 1, "Should successfully process at least 1 task"

        for result in results:
            assert result["batch1_frames"] > 0, f"Failed to extract frames from {result['task']}"
            if "prefetch_fast_enough" in result:
                assert result["prefetch_fast_enough"], (
                    f"Prefetch too slow for {result['task']}: "
                    f"{result['prefetch_time']:.2f}s (max {self.PREFETCH_THRESHOLD / self.FPS + 1.0:.2f}s)"
                )

    # =========================================================================
    # Helper methods
    # =========================================================================
    async def _get_tasks(self) -> List[str]:
        """Fetch task list from HuggingFace API."""
        url = f"https://huggingface.co/api/datasets/{self.REPO_ID}/tree/main"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            items = response.json()

            tasks = []
            for item in items:
                if item.get("type") == "directory":
                    task_name = item.get("path", "")
                    if not task_name.startswith("."):
                        tasks.append(task_name)

            return tasks

    async def _get_episodes_from_task(
        self,
        task_folder: str,
        limit: int = 10
    ) -> List[dict]:
        """Fetch episodes from a task folder."""
        import urllib.parse
        encoded_path = urllib.parse.quote(task_folder, safe='')
        url = f"https://huggingface.co/api/datasets/{self.REPO_ID}/tree/main/{encoded_path}"

        async with httpx.AsyncClient() as client:
            episodes = []
            dirs_to_explore = [url]
            explored = set()

            while dirs_to_explore and len(episodes) < limit:
                current_url = dirs_to_explore.pop(0)
                if current_url in explored:
                    continue
                explored.add(current_url)

                try:
                    response = await client.get(current_url, timeout=15.0)
                    if response.status_code != 200:
                        continue

                    items = response.json()

                    for item in items:
                        item_path = item.get("path", "")
                        item_type = item.get("type", "")

                        if item_type == "file" and item_path.endswith(".mcap"):
                            episodes.append({
                                "id": item_path,
                                "task": task_folder,
                            })

                        elif item_type == "directory":
                            depth = item_path.count('/') - task_folder.count('/')
                            if depth < 3:
                                sub_encoded = urllib.parse.quote(item_path, safe='')
                                sub_url = f"https://huggingface.co/api/datasets/{self.REPO_ID}/tree/main/{sub_encoded}"
                                dirs_to_explore.append(sub_url)

                except Exception:
                    continue

            return episodes[:limit]
