"""
Multi-Subdataset Adapter for GR00T-style HuggingFace datasets.

Handles repos like nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim that
contain 100+ subdatasets, each being a complete LeRobot dataset with its own
meta/info.json, data/, and videos/ directories.

The adapter presents top-level directories as "tasks" (actually subdatasets),
then delegates to a scoped LeRobotAdapter for the selected subdataset.
"""
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .base import StreamingAdapter, TaskRef, EpisodeRef, FrameResolution

logger = logging.getLogger(__name__)


def _get_hf_headers() -> dict:
    """Get HuggingFace auth headers."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        for token_path in [
            Path.home() / ".huggingface" / "token",
            Path.home() / ".cache" / "huggingface" / "token",
        ]:
            if token_path.exists():
                hf_token = token_path.read_text().strip()
                break
    return {"Authorization": f"Bearer {hf_token}"} if hf_token else {}


class MultiSubdatasetAdapter(StreamingAdapter):
    """
    Adapter for repositories containing multiple subdatasets.

    Each top-level directory is a complete dataset (with its own meta/info.json).
    The adapter lists them as "tasks" and when a subdataset is selected, creates
    a scoped LeRobotAdapter to handle it.

    For GR00T, subdatasets look like:
    - bimanual_panda_gripper.Threading/
    - franka_panda.PickAndPlace/
    Each containing meta/, data/, videos/ subdirectories.
    """

    def __init__(self, repo_id: str, config: Dict[str, Any]):
        super().__init__(repo_id, config)
        self._subdataset_dirs: Optional[List[str]] = None
        self._scoped_adapters: Dict[str, "StreamingAdapter"] = {}

    async def _get_subdataset_dirs(self) -> List[str]:
        """Get list of subdataset directories."""
        if self._subdataset_dirs is not None:
            return self._subdataset_dirs

        headers = _get_hf_headers()
        url = f"https://huggingface.co/api/datasets/{self.repo_id}/tree/main"

        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, headers=headers, timeout=30.0)
            if resp.status_code != 200:
                self._subdataset_dirs = []
                return self._subdataset_dirs

            items = resp.json()
            self._subdataset_dirs = [
                item["path"]
                for item in items
                if item.get("type") == "directory"
                and not item["path"].startswith(".")
                and item["path"] not in ("meta", "data", "videos", ".cache", ".huggingface")
            ]

        return self._subdataset_dirs

    async def list_tasks(self) -> List[TaskRef]:
        """
        List subdatasets as tasks.

        Each subdataset directory becomes a "task" in the UI. The user selects
        one to drill down into its actual tasks/episodes.
        """
        dirs = await self._get_subdataset_dirs()
        return [
            TaskRef(
                name=d,
                description=f"Subdataset: {d}",
            )
            for d in sorted(dirs)
        ]

    async def list_episodes(
        self,
        task: str,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[EpisodeRef], int]:
        """
        List episodes within a subdataset.

        The 'task' parameter here is actually a subdataset name. We need to
        discover the real tasks within it and list episodes from the first
        (or only) task.

        For multi-subdataset repos, the UI flow is:
        1. Select dataset -> shows subdatasets as "tasks"
        2. Select subdataset -> shows episodes directly

        Since each subdataset may have its own task structure, we flatten
        all episodes from all tasks within the subdataset.
        """
        adapter = await self._get_scoped_adapter(task)
        if adapter is None:
            return [], 0

        # Get the real tasks within this subdataset
        real_tasks = await adapter.list_tasks()
        if not real_tasks:
            return [], 0

        # Aggregate episodes from all tasks in this subdataset
        # Prefix episode IDs with subdataset name so they're unambiguous
        # (multiple subdatasets each have episode_0, episode_1, etc.)
        all_episodes: List[EpisodeRef] = []
        for real_task in real_tasks:
            eps, _ = await adapter.list_episodes(real_task.name, limit=1000, offset=0)
            for ep in eps:
                all_episodes.append(EpisodeRef(
                    id=f"{task}/{ep.id}",
                    task_name=ep.task_name,
                    description=ep.description,
                    num_frames=ep.num_frames,
                    duration_seconds=ep.duration_seconds,
                    task_local_index=ep.task_local_index,
                ))

        total_count = len(all_episodes)
        return all_episodes[offset:offset + limit], total_count

    async def resolve_episode(self, episode_id: str) -> Optional[FrameResolution]:
        """
        Resolve an episode from a subdataset.

        Episode IDs are prefixed with the subdataset name:
        e.g., "bimanual_panda_gripper.Threading/episode_0"
        """
        # Parse subdataset prefix from episode_id
        subdataset, base_episode_id = self.parse_episode_id(episode_id)

        if subdataset:
            adapter = await self._get_scoped_adapter(subdataset)
            if adapter:
                return await adapter.resolve_episode(base_episode_id)

        # Fallback: try all cached scoped adapters with the full episode_id
        for prefix, adapter in self._scoped_adapters.items():
            result = await adapter.resolve_episode(episode_id)
            if result:
                return result

        return None

    @staticmethod
    def parse_episode_id(episode_id: str) -> Tuple[Optional[str], str]:
        """
        Parse a multi-subdataset episode_id into (subdataset_prefix, base_episode_id).

        e.g., "bimanual_panda_gripper.Threading/episode_0" -> ("bimanual_panda_gripper.Threading", "episode_0")
              "episode_0" -> (None, "episode_0")
        """
        import re
        # Match: anything/episode_N
        match = re.match(r'^(.+)/(episode_\d+)$', episode_id)
        if match:
            return match.group(1), match.group(2)
        return None, episode_id

    async def _get_scoped_adapter(self, subdataset: str) -> Optional["StreamingAdapter"]:
        """Get or create a scoped LeRobotAdapter for a subdataset."""
        if subdataset in self._scoped_adapters:
            return self._scoped_adapters[subdataset]

        # Verify this subdataset has meta/info.json
        headers = _get_hf_headers()
        info_url = (
            f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/"
            f"{urllib.parse.quote(subdataset, safe='')}/meta/info.json"
        )

        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(info_url, headers=headers, timeout=10.0)
            if resp.status_code != 200:
                logger.warning(f"Subdataset {subdataset} has no meta/info.json")
                return None

        # Create a scoped LeRobotAdapter
        from .lerobot_adapter import LeRobotAdapter

        scoped_config = dict(self.config)
        scoped_config["subdataset_prefix"] = subdataset

        adapter = LeRobotAdapter(
            self.repo_id,
            scoped_config,
            subdataset_prefix=subdataset,
        )
        self._scoped_adapters[subdataset] = adapter
        return adapter

    def get_capabilities(self) -> Dict[str, Any]:
        """Return capabilities (varies by subdataset, return superset)."""
        return {
            "has_video": True,
            "has_actions": True,
            "has_imu": False,
            "has_depth": False,
            "cameras": ["default"],
            "modalities": ["rgb", "actions"],
            "is_multi_subdataset": True,
        }
