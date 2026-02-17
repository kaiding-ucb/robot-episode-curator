"""
MCAP Adapter for ROS2-based MCAP datasets on HuggingFace.

Handles datasets like RealOmni/10Kh, MicroAGI, and other MCAP-based repos
where tasks are organized as directory hierarchies and episodes are .mcap files.
"""
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .base import StreamingAdapter, TaskRef, EpisodeRef, FrameResolution

logger = logging.getLogger(__name__)

_NON_TASK_DIRS = {"meta", "data", "videos", ".cache", ".huggingface", "logs", "stats"}


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


class MCAPAdapter(StreamingAdapter):
    """
    Adapter for MCAP (ROS2) datasets on HuggingFace.

    Task hierarchy: Top-level dirs -> Skill subdirs -> Agent dirs -> .mcap files.
    Example: Cooking_and_Kitchen_Clean/clean_bowl/agent_001/00001.mcap
    """

    def __init__(self, repo_id: str, config: Dict[str, Any]):
        super().__init__(repo_id, config)
        self._tree_cache: Optional[List[Dict]] = None

    async def _get_top_level_tree(self) -> List[Dict]:
        """Fetch and cache the top-level tree of the repo."""
        if self._tree_cache is not None:
            return self._tree_cache

        headers = _get_hf_headers()
        url = f"https://huggingface.co/api/datasets/{self.repo_id}/tree/main"
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, headers=headers, timeout=30.0)
            if resp.status_code == 200:
                self._tree_cache = resp.json()
                return self._tree_cache
        self._tree_cache = []
        return self._tree_cache

    async def list_tasks(self) -> List[TaskRef]:
        """
        List tasks by traversing the HF tree API.

        Top-level directories (excluding meta, data, etc.) are treated as
        task categories. If they contain subdirectories with .mcap files,
        those subdirectories are combined into task paths like
        "Cooking_and_Kitchen_Clean/clean_bowl".
        """
        items = await self._get_top_level_tree()
        headers = _get_hf_headers()

        tasks = []
        top_dirs = [
            item["path"]
            for item in items
            if item.get("type") == "directory"
            and not item["path"].startswith(".")
            and item["path"].lower() not in _NON_TASK_DIRS
        ]

        async with httpx.AsyncClient(follow_redirects=True) as client:
            for top_dir in top_dirs:
                # Check if this directory directly contains .mcap files
                # or if it has skill subdirectories
                encoded = urllib.parse.quote(top_dir, safe="")
                sub_url = f"https://huggingface.co/api/datasets/{self.repo_id}/tree/main/{encoded}"
                try:
                    sub_resp = await client.get(sub_url, headers=headers, timeout=15.0)
                    if sub_resp.status_code != 200:
                        tasks.append(TaskRef(name=top_dir))
                        continue

                    sub_items = sub_resp.json()
                    has_mcap = any(
                        i.get("type") == "file" and i["path"].endswith(".mcap")
                        for i in sub_items
                    )
                    sub_dirs = [
                        i["path"]
                        for i in sub_items
                        if i.get("type") == "directory"
                    ]

                    if has_mcap:
                        # This directory directly contains MCAP files
                        mcap_count = sum(
                            1 for i in sub_items
                            if i.get("type") == "file" and i["path"].endswith(".mcap")
                        )
                        tasks.append(TaskRef(
                            name=top_dir,
                            episode_count=mcap_count,
                        ))
                    elif sub_dirs:
                        # Has subdirectories -- each is a skill/task
                        for sub_dir in sub_dirs:
                            relative = sub_dir
                            if relative.startswith(top_dir + "/"):
                                pass  # Already full path
                            tasks.append(TaskRef(name=relative))
                    else:
                        # Empty or non-MCAP directory, still show it
                        tasks.append(TaskRef(name=top_dir))

                except Exception as e:
                    logger.warning(f"Failed to list subdirectory {top_dir}: {e}")
                    tasks.append(TaskRef(name=top_dir))

        return tasks

    async def list_episodes(
        self,
        task: str,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[EpisodeRef], int]:
        """
        List .mcap episode files within a task directory.

        Recursively traverses up to 3 levels deep to find .mcap files.
        Each .mcap file = 1 episode.
        """
        headers = _get_hf_headers()
        encoded = urllib.parse.quote(task, safe="")
        url = f"https://huggingface.co/api/datasets/{self.repo_id}/tree/main/{encoded}"

        all_episodes: List[EpisodeRef] = []
        dirs_to_visit = [task]
        visited = set()

        async with httpx.AsyncClient(follow_redirects=True) as client:
            while dirs_to_visit and len(all_episodes) < offset + limit + 50:
                current_dir = dirs_to_visit.pop(0)
                if current_dir in visited:
                    continue
                visited.add(current_dir)

                depth = current_dir.count("/") - task.count("/")
                if depth > 3:
                    continue

                enc = urllib.parse.quote(current_dir, safe="")
                dir_url = f"https://huggingface.co/api/datasets/{self.repo_id}/tree/main/{enc}"
                try:
                    resp = await client.get(dir_url, headers=headers, timeout=15.0)
                    if resp.status_code != 200:
                        continue
                    items = resp.json()
                except Exception:
                    continue

                for item in items:
                    item_path = item.get("path", "")
                    item_type = item.get("type", "")

                    if item_type == "file" and item_path.endswith(".mcap"):
                        relative_path = item_path
                        size_bytes = item.get("size", 0)
                        estimated_frames = estimate_mcap_frame_count(size_bytes)

                        all_episodes.append(EpisodeRef(
                            id=relative_path,
                            task_name=task,
                            description=f"File: {Path(item_path).name}",
                            num_frames=estimated_frames,
                        ))
                    elif item_type == "directory" and item_path not in visited:
                        dirs_to_visit.append(item_path)

        # Sort for consistent ordering
        all_episodes.sort(key=lambda e: e.id)
        total_count = len(all_episodes)

        return all_episodes[offset:offset + limit], total_count

    async def resolve_episode(self, episode_id: str) -> Optional[FrameResolution]:
        """Resolve an MCAP episode to its file path."""
        return FrameResolution(
            file_format="mcap",
            file_path=episode_id,
            frame_start=0,
            fps=30,
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Return MCAP dataset capabilities."""
        modalities = self.config.get("modalities", ["rgb", "depth", "imu", "actions"])
        return {
            "has_video": True,
            "has_actions": "actions" in modalities,
            "has_imu": "imu" in modalities,
            "has_depth": "depth" in modalities,
            "has_tactile": "tactile" in modalities,
            "cameras": ["camera0", "camera1", "camera2"],
            "modalities": modalities,
        }

    async def get_actions_data(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Extract action data from MCAP episode."""
        if not episode_id.endswith(".mcap"):
            return None

        from loaders.streaming_extractor import StreamingFrameExtractor
        extractor = StreamingFrameExtractor(self.repo_id)
        try:
            result = extractor.extract_actions_data(episode_id)
            return result
        except Exception as e:
            logger.error(f"Failed to extract MCAP actions: {e}")
            return {"error": str(e)}

    async def get_imu_data(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Extract IMU data from MCAP episode."""
        if not episode_id.endswith(".mcap"):
            return None

        from loaders.streaming_extractor import StreamingFrameExtractor
        extractor = StreamingFrameExtractor(self.repo_id)
        try:
            result = extractor.extract_imu_data(episode_id)
            return result
        except Exception as e:
            logger.error(f"Failed to extract MCAP IMU data: {e}")
            return {"error": str(e)}


def estimate_mcap_frame_count(file_size_bytes: int, fps: int = 30) -> Optional[int]:
    """
    Estimate frame count from MCAP file size.

    Heuristic: ~50KB per frame for compressed H.264 at 640x480.
    This is a rough estimate used for UI display before actual decoding.
    """
    if file_size_bytes <= 0:
        return None
    bytes_per_frame = 50 * 1024  # 50KB per frame estimate
    return max(1, file_size_bytes // bytes_per_frame)
