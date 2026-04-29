"""
Raw Video Adapter for HuggingFace datasets containing plain video files.

Handles repos that contain .mp4/.avi/.webm/.mov files organized in folder
hierarchies without LeRobot metadata. Tasks = directories, episodes = video files.
"""
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .base import EpisodeRef, FrameResolution, StreamingAdapter, TaskRef

logger = logging.getLogger(__name__)

_NON_TASK_DIRS = {"meta", "data", "videos", ".cache", ".huggingface", "logs", "stats"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".webm", ".mov", ".mkv"}


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


class RawVideoAdapter(StreamingAdapter):
    """
    Adapter for raw video datasets on HuggingFace.

    Treats directories as tasks and video files as episodes.
    No actions, IMU, or depth -- video-only.
    """

    async def list_tasks(self) -> List[TaskRef]:
        """
        List tasks by grouping video files by parent directory.

        If videos are at the root level, creates a single "default" task.
        """
        headers = _get_hf_headers()
        url = f"https://huggingface.co/api/datasets/{self.repo_id}/tree/main"

        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, headers=headers, timeout=30.0)
            if resp.status_code != 200:
                return []

            items = resp.json()

        # Check for root-level video files
        root_videos = [
            item for item in items
            if item.get("type") == "file"
            and Path(item["path"]).suffix.lower() in _VIDEO_EXTENSIONS
        ]

        dirs = [
            item["path"]
            for item in items
            if item.get("type") == "directory"
            and not item["path"].startswith(".")
            and item["path"].lower() not in _NON_TASK_DIRS
        ]

        tasks = []

        if root_videos:
            tasks.append(TaskRef(
                name="root",
                episode_count=len(root_videos),
                description="Videos in root directory",
            ))

        for d in dirs:
            tasks.append(TaskRef(name=d))

        return tasks

    async def list_episodes(
        self,
        task: str,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[EpisodeRef], int]:
        """List video files within a task directory."""
        headers = _get_hf_headers()
        all_episodes: List[EpisodeRef] = []

        if task == "root":
            # List root-level video files
            url = f"https://huggingface.co/api/datasets/{self.repo_id}/tree/main"
        else:
            encoded = urllib.parse.quote(task, safe="")
            url = f"https://huggingface.co/api/datasets/{self.repo_id}/tree/main/{encoded}"

        dirs_to_visit = [url]
        visited = set()

        async with httpx.AsyncClient(follow_redirects=True) as client:
            while dirs_to_visit and len(all_episodes) < offset + limit + 50:
                current_url = dirs_to_visit.pop(0)
                if current_url in visited:
                    continue
                visited.add(current_url)

                try:
                    resp = await client.get(current_url, headers=headers, timeout=15.0)
                    if resp.status_code != 200:
                        continue
                    items = resp.json()
                except Exception:
                    continue

                for item in items:
                    item_path = item.get("path", "")
                    item_type = item.get("type", "")

                    if item_type == "file":
                        ext = Path(item_path).suffix.lower()
                        if ext in _VIDEO_EXTENSIONS:
                            size_bytes = item.get("size", 0)
                            all_episodes.append(EpisodeRef(
                                id=item_path,
                                task_name=task,
                                description=f"Video: {Path(item_path).name}",
                                num_frames=estimate_video_frame_count(size_bytes),
                            ))
                    elif item_type == "directory":
                        # Only recurse 2 levels deep
                        depth = item_path.count("/") - (task.count("/") if task != "root" else 0)
                        if depth < 2:
                            enc = urllib.parse.quote(item_path, safe="")
                            sub_url = f"https://huggingface.co/api/datasets/{self.repo_id}/tree/main/{enc}"
                            if sub_url not in visited:
                                dirs_to_visit.append(sub_url)

        all_episodes.sort(key=lambda e: e.id)
        total_count = len(all_episodes)

        return all_episodes[offset:offset + limit], total_count

    async def resolve_episode(self, episode_id: str) -> Optional[FrameResolution]:
        """Resolve a video file to its path for frame extraction."""
        ext = Path(episode_id).suffix.lower()
        if ext not in _VIDEO_EXTENSIONS:
            return None

        return FrameResolution(
            file_format=ext.lstrip("."),
            file_path=episode_id,
            frame_start=0,
            fps=30,  # Default, will be detected from video metadata
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Return raw video capabilities (video-only, no actions/IMU)."""
        return {
            "has_video": True,
            "has_actions": False,
            "has_imu": False,
            "has_depth": False,
            "cameras": ["default"],
            "modalities": ["rgb"],
        }


def estimate_video_frame_count(file_size_bytes: int, fps: int = 30) -> Optional[int]:
    """
    Estimate frame count from video file size.

    Heuristic: ~30KB per frame for H.264 at 640x480.
    """
    if file_size_bytes <= 0:
        return None
    bytes_per_frame = 30 * 1024
    return max(1, file_size_bytes // bytes_per_frame)
