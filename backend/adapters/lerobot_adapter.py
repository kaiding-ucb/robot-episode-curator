"""
LeRobot Adapter for LeRobot v2.0/v2.1/v3.0 HuggingFace datasets.

Wraps existing LeRobot metadata-fetching functions from datasets.py
into the StreamingAdapter interface. The original functions remain in
datasets.py for backward compatibility -- this adapter delegates to them.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import StreamingAdapter, TaskRef, EpisodeRef, FrameResolution

logger = logging.getLogger(__name__)


class LeRobotAdapter(StreamingAdapter):
    """
    Adapter for LeRobot v2.0/v2.1/v3.0 datasets on HuggingFace.

    Delegates to the existing fetch_lerobot_* functions in datasets.py
    and resolve_lerobot_episode in episodes.py.
    """

    def __init__(self, repo_id: str, config: Dict[str, Any], subdataset_prefix: str = ""):
        super().__init__(repo_id, config)
        self.subdataset_prefix = subdataset_prefix
        self._info_cache: Optional[dict] = None

    @property
    def _effective_repo_id(self) -> str:
        return self.repo_id

    async def _get_info(self) -> Optional[dict]:
        """Get cached LeRobot info.json (scoped by subdataset_prefix if set)."""
        if self._info_cache is not None:
            return self._info_cache
        from api.routes.datasets import fetch_lerobot_info
        info = await fetch_lerobot_info(
            self._effective_repo_id, path_prefix=self.subdataset_prefix
        )
        self._info_cache = info
        return info

    async def list_tasks(self) -> List[TaskRef]:
        """List tasks using LeRobot metadata parquet files."""
        from api.routes.datasets import (
            fetch_lerobot_tasks_meta,
            fetch_lerobot_episode_task_map,
            fetch_lerobot_info,
        )

        repo_id = self._effective_repo_id
        prefix = self.subdataset_prefix
        tasks_df = await fetch_lerobot_tasks_meta(repo_id, path_prefix=prefix)

        if tasks_df is not None:
            task_col = "task_description"
            if task_col not in tasks_df.columns:
                for col in tasks_df.columns:
                    if col != "task_index":
                        task_col = col
                        break

            # Get episode-task mapping for counts
            ep_task_map = await fetch_lerobot_episode_task_map(repo_id, path_prefix=prefix)
            task_episode_counts: Dict[int, int] = {}
            if ep_task_map:
                for _, task_idx in ep_task_map.items():
                    task_episode_counts[task_idx] = task_episode_counts.get(task_idx, 0) + 1

            tasks = []
            for _, row in tasks_df.iterrows():
                task_idx = int(row["task_index"])
                task_name = str(row[task_col]).strip() if task_col in row.index else ""
                if not task_name:
                    task_name = f"Untitled (task {task_idx})"
                ep_count = task_episode_counts.get(task_idx)
                tasks.append(TaskRef(
                    name=task_name,
                    episode_count=ep_count,
                ))
            return tasks

        # Fallback: use info.json
        info = await self._get_info()
        if info:
            total_tasks = info.get("total_tasks", 1)
            total_episodes = info.get("total_episodes", 0)
            tasks = []
            for t_idx in range(total_tasks):
                ep_count = total_episodes if total_tasks == 1 else None
                tasks.append(TaskRef(
                    name=f"Untitled (task {t_idx})",
                    episode_count=ep_count,
                ))
            return tasks

        return []

    async def list_episodes(
        self,
        task: str,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[EpisodeRef], int]:
        """List episodes for a task with pagination."""
        from api.routes.datasets import (
            fetch_lerobot_tasks_meta,
            fetch_lerobot_episodes_meta,
            fetch_lerobot_episode_task_map,
            fetch_lerobot_info,
        )

        repo_id = self._effective_repo_id
        prefix = self.subdataset_prefix
        episodes_df = await fetch_lerobot_episodes_meta(repo_id, path_prefix=prefix)
        tasks_df = await fetch_lerobot_tasks_meta(repo_id, path_prefix=prefix)
        info = await self._get_info()

        if episodes_df is not None and tasks_df is not None:
            # Map task_name to task_index
            task_col = "task_description"
            if task_col not in tasks_df.columns:
                for col in tasks_df.columns:
                    if col != "task_index":
                        task_col = col
                        break

            task_index = None
            if task_col in tasks_df.columns:
                match = tasks_df[tasks_df[task_col] == task]
                if len(match) > 0:
                    task_index = int(match.iloc[0]["task_index"])

            # Handle "Untitled (task N)" fallback names
            if task_index is None:
                untitled_match = re.match(r"^Untitled \(task (\d+)\)$", task)
                if untitled_match:
                    candidate_idx = int(untitled_match.group(1))
                    if candidate_idx in tasks_df["task_index"].values:
                        task_index = candidate_idx

            if task_index is None:
                return [], 0

            # Get episode->task mapping
            ep_task_map = await fetch_lerobot_episode_task_map(repo_id, path_prefix=prefix)
            if ep_task_map is None:
                return [], 0

            task_ep_indices = sorted([
                ep_idx for ep_idx, t_idx in ep_task_map.items()
                if t_idx == task_index
            ])
            total_count = len(task_ep_indices)

            global_to_local = {ep_idx: local_idx for local_idx, ep_idx in enumerate(task_ep_indices)}

            # Get metadata for these episodes
            task_episodes = episodes_df[episodes_df["episode_index"].isin(task_ep_indices)]
            task_episodes = task_episodes.sort_values("episode_index")

            # Apply pagination
            page = task_episodes.iloc[offset:offset + limit]

            fps = info.get("fps", 30) if info else 30
            episodes = []
            for _, row in page.iterrows():
                ep_idx = int(row["episode_index"])
                length = int(row["length"]) if "length" in row.index else None
                duration = length / fps if length else None
                local_idx = global_to_local[ep_idx]
                episodes.append(EpisodeRef(
                    id=f"episode_{ep_idx}",
                    task_name=task,
                    num_frames=length,
                    duration_seconds=round(duration, 2) if duration else None,
                    task_local_index=local_idx,
                ))
            return episodes, total_count

        # v2.1 fallback: no meta/episodes/ or meta/tasks.parquet
        if info:
            task_index = 0
            untitled_match = re.match(r"^Untitled \(task (\d+)\)$", task)
            if untitled_match:
                task_index = int(untitled_match.group(1))

            total_episodes = info.get("total_episodes", 0)
            total_tasks = info.get("total_tasks", 1)

            if total_tasks <= 1 and task_index == 0:
                ep_start = offset
                ep_end = min(offset + limit, total_episodes)
                episodes = []
                for ep_idx in range(ep_start, ep_end):
                    episodes.append(EpisodeRef(
                        id=f"episode_{ep_idx}",
                        task_name=task,
                        num_frames=None,
                        duration_seconds=None,
                        task_local_index=ep_idx,
                    ))
                return episodes, total_episodes

            # Multi-task without metadata
            ep_task_map = await fetch_lerobot_episode_task_map(repo_id, path_prefix=prefix)
            if ep_task_map:
                task_ep_indices = sorted([
                    ep_idx for ep_idx, t_idx in ep_task_map.items()
                    if t_idx == task_index
                ])
                total_count = len(task_ep_indices)
                global_to_local = {ep_idx: local_idx for local_idx, ep_idx in enumerate(task_ep_indices)}
                page = task_ep_indices[offset:offset + limit]
                episodes = []
                for ep_idx in page:
                    episodes.append(EpisodeRef(
                        id=f"episode_{ep_idx}",
                        task_name=task,
                        num_frames=None,
                        duration_seconds=None,
                        task_local_index=global_to_local[ep_idx],
                    ))
                return episodes, total_count

            # Last resort: no task metadata at all, return all episodes
            # (common for v2.0 subdatasets without episode-task mapping)
            ep_start = offset
            ep_end = min(offset + limit, total_episodes)
            episodes = []
            for ep_idx in range(ep_start, ep_end):
                episodes.append(EpisodeRef(
                    id=f"episode_{ep_idx}",
                    task_name=task,
                    num_frames=None,
                    duration_seconds=None,
                    task_local_index=ep_idx,
                ))
            return episodes, total_episodes

        return [], 0

    async def resolve_episode(self, episode_id: str) -> Optional[FrameResolution]:
        """Resolve a LeRobot episode to video path and frame range."""
        from api.routes.episodes import resolve_lerobot_episode

        result = await resolve_lerobot_episode(
            self._effective_repo_id, episode_id, path_prefix=self.subdataset_prefix
        )
        if result is None:
            return None

        return FrameResolution(
            file_format="video",
            file_path=result["video_path"],
            frame_start=result.get("frame_start", 0),
            frame_end=result.get("frame_end"),
            num_frames=result.get("num_frames"),
            fps=result.get("fps", 30),
            video_key=result.get("video_key"),
            data_branch=result.get("data_branch"),
            single_episode_video=result.get("single_episode_video", False),
            metadata=result,
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Return LeRobot dataset capabilities."""
        modalities = self.config.get("modalities", ["rgb"])
        return {
            "has_video": True,
            "has_actions": "actions" in modalities,
            "has_imu": "imu" in modalities,
            "has_depth": "depth" in modalities,
            "cameras": ["default"],
            "modalities": modalities,
        }

    async def get_actions_data(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Get action data from LeRobot parquet files."""
        from api.routes.episodes import _get_lerobot_actions
        result = await _get_lerobot_actions(
            self._effective_repo_id, episode_id, dataset_id=None,
            path_prefix=self.subdataset_prefix
        )
        if result.error:
            return {"error": result.error}
        return {
            "timestamps": result.timestamps,
            "actions": result.actions,
            "dimension_labels": result.dimension_labels,
        }
