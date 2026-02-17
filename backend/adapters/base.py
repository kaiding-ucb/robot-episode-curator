"""
Base classes for streaming dataset adapters.

Defines the StreamingAdapter ABC that all format-specific adapters must implement,
plus shared data models (TaskRef, EpisodeRef, FrameResolution).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TaskRef:
    """Lightweight reference to a task within a dataset."""
    name: str
    episode_count: Optional[int] = None
    description: Optional[str] = None


@dataclass
class EpisodeRef:
    """Lightweight reference to an episode within a task."""
    id: str
    task_name: Optional[str] = None
    description: Optional[str] = None
    num_frames: Optional[int] = None
    duration_seconds: Optional[float] = None
    task_local_index: Optional[int] = None


@dataclass
class FrameResolution:
    """Resolution info for extracting frames from an episode."""
    file_format: str  # "video", "mcap", "parquet", etc.
    file_path: str  # Path within the HF repo
    frame_start: int = 0
    frame_end: Optional[int] = None
    num_frames: Optional[int] = None
    fps: int = 30
    video_key: Optional[str] = None
    data_branch: Optional[str] = None
    single_episode_video: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingAdapter(ABC):
    """
    Abstract base class for streaming dataset adapters.

    Each adapter handles a specific dataset format (LeRobot, MCAP, raw video, etc.)
    and provides a unified interface for listing tasks, episodes, and resolving
    frame locations.

    Adapters operate on HuggingFace datasets accessed via the HF API and
    streaming downloads -- they do NOT require local data.
    """

    def __init__(self, repo_id: str, config: Dict[str, Any]):
        """
        Initialize adapter with HF repo ID and dataset config.

        Args:
            repo_id: HuggingFace repository ID (e.g., "user/dataset")
            config: Dataset configuration from the registry
        """
        self.repo_id = repo_id
        self.config = config

    @abstractmethod
    async def list_tasks(self) -> List[TaskRef]:
        """
        List all tasks in the dataset.

        Returns:
            List of TaskRef objects
        """
        ...

    @abstractmethod
    async def list_episodes(
        self,
        task: str,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[EpisodeRef], int]:
        """
        List episodes for a specific task with pagination.

        Args:
            task: Task name to list episodes for
            limit: Maximum number of episodes to return
            offset: Number of episodes to skip

        Returns:
            Tuple of (episode_list, total_count)
        """
        ...

    @abstractmethod
    async def resolve_episode(self, episode_id: str) -> Optional[FrameResolution]:
        """
        Resolve an episode ID to a FrameResolution for frame extraction.

        Args:
            episode_id: Episode identifier

        Returns:
            FrameResolution with file path and frame range, or None if not found
        """
        ...

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return the capabilities of this dataset format.

        Returns:
            Dict with keys like:
                - has_video: bool
                - has_actions: bool
                - has_imu: bool
                - has_depth: bool
                - cameras: List[str]
                - modalities: List[str]
        """
        ...

    async def get_actions_data(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """
        Get action data for an episode.

        Returns:
            Dict with timestamps, actions, dimension_labels, or None if not available
        """
        return None

    async def get_imu_data(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """
        Get IMU data for an episode.

        Returns:
            Dict with timestamps, accel_x/y/z, gyro_x/y/z, or None if not available
        """
        return None

    async def get_modalities(self) -> List[str]:
        """
        Detect available modalities for this dataset.

        Returns:
            List of modality strings (e.g., ["rgb", "depth", "actions"])
        """
        caps = self.get_capabilities()
        return caps.get("modalities", ["rgb"])
