"""
Base classes for dataset loaders.
Defines the common interface that all loaders must implement.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np


@dataclass
class Episode:
    """
    Represents a single episode/trajectory from a robotics dataset.

    For teleop datasets: contains observations, actions, and states.
    For video datasets: contains video data and metadata.
    """

    id: str
    task_name: Optional[str] = None
    description: Optional[str] = None

    # Teleop data (LIBERO, Bridge, etc.)
    observations: Optional[np.ndarray] = None  # (T, H, W, C) images
    actions: Optional[np.ndarray] = None  # (T, action_dim) actions
    states: Optional[np.ndarray] = None  # (T, state_dim) robot states
    timestamps: Optional[np.ndarray] = None  # (T,) timestamps

    # Video data (Ego4D, Egocentric-10K)
    video_path: Optional[Path] = None
    video_bytes: Optional[bytes] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Computed properties
    @property
    def num_frames(self) -> int:
        """Number of frames/timesteps in episode."""
        if self.observations is not None:
            return len(self.observations)
        if self.actions is not None:
            return len(self.actions)
        return 0

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration in seconds (if timestamps available)."""
        if self.timestamps is not None and len(self.timestamps) > 1:
            return float(self.timestamps[-1] - self.timestamps[0])
        # Estimate from fps if available
        fps = self.metadata.get("fps", 30)
        if self.num_frames > 0:
            return self.num_frames / fps
        return None

    def get_frame(self, idx: int) -> Optional[np.ndarray]:
        """Get a single observation frame."""
        if self.observations is not None and 0 <= idx < len(self.observations):
            return self.observations[idx]
        return None

    def get_action(self, idx: int) -> Optional[np.ndarray]:
        """Get a single action."""
        if self.actions is not None and 0 <= idx < len(self.actions):
            return self.actions[idx]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "task_name": self.task_name,
            "description": self.description,
            "num_frames": self.num_frames,
            "duration_seconds": self.duration_seconds,
            "has_observations": self.observations is not None,
            "has_actions": self.actions is not None,
            "has_video": self.video_path is not None or self.video_bytes is not None,
            "metadata": self.metadata,
        }


@dataclass
class EpisodeMetadata:
    """Lightweight metadata about an episode (for listing)."""

    id: str
    task_name: Optional[str] = None
    description: Optional[str] = None
    num_frames: int = 0
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "task_name": self.task_name,
            "description": self.description,
            "num_frames": self.num_frames,
            "duration_seconds": self.duration_seconds,
            **self.metadata,
        }


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.

    All loaders must implement:
    - list_episodes(): List available episodes with metadata
    - load_episode(episode_id): Load full episode data
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize loader with data directory.

        Args:
            data_dir: Path to the dataset directory
        """
        self.data_dir = Path(data_dir)

    @abstractmethod
    def list_episodes(self) -> List[EpisodeMetadata]:
        """
        List all available episodes in the dataset.

        Returns:
            List of EpisodeMetadata objects
        """
        pass

    @abstractmethod
    def load_episode(self, episode_id: str) -> Episode:
        """
        Load a complete episode by ID.

        Args:
            episode_id: Unique identifier for the episode

        Returns:
            Episode object with full data
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset-level metadata.

        Returns:
            Dictionary with dataset info (name, size, etc.)
        """
        return {
            "data_dir": str(self.data_dir),
            "exists": self.data_dir.exists(),
        }


class StreamingLoader(ABC):
    """
    Abstract base class for streaming dataset loaders.

    Used for large datasets that shouldn't be fully loaded into memory.
    """

    def __init__(self, repo_id: str, streaming: bool = True):
        """
        Initialize streaming loader.

        Args:
            repo_id: HuggingFace repository ID or data path
            streaming: Whether to use streaming mode
        """
        self.repo_id = repo_id
        self.streaming = streaming

    @abstractmethod
    def stream_episodes(self) -> Iterator[Episode]:
        """
        Stream episodes one at a time.

        Yields:
            Episode objects
        """
        pass

    def load_episode(self, episode_id: str) -> Optional[Episode]:
        """
        Load a specific episode by ID (may require scanning).

        Args:
            episode_id: Unique identifier for the episode

        Returns:
            Episode if found, None otherwise
        """
        for episode in self.stream_episodes():
            if episode.id == episode_id:
                return episode
        return None
