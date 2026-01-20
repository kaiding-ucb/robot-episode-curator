"""
LeRobot format loader for datasets stored as Parquet files.

LeRobot format (used by HuggingFace robotics datasets) stores data as:
- observation.images.image: dict with 'bytes' and 'path' keys
- observation.state: state vector
- action: action vector
- episode_index, frame_index, timestamp
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image

from .base import DatasetLoader, Episode, EpisodeMetadata

logger = logging.getLogger(__name__)


class LeRobotLoader(DatasetLoader):
    """
    Loader for LeRobot format datasets (Parquet files).

    Used for:
    - HuggingFaceVLA/libero (LIBERO in LeRobot format)
    - physical-intelligence/libero
    - Other LeRobot-compatible datasets
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize LeRobot loader.

        Args:
            data_dir: Path to the dataset directory containing Parquet files
        """
        super().__init__(data_dir)
        self._episode_cache: Dict[str, EpisodeMetadata] = {}
        self._dataframe: Optional[pd.DataFrame] = None
        self._scan_complete = False

    def _load_dataframe(self) -> pd.DataFrame:
        """Load all parquet files into a single DataFrame."""
        if self._dataframe is not None:
            return self._dataframe

        parquet_files = list(self.data_dir.rglob("*.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found in {self.data_dir}")
            return pd.DataFrame()

        # Load all parquet files
        dfs = []
        for pq_file in sorted(parquet_files):
            try:
                df = pd.read_parquet(pq_file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {pq_file}: {e}")

        if not dfs:
            return pd.DataFrame()

        self._dataframe = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(self._dataframe)} frames from {len(parquet_files)} parquet files")
        return self._dataframe

    def _scan_dataset(self) -> None:
        """Scan the dataset to build episode index."""
        if self._scan_complete:
            return

        df = self._load_dataframe()
        if df.empty:
            self._scan_complete = True
            return

        # Group by episode
        episode_col = "episode_index" if "episode_index" in df.columns else "episode_id"
        if episode_col not in df.columns:
            logger.error(f"No episode column found in dataset")
            self._scan_complete = True
            return

        for episode_idx, group in df.groupby(episode_col):
            episode_id = f"episode_{int(episode_idx)}"

            # Get task name if available
            task_name = None
            if "task_index" in group.columns:
                task_idx = group["task_index"].iloc[0]
                task_name = f"Task {task_idx}"

            self._episode_cache[episode_id] = EpisodeMetadata(
                id=episode_id,
                task_name=task_name,
                description=f"LeRobot episode {episode_idx}",
                num_frames=len(group),
                metadata={
                    "episode_index": int(episode_idx),
                    "task_index": int(group["task_index"].iloc[0]) if "task_index" in group.columns else None,
                },
            )

        self._scan_complete = True
        logger.info(f"Found {len(self._episode_cache)} episodes in {self.data_dir}")

    def list_episodes(self) -> List[EpisodeMetadata]:
        """List all available episodes."""
        self._scan_dataset()
        return list(self._episode_cache.values())

    def _decode_image(self, img_data: Any) -> Optional[np.ndarray]:
        """Decode image from LeRobot format."""
        if img_data is None:
            return None

        if isinstance(img_data, dict):
            if "bytes" in img_data and img_data["bytes"]:
                img = Image.open(BytesIO(img_data["bytes"]))
                return np.array(img)
            elif "path" in img_data:
                # Load from path
                try:
                    img = Image.open(img_data["path"])
                    return np.array(img)
                except Exception:
                    return None
        elif isinstance(img_data, bytes):
            img = Image.open(BytesIO(img_data))
            return np.array(img)
        elif isinstance(img_data, np.ndarray):
            return img_data

        return None

    def load_episode(self, episode_id: str) -> Episode:
        """
        Load a complete episode by ID.

        Args:
            episode_id: Episode ID in format "episode_X"

        Returns:
            Episode with observations, actions, and states
        """
        self._scan_dataset()

        if episode_id not in self._episode_cache:
            raise ValueError(f"Episode not found: {episode_id}")

        metadata = self._episode_cache[episode_id]
        episode_idx = metadata.metadata["episode_index"]

        df = self._load_dataframe()
        episode_col = "episode_index" if "episode_index" in df.columns else "episode_id"
        episode_df = df[df[episode_col] == episode_idx].sort_values("frame_index" if "frame_index" in df.columns else "index")

        # Extract observations (images)
        observations = None
        image_cols = [c for c in episode_df.columns if "image" in c.lower()]
        if image_cols:
            # Use first image column
            img_col = image_cols[0]
            frames = []
            for img_data in episode_df[img_col]:
                frame = self._decode_image(img_data)
                if frame is not None:
                    frames.append(frame)
            if frames:
                observations = np.stack(frames, axis=0)

        # Extract actions
        actions = None
        if "action" in episode_df.columns:
            actions = np.stack(episode_df["action"].tolist(), axis=0)

        # Extract states
        states = None
        state_cols = [c for c in episode_df.columns if "state" in c.lower() and "observation" in c.lower()]
        if state_cols:
            state_col = state_cols[0]
            states = np.stack(episode_df[state_col].tolist(), axis=0)

        # Extract timestamps
        timestamps = None
        if "timestamp" in episode_df.columns:
            timestamps = episode_df["timestamp"].values

        return Episode(
            id=episode_id,
            task_name=metadata.task_name,
            description=metadata.description,
            observations=observations,
            actions=actions,
            states=states,
            timestamps=timestamps,
            metadata={
                **metadata.metadata,
                "format": "lerobot",
                "fps": 30,  # LeRobot typically uses 30Hz
            },
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset-level metadata."""
        self._scan_dataset()

        return {
            "data_dir": str(self.data_dir),
            "exists": self.data_dir.exists(),
            "format": "lerobot",
            "num_episodes": len(self._episode_cache),
        }
