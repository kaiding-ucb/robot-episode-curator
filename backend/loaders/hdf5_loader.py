"""
HDF5 Loader for LIBERO and LIBERO-PRO datasets.

LIBERO stores demonstrations in HDF5 files with the following structure:
- /data/demo_0/obs/agentview_rgb  # RGB images (T, H, W, C)
- /data/demo_0/obs/robot0_eef_pos  # End-effector position
- /data/demo_0/obs/robot0_eef_quat  # End-effector orientation
- /data/demo_0/obs/robot0_gripper_qpos  # Gripper state
- /data/demo_0/actions  # 7-DoF actions
- /data/demo_0/states  # Full state info
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np

from .base import DatasetLoader, Episode, EpisodeMetadata

logger = logging.getLogger(__name__)


class HDF5Loader(DatasetLoader):
    """
    Loader for LIBERO HDF5 demonstration files.

    LIBERO datasets are organized as:
    - libero_spatial/
        - task_name_1.hdf5
        - task_name_2.hdf5
    - libero_object/
        - ...
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize LIBERO HDF5 loader.

        Args:
            data_dir: Path to LIBERO data directory
        """
        super().__init__(data_dir)
        self._episode_cache: Dict[str, EpisodeMetadata] = {}
        self._scan_complete = False

    def _scan_dataset(self) -> None:
        """Scan the data directory to find all HDF5 files."""
        if self._scan_complete:
            return

        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            self._scan_complete = True
            return

        # Find all HDF5 files
        for hdf5_path in self.data_dir.rglob("*.hdf5"):
            # Extract task name from path
            relative_path = hdf5_path.relative_to(self.data_dir)
            task_suite = relative_path.parent.name if relative_path.parent != Path(".") else "default"
            task_name = hdf5_path.stem

            # Count demos in file
            try:
                with h5py.File(hdf5_path, "r") as f:
                    if "data" not in f:
                        continue

                    for demo_key in f["data"].keys():
                        if not demo_key.startswith("demo_"):
                            continue

                        demo_group = f["data"][demo_key]
                        episode_id = f"{task_suite}/{task_name}/{demo_key}"

                        # Get metadata
                        num_frames = 0
                        if "actions" in demo_group:
                            num_frames = len(demo_group["actions"])

                        self._episode_cache[episode_id] = EpisodeMetadata(
                            id=episode_id,
                            task_name=task_name.replace("_", " ").title(),
                            description=f"{task_suite}: {task_name}",
                            num_frames=num_frames,
                            metadata={
                                "hdf5_path": str(hdf5_path),
                                "demo_key": demo_key,
                                "task_suite": task_suite,
                            },
                        )
            except Exception as e:
                logger.error(f"Error scanning {hdf5_path}: {e}")

        self._scan_complete = True
        logger.info(f"Found {len(self._episode_cache)} episodes in {self.data_dir}")

    def list_episodes(self) -> List[EpisodeMetadata]:
        """List all available episodes."""
        self._scan_dataset()
        return list(self._episode_cache.values())

    def load_episode(self, episode_id: str) -> Episode:
        """
        Load a complete episode by ID.

        Args:
            episode_id: Episode ID in format "task_suite/task_name/demo_X"

        Returns:
            Episode with observations, actions, and states
        """
        self._scan_dataset()

        if episode_id not in self._episode_cache:
            raise ValueError(f"Episode not found: {episode_id}")

        metadata = self._episode_cache[episode_id]
        hdf5_path = Path(metadata.metadata["hdf5_path"])
        demo_key = metadata.metadata["demo_key"]

        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_key]

            # Load observations (images)
            observations = None
            obs_group = demo.get("obs", {})

            # Try different observation keys (LIBERO uses agentview_rgb)
            for obs_key in ["agentview_rgb", "agentview_image", "image", "rgb"]:
                if obs_key in obs_group:
                    observations = np.array(obs_group[obs_key])
                    break

            # If no single image key, try to find any RGB observation
            if observations is None:
                for key in obs_group.keys():
                    if "rgb" in key.lower() or "image" in key.lower():
                        observations = np.array(obs_group[key])
                        break

            # Load actions (7-DoF)
            actions = None
            if "actions" in demo:
                actions = np.array(demo["actions"])

            # Load states
            states = None
            if "states" in demo:
                states = np.array(demo["states"])

            # Generate timestamps (LIBERO doesn't store explicit timestamps)
            timestamps = None
            if actions is not None:
                # Assume 20Hz control frequency (common for LIBERO)
                timestamps = np.arange(len(actions)) / 20.0

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
                    "fps": 20,  # LIBERO typically runs at 20Hz
                },
            )

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset-level metadata."""
        self._scan_dataset()

        return {
            "data_dir": str(self.data_dir),
            "exists": self.data_dir.exists(),
            "num_episodes": len(self._episode_cache),
            "task_suites": list(
                set(m.metadata.get("task_suite", "unknown") for m in self._episode_cache.values())
            ),
        }
