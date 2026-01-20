"""
RLDS format loader for datasets stored as TFRecord files.

RLDS (Reinforcement Learning Datasets) format stores data as:
- episode_metadata: episode-level info (episode_id, file_path, etc.)
- steps: sequence of (observation, action, reward, discount, is_first, is_last)

Used for:
- Bridge V2
- Open X Embodiment datasets
"""
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from io import BytesIO

import numpy as np
from PIL import Image

from .base import DatasetLoader, Episode, EpisodeMetadata

logger = logging.getLogger(__name__)


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to numpy array."""
    img = Image.open(BytesIO(jpeg_bytes))
    return np.array(img)


class RLDSLoader(DatasetLoader):
    """
    Loader for RLDS format datasets (TFRecord files).

    Used for:
    - Bridge V2 (youliangtan/bridge_dataset)
    - Open X Embodiment datasets
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize RLDS loader.

        Args:
            data_dir: Path to the dataset directory containing TFRecord files
        """
        super().__init__(data_dir)
        self._episode_cache: Dict[str, EpisodeMetadata] = {}
        self._episode_data: Dict[str, Dict] = {}
        self._scan_complete = False
        self._tf = None
        self._tf_train = None
        self._dataset_info = None

    def _get_tf(self):
        """Lazy import tensorflow."""
        if self._tf is None:
            import tensorflow as tf
            self._tf = tf
        return self._tf

    def _load_dataset_info(self) -> Optional[Dict]:
        """Load dataset_info.json if available."""
        if self._dataset_info is not None:
            return self._dataset_info

        # Try to find dataset_info.json
        info_paths = [
            self.data_dir / "dataset_info.json",
            self.data_dir / "1.0.0" / "dataset_info.json",
        ]

        import json
        for info_path in info_paths:
            if info_path.exists():
                try:
                    with open(info_path, "r") as f:
                        self._dataset_info = json.load(f)
                    logger.info(f"Loaded dataset info from {info_path}")
                    return self._dataset_info
                except Exception as e:
                    logger.warning(f"Failed to load dataset_info.json: {e}")

        return None

    def _find_tfrecord_files(self) -> List[Path]:
        """Find all TFRecord files in the data directory."""
        patterns = ["*.tfrecord*", "1.0.0/*.tfrecord*"]
        files = []
        for pattern in patterns:
            files.extend(self.data_dir.glob(pattern))
        return sorted(files)

    def _parse_episode_metadata(self, raw_record: bytes) -> Optional[Dict]:
        """Parse episode metadata only (fast, no image decoding)."""
        tf = self._get_tf()
        if self._tf_train is None:
            self._tf_train = tf.train

        example = self._tf_train.Example()
        example.ParseFromString(raw_record)

        features = example.features.feature

        # Extract episode metadata
        episode_id = None
        if "episode_metadata/episode_id" in features:
            episode_id = features["episode_metadata/episode_id"].int64_list.value[0]

        # Count steps (from is_first or is_last)
        num_steps = 0
        if "steps/is_first" in features:
            num_steps = len(features["steps/is_first"].int64_list.value)

        # Extract language instruction
        language = None
        if "steps/language_instruction" in features:
            lang_bytes = features["steps/language_instruction"].bytes_list.value
            if lang_bytes:
                try:
                    language = lang_bytes[0].decode("utf-8")
                except Exception:
                    pass

        return {
            "episode_id": episode_id,
            "num_steps": num_steps,
            "language": language,
        }

    def _parse_episode_full(self, raw_record: bytes) -> Optional[Dict]:
        """Parse a single TFRecord episode with all data including images."""
        tf = self._get_tf()
        if self._tf_train is None:
            self._tf_train = tf.train

        example = self._tf_train.Example()
        example.ParseFromString(raw_record)

        features = example.features.feature

        # Extract episode metadata
        episode_id = None
        if "episode_metadata/episode_id" in features:
            episode_id = features["episode_metadata/episode_id"].int64_list.value[0]

        # Count steps (from is_first or is_last)
        num_steps = 0
        if "steps/is_first" in features:
            num_steps = len(features["steps/is_first"].int64_list.value)

        # Extract actions
        actions = None
        if "steps/action" in features:
            action_flat = features["steps/action"].float_list.value
            if num_steps > 0:
                action_dim = len(action_flat) // num_steps
                actions = np.array(action_flat).reshape(num_steps, action_dim)

        # Extract states
        states = None
        if "steps/observation/state" in features:
            state_flat = features["steps/observation/state"].float_list.value
            if num_steps > 0:
                state_dim = len(state_flat) // num_steps
                states = np.array(state_flat).reshape(num_steps, state_dim)

        # Extract images (image_0 is the main camera)
        images = []
        if "steps/observation/image_0" in features:
            img_bytes_list = features["steps/observation/image_0"].bytes_list.value
            for img_bytes in img_bytes_list:
                try:
                    img = _decode_jpeg(img_bytes)
                    images.append(img)
                except Exception as e:
                    logger.debug(f"Failed to decode image: {e}")

        # Extract language instruction
        language = None
        if "steps/language_instruction" in features:
            lang_bytes = features["steps/language_instruction"].bytes_list.value
            if lang_bytes:
                try:
                    language = lang_bytes[0].decode("utf-8")
                except Exception:
                    pass

        return {
            "episode_id": episode_id,
            "num_steps": num_steps,
            "actions": actions,
            "states": states,
            "images": np.stack(images, axis=0) if images else None,
            "language": language,
        }

    def _scan_dataset(self) -> None:
        """Scan the dataset to build episode index (fast, metadata only)."""
        if self._scan_complete:
            return

        tfrecord_files = self._find_tfrecord_files()

        if not tfrecord_files:
            logger.warning(f"No TFRecord files found in {self.data_dir}")
            self._scan_complete = True
            return

        # Try to use dataset_info.json for fast metadata (no TFRecord parsing needed)
        dataset_info = self._load_dataset_info()
        if dataset_info and "splits" in dataset_info:
            self._scan_from_metadata(dataset_info, tfrecord_files)
        else:
            # Fallback to slow TFRecord parsing
            self._scan_from_tfrecords(tfrecord_files)

        self._scan_complete = True
        logger.info(f"Found {len(self._episode_cache)} episodes in {self.data_dir}")

    def _scan_from_metadata(self, dataset_info: Dict, tfrecord_files: List[Path]) -> None:
        """Build episode index from dataset_info.json (fast, no TFRecord parsing)."""
        # Get shard lengths from train split
        train_split = None
        for split in dataset_info.get("splits", []):
            if split.get("name") == "train":
                train_split = split
                break

        if not train_split:
            logger.warning("No train split found in dataset_info.json")
            return

        shard_lengths = train_split.get("shardLengths", [])
        logger.info(f"Found {len(shard_lengths)} shards with {sum(int(x) for x in shard_lengths)} total episodes from metadata")

        # Only load episodes from downloaded files (match by shard index)
        # Limit to first 5 episodes across available files for fast loading
        episode_count = 0
        max_episodes = 5

        for tfrecord_path in tfrecord_files:
            # Extract shard index from filename (e.g., "...-00000-of-01024")
            filename = tfrecord_path.name
            try:
                # Parse shard index from filename like "bridge_dataset-train.tfrecord-00000-of-01024"
                parts = filename.split("-")
                shard_idx = int(parts[-3])  # Get the 00000 part
            except (IndexError, ValueError):
                shard_idx = 0

            if shard_idx < len(shard_lengths):
                num_episodes_in_shard = int(shard_lengths[shard_idx])
                episodes_to_add = min(num_episodes_in_shard, max_episodes - episode_count)

                for local_idx in range(episodes_to_add):
                    episode_id = f"bridge_{episode_count}"
                    self._episode_cache[episode_id] = EpisodeMetadata(
                        id=episode_id,
                        task_name="Bridge manipulation",
                        description=f"Bridge V2 episode {episode_count}",
                        num_frames=50,  # Approximate, actual value loaded on demand
                        metadata={
                            "tfrecord_path": str(tfrecord_path),
                            "tfrecord_index": local_idx,
                            "shard_idx": shard_idx,
                        },
                    )
                    episode_count += 1

                    if episode_count >= max_episodes:
                        return

    def _scan_from_tfrecords(self, tfrecord_files: List[Path]) -> None:
        """Scan TFRecords directly (slower fallback)."""
        tf = self._get_tf()
        logger.info(f"Found {len(tfrecord_files)} TFRecord files, scanning...")

        episode_count = 0
        max_episodes_per_file = 5  # Limit for fast scanning
        # Limit to first 1 file for fast scanning
        for tfrecord_path in tfrecord_files[:1]:
            try:
                raw_dataset = tf.data.TFRecordDataset([str(tfrecord_path)])
                local_idx = 0
                for raw_record in raw_dataset.take(max_episodes_per_file):
                    # Use metadata-only parsing for speed
                    parsed = self._parse_episode_metadata(raw_record.numpy())
                    if parsed is None:
                        continue

                    episode_id = f"bridge_{parsed['episode_id'] or episode_count}"
                    self._episode_cache[episode_id] = EpisodeMetadata(
                        id=episode_id,
                        task_name=parsed["language"] or "Bridge manipulation",
                        description=f"Bridge V2 episode {parsed['episode_id'] or episode_count}",
                        num_frames=parsed["num_steps"],
                        metadata={
                            "tfrecord_path": str(tfrecord_path),
                            "tfrecord_index": local_idx,
                            "language": parsed["language"],
                        },
                    )
                    episode_count += 1
                    local_idx += 1

            except Exception as e:
                logger.error(f"Error scanning {tfrecord_path}: {e}")

    def list_episodes(self) -> List[EpisodeMetadata]:
        """List all available episodes."""
        self._scan_dataset()
        return list(self._episode_cache.values())

    def load_episode(self, episode_id: str) -> Episode:
        """
        Load a complete episode by ID.

        Args:
            episode_id: Episode ID in format "bridge_X"

        Returns:
            Episode with observations, actions, and states
        """
        self._scan_dataset()

        if episode_id not in self._episode_cache:
            raise ValueError(f"Episode not found: {episode_id}")

        metadata = self._episode_cache[episode_id]

        # Parse full episode with images from TFRecord
        tf = self._get_tf()
        tfrecord_path = metadata.metadata["tfrecord_path"]
        raw_dataset = tf.data.TFRecordDataset([tfrecord_path])

        target_idx = metadata.metadata.get("tfrecord_index", 0)
        parsed = None
        for i, raw_record in enumerate(raw_dataset):
            if i == target_idx:
                parsed = self._parse_episode_full(raw_record.numpy())
                break

        if parsed is None:
            raise ValueError(f"Could not load episode {episode_id}")

        # Generate timestamps
        timestamps = None
        if parsed["num_steps"] > 0:
            # Bridge uses 5Hz control
            timestamps = np.arange(parsed["num_steps"]) / 5.0

        return Episode(
            id=episode_id,
            task_name=metadata.task_name,
            description=metadata.description,
            observations=parsed["images"],
            actions=parsed["actions"],
            states=parsed["states"],
            timestamps=timestamps,
            metadata={
                **metadata.metadata,
                "format": "rlds",
                "fps": 5,  # Bridge V2 uses 5Hz
            },
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset-level metadata."""
        self._scan_dataset()

        return {
            "data_dir": str(self.data_dir),
            "exists": self.data_dir.exists(),
            "format": "rlds",
            "num_episodes": len(self._episode_cache),
        }
