"""
WebDataset Loader for Egocentric-10K and similar datasets.

Egocentric-10K uses WebDataset format (TAR files) hosted on HuggingFace.
Structure:
- factory_XXX/workers/worker_YYY/
    - intrinsics.json
    - factoryXXX_workerYYY_partZZ.tar
        - video.mp4
        - metadata.json
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from .base import Episode, EpisodeMetadata, StreamingLoader

logger = logging.getLogger(__name__)


class WebDatasetLoader(StreamingLoader):
    """
    Loader for WebDataset format (Egocentric-10K, etc.).

    Supports both local files and HuggingFace streaming.
    """

    def __init__(
        self,
        source: Union[str, Path],
        streaming: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize WebDataset loader.

        Args:
            source: HuggingFace repo ID or local path
            streaming: Use streaming mode (no full download)
            cache_dir: Local cache directory for downloaded files
        """
        # Handle both Path and string
        if isinstance(source, Path):
            repo_id = str(source)
        else:
            repo_id = source

        super().__init__(repo_id, streaming)
        self.cache_dir = cache_dir
        self._dataset = None

    def _get_hf_dataset(self):
        """Get HuggingFace dataset handle."""
        if self._dataset is not None:
            return self._dataset

        try:
            from datasets import load_dataset

            # Custom features for datasets with schema mismatches
            custom_features = self._get_custom_features()

            # Try loading with custom features if available
            try:
                self._dataset = load_dataset(
                    self.repo_id,
                    streaming=self.streaming,
                    features=custom_features,
                )
            except Exception as e:
                logger.warning(f"Failed with custom features, trying default: {e}")
                self._dataset = load_dataset(
                    self.repo_id,
                    streaming=self.streaming,
                )
            return self._dataset
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset: {e}")
            return None

    def _get_custom_features(self):
        """Get custom features for known datasets with schema issues."""
        from datasets import Features, Value

        # Egocentric-10K has schema mismatch (json field is struct, not string)
        if "Egocentric-10K" in self.repo_id or "egocentric" in self.repo_id.lower():
            return Features({
                'mp4': Value('binary'),
                'json': {
                    'codec': Value('string'),
                    'duration_sec': Value('float64'),
                    'factory_id': Value('string'),
                    'fps': Value('float64'),
                    'height': Value('int64'),
                    'size_bytes': Value('int64'),
                    'video_index': Value('int64'),
                    'width': Value('int64'),
                    'worker_id': Value('string'),
                },
                '__key__': Value('string'),
                '__url__': Value('string'),
            })

        return None

    def _get_local_files(self) -> List[Path]:
        """Get list of local TAR/video files."""
        source_path = Path(self.repo_id)
        if not source_path.exists():
            return []

        files = []
        # Look for TAR files (WebDataset format)
        files.extend(source_path.rglob("*.tar"))
        # Also look for direct video files
        files.extend(source_path.rglob("*.mp4"))
        return sorted(files)

    def stream_episodes(self) -> Iterator[Episode]:
        """
        Stream episodes one at a time.

        For HuggingFace datasets, uses streaming mode.
        For local files, iterates through TAR/video files.
        """
        source_path = Path(self.repo_id)

        if source_path.exists():
            # Local files
            yield from self._stream_local()
        else:
            # HuggingFace dataset
            yield from self._stream_huggingface()

    def _stream_local(self) -> Iterator[Episode]:
        """Stream from local files."""
        source_path = Path(self.repo_id)

        # Try to find video files with metadata
        for video_path in source_path.rglob("*.mp4"):
            # Look for accompanying metadata
            metadata_path = video_path.with_suffix(".json")
            metadata = {}

            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata: {e}")

            # Generate episode ID from path
            relative_path = video_path.relative_to(source_path)
            episode_id = str(relative_path.with_suffix(""))

            yield Episode(
                id=episode_id,
                task_name=metadata.get("task_name"),
                video_path=video_path,
                metadata={
                    "factory_id": metadata.get("factory_id"),
                    "worker_id": metadata.get("worker_id"),
                    "fps": metadata.get("fps", 30),
                    "width": metadata.get("width", 1920),
                    "height": metadata.get("height", 1080),
                    "duration_sec": metadata.get("duration_sec"),
                    **metadata,
                },
            )

    def _stream_huggingface(self) -> Iterator[Episode]:
        """Stream from HuggingFace dataset."""
        dataset = self._get_hf_dataset()
        if dataset is None:
            return

        # Handle different dataset structures
        if hasattr(dataset, "keys"):
            # Dataset with splits
            split = "train" if "train" in dataset.keys() else list(dataset.keys())[0]
            data_iter = iter(dataset[split])
        else:
            data_iter = iter(dataset)

        for idx, sample in enumerate(data_iter):
            # Extract data from sample
            video_bytes = None
            metadata = {}

            # Handle different sample formats
            if isinstance(sample, dict):
                # WebDataset format: {"mp4": bytes, "json": {...}}
                video_bytes = sample.get("mp4") or sample.get("video")

                # Parse JSON metadata if present
                json_data = sample.get("json")
                if isinstance(json_data, (str, bytes)):
                    try:
                        metadata = json.loads(json_data)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                elif isinstance(json_data, dict):
                    metadata = json_data

                # Also check for direct metadata fields
                for key in ["factory_id", "worker_id", "fps", "width", "height"]:
                    if key in sample:
                        metadata[key] = sample[key]

            # Generate episode ID
            episode_id = metadata.get("__key__") or f"sample_{idx:06d}"

            yield Episode(
                id=episode_id,
                video_bytes=video_bytes,
                metadata={
                    "fps": metadata.get("fps", 30),
                    "width": metadata.get("width", 1920),
                    "height": metadata.get("height", 1080),
                    "factory_id": metadata.get("factory_id"),
                    "worker_id": metadata.get("worker_id"),
                    **metadata,
                },
            )

    def list_episodes(self) -> List[EpisodeMetadata]:
        """
        List available episodes (limited for streaming datasets).

        Note: For large streaming datasets, this only returns a sample.
        Uses Hub API for fast enumeration without downloading content.
        """
        source_path = Path(self.repo_id)
        max_to_list = 100  # Don't enumerate entire dataset

        # For local files, use local enumeration
        if source_path.exists():
            episodes = []
            for idx, episode in enumerate(self.stream_episodes()):
                if idx >= max_to_list:
                    break
                episodes.append(
                    EpisodeMetadata(
                        id=episode.id,
                        task_name=episode.task_name,
                        metadata=episode.metadata,
                    )
                )
            return episodes

        # For HuggingFace repos, use Hub API for fast listing
        return self._list_episodes_from_hub(max_to_list)

    def _list_episodes_from_hub(self, max_count: int) -> List[EpisodeMetadata]:
        """List episodes using HuggingFace Hub API (fast, no download)."""
        try:
            from huggingface_hub import list_repo_files

            files = list(list_repo_files(self.repo_id, repo_type='dataset'))

            # Filter to TAR files (each TAR typically contains episode(s))
            tar_files = sorted([f for f in files if f.endswith('.tar')])[:max_count]

            episodes = []
            for tar_path in tar_files:
                # Extract episode ID from TAR path
                # e.g., "factory_001/workers/worker_001/factory001_worker001_part00.tar"
                episode_id = Path(tar_path).stem  # Remove .tar extension

                # Parse metadata from path
                parts = tar_path.split('/')
                factory_id = None
                worker_id = None
                for part in parts:
                    if part.startswith('factory_'):
                        factory_id = part
                    elif part.startswith('worker_'):
                        worker_id = part

                episodes.append(
                    EpisodeMetadata(
                        id=episode_id,
                        task_name=f"{factory_id}/{worker_id}" if factory_id and worker_id else None,
                        metadata={
                            "factory_id": factory_id,
                            "worker_id": worker_id,
                            "tar_path": tar_path,
                        },
                    )
                )

            return episodes

        except Exception as e:
            logger.warning(f"Hub API listing failed, falling back to streaming: {e}")
            # Fallback to streaming (slow but reliable)
            episodes = []
            for idx, episode in enumerate(self.stream_episodes()):
                if idx >= max_count:
                    break
                episodes.append(
                    EpisodeMetadata(
                        id=episode.id,
                        task_name=episode.task_name,
                        metadata=episode.metadata,
                    )
                )
            return episodes

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset-level metadata."""
        return {
            "repo_id": self.repo_id,
            "streaming": self.streaming,
            "type": "webdataset",
        }
