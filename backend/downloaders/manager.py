"""
Download Manager for coordinating dataset downloads.

Manages multiple datasets and their download status.
"""
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import Downloader, DownloadResult, DownloadStatus
from .huggingface import HuggingFaceDownloader

logger = logging.getLogger(__name__)


# Path for persisting dynamic datasets
_DYNAMIC_REGISTRY_PATH = Path(os.environ.get(
    "DATA_VIEWER_CONFIG_DIR",
    Path.home() / ".config" / "data_viewer"
)) / "dynamic_datasets.json"

# Dynamic registry for user-added datasets via URL
# Format: { "dataset_id": { ...config... } }
_DYNAMIC_REGISTRY: Dict[str, Dict[str, Any]] = {}


# First-run seed. Only used when no dynamic_datasets.json exists yet, so the
# user lands on a working example instead of an empty sidebar; once seeded it
# behaves like any user-added entry (editable, removable).
_FIRST_RUN_SEED: Dict[str, Dict[str, Any]] = {
    "libero": {
        "name": "Libero (Lerobot)",
        "type": "video",
        "description": "HuggingFace dataset: lerobot/libero",
        "repo_id": "lerobot/libero",
        "format": "lerobot",
        "modalities": ["rgb"],
        "modality_config": None,
        "has_tasks": True,
        "streaming_recommended": True,
        "requires_auth": False,
        "downloader_class": None,
    },
}


def _load_dynamic_registry() -> None:
    """Load dynamic datasets from persistent storage, seeding on first run."""
    global _DYNAMIC_REGISTRY
    if _DYNAMIC_REGISTRY_PATH.exists():
        try:
            with open(_DYNAMIC_REGISTRY_PATH, "r") as f:
                _DYNAMIC_REGISTRY = json.load(f)
            logger.info(f"Loaded {len(_DYNAMIC_REGISTRY)} dynamic datasets from {_DYNAMIC_REGISTRY_PATH}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load dynamic registry: {e}")
            _DYNAMIC_REGISTRY = {}
    else:
        _DYNAMIC_REGISTRY = {k: dict(v) for k, v in _FIRST_RUN_SEED.items()}
        _save_dynamic_registry()
        logger.info(f"Seeded dynamic registry with {len(_DYNAMIC_REGISTRY)} default dataset(s)")


def _save_dynamic_registry() -> None:
    """Save dynamic datasets to persistent storage."""
    try:
        _DYNAMIC_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_DYNAMIC_REGISTRY_PATH, "w") as f:
            json.dump(_DYNAMIC_REGISTRY, f, indent=2)
        logger.info(f"Saved {len(_DYNAMIC_REGISTRY)} dynamic datasets to {_DYNAMIC_REGISTRY_PATH}")
    except IOError as e:
        logger.error(f"Failed to save dynamic registry: {e}")


# Load dynamic registry on module import
_load_dynamic_registry()


def add_dynamic_dataset(dataset_id: str, config: Dict[str, Any]) -> None:
    """
    Add a dataset to the dynamic registry and persist to disk.

    Args:
        dataset_id: Unique identifier for the dataset
        config: Dataset configuration dict containing:
            - name: Display name
            - type: "teleop" or "video"
            - repo_id: HuggingFace repository ID
            - format: "mcap", "webdataset", "lerobot", etc.
            - modalities: List of available modalities ["rgb", "depth", "imu", etc.]
            - modality_config: Dict of modality name to ModalityConfig
            - has_tasks: Whether dataset has task subdirectories
            - streaming_recommended: Whether to use streaming mode
    """
    _DYNAMIC_REGISTRY[dataset_id] = config
    _save_dynamic_registry()
    logger.info(f"Added dynamic dataset: {dataset_id}")


def remove_dynamic_dataset(dataset_id: str) -> bool:
    """
    Remove a dataset from the dynamic registry and persist to disk.

    Args:
        dataset_id: Dataset identifier to remove

    Returns:
        True if removed, False if not found
    """
    if dataset_id in _DYNAMIC_REGISTRY:
        del _DYNAMIC_REGISTRY[dataset_id]
        _save_dynamic_registry()
        logger.info(f"Removed dynamic dataset: {dataset_id}")
        return True
    return False


def get_dynamic_datasets() -> Dict[str, Dict[str, Any]]:
    """Get all dynamically registered datasets."""
    return _DYNAMIC_REGISTRY.copy()


def get_all_datasets() -> Dict[str, Dict[str, Any]]:
    """Get combined registry of static and dynamic datasets."""
    combined = DATASET_REGISTRY.copy()
    combined.update(_DYNAMIC_REGISTRY)
    return combined


# Static dataset registry intentionally empty after the LeRobot pivot —
# all datasets are user-added (dynamic) and fully deletable. The symbol
# remains for backward compatibility with code that imports it.
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {}


class DownloadManager:
    """
    Manages downloads for all supported datasets.
    """

    def __init__(self, data_root: Union[str, Path]):
        """
        Initialize download manager.

        Args:
            data_root: Root directory for all dataset downloads
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._downloaders: Dict[str, Downloader] = {}

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all supported datasets with their info.

        Returns:
            List of dataset info dictionaries
        """
        datasets = []
        # Use combined registry (static + dynamic)
        all_datasets = get_all_datasets()
        for dataset_id, config in all_datasets.items():
            status = self.get_status(dataset_id)
            datasets.append(
                {
                    "id": dataset_id,
                    "name": config["name"],
                    "type": config["type"],
                    "format": config.get("format"),
                    "description": config.get("description", ""),
                    "size_estimate_gb": config.get("size_estimate_gb", 0),
                    "requires_auth": config.get("requires_auth", False),
                    "requires_license": config.get("requires_license", False),
                    "streaming_recommended": config.get("streaming_recommended", False),
                    "modalities": config.get("modalities", ["rgb"]),
                    "has_tasks": config.get("has_tasks", True),
                    "status": status.get("status", "unknown"),
                }
            )
        return datasets

    def get_downloader(self, dataset_id: str) -> Optional[Downloader]:
        """
        Get or create downloader for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Downloader instance or None if not supported
        """
        if dataset_id in self._downloaders:
            return self._downloaders[dataset_id]

        all_datasets = get_all_datasets()
        if dataset_id not in all_datasets:
            logger.warning(f"Unknown dataset: {dataset_id}")
            return None

        config = all_datasets[dataset_id]
        downloader_class = config.get("downloader_class")

        if downloader_class is None:
            logger.warning(f"No downloader configured for: {dataset_id}")
            return None

        data_dir = self.data_root / dataset_id

        if downloader_class == HuggingFaceDownloader:
            # HuggingFace downloader needs repo_id
            repo_id = config.get("repo_id", dataset_id)
            streaming = config.get("streaming_recommended", False)
            downloader = HuggingFaceDownloader(
                repo_id=repo_id,
                data_dir=data_dir,
                streaming=streaming,
            )
        else:
            downloader = downloader_class(data_dir=data_dir)

        self._downloaders[dataset_id] = downloader
        return downloader

    def get_status(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get download status for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Status dictionary with status, size, etc.
        """
        all_datasets = get_all_datasets()
        if dataset_id not in all_datasets:
            return {"status": "unknown", "error": f"Unknown dataset: {dataset_id}"}

        all_datasets[dataset_id]

        # Check if data directory exists
        data_dir = self.data_root / dataset_id
        if not data_dir.exists():
            return {
                "status": DownloadStatus.NOT_DOWNLOADED.value,
                "size_bytes": 0,
                "size_mb": 0,
            }

        # Use downloader to get detailed status if available
        downloader = self.get_downloader(dataset_id)
        if downloader:
            return downloader.check_status()

        # Fallback: check directory size
        size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
        return {
            "status": DownloadStatus.READY.value if size > 0 else DownloadStatus.NOT_DOWNLOADED.value,
            "size_bytes": size,
            "size_mb": size / (1024 * 1024),
        }

    def download(
        self,
        dataset_id: str,
        **kwargs,
    ) -> DownloadResult:
        """
        Download a dataset.

        Args:
            dataset_id: Dataset identifier
            **kwargs: Additional arguments passed to downloader

        Returns:
            DownloadResult with success status
        """
        downloader = self.get_downloader(dataset_id)
        if downloader is None:
            return DownloadResult(
                success=False,
                error=f"No downloader available for: {dataset_id}",
            )

        return downloader.download(**kwargs)

    def check_disk_space(self) -> Dict[str, Any]:
        """
        Check available disk space at data root.

        Returns:
            Dictionary with available and total space
        """
        try:
            usage = shutil.disk_usage(self.data_root)
            return {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "available_bytes": usage.free,
                "total_gb": usage.total / (1024**3),
                "used_gb": usage.used / (1024**3),
                "available_gb": usage.free / (1024**3),
            }
        except Exception as e:
            logger.error(f"Failed to check disk space: {e}")
            return {
                "error": str(e),
                "available_bytes": 0,
                "available_gb": 0,
            }

    def get_data_path(self, dataset_id: str) -> Path:
        """Get the data directory path for a dataset."""
        return self.data_root / dataset_id
