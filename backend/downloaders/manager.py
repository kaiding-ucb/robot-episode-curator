"""
Download Manager for coordinating dataset downloads.

Manages multiple datasets and their download status.
"""
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import Downloader, DownloadResult, DownloadStatus
from .libero import LiberoDownloader
from .huggingface import HuggingFaceDownloader

logger = logging.getLogger(__name__)


# Dataset registry with download configurations
DATASET_REGISTRY = {
    "libero": {
        "name": "LIBERO",
        "type": "teleop",
        "description": "LIBERO benchmark - 130 manipulation tasks",
        "size_estimate_gb": 10,
        "downloader_class": LiberoDownloader,
        "requires_auth": False,
    },
    "egocentric_10k": {
        "name": "Egocentric-10K",
        "type": "video",
        "description": "10,000 hours of factory egocentric video",
        "size_estimate_gb": 16400,  # 16.4 TB
        "downloader_class": HuggingFaceDownloader,
        "repo_id": "builddotai/Egocentric-10K",
        "requires_auth": True,
        "streaming_recommended": True,
    },
    "realomni": {
        "name": "10Kh RealOmni-Open",
        "type": "teleop",
        "description": "Dual-arm teleoperation data with multiple sensors",
        "size_estimate_gb": 95000,  # 95 TB total
        "downloader_class": HuggingFaceDownloader,
        "repo_id": "genrobot2025/10Kh-RealOmin-OpenData",
        "requires_auth": True,
        "streaming_recommended": True,
    },
    "libero_lerobot": {
        "name": "LIBERO (LeRobot)",
        "type": "teleop",
        "description": "LIBERO benchmark in LeRobot format",
        "size_estimate_gb": 5,
        "downloader_class": HuggingFaceDownloader,
        "repo_id": "HuggingFaceVLA/libero",
        "requires_auth": False,
        "format": "lerobot",
    },
    "ego4d": {
        "name": "Ego4D",
        "type": "video",
        "description": "Egocentric video dataset from Meta",
        "size_estimate_gb": 7000,  # ~7 TB
        "requires_auth": True,
        "requires_license": True,
    },
}


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
        for dataset_id, config in DATASET_REGISTRY.items():
            status = self.get_status(dataset_id)
            datasets.append(
                {
                    "id": dataset_id,
                    "name": config["name"],
                    "type": config["type"],
                    "description": config.get("description", ""),
                    "size_estimate_gb": config.get("size_estimate_gb", 0),
                    "requires_auth": config.get("requires_auth", False),
                    "requires_license": config.get("requires_license", False),
                    "streaming_recommended": config.get("streaming_recommended", False),
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

        if dataset_id not in DATASET_REGISTRY:
            logger.warning(f"Unknown dataset: {dataset_id}")
            return None

        config = DATASET_REGISTRY[dataset_id]
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
        if dataset_id not in DATASET_REGISTRY:
            return {"status": "unknown", "error": f"Unknown dataset: {dataset_id}"}

        config = DATASET_REGISTRY[dataset_id]

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
