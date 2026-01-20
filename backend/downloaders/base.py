"""
Base classes for dataset downloaders.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union


class DownloadStatus(Enum):
    """Status of a dataset download."""

    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"


@dataclass
class DownloadResult:
    """Result of a download operation."""

    success: bool
    path: Optional[Path] = None
    size_bytes: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "path": str(self.path) if self.path else None,
            "size_bytes": self.size_bytes,
            "size_mb": self.size_bytes / (1024 * 1024) if self.size_bytes else 0,
            "error": self.error,
        }


@dataclass
class DownloadProgress:
    """Progress of an ongoing download."""

    total_bytes: int
    downloaded_bytes: int
    status: DownloadStatus
    message: str = ""

    @property
    def percent(self) -> float:
        """Download percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100


class Downloader(ABC):
    """
    Abstract base class for dataset downloaders.

    All downloaders must implement:
    - download(): Download the dataset
    - check_status(): Check if data already exists
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize downloader.

        Args:
            data_dir: Directory to download data to
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(
        self,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        **kwargs,
    ) -> DownloadResult:
        """
        Download the dataset.

        Args:
            progress_callback: Optional callback for progress updates
            **kwargs: Additional download options

        Returns:
            DownloadResult with success status and details
        """
        pass

    @abstractmethod
    def check_status(self) -> Dict[str, Any]:
        """
        Check the current status of the dataset.

        Returns:
            Dictionary with status, size, and other info
        """
        pass

    def get_size_on_disk(self) -> int:
        """Get total size of downloaded data in bytes."""
        if not self.data_dir.exists():
            return 0

        total = 0
        for f in self.data_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
