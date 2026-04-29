"""
Generic HuggingFace dataset downloader.

Used for:
- Egocentric-10K (builddotai/Egocentric-10K)
- 10Kh RealOmni-Open (genrobot2025/10Kh-RealOmin-OpenData)
- Other HuggingFace-hosted datasets
"""
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from .base import Downloader, DownloadProgress, DownloadResult, DownloadStatus

logger = logging.getLogger(__name__)


class HuggingFaceDownloader(Downloader):
    """
    Generic downloader for HuggingFace datasets.

    Supports both full download and streaming modes.
    """

    def __init__(
        self,
        repo_id: str,
        data_dir: Optional[Union[str, Path]] = None,
        streaming: bool = True,
    ):
        """
        Initialize HuggingFace downloader.

        Args:
            repo_id: HuggingFace repository ID (e.g., "builddotai/Egocentric-10K")
            data_dir: Directory to download data to (optional for streaming)
            streaming: Use streaming mode (recommended for large datasets)
        """
        if data_dir is None:
            data_dir = Path.home() / ".cache" / "data_viewer" / repo_id.replace("/", "_")

        super().__init__(data_dir)
        self.repo_id = repo_id
        self.streaming = streaming
        self._dataset = None

    def get_dataset(self):
        """
        Get the HuggingFace dataset object.

        Returns a streaming or loaded dataset depending on mode.
        """
        if self._dataset is not None:
            return self._dataset

        try:
            from datasets import load_dataset

            self._dataset = load_dataset(
                self.repo_id,
                streaming=self.streaming,
            )
            return self._dataset

        except Exception as e:
            logger.error(f"Failed to load dataset {self.repo_id}: {e}")
            raise

    def stream(self):
        """
        Get a streaming iterator over the dataset.

        Yields samples one at a time without downloading everything.
        """
        dataset = self.get_dataset()

        # Handle different dataset structures
        if hasattr(dataset, "keys"):
            # Dataset with splits
            split = "train" if "train" in dataset.keys() else list(dataset.keys())[0]
            return iter(dataset[split])
        else:
            return iter(dataset)

    def download(
        self,
        patterns: Optional[list] = None,
        limit: Optional[int] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        **kwargs,
    ) -> DownloadResult:
        """
        Download dataset files.

        Args:
            patterns: File patterns to download (e.g., ["factory_001/**/*.tar"])
            limit: Limit number of files to download
            progress_callback: Callback for progress updates

        Returns:
            DownloadResult with success status
        """
        if self.streaming:
            # In streaming mode, we don't download everything
            # Just verify we can access the dataset
            try:
                self.get_dataset()

                # Try to access first sample to verify
                next(self.stream())

                return DownloadResult(
                    success=True,
                    path=self.data_dir,
                    size_bytes=0,  # Streaming mode, no local files
                )
            except Exception as e:
                return DownloadResult(
                    success=False,
                    error=str(e),
                )

        # Full download mode
        try:
            from huggingface_hub import snapshot_download

            if progress_callback:
                progress_callback(
                    DownloadProgress(
                        total_bytes=0,
                        downloaded_bytes=0,
                        status=DownloadStatus.DOWNLOADING,
                        message=f"Downloading {self.repo_id}...",
                    )
                )

            # Download with patterns if specified
            download_kwargs = {
                "repo_id": self.repo_id,
                "repo_type": "dataset",
                "local_dir": self.data_dir,
            }

            if patterns:
                download_kwargs["allow_patterns"] = patterns

            snapshot_download(**download_kwargs)

            size = self.get_size_on_disk()

            return DownloadResult(
                success=True,
                path=self.data_dir,
                size_bytes=size,
            )

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return DownloadResult(
                success=False,
                error=str(e),
            )

    def check_status(self) -> Dict[str, Any]:
        """
        Check the current status of the dataset.

        For streaming datasets, checks if the dataset is accessible.
        For downloaded datasets, checks local files.
        """
        if self.streaming:
            # For streaming, check if we can access the dataset
            try:
                self.get_dataset()
                return {
                    "status": DownloadStatus.READY.value,
                    "streaming": True,
                    "repo_id": self.repo_id,
                }
            except Exception as e:
                return {
                    "status": DownloadStatus.ERROR.value,
                    "streaming": True,
                    "error": str(e),
                }

        # Check local files
        if not self.data_dir.exists():
            return {
                "status": DownloadStatus.NOT_DOWNLOADED.value,
                "size_bytes": 0,
            }

        size = self.get_size_on_disk()
        if size == 0:
            return {
                "status": DownloadStatus.NOT_DOWNLOADED.value,
                "size_bytes": 0,
            }

        return {
            "status": DownloadStatus.READY.value,
            "size_bytes": size,
            "size_mb": size / (1024 * 1024),
            "path": str(self.data_dir),
        }
