"""
API routes for download operations.

Endpoints:
- GET /api/downloads/status - Get status of all downloads
- POST /api/downloads/{dataset_id} - Start download
- GET /api/downloads/disk-space - Check available disk space
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel

from downloaders.manager import DownloadManager

logger = logging.getLogger(__name__)

router = APIRouter()


class DownloadStatusResponse(BaseModel):
    """Download status for a dataset."""

    dataset_id: str
    status: str
    size_bytes: int = 0
    size_mb: float = 0
    error: Optional[str] = None


class DiskSpaceResponse(BaseModel):
    """Disk space information."""

    total_gb: float
    used_gb: float
    available_gb: float


class DownloadRequest(BaseModel):
    """Request to start a download."""

    dataset: Optional[str] = None  # Specific subset (e.g., "libero_spatial")
    limit: Optional[int] = None  # Limit files for testing


def get_data_root(request: Request) -> Path:
    """Get data root from app state."""
    return request.app.state.data_root


@router.get("/status", response_model=List[DownloadStatusResponse])
async def get_all_download_status(request: Request):
    """
    Get download status for all datasets.
    """
    data_root = get_data_root(request)
    manager = DownloadManager(data_root)

    statuses = []
    for ds in manager.list_datasets():
        status = manager.get_status(ds["id"])
        statuses.append(
            DownloadStatusResponse(
                dataset_id=ds["id"],
                status=status.get("status", "unknown"),
                size_bytes=status.get("size_bytes", 0),
                size_mb=status.get("size_mb", 0),
                error=status.get("error"),
            )
        )

    return statuses


@router.get("/status/{dataset_id}", response_model=DownloadStatusResponse)
async def get_download_status(dataset_id: str, request: Request):
    """
    Get download status for a specific dataset.
    """
    data_root = get_data_root(request)
    manager = DownloadManager(data_root)
    status = manager.get_status(dataset_id)

    return DownloadStatusResponse(
        dataset_id=dataset_id,
        status=status.get("status", "unknown"),
        size_bytes=status.get("size_bytes", 0),
        size_mb=status.get("size_mb", 0),
        error=status.get("error"),
    )


@router.post("/{dataset_id}")
async def start_download(
    dataset_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    download_request: Optional[DownloadRequest] = None,
):
    """
    Start downloading a dataset.

    Downloads run in the background and status can be checked via GET /status.
    """
    data_root = get_data_root(request)
    manager = DownloadManager(data_root)

    # Check if download is already in progress
    status = manager.get_status(dataset_id)
    if status.get("status") == "downloading":
        raise HTTPException(
            status_code=409,
            detail=f"Download already in progress for: {dataset_id}",
        )

    # Start download in background
    download_kwargs = {}
    if download_request:
        if download_request.dataset:
            download_kwargs["dataset"] = download_request.dataset
        if download_request.limit:
            download_kwargs["limit"] = download_request.limit

    def do_download():
        try:
            result = manager.download(dataset_id, **download_kwargs)
            if not result.success:
                logger.error(f"Download failed: {result.error}")
        except Exception as e:
            logger.error(f"Download error: {e}")

    background_tasks.add_task(do_download)

    return {
        "message": f"Download started for: {dataset_id}",
        "status": "downloading",
    }


@router.get("/disk-space", response_model=DiskSpaceResponse)
async def get_disk_space(request: Request):
    """
    Get available disk space at data root.
    """
    data_root = get_data_root(request)
    manager = DownloadManager(data_root)
    space = manager.check_disk_space()

    return DiskSpaceResponse(
        total_gb=space.get("total_gb", 0),
        used_gb=space.get("used_gb", 0),
        available_gb=space.get("available_gb", 0),
    )
