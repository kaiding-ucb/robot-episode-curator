"""
API routes for download operations.

Endpoints:
- GET /api/downloads/status - Get status of all downloads
- POST /api/downloads/{dataset_id} - Start download
- GET /api/downloads/disk-space - Check available disk space
- GET /api/downloads/cache/episodes - List cached episodes
- GET /api/downloads/cache/stats - Get cache statistics
- DELETE /api/downloads/cache/episodes - Clear all cache
- DELETE /api/downloads/cache/episodes/{dataset_id}/{episode_id} - Delete specific episode cache
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel

from downloaders.manager import DownloadManager
from cache import get_encoded_frame_cache

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


# ============================================================================
# Episode Cache Management
# ============================================================================


class CachedEpisodeResponse(BaseModel):
    """Information about a cached episode."""

    dataset_id: str
    episode_id: str
    size_mb: float
    cached_at: float
    batch_count: int


class CacheStatsResponse(BaseModel):
    """Cache statistics."""

    total_size_mb: float
    episode_count: int
    batch_count: int


@router.get("/cache/episodes", response_model=List[CachedEpisodeResponse])
async def list_cached_episodes():
    """
    List all cached episodes with size and timestamp.
    """
    cache = get_encoded_frame_cache()
    episodes = cache.list_cached_episodes()

    return [
        CachedEpisodeResponse(
            dataset_id=ep.dataset_id,
            episode_id=ep.episode_id,
            size_mb=ep.size_mb,
            cached_at=ep.cached_at,
            batch_count=ep.batch_count,
        )
        for ep in episodes
    ]


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    Get cache statistics (total size, episode count, etc.).
    """
    cache = get_encoded_frame_cache()
    stats = cache.get_cache_stats()

    return CacheStatsResponse(
        total_size_mb=stats.get("total_size_mb", 0),
        episode_count=stats.get("episode_count", 0),
        batch_count=stats.get("batch_count", 0),
    )


def _get_dir_size_mb(path: Path) -> float:
    """Get total size of a directory in MB."""
    if not path.exists():
        return 0.0
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except (PermissionError, OSError):
        pass
    return total / (1024 * 1024)


@router.get("/cache/all")
async def get_all_cache_stats():
    """
    Get comprehensive statistics for ALL cache locations.

    This includes hidden caches that don't appear in Finder or Mac Storage Management:
    - Encoded frames cache (JSON files)
    - Streaming cache (downloaded tar/mp4/mcap files)
    - Decoded frames cache (pickle files - can be huge!)
    - HuggingFace hub cache
    """
    import os
    home = Path.home()
    cache_base = home / ".cache" / "data_viewer"
    hf_cache = home / ".cache" / "huggingface"

    caches = []

    # Encoded frames cache (what the UI currently shows)
    encoded_path = cache_base / "frames"
    caches.append({
        "name": "Encoded Frames",
        "path": str(encoded_path),
        "size_mb": round(_get_dir_size_mb(encoded_path), 2),
        "description": "WebP-encoded frames for quick loading",
        "safe_to_clear": True,
    })

    # Streaming cache (downloaded source files)
    streaming_path = cache_base / "streaming"
    # Exclude decoded_frames subdirectory
    streaming_size = 0.0
    if streaming_path.exists():
        for item in streaming_path.iterdir():
            if item.name != "decoded_frames":
                if item.is_file():
                    streaming_size += item.stat().st_size
                elif item.is_dir():
                    streaming_size += _get_dir_size_mb(item) * 1024 * 1024
    caches.append({
        "name": "Downloaded Episodes",
        "path": str(streaming_path),
        "size_mb": round(streaming_size / (1024 * 1024), 2),
        "description": "Downloaded tar/mp4/mcap files from HuggingFace",
        "safe_to_clear": True,
    })

    # Decoded frames cache (THE PROBLEMATIC ONE - can be 100GB+!)
    decoded_path = cache_base / "streaming" / "decoded_frames"
    caches.append({
        "name": "Decoded Frames (WARNING)",
        "path": str(decoded_path),
        "size_mb": round(_get_dir_size_mb(decoded_path), 2),
        "description": "Raw decoded video frames as pickle - CAN BE HUGE!",
        "safe_to_clear": True,
    })

    # HuggingFace cache
    caches.append({
        "name": "HuggingFace Hub",
        "path": str(hf_cache),
        "size_mb": round(_get_dir_size_mb(hf_cache), 2),
        "description": "HuggingFace model/dataset downloads",
        "safe_to_clear": True,
    })

    # Quality cache
    quality_path = cache_base / "quality"
    caches.append({
        "name": "Quality Metrics",
        "path": str(quality_path),
        "size_mb": round(_get_dir_size_mb(quality_path), 2),
        "description": "Computed quality metrics",
        "safe_to_clear": True,
    })

    total_mb = sum(c["size_mb"] for c in caches)

    return {
        "caches": caches,
        "total_size_mb": round(total_mb, 2),
        "total_size_gb": round(total_mb / 1024, 2),
    }


@router.delete("/cache/all/{cache_name}")
async def clear_specific_cache(cache_name: str):
    """
    Clear a specific cache by name.

    Valid names: encoded_frames, streaming, decoded_frames, huggingface, quality
    """
    import shutil
    home = Path.home()
    cache_base = home / ".cache" / "data_viewer"

    cache_paths = {
        "encoded_frames": cache_base / "frames",
        "streaming": cache_base / "streaming",
        "decoded_frames": cache_base / "streaming" / "decoded_frames",
        "huggingface": home / ".cache" / "huggingface",
        "quality": cache_base / "quality",
    }

    if cache_name not in cache_paths:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown cache: {cache_name}. Valid: {list(cache_paths.keys())}"
        )

    path = cache_paths[cache_name]
    if not path.exists():
        return {"message": f"Cache {cache_name} is already empty", "bytes_freed": 0}

    size_before = _get_dir_size_mb(path) * 1024 * 1024

    try:
        if cache_name == "decoded_frames":
            # Only clear pickle files, keep directory
            for f in path.glob("*.pkl"):
                f.unlink()
        else:
            # Clear all contents but keep directory
            for item in path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e}")

    return {
        "message": f"Cleared {cache_name} cache",
        "bytes_freed": int(size_before),
        "mb_freed": round(size_before / (1024 * 1024), 2),
    }


@router.delete("/cache/episodes/{dataset_id}/{episode_id:path}")
async def delete_episode_cache(dataset_id: str, episode_id: str):
    """
    Delete cached frames for a specific episode.
    """
    cache = get_encoded_frame_cache()
    bytes_freed = cache.delete_episode_cache(dataset_id, episode_id)

    return {
        "message": f"Deleted cache for {dataset_id}/{episode_id}",
        "bytes_freed": bytes_freed,
        "mb_freed": bytes_freed / (1024 * 1024),
    }


@router.delete("/cache/episodes")
async def clear_all_cache():
    """
    Clear all cached episodes.
    """
    cache = get_encoded_frame_cache()
    bytes_freed = cache.clear_all()

    return {
        "message": "Cleared all episode cache",
        "bytes_freed": bytes_freed,
        "mb_freed": bytes_freed / (1024 * 1024),
    }
