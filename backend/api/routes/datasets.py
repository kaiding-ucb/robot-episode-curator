"""
API routes for dataset operations.

Endpoints:
- GET /api/datasets - List all datasets
- GET /api/datasets/{id} - Get dataset info
- GET /api/datasets/{id}/episodes - List episodes in dataset
- GET /api/datasets/{id}/tasks - List tasks in dataset
- GET /api/datasets/{id}/tasks/{task_name}/episodes - List episodes for a task
"""
import logging
from pathlib import Path
from typing import List, Optional
from collections import defaultdict

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from downloaders.manager import DownloadManager, DATASET_REGISTRY
from loaders import HDF5Loader, WebDatasetLoader, LeRobotLoader, RLDSLoader

logger = logging.getLogger(__name__)

router = APIRouter()


class DatasetInfo(BaseModel):
    """Dataset information model."""

    id: str
    name: str
    type: str  # "teleop" or "video"
    description: Optional[str] = None
    status: str  # "ready", "not_downloaded", "downloading"
    size_mb: Optional[float] = None
    num_episodes: Optional[int] = None


class EpisodeInfo(BaseModel):
    """Episode metadata model."""

    id: str
    task_name: Optional[str] = None
    description: Optional[str] = None
    num_frames: Optional[int] = None
    duration_seconds: Optional[float] = None


class TaskInfo(BaseModel):
    """Task metadata model."""

    name: str
    episode_count: Optional[int] = None
    description: Optional[str] = None


class TaskListResponse(BaseModel):
    """Response for task list endpoint."""

    tasks: List[TaskInfo]
    total_tasks: int
    source: str  # "huggingface_api", "episode_scan", "config"


def get_data_root(request: Request) -> Path:
    """Get data root from app state."""
    return request.app.state.data_root


def get_loader(dataset_id: str, data_root: Path):
    """Get appropriate loader for a dataset."""
    if dataset_id not in DATASET_REGISTRY:
        return None

    config = DATASET_REGISTRY[dataset_id]
    data_dir = data_root / dataset_id

    # Check format first
    data_format = config.get("format")
    if data_format == "lerobot":
        return LeRobotLoader(data_dir)
    elif data_format == "rlds":
        return RLDSLoader(data_dir)

    # Fall back to default loaders based on dataset type
    if dataset_id in ["libero", "libero_pro"]:
        return HDF5Loader(data_dir)
    elif config.get("type") == "video" or config.get("streaming_recommended"):
        repo_id = config.get("repo_id", dataset_id)
        return WebDatasetLoader(repo_id, streaming=True)

    return None


@router.get("", response_model=List[DatasetInfo])
async def list_datasets(request: Request):
    """
    List all available datasets with their status.
    """
    data_root = get_data_root(request)
    manager = DownloadManager(data_root)

    datasets = []
    for ds in manager.list_datasets():
        datasets.append(
            DatasetInfo(
                id=ds["id"],
                name=ds["name"],
                type=ds["type"],
                description=ds.get("description"),
                status=ds.get("status", "unknown"),
                size_mb=ds.get("size_mb"),
            )
        )

    return datasets


@router.get("/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str, request: Request):
    """
    Get information about a specific dataset.
    """
    if dataset_id not in DATASET_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    data_root = get_data_root(request)
    manager = DownloadManager(data_root)
    status = manager.get_status(dataset_id)

    config = DATASET_REGISTRY[dataset_id]

    # Try to count episodes if data is available
    num_episodes = None
    if status.get("status") == "ready":
        loader = get_loader(dataset_id, data_root)
        if loader:
            try:
                episodes = loader.list_episodes()
                num_episodes = len(episodes)
            except Exception as e:
                logger.warning(f"Could not count episodes: {e}")

    return DatasetInfo(
        id=dataset_id,
        name=config["name"],
        type=config["type"],
        description=config.get("description"),
        status=status.get("status", "unknown"),
        size_mb=status.get("size_mb"),
        num_episodes=num_episodes,
    )


@router.get("/{dataset_id}/episodes", response_model=List[EpisodeInfo])
async def list_episodes(
    dataset_id: str,
    request: Request,
    limit: int = 100,
    offset: int = 0,
):
    """
    List episodes in a dataset.

    Args:
        dataset_id: Dataset identifier
        limit: Maximum number of episodes to return
        offset: Number of episodes to skip
    """
    if dataset_id not in DATASET_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    config = DATASET_REGISTRY[dataset_id]

    data_root = get_data_root(request)
    loader = get_loader(dataset_id, data_root)

    if loader is None:
        raise HTTPException(
            status_code=501,
            detail=f"Loader not implemented for: {dataset_id}",
        )

    try:
        all_episodes = loader.list_episodes()
    except Exception as e:
        logger.error(f"Failed to list episodes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list episodes: {str(e)}",
        )

    # Apply pagination
    episodes = all_episodes[offset : offset + limit]

    return [
        EpisodeInfo(
            id=ep.id,
            task_name=ep.task_name,
            description=ep.description,
            num_frames=ep.num_frames,
            duration_seconds=ep.duration_seconds,
        )
        for ep in episodes
    ]


async def get_tasks_from_huggingface(repo_id: str) -> List[TaskInfo]:
    """
    Get task list from HuggingFace API by listing top-level directories.

    This works for datasets organized by task folders (like 10Kh-RealOmin-OpenData).
    """
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            items = response.json()

            # Extract directories as tasks (exclude files like README.md, .gitattributes)
            tasks = []
            for item in items:
                if item.get("type") == "directory":
                    task_name = item.get("path", "")
                    # Skip hidden directories
                    if not task_name.startswith("."):
                        tasks.append(TaskInfo(
                            name=task_name,
                            episode_count=None,  # Unknown without scanning
                            description=None,
                        ))

            return tasks
        except httpx.HTTPStatusError as e:
            logger.warning(f"HuggingFace API returned {e.response.status_code} for {repo_id}")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch tasks from HuggingFace: {e}")
            raise


async def get_episodes_from_huggingface(
    repo_id: str,
    task_folder: str,
    limit: int = 10,
    offset: int = 0,
) -> List[EpisodeInfo]:
    """
    Get episodes from a task folder in HuggingFace dataset.

    Recursively lists files in the task folder and returns them as episodes.
    Supports .mcap, .tar, .mp4 and other data files.
    """
    # URL encode the task folder path
    import urllib.parse
    encoded_path = urllib.parse.quote(task_folder, safe='')
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{encoded_path}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            items = response.json()

            episodes = []
            episode_extensions = {'.mcap', '.tar', '.mp4', '.hdf5', '.h5', '.parquet'}

            # Process items - may need to recurse into subdirectories
            items_to_process = list(items)
            processed_dirs = set()

            while items_to_process and len(episodes) < offset + limit + 50:  # Get extra for pagination
                item = items_to_process.pop(0)
                item_path = item.get("path", "")
                item_type = item.get("type", "")

                if item_type == "file":
                    # Check if it's an episode file
                    ext = Path(item_path).suffix.lower()
                    if ext in episode_extensions:
                        # Extract episode ID from path
                        relative_path = item_path
                        if relative_path.startswith(task_folder + "/"):
                            relative_path = relative_path[len(task_folder) + 1:]

                        episode_id = Path(relative_path).stem
                        size_bytes = item.get("size", 0)

                        episodes.append(EpisodeInfo(
                            id=f"{task_folder}/{relative_path}",
                            task_name=task_folder,
                            description=f"File: {Path(item_path).name}",
                            num_frames=None,  # Unknown without loading
                            duration_seconds=None,
                        ))

                elif item_type == "directory" and item_path not in processed_dirs:
                    # Recurse into subdirectory (limit depth)
                    depth = item_path.count('/') - task_folder.count('/')
                    if depth < 3:  # Max 3 levels deep
                        processed_dirs.add(item_path)
                        try:
                            sub_encoded = urllib.parse.quote(item_path, safe='')
                            sub_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{sub_encoded}"
                            sub_response = await client.get(sub_url, timeout=15.0)
                            if sub_response.status_code == 200:
                                sub_items = sub_response.json()
                                items_to_process.extend(sub_items)
                        except Exception as e:
                            logger.warning(f"Failed to list subdirectory {item_path}: {e}")

            # Sort episodes by ID for consistent ordering
            episodes.sort(key=lambda e: e.id)

            # Apply pagination
            return episodes[offset:offset + limit]

        except httpx.HTTPStatusError as e:
            logger.warning(f"HuggingFace API returned {e.response.status_code} for {repo_id}/{task_folder}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch episodes from HuggingFace: {e}")
            return []


def get_tasks_from_episodes(episodes) -> List[TaskInfo]:
    """
    Extract unique tasks from episode list with counts.

    Works for downloaded datasets where we can enumerate all episodes.
    """
    task_counts = defaultdict(int)

    for ep in episodes:
        task_name = ep.task_name or "Unknown"
        task_counts[task_name] += 1

    return [
        TaskInfo(name=name, episode_count=count)
        for name, count in sorted(task_counts.items())
    ]


@router.get("/{dataset_id}/tasks", response_model=TaskListResponse)
async def list_tasks(dataset_id: str, request: Request):
    """
    List all tasks in a dataset.

    For HuggingFace streaming datasets, uses HF API to get folder structure.
    For downloaded datasets, scans episodes and extracts unique task names.
    """
    if dataset_id not in DATASET_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    config = DATASET_REGISTRY[dataset_id]
    data_root = get_data_root(request)

    # Try HuggingFace API first for streaming datasets
    if config.get("streaming_recommended") and config.get("repo_id"):
        try:
            tasks = await get_tasks_from_huggingface(config["repo_id"])
            if tasks:
                return TaskListResponse(
                    tasks=tasks,
                    total_tasks=len(tasks),
                    source="huggingface_api",
                )
        except Exception as e:
            logger.warning(f"HuggingFace API failed, falling back to episode scan: {e}")

    # Fall back to episode scanning for downloaded datasets
    loader = get_loader(dataset_id, data_root)
    if loader is None:
        raise HTTPException(
            status_code=501,
            detail=f"Cannot list tasks - no loader available for: {dataset_id}",
        )

    try:
        episodes = loader.list_episodes()
        tasks = get_tasks_from_episodes(episodes)

        return TaskListResponse(
            tasks=tasks,
            total_tasks=len(tasks),
            source="episode_scan",
        )
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tasks: {str(e)}",
        )


@router.get("/{dataset_id}/tasks/{task_name:path}/episodes", response_model=List[EpisodeInfo])
async def list_task_episodes(
    dataset_id: str,
    task_name: str,
    request: Request,
    limit: int = 10,
    offset: int = 0,
):
    """
    List episodes for a specific task.

    For streaming datasets, uses HuggingFace API to list files in task folder.
    For downloaded datasets, filters from episode list.

    Args:
        dataset_id: Dataset identifier
        task_name: Task name to filter by
        limit: Maximum episodes to return (default 10)
        offset: Number of episodes to skip
    """
    if dataset_id not in DATASET_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    config = DATASET_REGISTRY[dataset_id]
    data_root = get_data_root(request)

    # For streaming HuggingFace datasets, use API to list files in task folder
    if config.get("streaming_recommended") and config.get("repo_id"):
        try:
            episodes = await get_episodes_from_huggingface(
                config["repo_id"],
                task_name,
                limit=limit,
                offset=offset,
            )
            if episodes:
                return episodes
            # If no episodes found via API, fall through to loader
            logger.info(f"No episodes found via HF API for {task_name}, trying loader")
        except Exception as e:
            logger.warning(f"HuggingFace API failed for episodes, falling back to loader: {e}")

    # Fall back to loader for downloaded datasets
    loader = get_loader(dataset_id, data_root)
    if loader is None:
        raise HTTPException(
            status_code=501,
            detail=f"Loader not implemented for: {dataset_id}",
        )

    try:
        all_episodes = loader.list_episodes()

        # Filter by task name
        filtered = [
            ep for ep in all_episodes
            if ep.task_name == task_name or (ep.task_name is None and task_name == "Unknown")
        ]

        # Apply pagination
        episodes = filtered[offset : offset + limit]

        return [
            EpisodeInfo(
                id=ep.id,
                task_name=ep.task_name,
                description=ep.description,
                num_frames=ep.num_frames,
                duration_seconds=ep.duration_seconds,
            )
            for ep in episodes
        ]
    except Exception as e:
        logger.error(f"Failed to list episodes for task {task_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list episodes: {str(e)}",
        )
