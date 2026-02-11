"""
API routes for dataset operations.

Endpoints:
- GET /api/datasets - List all datasets
- GET /api/datasets/{id} - Get dataset info
- GET /api/datasets/{id}/overview - Get rich metadata overview
- GET /api/datasets/{id}/episodes - List episodes in dataset
- GET /api/datasets/{id}/tasks - List tasks in dataset
- GET /api/datasets/{id}/tasks/{task_name}/episodes - List episodes for a task
- POST /api/datasets/probe - Probe a HuggingFace URL to detect format and modalities
- POST /api/datasets - Add a new dataset from HuggingFace URL
- DELETE /api/datasets/{id} - Remove a dynamically added dataset
"""
import io
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import httpx
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from downloaders.manager import (
    DownloadManager,
    DATASET_REGISTRY,
    add_dynamic_dataset,
    remove_dynamic_dataset,
    get_all_datasets,
)
from loaders import HDF5Loader, WebDatasetLoader, LeRobotLoader, RLDSLoader
from loaders.base import Modality
from loaders.mcap_utils import detect_mcap_modalities

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


class DatasetOverview(BaseModel):
    """Rich metadata for dataset overview display."""

    repo_id: str
    name: str
    description: Optional[str] = None
    readme_summary: Optional[str] = None
    license: Optional[str] = None
    dataset_tags: List[str] = []

    # From HF repo info
    size_bytes: Optional[int] = None
    gated: bool = False
    downloads_last_month: Optional[int] = None

    # Parsed from README or detected
    environment: Optional[str] = None
    perspective: Optional[str] = None
    format_detected: Optional[str] = None

    # Scale and modalities
    modalities: List[str] = []
    estimated_hours: Optional[float] = None
    estimated_clips: Optional[int] = None
    task_count: Optional[int] = None

    # Cache metadata
    cached_at: Optional[str] = None


# Overview cache with TTL
_OVERVIEW_CACHE: Dict[str, Tuple[DatasetOverview, datetime]] = {}
OVERVIEW_CACHE_TTL_HOURS = 24

# LeRobot metadata cache: {repo_id: (data, fetched_at)}
_LEROBOT_INFO_CACHE: Dict[str, Tuple[dict, datetime]] = {}
_LEROBOT_EPISODES_CACHE: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
_LEROBOT_TASKS_CACHE: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
LEROBOT_CACHE_TTL_HOURS = 24


def _get_hf_headers() -> dict:
    """Get HuggingFace auth headers if token is available."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {hf_token}"} if hf_token else {}


async def fetch_lerobot_info(repo_id: str) -> Optional[dict]:
    """
    Fetch and cache meta/info.json from a LeRobot HuggingFace dataset.

    Returns dataset info including fps, total_episodes, chunks_size, video path template, etc.
    """
    if repo_id in _LEROBOT_INFO_CACHE:
        cached, fetched_at = _LEROBOT_INFO_CACHE[repo_id]
        if datetime.utcnow() - fetched_at < timedelta(hours=LEROBOT_CACHE_TTL_HOURS):
            return cached

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/info.json"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            resp = await client.get(url, headers=_get_hf_headers(), timeout=30.0)
            if resp.status_code == 200:
                data = resp.json()
                _LEROBOT_INFO_CACHE[repo_id] = (data, datetime.utcnow())
                return data
        except Exception as e:
            logger.warning(f"Failed to fetch LeRobot info.json for {repo_id}: {e}")
    return None


async def fetch_lerobot_episodes_meta(repo_id: str) -> Optional[pd.DataFrame]:
    """
    Fetch and cache episode metadata parquet from a LeRobot HuggingFace dataset.

    Returns DataFrame with episode_index, length (frame count), task_index, and
    video frame range columns.
    """
    if repo_id in _LEROBOT_EPISODES_CACHE:
        cached, fetched_at = _LEROBOT_EPISODES_CACHE[repo_id]
        if datetime.utcnow() - fetched_at < timedelta(hours=LEROBOT_CACHE_TTL_HOURS):
            return cached

    # LeRobot v3 stores episode metadata under meta/episodes/chunk-{N}/file-{N}.parquet
    # Discover chunks by listing meta/episodes/
    async with httpx.AsyncClient(follow_redirects=True) as client:
        headers = _get_hf_headers()
        dfs = []

        # List chunk directories
        tree_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/meta/episodes"
        try:
            resp = await client.get(tree_url, headers=headers, timeout=30.0)
            if resp.status_code != 200:
                logger.warning(f"Could not list LeRobot episodes metadata for {repo_id}")
                return None
            items = resp.json()
        except Exception as e:
            logger.warning(f"Failed to list LeRobot episodes metadata for {repo_id}: {e}")
            return None

        # For each chunk directory, list and download parquet files
        for item in items:
            if item.get("type") != "directory":
                continue
            chunk_path = item["path"]

            # List files in this chunk
            try:
                chunk_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{chunk_path}"
                chunk_resp = await client.get(chunk_url, headers=headers, timeout=15.0)
                if chunk_resp.status_code != 200:
                    continue
                chunk_items = chunk_resp.json()
            except Exception:
                continue

            for file_item in chunk_items:
                if file_item.get("type") != "file" or not file_item["path"].endswith(".parquet"):
                    continue
                file_path = file_item["path"]

                # Download parquet file
                try:
                    dl_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_path}"
                    dl_resp = await client.get(dl_url, headers=headers, timeout=60.0)
                    if dl_resp.status_code == 200:
                        df = pd.read_parquet(io.BytesIO(dl_resp.content))
                        dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to download {file_path}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("episode_index").reset_index(drop=True)
    _LEROBOT_EPISODES_CACHE[repo_id] = (combined, datetime.utcnow())
    logger.info(f"Cached LeRobot episodes metadata for {repo_id}: {len(combined)} episodes")
    return combined


async def fetch_lerobot_tasks_meta(repo_id: str) -> Optional[pd.DataFrame]:
    """
    Fetch and cache meta/tasks.parquet from a LeRobot HuggingFace dataset.

    Returns DataFrame with task_index and task descriptions.
    """
    if repo_id in _LEROBOT_TASKS_CACHE:
        cached, fetched_at = _LEROBOT_TASKS_CACHE[repo_id]
        if datetime.utcnow() - fetched_at < timedelta(hours=LEROBOT_CACHE_TTL_HOURS):
            return cached

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/tasks.parquet"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            resp = await client.get(url, headers=_get_hf_headers(), timeout=30.0)
            if resp.status_code == 200:
                df = pd.read_parquet(io.BytesIO(resp.content))
                # In LeRobot v3 tasks.parquet, the task description is the DataFrame
                # index and task_index is a column. Normalize to have both as columns.
                if df.index.name is None and "task_index" in df.columns:
                    df = df.reset_index()
                    df = df.rename(columns={"index": "task_description"})
                _LEROBOT_TASKS_CACHE[repo_id] = (df, datetime.utcnow())
                logger.info(f"Cached LeRobot tasks metadata for {repo_id}: {len(df)} tasks")
                return df
        except Exception as e:
            logger.warning(f"Failed to fetch LeRobot tasks.parquet for {repo_id}: {e}")
    return None


_LEROBOT_EP_TASK_MAP_CACHE: Dict[str, Tuple[Dict[int, int], datetime]] = {}


async def fetch_lerobot_episode_task_map(repo_id: str) -> Optional[Dict[int, int]]:
    """
    Fetch the episode_index -> task_index mapping for a LeRobot dataset.

    Uses the HuggingFace datasets server filter API to efficiently get only
    the first frame of each episode (frame_index=0), extracting episode_index
    and task_index columns.

    Returns dict mapping episode_index -> task_index, or None on failure.
    """
    if repo_id in _LEROBOT_EP_TASK_MAP_CACHE:
        cached, fetched_at = _LEROBOT_EP_TASK_MAP_CACHE[repo_id]
        if datetime.utcnow() - fetched_at < timedelta(hours=LEROBOT_CACHE_TTL_HOURS):
            return cached

    mapping: Dict[int, int] = {}
    offset = 0
    page_size = 100

    async with httpx.AsyncClient(follow_redirects=True) as client:
        headers = _get_hf_headers()
        while True:
            url = (
                f"https://datasets-server.huggingface.co/filter"
                f"?dataset={repo_id}&config=default&split=train"
                f"&where=frame_index=0&offset={offset}&length={page_size}"
            )
            try:
                resp = await client.get(url, headers=headers, timeout=30.0)
                if resp.status_code != 200:
                    logger.warning(f"HF filter API returned {resp.status_code} for {repo_id}")
                    break
                data = resp.json()
                rows = data.get("rows", [])
                if not rows:
                    break
                for row in rows:
                    r = row["row"]
                    ep_idx = r.get("episode_index")
                    task_idx = r.get("task_index")
                    if ep_idx is not None and task_idx is not None:
                        mapping[int(ep_idx)] = int(task_idx)
                offset += len(rows)
                total = data.get("num_rows_total", 0)
                if offset >= total:
                    break
            except Exception as e:
                logger.warning(f"Failed to fetch episode-task mapping at offset {offset}: {e}")
                break

    if mapping:
        _LEROBOT_EP_TASK_MAP_CACHE[repo_id] = (mapping, datetime.utcnow())
        logger.info(f"Cached episode-task mapping for {repo_id}: {len(mapping)} episodes")
        return mapping
    return None


async def is_lerobot_dataset(config: dict) -> bool:
    """Check if a streaming dataset is in LeRobot v3 format by probing meta/info.json."""
    if config.get("format") == "lerobot":
        return True
    repo_id = config.get("repo_id")
    if not repo_id:
        return False
    info = await fetch_lerobot_info(repo_id)
    return info is not None and "codebase_version" in info


def get_data_root(request: Request) -> Path:
    """Get data root from app state."""
    return request.app.state.data_root


def get_loader(dataset_id: str, data_root: Path):
    """Get appropriate loader for a dataset."""
    all_datasets = get_all_datasets()
    if dataset_id not in all_datasets:
        return None

    config = all_datasets[dataset_id]
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
    all_datasets = get_all_datasets()
    if dataset_id not in all_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    data_root = get_data_root(request)
    manager = DownloadManager(data_root)
    status = manager.get_status(dataset_id)

    config = all_datasets[dataset_id]

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
    all_datasets = get_all_datasets()
    if dataset_id not in all_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    config = all_datasets[dataset_id]

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
    all_datasets = get_all_datasets()
    if dataset_id not in all_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    config = all_datasets[dataset_id]
    data_root = get_data_root(request)

    # For LeRobot streaming datasets, use metadata parquet for tasks
    if config.get("streaming_recommended") and config.get("repo_id"):
        if await is_lerobot_dataset(config):
            repo_id = config["repo_id"]
            tasks_df = await fetch_lerobot_tasks_meta(repo_id)
            episodes_df = await fetch_lerobot_episodes_meta(repo_id)
            if tasks_df is not None:
                # Use task_description column (or fallback to first non-task_index column)
                task_col = "task_description"
                if task_col not in tasks_df.columns:
                    for col in tasks_df.columns:
                        if col != "task_index":
                            task_col = col
                            break

                # Get episode-task mapping for counts
                ep_task_map = await fetch_lerobot_episode_task_map(repo_id)
                task_episode_counts: Dict[int, int] = {}
                if ep_task_map:
                    for _, task_idx in ep_task_map.items():
                        task_episode_counts[task_idx] = task_episode_counts.get(task_idx, 0) + 1

                tasks = []
                for _, row in tasks_df.iterrows():
                    task_idx = int(row["task_index"])
                    task_name = str(row[task_col]).strip() if task_col in row.index else ""
                    if not task_name:
                        task_name = f"Untitled (task {task_idx})"
                    ep_count = task_episode_counts.get(task_idx)
                    tasks.append(TaskInfo(
                        name=task_name,
                        episode_count=ep_count,
                    ))
                return TaskListResponse(
                    tasks=tasks,
                    total_tasks=len(tasks),
                    source="lerobot_metadata",
                )

    # Try HuggingFace API for other streaming datasets
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
    all_datasets = get_all_datasets()
    if dataset_id not in all_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    config = all_datasets[dataset_id]
    data_root = get_data_root(request)

    # For LeRobot streaming datasets, use metadata for episode listing
    if config.get("streaming_recommended") and config.get("repo_id"):
        if await is_lerobot_dataset(config):
            repo_id = config["repo_id"]
            episodes_df = await fetch_lerobot_episodes_meta(repo_id)
            tasks_df = await fetch_lerobot_tasks_meta(repo_id)
            info = await fetch_lerobot_info(repo_id)

            if episodes_df is not None and tasks_df is not None:
                # Map task_name to task_index
                task_col = "task_description"
                if task_col not in tasks_df.columns:
                    for col in tasks_df.columns:
                        if col != "task_index":
                            task_col = col
                            break

                task_index = None
                if task_col in tasks_df.columns:
                    match = tasks_df[tasks_df[task_col] == task_name]
                    if len(match) > 0:
                        task_index = int(match.iloc[0]["task_index"])

                # Handle "Untitled (task N)" fallback names
                if task_index is None:
                    import re
                    untitled_match = re.match(r"^Untitled \(task (\d+)\)$", task_name)
                    if untitled_match:
                        candidate_idx = int(untitled_match.group(1))
                        if candidate_idx in tasks_df["task_index"].values:
                            task_index = candidate_idx

                if task_index is None:
                    return []

                # Get episode->task mapping to filter episodes for this task
                ep_task_map = await fetch_lerobot_episode_task_map(repo_id)
                if ep_task_map is None:
                    return []

                # Filter episode indices belonging to this task
                task_ep_indices = sorted([
                    ep_idx for ep_idx, t_idx in ep_task_map.items()
                    if t_idx == task_index
                ])

                # Get metadata for these episodes
                task_episodes = episodes_df[episodes_df["episode_index"].isin(task_ep_indices)]
                task_episodes = task_episodes.sort_values("episode_index")

                # Apply pagination
                page = task_episodes.iloc[offset:offset + limit]

                fps = info.get("fps", 30) if info else 30
                episodes = []
                for _, row in page.iterrows():
                    ep_idx = int(row["episode_index"])
                    length = int(row["length"]) if "length" in row.index else None
                    duration = length / fps if length else None
                    episodes.append(EpisodeInfo(
                        id=f"episode_{ep_idx}",
                        task_name=task_name,
                        num_frames=length,
                        duration_seconds=round(duration, 2) if duration else None,
                    ))
                return episodes

    # For other streaming HuggingFace datasets, use API to list files in task folder
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


# === Dataset Overview Functions ===


async def fetch_repo_readme(repo_id: str) -> Optional[str]:
    """Fetch README.md content from HuggingFace."""
    url = f"https://huggingface.co/datasets/{repo_id}/raw/main/README.md"
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            logger.warning(f"Failed to fetch README for {repo_id}: {e}")
    return None


async def fetch_repo_info(repo_id: str) -> Dict[str, Any]:
    """Fetch dataset info from HuggingFace API."""
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch repo info for {repo_id}: {e}")
    return {}


async def fetch_repo_files_size(repo_id: str) -> Optional[int]:
    """Calculate total size from file listing."""
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            if response.status_code == 200:
                items = response.json()
                total_size = sum(item.get("size", 0) for item in items if item.get("type") == "file")
                return total_size if total_size > 0 else None
        except Exception as e:
            logger.warning(f"Failed to fetch file sizes for {repo_id}: {e}")
    return None


def parse_readme_metadata(content: str) -> Dict[str, Any]:
    """Extract structured metadata from README text."""
    result: Dict[str, Any] = {}

    # Extract summary (first paragraph after any YAML front matter)
    lines = content.split("\n")
    in_yaml = False
    summary_lines = []
    for line in lines:
        if line.strip() == "---":
            in_yaml = not in_yaml
            continue
        if in_yaml:
            continue
        # Skip headers and empty lines for summary
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("|"):
            summary_lines.append(stripped)
            if len(" ".join(summary_lines)) > 500:
                break
    if summary_lines:
        result["readme_summary"] = " ".join(summary_lines)[:500]

    # Extract environment
    content_lower = content.lower()
    if any(word in content_lower for word in ["household", "home", "kitchen", "domestic"]):
        result["environment"] = "Household"
    elif any(word in content_lower for word in ["factory", "industrial", "manufacturing", "assembly"]):
        result["environment"] = "Factory"
    elif any(word in content_lower for word in ["laboratory", "lab", "research"]):
        result["environment"] = "Laboratory"
    elif any(word in content_lower for word in ["outdoor", "field", "nature"]):
        result["environment"] = "Outdoor"

    # Extract perspective
    if any(word in content_lower for word in ["egocentric", "first-person", "first person", "ego-view"]):
        result["perspective"] = "Egocentric"
    elif any(word in content_lower for word in ["third-person", "third person", "external view"]):
        result["perspective"] = "Third-person"
    elif any(word in content_lower for word in ["robotic", "robot arm", "manipulator"]):
        result["perspective"] = "Robotic"

    # Extract hours
    hours_match = re.search(r"(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:k|K)?\s*(?:hours?|hrs?|h)\b", content)
    if hours_match:
        hours_str = hours_match.group(1).replace(",", "")
        hours = float(hours_str)
        if "k" in hours_match.group(0).lower():
            hours *= 1000
        result["estimated_hours"] = hours

    # Extract clips/episodes count
    clips_match = re.search(r"(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:million|M|k|K)?\s*(?:clips?|episodes?|videos?|samples?)\b", content)
    if clips_match:
        clips_str = clips_match.group(1).replace(",", "")
        clips = float(clips_str)
        match_text = clips_match.group(0).lower()
        if "million" in match_text or match_text.endswith("m"):
            clips *= 1_000_000
        elif "k" in match_text:
            clips *= 1000
        result["estimated_clips"] = int(clips)

    # Extract modalities
    modalities = []
    if any(word in content_lower for word in ["rgb", "color image", "video"]):
        modalities.append("rgb")
    if any(word in content_lower for word in ["depth", "16-bit depth", "depth map"]):
        modalities.append("depth")
    if any(word in content_lower for word in ["imu", "inertial", "accelerometer", "gyroscope"]):
        modalities.append("imu")
    if any(word in content_lower for word in ["tactile", "touch sensor", "force sensor"]):
        modalities.append("tactile")
    if any(word in content_lower for word in ["action", "trajectory", "end-effector"]):
        modalities.append("actions")
    if modalities:
        result["modalities"] = modalities

    return result


def parse_yaml_front_matter(content: str) -> Dict[str, Any]:
    """Extract YAML front matter from README."""
    result: Dict[str, Any] = {}
    if not content.startswith("---"):
        return result

    try:
        end_idx = content.find("---", 3)
        if end_idx == -1:
            return result
        yaml_content = content[3:end_idx].strip()

        # Simple YAML parsing for common fields
        for line in yaml_content.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip().strip('"').strip("'")

                if key == "license":
                    result["license"] = value
                elif key == "tags":
                    # Handle list format
                    pass  # Tags usually span multiple lines
    except Exception as e:
        logger.warning(f"Failed to parse YAML front matter: {e}")

    return result


def detect_format_from_files(files: List[str]) -> Optional[str]:
    """Detect dataset format from file extensions."""
    extensions = set()
    for f in files:
        ext = Path(f).suffix.lower()
        if ext:
            extensions.add(ext)

    if ".mcap" in extensions:
        return "MCAP"
    elif ".tar" in extensions:
        return "WebDataset"
    elif ".parquet" in extensions:
        return "LeRobot"
    elif ".hdf5" in extensions or ".h5" in extensions:
        return "HDF5"
    elif ".tfrecord" in extensions:
        return "RLDS"
    elif ".mp4" in extensions or ".webm" in extensions:
        return "Video"
    return None


@router.get("/{dataset_id}/overview", response_model=DatasetOverview)
async def get_dataset_overview(
    dataset_id: str,
    request: Request,
    refresh: bool = False,
):
    """
    Get rich metadata overview for a dataset.

    Fetches and caches metadata from HuggingFace including:
    - README content and summary
    - Repository info (size, gated status, downloads)
    - Parsed metadata (environment, perspective, scale)
    - Detected format and modalities

    Args:
        dataset_id: Dataset identifier
        refresh: Force refresh cached metadata
    """
    # Check cache first
    if not refresh and dataset_id in _OVERVIEW_CACHE:
        cached_overview, cached_at = _OVERVIEW_CACHE[dataset_id]
        if datetime.utcnow() - cached_at < timedelta(hours=OVERVIEW_CACHE_TTL_HOURS):
            return cached_overview

    # Get dataset config
    all_datasets = get_all_datasets()
    if dataset_id not in all_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    config = all_datasets[dataset_id]
    repo_id = config.get("repo_id")

    # Start building overview with registry data
    overview = DatasetOverview(
        repo_id=repo_id or dataset_id,
        name=config.get("name", dataset_id),
        description=config.get("description"),
        modalities=config.get("modalities", ["rgb"]),
    )

    # If no repo_id, return basic info from registry
    if not repo_id:
        overview.cached_at = datetime.utcnow().isoformat()
        _OVERVIEW_CACHE[dataset_id] = (overview, datetime.utcnow())
        return overview

    # Fetch HuggingFace metadata
    readme_content = await fetch_repo_readme(repo_id)
    repo_info = await fetch_repo_info(repo_id)

    # Parse repo info
    if repo_info:
        # Handle gated field - can be bool, "auto", "manual", etc.
        gated_value = repo_info.get("gated", False)
        overview.gated = gated_value not in (False, None, "")
        overview.downloads_last_month = repo_info.get("downloads", 0)

        # Extract license from cardData
        card_data = repo_info.get("cardData", {})
        if isinstance(card_data, dict):
            overview.license = card_data.get("license")
            tags = card_data.get("tags", [])
            if isinstance(tags, list):
                overview.dataset_tags = tags[:10]

    # Parse README content
    if readme_content:
        yaml_data = parse_yaml_front_matter(readme_content)
        if yaml_data.get("license"):
            overview.license = yaml_data["license"]

        parsed = parse_readme_metadata(readme_content)
        if parsed.get("readme_summary"):
            overview.readme_summary = parsed["readme_summary"]
        if parsed.get("environment"):
            overview.environment = parsed["environment"]
        if parsed.get("perspective"):
            overview.perspective = parsed["perspective"]
        if parsed.get("estimated_hours"):
            overview.estimated_hours = parsed["estimated_hours"]
        if parsed.get("estimated_clips"):
            overview.estimated_clips = parsed["estimated_clips"]
        if parsed.get("modalities"):
            overview.modalities = parsed["modalities"]

    # Get file size
    size_bytes = await fetch_repo_files_size(repo_id)
    if size_bytes:
        overview.size_bytes = size_bytes

    # Detect format from registry or file listing
    if config.get("format"):
        overview.format_detected = config["format"].upper()
    else:
        # Try to get from file listing
        url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, timeout=15.0)
                if response.status_code == 200:
                    files = [item.get("path", "") for item in response.json()]
                    detected = detect_format_from_files(files)
                    if detected:
                        overview.format_detected = detected
            except Exception:
                pass

    # Get task count
    try:
        tasks = await get_tasks_from_huggingface(repo_id)
        overview.task_count = len(tasks)
    except Exception:
        pass

    # Cache and return
    overview.cached_at = datetime.utcnow().isoformat()
    _OVERVIEW_CACHE[dataset_id] = (overview, datetime.utcnow())
    return overview


# === Probe and Add Dataset Endpoints ===


class ProbeRequest(BaseModel):
    """Request to probe a HuggingFace dataset URL."""
    url: str  # HuggingFace URL like https://huggingface.co/datasets/user/repo


class ProbeResponse(BaseModel):
    """Response from probing a HuggingFace dataset."""
    repo_id: str
    name: str
    format_detected: Optional[str] = None  # "mcap", "webdataset", "lerobot", "rlds", "hdf5"
    has_tasks: bool = True  # True if has subdirectories, False if flat
    modalities: List[str] = ["rgb"]  # Detected modalities
    modality_config: Optional[Dict[str, Any]] = None
    sample_files: List[str] = []
    error: Optional[str] = None


class AddDatasetRequest(BaseModel):
    """Request to add a new dataset."""
    url: str
    name: Optional[str] = None  # Override detected name
    dataset_id: Optional[str] = None  # Override generated ID


class AddDatasetResponse(BaseModel):
    """Response after adding a dataset."""
    dataset_id: str
    name: str
    success: bool
    error: Optional[str] = None


def parse_huggingface_url(url: str) -> Optional[str]:
    """
    Parse a HuggingFace URL to extract the repo_id.

    Accepts formats:
    - https://huggingface.co/datasets/user/repo
    - huggingface.co/datasets/user/repo
    - user/repo (assumed to be HF repo)

    Returns:
        repo_id string (e.g., "user/repo") or None if invalid
    """
    url = url.strip()

    # Full HuggingFace URL
    hf_match = re.match(
        r"(?:https?://)?(?:www\.)?huggingface\.co/datasets/([^/]+/[^/\s]+)",
        url
    )
    if hf_match:
        return hf_match.group(1).rstrip("/")

    # Direct repo_id format (user/repo)
    if "/" in url and not url.startswith("http"):
        parts = url.split("/")
        if len(parts) == 2:
            return url

    return None


def generate_dataset_id(repo_id: str) -> str:
    """Generate a dataset ID from repo_id."""
    # Take the repo name, lowercase, replace special chars
    name = repo_id.split("/")[-1].lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


async def probe_huggingface_dataset(repo_id: str) -> ProbeResponse:
    """
    Probe a HuggingFace dataset to detect its format and modalities.

    This function:
    1. Lists files in the repo
    2. Detects format from file extensions
    3. For MCAP: Downloads a sample file and scans channels for modalities
    4. Determines if the dataset has task subdirectories or is flat

    Args:
        repo_id: HuggingFace repository ID (e.g., "user/repo")

    Returns:
        ProbeResponse with detected information
    """
    name = repo_id.split("/")[-1]
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            items = response.json()
        except httpx.HTTPStatusError as e:
            return ProbeResponse(
                repo_id=repo_id,
                name=name,
                error=f"HuggingFace API error: {e.response.status_code}",
            )
        except Exception as e:
            return ProbeResponse(
                repo_id=repo_id,
                name=name,
                error=f"Failed to access HuggingFace: {str(e)}",
            )

    # Analyze the file structure
    directories = []
    files = []
    sample_files = []

    for item in items:
        item_type = item.get("type", "")
        item_path = item.get("path", "")

        if item_type == "directory":
            if not item_path.startswith("."):
                directories.append(item_path)
        elif item_type == "file":
            files.append(item_path)
            if len(sample_files) < 5:
                sample_files.append(item_path)

    # Determine if flat or hierarchical
    has_tasks = len(directories) > 0

    # Detect format from file extensions
    format_detected = None
    modalities = [Modality.RGB.value]  # Default to RGB
    modality_config = None

    # Check top-level files first
    all_files = files.copy()

    # If hierarchical, also check first subdirectory for files
    if has_tasks and directories:
        try:
            first_dir = directories[0]
            import urllib.parse
            encoded = urllib.parse.quote(first_dir, safe="")
            sub_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{encoded}"

            async with httpx.AsyncClient() as client:
                sub_response = await client.get(sub_url, timeout=15.0)
                if sub_response.status_code == 200:
                    sub_items = sub_response.json()
                    for item in sub_items:
                        if item.get("type") == "file":
                            all_files.append(item.get("path", ""))
        except Exception as e:
            logger.warning(f"Failed to probe subdirectory: {e}")

    # Detect format from file extensions
    extensions = set()
    for f in all_files:
        ext = Path(f).suffix.lower()
        if ext:
            extensions.add(ext)

    if ".mcap" in extensions:
        format_detected = "mcap"
        # For MCAP, we need to download a sample to detect modalities
        modalities, modality_config = await _probe_mcap_modalities(repo_id, all_files)
    elif ".tar" in extensions or ".tar.gz" in extensions:
        format_detected = "webdataset"
    elif ".parquet" in extensions:
        format_detected = "lerobot"
    elif ".hdf5" in extensions or ".h5" in extensions:
        format_detected = "hdf5"
    elif ".tfrecord" in extensions:
        format_detected = "rlds"
    elif ".mp4" in extensions or ".webm" in extensions or ".avi" in extensions:
        format_detected = "video"

    return ProbeResponse(
        repo_id=repo_id,
        name=name,
        format_detected=format_detected,
        has_tasks=has_tasks,
        modalities=modalities,
        modality_config=modality_config,
        sample_files=sample_files[:5],
    )


async def _probe_mcap_modalities(
    repo_id: str, files: List[str]
) -> tuple[List[str], Optional[Dict[str, Any]]]:
    """
    Download a sample MCAP file and detect its modalities.

    Returns:
        Tuple of (modality_list, modality_config_dict)
    """
    # Find a .mcap file to probe
    mcap_files = [f for f in files if f.endswith(".mcap")]
    if not mcap_files:
        return [Modality.RGB.value], None

    # Download the first MCAP file (or a small one)
    mcap_path = mcap_files[0]
    download_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{mcap_path}"

    try:
        # Get HuggingFace token from environment
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        async with httpx.AsyncClient() as client:
            # Stream download to temp file
            async with client.stream("GET", download_url, headers=headers, timeout=60.0) as response:
                if response.status_code != 200:
                    logger.warning(f"Failed to download MCAP sample: {response.status_code}")
                    return [Modality.RGB.value], None

                with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        tmp.write(chunk)
                        # Limit download to 50MB for probing
                        if tmp.tell() > 50 * 1024 * 1024:
                            break
                    tmp_path = tmp.name

        # Detect modalities from MCAP
        modality_configs = detect_mcap_modalities(Path(tmp_path))

        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        if modality_configs:
            modalities = list(modality_configs.keys())
            config_dict = {k: v.to_dict() for k, v in modality_configs.items()}
            return modalities, config_dict

    except Exception as e:
        logger.error(f"Failed to probe MCAP modalities: {e}")

    return [Modality.RGB.value], None


@router.post("/probe", response_model=ProbeResponse)
async def probe_dataset(request_body: ProbeRequest):
    """
    Probe a HuggingFace dataset URL to detect format and modalities.

    This endpoint:
    1. Parses the URL to extract the repo_id
    2. Queries HuggingFace API to list files
    3. Detects the data format from file extensions
    4. For MCAP files, downloads a sample to detect available modalities
    5. Determines if the dataset has task subdirectories or is flat

    Returns format, modalities, and structure information for UI display.
    """
    repo_id = parse_huggingface_url(request_body.url)
    if not repo_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid HuggingFace URL. Expected format: https://huggingface.co/datasets/user/repo",
        )

    result = await probe_huggingface_dataset(repo_id)
    return result


@router.post("", response_model=AddDatasetResponse)
async def add_dataset(request_body: AddDatasetRequest):
    """
    Add a new dataset from a HuggingFace URL.

    This endpoint:
    1. Probes the URL to detect format and modalities
    2. Creates a dataset configuration
    3. Adds the dataset to the dynamic registry

    The dataset will appear in the list and can be browsed/streamed.
    """
    repo_id = parse_huggingface_url(request_body.url)
    if not repo_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid HuggingFace URL",
        )

    # Check if already exists
    dataset_id = request_body.dataset_id or generate_dataset_id(repo_id)
    all_datasets = get_all_datasets()
    if dataset_id in all_datasets:
        return AddDatasetResponse(
            dataset_id=dataset_id,
            name=all_datasets[dataset_id]["name"],
            success=False,
            error=f"Dataset '{dataset_id}' already exists",
        )

    # Probe to get details
    probe_result = await probe_huggingface_dataset(repo_id)

    if probe_result.error:
        return AddDatasetResponse(
            dataset_id=dataset_id,
            name=repo_id.split("/")[-1],
            success=False,
            error=probe_result.error,
        )

    # Determine dataset type
    dataset_type = "video"
    if probe_result.format_detected in ["mcap", "lerobot", "rlds", "hdf5"]:
        dataset_type = "teleop"
    if Modality.ACTIONS.value in probe_result.modalities:
        dataset_type = "teleop"

    # Create config
    name = request_body.name or probe_result.name
    config = {
        "name": name,
        "type": dataset_type,
        "description": f"HuggingFace dataset: {repo_id}",
        "repo_id": repo_id,
        "format": probe_result.format_detected,
        "modalities": probe_result.modalities,
        "modality_config": probe_result.modality_config,
        "has_tasks": probe_result.has_tasks,
        "streaming_recommended": True,
        "requires_auth": False,
        "downloader_class": None,  # Use HuggingFace streaming
    }

    add_dynamic_dataset(dataset_id, config)

    return AddDatasetResponse(
        dataset_id=dataset_id,
        name=name,
        success=True,
    )


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    Remove a dynamically added dataset.

    Note: Only datasets added via the /api/datasets POST endpoint can be removed.
    Built-in datasets cannot be removed.
    """
    if dataset_id in DATASET_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot remove built-in dataset: {dataset_id}",
        )

    if remove_dynamic_dataset(dataset_id):
        return {"success": True, "message": f"Removed dataset: {dataset_id}"}

    raise HTTPException(
        status_code=404,
        detail=f"Dataset not found: {dataset_id}",
    )
