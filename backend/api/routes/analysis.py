"""
Dataset Analysis API routes.

Provides high-level dataset analysis endpoints:
- Frame count distribution (zero download, uses HF tree API file sizes)
- Batch actions/IMU signal comparison (parallel MCAP downloads, no video decode)
"""
import asyncio
import json
import logging
import math
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Models ---

class EpisodeFrameCount(BaseModel):
    episode_id: str
    estimated_frames: int
    size_bytes: int
    file_name: str


class FrameCountDistribution(BaseModel):
    task_name: str
    episodes: List[EpisodeFrameCount]
    total_episodes: int
    mean_frames: float
    std_frames: float
    min_frames: int
    max_frames: int
    outlier_episode_ids: List[str]
    source: str  # "file_size_estimate" or "metadata"


class EpisodeSignals(BaseModel):
    episode_id: str
    actions: Optional[Dict[str, Any]] = None
    imu: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# --- HF Token Helper ---

def _get_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    cache_token = Path.home() / ".cache" / "huggingface" / "token"
    if cache_token.exists():
        return cache_token.read_text().strip()
    return None


def _get_hf_headers() -> Dict[str, str]:
    headers = {}
    token = _get_hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# --- Frame Count Estimation ---

# Bytes per frame heuristic for MCAP with H.264 compressed video
MCAP_BYTES_PER_FRAME = 50 * 1024  # ~50KB per compressed frame


async def _collect_episode_files(
    client: httpx.AsyncClient,
    repo_id: str,
    task_folder: str,
    headers: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Recursively collect episode files from HF tree API, capturing file sizes.
    """
    encoded_path = urllib.parse.quote(task_folder, safe="")
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{encoded_path}"

    episode_extensions = {".mcap", ".tar", ".mp4", ".hdf5", ".h5", ".parquet"}
    episodes = []

    try:
        response = await client.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        items = response.json()

        items_to_process = list(items)
        processed_dirs = set()

        while items_to_process:
            item = items_to_process.pop(0)
            item_path = item.get("path", "")
            item_type = item.get("type", "")

            if item_type == "file":
                ext = Path(item_path).suffix.lower()
                if ext in episode_extensions:
                    relative_path = item_path
                    if relative_path.startswith(task_folder + "/"):
                        relative_path = relative_path[len(task_folder) + 1:]

                    episodes.append({
                        "episode_id": f"{task_folder}/{relative_path}",
                        "file_name": Path(item_path).name,
                        "size_bytes": item.get("size", 0),
                        "path": item_path,
                    })

            elif item_type == "directory" and item_path not in processed_dirs:
                depth = item_path.count("/") - task_folder.count("/")
                if depth < 3:
                    processed_dirs.add(item_path)
                    try:
                        sub_encoded = urllib.parse.quote(item_path, safe="")
                        sub_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{sub_encoded}"
                        sub_response = await client.get(sub_url, headers=headers, timeout=15.0)
                        if sub_response.status_code == 200:
                            items_to_process.extend(sub_response.json())
                    except Exception as e:
                        logger.warning(f"Failed to list subdirectory {item_path}: {e}")

        episodes.sort(key=lambda e: e["episode_id"])
        return episodes

    except Exception as e:
        logger.error(f"Failed to collect episode files: {e}")
        raise


def _compute_statistics(
    episodes: List[EpisodeFrameCount],
) -> Dict[str, Any]:
    """Compute frame count statistics and identify outliers."""
    if not episodes:
        return {
            "mean_frames": 0.0,
            "std_frames": 0.0,
            "min_frames": 0,
            "max_frames": 0,
            "outlier_episode_ids": [],
        }

    counts = [ep.estimated_frames for ep in episodes]
    mean = sum(counts) / len(counts)
    variance = sum((c - mean) ** 2 for c in counts) / max(len(counts) - 1, 1)
    std = math.sqrt(variance)

    # Outliers: episodes beyond 2 standard deviations from mean
    outliers = []
    if std > 0:
        for ep in episodes:
            z_score = abs(ep.estimated_frames - mean) / std
            if z_score > 2.0:
                outliers.append(ep.episode_id)

    return {
        "mean_frames": round(mean, 1),
        "std_frames": round(std, 1),
        "min_frames": min(counts),
        "max_frames": max(counts),
        "outlier_episode_ids": outliers,
    }


# --- Endpoints ---

@router.get("/{dataset_id}/analysis/frame-counts")
async def get_frame_count_distribution(
    dataset_id: str,
    task_name: str = Query(..., description="Task folder name"),
):
    """
    Get frame count distribution for all episodes in a task.

    Zero download required — uses HF tree API file sizes with heuristic estimation.
    """
    from downloaders.manager import get_all_datasets

    all_datasets = get_all_datasets()
    dataset_info = all_datasets.get(dataset_id)
    if not dataset_info:
        return {"error": f"Dataset '{dataset_id}' not found"}

    repo_id = dataset_info.get("repo_id", "")
    if not repo_id:
        return {"error": f"No repo_id for dataset '{dataset_id}'"}

    headers = _get_hf_headers()

    async with httpx.AsyncClient() as client:
        raw_episodes = await _collect_episode_files(client, repo_id, task_name, headers)

    # Estimate frame counts from file sizes
    episode_counts = []
    for ep in raw_episodes:
        size = ep["size_bytes"]
        estimated = max(1, size // MCAP_BYTES_PER_FRAME) if size > 0 else 0
        episode_counts.append(EpisodeFrameCount(
            episode_id=ep["episode_id"],
            estimated_frames=estimated,
            size_bytes=size,
            file_name=ep["file_name"],
        ))

    stats = _compute_statistics(episode_counts)

    return FrameCountDistribution(
        task_name=task_name,
        episodes=episode_counts,
        total_episodes=len(episode_counts),
        source="file_size_estimate",
        **stats,
    )


@router.get("/{dataset_id}/analysis/signals")
async def get_signals_comparison(
    request: Request,
    dataset_id: str,
    task_name: str = Query(..., description="Task folder name"),
    max_episodes: int = Query(5, ge=1, le=20, description="Max episodes to analyze"),
):
    """
    Stream actions and IMU data for multiple episodes via SSE.

    Downloads MCAP files in parallel (no video decode) and streams results progressively.
    """
    from downloaders.manager import get_all_datasets

    all_datasets = get_all_datasets()
    dataset_info = all_datasets.get(dataset_id)
    if not dataset_info:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': f'Dataset {dataset_id} not found'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    repo_id = dataset_info.get("repo_id", "")

    async def signal_stream():
        try:
            # First, get episode list from HF tree API
            headers = _get_hf_headers()
            async with httpx.AsyncClient() as client:
                raw_episodes = await _collect_episode_files(client, repo_id, task_name, headers)

            episodes_to_analyze = raw_episodes[:max_episodes]
            total = len(episodes_to_analyze)

            yield f"data: {json.dumps({'type': 'total', 'total_episodes': total})}\n\n"

            # Import extractor
            from loaders.streaming_extractor import StreamingFrameExtractor
            extractor = StreamingFrameExtractor(repo_id)

            # Semaphore for parallel downloads (limit concurrency)
            semaphore = asyncio.Semaphore(3)

            async def process_episode(idx: int, ep: Dict[str, Any]):
                episode_id = ep["episode_id"]
                episode_path = ep["path"]

                async with semaphore:
                    # Check for client disconnection
                    if await request.is_disconnected():
                        return None

                    yield_data = {
                        "type": "progress",
                        "phase": "downloading",
                        "episode_index": idx,
                        "total": total,
                        "episode_id": episode_id,
                    }

                    # Download + extract in thread pool (blocking I/O)
                    try:
                        loop = asyncio.get_event_loop()

                        # Extract actions
                        actions_data = await loop.run_in_executor(
                            None, extractor.extract_actions_data, episode_path
                        )

                        # Extract IMU
                        imu_data = await loop.run_in_executor(
                            None, extractor.extract_imu_data, episode_path
                        )

                        # Downsample if too many points (>2000 per episode)
                        actions_data = _downsample_actions(actions_data, max_points=2000)
                        imu_data = _downsample_imu(imu_data, max_points=2000)

                        return {
                            "type": "episode_data",
                            "episode_id": episode_id,
                            "episode_index": idx,
                            "actions": actions_data,
                            "imu": imu_data,
                        }

                    except Exception as e:
                        logger.error(f"Failed to process episode {episode_id}: {e}")
                        return {
                            "type": "episode_data",
                            "episode_id": episode_id,
                            "episode_index": idx,
                            "actions": {"error": str(e)},
                            "imu": {"error": str(e)},
                        }

            # Process episodes with concurrency control
            # We can't use asyncio.gather with the semaphore yielding pattern,
            # so process sequentially but downloads are cached after first fetch
            for idx, ep in enumerate(episodes_to_analyze):
                if await request.is_disconnected():
                    break

                episode_id = ep["episode_id"]
                episode_path = ep["path"]

                # Send progress
                yield f"data: {json.dumps({'type': 'progress', 'phase': 'processing', 'episode_index': idx, 'total': total, 'episode_id': episode_id})}\n\n"

                try:
                    loop = asyncio.get_event_loop()

                    # Extract actions (downloads MCAP if not cached, no video decode)
                    actions_data = await loop.run_in_executor(
                        None, extractor.extract_actions_data, episode_path
                    )

                    # Extract IMU (file already cached from actions extraction)
                    imu_data = await loop.run_in_executor(
                        None, extractor.extract_imu_data, episode_path
                    )

                    # Downsample if too many points
                    actions_data = _downsample_actions(actions_data, max_points=2000)
                    imu_data = _downsample_imu(imu_data, max_points=2000)

                    result = {
                        "type": "episode_data",
                        "episode_id": episode_id,
                        "episode_index": idx,
                        "actions": actions_data,
                        "imu": imu_data,
                    }

                except Exception as e:
                    logger.error(f"Failed to process episode {episode_id}: {e}")
                    result = {
                        "type": "episode_data",
                        "episode_id": episode_id,
                        "episode_index": idx,
                        "actions": {"error": str(e)},
                        "imu": {"error": str(e)},
                    }

                yield f"data: {json.dumps(result)}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Signal analysis failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(signal_stream(), media_type="text/event-stream")


# --- Downsampling helpers ---

def _downsample_actions(data: Dict[str, Any], max_points: int = 2000) -> Dict[str, Any]:
    """Downsample action data if it exceeds max_points."""
    if "error" in data:
        return data
    actions = data.get("actions", [])
    timestamps = data.get("timestamps", [])
    if len(actions) <= max_points:
        return data

    step = len(actions) / max_points
    indices = [int(i * step) for i in range(max_points)]
    return {
        "timestamps": [timestamps[i] for i in indices],
        "actions": [actions[i] for i in indices],
        "dimension_labels": data.get("dimension_labels"),
    }


def _downsample_imu(data: Dict[str, Any], max_points: int = 2000) -> Dict[str, Any]:
    """Downsample IMU data if it exceeds max_points."""
    if "error" in data:
        return data
    timestamps = data.get("timestamps", [])
    if len(timestamps) <= max_points:
        return data

    step = len(timestamps) / max_points
    indices = [int(i * step) for i in range(max_points)]
    result = {"timestamps": [timestamps[i] for i in indices]}
    for key in ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]:
        arr = data.get(key, [])
        if arr:
            result[key] = [arr[i] for i in indices]
        else:
            result[key] = []
    return result
