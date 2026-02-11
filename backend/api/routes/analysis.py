"""
Dataset Analysis API routes.

Provides high-level dataset analysis endpoints:
- Format detection and capability reporting
- Frame count distribution (zero download, uses HF tree API file sizes)
- Batch actions/IMU signal comparison (parallel downloads, no video decode)

Supports multiple formats:
- MCAP (RealOmni): Uses StreamingFrameExtractor for actions + IMU
- LeRobot/Parquet (Libero, etc.): Downloads tiny Parquet files for actions (no IMU)
"""
import asyncio
import io
import json
import logging
import math
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
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
    source_note: str = ""


class EpisodeSignals(BaseModel):
    episode_id: str
    actions: Optional[Dict[str, Any]] = None
    imu: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DatasetCapabilities(BaseModel):
    format: str
    has_actions: bool
    has_imu: bool
    supports_frame_counts: bool
    supports_signal_comparison: bool
    signal_comparison_note: str


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


# --- Format Detection & Capabilities ---

# Bytes-per-frame heuristics by format
BYTES_PER_FRAME = {
    "mcap": 50 * 1024,       # ~50KB per compressed frame (with video+depth)
    "webdataset": None,       # Use duration-based estimate
    "video": None,            # Use duration-based estimate
    "lerobot": None,          # Use parquet row counts
    "hdf5": 20 * 1024,       # ~20KB per frame
}

# Size threshold for signal extraction (500 MB)
SIGNAL_SIZE_THRESHOLD_MB = 500


def _detect_format(dataset_info: Dict[str, Any], episode_files: List[Dict[str, Any]]) -> str:
    """
    Detect the dataset format from config or file extensions.

    Returns: "mcap", "lerobot", "webdataset", "video", "hdf5", or "unknown"
    """
    fmt = dataset_info.get("format")
    if fmt:
        return fmt

    # Infer from file extensions
    extensions = {Path(ep["path"]).suffix.lower() for ep in episode_files}
    if ".parquet" in extensions:
        return "lerobot"
    if ".mcap" in extensions:
        return "mcap"
    if ".tar" in extensions:
        return "webdataset"
    if extensions & {".mp4", ".avi", ".mov", ".mkv"}:
        return "video"
    if extensions & {".hdf5", ".h5"}:
        return "hdf5"
    return "unknown"


def _get_format_metadata(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get format metadata from registry config.

    Returns: {format, has_actions, has_imu, max_file_size_mb}
    """
    fmt = dataset_info.get("format") or "unknown"
    return {
        "format": fmt,
        "has_actions": dataset_info.get("has_actions", fmt in ("mcap", "lerobot")),
        "has_imu": dataset_info.get("has_imu", False),
        "max_file_size_mb": dataset_info.get("max_episode_size_mb"),
    }


def _get_capabilities(dataset_info: Dict[str, Any]) -> DatasetCapabilities:
    """Build capabilities object for a dataset."""
    meta = _get_format_metadata(dataset_info)
    fmt = meta["format"]
    has_actions = meta["has_actions"]
    has_imu = meta["has_imu"]
    max_size_mb = meta["max_file_size_mb"]

    supports_frame_counts = True
    supports_signals = True
    note = ""

    if not has_actions and fmt != "unknown":
        supports_signals = False
        if fmt in ("webdataset", "video"):
            note = "Video-only dataset — no robot action or IMU signals available"
        else:
            note = "This dataset does not contain action signals"
    elif max_size_mb is not None and max_size_mb > SIGNAL_SIZE_THRESHOLD_MB:
        supports_signals = False
        note = f"Episode files are too large for signal extraction (~{max_size_mb:,.0f} MB each). Frame count analysis is still available."
    elif fmt in ("webdataset", "video"):
        supports_signals = False
        note = "Signal comparison is not supported for this format"

    return DatasetCapabilities(
        format=fmt,
        has_actions=has_actions,
        has_imu=has_imu,
        supports_frame_counts=supports_frame_counts,
        supports_signal_comparison=supports_signals,
        signal_comparison_note=note,
    )


def _episode_index_from_id(episode_id: str) -> Optional[int]:
    """Extract the numeric episode index from an episode_id like 'episode_18'."""
    import re
    m = re.match(r"episode_(\d+)", episode_id)
    return int(m.group(1)) if m else None


def _estimate_frames(size_bytes: int, fmt: str) -> int:
    """Estimate frame count from file size based on format."""
    if fmt == "mcap":
        bpf = BYTES_PER_FRAME["mcap"]
        return max(1, size_bytes // bpf) if size_bytes > 0 else 0
    elif fmt == "hdf5":
        bpf = BYTES_PER_FRAME["hdf5"]
        return max(1, size_bytes // bpf) if size_bytes > 0 else 0
    elif fmt in ("webdataset", "video"):
        # Estimate from typical video bitrate (~5 Mbps, 30fps)
        if size_bytes > 0:
            duration_secs = size_bytes / (5 * 1024 * 1024 / 8)  # 5 Mbps bitrate
            return max(1, int(duration_secs * 30))  # 30 fps
        return 0
    elif fmt == "lerobot":
        # Parquet: rough estimate (~200 bytes/row for robotics data)
        return max(1, size_bytes // 200) if size_bytes > 0 else 0
    else:
        # Default fallback: use MCAP heuristic
        bpf = BYTES_PER_FRAME["mcap"]
        return max(1, size_bytes // bpf) if size_bytes > 0 else 0


def _get_source_note(fmt: str) -> str:
    """Return human-readable explanation of estimation method."""
    notes = {
        "mcap": "Frame counts estimated from file sizes (~50KB/frame for MCAP with compressed video).",
        "webdataset": "Frame counts estimated from file sizes assuming ~5 Mbps video bitrate at 30 fps.",
        "video": "Frame counts estimated from file sizes assuming ~5 Mbps video bitrate at 30 fps.",
        "lerobot": "Frame counts estimated from Parquet file sizes (~200 bytes/row).",
        "hdf5": "Frame counts estimated from file sizes (~20KB/frame for HDF5).",
    }
    base = notes.get(fmt, "Frame counts estimated from file sizes.")
    return f"{base} A well-curated dataset should approximate a normal distribution."


# --- LeRobot/Parquet Signal Extraction ---

_SIGNAL_COLUMNS = ["action", "episode_index", "frame_index", "timestamp", "index", "episode_id"]


async def _download_parquet(
    client: httpx.AsyncClient,
    repo_id: str,
    file_path: str,
    headers: Dict[str, str],
) -> pd.DataFrame:
    """
    Download only signal-related columns from a HF Parquet file.

    Uses pyarrow with HfFileSystem to fetch only needed columns via HTTP range
    requests, reducing downloads from ~60MB (with images) to ~1-2MB.
    Falls back to full download for small files or on error.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    token = _get_hf_token()
    loop = asyncio.get_event_loop()

    def _read_columns():
        fs = HfFileSystem(token=token)
        hf_path = f"datasets/{repo_id}/{file_path}"
        schema = pq.read_schema(hf_path, filesystem=fs)
        available = set(schema.names)
        cols = [c for c in _SIGNAL_COLUMNS if c in available]
        if not cols:
            return pd.DataFrame()
        table = pq.read_table(hf_path, filesystem=fs, columns=cols)
        return table.to_pandas()

    try:
        return await loop.run_in_executor(None, _read_columns)
    except Exception as e:
        logger.warning(f"Column-filtered read failed for {file_path}, falling back to full download: {e}")
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_path}"
        response = await client.get(url, headers=headers, timeout=120.0, follow_redirects=True)
        response.raise_for_status()
        return pd.read_parquet(io.BytesIO(response.content))


async def _extract_lerobot_episodes(
    client: httpx.AsyncClient,
    repo_id: str,
    parquet_files: List[Dict[str, Any]],
    headers: Dict[str, str],
    max_episodes: int,
    filter_episodes: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Download LeRobot Parquet files and extract per-episode action data.

    Parquet files are tiny (~50KB, no images) so this is fast.
    Groups rows by episode_index and returns action vectors per episode.
    Downloads incrementally — stops once enough episodes are collected.

    If filter_episodes is provided, only episodes with indices in that set
    are counted toward max_episodes and included in the result.
    """
    all_dfs = []
    relevant_episodes = set()

    for pf in parquet_files:
        try:
            df = await _download_parquet(client, repo_id, pf["path"], headers)
            all_dfs.append(df)

            # Track unique episodes seen so far (only relevant ones)
            ep_col = "episode_index" if "episode_index" in df.columns else "episode_id"
            if ep_col in df.columns:
                found = set(int(x) for x in df[ep_col].unique())
                if filter_episodes is not None:
                    relevant_episodes.update(found & filter_episodes)
                else:
                    relevant_episodes.update(found)

            # Stop downloading once we have enough relevant episodes
            if len(relevant_episodes) >= max_episodes:
                logger.info(f"Downloaded {len(all_dfs)} parquet files, found {len(relevant_episodes)} relevant episodes (need {max_episodes})")
                break

        except Exception as e:
            logger.warning(f"Failed to download {pf['path']}: {e}")

    if not all_dfs:
        return []

    combined = pd.concat(all_dfs, ignore_index=True)

    # Group by episode_index
    episode_col = "episode_index" if "episode_index" in combined.columns else None
    if episode_col is None:
        # Try episode_id
        if "episode_id" in combined.columns:
            episode_col = "episode_id"
        else:
            logger.error(f"No episode column found. Columns: {list(combined.columns)}")
            return []

    # Sort by episode and frame index
    sort_cols = [episode_col]
    if "frame_index" in combined.columns:
        sort_cols.append("frame_index")
    elif "index" in combined.columns:
        sort_cols.append("index")
    combined = combined.sort_values(sort_cols)

    episodes = []
    for ep_idx, group in combined.groupby(episode_col):
        if len(episodes) >= max_episodes:
            break

        ep_idx_int = int(ep_idx)

        # Skip episodes not in filter set
        if filter_episodes is not None and ep_idx_int not in filter_episodes:
            continue

        episode_id = f"episode_{ep_idx_int}"

        # Extract timestamps
        timestamps = []
        if "timestamp" in group.columns:
            timestamps = group["timestamp"].tolist()
        elif "frame_index" in group.columns:
            # Synthesize timestamps from frame index (assume 10Hz default for LeRobot)
            timestamps = (group["frame_index"] / 10.0).tolist()
        else:
            timestamps = [i / 10.0 for i in range(len(group))]

        # Extract actions
        actions_data: Dict[str, Any] = {"timestamps": [], "actions": [], "dimension_labels": None}
        if "action" in group.columns:
            action_list = group["action"].tolist()
            # Convert numpy arrays to lists
            actions = []
            for a in action_list:
                if isinstance(a, np.ndarray):
                    actions.append(a.tolist())
                elif isinstance(a, (list, tuple)):
                    actions.append(list(a))
                else:
                    actions.append([float(a)])
            actions_data["timestamps"] = timestamps
            actions_data["actions"] = actions

            # Infer dimension labels
            if actions:
                num_dims = len(actions[0])
                if num_dims == 7:
                    actions_data["dimension_labels"] = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
                elif num_dims == 6:
                    actions_data["dimension_labels"] = ["x", "y", "z", "rx", "ry", "rz"]
                elif num_dims == 3:
                    actions_data["dimension_labels"] = ["x", "y", "z"]
        else:
            actions_data = {"error": "No 'action' column in Parquet data"}

        episodes.append({
            "episode_id": episode_id,
            "episode_index": len(episodes),
            "global_episode_index": ep_idx_int,
            "actions": actions_data,
            "imu": None,  # LeRobot datasets don't have IMU
        })

    return episodes


# --- Episode Collection ---

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

@router.get("/{dataset_id}/analysis/capabilities")
async def get_dataset_capabilities(dataset_id: str):
    """
    Get analysis capabilities for a dataset.

    Returns what types of analysis are supported based on format and file sizes.
    """
    from downloaders.manager import get_all_datasets

    all_datasets = get_all_datasets()
    dataset_info = all_datasets.get(dataset_id)
    if not dataset_info:
        return {"error": f"Dataset '{dataset_id}' not found"}

    return _get_capabilities(dataset_info)


async def _lerobot_frame_counts_for_task(
    repo_id: str, task_name: str,
) -> Optional[FrameCountDistribution]:
    """
    Get exact frame counts for a LeRobot dataset task using metadata parquet.

    Returns None if metadata is unavailable (caller should fall back to file-size estimation).
    """
    from api.routes.datasets import (
        fetch_lerobot_episodes_meta,
        fetch_lerobot_tasks_meta,
        fetch_lerobot_episode_task_map,
        fetch_lerobot_info,
    )
    import re

    episodes_df = await fetch_lerobot_episodes_meta(repo_id)
    tasks_df = await fetch_lerobot_tasks_meta(repo_id)
    info = await fetch_lerobot_info(repo_id)
    if episodes_df is None or tasks_df is None:
        return None

    # Resolve task_name to task_index
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
        untitled_match = re.match(r"^Untitled \(task (\d+)\)$", task_name)
        if untitled_match:
            candidate_idx = int(untitled_match.group(1))
            if candidate_idx in tasks_df["task_index"].values:
                task_index = candidate_idx

    if task_index is None:
        return None

    # Get episodes for this task
    ep_task_map = await fetch_lerobot_episode_task_map(repo_id)
    if ep_task_map is None:
        return None

    task_ep_indices = sorted([
        ep_idx for ep_idx, t_idx in ep_task_map.items()
        if t_idx == task_index
    ])

    task_episodes = episodes_df[episodes_df["episode_index"].isin(task_ep_indices)]
    task_episodes = task_episodes.sort_values("episode_index")

    fps = info.get("fps", 30) if info else 30

    episode_counts = []
    for local_idx, (_, row) in enumerate(task_episodes.iterrows()):
        ep_idx = int(row["episode_index"])
        length = int(row["length"]) if "length" in row.index else 0
        episode_counts.append(EpisodeFrameCount(
            episode_id=f"episode_{local_idx}",
            estimated_frames=length,
            size_bytes=0,
            file_name=f"episode_{local_idx} ({length} frames, {length/fps:.1f}s)",
        ))

    stats = _compute_statistics(episode_counts)

    return FrameCountDistribution(
        task_name=task_name,
        episodes=episode_counts,
        total_episodes=len(episode_counts),
        source="metadata",
        source_note="Exact frame counts from LeRobot episode metadata (no estimation needed).",
        **stats,
    )


@router.get("/{dataset_id}/analysis/frame-counts")
async def get_frame_count_distribution(
    dataset_id: str,
    task_name: str = Query(..., description="Task folder name"),
):
    """
    Get frame count distribution for all episodes in a task.

    For LeRobot datasets, uses exact frame counts from metadata parquet.
    For other formats, uses HF tree API file sizes with format-specific heuristics.
    """
    from downloaders.manager import get_all_datasets
    from api.routes.datasets import is_lerobot_dataset

    all_datasets = get_all_datasets()
    dataset_info = all_datasets.get(dataset_id)
    if not dataset_info:
        return {"error": f"Dataset '{dataset_id}' not found"}

    repo_id = dataset_info.get("repo_id", "")
    if not repo_id:
        return {"error": f"No repo_id for dataset '{dataset_id}'"}

    # For LeRobot datasets, use exact metadata instead of file-size estimation
    if dataset_info.get("streaming_recommended") and await is_lerobot_dataset(dataset_info):
        result = await _lerobot_frame_counts_for_task(repo_id, task_name)
        if result is not None:
            return result
        logger.warning(f"LeRobot metadata unavailable for {dataset_id}, falling back to file-size estimation")

    headers = _get_hf_headers()

    try:
        async with httpx.AsyncClient() as client:
            raw_episodes = await _collect_episode_files(client, repo_id, task_name, headers)
    except Exception as e:
        logger.error(f"Failed to collect episode files for task '{task_name}': {e}")
        return {"error": f"Task '{task_name}' not found in dataset '{dataset_id}'"}

    if not raw_episodes:
        return {"error": f"No episode files found for task '{task_name}'"}

    # Detect format for appropriate heuristic
    fmt = _detect_format(dataset_info, raw_episodes)

    # Estimate frame counts using format-specific heuristic
    episode_counts = []
    for ep in raw_episodes:
        size = ep["size_bytes"]
        estimated = _estimate_frames(size, fmt)
        episode_counts.append(EpisodeFrameCount(
            episode_id=ep["episode_id"],
            estimated_frames=estimated,
            size_bytes=size,
            file_name=ep["file_name"],
        ))

    stats = _compute_statistics(episode_counts)
    source_note = _get_source_note(fmt)

    return FrameCountDistribution(
        task_name=task_name,
        episodes=episode_counts,
        total_episodes=len(episode_counts),
        source="file_size_estimate",
        source_note=source_note,
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

    Checks dataset capabilities first; sends no_signals event for unsupported formats.
    Supports:
    - MCAP (RealOmni): Downloads MCAP files, extracts actions + IMU
    - LeRobot/Parquet (Libero, etc.): Downloads tiny Parquet files, extracts actions
    """
    from downloaders.manager import get_all_datasets

    all_datasets = get_all_datasets()
    dataset_info = all_datasets.get(dataset_id)
    if not dataset_info:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': f'Dataset {dataset_id} not found'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    # Check capabilities before attempting extraction
    capabilities = _get_capabilities(dataset_info)
    if not capabilities.supports_signal_comparison:
        async def no_signals_gen():
            yield f"data: {json.dumps({'type': 'no_signals', 'reason': capabilities.signal_comparison_note})}\n\n"
        return StreamingResponse(no_signals_gen(), media_type="text/event-stream")

    repo_id = dataset_info.get("repo_id", "")

    async def signal_stream():
        try:
            headers = _get_hf_headers()
            fmt = dataset_info.get("format") or "unknown"

            # For LeRobot datasets, use metadata to find data parquet files
            is_lerobot = False
            if dataset_info.get("streaming_recommended"):
                from api.routes.datasets import is_lerobot_dataset
                is_lerobot = await is_lerobot_dataset(dataset_info)

            if is_lerobot:
                # --- LeRobot/Parquet path using metadata ---
                fmt = "lerobot"

                # Get task's episode indices and metadata
                from api.routes.datasets import (
                    fetch_lerobot_tasks_meta,
                    fetch_lerobot_episode_task_map,
                    fetch_lerobot_episodes_meta,
                )
                import re as _re

                tasks_df = await fetch_lerobot_tasks_meta(repo_id)
                ep_task_map = await fetch_lerobot_episode_task_map(repo_id)

                task_ep_indices = None
                if tasks_df is not None and ep_task_map is not None:
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
                    if task_index is None:
                        untitled_match = _re.match(r"^Untitled \(task (\d+)\)$", task_name)
                        if untitled_match:
                            candidate_idx = int(untitled_match.group(1))
                            if candidate_idx in tasks_df["task_index"].values:
                                task_index = candidate_idx

                    if task_index is not None:
                        task_ep_indices = set(
                            ep_idx for ep_idx, t_idx in ep_task_map.items()
                            if t_idx == task_index
                        )

                # Fetch episode metadata for true frame counts
                episodes_meta_df = await fetch_lerobot_episodes_meta(repo_id)
                ep_frame_counts: Dict[int, int] = {}
                if episodes_meta_df is not None and "length" in episodes_meta_df.columns:
                    for _, row in episodes_meta_df.iterrows():
                        ep_frame_counts[int(row["episode_index"])] = int(row["length"])

                # List data parquet files from data/ directory
                yield f"data: {json.dumps({'type': 'progress', 'phase': 'processing', 'episode_index': 0, 'total': max_episodes, 'episode_id': 'downloading parquet data...'})}\n\n"

                async with httpx.AsyncClient(timeout=60.0) as client:
                    data_files = await _collect_episode_files(client, repo_id, "data", headers)

                async with httpx.AsyncClient(timeout=60.0) as client:
                    episodes = await _extract_lerobot_episodes(
                        client, repo_id, data_files, headers,
                        max_episodes=max_episodes,
                        filter_episodes=task_ep_indices,
                    )

                # Re-index to task-local indices, preserving true frame counts
                for i, ep in enumerate(episodes):
                    global_idx = ep.get("global_episode_index")
                    ep["total_frames"] = ep_frame_counts.get(global_idx) if global_idx is not None else None
                    ep["episode_index"] = i
                    ep["episode_id"] = f"episode_{i}"

                total = len(episodes)
                yield f"data: {json.dumps({'type': 'total', 'total_episodes': total})}\n\n"

                for ep in episodes:
                    if await request.is_disconnected():
                        break

                    actions_data = _downsample_actions(ep["actions"], max_points=2000)
                    result = {
                        "type": "episode_data",
                        "episode_id": ep["episode_id"],
                        "episode_index": ep["episode_index"],
                        "actions": actions_data,
                        "imu": ep["imu"],
                        "total_frames": ep.get("total_frames"),
                    }
                    yield f"data: {json.dumps(result)}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            else:
                # Non-LeRobot: use directory-based file collection
                async with httpx.AsyncClient(timeout=60.0) as client:
                    raw_episodes = await _collect_episode_files(client, repo_id, task_name, headers)

                fmt = _detect_format(dataset_info, raw_episodes)
                logger.info(f"Signal analysis for {dataset_id}: format={fmt}, {len(raw_episodes)} files found")

                if fmt == "mcap":
                    # --- MCAP path (original) ---
                    episodes_to_analyze = raw_episodes[:max_episodes]
                    total = len(episodes_to_analyze)

                    yield f"data: {json.dumps({'type': 'total', 'total_episodes': total})}\n\n"

                    from loaders.streaming_extractor import StreamingFrameExtractor
                    extractor = StreamingFrameExtractor(repo_id)

                    for idx, ep in enumerate(episodes_to_analyze):
                        if await request.is_disconnected():
                            break

                        episode_id = ep["episode_id"]
                        episode_path = ep["path"]

                        yield f"data: {json.dumps({'type': 'progress', 'phase': 'processing', 'episode_index': idx, 'total': total, 'episode_id': episode_id})}\n\n"

                        try:
                            loop = asyncio.get_event_loop()

                            actions_data = await loop.run_in_executor(
                                None, extractor.extract_actions_data, episode_path
                            )
                            imu_data = await loop.run_in_executor(
                                None, extractor.extract_imu_data, episode_path
                            )

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

                else:
                    yield f"data: {json.dumps({'type': 'no_signals', 'reason': f'Signal comparison is not supported for {fmt} format'})}\n\n"

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
