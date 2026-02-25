"""
API routes for episode operations.

Endpoints:
- GET /api/episodes/{id} - Get episode metadata
- GET /api/episodes/{id}/frames - Get frames from episode
- GET /api/episodes/{id}/frames/stream - Stream frames via SSE (Server-Sent Events)
- GET /api/episodes/{id}/imu - Get IMU data from episode
"""
import asyncio
import base64
import concurrent.futures
import json
import logging
import math
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from downloaders.manager import DATASET_REGISTRY, get_all_datasets
from loaders import HDF5Loader, WebDatasetLoader, LeRobotLoader, RLDSLoader
from loaders.streaming_extractor import StreamingFrameExtractor, cleanup_decoded_frames
from cache import get_encoded_frame_cache
from adapters import FormatRegistry

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared thread pool for CPU-bound operations (video decoding, encoding)
# 8 workers provides headroom for 5 parallel caching + 3 for API calls
_HEAVY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="heavy_ops"
)

# Dedicated thread pool for parallel frame encoding within batches
# More workers = faster encoding (WebP encoding releases GIL)
import multiprocessing
_ENCODE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(8, multiprocessing.cpu_count()),
    thread_name_prefix="encode"
)

# Semaphore to limit concurrent caching operations to prevent resource exhaustion
_CACHING_SEMAPHORE = asyncio.Semaphore(5)

# Background caching state - tracks which episodes are being cached
# and their progress (0-100%)
_caching_tasks: Dict[str, asyncio.Task] = {}
_caching_progress: Dict[str, Dict[str, Any]] = {}  # {key: {progress: 0-100, status: str, total_frames: int}}


def _compute_stride_target(total_frames: int) -> int:
    """Compute stride target based on episode size.

    Balances storage savings with playback quality.
    Target ~300 frames for most episodes (10s at 30fps playback).
    """
    if total_frames <= 300:
        return total_frames
    elif total_frames <= 1000:
        return 300
    elif total_frames <= 5000:
        return 300
    else:
        return 300


class EpisodeDetail(BaseModel):
    """Detailed episode information."""

    id: str
    task_name: Optional[str] = None
    description: Optional[str] = None
    num_frames: int
    duration_seconds: Optional[float] = None
    has_observations: bool = False
    has_actions: bool = False
    has_video: bool = False
    metadata: dict = {}


class FrameData(BaseModel):
    """Single frame data."""

    frame_idx: int
    timestamp: Optional[float] = None
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    action: Optional[List[float]] = None


class FramesResponse(BaseModel):
    """Response containing frames and metadata."""

    frames: List[FrameData]
    total_frames: Optional[int] = None
    from_cache: bool = False


def get_data_root(request: Request) -> Path:
    """Get data root from app state."""
    return request.app.state.data_root


def parse_episode_id(episode_id: str) -> tuple:
    """
    Parse episode ID to extract dataset and internal ID.

    Episode IDs are typically: dataset_id/path/to/episode
    """
    all_datasets = get_all_datasets()
    parts = episode_id.split("/", 1)
    if len(parts) == 1:
        # Assume it's just the episode ID, try to find dataset
        return None, episode_id

    # First part might be dataset ID
    if parts[0] in all_datasets:
        return parts[0], parts[1]

    # Or it might be a task suite (e.g., libero_spatial)
    for dataset_id, config in all_datasets.items():
        if parts[0].startswith(dataset_id.replace("_", "")):
            return dataset_id, episode_id

    return None, episode_id


def get_loader_for_dataset(dataset_id: str, data_root: Path):
    """Get the appropriate loader for a dataset."""
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

    # Fall back to default loaders
    if dataset_id in ["libero", "libero_pro"]:
        return HDF5Loader(data_dir)

    return None


def get_loader_for_episode(episode_id: str, data_root: Path, dataset_id: str = None):
    """Get the appropriate loader for an episode."""
    # If dataset_id is provided, use it directly
    if dataset_id:
        loader = get_loader_for_dataset(dataset_id, data_root)
        if loader:
            return loader, episode_id

    # Parse episode_id to extract dataset info
    parsed_dataset_id, internal_id = parse_episode_id(episode_id)

    # First, try to determine which dataset based on episode_id prefix
    # LIBERO episodes are formatted as: libero_{suite}/{task_name}/{demo_key}
    episode_parts = episode_id.split("/")

    # Check if this is a LIBERO HDF5 episode (starts with libero_*)
    if episode_parts[0].startswith("libero_") and not episode_parts[0].startswith("libero_lerobot"):
        # This is a LIBERO episode - try libero and libero_pro loaders only
        for ds_id in ["libero", "libero_pro"]:
            data_dir = data_root / ds_id
            if data_dir.exists():
                loader = HDF5Loader(data_dir)
                try:
                    episodes = loader.list_episodes()
                    for ep in episodes:
                        if ep.id == episode_id or ep.id == internal_id:
                            return loader, ep.id
                except Exception:
                    pass
        return None, None

    # Check if this is a LeRobot episode (format: episode_X)
    if episode_id.startswith("episode_"):
        data_dir = data_root / "libero_lerobot"
        if data_dir.exists():
            loader = LeRobotLoader(data_dir)
            try:
                episodes = loader.list_episodes()
                for ep in episodes:
                    if ep.id == episode_id:
                        return loader, ep.id
            except Exception:
                pass

    # Check if this is a Bridge episode (format: bridge_X)
    if episode_id.startswith("bridge_"):
        data_dir = data_root / "bridge_v2"
        if data_dir.exists():
            loader = RLDSLoader(data_dir)
            try:
                episodes = loader.list_episodes()
                for ep in episodes:
                    if ep.id == episode_id:
                        return loader, ep.id
            except Exception:
                pass

    # For other datasets, try to match by dataset_id first
    all_datasets = get_all_datasets()
    if parsed_dataset_id and parsed_dataset_id in all_datasets:
        loader = get_loader_for_dataset(parsed_dataset_id, data_root)
        if loader:
            try:
                episodes = loader.list_episodes()
                for ep in episodes:
                    if ep.id == episode_id or ep.id == internal_id:
                        return loader, ep.id
            except Exception:
                pass

    # Fallback: try each local dataset's loader (skip streaming datasets)
    for ds_id in all_datasets:
        config = all_datasets[ds_id]

        # Skip streaming datasets - they're too slow to enumerate
        if config.get("streaming_recommended") or config.get("type") == "video":
            continue

        data_dir = data_root / ds_id
        loader = get_loader_for_dataset(ds_id, data_root)

        if loader:
            try:
                episodes = loader.list_episodes()
                for ep in episodes:
                    if ep.id == episode_id or ep.id == internal_id:
                        return loader, ep.id
            except Exception:
                pass

    return None, None


RESOLUTION_MAP = {
    "low": (320, 240),
    "medium": (640, 480),
    "high": (960, 720),
    "original": None,
}


def encode_image_with_options(
    image: np.ndarray,
    resolution: str = "medium",
    quality: int = 70,
) -> str:
    """Encode numpy image array to base64 WebP with resolution and quality options."""
    try:
        from PIL import Image
        import cv2

        # Ensure correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Resize if needed - use INTER_LINEAR for speed (2x faster than INTER_AREA)
        target_size = RESOLUTION_MAP.get(resolution)
        if target_size and (image.shape[1] > target_size[0] or image.shape[0] > target_size[1]):
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Encode to WebP with method=0 for fastest compression
        buffer = BytesIO()
        pil_image.save(buffer, format="WEBP", quality=quality, method=0)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")
    except ImportError:
        # Fallback without PIL - return raw base64
        return base64.b64encode(image.tobytes()).decode("utf-8")


def encode_image_base64(image: np.ndarray) -> str:
    """Encode numpy image array to base64 JPEG (legacy compatibility)."""
    return encode_image_with_options(image, resolution="original", quality=85)


def _is_lerobot_episode_id(episode_id: str) -> bool:
    """Check if episode_id matches the LeRobot format: episode_N or prefix/episode_N."""
    import re
    return bool(re.match(r'^(.+/)?episode_\d+$', episode_id))


def _parse_lerobot_episode_id(episode_id: str) -> tuple:
    """
    Parse a LeRobot episode_id into (path_prefix, base_episode_id).

    e.g., "subdir/episode_0" -> ("subdir", "episode_0")
          "episode_0" -> ("", "episode_0")
    """
    import re
    match = re.match(r'^(.+)/(episode_\d+)$', episode_id)
    if match:
        return match.group(1), match.group(2)
    return "", episode_id


# Cache for LeRobot episode resolution (avoids repeated HF API calls)
_lerobot_resolution_cache: Dict[str, Dict[str, Any]] = {}


async def resolve_lerobot_episode(
    repo_id: str,
    episode_id: str,
    path_prefix: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Resolve a LeRobot episode_N ID to the video chunk path and frame range.

    Uses episode metadata from meta/episodes/ which contains explicit
    video chunk/file indices and from/to timestamps per episode.

    When path_prefix is set (for multi-subdataset repos), all paths are
    scoped under the subdataset prefix.

    Returns dict with:
        - video_path: path to the MP4 file in the HF repo
        - frame_start: start frame index within the video file
        - frame_end: end frame index within the video file (exclusive)
        - num_frames: number of frames in this episode
        - fps: frames per second
        - video_key: the observation key used for video
    Or None if resolution fails.
    """
    cache_key = f"{repo_id}/{path_prefix}/{episode_id}" if path_prefix else f"{repo_id}/{episode_id}"
    if cache_key in _lerobot_resolution_cache:
        return _lerobot_resolution_cache[cache_key]

    from .datasets import fetch_lerobot_info, fetch_lerobot_episodes_meta, detect_lerobot_data_branch

    # Parse episode index
    import re
    match = re.match(r'^episode_(\d+)$', episode_id)
    if not match:
        return None
    target_ep_idx = int(match.group(1))

    # Fetch metadata
    info = await fetch_lerobot_info(repo_id, path_prefix=path_prefix)
    if info is None:
        logger.warning(f"Could not fetch LeRobot info.json for {repo_id}")
        return None

    fps = info.get("fps", 30)
    video_path_template = info.get("video_path", "")

    # Determine the first video key from features
    video_key = None
    features = info.get("features", {})
    for feat_name, feat_info in features.items():
        if feat_info.get("dtype") == "video":
            video_key = feat_name
            break

    if not video_key or not video_path_template:
        logger.warning(f"No video feature or video_path template found in LeRobot info for {repo_id}")
        return None

    # Detect which branch has the actual data files
    data_branch = await detect_lerobot_data_branch(repo_id, path_prefix=path_prefix)

    episodes_df = await fetch_lerobot_episodes_meta(repo_id, path_prefix=path_prefix)

    if episodes_df is not None:
        # v3 path: use episode metadata for chunk/file indices and timestamps
        ep_row = episodes_df[episodes_df["episode_index"] == target_ep_idx]
        if len(ep_row) == 0:
            logger.warning(f"Episode {target_ep_idx} not found in metadata for {repo_id}")
            return None
        ep_row = ep_row.iloc[0]

        num_frames = int(ep_row["length"]) if "length" in ep_row.index else None
        if num_frames is None:
            return None

        # Get chunk_index and file_index from episode metadata
        vid_chunk_col = f"videos/{video_key}/chunk_index"
        vid_file_col = f"videos/{video_key}/file_index"
        vid_from_ts_col = f"videos/{video_key}/from_timestamp"
        vid_to_ts_col = f"videos/{video_key}/to_timestamp"

        chunk_index = int(ep_row[vid_chunk_col]) if vid_chunk_col in ep_row.index else target_ep_idx // info.get("chunks_size", 1000)
        file_index = int(ep_row[vid_file_col]) if vid_file_col in ep_row.index else 0

        # Build the video path - try template variables for both v3 and v2.1 naming
        try:
            video_path = video_path_template.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
                episode_chunk=chunk_index,
                episode_index=target_ep_idx,
            )
        except KeyError:
            video_path = video_path_template.format(
                video_key=video_key,
                episode_chunk=chunk_index,
                episode_index=target_ep_idx,
            )

        # Prepend subdataset prefix to video path for multi-subdataset repos
        if path_prefix:
            video_path = f"{path_prefix}/{video_path}"

        # Compute frame range using timestamps and FPS
        if vid_from_ts_col in ep_row.index and vid_to_ts_col in ep_row.index:
            from_ts = float(ep_row[vid_from_ts_col])
            to_ts = float(ep_row[vid_to_ts_col])
            frame_start = round(from_ts * fps)
            frame_end = round(to_ts * fps)
        else:
            # Fallback: compute from cumulative episode lengths in same chunk
            chunks_size = info.get("chunks_size", 1000)
            chunk_episodes = episodes_df[
                (episodes_df["episode_index"] >= chunk_index * chunks_size) &
                (episodes_df["episode_index"] < (chunk_index + 1) * chunks_size)
            ].sort_values("episode_index")

            frame_start = 0
            for _, row in chunk_episodes.iterrows():
                if int(row["episode_index"]) == target_ep_idx:
                    break
                frame_start += int(row["length"])
            frame_end = frame_start + num_frames

        result = {
            "video_path": video_path,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "num_frames": num_frames,
            "fps": fps,
            "video_key": video_key,
            "data_branch": data_branch,
        }

        _lerobot_resolution_cache[cache_key] = result
        logger.info(
            f"Resolved LeRobot {episode_id} -> {video_path} "
            f"frames [{frame_start}:{frame_end}] ({num_frames} frames)"
        )
        return result

    # v2.1 fallback: no meta/episodes/ metadata, each episode is a separate video file
    chunks_size = info.get("chunks_size", 1000)
    episode_chunk = target_ep_idx // chunks_size

    try:
        video_path = video_path_template.format(
            video_key=video_key,
            episode_chunk=episode_chunk,
            episode_index=target_ep_idx,
            chunk_index=episode_chunk,
            file_index=0,
        )
    except KeyError as e:
        logger.warning(f"Failed to format video_path template for {repo_id}: {e}")
        return None

    # Prepend subdataset prefix for multi-subdataset repos
    if path_prefix:
        video_path = f"{path_prefix}/{video_path}"

    result = {
        "video_path": video_path,
        "frame_start": 0,
        "frame_end": None,
        "num_frames": None,
        "fps": fps,
        "video_key": video_key,
        "single_episode_video": True,
        "data_branch": data_branch,
    }

    _lerobot_resolution_cache[cache_key] = result
    logger.info(
        f"Resolved LeRobot v2.1 {episode_id} -> {video_path} "
        f"(single episode video, branch={data_branch})"
    )
    return result


def is_streaming_episode(episode_id: str, dataset_id: Optional[str]) -> tuple:
    """
    Check if an episode is from a streaming HuggingFace dataset.

    Returns:
        Tuple of (is_streaming, repo_id, file_path)
    """
    all_datasets = get_all_datasets()

    # If dataset_id is provided, check if it's streaming
    if dataset_id and dataset_id in all_datasets:
        config = all_datasets[dataset_id]
        if config.get("streaming_recommended") and config.get("repo_id"):
            # Episode ID format for streaming: task_folder/subfolder/file.mcap
            return True, config["repo_id"], episode_id

    # Check if episode_id matches a streaming dataset pattern
    for ds_id, config in all_datasets.items():
        if config.get("streaming_recommended") and config.get("repo_id"):
            # Check if episode_id looks like a HuggingFace path
            if "/" in episode_id and (
                episode_id.endswith(".mcap") or
                episode_id.endswith(".tar") or
                episode_id.endswith(".mp4")
            ):
                return True, config["repo_id"], episode_id

    return False, None, None


async def get_streaming_frames(
    repo_id: str,
    file_path: str,
    start: int,
    end: int,
    resolution: str = "medium",
    quality: int = 70,
    dataset_id: str = None,
    stream: str = "rgb",
    revision: str = None,
) -> FramesResponse:
    """
    Get frames from a streaming HuggingFace episode.

    Uses full-episode caching to survive browser refresh regardless of batch boundaries.

    Args:
        repo_id: HuggingFace repository ID
        file_path: Path to the episode file within the repo
        start: Start frame index
        end: End frame index
        resolution: Image resolution (low/medium/high/original)
        quality: JPEG quality (10-100)
        dataset_id: Dataset identifier for caching
        stream: Which stream to extract: "rgb" or "depth"
        revision: Branch/tag to download from (e.g., "v2.0")
    """
    cache = get_encoded_frame_cache()
    effective_dataset_id = dataset_id or repo_id.replace("/", "_")

    # Include stream in cache key to cache different streams separately
    cache_key_suffix = f":{stream}" if stream != "rgb" else ""
    episode_key = cache.get_episode_cache_key(
        effective_dataset_id, file_path + cache_key_suffix, resolution, quality
    )

    # Check if full episode is cached
    cached_episode = cache.get_episode_frames(episode_key, effective_dataset_id, file_path)
    if cached_episode:
        # Return requested slice from cached full episode
        all_frames = cached_episode["frames"]
        total_frames = cached_episode["total"]

        # Clamp range to available frames
        actual_end = min(end, len(all_frames))
        slice_frames = all_frames[start:actual_end]

        frames = [
            FrameData(
                frame_idx=f["frame_idx"],
                timestamp=f.get("timestamp"),
                image_base64=f.get("image_base64"),
                action=f.get("action"),
            )
            for f in slice_frames
        ]
        logger.info(f"Serving frames {start}-{actual_end} from full episode cache ({total_frames} total)")
        return FramesResponse(frames=frames, total_frames=total_frames, from_cache=True)

    # Not cached: decode ALL frames for the episode, then cache
    extractor = StreamingFrameExtractor(repo_id)

    try:
        # Extract ALL frames (from 0 to a large number to get everything)
        logger.info(f"Decoding full episode for caching: {file_path} (stream={stream}, revision={revision})")
        raw_frames, total_frames, _ = extractor.extract_frames_with_count(
            file_path, 0, 999999, stream=stream, revision=revision
        )

        # Log decode result
        logger.info(f"Decoded {len(raw_frames)} frames for episode {file_path} (stream={stream})")

        # Convert ALL frames to FrameData with resolution/quality options
        all_frames_for_cache = []
        for frame_idx, timestamp, image in raw_frames:
            image_base64 = encode_image_with_options(image, resolution, quality)
            all_frames_for_cache.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "image_base64": image_base64,
                "action": None,  # Streaming datasets typically don't have action data
            })

        # Store FULL episode in cache
        cache.store_episode_frames(
            episode_key,
            all_frames_for_cache,
            len(all_frames_for_cache),  # Use actual decoded count as total
            {
                "dataset_id": effective_dataset_id,
                "episode_id": file_path,
                "resolution": resolution,
                "quality": quality,
            },
        )

        # Return only the requested slice
        actual_end = min(end, len(all_frames_for_cache))
        slice_frames = all_frames_for_cache[start:actual_end]

        frames = [
            FrameData(
                frame_idx=f["frame_idx"],
                timestamp=f.get("timestamp"),
                image_base64=f.get("image_base64"),
                action=f.get("action"),
            )
            for f in slice_frames
        ]

        return FramesResponse(frames=frames, total_frames=len(all_frames_for_cache), from_cache=False)

    except Exception as e:
        logger.error(f"Failed to extract streaming frames: {e}")
        raise


# NOTE: /frame-at route must come BEFORE /frames and the catch-all route
@router.get("/{episode_id:path}/frame-at")
async def get_frame_at(
    episode_id: str,
    request: Request,
    dataset_id: Optional[str] = Query(None, description="Dataset ID"),
    frame_index: Optional[int] = Query(None, ge=0, description="Frame index to extract"),
    timestamp: Optional[float] = Query(None, ge=0.0, description="Timestamp in seconds"),
    resolution: str = Query("low", description="Image resolution"),
    quality: int = Query(70, ge=10, le=100, description="WebP quality"),
):
    """
    Get a single frame at a specific index or timestamp.

    Optimised for the frame comparison strip: checks the encoded-frame cache
    first, and only falls back to the full /frames endpoint logic when the
    episode has not been cached yet.

    Priority:
    1. Cached episode frames (instant)
    2. Existing /frames endpoint as fallback (full decode + cache)
    """
    if frame_index is None and timestamp is None:
        raise HTTPException(
            status_code=400,
            detail="Either frame_index or timestamp must be provided",
        )

    cache = get_encoded_frame_cache()

    # Determine effective dataset_id
    is_streaming, repo_id, file_path = is_streaming_episode(episode_id, dataset_id)
    effective_dataset_id = dataset_id or (repo_id.replace("/", "_") if repo_id else "local")

    # Build cache key matching what /frames and /frames/stream use
    episode_key = cache.get_episode_cache_key(
        effective_dataset_id, episode_id, resolution, quality
    )
    cached_episode = cache.get_episode_frames(episode_key, effective_dataset_id, episode_id)

    target_idx = frame_index

    if cached_episode:
        all_frames = cached_episode["frames"]
        total_frames = cached_episode["total"]

        # Resolve timestamp to frame index if needed
        if target_idx is None and timestamp is not None and len(all_frames) > 0:
            # Find closest frame by timestamp
            best_idx = 0
            best_diff = float("inf")
            for i, f in enumerate(all_frames):
                ts = f.get("timestamp")
                if ts is not None:
                    diff = abs(ts - timestamp)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = i
            target_idx = best_idx

        if target_idx is not None and target_idx < len(all_frames):
            f = all_frames[target_idx]
            return {
                "frame_idx": f["frame_idx"],
                "timestamp": f.get("timestamp"),
                "image_base64": f.get("image_base64"),
                "total_frames": total_frames,
                "from_cache": True,
            }

    # Not cached — fall back to the /frames endpoint for a single frame
    # This will trigger a full decode + cache, but returns the frame we need
    if target_idx is None:
        # Estimate frame_index from timestamp: default 30fps
        target_idx = round(timestamp * 30) if timestamp is not None else 0

    # Delegate to the existing frames endpoint
    frames_response = await get_frames(
        episode_id=episode_id,
        request=request,
        start=target_idx,
        end=target_idx + 1,
        include_actions=False,
        dataset_id=dataset_id,
        resolution=resolution,
        quality=quality,
        stream="rgb",
    )

    if frames_response.frames:
        f = frames_response.frames[0]
        return {
            "frame_idx": f.frame_idx,
            "timestamp": f.timestamp,
            "image_base64": f.image_base64,
            "total_frames": frames_response.total_frames,
            "from_cache": False,
        }

    return {
        "frame_idx": target_idx,
        "timestamp": None,
        "image_base64": None,
        "total_frames": frames_response.total_frames,
        "from_cache": False,
    }


# NOTE: /frames route must come BEFORE the catch-all /{episode_id:path} route
# to prevent the path converter from matching "frames" as part of the episode ID
@router.get("/{episode_id:path}/frames", response_model=FramesResponse)
async def get_frames(
    episode_id: str,
    request: Request,
    start: int = Query(0, ge=0),
    end: int = Query(10, ge=1),
    include_actions: bool = True,
    dataset_id: Optional[str] = Query(None, description="Dataset ID to load episode from"),
    resolution: str = Query("low", description="Image resolution: low (320x240), medium (640x480), high (960x720), original"),
    quality: int = Query(70, ge=10, le=100, description="WebP quality (10-100)"),
    stream: str = Query("rgb", description="Stream to extract: rgb or depth"),
):
    """
    Get frames from an episode.

    Args:
        episode_id: Episode identifier
        start: Start frame index (inclusive)
        end: End frame index (exclusive)
        include_actions: Include action data with frames
        dataset_id: Optional dataset ID (used to directly select the loader)
        resolution: Image resolution for streaming optimization
        quality: WebP encoding quality (lower = faster, smaller)
        stream: Which modality stream to extract (rgb or depth)
    """
    # Check if this is a streaming episode
    is_streaming, repo_id, file_path = is_streaming_episode(episode_id, dataset_id)

    # Handle LeRobot episode_N format via video chunk extraction
    if is_streaming and _is_lerobot_episode_id(episode_id):
        ep_path_prefix, base_episode_id = _parse_lerobot_episode_id(episode_id)
        resolution_info = await resolve_lerobot_episode(repo_id, base_episode_id, path_prefix=ep_path_prefix)
        if resolution_info:
            cache = get_encoded_frame_cache()
            effective_dataset_id = dataset_id or repo_id.replace("/", "_")
            cache_key_suffix = f":{stream}" if stream != "rgb" else ""
            episode_key = cache.get_episode_cache_key(
                effective_dataset_id, episode_id + cache_key_suffix, resolution, quality
            )
            cached_episode = cache.get_episode_frames(episode_key, effective_dataset_id, episode_id)
            if cached_episode:
                all_frames = cached_episode["frames"]
                total_frames = cached_episode["total"]
                actual_end = min(end, len(all_frames))
                frames = [
                    FrameData(
                        frame_idx=f["frame_idx"],
                        timestamp=f.get("timestamp"),
                        image_base64=f.get("image_base64"),
                        action=f.get("action"),
                    )
                    for f in all_frames[start:actual_end]
                ]
                return FramesResponse(frames=frames, total_frames=total_frames, from_cache=True)

            # Not cached - extract from video
            video_path = resolution_info["video_path"]
            fps = resolution_info["fps"]
            data_branch = resolution_info.get("data_branch")

            if resolution_info.get("single_episode_video"):
                # v2.1: each episode is a separate video file, use streaming extraction
                return await get_streaming_frames(
                    repo_id, video_path, start, end, resolution, quality,
                    dataset_id, stream, revision=data_branch,
                )

            # v3: extract from video chunk using frame offsets
            frame_start = resolution_info["frame_start"]
            frame_end = resolution_info["frame_end"]
            num_frames = resolution_info["num_frames"]

            extractor = StreamingFrameExtractor(repo_id)
            local_path = extractor.download_file(video_path, revision=data_branch)

            actual_start = frame_start + start
            actual_end_vid = min(frame_start + end, frame_end)
            raw_frames = extractor.extract_frames_from_video(local_path, actual_start, actual_end_vid)

            frames = []
            for abs_idx, _, image in raw_frames:
                ep_idx = abs_idx - frame_start
                image_base64 = encode_image_with_options(image, resolution, quality)
                frames.append(FrameData(
                    frame_idx=ep_idx,
                    timestamp=ep_idx / fps,
                    image_base64=image_base64,
                ))
            return FramesResponse(frames=frames, total_frames=num_frames, from_cache=False)

    if is_streaming:
        logger.info(f"Loading streaming episode from {repo_id}: {file_path} (res={resolution}, q={quality}, stream={stream})")
        try:
            return await get_streaming_frames(repo_id, file_path, start, end, resolution, quality, dataset_id, stream)
        except Exception as e:
            logger.error(f"Streaming frame extraction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract frames from streaming episode: {str(e)}"
            )

    # Fall back to local loader
    data_root = get_data_root(request)
    loader, resolved_id = get_loader_for_episode(episode_id, data_root, dataset_id)

    if loader is None:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")

    # Determine effective dataset_id for caching
    effective_dataset_id = dataset_id
    if not effective_dataset_id:
        # Try to extract from episode_id
        all_datasets_local = get_all_datasets()
        parts = episode_id.split("/", 1)
        if parts[0] in all_datasets_local:
            effective_dataset_id = parts[0]
        else:
            effective_dataset_id = "local"

    # Use full-episode cache key (no start/end) to survive browser refresh
    cache = get_encoded_frame_cache()
    episode_key = cache.get_episode_cache_key(effective_dataset_id, episode_id, resolution, quality)

    # Check if full episode is cached
    cached_episode = cache.get_episode_frames(episode_key, effective_dataset_id, episode_id)
    if cached_episode:
        # Return requested slice from cached full episode
        all_frames = cached_episode["frames"]
        total_frames = cached_episode["total"]

        # Clamp range to available frames
        actual_end = min(end, len(all_frames))
        slice_frames = all_frames[start:actual_end]

        frames = [
            FrameData(
                frame_idx=f["frame_idx"],
                timestamp=f.get("timestamp"),
                image_base64=f.get("image_base64"),
                action=f.get("action"),
            )
            for f in slice_frames
        ]
        logger.info(f"Serving frames {start}-{actual_end} from full episode cache ({total_frames} total)")
        return FramesResponse(frames=frames, total_frames=total_frames, from_cache=True)

    # Not cached: load episode and cache ALL frames
    try:
        episode = loader.load_episode(resolved_id)
    except Exception as e:
        logger.error(f"Failed to load episode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load episode: {str(e)}")

    max_frames = episode.num_frames
    if start >= max_frames:
        return FramesResponse(frames=[], total_frames=max_frames, from_cache=False)

    # Encode ALL frames for the episode (not just requested range)
    logger.info(f"Encoding full episode for caching: {episode_id} ({max_frames} frames)")
    all_frames_for_cache = []
    for idx in range(max_frames):
        cache_frame = {"frame_idx": idx}

        # Get timestamp
        if episode.timestamps is not None and idx < len(episode.timestamps):
            cache_frame["timestamp"] = float(episode.timestamps[idx])
        else:
            # Estimate from frame rate
            fps = episode.metadata.get("fps", 30)
            cache_frame["timestamp"] = idx / fps

        # Get image
        if episode.observations is not None and idx < len(episode.observations):
            image = episode.observations[idx]
            cache_frame["image_base64"] = encode_image_with_options(image, resolution, quality)

        # Get action
        if include_actions and episode.actions is not None and idx < len(episode.actions):
            cache_frame["action"] = episode.actions[idx].tolist()

        all_frames_for_cache.append(cache_frame)

    # Store FULL episode in cache
    cache.store_episode_frames(
        episode_key,
        all_frames_for_cache,
        max_frames,
        {
            "dataset_id": effective_dataset_id,
            "episode_id": episode_id,
            "resolution": resolution,
            "quality": quality,
        },
    )

    # Return only the requested slice
    actual_end = min(end, max_frames)
    slice_frames = all_frames_for_cache[start:actual_end]

    frames = [
        FrameData(
            frame_idx=f["frame_idx"],
            timestamp=f.get("timestamp"),
            image_base64=f.get("image_base64"),
            action=f.get("action"),
        )
        for f in slice_frames
    ]

    return FramesResponse(frames=frames, total_frames=max_frames, from_cache=False)


# === SSE STREAMING ENDPOINT ===


async def stream_frames_generator(
    repo_id: str,
    file_path: str,
    start: int,
    end: int,
    resolution: str,
    quality: int,
    stream: str,
    stride: int = 1,
    is_disconnected: callable = None,
    dataset_id: str = None,
    revision: str = None,
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events for progressive frame streaming.

    Yields frames one-by-one as they are decoded, allowing the client
    to display frames before the entire episode is decoded.

    Also caches frames after streaming completes so they appear in "Cached Episodes".

    Args:
        stride: Extract every Nth frame. If 1 and total_frames > 500, auto-computed.
        is_disconnected: Optional async callable to check if client disconnected
        dataset_id: Dataset ID for caching
        revision: Branch/tag to download from (e.g., "v2.0")
    """
    import concurrent.futures

    extractor = StreamingFrameExtractor(repo_id)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # Collect frames for caching
    frames_for_cache = []
    total_frames_count = 0
    cache = get_encoded_frame_cache()
    effective_dataset_id = dataset_id or repo_id.replace("/", "_")
    loop = asyncio.get_event_loop()

    async def check_disconnected():
        """Check if client has disconnected."""
        if is_disconnected:
            try:
                return await is_disconnected()
            except Exception:
                return False
        return False

    async def run_in_thread(func, *args, **kwargs):
        """Run blocking function in thread pool with disconnection checking."""
        future = loop.run_in_executor(executor, lambda: func(*args, **kwargs))

        # Poll for completion while checking for disconnection
        while not future.done():
            if await check_disconnected():
                future.cancel()
                logger.info(f"Cancelled blocking operation due to client disconnect: {file_path}")
                return None
            try:
                # Wait a short time for the future
                return await asyncio.wait_for(asyncio.shield(future), timeout=0.5)
            except asyncio.TimeoutError:
                # Future not done yet, loop and check disconnection again
                continue
        return future.result()

    try:
        # Check for disconnection before heavy work
        if await check_disconnected():
            logger.info(f"Client disconnected before download: {file_path}")
            return

        yield f"data: {json.dumps({'type': 'status', 'status': 'downloading'})}\n\n"

        # Download the file in thread pool (allows disconnection checking)
        local_path = await run_in_thread(extractor.download_file, file_path, revision=revision)
        if local_path is None:
            return  # Client disconnected during download
        suffix = local_path.suffix.lower()

        if suffix != '.mcap':
            # Check for disconnection
            if await check_disconnected():
                logger.info(f"Client disconnected before decoding: {file_path}")
                return

            yield f"data: {json.dumps({'type': 'status', 'status': 'extracting'})}\n\n"

            # For non-MCAP files, fall back to batch decoding (in thread pool)
            # When stride==1, let the decoder skip frames via auto_stride_target
            result = await run_in_thread(
                extractor.extract_frames_with_count,
                file_path, start, end, stream=stream,
                auto_stride_target=150 if stride == 1 else None,
            )
            if result is None:
                return  # Client disconnected during extraction
            raw_frames, total_frames, stride_used = result

            # Use decoder-computed stride when auto_stride was applied
            if stride_used > 1 and stride == 1:
                stride = stride_used
                logger.info(f"Auto-stride: {total_frames} frames -> stride={stride}")

            total_frames_count = total_frames
            yield f"data: {json.dumps({'type': 'total', 'total_frames': total_frames, 'stride': stride})}\n\n"

            # Apply post-hoc stride filtering only for explicit caller stride
            # (auto_stride is already applied during decode)
            if stride > 1 and stride_used == 1:
                raw_frames = [(idx, ts, img) for idx, ts, img in raw_frames if idx % stride == 0]

            for frame_idx, timestamp, image in raw_frames:
                # Check for disconnection periodically (every 10 frames)
                if frame_idx % 10 == 0 and await check_disconnected():
                    logger.info(f"Client disconnected during streaming: {file_path} at frame {frame_idx}")
                    return

                image_base64 = encode_image_with_options(image, resolution, quality)
                frame_data = {
                    'type': 'frame',
                    'index': frame_idx,
                    'timestamp': timestamp,
                    'data': image_base64,
                }
                # Collect for caching
                frames_for_cache.append({
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "image_base64": image_base64,
                    "action": None,
                })
                yield f"data: {json.dumps(frame_data)}\n\n"
                # Allow other tasks to run
                await asyncio.sleep(0)

            # NOTE: No partial caching here - full episode caching is handled
            # by the background caching endpoint POST /episodes/{id}/cache

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Check for disconnection before MCAP processing
        if await check_disconnected():
            logger.info(f"Client disconnected before MCAP processing: {file_path}")
            return

        yield f"data: {json.dumps({'type': 'status', 'status': 'extracting'})}\n\n"

        # For MCAP files, we can stream frames as they're decoded
        # First, get total frame count (in thread pool)
        total_frames = await run_in_thread(extractor.get_frame_count, file_path)
        if total_frames is None:
            return  # Client disconnected
        total_frames_count = total_frames

        # Auto-compute stride for large MCAP episodes
        target = _compute_stride_target(total_frames)
        if stride == 1 and total_frames > target:
            stride = math.ceil(total_frames / target)
            logger.info(f"Auto-stride (MCAP): {total_frames} frames -> stride={stride} (target={target})")

        yield f"data: {json.dumps({'type': 'total', 'total_frames': total_frames, 'stride': stride})}\n\n"

        # Extract frames progressively (in thread pool)
        # For H.264, we still need to decode from the beginning, but we can yield as we go
        # Pass stride to MCAP extractor — JPEG/PNG frames skip decode, H.264 filters post-decode
        raw_frames = await run_in_thread(
            extractor.extract_frames_from_mcap,
            local_path, start, end,
            episode_path=file_path,
            stream=stream,
            stride=stride,
        )
        if raw_frames is None:
            return  # Client disconnected during extraction

        # Stride already applied during extraction — no post-filtering needed

        for frame_idx, timestamp, image in raw_frames:
            # Check for disconnection periodically (every 10 frames)
            if frame_idx % 10 == 0 and await check_disconnected():
                logger.info(f"Client disconnected during MCAP streaming: {file_path} at frame {frame_idx}")
                return

            image_base64 = encode_image_with_options(image, resolution, quality)
            frame_data = {
                'type': 'frame',
                'index': frame_idx,
                'timestamp': timestamp,
                'data': image_base64,
            }
            # Collect for caching
            frames_for_cache.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "image_base64": image_base64,
                "action": None,
            })
            yield f"data: {json.dumps(frame_data)}\n\n"
            # Allow other tasks to run
            await asyncio.sleep(0)

        # NOTE: No partial caching here - full episode caching is handled
        # by the background caching endpoint POST /episodes/{id}/cache

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except GeneratorExit:
        # Client closed connection
        logger.info(f"Client closed connection (GeneratorExit): {file_path}")
    except Exception as e:
        logger.error(f"SSE streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    finally:
        # Clean up executor
        executor.shutdown(wait=False, cancel_futures=True)


async def serve_cached_frames_as_sse(
    cached_episode: dict,
    start: int,
    end: int,
    file_path: str,
) -> AsyncGenerator[str, None]:
    """
    Serve cached frames as SSE events (much faster than re-decoding).

    Stride is inferred from the mismatch between total_frames and cached frame count.
    Cached frames are already strided — no further filtering needed.
    """
    all_frames = cached_episode["frames"]
    total_frames = cached_episode["total"]

    # Infer stride from cache: if total > cached count, stride was applied during caching
    stride = 1
    if total_frames and len(all_frames) > 0 and total_frames > len(all_frames):
        stride = math.ceil(total_frames / len(all_frames))

    yield f"data: {json.dumps({'type': 'total', 'total_frames': total_frames, 'stride': stride})}\n\n"

    # Clamp range to available frames
    actual_end = min(end, len(all_frames))
    for f in all_frames[start:actual_end]:
        frame_data = {
            'type': 'frame',
            'index': f["frame_idx"],
            'timestamp': f.get("timestamp"),
            'data': f.get("image_base64"),
        }
        yield f"data: {json.dumps(frame_data)}\n\n"
        # Allow other tasks to run
        await asyncio.sleep(0)

    logger.info(f"Served {actual_end - start} frames from cache via SSE: {file_path}")
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def stream_lerobot_frames_generator(
    repo_id: str,
    episode_id: str,
    resolution_info: Dict[str, Any],
    start: int,
    end: int,
    resolution: str,
    quality: int,
    stride: int = 1,
    is_disconnected: callable = None,
    dataset_id: str = None,
) -> AsyncGenerator[str, None]:
    """
    SSE generator for LeRobot episodes.

    Downloads the correct video chunk and extracts only the frames
    belonging to the specific episode.

    Args:
        stride: Extract every Nth frame. If 1 and num_frames > 500, auto-computed.
    """
    import concurrent.futures

    video_path = resolution_info["video_path"]
    frame_start = resolution_info["frame_start"]
    frame_end = resolution_info["frame_end"]
    num_frames = resolution_info["num_frames"]
    fps = resolution_info["fps"]

    extractor = StreamingFrameExtractor(repo_id)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()

    frames_for_cache = []
    cache = get_encoded_frame_cache()
    effective_dataset_id = dataset_id or repo_id.replace("/", "_")

    async def check_disconnected():
        if is_disconnected:
            try:
                return await is_disconnected()
            except Exception:
                return False
        return False

    try:
        # Auto-compute stride for large episodes
        target = _compute_stride_target(num_frames) if num_frames else 150
        if stride == 1 and num_frames and num_frames > target:
            stride = math.ceil(num_frames / target)
            logger.info(f"Auto-stride (LeRobot): {num_frames} frames -> stride={stride} (target={target})")

        # Send total frames first (with stride info)
        yield f"data: {json.dumps({'type': 'total', 'total_frames': num_frames, 'stride': stride})}\n\n"

        if await check_disconnected():
            return

        yield f"data: {json.dumps({'type': 'status', 'status': 'downloading'})}\n\n"

        # Download the video chunk file (use data_branch for datasets on non-main branches)
        data_branch = resolution_info.get("data_branch")
        local_path = await loop.run_in_executor(
            executor, lambda: extractor.download_file(video_path, revision=data_branch)
        )
        if local_path is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to download video'})}\n\n"
            return

        if await check_disconnected():
            return

        yield f"data: {json.dumps({'type': 'status', 'status': 'extracting'})}\n\n"

        # Extract only this episode's frames from the video chunk
        # frame_start/frame_end are absolute positions within the video file
        actual_start = frame_start + start
        actual_end = min(frame_start + end, frame_end)

        raw_frames = await loop.run_in_executor(
            executor,
            extractor.extract_frames_from_video,
            local_path,
            actual_start,
            actual_end,
            stride,  # Pass stride to decoder — skips non-stride frames during decode
        )

        if raw_frames is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to extract frames'})}\n\n"
            return

        # Stride already applied during decode — no post-filtering needed

        for abs_frame_idx, timestamp, image in raw_frames:
            # Re-index to episode-relative (0-based within the episode)
            ep_frame_idx = abs_frame_idx - frame_start

            if ep_frame_idx % 10 == 0 and await check_disconnected():
                logger.info(f"Client disconnected during LeRobot streaming: {episode_id} at frame {ep_frame_idx}")
                return

            image_base64 = encode_image_with_options(image, resolution, quality)
            ep_timestamp = ep_frame_idx / fps

            frame_data = {
                'type': 'frame',
                'index': ep_frame_idx,
                'timestamp': ep_timestamp,
                'data': image_base64,
            }
            frames_for_cache.append({
                "frame_idx": ep_frame_idx,
                "timestamp": ep_timestamp,
                "image_base64": image_base64,
                "action": None,
            })
            yield f"data: {json.dumps(frame_data)}\n\n"
            await asyncio.sleep(0)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except GeneratorExit:
        logger.info(f"Client closed connection (GeneratorExit): {episode_id}")
    except Exception as e:
        logger.error(f"LeRobot SSE streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


@router.get("/{episode_id:path}/frames/stream")
async def stream_frames(
    episode_id: str,
    request: Request,
    dataset_id: Optional[str] = Query(None, description="Dataset ID"),
    start: int = Query(0, ge=0),
    end: int = Query(100, ge=1),
    resolution: str = Query("low", description="Image resolution"),
    quality: int = Query(70, ge=10, le=100, description="WebP quality"),
    stream: str = Query("rgb", description="Stream: rgb or depth"),
    stride: int = Query(1, ge=1, description="Extract every Nth frame (for large episodes)"),
):
    """
    Stream frames using Server-Sent Events (SSE).

    This endpoint returns frames progressively as they're decoded,
    allowing the client to display frames before the entire episode
    is ready. Much better user experience for large episodes.

    If frames are already cached, serves them immediately from cache.

    SSE Event Types:
    - total: Contains total_frames count
    - frame: Contains frame index, timestamp, and base64-encoded image
    - done: Signals completion
    - error: Contains error message

    Args:
        episode_id: Episode identifier
        request: FastAPI Request object for disconnection detection
        dataset_id: Optional dataset ID
        start: Start frame index
        end: End frame index
        resolution: Image resolution (low/medium/high/original)
        quality: WebP quality (10-100)
        stream: Which stream to extract (rgb or depth)
        stride: Extract every Nth frame (auto-computed for large episodes if set to 1)
    """
    # Check if this is a streaming episode
    is_streaming, repo_id, file_path = is_streaming_episode(episode_id, dataset_id)

    if not is_streaming:
        raise HTTPException(
            status_code=400,
            detail="SSE streaming only available for HuggingFace streaming datasets"
        )

    # Handle LeRobot episode_N format (including multi-subdataset prefix/episode_N)
    if _is_lerobot_episode_id(episode_id):
        ep_path_prefix, base_episode_id = _parse_lerobot_episode_id(episode_id)
        resolution_info = await resolve_lerobot_episode(repo_id, base_episode_id, path_prefix=ep_path_prefix)
        if resolution_info:
            data_branch = resolution_info.get("data_branch")

            # v2.1 single-episode videos: use generic streaming with resolved video path
            if resolution_info.get("single_episode_video"):
                video_path = resolution_info["video_path"]
                # Check cache using episode_id as key
                cache = get_encoded_frame_cache()
                effective_dataset_id = dataset_id or repo_id.replace("/", "_")
                cache_key_suffix = f":{stream}" if stream != "rgb" else ""
                episode_key = cache.get_episode_cache_key(
                    effective_dataset_id, episode_id + cache_key_suffix, resolution, quality
                )
                cached_episode = cache.get_episode_frames(episode_key, effective_dataset_id, episode_id)
                if cached_episode:
                    cached_frame_count = len(cached_episode['frames'])
                    has_enough_frames = cached_frame_count >= end or cached_frame_count >= cached_episode.get('total', 0)
                    if has_enough_frames and start < cached_frame_count:
                        return StreamingResponse(
                            serve_cached_frames_as_sse(cached_episode, start, end, episode_id),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
                        )

                async def is_disconnected_v21():
                    return await request.is_disconnected()

                return StreamingResponse(
                    stream_frames_generator(
                        repo_id, video_path, start, end, resolution, quality, stream,
                        stride=stride,
                        is_disconnected=is_disconnected_v21,
                        dataset_id=dataset_id,
                        revision=data_branch,
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
                )

            # v3: use episode metadata for chunk/frame-level extraction
            # Check cache first (keyed by episode_id, not video chunk path)
            cache = get_encoded_frame_cache()
            effective_dataset_id = dataset_id or repo_id.replace("/", "_")
            cache_key_suffix = f":{stream}" if stream != "rgb" else ""
            episode_key = cache.get_episode_cache_key(
                effective_dataset_id, episode_id + cache_key_suffix, resolution, quality
            )

            cached_episode = cache.get_episode_frames(episode_key, effective_dataset_id, episode_id)
            if cached_episode:
                cached_frame_count = len(cached_episode['frames'])
                has_enough_frames = cached_frame_count >= end or cached_frame_count >= cached_episode.get('total', 0)
                if has_enough_frames and start < cached_frame_count:
                    logger.info(f"SSE serving LeRobot from cache: {episode_id} ({cached_frame_count} frames)")
                    return StreamingResponse(
                        serve_cached_frames_as_sse(cached_episode, start, end, episode_id),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                        }
                    )

            # Not cached - stream from video chunk
            async def is_disconnected():
                return await request.is_disconnected()

            return StreamingResponse(
                stream_lerobot_frames_generator(
                    repo_id, episode_id, resolution_info,
                    start, end, resolution, quality,
                    stride=stride,
                    is_disconnected=is_disconnected,
                    dataset_id=dataset_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

    # Check cache FIRST - serve from cache if available (much faster!)
    cache = get_encoded_frame_cache()
    effective_dataset_id = dataset_id or repo_id.replace("/", "_")
    cache_key_suffix = f":{stream}" if stream != "rgb" else ""
    episode_key = cache.get_episode_cache_key(
        effective_dataset_id, file_path + cache_key_suffix, resolution, quality
    )

    cached_episode = cache.get_episode_frames(episode_key, effective_dataset_id, file_path)
    if cached_episode:
        cached_frame_count = len(cached_episode['frames'])
        # Only serve from cache if we have enough frames for the requested range
        # Cache must have frames up to at least 'end' (or all frames if end > total)
        has_enough_frames = cached_frame_count >= end or cached_frame_count >= cached_episode.get('total', 0)

        if has_enough_frames and start < cached_frame_count:
            logger.info(f"SSE serving from cache: {file_path} ({cached_frame_count} frames, requested {start}-{end})")
            return StreamingResponse(
                serve_cached_frames_as_sse(cached_episode, start, end, file_path),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        else:
            logger.info(f"Cache incomplete for {file_path}: have {cached_frame_count} frames, need up to {end}")

    # Not cached - stream and cache as we go
    # Create disconnection checker from request
    async def is_disconnected():
        return await request.is_disconnected()

    return StreamingResponse(
        stream_frames_generator(
            repo_id, file_path, start, end, resolution, quality, stream,
            stride=stride,
            is_disconnected=is_disconnected,
            dataset_id=dataset_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


class IMUData(BaseModel):
    """IMU sensor data from an episode."""
    timestamps: List[float] = []
    accel_x: List[float] = []
    accel_y: List[float] = []
    accel_z: List[float] = []
    gyro_x: List[float] = []
    gyro_y: List[float] = []
    gyro_z: List[float] = []
    error: Optional[str] = None


class ActionsData(BaseModel):
    """Action data from an episode."""
    timestamps: List[float] = []
    actions: List[List[float]] = []  # 2D array: [frame][dimension]
    dimension_labels: Optional[List[str]] = None
    error: Optional[str] = None


@router.get("/{episode_id:path}/imu", response_model=IMUData)
async def get_imu_data(
    episode_id: str,
    dataset_id: Optional[str] = Query(None, description="Dataset ID to load episode from"),
):
    """
    Get IMU data from an episode.

    Only available for MCAP episodes that contain IMU topics.

    Args:
        episode_id: Episode identifier
        dataset_id: Optional dataset ID (used to directly select the loader)

    Returns:
        IMU sensor data including accelerometer and gyroscope readings
    """
    # Check if this is a streaming episode
    is_streaming, repo_id, file_path = is_streaming_episode(episode_id, dataset_id)

    if not is_streaming:
        return IMUData(error="IMU data only available for streaming MCAP episodes")

    # Only MCAP files have IMU data
    if not file_path.endswith(".mcap"):
        return IMUData(error="IMU data only available for MCAP files")

    extractor = StreamingFrameExtractor(repo_id)

    try:
        imu_data = extractor.extract_imu_data(file_path)

        if "error" in imu_data:
            return IMUData(error=imu_data["error"])

        return IMUData(
            timestamps=imu_data.get("timestamps", []),
            accel_x=imu_data.get("accel_x", []),
            accel_y=imu_data.get("accel_y", []),
            accel_z=imu_data.get("accel_z", []),
            gyro_x=imu_data.get("gyro_x", []),
            gyro_y=imu_data.get("gyro_y", []),
            gyro_z=imu_data.get("gyro_z", []),
        )
    except Exception as e:
        logger.error(f"Failed to extract IMU data: {e}")
        return IMUData(error=str(e))


async def _get_lerobot_actions(
    repo_id: str,
    episode_id: str,
    dataset_id: Optional[str] = None,
    path_prefix: str = "",
) -> ActionsData:
    """
    Extract action data from a streaming LeRobot dataset's parquet files.

    Uses the same parquet download infrastructure as signal analysis.
    When path_prefix is set, scopes metadata and data paths under the subdataset prefix.
    """
    import re
    import httpx

    from .datasets import fetch_lerobot_info, detect_lerobot_data_branch
    from .analysis import _build_lerobot_data_file_list, _download_parquet, _get_hf_token as get_hf_token

    match = re.match(r'^episode_(\d+)$', episode_id)
    if not match:
        return ActionsData(error=f"Invalid LeRobot episode ID: {episode_id}")
    target_ep_idx = int(match.group(1))

    info = await fetch_lerobot_info(repo_id, path_prefix=path_prefix)
    if info is None:
        return ActionsData(error=f"Could not fetch info.json for {repo_id}")

    data_branch = await detect_lerobot_data_branch(repo_id, path_prefix=path_prefix) or "main"
    data_files = _build_lerobot_data_file_list(info, [target_ep_idx], path_prefix=path_prefix)
    if not data_files:
        return ActionsData(error="Could not determine parquet file path for episode")

    token = None
    try:
        token = get_hf_token()
    except Exception:
        pass
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            df = await _download_parquet(client, repo_id, data_files[0]["path"], headers, branch=data_branch)
    except Exception as e:
        logger.error(f"Failed to download parquet for LeRobot actions: {e}")
        return ActionsData(error=f"Failed to download episode data: {e}")

    if df.empty:
        return ActionsData(error="No data in parquet file")

    # Filter to target episode
    ep_col = "episode_index" if "episode_index" in df.columns else "episode_id"
    if ep_col not in df.columns:
        return ActionsData(error=f"No episode column found in parquet data")
    episode_df = df[df[ep_col] == target_ep_idx]
    if episode_df.empty:
        return ActionsData(error=f"Episode {target_ep_idx} not found in parquet data")

    sort_col = "frame_index" if "frame_index" in episode_df.columns else "index"
    if sort_col in episode_df.columns:
        episode_df = episode_df.sort_values(sort_col)

    # Auto-stride: subsample rows for large episodes (consistent with video streaming)
    if len(episode_df) > 500:
        signal_stride = math.ceil(len(episode_df) / 500)
        episode_df = episode_df.iloc[::signal_stride].reset_index(drop=True)

    # Extract timestamps
    fps = info.get("fps", 30)
    if "timestamp" in episode_df.columns:
        timestamps = episode_df["timestamp"].tolist()
    elif "frame_index" in episode_df.columns:
        timestamps = (episode_df["frame_index"] / fps).tolist()
    else:
        timestamps = [i / fps for i in range(len(episode_df))]

    # Extract actions
    action_col = None
    for candidate in ["action", "observation.state", "end_pose", "start_pos"]:
        if candidate in episode_df.columns:
            action_col = candidate
            break

    if action_col is None:
        return ActionsData(error="No action or state column in parquet data")

    action_list = episode_df[action_col].tolist()
    actions = []
    for a in action_list:
        if isinstance(a, np.ndarray):
            actions.append(a.tolist())
        elif isinstance(a, (list, tuple)):
            actions.append(list(a))
        else:
            actions.append([float(a)])

    # Infer dimension labels
    dimension_labels = None
    if actions:
        num_dims = len(actions[0])
        if num_dims == 7:
            dimension_labels = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
        elif num_dims == 8:
            dimension_labels = ["x", "y", "z", "rx", "ry", "rz", "state", "gripper"]
        elif num_dims == 6:
            dimension_labels = ["x", "y", "z", "rx", "ry", "rz"]
        elif num_dims == 3:
            dimension_labels = ["x", "y", "z"]

    return ActionsData(
        timestamps=timestamps,
        actions=actions,
        dimension_labels=dimension_labels,
    )


@router.get("/{episode_id:path}/actions", response_model=ActionsData)
async def get_actions_data(
    episode_id: str,
    request: Request,
    dataset_id: Optional[str] = Query(None, description="Dataset ID to load episode from"),
):
    """
    Get action data from an episode.

    For streaming MCAP episodes, extracts action data from action/command topics.
    For HDF5 episodes (LIBERO), reads from the actions array.

    Args:
        episode_id: Episode identifier
        dataset_id: Optional dataset ID (used to directly select the loader)

    Returns:
        Action data including timestamps and action vectors
    """
    # Check if this is a streaming episode
    is_streaming, repo_id, file_path = is_streaming_episode(episode_id, dataset_id)

    if is_streaming:
        # Handle LeRobot parquet-based episodes
        if _is_lerobot_episode_id(episode_id):
            ep_path_prefix, base_episode_id = _parse_lerobot_episode_id(episode_id)
            return await _get_lerobot_actions(repo_id, base_episode_id, dataset_id, path_prefix=ep_path_prefix)

        # Only MCAP files have action data via topic extraction
        if not file_path.endswith(".mcap"):
            return ActionsData(error="Action data only available for MCAP files")

        extractor = StreamingFrameExtractor(repo_id)

        try:
            actions_result = extractor.extract_actions_data(file_path)

            if "error" in actions_result and actions_result["error"]:
                return ActionsData(error=actions_result["error"])

            return ActionsData(
                timestamps=actions_result.get("timestamps", []),
                actions=actions_result.get("actions", []),
                dimension_labels=actions_result.get("dimension_labels"),
            )
        except Exception as e:
            logger.error(f"Failed to extract actions data: {e}")
            return ActionsData(error=str(e))

    # For local episodes (HDF5), read actions from the episode
    data_root = get_data_root(request)
    loader, resolved_id = get_loader_for_episode(episode_id, data_root, dataset_id)

    if loader is None:
        return ActionsData(error=f"Episode not found: {episode_id}")

    try:
        episode = loader.load_episode(resolved_id)

        if episode.actions is None or len(episode.actions) == 0:
            return ActionsData(error="No action data in episode")

        # Convert actions to list format
        actions_list = episode.actions.tolist()

        # Generate timestamps based on frame count and fps
        fps = episode.metadata.get("fps", 30)
        timestamps = [i / fps for i in range(len(actions_list))]

        # Determine dimension labels based on action dimensions
        num_dims = len(actions_list[0]) if actions_list else 0
        dimension_labels = None
        if num_dims == 7:
            dimension_labels = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
        elif num_dims == 6:
            dimension_labels = ["x", "y", "z", "rx", "ry", "rz"]
        elif num_dims == 3:
            dimension_labels = ["x", "y", "z"]

        return ActionsData(
            timestamps=timestamps,
            actions=actions_list,
            dimension_labels=dimension_labels,
        )
    except Exception as e:
        logger.error(f"Failed to load actions from episode: {e}")
        return ActionsData(error=str(e))


# =============================================================================
# Background Caching Endpoints
# =============================================================================


class CachingStatusResponse(BaseModel):
    """Response for caching status endpoint."""
    status: str  # "not_cached" | "caching" | "cached" | "error" | "not_applicable"
    progress: Optional[int] = None  # 0-100 when caching
    total_frames: Optional[int] = None
    error: Optional[str] = None


@router.post("/{episode_id:path}/cache")
async def start_background_caching(
    episode_id: str,
    dataset_id: Optional[str] = Query(None, description="Dataset ID"),
    resolution: str = Query("low", description="Image resolution"),
    quality: int = Query(70, ge=10, le=100, description="WebP quality"),
    stream: str = Query("rgb", description="Stream: rgb or depth"),
) -> CachingStatusResponse:
    """
    Start caching an entire episode in the background.

    Called when user starts playback. Returns immediately while
    caching proceeds in background. If episode is already fully cached,
    returns immediately with status="cached".

    This endpoint triggers a background task that:
    1. Decodes ALL frames from the episode
    2. Encodes them to WebP format
    3. Stores them in the encoded frames cache
    4. Cleans up decoded frames (pickle files) after completion

    Args:
        episode_id: Episode identifier
        dataset_id: Optional dataset ID
        resolution: Image resolution (low/medium/high/original)
        quality: WebP quality (10-100)
        stream: Which stream to cache (rgb or depth)
    """
    # Check if this is a streaming episode
    is_streaming, repo_id, file_path = is_streaming_episode(episode_id, dataset_id)

    if not is_streaming:
        return CachingStatusResponse(status="not_applicable", error="Only streaming episodes need caching")

    # LeRobot episodes (episode_N format) need special resolution via the SSE endpoint
    # (resolve_lerobot_episode finds the correct video chunk path). The generic
    # StreamingFrameExtractor can't handle them, so skip background caching and let
    # the SSE stream_frames endpoint handle them directly.
    if _is_lerobot_episode_id(episode_id):
        return CachingStatusResponse(status="not_applicable")

    cache = get_encoded_frame_cache()
    effective_dataset_id = dataset_id or repo_id.replace("/", "_")
    cache_key_suffix = f":{stream}" if stream != "rgb" else ""
    episode_key = cache.get_episode_cache_key(
        effective_dataset_id, file_path + cache_key_suffix, resolution, quality
    )

    # Check if already fully cached
    cached = cache.get_episode_frames(episode_key, effective_dataset_id, file_path)
    if cached and cached.get("full_episode"):
        return CachingStatusResponse(
            status="cached",
            total_frames=cached.get("total", len(cached.get("frames", [])))
        )

    # Create a unique task key
    task_key = f"{effective_dataset_id}/{file_path}/{stream}/{resolution}/{quality}"

    # Check if already caching
    if task_key in _caching_tasks:
        task = _caching_tasks[task_key]
        if not task.done():
            progress_info = _caching_progress.get(task_key, {})
            return CachingStatusResponse(
                status="caching",
                progress=progress_info.get("progress", 0),
                total_frames=progress_info.get("total_frames")
            )

    # Define the background caching task
    async def cache_full_episode():
        try:
            _caching_progress[task_key] = {"progress": 0, "status": "caching", "total_frames": 0}
            logger.info(f"Starting background caching for: {file_path}")

            extractor = StreamingFrameExtractor(repo_id)

            # For TAR files, extract video once and reuse across all batches
            # This avoids re-extracting the video from the TAR archive for every batch
            local_path = await asyncio.to_thread(extractor.download_file, file_path)
            tar_video_path = None
            if local_path.suffix.lower() == '.tar':
                tar_video_path = await asyncio.to_thread(
                    extractor.extract_video_from_tar_once, local_path
                )
                logger.info(f"Pre-extracted video from TAR: {tar_video_path}")

            # Get total frame count (uses cached download, no re-download)
            total_frames = await asyncio.to_thread(extractor.get_frame_count, file_path)
            _caching_progress[task_key]["total_frames"] = total_frames
            logger.info(f"Background caching {file_path}: {total_frames} total frames")

            # Compute stride to avoid caching more frames than needed
            cache_stride = 1
            target = _compute_stride_target(total_frames)
            if total_frames > target:
                cache_stride = math.ceil(total_frames / target)
                logger.info(f"Background caching with stride={cache_stride} (target={target})")

            # Pipelined batch processing: extract batch N+1 while encoding batch N
            # Larger batch size reduces overhead from task switching
            BATCH_SIZE = 500
            all_frames = []

            def extract_batch_raw(start, end):
                """Extract raw frames from episode (blocking)."""
                if tar_video_path:
                    # Use pre-extracted video path instead of re-extracting from TAR
                    return list(extractor.extract_frames_from_video(
                        tar_video_path, start, end, cache_stride
                    ))
                else:
                    return list(extractor.extract_frames(
                        file_path, start=start, end=end, stream=stream,
                        force_full_extraction=True, stride=cache_stride
                    ))

            def encode_batch(raw_frames):
                """Encode raw frames to base64 WebP in parallel (blocking)."""
                if not raw_frames:
                    return []

                def encode_single(frame_data):
                    frame_idx, timestamp, image = frame_data
                    image_base64 = encode_image_with_options(image, resolution, quality)
                    return {
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                        "image_base64": image_base64,
                        "action": None,
                    }

                futures = [_ENCODE_EXECUTOR.submit(encode_single, f) for f in raw_frames]
                return [future.result() for future in futures]

            # Pipeline: extract batch N+1 while encoding batch N
            prev_encode_task = None

            for batch_start in range(0, total_frames, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_frames)

                # Extract current batch in thread pool
                raw_frames = await asyncio.to_thread(extract_batch_raw, batch_start, batch_end)

                # Wait for previous encode task to finish and collect results
                if prev_encode_task:
                    all_frames.extend(await prev_encode_task)

                # Start encoding this batch in background while we extract the next
                prev_encode_task = asyncio.create_task(
                    asyncio.to_thread(encode_batch, raw_frames)
                )

                # Update progress
                progress = int((batch_start / total_frames) * 100) if total_frames > 0 else 0
                _caching_progress[task_key]["progress"] = min(progress, 99)

                # Yield control to event loop between batches
                await asyncio.sleep(0)

            # Collect final batch
            if prev_encode_task:
                all_frames.extend(await prev_encode_task)

            # Clean up pre-extracted TAR video temp file
            if tar_video_path:
                try:
                    tar_video_path.unlink(missing_ok=True)
                except Exception:
                    pass

            # Store as full episode — use original total_frames (not strided count)
            # so the frontend knows the true episode length for timeline/playback
            cache.store_episode_frames(
                episode_key,
                all_frames,
                total_frames,
                {
                    "dataset_id": effective_dataset_id,
                    "episode_id": file_path,
                    "resolution": resolution,
                    "quality": quality,
                },
            )

            logger.info(f"Background caching complete: {file_path} ({len(all_frames)} frames)")
            _caching_progress[task_key] = {"progress": 100, "status": "complete", "total_frames": len(all_frames)}

            # Cleanup decoded frames after successful encoding
            cleanup_decoded_frames(repo_id, file_path, stream)

        except Exception as e:
            logger.error(f"Background caching failed for {file_path}: {e}")
            _caching_progress[task_key] = {"progress": 0, "status": "error", "error": str(e)}
            # Clean up TAR video temp file on error too
            if tar_video_path:
                try:
                    tar_video_path.unlink(missing_ok=True)
                except Exception:
                    pass

    # Start the background task
    task = asyncio.create_task(cache_full_episode())
    _caching_tasks[task_key] = task

    return CachingStatusResponse(status="started", progress=0)


@router.get("/{episode_id:path}/cache/status")
async def get_caching_status(
    episode_id: str,
    dataset_id: Optional[str] = Query(None, description="Dataset ID"),
    resolution: str = Query("low", description="Image resolution"),
    quality: int = Query(70, ge=10, le=100, description="WebP quality"),
    stream: str = Query("rgb", description="Stream: rgb or depth"),
) -> CachingStatusResponse:
    """
    Get the caching status for an episode.

    Returns the current status of caching:
    - "not_cached": Episode has not been cached
    - "caching": Background caching in progress (includes progress %)
    - "cached": Episode is fully cached
    - "error": Caching failed
    - "not_applicable": Not a streaming episode
    """
    is_streaming, repo_id, file_path = is_streaming_episode(episode_id, dataset_id)

    if not is_streaming:
        return CachingStatusResponse(status="not_applicable")

    # LeRobot episodes use direct SSE streaming, not background caching
    if _is_lerobot_episode_id(episode_id):
        # Still check if frames happen to be cached (from SSE streaming)
        cache = get_encoded_frame_cache()
        effective_dataset_id = dataset_id or repo_id.replace("/", "_")
        cache_key_suffix = f":{stream}" if stream != "rgb" else ""
        episode_key = cache.get_episode_cache_key(
            effective_dataset_id, episode_id + cache_key_suffix, resolution, quality
        )
        cached = cache.get_episode_frames(episode_key, effective_dataset_id, episode_id)
        if cached and cached.get("full_episode"):
            return CachingStatusResponse(
                status="cached",
                progress=100,
                total_frames=cached.get("total", len(cached.get("frames", [])))
            )
        return CachingStatusResponse(status="not_applicable")

    cache = get_encoded_frame_cache()
    effective_dataset_id = dataset_id or repo_id.replace("/", "_")
    cache_key_suffix = f":{stream}" if stream != "rgb" else ""
    episode_key = cache.get_episode_cache_key(
        effective_dataset_id, file_path + cache_key_suffix, resolution, quality
    )

    # Check if fully cached
    cached = cache.get_episode_frames(episode_key, effective_dataset_id, file_path)
    if cached and cached.get("full_episode"):
        return CachingStatusResponse(
            status="cached",
            progress=100,
            total_frames=cached.get("total", len(cached.get("frames", [])))
        )

    # Check if caching in progress
    task_key = f"{effective_dataset_id}/{file_path}/{stream}/{resolution}/{quality}"
    if task_key in _caching_progress:
        progress_info = _caching_progress[task_key]
        status = progress_info.get("status", "caching")

        if status == "complete":
            return CachingStatusResponse(
                status="cached",
                progress=100,
                total_frames=progress_info.get("total_frames")
            )
        elif status == "error":
            return CachingStatusResponse(
                status="error",
                error=progress_info.get("error")
            )
        else:
            return CachingStatusResponse(
                status="caching",
                progress=progress_info.get("progress", 0),
                total_frames=progress_info.get("total_frames")
            )

    return CachingStatusResponse(status="not_cached")


# NOTE: This catch-all route MUST be LAST to avoid matching /imu, /actions, etc.
@router.get("/{episode_id:path}", response_model=EpisodeDetail)
async def get_episode(
    episode_id: str,
    request: Request,
    dataset_id: Optional[str] = Query(None, description="Dataset ID to load episode from"),
):
    """
    Get detailed information about a specific episode.

    Args:
        episode_id: Episode identifier
        dataset_id: Optional dataset ID (used to directly select the loader)
    """
    data_root = get_data_root(request)
    loader, resolved_id = get_loader_for_episode(episode_id, data_root, dataset_id)

    if loader is None:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")

    try:
        episode = loader.load_episode(resolved_id)
    except Exception as e:
        logger.error(f"Failed to load episode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load episode: {str(e)}")

    return EpisodeDetail(
        id=episode.id,
        task_name=episode.task_name,
        description=episode.description,
        num_frames=episode.num_frames,
        duration_seconds=episode.duration_seconds,
        has_observations=episode.observations is not None,
        has_actions=episode.actions is not None,
        has_video=episode.video_path is not None or episode.video_bytes is not None,
        metadata=episode.metadata,
    )
