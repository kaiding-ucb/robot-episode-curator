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
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from downloaders.manager import DATASET_REGISTRY, get_all_datasets
from loaders import HDF5Loader, WebDatasetLoader, LeRobotLoader, RLDSLoader
from loaders.streaming_extractor import StreamingFrameExtractor
from cache import get_encoded_frame_cache

logger = logging.getLogger(__name__)

router = APIRouter()


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

        # Resize if needed
        target_size = RESOLUTION_MAP.get(resolution)
        if target_size and (image.shape[1] > target_size[0] or image.shape[0] > target_size[1]):
            # Use cv2.resize with INTER_AREA for quality downsampling
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Encode to WebP with specified quality (40-50% smaller than JPEG)
        buffer = BytesIO()
        pil_image.save(buffer, format="WEBP", quality=quality)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")
    except ImportError:
        # Fallback without PIL - return raw base64
        return base64.b64encode(image.tobytes()).decode("utf-8")


def encode_image_base64(image: np.ndarray) -> str:
    """Encode numpy image array to base64 JPEG (legacy compatibility)."""
    return encode_image_with_options(image, resolution="original", quality=85)


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
        logger.info(f"Decoding full episode for caching: {file_path} (stream={stream})")
        raw_frames, total_frames = extractor.extract_frames_with_count(
            file_path, 0, 999999, stream=stream
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
    is_disconnected: callable = None,
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events for progressive frame streaming.

    Yields frames one-by-one as they are decoded, allowing the client
    to display frames before the entire episode is decoded.

    Args:
        is_disconnected: Optional async callable to check if client disconnected
    """
    import concurrent.futures

    extractor = StreamingFrameExtractor(repo_id)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
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

        # Download the file in thread pool (allows disconnection checking)
        local_path = await run_in_thread(extractor.download_file, file_path)
        if local_path is None:
            return  # Client disconnected during download
        suffix = local_path.suffix.lower()

        if suffix != '.mcap':
            # Check for disconnection
            if await check_disconnected():
                logger.info(f"Client disconnected before decoding: {file_path}")
                return

            # For non-MCAP files, fall back to batch decoding (in thread pool)
            result = await run_in_thread(
                extractor.extract_frames_with_count,
                file_path, start, end, stream=stream
            )
            if result is None:
                return  # Client disconnected during extraction
            raw_frames, total_frames = result

            yield f"data: {json.dumps({'type': 'total', 'total_frames': total_frames})}\n\n"

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
                yield f"data: {json.dumps(frame_data)}\n\n"
                # Allow other tasks to run
                await asyncio.sleep(0)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Check for disconnection before MCAP processing
        if await check_disconnected():
            logger.info(f"Client disconnected before MCAP processing: {file_path}")
            return

        # For MCAP files, we can stream frames as they're decoded
        # First, get total frame count (in thread pool)
        total_frames = await run_in_thread(extractor.get_frame_count, file_path)
        if total_frames is None:
            return  # Client disconnected
        yield f"data: {json.dumps({'type': 'total', 'total_frames': total_frames})}\n\n"

        # Extract frames progressively (in thread pool)
        # For H.264, we still need to decode from the beginning, but we can yield as we go
        raw_frames = await run_in_thread(
            extractor.extract_frames_from_mcap,
            local_path, start, end,
            episode_path=file_path,
            stream=stream
        )
        if raw_frames is None:
            return  # Client disconnected during extraction

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
            yield f"data: {json.dumps(frame_data)}\n\n"
            # Allow other tasks to run
            await asyncio.sleep(0)

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
):
    """
    Stream frames using Server-Sent Events (SSE).

    This endpoint returns frames progressively as they're decoded,
    allowing the client to display frames before the entire episode
    is ready. Much better user experience for large episodes.

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
    """
    # Check if this is a streaming episode
    is_streaming, repo_id, file_path = is_streaming_episode(episode_id, dataset_id)

    if not is_streaming:
        raise HTTPException(
            status_code=400,
            detail="SSE streaming only available for HuggingFace streaming datasets"
        )

    # Create disconnection checker from request
    async def is_disconnected():
        return await request.is_disconnected()

    return StreamingResponse(
        stream_frames_generator(
            repo_id, file_path, start, end, resolution, quality, stream,
            is_disconnected=is_disconnected
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
