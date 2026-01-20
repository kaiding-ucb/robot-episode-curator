"""
API routes for episode operations.

Endpoints:
- GET /api/episodes/{id} - Get episode metadata
- GET /api/episodes/{id}/frames - Get frames from episode
"""
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel

from downloaders.manager import DATASET_REGISTRY
from loaders import HDF5Loader, WebDatasetLoader, LeRobotLoader, RLDSLoader
from loaders.streaming_extractor import StreamingFrameExtractor

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


def get_data_root(request: Request) -> Path:
    """Get data root from app state."""
    return request.app.state.data_root


def parse_episode_id(episode_id: str) -> tuple:
    """
    Parse episode ID to extract dataset and internal ID.

    Episode IDs are typically: dataset_id/path/to/episode
    """
    parts = episode_id.split("/", 1)
    if len(parts) == 1:
        # Assume it's just the episode ID, try to find dataset
        return None, episode_id

    # First part might be dataset ID
    if parts[0] in DATASET_REGISTRY:
        return parts[0], parts[1]

    # Or it might be a task suite (e.g., libero_spatial)
    for dataset_id, config in DATASET_REGISTRY.items():
        if parts[0].startswith(dataset_id.replace("_", "")):
            return dataset_id, episode_id

    return None, episode_id


def get_loader_for_dataset(dataset_id: str, data_root: Path):
    """Get the appropriate loader for a dataset."""
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
    if parsed_dataset_id and parsed_dataset_id in DATASET_REGISTRY:
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
    for ds_id in DATASET_REGISTRY:
        config = DATASET_REGISTRY[ds_id]

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


def encode_image_base64(image: np.ndarray) -> str:
    """Encode numpy image array to base64 JPEG."""
    try:
        from PIL import Image

        # Ensure correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Encode to JPEG
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")
    except ImportError:
        # Fallback without PIL - return raw base64
        return base64.b64encode(image.tobytes()).decode("utf-8")


def is_streaming_episode(episode_id: str, dataset_id: Optional[str]) -> tuple:
    """
    Check if an episode is from a streaming HuggingFace dataset.

    Returns:
        Tuple of (is_streaming, repo_id, file_path)
    """
    # If dataset_id is provided, check if it's streaming
    if dataset_id and dataset_id in DATASET_REGISTRY:
        config = DATASET_REGISTRY[dataset_id]
        if config.get("streaming_recommended") and config.get("repo_id"):
            # Episode ID format for streaming: task_folder/subfolder/file.mcap
            return True, config["repo_id"], episode_id

    # Check if episode_id matches a streaming dataset pattern
    for ds_id, config in DATASET_REGISTRY.items():
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
) -> FramesResponse:
    """
    Get frames from a streaming HuggingFace episode.
    """
    extractor = StreamingFrameExtractor(repo_id)

    try:
        # Extract frames and get total count
        raw_frames, total_frames = extractor.extract_frames_with_count(file_path, start, end)

        # Convert to FrameData
        frames = []
        for frame_idx, timestamp, image in raw_frames:
            frame_data = FrameData(
                frame_idx=frame_idx,
                timestamp=timestamp,
                image_base64=encode_image_base64(image),
                action=None,  # Streaming datasets typically don't have action data
            )
            frames.append(frame_data)

        return FramesResponse(frames=frames, total_frames=total_frames)

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
):
    """
    Get frames from an episode.

    Args:
        episode_id: Episode identifier
        start: Start frame index (inclusive)
        end: End frame index (exclusive)
        include_actions: Include action data with frames
        dataset_id: Optional dataset ID (used to directly select the loader)
    """
    # Check if this is a streaming episode
    is_streaming, repo_id, file_path = is_streaming_episode(episode_id, dataset_id)

    if is_streaming:
        logger.info(f"Loading streaming episode from {repo_id}: {file_path}")
        try:
            return await get_streaming_frames(repo_id, file_path, start, end)
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

    try:
        episode = loader.load_episode(resolved_id)
    except Exception as e:
        logger.error(f"Failed to load episode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load episode: {str(e)}")

    # Validate range
    max_frames = episode.num_frames
    if start >= max_frames:
        return FramesResponse(frames=[], total_frames=max_frames)

    end = min(end, max_frames)

    frames = []
    for idx in range(start, end):
        frame_data = FrameData(frame_idx=idx)

        # Get timestamp
        if episode.timestamps is not None and idx < len(episode.timestamps):
            frame_data.timestamp = float(episode.timestamps[idx])
        else:
            # Estimate from frame rate
            fps = episode.metadata.get("fps", 30)
            frame_data.timestamp = idx / fps

        # Get image
        if episode.observations is not None and idx < len(episode.observations):
            image = episode.observations[idx]
            frame_data.image_base64 = encode_image_base64(image)

        # Get action
        if include_actions and episode.actions is not None and idx < len(episode.actions):
            frame_data.action = episode.actions[idx].tolist()

        frames.append(frame_data)

    return FramesResponse(frames=frames, total_frames=max_frames)


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
