"""
Rerun API routes for generating .rrd recordings from episode data.

Converts episode data to Rerun's native format for
advanced multi-modal visualization.
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rerun", tags=["rerun"])

# Cache directory for generated RRD files
RRD_CACHE_DIR = Path(os.environ.get("RRD_CACHE_DIR", Path.home() / ".cache" / "data_viewer" / "rerun"))
RRD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Static file serving base URL
STATIC_BASE_URL = "/api/rerun/files"


def _get_rrd_cache_path(dataset_id: str, episode_id: str) -> Path:
    """Get the cache path for an RRD file."""
    import hashlib
    key = f"{dataset_id}|{episode_id}"
    hash_key = hashlib.sha256(key.encode()).hexdigest()[:16]
    safe_name = episode_id.replace("/", "_").replace("\\", "_")[:50]
    return RRD_CACHE_DIR / f"{safe_name}_{hash_key}.rrd"


def _get_loader_for_episode(dataset_id: str, episode_id: str, data_root: Path):
    """Get appropriate loader and episode for the given IDs."""
    from downloaders.manager import DATASET_REGISTRY
    from loaders import HDF5Loader, LeRobotLoader

    # Check if dataset exists in registry
    if dataset_id not in DATASET_REGISTRY:
        return None, None, False

    config = DATASET_REGISTRY[dataset_id]
    data_dir = data_root / dataset_id

    # For LIBERO HDF5 format
    if dataset_id in ["libero", "libero_pro"]:
        if data_dir.exists():
            loader = HDF5Loader(data_dir)
            return loader, episode_id, False
        return None, None, False

    # For LeRobot format
    if config.get("format") == "lerobot":
        if data_dir.exists():
            loader = LeRobotLoader(data_dir)
            return loader, episode_id, False
        return None, None, False

    # For streaming datasets (RealOmni, Egocentric-10K)
    if config.get("streaming_recommended", False):
        return None, episode_id, True  # Return True to indicate streaming

    return None, None, False


async def _generate_rrd_streaming(
    dataset_id: str,
    episode_id: str,
    cache_path: Path,
    max_frames: int,
    include_actions: bool
):
    """Generate RRD from streaming HuggingFace dataset."""
    import rerun as rr
    import numpy as np

    from loaders.streaming_extractor import StreamingFrameExtractor
    from loaders import get_repo_id_for_dataset

    repo_id = get_repo_id_for_dataset(dataset_id)
    if not repo_id:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset_id}")

    extractor = StreamingFrameExtractor(repo_id)

    logger.info(f"Extracting streaming frames for RRD: {episode_id}")

    # Extract frames from streaming dataset
    frames = extractor.extract_frames(episode_id, start=0, end=max_frames)

    if not frames:
        raise HTTPException(status_code=404, detail="No frames found in episode")

    # Initialize Rerun recording stream
    rec = rr.RecordingStream(
        application_id=f"data_viewer/{dataset_id}",
        recording_id=f"{dataset_id}/{episode_id}",
    )

    # Log frames
    for frame_idx, timestamp, image_array in frames:
        rec.set_time("frame", sequence=frame_idx)
        rec.set_time("time", timestamp=timestamp)
        rec.log("camera/rgb", rr.Image(image_array))

    frame_count = len(frames)

    # Save to file
    rec.save(str(cache_path))
    file_size_kb = cache_path.stat().st_size / 1024
    logger.info(f"Generated streaming RRD: {cache_path} ({file_size_kb:.1f} KB)")

    return JSONResponse({
        "status": "generated",
        "rrd_url": f"{STATIC_BASE_URL}/{cache_path.name}",
        "episode_id": episode_id,
        "dataset_id": dataset_id,
        "num_frames": frame_count,
        "file_size_kb": file_size_kb,
        "source": "streaming"
    })


@router.post("/generate/{episode_id:path}")
async def generate_rrd(
    request: Request,
    episode_id: str,
    dataset_id: str = Query(..., description="Dataset ID"),
    force: bool = Query(False, description="Force regeneration even if cached"),
    include_actions: bool = Query(True, description="Include action data if available"),
    max_frames: int = Query(100, description="Maximum frames to include (for performance)")
):
    """
    Generate a Rerun recording (.rrd) file from episode data.

    Returns the URL to access the generated RRD file.
    """
    try:
        import rerun as rr
        import numpy as np
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="rerun-sdk not installed. Install with: pip install rerun-sdk"
        )

    cache_path = _get_rrd_cache_path(dataset_id, episode_id)

    # Check cache first
    if cache_path.exists() and not force:
        logger.info(f"Using cached RRD: {cache_path}")
        return JSONResponse({
            "status": "cached",
            "rrd_url": f"{STATIC_BASE_URL}/{cache_path.name}",
            "episode_id": episode_id,
            "dataset_id": dataset_id
        })

    # Get data root from app state
    data_root = request.app.state.data_root

    # Get loader for this episode
    loader, ep_id, is_streaming = _get_loader_for_episode(dataset_id, episode_id, data_root)

    if loader is None and not is_streaming:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found or not downloaded. Download it first via the Data Manager."
        )

    try:
        # Get episode data
        logger.info(f"Loading episode for RRD generation: {episode_id}")

        # Handle streaming datasets differently
        if is_streaming:
            return await _generate_rrd_streaming(
                dataset_id, episode_id, cache_path, max_frames, include_actions
            )

        # Load full episode from local loader
        episode = loader.load_episode(ep_id)

        if episode.observations is None or len(episode.observations) == 0:
            raise HTTPException(status_code=404, detail="No frames found in episode")

        # Initialize Rerun recording stream
        rec = rr.RecordingStream(
            application_id=f"data_viewer/{dataset_id}",
            recording_id=f"{dataset_id}/{episode_id}",
        )

        # Limit frames for performance
        num_frames = min(len(episode.observations), max_frames)

        # Log frames as images
        for i in range(num_frames):
            # Get timestamp
            timestamp = episode.timestamps[i] if episode.timestamps is not None and i < len(episode.timestamps) else i / 20.0

            # Set timeline (new API in 0.28+)
            rec.set_time("frame", sequence=i)
            rec.set_time("time", timestamp=timestamp)

            # Log image
            image = episode.observations[i]
            rec.log("camera/rgb", rr.Image(image))

            # Log action if available and requested
            if include_actions and episode.actions is not None and i < len(episode.actions):
                action = episode.actions[i]
                # Log action as series of scalars
                for j, val in enumerate(action):
                    rec.log(f"action/dim_{j}", rr.Scalars([float(val)]))

        frame_count = num_frames

        # Save to file
        rec.save(str(cache_path))
        file_size_kb = cache_path.stat().st_size / 1024
        logger.info(f"Generated RRD file: {cache_path} ({file_size_kb:.1f} KB)")

        return JSONResponse({
            "status": "generated",
            "rrd_url": f"{STATIC_BASE_URL}/{cache_path.name}",
            "episode_id": episode_id,
            "dataset_id": dataset_id,
            "num_frames": frame_count,
            "file_size_kb": file_size_kb
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate RRD: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate RRD: {str(e)}")


@router.get("/files/{filename}")
async def serve_rrd_file(filename: str):
    """Serve a cached RRD file."""
    file_path = RRD_CACHE_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="RRD file not found")

    # Validate the file is within the cache directory (security)
    try:
        file_path.resolve().relative_to(RRD_CACHE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=3600"
        }
    )


@router.get("/status")
async def rerun_status():
    """Check Rerun integration status."""
    try:
        import rerun as rr
        version = rr.__version__
        return {
            "available": True,
            "version": version,
            "cache_dir": str(RRD_CACHE_DIR),
            "cached_files": len(list(RRD_CACHE_DIR.glob("*.rrd")))
        }
    except ImportError:
        return {
            "available": False,
            "error": "rerun-sdk not installed"
        }


@router.delete("/cache")
async def clear_cache():
    """Clear the RRD cache."""
    count = 0
    for rrd_file in RRD_CACHE_DIR.glob("*.rrd"):
        try:
            rrd_file.unlink()
            count += 1
        except Exception as e:
            logger.warning(f"Failed to delete {rrd_file}: {e}")

    return {"deleted": count}
