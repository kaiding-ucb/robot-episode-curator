"""Per-episode MP4 clip cache. Streams the per-episode byte range from the
HF chunk MP4 via ffmpeg HTTP-range reads — no full chunk download — and
caches the small slice on disk for reuse by repeated Gemini analyses."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.video_slice import slice_remote_mp4, FfmpegError

logger = logging.getLogger(__name__)

def _read_hf_token_file() -> Optional[str]:
    for p in (Path.home() / ".huggingface" / "token", Path.home() / ".cache" / "huggingface" / "token"):
        if p.exists():
            try:
                t = p.read_text().strip()
                if t:
                    return t
            except OSError:
                continue
    return None


HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or _read_hf_token_file()
    or ""
)
CACHE_ROOT = Path(os.environ.get("DATA_VIEWER_CACHE", Path.home() / ".cache" / "data_viewer")) / "phase_aware_gemini" / "clips"


def _safe_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def cache_path_for(repo_id: str, episode_idx: int) -> Path:
    return CACHE_ROOT / _safe_repo_id(repo_id) / f"episode_{episode_idx}.mp4"


def _find_episode_meta_row(repo_id: str, episode_idx: int) -> Optional[pd.Series]:
    """Find the meta/episodes row for `episode_idx` via the existing async cache."""
    import asyncio
    from api.routes.datasets import fetch_lerobot_episodes_meta

    try:
        df = asyncio.run(fetch_lerobot_episodes_meta(repo_id))
    except Exception as e:
        logger.warning(f"video_cache: episodes meta fetch failed for {repo_id}: {e}")
        return None
    if df is None:
        return None
    rows = df[df["episode_index"] == episode_idx]
    if len(rows) == 0:
        return None
    return rows.iloc[0]


def _pick_primary_video_key(row: pd.Series) -> Optional[str]:
    """Pick the first available image video key from the meta row. Libero has
    'observation.images.image' (and 'image2'), UMI has 'observation.image', etc."""
    candidates = []
    for col in row.index:
        if col.startswith("videos/") and col.endswith("/file_index"):
            key = col[len("videos/"):-len("/file_index")]
            # Skip depth/wrist/image2 when possible — prefer the main "image"
            rank = 0
            if "depth" in key.lower():
                rank += 100
            if "wrist" in key.lower():
                rank += 10
            if key.endswith("image2"):
                rank += 5
            candidates.append((rank, key))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


async def get_episode_clip(
    repo_id: str,
    episode_idx: int,
    video_path_template: str,
    *,
    force: bool = False,
    timing: Optional[dict] = None,
) -> Optional[Path]:
    """Return a cached MP4 clip for one episode. Downloads the HF chunk MP4
    and ffmpeg-slices to this episode's timestamp range if not cached.

    Args:
        repo_id: e.g. "lerobot/libero"
        episode_idx: global episode_index
        video_path_template: from info.json, e.g. "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        force: re-extract even if cached
        timing: if provided, accumulates latency breakdown (keys: 'meta', 'chunk_dl', 'ffmpeg')
    """
    out = cache_path_for(repo_id, episode_idx)
    if out.exists() and out.stat().st_size > 1024 and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    row = _find_episode_meta_row(repo_id, episode_idx)
    if timing is not None:
        timing["meta"] = timing.get("meta", 0.0) + (time.time() - t0)
    if row is None:
        logger.warning(f"video_cache: no meta row for {repo_id}/episode_{episode_idx}")
        return None

    video_key = _pick_primary_video_key(row)
    if video_key is None:
        logger.warning(f"video_cache: no video key on meta row for {repo_id}/episode_{episode_idx}")
        return None

    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    t0_col = f"videos/{video_key}/from_timestamp"
    t1_col = f"videos/{video_key}/to_timestamp"
    try:
        chunk_idx = int(row[chunk_col])
        file_idx = int(row[file_col])
        t_start = float(row[t0_col])
        t_end = float(row[t1_col])
    except Exception as e:
        logger.warning(f"video_cache: missing video fields in meta row: {e}")
        return None

    rel = video_path_template.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}"

    t0 = time.time()
    try:
        slice_remote_mp4(
            url=url,
            hf_token=HF_TOKEN or None,
            t_start=t_start,
            t_end=t_end,
            out_path=out,
            pre_roll=0.0,
        )
    except FfmpegError as e:
        logger.warning(f"video_cache: slice failed for episode_{episode_idx} ({rel}): {e}")
        return None
    if timing is not None:
        # Combined network + ffmpeg time — no separate chunk download phase any more.
        timing["ffmpeg"] = timing.get("ffmpeg", 0.0) + (time.time() - t0)

    # Cap the Gemini clip cache at 500 MB.
    try:
        from api.routes.cache_admin import enforce_lru_cap
        enforce_lru_cap(CACHE_ROOT, 500 * 1024 * 1024)
    except Exception as e:
        logger.warning(f"LRU cap on Gemini clip cache failed: {e}")

    return out
