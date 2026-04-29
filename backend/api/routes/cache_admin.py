"""
Cache admin endpoints — total size + clear-all.

Used by the sidebar "Clear cache" button (replaces the legacy Manage Downloads
panel). Walks the disk caches written by edge_frames, first-frames, decoded
frames, and the LeRobot metadata cache.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


CACHE_BASE = Path.home() / ".cache" / "data_viewer"

# Sub-caches we know about. Keep additive — unknown subdirs under CACHE_BASE
# are also accounted for in size and cleared in the global delete.
KNOWN_CACHES: List[str] = [
    "edge_frames",
    "first_frames",
    "decoded_frames",
    "metadata",
    "video_clips",
]


class CacheBucket(BaseModel):
    name: str
    bytes: int


class CacheSizeResponse(BaseModel):
    total_bytes: int
    total_mb: float
    buckets: List[CacheBucket]


class CacheClearResponse(BaseModel):
    cleared_bytes: int
    cleared_mb: float


def _dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    total = 0
    for child in p.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def enforce_lru_cap(directory: Path, max_bytes: int) -> int:
    """Delete oldest files (by mtime) under `directory` until total size ≤ max_bytes.

    Returns the number of files removed. Safe to call from anywhere — never
    raises, never deletes outside `directory`.
    """
    if not directory.exists():
        return 0
    try:
        files = []
        for child in directory.rglob("*"):
            try:
                if child.is_file():
                    files.append((child.stat().st_mtime, child.stat().st_size, child))
            except OSError:
                continue
        files.sort(key=lambda t: t[0])  # oldest first
        total = sum(sz for _, sz, _ in files)
        removed = 0
        for _, sz, path in files:
            if total <= max_bytes:
                break
            try:
                path.unlink(missing_ok=True)
                total -= sz
                removed += 1
            except OSError as e:
                logger.warning(f"enforce_lru_cap: unlink failed for {path}: {e}")
        if removed:
            logger.info(
                f"enforce_lru_cap({directory.name}): removed {removed} file(s) "
                f"to keep ≤ {max_bytes / (1024 * 1024):.0f} MB"
            )
        return removed
    except Exception as e:
        logger.warning(f"enforce_lru_cap failed for {directory}: {e}")
        return 0


@router.get("/size", response_model=CacheSizeResponse)
async def cache_size():
    """Return total disk usage of all known caches."""
    buckets: List[CacheBucket] = []
    total = 0
    seen_names = set()

    if CACHE_BASE.exists():
        # Sum any subdirectory of CACHE_BASE so newly-added caches are picked up.
        for child in CACHE_BASE.iterdir():
            if not child.is_dir():
                continue
            size = _dir_size(child)
            buckets.append(CacheBucket(name=child.name, bytes=size))
            seen_names.add(child.name)
            total += size

    # Stable ordering: known buckets first, then any extras.
    def _rank(b: CacheBucket) -> tuple:
        try:
            return (0, KNOWN_CACHES.index(b.name))
        except ValueError:
            return (1, b.name)

    buckets.sort(key=_rank)

    return CacheSizeResponse(
        total_bytes=total,
        total_mb=round(total / (1024 * 1024), 2),
        buckets=buckets,
    )


@router.delete("")
async def cache_clear() -> CacheClearResponse:
    """Delete every subdirectory under ~/.cache/data_viewer."""
    if not CACHE_BASE.exists():
        return CacheClearResponse(cleared_bytes=0, cleared_mb=0.0)

    total_before = _dir_size(CACHE_BASE)
    failures: List[str] = []
    for child in list(CACHE_BASE.iterdir()):
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=False)
            else:
                child.unlink(missing_ok=True)
        except Exception as e:
            failures.append(f"{child.name}: {e}")

    if failures:
        logger.warning(f"cache_clear: {len(failures)} failure(s): {failures[:3]}")

    return CacheClearResponse(
        cleared_bytes=total_before,
        cleared_mb=round(total_before / (1024 * 1024), 2),
    )
