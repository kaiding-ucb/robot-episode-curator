"""
Caching utilities for the Data Viewer backend.

Provides persistent file-based caching for:
- Quality analysis results
- Decoded episode frames
- Episode metadata
"""
import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional, TypeVar, Callable
from functools import wraps

logger = logging.getLogger(__name__)

# Default cache directory
CACHE_DIR = Path(os.environ.get("DATA_VIEWER_CACHE_DIR", Path.home() / ".cache" / "data_viewer"))

# Cache subdirectories
QUALITY_CACHE_DIR = CACHE_DIR / "quality"
FRAMES_CACHE_DIR = CACHE_DIR / "frames"
METADATA_CACHE_DIR = CACHE_DIR / "metadata"

# Default TTL (time-to-live) in seconds - 7 days
DEFAULT_TTL = 7 * 24 * 60 * 60

T = TypeVar('T')


def ensure_cache_dirs():
    """Create cache directories if they don't exist."""
    for cache_dir in [QUALITY_CACHE_DIR, FRAMES_CACHE_DIR, METADATA_CACHE_DIR]:
        cache_dir.mkdir(parents=True, exist_ok=True)


def get_cache_key(*args) -> str:
    """Generate a cache key from arguments."""
    key_str = "|".join(str(arg) for arg in args)
    return hashlib.sha256(key_str.encode()).hexdigest()[:32]


class FileCache:
    """
    Simple file-based cache with TTL support.

    Stores cached data as JSON or pickle files on disk.
    """

    def __init__(self, cache_dir: Path, ttl: int = DEFAULT_TTL, use_pickle: bool = False):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live in seconds (default: 7 days)
            use_pickle: Use pickle instead of JSON (for complex objects like numpy arrays)
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.use_pickle = use_pickle
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        ext = ".pkl" if self.use_pickle else ".json"
        return self.cache_dir / f"{key}{ext}"

    def _get_meta_path(self, key: str) -> Path:
        """Get the metadata file path for a cache key."""
        return self.cache_dir / f"{key}.meta"

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        if not cache_path.exists():
            return None

        # Check TTL
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    created_at = meta.get('created_at', 0)
                    if time.time() - created_at > self.ttl:
                        logger.debug(f"Cache expired for key: {key}")
                        self.delete(key)
                        return None
            except Exception:
                pass

        # Load cached data
        try:
            if self.use_pickle:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(cache_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache for key {key}: {e}")
            return None

    def set(self, key: str, value: Any) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache

        Returns:
            True if successful
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        try:
            # Save data
            if self.use_pickle:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
            else:
                with open(cache_path, 'w') as f:
                    json.dump(value, f)

            # Save metadata
            with open(meta_path, 'w') as f:
                json.dump({'created_at': time.time()}, f)

            logger.debug(f"Cached value for key: {key}")
            return True

        except Exception as e:
            logger.warning(f"Failed to cache value for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        try:
            if cache_path.exists():
                cache_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
            return True
        except Exception:
            return False

    def clear(self) -> int:
        """Clear all cache entries. Returns number of entries cleared."""
        count = 0
        for path in self.cache_dir.glob("*"):
            try:
                path.unlink()
                count += 1
            except Exception:
                pass
        return count // 2  # Divide by 2 because each entry has data + meta file

    def has(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        return self.get(key) is not None


# Global cache instances
_quality_cache: Optional[FileCache] = None
_frames_cache: Optional[FileCache] = None


def get_quality_cache() -> FileCache:
    """Get the quality analysis cache (JSON-based)."""
    global _quality_cache
    if _quality_cache is None:
        ensure_cache_dirs()
        _quality_cache = FileCache(QUALITY_CACHE_DIR, ttl=DEFAULT_TTL, use_pickle=False)
    return _quality_cache


def get_frames_cache() -> FileCache:
    """Get the frames cache (pickle-based for numpy arrays)."""
    global _frames_cache
    if _frames_cache is None:
        ensure_cache_dirs()
        _frames_cache = FileCache(FRAMES_CACHE_DIR, ttl=DEFAULT_TTL, use_pickle=True)
    return _frames_cache


def cache_quality_result(dataset_id: str, episode_id: str, result: dict) -> bool:
    """
    Cache a quality analysis result.

    Args:
        dataset_id: Dataset identifier
        episode_id: Episode identifier
        result: Quality result dictionary

    Returns:
        True if cached successfully
    """
    cache = get_quality_cache()
    key = get_cache_key("quality", dataset_id, episode_id)
    return cache.set(key, result)


def get_cached_quality_result(dataset_id: str, episode_id: str) -> Optional[dict]:
    """
    Get a cached quality analysis result.

    Args:
        dataset_id: Dataset identifier
        episode_id: Episode identifier

    Returns:
        Cached quality result or None
    """
    cache = get_quality_cache()
    key = get_cache_key("quality", dataset_id, episode_id)
    return cache.get(key)


def cache_quality_events(dataset_id: str, episode_id: str, events: dict) -> bool:
    """
    Cache quality events for an episode.

    Args:
        dataset_id: Dataset identifier
        episode_id: Episode identifier
        events: Quality events dictionary

    Returns:
        True if cached successfully
    """
    cache = get_quality_cache()
    key = get_cache_key("quality_events", dataset_id, episode_id)
    return cache.set(key, events)


def get_cached_quality_events(dataset_id: str, episode_id: str) -> Optional[dict]:
    """
    Get cached quality events for an episode.

    Args:
        dataset_id: Dataset identifier
        episode_id: Episode identifier

    Returns:
        Cached quality events or None
    """
    cache = get_quality_cache()
    key = get_cache_key("quality_events", dataset_id, episode_id)
    return cache.get(key)


def cache_frames(repo_id: str, episode_path: str, frames_data: list) -> bool:
    """
    Cache decoded frames for an episode.

    Args:
        repo_id: HuggingFace repo ID
        episode_path: Path to episode within repo
        frames_data: List of frame data (will be pickled)

    Returns:
        True if cached successfully
    """
    cache = get_frames_cache()
    key = get_cache_key("frames", repo_id, episode_path)
    return cache.set(key, frames_data)


def get_cached_frames(repo_id: str, episode_path: str) -> Optional[list]:
    """
    Get cached frames for an episode.

    Args:
        repo_id: HuggingFace repo ID
        episode_path: Path to episode within repo

    Returns:
        Cached frames list or None
    """
    cache = get_frames_cache()
    key = get_cache_key("frames", repo_id, episode_path)
    return cache.get(key)


def clear_episode_cache(dataset_id: str, episode_id: str):
    """Clear all cached data for a specific episode."""
    quality_cache = get_quality_cache()

    # Clear quality results
    quality_key = get_cache_key("quality", dataset_id, episode_id)
    quality_cache.delete(quality_key)

    # Clear quality events
    events_key = get_cache_key("quality_events", dataset_id, episode_id)
    quality_cache.delete(events_key)

    logger.info(f"Cleared cache for episode: {dataset_id}/{episode_id}")


def get_cache_stats() -> dict:
    """Get cache statistics."""
    quality_cache = get_quality_cache()
    frames_cache = get_frames_cache()

    def count_entries(cache_dir: Path) -> int:
        return len(list(cache_dir.glob("*.json"))) + len(list(cache_dir.glob("*.pkl")))

    def get_size(cache_dir: Path) -> int:
        return sum(f.stat().st_size for f in cache_dir.glob("*") if f.is_file())

    return {
        "quality_entries": count_entries(QUALITY_CACHE_DIR),
        "quality_size_mb": get_size(QUALITY_CACHE_DIR) / (1024 * 1024),
        "frames_entries": count_entries(FRAMES_CACHE_DIR),
        "frames_size_mb": get_size(FRAMES_CACHE_DIR) / (1024 * 1024),
    }
