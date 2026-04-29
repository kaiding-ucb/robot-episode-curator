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
from typing import Any, Optional, TypeVar

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
    get_quality_cache()
    get_frames_cache()

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


# Encoded frames cache directory
ENCODED_FRAMES_CACHE_DIR = CACHE_DIR / "frames" / "encoded"


class CachedEpisodeInfo:
    """Information about a cached episode."""

    def __init__(
        self,
        dataset_id: str,
        episode_id: str,
        size_bytes: int,
        cached_at: float,
        batch_count: int,
    ):
        self.dataset_id = dataset_id
        self.episode_id = episode_id
        self.size_bytes = size_bytes
        self.size_mb = size_bytes / (1024 * 1024)
        self.cached_at = cached_at
        self.batch_count = batch_count


class EncodedFrameCache:
    """
    Caches encoded WebP frames to disk.

    Stores pre-encoded base64 WebP frames to avoid re-encoding on subsequent requests.
    Frames are stored per-batch (e.g., frames 0-30, 30-60) for each episode.
    """

    def __init__(self, cache_dir: Path = None):
        """
        Initialize the encoded frame cache.

        Args:
            cache_dir: Directory to store cache files. Defaults to ~/.cache/data_viewer/frames/encoded
        """
        self.cache_dir = cache_dir or ENCODED_FRAMES_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(
        self,
        dataset_id: str,
        episode_id: str,
        resolution: str,
        quality: int,
        start: int,
        end: int,
    ) -> str:
        """
        Generate unique key for a frame batch (DEPRECATED - use get_episode_cache_key).

        Args:
            dataset_id: Dataset identifier
            episode_id: Episode identifier
            resolution: Image resolution (low/medium/high/original)
            quality: WebP quality (10-100)
            start: Start frame index
            end: End frame index

        Returns:
            Unique cache key string
        """
        key_str = f"{dataset_id}|{episode_id}|{resolution}|{quality}|{start}|{end}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get_episode_cache_key(
        self,
        dataset_id: str,
        episode_id: str,
        resolution: str,
        quality: int,
    ) -> str:
        """
        Generate unique key for a full episode (no start/end).

        This key is used for full-episode caching, which survives browser refresh
        regardless of what batch range the frontend requests.

        Args:
            dataset_id: Dataset identifier
            episode_id: Episode identifier
            resolution: Image resolution (low/medium/high/original)
            quality: WebP quality (10-100)

        Returns:
            Unique cache key string
        """
        key_str = f"{dataset_id}|{episode_id}|{resolution}|{quality}|full"
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def _get_episode_dir(self, dataset_id: str, episode_id: str) -> Path:
        """Get the directory for a specific episode's cache files."""
        # Sanitize episode_id for filesystem use
        safe_episode_id = episode_id.replace("/", "__").replace("\\", "__")
        return self.cache_dir / dataset_id / safe_episode_id

    def _get_cache_path(self, key: str, dataset_id: str, episode_id: str) -> Path:
        """Get the file path for a cache key."""
        episode_dir = self._get_episode_dir(dataset_id, episode_id)
        episode_dir.mkdir(parents=True, exist_ok=True)
        return episode_dir / f"{key}.json"

    def get_frames(self, key: str, dataset_id: str, episode_id: str) -> Optional[dict]:
        """
        Return cached frames or None (DEPRECATED - use get_episode_frames).

        Args:
            key: Cache key from get_cache_key()
            dataset_id: Dataset identifier
            episode_id: Episode identifier

        Returns:
            Cached data dict with 'frames' and 'total' keys, or None if not found
        """
        cache_path = self._get_cache_path(key, dataset_id, episode_id)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                logger.debug(f"Cache hit for key: {key}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load cached frames for key {key}: {e}")
            return None

    def get_episode_frames(self, key: str, dataset_id: str, episode_id: str) -> Optional[dict]:
        """
        Return all cached frames for an episode, or None if not found.

        Args:
            key: Cache key from get_episode_cache_key()
            dataset_id: Dataset identifier
            episode_id: Episode identifier

        Returns:
            Cached data dict with 'frames' (all frames) and 'total' keys, or None
        """
        cache_path = self._get_cache_path(key, dataset_id, episode_id)

        if not cache_path.exists():
            logger.debug(f"Episode cache miss for {dataset_id}/{episode_id}")
            return None

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                logger.info(f"Episode cache hit: {dataset_id}/{episode_id} ({len(data.get('frames', []))} frames)")
                return data
        except Exception as e:
            logger.warning(f"Failed to load episode cache for key {key}: {e}")
            return None

    def store_frames(
        self,
        key: str,
        frames: list,
        total_frames: int,
        metadata: dict,
    ) -> bool:
        """
        Store frames with metadata (DEPRECATED - use store_episode_frames).

        Args:
            key: Cache key from get_cache_key()
            frames: List of frame data dicts
            total_frames: Total frames in episode
            metadata: Metadata dict with dataset_id, episode_id, resolution, quality, etc.

        Returns:
            True if stored successfully
        """
        dataset_id = metadata.get("dataset_id", "unknown")
        episode_id = metadata.get("episode_id", "unknown")
        cache_path = self._get_cache_path(key, dataset_id, episode_id)

        try:
            data = {
                "frames": frames,
                "total": total_frames,
                "metadata": metadata,
                "cached_at": time.time(),
            }

            with open(cache_path, "w") as f:
                json.dump(data, f)

            logger.debug(f"Cached {len(frames)} frames for key: {key}")
            return True

        except Exception as e:
            logger.warning(f"Failed to cache frames for key {key}: {e}")
            return False

    def store_episode_frames(
        self,
        key: str,
        all_frames: list,
        total_frames: int,
        metadata: dict,
    ) -> bool:
        """
        Store ALL frames for an episode.

        This stores the complete episode so that any subsequent requests
        (regardless of batch boundaries) can be served from cache.

        Args:
            key: Cache key from get_episode_cache_key()
            all_frames: List of ALL frame data dicts for the episode
            total_frames: Total frames in episode
            metadata: Metadata dict with dataset_id, episode_id, resolution, quality

        Returns:
            True if stored successfully
        """
        dataset_id = metadata.get("dataset_id", "unknown")
        episode_id = metadata.get("episode_id", "unknown")
        cache_path = self._get_cache_path(key, dataset_id, episode_id)

        try:
            data = {
                "frames": all_frames,
                "total": total_frames,
                "metadata": metadata,
                "cached_at": time.time(),
                "full_episode": True,
            }

            with open(cache_path, "w") as f:
                json.dump(data, f)

            size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"Cached full episode: {dataset_id}/{episode_id} ({total_frames} frames, {size_mb:.2f} MB)")
            return True

        except Exception as e:
            logger.warning(f"Failed to cache full episode for key {key}: {e}")
            return False

    def list_cached_episodes(self) -> list:
        """
        Return all FULLY cached episodes with size and timestamp.

        Only returns episodes where full_episode=True in the cache file,
        meaning ALL frames have been encoded and cached. Partial caches
        are not included.

        Returns:
            List of CachedEpisodeInfo objects
        """
        episodes = []

        if not self.cache_dir.exists():
            return episodes

        # Iterate through dataset directories
        for dataset_dir in self.cache_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_id = dataset_dir.name

            # Iterate through episode directories
            for episode_dir in dataset_dir.iterdir():
                if not episode_dir.is_dir():
                    continue

                # Restore original episode_id from safe name
                episode_id = episode_dir.name.replace("__", "/")

                # Look for a cache file with full_episode=True
                total_size = 0
                latest_timestamp = 0
                full_episode_found = False

                for cache_file in episode_dir.glob("*.json"):
                    try:
                        stat = cache_file.stat()
                        total_size += stat.st_size
                        latest_timestamp = max(latest_timestamp, stat.st_mtime)

                        # Check if this cache file has full_episode flag
                        with open(cache_file, "r") as f:
                            data = json.load(f)
                            if data.get("full_episode", False):
                                full_episode_found = True
                    except Exception:
                        continue

                # Only include episodes that are fully cached
                if full_episode_found:
                    episodes.append(
                        CachedEpisodeInfo(
                            dataset_id=dataset_id,
                            episode_id=episode_id,
                            size_bytes=total_size,
                            cached_at=latest_timestamp,
                            batch_count=1,  # Now represents "full episode cached"
                        )
                    )

        # Sort by most recently cached
        episodes.sort(key=lambda e: e.cached_at, reverse=True)
        return episodes

    def delete_episode_cache(self, dataset_id: str, episode_id: str) -> int:
        """
        Delete all cached batches for an episode.

        Args:
            dataset_id: Dataset identifier
            episode_id: Episode identifier

        Returns:
            Bytes freed
        """
        episode_dir = self._get_episode_dir(dataset_id, episode_id)

        if not episode_dir.exists():
            return 0

        bytes_freed = 0
        try:
            for cache_file in episode_dir.glob("*.json"):
                try:
                    bytes_freed += cache_file.stat().st_size
                    cache_file.unlink()
                except Exception:
                    pass

            # Try to remove the episode directory if empty
            try:
                episode_dir.rmdir()
            except Exception:
                pass

            # Try to remove the dataset directory if empty
            try:
                dataset_dir = self.cache_dir / dataset_id
                dataset_dir.rmdir()
            except Exception:
                pass

            logger.info(f"Deleted cache for episode: {dataset_id}/{episode_id}, freed {bytes_freed} bytes")

        except Exception as e:
            logger.warning(f"Error deleting episode cache: {e}")

        return bytes_freed

    def clear_all(self) -> int:
        """
        Clear entire encoded frames cache.

        Returns:
            Bytes freed
        """
        bytes_freed = 0

        if not self.cache_dir.exists():
            return 0

        try:
            for dataset_dir in list(self.cache_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue

                for episode_dir in list(dataset_dir.iterdir()):
                    if not episode_dir.is_dir():
                        continue

                    for cache_file in list(episode_dir.glob("*.json")):
                        try:
                            bytes_freed += cache_file.stat().st_size
                            cache_file.unlink()
                        except Exception:
                            pass

                    try:
                        episode_dir.rmdir()
                    except Exception:
                        pass

                try:
                    dataset_dir.rmdir()
                except Exception:
                    pass

            logger.info(f"Cleared all encoded frame cache, freed {bytes_freed} bytes")

        except Exception as e:
            logger.warning(f"Error clearing encoded frame cache: {e}")

        return bytes_freed

    def get_cache_stats(self) -> dict:
        """
        Return cache statistics.

        Returns:
            Dict with total_size_mb, episode_count, batch_count
        """
        episodes = self.list_cached_episodes()
        total_size = sum(e.size_bytes for e in episodes)
        total_batches = sum(e.batch_count for e in episodes)

        return {
            "total_size_mb": total_size / (1024 * 1024),
            "episode_count": len(episodes),
            "batch_count": total_batches,
        }

    async def list_cached_episodes_async(self) -> list:
        """
        Async version of list_cached_episodes that runs in executor.
        Use this from async endpoints to avoid blocking the event loop.
        """
        import asyncio

        from api.routes.episodes import _HEAVY_EXECUTOR
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_HEAVY_EXECUTOR, self.list_cached_episodes)

    async def get_cache_stats_async(self) -> dict:
        """
        Async version of get_cache_stats that runs in executor.
        Use this from async endpoints to avoid blocking the event loop.
        """
        import asyncio

        from api.routes.episodes import _HEAVY_EXECUTOR
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_HEAVY_EXECUTOR, self.get_cache_stats)


# Global encoded frame cache instance
_encoded_frame_cache: Optional[EncodedFrameCache] = None


def get_encoded_frame_cache() -> EncodedFrameCache:
    """Get the global encoded frame cache instance."""
    global _encoded_frame_cache
    if _encoded_frame_cache is None:
        _encoded_frame_cache = EncodedFrameCache()
    return _encoded_frame_cache
