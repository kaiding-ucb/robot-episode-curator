"""
Streaming Frame Extractor for HuggingFace datasets.

Extracts frames from streaming datasets (MCAP, WebDataset/TAR, video files)
by downloading individual episode files on demand.

Includes persistent caching to avoid re-downloading and re-decoding on subsequent requests.
Supports multiple modalities: RGB, depth, IMU.
"""
import logging
import math
import os
import pickle
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# Depth colormap lookup
DEPTH_COLORMAPS = {
    "viridis": None,  # Will be cv2.COLORMAP_VIRIDIS
    "jet": None,  # Will be cv2.COLORMAP_JET
    "plasma": None,  # Will be cv2.COLORMAP_PLASMA
    "magma": None,  # Will be cv2.COLORMAP_MAGMA
}


def _get_depth_colormap(name: str):
    """Get OpenCV colormap constant by name."""
    try:
        import cv2
        colormap_map = {
            "viridis": cv2.COLORMAP_VIRIDIS,
            "jet": cv2.COLORMAP_JET,
            "plasma": cv2.COLORMAP_PLASMA,
            "magma": cv2.COLORMAP_MAGMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "turbo": cv2.COLORMAP_TURBO,
        }
        return colormap_map.get(name, cv2.COLORMAP_VIRIDIS)
    except ImportError:
        return None


def colorize_depth(depth_data: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """
    Colorize a depth image for visualization.

    Args:
        depth_data: Depth data as numpy array (16-bit or float)
        colormap: Name of colormap to use

    Returns:
        RGB colorized depth image (H, W, 3) uint8
    """
    try:
        import cv2

        # Normalize to 0-255
        if depth_data.dtype == np.float32 or depth_data.dtype == np.float64:
            # Float depth (assume meters, normalize by max)
            depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
        elif depth_data.dtype == np.uint16:
            # 16-bit depth (common in RGB-D cameras)
            depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
        else:
            depth_normalized = depth_data

        depth_8bit = depth_normalized.astype(np.uint8)

        # Apply colormap
        cmap = _get_depth_colormap(colormap)
        colored = cv2.applyColorMap(depth_8bit, cmap)

        # Convert BGR to RGB
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    except ImportError:
        logger.warning("OpenCV not available for depth colorization")
        # Fallback: return grayscale expanded to 3 channels
        if depth_data.dtype != np.uint8:
            depth_normalized = ((depth_data - depth_data.min()) /
                               (depth_data.max() - depth_data.min()) * 255).astype(np.uint8)
        else:
            depth_normalized = depth_data
        return np.stack([depth_normalized] * 3, axis=-1)

# Cache directory for downloaded files
CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", Path.home() / ".cache" / "data_viewer" / "streaming"))

# Persistent frame cache directory (for decoded frames)
FRAME_CACHE_DIR = CACHE_DIR / "decoded_frames"
FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Memory-limited LRU cache for decoded frames
# Only cache a few recent episodes to prevent memory explosion
# Each frame is ~3MB at 720p, so 1000 frames = ~3GB max per episode
_FRAME_CACHE: dict = {}
_FRAME_CACHE_ORDER: list = []  # Track access order for LRU eviction
_FRAME_CACHE_MAX_EPISODES = 2  # Only keep 2 most recent episodes in memory
_FRAME_CACHE_MAX_FRAMES_PER_EPISODE = 500  # Max frames to cache per episode


def _evict_frame_cache_if_needed():
    """Evict oldest episodes from cache if we exceed limits."""
    global _FRAME_CACHE, _FRAME_CACHE_ORDER

    while len(_FRAME_CACHE_ORDER) > _FRAME_CACHE_MAX_EPISODES:
        oldest_key = _FRAME_CACHE_ORDER.pop(0)
        if oldest_key in _FRAME_CACHE:
            # Log what we're evicting
            evicted_frames = len(_FRAME_CACHE[oldest_key])
            del _FRAME_CACHE[oldest_key]
            logger.info(f"Evicted {evicted_frames} frames from memory cache: {oldest_key}")


def _update_frame_cache_access(cache_key: str):
    """Update LRU order when a cache key is accessed."""
    global _FRAME_CACHE_ORDER

    if cache_key in _FRAME_CACHE_ORDER:
        _FRAME_CACHE_ORDER.remove(cache_key)
    _FRAME_CACHE_ORDER.append(cache_key)


def _get_frame_cache_path(repo_id: str, episode_path: str) -> Path:
    """Get the persistent cache path for decoded frames."""
    import hashlib
    key = f"{repo_id}|{episode_path}"
    hash_key = hashlib.sha256(key.encode()).hexdigest()[:32]
    return FRAME_CACHE_DIR / f"{hash_key}.pkl"


def _load_persistent_frame_cache(repo_id: str, episode_path: str) -> Optional[List]:
    """Load frames from persistent cache if available.

    WARNING: This loads ALL frames into memory. For large episodes (>1GB cache files),
    we skip loading to prevent memory explosion.
    """
    cache_path = _get_frame_cache_path(repo_id, episode_path)
    if cache_path.exists():
        try:
            # Skip loading if cache file is too large (>500MB = potentially huge memory usage)
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            if cache_size_mb > 500:
                logger.warning(f"Skipping persistent cache load - file too large ({cache_size_mb:.1f}MB): {episode_path}")
                return None

            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"Loaded {len(data)} frames from persistent cache for {episode_path} ({cache_size_mb:.1f}MB)")
                return data
        except Exception as e:
            logger.warning(f"Failed to load persistent frame cache: {e}")
    return None


def _save_persistent_frame_cache(repo_id: str, episode_path: str, frames: List) -> bool:
    """Save frames to persistent cache.

    WARNING: Only saves if episode is reasonably sized (<2000 frames) to prevent
    creating huge cache files that cause memory issues when loaded.
    """
    # Don't save huge episodes - they'll cause memory issues when loading
    if len(frames) > 2000:
        logger.info(f"Skipping persistent cache save - too many frames ({len(frames)}): {episode_path}")
        return False

    # Size guard: estimate raw numpy size and reject if too large
    if frames and hasattr(frames[0][2], 'nbytes'):
        estimated_mb = (frames[0][2].nbytes * len(frames)) / (1024 * 1024)
        if estimated_mb > 200:
            logger.info(f"Skipping persistent cache save - estimated {estimated_mb:.0f}MB exceeds 200MB limit: {episode_path}")
            return False

    cache_path = _get_frame_cache_path(repo_id, episode_path)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(frames, f)
        logger.info(f"Saved {len(frames)} frames to persistent cache for {episode_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save persistent frame cache: {e}")
        return False


def cleanup_decoded_frames(repo_id: str, episode_path: str, stream: str = "rgb") -> bool:
    """
    Delete decoded frame pickle files after WebP encoding completes.

    Called automatically after an episode is fully cached to free storage.
    Decoded frames are intermediate data that can always be regenerated
    from the source files if needed.

    Args:
        repo_id: HuggingFace repository ID
        episode_path: Path to the episode file
        stream: Stream type (rgb or depth)

    Returns:
        True if cleanup was successful or file didn't exist
    """
    cache_key = f"{episode_path}:{stream}"
    cache_path = _get_frame_cache_path(repo_id, cache_key)

    if cache_path.exists():
        try:
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            cache_path.unlink()
            logger.info(f"Cleaned up decoded frames ({size_mb:.1f} MB): {episode_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cleanup decoded frames: {e}")
            return False
    return True


def cleanup_all_decoded_frames() -> dict:
    """
    Delete ALL decoded frame pickle files.

    Useful for storage management - decoded frames are intermediate
    and can always be regenerated from source files.

    Returns:
        Dict with files_deleted and bytes_freed
    """
    count = 0
    bytes_freed = 0

    for pkl_file in FRAME_CACHE_DIR.glob("*.pkl"):
        try:
            bytes_freed += pkl_file.stat().st_size
            pkl_file.unlink()
            count += 1
        except Exception:
            pass

    logger.info(f"Cleaned up {count} decoded frame files ({bytes_freed / 1024 / 1024:.1f} MB)")
    return {"files_deleted": count, "bytes_freed": bytes_freed}


class StreamingFrameExtractor:
    """
    Extracts frames from streaming HuggingFace datasets.

    Supports:
    - MCAP files (RealOmni format with protobuf CompressedImage)
    - WebDataset TAR files (Egocentric-10K with video.mp4)
    - Direct video files (MP4, etc.)
    """

    # Topic pattern constants for MCAP signal extraction
    _ACTION_TOPIC_PATTERNS = [
        "action", "command", "control", "cmd", "joint",
        "gripper", "target", "eef_pose", "end_effector",
    ]
    _IMU_TOPIC_PATTERNS = ["imu", "accelerometer", "gyroscope", "accel", "gyro"]
    _SKIP_TOPIC_PATTERNS = ["camera", "image", "rgb", "depth", "compressed"]

    def __init__(self, repo_id: str, cache_dir: Optional[Path] = None):
        self.repo_id = repo_id
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cached_decoder_factories = None

    def get_cached_path(self, file_path: str) -> Path:
        """Get the local cache path for a file."""
        # Create a safe filename from the path
        safe_name = file_path.replace("/", "_").replace("\\", "_")
        return self.cache_dir / self.repo_id.replace("/", "_") / safe_name

    def _get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token from environment or token file."""
        # Try environment variable first
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if token:
            return token

        # Try token file
        token_file = Path.home() / ".huggingface" / "token"
        if token_file.exists():
            return token_file.read_text().strip()

        # Try cache token file (alternative location)
        cache_token = Path.home() / ".cache" / "huggingface" / "token"
        if cache_token.exists():
            return cache_token.read_text().strip()

        return None

    def _open_hf_remote(self, file_path: str):
        """
        Open a remote HF file as a seekable file-like object via HfFileSystem.

        The returned object supports seek() and read(), making it compatible
        with mcap's make_reader (which auto-selects SeekingReader for seekable
        streams, enabling HTTP range requests for selective chunk reads).

        Args:
            file_path: Path within the repo (e.g. "task/episode/00001.mcap")

        Returns:
            A seekable file-like object backed by HTTP range requests.
        """
        from huggingface_hub import HfFileSystem

        token = self._get_hf_token()
        fs = HfFileSystem(token=token)
        return fs.open(f"datasets/{self.repo_id}/{file_path}", "rb")

    def download_file(self, file_path: str, revision: Optional[str] = None) -> Path:
        """
        Download a file from HuggingFace if not already cached.

        Args:
            file_path: Path within the HuggingFace repo (e.g., "Cooking_and_Kitchen_Clean/clean_bowl/00001/00001.mcap")
            revision: Branch/tag/commit to download from (e.g., "v2.0"). Defaults to "main".

        Returns:
            Local path to the downloaded file
        """
        cached_path = self.get_cached_path(file_path)

        if cached_path.exists():
            logger.info(f"Using cached file: {cached_path}")
            return cached_path

        # Download from HuggingFace
        try:
            from huggingface_hub import hf_hub_download

            branch_info = f" (revision={revision})" if revision and revision != "main" else ""
            logger.info(f"Downloading {file_path} from {self.repo_id}{branch_info}...")

            # Create parent directory
            cached_path.parent.mkdir(parents=True, exist_ok=True)

            # Get authentication token for gated repos
            token = self._get_hf_token()

            # Download to cache
            download_kwargs = {
                "repo_id": self.repo_id,
                "filename": file_path,
                "repo_type": "dataset",
                "local_dir": cached_path.parent,
                "local_dir_use_symlinks": False,
                "token": token,
            }
            if revision and revision != "main":
                download_kwargs["revision"] = revision

            downloaded_path = hf_hub_download(**download_kwargs)

            # Move to our cache location if different
            downloaded = Path(downloaded_path)
            if downloaded != cached_path:
                if downloaded.exists():
                    import shutil
                    shutil.copy2(downloaded, cached_path)

            logger.info(f"Downloaded to: {cached_path}")
            return cached_path

        except Exception as e:
            logger.error(f"Failed to download {file_path}: {e}")
            raise

    def extract_frames_from_mcap(
        self,
        file_path: Path,
        start: int = 0,
        end: int = 10,
        camera_topic: str = "/robot0/sensor/camera0/compressed",
        episode_path: str = None,
        stream: str = "rgb",
        depth_colormap: str = "viridis",
        force_full_extraction: bool = False,
        stride: int = 1,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract frames from an MCAP file.

        Args:
            file_path: Path to the MCAP file
            start: Start frame index
            end: End frame index
            camera_topic: Topic name for camera images
            episode_path: Original episode path (for persistent caching)
            stream: Which stream to extract: "rgb" or "depth"
            depth_colormap: Colormap for depth visualization

        Returns:
            List of (frame_idx, timestamp, image_array) tuples
        """
        global _FRAME_CACHE

        # Include stream in cache key to cache different streams separately
        cache_key = f"{file_path}:{stream}"

        # Check in-memory cache first
        if cache_key in _FRAME_CACHE:
            cached_frames = _FRAME_CACHE[cache_key]
            _update_frame_cache_access(cache_key)
            if len(cached_frames) >= end:
                logger.info(f"Using in-memory cached frames for {file_path} (range {start}-{end})")
                return cached_frames[start:end]
            # If cache has frames covering the requested start, return what we have
            # This handles partial decode scenarios where decode stops early
            # BUT skip this for force_full_extraction (background caching needs ALL frames)
            if len(cached_frames) > start and not force_full_extraction:
                actual_end = min(end, len(cached_frames))
                logger.info(f"Cache partial hit: returning frames {start}-{actual_end} of {len(cached_frames)} cached for {file_path}")
                return cached_frames[start:actual_end]
            # If cache doesn't have enough frames, fall through to persistent cache
            logger.info(f"In-memory cache has {len(cached_frames)} frames, need up to {end}, checking persistent cache")

        # Check persistent cache (include stream in cache key)
        persistent_cache_key = f"{episode_path}:{stream}" if episode_path else None
        if persistent_cache_key:
            persistent_frames = _load_persistent_frame_cache(self.repo_id, persistent_cache_key)
            if persistent_frames:
                # Only cache limited frames in memory to prevent memory explosion
                # Return directly from persistent cache (disk) for frames beyond limit
                frames_to_cache = persistent_frames[:_FRAME_CACHE_MAX_FRAMES_PER_EPISODE]
                _FRAME_CACHE[cache_key] = frames_to_cache
                _update_frame_cache_access(cache_key)
                _evict_frame_cache_if_needed()

                if len(persistent_frames) >= end:
                    logger.info(f"Loaded from persistent cache for {file_path}:{stream} (range {start}-{end})")
                    return persistent_frames[start:end]
                # Return partial frames if they cover the requested start
                # BUT skip this for force_full_extraction (background caching needs ALL frames)
                if len(persistent_frames) > start and not force_full_extraction:
                    actual_end = min(end, len(persistent_frames))
                    logger.info(f"Persistent cache partial hit: returning frames {start}-{actual_end} of {len(persistent_frames)} cached")
                    return persistent_frames[start:actual_end]

        # Define topic patterns for different streams
        if stream == "depth":
            topic_patterns = ["depth", "range", "disparity"]
        else:  # rgb is default
            topic_patterns = ["rgb", "color", "image", "camera", "compressed"]

        try:
            from mcap.reader import make_reader

            # Try to import decoder factories - ROS2 (CDR) and Protobuf
            decoder_factories = []
            try:
                from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory
                decoder_factories.append(Ros2DecoderFactory())
                logger.debug("ROS2 decoder factory available")
            except ImportError:
                logger.debug("mcap-ros2-support not installed, ROS2 MCAP decoding unavailable")

            try:
                from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory
                decoder_factories.append(ProtobufDecoderFactory())
                logger.debug("Protobuf decoder factory available")
            except ImportError:
                logger.debug("mcap-protobuf-support not installed")

            if not decoder_factories:
                raise ImportError("No MCAP decoder factories available. Install mcap-ros2-support or mcap-protobuf-support.")

            # For H.264, we need to decode ALL frames from the beginning
            # because H.264 uses inter-frame compression
            frames = []
            h264_data = []
            timestamps = []
            _frame_counter = 0  # Track raw frame index for stride filtering

            def matches_topic_patterns(topic: str, patterns: List[str]) -> bool:
                """Check if topic matches any of the patterns."""
                topic_lower = topic.lower()
                # For depth, we want to match depth topics but NOT rgb topics
                if stream == "depth":
                    return any(p in topic_lower for p in patterns) and "rgb" not in topic_lower
                # For rgb, we want to match rgb topics but NOT depth topics
                return any(p in topic_lower for p in patterns) and "depth" not in topic_lower

            with open(file_path, "rb") as f:
                reader = make_reader(f, decoder_factories=decoder_factories)
                msg_count = 0

                for schema, channel, message, decoded_message in reader.iter_decoded_messages():
                    msg_count += 1
                    # Check if this topic matches our stream pattern
                    topic_matches = matches_topic_patterns(channel.topic, topic_patterns)
                    if msg_count <= 5:
                        logger.debug(f"Message {msg_count}: topic={channel.topic}, matches={topic_matches}")

                    if topic_matches:
                        if hasattr(decoded_message, 'data') and hasattr(decoded_message, 'format'):
                            img_format = decoded_message.format.lower()
                            img_data = decoded_message.data
                            timestamp = message.log_time / 1e9

                            if len(frames) < 3:
                                logger.info(f"Processing frame {len(frames)}: format='{img_format}', data_len={len(img_data)}")

                            if 'h264' in img_format:
                                # Collect ALL H.264 NAL units (don't stop early)
                                h264_data.append(img_data)
                                timestamps.append(timestamp)
                            elif 'jpeg' in img_format or 'jpg' in img_format or 'png' in img_format:
                                # Apply stride: skip non-stride frames (cheap — no decode)
                                if stride > 1 and _frame_counter % stride != 0:
                                    _frame_counter += 1
                                    continue
                                _frame_counter += 1
                                # Direct decode for JPEG/PNG
                                import cv2
                                nparr = np.frombuffer(img_data, np.uint8)
                                # For depth, use UNCHANGED to preserve 16-bit
                                decode_flag = cv2.IMREAD_UNCHANGED if stream == "depth" else cv2.IMREAD_COLOR
                                image = cv2.imdecode(nparr, decode_flag)
                                if image is not None:
                                    if stream == "depth":
                                        # Colorize depth for visualization
                                        image = colorize_depth(image, depth_colormap)
                                    else:
                                        # RGB: Convert BGR to RGB
                                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    frames.append((len(frames), timestamp, image))
                                    # Early termination — stop when we have enough strided frames
                                    target_count = end if stride <= 1 else math.ceil(end / stride)
                                    if len(frames) >= target_count:
                                        logger.info(f"Reached strided frame limit ({target_count}), stopping extraction")
                                        break
                            elif img_format in ['16uc1', 'mono16', '16uc', 'depth']:
                                # Apply stride for depth too
                                if stride > 1 and _frame_counter % stride != 0:
                                    _frame_counter += 1
                                    continue
                                _frame_counter += 1
                                # Raw 16-bit depth data
                                import cv2
                                depth_array = np.frombuffer(img_data, dtype=np.uint16)
                                # Try to reshape (may need to know dimensions)
                                # Common depth camera resolutions
                                for h, w in [(480, 640), (720, 1280), (240, 320), (424, 512)]:
                                    if depth_array.size == h * w:
                                        depth_array = depth_array.reshape((h, w))
                                        break
                                if len(depth_array.shape) == 2:
                                    image = colorize_depth(depth_array, depth_colormap)
                                    frames.append((len(frames), timestamp, image))
                                    target_count = end if stride <= 1 else math.ceil(end / stride)
                                    if len(frames) >= target_count:
                                        logger.info(f"Reached strided frame limit ({target_count}), stopping extraction")
                                        break

                    # Check if we have enough frames (handles break from inner conditions)
                    if len(frames) >= end:
                        break

            # If we have H.264 data, decode ALL frames from the beginning
            if h264_data and not frames:
                logger.info(f"Decoding {len(h264_data)} H.264 NAL units from MCAP (stream={stream})")
                # Decode all frames (start=0) and cache them
                frames = self._decode_h264_frames(h264_data, timestamps, 0, len(h264_data))

                # For depth stream, colorize all decoded frames
                if stream == "depth" and frames:
                    colorized_frames = []
                    for idx, ts, img in frames:
                        colorized = colorize_depth(img, depth_colormap)
                        colorized_frames.append((idx, ts, colorized))
                    frames = colorized_frames

                # Log decode completion status
                logger.info(f"H.264 decode complete: {len(frames)} frames from {len(h264_data)} NAL units")
                if len(frames) < len(h264_data):
                    logger.warning(f"Incomplete decode: expected ~{len(h264_data)} frames, got {len(frames)} frames")

                # Apply stride to H.264 decoded frames (H.264 must decode all, but only cache strided subset)
                if stride > 1:
                    frames = frames[::stride]
                    logger.info(f"Applied stride={stride} to H.264 frames: {len(frames)} frames after filtering")

                # Cache the decoded frames for future requests (both in-memory and persistent)
                if frames:
                    # Limit in-memory cache to prevent memory explosion
                    # Only cache first N frames in memory; full episode is in persistent cache
                    frames_to_cache = frames[:_FRAME_CACHE_MAX_FRAMES_PER_EPISODE]
                    _FRAME_CACHE[cache_key] = frames_to_cache
                    _update_frame_cache_access(cache_key)
                    _evict_frame_cache_if_needed()
                    logger.info(f"Cached {len(frames_to_cache)} of {len(frames)} frames in memory for {file_path}:{stream}")

            # Return requested range
            if frames:
                return frames[start:end]

            return frames

        except ImportError as e:
            logger.error(f"MCAP library not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to extract frames from MCAP: {e}")
            raise

    def _decode_h264_frames(
        self,
        h264_data: List[bytes],
        timestamps: List[float],
        start: int,
        end: int
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Decode H.264 NAL units to frames using PyAV or ffmpeg.
        """
        import tempfile
        import cv2

        frames = []

        # Write H.264 data to temp file
        with tempfile.NamedTemporaryFile(suffix='.h264', delete=False) as tmp:
            tmp_path = tmp.name
            for data in h264_data:
                tmp.write(data)

        try:
            # Try PyAV first (better H.264 support)
            try:
                import av

                container = av.open(tmp_path)
                stream = container.streams.video[0]
                frame_idx = 0
                decode_errors = 0

                try:
                    for frame in container.decode(stream):
                        if frame_idx >= end:
                            break

                        if frame_idx >= start:
                            # Convert to numpy array (RGB)
                            img = frame.to_ndarray(format='rgb24')
                            timestamp = timestamps[frame_idx] if frame_idx < len(timestamps) else frame_idx / 30.0
                            frames.append((frame_idx, timestamp, img))

                        frame_idx += 1
                except av.AVError as e:
                    decode_errors += 1
                    logger.warning(f"H.264 decode stopped at frame {frame_idx}: {e}")
                except Exception as e:
                    decode_errors += 1
                    logger.warning(f"H.264 decode error at frame {frame_idx}: {e}")

                if decode_errors > 0:
                    logger.info(f"Decode completed with errors: {len(frames)} frames decoded before error")

                container.close()

            except ImportError:
                # Fall back to OpenCV
                logger.warning("PyAV not available, using OpenCV for H.264 decoding")
                cap = cv2.VideoCapture(tmp_path)
                frame_idx = 0

                while frame_idx < end:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx >= start:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        timestamp = timestamps[frame_idx] if frame_idx < len(timestamps) else frame_idx / 30.0
                        frames.append((frame_idx, timestamp, frame_rgb))

                    frame_idx += 1

                cap.release()

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

        return frames

    def _decode_mcap_image(self, data: bytes, schema) -> Optional[np.ndarray]:
        """
        Decode image data from MCAP message.

        Handles various formats:
        - JPEG/PNG compressed images
        - H.264 video frames (individual keyframes)
        - Raw image data
        """
        try:
            # Try decoding as JPEG/PNG first (most common in CompressedImage)
            import cv2

            # Try direct decode
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is not None:
                # Convert BGR to RGB
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # If that fails, try protobuf decode
            try:
                from mcap_protobuf.decoder import DecoderFactory
                # The data might be protobuf-encoded CompressedImage
                # Try to extract the image bytes

                # Simple heuristic: look for JPEG/PNG magic bytes in the data
                jpeg_start = data.find(b'\xff\xd8\xff')
                if jpeg_start >= 0:
                    image_data = data[jpeg_start:]
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is not None:
                        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                png_start = data.find(b'\x89PNG')
                if png_start >= 0:
                    image_data = data[png_start:]
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is not None:
                        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            except Exception:
                pass

            return None

        except Exception as e:
            logger.warning(f"Failed to decode image: {e}")
            return None

    def extract_frames_from_video(
        self,
        file_path: Path,
        start: int = 0,
        end: int = 10,
        stride: int = 1,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract frames from a video file (MP4, etc.).

        Args:
            file_path: Path to the video file
            start: Start frame index
            end: End frame index
            stride: Extract every Nth frame (1 = all frames, 2 = every other frame, etc.)

        Returns:
            List of (frame_idx, timestamp, image_array) tuples
        """
        stride = max(1, stride)
        try:
            import cv2

            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                # Try PyAV as fallback (handles AV1 and other codecs)
                return self._extract_frames_pyav(file_path, start, end, stride)

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frames = []
            frame_idx = 0

            # Seek to start frame
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                frame_idx = start

            while frame_idx < end:
                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_idx - start) % stride == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp = frame_idx / fps
                    frames.append((frame_idx, timestamp, frame_rgb))
                frame_idx += 1

            cap.release()

            # If no frames extracted, video might use a codec OpenCV can't decode (e.g. AV1)
            if not frames and start < end:
                logger.info(f"OpenCV extracted 0 frames, trying PyAV fallback for {file_path}")
                return self._extract_frames_pyav(file_path, start, end, stride)

            return frames

        except ImportError:
            logger.error("opencv-python not installed. Install with: pip install opencv-python")
            raise
        except Exception as e:
            logger.error(f"Failed to extract frames from video: {e}")
            raise

    def _extract_frames_pyav(
        self,
        file_path: Path,
        start: int = 0,
        end: int = 10,
        stride: int = 1,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        PyAV fallback for extracting frames from videos with codecs
        that OpenCV can't handle (e.g., AV1).
        """
        try:
            import av

            container = av.open(str(file_path))
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 30.0

            frames = []
            frame_idx = 0

            for frame in container.decode(stream):
                if frame_idx >= end:
                    break
                if frame_idx >= start and (frame_idx - start) % stride == 0:
                    img = frame.to_ndarray(format="rgb24")
                    timestamp = frame_idx / fps
                    frames.append((frame_idx, timestamp, img))
                frame_idx += 1

            container.close()
            return frames

        except ImportError:
            logger.error("PyAV not installed. Install with: pip install av")
            return []
        except Exception as e:
            logger.error(f"PyAV fallback failed for {file_path}: {e}")
            return []

    def extract_video_from_tar_once(self, file_path: Path) -> Path:
        """
        Extract the video file from a TAR archive to a temp file.

        Returns the path to the extracted video file. Caller is responsible
        for cleaning up the temp file when done.

        This avoids re-extracting the video from the TAR for every batch
        during background caching (significant speedup for large episodes).
        """
        import tarfile

        with tarfile.open(file_path, 'r') as tar:
            video_member = None
            for member in tar.getmembers():
                if member.name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_member = member
                    break

            if video_member is None:
                raise ValueError(f"No video file found in TAR: {file_path}")

            tmp = tempfile.NamedTemporaryFile(
                suffix=Path(video_member.name).suffix, delete=False
            )
            video_file = tar.extractfile(video_member)
            if video_file:
                tmp.write(video_file.read())
            tmp.close()
            return Path(tmp.name)

    def extract_frames_from_tar(
        self,
        file_path: Path,
        start: int = 0,
        end: int = 10,
        stride: int = 1,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract frames from a WebDataset TAR file containing video.

        Args:
            file_path: Path to the TAR file
            start: Start frame index
            end: End frame index
            stride: Extract every Nth frame (1 = all frames)

        Returns:
            List of (frame_idx, timestamp, image_array) tuples
        """
        import tarfile
        import tempfile

        try:
            with tarfile.open(file_path, 'r') as tar:
                # Find video file in TAR
                video_member = None
                for member in tar.getmembers():
                    if member.name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        video_member = member
                        break

                if video_member is None:
                    raise ValueError(f"No video file found in TAR: {file_path}")

                # Extract video to temp file
                with tempfile.NamedTemporaryFile(suffix=Path(video_member.name).suffix, delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                    # Extract video
                    video_file = tar.extractfile(video_member)
                    if video_file:
                        tmp.write(video_file.read())

                try:
                    # Extract frames from video
                    return self.extract_frames_from_video(tmp_path, start, end, stride)
                finally:
                    # Clean up temp file
                    tmp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Failed to extract frames from TAR: {e}")
            raise

    def extract_frames(
        self,
        episode_path: str,
        start: int = 0,
        end: int = 10,
        stream: str = "rgb",
        depth_colormap: str = "viridis",
        force_full_extraction: bool = False,
        stride: int = 1,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract frames from an episode file (auto-detects format).

        Uses persistent caching to avoid re-downloading and re-decoding
        on subsequent requests.

        Args:
            episode_path: Path within the HuggingFace repo
            start: Start frame index
            end: End frame index
            stream: Which stream to extract: "rgb" or "depth"
            depth_colormap: Colormap for depth visualization
            force_full_extraction: If True, bypass partial cache returns (for background caching)
            stride: Extract every Nth frame (1 = all frames)

        Returns:
            List of (frame_idx, timestamp, image_array) tuples
        """
        # Download the file
        local_path = self.download_file(episode_path)

        # Detect format and extract
        suffix = local_path.suffix.lower()

        if suffix == '.mcap':
            return self.extract_frames_from_mcap(
                local_path, start, end,
                episode_path=episode_path,
                stream=stream,
                depth_colormap=depth_colormap,
                force_full_extraction=force_full_extraction,
                stride=stride,
            )
        elif suffix == '.tar':
            return self.extract_frames_from_tar(local_path, start, end, stride)
        elif suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return self.extract_frames_from_video(local_path, start, end, stride)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def extract_imu_data(
        self,
        episode_path: str,
    ) -> Dict[str, Any]:
        """
        Extract IMU data from an MCAP episode file.

        Args:
            episode_path: Path within the HuggingFace repo

        Returns:
            Dictionary with IMU data:
            {
                "timestamps": List[float],
                "accel_x": List[float],
                "accel_y": List[float],
                "accel_z": List[float],
                "gyro_x": List[float],
                "gyro_y": List[float],
                "gyro_z": List[float]
            }
        """
        local_path = self.download_file(episode_path)
        suffix = local_path.suffix.lower()

        if suffix != '.mcap':
            return {"error": f"IMU extraction only supported for MCAP files, got {suffix}"}

        try:
            from mcap.reader import make_reader
            from mcap_protobuf.decoder import DecoderFactory

            imu_data = {
                "timestamps": [],
                "accel_x": [],
                "accel_y": [],
                "accel_z": [],
                "gyro_x": [],
                "gyro_y": [],
                "gyro_z": [],
            }

            with open(local_path, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])

                for schema, channel, message, decoded_message in reader.iter_decoded_messages():
                    topic_lower = channel.topic.lower()
                    if any(p in topic_lower for p in self._IMU_TOPIC_PATTERNS):
                        timestamp = message.log_time / 1e9

                        # Try to extract IMU data from decoded message
                        # ROS sensor_msgs/Imu format
                        if hasattr(decoded_message, 'linear_acceleration'):
                            accel = decoded_message.linear_acceleration
                            imu_data["timestamps"].append(timestamp)
                            imu_data["accel_x"].append(getattr(accel, 'x', 0.0))
                            imu_data["accel_y"].append(getattr(accel, 'y', 0.0))
                            imu_data["accel_z"].append(getattr(accel, 'z', 0.0))

                            if hasattr(decoded_message, 'angular_velocity'):
                                gyro = decoded_message.angular_velocity
                                imu_data["gyro_x"].append(getattr(gyro, 'x', 0.0))
                                imu_data["gyro_y"].append(getattr(gyro, 'y', 0.0))
                                imu_data["gyro_z"].append(getattr(gyro, 'z', 0.0))
                            else:
                                imu_data["gyro_x"].append(0.0)
                                imu_data["gyro_y"].append(0.0)
                                imu_data["gyro_z"].append(0.0)

                        # Foxglove IMUMeasurement format (used by RealOmni)
                        elif hasattr(decoded_message, 'linear_acceleration_x'):
                            imu_data["timestamps"].append(timestamp)
                            imu_data["accel_x"].append(getattr(decoded_message, 'linear_acceleration_x', 0.0))
                            imu_data["accel_y"].append(getattr(decoded_message, 'linear_acceleration_y', 0.0))
                            imu_data["accel_z"].append(getattr(decoded_message, 'linear_acceleration_z', 0.0))
                            imu_data["gyro_x"].append(getattr(decoded_message, 'angular_velocity_x', 0.0))
                            imu_data["gyro_y"].append(getattr(decoded_message, 'angular_velocity_y', 0.0))
                            imu_data["gyro_z"].append(getattr(decoded_message, 'angular_velocity_z', 0.0))

            logger.info(f"Extracted {len(imu_data['timestamps'])} IMU samples from {episode_path}")
            return imu_data

        except ImportError as e:
            logger.error(f"MCAP library not installed: {e}")
            return {"error": "MCAP library not installed"}
        except Exception as e:
            logger.error(f"Failed to extract IMU data: {e}")
            return {"error": str(e)}

    def extract_actions_data(
        self,
        episode_path: str,
    ) -> Dict[str, Any]:
        """
        Extract action data from an MCAP episode file.

        Args:
            episode_path: Path within the HuggingFace repo

        Returns:
            Dictionary with action data:
            {
                "timestamps": List[float],
                "actions": List[List[float]],  # 2D array: [frame][dimension]
                "dimension_labels": List[str],  # Optional dimension labels
                "error": Optional[str]
            }
        """
        local_path = self.download_file(episode_path)
        suffix = local_path.suffix.lower()

        if suffix != '.mcap':
            return {"error": f"Action extraction only supported for MCAP files, got {suffix}"}

        try:
            from mcap.reader import make_reader

            # Try to import decoder factories
            decoder_factories = []
            try:
                from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory
                decoder_factories.append(Ros2DecoderFactory())
            except ImportError:
                pass
            try:
                from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory
                decoder_factories.append(ProtobufDecoderFactory())
            except ImportError:
                pass

            if not decoder_factories:
                return {"error": "No MCAP decoder factories available"}

            actions_data = {
                "timestamps": [],
                "actions": [],
                "dimension_labels": None,
            }
            detected_msg_type = None

            with open(local_path, "rb") as f:
                reader = make_reader(f, decoder_factories=decoder_factories)

                for schema, channel, message, decoded_message in reader.iter_decoded_messages():
                    topic_lower = channel.topic.lower()

                    # Skip camera/image topics
                    if any(p in topic_lower for p in self._SKIP_TOPIC_PATTERNS):
                        continue

                    # Check if this looks like an action topic
                    if any(p in topic_lower for p in self._ACTION_TOPIC_PATTERNS):
                        timestamp = message.log_time / 1e9

                        # Try to extract action data from decoded message
                        action_vector, msg_type = self._extract_action_vector(decoded_message)

                        if action_vector is not None:
                            actions_data["timestamps"].append(timestamp)
                            actions_data["actions"].append(action_vector)
                            if detected_msg_type is None:
                                detected_msg_type = msg_type

            # Infer dimension labels based on detected message type
            if actions_data["actions"]:
                num_dims = len(actions_data["actions"][0])
                if detected_msg_type == "pose":
                    # PoseInFrame / Pose: position + quaternion (no gripper)
                    if num_dims == 7:
                        actions_data["dimension_labels"] = ["x", "y", "z", "qx", "qy", "qz", "qw"]
                    elif num_dims == 6:
                        actions_data["dimension_labels"] = ["x", "y", "z", "qx", "qy", "qz"]
                elif num_dims == 7:
                    actions_data["dimension_labels"] = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
                elif num_dims == 6:
                    actions_data["dimension_labels"] = ["x", "y", "z", "rx", "ry", "rz"]
                elif num_dims == 3:
                    actions_data["dimension_labels"] = ["x", "y", "z"]

            logger.info(f"Extracted {len(actions_data['timestamps'])} action samples from {episode_path}")
            return actions_data

        except ImportError as e:
            logger.error(f"MCAP library not installed: {e}")
            return {"error": "MCAP library not installed"}
        except Exception as e:
            logger.error(f"Failed to extract actions data: {e}")
            return {"error": str(e)}

    def _extract_action_vector(self, decoded_message) -> tuple:
        """
        Extract action vector from a decoded MCAP message.

        Handles various ROS and Foxglove message types that might contain action data.

        Returns:
            Tuple of (action_vector, msg_type) where msg_type is one of:
            "array", "joint", "twist", "pose", or None if extraction fails.
        """
        # Try common message attribute patterns

        # Float64MultiArray / Float32MultiArray
        if hasattr(decoded_message, 'data'):
            data = decoded_message.data
            if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                try:
                    return [float(x) for x in data], "array"
                except (TypeError, ValueError):
                    pass

        # JointState message
        if hasattr(decoded_message, 'position') and not hasattr(decoded_message, 'orientation'):
            positions = decoded_message.position
            if hasattr(positions, '__iter__'):
                try:
                    return [float(x) for x in positions], "joint"
                except (TypeError, ValueError):
                    pass

        # Twist message (linear + angular velocity)
        if hasattr(decoded_message, 'linear') and hasattr(decoded_message, 'angular'):
            try:
                linear = decoded_message.linear
                angular = decoded_message.angular
                return [
                    getattr(linear, 'x', 0.0),
                    getattr(linear, 'y', 0.0),
                    getattr(linear, 'z', 0.0),
                    getattr(angular, 'x', 0.0),
                    getattr(angular, 'y', 0.0),
                    getattr(angular, 'z', 0.0),
                ], "twist"
            except (TypeError, ValueError):
                pass

        # Foxglove PoseInFrame message (used by RealOmni)
        if hasattr(decoded_message, 'pose'):
            try:
                pose = decoded_message.pose
                pos = getattr(pose, 'position', None)
                ori = getattr(pose, 'orientation', None)
                if pos is not None and ori is not None:
                    return [
                        getattr(pos, 'x', 0.0),
                        getattr(pos, 'y', 0.0),
                        getattr(pos, 'z', 0.0),
                        getattr(ori, 'x', 0.0),
                        getattr(ori, 'y', 0.0),
                        getattr(ori, 'z', 0.0),
                        getattr(ori, 'w', 1.0),
                    ], "pose"
            except (TypeError, ValueError):
                pass

        # ROS Pose message (position + orientation directly)
        if hasattr(decoded_message, 'position') and hasattr(decoded_message, 'orientation'):
            try:
                pos = decoded_message.position
                ori = decoded_message.orientation
                return [
                    getattr(pos, 'x', 0.0),
                    getattr(pos, 'y', 0.0),
                    getattr(pos, 'z', 0.0),
                    getattr(ori, 'x', 0.0),
                    getattr(ori, 'y', 0.0),
                    getattr(ori, 'z', 0.0),
                    getattr(ori, 'w', 1.0),
                ], "pose"
            except (TypeError, ValueError):
                pass

        return None, None

    def _extract_imu_sample(
        self, decoded_message, timestamp: float, imu_data: Dict[str, List]
    ) -> bool:
        """
        Extract a single IMU sample from a decoded MCAP message into imu_data.

        Handles both ROS sensor_msgs/Imu and Foxglove IMUMeasurement formats.

        Returns:
            True if a sample was extracted, False otherwise.
        """
        # ROS sensor_msgs/Imu format
        if hasattr(decoded_message, 'linear_acceleration'):
            accel = decoded_message.linear_acceleration
            imu_data["timestamps"].append(timestamp)
            imu_data["accel_x"].append(getattr(accel, 'x', 0.0))
            imu_data["accel_y"].append(getattr(accel, 'y', 0.0))
            imu_data["accel_z"].append(getattr(accel, 'z', 0.0))

            if hasattr(decoded_message, 'angular_velocity'):
                gyro = decoded_message.angular_velocity
                imu_data["gyro_x"].append(getattr(gyro, 'x', 0.0))
                imu_data["gyro_y"].append(getattr(gyro, 'y', 0.0))
                imu_data["gyro_z"].append(getattr(gyro, 'z', 0.0))
            else:
                imu_data["gyro_x"].append(0.0)
                imu_data["gyro_y"].append(0.0)
                imu_data["gyro_z"].append(0.0)
            return True

        # Foxglove IMUMeasurement format (used by RealOmni)
        elif hasattr(decoded_message, 'linear_acceleration_x'):
            imu_data["timestamps"].append(timestamp)
            imu_data["accel_x"].append(getattr(decoded_message, 'linear_acceleration_x', 0.0))
            imu_data["accel_y"].append(getattr(decoded_message, 'linear_acceleration_y', 0.0))
            imu_data["accel_z"].append(getattr(decoded_message, 'linear_acceleration_z', 0.0))
            imu_data["gyro_x"].append(getattr(decoded_message, 'angular_velocity_x', 0.0))
            imu_data["gyro_y"].append(getattr(decoded_message, 'angular_velocity_y', 0.0))
            imu_data["gyro_z"].append(getattr(decoded_message, 'angular_velocity_z', 0.0))
            return True

        return False

    def _count_signal_messages(self, local_path: Path) -> Tuple[int, int, list]:
        """
        Cheap counting pass over MCAP messages (no decode) to determine
        action and IMU message counts for stride calculation.

        Returns:
            Tuple of (action_count, imu_count, signal_topics)
            where signal_topics is the list of topic names matching action/IMU patterns.
        """
        from mcap.reader import make_reader

        action_count = 0
        imu_count = 0
        signal_topics: set = set()

        with open(local_path, "rb") as f:
            reader = make_reader(f)
            for _, channel, _, in reader.iter_messages():
                topic_lower = channel.topic.lower()
                if any(p in topic_lower for p in self._SKIP_TOPIC_PATTERNS):
                    continue
                if any(p in topic_lower for p in self._ACTION_TOPIC_PATTERNS):
                    action_count += 1
                    signal_topics.add(channel.topic)
                elif any(p in topic_lower for p in self._IMU_TOPIC_PATTERNS):
                    imu_count += 1
                    signal_topics.add(channel.topic)

        return action_count, imu_count, list(signal_topics)

    def _get_signal_counts_from_summary(
        self, stream
    ) -> Optional[Tuple[int, int, List[str]]]:
        """
        Read action/IMU message counts from MCAP footer summary statistics.

        Uses summary.statistics.channel_message_counts (keyed by channel_id)
        and summary.channels for topic-name mapping. No message iteration.

        Args:
            stream: An open seekable file-like object (local file or HfFileSystem).
                    Must support seek(). The stream is consumed by make_reader.

        Returns:
            (action_count, imu_count, signal_topics) or None if summary unavailable.
        """
        try:
            from mcap.reader import make_reader

            reader = make_reader(stream)
            summary = reader.get_summary()
            if summary is None:
                return None

            statistics = summary.statistics
            channels = summary.channels  # {channel_id: Channel}

            if statistics is None or channels is None:
                return None

            # channel_message_counts: {channel_id: count}
            counts_by_id = statistics.channel_message_counts or {}

            # Build topic → count mapping
            action_count = 0
            imu_count = 0
            signal_topics: list = []

            for ch_id, ch in channels.items():
                topic_lower = ch.topic.lower()
                count = counts_by_id.get(ch_id, 0)

                if any(p in topic_lower for p in self._SKIP_TOPIC_PATTERNS):
                    continue
                if any(p in topic_lower for p in self._ACTION_TOPIC_PATTERNS):
                    action_count += count
                    signal_topics.append(ch.topic)
                elif any(p in topic_lower for p in self._IMU_TOPIC_PATTERNS):
                    imu_count += count
                    signal_topics.append(ch.topic)

            return action_count, imu_count, signal_topics
        except Exception as e:
            logger.warning(f"Failed to read MCAP summary: {e}")
            return None

    def extract_signals_data(
        self,
        episode_path: str,
        max_actions: int = 500,
        max_imu: int = 500,
    ) -> Dict[str, Any]:
        """
        Extract both action and IMU signal data from an MCAP episode in a
        single stride-aware pass.

        Pass 1: Count messages (no decode) to compute per-signal strides.
        Pass 2: Decode only the messages that survive the stride filter.

        Args:
            episode_path: Path within the HuggingFace repo
            max_actions: Target maximum number of action samples
            max_imu: Target maximum number of IMU samples

        Returns:
            {
                "actions": {timestamps, actions, dimension_labels},
                "imu": {timestamps, accel_x/y/z, gyro_x/y/z},
                "action_stride": int,
                "imu_stride": int,
            }
        """
        local_path = self.download_file(episode_path)
        suffix = local_path.suffix.lower()

        if suffix != '.mcap':
            return {
                "actions": {"error": f"Signal extraction only supported for MCAP files, got {suffix}"},
                "imu": {"error": f"Signal extraction only supported for MCAP files, got {suffix}"},
                "action_stride": 1,
                "imu_stride": 1,
            }

        try:
            from mcap.reader import make_reader

            decoder_factories = self._get_decoder_factories()

            # Pass 1: count messages to compute strides + collect signal topic names
            # Try summary-based counting first (reads only footer, no iteration)
            summary_result = None
            with open(local_path, "rb") as f:
                summary_result = self._get_signal_counts_from_summary(f)

            if summary_result is not None:
                action_count, imu_count, signal_topics = summary_result
            else:
                action_count, imu_count, signal_topics = self._count_signal_messages(local_path)

            action_stride = max(1, math.ceil(action_count / max_actions)) if action_count > max_actions else 1
            imu_stride = max(1, math.ceil(imu_count / max_imu)) if imu_count > max_imu else 1

            logger.info(
                f"Signal extraction: {action_count} action msgs (stride={action_stride}), "
                f"{imu_count} IMU msgs (stride={imu_stride}), "
                f"filtering to topics: {signal_topics}"
            )

            # Pass 2: single decoded loop with stride filtering
            actions_data = {
                "timestamps": [],
                "actions": [],
                "dimension_labels": None,
            }
            imu_data = {
                "timestamps": [],
                "accel_x": [],
                "accel_y": [],
                "accel_z": [],
                "gyro_x": [],
                "gyro_y": [],
                "gyro_z": [],
            }

            action_idx = 0
            imu_idx = 0

            with open(local_path, "rb") as f:
                reader = make_reader(f, decoder_factories=decoder_factories)

                # Filter to only action/IMU topics — skips decoding video frames entirely
                for schema, channel, message, decoded_message in reader.iter_decoded_messages(
                    topics=signal_topics if signal_topics else None
                ):
                    topic_lower = channel.topic.lower()

                    # Action topics
                    if any(p in topic_lower for p in self._ACTION_TOPIC_PATTERNS):
                        if action_idx % action_stride == 0:
                            timestamp = message.log_time / 1e9
                            action_result = self._extract_action_vector(decoded_message)
                            if action_result is not None:
                                values, _msg_type = action_result
                                if values is not None:
                                    actions_data["timestamps"].append(timestamp)
                                    actions_data["actions"].append(values)
                        action_idx += 1

                    # IMU topics
                    elif any(p in topic_lower for p in self._IMU_TOPIC_PATTERNS):
                        if imu_idx % imu_stride == 0:
                            timestamp = message.log_time / 1e9
                            self._extract_imu_sample(decoded_message, timestamp, imu_data)
                        imu_idx += 1

            # Infer dimension labels for actions
            if actions_data["actions"]:
                num_dims = len(actions_data["actions"][0])
                if num_dims == 7:
                    actions_data["dimension_labels"] = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
                elif num_dims == 6:
                    actions_data["dimension_labels"] = ["x", "y", "z", "rx", "ry", "rz"]
                elif num_dims == 3:
                    actions_data["dimension_labels"] = ["x", "y", "z"]

            logger.info(
                f"Extracted {len(actions_data['timestamps'])} action samples, "
                f"{len(imu_data['timestamps'])} IMU samples from {episode_path}"
            )

            return {
                "actions": actions_data,
                "imu": imu_data,
                "action_stride": action_stride,
                "imu_stride": imu_stride,
                "raw_action_count": action_count,
                "raw_imu_count": imu_count,
            }

        except ImportError as e:
            logger.error(f"MCAP library not installed: {e}")
            return {
                "actions": {"error": "MCAP library not installed"},
                "imu": {"error": "MCAP library not installed"},
                "action_stride": 1,
                "imu_stride": 1,
            }
        except Exception as e:
            logger.error(f"Failed to extract signals data: {e}")
            return {
                "actions": {"error": str(e)},
                "imu": {"error": str(e)},
                "action_stride": 1,
                "imu_stride": 1,
            }

    def _get_decoder_factories(self) -> list:
        """
        Import and return all available MCAP decoder factories.

        Returns list of decoder factory instances (Protobuf, ROS2, etc.).
        Cached on first call to avoid repeated imports.
        Raises ImportError if no factories are available.
        """
        if self._cached_decoder_factories is not None:
            return self._cached_decoder_factories

        decoder_factories = []
        try:
            from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory
            decoder_factories.append(Ros2DecoderFactory())
        except ImportError:
            pass
        try:
            from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory
            decoder_factories.append(ProtobufDecoderFactory())
        except ImportError:
            pass

        if not decoder_factories:
            raise ImportError(
                "No MCAP decoder factories available. "
                "Install mcap-ros2-support or mcap-protobuf-support."
            )
        self._cached_decoder_factories = decoder_factories
        return decoder_factories

    def extract_signals_remote(
        self,
        episode_path: str,
        max_actions: int = 500,
        max_imu: int = 500,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """
        Extract action and IMU signals from a remote MCAP file via HTTP range requests.

        Uses HfFileSystem for seekable HTTP access. The mcap SeekingReader reads
        only the footer (~8KB) to get summary statistics and then fetches only
        signal-bearing chunks (~1-3MB), skipping all video data (~35-72MB).

        Falls back to full download + extract_signals_data() on any failure.

        Args:
            episode_path: Path within the HuggingFace repo
            max_actions: Target maximum number of action samples
            max_imu: Target maximum number of IMU samples
            cancel_event: Optional threading.Event; when set, extraction aborts early.

        Returns:
            Same format as extract_signals_data():
            {
                "actions": {timestamps, actions, dimension_labels},
                "imu": {timestamps, accel_x/y/z, gyro_x/y/z},
                "action_stride": int,
                "imu_stride": int,
            }

        Raises:
            concurrent.futures.CancelledError: If cancel_event is set during extraction.
        """
        import concurrent.futures
        import time

        def _check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise concurrent.futures.CancelledError(
                    f"Extraction cancelled for {episode_path}"
                )

        try:
            from mcap.reader import make_reader

            decoder_factories = self._get_decoder_factories()

            t0 = time.time()
            _check_cancelled()

            # Step 1: Read summary from footer (1-2 HTTP range requests, ~8KB)
            remote_file = self._open_hf_remote(episode_path)
            reader = make_reader(remote_file, decoder_factories=decoder_factories)
            summary = reader.get_summary()
            remote_file.close()

            _check_cancelled()

            if summary is None:
                logger.info(f"No MCAP summary for {episode_path}, falling back to full download")
                return self.extract_signals_data(episode_path, max_actions, max_imu)

            # Step 2: Extract counts and signal topics directly from summary
            statistics = summary.statistics
            channels = summary.channels
            if statistics is None or channels is None:
                logger.info(f"No statistics/channels in summary for {episode_path}, falling back")
                return self.extract_signals_data(episode_path, max_actions, max_imu)

            counts_by_id = statistics.channel_message_counts or {}
            action_count = 0
            imu_count = 0
            signal_topics: list = []

            for ch_id, ch in channels.items():
                topic_lower = ch.topic.lower()
                count = counts_by_id.get(ch_id, 0)
                if any(p in topic_lower for p in self._SKIP_TOPIC_PATTERNS):
                    continue
                if any(p in topic_lower for p in self._ACTION_TOPIC_PATTERNS):
                    action_count += count
                    signal_topics.append(ch.topic)
                elif any(p in topic_lower for p in self._IMU_TOPIC_PATTERNS):
                    imu_count += count
                    signal_topics.append(ch.topic)

            # Re-open for message iteration (separate connection avoids seek issues)
            remote_file = self._open_hf_remote(episode_path)
            reader = make_reader(remote_file, decoder_factories=decoder_factories)

            t_summary = time.time()
            logger.info(
                f"Remote summary for {episode_path}: "
                f"{action_count} actions, {imu_count} IMU, "
                f"topics={signal_topics} ({t_summary - t0:.1f}s)"
            )

            # Step 3: Compute strides
            action_stride = max(1, math.ceil(action_count / max_actions)) if action_count > max_actions else 1
            imu_stride = max(1, math.ceil(imu_count / max_imu)) if imu_count > max_imu else 1

            # Step 4: Iterate decoded messages for signal topics only
            actions_data = {
                "timestamps": [],
                "actions": [],
                "dimension_labels": None,
            }
            imu_data = {
                "timestamps": [],
                "accel_x": [], "accel_y": [], "accel_z": [],
                "gyro_x": [], "gyro_y": [], "gyro_z": [],
            }

            action_idx = 0
            imu_idx = 0
            msg_count = 0

            for schema, channel, message, decoded_message in reader.iter_decoded_messages(
                topics=signal_topics if signal_topics else None
            ):
                # Check cancellation every 100 messages to avoid overhead
                msg_count += 1
                if msg_count % 100 == 0:
                    _check_cancelled()

                topic_lower = channel.topic.lower()

                if any(p in topic_lower for p in self._ACTION_TOPIC_PATTERNS):
                    if action_idx % action_stride == 0:
                        timestamp = message.log_time / 1e9
                        action_result = self._extract_action_vector(decoded_message)
                        if action_result is not None:
                            values, _msg_type = action_result
                            if values is not None:
                                actions_data["timestamps"].append(timestamp)
                                actions_data["actions"].append(values)
                    action_idx += 1

                elif any(p in topic_lower for p in self._IMU_TOPIC_PATTERNS):
                    if imu_idx % imu_stride == 0:
                        timestamp = message.log_time / 1e9
                        self._extract_imu_sample(decoded_message, timestamp, imu_data)
                    imu_idx += 1

            remote_file.close()

            # Infer dimension labels
            if actions_data["actions"]:
                num_dims = len(actions_data["actions"][0])
                if num_dims == 7:
                    actions_data["dimension_labels"] = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
                elif num_dims == 6:
                    actions_data["dimension_labels"] = ["x", "y", "z", "rx", "ry", "rz"]
                elif num_dims == 3:
                    actions_data["dimension_labels"] = ["x", "y", "z"]

            t_done = time.time()
            logger.info(
                f"Remote signal extraction complete for {episode_path}: "
                f"{len(actions_data['timestamps'])} actions, "
                f"{len(imu_data['timestamps'])} IMU samples "
                f"({t_done - t0:.1f}s total, {t_summary - t0:.1f}s summary)"
            )

            return {
                "actions": actions_data,
                "imu": imu_data,
                "action_stride": action_stride,
                "imu_stride": imu_stride,
                "raw_action_count": action_count,
                "raw_imu_count": imu_count,
            }

        except concurrent.futures.CancelledError:
            logger.info(f"Remote signal extraction cancelled for {episode_path}")
            raise
        except Exception as e:
            logger.warning(
                f"Remote signal extraction failed for {episode_path}: {e}, "
                f"falling back to full download"
            )
            _check_cancelled()  # Don't fall back if we're cancelled
            return self.extract_signals_data(episode_path, max_actions, max_imu)

    def extract_first_frame_remote(
        self,
        episode_path: str,
        thumb_width: int = 160,
        thumb_height: int = 120,
        cancel_event: Optional[threading.Event] = None,
    ) -> Optional[str]:
        """
        Extract the first video frame from a remote MCAP file as a base64 JPEG thumbnail.

        Uses HTTP range requests to read only the MCAP footer (~8KB) for metadata,
        then fetches only the first video message (~1-2KB for JPEG, or ~50-100KB for
        H.264 keyframe). This avoids downloading the entire 40-80MB file.

        Args:
            episode_path: Path within the HuggingFace repo
            thumb_width: Thumbnail width in pixels
            thumb_height: Thumbnail height in pixels
            cancel_event: Optional threading.Event; when set, extraction aborts early.

        Returns:
            Base64-encoded JPEG string, or None on failure.
        """
        import base64
        import concurrent.futures
        import time

        def _check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise concurrent.futures.CancelledError(
                    f"First frame extraction cancelled for {episode_path}"
                )

        try:
            import cv2
            from mcap.reader import make_reader

            decoder_factories = self._get_decoder_factories()
            t0 = time.time()
            _check_cancelled()

            # Step 1: Read summary to find video/image topics
            remote_file = self._open_hf_remote(episode_path)
            reader = make_reader(remote_file, decoder_factories=decoder_factories)
            summary = reader.get_summary()
            remote_file.close()

            _check_cancelled()

            if summary is None or summary.channels is None:
                logger.warning(f"No MCAP summary for first frame: {episode_path}")
                return None

            # Video topics are the ones matching _SKIP_TOPIC_PATTERNS (camera/image/rgb/depth)
            video_topics = []
            for _ch_id, ch in summary.channels.items():
                topic_lower = ch.topic.lower()
                if any(p in topic_lower for p in self._SKIP_TOPIC_PATTERNS):
                    # Prefer RGB/camera over depth
                    if "depth" not in topic_lower:
                        video_topics.insert(0, ch.topic)
                    else:
                        video_topics.append(ch.topic)

            if not video_topics:
                logger.warning(f"No video topics found in {episode_path}")
                return None

            target_topic = video_topics[0]

            # Step 2: Re-open and iterate to find the first message on the video topic
            _check_cancelled()
            remote_file = self._open_hf_remote(episode_path)
            reader = make_reader(remote_file, decoder_factories=decoder_factories)

            frame_img = None
            for _schema, channel, _message, decoded_message in reader.iter_decoded_messages(
                topics=[target_topic]
            ):
                _check_cancelled()
                # Try to extract image data from the decoded message
                frame_img = self._decode_first_frame_message(decoded_message)
                break  # Only need the first message

            remote_file.close()

            if frame_img is None:
                logger.warning(f"Could not decode first frame from {episode_path}")
                return None

            # Step 3: Resize to thumbnail and encode as JPEG base64
            thumb = cv2.resize(frame_img, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
            # Convert RGB to BGR for cv2 encoding
            if len(thumb.shape) == 3 and thumb.shape[2] == 3:
                thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
            else:
                thumb_bgr = thumb
            _, jpeg_buf = cv2.imencode(".jpg", thumb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

            t_done = time.time()
            logger.info(
                f"First frame extracted from {episode_path} "
                f"({thumb_width}x{thumb_height}, {len(b64)} bytes b64, {t_done - t0:.1f}s)"
            )
            return b64

        except concurrent.futures.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"First frame extraction failed for {episode_path}: {e}")
            return None

    def _decode_first_frame_message(self, decoded_message) -> Optional[np.ndarray]:
        """
        Decode a single MCAP video/image message into an RGB numpy array.

        Handles:
        - ROS CompressedImage (JPEG/PNG data field)
        - ROS Image (raw pixel data)
        - Protobuf messages with image data fields
        - Raw H.264 NAL units
        """
        try:
            import cv2

            # Try CompressedImage-style (has .data and .format)
            raw_data = None
            if hasattr(decoded_message, "data"):
                raw_data = bytes(decoded_message.data)
            elif isinstance(decoded_message, dict) and "data" in decoded_message:
                raw_data = bytes(decoded_message["data"])

            if raw_data is None or len(raw_data) == 0:
                return None

            # Check if it's JPEG or PNG by magic bytes
            is_jpeg = raw_data[:2] == b"\xff\xd8"
            is_png = raw_data[:4] == b"\x89PNG"

            if is_jpeg or is_png:
                arr = np.frombuffer(raw_data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Try H.264 decode (NAL unit)
            if raw_data[:4] == b"\x00\x00\x00\x01" or raw_data[:3] == b"\x00\x00\x01":
                return self._decode_single_h264_nal(raw_data)

            # Try raw Image (has width/height/encoding fields)
            width = getattr(decoded_message, "width", None)
            height = getattr(decoded_message, "height", None)
            encoding = getattr(decoded_message, "encoding", "")

            if width and height and width > 0 and height > 0:
                if encoding in ("rgb8", "8UC3"):
                    img = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 3)
                    return img.copy()
                elif encoding in ("bgr8",):
                    img = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 3)
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif encoding in ("mono8", "8UC1"):
                    img = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width)
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            return None
        except Exception as e:
            logger.warning(f"Failed to decode frame message: {e}")
            return None

    def _decode_single_h264_nal(self, nal_data: bytes) -> Optional[np.ndarray]:
        """Decode a single H.264 NAL unit to an RGB frame using PyAV."""
        try:
            import av
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".h264", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(nal_data)

            try:
                container = av.open(tmp_path)
                for frame in container.decode(container.streams.video[0]):
                    img = frame.to_ndarray(format="rgb24")
                    container.close()
                    return img
                container.close()
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"H.264 single-NAL decode failed: {e}")
        return None

    def get_frame_count(self, episode_path: str) -> int:
        """
        Get the number of frames in an episode.

        For large files, this may require downloading.
        """
        global _FRAME_CACHE

        local_path = self.download_file(episode_path)
        suffix = local_path.suffix.lower()
        cache_key = str(local_path)

        # Check cache first for MCAP files
        if suffix == '.mcap' and cache_key in _FRAME_CACHE:
            return len(_FRAME_CACHE[cache_key])

        if suffix == '.mcap':
            # Count messages in MCAP
            try:
                from mcap.reader import make_reader

                count = 0
                with open(local_path, "rb") as f:
                    reader = make_reader(f)
                    for schema, channel, message in reader.iter_messages():
                        if "camera" in channel.topic.lower():
                            count += 1
                return count
            except Exception:
                return 0

        elif suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.tar']:
            # For video, get frame count
            try:
                import cv2

                if suffix == '.tar':
                    # Extract video from TAR first
                    import tarfile
                    with tarfile.open(local_path, 'r') as tar:
                        for member in tar.getmembers():
                            if member.name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                                with tempfile.NamedTemporaryFile(suffix=Path(member.name).suffix, delete=False) as tmp:
                                    video_file = tar.extractfile(member)
                                    if video_file:
                                        tmp.write(video_file.read())
                                        tmp_path = tmp.name

                                cap = cv2.VideoCapture(tmp_path)
                                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                cap.release()
                                Path(tmp_path).unlink(missing_ok=True)
                                return count
                else:
                    cap = cv2.VideoCapture(str(local_path))
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    return count
            except Exception:
                return 0

        return 0

    def extract_frames_with_count(
        self,
        episode_path: str,
        start: int = 0,
        end: int = 10,
        stream: str = "rgb",
        depth_colormap: str = "viridis",
        revision: Optional[str] = None,
        auto_stride_target: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, float, np.ndarray]], int, int]:
        """
        Extract frames from an episode file and return total frame count.

        Uses persistent caching to avoid re-downloading and re-decoding
        on subsequent requests.

        Args:
            episode_path: Path within the HuggingFace repo
            start: Start frame index
            end: End frame index
            stream: Which stream to extract: "rgb" or "depth"
            depth_colormap: Colormap for depth visualization
            revision: Branch/tag to download from (e.g., "v2.0")
            auto_stride_target: If set, automatically compute stride to
                keep frame count near this target. Stride is passed to
                the video decoder so non-strided frames are never decoded.

        Returns:
            Tuple of (frames_list, total_frame_count, stride_used)
        """
        # Download the file first
        local_path = self.download_file(episode_path, revision=revision)
        suffix = local_path.suffix.lower()

        # Get total frame count
        total_frames = self.get_frame_count(episode_path)

        # Compute stride if auto_stride_target is set
        stride = 1
        if auto_stride_target and total_frames > auto_stride_target:
            stride = math.ceil(total_frames / auto_stride_target)

        # Extract frames (with episode_path for persistent caching)
        if suffix == '.mcap':
            frames = self.extract_frames_from_mcap(
                local_path, start, end,
                episode_path=episode_path,
                stream=stream,
                depth_colormap=depth_colormap,
                stride=stride,
            )
            # Stride applied during extraction — no post-hoc filtering needed
        elif suffix == '.tar':
            frames = self.extract_frames_from_tar(local_path, start, end, stride)
        elif suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            frames = self.extract_frames_from_video(local_path, start, end, stride)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return frames, total_frames, stride
