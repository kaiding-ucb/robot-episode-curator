"""
Streaming Frame Extractor for HuggingFace datasets.

Extracts frames from streaming datasets (MCAP, WebDataset/TAR, video files)
by downloading individual episode files on demand.

Includes persistent caching to avoid re-downloading and re-decoding on subsequent requests.
"""
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Cache directory for downloaded files
CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", Path.home() / ".cache" / "data_viewer" / "streaming"))

# Persistent frame cache directory (for decoded frames)
FRAME_CACHE_DIR = CACHE_DIR / "decoded_frames"
FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache for decoded frames (episode_path -> list of frames)
# This avoids re-decoding H.264 on every batch request within a session
_FRAME_CACHE: dict = {}


def _get_frame_cache_path(repo_id: str, episode_path: str) -> Path:
    """Get the persistent cache path for decoded frames."""
    import hashlib
    key = f"{repo_id}|{episode_path}"
    hash_key = hashlib.sha256(key.encode()).hexdigest()[:32]
    return FRAME_CACHE_DIR / f"{hash_key}.pkl"


def _load_persistent_frame_cache(repo_id: str, episode_path: str) -> Optional[List]:
    """Load frames from persistent cache if available."""
    cache_path = _get_frame_cache_path(repo_id, episode_path)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"Loaded {len(data)} frames from persistent cache for {episode_path}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load persistent frame cache: {e}")
    return None


def _save_persistent_frame_cache(repo_id: str, episode_path: str, frames: List) -> bool:
    """Save frames to persistent cache."""
    cache_path = _get_frame_cache_path(repo_id, episode_path)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(frames, f)
        logger.info(f"Saved {len(frames)} frames to persistent cache for {episode_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save persistent frame cache: {e}")
        return False


class StreamingFrameExtractor:
    """
    Extracts frames from streaming HuggingFace datasets.

    Supports:
    - MCAP files (RealOmni format with protobuf CompressedImage)
    - WebDataset TAR files (Egocentric-10K with video.mp4)
    - Direct video files (MP4, etc.)
    """

    def __init__(self, repo_id: str, cache_dir: Optional[Path] = None):
        self.repo_id = repo_id
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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

    def download_file(self, file_path: str) -> Path:
        """
        Download a file from HuggingFace if not already cached.

        Args:
            file_path: Path within the HuggingFace repo (e.g., "Cooking_and_Kitchen_Clean/clean_bowl/00001/00001.mcap")

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

            logger.info(f"Downloading {file_path} from {self.repo_id}...")

            # Create parent directory
            cached_path.parent.mkdir(parents=True, exist_ok=True)

            # Get authentication token for gated repos
            token = self._get_hf_token()

            # Download to cache
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=file_path,
                repo_type="dataset",
                local_dir=cached_path.parent,
                local_dir_use_symlinks=False,
                token=token,
            )

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
        episode_path: str = None
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract frames from an MCAP file.

        Args:
            file_path: Path to the MCAP file
            start: Start frame index
            end: End frame index
            camera_topic: Topic name for camera images
            episode_path: Original episode path (for persistent caching)

        Returns:
            List of (frame_idx, timestamp, image_array) tuples
        """
        global _FRAME_CACHE

        cache_key = str(file_path)

        # Check in-memory cache first
        if cache_key in _FRAME_CACHE:
            cached_frames = _FRAME_CACHE[cache_key]
            if len(cached_frames) >= end:
                logger.info(f"Using in-memory cached frames for {file_path} (range {start}-{end})")
                return cached_frames[start:end]
            # If cache doesn't have enough frames, we need to decode more
            logger.info(f"In-memory cache has {len(cached_frames)} frames, need up to {end}")

        # Check persistent cache
        if episode_path:
            persistent_frames = _load_persistent_frame_cache(self.repo_id, episode_path)
            if persistent_frames and len(persistent_frames) >= end:
                # Load into in-memory cache for faster subsequent access
                _FRAME_CACHE[cache_key] = persistent_frames
                logger.info(f"Loaded from persistent cache for {file_path} (range {start}-{end})")
                return persistent_frames[start:end]

        try:
            from mcap.reader import make_reader
            from mcap_protobuf.decoder import DecoderFactory

            # For H.264, we need to decode ALL frames from the beginning
            # because H.264 uses inter-frame compression
            frames = []
            h264_data = []
            timestamps = []

            with open(file_path, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])

                for schema, channel, message, decoded_message in reader.iter_decoded_messages():
                    # Look for camera topic (prefer robot0)
                    if "/robot0/sensor/camera0/compressed" in channel.topic:
                        if hasattr(decoded_message, 'data') and hasattr(decoded_message, 'format'):
                            img_format = decoded_message.format.lower()
                            img_data = decoded_message.data
                            timestamp = message.log_time / 1e9

                            if img_format == 'h264':
                                # Collect ALL H.264 NAL units (don't stop early)
                                h264_data.append(img_data)
                                timestamps.append(timestamp)
                            elif img_format in ['jpeg', 'jpg', 'png']:
                                # Direct decode for JPEG/PNG
                                import cv2
                                nparr = np.frombuffer(img_data, np.uint8)
                                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                if image is not None:
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    frames.append((len(frames), timestamp, image))

            # If we have H.264 data, decode ALL frames from the beginning
            if h264_data and not frames:
                logger.info(f"Decoding {len(h264_data)} H.264 NAL units from MCAP")
                # Decode all frames (start=0) and cache them
                frames = self._decode_h264_frames(h264_data, timestamps, 0, len(h264_data))

                # Cache the decoded frames for future requests (both in-memory and persistent)
                if frames:
                    _FRAME_CACHE[cache_key] = frames
                    logger.info(f"Cached {len(frames)} decoded frames in memory for {file_path}")

                    # Save to persistent cache for future sessions
                    if episode_path:
                        _save_persistent_frame_cache(self.repo_id, episode_path, frames)

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

                for frame in container.decode(stream):
                    if frame_idx >= end:
                        break

                    if frame_idx >= start:
                        # Convert to numpy array (RGB)
                        img = frame.to_ndarray(format='rgb24')
                        timestamp = timestamps[frame_idx] if frame_idx < len(timestamps) else frame_idx / 30.0
                        frames.append((frame_idx, timestamp, img))

                    frame_idx += 1

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
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract frames from a video file (MP4, etc.).

        Args:
            file_path: Path to the video file
            start: Start frame index
            end: End frame index

        Returns:
            List of (frame_idx, timestamp, image_array) tuples
        """
        try:
            import cv2

            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {file_path}")

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

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_idx / fps
                frames.append((frame_idx, timestamp, frame_rgb))
                frame_idx += 1

            cap.release()
            return frames

        except ImportError:
            logger.error("opencv-python not installed. Install with: pip install opencv-python")
            raise
        except Exception as e:
            logger.error(f"Failed to extract frames from video: {e}")
            raise

    def extract_frames_from_tar(
        self,
        file_path: Path,
        start: int = 0,
        end: int = 10,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract frames from a WebDataset TAR file containing video.

        Args:
            file_path: Path to the TAR file
            start: Start frame index
            end: End frame index

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
                    return self.extract_frames_from_video(tmp_path, start, end)
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
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract frames from an episode file (auto-detects format).

        Uses persistent caching to avoid re-downloading and re-decoding
        on subsequent requests.

        Args:
            episode_path: Path within the HuggingFace repo
            start: Start frame index
            end: End frame index

        Returns:
            List of (frame_idx, timestamp, image_array) tuples
        """
        # Download the file
        local_path = self.download_file(episode_path)

        # Detect format and extract
        suffix = local_path.suffix.lower()

        if suffix == '.mcap':
            return self.extract_frames_from_mcap(local_path, start, end, episode_path=episode_path)
        elif suffix == '.tar':
            return self.extract_frames_from_tar(local_path, start, end)
        elif suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return self.extract_frames_from_video(local_path, start, end)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

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
    ) -> Tuple[List[Tuple[int, float, np.ndarray]], int]:
        """
        Extract frames from an episode file and return total frame count.

        Uses persistent caching to avoid re-downloading and re-decoding
        on subsequent requests.

        Args:
            episode_path: Path within the HuggingFace repo
            start: Start frame index
            end: End frame index

        Returns:
            Tuple of (frames_list, total_frame_count)
        """
        # Download the file first
        local_path = self.download_file(episode_path)
        suffix = local_path.suffix.lower()

        # Get total frame count
        total_frames = self.get_frame_count(episode_path)

        # Extract frames (with episode_path for persistent caching)
        if suffix == '.mcap':
            frames = self.extract_frames_from_mcap(local_path, start, end, episode_path=episode_path)
        elif suffix == '.tar':
            frames = self.extract_frames_from_tar(local_path, start, end)
        elif suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            frames = self.extract_frames_from_video(local_path, start, end)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return frames, total_frames
