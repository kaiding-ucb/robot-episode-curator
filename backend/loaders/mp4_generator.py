"""
MP4 Generator - Creates preview videos from MCAP files.

Generates browser-native MP4 videos from MCAP files for instant playback
with native seeking support. Uses ffmpeg for H.264 encoding.
"""
import asyncio
import logging
import os
from pathlib import Path
from typing import Callable, Optional, AsyncGenerator
import numpy as np

logger = logging.getLogger(__name__)

# Cache directory for generated MP4 files
MP4_CACHE_DIR = Path(os.environ.get(
    "MP4_CACHE_DIR",
    Path.home() / ".cache" / "data_viewer" / "mp4_previews"
))
MP4_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class MP4Generator:
    """Generate MP4 preview videos from MCAP files."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or MP4_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_mp4_path(self, dataset_id: str, episode_id: str) -> Path:
        """Get cached MP4 path for an episode."""
        safe_name = f"{dataset_id}_{episode_id}".replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_name}.mp4"

    def has_cached_mp4(self, dataset_id: str, episode_id: str) -> bool:
        """Check if MP4 preview already exists."""
        return self.get_mp4_path(dataset_id, episode_id).exists()

    def get_mp4_size(self, dataset_id: str, episode_id: str) -> int:
        """Get size of cached MP4 in bytes."""
        mp4_path = self.get_mp4_path(dataset_id, episode_id)
        if mp4_path.exists():
            return mp4_path.stat().st_size
        return 0

    async def generate_mp4_from_mcap(
        self,
        mcap_path: Path,
        dataset_id: str,
        episode_id: str,
        resolution: str = "720p",
        fps: int = 30,
        on_progress: Optional[Callable[[int, Optional[int]], None]] = None
    ) -> Path:
        """
        Generate MP4 from MCAP file.

        Args:
            mcap_path: Path to source MCAP file
            dataset_id: Dataset identifier
            episode_id: Episode identifier
            resolution: "1080p", "720p", "480p", or "360p"
            fps: Output frame rate
            on_progress: Callback with (current_frame, total_frames)

        Returns:
            Path to generated MP4 file
        """
        output_path = self.get_mp4_path(dataset_id, episode_id)

        # Resolution presets
        res_map = {
            "1080p": "1920:1080",
            "720p": "1280:720",
            "480p": "854:480",
            "360p": "640:360"
        }
        scale = res_map.get(resolution, "1280:720")

        logger.info(f"Generating MP4 from {mcap_path} -> {output_path}")

        # Create temp file for output (atomic write)
        temp_output = output_path.with_suffix('.mp4.tmp')

        # ffmpeg command: read JPEG frames from stdin, encode to H.264
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f", "image2pipe",
            "-framerate", str(fps),
            "-i", "-",  # Read from stdin
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",  # Good quality/size balance
            "-vf", f"scale={scale}:force_original_aspect_ratio=decrease,pad={scale}:(ow-iw)/2:(oh-ih)/2",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",  # Enable streaming
            str(temp_output)
        ]

        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Extract frames from MCAP and pipe to ffmpeg
        frame_count = 0
        try:
            async for frame_data in self._extract_jpeg_frames_from_mcap(mcap_path):
                if process.stdin:
                    process.stdin.write(frame_data)
                    await process.stdin.drain()
                frame_count += 1

                if on_progress and frame_count % 30 == 0:
                    on_progress(frame_count, None)

            if process.stdin:
                process.stdin.close()

            await process.wait()

            if process.returncode != 0:
                stderr = await process.stderr.read() if process.stderr else b""
                raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

            # Atomic rename
            temp_output.rename(output_path)
            logger.info(f"Generated MP4: {output_path} ({frame_count} frames)")

            return output_path

        except Exception as e:
            # Clean up temp file on error
            if temp_output.exists():
                temp_output.unlink()
            raise

    async def _extract_jpeg_frames_from_mcap(
        self,
        mcap_path: Path
    ) -> AsyncGenerator[bytes, None]:
        """Extract raw JPEG frames from MCAP file."""
        import cv2

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
                raise ImportError("No MCAP decoder factories available")

            # Topic patterns for RGB images
            rgb_patterns = ["rgb", "color", "image", "camera", "compressed"]

            def is_rgb_topic(topic: str) -> bool:
                topic_lower = topic.lower()
                return any(p in topic_lower for p in rgb_patterns) and "depth" not in topic_lower

            with open(mcap_path, "rb") as f:
                reader = make_reader(f, decoder_factories=decoder_factories)

                for schema, channel, message, decoded_message in reader.iter_decoded_messages():
                    if not is_rgb_topic(channel.topic):
                        continue

                    if hasattr(decoded_message, 'data') and hasattr(decoded_message, 'format'):
                        img_format = decoded_message.format.lower()
                        img_data = bytes(decoded_message.data)

                        if 'jpeg' in img_format or 'jpg' in img_format:
                            # Already JPEG, yield directly
                            yield img_data
                        elif 'png' in img_format:
                            # PNG - decode and re-encode as JPEG for consistency
                            nparr = np.frombuffer(img_data, np.uint8)
                            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if image is not None:
                                _, jpeg_data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                yield jpeg_data.tobytes()
                        elif 'h264' not in img_format:
                            # Try to decode as raw image
                            nparr = np.frombuffer(img_data, np.uint8)
                            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if image is not None:
                                _, jpeg_data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                yield jpeg_data.tobytes()

        except Exception as e:
            logger.error(f"Failed to extract frames from MCAP: {e}")
            raise


# Global generator instance
_generator: Optional[MP4Generator] = None


def get_mp4_generator() -> MP4Generator:
    """Get or create the global MP4 generator instance."""
    global _generator
    if _generator is None:
        _generator = MP4Generator()
    return _generator
