"""
MCAP file utilities for modality detection and metadata extraction.

Provides functions to scan MCAP channels and detect available modalities
(RGB, depth, IMU, actions) from topic names and message types.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Modality, ModalityConfig

logger = logging.getLogger(__name__)


def detect_mcap_modalities(mcap_path: Path) -> Dict[str, ModalityConfig]:
    """
    Scan MCAP channels to detect available modalities.

    Args:
        mcap_path: Path to the MCAP file

    Returns:
        Dictionary mapping modality names to their configurations
    """
    try:
        from mcap.reader import make_reader
    except ImportError:
        logger.warning("mcap package not installed, cannot detect modalities")
        return {}

    modalities: Dict[str, ModalityConfig] = {}

    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()

            if summary is None:
                logger.warning(f"MCAP file has no summary: {mcap_path}")
                return {}

            for channel_id, channel in summary.channels.items():
                topic = channel.topic.lower()
                schema = summary.schemas.get(channel.schema_id)
                schema_name = schema.name if schema else ""

                # Detect RGB camera topics
                if _is_rgb_topic(topic, schema_name):
                    # If we already have rgb, keep the first one
                    if Modality.RGB.value not in modalities:
                        modalities[Modality.RGB.value] = ModalityConfig(
                            topic=channel.topic,
                            type="image"
                        )

                # Detect depth camera topics
                elif _is_depth_topic(topic, schema_name):
                    if Modality.DEPTH.value not in modalities:
                        modalities[Modality.DEPTH.value] = ModalityConfig(
                            topic=channel.topic,
                            type="image",
                            colormap="viridis"
                        )

                # Detect IMU topics
                elif _is_imu_topic(topic, schema_name):
                    if Modality.IMU.value not in modalities:
                        modalities[Modality.IMU.value] = ModalityConfig(
                            topic=channel.topic,
                            type="timeseries"
                        )

                # Detect action topics
                elif _is_action_topic(topic, schema_name):
                    if Modality.ACTIONS.value not in modalities:
                        modalities[Modality.ACTIONS.value] = ModalityConfig(
                            topic=channel.topic,
                            type="vector"
                        )

                # Detect state topics
                elif _is_state_topic(topic, schema_name):
                    if Modality.STATES.value not in modalities:
                        modalities[Modality.STATES.value] = ModalityConfig(
                            topic=channel.topic,
                            type="vector"
                        )

    except Exception as e:
        logger.error(f"Failed to read MCAP file {mcap_path}: {e}")

    return modalities


def _is_rgb_topic(topic: str, schema_name: str) -> bool:
    """Check if a topic is an RGB camera topic."""
    rgb_keywords = ["rgb", "color", "image", "camera", "compressed"]
    depth_keywords = ["depth"]

    # Must have RGB indicator and NOT have depth indicator
    has_rgb = any(kw in topic for kw in rgb_keywords)
    has_depth = any(kw in topic for kw in depth_keywords)

    # Check schema for image types
    is_image_schema = any(t in schema_name.lower() for t in [
        "compressedimage", "image", "sensor_msgs/image"
    ])

    return (has_rgb or is_image_schema) and not has_depth


def _is_depth_topic(topic: str, schema_name: str) -> bool:
    """Check if a topic is a depth camera topic."""
    depth_keywords = ["depth", "disparity", "range"]
    return any(kw in topic for kw in depth_keywords)


def _is_imu_topic(topic: str, schema_name: str) -> bool:
    """Check if a topic is an IMU topic."""
    imu_keywords = ["imu", "accelerometer", "gyroscope", "accel", "gyro"]
    return any(kw in topic for kw in imu_keywords)


def _is_action_topic(topic: str, schema_name: str) -> bool:
    """Check if a topic is an action/command topic."""
    action_keywords = ["action", "command", "control", "cmd", "joint_command"]
    return any(kw in topic for kw in action_keywords)


def _is_state_topic(topic: str, schema_name: str) -> bool:
    """Check if a topic is a robot state topic."""
    state_keywords = ["state", "joint_state", "robot_state", "pose", "odom"]
    return any(kw in topic for kw in state_keywords)


def list_mcap_channels(mcap_path: Path) -> List[Dict[str, Any]]:
    """
    List all channels in an MCAP file.

    Args:
        mcap_path: Path to the MCAP file

    Returns:
        List of channel info dictionaries
    """
    try:
        from mcap.reader import make_reader
    except ImportError:
        logger.warning("mcap package not installed")
        return []

    channels = []

    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()

            if summary is None:
                return []

            for channel_id, channel in summary.channels.items():
                schema = summary.schemas.get(channel.schema_id)
                channels.append({
                    "id": channel_id,
                    "topic": channel.topic,
                    "schema_name": schema.name if schema else None,
                    "message_count": summary.statistics.channel_message_counts.get(channel_id, 0)
                    if summary.statistics else None
                })

    except Exception as e:
        logger.error(f"Failed to list MCAP channels from {mcap_path}: {e}")

    return channels


def get_mcap_metadata(mcap_path: Path) -> Dict[str, Any]:
    """
    Get metadata from an MCAP file.

    Args:
        mcap_path: Path to the MCAP file

    Returns:
        Dictionary with file metadata
    """
    try:
        from mcap.reader import make_reader
    except ImportError:
        return {"error": "mcap package not installed"}

    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()

            if summary is None:
                return {"error": "No summary in MCAP file"}

            stats = summary.statistics
            return {
                "message_count": stats.message_count if stats else None,
                "channel_count": len(summary.channels),
                "schema_count": len(summary.schemas),
                "start_time": stats.message_start_time if stats else None,
                "end_time": stats.message_end_time if stats else None,
                "duration_ns": (stats.message_end_time - stats.message_start_time)
                if stats and stats.message_start_time and stats.message_end_time else None,
            }

    except Exception as e:
        logger.error(f"Failed to get MCAP metadata from {mcap_path}: {e}")
        return {"error": str(e)}
