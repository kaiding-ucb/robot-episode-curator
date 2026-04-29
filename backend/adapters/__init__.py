"""
Streaming dataset adapters for various HuggingFace dataset formats.

Usage:
    from adapters import FormatRegistry

    adapter = await FormatRegistry.get_adapter(dataset_id, config)
    if adapter:
        tasks = await adapter.list_tasks()
        episodes, total = await adapter.list_episodes(task_name, limit=10)
        resolution = await adapter.resolve_episode(episode_id)
"""
from .base import EpisodeRef, FrameResolution, StreamingAdapter, TaskRef
from .registry import FormatRegistry

__all__ = [
    "StreamingAdapter",
    "TaskRef",
    "EpisodeRef",
    "FrameResolution",
    "FormatRegistry",
]
