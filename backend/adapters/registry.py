"""
Format Registry for auto-detecting and caching dataset adapters.

The FormatRegistry is the single entry point for getting the right adapter
for any HuggingFace dataset. It probes the repo structure and caches the
result so subsequent calls are instant.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import httpx

from .base import StreamingAdapter

logger = logging.getLogger(__name__)

# Adapter cache: repo_id -> adapter instance (or None if detection failed)
_ADAPTER_CACHE: Dict[str, Optional[StreamingAdapter]] = {}


def _get_hf_headers() -> dict:
    """Get HuggingFace auth headers if token is available."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        for token_path in [
            Path.home() / ".huggingface" / "token",
            Path.home() / ".cache" / "huggingface" / "token",
        ]:
            if token_path.exists():
                hf_token = token_path.read_text().strip()
                break
    return {"Authorization": f"Bearer {hf_token}"} if hf_token else {}


class FormatRegistry:
    """
    Detects dataset format and returns the appropriate StreamingAdapter.

    Detection algorithm:
    1. Check config for explicit format
    2. Probe meta/info.json for LeRobot codebase_version
       - If found, check if multi-subdataset (GR00T pattern)
       - Return LeRobotAdapter or MultiSubdatasetAdapter
    3. Probe for .mcap files -> MCAPAdapter
    4. Probe for .mp4/.avi/.webm without meta/info.json -> RawVideoAdapter
    5. Fallback: None (use existing code path)
    """

    @staticmethod
    async def get_adapter(
        dataset_id: str,
        config: Dict,
    ) -> Optional[StreamingAdapter]:
        """
        Get the appropriate adapter for a dataset.

        Uses cached adapter if available, otherwise auto-detects format.

        Args:
            dataset_id: Internal dataset identifier
            config: Dataset configuration from the registry

        Returns:
            StreamingAdapter instance, or None if no adapter matches
        """
        repo_id = config.get("repo_id")
        if not repo_id:
            return None

        # Only streaming datasets get adapters
        if not config.get("streaming_recommended"):
            return None

        # Check cache
        if repo_id in _ADAPTER_CACHE:
            return _ADAPTER_CACHE[repo_id]

        # Try explicit format from config first
        fmt = config.get("format")
        adapter = await FormatRegistry._create_adapter_for_format(
            fmt, repo_id, config
        )

        if adapter is None and fmt is None:
            # Auto-detect format by probing the repo
            adapter = await FormatRegistry._auto_detect(repo_id, config)

        _ADAPTER_CACHE[repo_id] = adapter
        if adapter:
            logger.info(
                f"Registered adapter {type(adapter).__name__} for {repo_id}"
            )
        return adapter

    @staticmethod
    async def _create_adapter_for_format(
        fmt: Optional[str],
        repo_id: str,
        config: Dict,
    ) -> Optional[StreamingAdapter]:
        """Create adapter from an explicit format string."""
        if fmt is None:
            return None

        fmt_lower = fmt.lower()

        if fmt_lower in ("lerobot", "lerobot_v2", "lerobot_v3"):
            # Check for multi-subdataset pattern first
            if await FormatRegistry._is_multi_subdataset(repo_id):
                from .multi_subdataset_adapter import MultiSubdatasetAdapter
                return MultiSubdatasetAdapter(repo_id, config)
            from .lerobot_adapter import LeRobotAdapter
            return LeRobotAdapter(repo_id, config)

        if fmt_lower == "mcap":
            from .mcap_adapter import MCAPAdapter
            return MCAPAdapter(repo_id, config)

        if fmt_lower in ("video", "mp4", "raw_video"):
            from .raw_video_adapter import RawVideoAdapter
            return RawVideoAdapter(repo_id, config)

        return None

    @staticmethod
    async def _auto_detect(
        repo_id: str,
        config: Dict,
    ) -> Optional[StreamingAdapter]:
        """Auto-detect format by probing the HF repo structure."""
        headers = _get_hf_headers()

        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                # Fetch top-level tree
                tree_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
                resp = await client.get(tree_url, headers=headers, timeout=15.0)
                if resp.status_code != 200:
                    return None

                items = resp.json()
                dirs = {
                    item["path"]
                    for item in items
                    if item.get("type") == "directory"
                }
                files = {
                    item["path"]
                    for item in items
                    if item.get("type") == "file"
                }
                extensions = {Path(f).suffix.lower() for f in files}

                # 0. Check multi-subdataset FIRST (repos with NO root meta/
                #    but subdirs each have their own meta/info.json, e.g. GR00T)
                non_meta_dirs_early = dirs - {
                    "meta", "data", "videos", ".cache",
                    ".huggingface", "logs", "stats",
                }
                if "meta" not in dirs and len(non_meta_dirs_early) >= 2:
                    if await FormatRegistry._is_multi_subdataset(repo_id):
                        from .multi_subdataset_adapter import (
                            MultiSubdatasetAdapter,
                        )

                        return MultiSubdatasetAdapter(repo_id, config)

                # 1. Check for LeRobot (meta/ directory with info.json)
                if "meta" in dirs:
                    info_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/info.json"
                    info_resp = await client.get(
                        info_url, headers=headers, timeout=10.0
                    )
                    if info_resp.status_code == 200:
                        info = info_resp.json()
                        if "codebase_version" in info:
                            # Check multi-subdataset
                            if await FormatRegistry._is_multi_subdataset(
                                repo_id
                            ):
                                from .multi_subdataset_adapter import (
                                    MultiSubdatasetAdapter,
                                )
                                return MultiSubdatasetAdapter(
                                    repo_id, config
                                )
                            from .lerobot_adapter import LeRobotAdapter
                            return LeRobotAdapter(repo_id, config)

                # 2. Check for MCAP files (recurse one level if needed)
                if ".mcap" in extensions:
                    from .mcap_adapter import MCAPAdapter
                    return MCAPAdapter(repo_id, config)

                # Check first subdirectory for MCAP files
                non_meta_dirs = dirs - {
                    "meta", "data", "videos", ".cache",
                    ".huggingface", "logs", "stats",
                }
                for d in sorted(non_meta_dirs)[:3]:
                    import urllib.parse
                    enc = urllib.parse.quote(d, safe="")
                    sub_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{enc}"
                    sub_resp = await client.get(
                        sub_url, headers=headers, timeout=10.0
                    )
                    if sub_resp.status_code == 200:
                        sub_items = sub_resp.json()
                        sub_exts = {
                            Path(i["path"]).suffix.lower()
                            for i in sub_items
                            if i.get("type") == "file"
                        }
                        if ".mcap" in sub_exts:
                            from .mcap_adapter import MCAPAdapter
                            return MCAPAdapter(repo_id, config)

                # 3. Check for raw video files
                video_exts = {".mp4", ".avi", ".webm", ".mov", ".mkv"}
                if extensions & video_exts:
                    from .raw_video_adapter import RawVideoAdapter
                    return RawVideoAdapter(repo_id, config)

        except Exception as e:
            logger.warning(f"Auto-detection failed for {repo_id}: {e}")

        return None

    @staticmethod
    async def _is_multi_subdataset(repo_id: str) -> bool:
        """
        Check if a repo contains multiple subdatasets (GR00T pattern).

        A multi-subdataset repo has multiple top-level directories, each
        containing its own meta/info.json.
        """
        headers = _get_hf_headers()

        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                tree_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
                resp = await client.get(tree_url, headers=headers, timeout=15.0)
                if resp.status_code != 200:
                    return False

                items = resp.json()
                dirs = [
                    item["path"]
                    for item in items
                    if item.get("type") == "directory"
                    and not item["path"].startswith(".")
                    and item["path"] not in ("meta", "data", "videos")
                ]

                if len(dirs) < 2:
                    return False

                # Check if at least 2 subdirectories have their own meta/info.json
                import urllib.parse
                found = 0
                for d in dirs[:5]:  # Probe up to 5 dirs
                    enc = urllib.parse.quote(d, safe="")
                    info_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{enc}/meta/info.json"
                    info_resp = await client.get(
                        info_url, headers=headers, timeout=10.0
                    )
                    if info_resp.status_code == 200:
                        found += 1
                        if found >= 2:
                            logger.info(
                                f"Detected multi-subdataset pattern for {repo_id} "
                                f"({len(dirs)} top-level dirs)"
                            )
                            return True

        except Exception as e:
            logger.warning(f"Multi-subdataset check failed for {repo_id}: {e}")

        return False

    @staticmethod
    def invalidate(repo_id: str) -> None:
        """Remove a cached adapter (e.g., when a dataset is deleted)."""
        _ADAPTER_CACHE.pop(repo_id, None)

    @staticmethod
    def invalidate_all() -> None:
        """Clear entire adapter cache."""
        _ADAPTER_CACHE.clear()
