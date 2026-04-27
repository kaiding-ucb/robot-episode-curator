"""Episode-level action+state loader for phase-aware analysis.

Downloads LeRobot v3 data parquets directly from HF and returns raw numpy
arrays per episode (action + observation.state). Unlike the signal-extraction
path in `backend/api/routes/analysis.py`, this does no resampling — the
phase-aware pipeline needs original frame-rate data.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "REDACTED-HF-TOKEN")


async def _list_data_files(repo_id: str, branch: str = "main") -> list[str]:
    """List all data/chunk-*/file-*.parquet files in the repo via HF tree API."""
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{branch}/data?recursive=1"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        tree = resp.json()
    files = [t["path"] for t in tree if t.get("type") == "file" and t.get("path", "").endswith(".parquet")]
    files.sort()
    return files


async def _probe_columns(repo_id: str, first_file: str, branch: str, headers: dict) -> set[str]:
    """Read just the schema of the first parquet to learn available columns.
    Needed because datasets differ: Libero has ['action', 'observation.state'];
    UMI has ['observation.state', 'gripper_width'] with no action column."""
    import io
    import pyarrow.parquet as pq_
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}/{first_file}"
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return set()
        # Parse schema only (no rows) for speed
        buf = io.BytesIO(resp.content)
        try:
            schema = pq_.ParquetFile(buf).schema_arrow
            return set(schema.names)
        except Exception:
            try:
                return set(pd.read_parquet(io.BytesIO(resp.content)).columns)
            except Exception:
                return set()


async def _download_parquet(url: str, headers: dict, read_cols: list[str]) -> Optional[pd.DataFrame]:
    import io
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return None
        return pd.read_parquet(io.BytesIO(resp.content), columns=read_cols)


def _synthesize_action_from_state(
    state: np.ndarray,
    gripper_width: Optional[np.ndarray],
) -> np.ndarray:
    """When a dataset has no explicit action column (UMI-style), synthesize
    a 7-dim action vector our phase-aware pipeline can consume:
      action[:, 0:3] = delta-position
      action[:, 3:6] = delta-orientation (state must have >=6 dims)
      action[:, 6]   = -gripper_width  (flipped so narrow=low matches the
                                        "closed-is-low" convention phase_aware
                                        expects)
    """
    T = len(state)
    action = np.zeros((T, 7), dtype=float)
    if T > 1:
        s_dim = state.shape[1]
        action[1:, 0:3] = np.diff(state[:, 0:3], axis=0)
        if s_dim >= 6:
            action[1:, 3:6] = np.diff(state[:, 3:6], axis=0)
    if gripper_width is not None:
        action[:, 6] = -gripper_width
    return action


async def load_episodes_action_state(
    repo_id: str,
    target_episode_ids: list[int],
    branch: str = "main",
    max_episodes: int = 200,
) -> list[dict]:
    """Load raw action + observation.state arrays for a set of episodes.

    Handles both dataset variants:
      - Libero-style: `action` (T,D) + `observation.state` (T,S) both present
      - UMI-style:    `observation.state` (T,S) + separate `gripper_width` (T,)
                      column; `action` is synthesized from state deltas.

    Args:
        repo_id: HF dataset repo
        target_episode_ids: global episode_index values to load
        branch: git branch
        max_episodes: safety cap

    Returns:
        list of {"episode_id", "action": (T,7), "state": (T,S)}
    """
    if not target_episode_ids:
        return []
    targets = set(target_episode_ids[:max_episodes])
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    try:
        data_files = await _list_data_files(repo_id, branch=branch)
    except Exception as e:
        logger.error(f"Could not list data files for {repo_id}: {e}")
        return []
    if not data_files:
        return []

    # Probe the first file's schema so we know which columns this dataset uses.
    schema = await _probe_columns(repo_id, data_files[0], branch, headers)
    has_action = "action" in schema
    has_gripper_width = "gripper_width" in schema
    read_cols = ["episode_index", "frame_index", "observation.state"]
    if has_action:
        read_cols.append("action")
    if has_gripper_width:
        read_cols.append("gripper_width")
    logger.info(
        f"loader: repo={repo_id} schema_cols={sorted(schema)[:20]}… "
        f"read_cols={read_cols} has_action={has_action} has_gripper_width={has_gripper_width}"
    )

    results: dict[int, dict] = {}
    sem = asyncio.Semaphore(6)

    async def _probe(path: str) -> tuple[str, Optional[pd.DataFrame]]:
        async with sem:
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}/{path}"
            df = await _download_parquet(url, headers, read_cols)
            return path, df

    # Download incrementally; stop once all targets found
    for batch_start in range(0, len(data_files), 8):
        if not targets:
            break
        batch = data_files[batch_start:batch_start + 8]
        batch_results = await asyncio.gather(*[_probe(p) for p in batch])
        for path, df in batch_results:
            if df is None:
                continue
            present = set(int(x) for x in df["episode_index"].unique()) & targets
            if not present:
                continue
            for gid in present:
                ep_df = df[df.episode_index == gid].sort_values("frame_index")
                try:
                    state = np.stack([np.asarray(s, dtype=float) for s in ep_df["observation.state"].tolist()])
                    if has_action:
                        action = np.stack([np.asarray(a, dtype=float) for a in ep_df["action"].tolist()])
                    else:
                        # UMI path: synthesize action from state deltas + gripper_width column
                        gw_col = None
                        if has_gripper_width:
                            gw_col = np.asarray(ep_df["gripper_width"].tolist(), dtype=float).reshape(-1)
                        action = _synthesize_action_from_state(state, gw_col)
                except Exception as e:
                    logger.warning(f"episode_{gid} arrays failed to stack: {e}")
                    continue
                results[gid] = {
                    "episode_id": f"episode_{gid}",
                    "action": action,
                    "state": state,
                }
                targets.discard(gid)

    return [results[gid] for gid in sorted(results.keys())]
