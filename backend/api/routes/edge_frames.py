"""
Starting & Ending Frames endpoint.

GET /api/datasets/{dataset_id}/tasks/{task_name}/edge-frames/stream?position=start|end&limit=50
Streams thumbnails of the first or last frame of each episode in a task,
using PyAV HTTP-range seeks (no full-episode downloads). LeRobot v3 only.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from downloaders.manager import get_all_datasets

logger = logging.getLogger(__name__)
router = APIRouter()

EDGE_FRAMES_CACHE_DIR = Path.home() / ".cache" / "data_viewer" / "edge_frames"
EDGE_FRAMES_TTL = timedelta(days=30)
THUMB_W, THUMB_H = 240, 180
JPEG_QUALITY = 70
DEFAULT_LIMIT = 50
MAX_LIMIT = 100
PER_EPISODE_TIMEOUT_S = 8.0
CHUNK_CONCURRENCY = 4


def _safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _disk_path(repo_id: str, episode_idx: int, position: str, video_key: str) -> Path:
    return (
        EDGE_FRAMES_CACHE_DIR
        / _safe_id(repo_id)
        / _safe_id(video_key)
        / f"ep{episode_idx:06d}_{position}.b64"
    )


def _read_disk_cache(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        if datetime.utcnow() - mtime > EDGE_FRAMES_TTL:
            p.unlink(missing_ok=True)
            return None
        return p.read_text()
    except Exception as e:
        logger.warning(f"edge_frames: cache read failed for {p}: {e}")
        return None


def _write_disk_cache(p: Path, b64: str) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(b64)
    except Exception as e:
        logger.warning(f"edge_frames: cache write failed for {p}: {e}")


def _hf_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    for path in (
        Path.home() / ".huggingface" / "token",
        Path.home() / ".cache" / "huggingface" / "token",
    ):
        if path.exists():
            try:
                return path.read_text().strip()
            except Exception:
                pass
    return None


def _pick_video_key(info: dict) -> Optional[str]:
    """Pick the canonical camera feature key (first dtype:video|image)."""
    feats = info.get("features") or {}
    for key, meta in feats.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("dtype") in ("video", "image"):
            return key
    return None


def _encode_jpeg_b64(rgb) -> str:
    """Resize an RGB ndarray to a thumbnail and base64-encode as JPEG."""
    import cv2
    import numpy as np

    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        bgr = arr
    h, w = bgr.shape[:2]
    scale = min(THUMB_W / max(w, 1), THUMB_H / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    if (new_w, new_h) != (w, h):
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_edge_frame_sync(
    url: str,
    headers: Optional[str],
    t_start: float,
    t_end: float,
    position: str,
    fps: float,
) -> str:
    """Open MP4 over HTTP, seek to the start or end timestamp, decode 1 frame, return JPEG b64."""
    import av

    opts = {}
    if headers:
        opts["headers"] = headers
    container = av.open(url, options=opts)
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base) if stream.time_base else None
        if time_base is None:
            raise RuntimeError("video stream has no time_base")

        if position == "end":
            target_t = max(t_start, t_end - 1.0 / max(fps, 1.0))
        else:
            target_t = t_start

        seek_pts = int(target_t / time_base)
        container.seek(seek_pts, any_frame=False, backward=True, stream=stream)

        chosen = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts_s = float(frame.pts) * time_base
            if position == "end":
                if pts_s >= t_end - 1e-6 and chosen is not None:
                    break
                if t_start - 0.05 <= pts_s < t_end + 1e-6:
                    chosen = frame
                    continue
                if pts_s >= t_end:
                    break
            else:
                if pts_s + 1e-6 < t_start:
                    continue
                chosen = frame
                break

        if chosen is None:
            raise RuntimeError(f"no frame decoded for position={position} in [{t_start}, {t_end}]")
        rgb = chosen.to_ndarray(format="rgb24")
        return _encode_jpeg_b64(rgb)
    finally:
        container.close()


@router.get("/{dataset_id}/tasks/{task_name:path}/edge-frames/stream")
async def edge_frames_stream(
    dataset_id: str,
    task_name: str,
    request: Request,
    position: str = Query("start", pattern="^(start|end)$"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
):
    """
    SSE stream of starting or ending frame thumbnails for episodes in a task.
    """
    all_datasets = get_all_datasets()
    if dataset_id not in all_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    config = all_datasets[dataset_id]
    repo_id = config.get("repo_id")
    if not repo_id:
        raise HTTPException(status_code=400, detail="Dataset has no repo_id")

    # Resolve LeRobot meta via existing cached helpers.
    from api.routes.datasets import (
        fetch_lerobot_info,
        fetch_lerobot_tasks_meta,
        fetch_lerobot_episodes_meta,
        get_episode_task_map,
        is_lerobot_dataset,
    )

    if not await is_lerobot_dataset(config):
        raise HTTPException(
            status_code=400,
            detail="Edge frames are only available for LeRobot datasets.",
        )

    info = await fetch_lerobot_info(repo_id)
    if info is None:
        raise HTTPException(status_code=502, detail="Could not fetch LeRobot info.json")

    episodes_df = await fetch_lerobot_episodes_meta(repo_id)
    if episodes_df is None:
        raise HTTPException(status_code=502, detail="Could not fetch LeRobot episodes meta")

    all_tasks_mode = task_name in ("_all_", "")

    if all_tasks_mode:
        all_indices = sorted(int(x) for x in episodes_df["episode_index"].tolist())
        total_for_task = len(all_indices)
        selected_indices = all_indices[:limit]
    else:
        tasks_df = await fetch_lerobot_tasks_meta(repo_id)
        if tasks_df is None:
            raise HTTPException(status_code=502, detail="Could not fetch LeRobot tasks meta")

        # Resolve task_index from task_name
        task_col = "task_description" if "task_description" in tasks_df.columns else None
        if task_col is None:
            for col in tasks_df.columns:
                if col != "task_index":
                    task_col = col
                    break

        task_index: Optional[int] = None
        if task_col is not None:
            match = tasks_df[tasks_df[task_col] == task_name]
            if len(match) > 0:
                task_index = int(match.iloc[0]["task_index"])
        if task_index is None:
            m = re.match(r"^Untitled \(task (\d+)\)$", task_name)
            if m:
                candidate = int(m.group(1))
                if candidate in tasks_df["task_index"].values:
                    task_index = candidate
        if task_index is None:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_name}")

        ep_task_map = await get_episode_task_map(repo_id, episodes_df=episodes_df)
        if ep_task_map is None:
            raise HTTPException(status_code=502, detail="Could not derive episode-task map")

        task_ep_indices = sorted(
            [ep_idx for ep_idx, t_idx in ep_task_map.items() if t_idx == task_index]
        )
        total_for_task = len(task_ep_indices)
        selected_indices = task_ep_indices[:limit]

    video_path_template = info.get("video_path")
    video_key = _pick_video_key(info)
    fps = float(info.get("fps") or 30)

    if not video_path_template or not video_key:
        raise HTTPException(
            status_code=400,
            detail="Dataset info.json has no video_path / video features.",
        )

    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    t0_col = f"videos/{video_key}/from_timestamp"
    t1_col = f"videos/{video_key}/to_timestamp"
    needed = (chunk_col, file_col, t0_col, t1_col)
    if any(c not in episodes_df.columns for c in needed):
        raise HTTPException(
            status_code=400,
            detail=f"Episode meta missing video range columns for key '{video_key}'.",
        )

    df = episodes_df[episodes_df["episode_index"].isin(selected_indices)].copy()
    df = df.set_index("episode_index").reindex(selected_indices).reset_index()

    token = _hf_token()
    auth_header = f"Authorization: Bearer {token}\r\n" if token else None

    # Group episodes by (chunk, file)
    groups: Dict[Tuple[int, int], List[int]] = {}
    metas: Dict[int, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        try:
            ep_idx = int(row["episode_index"])
            chunk_idx = int(row[chunk_col])
            file_idx = int(row[file_col])
            t_start = float(row[t0_col])
            t_end = float(row[t1_col])
            length = int(row["length"]) if "length" in row.index and not row["length"] is None else None
        except Exception as e:
            logger.warning(f"edge_frames: bad episode meta row: {e}")
            continue
        groups.setdefault((chunk_idx, file_idx), []).append(ep_idx)
        metas[ep_idx] = {
            "chunk_idx": chunk_idx,
            "file_idx": file_idx,
            "t_start": t_start,
            "t_end": t_end,
            "length": length,
        }

    async def event_gen():
        # Header event
        yield "data: " + json.dumps({
            "type": "total",
            "total": len(selected_indices),
            "total_for_task": total_for_task,
            "task_name": task_name,
            "position": position,
            "video_key": video_key,
        }) + "\n\n"
        # Episode meta events (cheap — emit immediately so UI can render skeleton tiles)
        for ep_idx in selected_indices:
            m = metas.get(ep_idx)
            if m is None:
                continue
            yield "data: " + json.dumps({
                "type": "episode_meta",
                "episode_id": f"episode_{ep_idx}",
                "episode_index": ep_idx,
                "total_frames": m["length"],
            }) + "\n\n"

        # Queue of (ep_idx, b64 or error) results
        result_q: asyncio.Queue = asyncio.Queue()
        sem = asyncio.Semaphore(CHUNK_CONCURRENCY)
        loop = asyncio.get_event_loop()

        async def process_chunk(chunk_key: Tuple[int, int], eps: List[int]):
            chunk_idx, file_idx = chunk_key
            rel = video_path_template.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}"

            async with sem:
                for ep_idx in eps:
                    if await request.is_disconnected():
                        return
                    m = metas[ep_idx]
                    cache_p = _disk_path(repo_id, ep_idx, position, video_key)
                    cached = _read_disk_cache(cache_p)
                    if cached is not None:
                        await result_q.put((ep_idx, cached, None))
                        continue
                    try:
                        b64 = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                _decode_edge_frame_sync,
                                url,
                                auth_header,
                                m["t_start"],
                                m["t_end"],
                                position,
                                fps,
                            ),
                            timeout=PER_EPISODE_TIMEOUT_S,
                        )
                        _write_disk_cache(cache_p, b64)
                        await result_q.put((ep_idx, b64, None))
                    except asyncio.TimeoutError:
                        await result_q.put((ep_idx, None, "timeout"))
                    except Exception as e:
                        logger.warning(
                            f"edge_frames: decode failed ep={ep_idx} pos={position}: {e}"
                        )
                        await result_q.put((ep_idx, None, str(e)[:160]))

        tasks = [
            asyncio.create_task(process_chunk(key, eps))
            for key, eps in groups.items()
        ]

        produced = 0
        target = sum(len(v) for v in groups.values())

        while produced < target:
            try:
                ep_idx, b64, err = await asyncio.wait_for(result_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if all(t.done() for t in tasks) and result_q.empty():
                    break
                continue
            produced += 1
            if b64 is not None:
                yield "data: " + json.dumps({
                    "type": "frame",
                    "episode_id": f"episode_{ep_idx}",
                    "episode_index": ep_idx,
                    "image_b64": b64,
                }) + "\n\n"
            else:
                yield "data: " + json.dumps({
                    "type": "error",
                    "episode_id": f"episode_{ep_idx}",
                    "episode_index": ep_idx,
                    "message": err or "decode failed",
                }) + "\n\n"

        for t in tasks:
            if not t.done():
                t.cancel()
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
