"""
Stream-only MP4 slicing from a remote (HuggingFace) URL.

Uses ffmpeg with `-c copy` (stream copy, no re-encode) and `+faststart` so the
output starts playing as soon as a few KB are ready. Authenticates via the
`-headers` flag so the redirect to HF's CDN keeps the token.

This is the consistent path used by the Rerun viewer and the Gemini analysis
pipeline — neither downloads the full chunk MP4 anymore.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class FfmpegError(RuntimeError):
    pass


def slice_remote_mp4(
    url: str,
    hf_token: Optional[str],
    t_start: float,
    t_end: float,
    out_path: Path,
    *,
    pre_roll: float = 1.0,
    timeout_s: float = 60.0,
) -> Path:
    """Slice [t_start, t_end] from a remote MP4 via ffmpeg HTTP range reads.

    Args:
        url: Full URL to the MP4 (e.g., HF resolve/main/...).
        hf_token: Bearer token forwarded via `-headers`. None for public files.
        t_start: Start time in seconds (within the source video).
        t_end: End time in seconds.
        out_path: Where the sliced MP4 is written. Parent dirs are created.
        pre_roll: Seconds of lookback before t_start so input-side seek lands
            on the previous keyframe. Output may include a bit of pre-roll
            but `-c copy` requires keyframe alignment, and 1s is harmless for
            visualization.
        timeout_s: ffmpeg subprocess timeout.

    Returns:
        out_path on success.

    Raises:
        FfmpegError on ffmpeg failure or timeout.
    """
    if not _ffmpeg_available():
        raise FfmpegError("ffmpeg is not installed or not on PATH")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seek_start = max(0.0, t_start - pre_roll)
    duration = max(0.1, t_end - seek_start)

    headers = ""
    if hf_token:
        # ffmpeg requires CRLF-terminated header lines; passes through HTTP redirects.
        headers = f"Authorization: Bearer {hf_token}\r\n"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-nostdin",
        # Input options — input-side seek for speed (ffmpeg snaps to the prior keyframe).
        "-ss", f"{seek_start:.3f}",
        "-t", f"{duration:.3f}",
    ]
    if headers:
        cmd += ["-headers", headers]
    cmd += [
        "-i", url,
        "-map", "0:v:0",
        "-c", "copy",
        "-movflags", "+faststart",
        # Force mp4 container even with stream copy.
        "-f", "mp4",
        str(out_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise FfmpegError(f"ffmpeg slice timed out after {timeout_s}s") from e
    except FileNotFoundError as e:
        raise FfmpegError("ffmpeg binary not found") from e

    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")[:500] if proc.stderr else ""
        raise FfmpegError(
            f"ffmpeg slice failed (rc={proc.returncode}) for {url} "
            f"[{seek_start:.2f}, {seek_start + duration:.2f}s]: {err}"
        )

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise FfmpegError(f"ffmpeg produced empty output at {out_path}")

    return out_path


def slice_remote_mp4_to_bytes(
    url: str,
    hf_token: Optional[str],
    t_start: float,
    t_end: float,
    *,
    pre_roll: float = 1.0,
    timeout_s: float = 60.0,
) -> bytes:
    """Same as slice_remote_mp4 but returns raw bytes (no on-disk artifact)."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        slice_remote_mp4(url, hf_token, t_start, t_end, tmp_path,
                         pre_roll=pre_roll, timeout_s=timeout_s)
        return tmp_path.read_bytes()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
