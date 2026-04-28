"""
Settings endpoints — HuggingFace token management.

The backend already reads the HF token from $HF_TOKEN, $HUGGING_FACE_HUB_TOKEN,
~/.huggingface/token, or ~/.cache/huggingface/token. This module exposes a
small UI to set/clear the file-based token and report status, never returning
the literal value.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

TOKEN_FILE = Path.home() / ".huggingface" / "token"


class HfTokenStatus(BaseModel):
    has_token: bool
    source: str  # "env" | "file" | "none"
    masked: Optional[str] = None
    username: Optional[str] = None


class HfTokenUpdate(BaseModel):
    token: str = Field(..., min_length=4, max_length=200)


def _read_token_from_file() -> Optional[str]:
    for p in (TOKEN_FILE, Path.home() / ".cache" / "huggingface" / "token"):
        if p.exists():
            try:
                t = p.read_text().strip()
                if t:
                    return t
            except OSError:
                continue
    return None


def _current_token() -> tuple[Optional[str], str]:
    """Return (token, source). Source is "env", "file", or "none"."""
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t, "env"
    t = _read_token_from_file()
    if t:
        return t, "file"
    return None, "none"


def _mask(token: str) -> str:
    if len(token) <= 10:
        return "***"
    return f"{token[:4]}…{token[-4:]}"


async def _whoami(token: str) -> Optional[str]:
    """Validate via HF whoami-v2; return username or None."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://huggingface.co/api/whoami-v2", headers=headers)
            if resp.status_code == 200:
                payload = resp.json()
                if isinstance(payload, dict):
                    return payload.get("name") or payload.get("fullname") or "<unknown>"
    except Exception as e:
        logger.warning(f"whoami failed: {e}")
    return None


@router.get("/hf-token", response_model=HfTokenStatus)
async def get_hf_token_status():
    token, source = _current_token()
    if not token:
        return HfTokenStatus(has_token=False, source="none")
    username = await _whoami(token)
    return HfTokenStatus(
        has_token=True,
        source=source,
        masked=_mask(token),
        username=username,
    )


@router.post("/hf-token", response_model=HfTokenStatus)
async def set_hf_token(body: HfTokenUpdate):
    token = body.token.strip()
    if not token.startswith("hf_"):
        raise HTTPException(
            status_code=400,
            detail="Token must start with 'hf_' (HuggingFace user access token).",
        )
    username = await _whoami(token)
    if username is None:
        raise HTTPException(status_code=401, detail="Token rejected by HuggingFace whoami.")

    try:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(token + "\n")
        try:
            os.chmod(TOKEN_FILE, 0o600)
        except OSError:
            pass
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write token file: {e}")

    return HfTokenStatus(
        has_token=True,
        source="file",
        masked=_mask(token),
        username=username,
    )


@router.delete("/hf-token", response_model=HfTokenStatus)
async def delete_hf_token():
    if TOKEN_FILE.exists():
        try:
            TOKEN_FILE.unlink()
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Failed to remove token file: {e}")
    # Re-evaluate: maybe an env-var token still exists.
    token, source = _current_token()
    if token:
        username = await _whoami(token)
        return HfTokenStatus(has_token=True, source=source, masked=_mask(token), username=username)
    return HfTokenStatus(has_token=False, source="none")
