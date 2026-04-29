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
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

TOKEN_FILE = Path.home() / ".huggingface" / "token"
GEMINI_TOKEN_FILE = Path.home() / ".config" / "data_viewer" / "gemini_key"


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


# === Gemini API Key ===


class GeminiKeyStatus(BaseModel):
    has_key: bool
    source: str  # "env" | "file" | "none"
    masked: Optional[str] = None


class GeminiKeyUpdate(BaseModel):
    key: str = Field(..., min_length=4, max_length=200)


def _read_gemini_key_from_file() -> Optional[str]:
    if GEMINI_TOKEN_FILE.exists():
        try:
            t = GEMINI_TOKEN_FILE.read_text().strip()
            if t:
                return t
        except OSError:
            pass
    return None


def _current_gemini_key() -> tuple[Optional[str], str]:
    t = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if t:
        return t, "env"
    t = _read_gemini_key_from_file()
    if t:
        return t, "file"
    return None, "none"


async def _validate_gemini_key(key: str) -> tuple[bool, Optional[str]]:
    """Best-effort key check.

    Hits the public ListModels endpoint. We treat 401/403 as a hard reject
    (the key is genuinely bad), but any other response — 200, 4xx, network
    error — is "inconclusive": save the key and let the actual Gemini call
    surface a richer error if it fails. This avoids false-rejections from
    transient network issues or restricted-API keys that still work for
    Gemini-only access.

    Returns (ok, reason_if_rejected).
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params={"key": key})
            if resp.status_code in (401, 403):
                msg = ""
                try:
                    err = resp.json().get("error", {})
                    msg = err.get("message") or ""
                except Exception:
                    pass
                return False, msg or f"HTTP {resp.status_code}"
            return True, None
    except Exception as e:
        logger.warning(f"gemini key check failed: {e}")
        return True, None  # Inconclusive — accept and let real calls surface errors


@router.get("/gemini-token", response_model=GeminiKeyStatus)
async def get_gemini_status():
    key, source = _current_gemini_key()
    if not key:
        return GeminiKeyStatus(has_key=False, source="none")
    return GeminiKeyStatus(has_key=True, source=source, masked=_mask(key))


@router.post("/gemini-token", response_model=GeminiKeyStatus)
async def set_gemini_key(body: GeminiKeyUpdate):
    key = body.key.strip()
    if len(key) < 20:
        raise HTTPException(
            status_code=400,
            detail="Gemini key looks too short. Get one from https://aistudio.google.com/apikey.",
        )
    ok, reason = await _validate_gemini_key(key)
    if not ok:
        raise HTTPException(
            status_code=401,
            detail=f"Key rejected by Google API: {reason}",
        )
    try:
        GEMINI_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        GEMINI_TOKEN_FILE.write_text(key + "\n")
        try:
            os.chmod(GEMINI_TOKEN_FILE, 0o600)
        except OSError:
            pass
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write key file: {e}")
    return GeminiKeyStatus(has_key=True, source="file", masked=_mask(key))


@router.delete("/gemini-token", response_model=GeminiKeyStatus)
async def delete_gemini_key():
    if GEMINI_TOKEN_FILE.exists():
        try:
            GEMINI_TOKEN_FILE.unlink()
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Failed to remove key file: {e}")
    key, source = _current_gemini_key()
    if key:
        return GeminiKeyStatus(has_key=True, source=source, masked=_mask(key))
    return GeminiKeyStatus(has_key=False, source="none")


# === HF token DELETE (kept below Gemini for grouping) ===


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
