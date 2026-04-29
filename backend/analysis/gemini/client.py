"""Thin wrapper over google-genai SDK for uploading video files and running
constrained-JSON generation. Keeps the rest of the module SDK-agnostic."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")


def _read_gemini_token_file() -> Optional[str]:
    """Read user-saved Gemini key from disk if env var isn't set.

    Mirrors the HF-token resolution pattern: env var first, then a small
    file under ~/.config/data_viewer/. Token is written by the
    /api/settings/gemini-token endpoint.
    """
    p = Path.home() / ".config" / "data_viewer" / "gemini_key"
    if p.exists():
        try:
            t = p.read_text().strip()
            if t:
                return t
        except OSError:
            pass
    return None


def _resolve_api_key() -> str:
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or _read_gemini_token_file()
        or ""
    )


class GeminiUnavailable(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        try:
            from google import genai  # noqa: F401
        except ImportError as e:
            raise GeminiUnavailable(f"google-genai not installed: {e}")
        resolved = api_key or _resolve_api_key()
        if not resolved:
            raise GeminiUnavailable(
                "Gemini API key not configured. Set GEMINI_API_KEY or save a key "
                "via the homepage settings."
            )
        self.api_key = resolved
        self.model = model
        self._client = None
        self._uploaded_cache: dict[Path, Any] = {}

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def upload_file(self, path: Path) -> Any:
        """Upload a file to Gemini Files API. De-duplicates by path within the
        process (same file won't be re-uploaded across calls in one run)."""
        if path in self._uploaded_cache:
            return self._uploaded_cache[path]

        def _up():
            f = self.client.files.upload(file=str(path))
            # Wait until ACTIVE
            for _ in range(60):
                got = self.client.files.get(name=f.name)
                if got.state.name == "ACTIVE":
                    return got
                if got.state.name == "FAILED":
                    raise RuntimeError(f"upload FAILED: {got}")
                time.sleep(1)
            raise TimeoutError(f"upload never reached ACTIVE: {f.name}")

        uploaded = await asyncio.to_thread(_up)
        self._uploaded_cache[path] = uploaded
        return uploaded

    async def generate_json(
        self,
        uploaded_files: list[Any],
        system_instruction: str,
        user_prompt: str,
        response_schema: dict,
    ) -> dict:
        """Call Gemini with a list of uploaded videos + text prompt, request
        JSON output, and parse + return."""
        from google.genai import types

        content_parts = [
            types.Part(file_data=types.FileData(file_uri=f.uri, mime_type="video/mp4"))
            for f in uploaded_files
        ]
        content_parts.append(types.Part.from_text(text=user_prompt))

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        def _call():
            return self.client.models.generate_content(
                model=self.model,
                contents=content_parts,
                config=config,
            )

        resp = await asyncio.to_thread(_call)
        text = resp.text or ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Gemini returned non-JSON: {text[:500]}")
            raise
        usage = getattr(resp, "usage_metadata", None)
        usage_dict = {}
        if usage is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "thought_tokens": getattr(usage, "thoughts_token_count", None),
                "response_tokens": getattr(usage, "candidates_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
            }
        return {"parsed": parsed, "usage": usage_dict, "raw_text": text}
