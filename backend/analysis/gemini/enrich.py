"""Top-level Gemini enrichment. Takes a PhaseAwareResult dict, runs K cluster-
characterization calls + batched flag-enrichment calls, merges results back in.

Results are cached at ~/.cache/data_viewer/phase_aware_gemini/responses/ keyed
on (dataset_id, task_name, algo_version, prompt_version).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from . import prompts as prompt_mod
from .client import GeminiClient, GeminiUnavailable
from .video_cache import get_episode_clip

logger = logging.getLogger(__name__)

CACHE_ROOT = Path(os.environ.get("DATA_VIEWER_CACHE", Path.home() / ".cache" / "data_viewer")) / "phase_aware_gemini" / "responses"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

MAX_VIDEOS_PER_CALL = 10  # Gemini 3 Flash hard limit
DEFAULT_FLAGGED_CAP = 20  # decision #4
# v3 cluster-char prompt: describe the medoid only, so we only need to send
# 1 medoid video per cluster + 1 contrast medoid per other cluster.
# Sending more wastes upload time and gives Gemini ammunition to violate the
# "do not name non-medoid members" rule by citing the extra videos.
CLUSTER_CHAR_TARGET_VIDEOS = 1
CLUSTER_CHAR_CONTRAST_VIDEOS = 1
ALGO_VERSION_KEY = "phase_aware_v1"  # bump when backend/analysis/phase_aware.py changes materially


class GeminiEnrichmentError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# Caching
# ----------------------------------------------------------------------------

def _cache_key(dataset_id: str, task_name: str, cohort_size: int) -> str:
    h = hashlib.sha256()
    h.update(dataset_id.encode())
    h.update(b"\x00")
    h.update(task_name.encode())
    h.update(b"\x00")
    h.update(f"cohort={cohort_size}".encode())
    h.update(b"\x00")
    h.update(ALGO_VERSION_KEY.encode())
    h.update(b"\x00")
    h.update(prompt_mod.prompt_version_hash().encode())
    return h.hexdigest()[:16]


def _cache_path(dataset_id: str, task_name: str, cohort_size: int) -> Path:
    return CACHE_ROOT / f"{_cache_key(dataset_id, task_name, cohort_size)}.json"


def load_cached(dataset_id: str, task_name: str, cohort_size: int) -> Optional[dict]:
    p = _cache_path(dataset_id, task_name, cohort_size)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save_cached(dataset_id: str, task_name: str, cohort_size: int, data: dict) -> None:
    p = _cache_path(dataset_id, task_name, cohort_size)
    p.write_text(json.dumps(data, indent=2, default=str))


def clear_cache(dataset_id: Optional[str] = None, task_name: Optional[str] = None,
                cohort_size: Optional[int] = None) -> int:
    """Clear the whole cache or a specific cache entry. When dataset+task are
    given without cohort, clears all cohort variants for that pair."""
    if dataset_id and task_name and cohort_size is not None:
        p = _cache_path(dataset_id, task_name, cohort_size)
        if p.exists():
            p.unlink()
            return 1
        return 0
    if dataset_id and task_name:
        # Clear all cohort variants by loading and checking provenance
        n = 0
        for p in CACHE_ROOT.glob("*.json"):
            try:
                content = json.loads(p.read_text())
                if content.get("task_name") == task_name:
                    p.unlink()
                    n += 1
            except Exception:
                continue
        return n
    n = 0
    for p in CACHE_ROOT.glob("*.json"):
        p.unlink()
        n += 1
    return n


# ----------------------------------------------------------------------------
# Candidate selection
# ----------------------------------------------------------------------------

@dataclass
class _Plan:
    cluster_calls: list[dict]  # [{target_cluster_id, target_ids, contrast_ids_by_cluster}]
    flag_batches: list[dict]   # [{normal_ids, flagged_episodes}]
    all_unique_ids: list[str]  # all episode_ids needing videos
    flagged_capped: bool
    flagged_shown: int
    flagged_total: int
    # episode_id -> {frames, num_cycles, raw_gripper_events, phase_action_summary}
    # Used by both cluster-char and flag-enrich renderers so both prompts can
    # ground their observations in the same telemetry.
    episode_contexts: dict[str, dict] = field(default_factory=dict)


def _select_targets_for_cluster(cluster: dict, n_videos: int) -> list[str]:
    """Pick up to n_videos episode_ids from a cluster, medoid first then fan out."""
    members = cluster["members"]
    medoid = cluster["medoid"]
    ordered = [medoid] + [m for m in members if m != medoid]
    return ordered[:n_videos]


def _plan_calls(result: dict, flagged_cap: int = DEFAULT_FLAGGED_CAP) -> _Plan:
    clusters = result.get("clusters", [])
    episodes = result.get("episodes", [])
    all_flagged = [e for e in episodes if e.get("anomaly", {}).get("is_anomaly")]
    flagged_shown = all_flagged[:flagged_cap]
    unique: set[str] = set()

    # Cluster-char calls
    cluster_calls = []
    for target in clusters:
        target_ids = _select_targets_for_cluster(target, CLUSTER_CHAR_TARGET_VIDEOS)
        contrast_by_cluster: dict[str, list[str]] = {}
        for other in clusters:
            if other["id"] == target["id"]:
                continue
            contrast_by_cluster[other["id"]] = _select_targets_for_cluster(other, CLUSTER_CHAR_CONTRAST_VIDEOS)
        total_videos = len(target_ids) + sum(len(v) for v in contrast_by_cluster.values())
        if total_videos > MAX_VIDEOS_PER_CALL:
            # Trim contrast videos to fit
            slack = MAX_VIDEOS_PER_CALL - len(target_ids)
            contrast_by_cluster = {k: v[:max(1, slack // max(1, len(contrast_by_cluster)))]
                                   for k, v in contrast_by_cluster.items()}
        cluster_calls.append({
            "target_cluster_id": target["id"],
            "target_cluster_size": len(target["members"]),
            "target_cluster_stat_label": target.get("label", "(auto)"),
            "target_dominant_features": target.get("dominant_features", []),
            "target_ids": target_ids,
            "medoid_episode_id": target.get("medoid", target_ids[0] if target_ids else ""),
            "contrast_by_cluster": contrast_by_cluster,
        })
        for v in target_ids:
            unique.add(v)
        for vs in contrast_by_cluster.values():
            for v in vs:
                unique.add(v)

    # Flag-enrich calls (batched).
    # Each call: 3 normal reference (cluster medoids) + up to 7 flagged.
    normal_ids = [c["medoid"] for c in clusters][:3]
    if not normal_ids and episodes:
        normal_ids = [e["episode_id"] for e in episodes[:3] if not e.get("anomaly", {}).get("is_anomaly")]

    for v in normal_ids:
        unique.add(v)

    # Build quick lookup: episode_id -> {frames, num_cycles, raw_gripper_events, phase_action_summary}
    def _ep_ctx(ep: dict) -> dict:
        return {
            "frames": ep.get("frames", 0),
            "num_cycles": ep.get("num_cycles", 0),
            "raw_gripper_events": ep.get("raw_gripper_events", []),
            "phase_action_summary": ep.get("phase_action_summary", []),
        }

    ep_ctx_by_id = {e["episode_id"]: _ep_ctx(e) for e in episodes}

    flag_batches = []
    per_batch = MAX_VIDEOS_PER_CALL - len(normal_ids)
    if per_batch < 1:
        per_batch = 1
    for i in range(0, len(flagged_shown), per_batch):
        batch = flagged_shown[i:i + per_batch]
        flag_batches.append({
            "normal_ids": normal_ids,
            "normal_contexts": {eid: ep_ctx_by_id.get(eid, {}) for eid in normal_ids},
            "flagged_episodes": [
                {
                    "episode_id": e["episode_id"],
                    **_ep_ctx(e),
                    "reasons": [
                        {k: v for k, v in r.items() if k in ("signal", "phase", "cycle", "explanation")}
                        for r in e.get("anomaly", {}).get("reasons", [])
                    ],
                }
                for e in batch
            ],
        })
        for e in batch:
            unique.add(e["episode_id"])

    return _Plan(
        episode_contexts=ep_ctx_by_id,
        cluster_calls=cluster_calls,
        flag_batches=flag_batches,
        all_unique_ids=sorted(unique),
        flagged_capped=len(all_flagged) > flagged_cap,
        flagged_shown=len(flagged_shown),
        flagged_total=len(all_flagged),
    )


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

async def enrich_with_gemini(
    result: dict,
    *,
    dataset_id: str,
    task_name: str,
    repo_id: str,
    video_path_template: str,
    flagged_cap: int = DEFAULT_FLAGGED_CAP,
    use_cache: bool = True,
    cohort_size: Optional[int] = None,
    on_progress: Optional["Callable[[str, dict], Awaitable[None]]"] = None,
) -> dict:
    """Enrich a phase-aware result dict with Gemini observations.

    Returns the same dict shape with these added fields:
      - `gemini`: {enriched: bool, cached: bool, timings: {...}, token_usage: {...},
                   flagged_capped: bool, flagged_shown: int, flagged_total: int,
                   error: Optional[str]}
      - on each cluster: `gemini_label`, `gemini_description`, `gemini_evidence`, `gemini_confidence`
      - on each episode: `gemini_severity`, `gemini_confirmation`, `gemini_observations`
      - novel Gemini anomalies are appended to the episode's `anomaly.reasons`
        with signal="gemini"
    """
    t0_total = time.time()

    # Infer cohort_size from the result if caller didn't provide it explicitly.
    if cohort_size is None:
        cohort_size = int(result.get("cohort_size", len(result.get("episodes", []))))

    async def _emit(event_type: str, data: dict) -> None:
        if on_progress is not None:
            try:
                await on_progress(event_type, data)
            except Exception as e:
                logger.warning(f"on_progress callback raised: {e}")

    if use_cache:
        cached = load_cached(dataset_id, task_name, cohort_size)
        if cached is not None:
            cached["gemini"] = cached.get("gemini", {})
            cached["gemini"]["cached"] = True
            cached["gemini"]["timings"] = {"total_s": 0.0, "cache_hit": True}
            # Replay cached enrichment as progress events so the UI gets
            # uniform updates whether it's a hot run or a cache hit.
            for c in cached.get("clusters", []):
                if c.get("gemini_label") or c.get("gemini_description"):
                    await _emit("cluster", {
                        "cluster_id": c["id"],
                        "gemini_label": c.get("gemini_label"),
                        "gemini_description": c.get("gemini_description"),
                        "gemini_evidence": c.get("gemini_evidence"),
                        "gemini_confidence": c.get("gemini_confidence"),
                    })
            cache_episode_payload = []
            for e in cached.get("episodes", []):
                if e.get("gemini_severity") or e.get("gemini_confirmation"):
                    cache_episode_payload.append({
                        "episode_id": e["episode_id"],
                        "gemini_severity": e.get("gemini_severity"),
                        "gemini_confirmation": e.get("gemini_confirmation"),
                        "gemini_observations": e.get("gemini_observations") or [],
                        "anomaly_reasons": e.get("anomaly", {}).get("reasons", []),
                    })
            if cache_episode_payload:
                await _emit("flag_batch", {"episodes": cache_episode_payload, "cached": True})
            return cached

    # Plan calls
    plan = _plan_calls(result, flagged_cap=flagged_cap)
    if not plan.cluster_calls and not plan.flag_batches:
        return {**result, "gemini": {"enriched": False, "reason": "nothing to enrich"}}

    # Clients + video cache
    try:
        client = GeminiClient()
    except GeminiUnavailable as e:
        return {**result, "gemini": {"enriched": False, "error": f"Gemini unavailable: {e}"}}

    timings: dict[str, Any] = {"clip_meta_s": 0.0, "clip_download_s": 0.0, "clip_ffmpeg_s": 0.0,
                               "upload_s": 0.0, "cluster_calls_s": [], "flag_calls_s": [],
                               "parse_merge_s": 0.0}

    # 1. Ensure clips for every unique episode_id we'll reference.
    ep_idx_map: dict[str, int] = {}
    for eid in plan.all_unique_ids:
        if eid.startswith("episode_"):
            try:
                ep_idx_map[eid] = int(eid.split("_")[-1])
            except ValueError:
                pass

    t0 = time.time()
    clip_timing: dict[str, float] = {}
    clip_paths: dict[str, Path] = {}
    # Parallelize clip extraction (bounded)
    sem = asyncio.Semaphore(4)

    clips_done = [0]
    total_clips = len(ep_idx_map)

    async def _ensure_clip(eid: str, idx: int):
        async with sem:
            p = await get_episode_clip(repo_id, idx, video_path_template, timing=clip_timing)
            if p is not None:
                clip_paths[eid] = p
            clips_done[0] += 1
            await _emit("clip_progress", {
                "stage": "extracted", "episode_id": eid,
                "done": clips_done[0], "total": total_clips,
            })

    await asyncio.gather(*[_ensure_clip(eid, idx) for eid, idx in ep_idx_map.items()])
    timings["clip_meta_s"] = round(clip_timing.get("meta", 0.0), 2)
    timings["clip_download_s"] = round(clip_timing.get("chunk_dl", 0.0), 2)
    timings["clip_ffmpeg_s"] = round(clip_timing.get("ffmpeg", 0.0), 2)
    timings["clip_total_wall_s"] = round(time.time() - t0, 2)
    timings["clips_prepared"] = len(clip_paths)

    if not clip_paths:
        return {**result, "gemini": {"enriched": False, "error": "No clips could be prepared"}}

    # 2. Upload every clip once (dedup within process)
    t0 = time.time()
    uploaded: dict[str, Any] = {}

    uploads_done = [0]
    total_uploads = len(clip_paths)

    async def _upload(eid: str, p: Path):
        up = await client.upload_file(p)
        uploaded[eid] = up
        uploads_done[0] += 1
        await _emit("clip_progress", {
            "stage": "uploaded", "episode_id": eid,
            "done": uploads_done[0], "total": total_uploads,
        })

    await asyncio.gather(*[_upload(eid, p) for eid, p in clip_paths.items()])
    timings["upload_s"] = round(time.time() - t0, 2)
    timings["uploads"] = len(uploaded)

    cluster_char_tpl = prompt_mod.cluster_char()
    flag_enrich_tpl = prompt_mod.flag_enrich()

    total_tokens = {"prompt": 0, "response": 0, "thought": 0, "total": 0}
    errors: list[str] = []

    # 3. Cluster-char calls (in parallel, but throttled)
    cluster_results: dict[str, dict] = {}

    async def _run_cluster_call(call: dict):
        target_ids = [eid for eid in call["target_ids"] if eid in uploaded]
        contrast_flat = []
        for cid, eids in call["contrast_by_cluster"].items():
            for eid in eids:
                if eid in uploaded:
                    contrast_flat.append(uploaded[eid])
        if not target_ids:
            return
        files = [uploaded[eid] for eid in target_ids] + contrast_flat
        if len(files) > MAX_VIDEOS_PER_CALL:
            files = files[:MAX_VIDEOS_PER_CALL]

        cohort = result.get("cohort_size", 0)
        prompt = prompt_mod.render_cluster_char(
            template=cluster_char_tpl,
            task_name=task_name,
            fps=float(result.get("fps", 10.0)),
            target_cluster_id=call["target_cluster_id"],
            target_cluster_size=call["target_cluster_size"],
            target_cluster_stat_label=call["target_cluster_stat_label"],
            target_dominant_features=call["target_dominant_features"],
            cohort_size=cohort,
            medoid_episode_id=call.get("medoid_episode_id", ""),
            target_episode_ids=target_ids,
            contrast_episode_ids_by_cluster={
                cid: [eid for eid in eids if eid in uploaded]
                for cid, eids in call["contrast_by_cluster"].items()
            },
            episode_contexts=plan.episode_contexts,
        )
        t = time.time()
        try:
            out = await client.generate_json(
                uploaded_files=files,
                system_instruction=cluster_char_tpl.system,
                user_prompt=prompt,
                response_schema=cluster_char_tpl.response_schema,
            )
            dt = round(time.time() - t, 2)
            timings["cluster_calls_s"].append({"cluster_id": call["target_cluster_id"], "wall_s": dt})
            parsed = out["parsed"]
            cluster_results[call["target_cluster_id"]] = parsed
            for k in ("prompt_tokens", "response_tokens", "thought_tokens", "total_tokens"):
                v = out["usage"].get(k)
                if v:
                    short = k.replace("_tokens", "")
                    total_tokens[short] = total_tokens.get(short, 0) + v
            await _emit("cluster", {
                "cluster_id": call["target_cluster_id"],
                "gemini_label": parsed.get("proposed_label"),
                "gemini_description": parsed.get("description"),
                "gemini_evidence": parsed.get("evidence", []),
                "gemini_confidence": parsed.get("confidence"),
            })
        except Exception as e:
            logger.error(f"cluster-char call {call['target_cluster_id']} failed: {e}", exc_info=True)
            errors.append(f"cluster {call['target_cluster_id']}: {e}")

    await asyncio.gather(*[_run_cluster_call(c) for c in plan.cluster_calls])

    # 4. Flag-enrich calls (in parallel, throttled)
    flag_results: dict[str, dict] = {}  # episode_id -> {confirmation, severity, novel_observations}

    async def _run_flag_call(batch: dict):
        normal_files = [uploaded[eid] for eid in batch["normal_ids"] if eid in uploaded]
        flagged_eps = [fe for fe in batch["flagged_episodes"] if fe["episode_id"] in uploaded]
        if not flagged_eps:
            return
        flagged_files = [uploaded[fe["episode_id"]] for fe in flagged_eps]
        files = normal_files + flagged_files
        if len(files) > MAX_VIDEOS_PER_CALL:
            files = files[:MAX_VIDEOS_PER_CALL]
            flagged_eps = flagged_eps[:MAX_VIDEOS_PER_CALL - len(normal_files)]
        prompt = prompt_mod.render_flag_enrich(
            template=flag_enrich_tpl,
            task_name=task_name,
            fps=float(result.get("fps", 10.0)),
            normal_episode_ids=[eid for eid in batch["normal_ids"] if eid in uploaded],
            normal_episode_contexts=batch.get("normal_contexts", {}),
            flagged_episodes=flagged_eps,
        )
        t = time.time()
        try:
            out = await client.generate_json(
                uploaded_files=files,
                system_instruction=flag_enrich_tpl.system,
                user_prompt=prompt,
                response_schema=flag_enrich_tpl.response_schema,
            )
            dt = round(time.time() - t, 2)
            timings["flag_calls_s"].append({"batch_size": len(flagged_eps), "wall_s": dt})
            parsed = out["parsed"]
            episodes_parsed = parsed.get("episodes", []) if isinstance(parsed, dict) else []
            logger.info(
                f"flag_enrich batch ({len(flagged_eps)} flagged → {len(episodes_parsed)} returned): "
                f"{[ep.get('episode_id') for ep in episodes_parsed]}"
            )
            for ep in episodes_parsed:
                flag_results[ep["episode_id"]] = ep
            for k in ("prompt_tokens", "response_tokens", "thought_tokens", "total_tokens"):
                v = out["usage"].get(k)
                if v:
                    short = k.replace("_tokens", "")
                    total_tokens[short] = total_tokens.get(short, 0) + v
            await _emit("flag_batch", {
                "episodes": [
                    {
                        "episode_id": ep["episode_id"],
                        "gemini_severity": ep.get("severity"),
                        "gemini_confirmation": ep.get("confirmation"),
                        "gemini_observations": ep.get("novel_observations") or [],
                    }
                    for ep in episodes_parsed
                ],
            })
        except Exception as e:
            logger.error(f"flag-enrich call failed: {e}", exc_info=True)
            errors.append(f"flag batch: {e}")

    await asyncio.gather(*[_run_flag_call(b) for b in plan.flag_batches])

    # 5. Merge results
    t0 = time.time()
    out_result = {**result}
    out_result["clusters"] = [dict(c) for c in result.get("clusters", [])]
    out_result["episodes"] = [dict(e) for e in result.get("episodes", [])]

    for c in out_result["clusters"]:
        info = cluster_results.get(c["id"])
        if info:
            c["gemini_label"] = info.get("proposed_label")
            c["gemini_description"] = info.get("description")
            c["gemini_evidence"] = info.get("evidence", [])
            c["gemini_confidence"] = info.get("confidence")

    for e in out_result["episodes"]:
        info = flag_results.get(e["episode_id"])
        if info:
            e["gemini_severity"] = info.get("severity")
            e["gemini_confirmation"] = info.get("confirmation")
            novel = info.get("novel_observations", []) or []
            e["gemini_observations"] = novel
            # Append novel Gemini observations as AnomalyReason entries (signal="gemini")
            reasons = e.setdefault("anomaly", {}).setdefault("reasons", [])
            for n in novel:
                reasons.append({
                    "signal": "gemini",
                    "phase": n.get("phase"),
                    "cycle": n.get("cycle"),
                    "feature": None,
                    "magnitude": 0.0,
                    "explanation": f"AI-observed at {n.get('timestamp','?')}: {n.get('observation','')}",
                })
        else:
            e["gemini_observations"] = []

    timings["parse_merge_s"] = round(time.time() - t0, 2)
    timings["total_s"] = round(time.time() - t0_total, 2)

    out_result["gemini"] = {
        "enriched": True,
        "cached": False,
        "timings": timings,
        "token_usage": total_tokens,
        "flagged_shown": plan.flagged_shown,
        "flagged_total": plan.flagged_total,
        "flagged_capped": plan.flagged_capped,
        "errors": errors,
        "prompt_version": prompt_mod.prompt_version_hash(),
    }

    if not errors:
        _save_cached(dataset_id, task_name, cohort_size, out_result)

    return out_result
