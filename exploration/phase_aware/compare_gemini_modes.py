#!/usr/bin/env python3
"""A/B comparison: exhaustive vs flagged-only Gemini enrichment.

Mode A (exhaustive):
  Upload all 38 episode clips. Pick 3 normals as reference. Ask Gemini to
  independently rate the remaining 35 as normal/stylistic/suspicious/mistake.
  (Independent of our statistical flags.)

Mode B (flagged-only, current production):
  Upload 14 clips (cluster medoids + unique flagged). Ask Gemini to confirm
  our stat flags and surface novel observations within the flagged set.

For both, record:
  - Latency breakdown (clip prep, upload, per-call wall, total)
  - Token usage
  - Flag set (episodes Gemini rated suspicious/mistake)
  - Overlap with our statistical flags
  - Novel findings (episodes Gemini flagged that stats didn't)

Prereq: backend at localhost:8000; HF parquet + video files either cached on
disk or reachable via token.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

HERE = Path(__file__).parent
OUT = HERE / "compare_output"
OUT.mkdir(exist_ok=True)

TASK_NAME = (
    "put the white mug on the left plate and put the yellow and white mug "
    "on the right plate"
)
DATASET_ID = "libero"
REPO_ID = "lerobot/libero"
API = "http://localhost:8000/api"


def fetch_phase_aware() -> dict:
    import urllib.parse
    url = (
        f"{API}/datasets/{DATASET_ID}/analysis/phase-aware"
        f"?task_name={urllib.parse.quote(TASK_NAME)}"
    )
    with urllib.request.urlopen(url, timeout=300) as r:
        return json.loads(r.read())


async def _load_info() -> dict:
    from huggingface_hub import hf_hub_download
    p = await asyncio.to_thread(hf_hub_download, REPO_ID, "meta/info.json", repo_type="dataset")
    return json.loads(Path(p).read_text())


async def run_mode_B(result: dict, video_path_template: str) -> dict:
    """Current production pipeline: cluster-char + flag-enrich on flagged subset."""
    from analysis.gemini.enrich import clear_cache, enrich_with_gemini
    # Clear Gemini cache for fair comparison
    clear_cache(DATASET_ID, TASK_NAME)
    t0 = time.time()
    enriched = await enrich_with_gemini(
        result,
        dataset_id=DATASET_ID,
        task_name=TASK_NAME,
        repo_id=REPO_ID,
        video_path_template=video_path_template,
        use_cache=False,  # force cold run for fair latency
    )
    wall = round(time.time() - t0, 2)
    g = enriched.get("gemini", {})
    return {
        "mode": "B_flagged_only",
        "wall_s": wall,
        "timings": g.get("timings", {}),
        "tokens": g.get("token_usage", {}),
        "errors": g.get("errors", []),
        "episodes": enriched.get("episodes", []),
        "clusters": enriched.get("clusters", []),
    }


async def run_mode_A(result: dict, video_path_template: str) -> dict:
    """Exhaustive: upload all 38 clips, rate them all independently."""
    from analysis.gemini.client import GeminiClient
    from analysis.gemini import prompts as prompt_mod
    from analysis.gemini.video_cache import get_episode_clip

    all_eps = result.get("episodes", [])
    clusters = result.get("clusters", [])

    # Pick 3 reference normals = cluster medoids (not flagged)
    normal_ids = []
    for c in clusters:
        m = c["medoid"]
        is_flagged = any(e["episode_id"] == m and e["anomaly"]["is_anomaly"] for e in all_eps)
        if not is_flagged:
            normal_ids.append(m)
        if len(normal_ids) >= 3:
            break
    # pad with unflagged episodes if K < 3
    if len(normal_ids) < 3:
        for e in all_eps:
            if not e["anomaly"]["is_anomaly"] and e["episode_id"] not in normal_ids:
                normal_ids.append(e["episode_id"])
                if len(normal_ids) >= 3:
                    break

    unlabeled_ids = [e["episode_id"] for e in all_eps if e["episode_id"] not in normal_ids]

    all_ids = normal_ids + unlabeled_ids
    ep_idx = {eid: int(eid.split("_")[-1]) for eid in all_ids}

    print(f"  Mode A cohort: {len(all_ids)} total ({len(normal_ids)} ref + {len(unlabeled_ids)} unlabeled)")

    timings = {"clip_meta_s": 0.0, "clip_download_s": 0.0, "clip_ffmpeg_s": 0.0,
               "upload_s": 0.0, "call_wall_s_per_batch": [], "parse_merge_s": 0.0}

    # 1. Clip prep (parallelized)
    t_a = time.time()
    clip_timing: dict[str, float] = {}
    clip_paths: dict[str, Path] = {}
    sem = asyncio.Semaphore(4)

    async def _ensure_clip(eid: str, idx: int):
        async with sem:
            p = await get_episode_clip(REPO_ID, idx, video_path_template, timing=clip_timing)
            if p is not None:
                clip_paths[eid] = p

    await asyncio.gather(*[_ensure_clip(eid, idx) for eid, idx in ep_idx.items()])
    timings["clip_meta_s"] = round(clip_timing.get("meta", 0.0), 2)
    timings["clip_download_s"] = round(clip_timing.get("chunk_dl", 0.0), 2)
    timings["clip_ffmpeg_s"] = round(clip_timing.get("ffmpeg", 0.0), 2)
    timings["clip_total_wall_s"] = round(time.time() - t_a, 2)

    # 2. Upload
    client = GeminiClient()
    t_a = time.time()
    uploaded: dict[str, Any] = {}

    async def _up(eid: str, p: Path):
        uploaded[eid] = await client.upload_file(p)

    await asyncio.gather(*[_up(eid, p) for eid, p in clip_paths.items()])
    timings["upload_s"] = round(time.time() - t_a, 2)
    timings["uploads"] = len(uploaded)

    # 3. Batched exhaustive scan: 3 normals + 7 unlabeled per batch
    tpl = prompt_mod.exhaustive_scan()
    BATCH = 7
    batches = [unlabeled_ids[i:i + BATCH] for i in range(0, len(unlabeled_ids), BATCH)]
    results_by_ep: dict[str, dict] = {}
    tokens = {"prompt": 0, "response": 0, "thought": 0, "total": 0}
    errors: list[str] = []

    async def _run_batch(batch_ids: list[str]):
        normal_files = [uploaded[e] for e in normal_ids if e in uploaded]
        batch_files = [uploaded[e] for e in batch_ids if e in uploaded]
        files = normal_files + batch_files
        if len(files) > 10:
            files = files[:10]
        prompt = prompt_mod.render_exhaustive_scan(
            template=tpl,
            task_name=TASK_NAME,
            normal_episode_ids=[e for e in normal_ids if e in uploaded],
            unlabeled_episode_ids=batch_ids,
        )
        t0 = time.time()
        try:
            out = await client.generate_json(
                uploaded_files=files,
                system_instruction=tpl.system,
                user_prompt=prompt,
                response_schema=tpl.response_schema,
            )
            dt = round(time.time() - t0, 2)
            timings["call_wall_s_per_batch"].append({"batch_size": len(batch_ids), "wall_s": dt})
            parsed = out["parsed"]
            for ep in parsed.get("episodes", []):
                results_by_ep[ep["episode_id"]] = ep
            for k in ("prompt_tokens", "response_tokens", "thought_tokens", "total_tokens"):
                v = out["usage"].get(k)
                if v:
                    short = k.replace("_tokens", "")
                    tokens[short] = tokens.get(short, 0) + v
        except Exception as e:
            errors.append(f"batch {batch_ids[0]}…: {e}")

    t_a = time.time()
    await asyncio.gather(*[_run_batch(b) for b in batches])
    timings["all_calls_wall_s"] = round(time.time() - t_a, 2)
    timings["num_batches"] = len(batches)

    t_a = time.time()
    # Rebuild enriched episodes list
    enriched_eps = []
    for e in all_eps:
        eid = e["episode_id"]
        entry = {**e}
        if eid in normal_ids:
            entry["gemini_exhaustive_rating"] = "reference"
            entry["gemini_exhaustive_observation"] = "reference video"
        else:
            r = results_by_ep.get(eid)
            entry["gemini_exhaustive_rating"] = r["rating"] if r else None
            entry["gemini_exhaustive_observation"] = r["observation"] if r else None
        enriched_eps.append(entry)
    timings["parse_merge_s"] = round(time.time() - t_a, 2)

    return {
        "mode": "A_exhaustive",
        "wall_s": round(sum([timings["clip_total_wall_s"], timings["upload_s"], timings["all_calls_wall_s"], timings["parse_merge_s"]]), 2),
        "timings": timings,
        "tokens": tokens,
        "errors": errors,
        "episodes": enriched_eps,
        "normal_reference_ids": normal_ids,
    }


def compare(result_A: dict, result_B: dict, stat_flag_ids: list[str]) -> dict:
    # Mode A: episodes Gemini rated suspicious OR mistake (from exhaustive scan)
    A_flagged = [e for e in result_A["episodes"]
                 if e.get("gemini_exhaustive_rating") in ("suspicious", "mistake")]
    A_flagged_ids = [e["episode_id"] for e in A_flagged]

    # Mode B: our stat-flagged episodes with severity suspicious/mistake per Gemini
    B_gemini_sus_or_mistake = [e for e in result_B["episodes"]
                               if e.get("gemini_severity") in ("suspicious", "mistake")]
    B_flagged_ids = [e["episode_id"] for e in B_gemini_sus_or_mistake]

    # Statistically flagged regardless of Gemini
    stat_set = set(stat_flag_ids)
    A_set = set(A_flagged_ids)
    B_set = set(B_flagged_ids)

    return {
        "stat_flagged": sorted(stat_set),
        "A_exhaustive_gemini_flagged": sorted(A_set),
        "B_flagged_only_gemini_flagged": sorted(B_set),
        # What A found that B couldn't have (not in stat_set)
        "A_novel_findings": sorted(A_set - stat_set),
        # What stats caught that Gemini in A disagrees with
        "A_disagrees_with_stats": sorted(stat_set - A_set),
        # What stats caught AND Gemini in A confirms
        "A_confirms_stats": sorted(stat_set & A_set),
        "counts": {
            "stat": len(stat_set),
            "A": len(A_set),
            "B": len(B_set),
            "A_novel": len(A_set - stat_set),
        },
    }


async def main() -> int:
    print("=" * 78)
    print("Gemini mode A/B comparison on Libero mug-left-plate task")
    print("=" * 78)

    # Get phase-aware baseline (shared between both modes)
    print("\n[baseline] Fetching phase-aware result...")
    t0 = time.time()
    result = fetch_phase_aware()
    print(f"  done in {time.time()-t0:.2f}s — cohort={result['cohort_size']}, "
          f"clusters={len(result['clusters'])}, "
          f"flagged={sum(1 for e in result['episodes'] if e['anomaly']['is_anomaly'])}")

    stat_flag_ids = [e["episode_id"] for e in result["episodes"] if e["anomaly"]["is_anomaly"]]
    print(f"  stat-flagged: {stat_flag_ids}")

    info = await _load_info()
    vpt = info.get("video_path")
    if not vpt:
        print("no video_path — abort")
        return 1

    print("\n[Mode A] Exhaustive scan — all 38 episodes...")
    t_a0 = time.time()
    A = await run_mode_A(result, vpt)
    A["end_to_end_wall_s"] = round(time.time() - t_a0, 2)
    print(f"  A done in {A['end_to_end_wall_s']}s · timings: {json.dumps(A['timings'])}")
    print(f"  A tokens: {A['tokens']}")

    print("\n[Mode B] Flagged-only — cluster + flag enrichment...")
    t_b0 = time.time()
    B = await run_mode_B(result, vpt)
    B["end_to_end_wall_s"] = round(time.time() - t_b0, 2)
    print(f"  B done in {B['end_to_end_wall_s']}s · timings: {json.dumps(B['timings'])}")
    print(f"  B tokens: {B['tokens']}")

    comp = compare(A, B, stat_flag_ids)
    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    print(json.dumps(comp, indent=2))

    out = {
        "stat_flag_ids": stat_flag_ids,
        "mode_A": {
            "wall_s": A["wall_s"],
            "end_to_end_wall_s": A["end_to_end_wall_s"],
            "timings": A["timings"],
            "tokens": A["tokens"],
            "errors": A["errors"],
            "ratings": {
                e["episode_id"]: {
                    "rating": e.get("gemini_exhaustive_rating"),
                    "observation": e.get("gemini_exhaustive_observation"),
                }
                for e in A["episodes"]
            },
        },
        "mode_B": {
            "wall_s": B["wall_s"],
            "end_to_end_wall_s": B["end_to_end_wall_s"],
            "timings": B["timings"],
            "tokens": B["tokens"],
            "errors": B["errors"],
            "severities": {
                e["episode_id"]: {
                    "severity": e.get("gemini_severity"),
                    "confirmation": e.get("gemini_confirmation"),
                }
                for e in B["episodes"] if e["anomaly"]["is_anomaly"]
            },
        },
        "comparison": comp,
    }
    out_path = OUT / "compare_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
