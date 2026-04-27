#!/usr/bin/env python3
"""Run Mode B (phase-aware + Gemini enrichment with gripper events + phase
timings) on the first 10 episodes of UMI Cup in the Wild.

UMI differs from Libero:
  - No `action` column. We synthesize one from state deltas.
  - Gripper lives in its own `gripper_width` column (physical width in meters).
  - Task is single pick-place (expect num_cycles=1, not 2).
  - 1 single big parquet file (not sharded per chunk).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from analysis import analyze_task
from analysis.gemini import enrich_with_gemini, GeminiEnrichmentError
from analysis.gemini.enrich import clear_cache

HERE = Path(__file__).parent
OUT = HERE / "umi_output"
OUT.mkdir(exist_ok=True)

HF_TOKEN = "REDACTED-HF-TOKEN"
REPO_ID = "lerobot/umi_cup_in_the_wild"
TASK_NAME = "Put the cup on the plate."
DATASET_ID = "umi_cup_in_the_wild"
TARGET_EPISODES = list(range(10))  # first 10


def load_umi_first_10() -> list[dict]:
    """Download UMI's single parquet file and slice out the first 10 episodes."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO_ID, "data/chunk-000/file-000.parquet",
                           repo_type="dataset", token=HF_TOKEN)
    cols = ["episode_index", "frame_index", "observation.state", "gripper_width"]
    df = pq.read_table(path, columns=cols).to_pandas()

    out: list[dict] = []
    for gid in TARGET_EPISODES:
        rows = df[df.episode_index == gid].sort_values("frame_index")
        if rows.empty:
            continue
        state_rows = np.stack([np.asarray(s, dtype=float) for s in rows["observation.state"].values])
        # observation.state is 7-dim; layout is [x, y, z, rx, ry, rz, ?]. We
        # use the first 6 (xyz + rpy) for arm motion. Gripper lives in its
        # own column.
        gripper_width = np.asarray(rows["gripper_width"].values, dtype=float).reshape(-1)

        # Synthesize action = delta-state + gripper proxy:
        #   action[:, 0:3] = delta position
        #   action[:, 3:6] = delta rotation
        #   action[:, 6]   = -gripper_width  (flip so narrow=low, matching Libero
        #                                     convention where close command is low)
        T = len(state_rows)
        action = np.zeros((T, 7), dtype=float)
        if T > 1:
            action[1:, 0:3] = np.diff(state_rows[:, 0:3], axis=0)
            action[1:, 3:6] = np.diff(state_rows[:, 3:6], axis=0)
        # Copy first frame's delta as zeros (np.zeros default); that's fine.
        action[:, 6] = -gripper_width

        # State for phase_aware is a (T, S) array where [:, 2] is used as z-axis
        # for Lift/Transit/Place splitting. UMI state[:, 2] IS the z coord.
        out.append({
            "episode_id": f"episode_{gid}",
            "action": action,
            "state": state_rows,
        })
    return out


async def main() -> int:
    print("=" * 78)
    print("Mode B on UMI Cup in the Wild — first 10 episodes")
    print("=" * 78)

    print("\n[1/4] Loading UMI episodes from HF parquet...")
    t0 = time.time()
    episodes_raw = load_umi_first_10()
    print(f"  loaded {len(episodes_raw)} episodes in {time.time()-t0:.1f}s")
    for e in episodes_raw[:3]:
        print(f"    {e['episode_id']}: action={e['action'].shape} state={e['state'].shape} "
              f"gripper_range=[{-e['action'][:,6].max():.3f}, {-e['action'][:,6].min():.3f}]m")

    print("\n[2/4] Running phase-aware analysis (fps=10)...")
    t0 = time.time()
    result = analyze_task(episodes_raw, TASK_NAME, fps=10.0).to_dict()
    print(f"  done in {time.time()-t0:.2f}s — cohort={result['cohort_size']}, "
          f"clusters={len(result['clusters'])}, "
          f"flagged={sum(1 for e in result['episodes'] if e['anomaly']['is_anomaly'])}")

    print("\n  Phase detection results per episode:")
    for e in result["episodes"]:
        raw_events = e.get("raw_gripper_events", [])
        xtra = len(raw_events) - 2 * e["num_cycles"]
        flagged = "FLAG" if e["anomaly"]["is_anomaly"] else "-"
        print(f"    {e['episode_id']:<14}  frames={e['frames']:<4}  "
              f"cycles={e['num_cycles']}  raw_events={len(raw_events)} (xtra={xtra:+d})  {flagged}")
        if e["anomaly"]["is_anomaly"]:
            for r in e["anomaly"]["reasons"]:
                print(f"      → [{r['signal']}] {r['explanation'][:120]}")

    # Get video_path template for clip extraction
    from huggingface_hub import hf_hub_download
    info_path = hf_hub_download(REPO_ID, "meta/info.json", repo_type="dataset", token=HF_TOKEN)
    info = json.loads(Path(info_path).read_text())
    vpt = info.get("video_path")
    if not vpt:
        print("\nERROR: UMI has no video_path — skipping Gemini step")
        (OUT / "umi_phase_aware_only.json").write_text(json.dumps(result, indent=2, default=str))
        return 0

    print(f"\n  video_path template: {vpt}")

    # Clear any prior cache for this task
    clear_cache(DATASET_ID, TASK_NAME)

    print("\n[3/4] Running Gemini Mode B enrichment...")
    t0 = time.time()
    try:
        enriched = await enrich_with_gemini(
            result,
            dataset_id=DATASET_ID,
            task_name=TASK_NAME,
            repo_id=REPO_ID,
            video_path_template=vpt,
            use_cache=False,
        )
    except GeminiEnrichmentError as e:
        print(f"  Gemini enrichment failed: {e}")
        return 1
    dt = time.time() - t0
    g = enriched.get("gemini", {})
    print(f"  done in {dt:.1f}s (enriched={g.get('enriched')}, errors={g.get('errors', [])})")
    print(f"  tokens: {g.get('token_usage', {})}")
    print(f"  timings: {g.get('timings', {})}")

    print("\n[4/4] Results:")
    print("\n=== Clusters ===")
    for c in enriched["clusters"]:
        stat = c.get("label", "?")
        gem = c.get("gemini_label", "(no Gemini label)")
        conf = c.get("gemini_confidence", "?")
        print(f"  {c['id']}: stat=\"{stat}\"  →  AI=\"{gem}\" ({conf})")
        desc = c.get("gemini_description", "")
        if desc:
            print(f"    {desc[:220]}")

    print("\n=== Per-flagged-episode Gemini verdict ===")
    any_flagged = False
    for e in enriched["episodes"]:
        if not e["anomaly"]["is_anomaly"]:
            continue
        any_flagged = True
        sev = e.get("gemini_severity") or "?"
        cf = e.get("gemini_confirmation") or ""
        obs = e.get("gemini_observations") or []
        raw = e.get("raw_gripper_events", [])
        xtra = len(raw) - 2 * e["num_cycles"]
        print(f"\n  {e['episode_id']}  cycles={e['num_cycles']}  raw_events={len(raw)}  "
              f"(xtra={xtra:+d})  severity={sev}")
        if cf:
            print(f"    confirm: {cf}")
        for o in obs:
            print(f"    [novel] {o.get('phase','?')}c{o.get('cycle','?')} @ "
                  f"{o.get('timestamp','?')}: {o.get('observation','')}")

    if not any_flagged:
        print("\n  (no episodes flagged statistically)")
        print("  AI pass still ran for cluster characterization. Checking "
              "for any Gemini-surfaced observations on unflagged episodes...")
        for e in enriched["episodes"]:
            obs = e.get("gemini_observations") or []
            if obs:
                print(f"  {e['episode_id']}:")
                for o in obs:
                    print(f"    [novel] {o.get('phase','?')}c{o.get('cycle','?')} @ "
                          f"{o.get('timestamp','?')}: {o.get('observation','')}")

    out_path = OUT / "umi_first_10_mode_b.json"
    out_path.write_text(json.dumps(enriched, indent=2, default=str))
    print(f"\nSaved full result: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
