#!/usr/bin/env python3
"""Detailed per-episode verification on the Pilot C 10-episode sample.

Runs the phase-aware algorithm on the full 38-episode cohort (needed for valid
envelope/MAD baselines), then prints a focused report for the 10 episodes the
user is asking about:
  - phase breakdown (phase name × duration)
  - cluster membership
  - shape feature snapshot
  - anomaly verdict + reasons
  - side-by-side comparison: old envelope / Gemini / phase-aware
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prototype import (  # noqa: E402
    load_task_episode_ids, load_action_state, run_analysis,
    OLD_ENVELOPE_FLAGS, GEMINI_FLAGS, PILOT_C_EPISODES, SHAPE_FEATURE_NAMES,
)


def main() -> int:
    print("=" * 78)
    print("VERIFICATION: phase-aware on Libero mug-left-plate, 10-episode sample")
    print("=" * 78)
    print("Task: 'put the white mug on the left plate and put the yellow and white mug")
    print("       on the right plate' (38 total; showing first 10 from Analysis modal)")
    print()

    # Full cohort needed for envelope/MAD statistics
    all_ids = load_task_episode_ids()
    ep_data = load_action_state(all_ids)
    result = run_analysis(ep_data)

    pc = PILOT_C_EPISODES  # [0, 18, 22, 33, 58, 85, 88, 105, 107, 114]

    # Ground truth from prior runs
    old = OLD_ENVELOPE_FLAGS  # {0, 18, 85, 114}
    gem = GEMINI_FLAGS        # {18}
    new = set(result["anomalies"].keys()) & set(pc)

    print("--- SUMMARY TABLE ------------------------------------------------------------")
    print(f"{'ep':<6}{'cluster':<10}{'cycles':<8}{'frames':<8}{'old':<6}{'gemini':<8}{'phase-aware':<14}{'reason (if flagged)'}")
    print("-" * 78)
    for eid in pc:
        phases_seg = result["phases"][eid]
        cluster = result["cluster_of"].get(eid, "?")
        frames = ep_data[eid]["action"].shape[0]
        f_old = "FLAG" if eid in old else "—"
        f_gem = "FLAG" if eid in gem else "—"
        reasons = result["anomalies"].get(eid, [])
        f_new = "FLAG" if reasons else "—"
        # One-line reason (short form)
        if reasons:
            r = reasons[0]
            if r.signal == "envelope":
                short = f"env {r.feature}@{r.phase}c{r.cycle} |z|={r.magnitude}"
            elif r.signal == "duration":
                short = f"dur {r.phase}c{r.cycle} z={r.magnitude}"
            elif r.signal == "cycle_count":
                short = f"cycle-count {r.magnitude} off mode"
            else:
                short = f"{r.signal} {r.feature}"
            if len(reasons) > 1:
                short += f" (+{len(reasons)-1} more)"
        else:
            short = ""
        print(f"episode_{eid:<4}{cluster:<10}{phases_seg.num_cycles:<8}{frames:<8}{f_old:<6}{f_gem:<8}{f_new:<14}{short}")
    print("-" * 78)
    print()

    # Agreement summary
    print("--- AGREEMENT METRICS --------------------------------------------------------")
    print(f"Old envelope flags:   {sorted(old)} ({len(old)}/10)")
    print(f"Gemini flags:         {sorted(gem)} ({len(gem)}/10)")
    print(f"Phase-aware flags:    {sorted(new)} ({len(new)}/10)")
    print()
    print(f"Phase-aware ∩ Gemini: {sorted(new & gem)}")
    print(f"Phase-aware ∩ Old:    {sorted(new & old)}")
    print(f"Old ∖ Phase-aware:    {sorted(old - new)}  (false positives we removed)")
    print(f"Gemini ∖ Phase-aware: {sorted(gem - new)}  (Gemini-flagged but we missed)")
    print()

    # Per-episode deep dive
    print("--- PER-EPISODE DETAIL -------------------------------------------------------")
    for eid in pc:
        phases_seg = result["phases"][eid]
        feats = result["features"][eid]
        cluster = result["cluster_of"].get(eid, "?")
        reasons = result["anomalies"].get(eid, [])
        verdict_old = "FLAG" if eid in old else "ok"
        verdict_gem = "FLAG" if eid in gem else "ok"
        verdict_new = "FLAG" if reasons else "ok"

        print(f"\nepisode_{eid}  cluster={cluster}  cycles={phases_seg.num_cycles}  frames={ep_data[eid]['action'].shape[0]}")
        print(f"  Old envelope: {verdict_old}   Gemini: {verdict_gem}   Phase-aware: {verdict_new}")

        # Phase breakdown
        print(f"  Phase breakdown:")
        for p in phases_seg.phases:
            dur = p.end - p.start
            print(f"    {p.name:<10} cycle={p.cycle}  frames[{p.start},{p.end})  duration={dur}")

        # Shape features
        print(f"  Shape features:")
        snap = [f"{n}={feats[n]}" for n in SHAPE_FEATURE_NAMES[:6]]
        print(f"    {' | '.join(snap)}")
        snap2 = [f"{n}={feats[n]}" for n in SHAPE_FEATURE_NAMES[6:]]
        print(f"    {' | '.join(snap2)}")

        # Anomaly reasons
        if reasons:
            print(f"  Anomaly reasons:")
            for r in reasons:
                print(f"    [{r.signal}] {r.explanation}")
    print()

    # Save a condensed JSON the user can diff against future runs
    out_json = {
        "task": "put the white mug on the left plate and put the yellow and white mug on the right plate",
        "episodes_examined": pc,
        "cohort_size_for_envelope": len(ep_data),
        "old_envelope_flags": sorted(old),
        "gemini_flags": sorted(gem),
        "phase_aware_flags": sorted(new),
        "per_episode": [
            {
                "episode_id": f"episode_{eid}",
                "cluster": result["cluster_of"].get(eid),
                "num_cycles": result["phases"][eid].num_cycles,
                "frames": int(ep_data[eid]["action"].shape[0]),
                "phases": [
                    {"name": p.name, "cycle": p.cycle, "start": p.start, "end": p.end,
                     "duration": p.end - p.start}
                    for p in result["phases"][eid].phases
                ],
                "shape_features": result["features"][eid],
                "flagged_old": eid in old,
                "flagged_gemini": eid in gem,
                "flagged_phase_aware": eid in new,
                "anomaly_reasons": [
                    {"signal": r.signal, "phase": r.phase, "cycle": r.cycle,
                     "feature": r.feature, "magnitude": r.magnitude,
                     "explanation": r.explanation}
                    for r in result["anomalies"].get(eid, [])
                ],
            } for eid in pc
        ],
    }
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    out_path = Path(__file__).parent / "verify_pilot_c_results.json"
    out_path.write_text(json.dumps(out_json, indent=2, default=_default))
    print(f"Saved {out_path} ({out_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
