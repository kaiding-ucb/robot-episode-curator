#!/usr/bin/env python3
"""
Cross-dataset sanity check for phase-aware detection.

For each (dataset, task), load 30-50 episodes via the existing backend, run
phase detection, and report:
  - fraction of episodes successfully phase-segmented via gripper signal
  - phase-name inventory (canonical pick-place vs generic fallback)
  - anomaly count / cohort size
  - any obvious breakage
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from prototype import (  # noqa: E402
    detect_phases, run_analysis, GRIPPER_DIM,
)

API = "http://localhost:8000/api"
HF_BASE_UMI = "https://huggingface.co/datasets/lerobot/umi_cup_in_the_wild/resolve/main"
HF_TOKEN = "REDACTED-HF-TOKEN"

HERE = Path(__file__).parent
UMI_DATA = HERE / "_umi_cache"
UMI_DATA.mkdir(exist_ok=True)


def _hf_download(url: str, dst: Path):
    import urllib.request
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    with urllib.request.urlopen(req, timeout=300) as r, dst.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def load_episodes_from_hf_datasets_server(limit: int = 40) -> dict[int, dict]:
    """UMI has only observation.state; no action. Use state[:,-1] (gripper_width)
    as gripper surrogate. Fetch via HF datasets-server API (same path the
    backend uses for envelope stats)."""
    # Pull ~first 40 episodes' actions + state from the HF datasets server
    out: dict[int, dict] = {}
    offset = 0
    while len(out) < limit:
        url = (
            f"https://datasets-server.huggingface.co/rows"
            f"?dataset=lerobot%2Fumi_cup_in_the_wild"
            f"&config=default&split=train&offset={offset}&length=100"
        )
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        rows = data.get("rows", [])
        if not rows:
            break
        for row in rows:
            r_ = row.get("row", {})
            eid = r_.get("episode_index")
            if eid is None or eid >= limit:
                continue
            state = r_.get("observation.state")
            if state is None:
                continue
            out.setdefault(int(eid), {"frames": []})["frames"].append({
                "t": r_.get("timestamp"), "state": state,
            })
        offset += 100
        if offset > 10000:
            break
    # Convert to action/state arrays. Synthesize an "action" = observation.state
    # deltas (UMI has no action column).
    final: dict[int, dict] = {}
    for eid, d in out.items():
        frames = sorted(d["frames"], key=lambda f: f["t"] or 0)
        state = np.array([f["state"] for f in frames], dtype=float)
        if len(state) < 10:
            continue
        action = np.zeros_like(state)
        action[1:] = np.diff(state, axis=0)
        final[eid] = {"action": action, "state": state}
    return final


def main() -> int:
    print("=" * 70)
    print("Cross-dataset sanity: UMI Cup (state-only, no action column)")
    print("=" * 70)

    try:
        ep_data = load_episodes_from_hf_datasets_server(limit=40)
    except Exception as e:
        print(f"UMI data load failed: {e}")
        return 1
    print(f"Loaded {len(ep_data)} UMI episodes")

    if not ep_data:
        print("  no episodes; aborting")
        return 1

    # Probe: UMI state is [x, y, z, qx, qy, qz, qw, gripper_width]
    # So observation.state[-1] is gripper width, in meters (0.02-0.08 roughly).
    # Our detect_phases looks at action[:,6] for gripper. Map state[-1] into action[:,6].
    first_eid = next(iter(ep_data))
    state_dim = ep_data[first_eid]["state"].shape[1]
    print(f"  state dim: {state_dim} — assuming state[-1] is gripper width")

    for eid, d in ep_data.items():
        # Swap synthesized delta-action for something our detector can use:
        # put the gripper-width into action[:, GRIPPER_DIM] (with sign flip
        # since UMI gripper OPEN is wide and CLOSED is narrow — matches
        # Libero convention if we invert).
        synth_action = np.zeros((len(d["state"]), max(GRIPPER_DIM + 1, 7)))
        # delta end-effector xyz
        synth_action[1:, 0:3] = np.diff(d["state"][:, 0:3], axis=0)
        # pretend rotation channels are zeros
        # gripper: narrow = closed, wide = open. In Libero convention, closed = low.
        synth_action[:, GRIPPER_DIM] = -d["state"][:, -1]  # flip so narrow = low
        d["action"] = synth_action

    result = run_analysis(ep_data)
    n_gripper = sum(1 for s in result["phases"].values() if s.method == "gripper")
    n_vel = sum(1 for s in result["phases"].values() if s.method == "velocity_minima")
    phase_names = set()
    cycle_counts = []
    for s in result["phases"].values():
        cycle_counts.append(s.num_cycles)
        for p in s.phases:
            phase_names.add(p.name)

    print(f"\nPhase detection: {n_gripper} gripper / {n_vel} velocity-minima fallback")
    print(f"Cycle counts: min={min(cycle_counts) if cycle_counts else 0} max={max(cycle_counts) if cycle_counts else 0}")
    print(f"Phase name inventory: {sorted(phase_names)}")
    print(f"Anomalies: {len(result['anomalies'])} / {len(ep_data)}")
    print(f"Clusters: {[(c.id, c.label, len(c.members)) for c in result['clusters']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
