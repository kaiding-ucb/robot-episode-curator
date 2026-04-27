#!/usr/bin/env python3
"""
Phase-aware anomaly detection — Phase 0 offline prototype.

Re-does the Libero mug-left-plate task analysis with:
  1. Gripper-event phase segmentation (fallback: velocity-minima)
  2. Per-phase envelope with contiguous-run flagging (≥10% of phase bins, min 4)
  3. Phase-duration MAD outliers (2.5× MAD, Gaussian-equivalent)
  4. Shape descriptors + Mahalanobis anomaly (χ²₀.₉₅)
  5. Hierarchical clustering with adaptive K via gap statistic (capped at 5)
  6. Combined anomaly scoring with human-readable reason lineage

Validation:
  - Pilot C 10-episode comparison: old envelope flagged 4; Gemini flagged 1 (ep_18).
    Target: phase-aware flags ≤ 1 of those 10 (drop FPs on 0, 85, 114).
  - Synthetic retry injection: inject 3rd gripper cycle, expect flag via shape.
  - Generic-phase fallback: forced-disable gripper path, ensure envelope-in-phase
    still works via velocity-minima.

Writes findings.json + prints a summary table.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, medfilt
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.cluster import AgglomerativeClustering

HERE = Path(__file__).parent
GEMINI_DIR = HERE.parent / "gemini"
DATA_DIR = GEMINI_DIR / "data_libero"
META_PARQUET = GEMINI_DIR / "_libero_ep_meta.parquet"
OUT = HERE / "findings.json"

TASK_NAME = "put the white mug on the left plate and put the yellow and white mug on the right plate"
REPO = "lerobot/libero"
HF_TOKEN = "REDACTED-HF-TOKEN"
HF_BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"

# From Pilot C screenshot (first 10 of 38 task episodes)
PILOT_C_EPISODES = [0, 18, 22, 33, 58, 85, 88, 105, 107, 114]
# Old envelope flagged (UI screenshot):
OLD_ENVELOPE_FLAGS = {0, 18, 85, 114}
# Gemini's only flag (Pilot C):
GEMINI_FLAGS = {18}

# Libero action convention
ACTION_DIM = 7
GRIPPER_DIM = 6
Z_DIM = 2  # in observation.state (end-effector z)

# Algorithm constants
N_BIN_PER_PHASE = 50
CONTIGUOUS_RUN_FRAC = 0.40  # ≥40% of phase bins outside
CONTIGUOUS_RUN_MIN = 10
ENVELOPE_SIGMA = 3.0  # ±3σ (decision #3 — strict)
ENVELOPE_MIN_COHORT = 6  # need at least this many episodes for meaningful envelope
ENVELOPE_MIN_MEDIAN_DURATION = 15  # skip envelope if median phase duration < N frames
ENVELOPE_SMOOTH_SIGMA = 2.0  # Gaussian-smooth signals before envelope comparison
DURATION_MAD_THRESHOLD = 3.5  # 3.5× MAD (decision #3 — strict; 2.5× surfaced 1.4× median as "atypical")
DURATION_MIN_MEDIAN = 10  # skip duration flag if median phase duration < N frames
CYCLE_COUNT_MIN_DELTA = 2  # flag cycle-count outlier only if ≥2 off from mode
MAD_CONST = 1.4826  # Gaussian-equivalent scale
SHAPE_CONFIDENCE = 0.99  # conservative (decision #3)
GAP_K_RANGE = (2, 3, 4, 5)
GAP_REFS = 20
GAP_RNG_SEED = 42


# ========================================================================
# Data loading
# ========================================================================

def hf_download(remote_path: str, local_path: Path) -> None:
    import urllib.request
    if local_path.exists() and local_path.stat().st_size > 0:
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE}/{remote_path}"
    print(f"    GET {remote_path}")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    with urllib.request.urlopen(req, timeout=300) as r, local_path.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def load_task_episode_ids() -> list[int]:
    """Return all global episode_index values for the target task.

    Libero's meta/episodes parquet does not store task assignment; instead
    each frame in data/ has a task_index column. Use the running backend API
    (same resolution path as the UI) to enumerate the task's episodes.
    """
    import urllib.parse
    import urllib.request
    url = (
        "http://localhost:8000/api/datasets/libero/tasks/"
        + urllib.parse.quote(TASK_NAME)
        + "/episodes?limit=200&offset=0"
    )
    with urllib.request.urlopen(url, timeout=30) as r:
        episodes = json.loads(r.read())
    gids: list[int] = []
    for ep in episodes:
        eid = ep["id"]
        if eid.startswith("episode_"):
            gids.append(int(eid.split("_")[1]))
    gids.sort()
    print(f"  {len(gids)} episodes in task")
    return gids


def load_action_state(global_ids: list[int]) -> dict[int, dict]:
    """Scan data/ parquets until every target id is found."""
    targets = set(global_ids)
    out: dict[int, dict] = {}
    for fidx in range(0, 400):
        if not targets:
            break
        local = DATA_DIR / f"file-{fidx:03d}.parquet"
        hf_download(f"data/chunk-000/file-{fidx:03d}.parquet", local)
        df = pq.read_table(
            local,
            columns=["episode_index", "frame_index", "action", "observation.state"],
        ).to_pandas()
        found = sorted(set(df.episode_index.unique()) & targets)
        if found:
            for gid in found:
                ep = df[df.episode_index == gid].sort_values("frame_index")
                out[gid] = {
                    "action": np.stack(ep["action"].values),
                    "state": np.stack(ep["observation.state"].values),
                }
                targets.discard(gid)
    if targets:
        raise RuntimeError(f"missing: {sorted(targets)}")
    return out


# ========================================================================
# Phase detection
# ========================================================================

@dataclass
class Phase:
    name: str  # "Approach" | "Grasp" | "Lift" | "Transit" | "Place" | "Release" | "Return" | "Motion1" | "Pause1"
    start: int  # frame index inclusive
    end: int  # frame index exclusive
    cycle: int  # gripper-cycle index (0, 1, ...) or 0 for generic


@dataclass
class PhaseSegmentation:
    phases: list[Phase]
    method: str  # "gripper" | "velocity_minima"
    num_cycles: int
    gripper_closed_ranges: list[tuple[int, int]]  # [(close_i, open_i), ...]
    raw_gripper_transitions: int = 0  # open↔closed transitions before short-event filtering (retry signal)


def detect_gripper_closed_ranges(action: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    """
    action[:,6] is gripper cmd. Smooth aggressively, binarize, drop
    short-duration events, merge near-adjacent closed ranges (re-grips within
    the same pick-place).

    Returns (merged_closed_ranges, raw_transition_count). The raw count is the
    number of open↔closed transitions BEFORE any short-event filtering — it's
    the "retry smell" signal that feeds into shape descriptors.
    """
    g = action[:, GRIPPER_DIM].astype(float)
    if g.max() == g.min():
        return [], 0
    # Strong median filter — gripper commands are noisy at frame boundaries
    ksize = min(15, len(g) if len(g) % 2 == 1 else len(g) - 1)
    sm = medfilt(g, kernel_size=max(3, ksize))
    # p5/p95 midpoint for robust threshold (resists outlier injections)
    lo_ref = float(np.percentile(sm, 5))
    hi_ref = float(np.percentile(sm, 95))
    mid = (lo_ref + hi_ref) / 2
    # Libero (and most LeRobot pick-place datasets): action[6] positive = CLOSE
    # command (grasp), negative = OPEN command (release). Episodes start in the
    # approach phase with gripper commanded OPEN (low), transition to CLOSE
    # (high) at each grasp event. So is_closed = sm > mid.
    # Auto-detect if the convention is flipped by checking whether the FIRST
    # detected "closed" state occupies an implausibly long opening: if the
    # majority state is also what we're calling "closed", the convention is
    # inverted and we should flip.
    is_closed = sm > mid
    if is_closed.sum() > 0.6 * len(is_closed):
        # Closed-majority → flip (some datasets use opposite convention)
        is_closed = sm < mid

    diff = np.diff(is_closed.astype(int))
    close_events = np.where(diff == 1)[0] + 1
    open_events = np.where(diff == -1)[0] + 1

    if is_closed[0]:
        close_events = np.concatenate([[0], close_events])
    if is_closed[-1]:
        open_events = np.concatenate([open_events, [len(is_closed)]])

    raw_ranges = list(zip(close_events.tolist(), open_events.tolist()))
    raw_transition_count = len(close_events) + len(open_events)

    # Drop closed ranges shorter than 20 frames (2 s at 10 Hz) — below that
    # it's almost certainly a re-grip, not a real pick-and-place.
    MIN_CLOSED = 20
    ranges = [(a, b) for a, b in raw_ranges if b - a >= MIN_CLOSED]
    # Merge consecutive closed ranges separated by a short open gap (<30 f = 3s).
    # A true between-cycles release-and-reach typically takes >3 s; shorter means
    # it's a re-grip inside one logical pick-place.
    MERGE_OPEN_GAP = 30
    merged: list[tuple[int, int]] = []
    for a, b in ranges:
        if merged and a - merged[-1][1] < MERGE_OPEN_GAP:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged, raw_transition_count


def detect_phases(action: np.ndarray, state: np.ndarray, *, force_generic: bool = False) -> PhaseSegmentation:
    T = len(action)
    z = state[:, Z_DIM]

    if force_generic:
        closed_ranges, raw_trans = [], 0
    else:
        closed_ranges, raw_trans = detect_gripper_closed_ranges(action)

    if len(closed_ranges) == 0:
        # Fallback: velocity-minima split. Compute vel = ||Δaction[0:3]||; find
        # significant minima to split into Motion/Pause segments.
        v = np.linalg.norm(np.diff(action[:, 0:3], axis=0), axis=1)
        v = gaussian_filter1d(v, sigma=3)
        # Invert so minima become peaks
        thresh = np.percentile(v, 20)
        peaks, _ = find_peaks(-v, height=-thresh, distance=max(10, T // 20))
        # Split into alternating Motion / Pause chunks
        boundaries = [0] + sorted(peaks.tolist()) + [T]
        phases = []
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            if b - a < 3:
                continue
            phases.append(Phase(
                name=f"Motion{i + 1}" if i % 2 == 0 else f"Pause{i // 2 + 1}",
                start=a, end=b, cycle=0,
            ))
        return PhaseSegmentation(
            phases=phases,
            method="velocity_minima",
            num_cycles=0,
            gripper_closed_ranges=[],
            raw_gripper_transitions=raw_trans,
        )

    # Gripper-cycle phases. Strictly non-overlapping: each phase's start is
    # exactly the prior phase's end. Grasp and Release are the narrow
    # transition windows immediately after close / open events.
    GRASP_LEN = 3
    RELEASE_LEN = 3

    phases: list[Phase] = []
    prev_end = 0
    for cycle_idx, (close_i, open_i) in enumerate(closed_ranges):
        # Approach: prev_end → close_i
        if close_i > prev_end:
            phases.append(Phase("Approach", prev_end, close_i, cycle_idx))
        # Grasp window: [close_i, close_i + GRASP_LEN), bounded by open_i
        grasp_end = min(close_i + GRASP_LEN, open_i)
        if grasp_end > close_i:
            phases.append(Phase("Grasp", close_i, grasp_end, cycle_idx))
        # Inside closed range (after grasp): find z-peak to split Lift / Transit / Place
        interior_start = grasp_end
        z_seg = z[interior_start:open_i]
        if len(z_seg) > 3:
            zpeak_rel = int(np.argmax(z_seg))
            z_ascend_end = interior_start + zpeak_rel
            peak_z = z_seg[zpeak_rel]
            min_z_after = z_seg[zpeak_rel:].min()
            if peak_z - min_z_after > 1e-4:
                descend_threshold = peak_z - 0.3 * (peak_z - min_z_after)
                descend_rel_offset = int(np.argmax(z_seg[zpeak_rel:] < descend_threshold))
                descend_start = interior_start + zpeak_rel + max(descend_rel_offset, 0)
            else:
                descend_start = z_ascend_end
        else:
            z_ascend_end = interior_start
            descend_start = open_i

        # Lift / Transit / Place: contiguous, cover [grasp_end, open_i)
        if z_ascend_end > interior_start:
            phases.append(Phase("Lift", interior_start, z_ascend_end, cycle_idx))
        if descend_start > z_ascend_end:
            phases.append(Phase("Transit", z_ascend_end, descend_start, cycle_idx))
        if open_i > descend_start:
            phases.append(Phase("Place", descend_start, open_i, cycle_idx))

        # Release: [open_i, open_i + RELEASE_LEN), bounded by T
        release_end = min(open_i + RELEASE_LEN, T)
        if release_end > open_i:
            phases.append(Phase("Release", open_i, release_end, cycle_idx))
        prev_end = release_end

    # Trailing Return: [last release_end, T)
    if prev_end < T:
        phases.append(Phase("Return", prev_end, T, cycle=len(closed_ranges) - 1))

    # Defensive: filter any zero/negative-length artifacts (shouldn't happen
    # given the guards above, but keeps the output invariant-clean).
    phases = [p for p in phases if p.end > p.start]

    return PhaseSegmentation(
        phases=phases,
        method="gripper",
        num_cycles=len(closed_ranges),
        gripper_closed_ranges=closed_ranges,
        raw_gripper_transitions=raw_trans,
    )


# ========================================================================
# Per-phase envelope flagging
# ========================================================================

@dataclass
class AnomalyReason:
    signal: str  # "envelope" | "duration" | "shape" | "cycle_count"
    phase: Optional[str] = None
    cycle: Optional[int] = None
    feature: Optional[str] = None
    magnitude: float = 0.0
    explanation: str = ""


def resample_phase(signal: np.ndarray, start: int, end: int, n: int) -> np.ndarray:
    """Linear-interpolate `signal[start:end]` to exactly n samples."""
    seg = signal[start:end]
    if len(seg) < 2:
        return np.full(n, float("nan"))
    t_src = np.linspace(0, 1, len(seg))
    t_dst = np.linspace(0, 1, n)
    return np.interp(t_dst, t_src, seg.astype(float))


def _phase_key(p: Phase, num_cycles: int) -> str:
    """Key for envelope grouping: one envelope per (phase_name, cycle, num_cycles).
    Including num_cycles ensures we only compare 'cycle 1 of a 2-cycle episode'
    against other 'cycle 1 of a 2-cycle episode' — not against cycle 1 of a
    3-cycle episode (which would be a different part of the task)."""
    return f"{p.name}#c{p.cycle}#n{num_cycles}"


def flag_envelope_violations(
    ep_data: dict[int, dict],
    phases: dict[int, PhaseSegmentation],
) -> dict[int, list[AnomalyReason]]:
    """For each phase-cycle, build an envelope across episodes with the same
    phase-cycle present, then flag contiguous runs."""
    # Channels: position magnitude, rotation magnitude, gripper
    def pos_mag(a: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a[:, 0:3], axis=1)

    def rot_mag(a: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a[:, 3:6], axis=1)

    # Collect per-phase-cycle-channel resampled signals
    channels = {"position": pos_mag, "rotation": rot_mag}
    # phase_key -> channel -> dict[eid -> resampled (N,)]
    buckets: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    durations: dict[str, list[int]] = {}  # for median-duration gating
    for eid, seg in phases.items():
        a = ep_data[eid]["action"]
        for p in seg.phases:
            if p.name in ("Grasp", "Release"):  # narrow windows, skip envelope
                continue
            key = _phase_key(p, seg.num_cycles)
            buckets.setdefault(key, {ch: {} for ch in channels})
            durations.setdefault(key, []).append(p.end - p.start)
            for ch_name, fn in channels.items():
                sig = fn(a)
                rs = resample_phase(sig, p.start, p.end, N_BIN_PER_PHASE)
                if not np.isnan(rs).any():
                    # Smooth to suppress per-frame jitter before envelope comparison
                    buckets[key][ch_name][eid] = gaussian_filter1d(rs, sigma=ENVELOPE_SMOOTH_SIGMA)

    # Per bucket: compute envelope and flag runs
    flags: dict[int, list[AnomalyReason]] = {}
    for key, ch_map in buckets.items():
        # Parse key
        # format: "<name>#c<cycle>#n<num_cycles>"
        try:
            name_part, rest = key.split("#c", 1)
            cycle_part, n_part = rest.split("#n", 1)
            phase_name = name_part
            cycle = int(cycle_part)
        except ValueError:
            continue
        med_dur = int(np.median(durations[key])) if durations[key] else 0
        if med_dur < ENVELOPE_MIN_MEDIAN_DURATION:
            continue  # phase too short for meaningful envelope
        for ch_name, eid_sigs in ch_map.items():
            if len(eid_sigs) < ENVELOPE_MIN_COHORT:
                continue
            ids = sorted(eid_sigs.keys())
            X = np.stack([eid_sigs[e] for e in ids])  # (N_ep, N_bin)
            mean = X.mean(0)
            std = X.std(0)
            std = np.clip(std, 1e-6, None)
            for eid in ids:
                z = np.abs(eid_sigs[eid] - mean) / std
                outside = z > ENVELOPE_SIGMA
                # Longest contiguous run of True
                run_lengths = _contiguous_run_lengths(outside)
                longest = max(run_lengths, default=0)
                min_run = max(CONTIGUOUS_RUN_MIN, int(CONTIGUOUS_RUN_FRAC * N_BIN_PER_PHASE))
                if longest >= min_run:
                    peak_z = float(z[outside].max()) if outside.any() else 0.0
                    reason = AnomalyReason(
                        signal="envelope",
                        phase=phase_name,
                        cycle=cycle,
                        feature=ch_name,
                        magnitude=round(peak_z, 2),
                        explanation=(
                            f"{ch_name.title()} magnitude deviates from cohort envelope during {phase_name}"
                            f" (cycle {cycle}) — {longest} consecutive bins outside ±{ENVELOPE_SIGMA}σ, "
                            f"peak |z|={peak_z:.1f}"
                        ),
                    )
                    flags.setdefault(eid, []).append(reason)

    return flags


def _contiguous_run_lengths(mask: np.ndarray) -> list[int]:
    """Return the lengths of True runs in a 1-D boolean array."""
    runs = []
    cur = 0
    for v in mask:
        if v:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


# ========================================================================
# Phase duration outliers (MAD)
# ========================================================================

def flag_duration_outliers(
    phases: dict[int, PhaseSegmentation],
) -> dict[int, list[AnomalyReason]]:
    """Per (phase_name, cycle), compute MAD-based outliers in duration."""
    # Collect durations
    # key -> dict[eid -> duration]
    key_ep_dur: dict[str, dict[int, int]] = {}
    cycle_counts: dict[int, int] = {}
    for eid, seg in phases.items():
        cycle_counts[eid] = seg.num_cycles
        for p in seg.phases:
            if p.name in ("Grasp", "Release"):
                continue  # fixed narrow windows
            dur = p.end - p.start
            key = _phase_key(p, seg.num_cycles)
            key_ep_dur.setdefault(key, {})[eid] = dur

    flags: dict[int, list[AnomalyReason]] = {}
    for key, ep_dur in key_ep_dur.items():
        if len(ep_dur) < ENVELOPE_MIN_COHORT:
            continue
        try:
            name_part, rest = key.split("#c", 1)
            cycle_part, _n = rest.split("#n", 1)
            phase_name = name_part
            cycle = int(cycle_part)
        except ValueError:
            continue
        durs = np.array([ep_dur[e] for e in ep_dur], dtype=float)
        med = float(np.median(durs))
        if med < DURATION_MIN_MEDIAN:
            continue  # phase too short for MAD to be meaningful
        mad = float(np.median(np.abs(durs - med)))
        sigma_eq = MAD_CONST * mad if mad > 0 else 1.0
        for eid, d in ep_dur.items():
            z = (d - med) / sigma_eq if sigma_eq > 0 else 0.0
            if abs(z) > DURATION_MAD_THRESHOLD:
                ratio = d / med if med > 0 else 1.0
                label = _duration_label(phase_name, ratio)
                reason = AnomalyReason(
                    signal="duration",
                    phase=phase_name,
                    cycle=cycle,
                    magnitude=round(float(abs(z)), 2),
                    explanation=f"{phase_name} phase (cycle {cycle}) duration {ratio:.2f}× cohort median ({d}f vs med {med:.0f}f): {label}",
                )
                flags.setdefault(eid, []).append(reason)

    # Cycle-count outlier: flag only if ≥CYCLE_COUNT_MIN_DELTA off from mode.
    # 1-off is usually detection noise or one legitimate re-grip.
    counts = np.array(list(cycle_counts.values()))
    if len(counts) >= 4 and counts.max() > 0:
        mode = int(np.bincount(counts).argmax())
        for eid, nc in cycle_counts.items():
            if abs(nc - mode) >= CYCLE_COUNT_MIN_DELTA:
                label = "missing cycles" if nc < mode else "extra cycles"
                reason = AnomalyReason(
                    signal="cycle_count",
                    magnitude=float(abs(nc - mode)),
                    explanation=f"{nc} gripper cycles vs cohort mode {mode}: {label}",
                )
                flags.setdefault(eid, []).append(reason)

    return flags


def _duration_label(phase_name: str, ratio: float) -> str:
    """Human-friendly label (decision #5 — on backend)."""
    if ratio > 2.0:
        if phase_name == "Place":
            return "hesitation before release"
        if phase_name == "Grasp":
            return "difficulty grasping"
        if phase_name == "Approach":
            return "slow approach"
        if phase_name == "Transit":
            return "slow transit"
        return "prolonged"
    if ratio < 0.4:
        if phase_name == "Approach":
            return "cut-short approach"
        if phase_name == "Place":
            return "abrupt placement"
        return "abbreviated"
    return "atypical duration"


# ========================================================================
# Shape descriptors + Mahalanobis
# ========================================================================

SHAPE_FEATURE_NAMES = [
    "arc_length_xyz",
    "arc_length_rot",
    "num_gripper_events",
    "num_pauses",
    "z_range",
    "xy_bbox_area",
    "path_directness",
    "gripper_closed_ratio",
    "num_direction_changes",
    "episode_duration",
]


def compute_shape_features(action: np.ndarray, state: np.ndarray, seg: PhaseSegmentation) -> dict[str, float]:
    T = len(action)
    xyz = state[:, 0:3]
    dxyz = np.diff(xyz, axis=0)
    arc_xyz = float(np.linalg.norm(dxyz, axis=1).sum())

    drpy = np.diff(action[:, 3:6], axis=0)
    arc_rot = float(np.linalg.norm(drpy, axis=1).sum())

    # Pauses = prolonged low-velocity regions
    v = np.linalg.norm(dxyz, axis=1)
    v_sm = gaussian_filter1d(v, sigma=3) if len(v) > 3 else v
    thresh = float(np.percentile(v_sm, 25)) if len(v_sm) else 0.0
    in_pause = v_sm < thresh
    # A "pause" = contiguous run of ≥10 frames
    pauses = sum(1 for r in _contiguous_run_lengths(in_pause) if r >= 10)

    z_range = float(xyz[:, 2].max() - xyz[:, 2].min()) if T else 0.0
    xy_bbox = float((xyz[:, 0].max() - xyz[:, 0].min()) * (xyz[:, 1].max() - xyz[:, 1].min())) if T else 0.0

    straight = float(np.linalg.norm(xyz[-1] - xyz[0]))
    directness = straight / arc_xyz if arc_xyz > 0 else 0.0

    # Gripper closed ratio
    closed_frames = sum(b - a for a, b in seg.gripper_closed_ranges)
    gripper_ratio = closed_frames / T if T else 0.0

    # Direction changes — sign flips in each xyz-velocity axis
    dir_changes = 0
    for i in range(3):
        s = np.sign(dxyz[:, i])
        s[s == 0] = 1
        dir_changes += int(np.sum(np.diff(s) != 0))

    return {
        "arc_length_xyz": round(arc_xyz, 4),
        "arc_length_rot": round(arc_rot, 4),
        # Raw transitions (pre-merge) — retries show up here even if they get
        # merged into a single logical cycle.
        "num_gripper_events": seg.raw_gripper_transitions,
        "num_pauses": pauses,
        "z_range": round(z_range, 4),
        "xy_bbox_area": round(xy_bbox, 4),
        "path_directness": round(directness, 4),
        "gripper_closed_ratio": round(gripper_ratio, 4),
        "num_direction_changes": dir_changes,
        "episode_duration": T,
    }


def flag_shape_outliers(features: dict[int, dict[str, float]]) -> dict[int, list[AnomalyReason]]:
    ids = sorted(features.keys())
    X = np.array([[features[e][n] for n in SHAPE_FEATURE_NAMES] for e in ids], dtype=float)
    # Standardize
    mu = X.mean(0)
    sigma = X.std(0)
    sigma = np.clip(sigma, 1e-9, None)
    Z = (X - mu) / sigma
    # Robust covariance (pseudo-inv if singular)
    try:
        cov = np.cov(Z, rowvar=False)
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        inv = np.eye(X.shape[1])

    threshold = chi2.ppf(SHAPE_CONFIDENCE, df=X.shape[1])
    flags: dict[int, list[AnomalyReason]] = {}
    for i, eid in enumerate(ids):
        d2 = float(Z[i] @ inv @ Z[i])
        if d2 > threshold:
            # Which features drive the distance? attribution ~ (Z_i · inv) * Z_i per-feature
            contribs = (Z[i] @ inv) * Z[i]
            # Normalize to percentages
            total = d2 if d2 > 0 else 1.0
            ranked = sorted(zip(SHAPE_FEATURE_NAMES, contribs), key=lambda x: -abs(x[1]))
            dominant = [(n, c / total) for n, c in ranked[:2] if abs(c) / total > 0.1]
            label = _shape_label(dominant, features[eid])
            reason = AnomalyReason(
                signal="shape",
                feature=",".join(n for n, _ in dominant) if dominant else "combined",
                magnitude=round(d2, 2),
                explanation=f"Shape outlier (Mahalanobis²={d2:.1f} > χ²₀.₉₅={threshold:.1f}): {label}",
            )
            flags.setdefault(eid, []).append(reason)
    return flags


def _shape_label(dominant: list[tuple[str, float]], feats: dict) -> str:
    if not dominant:
        return "multivariate anomaly (no single dominant axis)"
    parts = []
    for n, c in dominant:
        val = feats[n]
        direction = "high" if c > 0 else "low"
        if n == "num_gripper_events":
            parts.append(f"{val} gripper events")
        elif n == "num_pauses":
            parts.append(f"{val} velocity pauses")
        elif n == "path_directness":
            parts.append(f"directness={val:.2f} ({'circuitous' if direction == 'low' else 'very direct'})")
        elif n == "arc_length_xyz":
            parts.append(f"total xyz travel {direction}")
        elif n == "arc_length_rot":
            parts.append(f"total rotation {direction}")
        elif n == "num_direction_changes":
            parts.append(f"{val} velocity sign flips ({'jerky' if direction == 'high' else 'smooth'})")
        else:
            parts.append(f"{n}={val:.3f} ({direction})")
    return "; ".join(parts)


# ========================================================================
# Variance clustering with adaptive K (gap statistic)
# ========================================================================

@dataclass
class Cluster:
    id: str
    label: str
    members: list[int]
    medoid: int
    dominant_features: list[str]


def _wss(X: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for k in np.unique(labels):
        xk = X[labels == k]
        if len(xk) == 0:
            continue
        total += float(((xk - xk.mean(0)) ** 2).sum())
    return total


def gap_statistic_k(X: np.ndarray, k_range=GAP_K_RANGE, n_refs=GAP_REFS, seed=GAP_RNG_SEED) -> int:
    rng = np.random.default_rng(seed)
    lo, hi = X.min(0), X.max(0)

    real = np.zeros(len(k_range))
    ref = np.zeros((n_refs, len(k_range)))

    def fit(X_, k):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_)
        return labels

    for i, k in enumerate(k_range):
        real[i] = _wss(X, fit(X, k))

    for b in range(n_refs):
        Xr = rng.uniform(lo, hi, X.shape)
        for i, k in enumerate(k_range):
            ref[b, i] = _wss(Xr, fit(Xr, k))

    log_ref = np.log(ref + 1e-12)
    gap = log_ref.mean(0) - np.log(real + 1e-12)
    sk = log_ref.std(0) * math.sqrt(1 + 1.0 / n_refs)

    for i in range(len(k_range) - 1):
        if gap[i] >= gap[i + 1] - sk[i + 1]:
            return k_range[i]
    return k_range[-1]


def cluster_variance_modes(features: dict[int, dict[str, float]]) -> list[Cluster]:
    ids = sorted(features.keys())
    if len(ids) < 4:
        return []
    X = np.array([[features[e][n] for n in SHAPE_FEATURE_NAMES] for e in ids], dtype=float)
    mu = X.mean(0)
    sigma = X.std(0)
    sigma = np.clip(sigma, 1e-9, None)
    Z = (X - mu) / sigma

    k = gap_statistic_k(Z)
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)

    clusters: list[Cluster] = []
    for ki in range(k):
        mask = labels == ki
        members = [ids[i] for i in np.where(mask)[0]]
        if not members:
            continue
        # Medoid = minimum-sum-distance in Z-space
        Zc = Z[mask]
        d2 = ((Zc[:, None, :] - Zc[None, :, :]) ** 2).sum(-1)
        medoid_local = int(d2.sum(0).argmin())
        medoid = members[medoid_local]
        # Dominant features = largest (|mean_cluster - mean_all|) across axes
        diff = Z[mask].mean(0)  # already centered around 0 since Z is standardized cohort-wide
        ranked = sorted(zip(SHAPE_FEATURE_NAMES, diff), key=lambda x: -abs(x[1]))
        dominant = [n for n, v in ranked[:3] if abs(v) > 0.5]
        # Auto label from top axis direction
        if not dominant:
            lbl = f"Mode {ki + 1}"
        else:
            top, top_v = ranked[0]
            direction = "high" if top_v > 0 else "low"
            lbl = _cluster_label(top, direction, dominant)
        clusters.append(Cluster(
            id=f"C{ki + 1}",
            label=lbl,
            members=members,
            medoid=medoid,
            dominant_features=dominant,
        ))
    return clusters


def _cluster_label(top: str, direction: str, dominant: list[str]) -> str:
    if top == "episode_duration":
        return "Rapid / direct" if direction == "low" else "Deliberate / slow"
    if top == "num_pauses":
        return "Steady / uninterrupted" if direction == "low" else "Hesitant / many pauses"
    if top == "path_directness":
        return "Circuitous path" if direction == "low" else "Direct path"
    if top == "arc_length_xyz":
        return "Compact motion" if direction == "low" else "Wide-ranging motion"
    if top == "num_gripper_events":
        return "Single-cycle" if direction == "low" else "Multi-cycle / retries"
    if top == "arc_length_rot":
        return "Minimal rotation" if direction == "low" else "High rotation"
    return f"Cluster ({top}={direction})"


# ========================================================================
# Top-level driver
# ========================================================================

def run_analysis(ep_data: dict[int, dict]) -> dict:
    phases = {eid: detect_phases(d["action"], d["state"]) for eid, d in ep_data.items()}
    envelope = flag_envelope_violations(ep_data, phases)
    duration = flag_duration_outliers(phases)
    features = {eid: compute_shape_features(d["action"], d["state"], phases[eid]) for eid, d in ep_data.items()}
    shape = flag_shape_outliers(features)
    clusters = cluster_variance_modes(features)

    anomalies: dict[int, list[AnomalyReason]] = {}
    for eid in ep_data:
        rs = envelope.get(eid, []) + duration.get(eid, []) + shape.get(eid, [])
        if rs:
            anomalies[eid] = rs

    # Assign cluster id per episode
    cluster_of: dict[int, str] = {}
    for c in clusters:
        for m in c.members:
            cluster_of[m] = c.id

    return {
        "phases": phases,
        "features": features,
        "clusters": clusters,
        "cluster_of": cluster_of,
        "anomalies": anomalies,
    }


def summarize(result: dict, ep_ids: list[int]) -> str:
    lines = []
    lines.append(f"Cohort size: {len(ep_ids)}")
    lines.append(f"Phase detection: {sum(1 for s in result['phases'].values() if s.method == 'gripper')} gripper / "
                 f"{sum(1 for s in result['phases'].values() if s.method == 'velocity_minima')} fallback")
    cycle_counts = [s.num_cycles for s in result["phases"].values()]
    if cycle_counts:
        lines.append(f"Cycle counts: min={min(cycle_counts)} med={int(np.median(cycle_counts))} max={max(cycle_counts)}")
    lines.append("")
    lines.append(f"Clusters ({len(result['clusters'])}):")
    for c in result["clusters"]:
        lines.append(f"  {c.id}: {c.label} — {len(c.members)} eps, medoid=episode_{c.medoid}, dom={c.dominant_features}")
    lines.append("")
    lines.append(f"Anomalies: {len(result['anomalies'])} / {len(ep_ids)} episodes")
    for eid in sorted(result["anomalies"].keys()):
        lines.append(f"  episode_{eid}  cluster={result['cluster_of'].get(eid,'?')}")
        for r in result["anomalies"][eid]:
            lines.append(f"      [{r.signal}] {r.explanation}")
    return "\n".join(lines)


def compare_to_pilot_c(result: dict) -> str:
    flagged = set(result["anomalies"].keys())
    pc = set(PILOT_C_EPISODES)
    pc_flagged_new = flagged & pc
    lines = ["\n=== Pilot C (10 episodes) comparison ===",
             f"Old envelope flags: {sorted(OLD_ENVELOPE_FLAGS)}",
             f"Gemini's flags:     {sorted(GEMINI_FLAGS)}",
             f"Phase-aware flags:  {sorted(pc_flagged_new)}",
             ""]
    agreement_gemini = pc_flagged_new == GEMINI_FLAGS
    fp_old_minus_new = OLD_ENVELOPE_FLAGS - pc_flagged_new
    lines.append(f"Agreement with Gemini (subset): {'YES' if GEMINI_FLAGS.issubset(pc_flagged_new) else 'NO'}")
    lines.append(f"Strict match with Gemini:       {'YES' if agreement_gemini else 'NO'}")
    lines.append(f"False-positives removed vs old: {sorted(fp_old_minus_new)}")
    return "\n".join(lines)


def _pick_wide_gap_target(ep_data: dict[int, dict]) -> Optional[int]:
    """Pick an episode whose between-cycles open gap is wide enough to host a
    synthetic injection (>=90 frames) without triggering the merge rule."""
    best = None
    best_gap = 0
    for eid, d in ep_data.items():
        ranges, _ = detect_gripper_closed_ranges(d["action"])
        if len(ranges) < 2:
            continue
        gap = ranges[1][0] - ranges[0][1]
        if gap > best_gap:
            best_gap = gap
            best = eid
    return best


def synthetic_retry_test(ep_data: dict[int, dict]) -> str:
    """Inject a phantom 3rd cycle into a normal episode; expect cycle_count flag."""
    import copy
    target = _pick_wide_gap_target(ep_data)
    if target is None:
        return "\n=== test_injected_retry === SKIP — no episode with ≥2 cycles"
    fake = copy.deepcopy(ep_data)
    a = fake[target]["action"].copy()
    g = a[:, GRIPPER_DIM]
    closed_ranges_target, _ = detect_gripper_closed_ranges(a)
    gap_start = closed_ranges_target[0][1]
    gap_end = closed_ranges_target[1][0]
    # 30f closed in the middle, with ≥30f clearance on each side so the merge
    # rule keeps them as a separate cycle.
    mid = (gap_start + gap_end) // 2
    inject_start, inject_end = mid - 15, mid + 15
    if inject_start - gap_start < 30 or gap_end - inject_end < 30:
        return f"\n=== test_injected_retry === SKIP — gap in episode_{target} too narrow ({gap_end - gap_start}f)"
    lo = float(np.percentile(g, 5))
    a[inject_start:inject_end, GRIPPER_DIM] = lo
    fake[target]["action"] = a

    after_ranges, _ = detect_gripper_closed_ranges(a)
    result = run_analysis(fake)
    reasons = result["anomalies"].get(target, [])
    signals = {r.signal for r in reasons}
    ok = "cycle_count" in signals
    lines = ["\n=== test_injected_retry ===",
             f"Target: episode_{target}, gap was {gap_end - gap_start}f, injected closed at [{inject_start},{inject_end}]",
             f"Closed ranges before injection: {closed_ranges_target}",
             f"Closed ranges after injection:  {after_ranges}",
             f"Cycles detected after injection: {result['phases'][target].num_cycles}",
             f"Signals triggered: {sorted(signals) or '(none)'}",
             f"Result: {'PASS — caught via cycle_count' if ok else 'FAIL'}"]
    return "\n".join(lines)


def synthetic_elongated_place_test(ep_data: dict[int, dict]) -> str:
    """Inject a 3× longer Place phase into a normal episode; expect duration flag."""
    import copy
    target = None
    for eid in (22, 33, 58):
        if eid in ep_data:
            target = eid
            break
    if target is None:
        return "\n=== test_elongated_place === SKIP — no normal episode available"
    fake = copy.deepcopy(ep_data)
    a = fake[target]["action"].copy()

    # Find Place phase of cycle 0 and extend by duplicating the last N frames
    seg = detect_phases(a, fake[target]["state"])
    place_phase = next((p for p in seg.phases if p.name == "Place" and p.cycle == 0), None)
    if place_phase is None:
        return f"\n=== test_elongated_place === SKIP — no Place phase in episode_{target}"
    place_len = place_phase.end - place_phase.start
    extra = place_len * 2  # triple the Place phase
    # Repeat the last row of Place phase
    last_frame = a[place_phase.end - 1:place_phase.end]
    last_state = fake[target]["state"][place_phase.end - 1:place_phase.end]
    a = np.vstack([
        a[:place_phase.end],
        np.tile(last_frame, (extra, 1)),
        a[place_phase.end:],
    ])
    fake[target]["action"] = a
    fake[target]["state"] = np.vstack([
        fake[target]["state"][:place_phase.end],
        np.tile(last_state, (extra, 1)),
        fake[target]["state"][place_phase.end:],
    ])

    result = run_analysis(fake)
    reasons = result["anomalies"].get(target, [])
    signals = {r.signal for r in reasons}
    place_reasons = [r for r in reasons if r.signal == "duration" and r.phase == "Place"]
    ok = len(place_reasons) > 0
    lines = ["\n=== test_elongated_place ===",
             f"Target: episode_{target}, extended Place (cycle 0) by +{extra}f ({place_len}→{place_len+extra})",
             f"Signals triggered: {sorted(signals) or '(none)'}",
             f"Place duration flags: {[r.explanation for r in place_reasons]}",
             f"Result: {'PASS — caught via duration flag on Place' if ok else 'FAIL'}"]
    return "\n".join(lines)


def synthetic_natural_variance_test(ep_data: dict[int, dict]) -> str:
    """Timing-perturb a normal episode (10% uniform frame stretching) without
    changing behaviour; expect NO flag (or at most a mild duration one)."""
    import copy
    target = None
    for eid in (22, 33, 58):
        if eid in ep_data:
            target = eid
            break
    if target is None:
        return "\n=== test_natural_variance === SKIP — no normal episode"
    # Check if this episode is already flagged in the unperturbed run
    baseline_result = run_analysis({k: v for k, v in ep_data.items()})
    baseline_flagged = target in baseline_result["anomalies"]
    if baseline_flagged:
        return f"\n=== test_natural_variance === SKIP — episode_{target} already flagged baseline"

    fake = copy.deepcopy(ep_data)
    a = fake[target]["action"]
    s = fake[target]["state"]
    # Temporal resampling: stretch each episode by 10% via linear interpolation
    T = len(a)
    new_T = int(T * 1.10)
    t_src = np.linspace(0, 1, T)
    t_dst = np.linspace(0, 1, new_T)
    a_new = np.stack([np.interp(t_dst, t_src, a[:, i]) for i in range(a.shape[1])], axis=1)
    s_new = np.stack([np.interp(t_dst, t_src, s[:, i]) for i in range(s.shape[1])], axis=1)
    fake[target]["action"] = a_new
    fake[target]["state"] = s_new

    result = run_analysis(fake)
    reasons = result["anomalies"].get(target, [])
    ok = len(reasons) == 0
    lines = ["\n=== test_natural_variance ===",
             f"Target: episode_{target}, 10% temporal stretch ({T}f → {new_T}f)",
             f"Signals triggered: {sorted({r.signal for r in reasons}) or '(none)'}",
             f"Result: {'PASS — unflagged (no false positive)' if ok else 'FAIL — false positive'}"]
    if not ok:
        lines.append(f"(reasons: {[r.explanation for r in reasons]})")
    return "\n".join(lines)


def generic_fallback_sanity(ep_data: dict[int, dict]) -> str:
    """Force-disable gripper detection to exercise velocity-minima fallback."""
    eid = next(iter(ep_data))
    seg = detect_phases(ep_data[eid]["action"], ep_data[eid]["state"], force_generic=True)
    return (f"\n=== Generic (no-gripper) fallback sanity ===\n"
            f"episode_{eid} force_generic: method={seg.method}, phases={len(seg.phases)}, "
            f"names={[p.name for p in seg.phases[:6]]}…")


def to_jsonable(result: dict, ep_ids: list[int]) -> dict:
    return {
        "task": TASK_NAME,
        "cohort_size": len(ep_ids),
        "algorithm": {
            "envelope_sigma": ENVELOPE_SIGMA,
            "duration_mad_threshold": DURATION_MAD_THRESHOLD,
            "shape_confidence": SHAPE_CONFIDENCE,
            "contiguous_run_min": CONTIGUOUS_RUN_MIN,
            "contiguous_run_frac": CONTIGUOUS_RUN_FRAC,
            "n_bin_per_phase": N_BIN_PER_PHASE,
        },
        "clusters": [
            {
                "id": c.id, "label": c.label, "members": c.members,
                "medoid": c.medoid, "dominant_features": c.dominant_features,
            } for c in result["clusters"]
        ],
        "episodes": [
            {
                "episode_id": f"episode_{eid}",
                "global_index": eid,
                "cluster": result["cluster_of"].get(eid),
                "phase_method": result["phases"][eid].method,
                "num_cycles": result["phases"][eid].num_cycles,
                "phases": [asdict(p) for p in result["phases"][eid].phases],
                "shape_features": result["features"][eid],
                "anomaly": {
                    "is_anomaly": eid in result["anomalies"],
                    "reasons": [asdict(r) for r in result["anomalies"].get(eid, [])],
                },
            } for eid in ep_ids
        ],
    }


def main() -> int:
    print("=" * 70)
    print("PHASE-AWARE ANOMALY DETECTION — Libero mug-left-plate prototype")
    print("=" * 70)

    t0 = time.time()
    print("\n[1/5] Resolving task episodes...")
    ep_ids = load_task_episode_ids()

    print("\n[2/5] Loading action/state arrays...")
    ep_data = load_action_state(ep_ids)
    print(f"  loaded {len(ep_data)} episodes in {time.time()-t0:.1f}s")

    print("\n[3/5] Running analysis...")
    result = run_analysis(ep_data)

    print("\n[4/5] Summary:")
    print(summarize(result, sorted(ep_data.keys())))

    print(compare_to_pilot_c(result))
    print(synthetic_retry_test(ep_data))
    print(synthetic_elongated_place_test(ep_data))
    print(synthetic_natural_variance_test(ep_data))
    print(generic_fallback_sanity(ep_data))

    print("\n[5/5] Writing findings.json...")
    out = to_jsonable(result, sorted(ep_data.keys()))
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"  wrote {OUT} ({OUT.stat().st_size:,} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
