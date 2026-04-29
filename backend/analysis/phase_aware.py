"""Phase-aware anomaly detection core.

Ported from `exploration/phase_aware/prototype.py` after Phase 0 validation
on Libero. See `exploration/phase_aware/FINDINGS.md` for rationale behind the
algorithm choices and thresholds.

Inputs:
    episodes: list of {"episode_id": "episode_N", "action": (T, D), "state": (T, S)}
    task_name: string (for labeling)

Outputs:
    PhaseAwareResult with per-episode phase segmentation, shape features,
    anomaly reasons (signal lineage), and variance clusters.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, medfilt
from scipy.stats import chi2
from sklearn.cluster import AgglomerativeClustering

# ========================================================================
# Algorithm constants (mirrored from prototype.py — see Phase 0 FINDINGS.md)
# ========================================================================

# The LeRobot convention places the gripper signal at the LAST action dimension.
# This is true for Libero/Franka (7-D: xyz + rpy + gripper, idx 6) and SO-ARM101
# (6-D: 5 joints + gripper, idx 5), so we derive it per-array rather than hardcode.
STATE_DIM_Z = 2

N_BIN_PER_PHASE = 50
CONTIGUOUS_RUN_FRAC = 0.40
CONTIGUOUS_RUN_MIN = 10
ENVELOPE_SIGMA = 3.0
ENVELOPE_MIN_COHORT = 6
ENVELOPE_MIN_MEDIAN_DURATION = 15
ENVELOPE_SMOOTH_SIGMA = 2.0
DURATION_MAD_THRESHOLD = 3.5
DURATION_MIN_MEDIAN = 10
CYCLE_COUNT_MIN_DELTA = 2
MAD_CONST = 1.4826
SHAPE_CONFIDENCE = 0.99
GAP_K_RANGE = (2, 3, 4, 5)
GAP_REFS = 20
GAP_RNG_SEED = 42

GRIPPER_MIN_CLOSED = 20
GRIPPER_MERGE_OPEN_GAP = 30
GRASP_WINDOW = 3
RELEASE_WINDOW = 3

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


# ========================================================================
# Dataclasses
# ========================================================================

@dataclass
class Phase:
    name: str
    start: int
    end: int
    cycle: int

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass
class AnomalyReason:
    signal: str  # "envelope" | "duration" | "shape" | "cycle_count"
    phase: Optional[str] = None
    cycle: Optional[int] = None
    feature: Optional[str] = None
    magnitude: float = 0.0
    explanation: str = ""


@dataclass
class PhaseSegmentation:
    phases: list[Phase]
    method: str  # "gripper" | "velocity_minima"
    num_cycles: int
    gripper_closed_ranges: list[tuple[int, int]]
    raw_gripper_transitions: int = 0
    # Pre-merge gripper events (includes micro-regrips later absorbed into one
    # logical pick-place). Each is (kind, frame_idx) where kind is "close" | "open".
    raw_gripper_events: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class Cluster:
    id: str
    label: str
    members: list[str]  # episode_ids
    medoid: str
    dominant_features: list[str]
    dominant_features_human: list[str] = field(default_factory=list)


@dataclass
class EpisodeReport:
    episode_id: str
    cluster: Optional[str]
    num_cycles: int
    frames: int
    phases: list[Phase]
    shape_features: dict[str, float]
    anomaly: dict  # {"is_anomaly": bool, "reasons": [AnomalyReason]}
    # Pre-merge gripper event log: [("close" | "open", frame_idx), …]
    raw_gripper_events: list[tuple[str, int]] = field(default_factory=list)
    # Per-phase summary of position / rotation magnitudes (mean, max) — helps
    # Gemini correlate video observations with signal intensity.
    phase_action_summary: list[dict] = field(default_factory=list)


@dataclass
class PhaseAwareResult:
    task_name: str
    cohort_size: int
    fps: float
    algorithm: dict
    clusters: list[Cluster]
    episodes: list[EpisodeReport]

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "cohort_size": self.cohort_size,
            "fps": self.fps,
            "algorithm": self.algorithm,
            "clusters": [asdict(c) for c in self.clusters],
            "episodes": [
                {
                    "episode_id": e.episode_id,
                    "cluster": e.cluster,
                    "num_cycles": e.num_cycles,
                    "frames": e.frames,
                    "phases": [asdict(p) for p in e.phases],
                    "shape_features": e.shape_features,
                    "anomaly": {
                        "is_anomaly": e.anomaly["is_anomaly"],
                        "reasons": [asdict(r) for r in e.anomaly["reasons"]],
                    },
                    "raw_gripper_events": [
                        {"type": t, "frame": f} for t, f in e.raw_gripper_events
                    ],
                    "phase_action_summary": e.phase_action_summary,
                }
                for e in self.episodes
            ],
        }


# ========================================================================
# Phase segmentation
# ========================================================================

def _detect_gripper_closed_ranges(action: np.ndarray) -> tuple[list[tuple[int, int]], int, list[tuple[str, int]]]:
    """Return (merged_closed_ranges, raw_transition_count, raw_events).

    Libero/LeRobot pick-place: action[GRIPPER] positive = close command,
    negative = open command. Convention is auto-flipped if the "closed"
    state turns out to be majority of the episode.

    `raw_events` is the full pre-merge transition log, each entry
    ("close" | "open", frame_idx). Used to surface mid-cycle re-grips
    (which the merge step collapses into a single logical cycle) as context
    for downstream semantic analysis.
    """
    if action.ndim != 2 or action.shape[1] < 1:
        return [], 0, []
    gripper_dim = action.shape[1] - 1
    g = action[:, gripper_dim].astype(float)
    if g.max() == g.min():
        return [], 0, []
    ksize = min(15, len(g) if len(g) % 2 == 1 else len(g) - 1)
    sm = medfilt(g, kernel_size=max(3, ksize))
    lo_ref = float(np.percentile(sm, 5))
    hi_ref = float(np.percentile(sm, 95))
    mid = (lo_ref + hi_ref) / 2
    is_closed = sm > mid
    if is_closed.sum() > 0.6 * len(is_closed):
        is_closed = sm < mid  # Convention flipped

    diff = np.diff(is_closed.astype(int))
    close_events = np.where(diff == 1)[0] + 1
    open_events = np.where(diff == -1)[0] + 1

    if is_closed[0]:
        close_events = np.concatenate([[0], close_events])
    if is_closed[-1]:
        open_events = np.concatenate([open_events, [len(is_closed)]])

    raw_ranges = list(zip(close_events.tolist(), open_events.tolist()))
    raw_transition_count = len(close_events) + len(open_events)

    # Build a chronologically-ordered raw event log for downstream analysis.
    events: list[tuple[str, int]] = []
    for f in close_events.tolist():
        events.append(("close", int(f)))
    for f in open_events.tolist():
        events.append(("open", int(f)))
    events.sort(key=lambda ev: ev[1])

    ranges = [(a, b) for a, b in raw_ranges if b - a >= GRIPPER_MIN_CLOSED]
    merged: list[tuple[int, int]] = []
    for a, b in ranges:
        if merged and a - merged[-1][1] < GRIPPER_MERGE_OPEN_GAP:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged, raw_transition_count, events


def detect_phases(action: np.ndarray, state: np.ndarray, *, force_generic: bool = False) -> PhaseSegmentation:
    T = len(action)
    z = state[:, STATE_DIM_Z] if state.shape[1] > STATE_DIM_Z else np.zeros(T)

    if force_generic:
        closed_ranges, raw_trans, raw_events = [], 0, []
    else:
        closed_ranges, raw_trans, raw_events = _detect_gripper_closed_ranges(action)

    if len(closed_ranges) == 0:
        # Fallback: velocity-minima split
        v = np.linalg.norm(np.diff(action[:, 0:3], axis=0), axis=1)
        v = gaussian_filter1d(v, sigma=3) if len(v) > 3 else v
        thresh = float(np.percentile(v, 20)) if len(v) else 0.0
        peaks, _ = find_peaks(-v, height=-thresh, distance=max(10, T // 20))
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
        return PhaseSegmentation(phases=phases, method="velocity_minima",
                                 num_cycles=0, gripper_closed_ranges=[],
                                 raw_gripper_transitions=raw_trans,
                                 raw_gripper_events=raw_events)

    # Gripper-cycle phases. Strictly chained (end of one = start of next).
    phases: list[Phase] = []
    prev_end = 0
    for cycle_idx, (close_i, open_i) in enumerate(closed_ranges):
        # Approach: prev_end → close_i
        if close_i > prev_end:
            phases.append(Phase("Approach", prev_end, close_i, cycle_idx))
        # Grasp: [close_i, close_i + GRASP_WINDOW)
        grasp_end = min(close_i + GRASP_WINDOW, open_i)
        if grasp_end > close_i:
            phases.append(Phase("Grasp", close_i, grasp_end, cycle_idx))
        # Lift / Transit / Place: split the interior by z-peak
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

        if z_ascend_end > interior_start:
            phases.append(Phase("Lift", interior_start, z_ascend_end, cycle_idx))
        if descend_start > z_ascend_end:
            phases.append(Phase("Transit", z_ascend_end, descend_start, cycle_idx))
        if open_i > descend_start:
            phases.append(Phase("Place", descend_start, open_i, cycle_idx))

        # Release: [open_i, open_i + RELEASE_WINDOW)
        release_end = min(open_i + RELEASE_WINDOW, T)
        if release_end > open_i:
            phases.append(Phase("Release", open_i, release_end, cycle_idx))
        prev_end = release_end

    if prev_end < T:
        phases.append(Phase("Return", prev_end, T, cycle=len(closed_ranges) - 1))

    phases = [p for p in phases if p.end > p.start]

    return PhaseSegmentation(phases=phases, method="gripper",
                             num_cycles=len(closed_ranges),
                             gripper_closed_ranges=closed_ranges,
                             raw_gripper_transitions=raw_trans,
                             raw_gripper_events=raw_events)


# ========================================================================
# Envelope flagging
# ========================================================================

def _resample_phase(signal: np.ndarray, start: int, end: int, n: int) -> np.ndarray:
    seg = signal[start:end]
    if len(seg) < 2:
        return np.full(n, float("nan"))
    t_src = np.linspace(0, 1, len(seg))
    t_dst = np.linspace(0, 1, n)
    return np.interp(t_dst, t_src, seg.astype(float))


def _phase_key(p: Phase, num_cycles: int) -> str:
    return f"{p.name}#c{p.cycle}#n{num_cycles}"


def _contiguous_run_lengths(mask: np.ndarray) -> list[int]:
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


def _flag_envelope_violations(
    episodes: dict[str, dict],
    phases: dict[str, PhaseSegmentation],
) -> dict[str, list[AnomalyReason]]:
    def pos_mag(a: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a[:, 0:3], axis=1)

    def rot_mag(a: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a[:, 3:6], axis=1) if a.shape[1] >= 6 else np.zeros(len(a))

    channels = {"position": pos_mag, "rotation": rot_mag}
    buckets: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    durations: dict[str, list[int]] = {}
    for eid, seg in phases.items():
        a = episodes[eid]["action"]
        for p in seg.phases:
            if p.name in ("Grasp", "Release"):
                continue
            key = _phase_key(p, seg.num_cycles)
            buckets.setdefault(key, {ch: {} for ch in channels})
            durations.setdefault(key, []).append(p.end - p.start)
            for ch_name, fn in channels.items():
                sig = fn(a)
                rs = _resample_phase(sig, p.start, p.end, N_BIN_PER_PHASE)
                if not np.isnan(rs).any():
                    buckets[key][ch_name][eid] = gaussian_filter1d(rs, sigma=ENVELOPE_SMOOTH_SIGMA)

    flags: dict[str, list[AnomalyReason]] = {}
    for key, ch_map in buckets.items():
        try:
            name_part, rest = key.split("#c", 1)
            cycle_part, _n = rest.split("#n", 1)
            phase_name = name_part
            cycle = int(cycle_part)
        except ValueError:
            continue
        med_dur = int(np.median(durations[key])) if durations[key] else 0
        if med_dur < ENVELOPE_MIN_MEDIAN_DURATION:
            continue
        for ch_name, eid_sigs in ch_map.items():
            if len(eid_sigs) < ENVELOPE_MIN_COHORT:
                continue
            ids = sorted(eid_sigs.keys())
            X = np.stack([eid_sigs[e] for e in ids])
            mean = X.mean(0)
            std = np.clip(X.std(0), 1e-6, None)
            for eid in ids:
                z = np.abs(eid_sigs[eid] - mean) / std
                outside = z > ENVELOPE_SIGMA
                run_lengths = _contiguous_run_lengths(outside)
                longest = max(run_lengths, default=0)
                min_run = max(CONTIGUOUS_RUN_MIN, int(CONTIGUOUS_RUN_FRAC * N_BIN_PER_PHASE))
                if longest >= min_run:
                    peak_z = float(z[outside].max()) if outside.any() else 0.0
                    # Describe what the operator did visibly, not the
                    # statistical machinery. Drop σ/χ² notation; keep the
                    # actionable numbers.
                    run_pct = int(round(100 * longest / N_BIN_PER_PHASE))
                    reason = AnomalyReason(
                        signal="envelope",
                        phase=phase_name,
                        cycle=cycle,
                        feature=ch_name,
                        magnitude=round(peak_z, 2),
                        explanation=(
                            f"Unusual {ch_name} during {phase_name} (cycle {cycle}): "
                            f"peak {peak_z:.1f}× typical range, sustained for {run_pct}% of the phase"
                        ),
                    )
                    flags.setdefault(eid, []).append(reason)
    return flags


# ========================================================================
# Duration flagging
# ========================================================================

def _duration_label(phase_name: str, ratio: float) -> str:
    if ratio > 2.0:
        if phase_name == "Place":
            return "hesitation before release"
        if phase_name == "Grasp":
            return "difficulty grasping"
        if phase_name == "Approach":
            return "slow approach"
        if phase_name == "Transit":
            return "slow transit"
        if phase_name == "Lift":
            return "prolonged lift"
        if phase_name == "Return":
            return "long idle after release"
        return "prolonged"
    if ratio < 0.4:
        if phase_name == "Approach":
            return "cut-short approach"
        if phase_name == "Place":
            return "abrupt placement"
        if phase_name == "Return":
            return "immediate reset after release"
        return "abbreviated"
    if phase_name == "Return":
        return "longer-than-typical idle after release"
    return "slightly off-typical pacing"


def _flag_duration_outliers(
    phases: dict[str, PhaseSegmentation],
) -> dict[str, list[AnomalyReason]]:
    key_ep_dur: dict[str, dict[str, int]] = {}
    cycle_counts: dict[str, int] = {}
    for eid, seg in phases.items():
        cycle_counts[eid] = seg.num_cycles
        for p in seg.phases:
            if p.name in ("Grasp", "Release"):
                continue
            key = _phase_key(p, seg.num_cycles)
            key_ep_dur.setdefault(key, {})[eid] = p.end - p.start

    flags: dict[str, list[AnomalyReason]] = {}
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
        durs = np.array(list(ep_dur.values()), dtype=float)
        med = float(np.median(durs))
        if med < DURATION_MIN_MEDIAN:
            continue
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
                    explanation=f"{label.capitalize()}: {phase_name} phase (cycle {cycle}) took {ratio:.1f}× typical time ({d}f vs typical {med:.0f}f)",
                )
                flags.setdefault(eid, []).append(reason)

    counts = np.array(list(cycle_counts.values()))
    if len(counts) >= 4 and counts.max() > 0:
        mode = int(np.bincount(counts).argmax())
        # Adaptive threshold: for single-pick tasks (mode=1) OR small cohorts,
        # a 1-off cycle count IS the signal (e.g., UMI's episode_9 had 2 cycles
        # while all others had 1 — legitimate re-attempt anomaly). For larger
        # multi-pick cohorts (mode >= 2), require ≥2 off to cut noise.
        effective_min_delta = 1 if mode <= 1 or len(counts) < 15 else CYCLE_COUNT_MIN_DELTA
        for eid, nc in cycle_counts.items():
            if abs(nc - mode) >= effective_min_delta:
                label = "missing cycles" if nc < mode else "extra cycles"
                reason = AnomalyReason(
                    signal="cycle_count",
                    magnitude=float(abs(nc - mode)),
                    explanation=f"{label.capitalize()}: {nc} pick-place cycle{'s' if nc != 1 else ''} vs typical {mode}",
                )
                flags.setdefault(eid, []).append(reason)

    return flags


# ========================================================================
# Shape descriptors + Mahalanobis
# ========================================================================

def _compute_shape_features(action: np.ndarray, state: np.ndarray, seg: PhaseSegmentation) -> dict[str, float]:
    T = len(action)
    xyz = state[:, 0:3] if state.shape[1] >= 3 else np.zeros((T, 3))
    dxyz = np.diff(xyz, axis=0) if T > 1 else np.zeros((0, 3))
    arc_xyz = float(np.linalg.norm(dxyz, axis=1).sum()) if len(dxyz) else 0.0
    drpy = np.diff(action[:, 3:6], axis=0) if action.shape[1] >= 6 and T > 1 else np.zeros((0, 3))
    arc_rot = float(np.linalg.norm(drpy, axis=1).sum()) if len(drpy) else 0.0

    v = np.linalg.norm(dxyz, axis=1) if len(dxyz) else np.zeros(1)
    v_sm = gaussian_filter1d(v, sigma=3) if len(v) > 3 else v
    thresh = float(np.percentile(v_sm, 25)) if len(v_sm) else 0.0
    in_pause = v_sm < thresh
    pauses = sum(1 for r in _contiguous_run_lengths(in_pause) if r >= 10)

    z_range = float(xyz[:, 2].max() - xyz[:, 2].min()) if T else 0.0
    xy_bbox = float((xyz[:, 0].max() - xyz[:, 0].min()) * (xyz[:, 1].max() - xyz[:, 1].min())) if T else 0.0
    straight = float(np.linalg.norm(xyz[-1] - xyz[0])) if T else 0.0
    directness = straight / arc_xyz if arc_xyz > 0 else 0.0

    closed_frames = sum(b - a for a, b in seg.gripper_closed_ranges)
    gripper_ratio = closed_frames / T if T else 0.0

    dir_changes = 0
    for i in range(3):
        if i >= dxyz.shape[1]:
            break
        s = np.sign(dxyz[:, i])
        s[s == 0] = 1
        dir_changes += int(np.sum(np.diff(s) != 0)) if len(s) > 1 else 0

    return {
        "arc_length_xyz": round(arc_xyz, 4),
        "arc_length_rot": round(arc_rot, 4),
        "num_gripper_events": seg.raw_gripper_transitions,
        "num_pauses": pauses,
        "z_range": round(z_range, 4),
        "xy_bbox_area": round(xy_bbox, 4),
        "path_directness": round(directness, 4),
        "gripper_closed_ratio": round(gripper_ratio, 4),
        "num_direction_changes": dir_changes,
        "episode_duration": T,
    }


def _shape_label(dominant: list[tuple[str, float]], feats: dict) -> str:
    if not dominant:
        return "multi-axis deviation without one dominant cause"
    parts = []
    for n, c in dominant:
        val = feats[n]
        direction = "high" if c > 0 else "low"
        if n == "num_gripper_events":
            parts.append(f"{val} gripper open/close events (more than typical)")
        elif n == "num_pauses":
            plural = "pauses" if val != 1 else "pause"
            parts.append(f"{val} velocity {plural} mid-trajectory ({'many' if direction == 'high' else 'few'})")
        elif n == "path_directness":
            parts.append(
                "path wanders noticeably (not direct start-to-end)"
                if direction == "low"
                else "path is unusually straight"
            )
        elif n == "arc_length_xyz":
            parts.append("much more total translation than typical" if direction == "high" else "much less total translation than typical")
        elif n == "arc_length_rot":
            parts.append("much more wrist rotation than typical" if direction == "high" else "much less wrist rotation than typical")
        elif n == "num_direction_changes":
            parts.append(f"jerky motion ({val} velocity sign flips)" if direction == "high" else f"unusually smooth motion ({val} velocity sign flips)")
        elif n == "episode_duration":
            parts.append("much longer than typical" if direction == "high" else "much shorter than typical")
        elif n == "z_range":
            parts.append("wider vertical range than typical" if direction == "high" else "lower ceiling than typical")
        elif n == "xy_bbox_area":
            parts.append("wider workspace than typical" if direction == "high" else "compact workspace")
        elif n == "gripper_closed_ratio":
            parts.append("gripper held closed longer than typical" if direction == "high" else "gripper held closed less than typical")
        else:
            parts.append(f"{n} is {direction} (={val:.3f})")
    return "; ".join(parts)


def _flag_shape_outliers(features: dict[str, dict[str, float]]) -> dict[str, list[AnomalyReason]]:
    ids = sorted(features.keys())
    if not ids:
        return {}
    X = np.array([[features[e][n] for n in SHAPE_FEATURE_NAMES] for e in ids], dtype=float)
    mu = X.mean(0)
    sigma = np.clip(X.std(0), 1e-9, None)
    Z = (X - mu) / sigma
    try:
        cov = np.cov(Z, rowvar=False)
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        inv = np.eye(X.shape[1])

    threshold = chi2.ppf(SHAPE_CONFIDENCE, df=X.shape[1])
    flags: dict[str, list[AnomalyReason]] = {}
    for i, eid in enumerate(ids):
        d2 = float(Z[i] @ inv @ Z[i])
        if d2 > threshold:
            contribs = (Z[i] @ inv) * Z[i]
            total = d2 if d2 > 0 else 1.0
            ranked = sorted(zip(SHAPE_FEATURE_NAMES, contribs), key=lambda x: -abs(x[1]))
            dominant = [(n, c / total) for n, c in ranked[:2] if abs(c) / total > 0.1]
            label = _shape_label(dominant, features[eid])
            reason = AnomalyReason(
                signal="shape",
                feature=",".join(n for n, _ in dominant) if dominant else "combined",
                magnitude=round(d2, 2),
                explanation=f"Unusual trajectory shape: {label}",
            )
            flags.setdefault(eid, []).append(reason)
    return flags


# ========================================================================
# Variance clustering
# ========================================================================

def _wss(X: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for k in np.unique(labels):
        xk = X[labels == k]
        if len(xk) == 0:
            continue
        total += float(((xk - xk.mean(0)) ** 2).sum())
    return total


def _gap_statistic_k(X: np.ndarray, k_range=GAP_K_RANGE, n_refs=GAP_REFS, seed=GAP_RNG_SEED) -> int:
    rng = np.random.default_rng(seed)
    lo, hi = X.min(0), X.max(0)
    real = np.zeros(len(k_range))
    ref = np.zeros((n_refs, len(k_range)))

    def fit(X_, k):
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_)

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


# Feature name → human-readable description. Used on cluster tag chips in the
# UI ("Vertical range" instead of `z_range`) and as the fallback text when
# building cluster titles. Kept in one place so the backend and any future
# debug tooling agree on wording.
FEATURE_HUMAN: dict[str, str] = {
    "episode_duration": "Duration",
    "num_pauses": "Mid-motion pauses",
    "path_directness": "Path directness",
    "arc_length_xyz": "Total travel",
    "num_gripper_events": "Gripper events",
    "arc_length_rot": "Wrist rotation",
    "xy_bbox_area": "Workspace footprint",
    "z_range": "Vertical range",
    "gripper_closed_ratio": "Carrying time",
    "num_direction_changes": "Motion smoothness",
}

# Exhaustive cluster-title lookup. Every (feature, direction) pair produces
# a short behavioural phrase — avoids the old fallback "Cluster (foo=low)"
# that made C1/C5 unreadable when the dominant axis wasn't one of the six
# we'd previously hand-written.
CLUSTER_TITLE: dict[tuple[str, str], str] = {
    ("episode_duration",     "low"):  "Rapid execution",
    ("episode_duration",     "high"): "Deliberate / slow",
    ("num_pauses",           "low"):  "Steady / uninterrupted",
    ("num_pauses",           "high"): "Hesitant / many pauses",
    ("path_directness",      "low"):  "Indirect / wandering path",
    ("path_directness",      "high"): "Direct / efficient path",
    ("arc_length_xyz",       "low"):  "Compact motion",
    ("arc_length_xyz",       "high"): "Wide-ranging motion",
    ("num_gripper_events",   "low"):  "Clean single-cycle pick-place",
    ("num_gripper_events",   "high"): "Multi-cycle / re-grip attempts",
    ("arc_length_rot",       "low"):  "Minimal wrist rotation",
    ("arc_length_rot",       "high"): "Heavy wrist rotation",
    ("xy_bbox_area",         "low"):  "Narrow workspace",
    ("xy_bbox_area",         "high"): "Wide workspace",
    ("z_range",              "low"):  "Low-clearance / shallow lift",
    ("z_range",              "high"): "High-clearance / tall lift",
    ("gripper_closed_ratio", "low"):  "Brief grasping",
    ("gripper_closed_ratio", "high"): "Extended carrying time",
    ("num_direction_changes","low"):  "Smooth motion",
    ("num_direction_changes","high"): "Jerky motion",
}


def _cluster_label(top: str, direction: str) -> str:
    title = CLUSTER_TITLE.get((top, direction))
    if title is not None:
        return title
    # Unknown feature (shouldn't happen with the current SHAPE_FEATURE_NAMES);
    # fall back to a sensible human-ish description rather than "Cluster(...)".
    human = FEATURE_HUMAN.get(top, top)
    return f"{human} {'high' if direction == 'high' else 'low'}"


def _cluster_variance_modes(features: dict[str, dict[str, float]]) -> list[Cluster]:
    ids = sorted(features.keys())
    if len(ids) < 4:
        return []
    X = np.array([[features[e][n] for n in SHAPE_FEATURE_NAMES] for e in ids], dtype=float)
    mu = X.mean(0)
    sigma = np.clip(X.std(0), 1e-9, None)
    Z = (X - mu) / sigma

    k = _gap_statistic_k(Z)
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)

    clusters: list[Cluster] = []
    for ki in range(k):
        mask = labels == ki
        members = [ids[i] for i in np.where(mask)[0]]
        if not members:
            continue
        Zc = Z[mask]
        d2 = ((Zc[:, None, :] - Zc[None, :, :]) ** 2).sum(-1)
        medoid_local = int(d2.sum(0).argmin())
        medoid = members[medoid_local]
        diff = Z[mask].mean(0)
        ranked = sorted(zip(SHAPE_FEATURE_NAMES, diff), key=lambda x: -abs(x[1]))
        dominant = [n for n, v in ranked[:3] if abs(v) > 0.5]
        if not dominant:
            lbl = f"Mode {ki + 1}"
        else:
            top, top_v = ranked[0]
            direction = "high" if top_v > 0 else "low"
            lbl = _cluster_label(top, direction)
        clusters.append(Cluster(
            id=f"C{ki + 1}",
            label=lbl,
            members=members,
            medoid=medoid,
            dominant_features=dominant,
            dominant_features_human=[FEATURE_HUMAN.get(d, d) for d in dominant],
        ))
    return clusters


# ========================================================================
# Top-level driver
# ========================================================================

def _compute_phase_action_summary(action: np.ndarray, seg: PhaseSegmentation) -> list[dict]:
    """For each non-narrow phase, compute mean/peak position & rotation
    magnitudes. Helps Gemini correlate video observations with signal intensity.
    """
    if action.shape[1] < 6:
        return []
    pos_mag = np.linalg.norm(action[:, 0:3], axis=1)
    rot_mag = np.linalg.norm(action[:, 3:6], axis=1)
    out = []
    for p in seg.phases:
        if p.end <= p.start:
            continue
        pm_seg = pos_mag[p.start:p.end]
        rm_seg = rot_mag[p.start:p.end]
        if len(pm_seg) == 0:
            continue
        out.append({
            "name": p.name,
            "cycle": p.cycle,
            "frame_start": int(p.start),
            "frame_end": int(p.end),
            "duration_frames": int(p.end - p.start),
            "pos_mag_mean": round(float(pm_seg.mean()), 3),
            "pos_mag_max": round(float(pm_seg.max()), 3),
            "rot_mag_mean": round(float(rm_seg.mean()), 3),
            "rot_mag_max": round(float(rm_seg.max()), 3),
        })
    return out


def analyze_task(
    episodes: list[dict],
    task_name: str,
    fps: float = 10.0,
) -> PhaseAwareResult:
    """Run the full phase-aware pipeline.

    Args:
        episodes: [{"episode_id": "episode_N", "action": np.ndarray (T, D),
                    "state": np.ndarray (T, S)}]
        task_name: task description (for labeling)
        fps: frames-per-second of the action/state signal (used to convert
             frame indices to seconds when rendering prompts for Gemini)

    Returns:
        PhaseAwareResult
    """
    # Index episodes by id
    ep_by_id: dict[str, dict] = {}
    for e in episodes:
        if "action" not in e or "state" not in e:
            continue
        ep_by_id[e["episode_id"]] = {
            "action": np.asarray(e["action"], dtype=float),
            "state": np.asarray(e["state"], dtype=float),
        }

    phases = {eid: detect_phases(d["action"], d["state"]) for eid, d in ep_by_id.items()}
    envelope = _flag_envelope_violations(ep_by_id, phases)
    duration = _flag_duration_outliers(phases)
    features = {eid: _compute_shape_features(d["action"], d["state"], phases[eid])
                for eid, d in ep_by_id.items()}
    shape_flags = _flag_shape_outliers(features)
    clusters = _cluster_variance_modes(features)

    cluster_of: dict[str, str] = {}
    for c in clusters:
        for m in c.members:
            cluster_of[m] = c.id

    reports: list[EpisodeReport] = []
    for eid in sorted(ep_by_id.keys(), key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else s):
        reasons = envelope.get(eid, []) + duration.get(eid, []) + shape_flags.get(eid, [])
        summary = _compute_phase_action_summary(ep_by_id[eid]["action"], phases[eid])
        reports.append(EpisodeReport(
            episode_id=eid,
            cluster=cluster_of.get(eid),
            num_cycles=phases[eid].num_cycles,
            frames=int(len(ep_by_id[eid]["action"])),
            phases=phases[eid].phases,
            shape_features=features[eid],
            anomaly={"is_anomaly": len(reasons) > 0, "reasons": reasons},
            raw_gripper_events=phases[eid].raw_gripper_events,
            phase_action_summary=summary,
        ))

    return PhaseAwareResult(
        task_name=task_name,
        cohort_size=len(reports),
        fps=fps,
        algorithm={
            "envelope_sigma": ENVELOPE_SIGMA,
            "duration_mad_threshold": DURATION_MAD_THRESHOLD,
            "shape_confidence": SHAPE_CONFIDENCE,
            "contiguous_run_min": CONTIGUOUS_RUN_MIN,
            "contiguous_run_frac": CONTIGUOUS_RUN_FRAC,
            "n_bin_per_phase": N_BIN_PER_PHASE,
            "cycle_count_min_delta": CYCLE_COUNT_MIN_DELTA,
        },
        clusters=clusters,
        episodes=reports,
    )
