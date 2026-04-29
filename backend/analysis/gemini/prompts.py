"""Render Gemini prompts from YAML templates. Decision #7: prompts live in
YAML so they can be iterated without a deploy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class PromptTemplate:
    name: str
    version: int
    system: str
    user_template: str
    response_schema: dict[str, Any]


def _load(name: str) -> PromptTemplate:
    path = PROMPTS_DIR / f"{name}.yaml"
    with path.open() as f:
        raw = yaml.safe_load(f)
    return PromptTemplate(
        name=name,
        version=int(raw.get("version", 1)),
        system=raw["system"].strip(),
        user_template=raw["user_template"],
        response_schema=raw["response_schema"],
    )


def cluster_char() -> PromptTemplate:
    return _load("cluster_char")


def flag_enrich() -> PromptTemplate:
    return _load("flag_enrich")


def exhaustive_scan() -> PromptTemplate:
    return _load("exhaustive_scan")


def render_exhaustive_scan(
    *,
    template: PromptTemplate,
    task_name: str,
    normal_episode_ids: list[str],
    unlabeled_episode_ids: list[str],
) -> str:
    lines_n = []
    idx = 1
    for eid in normal_episode_ids:
        lines_n.append(f"  Video {idx}: {eid}")
        idx += 1
    n_normals = idx - 1
    normal_block = "\n".join(lines_n)
    first_unlabeled = idx
    lines_u = []
    for eid in unlabeled_episode_ids:
        lines_u.append(f"  Video {idx}: {eid}")
        idx += 1
    last_unlabeled = idx - 1
    unlabeled_block = "\n".join(lines_u)
    return template.user_template.format(
        task_name=task_name,
        n_normals=n_normals,
        normal_videos_block=normal_block,
        first_unlabeled=first_unlabeled,
        last_unlabeled=last_unlabeled,
        unlabeled_videos_block=unlabeled_block,
    )


def prompt_version_hash() -> str:
    """Cache-key fragment — bump when prompts change."""
    import hashlib
    h = hashlib.sha256()
    for p in sorted(PROMPTS_DIR.glob("*.yaml")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def render_cluster_char(
    *,
    template: PromptTemplate,
    task_name: str,
    fps: float,
    target_cluster_id: str,
    target_cluster_size: int,
    target_cluster_stat_label: str,
    target_dominant_features: list[str],
    cohort_size: int,
    medoid_episode_id: str,
    target_episode_ids: list[str],
    contrast_episode_ids_by_cluster: dict[str, list[str]],
    episode_contexts: dict[str, dict],  # episode_id -> {frames, num_cycles, raw_gripper_events, phase_action_summary}
) -> str:
    """Build the user prompt for one cluster-characterization call.

    v3 (2026-04-27): description and evidence must refer ONLY to the medoid
    (and optionally contrast medoids) — never to other target-cluster
    members. Eliminates the "ep_X is flagged" reading users got when Gemini
    cited multiple cluster members in a description.

    v2 retained: every video has a telemetry block; claims must be grounded
    in phase boundaries / gripper events, not video eyeballing.
    """
    def _emit(eid: str, prefix: str = "") -> list[str]:
        out = [f"  Video {idx_holder[0]}: {eid}{prefix}"]
        ctx = episode_contexts.get(eid)
        if ctx:
            out.append(_format_episode_context(
                episode_id=eid,
                frames=ctx["frames"],
                fps=fps,
                num_cycles=ctx["num_cycles"],
                raw_gripper_events=ctx.get("raw_gripper_events", []),
                phase_action_summary=ctx.get("phase_action_summary", []),
            ))
        idx_holder[0] += 1
        return out

    idx_holder = [1]  # mutable counter shared with _emit
    target_lines: list[str] = []
    for eid in target_episode_ids:
        target_lines.extend(_emit(eid))
    n_target = idx_holder[0] - 1
    target_block = "\n".join(target_lines)

    contrast_lines: list[str] = []
    for cid, eids in contrast_episode_ids_by_cluster.items():
        for eid in eids:
            contrast_lines.extend(_emit(eid, prefix=f" (from cluster {cid})"))
    contrast_block = "\n".join(contrast_lines) if contrast_lines else "  (this is the only cluster)"

    return template.user_template.format(
        task_name=task_name,
        fps=fps,
        target_cluster_id=target_cluster_id,
        target_cluster_size=target_cluster_size,
        target_cluster_stat_label=target_cluster_stat_label,
        target_dominant_features=", ".join(target_dominant_features) if target_dominant_features else "(none)",
        cohort_size=cohort_size,
        medoid_episode_id=medoid_episode_id,
        n_target=n_target,
        target_videos_block=target_block,
        contrast_videos_block=contrast_block,
    )


def _format_episode_context(
    *,
    episode_id: str,
    frames: int,
    fps: float,
    num_cycles: int,
    raw_gripper_events: list[dict],
    phase_action_summary: list[dict],
) -> str:
    """Render a compact per-episode context block for the flag-enrich prompt.
    Surfaces gripper events + phase timings in seconds + per-phase magnitudes
    so Gemini can correlate video with signal."""
    def sec(f):
        return f / fps if fps else f
    lines: list[str] = []
    lines.append(f"    frames={frames} @ {fps:g}Hz · cycles={num_cycles}")

    if phase_action_summary:
        parts = []
        for p in phase_action_summary:
            if p["name"] in ("Grasp", "Release"):
                # Narrow windows — include name + timing only, skip magnitudes
                parts.append(
                    f"{p['name']}c{p['cycle']} {sec(p['frame_start']):.1f}-{sec(p['frame_end']):.1f}s"
                )
            else:
                parts.append(
                    f"{p['name']}c{p['cycle']} {sec(p['frame_start']):.1f}-{sec(p['frame_end']):.1f}s "
                    f"(posM={p['pos_mag_mean']}/{p['pos_mag_max']},"
                    f"rotM={p['rot_mag_mean']}/{p['rot_mag_max']})"
                )
        lines.append("    phases: " + ", ".join(parts))

    if raw_gripper_events:
        events_str = ", ".join(
            f"{ev['type']}@{sec(ev['frame']):.2f}s"
            for ev in raw_gripper_events
        )
        expected = num_cycles * 2
        extra = len(raw_gripper_events) - expected
        note = ""
        if extra > 0:
            note = f" ← {extra} extra event(s) vs 2×cycles={expected}: probable re-grip / failed-grasp"
        lines.append(f"    gripper events ({len(raw_gripper_events)}): [{events_str}]{note}")

    return "\n".join(lines)


def render_flag_enrich(
    *,
    template: PromptTemplate,
    task_name: str,
    fps: float,
    normal_episode_ids: list[str],
    normal_episode_contexts: dict[str, dict],  # episode_id -> {frames, num_cycles, raw_gripper_events, phase_action_summary}
    flagged_episodes: list[dict],  # [{"episode_id": str, "frames": int, "num_cycles": int, "raw_gripper_events": [...], "phase_action_summary": [...], "reasons": [...]}]
) -> str:
    """Build the user prompt for one flag-enrichment call, now with
    gripper-event / phase-timing context per episode."""
    lines_n = []
    idx = 1
    for eid in normal_episode_ids:
        lines_n.append(f"  Video {idx}: {eid}")
        ctx = normal_episode_contexts.get(eid)
        if ctx:
            lines_n.append(_format_episode_context(
                episode_id=eid,
                frames=ctx["frames"],
                fps=fps,
                num_cycles=ctx["num_cycles"],
                raw_gripper_events=ctx.get("raw_gripper_events", []),
                phase_action_summary=ctx.get("phase_action_summary", []),
            ))
        idx += 1
    n_normals = idx - 1 - sum(1 for e in normal_episode_ids if normal_episode_contexts.get(e))  # still the number of videos
    # Actually: n_normals should be the count of video lines, which is len(normal_episode_ids).
    n_normals = len(normal_episode_ids)
    normal_block = "\n".join(lines_n)

    first_flagged_video = idx
    lines_f = []
    for ep in flagged_episodes:
        reason_summary = "; ".join(r["explanation"] for r in ep["reasons"]) or "(no reasons)"
        lines_f.append(f"  Video {idx}: {ep['episode_id']} — stat flag: \"{reason_summary}\"")
        lines_f.append(_format_episode_context(
            episode_id=ep["episode_id"],
            frames=ep["frames"],
            fps=fps,
            num_cycles=ep["num_cycles"],
            raw_gripper_events=ep.get("raw_gripper_events", []),
            phase_action_summary=ep.get("phase_action_summary", []),
        ))
        idx += 1
    last_flagged_video = idx - 1
    flagged_block = "\n".join(lines_f)

    return template.user_template.format(
        task_name=task_name,
        fps=fps,
        n_normals=n_normals,
        normal_videos_block=normal_block,
        first_flagged_video=first_flagged_video,
        last_flagged_video=last_flagged_video,
        flagged_videos_block=flagged_block,
    )
