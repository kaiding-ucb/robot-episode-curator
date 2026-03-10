"""
Exp 3: Quality Events on Timeline

Goal: Can quality events be meaningfully displayed in Rerun?

Streams an episode, computes quality metrics (stalls, divergence,
action discontinuities), and logs them as Rerun events.
"""

import argparse
import io
import sys
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

HF_TOKEN = "REDACTED-HF-TOKEN"
REPO_ID = "lerobot/libero_spatial_image"


def compute_quality_metrics(actions: np.ndarray, fps: int = 10):
    """
    Compute quality metrics from action array.
    Lightweight version of backend/quality/ modules.
    Returns dict of metric arrays and event lists.
    """
    n_frames = len(actions)
    if n_frames < 3:
        return {"events": [], "action_magnitude": np.zeros(n_frames), "smoothness": np.zeros(n_frames)}

    # Action magnitude per frame
    action_mag = np.linalg.norm(actions, axis=1)

    # Action deltas (velocity of action change)
    action_deltas = np.diff(actions, axis=0)
    delta_mag = np.linalg.norm(action_deltas, axis=1)

    # Smoothness: second derivative of actions (jerk)
    if n_frames >= 3:
        action_jerk = np.diff(action_deltas, axis=0)
        jerk_mag = np.linalg.norm(action_jerk, axis=1)
        # Pad to match frame count
        smoothness = np.concatenate([[0], delta_mag])
        jerk = np.concatenate([[0, 0], jerk_mag])
    else:
        smoothness = np.zeros(n_frames)
        jerk = np.zeros(n_frames)

    # Detect events
    events = []

    # 1. Stall detection: low action magnitude for consecutive frames
    stall_threshold = np.percentile(action_mag, 15)
    in_stall = False
    stall_start = 0
    for i in range(n_frames):
        if action_mag[i] < stall_threshold * 0.5:
            if not in_stall:
                stall_start = i
                in_stall = True
        else:
            if in_stall and (i - stall_start) >= 5:
                events.append({
                    "frame": stall_start,
                    "end_frame": i,
                    "type": "stall",
                    "severity": "warn",
                    "message": f"Stall detected: frames {stall_start}-{i} ({i-stall_start} frames)",
                })
            in_stall = False

    # 2. Action discontinuity: sudden large jumps
    if len(delta_mag) > 0:
        jump_threshold = np.percentile(delta_mag, 95) * 2
        for i in range(len(delta_mag)):
            if delta_mag[i] > jump_threshold:
                events.append({
                    "frame": i + 1,
                    "type": "discontinuity",
                    "severity": "error",
                    "message": f"Action jump at frame {i+1} (delta={delta_mag[i]:.3f}, threshold={jump_threshold:.3f})",
                })

    # 3. Gripper events (assuming last dim is gripper)
    gripper = actions[:, -1]
    for i in range(1, n_frames):
        if abs(gripper[i] - gripper[i-1]) > 0.5:
            action_type = "close" if gripper[i] > gripper[i-1] else "open"
            events.append({
                "frame": i,
                "type": "gripper",
                "severity": "info",
                "message": f"Gripper {action_type} at frame {i}",
            })

    # 4. Compute running divergence from mean trajectory
    mean_action = np.mean(actions, axis=0)
    divergence = np.array([np.linalg.norm(a - mean_action) for a in actions])

    return {
        "events": events,
        "action_magnitude": action_mag,
        "smoothness": smoothness,
        "jerk": jerk,
        "divergence": divergence,
        "delta_magnitude": np.concatenate([[0], delta_mag]),
    }


def make_quality_blueprint():
    """Blueprint focused on quality visualization."""
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Camera", origin="camera/rgb"),
                rrb.TextLogView(name="Quality Events", origin="events"),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="Action Magnitude", origin="metrics/action_magnitude"),
                rrb.TimeSeriesView(name="Action Smoothness", origin="metrics/smoothness"),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="Divergence from Mean", origin="metrics/divergence"),
                rrb.TimeSeriesView(name="Action Delta", origin="metrics/delta_magnitude"),
            ),
            row_shares=[4, 3, 3],
        ),
        rrb.TimePanel(state="expanded"),
    )


def stream_quality_episode(episode_idx: int = 0, max_frames: int = 0):
    """Stream episode with quality analysis overlay."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import json

    # Get info
    info_path = hf_hub_download(
        REPO_ID, "meta/info.json",
        repo_type="dataset", token=HF_TOKEN,
    )
    with open(info_path) as f:
        info = json.load(f)

    fps = info.get("fps", 10)

    # Stream episode
    ds = load_dataset(REPO_ID, split="train", streaming=True, token=HF_TOKEN)

    episode_data = []
    for row in ds:
        if row["episode_index"] == episode_idx:
            episode_data.append(row)
        elif len(episode_data) > 0:
            break
        if max_frames > 0 and len(episode_data) >= max_frames:
            break

    if not episode_data:
        sys.exit(f"Episode {episode_idx} not found")

    total_frames = len(episode_data)

    # Collect all actions for quality analysis
    actions_list = []
    for row in episode_data:
        a = row.get("action")
        if a is not None:
            actions_list.append(np.array(a, dtype=np.float32))

    actions_array = np.array(actions_list) if actions_list else np.zeros((total_frames, 7))

    # Compute quality metrics
    metrics = compute_quality_metrics(actions_array, fps)

    # Initialize Rerun
    blueprint = make_quality_blueprint()
    rr.init(
        f"data_viewer/{REPO_ID}/quality",
        recording_id=f"episode_{episode_idx}_quality",
        default_blueprint=blueprint,
        spawn=False,
    )

    output_path = f"exploration/rerun/exp3_episode_{episode_idx}_quality.rrd"
    rr.save(output_path)

    t_start = time.time()

    # Log frames and metrics together
    for i, row in enumerate(episode_data):
        frame_idx = row["frame_index"]
        timestamp = row.get("timestamp", frame_idx / fps)

        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("time", timestamp=float(timestamp))

        # Camera
        img = row.get("observation.images.image")
        if img is not None:
            if isinstance(img, dict) and "bytes" in img:
                pil_img = Image.open(io.BytesIO(img["bytes"]))
                img_array = np.array(pil_img)
            elif isinstance(img, Image.Image):
                img_array = np.array(img)
            else:
                img_array = np.array(img)
            rr.log("camera/rgb", rr.Image(img_array))

        # Continuous metrics
        if i < len(metrics["action_magnitude"]):
            rr.log("metrics/action_magnitude", rr.Scalars([float(metrics["action_magnitude"][i])]))
        if i < len(metrics["smoothness"]):
            rr.log("metrics/smoothness", rr.Scalars([float(metrics["smoothness"][i])]))
        if i < len(metrics["divergence"]):
            rr.log("metrics/divergence", rr.Scalars([float(metrics["divergence"][i])]))
        if i < len(metrics["delta_magnitude"]):
            rr.log("metrics/delta_magnitude", rr.Scalars([float(metrics["delta_magnitude"][i])]))
        if i < len(metrics["jerk"]):
            rr.log("metrics/jerk", rr.Scalars([float(metrics["jerk"][i])]))

    # Log discrete events
    severity_map = {
        "info": rr.TextLogLevel.INFO,
        "warn": rr.TextLogLevel.WARN,
        "error": rr.TextLogLevel.ERROR,
    }

    for event in metrics["events"]:
        rr.set_time("frame", sequence=event["frame"])
        rr.log("events", rr.TextLog(
            event["message"],
            level=severity_map.get(event["severity"], rr.TextLogLevel.INFO),
        ))

    elapsed = time.time() - t_start

    sys.stdout.write(
        f"\nExp 3 Complete:\n"
        f"  Episode: {episode_idx}\n"
        f"  Frames: {total_frames}\n"
        f"  Quality events: {len(metrics['events'])}\n"
        f"  Event types: {set(e['type'] for e in metrics['events'])}\n"
        f"  Time: {elapsed:.1f}s\n"
        f"  Output: {output_path}\n"
    )

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 3: Quality Events on Timeline")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0=all)")
    args = parser.parse_args()

    stream_quality_episode(episode_idx=args.episode, max_frames=args.max_frames)
