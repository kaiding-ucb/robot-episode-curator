"""
Exp 1: Rich Single-Episode Viewer

Goal: Can Rerun replace EpisodeViewer + ActionsChart + EnhancedTimeline?

Streams a LeRobot episode from HuggingFace (lerobot/libero_spatial_image),
logs ALL frames with semantic action names and a programmatic Blueprint.
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


def classify_action_dimensions(labels: list[str] | None, dims: int):
    """
    Port of frontend/src/utils/actionClassification.ts.
    Returns dict with group1, group2, grippers.
    """
    if not labels:
        if dims == 7:
            return {
                "group1": {"label": "Position", "names": ["x", "y", "z"], "indices": [0, 1, 2]},
                "group2": {"label": "Rotation", "names": ["rx", "ry", "rz"], "indices": [3, 4, 5]},
                "grippers": [{"label": "Gripper", "index": 6}],
            }
        mid = dims // 2
        return {
            "group1": {"label": f"Dims(0-{mid-1})", "names": [f"d{i}" for i in range(mid)], "indices": list(range(mid))},
            "group2": {"label": f"Dims({mid}-{dims-1})", "names": [f"d{i}" for i in range(mid, dims)], "indices": list(range(mid, dims))},
            "grippers": [],
        }

    lower = [l.lower() for l in labels]

    # Find grippers
    gripper_indices = [i for i, l in enumerate(lower) if "gripper" in l or "grip" in l]
    non_gripper = [i for i in range(dims) if i not in gripper_indices]

    # Cartesian detection
    pos_map = {"x": None, "y": None, "z": None}
    rot_map = {"roll": None, "pitch": None, "yaw": None, "rx": None, "ry": None, "rz": None}

    for i, l in enumerate(lower):
        if i in gripper_indices:
            continue
        # Position
        if l in ("x",) or "pos_x" in l:
            pos_map["x"] = i
        elif l in ("y",) or "pos_y" in l:
            pos_map["y"] = i
        elif l in ("z",) or "pos_z" in l:
            pos_map["z"] = i
        # Rotation
        elif l in ("roll",) or "rot_x" in l or l == "rx":
            rot_map["roll"] = i
        elif l in ("pitch",) or "rot_y" in l or l == "ry":
            rot_map["pitch"] = i
        elif l in ("yaw",) or "rot_z" in l or l == "rz":
            rot_map["yaw"] = i

    pos_indices = [v for v in [pos_map["x"], pos_map["y"], pos_map["z"]] if v is not None]
    rot_indices = [v for v in [rot_map.get("roll"), rot_map.get("pitch"), rot_map.get("yaw")] if v is not None]

    if len(pos_indices) == 3 and len(rot_indices) == 3:
        pos_names = ["x", "y", "z"]
        rot_names = ["roll", "pitch", "yaw"]
        grippers = [{"label": "Gripper", "index": idx} for idx in gripper_indices]
        return {
            "group1": {"label": "Position", "names": pos_names, "indices": pos_indices},
            "group2": {"label": "Rotation", "names": rot_names, "indices": rot_indices},
            "grippers": grippers,
        }

    # Fallback: split non-gripper in half
    mid = len(non_gripper) // 2
    return {
        "group1": {"label": "Arm", "names": [labels[i] for i in non_gripper[:mid]], "indices": non_gripper[:mid]},
        "group2": {"label": "Wrist", "names": [labels[i] for i in non_gripper[mid:]], "indices": non_gripper[mid:]},
        "grippers": [{"label": "Gripper", "index": idx} for idx in gripper_indices],
    }


def make_blueprint(has_wrist_camera: bool = True):
    """Create a programmatic Blueprint: cameras on top, action plots below."""
    camera_views = [rrb.Spatial2DView(name="RGB Camera", origin="camera/rgb")]
    if has_wrist_camera:
        camera_views.append(rrb.Spatial2DView(name="Wrist Camera", origin="camera/wrist"))

    action_views = [
        rrb.TimeSeriesView(
            name="Position (x, y, z)",
            origin="action/position",
        ),
        rrb.TimeSeriesView(
            name="Rotation (roll, pitch, yaw)",
            origin="action/rotation",
        ),
        rrb.TimeSeriesView(
            name="Gripper",
            origin="action/gripper",
        ),
    ]

    state_views = [
        rrb.TimeSeriesView(
            name="Robot State",
            origin="state",
        ),
    ]

    event_views = [
        rrb.TextLogView(
            name="Quality Events",
            origin="events",
        ),
    ]

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*camera_views),
            rrb.Horizontal(*action_views),
            rrb.Horizontal(*state_views, *event_views),
            row_shares=[4, 3, 2],
        ),
        rrb.TimePanel(state="expanded"),
    )


def stream_episode(episode_idx: int = 0, max_frames: int = 0):
    """Stream a single episode from HuggingFace and log to Rerun."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import json

    # Get dataset info for action labels
    info_path = hf_hub_download(
        REPO_ID, "meta/info.json",
        repo_type="dataset", token=HF_TOKEN,
    )
    with open(info_path) as f:
        info = json.load(f)

    fps = info.get("fps", 10)
    action_feature = info.get("features", {}).get("action", {})
    action_names_raw = action_feature.get("names", {})
    action_labels = None
    if isinstance(action_names_raw, dict) and "motors" in action_names_raw:
        action_labels = action_names_raw["motors"]
    elif isinstance(action_names_raw, list):
        action_labels = action_names_raw

    action_dims = action_feature.get("shape", [7])[0]
    classification = classify_action_dimensions(action_labels, action_dims)

    state_feature = info.get("features", {}).get("observation.state", {})
    state_names_raw = state_feature.get("names", {})
    state_labels = None
    if isinstance(state_names_raw, dict) and "motors" in state_names_raw:
        state_labels = state_names_raw["motors"]

    has_wrist = "observation.images.wrist_image" in info.get("features", {})

    # Stream parquet data
    ds = load_dataset(REPO_ID, split="train", streaming=True, token=HF_TOKEN)

    # Filter to target episode
    episode_data = []
    for row in ds:
        if row["episode_index"] == episode_idx:
            episode_data.append(row)
        elif len(episode_data) > 0:
            # Past our episode, stop
            break
        if max_frames > 0 and len(episode_data) >= max_frames:
            break

    if not episode_data:
        sys.exit(f"Episode {episode_idx} not found in dataset")

    total_frames = len(episode_data)

    # Get task info
    tasks_path = hf_hub_download(
        REPO_ID, "meta/tasks.parquet",
        repo_type="dataset", token=HF_TOKEN,
    )
    import pandas as pd
    tasks_df = pd.read_parquet(tasks_path)
    task_idx = episode_data[0].get("task_index", 0)
    task_name = tasks_df.index[task_idx] if task_idx < len(tasks_df) else f"task_{task_idx}"

    # Initialize Rerun
    blueprint = make_blueprint(has_wrist_camera=has_wrist)
    rr.init(
        f"data_viewer/{REPO_ID}",
        recording_id=f"episode_{episode_idx}",
        default_blueprint=blueprint,
        spawn=False,
    )

    output_path = f"exploration/rerun/exp1_episode_{episode_idx}.rrd"
    rr.save(output_path)

    # Log metadata as annotation context
    rr.log("metadata", rr.TextDocument(
        f"# Episode {episode_idx}\n"
        f"**Task:** {task_name}\n"
        f"**Frames:** {total_frames}\n"
        f"**FPS:** {fps}\n"
        f"**Dataset:** {REPO_ID}\n"
        f"**Actions:** {action_labels}\n",
        media_type=rr.MediaType.MARKDOWN,
    ), static=True)

    t_start = time.time()

    for row in episode_data:
        frame_idx = row["frame_index"]
        timestamp = row.get("timestamp", frame_idx / fps)

        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("time", timestamp=float(timestamp))

        # Log main camera image
        img = row.get("observation.images.image")
        if img is not None:
            if isinstance(img, dict) and "bytes" in img:
                pil_img = Image.open(io.BytesIO(img["bytes"]))
                img_array = np.array(pil_img)
            elif isinstance(img, Image.Image):
                img_array = np.array(img)
            elif isinstance(img, np.ndarray):
                img_array = img
            else:
                img_array = np.array(img)
            rr.log("camera/rgb", rr.Image(img_array))

        # Log wrist camera if available
        wrist_img = row.get("observation.images.wrist_image")
        if wrist_img is not None:
            if isinstance(wrist_img, dict) and "bytes" in wrist_img:
                pil_img = Image.open(io.BytesIO(wrist_img["bytes"]))
                wrist_array = np.array(pil_img)
            elif isinstance(wrist_img, Image.Image):
                wrist_array = np.array(wrist_img)
            elif isinstance(wrist_img, np.ndarray):
                wrist_array = wrist_img
            else:
                wrist_array = np.array(wrist_img)
            rr.log("camera/wrist", rr.Image(wrist_array))

        # Log actions with semantic names
        action = row.get("action")
        if action is not None:
            action = np.array(action, dtype=np.float32)
            # Position group
            g1 = classification["group1"]
            for name, idx in zip(g1["names"], g1["indices"]):
                rr.log(f"action/position/{name}", rr.Scalars([float(action[idx])]))

            # Rotation group
            g2 = classification["group2"]
            for name, idx in zip(g2["names"], g2["indices"]):
                rr.log(f"action/rotation/{name}", rr.Scalars([float(action[idx])]))

            # Grippers
            for grip in classification["grippers"]:
                rr.log("action/gripper", rr.Scalars([float(action[grip["index"]])]))

        # Log robot state
        state = row.get("observation.state")
        if state is not None:
            state = np.array(state, dtype=np.float32)
            if state_labels:
                for i, label in enumerate(state_labels):
                    if i < len(state):
                        rr.log(f"state/{label}", rr.Scalars([float(state[i])]))
            else:
                for i, val in enumerate(state):
                    rr.log(f"state/dim_{i}", rr.Scalars([float(val)]))

    elapsed = time.time() - t_start

    # Log quality events as simple examples
    # (In production, these would come from backend/quality/ modules)
    rr.set_time("frame", sequence=0)
    rr.log("events", rr.TextLog("Episode start", level=rr.TextLogLevel.INFO))
    rr.set_time("frame", sequence=total_frames - 1)
    rr.log("events", rr.TextLog("Episode end", level=rr.TextLogLevel.INFO))

    # Detect potential stall (consecutive similar actions)
    if total_frames > 10:
        actions_array = []
        for row in episode_data:
            a = row.get("action")
            if a is not None:
                actions_array.append(np.array(a, dtype=np.float32))

        if len(actions_array) > 10:
            actions_np = np.array(actions_array)
            diffs = np.linalg.norm(np.diff(actions_np, axis=0), axis=1)
            # Find stall regions (very low action change)
            stall_threshold = np.percentile(diffs, 10)
            stall_frames = np.where(diffs < stall_threshold * 0.5)[0]
            for sf in stall_frames[:5]:  # Log up to 5 stall events
                rr.set_time("frame", sequence=int(sf))
                rr.log("events", rr.TextLog(
                    f"Low action change at frame {sf} (delta={diffs[sf]:.4f})",
                    level=rr.TextLogLevel.WARN,
                ))

    sys.stdout.write(
        f"\nExp 1 Complete:\n"
        f"  Episode: {episode_idx} ({task_name})\n"
        f"  Frames: {total_frames}\n"
        f"  Time: {elapsed:.1f}s ({total_frames/elapsed:.0f} frames/sec)\n"
        f"  Output: {output_path}\n"
        f"  Action labels: {action_labels}\n"
        f"  Classification: {classification['group1']['label']} / {classification['group2']['label']}\n"
    )

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 1: Rich Single-Episode Viewer")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0=all)")
    args = parser.parse_args()

    stream_episode(episode_idx=args.episode, max_frames=args.max_frames)
