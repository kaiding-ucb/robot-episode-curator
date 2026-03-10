"""
Exp 2: Multi-Episode Signal Comparison

Goal: Can Rerun replace SignalComparisonChart?

Streams multiple episodes from the same task and overlays
action signals for comparison.
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


def make_comparison_blueprint(num_episodes: int, episode_indices: list[int]):
    """Blueprint for multi-episode signal comparison."""
    # One TimeSeriesView per signal group, all episodes overlaid
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.TimeSeriesView(
                name="Position Magnitude (all episodes)",
                origin="/",
                contents=[f"episode_{i}/action/position_mag" for i in episode_indices],
            ),
            rrb.TimeSeriesView(
                name="Rotation Magnitude (all episodes)",
                origin="/",
                contents=[f"episode_{i}/action/rotation_mag" for i in episode_indices],
            ),
            rrb.TimeSeriesView(
                name="Gripper (all episodes)",
                origin="/",
                contents=[f"episode_{i}/action/gripper" for i in episode_indices],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Position X",
                    origin="/",
                    contents=[f"episode_{i}/action/position/x" for i in episode_indices],
                ),
                rrb.TimeSeriesView(
                    name="Position Y",
                    origin="/",
                    contents=[f"episode_{i}/action/position/y" for i in episode_indices],
                ),
                rrb.TimeSeriesView(
                    name="Position Z",
                    origin="/",
                    contents=[f"episode_{i}/action/position/z" for i in episode_indices],
                ),
            ),
            row_shares=[2, 2, 2, 3],
        ),
        rrb.TimePanel(state="expanded"),
    )


def stream_multi_episodes(
    task_index: int = 0,
    num_episodes: int = 5,
    max_frames_per_episode: int = 0,
):
    """Stream multiple episodes from the same task."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import json
    import pandas as pd

    # Get info
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

    # Get tasks
    tasks_path = hf_hub_download(
        REPO_ID, "meta/tasks.parquet",
        repo_type="dataset", token=HF_TOKEN,
    )
    tasks_df = pd.read_parquet(tasks_path)
    task_name = tasks_df.index[task_index] if task_index < len(tasks_df) else f"task_{task_index}"

    # Stream dataset and collect episodes for this task
    ds = load_dataset(REPO_ID, split="train", streaming=True, token=HF_TOKEN)

    episodes_data: dict[int, list] = {}
    current_ep = -1

    for row in ds:
        if row.get("task_index") != task_index:
            if len(episodes_data) >= num_episodes:
                break
            continue

        ep_idx = row["episode_index"]
        if ep_idx not in episodes_data:
            if len(episodes_data) >= num_episodes:
                break
            episodes_data[ep_idx] = []
            current_ep = ep_idx

        if max_frames_per_episode > 0 and len(episodes_data[ep_idx]) >= max_frames_per_episode:
            continue

        episodes_data[ep_idx].append(row)

    if not episodes_data:
        sys.exit(f"No episodes found for task_index={task_index}")

    episode_indices = sorted(episodes_data.keys())

    # Initialize Rerun
    blueprint = make_comparison_blueprint(len(episode_indices), episode_indices)
    rr.init(
        f"data_viewer/{REPO_ID}/comparison",
        recording_id=f"task_{task_index}_comparison",
        default_blueprint=blueprint,
        spawn=False,
    )

    output_path = f"exploration/rerun/exp2_task_{task_index}_comparison.rrd"
    rr.save(output_path)

    # Log metadata
    rr.log("metadata", rr.TextDocument(
        f"# Multi-Episode Comparison\n"
        f"**Task:** {task_name}\n"
        f"**Episodes:** {len(episode_indices)} ({episode_indices})\n"
        f"**Dataset:** {REPO_ID}\n",
        media_type=rr.MediaType.MARKDOWN,
    ), static=True)

    t_start = time.time()

    for ep_idx in episode_indices:
        rows = episodes_data[ep_idx]

        for row in rows:
            frame_idx = row["frame_index"]
            timestamp = row.get("timestamp", frame_idx / fps)

            rr.set_time("frame", sequence=frame_idx)
            rr.set_time("time", timestamp=float(timestamp))

            action = row.get("action")
            if action is None:
                continue

            action = np.array(action, dtype=np.float32)

            # Semantic action logging (assuming 7D: x,y,z,roll,pitch,yaw,gripper)
            if action_labels and len(action_labels) == len(action):
                names = action_labels
            elif len(action) == 7:
                names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            else:
                names = [f"d{i}" for i in range(len(action))]

            # Position components (first 3)
            pos = action[:3]
            pos_mag = float(np.linalg.norm(pos))
            rr.log(f"episode_{ep_idx}/action/position_mag", rr.Scalars([pos_mag]))

            for i, name in enumerate(names[:3]):
                rr.log(f"episode_{ep_idx}/action/position/{name}", rr.Scalars([float(action[i])]))

            # Rotation components (3:6)
            if len(action) > 3:
                rot = action[3:6] if len(action) >= 6 else action[3:]
                rot_mag = float(np.linalg.norm(rot))
                rr.log(f"episode_{ep_idx}/action/rotation_mag", rr.Scalars([rot_mag]))

            # Gripper (last component)
            if len(action) >= 7:
                rr.log(f"episode_{ep_idx}/action/gripper", rr.Scalars([float(action[6])]))

    elapsed = time.time() - t_start

    frame_counts = {ep: len(rows) for ep, rows in episodes_data.items()}

    sys.stdout.write(
        f"\nExp 2 Complete:\n"
        f"  Task: {task_name} (index={task_index})\n"
        f"  Episodes: {len(episode_indices)}\n"
        f"  Frame counts: {frame_counts}\n"
        f"  Time: {elapsed:.1f}s\n"
        f"  Output: {output_path}\n"
    )

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 2: Multi-Episode Signal Comparison")
    parser.add_argument("--task", type=int, default=0, help="Task index")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames per episode (0=all)")
    args = parser.parse_args()

    stream_multi_episodes(
        task_index=args.task,
        num_episodes=args.num_episodes,
        max_frames_per_episode=args.max_frames,
    )
