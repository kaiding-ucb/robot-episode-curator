"""
Exp 4: Blueprint + Web Viewer

Goal: Can programmatic Blueprints work with the web viewer?

Creates an RRD with an embedded Blueprint and tests if
@rerun-io/web-viewer-react v0.28.2 picks it up.

Also tests: rrd file size at various frame counts,
performance characteristics.
"""

import argparse
import io
import os
import sys
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

HF_TOKEN = "REDACTED-HF-TOKEN"
REPO_ID = "lerobot/libero_spatial_image"


def make_full_blueprint():
    """
    Blueprint that mimics the current app layout:
    - Top: Camera views
    - Middle: Action plots (Position, Rotation, Gripper)
    - Bottom: State + Events
    """
    return rrb.Blueprint(
        rrb.Vertical(
            # Camera row
            rrb.Horizontal(
                rrb.Spatial2DView(name="Main Camera", origin="camera/rgb"),
                rrb.Spatial2DView(name="Wrist Camera", origin="camera/wrist"),
            ),
            # Action plots
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Position (x,y,z)",
                    origin="action/position",
                ),
                rrb.TimeSeriesView(
                    name="Rotation (roll,pitch,yaw)",
                    origin="action/rotation",
                ),
                rrb.TimeSeriesView(
                    name="Gripper",
                    origin="action/gripper",
                ),
            ),
            # Metadata + Events
            rrb.Horizontal(
                rrb.TextDocumentView(name="Episode Info", origin="metadata"),
                rrb.TextLogView(name="Events", origin="events"),
            ),
            row_shares=[5, 3, 2],
        ),
        rrb.TimePanel(state="expanded"),
    )


def generate_blueprint_rrd(max_frames: int = 50):
    """Generate an RRD with embedded Blueprint for web viewer testing."""
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

    # Stream first episode
    ds = load_dataset(REPO_ID, split="train", streaming=True, token=HF_TOKEN)

    episode_data = []
    for row in ds:
        if row["episode_index"] == 0:
            episode_data.append(row)
        elif len(episode_data) > 0:
            break
        if max_frames > 0 and len(episode_data) >= max_frames:
            break

    if not episode_data:
        sys.exit("No data found")

    total_frames = len(episode_data)

    # Initialize with Blueprint
    blueprint = make_full_blueprint()
    rr.init(
        f"data_viewer/{REPO_ID}/blueprint_test",
        recording_id="blueprint_test",
        default_blueprint=blueprint,
        spawn=False,
    )

    output_path = "exploration/rerun/exp4_blueprint.rrd"
    rr.save(output_path)

    # Log metadata
    rr.log("metadata", rr.TextDocument(
        f"# Blueprint Test\n"
        f"**Frames:** {total_frames}\n"
        f"**FPS:** {fps}\n"
        f"**Dataset:** {REPO_ID}\n\n"
        f"This RRD tests whether the embedded Blueprint\n"
        f"is picked up by the web viewer component.\n\n"
        f"**Expected layout:**\n"
        f"- Top: Two camera views side by side\n"
        f"- Middle: Three action plot panels\n"
        f"- Bottom: Episode info + Events log\n",
        media_type=rr.MediaType.MARKDOWN,
    ), static=True)

    t_start = time.time()

    for row in episode_data:
        frame_idx = row["frame_index"]
        timestamp = row.get("timestamp", frame_idx / fps)

        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("time", timestamp=float(timestamp))

        # Main camera
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

        # Wrist camera
        wrist_img = row.get("observation.images.wrist_image")
        if wrist_img is not None:
            if isinstance(wrist_img, dict) and "bytes" in wrist_img:
                pil_img = Image.open(io.BytesIO(wrist_img["bytes"]))
                wrist_array = np.array(pil_img)
            elif isinstance(wrist_img, Image.Image):
                wrist_array = np.array(wrist_img)
            else:
                wrist_array = np.array(wrist_img)
            rr.log("camera/wrist", rr.Image(wrist_array))

        # Actions (7D: x,y,z,roll,pitch,yaw,gripper)
        action = row.get("action")
        if action is not None:
            action = np.array(action, dtype=np.float32)
            names_pos = ["x", "y", "z"]
            names_rot = ["roll", "pitch", "yaw"]

            for i, name in enumerate(names_pos):
                if i < len(action):
                    rr.log(f"action/position/{name}", rr.Scalars([float(action[i])]))

            for i, name in enumerate(names_rot):
                idx = i + 3
                if idx < len(action):
                    rr.log(f"action/rotation/{name}", rr.Scalars([float(action[idx])]))

            if len(action) >= 7:
                rr.log("action/gripper", rr.Scalars([float(action[6])]))

    # Log some events
    rr.set_time("frame", sequence=0)
    rr.log("events", rr.TextLog("Recording started", level=rr.TextLogLevel.INFO))
    rr.set_time("frame", sequence=total_frames // 2)
    rr.log("events", rr.TextLog("Midpoint reached", level=rr.TextLogLevel.INFO))
    rr.set_time("frame", sequence=total_frames - 1)
    rr.log("events", rr.TextLog("Recording complete", level=rr.TextLogLevel.INFO))

    elapsed = time.time() - t_start

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    sys.stdout.write(
        f"\nExp 4 Complete:\n"
        f"  Frames: {total_frames}\n"
        f"  Time: {elapsed:.1f}s\n"
        f"  File size: {file_size_mb:.2f} MB\n"
        f"  Output: {output_path}\n"
        f"  Frames/sec: {total_frames/elapsed:.0f}\n"
        f"  MB/frame: {file_size_mb/total_frames:.3f}\n\n"
        f"  Blueprint embedded: YES\n"
        f"  Web viewer test: Load this .rrd in the RerunViewer component\n"
        f"  or open with: rerun {output_path}\n"
    )

    return output_path


def benchmark_sizes():
    """Test RRD sizes at various frame counts for performance analysis."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import json

    info_path = hf_hub_download(
        REPO_ID, "meta/info.json",
        repo_type="dataset", token=HF_TOKEN,
    )
    with open(info_path) as f:
        info = json.load(f)

    fps = info.get("fps", 10)

    # Stream enough data for largest test
    ds = load_dataset(REPO_ID, split="train", streaming=True, token=HF_TOKEN)
    all_rows = []
    for row in ds:
        if row["episode_index"] == 0:
            all_rows.append(row)
        elif len(all_rows) > 0:
            break

    frame_counts = [10, 50, 100, 200, len(all_rows)]
    results = []

    for count in frame_counts:
        if count > len(all_rows):
            count = len(all_rows)

        rr.init(
            f"benchmark_{count}",
            recording_id=f"bench_{count}",
            spawn=False,
        )

        out_path = f"exploration/rerun/exp4_bench_{count}frames.rrd"
        rr.save(out_path)

        t0 = time.time()
        for i, row in enumerate(all_rows[:count]):
            frame_idx = row["frame_index"]
            rr.set_time("frame", sequence=frame_idx)

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

            action = row.get("action")
            if action is not None:
                action = np.array(action, dtype=np.float32)
                for j in range(min(len(action), 7)):
                    rr.log(f"action/dim_{j}", rr.Scalars([float(action[j])]))

        elapsed = time.time() - t0
        size_mb = os.path.getsize(out_path) / (1024 * 1024)

        results.append({
            "frames": count,
            "size_mb": size_mb,
            "time_s": elapsed,
            "fps": count / elapsed if elapsed > 0 else 0,
            "mb_per_frame": size_mb / count if count > 0 else 0,
        })

        # Clean up benchmark files
        os.unlink(out_path)

    sys.stdout.write("\nBenchmark Results:\n")
    sys.stdout.write(f"{'Frames':>8} {'Size(MB)':>10} {'Time(s)':>8} {'FPS':>8} {'MB/frame':>10}\n")
    sys.stdout.write("-" * 50 + "\n")
    for r in results:
        sys.stdout.write(
            f"{r['frames']:>8} {r['size_mb']:>10.2f} {r['time_s']:>8.1f} "
            f"{r['fps']:>8.0f} {r['mb_per_frame']:>10.3f}\n"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 4: Blueprint + Web Viewer")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames for blueprint test")
    parser.add_argument("--benchmark", action="store_true", help="Run size benchmark")
    args = parser.parse_args()

    generate_blueprint_rrd(max_frames=args.max_frames)

    if args.benchmark:
        benchmark_sizes()
