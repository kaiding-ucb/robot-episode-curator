"""
Rerun API routes for generating .rrd recordings from episode data.

Converts episode data to Rerun's native format for
advanced multi-modal visualization with semantic action names,
programmatic Blueprints, and quality event overlays.
"""

import asyncio
import io
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)

# Dedup in-flight generation per (dataset_id, episode_id). Rapid episode
# clicks previously started N concurrent blocking parquet reads; now they
# share the same task and the event loop stays responsive.
_rrd_inflight: dict[str, asyncio.Task] = {}
_rrd_inflight_lock = asyncio.Lock()

router = APIRouter(prefix="/rerun", tags=["rerun"])

# Cache directory for generated RRD files
RRD_CACHE_DIR = Path(os.environ.get("RRD_CACHE_DIR", Path.home() / ".cache" / "data_viewer" / "rerun"))
RRD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Static file serving base URL
STATIC_BASE_URL = "/api/rerun/files"

def _read_hf_token_file() -> "str | None":
    for p in (Path.home() / ".huggingface" / "token", Path.home() / ".cache" / "huggingface" / "token"):
        if p.exists():
            try:
                t = p.read_text().strip()
                if t:
                    return t
            except OSError:
                continue
    return None


HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or _read_hf_token_file()
    or ""
)


def _get_rrd_cache_path(dataset_id: str, episode_id: str) -> Path:
    """Get the cache path for an RRD file. Ensures the cache dir exists at
    call time — the user can wipe ~/.cache/data_viewer at any moment via the
    sidebar Clear cache button, so the import-time mkdir alone isn't enough.
    """
    import hashlib
    RRD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{dataset_id}|{episode_id}"
    hash_key = hashlib.sha256(key.encode()).hexdigest()[:16]
    safe_name = episode_id.replace("/", "_").replace("\\", "_")[:50]
    return RRD_CACHE_DIR / f"{safe_name}_{hash_key}.rrd"


def _classify_action_dimensions(labels: list | None, dims: int) -> dict:
    """
    Classify action dimensions into semantic groups.
    Port of frontend/src/utils/actionClassification.ts.
    """
    # Some datasets (e.g. lerobot/libero) list a single aggregate label like
    # ['actions'] for a 7-dim vector. Treat that as "no labels" rather than
    # trying to index past the end.
    if not labels or len(labels) < dims:
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
    gripper_indices = [i for i, l in enumerate(lower) if "gripper" in l or "grip" in l]
    non_gripper = [i for i in range(dims) if i not in gripper_indices]

    # Cartesian detection
    pos_map = {"x": None, "y": None, "z": None}
    rot_names_found = {"roll": None, "pitch": None, "yaw": None}

    for i, l in enumerate(lower):
        if i in gripper_indices:
            continue
        if l in ("x",) or "pos_x" in l:
            pos_map["x"] = i
        elif l in ("y",) or "pos_y" in l:
            pos_map["y"] = i
        elif l in ("z",) or "pos_z" in l:
            pos_map["z"] = i
        elif l in ("roll",) or "rot_x" in l or l == "rx":
            rot_names_found["roll"] = i
        elif l in ("pitch",) or "rot_y" in l or l == "ry":
            rot_names_found["pitch"] = i
        elif l in ("yaw",) or "rot_z" in l or l == "rz":
            rot_names_found["yaw"] = i

    pos_indices = [v for v in pos_map.values() if v is not None]
    rot_indices = [v for v in rot_names_found.values() if v is not None]

    if len(pos_indices) == 3 and len(rot_indices) == 3:
        grippers = [{"label": "Gripper", "index": idx} for idx in gripper_indices]
        return {
            "group1": {"label": "Position", "names": ["x", "y", "z"], "indices": pos_indices},
            "group2": {"label": "Rotation", "names": ["roll", "pitch", "yaw"], "indices": rot_indices},
            "grippers": grippers,
        }

    # Fallback: split non-gripper in half
    mid = len(non_gripper) // 2
    return {
        "group1": {"label": "Arm", "names": [labels[i] for i in non_gripper[:mid]], "indices": non_gripper[:mid]},
        "group2": {"label": "Wrist", "names": [labels[i] for i in non_gripper[mid:]], "indices": non_gripper[mid:]},
        "grippers": [{"label": "Gripper", "index": idx} for idx in gripper_indices],
    }


def _discover_image_keys(features: dict) -> list[str]:
    """Return sorted list of non-depth camera keys from info.json features.

    Handles both LeRobot conventions:
      - `observation.images.{name}` (plural, newer — Libero, DROID)
      - `observation.image`           (singular — UMI Cup in the Wild)
    """
    keys = []
    for name, spec in (features or {}).items():
        if not isinstance(spec, dict):
            continue
        is_camera_key = name.startswith("observation.images.") or name == "observation.image"
        if not is_camera_key:
            continue
        if spec.get("dtype") not in ("video", "image"):
            continue
        sub = spec.get("info", {}) or {}
        is_depth = sub.get("video.is_depth_map", False) or "depth" in name.lower()
        if is_depth:
            continue
        keys.append(name)
    return sorted(keys)


def _camera_short_name(img_key: str) -> str:
    """Short, display-friendly name for a camera image key."""
    if img_key.startswith("observation.images."):
        return img_key.removeprefix("observation.images.")
    if img_key == "observation.image":
        return "rgb"
    return img_key


def _camera_entity(img_key: str) -> str:
    """Map an info.json image key to a Rerun entity path."""
    return f"camera/{_camera_short_name(img_key)}"


# Lazy cache: {repo_id: {episode_idx: file_path_within_repo}}. Populated as
# episodes are discovered; persists within one backend process. Avoids repeated
# outward-spiral search when the meta parquet's data/file_index is unreliable
# (e.g., Libero — meta says ep_235→file_015, actual is file_091).
_EP_TO_FILE_CACHE: dict[str, dict[int, str]] = {}


def _remember_episodes_in_file(repo_id: str, file_rel: str, episodes_in_file: list[int]) -> None:
    cache = _EP_TO_FILE_CACHE.setdefault(repo_id, {})
    for eid in episodes_in_file:
        cache[int(eid)] = file_rel


def _load_lerobot_episode_direct(
    repo_id: str,
    episode_idx: int,
    max_frames: int = 0,
    columns: list[str] | None = None,
) -> list[dict]:
    """
    Load a LeRobot episode by downloading the right parquet file directly.
    Much faster than streaming through the entire dataset.

    Args:
        columns: If set, only read these columns from parquet (plus episode_index
                 and frame_index which are always included). Dramatically faster
                 when image columns can be skipped.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download, HfApi

    api = HfApi(token=HF_TOKEN)

    # List parquet files to find the one containing our episode
    # LeRobot v3 format: data/chunk-{chunk}/file-{file}.parquet
    # Each file contains ~5-10 episodes
    try:
        all_files = [
            f for f in api.list_repo_files(repo_id, repo_type="dataset", token=HF_TOKEN)
            if f.startswith("data/") and f.endswith(".parquet")
        ]
        all_files.sort()
    except Exception as e:
        logger.warning(f"Could not list files for {repo_id}: {e}")
        return []

    if not all_files:
        return []

    # Build column list for selective reads
    read_cols = None
    if columns is not None:
        read_cols = list(set(["episode_index", "frame_index"] + columns))

    def _read_episode_from_file(file_path: str, file_rel: str) -> list[dict] | None:
        """Read a specific episode from a parquet file, using column filter.
        Side-effect: record all episodes found in this file into the cache
        so future lookups can skip the search."""
        try:
            df = pd.read_parquet(file_path, columns=read_cols)
        except Exception:
            df = pd.read_parquet(file_path)
        try:
            unique_eps = df["episode_index"].unique().tolist()
            _remember_episodes_in_file(repo_id, file_rel, unique_eps)
        except Exception:
            pass
        if episode_idx not in df["episode_index"].values:
            return None
        ep_df = df[df["episode_index"] == episode_idx].sort_values("frame_index")
        if max_frames > 0:
            ep_df = ep_df.head(max_frames)
        return ep_df.to_dict("records")

    # Lazy cache hit: we've previously read a file that contained this episode.
    # Trust it — HF hub files are immutable within a branch.
    cached_file = _EP_TO_FILE_CACHE.get(repo_id, {}).get(episode_idx)
    if cached_file and cached_file in all_files:
        try:
            path = hf_hub_download(repo_id, cached_file, repo_type="dataset", token=HF_TOKEN)
            result = _read_episode_from_file(path, cached_file)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"Cached file {cached_file} for ep {episode_idx} failed: {e}")

    # Binary search: load first file to learn episodes_per_file, then jump
    num_files = len(all_files)

    try:
        first_path = hf_hub_download(repo_id, all_files[0], repo_type="dataset", token=HF_TOKEN)
        first_df = pd.read_parquet(first_path, columns=["episode_index"])
        _remember_episodes_in_file(repo_id, all_files[0], first_df["episode_index"].unique().tolist())
        eps_per_file = len(first_df["episode_index"].unique())
        first_ep = first_df["episode_index"].min()

        # Check if target is in first file
        if episode_idx in first_df["episode_index"].values:
            result = _read_episode_from_file(first_path, all_files[0])
            if result is not None:
                return result

        estimated_file = min(max(0, (episode_idx - first_ep) // max(eps_per_file, 1)), num_files - 1)
    except Exception:
        estimated_file = min(episode_idx // 7, num_files - 1)

    # Outward spiral. No cap — if the estimate is wildly off (as happens when
    # eps-per-file is non-uniform, e.g. Libero), we still need to find the
    # episode. Each downloaded file caches every episode it contains, so this
    # pays off across repeated calls.
    for delta in range(0, num_files):
        for idx in [estimated_file + delta, estimated_file - delta]:
            if idx < 0 or idx >= num_files:
                continue
            # Skip if our lazy cache says this file doesn't contain the target
            # (we recorded its episodes on a previous pass)
            known_eps = _EP_TO_FILE_CACHE.get(repo_id, {})
            file_rel = all_files[idx]
            if file_rel in known_eps.values() and episode_idx not in {k for k, v in known_eps.items() if v == file_rel}:
                continue
            try:
                path = hf_hub_download(repo_id, file_rel, repo_type="dataset", token=HF_TOKEN)
                result = _read_episode_from_file(path, file_rel)
                if result is not None:
                    logger.info(f"Found episode {episode_idx} in {file_rel} ({len(result)} frames, {delta} files from estimate)")
                    return result
            except Exception as e:
                logger.warning(f"Error reading {file_rel}: {e}")
                continue

    logger.warning(f"Episode {episode_idx} not found in any of {num_files} files of {repo_id}")
    return []


def _make_blueprint(camera_names: list[str] | None = None, has_state: bool = False, has_action: bool = False):
    """Episode blueprint: cameras row on top, action + state timeseries at the bottom.

    Shape is always two rows:
        Row 1  — all cameras, side-by-side
        Row 2  — [Action time-series] + [State time-series]   (each may be absent)

    Explicitly hides the Recordings/Blueprint side panel and the Selection panel
    so the viewer shows only cameras + action/state charts + time controls.
    """
    import rerun.blueprint as rrb

    names = camera_names or ["rgb"]
    camera_views = [rrb.Spatial2DView(name=n, origin=f"camera/{n}") for n in names]

    bottom_views: list = []
    if has_action:
        bottom_views.append(rrb.TimeSeriesView(name="Action", origin="action"))
    if has_state:
        bottom_views.append(rrb.TimeSeriesView(name="State", origin="state"))

    # rrb.Vertical requires Container children (Horizontal/Grid/Tabs); raw
    # Views get interpreted as side-by-side peers rather than stacked rows.
    # Always wrap each row in rrb.Horizontal so layout is stable for N=1.
    if bottom_views:
        viewport = rrb.Vertical(
            rrb.Horizontal(*camera_views),
            rrb.Horizontal(*bottom_views),
            row_shares=[1, 1],
        )
    else:
        # No action/state available — just show cameras in a row
        viewport = rrb.Horizontal(*camera_views)

    return rrb.Blueprint(
        viewport,
        rrb.BlueprintPanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
        rrb.TimePanel(state="collapsed"),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _get_loader_for_episode(dataset_id: str, episode_id: str, data_root: Path):
    """Get appropriate loader and episode for the given IDs."""
    from downloaders.manager import get_all_datasets
    from loaders import HDF5Loader, LeRobotLoader

    all_datasets = get_all_datasets()

    if dataset_id not in all_datasets:
        return None, None, False, None

    config = all_datasets[dataset_id]
    data_dir = data_root / dataset_id

    # Check if episode_id looks like a LeRobot streaming episode (episode_N format)
    is_lerobot_episode = episode_id.startswith("episode_") and episode_id.split("_")[-1].isdigit()

    # For streaming datasets with repo_id — prefer streaming for LeRobot-style episodes
    if is_lerobot_episode and config.get("repo_id"):
        return None, episode_id, True, config

    # For LIBERO HDF5 format (local only)
    if dataset_id in ["libero", "libero_pro"] and data_dir.exists():
        loader = HDF5Loader(data_dir)
        return loader, episode_id, False, config

    # For LeRobot format (local)
    if config.get("format") == "lerobot" and data_dir.exists():
        loader = LeRobotLoader(data_dir)
        return loader, episode_id, False, config

    # For streaming datasets (HuggingFace)
    if config.get("streaming_recommended", False) or config.get("repo_id"):
        return None, episode_id, True, config

    return None, None, False, config


def _generate_rrd_lerobot_streaming(
    dataset_id: str,
    episode_id: str,
    cache_path: Path,
    config: dict,
    max_frames: int,
    include_actions: bool,
):
    """Generate enriched RRD from streaming LeRobot HuggingFace dataset."""
    import json

    import numpy as np
    import rerun as rr
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from PIL import Image

    repo_id = config.get("repo_id", dataset_id)

    # Try to get dataset info for action labels and FPS
    fps = 10
    action_labels = None
    state_labels = None
    has_state = False
    image_keys: list[str] = []
    task_name = None

    try:
        info_path = hf_hub_download(
            repo_id, "meta/info.json",
            repo_type="dataset", token=HF_TOKEN,
        )
        with open(info_path) as f:
            info = json.load(f)

        fps = info.get("fps", 10)
        features = info.get("features", {})

        # Extract action labels
        action_feature = features.get("action", {})
        action_names_raw = action_feature.get("names", {})
        if isinstance(action_names_raw, dict) and "motors" in action_names_raw:
            action_labels = action_names_raw["motors"]
        elif isinstance(action_names_raw, list):
            action_labels = action_names_raw

        # Extract state labels
        state_feature = features.get("observation.state", {})
        state_names_raw = state_feature.get("names", {})
        if isinstance(state_names_raw, dict) and "motors" in state_names_raw:
            state_labels = state_names_raw["motors"]
        has_state = "observation.state" in features

        image_keys = _discover_image_keys(features)

        # Get task name
        try:
            import pandas as pd
            tasks_path = hf_hub_download(
                repo_id, "meta/tasks.parquet",
                repo_type="dataset", token=HF_TOKEN,
            )
            tasks_df = pd.read_parquet(tasks_path)
        except Exception:
            tasks_df = None

    except Exception as e:
        logger.warning(f"Could not fetch LeRobot metadata for {repo_id}: {e}")

    # Parse episode index from episode_id (e.g., "episode_0" -> 0)
    episode_idx = 0
    ep_id_str = episode_id.split("/")[-1] if "/" in episode_id else episode_id
    if ep_id_str.startswith("episode_"):
        try:
            episode_idx = int(ep_id_str.replace("episode_", ""))
        except ValueError:
            pass

    # Check if images are video-type (not embedded in parquet)
    # If so, try the _image variant of the dataset which has embedded images
    images_are_video = False
    for feat_name, feat_info in info.get("features", {}).items():
        if "images" in feat_name and feat_info.get("dtype") == "video":
            images_are_video = True
            break

    streaming_repo = repo_id
    if images_are_video:
        # Try _image variant (e.g., lerobot/libero -> lerobot/libero_spatial_image)
        # Common LeRobot pattern: dataset_image has embedded images
        image_variants = [
            f"{repo_id}_image",
            f"{repo_id}_spatial_image",
        ]
        for variant in image_variants:
            try:
                variant_info_path = hf_hub_download(
                    variant, "meta/info.json",
                    repo_type="dataset", token=HF_TOKEN,
                )
                with open(variant_info_path) as f:
                    variant_info = json.load(f)
                # Only accept a variant that is a true MIRROR of the main repo
                # (same episode count AND same total frame count). Some *_image
                # repos (e.g. lerobot/libero_spatial_image) are subsets — accepting
                # them here silently serves the wrong episode with a fraction of
                # the expected frames.
                same_episodes = (
                    variant_info.get("total_episodes") == info.get("total_episodes")
                    and variant_info.get("total_frames") == info.get("total_frames")
                )
                if not same_episodes:
                    logger.info(
                        f"Skipping variant {variant} (not a mirror of {repo_id}: "
                        f"{variant_info.get('total_episodes')}ep/{variant_info.get('total_frames')}f "
                        f"vs {info.get('total_episodes')}ep/{info.get('total_frames')}f)"
                    )
                    continue
                # Check this variant has actual image data
                for vf_name, vf_info in variant_info.get("features", {}).items():
                    if "images" in vf_name and vf_info.get("dtype") == "image":
                        streaming_repo = variant
                        info = variant_info
                        # Re-extract metadata from variant
                        fps = info.get("fps", 10)
                        features = info.get("features", {})
                        action_feature = features.get("action", {})
                        action_names_raw = action_feature.get("names", {})
                        if isinstance(action_names_raw, dict) and "motors" in action_names_raw:
                            action_labels = action_names_raw["motors"]
                        elif isinstance(action_names_raw, list):
                            action_labels = action_names_raw
                        state_feature = features.get("observation.state", {})
                        state_names_raw = state_feature.get("names", {})
                        if isinstance(state_names_raw, dict) and "motors" in state_names_raw:
                            state_labels = state_names_raw["motors"]
                        image_keys = _discover_image_keys(features)
                        has_state = "observation.state" in features
                        logger.info(f"Using image variant: {streaming_repo} instead of {repo_id}")
                        break
                if streaming_repo != repo_id:
                    break
            except Exception:
                continue

    # Load episode data — try direct parquet file download first (much faster)
    logger.info(f"Loading LeRobot episode {episode_idx} from {streaming_repo}")
    episode_data = _load_lerobot_episode_direct(streaming_repo, episode_idx, max_frames)

    if not episode_data:
        raise HTTPException(status_code=404, detail=f"Episode {episode_idx} not found in {streaming_repo}")

    total_frames = len(episode_data)

    # If the image variant doesn't have action/state data, load from the original repo
    action_lookup: dict[int, dict] | None = None
    sample_action = episode_data[0].get("action")
    if sample_action is None and streaming_repo != repo_id:
        logger.info(f"Image variant missing actions — loading from original repo: {repo_id}")
        action_data_source = _load_lerobot_episode_direct(repo_id, episode_idx, max_frames)
        if action_data_source:
            action_lookup = {row["frame_index"]: row for row in action_data_source}
            # Use first row from original to detect action dims
            sample_action = action_data_source[0].get("action")

    # Get task name from first row
    if tasks_df is not None:
        task_idx = episode_data[0].get("task_index", 0)
        if task_idx is None and action_lookup:
            # task_index might also be missing from image variant
            first_orig = next(iter(action_lookup.values()), None)
            if first_orig:
                task_idx = first_orig.get("task_index", 0)
        if task_idx is not None and task_idx < len(tasks_df):
            task_name = tasks_df.index[task_idx]

    # Classify action dimensions
    action_dims = 7  # default
    if sample_action is not None:
        action_dims = len(sample_action) if hasattr(sample_action, '__len__') else 7
    classification = _classify_action_dimensions(action_labels, action_dims)

    # For video-type image columns that aren't in the parquet (DROID / UMI
    # style), decode frames directly from the HF chunk MP4 files via PyAV.
    video_decoded_frames: dict[str, dict[int, "np.ndarray"]] = {}
    first_row_has_image = any(episode_data[0].get(k) is not None for k in image_keys) if episode_data and image_keys else False
    if image_keys and not first_row_has_image:
        try:
            video_decoded_frames = _extract_lerobot_video_frames(
                repo_id, episode_idx, info, image_keys, max_frames=max_frames,
            )
        except Exception as e:
            logger.warning(f"Video frame extraction failed for {repo_id}/{episode_idx}: {e}")

    # Create Blueprint and initialize Rerun
    # A camera view only makes sense if we actually have frames for it. Drop
    # keys that produced no frames so the blueprint row stays tight.
    effective_image_keys = [
        k for k in image_keys
        if (first_row_has_image or (video_decoded_frames and video_decoded_frames.get(k)))
    ]
    camera_names = [_camera_short_name(k) for k in effective_image_keys] or ["rgb"]
    has_action_column = sample_action is not None
    blueprint = _make_blueprint(
        camera_names=camera_names,
        has_state=has_state,
        has_action=has_action_column,
    )
    rr.init(
        f"data_viewer/{repo_id}",
        recording_id=f"{dataset_id}/{episode_id}",
        default_blueprint=blueprint,
        spawn=False,
    )
    # NOTE: don't `rr.save()` here. Calling save() BEFORE logging turns the
    # recording into a streaming-to-file sink; the file keeps growing while
    # FastAPI tries to serve it with a stale Content-Length header, causing
    # "Response content longer than Content-Length" on the server and
    # "Failed to fetch .rrd file" in the Rerun viewer. Instead we log
    # everything to memory and rr.save() once at the end of this function.

    # (Episode metadata is surfaced in the app header — no Rerun panel needed.)

    # Log episode data
    for row in episode_data:
        frame_idx = row.get("frame_index", 0)
        timestamp = row.get("timestamp", frame_idx / fps)

        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("time", timestamp=float(timestamp))

        # All observation.images.* cameras (dynamic — matches info.json features).
        # Sources in order of preference:
        #   1. An "images" column already in the parquet (libero_*_image variant)
        #   2. A frame decoded from the HF chunk MP4 (droid_100, umi, raw libero)
        local_idx = frame_idx  # episodes are 0-indexed by frame_index
        for img_key in image_keys:
            img = row.get(img_key)
            arr = _decode_image(img) if img is not None else None
            if arr is None:
                arr = (video_decoded_frames.get(img_key) or {}).get(local_idx)
            if arr is None:
                continue
            rr.log(_camera_entity(img_key), rr.Image(arr))

        # Actions with semantic names
        if include_actions:
            action = row.get("action")
            # Fallback to original repo's action data if image variant lacks it
            if action is None and action_lookup:
                orig_row = action_lookup.get(frame_idx)
                if orig_row:
                    action = orig_row.get("action")
            if action is not None:
                action = np.array(action, dtype=np.float32)
                g1 = classification["group1"]
                for name, idx in zip(g1["names"], g1["indices"]):
                    if idx < len(action):
                        rr.log(f"action/position/{name}", rr.Scalars([float(action[idx])]))

                g2 = classification["group2"]
                for name, idx in zip(g2["names"], g2["indices"]):
                    if idx < len(action):
                        rr.log(f"action/rotation/{name}", rr.Scalars([float(action[idx])]))

                for grip in classification["grippers"]:
                    if grip["index"] < len(action):
                        rr.log("action/gripper", rr.Scalars([float(action[grip["index"]])]))

        # Robot state
        if has_state:
            state = row.get("observation.state")
            # Fallback to original repo's state data
            if state is None and action_lookup:
                orig_row = action_lookup.get(frame_idx)
                if orig_row:
                    state = orig_row.get("observation.state")
            if state is not None:
                state = np.array(state, dtype=np.float32)
                if state_labels:
                    for i, label in enumerate(state_labels):
                        if i < len(state):
                            rr.log(f"state/{label}", rr.Scalars([float(state[i])]))
                else:
                    for i, val in enumerate(state):
                        rr.log(f"state/dim_{i}", rr.Scalars([float(val)]))

    # Quality events intentionally not logged: Rerun auto-adds a DataframeView
    # for the `events` entity, which clutters the viewer. The Analysis modal
    # surfaces the same signals in a purpose-built UI.

    # Finalise the file: save everything accumulated via rr.init+log to disk
    # in one atomic shot. This is the step that guarantees Content-Length
    # matches file size by the time FastAPI serves it.
    rr.save(str(cache_path), default_blueprint=blueprint)

    file_size_kb = cache_path.stat().st_size / 1024
    logger.info(f"Generated enriched RRD: {cache_path} ({file_size_kb:.1f} KB, {total_frames} frames)")

    return {
        "status": "generated",
        "rrd_url": f"{STATIC_BASE_URL}/{cache_path.name}",
        "episode_id": episode_id,
        "dataset_id": dataset_id,
        "num_frames": total_frames,
        "file_size_kb": file_size_kb,
        "source": "streaming_lerobot",
    }


def _extract_lerobot_video_frames(
    repo_id: str,
    episode_idx: int,
    info: dict,
    image_keys: list[str],
    max_frames: int = 0,
) -> dict[str, dict[int, "np.ndarray"]]:
    """For LeRobot v3 datasets whose image columns are video-type (no image
    variant on HF), decode frames directly from the per-chunk MP4 files.

    Returns {img_key: {frame_index_within_episode: rgb np.ndarray}}.
    Frame 0 corresponds to the episode's first frame, not a global offset.
    """
    import numpy as np
    import pandas as pd
    import av
    from huggingface_hub import hf_hub_download

    video_path_template = info.get("video_path")
    fps = info.get("fps", 10)
    if not video_path_template:
        return {}

    # Locate the episodes metadata parquet(s). We don't know which chunk/file
    # the episode lives in, so scan meta/episodes/*/*.parquet until we find it.
    episode_meta_row = None
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        meta_files = [
            f for f in api.list_repo_files(repo_id, repo_type="dataset", token=HF_TOKEN)
            if f.startswith("meta/episodes/") and f.endswith(".parquet")
        ]
        meta_files.sort()
    except Exception as e:
        logger.warning(f"_extract_lerobot_video_frames: could not list meta files for {repo_id}: {e}")
        return {}

    for meta_rel in meta_files:
        try:
            meta_local = hf_hub_download(repo_id, meta_rel, repo_type="dataset", token=HF_TOKEN)
            df = pd.read_parquet(meta_local)
            if episode_idx in df["episode_index"].values:
                episode_meta_row = df[df["episode_index"] == episode_idx].iloc[0]
                break
        except Exception:
            continue
    if episode_meta_row is None:
        logger.warning(f"_extract_lerobot_video_frames: episode {episode_idx} not found in meta for {repo_id}")
        return {}

    frames_by_key: dict[str, dict[int, np.ndarray]] = {}

    for img_key in image_keys:
        chunk_col = f"videos/{img_key}/chunk_index"
        file_col = f"videos/{img_key}/file_index"
        t0_col = f"videos/{img_key}/from_timestamp"
        t1_col = f"videos/{img_key}/to_timestamp"

        if any(c not in episode_meta_row.index for c in (chunk_col, file_col, t0_col, t1_col)):
            continue

        chunk_idx = int(episode_meta_row[chunk_col])
        file_idx = int(episode_meta_row[file_col])
        t_start = float(episode_meta_row[t0_col])
        t_end = float(episode_meta_row[t1_col])

        rel = video_path_template.format(video_key=img_key, chunk_index=chunk_idx, file_index=file_idx)
        try:
            mp4_local = hf_hub_download(repo_id, rel, repo_type="dataset", token=HF_TOKEN)
        except Exception as e:
            logger.warning(f"Could not download {rel}: {e}")
            continue

        try:
            container = av.open(mp4_local)
            stream = container.streams.video[0]
            # seek to t_start (in AV_TIME_BASE microseconds when no stream is given,
            # or stream time_base when given). Use the stream.time_base for precision.
            if stream.time_base is not None:
                seek_pts = int(t_start / float(stream.time_base))
                container.seek(seek_pts, any_frame=False, backward=True, stream=stream)

            target = {}
            expected_frames = max(1, int(round((t_end - t_start) * fps)))
            if max_frames > 0:
                expected_frames = min(expected_frames, max_frames)

            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                pts_s = float(frame.pts * stream.time_base)
                if pts_s < t_start - 0.05:
                    continue
                if pts_s >= t_end - 1e-6:
                    break
                # Snap to the nearest episode-local frame index
                local_idx = int(round((pts_s - t_start) * fps))
                if local_idx < 0 or local_idx >= expected_frames:
                    continue
                if local_idx in target:
                    continue  # keep first decoded frame at this slot
                target[local_idx] = frame.to_ndarray(format="rgb24")
            container.close()
            frames_by_key[img_key] = target
            logger.info(
                f"_extract_lerobot_video_frames: decoded {len(target)}/{expected_frames} frames "
                f"for {img_key} (ep {episode_idx} from {rel})"
            )
        except Exception as e:
            logger.warning(f"PyAV decode failed for {rel}: {e}")

    return frames_by_key


def _decode_image(img) -> "np.ndarray | None":
    """Decode an image from various formats to numpy array."""
    import numpy as np
    from PIL import Image

    if img is None:
        return None
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, Image.Image):
        return np.array(img)
    if isinstance(img, dict) and "bytes" in img:
        pil_img = Image.open(io.BytesIO(img["bytes"]))
        return np.array(pil_img)
    try:
        return np.array(img)
    except Exception:
        return None


def _log_quality_events(
    episode_data: list,
    fps: int,
    action_lookup: dict[int, dict] | None = None,
):
    """Log basic quality events (stalls, discontinuities, gripper changes)."""
    import numpy as np
    import rerun as rr

    actions_list = []
    for row in episode_data:
        a = row.get("action")
        if a is None and action_lookup:
            orig = action_lookup.get(row.get("frame_index", -1))
            if orig:
                a = orig.get("action")
        if a is not None:
            actions_list.append(np.array(a, dtype=np.float32))

    if len(actions_list) < 3:
        return

    actions = np.array(actions_list)
    n = len(actions)

    # Action deltas
    deltas = np.diff(actions, axis=0)
    delta_mag = np.linalg.norm(deltas, axis=1)

    # Gripper events (assuming last dim)
    gripper = actions[:, -1]
    for i in range(1, n):
        if abs(gripper[i] - gripper[i - 1]) > 0.5:
            action_type = "close" if gripper[i] > gripper[i - 1] else "open"
            frame_idx = episode_data[i].get("frame_index", i)
            rr.set_time("frame", sequence=frame_idx)
            rr.log("events", rr.TextLog(
                f"Gripper {action_type} at frame {frame_idx}",
                level=rr.TextLogLevel.INFO,
            ))

    # Action discontinuities
    if len(delta_mag) > 0:
        p95 = np.percentile(delta_mag, 95)
        threshold = p95 * 2
        for i in range(len(delta_mag)):
            if delta_mag[i] > threshold:
                frame_idx = episode_data[i + 1].get("frame_index", i + 1)
                rr.set_time("frame", sequence=frame_idx)
                rr.log("events", rr.TextLog(
                    f"Action jump at frame {frame_idx} (delta={delta_mag[i]:.3f})",
                    level=rr.TextLogLevel.WARN,
                ))


def _generate_rrd_streaming(
    dataset_id: str,
    episode_id: str,
    cache_path: Path,
    max_frames: int,
    include_actions: bool
):
    """Generate RRD from streaming HuggingFace dataset (non-LeRobot fallback)."""
    import rerun as rr

    from loaders.streaming_extractor import StreamingFrameExtractor
    from loaders import get_repo_id_for_dataset

    repo_id = get_repo_id_for_dataset(dataset_id)
    if not repo_id:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset_id}")

    extractor = StreamingFrameExtractor(repo_id)

    logger.info(f"Extracting streaming frames for RRD: {episode_id}")
    frames = extractor.extract_frames(episode_id, start=0, end=max_frames if max_frames > 0 else 500)

    if not frames:
        raise HTTPException(status_code=404, detail="No frames found in episode")

    rec = rr.RecordingStream(
        application_id=f"data_viewer/{dataset_id}",
        recording_id=f"{dataset_id}/{episode_id}",
    )

    for frame_idx, timestamp, image_array in frames:
        rec.set_time("frame", sequence=frame_idx)
        rec.set_time("time", timestamp=timestamp)
        rec.log("camera/rgb", rr.Image(image_array))

    frame_count = len(frames)
    rec.save(str(cache_path))
    file_size_kb = cache_path.stat().st_size / 1024
    logger.info(f"Generated streaming RRD: {cache_path} ({file_size_kb:.1f} KB)")

    return {
        "status": "generated",
        "rrd_url": f"{STATIC_BASE_URL}/{cache_path.name}",
        "episode_id": episode_id,
        "dataset_id": dataset_id,
        "num_frames": frame_count,
        "file_size_kb": file_size_kb,
        "source": "streaming"
    }


def _is_lerobot_dataset(config: dict) -> bool:
    """Check if a dataset is in LeRobot format (has info.json with features)."""
    repo_id = config.get("repo_id", "")
    if not repo_id:
        return False
    # LeRobot datasets typically have lerobot/ prefix or known format
    if "lerobot/" in repo_id or config.get("format") == "lerobot":
        return True
    # Try to detect by checking for meta/info.json
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id, "meta/info.json", repo_type="dataset", token=HF_TOKEN)
        return True
    except Exception:
        return False


@router.post("/generate/{episode_id:path}")
async def generate_rrd(
    request: Request,
    episode_id: str,
    dataset_id: str = Query(..., description="Dataset ID"),
    force: bool = Query(False, description="Force regeneration even if cached"),
    include_actions: bool = Query(True, description="Include action data if available"),
    max_frames: int = Query(0, description="Maximum frames (0=all)"),
):
    """
    Generate a Rerun recording (.rrd) file from episode data.

    Returns the URL to access the generated RRD file.
    """
    try:
        import rerun as rr
        import numpy as np
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="rerun-sdk not installed. Install with: pip install rerun-sdk"
        )

    cache_path = _get_rrd_cache_path(dataset_id, episode_id)

    # Check cache first
    if cache_path.exists() and not force:
        logger.info(f"Using cached RRD: {cache_path}")
        return JSONResponse({
            "status": "cached",
            "rrd_url": f"{STATIC_BASE_URL}/{cache_path.name}",
            "episode_id": episode_id,
            "dataset_id": dataset_id
        })

    data_root = request.app.state.data_root
    loader, ep_id, is_streaming, config = _get_loader_for_episode(dataset_id, episode_id, data_root)

    if loader is None and not is_streaming:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found or not downloaded. Download it first via the Data Manager."
        )

    try:
        logger.info(f"Loading episode for RRD generation: {episode_id}")

        # Dedup in-flight generations per (dataset, episode). Rapid clicks
        # now share the same task instead of starting N blocking reads.
        dedup_key = f"{dataset_id}|{episode_id}"

        async def _run_streaming_lerobot():
            return await asyncio.to_thread(
                _generate_rrd_lerobot_streaming,
                dataset_id, episode_id, cache_path, config,
                max_frames, include_actions,
            )

        async def _run_streaming_generic():
            return await asyncio.to_thread(
                _generate_rrd_streaming,
                dataset_id, episode_id, cache_path,
                max_frames if max_frames > 0 else 500, include_actions,
            )

        runner = None
        if is_streaming and config and _is_lerobot_dataset(config):
            runner = _run_streaming_lerobot
        elif is_streaming:
            runner = _run_streaming_generic

        if runner is not None:
            async with _rrd_inflight_lock:
                existing = _rrd_inflight.get(dedup_key)
                if existing is not None and not existing.done():
                    task = existing
                else:
                    task = asyncio.create_task(runner())
                    _rrd_inflight[dedup_key] = task

                    def _cleanup(t: asyncio.Task, key: str = dedup_key) -> None:
                        _rrd_inflight.pop(key, None)

                    task.add_done_callback(_cleanup)
            result = await task
            return JSONResponse(result)

        # Local loader (HDF5, LeRobot parquet, etc.)
        episode = loader.load_episode(ep_id)

        if episode.observations is None or len(episode.observations) == 0:
            raise HTTPException(status_code=404, detail="No frames found in episode")

        num_frames = len(episode.observations)
        if max_frames > 0:
            num_frames = min(num_frames, max_frames)

        # Create blueprint (local HDF5/LeRobot loaders expose a single camera as `camera/rgb`)
        blueprint = _make_blueprint(
            camera_names=["rgb"],
            has_state=False,
            has_action=(episode.actions is not None and len(episode.actions) > 0),
        )
        rr.init(
            f"data_viewer/{dataset_id}",
            recording_id=f"{dataset_id}/{episode_id}",
            default_blueprint=blueprint,
            spawn=False,
        )
        # Do not `rr.save()` here — save at end so Content-Length matches.

        for i in range(num_frames):
            timestamp = episode.timestamps[i] if episode.timestamps is not None and i < len(episode.timestamps) else i / 20.0

            rr.set_time("frame", sequence=i)
            rr.set_time("time", timestamp=timestamp)

            image = episode.observations[i]
            rr.log("camera/rgb", rr.Image(image))

            if include_actions and episode.actions is not None and i < len(episode.actions):
                action = episode.actions[i]
                classification = _classify_action_dimensions(None, len(action))
                g1 = classification["group1"]
                for name, idx in zip(g1["names"], g1["indices"]):
                    if idx < len(action):
                        rr.log(f"action/position/{name}", rr.Scalars([float(action[idx])]))
                g2 = classification["group2"]
                for name, idx in zip(g2["names"], g2["indices"]):
                    if idx < len(action):
                        rr.log(f"action/rotation/{name}", rr.Scalars([float(action[idx])]))
                for grip in classification["grippers"]:
                    if grip["index"] < len(action):
                        rr.log("action/gripper", rr.Scalars([float(action[grip["index"]])]))

        rr.save(str(cache_path), default_blueprint=blueprint)

        frame_count = num_frames
        file_size_kb = cache_path.stat().st_size / 1024
        logger.info(f"Generated RRD file: {cache_path} ({file_size_kb:.1f} KB)")

        return JSONResponse({
            "status": "generated",
            "rrd_url": f"{STATIC_BASE_URL}/{cache_path.name}",
            "episode_id": episode_id,
            "dataset_id": dataset_id,
            "num_frames": frame_count,
            "file_size_kb": file_size_kb
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate RRD: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate RRD: {str(e)}")


@router.get("/files/{filename}")
async def serve_rrd_file(filename: str):
    """Serve a cached RRD file."""
    file_path = RRD_CACHE_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="RRD file not found")

    # Validate the file is within the cache directory (security)
    try:
        file_path.resolve().relative_to(RRD_CACHE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=3600"
        }
    )


@router.get("/status")
async def rerun_status():
    """Check Rerun integration status."""
    try:
        import rerun as rr
        version = rr.__version__
        return {
            "available": True,
            "version": version,
            "cache_dir": str(RRD_CACHE_DIR),
            "cached_files": len(list(RRD_CACHE_DIR.glob("*.rrd")))
        }
    except ImportError:
        return {
            "available": False,
            "error": "rerun-sdk not installed"
        }


def _get_comparison_cache_path(dataset_id: str, task_name: str, max_episodes: int) -> Path:
    """Get cache path for a comparison RRD file."""
    import hashlib
    key = f"comparison|{dataset_id}|{task_name}|{max_episodes}"
    hash_key = hashlib.sha256(key.encode()).hexdigest()[:16]
    safe_task = task_name.replace(" ", "_")[:40]
    return RRD_CACHE_DIR / f"comparison_{safe_task}_{hash_key}.rrd"


def _make_comparison_blueprint(episode_indices: list[int]):
    """Create a Blueprint for multi-episode signal comparison."""
    import rerun.blueprint as rrb

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


async def _resolve_task_episodes(
    repo_id: str, task_name: str, max_episodes: int
) -> tuple[list[int], str | None]:
    """Resolve task name to episode indices for a LeRobot dataset.

    Reuses the proven resolution chain from datasets.py (meta parquet →
    data parquet → datasets-server API).

    Returns (episode_indices, resolved_task_name).
    """
    import re

    from api.routes.datasets import (
        fetch_lerobot_episodes_meta,
        fetch_lerobot_tasks_meta,
        get_episode_task_map,
    )

    # Fetch metadata
    episodes_df = await fetch_lerobot_episodes_meta(repo_id)
    tasks_df = await fetch_lerobot_tasks_meta(repo_id)

    if tasks_df is None:
        logger.warning(f"Could not load tasks for {repo_id}")
        return [], None

    # Resolve task_name → task_index
    task_col = "task_description"
    if task_col not in tasks_df.columns:
        for col in tasks_df.columns:
            if col != "task_index":
                task_col = col
                break

    task_index = None
    if task_col in tasks_df.columns:
        match = tasks_df[tasks_df[task_col] == task_name]
        if len(match) > 0:
            task_index = int(match.iloc[0]["task_index"])

    # Handle "Untitled (task N)" fallback
    if task_index is None:
        untitled_match = re.match(r"^Untitled \(task (\d+)\)$", task_name)
        if untitled_match:
            candidate_idx = int(untitled_match.group(1))
            if candidate_idx in tasks_df["task_index"].values:
                task_index = candidate_idx

    if task_index is None:
        logger.warning(f"Could not resolve task '{task_name}' to index for {repo_id}")
        return [], None

    # Use the full resolution chain (meta → data parquet → API)
    ep_task_map = await get_episode_task_map(repo_id, episodes_df=episodes_df)
    if ep_task_map is None:
        logger.warning(f"Could not build episode-task map for {repo_id}")
        return [], None

    episode_indices = sorted([
        ep_idx for ep_idx, t_idx in ep_task_map.items()
        if t_idx == task_index
    ])

    return episode_indices[:max_episodes], task_name


@router.post("/generate-comparison")
async def generate_comparison_rrd(
    request: Request,
    dataset_id: str = Query(..., description="Dataset ID"),
    task_name: str = Query(..., description="Task name"),
    max_episodes: int = Query(5, ge=1, le=20, description="Max episodes"),
    max_frames_per_episode: int = Query(0, description="Max frames per episode (0=all)"),
    force: bool = Query(False, description="Force regeneration"),
):
    """Generate a multi-episode comparison .rrd for a task."""
    try:
        import rerun as rr
        import numpy as np
    except ImportError:
        raise HTTPException(status_code=500, detail="rerun-sdk not installed")

    from downloaders.manager import get_all_datasets

    all_datasets = get_all_datasets()
    config = all_datasets.get(dataset_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    repo_id = config.get("repo_id", dataset_id)

    cache_path = _get_comparison_cache_path(dataset_id, task_name, max_episodes)
    if cache_path.exists() and not force:
        return JSONResponse({
            "status": "cached",
            "rrd_url": f"{STATIC_BASE_URL}/{cache_path.name}",
            "dataset_id": dataset_id,
            "task_name": task_name,
        })

    # Resolve task → episodes
    episode_indices, resolved_task = await _resolve_task_episodes(repo_id, task_name, max_episodes)
    if not episode_indices:
        raise HTTPException(status_code=404, detail=f"No episodes found for task '{task_name}'")

    # Get metadata for action labels
    import json
    from huggingface_hub import hf_hub_download

    fps = 10
    action_labels = None
    try:
        info_path = hf_hub_download(
            repo_id, "meta/info.json",
            repo_type="dataset", token=HF_TOKEN,
        )
        with open(info_path) as f:
            info = json.load(f)
        fps = info.get("fps", 10)
        action_feature = info.get("features", {}).get("action", {})
        action_names_raw = action_feature.get("names", {})
        if isinstance(action_names_raw, dict) and "motors" in action_names_raw:
            action_labels = action_names_raw["motors"]
        elif isinstance(action_names_raw, list):
            action_labels = action_names_raw
    except Exception:
        pass

    # Load episodes in parallel (from original repo — no images needed for comparison)
    import concurrent.futures

    # Only need action signals for comparison — skip image columns for speed
    comparison_columns = ["action", "timestamp", "task_index", "observation.state"]

    def _load_ep(ep_idx: int) -> tuple[int, list[dict]]:
        return ep_idx, _load_lerobot_episode_direct(
            repo_id, ep_idx, max_frames_per_episode, columns=comparison_columns,
        )

    episodes_data: dict[int, list[dict]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(episode_indices), 4)) as pool:
        futures = {pool.submit(_load_ep, idx): idx for idx in episode_indices}
        for future in concurrent.futures.as_completed(futures):
            try:
                ep_idx, rows = future.result()
                if rows:
                    episodes_data[ep_idx] = rows
            except Exception as e:
                logger.warning(f"Failed to load episode {futures[future]}: {e}")

    if not episodes_data:
        raise HTTPException(status_code=404, detail="Could not load any episode data")

    actual_indices = sorted(episodes_data.keys())

    # Build blueprint and initialize Rerun
    blueprint = _make_comparison_blueprint(actual_indices)
    rr.init(
        f"data_viewer/{repo_id}/comparison",
        recording_id=f"{dataset_id}/comparison/{task_name}",
        default_blueprint=blueprint,
        spawn=False,
    )
    # Log first, save at end (same rationale as _generate_rrd_lerobot_streaming).

    # Log metadata
    rr.log("metadata", rr.TextDocument(
        f"# Multi-Episode Comparison\n"
        f"**Task:** {resolved_task}\n"
        f"**Episodes:** {len(actual_indices)} ({actual_indices})\n"
        f"**Dataset:** {repo_id}\n",
        media_type=rr.MediaType.MARKDOWN,
    ), static=True)

    # Determine action names
    sample_action = None
    for rows in episodes_data.values():
        for row in rows:
            if row.get("action") is not None:
                sample_action = row["action"]
                break
        if sample_action is not None:
            break

    action_dims = len(sample_action) if sample_action is not None and hasattr(sample_action, '__len__') else 7
    if action_labels and len(action_labels) == action_dims:
        names = action_labels
    elif action_dims == 7:
        names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    else:
        names = [f"d{i}" for i in range(action_dims)]

    # Log per-episode signals
    import numpy as np

    for ep_idx in actual_indices:
        rows = episodes_data[ep_idx]
        for row in rows:
            frame_idx = row.get("frame_index", 0)
            timestamp = row.get("timestamp", frame_idx / fps)

            rr.set_time("frame", sequence=frame_idx)
            rr.set_time("time", timestamp=float(timestamp))

            action = row.get("action")
            if action is None:
                continue

            action = np.array(action, dtype=np.float32)

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

            # Gripper (last component, if 7D)
            for grip_idx in range(6, len(action)):
                rr.log(f"episode_{ep_idx}/action/gripper", rr.Scalars([float(action[grip_idx])]))

    rr.save(str(cache_path), default_blueprint=blueprint)

    file_size_kb = cache_path.stat().st_size / 1024
    logger.info(
        f"Generated comparison RRD: {cache_path} "
        f"({file_size_kb:.1f} KB, {len(actual_indices)} episodes)"
    )

    return JSONResponse({
        "status": "generated",
        "rrd_url": f"{STATIC_BASE_URL}/{cache_path.name}",
        "dataset_id": dataset_id,
        "task_name": resolved_task,
        "num_episodes": len(actual_indices),
        "episode_indices": actual_indices,
        "file_size_kb": file_size_kb,
    })


@router.delete("/cache")
async def clear_cache():
    """Clear the RRD cache."""
    count = 0
    for rrd_file in RRD_CACHE_DIR.glob("*.rrd"):
        try:
            rrd_file.unlink()
            count += 1
        except Exception as e:
            logger.warning(f"Failed to delete {rrd_file}: {e}")

    return {"deleted": count}
