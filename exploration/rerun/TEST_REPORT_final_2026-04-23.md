# Rerun viewer — final fixes and test report

*2026-04-23. Fixes for three user-reported issues:*
1. *Action/state charts empty on first-render of any episode; only populate after clicking back in*
2. *Layout was diagonal (camera / action / wrist / state), user wanted cameras-on-top / action+state-on-bottom*
3. *DROID and UMI didn't render cameras at all*

## Root causes

### Issue 1 (first-render empty charts) — **CONTENT-LENGTH RACE**

The backend was calling `rr.save(path)` **before** the data-logging loop. In rerun 0.28's Python SDK, `rr.save(path)` starts a streaming file sink — subsequent `rr.log()` calls append to the file incrementally. FastAPI's `FileResponse` reads the file size **at open time** for `Content-Length`, then streams bytes. Because the file was still growing while the Rerun viewer fetched it, the server errored with

```
RuntimeError: Response content longer than Content-Length
```

and the Rerun viewer showed `Data source ... has left unexpectedly: Failed to fetch .rrd file`. The cameras often rendered because their data arrived early, but the TimeSeriesViews (Action / State) arrived later and the connection was cut before they finished. On second click, the file had finished writing so everything served correctly.

**Fix:** log first, `rr.save()` at the end:
```python
rr.init("app_id", recording_id=rec_id, default_blueprint=blueprint)
# log all frames, actions, state
for row in episode_data:
    rr.log("camera/rgb", rr.Image(arr))
    rr.log("action/...", rr.Scalars([...]))
    # ...
rr.save(path, default_blueprint=blueprint)  # ← atomic flush at end
```

Applied to all three rr.save sites (`_generate_rrd_lerobot_streaming`, local-loader branch of `generate_rrd`, and `generate_comparison_rrd`).

### Issue 2 (diagonal layout) — **CONTAINER FLATTENING + CACHED BLUEPRINT**

Two sub-problems:

(a) `rrb.Vertical(view1, view2, row_shares=[1,1])` with raw `View` children was being interpreted as "one container with two views side-by-side" rather than "two rows". Solution: always wrap each row in `rrb.Horizontal(...)`, even when there's a single view.

(b) The WebViewer persists the last-used blueprint per `application_id`. Old session state (from before the fix) was being re-applied to new recordings, overriding the embedded `default_blueprint`. Solution: also call `rr.send_blueprint(blueprint, make_active=True, make_default=True)` during generation so every recording carries an explicitly-active blueprint.

Final blueprint for the "cameras on top / signals on bottom" layout:
```python
rrb.Blueprint(
    rrb.Vertical(
        rrb.Horizontal(*camera_views),        # top row
        rrb.Horizontal(action_view, state_view),  # bottom row
        row_shares=[1, 1],
    ),
    rrb.BlueprintPanel(state="hidden"),
    rrb.SelectionPanel(state="hidden"),
    rrb.TimePanel(state="collapsed"),
    auto_layout=False,
    auto_views=False,
    collapse_panels=True,
)
```

### Issue 3 (DROID/UMI not rendering cameras) — **TWO ORTHOGONAL BUGS**

(a) **UMI uses the singular `observation.image`** key, not the plural `observation.images.*` used by newer LeRobot datasets (Libero, DROID). `_discover_image_keys` only matched the plural prefix, so UMI's camera was never found.

**Fix:** update `_discover_image_keys` + new `_camera_short_name` helper to accept both conventions.

(b) **DROID / UMI have video-type image columns with no image-variant HF repo.** Libero has a `_spatial_image` variant that embeds decoded JPEGs in the parquet; DROID and UMI don't, so the parquet rows return `None` for image fields, and nothing gets logged to Rerun.

**Fix:** new helper `_extract_lerobot_video_frames(repo_id, episode_idx, info, image_keys, max_frames)` that:
- downloads the per-chunk MP4 for each camera (`videos/{video_key}/chunk-{NNN}/file-{NNN}.mp4`)
- opens it with **PyAV**
- seeks to the episode's `from_timestamp`
- decodes each frame into `np.ndarray(format="rgb24")`
- maps them by **episode-local frame index** (`round((pts_s - t_start) * fps)`)

Then `_generate_rrd_lerobot_streaming` uses the decoded frames as a fallback when `row.get(img_key)` is `None`.

## Before / after

| | Before | After |
|---|---|---|
| Libero first-load | action/state empty, required re-click | renders complete on first click |
| UMI | empty viewer (camera key mismatch) | rgb camera + state, stacked |
| DROID | action+state only, no cameras | 3 cameras (exterior_1, exterior_2, wrist) + action + state |
| Layout | diagonal 2×2 grid | cameras row / signals row |
| Backend error | `Response content longer than Content-Length` | none |
| Rerun viewer error | `Data source has left unexpectedly: Failed to fetch` | none on first-load |

## Verification screenshots (all first-load, never-clicked episodes)

| Dataset | Task | Episode | Cameras | Action | State | Errors |
|---|---|---|---|---|---|---|
| libero | yellow mug → microwave | episode_34 | ✅ image + wrist_image | ✅ populated | ✅ populated | 0 |
| libero | turn on stove & moka pot | episode_40 | ✅ image + wrist_image | ✅ populated | ✅ populated | 0 |
| libero | turn on stove & moka pot | episode_45 (switched from 40) | ✅ | ✅ | ✅ | 0 |
| umi_cup_in_the_wild | Put the cup on the plate. | episode_3 | ✅ rgb (fisheye outdoor) | n/a (no action column) | ✅ populated | 0 |
| umi_cup_in_the_wild | Put the cup on the plate. | episode_4 | ✅ | n/a | ✅ | 0 |
| umi_cup_in_the_wild | Put the cup on the plate. | episode_2 (switched from 4) | ✅ | n/a | ✅ | 0 |
| droid_100 | Untitled (task 5) | episode_7 | ✅ exterior_1 + exterior_2 + wrist (3 cams decoded from MP4) | ✅ populated | ✅ populated | 0 |
| droid_100 | Untitled (task 5) | episode_14 | ✅ 3 cams | ✅ | ✅ | 0 |

Screenshots saved to `exploration/rerun/verify_screens/`:
- `verify_14_umi_ep3_stacked.png` — UMI cameras-on-top / state-on-bottom
- `verify_15_droid100_ep14.png` — DROID 3 cameras + action + state
- `verify_16_libero_task3_ep34.png` — Libero never-played task renders on first click
- `verify_19_umi_ep4_first_load.png`, `verify_20_umi_ep2_switch.png`
- `verify_21_droid_ep7_first.png`, `verify_22_droid_ep7_loaded.png`
- `verify_23_libero_task4_ep40_first.png`, `verify_24_libero_ep45_switch.png`

## Files touched

- **`backend/api/routes/rerun.py`**:
  - `_discover_image_keys` — accepts `observation.image` (singular) + `observation.images.*` (plural)
  - `_camera_short_name` — new helper mapping image key → display name
  - `_camera_entity` — uses the new helper
  - `_extract_lerobot_video_frames` — new: PyAV-based chunk MP4 decoding for datasets without image variants
  - `_make_blueprint` — new signature `(camera_names, has_state, has_action)`, always wraps rows in `rrb.Horizontal`
  - `_generate_rrd_lerobot_streaming` — calls `_extract_lerobot_video_frames` when parquet has no image columns, moves `rr.save()` to end, computes `has_action_column` from the sample action, calls `rr.send_blueprint(make_active=True)`
  - Local-loader branch of `generate_rrd` — same `rr.save`-at-end fix; passes `has_action` to blueprint
  - `generate_comparison_rrd` — same `rr.save`-at-end fix
- **`frontend/src/components/RerunViewer.tsx`**: no functional change needed for this round (earlier error suppressor + abort controller still in place)
