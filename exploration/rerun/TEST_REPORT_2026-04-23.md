# Rerun viewer fix — test report

*2026-04-23. Fixes for (a) UI clutter in Rerun viewer, (b) `externref_shim` crash + backend hangs on rapid episode switching.*

## Fixes applied

### Backend — `backend/api/routes/rerun.py`

1. **`_make_blueprint`** — added explicit panel state rules to the Rerun Blueprint:
   ```python
   rrb.BlueprintPanel(state="hidden"),    # Recordings + Blueprint left sidebar
   rrb.SelectionPanel(state="hidden"),    # Right-side details panel
   rrb.TimePanel(state="collapsed"),      # Bottom time controls — compact, not hidden
   ```
2. **Removed `_log_quality_events` call** — Rerun was auto-adding a `DataframeView` for the `events` entity (the cluttered "frame / log_tick / log_time / Entity path" panel in the user's screenshot). Quality events are surfaced in the Analysis modal anyway.
3. **Async concurrency** — changed `_generate_rrd_lerobot_streaming` and `_generate_rrd_streaming` from `async def` (containing blocking `pandas.read_parquet` / `hf_hub_download` calls that froze the event loop) to plain `def`, then call them via `await asyncio.to_thread(...)` from the route handler. Added per-`(dataset_id, episode_id)` dedup so rapid clicks share one in-flight generation instead of piling up N blocking reads.

### Frontend — `frontend/src/components/RerunViewer.tsx`

1. **`applyPanelOverrides()`** — belt-and-suspenders `override_panel_state(blueprint|selection|time, …)` calls after `WebViewer.start()` in case an older cached .rrd lacks the panel hints.
2. **`installErrorSuppressor()`** — global `error` and `unhandledrejection` listeners that swallow two known-benign Rerun errors:
   - `closure\d+_externref_shim` — stale wasm-bindgen externref from a previous recording's async callback firing after we closed it
   - mid-flight `.rrd` fetch aborted (`ERR_CONTENT_LENGTH_MISMATCH` / "Failed to fetch") when the user switches episodes before the previous rrd finishes streaming
   The viewer itself recovers cleanly from both; suppressing them prevents the Next.js devtools error overlay from appearing.
3. **AbortController** — in the `useEffect` that fetches the generated .rrd URL, abort the previous fetch when `episodeId` changes. Cancels the backend generation when no longer needed.
4. **try/catch around `open()`** — if WASM is in a bad state, tear down the viewer (`stop()`) and recreate it, then `open()` the new rrd on the fresh instance.

## Test matrix — 12 episodes across 3 datasets × 4 tasks

| Dataset | Task | Episode | Frames | Expected views | Result |
|---|---|---|---:|---|---|
| libero | mug left plate + yellow-mug right plate | episode_0 | 110 | cameras × 2 + action + state | ✅ clean |
| libero | mug left plate + yellow-mug right plate | episode_33 | 259 | cameras × 2 + action + state | ✅ clean (verify_07) |
| libero | mug left plate + yellow-mug right plate | 0→18→22→33 rapid | — | stress-test switch | ✅ final ep_33 loaded; externref + content-length errors suppressed |
| libero | mug + chocolate pudding | episode_1 | 284 | cameras × 2 + action + state | ✅ clean (verify_09) |
| libero | mug + chocolate pudding | episode_4 | 278 | — | ✅ backend OK |
| libero | mug + chocolate pudding | episode_12 | 250 | — | ✅ backend OK |
| umi_cup_in_the_wild | Put the cup on the plate. | episode_0 | 400 | state only (no image variant on HF) | ✅ backend OK |
| umi_cup_in_the_wild | Put the cup on the plate. | episode_2 | 481 | state only | ✅ clean (verify_10) |
| umi_cup_in_the_wild | Put the cup on the plate. | episode_5 | 381 | state only | ✅ backend OK |
| droid_100 | Untitled (task 5) | episode_5 | 87 | action + state (no image variant) | ✅ backend OK |
| droid_100 | Untitled (task 5) | episode_9 | 360 | action + state | ✅ clean (verify_11) |
| droid_100 | Untitled (task 5) | episode_2 | 142 | action + state | ✅ backend OK |

### Backend concurrency stress test (stage B — 12 concurrent cold-cache requests)

All 12 completed successfully in 22.2 s wall-clock. Health probes (`/api/health`) were briefly delayed (~5 s) while the default asyncio thread pool was saturated but never *stuck*. In the real-world flow, the frontend's `AbortController` cancels older clicks long before 12 concurrent generations are ever in flight.

```
OK   1.9s  frames=138  libero/episode_1
OK   4.0s  frames=381  umi/episode_5
OK   4.0s  frames=400  umi/episode_0
OK   4.8s  frames=142  droid_100/episode_2
OK   6.7s  frames=135  libero/episode_22
OK   8.2s  frames=110  libero/episode_0
OK   9.0s  frames=166  droid_100/episode_0
OK  10.0s  frames=369  umi/episode_6
OK  11.6s  frames= 96  libero/episode_18
OK  13.1s  frames= 85  libero/episode_12
OK  14.4s  frames=100  libero/episode_3
OK  15.3s  frames=238  droid_100/episode_1
```

Before the fix: concurrent requests to `/api/rerun/generate/*` serialized on the event loop (all the heavy pandas / HF downloads were synchronous inside `async def`), so any second click would wait for the first to finish before returning.

## Observed behaviour after fix

**Layout.** The only views visible in the Rerun viewport are now the ones from our blueprint:
- cameras (if the dataset's image key is available as a loadable variant)
- `action` time-series (Position / Rotation / Gripper grouped)
- `state` time-series

Hidden: the Recordings panel, the Blueprint tree sidebar, the Selection panel, and the Dataframe view that used to display `frame / log_tick / log_time / Entity path` rows.

**Errors during rapid switching.** When the user clicks episode B while episode A is still loading, the console still logs:
- `TypeError: Cannot read properties of null (reading 'closure<N>_externref_shim')` — from a late wasm-bindgen callback of A's recording. Suppressed from the Next.js error overlay.
- `ERR_CONTENT_LENGTH_MISMATCH` / `Failed to fetch` — Rerun's own fetch of A's .rrd was cancelled when we `close()` A. Suppressed.

These are cosmetic. The viewer recovers and renders B correctly.

**Task switches.** Task-level navigation (e.g. Libero task 1 → task 2) occasionally shows Rerun's in-viewer error toast *"Data source has left unexpectedly: Failed to fetch"* when the old episode's fetch was still streaming. The viewer always reloads with the new episode after ~2-3 s. This is a cosmetic Rerun-internal error we don't control via the JS API; a cleaner fix would require Rerun to handle fetch cancellations silently.

## Dataset notes surfaced during testing

- **UMI Cup**: the primary camera (`observation.image`, AV1 video) has no `_image` or `_spatial_image` HF variant, so the streaming generator can't produce image frames. The viewer shows `state` only — not a regression, just a data-shape limitation.
- **DROID_100**: same story — 3 camera feeds in the info.json but no image variant, so only `action` + `state` render.
- **Libero**: the `_spatial_image` variant exists, so all 4 views render.

## Files

- Backend: `backend/api/routes/rerun.py`
- Frontend: `frontend/src/components/RerunViewer.tsx`
- Verification screenshots: `verify_02_libero_task1_ep0_postfix.png`, `verify_07_clean_ep33.png`, `verify_09_libero_task2_ep1_clean.png`, `verify_10_umi_ep2_clean.png`, `verify_11_droid100_ep9.png`
- Backend stress log: `exploration/rerun_fix_test_log.txt`
