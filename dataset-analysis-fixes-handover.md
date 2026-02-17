# Dataset Analysis Fixes — Handover Document

## Reported Errors

### 1. HTTP 500 on Dataset Switching
**Severity:** High
**Reproduction:** Open Dataset Analysis modal → select RealOmni → switch to Egocentric-10K → HTTP 500 error appears. Same when switching to MicroAGI.

**Root cause:** Race condition in `useTasks` hook. When user changes dataset:
1. `setChosenDatasetId(newId)` + `setSelectedTask(null)` fires
2. `useTasks(newDatasetId)` starts async fetch, but OLD tasks remain in state
3. Auto-select effect fires: `tasks.length > 0 && !selectedTask` → selects old task name
4. Frame counts fetch fires with `(newDatasetId, oldTaskName)` → backend 404/500 (e.g., Libero task "Clutter Tidy-Up [Stage2]" sent to Egocentric-10K)

### 2. Per-Episode Visual Inconsistency
**Severity:** Medium
**Reproduction:** Open Dataset Analysis → select RealOmni → Signal Comparison → per-episode view. Some episodes appear compressed while others dominate the chart. Compare with Libero which looks consistent.

**Root cause:** `getBatchRange()` in `SignalComparisonChart.tsx` computes raw min/max across all episodes. A single episode with extreme values compresses all others into a tiny band.

### 3. False Gripper Label on Non-Gripper Datasets
**Severity:** Medium
**Reproduction:** Open Dataset Analysis → select RealOmni → Signal Comparison → run analysis. A "Gripper" signal line appears, but RealOmni uses MCAP PoseInFrame messages with position + quaternion orientation — dim 6 is quaternion-w, not gripper.

**Root cause (backend):** `streaming_extractor.py` labels all 7-dim action data as `["x","y","z","rx","ry","rz","gripper"]` regardless of source message type.

**Root cause (frontend):** `SignalComparisonChart.tsx` uses `dims > 6` as the sole check for showing gripper, instead of checking `dimension_labels` for an actual "gripper" label.

### 4. Droid 100 Capabilities Endpoint Crash
**Severity:** Low
**Reproduction:** Call `GET /api/datasets/droid_100/analysis/capabilities` → HTTP 500.

**Root cause:** Dynamic registry entry for droid_100 has `"format": null`. In `_get_format_metadata()`, `dataset_info.get("format", "unknown")` returns `None` (not `"unknown"`) because the key exists with value `null`.

---

## Fix Plan

### Fix 1: HTTP 500 Race Condition

**`frontend/src/hooks/useApi.ts`** — `useTasks` hook (~line 95-124):
- Add `setTasks([])` and `setTotalTasks(0)` immediately at the start of the effect body (before the async `fetchTasks()` call) when `datasetId` changes. This clears old tasks instantly, preventing the auto-select effect from picking a stale task name.

**`frontend/src/components/DatasetAnalysis.tsx`** — dataset onChange handler (~line 126-129):
- Add `cancelAnalysis()` call when dataset changes to close any in-flight SSE connections.

**`backend/api/routes/analysis.py`** — frame-counts endpoint (~line 469-520):
- Wrap `_collect_episode_files()` in try/except. On failure, return a graceful `FrameCountDistribution` response with `total_episodes=0` and a descriptive `source_note` instead of letting the exception propagate as HTTP 500.

### Fix 2: Robust Percentile Normalization

**`frontend/src/components/SignalComparisonChart.tsx`** — `getBatchRange()` function:
- Replace raw min/max with 2nd/98th percentile:
  ```typescript
  function getBatchRange(arrays: number[][]): { min: number; max: number } {
    const all: number[] = [];
    for (const arr of arrays) {
      for (const v of arr) all.push(v);
    }
    if (all.length === 0) return { min: 0, max: 1 };
    all.sort((a, b) => a - b);
    const lo = Math.floor(all.length * 0.02);
    const hi = Math.ceil(all.length * 0.98) - 1;
    const min = all[Math.max(0, lo)];
    const max = all[Math.min(all.length - 1, hi)];
    if (max <= min) return { min: min - 0.5, max: max + 0.5 };
    return { min, max };
  }
  ```
- In `normalizeBatch()`, clamp values outside the robust range to 0–1 bounds.

### Fix 3: Correct Gripper Detection

**`backend/loaders/streaming_extractor.py`** — `_extract_action_vector()` (~line 1037+):
- Change return type to `tuple` returning `(action_vector, msg_type)` where `msg_type` is one of: `"array"`, `"joint"`, `"twist"`, `"pose"`, or `None`.
- Each message type handler returns the appropriate msg_type string.

**`backend/loaders/streaming_extractor.py`** — `extract_actions_data()` (~line 960+):
- Track `detected_msg_type` from the first successful `_extract_action_vector()` call.
- After extraction loop, assign `dimension_labels` based on `detected_msg_type`:
  - `"pose"` + 7 dims → `["x","y","z","qx","qy","qz","qw"]`
  - `"pose"` + 6 dims → `["x","y","z","qx","qy","qz"]`
  - other + 7 dims → `["x","y","z","rx","ry","rz","gripper"]`
  - other + 6 dims → `["x","y","z","rx","ry","rz"]`
  - 3 dims → `["x","y","z"]`

**`frontend/src/components/SignalComparisonChart.tsx`**:
- Add helper:
  ```typescript
  function hasGripperDimension(labels: string[] | null | undefined): boolean {
    if (!labels) return false;
    return labels.some((l) => l.toLowerCase() === "gripper");
  }
  ```
- In `batchRanges` computation and `EpisodeChartRow`: change `if (dims > 6)` to `if (dims > 6 && hasGripperDimension(ep.actions.dimension_labels))`.

### Fix 4: Null Format in Dynamic Registry

**`backend/api/routes/analysis.py`** — `_get_format_metadata()`:
- Change `dataset_info.get("format", "unknown")` to `dataset_info.get("format") or "unknown"` to handle explicit `null` values.

---

## Files to Modify

| File | Fix |
|------|-----|
| `frontend/src/hooks/useApi.ts` | Clear tasks immediately on dataset change (Fix 1) |
| `frontend/src/components/DatasetAnalysis.tsx` | Cancel analysis on dataset switch (Fix 1) |
| `frontend/src/components/SignalComparisonChart.tsx` | Robust percentile normalization (Fix 2) + gripper label check (Fix 3) |
| `backend/api/routes/analysis.py` | Graceful error for missing task folders (Fix 1) + null format handling (Fix 4) |
| `backend/loaders/streaming_extractor.py` | Return msg_type from _extract_action_vector, correct dimension labels (Fix 3) |

---

## Environment Notes

- Quality worktree ports: **3002** (frontend) / **8002** (backend)
- Frontend `NEXT_PUBLIC_API_URL` defaults to port 8000 — must set to `http://localhost:8002/api` via `.env.local` or env var
- Main worktree backend may be running on port 8000 with unfixed code — ensure frontend points to 8002

## Verification

1. Start backend on port 8002, frontend on port 3002 with correct `NEXT_PUBLIC_API_URL`
2. Open Dataset Analysis modal and switch between: RealOmni → Egocentric-10K → Droid → MicroAGI — no HTTP 500
3. Run signal comparison on RealOmni — no false "Gripper" label (should show qw)
4. Run signal comparison on Libero — per-episode charts should look visually consistent (no extreme compression)
5. Egocentric-10K Signal Comparison tab should show "(N/A)"
6. `GET /api/datasets/droid_100/analysis/capabilities` should return 200
