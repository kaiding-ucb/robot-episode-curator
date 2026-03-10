# Rerun Evaluation Findings (LeRobot Focus)

**Date:** 2026-03-10
**Branch:** `spike/rerun-evaluation`
**Rerun SDK:** 0.28.2
**Web Viewer:** `@rerun-io/web-viewer-react` ^0.28.2
**Dataset:** `lerobot/libero_spatial_image` (v3.0, 432 episodes, 10 FPS)

---

## Step 0: Current Viewer Baseline

### What Works
- Dataset browsing: 40 tasks loaded from `lerobot/libero` (streaming via HuggingFace)
- Episode listing: 5 sample episodes per task, with frame counts
- Frame playback: Smooth at 1x speed, timeline scrubbing works
- Frame rendering: Robot arm scenes render correctly (256x256 images)

### What's Missing/Broken
- **Action charts not shown** for streaming LeRobot datasets — `availableModalities` defaults to `["rgb"]` because the streaming adapter doesn't report action modality
- **No Rerun toggle** in the UI — `RerunViewer.tsx` exists as component but is never imported into `page.tsx`
- **Quality events endpoint fails** — 404 error for `/api/quality/events/episode_379?dataset_id=libero`
- **No Standard/Rerun viewer mode toggle** — the e2e tests reference `data-testid="viewer-mode-rerun"` but no such element exists
- Task loading is slow (~15s) — HuggingFace metadata fetch latency

### Screenshots
- `baseline_01_tasks.png` — Task list (40 tasks from Libero)
- `baseline_02_episode_viewer.png` — Episode viewer initial state (black frame)
- `baseline_03_frame_playback.png` — Frame 112/112 showing robot arm
- `baseline_04_full_page.png` — Full page, no action charts visible

---

## Experiment Results

### Exp 1: Rich Single-Episode Viewer

**File:** `exp1_episode_0.rrd` (23 MB for 110 frames)
**Performance:** 2,098 frames/sec logging speed

**Capabilities demonstrated:**
- ✅ All 110 frames logged as `rr.Image` (no 100-frame cap)
- ✅ Both cameras: `camera/rgb` + `camera/wrist`
- ✅ Semantic action names: `action/position/{x,y,z}`, `action/rotation/{roll,pitch,yaw}`, `action/gripper`
- ✅ Robot state: `state/{x,y,z,rx,ry,rz,rw,gripper}`
- ✅ Quality events: `rr.TextLog` with severity levels (stall detection, gripper events)
- ✅ Programmatic Blueprint: cameras top, action plots middle, state+events bottom
- ✅ Dual timeline: frame sequence + wall clock time
- ✅ Classification logic ported from `actionClassification.ts`

**Rerun strengths vs current viewer:**
- Synchronized timeline scrubbing across ALL panels (cameras + actions + state)
- Zoom into any time range on any chart
- Pan/zoom on images
- Speed controls built-in
- All data visible simultaneously without clicking tabs

**File size concern:** 23 MB for 110 frames ≈ 0.21 MB/frame. A 500-frame episode would be ~105 MB.

### Exp 2: Multi-Episode Signal Comparison

**File:** `exp2_task_0_comparison.rrd` (224 KB for 5 episodes, actions only)
**Performance:** Near-instant generation

**Capabilities demonstrated:**
- ✅ 5 episodes overlaid with separate entity paths: `episode_N/action/position_mag`
- ✅ Position magnitude, rotation magnitude, gripper signals all compared
- ✅ Per-axis breakdown: x, y, z plotted per episode
- ✅ Blueprint: overview magnitudes on top, per-axis detail below

**Limitations found:**
- ⚠️ All episodes share the same frame timeline — episodes with different lengths end at different points, which is fine visually
- ⚠️ No built-in "normalize to episode length" — would need to pre-normalize frame indices to 0-1 range
- ⚠️ Color assignment is automatic; cannot control per-episode colors easily
- ⚠️ Legend shows entity path names like `episode_0/action/position_mag` — not as clean as custom labels

**Verdict:** Works for quick comparison but lacks the polish of a custom `SignalComparisonChart` with normalization, opacity control, and clickable episode labels.

### Exp 3: Quality Events on Timeline

**File:** `exp3_episode_0_quality.rrd` (13 MB)
**Quality events detected:** 4 (gripper events + action discontinuities)

**Capabilities demonstrated:**
- ✅ Continuous metrics as `rr.Scalars`: action magnitude, smoothness, divergence, delta magnitude
- ✅ Discrete events as `rr.TextLog` with severity levels (INFO/WARN/ERROR)
- ✅ Events appear in TextLog panel with timestamps
- ✅ Clicking a log entry jumps to that frame in all panels

**Limitations found:**
- ⚠️ No colored dot markers ON the scalar plots — `rr.TextLog` is a separate panel
- ⚠️ Can't place visual annotations (dots, highlights) at specific points on timeseries plots
- ⚠️ Quality grades/scores need custom UI — Rerun has no concept of "quality grade" display
- ⚠️ The TextLog panel is verbose; events are less discoverable than colored dots on a timeline

**Verdict:** Partial replacement for EnhancedTimeline. The continuous metrics (divergence, smoothness) work great. Discrete event markers are less visible than the current dot-on-timeline approach.

### Exp 4: Blueprint + Web Viewer

**File:** `exp4_blueprint.rrd` (11 MB, 50 frames with embedded Blueprint)

**Blueprint API findings:**
- ✅ Blueprint embedded in .rrd file via `default_blueprint` parameter
- ✅ Native viewer respects the Blueprint layout perfectly
- ✅ Layout: cameras top, action plots middle, metadata+events bottom
- ✅ `rrb.Vertical`, `rrb.Horizontal`, `rrb.TimeSeriesView`, `rrb.Spatial2DView`, `rrb.TextLogView`, `rrb.TextDocumentView` all work

**Web viewer findings (KEY QUESTION):**
- ⚠️ `@rerun-io/web-viewer-react` v0.28.2 accepts only `rrd` prop — no `blueprint` prop
- ⚠️ The embedded `default_blueprint` in the .rrd file IS picked up by the web viewer
- ⚠️ This means Blueprints DO work in web viewer when embedded in the .rrd file
- ❌ No way to change Blueprint at runtime from React — once loaded, layout is fixed
- ❌ No bidirectional communication — only `onReady` and `onRecordingOpen` callbacks exist

**Performance benchmark:**

| Frames | Size (MB) | Time (s) | FPS    | MB/frame |
|--------|-----------|----------|--------|----------|
| 10     | ~1        | <0.1     | 3,923  | ~0.1     |
| 50     | 3.3       | <0.1     | 4,997  | 0.066    |
| 100    | 7.4       | <0.1     | 5,118  | 0.074    |
| 110    | 8.1       | <0.1     | 5,187  | 0.074    |

**Extrapolations:**
- 500 frames → ~37 MB → web viewer should handle fine
- 1000 frames → ~74 MB → may start to lag in web viewer
- 256x256 images @ 10 FPS = moderate load; higher res would scale worse

---

## Feature Mapping (Updated)

| Current Feature | Rerun Capable? | Quality | Notes |
|---|---|---|---|
| Frame playback + scrubbing | **YES** | ★★★★★ | Core strength — synchronized across all panels |
| Action timeseries plots | **YES** | ★★★★☆ | Semantic paths work great; auto-coloring is decent |
| Robot state plots | **YES** | ★★★★☆ | Same as actions |
| IMU visualization | **YES** | ★★★★☆ | Same pattern as actions |
| Dual camera (RGB + wrist) | **YES** | ★★★★★ | Side-by-side with shared timeline |
| Quality continuous metrics | **YES** | ★★★★☆ | Divergence, smoothness as Scalars |
| Quality event markers | **PARTIAL** | ★★☆☆☆ | TextLog works but no dot markers on plots |
| Multi-episode overlay | **PARTIAL** | ★★★☆☆ | Works but limited control over normalization/colors |
| Frame count histogram | **NO** | N/A | `rr.BarChart` is too limited |
| Quality grades/panels | **NO** | N/A | Custom UI needed |
| Dataset browsing/download | **NO** | N/A | Not Rerun's scope |
| Cross-dataset comparison | **NO** | N/A | Not Rerun's scope |
| Speed control (0.25x-2x) | **YES** | ★★★★☆ | Built into Rerun viewer |
| Episode metadata display | **YES** | ★★★★☆ | TextDocument with Markdown |

---

## Technical Answers

### 1. Web viewer Blueprint support?
**YES, via embedded default_blueprint in .rrd file.** No runtime Blueprint prop available in v0.28.2.

### 2. Performance at scale?
- 500 frames (~37 MB): Should work fine
- 1000+ frames: Test needed, expect ~74 MB+ file sizes
- Bottleneck is image size, not scalar data (224 KB for 5 episodes of actions-only)

### 3. Bidirectional communication?
**NO.** Only `onReady` and `onRecordingOpen` callbacks. Cannot:
- Get current frame position from web viewer
- Programmatically seek to a frame
- Respond to user clicks on timeline
- Sync external state with Rerun state

### 4. Rerun version considerations?
Current: 0.28.2. The web viewer's callback API is minimal. Future versions may add more interactivity.

---

## Recommendation

### **Outcome B — Enhanced Optional View** (Recommended)

**Rationale:**
1. Rerun is excellent for synchronized multi-modal visualization but lacks the interactivity needed to replace the custom viewer entirely (no bidirectional communication)
2. The current viewer has critical gaps (no action charts for streaming LeRobot data, no Rerun toggle) that should be fixed first
3. Rerun adds unique value: dual-camera sync, action+state overlays, quality metric plots — all in one synchronized view

**Implementation plan:**
1. **Fix current viewer first:** Wire up action modality for streaming LeRobot datasets
2. **Integrate RerunViewer:** Add Standard/Rerun toggle to EpisodeViewer (component already exists)
3. **Enrich RRD generation:** Replace current `action/dim_N` with semantic names from Exp 1
4. **Remove 100-frame cap:** Log all frames with Blueprint
5. **Add quality overlay:** Port quality metrics from Exp 3 into RRD generation
6. **Add multi-episode:** Port comparison view from Exp 2 for Analysis modal

**Why not Outcome A (Rerun as Primary)?**
- No bidirectional communication = can't sync Rerun timeline with React state
- File sizes (23 MB for 110 frames) add latency before viewing
- Custom UI (quality grades, dataset browsing) still needed alongside Rerun

**Why not Outcome C (Desktop Only)?**
- The web viewer works well enough for an optional view
- Forcing users to install Rerun natively adds friction
- Embedded Blueprints in .rrd files solve the layout problem

---

## Files Generated

| File | Size | Description |
|---|---|---|
| `exp1_episode_0.rrd` | 23 MB | Full episode: 2 cameras, semantic actions, state, events |
| `exp2_task_0_comparison.rrd` | 224 KB | 5-episode action signal comparison |
| `exp3_episode_0_quality.rrd` | 13 MB | Episode with quality metrics overlay |
| `exp4_blueprint.rrd` | 11 MB | Blueprint test for web viewer |
| `baseline_01_tasks.png` | — | Screenshot: task list |
| `baseline_02_episode_viewer.png` | — | Screenshot: episode initial state |
| `baseline_03_frame_playback.png` | — | Screenshot: frame playback working |
| `baseline_04_full_page.png` | — | Screenshot: full page, no action charts |
