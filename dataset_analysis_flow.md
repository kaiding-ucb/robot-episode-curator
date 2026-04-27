# Dataset Analysis — End-to-End Flow

This doc traces what happens from the moment a user opens the Dataset Analysis modal and selects a task through the final rendered cluster cards + flagged-episode list. It's a map, not a spec — see the referenced source files for the precise logic.

## 10-second summary

```
User opens Dataset Analysis → picks task → hits "Action Insights" tab
      │
      ▼
Frontend fetches GET /api/datasets/{id}/analysis/phase-aware?task_name=…&cohort_size=N
      │
      ▼
Backend loads raw action+state arrays for first N episodes of the task (HF parquet)
      │
      ▼
Per-episode: detect phases via gripper events → shape features → per-phase stats
      │
      ▼
Cohort-level: per-phase envelope, phase-duration MAD, cycle-count, Mahalanobis-shape
      │      (each produces 0-many AnomalyReason entries per episode)
      ▼
Gap-statistic clustering on shape features (K chosen from {2…5})
      │
      ▼
JSON response → UI renders clusters + flagged-episode cards with reason lineage
      │
      ▼
(optional) User clicks "Run AI analysis"
      │
      ▼
Backend extracts MP4 clips (cached), uploads to Gemini, runs 2 prompts:
   - cluster-characterisation (one call per cluster)
   - flag-enrichment (batched, 7 flagged per call, includes gripper/phase context)
      │
      ▼
Responses merged into the result: gemini_label/description per cluster,
   severity/confirmation/novel-observations per flagged episode, plus new
   signal="gemini" reasons.
      │
      ▼
UI re-renders with AI-augmented cards.
```

---

## Layer by layer

### 1. User interaction (frontend)

**Entry:** `frontend/src/components/DatasetAnalysis.tsx` — the modal.

Three pieces of state that drive the flow:
- `activeTab`: `"frame-counts"` or `"signal-comparison"`. Phase-aware lives under **Action Insights** (signal-comparison).
- `signalView`: `"phase-aware"` (default) or `"envelope"` (legacy debug toggle, decision #4).
- `selectedTask`: name string from the Task dropdown.

When `activeTab === "signal-comparison"` and `signalView === "phase-aware"`, the modal renders `<PhaseAwarePanel datasetId taskName />`.

**Panel:** `frontend/src/components/PhaseAwarePanel.tsx`
- Local state: `cohortSize` (10/30/50), `data`, `loading`, `gemLoading`, `gemError`.
- On mount / `cohortSize` change: fires a GET to the phase-aware endpoint with `include_gemini=false`.
- On "Run AI analysis" click: re-fires the same URL with `include_gemini=true` and updates `data` with the enriched response.

---

### 2. Backend endpoint (first responder)

**Route:** `backend/api/routes/analysis.py::get_phase_aware`

```python
@router.get("/{dataset_id}/analysis/phase-aware")
async def get_phase_aware(
    request, dataset_id, task_name,
    cohort_size=30, refresh=False,
    include_gemini=False, flagged_cap=20,
):
```

Order of operations:

1. **Cache check.** `_phase_aware_cache[(dataset_id, task_name, cohort_size)]`, an in-memory dict. Hit if `refresh=False` and the cached payload already has `gemini.enriched=True` (or AI wasn't requested). Miss drops through.
2. **Dataset capability check.** `is_lerobot_dataset()` — we only support LeRobot-format datasets today; anything else returns `{method: "unsupported"}`.
3. **Resolve task episodes.** Calls `list_task_episodes(dataset_id, task_name, limit=cohort_size)` which hits the existing `/tasks/{task}/episodes` endpoint logic. Returns a list of `EpisodeInfo.id` strings like `episode_85`. Parses the global episode-index out (e.g. 85) and truncates to `cohort_size`.
4. **Probe FPS** from `meta/info.json` (default 10 Hz if missing).
5. **Load raw signals.** `backend/analysis/loader.py::load_episodes_action_state(repo_id, target_episode_ids)` — see §3.
6. **Phase-aware compute.** `backend/analysis/phase_aware.py::analyze_task(episodes, task_name, fps)` — see §4-§8.
7. **Cache + return.** If `include_gemini=false`, return the result dict. Otherwise see §10.

---

### 3. Raw signal loader

**File:** `backend/analysis/loader.py`

The same LeRobot format is structured very differently across datasets — this loader adapts at runtime, not via per-dataset code:

1. **List data files.** `GET huggingface.co/api/datasets/{repo_id}/tree/main/data?recursive=1` → e.g. Libero has 377 `data/chunk-000/file-*.parquet` files; UMI has a single file.
2. **Probe schema.** Read only the first file's schema (no rows). Determines whether the dataset has an explicit `action` column (Libero) or only `observation.state` + `gripper_width` (UMI).
3. **Batched download.** Concurrent HTTP GETs (semaphore limit = 6), 8 files per batch. For each file: read the subset of columns we know exist, extract target episodes, stop once all targets found.
4. **Synthesize action (UMI path).** If `action` column is missing, build a synthetic 7-dim action per frame:
   - `action[:, 0:3]` = `np.diff(state[:, 0:3])` (translation)
   - `action[:, 3:6]` = `np.diff(state[:, 3:6])` (rotation)
   - `action[:, 6]` = `-gripper_width` (flipped so narrow=low, matching Libero's closed-is-low convention)
5. **Returns** `[{"episode_id", "action": (T,7), "state": (T,S)}, …]`.

This is the single place where per-dataset variance is absorbed; everything downstream sees the uniform `{action, state}` shape.

---

### 4. Phase segmentation

**File:** `backend/analysis/phase_aware.py::detect_phases`

Each episode is segmented independently.

**Step A — detect gripper closed ranges.** `_detect_gripper_closed_ranges(action)`:
- Pull `action[:, 6]` as the gripper signal.
- `medfilt(kernel=15)` smoothing.
- Robust p5/p95-midpoint threshold (resilient to injection outliers).
- Auto-flip convention if the "closed" state ends up being the majority of the episode.
- Build raw transition log: `[("close", frame_idx), ("open", frame_idx), …]` — preserved on the segmentation output as `raw_gripper_events` so mid-cycle re-grips survive the merging step below.
- Drop closed-ranges shorter than 20 frames (2s @ 10Hz) — filters noise.
- Merge closed-ranges separated by < 30 frames of open time — "brief release then re-grip" is one logical pick-place.

**Step B — split each cycle into phases.** For every merged `(close_i, open_i)`:
- **Approach**: `[prev_end, close_i)`
- **Grasp**: `[close_i, close_i + 3)` (narrow event window)
- **Lift**: `[grasp_end, z_peak)` (from state `z`)
- **Transit**: `[z_peak, z_descend_start)` where descent_start = z drops below 70% of peak height
- **Place**: `[z_descend_start, open_i)`
- **Release**: `[open_i, open_i + 3)` (narrow event window)
- **Return**: `[last_release_end, T)` (trailing)

Phases chain strictly (end-of-one = start-of-next; no overlap).

**Step C — fallback (no gripper).** If the episode has no detectable gripper transitions (e.g. "push the block" tasks), fall back to velocity-minima segmentation: find low-velocity points in `action[:, 0:3]` deltas, split the episode at those points, label phases generically `Motion1 / Pause1 / Motion2 / …`.

Output: `PhaseSegmentation(phases=[Phase(name, start, end, cycle)], method="gripper"|"velocity_minima", num_cycles, gripper_closed_ranges, raw_gripper_transitions, raw_gripper_events)`.

---

### 5. Shape features (per-episode)

**File:** `phase_aware.py::_compute_shape_features`

10-dim feature vector per episode:

| feature | definition | intuition |
|---|---|---|
| `arc_length_xyz` | sum of \|Δpos\| | total translation distance |
| `arc_length_rot` | sum of \|Δrot\| | total wrist rotation |
| `num_gripper_events` | raw (pre-merge) transition count | 2×cycles normally; higher signals re-grips |
| `num_pauses` | count of ≥10-frame low-velocity runs | hesitation markers |
| `z_range` | max(z)−min(z) | vertical envelope |
| `xy_bbox_area` | (max_x−min_x)·(max_y−min_y) | horizontal workspace |
| `path_directness` | ‖end−start‖ / arc_length_xyz | 0=circuitous, 1=straight |
| `gripper_closed_ratio` | closed frames / total frames | fraction of time holding |
| `num_direction_changes` | velocity sign flips across x,y,z | jerky vs smooth |
| `episode_duration` | frame count | pacing |

These get standardized cohort-wide before being fed into both the Mahalanobis anomaly check and the gap-statistic clustering.

Also computed per-episode: **`phase_action_summary`** — for each phase segment, the mean/peak of position-magnitude and rotation-magnitude. Fed to Gemini as context; not used by the statistical detectors.

---

### 6. Anomaly detection (four signal types)

All four produce `AnomalyReason(signal, phase, cycle, feature, magnitude, explanation)` entries, accumulated per episode.

| signal | source function | threshold | when it fires |
|---|---|---|---|
| `envelope` | `_flag_envelope_violations` | ≥ 3σ sustained over ≥ 40% of phase bins | position/rotation magnitude spikes during a specific phase, compared to cohort bin-by-bin after smoothing |
| `duration` | `_flag_duration_outliers` | \|z\| > 3.5 (MAD-scaled) on phase duration | this phase of this cycle took far longer/shorter than the cohort median for that same phase+cycle |
| `cycle_count` | same fn | \|count − mode\| ≥ threshold (adaptive) | wrong number of pick-place cycles; threshold = 1 for single-pick tasks, 2 for multi-pick |
| `shape` | `_flag_shape_outliers` | Mahalanobis² > χ²₀.₉₉ | multivariate trajectory outlier on the 10 shape features |

Envelope needs phase-matched cohorts (cohort is grouped by `(phase_name, cycle, num_cycles)` so "cycle 1 of a 2-cycle episode" only compares against other 2-cycle episodes' cycle 1). Phases with median duration < 15 frames are skipped — envelope is meaningless on effectively-instantaneous segments.

Human-readable explanation strings (no σ/χ²) are generated at this layer via `_duration_label`, `_shape_label`, `CLUSTER_TITLE`, `FEATURE_HUMAN`.

---

### 7. Variance clustering

**File:** `phase_aware.py::_cluster_variance_modes`

Inputs: the 10-dim cohort feature matrix, already standardized.

1. **Choose K** via gap statistic (`_gap_statistic_k`). Candidate range is `{2, 3, 4, 5}`; K=1 is not a candidate by design (we always surface two or more modes). Algorithm: for each K, cluster real data with Ward-linkage hierarchical, compute WSS; bootstrap 20 uniform reference datasets for the same K, compute expected WSS; pick smallest K where `Gap(k) ≥ Gap(k+1) − s_{k+1}`.
2. **Fit final clustering** with chosen K.
3. **Medoid per cluster** = member with minimum sum-of-squared-distance to other members (in standardized space).
4. **Dominant features** = top 3 axes where this cluster's standardized mean deviates most from zero (>0.5 SD).
5. **Title** from `CLUSTER_TITLE` lookup, e.g. `("z_range", "high") → "High-clearance / tall lift"`.
6. **Human feature labels** from `FEATURE_HUMAN`, e.g. `z_range → "Vertical range"` — surfaced as `dominant_features_human` on the response so the UI can render readable tags.

---

### 8. Assembling the response

The endpoint's response shape (serialized from `PhaseAwareResult.to_dict()`):

```json
{
  "task_name": "…",
  "cohort_size": 30,
  "fps": 10.0,
  "method": "gripper",
  "algorithm": { "envelope_sigma": 3.0, "duration_mad_threshold": 3.5, … },
  "clusters": [
    {
      "id": "C1",
      "label": "High-clearance / tall lift",
      "members": ["episode_6", "episode_8", …],
      "medoid": "episode_18",
      "dominant_features": ["z_range", "num_pauses", "path_directness"],
      "dominant_features_human": ["Vertical range", "Mid-motion pauses", "Path directness"]
    }, …
  ],
  "episodes": [
    {
      "episode_id": "episode_121",
      "cluster": "C3",
      "num_cycles": 2,
      "frames": 292,
      "phases": [ { "name": "Approach", "cycle": 0, "start": 0, "end": 43 }, … ],
      "shape_features": { … },
      "raw_gripper_events": [ { "type": "close", "frame": 43 }, … ],
      "phase_action_summary": [ { "name": "Lift", "cycle": 0, "pos_mag_mean": 0.284, … }, … ],
      "anomaly": {
        "is_anomaly": true,
        "reasons": [
          { "signal": "duration", "phase": "Lift", "cycle": 0, "magnitude": 5.1,
            "explanation": "Prolonged lift: Lift phase (cycle 0) took 2.4× typical time (68f vs typical 28f)" }
        ]
      }
    }, …
  ]
}
```

Cached in `_phase_aware_cache` keyed on `(dataset_id, task_name, cohort_size)`.

---

### 9. Frontend rendering (phase-aware only path)

**File:** `PhaseAwarePanel.tsx`

Top to bottom:

1. **Cohort-size selector** — 10/30/50 buttons, drives the request. Sub-label shows `cohort: 30 · K=2 · 5 flagged`.
2. **AI semantic analysis banner** — until Gemini has enriched, shows the "Run AI analysis" button + cost disclaimer. After enrichment, shows a subtle `AI-enriched · N tokens` note.
3. **Variance clusters grid** — one card per cluster:
   - Title = `cluster.gemini_label || cluster.label` (stat label only when Gemini absent).
   - Stat label, member count, medoid listed below.
   - Gemini description (when present).
   - Dominant-feature tags with human labels (raw metric in tooltip).
   - Full member list (sorted numerically, medoid bolded).
4. **Flagged episodes list** — one card per episode where `anomaly.is_anomaly`:
   - Header: episode_id, cluster dot, cycle+frame counts, severity badge (if Gemini run).
   - Stat reasons (one per AnomalyReason, with colored signal badge).
   - Gemini confirmation (italicized `AI:` line, when present).
   - Novel AI-observed anomalies (`AI+novel` badge) — fed into the same reasons array by the enrichment merger.
   - "Open in viewer →" button that jumps the main Rerun viewer to that episode.

---

### 10. Optional: Gemini enrichment

Fires when the endpoint receives `include_gemini=true` AND the cached result doesn't already have `gemini.enriched === true`.

**Entry:** `backend/analysis/gemini/enrich.py::enrich_with_gemini`

**Cache check** — disk cache at `~/.cache/data_viewer/phase_aware_gemini/responses/{hash}.json`, keyed on `(dataset_id, task_name, cohort_size, algo_version, prompt_version)`. Hit returns instant.

**Candidate selection** (`_plan_calls`):
- For each cluster: `3 medoid-ish videos + 1 contrastive video from each other cluster` (capped at 10 videos per call).
- For flagged episodes: up to `flagged_cap` (default 20). Batched as `3 normals + up to 7 flagged` per call.
- Collects the union of all referenced episode_ids for upload.

**Clip preparation** (`video_cache.py::get_episode_clip`):
- Per-episode MP4 slice cached on disk at `~/.cache/data_viewer/phase_aware_gemini/clips/{repo_id}/episode_N.mp4`.
- Cache miss: download the relevant HF chunk MP4, look up the episode's `[t_start, t_end]` from `meta/episodes/*.parquet`, ffmpeg-slice. One-time cost; reused across all future Gemini runs.

**Uploading** — `GeminiClient.upload_file` for each clip (parallel, awaits ACTIVE state).

**Calls** — both prompt templates are YAML under `backend/analysis/gemini/prompts/`.
- **Cluster characterisation** (`cluster_char.yaml`): per-cluster call, returns `{proposed_label, description, evidence, confidence}`.
- **Flag enrichment** (`flag_enrich.yaml`): per-batch call. Critically, this is where per-episode *data context* is injected: phase timeline in seconds, **gripper event log with per-event timestamps**, per-phase position/rotation magnitude mean/peak. This is what lets Gemini correctly classify `episode_121`'s "slow lift" as a re-grip recovery rather than just a leisurely lift. Returns per-episode `{episode_id, confirmation, severity, novel_observations[]}`.

Both endpoints use Gemini's constrained-JSON mode (`response_mime_type="application/json"` + schema) to avoid parsing fragility.

**Merging** (`enrich.py` end-of-file):
- Cluster cards gain `gemini_label`, `gemini_description`, `gemini_evidence`, `gemini_confidence`.
- Flagged episodes gain `gemini_severity` ("stylistic"/"suspicious"/"mistake"), `gemini_confirmation`, `gemini_observations[]`.
- Every novel Gemini observation is also appended to the episode's `anomaly.reasons` as `{signal: "gemini", phase, cycle, magnitude: 0, explanation}` — so they render on the same card as stat reasons with a distinct badge.

**Persist to disk cache**, return to endpoint, endpoint returns to UI.

Typical latency: **55–80 s first run** (cold), **~0 s cached**. Typical cost: **$0.01–$0.04 per task** depending on cohort size.

---

## Cache map (four levels)

| cache | key | lifetime | location |
|---|---|---|---|
| Parquet file cache | HF hub cached download | persistent | `~/.cache/huggingface/` |
| Phase-aware in-memory | `(dataset_id, task_name, cohort_size)` | per backend process | `_phase_aware_cache` in `analysis.py` |
| Video clip | `(repo_id, episode_idx)` | persistent | `~/.cache/data_viewer/phase_aware_gemini/clips/` |
| Gemini response | `(dataset_id, task_name, cohort_size, algo_version, prompt_version)` | persistent | `~/.cache/data_viewer/phase_aware_gemini/responses/` |

`algo_version` and `prompt_version` bump automatically when the respective source files change (the latter via content-hash of YAMLs under `prompts/`). Server restart drops the in-memory cache only; disk caches survive.

---

## Dataset-specific adaptation points

The only per-dataset differences the code currently handles:

- **`loader.py`** — `action` column missing (UMI) → synthesize from state + gripper_width.
- **`phase_aware.py`** — adaptive `CYCLE_COUNT_MIN_DELTA` (1 for mode≤1 single-pick tasks, 2 for multi-pick).
- **`video_cache.py`** — picks `observation.image` or `observation.images.image` depending on what exists.

Nothing else requires per-dataset knowledge. Adding a new LeRobot dataset should Just Work provided (a) it has `meta/info.json` with feature/video metadata, (b) either `action` or `observation.state`+`gripper_width` in the data parquets, and (c) videos under `videos/{video_key}/chunk-*/file-*.mp4`.

---

## Key files at a glance

| role | file |
|---|---|
| Endpoint & orchestration | `backend/api/routes/analysis.py` |
| Raw signal loader | `backend/analysis/loader.py` |
| Phase segmentation + anomaly detection + clustering | `backend/analysis/phase_aware.py` |
| Gemini enrichment orchestrator | `backend/analysis/gemini/enrich.py` |
| Gemini upload/call client | `backend/analysis/gemini/client.py` |
| Gemini prompts | `backend/analysis/gemini/prompts/cluster_char.yaml`, `flag_enrich.yaml` |
| Per-episode clip extraction + cache | `backend/analysis/gemini/video_cache.py` |
| Frontend panel | `frontend/src/components/PhaseAwarePanel.tsx` |
| Modal host | `frontend/src/components/DatasetAnalysis.tsx` |

---

## Prior art / related docs

- Phase 0 algorithm validation + threshold tuning rationale: `exploration/phase_aware/FINDINGS.md`
- Mode-A (exhaustive) vs Mode-B (flagged-only) Gemini comparison: `exploration/phase_aware/compare_output/compare_results.json`
- UMI first-10 Mode B: `exploration/phase_aware/umi_output/umi_first_10_mode_b.json`
