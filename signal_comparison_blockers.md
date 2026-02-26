# Signal Comparison Performance Blockers

## Context
Signal comparison is slow for datasets with large frames per episode. The bottleneck spans the full pipeline: sequential backend downloads, over-sized JSON payloads, redundant frontend computations, and excessive re-renders during SSE streaming. The top 4 optimizations alone should cut perceived latency by ~3-5x.

---

## HIGH IMPACT

### 1. Backend: Sequential Parquet Downloads
**File:** `backend/api/routes/analysis.py` — `_extract_lerobot_episodes()` (lines 310-327)

Parquet files download one at a time in a for-loop. 5 files × ~2s each = ~10s wall time.

**Fix:** `asyncio.gather` with `Semaphore(4)` to download concurrently. Process results in order afterward for the early-exit episode counting logic. Each `_download_parquet()` call creates its own `HfFileSystem` — already safe for concurrency.

**Expected:** ~10s → ~3s

---

### 2. Backend: Over-Sized JSON Payloads
**Files:** `backend/api/routes/analysis.py`, `frontend/src/hooks/useDatasetAnalysis.ts`

Backend sends up to 500 frames/episode (3,500 floats as nested JSON arrays). Charts render at 400px wide — 200 points is visually identical.

**Fix:** Add `resolution` query param (default 200) to the signals endpoint. Pass through to `max_frames_per_episode` and `_downsample_actions(max_points=)`. Frontend appends `&resolution=200` to SSE URL.

**Expected:** ~2.5x smaller payloads, proportional reduction in client-side computation

---

### 3. Frontend: Triple Magnitude Computation
**File:** `frontend/src/components/SignalComparisonChart.tsx`

`computeMagnitude()` runs 3x per episode per render:
1. `batchRanges` useMemo (lines 791-806)
2. Overlay traces useMemo (lines 828-858)
3. Each `EpisodeChartRow` component (lines 311-338)

**Fix:** Single combined useMemo that computes all magnitudes once into a `Map<string, PrecomputedEpisodeSignals>`, then derives batchRanges and overlay traces from it. Pass precomputed data to `EpisodeChartRow` as a prop.

**Expected:** 3x reduction in float ops per render

---

### 4. Frontend: Cascading Re-Renders During SSE Streaming
**File:** `frontend/src/hooks/useDatasetAnalysis.ts` — `useSignalComparison()` (lines 153-163)

Every `episode_data` SSE event triggers `setState` → `new Map()` → full chart re-render with all useMemo recomputations. 10 episodes = 10 cascading re-renders.

**Fix:** Accumulate episodes in a `useRef<Map>` instead of calling setState. Flush to state via `requestAnimationFrame` (debounced — each new event resets the timer). On `done`, flush synchronously before setting phase.

**Expected:** 10 renders → 1-2 renders. Combined with fix #3, eliminates ~27 redundant magnitude passes.

---

## MEDIUM IMPACT

### 5. Frontend: Full-Resolution Overlay SVG Paths
**File:** `frontend/src/components/SignalComparisonChart.tsx` — `OverlayPanel` (line 541)

Individual episode traces in the overlay render at full data length (500+ point SVG paths) even though chart is 300px wide. The envelope already uses `RESAMPLE_LEN=200`.

**Fix:** Pre-compute overlay trace paths in a useMemo using `resampleToLength()` (already exists in the file) to match `RESAMPLE_LEN`.

**Expected:** SVG path complexity drops ~2.5x across 4 overlay panels

---

### 6. Frontend: Sequential Thumbnail Fetching
**File:** `frontend/src/components/SignalComparisonChart.tsx` — `useFirstFrames` (lines 684-689)

Starting position thumbnails fetch one at a time. 10 episodes × ~200ms = ~2s.

**Fix:** Worker-pool pattern with 3 concurrent fetchers sharing a counter index. `Promise.all(workers)` ensures optimal pipelining.

**Expected:** ~2s → ~0.8s for 10 thumbnails

---

### 7. Backend: No Signal Cache
**Files:** `backend/cache.py`, `backend/api/routes/analysis.py`

Re-analyzing the same dataset/task re-downloads all parquet files from HuggingFace every time. Signal data is deterministic per `(repo_id, branch, episode_index)`.

**Fix:** Add `SIGNALS_CACHE_DIR` to `cache.py` with a `FileCache` accessor. In `_extract_lerobot_episodes()`: check cache before download, store after extraction. Key: `get_cache_key("signal_ep", repo_id, branch, ep_idx)`.

**Expected:** Repeat analyses complete in milliseconds

---

## Implementation Notes

- Backend fixes (1, 2, 7) and frontend fixes (3, 4, 5, 6) are fully independent
- Fix #4 should come after fix #3 (otherwise batched updates still trigger expensive triple-magnitude computes)
- Fix #5 depends on fix #3 (overlay traces will reference precomputed data)
- Fix #6 is standalone, can be done anytime

### Critical Files
| File | Optimizations |
|------|--------------|
| `backend/api/routes/analysis.py` | 1, 2, 7 |
| `backend/cache.py` | 7 |
| `frontend/src/components/SignalComparisonChart.tsx` | 3, 5, 6 |
| `frontend/src/hooks/useDatasetAnalysis.ts` | 2 (URL param), 4 |

### Existing Utilities to Reuse
- `cache.py:FileCache` / `get_cache_key()` / `ensure_cache_dirs()` — for signal cache
- `SignalComparisonChart.tsx:resampleToLength()` — for overlay downsampling
- `SignalComparisonChart.tsx:computeMagnitude` / `computeIMUMagnitude` / `computeGripperSignal` — wrap in single pass
