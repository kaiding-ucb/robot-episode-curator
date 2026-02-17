# Plan: Add Starting Position Visual to Signal Comparison

## Context
The signal comparison view compares action/IMU signals across episodes but lacks a visual reference for what each episode's starting scene looks like. Adding first-frame thumbnails gives immediate visual context for understanding why signals might differ (e.g., different object placements, arm positions).

## Changes

### 1. Pass `datasetId` to SignalComparisonChart
**File:** `frontend/src/components/DatasetAnalysis.tsx` (line 321)

- Add `datasetId` prop to the `<SignalComparisonChart>` call:
  ```
  <SignalComparisonChart episodes={signalState.episodes} datasetId={datasetId} />
  ```

### 2. Add frame fetching + grid to SignalComparisonChart
**File:** `frontend/src/components/SignalComparisonChart.tsx`

**a) Update props interface** (line 6-8)
- Add `datasetId: string | null` to `SignalComparisonChartProps`

**b) Add `useState`/`useEffect` imports** (line 3)
- Import `useState` and `useEffect` alongside existing `useMemo`

**c) Add `useFirstFrames` hook** (new function, before main component ~line 589)
- Accepts `episodeList` and `datasetId`
- On mount/change: fetches frame 0 for each episode in parallel via `Promise.all`
- API: `GET ${API_BASE}/episodes/${episodeId}/frames?start=0&end=1&dataset_id=${datasetId}&resolution=low&quality=70`
- Uses `API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"`
- Returns `{ frameData: Map<string, string>, loading: boolean, errors: Set<string> }`
- Cleanup flag to prevent stale state updates

**d) Add `StartingPositionGrid` component** (new function, before main component)
- Props: `episodeList`, `frameData`, `loading`, `errors`
- Layout: `grid grid-cols-5 gap-3` (5 per row)
- Each cell:
  - `<img>` with `data:image/webp;base64,...` or error/loading placeholder
  - 2px border colored with `EPISODE_COLORS[index]` (matches overlay legend)
  - Semi-transparent label overlay at bottom with episode name
  - `aspect-video` for consistent sizing
- Loading state: spinner + "Loading first frames..."
- If no frames loaded: returns null (section hidden)

**e) Integrate into main component JSX** (line 682)
- Call `useFirstFrames(episodeList, datasetId)` after `episodeList` memo
- Render `<StartingPositionGrid>` above the existing overlay section
- Conditionally render only when `datasetId` is truthy

### Layout Order (after changes)
```
Starting Position (5 frames per row)  <-- NEW
Overlay (4 panels in 2x2 grid)
Per Episode (scrollable list)
```

## Files Modified
- `frontend/src/components/DatasetAnalysis.tsx` — 1 line (add datasetId prop)
- `frontend/src/components/SignalComparisonChart.tsx` — add ~80 lines (hook + component + integration)

## Verification
1. Start backend: `cd backend && python -m uvicorn main:app --port 8000`
2. Start frontend: `cd frontend && npm run dev`
3. Open Dataset Analysis, select a dataset/task, click "Start Analysis" with 5 episodes
4. Verify: 1 row of 5 first-frame thumbnails appears above the overlay section
5. Switch to 10 episodes and re-analyze: verify 2 rows of 5 frames
6. Verify: frame border colors match the overlay legend colors
7. Verify: episode labels on frames match the overlay legend labels
