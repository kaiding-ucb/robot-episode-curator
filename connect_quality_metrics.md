# Connect Quality Events to Metrics

## Problem

Quality event markers (Gripper, Pause, Direction, Jerk) on the timeline are disconnected from quality metrics (Motion Smoothness 40%, Action-Obs Sync 50%, etc.). Users can't understand WHY an episode scores high/low on a metric.

## Solution Overview

Create metric-specific events that explain each metric's score, and add interactive filtering so clicking a metric highlights its contributing events on the timeline.

---

## Implementation

### 1. Extend QualityEvent Data Structure

**File:** `backend/quality/diversity.py` (lines 20-27)

Add fields to track which metrics each event affects:

```python
@dataclass
class QualityEvent:
    frame: int
    event_type: str
    severity: str
    score: float
    description: str
    # NEW FIELDS:
    affected_metrics: List[str] = field(default_factory=list)  # e.g., ['motion_smoothness']
    metric_impact: Dict[str, float] = field(default_factory=dict)  # e.g., {'motion_smoothness': -0.05}
```

### 2. Add Metric-Specific Event Detectors

**File:** `backend/quality/temporal.py`

Add new detection functions that return events for each temporal metric:

| Function | Detects | Affects Metric |
|----------|---------|----------------|
| `detect_smoothness_events()` | Frames where jerk > 2x mean | `motion_smoothness` |
| `detect_sync_events()` | Frames with action-observation lag | `sync_score` |
| `detect_consistency_events()` | Position/rotation jumps > thresholds | `action_consistency` |

**File:** `backend/quality/diversity.py`

Add/update detectors:

| Function | Detects | Affects Metric |
|----------|---------|----------------|
| `detect_near_miss_events()` | High accel + recovery (move from compute_near_miss_ratio) | `near_miss_ratio` |
| Update `detect_high_jerk()` | Add `affected_metrics=['motion_smoothness']` | `motion_smoothness` |
| Update `detect_direction_changes()` | Add `affected_metrics=['transition_diversity', 'recovery_behavior_score']` | multiple |

### 3. Aggregate All Events in QualityScore

**File:** `backend/quality/aggregator.py`

Update `compute_quality_score()` to:
1. Collect events from both temporal and diversity modules
2. Build a `metric_event_map: Dict[str, List[int]]` mapping metric names to event indices
3. Include both in the returned `QualityScore`

### 4. Update API Response

**File:** `backend/api/routes/quality.py`

Update `QualityEventsResponse` to include:
- `metric_event_map: Dict[str, List[int]]`
- Extended event fields (`affected_metrics`, `metric_impact`)

### 5. Frontend: Make Metrics Clickable

**File:** `frontend/src/components/QualityPanel.tsx`

Changes to `MetricBar`:
- Add `onClick` handler and `isSelected` state
- Show event count badge (e.g., "8 issues")
- Highlight when selected (blue ring)

Add state management:
```typescript
const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
```

### 6. Frontend: Filter Timeline by Metric

**File:** `frontend/src/components/EnhancedTimeline.tsx`

Add props:
```typescript
selectedMetric?: string | null;
```

When metric selected:
- Filter events to only show those with `affected_metrics.includes(selectedMetric)`
- Fade or hide non-matching events
- Show indicator text: "Showing events affecting: Motion Smoothness"

### 7. Enhanced Tooltips

**File:** `frontend/src/components/EnhancedTimeline.tsx`

Update event marker tooltips to show:
- Event type and frame (existing)
- "Affects: motion_smoothness, action_consistency"
- "Impact: -5% smoothness"

### 8. Page-Level State Coordination

**File:** `frontend/src/app/page.tsx`

Add shared state to connect QualityPanel selection to EnhancedTimeline filtering:
```typescript
const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
// Pass to both QualityPanel and EpisodeViewer
```

---

## Critical Files to Modify

| File | Changes |
|------|---------|
| `backend/quality/diversity.py` | Extend QualityEvent, update existing detectors with affected_metrics |
| `backend/quality/temporal.py` | Add `detect_smoothness_events()`, `detect_sync_events()`, `detect_consistency_events()` |
| `backend/quality/aggregator.py` | Merge events, build metric_event_map |
| `backend/api/routes/quality.py` | Update response models |
| `frontend/src/types/quality.ts` | Add new TypeScript types |
| `frontend/src/components/QualityPanel.tsx` | Clickable MetricBar, event counts |
| `frontend/src/components/EnhancedTimeline.tsx` | Metric filtering, enhanced tooltips |
| `frontend/src/app/page.tsx` | selectedMetric state coordination |

---

## User Interaction Flow

1. User sees "Motion Smoothness: 40%" with badge "5 issues"
2. User clicks the metric bar
3. Bar highlights, timeline filters to show only smoothness-related events (red jerk markers)
4. User clicks event @42 on timeline
5. Video jumps to frame 42, tooltip shows "High jerk (3.2x avg) - Affects: Motion Smoothness (-8%)"
6. User clicks metric bar again to deselect, timeline shows all events

---

## Verification

1. **Backend:** Run existing quality tests, add tests for new event detectors
2. **API:** Verify `/api/quality/events/{episode_id}` returns `affected_metrics` and `metric_event_map`
3. **Frontend:**
   - Click each metric bar, verify timeline filters correctly
   - Hover over events, verify tooltips show metric impact
   - Click event badges, verify video jumps to correct frame
4. **End-to-end:** Load an episode with low Motion Smoothness, click the metric, verify the timeline shows jerk events at the problem frames
