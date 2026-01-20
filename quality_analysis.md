# Quality Analysis - Score Computation Documentation

This document explains how quality scores are computed in the right sidebar of the Quality Analysis panel.

---

## Overall Score Formula

**Location:** `backend/quality/aggregator.py:103-109`

The overall quality score is a weighted combination of 5 components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Temporal Quality | 35% | Motion smoothness, action consistency, completeness |
| Diversity | 30% | Recovery behaviors, transition diversity, near-misses |
| Visual Quality | 15% | Resolution, blur, exposure, contrast |
| Semantic | 10% | Hardcoded at 0.5 (no annotations available) |
| Completeness | 10% | From temporal metrics (trajectory_completeness) |

```python
overall = (
    0.35 * temporal.overall_temporal_score +
    0.30 * diversity.overall_diversity_score +
    0.15 * visual.overall_visual_score +
    0.10 * 0.5 +  # semantic (default)
    0.10 * temporal.trajectory_completeness
)
```

---

## Quality Grade

**Location:** `backend/quality/aggregator.py:51-62`

| Score Range | Grade |
|-------------|-------|
| >= 0.9 | A |
| >= 0.8 | B |
| >= 0.7 | C |
| >= 0.6 | D |
| < 0.6 | F |

---

## Temporal Metrics

**Location:** `backend/quality/temporal.py`

### Overall Temporal Score Weights (`temporal.py:218-223`)

| Metric | Weight |
|--------|--------|
| Motion Smoothness | 40% |
| Action Consistency | 25% |
| Trajectory Completeness | 20% |
| Frame Rate Consistency | 15% |

### Motion Smoothness (`temporal.py:25-68`)

Measures how smooth the robot's motion is using coefficient of variation (CV) of velocity.

**Algorithm:**
1. Compute velocity (first derivative of actions)
2. Compute acceleration (second derivative)
3. Calculate CV = std(velocity) / mean(|velocity|) for each dimension
4. Smoothness = 1.0 - (mean_CV / 2.0), clamped to [0, 1]
5. Apply acceleration penalty: subtract up to 0.3 for high acceleration variance

**Interpretation:**
- CV near 0 = perfectly smooth motion
- CV > 2 = very jerky motion

### Action Consistency (`temporal.py:108-157`)

Checks for physically plausible actions (no sudden jumps).

**For 7-DoF actions (xyz + rotation + gripper):**
- Position consistency (dims 0-2): Penalizes jumps > 0.2
- Rotation consistency (dims 3-5): Penalizes jumps > 1.0 rad
- Gripper consistency (dim 6): Allows smooth or binary transitions

**Weights:** 40% position + 30% rotation + 30% gripper

### Trajectory Completeness (`temporal.py:159-191`)

**If success_label available:**
- Success = 1.0
- Failure = 0.3

**Otherwise (heuristics):**
- Length score: min(1.0, num_actions / 10)
- Settled score: Checks if final velocity is low (robot stopped)
- Combined: 60% length + 40% settled

### Frame Rate Consistency (`temporal.py:71-105`)

If timestamps available:
- Computes CV of time deltas
- Checks deviation from expected FPS (default 30)
- Consistency = 1.0 - CV - (rate_deviation * 0.5)

Without timestamps: Returns 0.8 if >10 frames, else 0.5

---

## Diversity Metrics

**Location:** `backend/quality/diversity.py`

### Key Philosophy

> "Perfect" demonstrations are LOW quality for learning.
> "Messy" demonstrations with recovery behaviors are HIGH quality.

### Overall Diversity Score Weights (`diversity.py:341-347`)

| Metric | Weight |
|--------|--------|
| Recovery Behaviors | 35% |
| Near-Miss Handling | 20% |
| Transition Diversity | 20% |
| Action Coverage | 15% |
| Starting State Diversity | 10% |

### Recovery Behavior Score (`diversity.py:103-166`)

Detects error-correction patterns (most important metric for learning value).

**Algorithm:**
1. Detect velocity reversals (direction changes > 110 degrees)
2. Detect anomalies (actions > 2 std from local mean)
3. Find patterns: anomaly followed by reversal within 10 frames = recovery event
4. Calculate recovery density = events per second (at 30fps)

**Scoring based on recovery density:**
| Density | Score | Interpretation |
|---------|-------|----------------|
| < 0.1 | 0.2 | Too perfect, not enough learning signal |
| 0.1 - 0.5 | 0.4 - 1.0 | Increasing value |
| 0.5 - 2.0 | 1.0 | Ideal range |
| > 2.0 | Degraded | Too chaotic |

### Transition Diversity (`diversity.py:169-204`)

Measures variety of action changes (not just repeated motions).

**Algorithm:**
1. Compute action deltas (transitions)
2. Discretize each dimension into bins
3. Count unique transition patterns
4. Diversity = unique_patterns / (total_patterns * 0.3), capped at 1.0

### Near-Miss Ratio (`diversity.py:232-271`)

Estimates situations that were close to failure but recovered.

**Algorithm:**
1. Find high-acceleration frames (>90th percentile)
2. Check if motion becomes controlled after (acceleration drops to <30% of threshold)
3. Ratio = near_misses / high_accel_events

### Action Space Coverage (`diversity.py:207-229`)

How much of the action space is utilized.

**Algorithm:**
1. Compute range used in each dimension
2. Assume typical range is [-1, 1] = 2 per dimension
3. Coverage per dim = range / 2.0
4. Overall = geometric mean of all dimensions

### Starting State Diversity (`diversity.py:274-307`)

Requires multiple episodes to compute. Measures variety of initial conditions using pairwise distances between starting actions.

---

## Visual Metrics

**Location:** `backend/quality/visual.py`

### Overall Visual Score Weights (`visual.py:269`)

| Metric | Weight |
|--------|--------|
| Blur/Sharpness | 35% |
| Resolution | 25% |
| Exposure | 25% |
| Contrast | 15% |

### Blur Score (`visual.py:99-123`)

Uses Laplacian variance for blur detection.

**Algorithm:**
1. Convert to grayscale
2. Apply Laplacian kernel: [[0,1,0],[1,-4,1],[0,1,0]]
3. Compute variance of result

**Scoring:**
| Variance | Score | Quality |
|----------|-------|---------|
| < 50 | 0 - 0.5 | Very blurry |
| 50 - 500 | 0.5 - 1.0 | Moderately sharp |
| > 500 | 1.0 | Sharp |

### Resolution Score (`visual.py:25-50`)

Based on pixel count relative to 1080p reference (1920x1080).

```python
score = min(1.0, actual_pixels / reference_pixels)
```

Very low resolution (<10% of 1080p) gets a minimum score of 0.1.

### Exposure Score (`visual.py:154-197`)

Checks histogram for proper exposure.

**Algorithm:**
1. Check shadow clipping (very dark pixels, bins 0-4)
2. Check highlight clipping (very bright pixels, bins 251-255)
3. Calculate mean brightness (ideal: 0.4-0.6)
4. Combined: 60% brightness_score + 40% (1.0 - clip_penalty)

### Contrast Score (`visual.py:200-241`)

Uses standard deviation of pixel values.

**Scoring by std:**
| Std | Score | Quality |
|-----|-------|---------|
| < 0.1 | Low | Flat/washed out |
| 0.1 - 0.25 | Medium | Acceptable |
| 0.25 - 0.4 | 1.0 | Ideal |
| > 0.4 | Degraded | Too harsh |

---

## Quality Flags (Right Sidebar Badges)

**Location:** `frontend/src/components/QualityPanel.tsx:62-86`

| Flag | Condition | Meaning |
|------|-----------|---------|
| Has Recovery | `recovery_behavior_score > 0.5` | Episode contains error-correction patterns |
| Diverse | `overall_diversity_score > 0.6` | Good variety of actions/transitions |
| Smooth | `motion_smoothness > 0.7` | Robot motion is smooth |
| Too Perfect | No recovery AND not diverse | May be less valuable for training |

---

## Source Files Reference

| File | Purpose |
|------|---------|
| `backend/quality/aggregator.py` | Combines all metrics into unified score |
| `backend/quality/temporal.py` | Motion smoothness, action consistency, completeness |
| `backend/quality/diversity.py` | Recovery behaviors, transitions, near-misses |
| `backend/quality/visual.py` | Resolution, blur, exposure, contrast |
| `backend/api/routes/quality.py` | API endpoints for quality metrics |
| `frontend/src/components/QualityPanel.tsx` | Right sidebar UI component |
| `frontend/src/types/quality.ts` | TypeScript type definitions |

---

## Notes for Review

<!-- Add your comments below -->


