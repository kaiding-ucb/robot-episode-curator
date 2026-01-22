"""
Diversity Quality Metrics for robotics datasets.

These metrics assess the diversity and learning value of trajectories:
- Transition diversity (variety of physical interactions)
- Recovery behaviors (error → correction patterns)
- Near-miss detection (close calls that were corrected)
- Starting state diversity

IMPORTANT: "Perfect" demonstrations are LOW quality for learning.
"Messy" demonstrations with recovery behaviors are HIGH quality.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from scipy import signal
from scipy.spatial.distance import pdist, squareform


@dataclass
class QualityEvent:
    """A quality event detected in the trajectory."""
    frame: int
    event_type: str  # 'gripper', 'pause', 'direction_change', 'high_jerk', 'recovery', 'near_miss', 'high_divergence'
    severity: str  # 'info', 'warning', 'significant'
    score: float  # Event magnitude/importance (0-1)
    description: str
    affected_metrics: List[str] = None  # Which metrics this event affects
    metric_category: str = None  # 'transition' or 'divergence' for UI color coding

    def __post_init__(self):
        """Populate affected_metrics and metric_category from static mappings if not provided."""
        if self.affected_metrics is None:
            self.affected_metrics = EVENT_METRIC_MAP.get(self.event_type, [])
        if self.metric_category is None:
            self.metric_category = EVENT_CATEGORY_MAP.get(self.event_type, "transition")


# Static mapping of event types to the metrics they affect
EVENT_METRIC_MAP: Dict[str, List[str]] = {
    'gripper': ['recovery_behavior_score', 'transition_diversity'],
    'pause': ['recovery_behavior_score'],
    'direction_change': ['transition_diversity', 'recovery_behavior_score'],
    'speed_change': ['motion_smoothness', 'action_consistency'],
    'high_jerk': ['motion_smoothness'],
    'correction': ['recovery_behavior_score', 'near_miss_ratio'],
    'recovery': ['recovery_behavior_score', 'transition_diversity'],
    'near_miss': ['near_miss_ratio', 'recovery_behavior_score'],
    'high_divergence': ['action_divergence'],
}

# Static mapping of event types to metric categories (for UI color coding)
EVENT_CATEGORY_MAP: Dict[str, str] = {
    'gripper': 'transition',
    'pause': 'transition',
    'direction_change': 'transition',
    'speed_change': 'transition',
    'high_jerk': 'transition',
    'correction': 'transition',
    'recovery': 'transition',
    'near_miss': 'transition',
    'high_divergence': 'divergence',
}


@dataclass
class DiversityMetrics:
    """Container for diversity quality metrics."""
    transition_diversity: float  # 0-1, variety of action patterns
    recovery_behavior_score: float  # 0-1, presence of error corrections
    near_miss_ratio: float  # 0-1, close calls that were handled
    starting_state_diversity: float  # 0-1, variety of initial conditions
    action_space_coverage: float  # 0-1, how much of action space is used
    overall_diversity_score: float  # Weighted combination
    quality_events: List[QualityEvent]  # All detected quality events


def detect_gripper_events(actions: np.ndarray) -> List[QualityEvent]:
    """
    Detect gripper open/close events.

    The last dimension of actions typically controls the gripper:
    - Positive values (e.g., 1.0) = gripper open
    - Negative values (e.g., -1.0) = gripper closed

    Returns:
        List of QualityEvent for each gripper state change
    """
    if actions is None or len(actions) < 2:
        return []

    events = []
    gripper_dim = actions.shape[1] - 1  # Last dimension
    gripper_values = actions[:, gripper_dim]

    for t in range(1, len(gripper_values)):
        prev_val = gripper_values[t - 1]
        curr_val = gripper_values[t]

        # Detect sign change (gripper state change)
        if prev_val * curr_val < 0:  # Sign changed
            if curr_val < 0:
                action = "close"
                desc = "Gripper closed (grasping)"
            else:
                action = "open"
                desc = "Gripper opened (releasing)"

            events.append(QualityEvent(
                frame=t,
                event_type='gripper',
                severity='significant',
                score=1.0,
                description=desc
            ))

    return events


def detect_pause_events(actions: np.ndarray, threshold_percentile: float = 20) -> List[QualityEvent]:
    """
    Detect pause/hesitation events where motion nearly stops.

    Args:
        actions: Array of shape (T, action_dim)
        threshold_percentile: Percentile for determining "low" velocity

    Returns:
        List of QualityEvent for pause events
    """
    if actions is None or len(actions) < 5:
        return []

    events = []

    # Compute velocity magnitude (excluding gripper dimension)
    motion_dims = actions[:, :-1]  # Exclude last dim (gripper)
    velocity = np.diff(motion_dims, axis=0)
    vel_magnitude = np.linalg.norm(velocity, axis=1)

    # Dynamic threshold based on this trajectory's motion
    threshold = np.percentile(vel_magnitude, threshold_percentile)
    mean_vel = np.mean(vel_magnitude)

    if mean_vel < 1e-6:  # No motion at all
        return []

    # Find sequences of low velocity (pauses)
    in_pause = False
    pause_start = 0
    min_pause_frames = 2  # Minimum frames to count as pause

    for t in range(len(vel_magnitude)):
        is_low = vel_magnitude[t] < threshold * 0.5

        if is_low and not in_pause:
            in_pause = True
            pause_start = t
        elif not is_low and in_pause:
            pause_len = t - pause_start
            if pause_len >= min_pause_frames:
                # Score based on how much slower than average
                pause_vel = np.mean(vel_magnitude[pause_start:t])
                slowdown_ratio = 1.0 - (pause_vel / mean_vel)

                events.append(QualityEvent(
                    frame=pause_start + 1,  # +1 because velocity is diff
                    event_type='pause',
                    severity='info' if pause_len < 5 else 'significant',
                    score=float(np.clip(slowdown_ratio, 0, 1)),
                    description=f"Motion pause ({pause_len} frames)"
                ))
            in_pause = False

    return events


def detect_direction_changes(actions: np.ndarray, angle_threshold: float = 60) -> List[QualityEvent]:
    """
    Detect significant direction changes in motion.

    Args:
        actions: Array of shape (T, action_dim)
        angle_threshold: Minimum angle change in degrees to detect

    Returns:
        List of QualityEvent for direction changes
    """
    if actions is None or len(actions) < 4:
        return []

    events = []

    # Use position-related dimensions (exclude gripper)
    motion_dims = actions[:, :-1]
    velocity = np.diff(motion_dims, axis=0)

    # Smooth velocity slightly to reduce noise
    if len(velocity) > 3:
        kernel = np.array([0.25, 0.5, 0.25])
        velocity_smooth = np.array([
            np.convolve(velocity[:, d], kernel, mode='same')
            for d in range(velocity.shape[1])
        ]).T
    else:
        velocity_smooth = velocity

    cos_threshold = np.cos(np.radians(angle_threshold))

    for t in range(1, len(velocity_smooth) - 1):
        prev_vel = velocity_smooth[t - 1]
        curr_vel = velocity_smooth[t]

        prev_mag = np.linalg.norm(prev_vel)
        curr_mag = np.linalg.norm(curr_vel)

        # Only consider if both have meaningful magnitude
        min_mag = 0.01
        if prev_mag > min_mag and curr_mag > min_mag:
            cos_angle = np.dot(prev_vel, curr_vel) / (prev_mag * curr_mag + 1e-8)

            if cos_angle < cos_threshold:
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

                # Determine severity based on angle
                if angle_deg > 120:
                    severity = 'significant'
                    desc = f"Sharp direction reversal ({angle_deg:.0f}deg)"
                elif angle_deg > 90:
                    severity = 'warning'
                    desc = f"Direction reversal ({angle_deg:.0f}deg)"
                else:
                    severity = 'info'
                    desc = f"Direction change ({angle_deg:.0f}deg)"

                events.append(QualityEvent(
                    frame=t + 1,  # +1 because velocity is diff
                    event_type='direction_change',
                    severity=severity,
                    score=float(np.clip(angle_deg / 180.0, 0, 1)),
                    description=desc
                ))

    return events


def detect_speed_changes(actions: np.ndarray, change_threshold: float = 2.0) -> List[QualityEvent]:
    """
    Detect sudden speed changes (acceleration/deceleration).

    Args:
        actions: Array of shape (T, action_dim)
        change_threshold: Multiplier for speed change to be considered significant

    Returns:
        List of QualityEvent for speed changes
    """
    if actions is None or len(actions) < 5:
        return []

    events = []

    # Compute velocity magnitude (excluding gripper)
    motion_dims = actions[:, :-1]
    velocity = np.diff(motion_dims, axis=0)
    vel_magnitude = np.linalg.norm(velocity, axis=1)

    # Compute speed changes
    speed_changes = np.diff(vel_magnitude)

    # Use rolling average as baseline
    window = 3
    for t in range(window, len(speed_changes) - window):
        local_baseline = np.mean(np.abs(speed_changes[t-window:t]))
        if local_baseline < 1e-6:
            local_baseline = np.mean(np.abs(speed_changes)) + 1e-6

        change_ratio = abs(speed_changes[t]) / local_baseline

        if change_ratio > change_threshold:
            if speed_changes[t] > 0:
                desc = f"Sudden acceleration ({change_ratio:.1f}x)"
            else:
                desc = f"Sudden deceleration ({change_ratio:.1f}x)"

            severity = 'significant' if change_ratio > 4 else 'warning' if change_ratio > 3 else 'info'

            events.append(QualityEvent(
                frame=t + 2,  # +2 because of double diff
                event_type='speed_change',
                severity=severity,
                score=float(np.clip(change_ratio / 5.0, 0, 1)),
                description=desc
            ))

    return events


def detect_high_jerk(actions: np.ndarray, percentile_threshold: float = 95) -> List[QualityEvent]:
    """
    Detect high-jerk moments (third derivative) indicating jerky/corrective motion.

    Args:
        actions: Array of shape (T, action_dim)
        percentile_threshold: Percentile above which jerk is considered high

    Returns:
        List of QualityEvent for high jerk moments
    """
    if actions is None or len(actions) < 6:
        return []

    events = []

    # Compute jerk (third derivative) for motion dimensions
    motion_dims = actions[:, :-1]
    velocity = np.diff(motion_dims, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)

    jerk_magnitude = np.linalg.norm(jerk, axis=1)

    if len(jerk_magnitude) == 0:
        return []

    threshold = np.percentile(jerk_magnitude, percentile_threshold)
    mean_jerk = np.mean(jerk_magnitude)

    if mean_jerk < 1e-8:
        return []

    # Find high jerk frames (with some distance between detections)
    min_gap = 3  # Minimum frames between detections
    last_detection = -min_gap

    for t in range(len(jerk_magnitude)):
        if jerk_magnitude[t] > threshold and (t - last_detection) >= min_gap:
            jerk_ratio = jerk_magnitude[t] / mean_jerk

            events.append(QualityEvent(
                frame=t + 3,  # +3 because of triple diff
                event_type='high_jerk',
                severity='warning' if jerk_ratio > 3 else 'info',
                score=float(np.clip(jerk_ratio / 5.0, 0, 1)),
                description=f"Jerky motion ({jerk_ratio:.1f}x avg)"
            ))
            last_detection = t

    return events


def detect_all_quality_events(actions: np.ndarray) -> List[QualityEvent]:
    """
    Run all quality event detectors and merge results.

    Args:
        actions: Array of shape (T, action_dim)

    Returns:
        List of all detected QualityEvent, sorted by frame
    """
    if actions is None or len(actions) < 5:
        return []

    all_events = []

    # Run all detectors
    all_events.extend(detect_gripper_events(actions))
    all_events.extend(detect_pause_events(actions))
    all_events.extend(detect_direction_changes(actions))
    # Removed: speed_change events (too noisy for timeline)
    all_events.extend(detect_high_jerk(actions))
    # Recovery and near-miss events (important for physics test)
    all_events.extend(detect_recovery_events(actions))
    all_events.extend(detect_near_miss_events(actions))

    # Sort by frame
    all_events.sort(key=lambda e: e.frame)

    # Deduplicate events that are too close together (within 2 frames)
    # Keep the most significant one
    severity_order = {'significant': 3, 'warning': 2, 'info': 1}

    if len(all_events) <= 1:
        return all_events

    deduplicated = []
    i = 0
    while i < len(all_events):
        # Collect all events within 2 frames
        cluster = [all_events[i]]
        j = i + 1
        while j < len(all_events) and all_events[j].frame - all_events[i].frame <= 2:
            cluster.append(all_events[j])
            j += 1

        # Keep the most significant event from the cluster
        cluster.sort(key=lambda e: (-severity_order.get(e.severity, 0), -e.score))
        deduplicated.append(cluster[0])

        i = j

    return deduplicated


def compute_recovery_score(actions: np.ndarray) -> Tuple[float, List[QualityEvent]]:
    """
    Compute recovery behavior score based on detected quality events.

    Args:
        actions: Array of shape (T, action_dim)

    Returns:
        Tuple of (recovery_score, list of quality events)
    """
    if actions is None or len(actions) < 10:
        return 0.2, []

    events = detect_all_quality_events(actions)

    if len(events) == 0:
        return 0.2, []  # No events detected = too smooth

    # Count significant events
    significant_count = sum(1 for e in events if e.severity in ['significant', 'warning'])
    total_count = len(events)

    # Event density (events per 30 frames ~= 1 second at typical fps)
    event_density = total_count / (len(actions) / 30.0)

    # Score based on event density and significance
    # Ideal: 0.5-3 events per second
    if event_density < 0.2:
        base_score = 0.3 + 0.3 * (event_density / 0.2)
    elif event_density < 3.0:
        base_score = 0.6 + 0.4 * min(1.0, event_density / 2.0)
    else:
        base_score = max(0.5, 1.0 - (event_density - 3.0) * 0.1)

    # Bonus for having significant events (corrections, gripper actions)
    significance_bonus = min(0.2, significant_count * 0.05)

    score = min(1.0, base_score + significance_bonus)

    return score, events


def compute_transition_diversity(actions: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute diversity of action transitions.

    Measures how many different types of action changes occur,
    not just the same motion repeated.

    Args:
        actions: Array of shape (T, action_dim)
        num_bins: Number of bins for discretizing action space

    Returns:
        Diversity score between 0 and 1
    """
    if actions is None or len(actions) < 5:
        return 0.5

    # Compute action deltas (transitions)
    deltas = np.diff(actions, axis=0)

    # Discretize each dimension
    transition_patterns = []
    for delta in deltas:
        # Bin each dimension into integers
        bins = np.linspace(-1, 1, num_bins)
        pattern = tuple(int(np.digitize(d, bins)) for d in delta)
        transition_patterns.append(pattern)

    # Count unique patterns
    unique_patterns = len(set(transition_patterns))
    total_patterns = len(transition_patterns)

    # Diversity is ratio of unique to total, with diminishing returns
    diversity = min(1.0, unique_patterns / (total_patterns * 0.3))

    return float(diversity)


def compute_action_space_coverage(actions: np.ndarray) -> float:
    """
    Compute how much of the action space is utilized.

    Args:
        actions: Array of shape (T, action_dim)

    Returns:
        Coverage score between 0 and 1
    """
    if actions is None or len(actions) < 2:
        return 0.0

    # Compute range used in each dimension
    action_range = np.max(actions, axis=0) - np.min(actions, axis=0)

    # Assume typical action range is [-1, 1] = 2 for each dim
    coverage_per_dim = np.clip(action_range / 2.0, 0, 1)

    # Overall coverage (geometric mean to penalize dimensions not used)
    coverage = np.exp(np.mean(np.log(coverage_per_dim + 0.01)))

    return float(np.clip(coverage, 0, 1))


def compute_near_miss_ratio(actions: np.ndarray, observations: Optional[np.ndarray] = None) -> float:
    """
    Estimate near-miss events - situations that were close to failure but recovered.

    These are extremely valuable for learning robust policies.

    Args:
        actions: Array of shape (T, action_dim)
        observations: Optional array of observations for context

    Returns:
        Near-miss ratio between 0 and 1
    """
    if actions is None or len(actions) < 10:
        return 0.0

    # Detect high-acceleration moments (potential near-misses)
    velocity = np.diff(actions, axis=0)
    acceleration = np.diff(velocity, axis=0)
    accel_magnitude = np.linalg.norm(acceleration, axis=1)

    # High acceleration followed by controlled motion suggests near-miss
    accel_threshold = np.percentile(accel_magnitude, 90)
    high_accel_frames = np.where(accel_magnitude > accel_threshold)[0]

    near_misses = 0
    for frame in high_accel_frames:
        # Check if motion becomes controlled after the high acceleration
        if frame + 5 < len(accel_magnitude):
            post_accel = np.mean(accel_magnitude[frame + 1:frame + 5])
            if post_accel < accel_threshold * 0.3:  # Significant slowdown
                near_misses += 1

    # Ratio of near-misses to total high-accel events
    if len(high_accel_frames) > 0:
        ratio = near_misses / len(high_accel_frames)
    else:
        ratio = 0.0

    return float(ratio)


def compute_starting_state_diversity(episodes_actions: List[np.ndarray]) -> float:
    """
    Compute diversity of starting states across episodes.

    Args:
        episodes_actions: List of action arrays from multiple episodes

    Returns:
        Diversity score between 0 and 1
    """
    if not episodes_actions or len(episodes_actions) < 2:
        return 0.5

    # Get starting actions from each episode
    starts = []
    for actions in episodes_actions:
        if actions is not None and len(actions) > 0:
            starts.append(actions[0])

    if len(starts) < 2:
        return 0.5

    starts = np.array(starts)

    # Compute pairwise distances
    if len(starts) > 1:
        distances = pdist(starts)
        mean_distance = np.mean(distances)
        # Normalize by expected range
        diversity = np.clip(mean_distance / 0.5, 0, 1)
    else:
        diversity = 0.0

    return float(diversity)


def compute_diversity_metrics(actions: np.ndarray,
                             observations: Optional[np.ndarray] = None,
                             other_episodes: Optional[List[np.ndarray]] = None) -> DiversityMetrics:
    """
    Compute all diversity quality metrics for an episode.

    REMEMBER: High diversity and recovery behaviors = HIGH quality
    Perfect, smooth executions = LOW quality for learning

    Args:
        actions: Array of shape (T, action_dim) containing actions
        observations: Optional observations array
        other_episodes: Optional list of other episode actions for comparison

    Returns:
        DiversityMetrics dataclass with all metrics
    """
    transition_div = compute_transition_diversity(actions)
    recovery_score, quality_events = compute_recovery_score(actions)
    near_miss = compute_near_miss_ratio(actions, observations)
    action_coverage = compute_action_space_coverage(actions)

    # Starting state diversity requires multiple episodes
    if other_episodes:
        all_episodes = [actions] + other_episodes
        starting_div = compute_starting_state_diversity(all_episodes)
    else:
        starting_div = 0.5  # Unknown without comparison

    # Weighted overall score
    # Recovery behaviors are weighted highest per user requirements
    overall = (
        0.20 * transition_div +
        0.35 * recovery_score +  # Most important!
        0.20 * near_miss +
        0.15 * action_coverage +
        0.10 * starting_div
    )

    return DiversityMetrics(
        transition_diversity=transition_div,
        recovery_behavior_score=recovery_score,
        near_miss_ratio=near_miss,
        starting_state_diversity=starting_div,
        action_space_coverage=action_coverage,
        overall_diversity_score=overall,
        quality_events=quality_events
    )


def detect_recovery_events(actions: np.ndarray) -> List[QualityEvent]:
    """
    Detect recovery behavior events: high acceleration followed by controlled motion.

    Pattern: Error → Correction → Return to normal path

    Args:
        actions: Array of shape (T, action_dim)

    Returns:
        List of QualityEvent for recovery behaviors
    """
    if actions is None or len(actions) < 10:
        return []

    events = []

    # Compute acceleration magnitude
    velocity = np.diff(actions[:, :-1], axis=0)  # Exclude gripper
    acceleration = np.diff(velocity, axis=0)
    accel_magnitude = np.linalg.norm(acceleration, axis=1)

    if len(accel_magnitude) < 5:
        return []

    # Find high-acceleration moments
    threshold = np.percentile(accel_magnitude, 90)
    mean_accel = np.mean(accel_magnitude)

    if mean_accel < 1e-8:
        return []

    min_gap = 5  # Minimum frames between recovery events
    last_event_frame = -min_gap

    for frame in range(len(accel_magnitude) - 5):
        if accel_magnitude[frame] > threshold:
            # Check if followed by controlled motion (recovery pattern)
            post_accel = accel_magnitude[frame + 1:frame + 5]
            if np.mean(post_accel) < threshold * 0.3:  # Significant slowdown
                if frame - last_event_frame >= min_gap:
                    recovery_strength = accel_magnitude[frame] / mean_accel
                    events.append(QualityEvent(
                        frame=frame + 2,  # Offset for derivatives
                        event_type='recovery',
                        severity='significant' if recovery_strength > 3 else 'warning',
                        score=float(min(1.0, recovery_strength / 5.0)),
                        description=f"Recovery behavior ({recovery_strength:.1f}x avg acceleration)",
                        affected_metrics=['recovery_behavior_score', 'transition_diversity'],
                        metric_category='transition'
                    ))
                    last_event_frame = frame

    return events


def detect_near_miss_events(actions: np.ndarray) -> List[QualityEvent]:
    """
    Detect near-miss events: approaching failure boundary then correcting.

    Patterns:
    - Gripper closes partially then re-opens (slip recovery)
    - Rapid direction reversal (collision avoidance)

    Args:
        actions: Array of shape (T, action_dim)

    Returns:
        List of QualityEvent for near-miss situations
    """
    if actions is None or len(actions) < 10:
        return []

    events = []
    gripper_idx = actions.shape[1] - 1
    gripper = actions[:, gripper_idx]

    min_gap = 5
    last_event_frame = -min_gap

    # Detect partial gripper operations (slip recovery)
    for i in range(2, len(gripper) - 3):
        # Pattern: open → closing → re-open (or vice versa)
        # Indicates a slip or failed grasp attempt
        prev_change = gripper[i] - gripper[i - 2]
        next_change = gripper[i + 2] - gripper[i]

        # Direction reversal in gripper
        if prev_change * next_change < -0.1:  # Significant reversal
            if i - last_event_frame >= min_gap:
                events.append(QualityEvent(
                    frame=i,
                    event_type='near_miss',
                    severity='warning',
                    score=0.8,
                    description="Gripper correction (potential slip recovery)",
                    affected_metrics=['near_miss_ratio', 'recovery_behavior_score'],
                    metric_category='transition'
                ))
                last_event_frame = i

    return events


@dataclass
class TransitionDiversityResult:
    """Simplified transition diversity result for task-level aggregation."""
    episode_id: str
    has_recovery: bool
    recovery_count: int
    near_miss_count: int
    correction_count: int
    transition_events: List[QualityEvent]
    overall_transition_score: float


def compute_simplified_transition_metrics(
    episode_id: str,
    actions: np.ndarray
) -> TransitionDiversityResult:
    """
    Compute simplified transition diversity metrics for task-level aggregation.

    Focuses on two key signals:
    1. Recovery behaviors (error → correction)
    2. Near-miss handling (slip recovery, collision avoidance)

    Args:
        episode_id: ID of the episode
        actions: Array of shape (T, action_dim)

    Returns:
        TransitionDiversityResult with simplified metrics
    """
    if actions is None or len(actions) < 10:
        return TransitionDiversityResult(
            episode_id=episode_id,
            has_recovery=False,
            recovery_count=0,
            near_miss_count=0,
            correction_count=0,
            transition_events=[],
            overall_transition_score=0.2
        )

    # Detect recovery events
    recovery_events = detect_recovery_events(actions)

    # Detect near-miss events
    near_miss_events = detect_near_miss_events(actions)

    # Get direction changes that indicate corrections (>90 deg)
    direction_events = [
        e for e in detect_direction_changes(actions, angle_threshold=90)
        if e.severity in ['warning', 'significant']
    ]
    # Re-label as corrections with proper category
    correction_events = []
    for e in direction_events:
        correction_events.append(QualityEvent(
            frame=e.frame,
            event_type='correction',
            severity=e.severity,
            score=e.score,
            description=e.description.replace("Direction", "Correction"),
            affected_metrics=['recovery_behavior_score', 'near_miss_ratio'],
            metric_category='transition'
        ))

    # Merge all events
    all_events = recovery_events + near_miss_events + correction_events
    all_events.sort(key=lambda e: e.frame)

    # Counts
    recovery_count = len(recovery_events)
    near_miss_count = len(near_miss_events)
    correction_count = len(correction_events)

    has_recovery = recovery_count > 0 or near_miss_count > 0

    # Score: presence of recovery behaviors is HIGH quality
    # Base score + bonus for having events
    event_density = (recovery_count + near_miss_count * 0.5 + correction_count * 0.3) / max(1, len(actions) / 30.0)
    overall_score = min(1.0, 0.3 + event_density * 0.35)

    # Boost if we have actual recovery events
    if recovery_count > 0:
        overall_score = min(1.0, overall_score + 0.2)

    return TransitionDiversityResult(
        episode_id=episode_id,
        has_recovery=has_recovery,
        recovery_count=recovery_count,
        near_miss_count=near_miss_count,
        correction_count=correction_count,
        transition_events=all_events,
        overall_transition_score=overall_score
    )
