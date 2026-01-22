"""
Action Divergence Metrics for robotics datasets.

Measures how consistently experts perform the same task across episodes.
High divergence = crowdsourced without style guide = hard for model to learn.
Low divergence = consistent technique = easier to learn median path.

This is the "Expertise Test" for dataset quality.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from .diversity import QualityEvent


@dataclass
class TaskActionStatistics:
    """Pre-computed statistics for a task's action trajectories."""
    task_name: str
    num_episodes: int
    reference_length: int  # Normalized trajectory length (default 100 frames)
    median_trajectory: np.ndarray  # (reference_length, action_dim)
    std_trajectory: np.ndarray  # Per-frame standard deviation
    percentile_25: np.ndarray  # Lower bound for "normal" variance
    percentile_75: np.ndarray  # Upper bound for "normal" variance
    episode_ids: List[str] = field(default_factory=list)


@dataclass
class ActionDivergenceResult:
    """Action divergence for a single episode relative to task median."""
    episode_id: str
    task_name: str
    overall_divergence_score: float  # 0-1, higher = more divergent
    frame_divergences: np.ndarray  # Per-frame divergence values (normalized length)
    original_frame_divergences: np.ndarray  # Per-frame divergence (original length)
    high_divergence_frames: List[int]  # Original frame indices exceeding threshold
    divergence_events: List[QualityEvent] = field(default_factory=list)
    # Per-dimension breakdown
    dimension_names: List[str] = field(default_factory=list)  # e.g. ["pos_x", "pos_y", ...]
    dimension_means: List[float] = field(default_factory=list)  # Mean divergence per dimension


@dataclass
class TaskQualityMetrics:
    """Aggregated quality metrics for a task."""
    task_name: str
    dataset_id: str
    num_episodes: int

    # Action Divergence (Expertise Test)
    mean_divergence: float
    expertise_score: float  # 1 - mean_divergence (higher = better)
    divergence_distribution: Dict[str, int]  # 'low', 'medium', 'high' counts
    most_divergent_episodes: List[str]
    high_divergence_frame_density: float  # Avg high-div frames per episode

    # Transition Diversity (Physics Test)
    pct_with_recovery: float
    mean_recovery_count: float
    mean_near_miss_count: float
    has_any_recovery_episodes: bool
    physics_coverage_score: float

    # Quality assessment
    quality_assessment: str


# Divergence thresholds (in units of standard deviation)
DIVERGENCE_THRESHOLDS = {
    "low": 0.5,      # < 0.5 std from median = very consistent
    "medium": 1.5,   # 0.5-1.5 std = normal variance
    "high": 2.0,     # > 2.0 std = high divergence (flagged)
}


def normalize_trajectory(actions: np.ndarray, target_length: int = 100) -> np.ndarray:
    """
    Normalize trajectory to fixed length using linear interpolation.

    Args:
        actions: (T, action_dim) original actions
        target_length: Desired number of frames (default 100)

    Returns:
        (target_length, action_dim) normalized trajectory
    """
    if actions is None or len(actions) < 2:
        raise ValueError("Actions must have at least 2 frames")

    original_length = len(actions)
    if original_length == target_length:
        return actions.copy()

    # Create interpolation points
    original_indices = np.linspace(0, 1, original_length)
    target_indices = np.linspace(0, 1, target_length)

    # Interpolate each action dimension
    normalized = np.zeros((target_length, actions.shape[1]))
    for dim in range(actions.shape[1]):
        normalized[:, dim] = np.interp(target_indices, original_indices, actions[:, dim])

    return normalized


def compute_task_statistics(
    task_name: str,
    episode_actions: List[Tuple[str, np.ndarray]],
    reference_length: int = 100
) -> TaskActionStatistics:
    """
    Pre-compute median and variance statistics for a task.

    Algorithm:
    1. Normalize all trajectories to reference_length
    2. Compute per-frame median (robust to outliers)
    3. Compute per-frame standard deviation and percentiles

    Args:
        task_name: Name of the task
        episode_actions: List of (episode_id, actions) tuples
        reference_length: Target normalized length

    Returns:
        TaskActionStatistics with computed statistics
    """
    if len(episode_actions) < 2:
        raise ValueError("Need at least 2 episodes to compute task statistics")

    # Normalize all trajectories
    normalized = []
    episode_ids = []

    for ep_id, actions in episode_actions:
        if actions is not None and len(actions) >= 5:
            try:
                norm_traj = normalize_trajectory(actions, reference_length)
                normalized.append(norm_traj)
                episode_ids.append(ep_id)
            except ValueError:
                continue

    if len(normalized) < 2:
        raise ValueError(f"Insufficient valid trajectories for task {task_name}")

    # Stack into (num_episodes, reference_length, action_dim)
    trajectories = np.stack(normalized)

    return TaskActionStatistics(
        task_name=task_name,
        num_episodes=len(normalized),
        reference_length=reference_length,
        median_trajectory=np.median(trajectories, axis=0),
        std_trajectory=np.std(trajectories, axis=0),
        percentile_25=np.percentile(trajectories, 25, axis=0),
        percentile_75=np.percentile(trajectories, 75, axis=0),
        episode_ids=episode_ids
    )


def compute_episode_divergence(
    episode_id: str,
    actions: np.ndarray,
    task_stats: TaskActionStatistics,
    divergence_threshold_multiplier: float = 2.0
) -> ActionDivergenceResult:
    """
    Compute how much an episode diverges from the task median.

    Args:
        episode_id: ID of the episode
        actions: Episode actions array (T, action_dim)
        task_stats: Pre-computed task statistics
        divergence_threshold_multiplier: Frames with divergence > multiplier * std are "high"

    Returns:
        ActionDivergenceResult with per-frame divergence and events
    """
    # Standard dimension names for robotics actions (7 DOF typical)
    DEFAULT_DIMENSION_NAMES = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "gripper"]
    READABLE_DIMENSION_NAMES = {
        "pos_x": "X position",
        "pos_y": "Y position",
        "pos_z": "Z position",
        "rot_x": "Roll",
        "rot_y": "Pitch",
        "rot_z": "Yaw",
        "gripper": "Gripper",
    }

    if actions is None or len(actions) < 5:
        return ActionDivergenceResult(
            episode_id=episode_id,
            task_name=task_stats.task_name,
            overall_divergence_score=0.5,
            frame_divergences=np.zeros(task_stats.reference_length),
            original_frame_divergences=np.zeros(max(1, len(actions) if actions is not None else 1)),
            high_divergence_frames=[],
            divergence_events=[],
            dimension_names=[],
            dimension_means=[]
        )

    original_length = len(actions)

    # Normalize to match task reference length
    normalized = normalize_trajectory(actions, task_stats.reference_length)

    # Compute per-frame divergence (z-score like)
    diff = np.abs(normalized - task_stats.median_trajectory)
    std_safe = np.maximum(task_stats.std_trajectory, 1e-6)  # Avoid division by zero
    z_scores = diff / std_safe

    # Per-dimension divergence (mean across frames for each dimension)
    num_dims = actions.shape[1]
    dimension_names = DEFAULT_DIMENSION_NAMES[:num_dims] if num_dims <= len(DEFAULT_DIMENSION_NAMES) else [f"dim_{i}" for i in range(num_dims)]
    dimension_z_scores = z_scores  # Shape: (reference_length, num_dims)
    dimension_means = [float(np.mean(dimension_z_scores[:, d])) for d in range(num_dims)]

    # Aggregate across action dimensions (L2 norm of z-scores)
    frame_divergences = np.sqrt(np.sum(z_scores ** 2, axis=1)) / np.sqrt(actions.shape[1])

    # Map divergences back to original frame indices
    original_frame_divergences = np.interp(
        np.linspace(0, 1, original_length),
        np.linspace(0, 1, task_stats.reference_length),
        frame_divergences
    )

    # Identify high-divergence frames in normalized space
    threshold = divergence_threshold_multiplier
    high_div_mask = frame_divergences > threshold
    high_divergence_norm_frames = np.where(high_div_mask)[0]

    # Map back to original frame indices
    high_divergence_frames = [
        int(f * original_length / task_stats.reference_length)
        for f in high_divergence_norm_frames
    ]
    # Remove duplicates and sort
    high_divergence_frames = sorted(set(high_divergence_frames))

    # Create divergence events for timeline with dimension-specific descriptions
    events = []
    min_gap = 3  # Minimum frames between events to avoid clutter
    last_event_frame = -min_gap

    for orig_frame in high_divergence_frames:
        if orig_frame - last_event_frame >= min_gap:
            div_value = original_frame_divergences[orig_frame]
            severity = "significant" if div_value > threshold * 1.5 else "warning"

            # Find dominant dimension at this frame (map to normalized frame)
            norm_frame = min(int(orig_frame * task_stats.reference_length / original_length), task_stats.reference_length - 1)
            frame_z_scores = dimension_z_scores[norm_frame]
            dominant_dim_idx = int(np.argmax(frame_z_scores))
            dominant_dim_name = dimension_names[dominant_dim_idx] if dominant_dim_idx < len(dimension_names) else f"dim_{dominant_dim_idx}"
            dominant_z_score = frame_z_scores[dominant_dim_idx]

            # Generate human-readable description
            readable_name = READABLE_DIMENSION_NAMES.get(dominant_dim_name, dominant_dim_name)
            description = f"{readable_name} {dominant_z_score:.1f} std from median"

            events.append(QualityEvent(
                frame=orig_frame,
                event_type="high_divergence",
                severity=severity,
                score=float(min(1.0, div_value / (threshold * 2))),
                description=description,
                affected_metrics=[dominant_dim_name]
            ))
            last_event_frame = orig_frame

    # Overall score: 0 = identical to median, 1 = very divergent
    overall_divergence = float(np.mean(frame_divergences) / threshold)

    return ActionDivergenceResult(
        episode_id=episode_id,
        task_name=task_stats.task_name,
        overall_divergence_score=min(1.0, overall_divergence),
        frame_divergences=frame_divergences,
        original_frame_divergences=original_frame_divergences,
        high_divergence_frames=high_divergence_frames,
        divergence_events=events,
        dimension_names=dimension_names,
        dimension_means=dimension_means
    )


def categorize_divergence(score: float) -> str:
    """Categorize divergence score into low/medium/high."""
    if score < DIVERGENCE_THRESHOLDS["low"]:
        return "low"
    elif score < DIVERGENCE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "high"


def compute_task_quality_metrics(
    task_name: str,
    dataset_id: str,
    divergence_results: List[ActionDivergenceResult],
    transition_results: List[dict]  # List of {has_recovery, recovery_count, near_miss_count}
) -> TaskQualityMetrics:
    """
    Aggregate individual episode results into task-level metrics.

    Args:
        task_name: Name of the task
        dataset_id: Dataset identifier
        divergence_results: List of ActionDivergenceResult for each episode
        transition_results: List of transition diversity info per episode

    Returns:
        TaskQualityMetrics with aggregated scores
    """
    num_episodes = len(divergence_results)

    if num_episodes == 0:
        return TaskQualityMetrics(
            task_name=task_name,
            dataset_id=dataset_id,
            num_episodes=0,
            mean_divergence=0.5,
            expertise_score=0.5,
            divergence_distribution={"low": 0, "medium": 0, "high": 0},
            most_divergent_episodes=[],
            high_divergence_frame_density=0.0,
            pct_with_recovery=0.0,
            mean_recovery_count=0.0,
            mean_near_miss_count=0.0,
            has_any_recovery_episodes=False,
            physics_coverage_score=0.0,
            quality_assessment="Insufficient data"
        )

    # Compute divergence statistics
    divergence_scores = [r.overall_divergence_score for r in divergence_results]
    mean_divergence = float(np.mean(divergence_scores))
    expertise_score = max(0.0, 1.0 - mean_divergence)

    # Categorize each episode
    categories = [categorize_divergence(s) for s in divergence_scores]
    divergence_distribution = {
        "low": categories.count("low"),
        "medium": categories.count("medium"),
        "high": categories.count("high")
    }

    # Find most divergent episodes
    sorted_by_div = sorted(
        zip(divergence_scores, divergence_results),
        key=lambda x: x[0],
        reverse=True
    )
    most_divergent_episodes = [r.episode_id for _, r in sorted_by_div[:3]]

    # High divergence frame density
    total_high_div_frames = sum(len(r.high_divergence_frames) for r in divergence_results)
    high_divergence_frame_density = total_high_div_frames / num_episodes

    # Transition diversity statistics
    episodes_with_recovery = sum(1 for t in transition_results if t.get("has_recovery", False))
    pct_with_recovery = episodes_with_recovery / num_episodes if num_episodes > 0 else 0.0

    recovery_counts = [t.get("recovery_count", 0) for t in transition_results]
    near_miss_counts = [t.get("near_miss_count", 0) for t in transition_results]

    mean_recovery_count = float(np.mean(recovery_counts)) if recovery_counts else 0.0
    mean_near_miss_count = float(np.mean(near_miss_counts)) if near_miss_counts else 0.0

    has_any_recovery_episodes = episodes_with_recovery > 0

    # Physics coverage score (higher if has recovery behaviors)
    physics_coverage_score = min(1.0, 0.3 + pct_with_recovery * 0.5 + min(0.2, mean_recovery_count * 0.1))

    # Quality assessment
    if expertise_score > 0.7 and has_any_recovery_episodes:
        quality_assessment = "Excellent - consistent technique with recovery behaviors"
    elif expertise_score > 0.7:
        quality_assessment = "Good consistency but lacks recovery behaviors"
    elif has_any_recovery_episodes:
        quality_assessment = "Good physics coverage but high action variance"
    else:
        quality_assessment = "Needs review - high variance and no recovery behaviors"

    return TaskQualityMetrics(
        task_name=task_name,
        dataset_id=dataset_id,
        num_episodes=num_episodes,
        mean_divergence=mean_divergence,
        expertise_score=expertise_score,
        divergence_distribution=divergence_distribution,
        most_divergent_episodes=most_divergent_episodes,
        high_divergence_frame_density=high_divergence_frame_density,
        pct_with_recovery=pct_with_recovery,
        mean_recovery_count=mean_recovery_count,
        mean_near_miss_count=mean_near_miss_count,
        has_any_recovery_episodes=has_any_recovery_episodes,
        physics_coverage_score=physics_coverage_score,
        quality_assessment=quality_assessment
    )
