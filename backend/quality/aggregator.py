"""
Quality Aggregator - combines all quality metrics into a unified score.

Weighting based on user requirements:
- Temporal Quality: 55% (includes sync score for action-observation latency)
- Diversity (including recovery behaviors): 45%

Note: Visual quality removed as it's not useful for robotics dataset evaluation.
"""
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

from .temporal import TemporalMetrics, compute_temporal_metrics
from .diversity import DiversityMetrics, compute_diversity_metrics


@dataclass
class QualityScore:
    """Unified quality score for an episode."""
    # Component scores
    temporal: TemporalMetrics
    diversity: DiversityMetrics

    # Overall scores
    overall_score: float  # 0-1 unified score
    quality_grade: str  # A, B, C, D, F

    # Flags
    has_recovery_behaviors: bool
    is_diverse: bool
    is_smooth: bool
    is_well_synced: bool  # Action-observation synchronization

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'temporal': asdict(self.temporal),
            'diversity': asdict(self.diversity),
            'overall_score': self.overall_score,
            'quality_grade': self.quality_grade,
            'has_recovery_behaviors': self.has_recovery_behaviors,
            'is_diverse': self.is_diverse,
            'is_smooth': self.is_smooth,
            'is_well_synced': self.is_well_synced,
        }


def grade_from_score(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 0.9:
        return 'A'
    elif score >= 0.8:
        return 'B'
    elif score >= 0.7:
        return 'C'
    elif score >= 0.6:
        return 'D'
    else:
        return 'F'


def compute_quality_score(
    actions: Optional[np.ndarray] = None,
    observations: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    success_label: Optional[bool] = None,
    other_episodes: Optional[List[np.ndarray]] = None,
    weights: Optional[Dict[str, float]] = None
) -> QualityScore:
    """
    Compute unified quality score for an episode.

    Args:
        actions: Array of shape (T, action_dim) containing actions
        observations: Array of shape (T, H, W, C) containing images
        timestamps: Optional array of timestamps
        success_label: Optional success label
        other_episodes: Optional list of other episode actions for diversity comparison
        weights: Optional custom weights for components

    Returns:
        QualityScore with all metrics and overall score
    """
    # Default weights: only temporal and diversity matter for robotics datasets
    if weights is None:
        weights = {
            'temporal': 0.55,
            'diversity': 0.45,
        }

    # Compute component metrics
    temporal = compute_temporal_metrics(actions, timestamps, observations, success_label)
    diversity = compute_diversity_metrics(actions, observations, other_episodes)

    # Compute overall score (only temporal and diversity)
    overall = (
        weights['temporal'] * temporal.overall_temporal_score +
        weights['diversity'] * diversity.overall_diversity_score
    )

    # Determine flags
    has_recovery = diversity.recovery_behavior_score > 0.5
    is_diverse = diversity.overall_diversity_score > 0.6
    is_smooth = temporal.motion_smoothness > 0.7
    is_well_synced = temporal.sync_score > 0.7

    return QualityScore(
        temporal=temporal,
        diversity=diversity,
        overall_score=float(overall),
        quality_grade=grade_from_score(overall),
        has_recovery_behaviors=has_recovery,
        is_diverse=is_diverse,
        is_smooth=is_smooth,
        is_well_synced=is_well_synced
    )


@dataclass
class DatasetQualityStats:
    """Quality statistics for an entire dataset."""
    num_episodes: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    p10_score: float  # 10th percentile (worst 10%)
    p90_score: float  # 90th percentile (best 10%)

    # Grade distribution
    grade_counts: Dict[str, int]

    # Flag percentages
    pct_with_recovery: float
    pct_diverse: float
    pct_smooth: float
    pct_well_synced: float

    # Component averages
    avg_temporal: float
    avg_diversity: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def compute_dataset_quality_stats(quality_scores: List[QualityScore]) -> DatasetQualityStats:
    """
    Compute aggregate quality statistics for a dataset.

    Args:
        quality_scores: List of QualityScore objects for each episode

    Returns:
        DatasetQualityStats with aggregate metrics
    """
    if not quality_scores:
        return DatasetQualityStats(
            num_episodes=0,
            mean_score=0.0,
            std_score=0.0,
            min_score=0.0,
            max_score=0.0,
            p10_score=0.0,
            p90_score=0.0,
            grade_counts={'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0},
            pct_with_recovery=0.0,
            pct_diverse=0.0,
            pct_smooth=0.0,
            pct_well_synced=0.0,
            avg_temporal=0.0,
            avg_diversity=0.0,
        )

    scores = [q.overall_score for q in quality_scores]

    grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    for q in quality_scores:
        grade_counts[q.quality_grade] += 1

    return DatasetQualityStats(
        num_episodes=len(quality_scores),
        mean_score=float(np.mean(scores)),
        std_score=float(np.std(scores)),
        min_score=float(np.min(scores)),
        max_score=float(np.max(scores)),
        p10_score=float(np.percentile(scores, 10)),
        p90_score=float(np.percentile(scores, 90)),
        grade_counts=grade_counts,
        pct_with_recovery=sum(q.has_recovery_behaviors for q in quality_scores) / len(quality_scores),
        pct_diverse=sum(q.is_diverse for q in quality_scores) / len(quality_scores),
        pct_smooth=sum(q.is_smooth for q in quality_scores) / len(quality_scores),
        pct_well_synced=sum(q.is_well_synced for q in quality_scores) / len(quality_scores),
        avg_temporal=float(np.mean([q.temporal.overall_temporal_score for q in quality_scores])),
        avg_diversity=float(np.mean([q.diversity.overall_diversity_score for q in quality_scores])),
    )
