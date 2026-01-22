"""
Quality metrics module for robotics datasets.

Provides temporal, diversity, visual quality analysis
with emphasis on recovery behaviors and diversity.

Task-level metrics:
- Action Divergence (Expertise Test)
- Transition Diversity (Physics Test)
"""
from .temporal import TemporalMetrics, compute_temporal_metrics
from .diversity import (
    DiversityMetrics,
    QualityEvent,
    TransitionDiversityResult,
    compute_diversity_metrics,
    compute_simplified_transition_metrics,
    detect_recovery_events,
    detect_near_miss_events,
)
from .visual import VisualMetrics, compute_visual_metrics
from .aggregator import (
    QualityScore,
    DatasetQualityStats,
    compute_quality_score,
    compute_dataset_quality_stats
)
from .divergence import (
    TaskActionStatistics,
    ActionDivergenceResult,
    TaskQualityMetrics,
    normalize_trajectory,
    compute_task_statistics,
    compute_episode_divergence,
    compute_task_quality_metrics,
)

__all__ = [
    # Temporal
    'TemporalMetrics',
    'compute_temporal_metrics',
    # Diversity
    'DiversityMetrics',
    'QualityEvent',
    'TransitionDiversityResult',
    'compute_diversity_metrics',
    'compute_simplified_transition_metrics',
    'detect_recovery_events',
    'detect_near_miss_events',
    # Visual
    'VisualMetrics',
    'compute_visual_metrics',
    # Aggregator
    'QualityScore',
    'DatasetQualityStats',
    'compute_quality_score',
    'compute_dataset_quality_stats',
    # Task-level divergence
    'TaskActionStatistics',
    'ActionDivergenceResult',
    'TaskQualityMetrics',
    'normalize_trajectory',
    'compute_task_statistics',
    'compute_episode_divergence',
    'compute_task_quality_metrics',
]
