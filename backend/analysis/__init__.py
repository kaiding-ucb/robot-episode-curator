"""Phase-aware anomaly detection for robotics teleoperation episodes.

Entry point: analyze_task(episodes, task_name) -> PhaseAwareResult
"""
from .phase_aware import AnomalyReason, Cluster, Phase, PhaseAwareResult, analyze_task

__all__ = ["analyze_task", "PhaseAwareResult", "Phase", "AnomalyReason", "Cluster"]
