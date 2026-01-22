"""
API routes for quality metrics.

Endpoints:
- GET /api/quality/{episode_id} - Get quality metrics for an episode
- GET /api/quality/dataset/{dataset_id} - Get quality stats for a dataset
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel

import numpy as np

from quality import (
    compute_quality_score,
    compute_dataset_quality_stats,
    QualityScore,
    # Task-level metrics
    compute_task_statistics,
    compute_episode_divergence,
    compute_task_quality_metrics,
    compute_simplified_transition_metrics,
    QualityEvent as QualityEventDataclass,
)
from loaders import HDF5Loader, LeRobotLoader, RLDSLoader
from loaders.streaming_extractor import StreamingFrameExtractor
from downloaders.manager import DATASET_REGISTRY
from cache import (
    get_cached_quality_result,
    cache_quality_result,
    get_cached_quality_events,
    cache_quality_events,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class TemporalMetricsResponse(BaseModel):
    """Temporal quality metrics."""
    motion_smoothness: float
    frame_rate_consistency: float
    action_consistency: float
    trajectory_completeness: float
    sync_score: float
    overall_temporal_score: float


class DiversityMetricsResponse(BaseModel):
    """Diversity quality metrics."""
    transition_diversity: float
    recovery_behavior_score: float
    near_miss_ratio: float
    starting_state_diversity: float
    action_space_coverage: float
    overall_diversity_score: float


class QualityResponse(BaseModel):
    """Complete quality metrics response."""
    episode_id: str
    temporal: TemporalMetricsResponse
    diversity: DiversityMetricsResponse
    overall_score: float
    quality_grade: str
    has_recovery_behaviors: bool
    is_diverse: bool
    is_smooth: bool
    is_well_synced: bool


class DatasetQualityResponse(BaseModel):
    """Dataset-level quality statistics."""
    dataset_id: str
    num_episodes: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    p10_score: float
    p90_score: float
    grade_counts: Dict[str, int]
    pct_with_recovery: float
    pct_diverse: float
    pct_smooth: float
    pct_well_synced: float
    avg_temporal: float
    avg_diversity: float


class QualityEvent(BaseModel):
    """A quality event tied to a specific frame."""
    frame: int
    event_type: str  # "gripper", "pause", "direction_change", "high_jerk", "recovery", "near_miss", "high_divergence"
    severity: str  # "info", "warning", "significant"
    score: float
    description: str
    affected_metrics: List[str]  # Which metrics this event affects
    metric_category: Optional[str] = "transition"  # "transition" or "divergence" for UI color coding


class QualityEventsResponse(BaseModel):
    """Frame-level quality events for timeline visualization."""
    episode_id: str
    total_frames: int
    events: List[QualityEvent]


# ============== Task-Level Quality Models ==============

class TaskQualityResponse(BaseModel):
    """Task-level quality metrics aggregated across episodes."""
    task_name: str
    dataset_id: str
    num_episodes: int

    # Action Divergence (Expertise Test)
    mean_divergence: float
    expertise_score: float  # 1 - mean_divergence (higher = better)
    divergence_distribution: Dict[str, int]  # 'low', 'medium', 'high' counts
    most_divergent_episodes: List[str]
    high_divergence_frame_density: float

    # Transition Diversity (Physics Test)
    pct_with_recovery: float
    mean_recovery_count: float
    mean_near_miss_count: float
    has_any_recovery_episodes: bool
    physics_coverage_score: float

    # Quality assessment
    quality_assessment: str


class EpisodeDivergenceResponse(BaseModel):
    """Per-frame divergence for timeline heat visualization."""
    episode_id: str
    task_name: str
    overall_divergence_score: float
    frame_divergences: List[float]  # Divergence at each frame (original length)
    high_divergence_frames: List[int]
    events: List[QualityEvent]
    # Per-dimension breakdown
    dimension_names: List[str] = []  # e.g. ["pos_x", "pos_y", ...]
    dimension_means: List[float] = []  # Mean divergence per dimension


def get_data_root(request: Request) -> Path:
    """Get data root from app state."""
    return request.app.state.data_root


def get_loader(dataset_id: str, data_root: Path):
    """Get appropriate loader for dataset."""
    if dataset_id not in DATASET_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    config = DATASET_REGISTRY[dataset_id]
    data_dir = data_root / dataset_id

    # Check format first
    data_format = config.get("format")
    if data_format == "lerobot":
        return LeRobotLoader(data_dir)
    elif data_format == "rlds":
        return RLDSLoader(data_dir)

    # Fall back to default loaders
    if dataset_id in ["libero", "libero_pro"]:
        return HDF5Loader(data_dir)

    raise HTTPException(status_code=501, detail=f"Loader not implemented for: {dataset_id}")


def is_streaming_dataset(dataset_id: str) -> tuple:
    """Check if dataset is streaming and return repo_id if so."""
    if dataset_id not in DATASET_REGISTRY:
        return False, None

    config = DATASET_REGISTRY[dataset_id]
    if config.get("streaming_recommended") and config.get("repo_id"):
        return True, config["repo_id"]

    return False, None


def load_streaming_episode_data(repo_id: str, episode_id: str, max_frames: int = 100) -> tuple:
    """
    Load observations from a streaming episode (MCAP file).

    For performance, limits to max_frames evenly sampled from the episode.

    Returns:
        Tuple of (observations, actions, num_frames)
        actions will be None for streaming datasets
    """
    extractor = StreamingFrameExtractor(repo_id)

    # Get total frame count
    total_frames = extractor.get_frame_count(episode_id)

    # For quality analysis, sample evenly across the episode
    # This makes analysis fast while still capturing the episode's characteristics
    sample_count = min(max_frames, total_frames)

    # Extract frames (up to max_frames from the beginning for now)
    # TODO: Implement proper even sampling across the episode
    frames, _ = extractor.extract_frames_with_count(episode_id, 0, sample_count)

    if not frames:
        return np.array([]), None, total_frames

    # Convert to numpy array of observations
    observations = np.array([frame[2] for frame in frames])  # frame[2] is the image

    return observations, None, total_frames


# NOTE: /dataset and /events routes must come BEFORE the catch-all /{episode_id:path} route
@router.get("/events/{episode_id:path}", response_model=QualityEventsResponse)
async def get_quality_events(
    episode_id: str,
    request: Request,
    dataset_id: str = Query(..., description="Dataset ID")
):
    """
    Get frame-level quality events for timeline visualization.

    Returns recovery events, anomalies, and direction changes with their
    exact frame indices so the frontend can display them on the timeline.

    Results are cached to avoid recomputation on subsequent requests.
    """
    # Check cache first
    cached = get_cached_quality_events(dataset_id, episode_id)
    if cached is not None:
        logger.info(f"Using cached quality events for {dataset_id}/{episode_id}")
        return QualityEventsResponse(
            episode_id=cached["episode_id"],
            total_frames=cached["total_frames"],
            events=[QualityEvent(**e) for e in cached["events"]]
        )

    data_root = get_data_root(request)

    # Check if this is a streaming dataset
    is_streaming, repo_id = is_streaming_dataset(dataset_id)

    if is_streaming:
        try:
            observations, actions, total_frames = load_streaming_episode_data(repo_id, episode_id)
        except Exception as e:
            logger.error(f"Failed to load streaming episode for quality: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load streaming episode: {str(e)}")
    else:
        try:
            loader = get_loader(dataset_id, data_root)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Load episode
        try:
            episode = loader.load_episode(episode_id)
            observations = episode.observations
            actions = episode.actions
            total_frames = len(observations) if observations is not None else 0
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")

    # Compute quality to get events
    quality = compute_quality_score(
        actions=actions,
        observations=observations,
    )

    # Convert quality events to API response format
    events: List[QualityEvent] = []
    for quality_event in quality.diversity.quality_events:
        events.append(QualityEvent(
            frame=quality_event.frame,
            event_type=quality_event.event_type,
            severity=quality_event.severity,
            score=quality_event.score,
            description=quality_event.description,
            affected_metrics=quality_event.affected_metrics or [],
            metric_category=getattr(quality_event, 'metric_category', 'transition') or 'transition'
        ))

    # Cache the results
    cache_data = {
        "episode_id": episode_id,
        "total_frames": total_frames,
        "events": [e.model_dump() for e in events]
    }
    cache_quality_events(dataset_id, episode_id, cache_data)
    logger.info(f"Cached quality events for {dataset_id}/{episode_id}")

    # Already sorted by frame in detect_all_quality_events

    return QualityEventsResponse(
        episode_id=episode_id,
        total_frames=total_frames,
        events=events
    )


# ============== Task-Level Quality Endpoints ==============

@router.get("/task/{dataset_id}/{task_name}", response_model=TaskQualityResponse)
async def get_task_quality(
    dataset_id: str,
    task_name: str,
    request: Request,
    limit: int = Query(50, description="Maximum episodes per task to analyze")
):
    """
    Get quality metrics aggregated at the task level.

    Computes two key metrics:
    1. Action Divergence (Expertise Test) - consistency across episodes
    2. Transition Diversity (Physics Test) - presence of recovery behaviors

    Results are more meaningful than per-episode metrics for comparing datasets.
    """
    data_root = get_data_root(request)

    try:
        loader = get_loader(dataset_id, data_root)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # List all episodes and filter by task name
    all_episodes = loader.list_episodes()
    task_episodes = [
        ep for ep in all_episodes
        if ep.task_name and ep.task_name == task_name
    ][:limit]

    if len(task_episodes) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Task '{task_name}' needs at least 2 episodes for divergence analysis, found {len(task_episodes)}"
        )

    # Load actions for all episodes
    episode_actions = []
    for ep_meta in task_episodes:
        try:
            episode = loader.load_episode(ep_meta.id)
            if episode.actions is not None and len(episode.actions) >= 5:
                episode_actions.append((ep_meta.id, episode.actions))
        except Exception as e:
            logger.warning(f"Failed to load episode {ep_meta.id}: {e}")
            continue

    if len(episode_actions) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Could not load enough valid episodes for task '{task_name}'"
        )

    # Compute task statistics
    try:
        task_stats = compute_task_statistics(task_name, episode_actions)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Compute divergence for each episode
    divergence_results = []
    for ep_id, actions in episode_actions:
        div_result = compute_episode_divergence(ep_id, actions, task_stats)
        divergence_results.append(div_result)

    # Compute transition diversity for each episode
    transition_results = []
    for ep_id, actions in episode_actions:
        trans_result = compute_simplified_transition_metrics(ep_id, actions)
        transition_results.append({
            "has_recovery": trans_result.has_recovery,
            "recovery_count": trans_result.recovery_count,
            "near_miss_count": trans_result.near_miss_count,
        })

    # Aggregate to task level
    task_metrics = compute_task_quality_metrics(
        task_name, dataset_id, divergence_results, transition_results
    )

    return TaskQualityResponse(
        task_name=task_metrics.task_name,
        dataset_id=task_metrics.dataset_id,
        num_episodes=task_metrics.num_episodes,
        mean_divergence=task_metrics.mean_divergence,
        expertise_score=task_metrics.expertise_score,
        divergence_distribution=task_metrics.divergence_distribution,
        most_divergent_episodes=task_metrics.most_divergent_episodes,
        high_divergence_frame_density=task_metrics.high_divergence_frame_density,
        pct_with_recovery=task_metrics.pct_with_recovery,
        mean_recovery_count=task_metrics.mean_recovery_count,
        mean_near_miss_count=task_metrics.mean_near_miss_count,
        has_any_recovery_episodes=task_metrics.has_any_recovery_episodes,
        physics_coverage_score=task_metrics.physics_coverage_score,
        quality_assessment=task_metrics.quality_assessment,
    )


@router.get("/task/{dataset_id}/{task_name}/divergence/{episode_id:path}", response_model=EpisodeDivergenceResponse)
async def get_episode_divergence(
    dataset_id: str,
    task_name: str,
    episode_id: str,
    request: Request,
    limit: int = Query(50, description="Maximum episodes for task statistics")
):
    """
    Get per-frame divergence data for a specific episode.

    Used by the frontend to render a divergence heat map on the timeline,
    showing where this episode diverges from the task median.
    """
    data_root = get_data_root(request)

    try:
        loader = get_loader(dataset_id, data_root)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Load the target episode
    try:
        target_episode = loader.load_episode(episode_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")

    if target_episode.actions is None or len(target_episode.actions) < 5:
        raise HTTPException(status_code=400, detail="Episode has insufficient action data")

    # Get all episodes for this task to compute statistics
    all_episodes = loader.list_episodes()
    task_episodes = [
        ep for ep in all_episodes
        if ep.task_name and ep.task_name == task_name
    ][:limit]

    if len(task_episodes) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Task '{task_name}' needs at least 2 episodes for divergence analysis"
        )

    # Load actions for task statistics
    episode_actions = []
    for ep_meta in task_episodes:
        try:
            episode = loader.load_episode(ep_meta.id)
            if episode.actions is not None and len(episode.actions) >= 5:
                episode_actions.append((ep_meta.id, episode.actions))
        except Exception:
            continue

    if len(episode_actions) < 2:
        raise HTTPException(status_code=400, detail="Could not load enough episodes for task statistics")

    # Compute task statistics
    try:
        task_stats = compute_task_statistics(task_name, episode_actions)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Compute divergence for the target episode
    div_result = compute_episode_divergence(episode_id, target_episode.actions, task_stats)

    # Convert events to API format
    events = [
        QualityEvent(
            frame=e.frame,
            event_type=e.event_type,
            severity=e.severity,
            score=e.score,
            description=e.description,
            affected_metrics=e.affected_metrics or [],
            metric_category=e.metric_category or "divergence"
        )
        for e in div_result.divergence_events
    ]

    return EpisodeDivergenceResponse(
        episode_id=episode_id,
        task_name=task_name,
        overall_divergence_score=div_result.overall_divergence_score,
        frame_divergences=div_result.original_frame_divergences.tolist(),
        high_divergence_frames=div_result.high_divergence_frames,
        events=events,
        dimension_names=div_result.dimension_names,
        dimension_means=div_result.dimension_means
    )


@router.get("/dataset/{dataset_id}", response_model=DatasetQualityResponse)
async def get_dataset_quality(
    dataset_id: str,
    request: Request,
    limit: int = Query(100, description="Maximum episodes to analyze")
):
    """
    Get quality statistics for an entire dataset.
    """
    data_root = get_data_root(request)

    try:
        loader = get_loader(dataset_id, data_root)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # List episodes
    episodes = loader.list_episodes()[:limit]

    if not episodes:
        raise HTTPException(status_code=404, detail="No episodes found in dataset")

    # Compute quality for each episode
    quality_scores: List[QualityScore] = []
    for ep_meta in episodes:
        try:
            episode = loader.load_episode(ep_meta.id)
            quality = compute_quality_score(
                actions=episode.actions,
                observations=episode.observations,
            )
            quality_scores.append(quality)
        except Exception as e:
            logger.warning(f"Failed to compute quality for {ep_meta.id}: {e}")
            continue

    if not quality_scores:
        raise HTTPException(status_code=500, detail="Failed to compute quality for any episodes")

    # Aggregate statistics
    stats = compute_dataset_quality_stats(quality_scores)

    return DatasetQualityResponse(
        dataset_id=dataset_id,
        num_episodes=stats.num_episodes,
        mean_score=stats.mean_score,
        std_score=stats.std_score,
        min_score=stats.min_score,
        max_score=stats.max_score,
        p10_score=stats.p10_score,
        p90_score=stats.p90_score,
        grade_counts=stats.grade_counts,
        pct_with_recovery=stats.pct_with_recovery,
        pct_diverse=stats.pct_diverse,
        pct_smooth=stats.pct_smooth,
        pct_well_synced=stats.pct_well_synced,
        avg_temporal=stats.avg_temporal,
        avg_diversity=stats.avg_diversity,
    )


@router.get("/{episode_id:path}", response_model=QualityResponse)
async def get_episode_quality(
    episode_id: str,
    request: Request,
    dataset_id: str = Query(..., description="Dataset ID")
):
    """
    Get quality metrics for a specific episode.

    Results are cached to avoid recomputation on subsequent requests.
    """
    # Check cache first
    cached = get_cached_quality_result(dataset_id, episode_id)
    if cached is not None:
        logger.info(f"Using cached quality result for {dataset_id}/{episode_id}")
        return QualityResponse(
            episode_id=cached["episode_id"],
            temporal=TemporalMetricsResponse(**cached["temporal"]),
            diversity=DiversityMetricsResponse(**cached["diversity"]),
            overall_score=cached["overall_score"],
            quality_grade=cached["quality_grade"],
            has_recovery_behaviors=cached["has_recovery_behaviors"],
            is_diverse=cached["is_diverse"],
            is_smooth=cached["is_smooth"],
            is_well_synced=cached["is_well_synced"],
        )

    data_root = get_data_root(request)

    # Check if this is a streaming dataset
    is_streaming, repo_id = is_streaming_dataset(dataset_id)

    if is_streaming:
        try:
            observations, actions, _ = load_streaming_episode_data(repo_id, episode_id)
            timestamps = None
        except Exception as e:
            logger.error(f"Failed to load streaming episode for quality: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load streaming episode: {str(e)}")
    else:
        try:
            loader = get_loader(dataset_id, data_root)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Load episode
        try:
            episode = loader.load_episode(episode_id)
            observations = episode.observations
            actions = episode.actions
            timestamps = episode.timestamps if hasattr(episode, 'timestamps') else None
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")

    # Compute quality
    quality = compute_quality_score(
        actions=actions,
        observations=observations,
        timestamps=timestamps,
    )

    # Build response
    response = QualityResponse(
        episode_id=episode_id,
        temporal=TemporalMetricsResponse(
            motion_smoothness=quality.temporal.motion_smoothness,
            frame_rate_consistency=quality.temporal.frame_rate_consistency,
            action_consistency=quality.temporal.action_consistency,
            trajectory_completeness=quality.temporal.trajectory_completeness,
            sync_score=quality.temporal.sync_score,
            overall_temporal_score=quality.temporal.overall_temporal_score,
        ),
        diversity=DiversityMetricsResponse(
            transition_diversity=quality.diversity.transition_diversity,
            recovery_behavior_score=quality.diversity.recovery_behavior_score,
            near_miss_ratio=quality.diversity.near_miss_ratio,
            starting_state_diversity=quality.diversity.starting_state_diversity,
            action_space_coverage=quality.diversity.action_space_coverage,
            overall_diversity_score=quality.diversity.overall_diversity_score,
        ),
        overall_score=quality.overall_score,
        quality_grade=quality.quality_grade,
        has_recovery_behaviors=quality.has_recovery_behaviors,
        is_diverse=quality.is_diverse,
        is_smooth=quality.is_smooth,
        is_well_synced=quality.is_well_synced,
    )

    # Cache the result
    cache_data = {
        "episode_id": episode_id,
        "temporal": response.temporal.model_dump(),
        "diversity": response.diversity.model_dump(),
        "overall_score": response.overall_score,
        "quality_grade": response.quality_grade,
        "has_recovery_behaviors": response.has_recovery_behaviors,
        "is_diverse": response.is_diverse,
        "is_smooth": response.is_smooth,
        "is_well_synced": response.is_well_synced,
    }
    cache_quality_result(dataset_id, episode_id, cache_data)
    logger.info(f"Cached quality result for {dataset_id}/{episode_id}")

    return response
