"""
API routes for cross-dataset comparison.

Endpoints:
- GET /api/compare - Compare quality across multiple datasets
"""
import logging
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Query, Request
from loaders.hdf5_loader import HDF5Loader
from pydantic import BaseModel
from quality import compute_dataset_quality_stats, compute_quality_score

logger = logging.getLogger(__name__)

router = APIRouter()


class DatasetComparison(BaseModel):
    """Comparison metrics for a single dataset."""
    dataset_id: str
    num_episodes: int
    mean_score: float
    std_score: float
    avg_temporal: float
    avg_diversity: float
    avg_visual: float
    pct_with_recovery: float
    grade_distribution: Dict[str, int]


class ComparisonResponse(BaseModel):
    """Response for dataset comparison."""
    datasets: List[DatasetComparison]
    best_overall: str
    best_diversity: str
    best_temporal: str
    recommendation: str


def get_data_root(request: Request) -> Path:
    """Get data root from app state."""
    return request.app.state.data_root


def get_loader(dataset_id: str, data_root: Path):
    """Get appropriate loader for dataset."""
    if dataset_id == "libero":
        return HDF5Loader(data_root / "libero")
    else:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")


@router.get("", response_model=ComparisonResponse)
async def compare_datasets(
    request: Request,
    dataset_ids: List[str] = Query(..., description="List of dataset IDs to compare"),
    limit_per_dataset: int = Query(50, description="Max episodes to analyze per dataset")
):
    """
    Compare quality metrics across multiple datasets.

    Returns comparison data and recommendations for which dataset
    has the best learning value based on diversity and recovery behaviors.
    """
    data_root = get_data_root(request)

    if len(dataset_ids) < 1:
        raise HTTPException(status_code=400, detail="At least one dataset ID required")

    comparisons: List[DatasetComparison] = []

    for dataset_id in dataset_ids:
        try:
            loader = get_loader(dataset_id, data_root)
            episodes = loader.list_episodes()[:limit_per_dataset]

            if not episodes:
                logger.warning(f"No episodes found for {dataset_id}")
                continue

            # Compute quality for each episode
            quality_scores = []
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
                continue

            # Aggregate stats
            stats = compute_dataset_quality_stats(quality_scores)

            comparisons.append(DatasetComparison(
                dataset_id=dataset_id,
                num_episodes=stats.num_episodes,
                mean_score=stats.mean_score,
                std_score=stats.std_score,
                avg_temporal=stats.avg_temporal,
                avg_diversity=stats.avg_diversity,
                avg_visual=stats.avg_visual,
                pct_with_recovery=stats.pct_with_recovery,
                grade_distribution=stats.grade_counts,
            ))

        except HTTPException:
            logger.warning(f"Dataset not found: {dataset_id}")
            continue
        except Exception as e:
            logger.error(f"Error processing {dataset_id}: {e}")
            continue

    if not comparisons:
        raise HTTPException(status_code=404, detail="No datasets could be analyzed")

    # Find best datasets
    best_overall = max(comparisons, key=lambda x: x.mean_score)
    best_diversity = max(comparisons, key=lambda x: x.avg_diversity)
    best_temporal = max(comparisons, key=lambda x: x.avg_temporal)

    # Generate recommendation
    # Prioritize diversity and recovery behaviors for learning value
    learning_scores = [
        (c.dataset_id, c.avg_diversity * 0.5 + c.pct_with_recovery * 0.3 + c.avg_temporal * 0.2)
        for c in comparisons
    ]
    best_for_learning = max(learning_scores, key=lambda x: x[1])

    if best_for_learning[0] == best_overall.dataset_id:
        recommendation = f"{best_for_learning[0]} is recommended: best overall quality and learning value."
    else:
        recommendation = (
            f"{best_for_learning[0]} is recommended for learning value "
            f"(high diversity and recovery behaviors). "
            f"{best_overall.dataset_id} has highest overall score."
        )

    return ComparisonResponse(
        datasets=comparisons,
        best_overall=best_overall.dataset_id,
        best_diversity=best_diversity.dataset_id,
        best_temporal=best_temporal.dataset_id,
        recommendation=recommendation,
    )
