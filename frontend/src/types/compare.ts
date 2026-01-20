/**
 * Types for dataset comparison
 */

export interface DatasetComparison {
  dataset_id: string;
  num_episodes: number;
  mean_score: number;
  std_score: number;
  avg_temporal: number;
  avg_diversity: number;
  avg_visual: number;
  pct_with_recovery: number;
  grade_distribution: Record<string, number>;
}

export interface ComparisonResponse {
  datasets: DatasetComparison[];
  best_overall: string;
  best_diversity: string;
  best_temporal: string;
  recommendation: string;
}
