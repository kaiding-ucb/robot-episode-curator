/**
 * Quality metrics types for the frontend
 */

export interface TemporalMetrics {
  motion_smoothness: number;
  frame_rate_consistency: number;
  action_consistency: number;
  trajectory_completeness: number;
  sync_score: number;
  overall_temporal_score: number;
}

export interface DiversityMetrics {
  transition_diversity: number;
  recovery_behavior_score: number;
  near_miss_ratio: number;
  starting_state_diversity: number;
  action_space_coverage: number;
  overall_diversity_score: number;
}

export interface QualityScore {
  episode_id: string;
  temporal: TemporalMetrics;
  diversity: DiversityMetrics;
  overall_score: number;
  quality_grade: string;
  has_recovery_behaviors: boolean;
  is_diverse: boolean;
  is_smooth: boolean;
  is_well_synced: boolean;
}

export interface DatasetQualityStats {
  dataset_id: string;
  num_episodes: number;
  mean_score: number;
  std_score: number;
  min_score: number;
  max_score: number;
  p10_score: number;
  p90_score: number;
  grade_counts: Record<string, number>;
  pct_with_recovery: number;
  pct_diverse: number;
  pct_smooth: number;
  pct_well_synced: number;
  avg_temporal: number;
  avg_diversity: number;
}

export interface QualityEvent {
  frame: number;
  event_type: "gripper" | "pause" | "direction_change" | "speed_change" | "high_jerk" | "correction";
  severity: "info" | "warning" | "significant";
  score: number;
  description: string;
  affected_metrics: string[];  // Which metrics this event affects
}

export interface QualityEventsResponse {
  episode_id: string;
  total_frames: number;
  events: QualityEvent[];
}
