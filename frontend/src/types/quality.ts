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
  event_type:
    | "gripper"
    | "pause"
    | "direction_change"
    | "speed_change"
    | "high_jerk"
    | "correction"
    | "recovery"
    | "near_miss"
    | "high_divergence";
  severity: "info" | "warning" | "significant";
  score: number;
  description: string;
  affected_metrics: string[]; // Which metrics this event affects
  metric_category?: "transition" | "divergence"; // For UI color coding
}

export interface QualityEventsResponse {
  episode_id: string;
  total_frames: number;
  events: QualityEvent[];
}

// ============== Task-Level Quality Types ==============

export interface TaskQualityMetrics {
  task_name: string;
  dataset_id: string;
  num_episodes: number;

  // Action Divergence (Expertise Test)
  mean_divergence: number;
  expertise_score: number; // 1 - mean_divergence (higher = better)
  divergence_distribution: Record<"low" | "medium" | "high", number>;
  most_divergent_episodes: string[];
  high_divergence_frame_density: number;

  // Transition Diversity (Physics Test)
  pct_with_recovery: number;
  mean_recovery_count: number;
  mean_near_miss_count: number;
  has_any_recovery_episodes: boolean;
  physics_coverage_score: number;

  // Quality assessment
  quality_assessment: string;
}

export interface EpisodeDivergence {
  episode_id: string;
  task_name: string;
  overall_divergence_score: number;
  frame_divergences: number[]; // Per-frame divergence values
  high_divergence_frames: number[];
  events: QualityEvent[];
  // Per-dimension breakdown
  dimension_names: string[]; // e.g. ["pos_x", "pos_y", ...]
  dimension_means: number[]; // Mean divergence per dimension
}

// Human-readable dimension names for UI
export const READABLE_DIMENSION_NAMES: Record<string, string> = {
  pos_x: "Position X",
  pos_y: "Position Y",
  pos_z: "Position Z",
  rot_x: "Roll",
  rot_y: "Pitch",
  rot_z: "Yaw",
  gripper: "Gripper",
};
