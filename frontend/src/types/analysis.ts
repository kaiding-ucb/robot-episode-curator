/**
 * Types for Dataset Analysis feature
 */

export interface EpisodeFrameCount {
  episode_id: string;
  estimated_frames: number;
  size_bytes: number;
  file_name: string;
}

export interface FrameCountDistribution {
  task_name: string;
  episodes: EpisodeFrameCount[];
  total_episodes: number;
  mean_frames: number;
  std_frames: number;
  min_frames: number;
  max_frames: number;
  outlier_episode_ids: string[];
  source: string;
  source_note?: string;
  error?: string;
}

export interface EpisodeActions {
  timestamps: number[];
  actions: number[][];
  dimension_labels: string[] | null;
  error?: string;
}

export interface EpisodeIMU {
  timestamps: number[];
  accel_x: number[];
  accel_y: number[];
  accel_z: number[];
  gyro_x: number[];
  gyro_y: number[];
  gyro_z: number[];
  error?: string;
}

export interface EpisodeSignalData {
  episode_id: string;
  episode_index: number;
  actions: EpisodeActions;
  imu: EpisodeIMU;
  total_frames?: number | null;
  global_episode_index?: number | null;
  signal_stride?: number;
  raw_action_count?: number | null;
  first_frame?: string | null;
}

export interface EpisodeStub {
  episode_id: string;
  episode_index: number;
}

export interface SignalAnalysisState {
  episodes: Map<string, EpisodeSignalData>;
  phase: "idle" | "processing" | "complete" | "error" | "no_signals";
  progress: { current: number; total: number; currentEpisode: string };
  error: string | null;
  noSignalsReason: string | null;
  firstFrames: Map<string, string>;
  knownEpisodes: EpisodeStub[];
}

export interface DatasetCapabilities {
  format: string;
  has_actions: boolean;
  has_imu: boolean;
  supports_frame_counts: boolean;
  supports_signal_comparison: boolean;
  signal_comparison_note: string;
  supports_edge_frames?: boolean;
  edge_frames_note?: string;
}

export interface EdgeFrameItem {
  episode_id: string;
  episode_index: number;
  total_frames: number | null;
  image_b64: string | null;
  error?: string | null;
}

export type EdgeFramePosition = "start" | "end";

export interface EdgeFramesState {
  position: EdgeFramePosition;
  framesByPos: Record<EdgeFramePosition, Map<number, EdgeFrameItem>>;
  totalByPos: Record<EdgeFramePosition, number>;
  totalForTaskByPos: Record<EdgeFramePosition, number>;
  loadedByPos: Record<EdgeFramePosition, number>;
  phaseByPos: Record<EdgeFramePosition, "idle" | "loading" | "complete" | "error">;
  errorByPos: Record<EdgeFramePosition, string | null>;
}
