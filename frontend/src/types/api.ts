/**
 * API types for the Data Viewer frontend
 */

// === MODALITY TYPES ===
export type Modality = "rgb" | "depth" | "imu" | "actions" | "states";

export interface ModalityConfig {
  topic?: string;
  type?: "image" | "timeseries" | "vector";
  colormap?: string;
  key?: string;
}

export interface Dataset {
  id: string;
  name: string;
  type: "teleop" | "video";
  description?: string;
  episode_count?: number;
  size_mb?: number;
  modalities?: Modality[];
  modality_config?: Record<string, ModalityConfig>;
  has_tasks?: boolean;
}

export interface EpisodeMetadata {
  id: string;
  task_name?: string;
  num_frames: number;
  duration_sec?: number;
}

export interface Task {
  name: string;
  episode_count?: number;
  description?: string;
}

export interface TaskListResponse {
  tasks: Task[];
  total_tasks: number;
  source: "huggingface_api" | "episode_scan" | "config";
}

export interface Episode {
  id: string;
  task_name?: string;
  observations?: string; // base64 encoded or URL
  actions?: number[][];
  metadata?: Record<string, unknown>;
}

export interface Frame {
  index: number;
  timestamp?: number;
  image: string; // base64 encoded
  action?: number[];
}

export interface DownloadStatus {
  dataset_id: string;
  status: "not_downloaded" | "downloading" | "ready" | "error";
  size_bytes: number;
  size_mb: number;
  error?: string;
}

export interface DiskSpace {
  total_gb: number;
  used_gb: number;
  available_gb: number;
}

// === STREAMING OPTIMIZATION TYPES ===
export type ImageResolution = "low" | "medium" | "high" | "original";

export interface StreamingOptions {
  resolution?: ImageResolution;
  quality?: number; // 10-100
  stream?: "rgb" | "depth"; // Which stream to fetch
}

// === PROBE & ADD DATASET TYPES ===
export interface ProbeResponse {
  repo_id: string;
  name: string;
  format_detected?: string;
  has_tasks: boolean;
  modalities: Modality[];
  modality_config?: Record<string, ModalityConfig>;
  sample_files: string[];
  error?: string;
}

export interface AddDatasetResponse {
  dataset_id: string;
  name: string;
  success: boolean;
  error?: string;
}

// === IMU DATA TYPES ===
export interface IMUData {
  timestamps: number[];
  accel_x: number[];
  accel_y: number[];
  accel_z: number[];
  gyro_x: number[];
  gyro_y: number[];
  gyro_z: number[];
  error?: string;
}

// === ACTIONS DATA TYPES ===
export interface ActionsData {
  timestamps: number[];
  actions: number[][];  // 2D array: [frame][dimension]
  dimension_labels: string[] | null;
  error?: string;
}

// === CACHE MANAGEMENT TYPES ===
export interface CachedEpisode {
  dataset_id: string;
  episode_id: string;
  size_mb: number;
  cached_at: number; // Unix timestamp
  batch_count: number;
}

export interface CacheStats {
  total_size_mb: number;
  episode_count: number;
  batch_count: number;
}

export interface DeleteCacheResponse {
  bytes_freed: number;
  success: boolean;
}

// === MODALITY TYPES ===
export type Modality = "rgb" | "depth" | "imu" | "tactile" | "actions" | "states";

// === DATASET OVERVIEW TYPES ===
export interface DatasetOverview {
  repo_id: string;
  name: string;
  description?: string;
  readme_summary?: string;
  license?: string;
  dataset_tags: string[];

  // From HF repo info
  size_bytes?: number;
  gated: boolean;
  downloads_last_month?: number;

  // Parsed from README or detected
  environment?: string;
  perspective?: string;
  format_detected?: string;

  // Scale and modalities
  modalities: string[];
  estimated_hours?: number;
  estimated_clips?: number;
  task_count?: number;

  // Cache metadata
  cached_at?: string;
}
