/**
 * API types for the Data Viewer frontend
 */

export interface Dataset {
  id: string;
  name: string;
  type: "teleop" | "video";
  description?: string;
  episode_count?: number;
  size_mb?: number;
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
