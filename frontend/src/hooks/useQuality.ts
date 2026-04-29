/**
 * React hooks for quality metrics API
 *
 * Includes:
 * - Episode-level quality metrics
 * - Dataset-level quality stats
 * - Task-level quality metrics (Expertise + Physics tests)
 * - Episode divergence for timeline heat
 */
import { useState, useEffect } from "react";
import type {
  QualityScore,
  DatasetQualityStats,
  QualityEventsResponse,
  TaskQualityMetrics,
  EpisodeDivergence,
} from "@/types/quality";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

/**
 * Fetch quality metrics for an episode
 */
export function useEpisodeQuality(datasetId: string | null, episodeId: string | null) {
  const [quality, setQuality] = useState<QualityScore | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId || !episodeId) {
      setQuality(null);
      return;
    }

    async function fetchQuality() {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `${API_BASE}/quality/${episodeId}?dataset_id=${datasetId}`
        );
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        setQuality(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to fetch quality");
      } finally {
        setLoading(false);
      }
    }

    fetchQuality();
  }, [datasetId, episodeId]);

  return { quality, loading, error };
}

/**
 * Fetch quality stats for a dataset
 */
export function useDatasetQuality(datasetId: string | null, limit: number = 100) {
  const [stats, setStats] = useState<DatasetQualityStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId) {
      setStats(null);
      return;
    }

    async function fetchStats() {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `${API_BASE}/quality/dataset/${datasetId}?limit=${limit}`
        );
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        setStats(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to fetch dataset quality");
      } finally {
        setLoading(false);
      }
    }

    fetchStats();
  }, [datasetId, limit]);

  return { stats, loading, error };
}

/**
 * Fetch quality events for timeline visualization
 */
export function useQualityEvents(datasetId: string | null, episodeId: string | null) {
  const [events, setEvents] = useState<QualityEventsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId || !episodeId) {
      setEvents(null);
      return;
    }

    async function fetchEvents() {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `${API_BASE}/quality/events/${episodeId}?dataset_id=${datasetId}`
        );
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        setEvents(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to fetch quality events");
      } finally {
        setLoading(false);
      }
    }

    fetchEvents();
  }, [datasetId, episodeId]);

  return { events, loading, error };
}

// ============== Task-Level Quality Hooks ==============

/**
 * Fetch task-level quality metrics
 *
 * Computes two key metrics:
 * 1. Action Divergence (Expertise Test) - consistency across episodes
 * 2. Transition Diversity (Physics Test) - presence of recovery behaviors
 */
export function useTaskQuality(
  datasetId: string | null,
  taskName: string | null,
  limit: number = 50
) {
  const [metrics, setMetrics] = useState<TaskQualityMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId || !taskName) {
      setMetrics(null);
      return;
    }
    // Capture non-null values; the closure below cannot rely on the outer
    // narrowing carrying into the async scope.
    const did = datasetId;
    const tn = taskName;

    async function fetchTaskQuality() {
      setLoading(true);
      setError(null);
      try {
        const encodedTaskName = encodeURIComponent(tn);
        const res = await fetch(
          `${API_BASE}/quality/task/${did}/${encodedTaskName}?limit=${limit}`
        );
        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          throw new Error(errorData.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        setMetrics(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to fetch task quality");
      } finally {
        setLoading(false);
      }
    }

    fetchTaskQuality();
  }, [datasetId, taskName, limit]);

  return { metrics, loading, error };
}

/**
 * Fetch per-frame divergence data for an episode
 *
 * Used to render the divergence heat map on the timeline,
 * showing where this episode diverges from the task median.
 */
export function useEpisodeDivergence(
  datasetId: string | null,
  taskName: string | null,
  episodeId: string | null,
  limit: number = 50
) {
  const [divergence, setDivergence] = useState<EpisodeDivergence | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId || !taskName || !episodeId) {
      setDivergence(null);
      return;
    }
    const did = datasetId;
    const tn = taskName;
    const eid = episodeId;

    async function fetchDivergence() {
      setLoading(true);
      setError(null);
      try {
        const encodedTaskName = encodeURIComponent(tn);
        const encodedEpisodeId = encodeURIComponent(eid);
        const res = await fetch(
          `${API_BASE}/quality/task/${did}/${encodedTaskName}/divergence/${encodedEpisodeId}?limit=${limit}`
        );
        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          throw new Error(errorData.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        setDivergence(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to fetch episode divergence");
      } finally {
        setLoading(false);
      }
    }

    fetchDivergence();
  }, [datasetId, taskName, episodeId, limit]);

  return { divergence, loading, error };
}
