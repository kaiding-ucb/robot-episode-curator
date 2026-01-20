/**
 * React hooks for quality metrics API
 */
import { useState, useEffect } from "react";
import type { QualityScore, DatasetQualityStats, QualityEventsResponse } from "@/types/quality";

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
