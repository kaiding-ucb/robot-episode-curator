/**
 * Hook for dataset analysis — frame counts, signal comparison, and capabilities.
 */
import { useState, useCallback, useRef } from "react";
import type {
  FrameCountDistribution,
  EpisodeSignalData,
  SignalAnalysisState,
  DatasetCapabilities,
} from "@/types/analysis";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

/**
 * Fetch dataset analysis capabilities (format, supported analyses).
 */
export function useDatasetCapabilities() {
  const [capabilities, setCapabilities] = useState<DatasetCapabilities | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchCapabilities = useCallback(async (datasetId: string) => {
    setLoading(true);
    try {
      const res = await fetch(
        `${API_BASE}/datasets/${datasetId}/analysis/capabilities`
      );
      if (!res.ok) {
        setCapabilities(null);
        return;
      }
      const result = await res.json();
      if (result.error) {
        setCapabilities(null);
      } else {
        setCapabilities(result);
      }
    } catch {
      setCapabilities(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setCapabilities(null);
  }, []);

  return { capabilities, loading, fetchCapabilities, reset };
}

/**
 * Fetch frame count distribution for a task (zero download).
 */
export function useFrameCounts() {
  const [data, setData] = useState<FrameCountDistribution | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchFrameCounts = useCallback(
    async (datasetId: string, taskName: string) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `${API_BASE}/datasets/${datasetId}/analysis/frame-counts?task_name=${encodeURIComponent(taskName)}`
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const result = await res.json();
        if (result.error) {
          setError(result.error);
          setData(null);
        } else {
          setData(result);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to fetch frame counts");
        setData(null);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const reset = useCallback(() => {
    setData(null);
    setLoading(false);
    setError(null);
  }, []);

  return { data, loading, error, fetchFrameCounts, reset };
}

/**
 * Stream signal comparison data via SSE.
 */
export function useSignalComparison() {
  const [state, setState] = useState<SignalAnalysisState>({
    episodes: new Map(),
    phase: "idle",
    progress: { current: 0, total: 0, currentEpisode: "" },
    error: null,
    noSignalsReason: null,
  });
  const eventSourceRef = useRef<EventSource | null>(null);

  const startAnalysis = useCallback(
    (datasetId: string, taskName: string, maxEpisodes: number = 5) => {
      // Close any existing connection
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      setState({
        episodes: new Map(),
        phase: "processing",
        progress: { current: 0, total: 0, currentEpisode: "" },
        error: null,
        noSignalsReason: null,
      });

      const url = `${API_BASE}/datasets/${datasetId}/analysis/signals?task_name=${encodeURIComponent(taskName)}&max_episodes=${maxEpisodes}`;
      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "total") {
            setState((prev) => ({
              ...prev,
              progress: { ...prev.progress, total: data.total_episodes },
            }));
          } else if (data.type === "progress") {
            setState((prev) => ({
              ...prev,
              progress: {
                current: data.episode_index,
                total: data.total,
                currentEpisode: data.episode_id,
              },
            }));
          } else if (data.type === "episode_data") {
            const episodeData: EpisodeSignalData = {
              episode_id: data.episode_id,
              episode_index: data.episode_index,
              actions: data.actions,
              imu: data.imu,
              total_frames: data.total_frames ?? null,
            };
            setState((prev) => {
              const newEpisodes = new Map(prev.episodes);
              newEpisodes.set(data.episode_id, episodeData);
              return {
                ...prev,
                episodes: newEpisodes,
                progress: {
                  ...prev.progress,
                  current: data.episode_index + 1,
                },
              };
            });
          } else if (data.type === "done") {
            setState((prev) => ({ ...prev, phase: "complete" }));
            eventSource.close();
          } else if (data.type === "no_signals") {
            setState((prev) => ({
              ...prev,
              phase: "no_signals",
              noSignalsReason: data.reason || "Signal comparison is not available for this dataset.",
            }));
            eventSource.close();
          } else if (data.type === "error") {
            setState((prev) => ({
              ...prev,
              phase: "error",
              error: data.message,
            }));
            eventSource.close();
          }
        } catch {
          // Ignore malformed events
        }
      };

      eventSource.onerror = () => {
        setState((prev) => {
          // If we already got data, treat as complete
          if (prev.episodes.size > 0) {
            return { ...prev, phase: "complete" };
          }
          // If no_signals was already set, keep it
          if (prev.phase === "no_signals") {
            return prev;
          }
          return { ...prev, phase: "error", error: "Connection lost" };
        });
        eventSource.close();
      };
    },
    []
  );

  const cancelAnalysis = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setState((prev) => ({ ...prev, phase: "idle" }));
  }, []);

  const reset = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setState({
      episodes: new Map(),
      phase: "idle",
      progress: { current: 0, total: 0, currentEpisode: "" },
      error: null,
      noSignalsReason: null,
    });
  }, []);

  return { state, startAnalysis, cancelAnalysis, reset };
}
