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
 * Uses RAF-batched state updates to reduce re-renders during streaming.
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

  // Buffers for batching SSE events — accumulate without triggering renders
  const episodesBufferRef = useRef<Map<string, EpisodeSignalData>>(new Map());
  const progressRef = useRef<{ current: number; total: number; currentEpisode: string }>({ current: 0, total: 0, currentEpisode: "" });
  const rafIdRef = useRef<number | null>(null);

  // Flush buffered state to React in a single render
  const flushBuffer = useCallback(() => {
    rafIdRef.current = null;
    setState((prev) => ({
      ...prev,
      episodes: new Map(episodesBufferRef.current),
      progress: { ...progressRef.current },
    }));
  }, []);

  // Schedule a RAF flush (coalesces multiple events into one render)
  const scheduleFlush = useCallback(() => {
    if (rafIdRef.current != null) {
      cancelAnimationFrame(rafIdRef.current);
    }
    rafIdRef.current = requestAnimationFrame(flushBuffer);
  }, [flushBuffer]);

  // Cancel any pending RAF
  const cancelRaf = useCallback(() => {
    if (rafIdRef.current != null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
  }, []);

  const startAnalysis = useCallback(
    (datasetId: string, taskName: string, maxEpisodes: number = 5) => {
      // Close any existing connection
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      cancelRaf();

      // Reset buffers
      episodesBufferRef.current = new Map();
      progressRef.current = { current: 0, total: 0, currentEpisode: "" };

      setState({
        episodes: new Map(),
        phase: "processing",
        progress: { current: 0, total: 0, currentEpisode: "" },
        error: null,
        noSignalsReason: null,
      });

      const url = `${API_BASE}/datasets/${datasetId}/analysis/signals?task_name=${encodeURIComponent(taskName)}&max_episodes=${maxEpisodes}&resolution=200`;
      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "total") {
            progressRef.current = { ...progressRef.current, total: data.total_episodes };
            scheduleFlush();
          } else if (data.type === "progress") {
            progressRef.current = {
              current: data.episode_index,
              total: data.total,
              currentEpisode: data.episode_id,
            };
            scheduleFlush();
          } else if (data.type === "episode_data") {
            const episodeData: EpisodeSignalData = {
              episode_id: data.episode_id,
              episode_index: data.episode_index,
              actions: data.actions,
              imu: data.imu,
              total_frames: data.total_frames ?? null,
              global_episode_index: data.global_episode_index ?? null,
            };
            episodesBufferRef.current.set(data.episode_id, episodeData);
            progressRef.current = {
              ...progressRef.current,
              current: data.episode_index + 1,
            };
            scheduleFlush();
          } else if (data.type === "done") {
            cancelRaf();
            // Synchronous flush — ensure all buffered data is committed
            setState((prev) => ({
              ...prev,
              episodes: new Map(episodesBufferRef.current),
              progress: { ...progressRef.current },
              phase: "complete",
            }));
            eventSource.close();
          } else if (data.type === "no_signals") {
            cancelRaf();
            setState((prev) => ({
              ...prev,
              phase: "no_signals",
              noSignalsReason: data.reason || "Signal comparison is not available for this dataset.",
            }));
            eventSource.close();
          } else if (data.type === "error") {
            cancelRaf();
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
        cancelRaf();
        setState((prev) => {
          // If we already got buffered data, flush and treat as complete
          if (episodesBufferRef.current.size > 0) {
            return {
              ...prev,
              episodes: new Map(episodesBufferRef.current),
              progress: { ...progressRef.current },
              phase: "complete",
            };
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
    [cancelRaf, scheduleFlush]
  );

  const cancelAnalysis = useCallback(() => {
    cancelRaf();
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setState((prev) => ({ ...prev, phase: "idle" }));
  }, [cancelRaf]);

  const reset = useCallback(() => {
    cancelRaf();
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    episodesBufferRef.current = new Map();
    progressRef.current = { current: 0, total: 0, currentEpisode: "" };
    setState({
      episodes: new Map(),
      phase: "idle",
      progress: { current: 0, total: 0, currentEpisode: "" },
      error: null,
      noSignalsReason: null,
    });
  }, [cancelRaf]);

  return { state, startAnalysis, cancelAnalysis, reset };
}
