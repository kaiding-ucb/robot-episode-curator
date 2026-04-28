/**
 * Hook for dataset analysis — frame counts, signal comparison, and capabilities.
 */
import { useState, useCallback, useEffect, useRef } from "react";
import type {
  FrameCountDistribution,
  EpisodeSignalData,
  EpisodeStub,
  SignalAnalysisState,
  DatasetCapabilities,
  EdgeFrameItem,
  EdgeFramePosition,
  EdgeFramesState,
} from "@/types/analysis";
import type { MetaSummaryResponse } from "@/types/api";

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
 * Fetch LeRobot meta/info.json + tasks.parquet for the Summary tab.
 */
export function useMetaSummary() {
  const [data, setData] = useState<MetaSummaryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSummary = useCallback(async (datasetId: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/datasets/${datasetId}/meta-summary`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const result: MetaSummaryResponse = await res.json();
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch meta summary");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setLoading(false);
    setError(null);
  }, []);

  return { data, loading, error, fetchSummary, reset };
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
    firstFrames: new Map(),
    knownEpisodes: [],
  });
  const eventSourceRef = useRef<EventSource | null>(null);

  // Buffers for batching SSE events — accumulate without triggering renders
  const episodesBufferRef = useRef<Map<string, EpisodeSignalData>>(new Map());
  const progressRef = useRef<{ current: number; total: number; currentEpisode: string }>({ current: 0, total: 0, currentEpisode: "" });
  const firstFramesBufferRef = useRef<Map<string, string>>(new Map());
  const knownEpisodesRef = useRef<EpisodeStub[]>([]);
  const rafIdRef = useRef<number | null>(null);

  // Flush buffered state to React in a single render
  const flushBuffer = useCallback(() => {
    rafIdRef.current = null;
    setState((prev) => ({
      ...prev,
      episodes: new Map(episodesBufferRef.current),
      progress: { ...progressRef.current },
      firstFrames: new Map(firstFramesBufferRef.current),
      knownEpisodes: [...knownEpisodesRef.current],
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
      firstFramesBufferRef.current = new Map();
      knownEpisodesRef.current = [];

      setState({
        episodes: new Map(),
        phase: "processing",
        progress: { current: 0, total: 0, currentEpisode: "" },
        error: null,
        noSignalsReason: null,
        firstFrames: new Map(),
        knownEpisodes: [],
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
          } else if (data.type === "episode_list") {
            knownEpisodesRef.current = data.episodes as EpisodeStub[];
            scheduleFlush();
          } else if (data.type === "first_frame") {
            if (data.first_frame) {
              firstFramesBufferRef.current.set(data.episode_id, data.first_frame);
              scheduleFlush();
            }
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
              signal_stride: data.signal_stride,
              raw_action_count: data.raw_action_count ?? null,
              first_frame: data.first_frame ?? null,
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
              firstFrames: new Map(firstFramesBufferRef.current),
              knownEpisodes: [...knownEpisodesRef.current],
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
              firstFrames: new Map(firstFramesBufferRef.current),
              knownEpisodes: [...knownEpisodesRef.current],
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
    firstFramesBufferRef.current = new Map();
    knownEpisodesRef.current = [];
    setState({
      episodes: new Map(),
      phase: "idle",
      progress: { current: 0, total: 0, currentEpisode: "" },
      error: null,
      noSignalsReason: null,
      firstFrames: new Map(),
      knownEpisodes: [],
    });
  }, [cancelRaf]);

  return { state, startAnalysis, cancelAnalysis, reset };
}

/**
 * Stream starting/ending frame thumbnails for episodes in a task.
 * Caches both start & end results in-memory; toggling between them after both
 * positions have loaded is instant (no second fetch).
 */
const EDGE_FRAMES_LIMIT = 50;

function emptyEdgeState(): EdgeFramesState {
  return {
    position: "start",
    framesByPos: { start: new Map(), end: new Map() },
    totalByPos: { start: 0, end: 0 },
    totalForTaskByPos: { start: 0, end: 0 },
    loadedByPos: { start: 0, end: 0 },
    phaseByPos: { start: "idle", end: "idle" },
    errorByPos: { start: null, end: null },
  };
}

export function useEdgeFrames(datasetId: string | null, taskName: string | null) {
  const [state, setState] = useState<EdgeFramesState>(() => emptyEdgeState());
  const sourcesRef = useRef<Record<EdgeFramePosition, EventSource | null>>({
    start: null,
    end: null,
  });

  const closeAll = useCallback(() => {
    (Object.keys(sourcesRef.current) as EdgeFramePosition[]).forEach((p) => {
      sourcesRef.current[p]?.close();
      sourcesRef.current[p] = null;
    });
  }, []);

  const fetchPosition = useCallback(
    (pos: EdgeFramePosition) => {
      if (!datasetId || !taskName) return;
      sourcesRef.current[pos]?.close();
      sourcesRef.current[pos] = null;

      setState((prev) => ({
        ...prev,
        framesByPos: { ...prev.framesByPos, [pos]: new Map() },
        loadedByPos: { ...prev.loadedByPos, [pos]: 0 },
        phaseByPos: { ...prev.phaseByPos, [pos]: "loading" },
        errorByPos: { ...prev.errorByPos, [pos]: null },
      }));

      const url =
        `${API_BASE}/datasets/${datasetId}/tasks/${encodeURIComponent(taskName)}` +
        `/edge-frames/stream?position=${pos}&limit=${EDGE_FRAMES_LIMIT}`;
      const es = new EventSource(url);
      sourcesRef.current[pos] = es;

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "total") {
            setState((prev) => ({
              ...prev,
              totalByPos: { ...prev.totalByPos, [pos]: data.total },
              totalForTaskByPos: { ...prev.totalForTaskByPos, [pos]: data.total_for_task },
            }));
          } else if (data.type === "episode_meta") {
            setState((prev) => {
              const next = new Map(prev.framesByPos[pos]);
              const existing = next.get(data.episode_index);
              const item: EdgeFrameItem = {
                episode_id: data.episode_id,
                episode_index: data.episode_index,
                total_frames: data.total_frames,
                image_b64: existing?.image_b64 ?? null,
                error: existing?.error ?? null,
              };
              next.set(data.episode_index, item);
              return { ...prev, framesByPos: { ...prev.framesByPos, [pos]: next } };
            });
          } else if (data.type === "frame") {
            setState((prev) => {
              const next = new Map(prev.framesByPos[pos]);
              const existing = next.get(data.episode_index);
              next.set(data.episode_index, {
                episode_id: data.episode_id,
                episode_index: data.episode_index,
                total_frames: existing?.total_frames ?? null,
                image_b64: data.image_b64,
                error: null,
              });
              return {
                ...prev,
                framesByPos: { ...prev.framesByPos, [pos]: next },
                loadedByPos: { ...prev.loadedByPos, [pos]: prev.loadedByPos[pos] + 1 },
              };
            });
          } else if (data.type === "error") {
            setState((prev) => {
              const next = new Map(prev.framesByPos[pos]);
              const existing = next.get(data.episode_index);
              next.set(data.episode_index, {
                episode_id: data.episode_id,
                episode_index: data.episode_index,
                total_frames: existing?.total_frames ?? null,
                image_b64: null,
                error: data.message || "decode failed",
              });
              return {
                ...prev,
                framesByPos: { ...prev.framesByPos, [pos]: next },
                loadedByPos: { ...prev.loadedByPos, [pos]: prev.loadedByPos[pos] + 1 },
              };
            });
          } else if (data.type === "done") {
            setState((prev) => ({
              ...prev,
              phaseByPos: { ...prev.phaseByPos, [pos]: "complete" },
            }));
            es.close();
            sourcesRef.current[pos] = null;
          }
        } catch {
          // ignore malformed events
        }
      };

      es.onerror = () => {
        setState((prev) => {
          if (prev.phaseByPos[pos] === "complete") return prev;
          return {
            ...prev,
            phaseByPos: { ...prev.phaseByPos, [pos]: "error" },
            errorByPos: { ...prev.errorByPos, [pos]: "Connection lost" },
          };
        });
        es.close();
        sourcesRef.current[pos] = null;
      };
    },
    [datasetId, taskName],
  );

  const setPosition = useCallback(
    (pos: EdgeFramePosition) => {
      setState((prev) => ({ ...prev, position: pos }));
      // Lazy-fetch if we haven't started this position yet
      if (state.phaseByPos[pos] === "idle" && datasetId && taskName) {
        fetchPosition(pos);
      }
    },
    [datasetId, taskName, fetchPosition, state.phaseByPos],
  );

  // Reset + auto-fetch starting frames whenever dataset/task changes.
  useEffect(() => {
    closeAll();
    setState(emptyEdgeState());
    if (datasetId && taskName) {
      fetchPosition("start");
    }
    return () => {
      closeAll();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, taskName]);

  return { state, setPosition };
}

