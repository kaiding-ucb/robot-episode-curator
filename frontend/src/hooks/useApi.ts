/**
 * React hooks for API calls
 */
import { useState, useEffect, useCallback, useRef } from "react";
import type {
  Dataset,
  EpisodeMetadata,
  Frame,
  DownloadStatus,
  DiskSpace,
  Task,
  TaskListResponse,
  ImageResolution,
  StreamingOptions,
  CachedEpisode,
  CacheStats,
  DatasetOverview,
} from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

/**
 * Fetch datasets from the API
 */
export function useDatasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchDatasets() {
      try {
        const res = await fetch(`${API_BASE}/datasets`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setDatasets(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to fetch datasets");
      } finally {
        setLoading(false);
      }
    }
    fetchDatasets();
  }, []);

  return { datasets, loading, error };
}

/**
 * Fetch episodes for a dataset
 */
export function useEpisodes(datasetId: string | null) {
  const [episodes, setEpisodes] = useState<EpisodeMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId) {
      setEpisodes([]);
      return;
    }

    async function fetchEpisodes() {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/datasets/${datasetId}/episodes`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setEpisodes(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to fetch episodes");
      } finally {
        setLoading(false);
      }
    }
    fetchEpisodes();
  }, [datasetId]);

  return { episodes, loading, error };
}

/**
 * Fetch tasks for a dataset
 */
export function useTasks(datasetId: string | null) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [totalTasks, setTotalTasks] = useState(0);
  const [source, setSource] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId) {
      setTasks([]);
      setTotalTasks(0);
      setSource(null);
      return;
    }

    async function fetchTasks() {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/datasets/${datasetId}/tasks`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: TaskListResponse = await res.json();
        setTasks(data.tasks);
        setTotalTasks(data.total_tasks);
        setSource(data.source);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to fetch tasks");
      } finally {
        setLoading(false);
      }
    }
    fetchTasks();
  }, [datasetId]);

  return { tasks, totalTasks, source, loading, error };
}

/**
 * Fetch episodes for a specific task
 */
export function useTaskEpisodes(
  datasetId: string | null,
  taskName: string | null,
  limit: number = 5,
  offset: number = 0
) {
  const [episodes, setEpisodes] = useState<EpisodeMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);

  const fetchEpisodes = useCallback(async (newOffset: number = 0) => {
    if (!datasetId || !taskName) {
      setEpisodes([]);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const encodedTaskName = encodeURIComponent(taskName);
      const res = await fetch(
        `${API_BASE}/datasets/${datasetId}/tasks/${encodedTaskName}/episodes?limit=${limit}&offset=${newOffset}`
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: EpisodeMetadata[] = await res.json();

      if (newOffset === 0) {
        setEpisodes(data);
      } else {
        setEpisodes(prev => [...prev, ...data]);
      }

      // If we got exactly limit episodes, there might be more
      setHasMore(data.length === limit);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch episodes");
    } finally {
      setLoading(false);
    }
  }, [datasetId, taskName, limit]);

  useEffect(() => {
    fetchEpisodes(0);
  }, [datasetId, taskName, limit]);

  const loadMore = useCallback(() => {
    fetchEpisodes(episodes.length);
  }, [fetchEpisodes, episodes.length]);

  return { episodes, loading, error, hasMore, loadMore };
}

/**
 * Previous batch data for fallback during transitions
 */
interface PreviousBatch {
  frames: Frame[];
  range: { start: number; end: number };
  episodeId: string;
}

/**
 * Fetch frames for an episode with pre-fetching and fallback support
 */
export function useFrames(
  episodeId: string | null,
  start: number = 0,
  end: number = 10,
  datasetId: string | null = null,
  streamingOptions?: StreamingOptions
) {
  const [frames, setFrames] = useState<Frame[]>([]);
  const [totalFrames, setTotalFrames] = useState<number | null>(null);
  const [loadedRange, setLoadedRange] = useState<{ start: number; end: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Previous batch for fallback during transitions
  const [previousBatch, setPreviousBatch] = useState<PreviousBatch | null>(null);

  // Prefetched data cache
  const prefetchedDataRef = useRef<{
    frames: Frame[];
    range: { start: number; end: number };
    episodeId: string;
  } | null>(null);
  const [prefetchedRange, setPrefetchedRange] = useState<{ start: number; end: number } | null>(null);

  // Track previous episode to detect changes
  const prevEpisodeIdRef = useRef<string | null>(null);

  // Track in-flight prefetch to prevent duplicate requests
  const prefetchInFlightRef = useRef<string | null>(null);

  // Helper function to fetch frames
  const fetchFramesData = useCallback(async (
    epId: string,
    rangeStart: number,
    rangeEnd: number,
    dsId: string | null,
    options?: StreamingOptions
  ): Promise<{ frames: Frame[]; total: number | null }> => {
    const datasetParam = dsId ? `&dataset_id=${dsId}` : '';
    const resolutionParam = options?.resolution ? `&resolution=${options.resolution}` : '';
    const qualityParam = options?.quality ? `&quality=${options.quality}` : '';
    const res = await fetch(
      `${API_BASE}/episodes/${epId}/frames?start=${rangeStart}&end=${rangeEnd}${datasetParam}${resolutionParam}${qualityParam}`
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const framesArray = data.frames || data;
    const total = data.total_frames;

    const mappedFrames: Frame[] = framesArray.map((frame: {
      frame_idx: number;
      timestamp?: number;
      image_base64?: string;
      action?: number[];
    }) => ({
      index: frame.frame_idx,
      timestamp: frame.timestamp,
      image: frame.image_base64,
      action: frame.action,
    }));

    return { frames: mappedFrames, total };
  }, []);

  // Prefetch function for loading upcoming batches
  const prefetch = useCallback(async (prefetchStart: number, prefetchEnd: number) => {
    if (!episodeId || !datasetId) return;

    // Create a unique key for this prefetch request
    const prefetchKey = `${episodeId}:${prefetchStart}-${prefetchEnd}`;

    // Don't prefetch if we already have this range
    if (prefetchedDataRef.current?.range.start === prefetchStart &&
        prefetchedDataRef.current?.range.end === prefetchEnd &&
        prefetchedDataRef.current?.episodeId === episodeId) {
      return;
    }

    // Don't prefetch if this exact request is already in flight
    if (prefetchInFlightRef.current === prefetchKey) {
      return;
    }

    // Mark as in-flight
    prefetchInFlightRef.current = prefetchKey;

    try {
      const { frames: prefetchedFrames, total } = await fetchFramesData(
        episodeId,
        prefetchStart,
        prefetchEnd,
        datasetId,
        streamingOptions
      );

      prefetchedDataRef.current = {
        frames: prefetchedFrames,
        range: { start: prefetchStart, end: prefetchEnd },
        episodeId,
      };
      setPrefetchedRange({ start: prefetchStart, end: prefetchEnd });
    } catch {
      // Silently fail prefetch - it's just an optimization
    } finally {
      // Clear in-flight marker
      if (prefetchInFlightRef.current === prefetchKey) {
        prefetchInFlightRef.current = null;
      }
    }
  }, [episodeId, datasetId, streamingOptions, fetchFramesData]);

  useEffect(() => {
    // Detect episode change and clear ALL state including frames
    if (prevEpisodeIdRef.current !== episodeId) {
      setPreviousBatch(null);
      prefetchedDataRef.current = null;
      setPrefetchedRange(null);
      prefetchInFlightRef.current = null;
      prevEpisodeIdRef.current = episodeId;
      // Clear frames immediately when episode changes to avoid showing old episode
      setFrames([]);
      setLoadedRange(null);
    }

    if (!episodeId) {
      setFrames([]);
      setTotalFrames(null);
      setLoadedRange(null);
      return;
    }

    // Capture episodeId for use in async function (TypeScript narrowing)
    const currentEpisodeId = episodeId;

    async function fetchFrames() {
      // Check if we have prefetched data for this range
      if (prefetchedDataRef.current?.range.start === start &&
          prefetchedDataRef.current?.range.end === end &&
          prefetchedDataRef.current?.episodeId === currentEpisodeId) {
        // Store current batch as previous before switching
        if (frames.length > 0 && loadedRange) {
          setPreviousBatch({
            frames: [...frames],
            range: { ...loadedRange },
            episodeId: currentEpisodeId,
          });
        }

        // Use prefetched data
        setFrames(prefetchedDataRef.current.frames);
        setLoadedRange({ start, end });
        prefetchedDataRef.current = null;
        setPrefetchedRange(null);
        return;
      }

      setLoading(true);
      setError(null);

      // Store current batch as previous before fetching new batch
      if (frames.length > 0 && loadedRange) {
        setPreviousBatch({
          frames: [...frames],
          range: { ...loadedRange },
          episodeId: currentEpisodeId,
        });
      }

      try {
        const { frames: newFrames, total } = await fetchFramesData(currentEpisodeId, start, end, datasetId, streamingOptions);
        setFrames(newFrames);
        if (total !== undefined && total !== null) {
          setTotalFrames(total);
        }
        setLoadedRange({ start, end });
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to fetch frames");
      } finally {
        setLoading(false);
      }
    }
    fetchFrames();
  }, [episodeId, start, end, datasetId, streamingOptions, fetchFramesData]);

  // Function to clear previous batch (for manual seeks)
  const clearPreviousBatch = useCallback(() => {
    setPreviousBatch(null);
  }, []);

  return {
    frames,
    loading,
    error,
    loadedRange,
    totalFrames,
    previousBatch,
    prefetch,
    prefetchedRange,
    clearPreviousBatch,
  };
}

/**
 * Download status management
 */
export function useDownloads() {
  const [statuses, setStatuses] = useState<DownloadStatus[]>([]);
  const [diskSpace, setDiskSpace] = useState<DiskSpace | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const [statusRes, diskRes] = await Promise.all([
        fetch(`${API_BASE}/downloads/status`),
        fetch(`${API_BASE}/downloads/disk-space`),
      ]);

      if (!statusRes.ok) throw new Error(`HTTP ${statusRes.status}`);
      if (!diskRes.ok) throw new Error(`HTTP ${diskRes.status}`);

      const statusData = await statusRes.json();
      const diskData = await diskRes.json();

      setStatuses(statusData);
      setDiskSpace(diskData);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch status");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    // Poll for updates
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const startDownload = useCallback(
    async (datasetId: string, options?: { dataset?: string; limit?: number }) => {
      try {
        const res = await fetch(`${API_BASE}/downloads/${datasetId}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(options || {}),
        });
        if (!res.ok) {
          const data = await res.json();
          throw new Error(data.detail || `HTTP ${res.status}`);
        }
        // Refresh status
        await fetchStatus();
        return true;
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to start download");
        return false;
      }
    },
    [fetchStatus]
  );

  return { statuses, diskSpace, loading, error, startDownload, refresh: fetchStatus };
}

/**
 * Episode cache management
 */
export function useEpisodeCache() {
  const [cachedEpisodes, setCachedEpisodes] = useState<CachedEpisode[]>([]);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchCachedEpisodes = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [episodesRes, statsRes] = await Promise.all([
        fetch(`${API_BASE}/downloads/cache/episodes`),
        fetch(`${API_BASE}/downloads/cache/stats`),
      ]);

      if (!episodesRes.ok) throw new Error(`HTTP ${episodesRes.status}`);
      if (!statsRes.ok) throw new Error(`HTTP ${statsRes.status}`);

      const episodesData = await episodesRes.json();
      const statsData = await statsRes.json();

      setCachedEpisodes(episodesData);
      setCacheStats(statsData);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch cached episodes");
    } finally {
      setLoading(false);
    }
  }, []);

  const deleteEpisodeCache = useCallback(
    async (datasetId: string, episodeId: string) => {
      try {
        const encodedEpisodeId = encodeURIComponent(episodeId);
        const res = await fetch(
          `${API_BASE}/downloads/cache/episodes/${datasetId}/${encodedEpisodeId}`,
          { method: "DELETE" }
        );
        if (!res.ok) {
          const data = await res.json();
          throw new Error(data.detail || `HTTP ${res.status}`);
        }
        // Refresh the list
        await fetchCachedEpisodes();
        return true;
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to delete cache");
        return false;
      }
    },
    [fetchCachedEpisodes]
  );

  const clearAllCache = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/downloads/cache/episodes`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      // Refresh the list
      await fetchCachedEpisodes();
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to clear cache");
      return false;
    }
  }, [fetchCachedEpisodes]);

  return {
    cachedEpisodes,
    cacheStats,
    loading,
    error,
    fetchCachedEpisodes,
    deleteEpisodeCache,
    clearAllCache,
  };
}

/**
 * Fetch dataset overview metadata
 */
export function useDatasetOverview(datasetId: string | null) {
  const [overview, setOverview] = useState<DatasetOverview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchOverview = useCallback(async (refresh: boolean = false) => {
    if (!datasetId) {
      setOverview(null);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const refreshParam = refresh ? "?refresh=true" : "";
      const res = await fetch(`${API_BASE}/datasets/${datasetId}/overview${refreshParam}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setOverview(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch overview");
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => {
    fetchOverview(false);
  }, [fetchOverview]);

  const refresh = useCallback(() => {
    fetchOverview(true);
  }, [fetchOverview]);

  return { overview, loading, error, refresh };
}
