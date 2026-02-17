/**
 * Hook for streaming frames using Server-Sent Events (SSE)
 *
 * Provides progressive frame loading where frames appear as they're decoded,
 * rather than waiting for the entire batch to complete.
 */
import { useState, useEffect, useCallback, useRef } from "react";
import type { Frame, StreamingOptions } from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

interface UseStreamingFramesOptions extends StreamingOptions {
  enabled?: boolean;
}

interface UseStreamingFramesResult {
  frames: Map<number, Frame>;
  totalFrames: number | null;
  stride: number;
  progress: number;
  isLoading: boolean;
  isComplete: boolean;
  error: string | null;
  cancel: () => void;
}

/**
 * Hook that uses Server-Sent Events to stream frames progressively.
 *
 * Frames are added to a Map as they arrive, allowing immediate display
 * of early frames while later frames are still being decoded.
 */
export function useStreamingFrames(
  episodeId: string | null,
  datasetId: string | null,
  start: number,
  end: number,
  options: UseStreamingFramesOptions = {}
): UseStreamingFramesResult {
  const [frames, setFrames] = useState<Map<number, Frame>>(new Map());
  const [totalFrames, setTotalFrames] = useState<number | null>(null);
  const [stride, setStride] = useState<number>(1);
  const [isLoading, setIsLoading] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const prevEpisodeIdRef = useRef<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const { resolution = "low", quality = 70, stream = "rgb", enabled = true } = options;

  // Calculate progress (account for stride: fewer frames expected)
  const expectedFrames = totalFrames && totalFrames > 0
    ? Math.ceil(totalFrames / stride)
    : 0;
  const progress = expectedFrames > 0
    ? Math.min(frames.size / expectedFrames, 1)
    : 0;

  const cancel = useCallback(() => {
    // Close SSE connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    // Abort any fetch requests
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  }, []);

  useEffect(() => {
    // Cleanup on unmount or when dependencies change
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    // Cancel any previous request immediately
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    if (!episodeId || !datasetId || !enabled) {
      setFrames(new Map());
      setTotalFrames(null);
      setIsComplete(false);
      setError(null);
      prevEpisodeIdRef.current = null;
      return;
    }

    // Only clear frames when episode changes, NOT when batch (start/end) changes
    // This keeps the UI responsive during batch transitions
    const episodeChanged = prevEpisodeIdRef.current !== episodeId;
    if (episodeChanged) {
      setFrames(new Map());
      setTotalFrames(null);
      setStride(1);
      prevEpisodeIdRef.current = episodeId;
    }

    // Reset loading state for new request
    setIsLoading(true);
    setIsComplete(false);
    setError(null);

    // Build SSE URL
    const params = new URLSearchParams({
      dataset_id: datasetId,
      start: start.toString(),
      end: end.toString(),
      resolution,
      quality: quality.toString(),
      stream,
    });

    const url = `${API_BASE}/episodes/${episodeId}/frames/stream?${params}`;

    // Create EventSource for SSE
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case "total":
            setTotalFrames(data.total_frames);
            if (data.stride && data.stride > 1) {
              setStride(data.stride);
            }
            break;

          case "frame":
            setFrames((prev) => {
              const newMap = new Map(prev);
              newMap.set(data.index, {
                index: data.index,
                timestamp: data.timestamp,
                image: data.data,
              });
              return newMap;
            });
            break;

          case "done":
            setIsLoading(false);
            setIsComplete(true);
            eventSource.close();
            eventSourceRef.current = null;
            break;

          case "error":
            setError(data.message);
            setIsLoading(false);
            eventSource.close();
            eventSourceRef.current = null;
            break;
        }
      } catch (e) {
        console.error("Failed to parse SSE message:", e);
      }
    };

    eventSource.onerror = async (e) => {
      // Check if connection was closed normally (after 'done')
      if (eventSource.readyState === EventSource.CLOSED && isComplete) {
        return;
      }

      // SSE failed - try falling back to regular frames endpoint
      // This happens for non-streaming datasets (local HDF5, LeRobot, etc.)
      eventSource.close();
      eventSourceRef.current = null;

      try {
        // Create AbortController for the fetch request
        const abortController = new AbortController();
        abortControllerRef.current = abortController;

        const fallbackParams = new URLSearchParams({
          dataset_id: datasetId,
          start: start.toString(),
          end: end.toString(),
          resolution,
          quality: quality.toString(),
          stream,
        });
        const fallbackUrl = `${API_BASE}/episodes/${episodeId}/frames?${fallbackParams}`;
        const response = await fetch(fallbackUrl, { signal: abortController.signal });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        const framesArray = data.frames || data;
        const total = data.total_frames;

        // Convert to Map
        const framesMap = new Map<number, Frame>();
        for (const frame of framesArray) {
          framesMap.set(frame.frame_idx, {
            index: frame.frame_idx,
            timestamp: frame.timestamp,
            image: frame.image_base64,
            action: frame.action,
          });
        }

        setFrames(framesMap);
        if (total !== undefined && total !== null) {
          setTotalFrames(total);
        }
        setIsLoading(false);
        setIsComplete(true);
      } catch (fallbackError) {
        // Don't treat abort as an error - it's intentional cancellation
        if (fallbackError instanceof Error && fallbackError.name === 'AbortError') {
          return;
        }
        console.error("Fallback frames fetch failed:", fallbackError);
        setError("Connection error");
        setIsLoading(false);
      }
    };

    return () => {
      eventSource.close();
      eventSourceRef.current = null;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, [episodeId, datasetId, start, end, resolution, quality, stream, enabled]);

  return {
    frames,
    totalFrames,
    stride,
    progress,
    isLoading,
    isComplete,
    error,
    cancel,
  };
}

/**
 * Convert Map<number, Frame> to array for rendering
 */
export function framesToArray(frames: Map<number, Frame>): Frame[] {
  return Array.from(frames.values()).sort((a, b) => a.index - b.index);
}

/**
 * Get a specific frame from the map, or the nearest frame for strided data.
 * When stride > 1, exact frame indices may not exist in the map,
 * so we find the closest available frame.
 */
export function getFrame(frames: Map<number, Frame>, index: number): Frame | undefined {
  const exact = frames.get(index);
  if (exact) return exact;

  // Nearest-frame lookup for strided data
  if (frames.size === 0) return undefined;

  let best: Frame | undefined;
  let bestDist = Infinity;
  for (const [idx, frame] of frames) {
    const dist = Math.abs(idx - index);
    if (dist < bestDist) {
      bestDist = dist;
      best = frame;
    }
  }
  return best;
}
