/**
 * Hook for streaming frames using Server-Sent Events (SSE)
 *
 * Provides progressive frame loading where frames appear as they're decoded,
 * rather than waiting for the entire batch to complete.
 *
 * Performance optimizations:
 * - Blob URLs instead of base64 data URIs (3-5x faster rendering)
 * - Batched state updates (reduces React re-renders from ~500 to ~15-25)
 * - Sorted index array for O(log n) nearest-frame lookup
 */
import { useState, useEffect, useCallback, useRef } from "react";
import type { Frame, StreamingOptions } from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

interface UseStreamingFramesOptions extends StreamingOptions {
  enabled?: boolean;
}

export type StreamStatus = "downloading" | "extracting" | null;

interface UseStreamingFramesResult {
  frames: Map<number, Frame>;
  sortedIndices: number[];
  totalFrames: number | null;
  stride: number;
  progress: number;
  isLoading: boolean;
  isComplete: boolean;
  error: string | null;
  streamStatus: StreamStatus;
  cancel: () => void;
}

/** Convert base64 string to a Blob URL for fast rendering. */
function base64ToBlobUrl(base64: string): string {
  const byteChars = atob(base64);
  const byteNumbers = new Uint8Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++) {
    byteNumbers[i] = byteChars.charCodeAt(i);
  }
  const blob = new Blob([byteNumbers], { type: "image/webp" });
  return URL.createObjectURL(blob);
}

/** Revoke all blob URLs in a frame map to prevent memory leaks. */
function revokeAllBlobUrls(frames: Map<number, Frame>) {
  for (const frame of frames.values()) {
    if (frame.blobUrl) URL.revokeObjectURL(frame.blobUrl);
  }
}

/** Insert index into sorted array, maintaining sort order. O(log n). */
function insertSorted(arr: number[], val: number): void {
  // Fast path: appending in order (most common for sequential SSE)
  if (arr.length === 0 || val > arr[arr.length - 1]) {
    arr.push(val);
    return;
  }
  // Already present - skip
  if (arr.length > 0 && val === arr[arr.length - 1]) return;
  // Binary search for insertion point
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < val) lo = mid + 1;
    else hi = mid;
  }
  if (arr[lo] !== val) arr.splice(lo, 0, val);
}

/**
 * Hook that uses Server-Sent Events to stream frames progressively.
 *
 * Frames are accumulated in a mutable ref and periodically flushed
 * to React state, reducing re-renders from ~500 to ~15-25 during loading.
 */
export function useStreamingFrames(
  episodeId: string | null,
  datasetId: string | null,
  start: number,
  end: number,
  options: UseStreamingFramesOptions = {}
): UseStreamingFramesResult {
  const [frames, setFrames] = useState<Map<number, Frame>>(new Map());
  const [sortedIndices, setSortedIndices] = useState<number[]>([]);
  const [totalFrames, setTotalFrames] = useState<number | null>(null);
  const [stride, setStride] = useState<number>(1);
  const [isLoading, setIsLoading] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamStatus, setStreamStatus] = useState<StreamStatus>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const prevEpisodeIdRef = useRef<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Mutable refs for batched frame accumulation (avoids per-frame re-renders)
  const mutableFramesRef = useRef<Map<number, Frame>>(new Map());
  const mutableIndicesRef = useRef<number[]>([]);
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingCountRef = useRef(0);
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { resolution = "low", quality = 70, stream = "rgb", enabled = true } = options;

  // Calculate progress (account for stride: fewer frames expected)
  const expectedFrames = totalFrames && totalFrames > 0
    ? Math.ceil(totalFrames / stride)
    : 0;
  const progress = expectedFrames > 0
    ? Math.min(frames.size / expectedFrames, 1)
    : 0;

  /** Flush accumulated frames from mutable ref into React state (single re-render). */
  const flushFrames = useCallback(() => {
    setFrames(new Map(mutableFramesRef.current));
    setSortedIndices([...mutableIndicesRef.current]);
    pendingCountRef.current = 0;
    if (flushTimerRef.current) {
      clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
  }, []);

  const cancel = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (flushTimerRef.current) {
      clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
    if (idleTimerRef.current) {
      clearTimeout(idleTimerRef.current);
      idleTimerRef.current = null;
    }
    setIsLoading(false);
    setStreamStatus(null);
  }, []);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current);
        flushTimerRef.current = null;
      }
      if (idleTimerRef.current) {
        clearTimeout(idleTimerRef.current);
        idleTimerRef.current = null;
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
    if (flushTimerRef.current) {
      clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }

    if (idleTimerRef.current) {
      clearTimeout(idleTimerRef.current);
      idleTimerRef.current = null;
    }

    if (!episodeId || !datasetId || !enabled) {
      revokeAllBlobUrls(mutableFramesRef.current);
      mutableFramesRef.current = new Map();
      mutableIndicesRef.current = [];
      pendingCountRef.current = 0;
      setFrames(new Map());
      setSortedIndices([]);
      setTotalFrames(null);
      setIsComplete(false);
      setError(null);
      setStreamStatus(null);
      prevEpisodeIdRef.current = null;
      return;
    }

    // Only clear frames when episode changes, NOT when batch (start/end) changes
    const episodeChanged = prevEpisodeIdRef.current !== episodeId;
    if (episodeChanged) {
      revokeAllBlobUrls(mutableFramesRef.current);
      mutableFramesRef.current = new Map();
      mutableIndicesRef.current = [];
      pendingCountRef.current = 0;
      setFrames(new Map());
      setSortedIndices([]);
      setTotalFrames(null);
      setStride(1);
      prevEpisodeIdRef.current = episodeId;
    }

    setIsLoading(true);
    setIsComplete(false);
    setError(null);
    setStreamStatus(null);

    // Safety idle timeout: close SSE if no data received for 120s
    const IDLE_TIMEOUT_MS = 120_000;
    const resetIdleTimer = () => {
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
      idleTimerRef.current = setTimeout(() => {
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
          eventSourceRef.current = null;
        }
        setError("Connection timed out (no data received for 120s)");
        setIsLoading(false);
        setStreamStatus(null);
      }, IDLE_TIMEOUT_MS);
    };
    resetIdleTimer();

    const params = new URLSearchParams({
      dataset_id: datasetId,
      start: start.toString(),
      end: end.toString(),
      resolution,
      quality: quality.toString(),
      stream,
    });

    const url = `${API_BASE}/episodes/${episodeId}/frames/stream?${params}`;
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        resetIdleTimer(); // Reset idle timeout on every message

        switch (data.type) {
          case "status":
            setStreamStatus(data.status as StreamStatus);
            break;

          case "total":
            setTotalFrames(data.total_frames);
            if (data.stride && data.stride > 1) {
              setStride(data.stride);
            }
            break;

          case "frame": {
            // Convert base64 to Blob URL once (avoids re-parsing on every render)
            const blobUrl = base64ToBlobUrl(data.data);
            const frame: Frame = {
              index: data.index,
              timestamp: data.timestamp,
              image: data.data,
              blobUrl,
            };
            // Accumulate in mutable ref (no React re-render)
            mutableFramesRef.current.set(data.index, frame);
            insertSorted(mutableIndicesRef.current, data.index);
            pendingCountRef.current++;

            // Batch flush: every 50 frames or schedule a 100ms timer
            if (pendingCountRef.current >= 50) {
              flushFrames();
            } else if (!flushTimerRef.current) {
              flushTimerRef.current = setTimeout(flushFrames, 100);
            }
            break;
          }

          case "done":
            // Final flush of any remaining frames
            flushFrames();
            setIsLoading(false);
            setIsComplete(true);
            setStreamStatus(null);
            if (idleTimerRef.current) { clearTimeout(idleTimerRef.current); idleTimerRef.current = null; }
            eventSource.close();
            eventSourceRef.current = null;
            break;

          case "error":
            flushFrames();
            setError(data.message);
            setIsLoading(false);
            setStreamStatus(null);
            if (idleTimerRef.current) { clearTimeout(idleTimerRef.current); idleTimerRef.current = null; }
            eventSource.close();
            eventSourceRef.current = null;
            break;
        }
      } catch (e) {
        console.error("Failed to parse SSE message:", e);
      }
    };

    eventSource.onerror = async () => {
      // Check if connection was closed normally (after 'done')
      if (eventSource.readyState === EventSource.CLOSED && isComplete) {
        return;
      }

      eventSource.close();
      eventSourceRef.current = null;

      try {
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

        const respData = await response.json();
        const framesArray = respData.frames || respData;
        const total = respData.total_frames;

        // Build map with blob URLs
        revokeAllBlobUrls(mutableFramesRef.current);
        const framesMap = new Map<number, Frame>();
        const indices: number[] = [];
        for (const frame of framesArray) {
          const blobUrl = base64ToBlobUrl(frame.image_base64);
          framesMap.set(frame.frame_idx, {
            index: frame.frame_idx,
            timestamp: frame.timestamp,
            image: frame.image_base64,
            blobUrl,
            action: frame.action,
          });
          indices.push(frame.frame_idx);
        }
        indices.sort((a, b) => a - b);

        mutableFramesRef.current = framesMap;
        mutableIndicesRef.current = indices;
        setFrames(framesMap);
        setSortedIndices(indices);
        if (total !== undefined && total !== null) {
          setTotalFrames(total);
        }
        setIsLoading(false);
        setIsComplete(true);
      } catch (fallbackError) {
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
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current);
        flushTimerRef.current = null;
      }
      if (idleTimerRef.current) {
        clearTimeout(idleTimerRef.current);
        idleTimerRef.current = null;
      }
    };
  }, [episodeId, datasetId, start, end, resolution, quality, stream, enabled, flushFrames]);

  return {
    frames,
    sortedIndices,
    totalFrames,
    stride,
    progress,
    isLoading,
    isComplete,
    error,
    streamStatus,
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
 * Uses binary search on sorted indices for O(log n) lookup instead of O(n).
 */
export function getFrame(
  frames: Map<number, Frame>,
  index: number,
  sortedIndices?: number[]
): Frame | undefined {
  // O(1) exact match
  const exact = frames.get(index);
  if (exact) return exact;

  if (!sortedIndices || sortedIndices.length === 0) {
    if (frames.size === 0) return undefined;
    // Fallback to O(n) if no sorted indices provided
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

  // O(log n) binary search for nearest index
  let lo = 0, hi = sortedIndices.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedIndices[mid] < index) lo = mid + 1;
    else hi = mid;
  }
  // Check lo and lo-1 for nearest
  let bestIdx = sortedIndices[lo];
  if (lo > 0 && Math.abs(sortedIndices[lo - 1] - index) < Math.abs(bestIdx - index)) {
    bestIdx = sortedIndices[lo - 1];
  }
  return frames.get(bestIdx);
}

/**
 * Binary search for the largest index <= target in a sorted array.
 * Returns the frame at that index, or null. O(log n).
 */
export function getFloorFrame(
  frames: Map<number, Frame>,
  target: number,
  sortedIndices: number[]
): Frame | null {
  if (sortedIndices.length === 0) return null;
  let lo = 0, hi = sortedIndices.length - 1;
  let result = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (sortedIndices[mid] <= target) {
      result = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return result >= 0 ? frames.get(sortedIndices[result]) || null : null;
}
