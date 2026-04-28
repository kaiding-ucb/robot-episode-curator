"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { useStreamingFrames, getFrame, getFloorFrame } from "@/hooks/useStreamingFrames";
import type { StreamingOptions, Modality } from "@/types/api";
import { useQualityEvents } from "@/hooks/useQuality";
import EnhancedTimeline from "./EnhancedTimeline";
import IMUChart from "./IMUChart";
import ActionsChart from "./ActionsChart";

interface EpisodeViewerProps {
  datasetId: string | null;
  episodeId: string | null;
  totalFrames: number;
  targetFrame?: number | null;
  onFrameChange?: (frame: number) => void;
  availableModalities?: Modality[];
}

export default function EpisodeViewer({
  datasetId,
  episodeId,
  totalFrames,
  targetFrame,
  onFrameChange,
  availableModalities = ["rgb"],
}: EpisodeViewerProps) {
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const activeStream = "rgb" as const;
  const [activeChartTab, setActiveChartTab] = useState<"none" | "actions" | "imu">("none");

  // Deferred loading: SSE only starts after user clicks Play
  const [playRequested, setPlayRequested] = useState(false);

  // Background caching status
  const [cachingStatus, setCachingStatus] = useState<{
    status: "not_cached" | "caching" | "cached" | "not_applicable" | "started" | "error" | null;
    progress?: number;
    totalFrames?: number;
  }>({ status: null });
  const cachingTriggeredRef = useRef(false);

  // Track if initial caching status check is done (to avoid starting SSE prematurely)
  const [initialStatusChecked, setInitialStatusChecked] = useState(false);

  // Check available modalities
  // depth playback not yet supported — toggle removed
  const hasImu = availableModalities.includes("imu");
  const hasActions = availableModalities.includes("actions");

  // Always use lowest resolution for fast loading
  const streamingOptions: StreamingOptions = useMemo(() => ({
    resolution: "low",  // 320x240
    quality: 30,        // Low quality for speed
    stream: activeStream,
  }), [activeStream]);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

  // Trigger background caching when play is clicked
  const triggerBackgroundCaching = useCallback(async () => {
    if (!episodeId || !datasetId || cachingTriggeredRef.current) return;
    // Don't trigger if already cached or caching in progress
    if (cachingStatus.status === "cached" || cachingStatus.status === "caching" ||
        cachingStatus.status === "started") return;

    cachingTriggeredRef.current = true;

    try {
      const params = new URLSearchParams({
        dataset_id: datasetId,
        resolution: streamingOptions.resolution || "low",
        quality: String(streamingOptions.quality || 70),
        stream: activeStream,
      });

      const res = await fetch(
        `${API_BASE}/episodes/${encodeURIComponent(episodeId)}/cache?${params}`,
        { method: "POST" }
      );
      const data = await res.json();
      setCachingStatus({
        status: data.status === "started" ? "caching" : data.status,
        progress: data.progress,
        totalFrames: data.total_frames,
      });
    } catch (err) {
      console.error("Failed to trigger background caching:", err);
    }
  }, [episodeId, datasetId, cachingStatus.status, streamingOptions, activeStream, API_BASE]);

  // Check caching status when episode loads (to resume showing progress for ongoing caching)
  // IMPORTANT: This must complete BEFORE SSE starts to avoid redundant streaming
  useEffect(() => {
    if (!episodeId || !datasetId) {
      setInitialStatusChecked(true);
      return;
    }

    // Reset for new episode
    setInitialStatusChecked(false);

    const checkInitialStatus = async () => {
      try {
        const params = new URLSearchParams({
          dataset_id: datasetId,
          resolution: streamingOptions.resolution || "low",
          quality: String(streamingOptions.quality || 70),
          stream: activeStream,
        });

        const res = await fetch(
          `${API_BASE}/episodes/${encodeURIComponent(episodeId)}/cache/status?${params}`
        );
        const data = await res.json();
        setCachingStatus({
          status: data.status,
          progress: data.progress,
          totalFrames: data.total_frames,
        });
        // If already caching, don't trigger again
        if (data.status === "caching" || data.status === "cached") {
          cachingTriggeredRef.current = true;
        } else {
          cachingTriggeredRef.current = false;
        }
      } catch (err) {
        console.error("Failed to check initial caching status:", err);
        setCachingStatus({ status: null });
        cachingTriggeredRef.current = false;
      } finally {
        setInitialStatusChecked(true);
      }
    };

    checkInitialStatus();
  }, [episodeId, datasetId, streamingOptions, activeStream, API_BASE]);

  // Poll caching status while caching is in progress
  useEffect(() => {
    if (!episodeId || !datasetId) return;
    // Poll when status is "caching" or "started" (which means caching just began)
    if (cachingStatus.status !== "caching" && cachingStatus.status !== "started") return;

    const pollStatus = async () => {
      try {
        const params = new URLSearchParams({
          dataset_id: datasetId,
          resolution: streamingOptions.resolution || "low",
          quality: String(streamingOptions.quality || 70),
          stream: activeStream,
        });

        const res = await fetch(
          `${API_BASE}/episodes/${encodeURIComponent(episodeId)}/cache/status?${params}`
        );
        const data = await res.json();
        setCachingStatus({
          status: data.status,
          progress: data.progress,
          totalFrames: data.total_frames,
        });
      } catch (err) {
        console.error("Failed to poll caching status:", err);
      }
    };

    // Poll immediately, then every 1 second for responsive UX during caching
    pollStatus();
    const interval = setInterval(pollStatus, 1000);
    return () => clearInterval(interval);
  }, [episodeId, datasetId, cachingStatus.status, streamingOptions, activeStream, API_BASE]);

  // Use SSE streaming for progressive frame loading
  // Frames appear as they're decoded (~2-3s for first frame) rather than waiting for full batch
  const batchSize = 750; // ~25 seconds at 30fps
  const batchStart = Math.floor(currentFrame / batchSize) * batchSize;

  // Determine if episode is actively being cached in background
  const isActivelyCaching = cachingStatus.status === "caching" || cachingStatus.status === "started";

  const {
    frames,
    sortedIndices,
    totalFrames: apiTotalFrames,
    stride,
    progress,
    isLoading: loading,
    isComplete,
    error,
    cancel,
  } = useStreamingFrames(
    episodeId,
    datasetId,
    batchStart,
    batchStart + batchSize,
    {
      ...streamingOptions,
      // Only start SSE when user has clicked Play AND no background caching is actively running.
      // This prevents triggering a multi-minute download just by browsing episodes,
      // and avoids double downloads (SSE + background caching simultaneously).
      // When caching completes ("cached") or fails ("error"/"not_cached"), SSE starts
      // and the stream_frames endpoint handles cache-first logic independently.
      enabled: playRequested && initialStatusChecked && !isActivelyCaching,
    }
  );

  // Use API-provided total frames if prop is 0 or unknown
  const effectiveTotalFrames = totalFrames > 0 ? totalFrames : (apiTotalFrames || 0);

  // Fetch quality events for timeline visualization
  const { events: qualityEventsData } = useQualityEvents(datasetId, episodeId);

  // Jump to target frame when it changes (from QualityPanel jump-to buttons)
  useEffect(() => {
    if (targetFrame !== null && targetFrame !== undefined && targetFrame !== currentFrame) {
      setCurrentFrame(targetFrame);
      setPlaying(false);
      onFrameChange?.(targetFrame);
    }
  }, [targetFrame, currentFrame, onFrameChange]);

  // Cancel streaming when episode changes or component unmounts
  useEffect(() => {
    return () => cancel();
  }, [episodeId, cancel]);

  // Get the frame to display from the streaming Map
  // Uses O(log n) binary search on sorted indices instead of O(n) linear scan
  const displayFrame = getFrame(frames, currentFrame, sortedIndices);

  // For streaming: show last available frame if current frame hasn't arrived yet
  // Uses O(log n) binary search instead of O(n) linear scan
  const lastLoadedFrame = useMemo(() => {
    return getFloorFrame(frames, currentFrame, sortedIndices);
  }, [frames, currentFrame, sortedIndices]);

  // Use current frame if available, otherwise fallback to last loaded frame
  const frameToShow = displayFrame || lastLoadedFrame;
  const isShowingFallback = !displayFrame && lastLoadedFrame !== null;


  // Track frame availability and loading state in refs so the interval callback can check them
  const hasFrameRef = useRef(false);
  const isLoadingRef = useRef(false);
  hasFrameRef.current = frameToShow !== null;
  isLoadingRef.current = loading;

  // Playback logic - continues during loading, uses fallback frames
  // When stride > 1, increment by stride so each tick shows a new actual frame
  const playbackStride = stride > 1 ? stride : 1;

  useEffect(() => {
    if (!playing) return;

    const interval = setInterval(() => {
      // Skip tick if no frame available yet (waiting for SSE)
      if (!hasFrameRef.current) {
        return;
      }

      setCurrentFrame((prev) => {
        const next = prev + playbackStride;
        if (next >= effectiveTotalFrames) {
          setPlaying(false);
          return prev;
        }
        return next;
      });
    }, 1000 / (30 * playbackSpeed)); // Assume 30fps base

    return () => clearInterval(interval);
  }, [playing, effectiveTotalFrames, playbackSpeed, playbackStride]);

  // Reset when episode changes
  useEffect(() => {
    setCurrentFrame(0);
    setPlaying(false);
    setPlayRequested(false); // Reset deferred loading on episode change
    // activeStream is always "rgb" (depth playback not supported)
    // Reset caching status to prevent stale state from enabling SSE on first render
    setCachingStatus({ status: null });
    cachingTriggeredRef.current = false; // Allow caching for new episode
  }, [episodeId]);

  const handleFrameChange = useCallback((frame: number) => {
    setCurrentFrame(frame);
    setPlaying(false);
    // Don't clear previous batch - we need it for display during batch transitions
    onFrameChange?.(frame);
  }, [onFrameChange]);

  // Direct caching trigger for uncached episode Play button
  // Calls triggerBackgroundCaching() directly before setting playing state,
  // avoiding race conditions with React's batched updates in togglePlayback
  const handleStartCaching = useCallback(async () => {
    setPlayRequested(true);
    await triggerBackgroundCaching();
    setPlaying(true);
  }, [triggerBackgroundCaching]);

  const togglePlayback = useCallback(() => {
    setPlayRequested(true);
    setPlaying((prev) => {
      // Trigger background caching when starting playback
      if (!prev) {
        triggerBackgroundCaching();
      }
      return !prev;
    });
  }, [triggerBackgroundCaching]);

  if (!episodeId) {
    return (
      <div
        className="flex items-center justify-center h-full text-gray-500"
        data-testid="no-episode-selected"
      >
        <div className="text-center">
          <svg
            className="w-16 h-16 mx-auto mb-4 text-gray-300"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
            />
          </svg>
          <p>Select an episode to view</p>
        </div>
      </div>
    );
  }

  // Overlay state logic (replaces early returns)
  const isReadyToStream = cachingStatus.status === "cached" || cachingStatus.status === "not_applicable";
  const needsCaching = initialStatusChecked && !isReadyToStream && !isActivelyCaching && frames.size === 0 && !loading;
  const showReadyToPlayOverlay = !playRequested;
  const showCachingOverlay = playRequested && isActivelyCaching && !frameToShow;
  const showLoadingOverlay = playRequested && loading && frames.size === 0 && !frameToShow && !isActivelyCaching;
  const showErrorOverlay = !!error;

  // Controls disabled before play
  const controlsDisabled = !playRequested;

  return (
    <div className="flex flex-col h-full" data-testid="episode-viewer">
      {/* Video Display */}
      <div className="flex-1 min-h-0 bg-black flex items-center justify-center relative overflow-hidden">
        {frameToShow ? (
          <img
            src={frameToShow.blobUrl || `data:image/webp;base64,${frameToShow.image}`}
            alt={`Frame ${currentFrame}`}
            className="max-w-full max-h-full object-contain"
            data-testid="frame-image"
          />
        ) : (
          <div className="text-gray-500">{playRequested ? "No frame available" : ""}</div>
        )}

        {/* Ready-to-play overlay — shown before user clicks Play */}
        {showReadyToPlayOverlay && (
          <div
            className="absolute inset-0 bg-black/60 flex items-center justify-center z-10"
            data-testid="ready-to-play-overlay"
          >
            <div className="text-center">
              <button
                onClick={handleStartCaching}
                className="w-20 h-20 rounded-full bg-gray-900 hover:bg-black dark:bg-gray-700 dark:hover:bg-gray-600 text-white flex items-center justify-center transition-colors mx-auto mb-3"
                data-testid="play-overlay-btn"
              >
                <svg className="w-10 h-10 ml-1" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </button>
              <p className="text-white text-sm opacity-75">
                {needsCaching && cachingStatus.status === "error"
                  ? "Previous caching failed. Click to retry."
                  : "Click to play"}
              </p>
            </div>
          </div>
        )}

        {/* Caching overlay — downloading/caching in progress, no frames yet */}
        {showCachingOverlay && (
          <div
            className="absolute inset-0 bg-black/60 flex items-center justify-center z-10"
            data-testid="caching-progress"
          >
            <div className="text-center text-white">
              <div className="animate-spin w-8 h-8 border-2 border-gray-300 dark:border-gray-600 border-t-transparent rounded-full mx-auto mb-2"></div>
              <p className="text-lg font-medium">Caching Episode...</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{cachingStatus.progress || 0}%</p>
              <p className="text-xs text-gray-400 mt-2">
                {cachingStatus.totalFrames
                  ? `${Math.round((cachingStatus.progress || 0) * cachingStatus.totalFrames / 100)} / ${cachingStatus.totalFrames} frames`
                  : "Preparing..."}
              </p>
            </div>
          </div>
        )}

        {/* Loading overlay — SSE streaming started, waiting for first frame */}
        {showLoadingOverlay && (
          <div
            className="absolute inset-0 bg-black/60 flex items-center justify-center z-10"
            data-testid="loading-frames"
          >
            <div className="text-center text-white">
              <div className="animate-spin w-8 h-8 border-2 border-gray-300 dark:border-gray-600 border-t-transparent rounded-full mx-auto mb-2"></div>
              <p>Loading frames...</p>
            </div>
          </div>
        )}

        {/* Error overlay */}
        {showErrorOverlay && (
          <div
            className="absolute inset-0 bg-black/60 flex items-center justify-center z-10"
            data-testid="error-frames"
          >
            <div className="text-center">
              <p className="text-red-400 mb-3">Error: {error}</p>
              <button
                onClick={() => { setPlayRequested(false); }}
                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-500 transition-colors text-sm"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {/* Frame info overlay */}
        <div className="absolute top-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
          Frame {currentFrame + 1} / {effectiveTotalFrames}
        </div>

        {/* Subsampled indicator */}
        {stride > 1 && (
          <div
            className="absolute top-2 right-2 bg-yellow-500/90 text-white text-xs font-medium px-3 py-1 rounded-full"
            data-testid="subsampled-indicator"
          >
            Subsampled {stride}x
          </div>
        )}

        {/* Streaming progress indicator */}
        {loading && frames.size > 0 && (
          <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded flex items-center gap-2">
            <div className="w-24 h-1 bg-gray-600 rounded overflow-hidden">
              <div
                className="h-full bg-gray-700 dark:bg-gray-300 transition-all duration-200"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
            <span>{Math.round(progress * 100)}%</span>
          </div>
        )}

        {/* Caching status indicator */}
        {(cachingStatus.status === "caching" || cachingStatus.status === "started") && (
          <div
            className="absolute top-2 left-1/2 -translate-x-1/2 bg-gray-900/90 dark:bg-gray-700/90 text-white text-xs px-3 py-1 rounded-full flex items-center gap-2"
            data-testid="caching-indicator"
          >
            <div className="animate-spin w-3 h-3 border border-white border-t-transparent rounded-full" />
            <span>Caching... {cachingStatus.progress || 0}%</span>
          </div>
        )}
        {cachingStatus.status === "cached" && (
          <div
            className="absolute top-2 left-1/2 -translate-x-1/2 bg-green-500/90 text-white text-xs px-3 py-1 rounded-full"
            data-testid="cached-indicator"
          >
            Cached
          </div>
        )}


      </div>

      {/* Modality Chart Tabs - only show if actions or IMU available */}
      {(hasActions || hasImu) && (
        <div className="border-t border-gray-200 dark:border-gray-700">
          {/* Tab Bar */}
          <div className="flex gap-1 px-4 py-2 bg-gray-100 dark:bg-gray-800">
            <button
              onClick={() => setActiveChartTab("none")}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                activeChartTab === "none"
                  ? "bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm"
                  : "text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
              data-testid="chart-tab-none"
            >
              None
            </button>
            {hasActions && (
              <button
                onClick={() => setActiveChartTab("actions")}
                className={`px-3 py-1 text-xs rounded transition-colors ${
                  activeChartTab === "actions"
                    ? "bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm"
                    : "text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
                }`}
                data-testid="chart-tab-actions"
              >
                Actions
              </button>
            )}
            {hasImu && (
              <button
                onClick={() => setActiveChartTab("imu")}
                className={`px-3 py-1 text-xs rounded transition-colors ${
                  activeChartTab === "imu"
                    ? "bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm"
                    : "text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
                }`}
                data-testid="chart-tab-imu"
              >
                IMU
              </button>
            )}
          </div>

          {/* Chart Content */}
          {activeChartTab === "actions" && hasActions && (
            <ActionsChart
              episodeId={episodeId}
              datasetId={datasetId}
              currentFrame={currentFrame}
              totalFrames={effectiveTotalFrames}
            />
          )}
          {activeChartTab === "imu" && hasImu && (
            <IMUChart
              episodeId={episodeId}
              datasetId={datasetId}
              currentFrame={currentFrame}
              totalFrames={effectiveTotalFrames}
            />
          )}
        </div>
      )}

      {/* Timeline Controls */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
        {/* Enhanced Timeline with Event Markers */}
        <div className={`mb-3 ${controlsDisabled ? "pointer-events-none opacity-50" : ""}`}>
          <EnhancedTimeline
            currentFrame={currentFrame}
            totalFrames={effectiveTotalFrames}
            events={qualityEventsData?.events || []}
            onFrameChange={handleFrameChange}
          />
        </div>

        {/* Playback Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {/* Play/Pause Button */}
            <button
              onClick={togglePlayback}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              data-testid="play-pause-btn"
            >
              {playing ? (
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                </svg>
              ) : (
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              )}
            </button>

            {/* Step Backward */}
            <button
              onClick={() => handleFrameChange(Math.max(0, currentFrame - 1))}
              disabled={controlsDisabled || currentFrame === 0}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors disabled:opacity-30"
              data-testid="step-back-btn"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 6h2v12H6zm3.5 6l8.5 6V6z" />
              </svg>
            </button>

            {/* Step Forward */}
            <button
              onClick={() => handleFrameChange(Math.min(effectiveTotalFrames - 1, currentFrame + 1))}
              disabled={controlsDisabled || currentFrame >= effectiveTotalFrames - 1}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors disabled:opacity-30"
              data-testid="step-forward-btn"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z" />
              </svg>
            </button>
          </div>

          {/* Playback Speed */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">Speed:</span>
            <select
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
              disabled={controlsDisabled}
              className="text-sm bg-gray-100 dark:bg-gray-800 rounded px-2 py-1 disabled:opacity-50"
              data-testid="speed-select"
            >
              <option value={0.25}>0.25x</option>
              <option value={0.5}>0.5x</option>
              <option value={1}>1x</option>
              <option value={2}>2x</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}
