"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { useStreamingFrames, getFrame } from "@/hooks/useStreamingFrames";
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
  const [activeStream, setActiveStream] = useState<"rgb" | "depth">("rgb");
  const [activeChartTab, setActiveChartTab] = useState<"none" | "actions" | "imu">("none");

  // Check available modalities
  const hasDepth = availableModalities.includes("depth");
  const hasImu = availableModalities.includes("imu");
  const hasActions = availableModalities.includes("actions");

  // Always use lowest resolution for fast loading
  const streamingOptions: StreamingOptions = useMemo(() => ({
    resolution: "low",  // 320x240
    quality: 30,        // Low quality for speed
    stream: activeStream,
  }), [activeStream]);

  // Use SSE streaming for progressive frame loading
  // Frames appear as they're decoded (~2-3s for first frame) rather than waiting for full batch
  const batchSize = 750; // ~25 seconds at 30fps
  const batchStart = Math.floor(currentFrame / batchSize) * batchSize;
  const {
    frames,
    totalFrames: apiTotalFrames,
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
    streamingOptions
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
  // Frames are added progressively as they arrive via SSE
  const displayFrame = getFrame(frames, currentFrame);

  // For streaming: show last available frame if current frame hasn't arrived yet
  const lastLoadedFrame = useMemo(() => {
    if (frames.size === 0) return null;
    // Find the highest frame index that's <= currentFrame
    let best = null;
    for (const [idx, frame] of frames) {
      if (idx <= currentFrame && (best === null || idx > best)) {
        best = idx;
      }
    }
    return best !== null ? frames.get(best) : null;
  }, [frames, currentFrame]);

  // Use current frame if available, otherwise fallback to last loaded frame
  const frameToShow = displayFrame || lastLoadedFrame;
  const isShowingFallback = !displayFrame && lastLoadedFrame !== null;


  // Track frame availability and loading state in refs so the interval callback can check them
  const hasFrameRef = useRef(false);
  const isLoadingRef = useRef(false);
  hasFrameRef.current = frameToShow !== null;
  isLoadingRef.current = loading;

  // Playback logic - continues during loading, uses fallback frames
  useEffect(() => {
    if (!playing) return;

    // Don't start playback if we don't have any frame to display
    if (!hasFrameRef.current) {
      return;
    }

    const interval = setInterval(() => {
      // Skip tick if no frame available (fallback will show last known frame)
      if (!hasFrameRef.current) {
        return;
      }

      setCurrentFrame((prev) => {
        if (prev >= effectiveTotalFrames - 1) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1000 / (30 * playbackSpeed)); // Assume 30fps base

    return () => clearInterval(interval);
  }, [playing, effectiveTotalFrames, playbackSpeed]); // Removed 'loading' dependency

  // Reset when episode changes
  useEffect(() => {
    setCurrentFrame(0);
    setPlaying(false);
    setActiveStream("rgb"); // Reset to RGB when episode changes
  }, [episodeId]);

  const handleFrameChange = useCallback((frame: number) => {
    setCurrentFrame(frame);
    setPlaying(false);
    // Don't clear previous batch - we need it for display during batch transitions
    onFrameChange?.(frame);
  }, [onFrameChange]);

  const togglePlayback = useCallback(() => {
    setPlaying((prev) => !prev);
  }, []);

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

  // Only show full loading screen if we have NO frames at all
  // If we have previous frames, show them with a loading overlay instead
  const showFullLoadingScreen = loading && frames.size === 0 && !frameToShow;

  if (showFullLoadingScreen) {
    return (
      <div
        className="flex items-center justify-center h-full text-gray-500"
        data-testid="loading-frames"
      >
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
          <p>Loading frames...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div
        className="flex items-center justify-center h-full text-red-500"
        data-testid="error-frames"
      >
        Error: {error}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full" data-testid="episode-viewer">
      {/* Video Display */}
      <div className="flex-1 min-h-0 bg-black flex items-center justify-center relative overflow-hidden">
        {frameToShow ? (
          <img
            key={`frame-${frameToShow.index}`}
            src={`data:image/webp;base64,${frameToShow.image}`}
            alt={`Frame ${currentFrame}`}
            className="max-w-full max-h-full object-contain"
            data-testid="frame-image"
          />
        ) : (
          <div className="text-gray-500">No frame available</div>
        )}

        {/* Frame info overlay */}
        <div className="absolute top-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
          Frame {currentFrame + 1} / {effectiveTotalFrames}
        </div>

        {/* Streaming progress indicator */}
        {loading && frames.size > 0 && (
          <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded flex items-center gap-2">
            <div className="w-24 h-1 bg-gray-600 rounded overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-200"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
            <span>{Math.round(progress * 100)}%</span>
          </div>
        )}

        {/* Stream toggle - only if depth is available */}
        {hasDepth && (
          <div className="absolute top-2 right-2 flex gap-1">
            <button
              onClick={() => setActiveStream("rgb")}
              className={`px-3 py-1 text-xs rounded-l transition-colors ${
                activeStream === "rgb"
                  ? "bg-blue-500 text-white"
                  : "bg-black/50 text-gray-300 hover:bg-black/70"
              }`}
              data-testid="stream-rgb-btn"
            >
              RGB
            </button>
            <button
              onClick={() => setActiveStream("depth")}
              className={`px-3 py-1 text-xs rounded-r transition-colors ${
                activeStream === "depth"
                  ? "bg-purple-500 text-white"
                  : "bg-black/50 text-gray-300 hover:bg-black/70"
              }`}
              data-testid="stream-depth-btn"
            >
              Depth
            </button>
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
        <div className="mb-3">
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
              disabled={currentFrame === 0}
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
              disabled={currentFrame >= effectiveTotalFrames - 1}
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
              className="text-sm bg-gray-100 dark:bg-gray-800 rounded px-2 py-1"
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
