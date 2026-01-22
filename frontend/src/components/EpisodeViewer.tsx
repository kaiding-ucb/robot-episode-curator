"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { useFrames } from "@/hooks/useApi";
import type { StreamingOptions } from "@/types/api";
import { useQualityEvents } from "@/hooks/useQuality";
import EnhancedTimeline from "./EnhancedTimeline";

interface EpisodeViewerProps {
  datasetId: string | null;
  episodeId: string | null;
  totalFrames: number;
  targetFrame?: number | null;
  onFrameChange?: (frame: number) => void;
  selectedMetric?: string | null;
}

export default function EpisodeViewer({
  datasetId,
  episodeId,
  totalFrames,
  targetFrame,
  onFrameChange,
  selectedMetric,
}: EpisodeViewerProps) {
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  // Always use lowest resolution for fast loading
  const streamingOptions: StreamingOptions = useMemo(() => ({
    resolution: "low",  // 320x240
    quality: 30,        // Low quality for speed
  }), []);

  // Fetch frames around the current position
  // Larger batches = fewer boundary transitions = smoother playback
  const batchSize = 750; // ~25 seconds at 30fps - covers most episodes in 1-2 batches
  const batchStart = Math.floor(currentFrame / batchSize) * batchSize;
  const {
    frames,
    loading,
    error,
    loadedRange,
    totalFrames: apiTotalFrames,
    previousBatch,
    prefetch,
    clearPreviousBatch,
  } = useFrames(
    episodeId,
    batchStart,
    batchStart + batchSize,
    datasetId,
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
      clearPreviousBatch(); // Clear fallback on manual seek
      onFrameChange?.(targetFrame);
    }
  }, [targetFrame, clearPreviousBatch]);

  // Pre-fetch next batch when approaching boundary
  // Trigger earlier to ensure prefetch completes before boundary
  useEffect(() => {
    if (!playing || !episodeId) return;
    const positionInBatch = currentFrame - batchStart;
    if (positionInBatch >= batchSize - 500) { // 500 frames before boundary (~17s at 30fps)
      prefetch(batchStart + batchSize, batchStart + batchSize * 2);
    }
  }, [currentFrame, playing, batchStart, batchSize, prefetch, episodeId]);

  // Get the frame to display from the loaded batch
  // Priority: 1) Current batch if ready, 2) Current frames if they cover the frame, 3) Previous batch fallback
  const isBatchReady = loadedRange && loadedRange.start === batchStart;
  let displayFrame = null;
  let isShowingFallback = false;

  if (isBatchReady) {
    // Add bounds check to prevent accessing out-of-bounds frame index
    const frameIndex = currentFrame - batchStart;
    displayFrame = frameIndex >= 0 && frameIndex < frames.length ? frames[frameIndex] : null;
  } else if (loadedRange && frames.length > 0) {
    // During loading/transition: use current frames if they cover the requested frame
    // This handles resolution changes (play/pause) where the batch range is same but quality differs
    const frameIndex = currentFrame - loadedRange.start;
    if (frameIndex >= 0 && frameIndex < frames.length) {
      displayFrame = frames[frameIndex];
      isShowingFallback = true;
    }
  }

  // If still no frame, try previous batch - but only if currentFrame is within its range
  if (!displayFrame && previousBatch?.episodeId === episodeId &&
             currentFrame >= previousBatch.range.start &&
             currentFrame < previousBatch.range.end) {
    // Find the correct frame from previousBatch by index
    const prevFrameIndex = currentFrame - previousBatch.range.start;
    if (prevFrameIndex >= 0 && prevFrameIndex < previousBatch.frames.length) {
      displayFrame = previousBatch.frames[prevFrameIndex];
      isShowingFallback = true;
    }
  }

  // Track frame availability and loading state in refs so the interval callback can check them
  const hasFrameRef = useRef(false);
  const isLoadingRef = useRef(false);
  hasFrameRef.current = displayFrame !== null || isShowingFallback;
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

  if (loading && frames.length === 0) {
    return (
      <div
        className="flex items-center justify-center h-full text-gray-500"
        data-testid="loading-frames"
      >
        Loading frames...
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
        {displayFrame ? (
          <img
            key={`frame-${currentFrame}`}
            src={`data:image/webp;base64,${displayFrame.image}`}
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

        {/* Action overlay (if available) */}
        {displayFrame?.action && (
          <div className="absolute bottom-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded font-mono">
            Action: [{displayFrame.action.map((a) => a.toFixed(2)).join(", ")}]
          </div>
        )}
      </div>

      {/* Timeline Controls */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
        {/* Enhanced Timeline with Event Markers */}
        <div className="mb-3">
          <EnhancedTimeline
            currentFrame={currentFrame}
            totalFrames={effectiveTotalFrames}
            events={qualityEventsData?.events || []}
            onFrameChange={handleFrameChange}
            selectedMetric={selectedMetric}
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
