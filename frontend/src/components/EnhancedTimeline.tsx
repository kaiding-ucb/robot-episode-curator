"use client";

import { useCallback, useMemo } from "react";
import type { QualityEvent } from "@/types/quality";

interface EnhancedTimelineProps {
  currentFrame: number;
  totalFrames: number;
  events: QualityEvent[];
  onFrameChange: (frame: number) => void;
  onEventClick?: (event: QualityEvent) => void;
  selectedMetric?: string | null;
  /** Per-frame divergence values for heat track (0-1 scale) */
  divergenceScores?: number[];
  /** Whether to show the divergence heat track */
  showDivergenceHeat?: boolean;
}

// Event colors with two categories:
// - Divergence events (warm colors - orange)
// - Transition events (cool colors - green, purple, cyan, etc.)
const EVENT_COLORS: Record<string, { bg: string; border: string }> = {
  // Transition events (Physics Test)
  gripper: { bg: "bg-green-500", border: "border-green-600" },
  pause: { bg: "bg-yellow-500", border: "border-yellow-600" },
  direction_change: { bg: "bg-cyan-500", border: "border-cyan-600" },
  high_jerk: { bg: "bg-red-500", border: "border-red-600" },
  correction: { bg: "bg-purple-500", border: "border-purple-600" },
  recovery: { bg: "bg-purple-500", border: "border-purple-600" },
  near_miss: { bg: "bg-pink-500", border: "border-pink-600" },
  // Divergence events (Expertise Test)
  high_divergence: { bg: "bg-orange-500", border: "border-orange-600" },
};

/** Divergence heat track component - shows where episode diverges from task median */
function DivergenceHeatTrack({
  divergenceScores,
  totalFrames,
}: {
  divergenceScores: number[];
  totalFrames: number;
}) {
  if (!divergenceScores || divergenceScores.length === 0) return null;

  // Downsample for performance if too many frames
  const maxSegments = 100;
  const step = Math.max(1, Math.floor(divergenceScores.length / maxSegments));
  const sampledScores: number[] = [];
  for (let i = 0; i < divergenceScores.length; i += step) {
    // Take max in each segment for visibility
    const end = Math.min(i + step, divergenceScores.length);
    const segment = divergenceScores.slice(i, end);
    sampledScores.push(Math.max(...segment));
  }

  return (
    <div
      className="h-2 w-full rounded mb-1 relative overflow-hidden bg-gray-200 dark:bg-gray-700"
      data-testid="divergence-heat-track"
      title="Divergence from task median (orange = high divergence)"
    >
      {sampledScores.map((score, idx) => {
        const position = (idx / sampledScores.length) * 100;
        const width = (1 / sampledScores.length) * 100 + 0.5; // Slight overlap to avoid gaps
        // Normalize score: 0-0.5 = low, 0.5-1.0 = medium, >1.0 = high
        const intensity = Math.min(1, score / 1.5);

        return (
          <div
            key={idx}
            className="absolute h-full"
            style={{
              left: `${position}%`,
              width: `${width}%`,
              backgroundColor: `rgba(249, 115, 22, ${intensity})`, // Orange with variable opacity
            }}
          />
        );
      })}
    </div>
  );
}

export default function EnhancedTimeline({
  currentFrame,
  totalFrames,
  events,
  onFrameChange,
  onEventClick,
  selectedMetric,
  divergenceScores,
  showDivergenceHeat = true,
}: EnhancedTimelineProps) {
  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onFrameChange(parseInt(e.target.value, 10));
    },
    [onFrameChange]
  );

  const handleTrackClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = x / rect.width;
      const frame = Math.round(percentage * (totalFrames - 1));
      onFrameChange(Math.max(0, Math.min(totalFrames - 1, frame)));
    },
    [totalFrames, onFrameChange]
  );

  const handleMarkerClick = useCallback(
    (event: QualityEvent, e: React.MouseEvent) => {
      e.stopPropagation();
      onFrameChange(event.frame);
      onEventClick?.(event);
    },
    [onFrameChange, onEventClick]
  );

  // Filter to only recovery and high_divergence events, then deduplicate
  const uniqueEvents = useMemo(() => {
    // Only show recovery (purple) and high_divergence (orange) events
    const relevantEvents = events.filter(e =>
      e.event_type === "recovery" || e.event_type === "high_divergence"
    );

    const seen = new Map<number, QualityEvent>();
    for (const event of relevantEvents) {
      // Prefer recovery over divergence at same frame
      const existing = seen.get(event.frame);
      if (!existing || (existing.event_type === "high_divergence" && event.event_type === "recovery")) {
        seen.set(event.frame, event);
      }
    }
    return Array.from(seen.values());
  }, [events]);

  return (
    <div className="relative" data-testid="enhanced-timeline">
      {/* Divergence heat track (above main timeline) */}
      {showDivergenceHeat && divergenceScores && divergenceScores.length > 0 && (
        <DivergenceHeatTrack divergenceScores={divergenceScores} totalFrames={totalFrames} />
      )}

      {/* Event markers track */}
      <div
        className="relative h-6 mb-1 cursor-pointer"
        onClick={handleTrackClick}
        data-testid="timeline-track"
      >
        {/* Background track */}
        <div className="absolute top-1/2 left-0 right-0 h-1 bg-gray-300 dark:bg-gray-600 rounded-full transform -translate-y-1/2" />

        {/* Progress indicator */}
        <div
          className="absolute top-1/2 left-0 h-1 bg-blue-500 rounded-full transform -translate-y-1/2"
          style={{ width: `${(currentFrame / Math.max(1, totalFrames - 1)) * 100}%` }}
        />

        {/* Event markers */}
        {uniqueEvents.map((event, idx) => {
          const position = (event.frame / Math.max(1, totalFrames - 1)) * 100;
          const colors = EVENT_COLORS[event.event_type] || EVENT_COLORS.direction_change;

          // Check if this event matches the selected metric
          const matchesMetric = !selectedMetric ||
            (event.affected_metrics && event.affected_metrics.includes(selectedMetric));
          const isHighlighted = selectedMetric && matchesMetric;
          const isFaded = selectedMetric && !matchesMetric;

          return (
            <div
              key={`${event.frame}-${event.event_type}-${idx}`}
              className={`absolute top-1/2 transform -translate-x-1/2 -translate-y-1/2 cursor-pointer group z-10
                ${isFaded ? 'opacity-30' : ''} ${isHighlighted ? 'z-20' : ''}`}
              style={{ left: `${position}%` }}
              onClick={(e) => handleMarkerClick(event, e)}
              data-testid={`event-marker-${event.frame}`}
              data-matches-metric={matchesMetric ? 'true' : 'false'}
            >
              {/* Marker dot */}
              <div
                className={`w-3 h-3 rounded-full ${colors.bg} border-2 ${colors.border}
                  hover:scale-150 transition-all shadow-sm
                  ${isHighlighted ? 'ring-2 ring-blue-400 ring-offset-1 scale-125' : ''}`}
              />

              {/* Tooltip on hover */}
              <div
                className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2
                  opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-20"
              >
                <div className="bg-gray-900 text-white text-xs px-2 py-1 rounded whitespace-nowrap shadow-lg max-w-xs">
                  <div className="font-semibold capitalize">
                    {event.event_type === "high_divergence" ? "High Divergence" : "Recovery"}
                  </div>
                  <div>Frame {event.frame}</div>
                  {/* Show description for high_divergence (contains dimension info) */}
                  {event.description && (
                    <div className={`mt-1 ${event.event_type === "high_divergence" ? "text-orange-300" : "text-purple-300"}`}>
                      {event.description}
                    </div>
                  )}
                </div>
                {/* Tooltip arrow */}
                <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900" />
              </div>
            </div>
          );
        })}

        {/* Current position indicator */}
        <div
          className="absolute top-1/2 transform -translate-x-1/2 -translate-y-1/2 w-4 h-4
            bg-white border-2 border-blue-500 rounded-full shadow-md z-20"
          style={{ left: `${(currentFrame / Math.max(1, totalFrames - 1)) * 100}%` }}
        />
      </div>

      {/* Hidden range input for accessibility and keyboard control */}
      <input
        type="range"
        min={0}
        max={Math.max(0, totalFrames - 1)}
        value={currentFrame}
        onChange={handleSliderChange}
        className="w-full h-2 opacity-0 absolute top-0 cursor-pointer"
        data-testid="timeline-slider"
        aria-label="Video timeline"
      />

      {/* Metric filter indicator */}
      {selectedMetric && (
        <div className="text-xs text-blue-600 dark:text-blue-400 mt-1" data-testid="metric-filter-indicator">
          Filtering by: <span className="font-medium capitalize">{selectedMetric.replace(/_/g, ' ')}</span>
          <span className="text-gray-400 ml-1">
            ({uniqueEvents.filter(e => e.affected_metrics?.includes(selectedMetric)).length} events)
          </span>
        </div>
      )}

      {/* Simplified event legend */}
      {(uniqueEvents.length > 0 || (showDivergenceHeat && divergenceScores?.length)) && !selectedMetric && (
        <div className="flex items-center gap-4 mt-1 text-xs text-gray-500">
          {/* Divergence */}
          {showDivergenceHeat && divergenceScores && divergenceScores.length > 0 && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-2 rounded bg-gradient-to-r from-orange-200 to-orange-500" />
              <span className="text-orange-600 dark:text-orange-400">Divergence</span>
            </div>
          )}

          {/* Recovery */}
          {uniqueEvents.some(e => e.event_type === "recovery") && (
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-purple-500" />
              <span className="text-purple-600 dark:text-purple-400">Recovery</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
