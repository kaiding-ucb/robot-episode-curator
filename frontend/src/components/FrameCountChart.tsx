"use client";

import { useMemo } from "react";
import type { FrameCountDistribution } from "@/types/analysis";

interface FrameCountChartProps {
  data: FrameCountDistribution;
}

interface HistogramBin {
  rangeStart: number;
  rangeEnd: number;
  count: number;
  episodes: string[];
  hasOutlier: boolean;
}

function buildHistogram(
  data: FrameCountDistribution,
  numBins: number = 30
): HistogramBin[] {
  if (!data.episodes.length) return [];

  const outlierSet = new Set(data.outlier_episode_ids);

  // Separate outlier and non-outlier frames to compute a meaningful bin range
  const mainEpisodes = data.episodes.filter((ep) => !outlierSet.has(ep.episode_id));
  const mainFrames = mainEpisodes.map((ep) => ep.estimated_frames);

  // Fall back to all frames if every episode is an outlier (or no outliers detected)
  const rangeFrames = mainFrames.length > 0 ? mainFrames : data.episodes.map((ep) => ep.estimated_frames);
  const min = Math.min(...rangeFrames);
  const max = Math.max(...rangeFrames);
  const range = max - min || 1;
  const binWidth = range / numBins;

  const bins: HistogramBin[] = [];
  for (let i = 0; i < numBins; i++) {
    bins.push({
      rangeStart: Math.round(min + i * binWidth),
      rangeEnd: Math.round(min + (i + 1) * binWidth),
      count: 0,
      episodes: [],
      hasOutlier: false,
    });
  }

  for (const ep of data.episodes) {
    // Clamp outliers to first/last bin so they still appear in the chart
    const rawIdx = Math.floor((ep.estimated_frames - min) / binWidth);
    const idx = Math.max(0, Math.min(rawIdx, numBins - 1));
    bins[idx].count++;
    bins[idx].episodes.push(ep.file_name);
    if (outlierSet.has(ep.episode_id)) {
      bins[idx].hasOutlier = true;
    }
  }

  return bins;
}

export default function FrameCountChart({ data }: FrameCountChartProps) {
  const { bins, maxCount, meanBinX, stdLeftX, stdRightX } = useMemo(() => {
    if (!data.episodes.length) {
      return { bins: [], maxCount: 0, meanBinX: 0, stdLeftX: 0, stdRightX: 0 };
    }

    const bins = buildHistogram(data, 40);
    const maxCount = Math.max(...bins.map((b) => b.count), 1);

    // Mean position as fraction of chart width
    const min = bins[0].rangeStart;
    const max = bins[bins.length - 1].rangeEnd;
    const range = max - min || 1;
    const meanBinX = ((data.mean_frames - min) / range) * 100;
    const stdLeftX = Math.max(0, ((data.mean_frames - data.std_frames - min) / range) * 100);
    const stdRightX = Math.min(100, ((data.mean_frames + data.std_frames + min) / range) * 100);

    return { bins, maxCount, meanBinX, stdLeftX, stdRightX };
  }, [data]);

  const chartWidth = 700;
  const chartHeight = 220;
  const padding = { top: 20, right: 20, bottom: 50, left: 50 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;
  const barWidth = bins.length > 0 ? innerWidth / bins.length : 0;

  // Y-axis tick values
  const yTicks = useMemo(() => {
    const ticks: number[] = [];
    const step = Math.max(1, Math.ceil(maxCount / 5));
    for (let v = 0; v <= maxCount; v += step) {
      ticks.push(v);
    }
    if (ticks[ticks.length - 1] < maxCount) ticks.push(maxCount);
    return ticks;
  }, [maxCount]);

  // X-axis tick values (frame counts)
  const xTicks = useMemo(() => {
    if (bins.length === 0) return [];
    const min = bins[0].rangeStart;
    const max = bins[bins.length - 1].rangeEnd;
    const range = max - min;
    const step = Math.max(1, Math.round(range / 6));
    const ticks: number[] = [];
    for (let v = min; v <= max; v += step) {
      ticks.push(v);
    }
    return ticks;
  }, [bins]);

  if (!data.episodes.length) {
    return (
      <div className="text-sm text-gray-500 text-center py-8">
        No episodes found in this task.
      </div>
    );
  }

  // Mean and std positions in pixel space, clamped to chart bounds
  const minFrame = bins[0].rangeStart;
  const maxFrame = bins[bins.length - 1].rangeEnd;
  const frameRange = maxFrame - minFrame || 1;
  const meanXRaw = ((data.mean_frames - minFrame) / frameRange) * innerWidth;
  const meanX = padding.left + Math.max(0, Math.min(innerWidth, meanXRaw));
  const meanOffChart = data.mean_frames < minFrame || data.mean_frames > maxFrame;
  const stdLeftPx = padding.left + Math.max(0, ((data.mean_frames - data.std_frames - minFrame) / frameRange) * innerWidth);
  const stdRightPx = padding.left + Math.min(innerWidth, ((data.mean_frames + data.std_frames - minFrame) / frameRange) * innerWidth);

  return (
    <div data-testid="frame-count-chart">
      {/* Stats summary */}
      <div className="flex flex-wrap gap-4 mb-3 text-sm">
        <span className="px-2 py-1 bg-blue-50 dark:bg-blue-900/30 rounded text-blue-700 dark:text-blue-300">
          {data.total_episodes} episodes
        </span>
        <span className="px-2 py-1 bg-gray-50 dark:bg-gray-800 rounded text-gray-600 dark:text-gray-300">
          Mean: {Math.round(data.mean_frames)} frames
        </span>
        <span className="px-2 py-1 bg-gray-50 dark:bg-gray-800 rounded text-gray-600 dark:text-gray-300">
          Std: {Math.round(data.std_frames)}
        </span>
        {data.outlier_episode_ids.length > 0 && (
          <span className="px-2 py-1 bg-amber-50 dark:bg-amber-900/30 rounded text-amber-700 dark:text-amber-300">
            {data.outlier_episode_ids.length} outlier{data.outlier_episode_ids.length !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Histogram chart */}
      <div className="overflow-x-auto">
        <svg
          width={chartWidth}
          height={chartHeight}
          className="bg-gray-50 dark:bg-gray-900/50 rounded"
        >
          {/* Std deviation band */}
          <rect
            x={stdLeftPx}
            y={padding.top}
            width={stdRightPx - stdLeftPx}
            height={innerHeight}
            fill="#3b82f6"
            opacity="0.06"
          />

          {/* Y-axis gridlines */}
          {yTicks.map((tick) => {
            const y = padding.top + innerHeight - (tick / maxCount) * innerHeight;
            return (
              <g key={`y-${tick}`}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={padding.left + innerWidth}
                  y2={y}
                  stroke="currentColor"
                  opacity="0.08"
                  strokeWidth="1"
                />
                <text
                  x={padding.left - 6}
                  y={y + 3}
                  textAnchor="end"
                  fontSize="9"
                  fill="currentColor"
                  opacity="0.4"
                >
                  {tick}
                </text>
              </g>
            );
          })}

          {/* Histogram bars */}
          <g>
            {bins.map((bin, i) => {
              const barH = maxCount > 0 ? (bin.count / maxCount) * innerHeight : 0;
              const x = padding.left + i * barWidth;
              const y = padding.top + innerHeight - barH;

              return (
                <g key={i}>
                  <title>
                    {bin.rangeStart}–{bin.rangeEnd} frames: {bin.count} episode{bin.count !== 1 ? "s" : ""}
                    {bin.hasOutlier ? " (contains outliers)" : ""}
                  </title>
                  <rect
                    x={x + 0.5}
                    y={y}
                    width={Math.max(barWidth - 1, 1)}
                    height={Math.max(barH, bin.count > 0 ? 1 : 0)}
                    rx="1"
                    fill={bin.hasOutlier ? "#f59e0b" : "#3b82f6"}
                    opacity={bin.hasOutlier ? 0.9 : 0.7}
                  />
                </g>
              );
            })}
          </g>

          {/* Mean line */}
          <line
            x1={meanX}
            y1={padding.top}
            x2={meanX}
            y2={padding.top + innerHeight}
            stroke="#ef4444"
            strokeWidth="1.5"
            strokeDasharray="4,3"
            opacity={meanOffChart ? 0.4 : 0.8}
          />
          <text
            x={meanX}
            y={padding.top - 6}
            textAnchor={data.mean_frames > maxFrame ? "end" : data.mean_frames < minFrame ? "start" : "middle"}
            fontSize="9"
            fill="#ef4444"
            fontWeight="500"
          >
            {data.mean_frames > maxFrame ? "mean →" : data.mean_frames < minFrame ? "← mean" : "mean"} ({Math.round(data.mean_frames)})
          </text>

          {/* X-axis labels */}
          {xTicks.map((tick) => {
            const x = padding.left + ((tick - minFrame) / frameRange) * innerWidth;
            return (
              <text
                key={`x-${tick}`}
                x={x}
                y={chartHeight - padding.bottom + 16}
                textAnchor="middle"
                fontSize="9"
                fill="currentColor"
                opacity="0.4"
              >
                {tick}
              </text>
            );
          })}

          {/* Axis labels */}
          <text
            x={padding.left + innerWidth / 2}
            y={chartHeight - 6}
            textAnchor="middle"
            fontSize="10"
            fill="currentColor"
            opacity="0.5"
          >
            Estimated Frame Count
          </text>
          <text
            x={14}
            y={padding.top + innerHeight / 2}
            textAnchor="middle"
            fontSize="10"
            fill="currentColor"
            opacity="0.5"
            transform={`rotate(-90, 14, ${padding.top + innerHeight / 2})`}
          >
            Episodes
          </text>
        </svg>
      </div>

      {/* Outlier list */}
      {data.outlier_episode_ids.length > 0 && (
        <div className="mt-3 space-y-1">
          <div className="text-xs font-medium text-amber-600 dark:text-amber-400">
            Outliers (beyond 2 std):
          </div>
          {data.episodes
            .filter((ep) => data.outlier_episode_ids.includes(ep.episode_id))
            .slice(0, 10)
            .map((ep) => {
              const deviation = data.mean_frames > 0
                ? Math.round(((ep.estimated_frames - data.mean_frames) / data.mean_frames) * 100)
                : 0;
              return (
                <div
                  key={ep.episode_id}
                  className="text-xs px-2 py-1 bg-amber-50 dark:bg-amber-900/20 rounded flex justify-between"
                >
                  <span className="text-gray-700 dark:text-gray-300 font-mono truncate">
                    {ep.file_name}
                  </span>
                  <span className="text-amber-600 dark:text-amber-400 ml-2 whitespace-nowrap">
                    ~{ep.estimated_frames} frames ({deviation > 0 ? "+" : ""}
                    {deviation}% from mean)
                  </span>
                </div>
              );
            })}
          {data.outlier_episode_ids.length > 10 && (
            <div className="text-xs text-gray-400 italic">
              ...and {data.outlier_episode_ids.length - 10} more outliers
            </div>
          )}
        </div>
      )}

      {/* Source note */}
      <div className="mt-2 text-xs text-gray-400 italic">
        {data.source_note || "Frame counts estimated from file sizes (~50KB/frame). A well-curated dataset should approximate a normal distribution."}
      </div>
    </div>
  );
}
