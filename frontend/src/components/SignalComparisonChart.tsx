"use client";

import { useMemo, useState, useEffect, useRef, useCallback } from "react";
import type { EpisodeSignalData, EpisodeStub } from "@/types/analysis";
import { classifyActionDimensions, type ActionGrouping } from "@/utils/actionClassification";

interface SignalComparisonChartProps {
  episodes: Map<string, EpisodeSignalData>;
  datasetId: string | null;
  firstFrames?: Map<string, string>;
  knownEpisodes?: EpisodeStub[];
  onNavigateToEpisode?: (datasetId: string, episodeId: string, numFrames: number, targetFrame?: number) => void;
}

function getEpisodeLabel(episodeId: string): string {
  const parts = episodeId.split("/");
  const last = parts[parts.length - 1];
  return last.replace(/\.[^.]+$/, "");
}

// Normalize data using provided min/max (batch-level)
function normalizeBatch(data: number[], min: number, max: number): number[] {
  if (data.length === 0) return [];
  const range = max - min || 1;
  return data.map((v) => (v - min) / range);
}

// Self-normalize (for overlay where each trace is independent)
function normalizeSelf(data: number[]): number[] {
  if (data.length === 0) return [];
  const min = Math.min(...data);
  const max = Math.max(...data);
  return normalizeBatch(data, min, max);
}

// Create SVG path from already-normalized (0-1) data points
function createPath(
  normalized: number[],
  height: number,
  width: number,
  yPadding: number = 2
): string {
  if (normalized.length === 0) return "";
  const usableHeight = height - yPadding * 2;
  const xStep = width / Math.max(normalized.length - 1, 1);

  return normalized
    .map((y, i) => {
      const clamped = Math.max(0, Math.min(1, y));
      const yPos = yPadding + usableHeight - clamped * usableHeight;
      return `${i === 0 ? "M" : "L"} ${i * xStep} ${yPos}`;
    })
    .join(" ");
}

// Compute magnitude from multi-dimensional action data
function computeMagnitude(actions: number[][], indices: number[]): number[] {
  return actions.map((a) => {
    let sum = 0;
    for (const i of indices) {
      if (i < a.length) sum += a[i] * a[i];
    }
    return Math.sqrt(sum);
  });
}

// Compute IMU magnitude from 3 axes
function computeIMUMagnitude(x: number[], y: number[], z: number[]): number[] {
  const len = Math.min(x.length, y.length, z.length);
  const result: number[] = [];
  for (let i = 0; i < len; i++) {
    result.push(Math.sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]));
  }
  return result;
}

// Compute gripper signal (last dimension, typically dim 6)
function computeGripperSignal(actions: number[][], dimIndex: number): number[] {
  return actions.map((a) => (dimIndex < a.length ? a[dimIndex] : 0));
}

// Get robust min/max across multiple arrays using percentiles
// This prevents a single extreme episode from compressing all others
function getBatchRange(arrays: number[][]): { min: number; max: number } {
  const all: number[] = [];
  for (const arr of arrays) {
    for (const v of arr) {
      all.push(v);
    }
  }
  if (all.length === 0) return { min: 0, max: 1 };

  all.sort((a, b) => a - b);

  // Use 2nd and 98th percentile for robust range
  const lo = Math.floor(all.length * 0.02);
  const hi = Math.ceil(all.length * 0.98) - 1;
  const min = all[Math.max(0, lo)];
  const max = all[Math.min(all.length - 1, hi)];

  if (max <= min) return { min: min - 0.5, max: max + 0.5 };
  return { min, max };
}

// Resample a data array to a fixed length using linear interpolation
function resampleToLength(data: number[], targetLen: number): number[] {
  if (data.length === 0) return [];
  if (data.length === 1) return new Array(targetLen).fill(data[0]);
  const result: number[] = [];
  for (let i = 0; i < targetLen; i++) {
    const t = (i / (targetLen - 1)) * (data.length - 1);
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, data.length - 1);
    const frac = t - lo;
    result.push(data[lo] * (1 - frac) + data[hi] * frac);
  }
  return result;
}

// Compute mean ± 2std envelope from multiple resampled traces
function computeEnvelope(traces: number[][]): {
  mean: number[];
  upper: number[];
  lower: number[];
  std: number[];
} {
  if (traces.length === 0) return { mean: [], upper: [], lower: [], std: [] };
  const len = traces[0].length;
  const mean: number[] = [];
  const upper: number[] = [];
  const lower: number[] = [];
  const stdArr: number[] = [];
  for (let i = 0; i < len; i++) {
    let sum = 0;
    for (const t of traces) sum += t[i];
    const mu = sum / traces.length;
    let sqSum = 0;
    for (const t of traces) sqSum += (t[i] - mu) * (t[i] - mu);
    const sigma = Math.sqrt(sqSum / traces.length);
    mean.push(mu);
    stdArr.push(sigma);
    upper.push(mu + 2 * sigma);
    lower.push(mu - 2 * sigma);
  }
  return { mean, upper, lower, std: stdArr };
}

// Compute fraction of resampled trace timesteps outside the envelope
function computeOutlierFraction(
  trace: number[],
  upper: number[],
  lower: number[]
): number {
  if (trace.length === 0) return 0;
  let outside = 0;
  for (let i = 0; i < trace.length; i++) {
    if (trace[i] > upper[i] || trace[i] < lower[i]) outside++;
  }
  return outside / trace.length;
}

const RESAMPLE_LEN = 200;

// Interpolate teal → yellow → red based on ratio 0-1
function bandWidthColor(ratio: number): string {
  const c = Math.max(0, Math.min(1, ratio));
  let r: number, g: number, b: number;
  if (c < 0.5) {
    const t = c * 2;
    r = Math.round(20 + t * (234 - 20));
    g = Math.round(184 + t * (179 - 184));
    b = Math.round(166 + t * (8 - 166));
  } else {
    const t = (c - 0.5) * 2;
    r = Math.round(234 + t * (239 - 234));
    g = Math.round(179 + t * (68 - 179));
    b = Math.round(8 + t * (68 - 8));
  }
  return `rgb(${r},${g},${b})`;
}

interface BandMetrics {
  bandCoverage: number;
  label: "Low Variance" | "Med Variance" | "High Variance";
  color: string;
}

function computeBandMetrics(
  envelope: { mean: number[]; upper: number[]; lower: number[]; std: number[] },
  batchRange: { min: number; max: number }
): BandMetrics {
  const len = envelope.mean.length;
  if (len === 0) return { bandCoverage: 0, label: "Low Variance", color: "#22c55e" };

  let bandWidthSum = 0;
  for (let i = 0; i < len; i++) {
    bandWidthSum += envelope.upper[i] - envelope.lower[i];
  }
  const meanBandWidth = bandWidthSum / len;
  const rangeSpan = batchRange.max - batchRange.min || 1;
  const bandCoverage = meanBandWidth / rangeSpan;

  let label: "Low Variance" | "Med Variance" | "High Variance";
  let color: string;
  if (bandCoverage < 0.25) {
    label = "Low Variance";
    color = "#22c55e";
  } else if (bandCoverage < 0.5) {
    label = "Med Variance";
    color = "#eab308";
  } else {
    label = "High Variance";
    color = "#ef4444";
  }

  return { bandCoverage, label, color };
}

// Colors for different episodes in overlay view
const EPISODE_COLORS = [
  "#3b82f6", // blue
  "#22c55e", // green
  "#f97316", // orange
  "#8b5cf6", // purple
  "#ef4444", // red
  "#06b6d4", // cyan
  "#ec4899", // pink
  "#eab308", // yellow
  "#14b8a6", // teal
  "#f43f5e", // rose
];

// Precomputed magnitude arrays per episode (computed once, reused everywhere)
interface PrecomputedEpisodeSignals {
  position: number[];   // group1 magnitude (position OR left arm OR joints 0-N/2)
  rotation: number[];   // group2 magnitude (rotation OR right arm OR joints N/2-M)
  gripper: number[];    // first gripper signal
  gripper2: number[];   // second gripper signal (dual-arm only)
  accel: number[];
  gyro: number[];
}

// Batch normalization ranges for per-episode charts
interface BatchRanges {
  position: { min: number; max: number };
  rotation: { min: number; max: number };
  gripper: { min: number; max: number };
  gripper2: { min: number; max: number };
  accel: { min: number; max: number };
  gyro: { min: number; max: number };
}

interface SignalTrace {
  label: string;
  data: number[];       // raw data
  normalized: number[]; // batch-normalized 0-1
  color: string;
  dashed?: boolean;
}

const CHART_HEIGHT = 48;
const CHART_WIDTH = 400;

function SignalChart({
  traces,
  height = CHART_HEIGHT,
  width = CHART_WIDTH,
}: {
  traces: SignalTrace[];
  height?: number;
  width?: number;
}) {
  const validTraces = traces.filter((t) => t.normalized.length > 0);
  if (validTraces.length === 0) {
    return (
      <div
        className="bg-gray-100 dark:bg-gray-700 rounded flex items-center justify-center"
        style={{ height, minWidth: width }}
      >
        <span className="text-[10px] text-gray-400">No data</span>
      </div>
    );
  }

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="bg-gray-100 dark:bg-gray-700/50 rounded"
      style={{ height, minWidth: width, width: "100%" }}
      preserveAspectRatio="none"
    >
      {validTraces.map((trace, i) => (
        <path
          key={`${trace.label}-${i}`}
          d={createPath(trace.normalized, height, width)}
          fill="none"
          stroke={trace.color}
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
          opacity="0.85"
          strokeDasharray={trace.dashed ? "3,2" : undefined}
        />
      ))}
    </svg>
  );
}

interface EpisodeChartRowProps {
  episodeId: string;
  episodeData: EpisodeSignalData;
  batchRanges: BatchRanges;
  precomputedSignals?: PrecomputedEpisodeSignals;
  classification?: ActionGrouping;
}

function EpisodeChartRow({ episodeId, episodeData, batchRanges, precomputedSignals, classification }: EpisodeChartRowProps) {
  const { actionTraces, imuTraces } = useMemo(() => {
    const actionTraces: SignalTrace[] = [];
    const imuTraces: SignalTrace[] = [];

    if (precomputedSignals) {
      // Use precomputed magnitude arrays — no recomputation
      if (precomputedSignals.position.length > 0) {
        actionTraces.push({
          label: classification?.group1.label ?? "Position",
          data: precomputedSignals.position,
          normalized: normalizeBatch(precomputedSignals.position, batchRanges.position.min, batchRanges.position.max),
          color: classification?.group1.color ?? "#3b82f6",
        });
      }
      if (precomputedSignals.rotation.length > 0) {
        actionTraces.push({
          label: classification?.group2.label ?? "Rotation",
          data: precomputedSignals.rotation,
          normalized: normalizeBatch(precomputedSignals.rotation, batchRanges.rotation.min, batchRanges.rotation.max),
          color: classification?.group2.color ?? "#8b5cf6",
        });
      }
      if (precomputedSignals.gripper.length > 0) {
        const gripLabel = classification?.grippers[0]?.label ?? "Gripper";
        actionTraces.push({
          label: gripLabel,
          data: precomputedSignals.gripper,
          normalized: normalizeBatch(precomputedSignals.gripper, batchRanges.gripper.min, batchRanges.gripper.max),
          color: classification?.grippers[0]?.color ?? "#22c55e",
          dashed: true,
        });
      }
      if (precomputedSignals.gripper2.length > 0) {
        const gripLabel = classification?.grippers[1]?.label ?? "Gripper 2";
        actionTraces.push({
          label: gripLabel,
          data: precomputedSignals.gripper2,
          normalized: normalizeBatch(precomputedSignals.gripper2, batchRanges.gripper2.min, batchRanges.gripper2.max),
          color: classification?.grippers[1]?.color ?? "#f97316",
          dashed: true,
        });
      }
      if (precomputedSignals.accel.length > 0) {
        imuTraces.push({
          label: "Accel",
          data: precomputedSignals.accel,
          normalized: normalizeBatch(precomputedSignals.accel, batchRanges.accel.min, batchRanges.accel.max),
          color: "#3b82f6",
        });
      }
      if (precomputedSignals.gyro.length > 0) {
        imuTraces.push({
          label: "Gyro",
          data: precomputedSignals.gyro,
          normalized: normalizeBatch(precomputedSignals.gyro, batchRanges.gyro.min, batchRanges.gyro.max),
          color: "#f97316",
        });
      }
    }

    return { actionTraces, imuTraces };
  }, [precomputedSignals, batchRanges, classification]);

  const hasError = episodeData.actions?.error || episodeData.imu?.error;
  const frameCount = episodeData.total_frames ?? episodeData.actions?.actions?.length ?? 0;
  const stride = episodeData.signal_stride ?? 1;
  const rawCount = episodeData.raw_action_count;

  return (
    <div className="flex items-stretch gap-3 py-2 border-b border-gray-100 dark:border-gray-800 last:border-b-0">
      <div className="w-28 flex-shrink-0 flex flex-col justify-center">
        <span className="text-xs font-mono text-gray-700 dark:text-gray-300 truncate">
          {getEpisodeLabel(episodeId)}
        </span>
        <span className="text-[10px] text-gray-400">
          {rawCount && stride > 1
            ? `${frameCount}/${rawCount} (stride ${stride})`
            : frameCount > 0 ? `${frameCount} frames` : ""}
        </span>
        {hasError && (
          <span className="text-[10px] text-red-400">Error</span>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <SignalChart traces={actionTraces} />
      </div>
      <div className="flex-1 min-w-0">
        <SignalChart traces={imuTraces} />
      </div>
    </div>
  );
}

// Detect contiguous high-variance regions from envelope std array
interface HotZone {
  start: number;
  end: number;
  peakVariance: number;
}

function detectHotZones(std: number[]): HotZone[] {
  if (std.length < 20) return [];

  // Normalize std relative to max
  const maxStd = Math.max(...std);
  if (maxStd < 1e-10) return [];
  const normalized = std.map((s) => s / maxStd);

  // Find threshold: top 30th percentile
  const sorted = [...normalized].sort((a, b) => a - b);
  const threshold = sorted[Math.floor(sorted.length * 0.7)];

  // Find contiguous segments above threshold
  const segments: HotZone[] = [];
  let segStart = -1;
  let segPeak = 0;
  for (let i = 0; i < normalized.length; i++) {
    if (normalized[i] >= threshold) {
      if (segStart < 0) segStart = i;
      segPeak = Math.max(segPeak, normalized[i]);
    } else if (segStart >= 0) {
      segments.push({ start: segStart, end: i - 1, peakVariance: segPeak });
      segStart = -1;
      segPeak = 0;
    }
  }
  if (segStart >= 0) {
    segments.push({ start: segStart, end: normalized.length - 1, peakVariance: segPeak });
  }

  // Merge segments with gap < 5
  const merged: HotZone[] = [];
  for (const seg of segments) {
    const last = merged[merged.length - 1];
    if (last && seg.start - last.end < 5) {
      last.end = seg.end;
      last.peakVariance = Math.max(last.peakVariance, seg.peakVariance);
    } else {
      merged.push({ ...seg });
    }
  }

  // Filter out short segments (< 10 indices)
  return merged.filter((s) => s.end - s.start >= 10);
}

// Overlay panel: all episodes on one chart, each a different color
// Uses batch normalization so all traces share the same scale
// Renders a mean ± 2std shaded envelope to highlight outliers
function OverlayPanel({
  title,
  traces,
  batchRange,
  onInspect,
}: {
  title: string;
  traces: { label: string; data: number[]; color: string }[];
  batchRange: { min: number; max: number };
  onInspect?: (resampledIndex: number) => void;
}) {
  const validTraces = traces.filter((t) => t.data.length > 0);

  // Compute envelope, outlier info, band quality metrics, and hot zones
  const { envelope, outliers, bandMetrics, hotZones, resampled } = useMemo(() => {
    if (validTraces.length < 2) return { envelope: null, outliers: [], bandMetrics: null, hotZones: [], resampled: [] };

    // Resample raw data to common length
    const resampled = validTraces.map((t) => resampleToLength(t.data, RESAMPLE_LEN));
    const env = computeEnvelope(resampled);

    // Find outlier episodes (>5% of timesteps outside band)
    const outliers: { label: string; pct: number; maxDeviationIdx: number }[] = [];
    for (let i = 0; i < resampled.length; i++) {
      const frac = computeOutlierFraction(resampled[i], env.upper, env.lower);
      if (frac > 0.05) {
        // Find index of maximum deviation from mean
        let maxDev = 0;
        let maxDevIdx = 0;
        for (let j = 0; j < resampled[i].length; j++) {
          const dev = Math.abs(resampled[i][j] - env.mean[j]);
          if (dev > maxDev) {
            maxDev = dev;
            maxDevIdx = j;
          }
        }
        outliers.push({ label: validTraces[i].label, pct: Math.round(frac * 100), maxDeviationIdx: maxDevIdx });
      }
    }

    const bandMetrics = computeBandMetrics(env, batchRange);
    const hotZones = detectHotZones(env.std);

    return { envelope: env, outliers, bandMetrics, hotZones, resampled };
  }, [validTraces, batchRange]);

  const h = 80;
  const w = 300;
  const yPadding = 2;
  const usableHeight = h - yPadding * 2;

  // Pre-compute envelope rendering data: segment colors, trapezoid paths, mean path, sigma bar.
  // Must run before any early return so the hook order stays stable across renders.
  const envelopeRender = useMemo(() => {
    if (!envelope) return null;
    const normUpper = normalizeBatch(envelope.upper, batchRange.min, batchRange.max);
    const normLower = normalizeBatch(envelope.lower, batchRange.min, batchRange.max);
    const normMean = normalizeBatch(envelope.mean, batchRange.min, batchRange.max);
    const xStep = w / Math.max(RESAMPLE_LEN - 1, 1);
    const rangeSpan = batchRange.max - batchRange.min || 1;

    const toY = (v: number) => {
      const clamped = Math.max(0, Math.min(1, v));
      return yPadding + usableHeight - clamped * usableHeight;
    };

    // Per-segment colors based on local band width
    const segColors = envelope.std.map((sigma) => {
      const localWidth = 4 * sigma;
      return bandWidthColor(localWidth / rangeSpan);
    });

    // Gradient-colored trapezoid segments
    const segments: { path: string; color: string }[] = [];
    for (let i = 0; i < RESAMPLE_LEN - 1; i++) {
      const x1 = i * xStep;
      const x2 = (i + 1) * xStep;
      segments.push({
        path: `M ${x1} ${toY(normUpper[i])} L ${x2} ${toY(normUpper[i + 1])} L ${x2} ${toY(normLower[i + 1])} L ${x1} ${toY(normLower[i])} Z`,
        color: segColors[i],
      });
    }

    // Mean line
    const meanPath = normMean
      .map((v, i) => `${i === 0 ? "M" : "L"} ${i * xStep} ${toY(v)}`)
      .join(" ");

    // Sigma profile bar data (normalized to max sigma)
    const maxSigma = Math.max(...envelope.std, 1e-10);
    const sigmaBar = envelope.std.map((s) => s / maxSigma);

    return { segments, meanPath, segColors, sigmaBar };
  }, [envelope, batchRange, usableHeight]);

  if (validTraces.length === 0) {
    return (
      <div className="bg-gray-50 dark:bg-gray-800 rounded p-3">
        <div className="text-xs text-gray-500 mb-1">{title}</div>
        <div className="text-xs text-gray-400 text-center py-4">No data</div>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 dark:bg-gray-800 rounded p-3">
      <div className="flex items-center gap-2 mb-1">
        <div className="text-xs text-gray-500">{title}</div>
        {bandMetrics && (
          <div className="flex items-center gap-1.5 ml-auto">
            <span
              className="text-[10px] font-semibold px-1.5 py-0.5 rounded"
              style={{
                color: bandMetrics.color,
                backgroundColor: `${bandMetrics.color}15`,
              }}
              title="The normal range covers this percentage of the full action range. Lower = more consistent episodes."
            >
              {bandMetrics.label} · {Math.round(bandMetrics.bandCoverage * 100)}%
            </span>
          </div>
        )}
      </div>
      {/* Inline chart legend */}
      {envelope && (
        <div className="flex items-center gap-3 mb-0.5" style={{ height: 14 }}>
          <div className="flex items-center gap-1">
            <svg width="16" height="6" viewBox="0 0 16 6">
              <line x1="0" y1="3" x2="16" y2="3" stroke="#e2e8f0" strokeWidth="1.5" />
            </svg>
            <span className="text-[10px] text-gray-400">median</span>
          </div>
          <div className="flex items-center gap-1">
            <svg width="14" height="8" viewBox="0 0 14 8">
              <rect x="0" y="0" width="14" height="8" fill={bandMetrics?.color || "#3b82f6"} opacity="0.3" rx="1" />
            </svg>
            <span className="text-[10px] text-gray-400">normal range</span>
          </div>
          <div className="flex items-center gap-1">
            <svg width="16" height="6" viewBox="0 0 16 6">
              <line x1="0" y1="3" x2="16" y2="3" stroke="#3b82f6" strokeWidth="0.8" opacity="0.5" />
            </svg>
            <span className="text-[10px] text-gray-400">episodes</span>
          </div>
        </div>
      )}
      {outliers.length > 0 && (
        <div className="mb-1">
          {outliers.map((o) => (
            <div
              key={o.label}
              className={`group text-[10px] text-amber-500 dark:text-amber-400 font-mono ${onInspect ? "cursor-pointer hover:underline" : ""}`}
              onClick={onInspect ? () => onInspect(o.maxDeviationIdx) : undefined}
              title={onInspect ? "Click to compare frames at the point of maximum deviation" : undefined}
            >
              {o.label}: outlier ({o.pct}% outside normal range)
              {onInspect && (
                <span className="text-amber-400/60 group-hover:text-amber-300 ml-1 transition-colors">
                  → inspect
                </span>
              )}
            </div>
          ))}
        </div>
      )}
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className={`w-full bg-gray-100 dark:bg-gray-700 ${envelopeRender?.sigmaBar ? "rounded-t" : "rounded"}`}
        style={{ height: 80 }}
        preserveAspectRatio="none"
      >
        {/* Gradient-colored ±2std band (trapezoid segments) */}
        {envelopeRender?.segments.map((seg, i) => (
          <path key={`seg-${i}`} d={seg.path} fill={seg.color} opacity="0.3" />
        ))}
        {/* Individual episode traces (faded) */}
        {validTraces.map((trace, i) => (
          <path
            key={`${trace.label}-${i}`}
            d={createPath(normalizeBatch(trace.data, batchRange.min, batchRange.max), h, w)}
            fill="none"
            stroke={trace.color}
            strokeWidth="0.8"
            vectorEffect="non-scaling-stroke"
            opacity="0.5"
          />
        ))}
        {/* Mean line */}
        {envelopeRender?.meanPath && (
          <path
            d={envelopeRender.meanPath}
            fill="none"
            stroke="#e2e8f0"
            strokeWidth="1.5"
            vectorEffect="non-scaling-stroke"
            opacity="0.9"
          />
        )}
      </svg>
      {/* Hot zone click hint */}
      {hotZones.length > 0 && onInspect && (
        <div className="text-[8px] text-red-400 bg-gray-100 dark:bg-gray-700 px-1 py-0.5 text-center">
          ▼ Click highlighted zones below to compare frames
        </div>
      )}
      {/* Sigma profile bar with hot zone overlays */}
      {envelopeRender?.sigmaBar && (
        <div className="flex items-stretch">
          <div
            className="flex items-center justify-center bg-gray-100 dark:bg-gray-700 rounded-bl text-[8px] text-gray-400 select-none shrink-0"
            style={{ width: 38 }}
            title="Shows how much actions vary between episodes at each timestep"
          >
            Variance
          </div>
        <svg
          viewBox={`0 0 ${w} 20`}
          className={`w-full bg-gray-100 dark:bg-gray-700 rounded-br ${onInspect ? "cursor-crosshair" : ""}`}
          style={{ height: 20 }}
          preserveAspectRatio="none"
          onClick={onInspect ? (e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const idx = Math.round((x / rect.width) * (RESAMPLE_LEN - 1));
            onInspect(Math.max(0, Math.min(RESAMPLE_LEN - 1, idx)));
          } : undefined}
        >
          {envelopeRender.sigmaBar.map((normSigma, i) => {
            const barX = (i / RESAMPLE_LEN) * w;
            const barW = w / RESAMPLE_LEN + 0.5;
            const barH = normSigma * 18;
            return (
              <rect
                key={`sb-${i}`}
                x={barX}
                y={20 - barH}
                width={barW}
                height={barH}
                fill={envelopeRender.segColors[i]}
                opacity="0.7"
              />
            );
          })}
          {/* Hot zone hover styles */}
          <defs>
            <style>{`
              .hot-zone-group:hover .hz-overlay { opacity: 0.4; }
              .hot-zone-group:hover .hz-border-top { stroke: #ef4444; stroke-opacity: 1; }
              .hot-zone-group:hover .hz-border-bottom { stroke: #ef4444; stroke-opacity: 1; }
            `}</style>
          </defs>
          {/* Hot zone overlays */}
          {hotZones.map((zone, i) => {
            const x1 = (zone.start / RESAMPLE_LEN) * w;
            const x2 = ((zone.end + 1) / RESAMPLE_LEN) * w;
            const zoneW = x2 - x1;
            return (
              <g
                key={`hz-${i}`}
                className={`hot-zone-group ${onInspect ? "cursor-pointer" : ""}`}
                onClick={onInspect ? (e) => {
                  e.stopPropagation();
                  const mid = Math.round((zone.start + zone.end) / 2);
                  onInspect(mid);
                } : undefined}
              >
                {/* Dark overlay to dim sigma bar underneath */}
                <rect
                  className="hz-overlay"
                  x={x1} y={0} width={zoneW} height={20}
                  fill="#1e293b" opacity="0.25"
                />
                {/* Dashed red top border */}
                <line
                  className="hz-border-top"
                  x1={x1} y1={1} x2={x2} y2={1}
                  stroke="#ef4444" strokeWidth="2" strokeDasharray="4 2"
                  vectorEffect="non-scaling-stroke" strokeOpacity="0.8"
                />
                {/* Solid red bottom border */}
                <line
                  className="hz-border-bottom"
                  x1={x1} y1={19} x2={x2} y2={19}
                  stroke="#ef4444" strokeWidth="1"
                  vectorEffect="non-scaling-stroke" strokeOpacity="0.7"
                />
                {/* Left bracket */}
                <line
                  x1={x1} y1={0} x2={x1} y2={20}
                  stroke="#ef4444" strokeWidth="1"
                  vectorEffect="non-scaling-stroke" opacity="0.6"
                />
                {/* Right bracket */}
                <line
                  x1={x2} y1={0} x2={x2} y2={20}
                  stroke="#ef4444" strokeWidth="1"
                  vectorEffect="non-scaling-stroke" opacity="0.6"
                />
              </g>
            );
          })}
        </svg>
        </div>
      )}
    </div>
  );
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

function useFirstFrames(
  episodeList: [string, EpisodeSignalData][],
  datasetId: string | null,
  sseFirstFrames?: Map<string, string>,
) {
  const [frameData, setFrameData] = useState<Map<string, string>>(new Map());
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<Set<string>>(new Set());

  // Persist accumulated results across re-renders so we don't re-fetch
  // episodes that already have frames when episodeList grows incrementally.
  const accumulatedRef = useRef<Map<string, string>>(new Map());
  const errorsRef = useRef<Set<string>>(new Set());
  const controllerRef = useRef<AbortController | null>(null);
  const fetchingRef = useRef<Set<string>>(new Set());
  const prevDatasetRef = useRef<string | null>(null);

  const fetchSingleFrame = useCallback(async (
    episodeId: string,
    episodeData: EpisodeSignalData,
    dsId: string,
    signal: AbortSignal,
  ) => {
    try {
      const frameEpisodeId =
        episodeData.global_episode_index != null
          ? `episode_${episodeData.global_episode_index}`
          : episodeId;
      const res = await fetch(
        `${API_BASE}/episodes/${encodeURIComponent(frameEpisodeId)}/frames?start=0&end=1&dataset_id=${encodeURIComponent(dsId)}&resolution=low&quality=70`,
        { signal }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const frame = data.frames?.[0]?.image_base64;
      if (frame) {
        accumulatedRef.current.set(episodeId, frame);
      } else {
        errorsRef.current.add(episodeId);
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      errorsRef.current.add(episodeId);
    } finally {
      fetchingRef.current.delete(episodeId);
    }
    if (!signal.aborted) {
      setFrameData(new Map(accumulatedRef.current));
      setErrors(new Set(errorsRef.current));
    }
  }, []);

  useEffect(() => {
    if (!datasetId || episodeList.length === 0) {
      accumulatedRef.current.clear();
      errorsRef.current.clear();
      fetchingRef.current.clear();
      setFrameData(new Map());
      setErrors(new Set());
      setLoading(false);
      return;
    }

    // Reset everything when dataset changes
    if (prevDatasetRef.current !== datasetId) {
      controllerRef.current?.abort();
      accumulatedRef.current.clear();
      errorsRef.current.clear();
      fetchingRef.current.clear();
      prevDatasetRef.current = datasetId;
    }

    // Ingest SSE-provided first frames (decoupled first_frame events)
    let hasNewSseFrames = false;
    if (sseFirstFrames) {
      for (const [id, frame] of sseFirstFrames) {
        if (frame && !accumulatedRef.current.has(id)) {
          accumulatedRef.current.set(id, frame);
          hasNewSseFrames = true;
        }
      }
    }
    // Also ingest any first_frame embedded in episode data (legacy path)
    for (const [id, epData] of episodeList) {
      if (epData.first_frame && !accumulatedRef.current.has(id)) {
        accumulatedRef.current.set(id, epData.first_frame);
        hasNewSseFrames = true;
      }
    }
    if (hasNewSseFrames) {
      setFrameData(new Map(accumulatedRef.current));
    }

    // Abort only the previous fetch sequence, not already-completed results
    controllerRef.current?.abort();
    fetchingRef.current.clear();  // Prevent aborted episodes from being permanently skipped
    const controller = new AbortController();
    controllerRef.current = controller;

    // Find episodes that need fetching via API (no SSE frame, not already loaded)
    const toFetch = episodeList.filter(([id, epData]) =>
      !epData.first_frame &&
      !accumulatedRef.current.has(id) &&
      !errorsRef.current.has(id) &&
      !fetchingRef.current.has(id)
    );

    if (toFetch.length === 0) {
      setLoading(false);
      return;
    }

    setLoading(true);

    // Fetch new episodes sequentially to avoid thread pool starvation
    const fetchNewFrames = async () => {
      for (const [episodeId, episodeData] of toFetch) {
        if (controller.signal.aborted) return;
        fetchingRef.current.add(episodeId);
        await fetchSingleFrame(episodeId, episodeData, datasetId, controller.signal);
      }
      if (!controller.signal.aborted) {
        setLoading(false);
      }
    };

    fetchNewFrames();
    return () => { controller.abort(); };
  }, [episodeList, datasetId, fetchSingleFrame, sseFirstFrames]);

  return { frameData, loading, errors };
}

function StartingPositionGrid({
  gridEpisodeIds,
  frameData,
  loading,
  errors,
}: {
  gridEpisodeIds: { id: string; index: number }[];
  frameData: Map<string, string>;
  loading: boolean;
  errors: Set<string>;
}) {
  // Show nothing only if no episodes to show at all
  if (gridEpisodeIds.length === 0 && !loading && frameData.size === 0 && errors.size === 0) return null;

  return (
    <div className="mb-4" data-testid="starting-position-grid">
      <div className="text-xs font-medium text-gray-500 mb-2 flex items-center gap-2">
        Starting Position
        {loading && (
          <svg className="animate-spin h-3 w-3 text-gray-400" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        )}
      </div>
      <div className="grid grid-cols-5 gap-3">
        {gridEpisodeIds.map(({ id }) => {
          const frame = frameData.get(id);
          const hasError = errors.has(id);
          const isPending = !frame && !hasError;

          return (
            <div
              key={id}
              className="relative aspect-video rounded overflow-hidden border border-gray-200 dark:border-gray-700"
            >
              {frame ? (
                <img
                  src={`data:image/${frame.startsWith("/9j/") ? "jpeg" : "webp"};base64,${frame}`}
                  alt={getEpisodeLabel(id)}
                  className="w-full h-full object-cover"
                />
              ) : hasError ? (
                <div className="w-full h-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
                  <span className="text-[10px] text-gray-400">No frame</span>
                </div>
              ) : isPending ? (
                <div className="w-full h-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center animate-pulse">
                  <svg className="animate-spin h-4 w-4 text-gray-400" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                </div>
              ) : (
                <div className="w-full h-full bg-gray-200 dark:bg-gray-700" />
              )}
              <div className="absolute bottom-0 left-0 right-0 bg-black/50 px-1 py-0.5">
                <span className="text-[10px] text-white font-mono truncate block">
                  {getEpisodeLabel(id)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Frame Comparison Strip — shows thumbnails from all episodes at a given resampled index
const FRAME_AT_API = `${API_BASE}/episodes`;

function FrameComparisonStrip({
  resampledIndex,
  episodes,
  episodeList,
  datasetId,
  onNavigate,
  onClose,
  onIndexChange,
}: {
  resampledIndex: number;
  episodes: Map<string, EpisodeSignalData>;
  episodeList: [string, EpisodeSignalData][];
  datasetId: string | null;
  onNavigate: (episodeId: string, frameNumber: number) => void;
  onClose: () => void;
  onIndexChange: (newIndex: number) => void;
}) {
  const [thumbnails, setThumbnails] = useState<Map<string, { image: string; frameIdx: number } | null>>(new Map());
  const [loading, setLoading] = useState<Set<string>>(new Set());
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const stripRef = useRef<HTMLDivElement>(null);

  // Keyboard scrubbing
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      const step = e.shiftKey ? 15 : 3;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        onIndexChange(Math.max(0, resampledIndex - step));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        onIndexChange(Math.min(RESAMPLE_LEN - 1, resampledIndex + step));
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [resampledIndex, onClose, onIndexChange]);

  // Debounced fetch of thumbnails for all episodes at the current index
  useEffect(() => {
    if (!datasetId) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    abortRef.current?.abort();

    debounceRef.current = setTimeout(() => {
      const controller = new AbortController();
      abortRef.current = controller;

      const fetchAll = async () => {
        const newLoading = new Set<string>();
        const newThumbnails = new Map(thumbnails);

        for (const [epId, epData] of episodeList) {
          const totalFrames = epData.total_frames ?? epData.actions?.actions?.length ?? 0;
          if (totalFrames === 0) continue;
          const frameIdx = Math.round((resampledIndex / (RESAMPLE_LEN - 1)) * Math.max(0, totalFrames - 1));
          newLoading.add(epId);

          // Resolve episode ID for the main viewer
          const viewerEpId = epData.global_episode_index != null
            ? `episode_${epData.global_episode_index}`
            : epId;

          try {
            const res = await fetch(
              `${FRAME_AT_API}/${encodeURIComponent(viewerEpId)}/frame-at?frame_index=${frameIdx}&dataset_id=${encodeURIComponent(datasetId)}&resolution=low&quality=70`,
              { signal: controller.signal }
            );
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            if (data.image_base64) {
              newThumbnails.set(epId, { image: data.image_base64, frameIdx });
            }
          } catch (err: unknown) {
            if (err instanceof DOMException && err.name === "AbortError") return;
          }
          newLoading.delete(epId);
        }

        if (!controller.signal.aborted) {
          setThumbnails(new Map(newThumbnails));
          setLoading(new Set());
        }
      };

      setLoading(new Set(episodeList.map(([id]) => id)));
      fetchAll();
    }, 300);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      abortRef.current?.abort();
    };
  }, [resampledIndex, datasetId, episodeList]);

  const progressPct = ((resampledIndex / (RESAMPLE_LEN - 1)) * 100).toFixed(0);

  return (
    <div
      ref={stripRef}
      className="mt-3 p-3 bg-gray-50 dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700"
      data-testid="frame-comparison-strip"
    >
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-medium text-gray-500">
          Frame Comparison
        </div>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 text-xs"
        >
          Close
        </button>
      </div>
      {/* Scrub bar */}
      <div className="flex items-center gap-2 mb-2">
        <button
          onClick={() => onIndexChange(Math.max(0, resampledIndex - 3))}
          className="text-gray-400 hover:text-gray-200 text-sm px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 transition-colors shrink-0"
          title="Previous frames (← arrow key, Shift for larger steps)"
        >
          ◀
        </button>
        <div className="relative flex-1 h-5 flex items-center group">
          <input
            type="range"
            min={0}
            max={RESAMPLE_LEN - 1}
            value={resampledIndex}
            onChange={(e) => onIndexChange(Number(e.target.value))}
            className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-gray-600 accent-blue-500
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
              [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:cursor-grab
              [&::-webkit-slider-thumb]:active:cursor-grabbing"
          />
          <div className="absolute -bottom-3 left-1/2 -translate-x-1/2 text-[9px] text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
            {progressPct}% · frame position
          </div>
        </div>
        <button
          onClick={() => onIndexChange(Math.min(RESAMPLE_LEN - 1, resampledIndex + 3))}
          className="text-gray-400 hover:text-gray-200 text-sm px-1.5 py-0.5 rounded bg-gray-700 hover:bg-gray-600 transition-colors shrink-0"
          title="Next frames (→ arrow key, Shift for larger steps)"
        >
          ▶
        </button>
      </div>
      <div className="grid grid-cols-5 gap-2">
        {episodeList.map(([epId, epData]) => {
          const totalFrames = epData.total_frames ?? epData.actions?.actions?.length ?? 0;
          const frameIdx = totalFrames > 0 ? Math.round((resampledIndex / (RESAMPLE_LEN - 1)) * Math.max(0, totalFrames - 1)) : 0;
          const thumb = thumbnails.get(epId);
          const isLoading = loading.has(epId);
          return (
            <div
              key={epId}
              className="relative rounded overflow-hidden"
            >
              <div className="aspect-video bg-gray-200 dark:bg-gray-700">
                {isLoading ? (
                  <div className="w-full h-full flex items-center justify-center animate-pulse">
                    <svg className="animate-spin h-4 w-4 text-gray-400" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                  </div>
                ) : thumb ? (
                  <img
                    src={`data:image/${thumb.image.startsWith("/9j/") ? "jpeg" : "webp"};base64,${thumb.image}`}
                    alt={getEpisodeLabel(epId)}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <span className="text-[10px] text-gray-400">No frame</span>
                  </div>
                )}
              </div>
              <div className="bg-black/60 px-1.5 py-1 flex items-center justify-between">
                <div>
                  <div className="text-[10px] text-white font-mono truncate">
                    {getEpisodeLabel(epId)}
                  </div>
                  <div className="text-[9px] text-gray-300">
                    frame {frameIdx}{totalFrames > 0 ? ` / ${totalFrames}` : ""}
                  </div>
                </div>
                <button
                  onClick={() => onNavigate(epId, frameIdx)}
                  className="text-[10px] bg-blue-500 hover:bg-blue-600 text-white px-1.5 py-0.5 rounded transition-colors"
                >
                  View
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function SignalComparisonChart({
  episodes,
  datasetId,
  firstFrames: sseFirstFrames,
  knownEpisodes,
  onNavigateToEpisode,
}: SignalComparisonChartProps) {
  const episodeList = useMemo(() => {
    return Array.from(episodes.entries()).sort(
      ([, a], [, b]) => a.episode_index - b.episode_index
    );
  }, [episodes]);

  // Build grid episode list: use knownEpisodes for early placeholder rendering,
  // fall back to episodeList once signal data arrives
  const gridEpisodeIds = useMemo(() => {
    if (episodeList.length > 0) {
      return episodeList.map(([id, ep]) => ({ id, index: ep.episode_index }));
    }
    if (knownEpisodes && knownEpisodes.length > 0) {
      return knownEpisodes.map((stub) => ({ id: stub.episode_id, index: stub.episode_index }));
    }
    return [];
  }, [episodeList, knownEpisodes]);

  const [inspectIndex, setInspectIndex] = useState<number | null>(null);

  const { frameData, loading: framesLoading, errors: frameErrors } = useFirstFrames(episodeList, datasetId, sseFirstFrames);

  // Derive action classification from first episode with labels (all episodes
  // in a task share the same action space). The React 19 compiler auto-memoizes
  // this; explicit useMemo here would conflict with the compiler analysis.
  let classification: ActionGrouping | undefined;
  for (const [, ep] of episodeList) {
    const actions = ep.actions;
    if (!actions || actions.error) continue;
    const rows = actions.actions;
    if (!rows || rows.length === 0) continue;
    classification = classifyActionDimensions(actions.dimension_labels ?? null, rows[0].length);
    break;
  }

  // Single precomputation pass: compute all magnitudes once per episode
  const precomputed = useMemo(() => {
    const map = new Map<string, PrecomputedEpisodeSignals>();
    for (const [id, ep] of episodeList) {
      const signals: PrecomputedEpisodeSignals = {
        position: [], rotation: [], gripper: [], gripper2: [], accel: [], gyro: [],
      };
      if (ep.actions && !ep.actions.error && ep.actions.actions?.length > 0 && classification) {
        signals.position = computeMagnitude(ep.actions.actions, classification.group1.indices);
        signals.rotation = computeMagnitude(ep.actions.actions, classification.group2.indices);
        if (classification.grippers.length > 0) {
          signals.gripper = computeGripperSignal(ep.actions.actions, classification.grippers[0].index);
        }
        if (classification.grippers.length > 1) {
          signals.gripper2 = computeGripperSignal(ep.actions.actions, classification.grippers[1].index);
        }
      }
      if (ep.imu && !ep.imu.error && ep.imu.timestamps?.length > 0) {
        signals.accel = computeIMUMagnitude(ep.imu.accel_x, ep.imu.accel_y, ep.imu.accel_z);
        signals.gyro = computeIMUMagnitude(ep.imu.gyro_x, ep.imu.gyro_y, ep.imu.gyro_z);
      }
      map.set(id, signals);
    }
    return map;
  }, [episodeList, classification]);

  // Derive batch ranges from precomputed magnitudes (no recomputation)
  const batchRanges: BatchRanges = useMemo(() => {
    const allPosition: number[][] = [];
    const allRotation: number[][] = [];
    const allGripper: number[][] = [];
    const allGripper2: number[][] = [];
    const allAccel: number[][] = [];
    const allGyro: number[][] = [];

    for (const [, signals] of precomputed) {
      if (signals.position.length > 0) allPosition.push(signals.position);
      if (signals.rotation.length > 0) allRotation.push(signals.rotation);
      if (signals.gripper.length > 0) allGripper.push(signals.gripper);
      if (signals.gripper2.length > 0) allGripper2.push(signals.gripper2);
      if (signals.accel.length > 0) allAccel.push(signals.accel);
      if (signals.gyro.length > 0) allGyro.push(signals.gyro);
    }

    return {
      position: getBatchRange(allPosition),
      rotation: getBatchRange(allRotation),
      gripper: getBatchRange(allGripper),
      gripper2: getBatchRange(allGripper2),
      accel: getBatchRange(allAccel),
      gyro: getBatchRange(allGyro),
    };
  }, [precomputed]);

  // Derive overlay traces from precomputed magnitudes (no recomputation)
  const { positionTraces, rotationTraces, accelTraces, gyroTraces } = useMemo(() => {
    const positionTraces: { label: string; data: number[]; color: string }[] = [];
    const rotationTraces: { label: string; data: number[]; color: string }[] = [];
    const accelTraces: { label: string; data: number[]; color: string }[] = [];
    const gyroTraces: { label: string; data: number[]; color: string }[] = [];

    episodeList.forEach(([id], i) => {
      const color = EPISODE_COLORS[i % EPISODE_COLORS.length];
      const label = getEpisodeLabel(id);
      const signals = precomputed.get(id);
      if (!signals) return;

      if (signals.position.length > 0) positionTraces.push({ label, data: signals.position, color });
      if (signals.rotation.length > 0) rotationTraces.push({ label, data: signals.rotation, color });
      if (signals.accel.length > 0) accelTraces.push({ label, data: signals.accel, color });
      if (signals.gyro.length > 0) gyroTraces.push({ label, data: signals.gyro, color });
    });

    return { positionTraces, rotationTraces, accelTraces, gyroTraces };
  }, [episodeList, precomputed]);

  // Check if any episode actually has gripper data (for conditional legend)
  const hasGripperData = useMemo(() => {
    for (const [, signals] of precomputed) {
      if (signals.gripper.length > 0) return true;
    }
    return false;
  }, [precomputed]);

  const hasGripper2Data = useMemo(() => {
    for (const [, signals] of precomputed) {
      if (signals.gripper2.length > 0) return true;
    }
    return false;
  }, [precomputed]);

  // Handle navigate from frame comparison strip
  const handleStripNavigate = useCallback((episodeId: string, frameNumber: number) => {
    if (!onNavigateToEpisode || !datasetId) return;
    const epData = episodes.get(episodeId);
    const totalFrames = epData?.total_frames ?? epData?.actions?.actions?.length ?? 0;
    // Resolve to viewer episode ID
    const viewerEpId = epData?.global_episode_index != null
      ? `episode_${epData.global_episode_index}`
      : episodeId;
    onNavigateToEpisode(datasetId, viewerEpId, totalFrames, frameNumber);
  }, [onNavigateToEpisode, datasetId, episodes]);

  if (episodeList.length === 0 && gridEpisodeIds.length === 0) {
    return null;
  }

  return (
    <div data-testid="signal-comparison-chart">
      {/* Starting position thumbnails — shows placeholders immediately via knownEpisodes */}
      {datasetId && gridEpisodeIds.length > 0 && (
        <StartingPositionGrid
          gridEpisodeIds={gridEpisodeIds}
          frameData={frameData}
          loading={framesLoading}
          errors={frameErrors}
        />
      )}

      {/* Overlay and per-episode charts — only render when signal data exists */}
      {episodeList.length > 0 && (
        <>
          {/* Overlay section */}
          <div className="mb-4">
            <div className="flex flex-wrap gap-x-3 gap-y-1 mb-2">
              <span className="text-xs font-medium text-gray-500">Overlay — </span>
              {episodeList.map(([id], i) => (
                <span key={id} className="flex items-center gap-1 text-xs">
                  <span
                    className="w-3 h-0.5 rounded"
                    style={{ backgroundColor: EPISODE_COLORS[i % EPISODE_COLORS.length] }}
                  />
                  <span className="text-gray-600 dark:text-gray-300 font-mono">
                    {getEpisodeLabel(id)}
                  </span>
                </span>
              ))}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <OverlayPanel title={`${classification?.group1.label ?? "Position"} Magnitude`} traces={positionTraces} batchRange={batchRanges.position} onInspect={setInspectIndex} />
              <OverlayPanel title="Accelerometer Magnitude" traces={accelTraces} batchRange={batchRanges.accel} onInspect={setInspectIndex} />
              <OverlayPanel title={`${classification?.group2.label ?? "Rotation"} Magnitude`} traces={rotationTraces} batchRange={batchRanges.rotation} onInspect={setInspectIndex} />
              <OverlayPanel title="Gyroscope Magnitude" traces={gyroTraces} batchRange={batchRanges.gyro} onInspect={setInspectIndex} />
            </div>

            {/* Frame Comparison Strip */}
            {inspectIndex !== null && datasetId && (
              <FrameComparisonStrip
                resampledIndex={inspectIndex}
                episodes={episodes}
                episodeList={episodeList}
                datasetId={datasetId}
                onNavigate={handleStripNavigate}
                onClose={() => setInspectIndex(null)}
                onIndexChange={setInspectIndex}
              />
            )}
          </div>

          {/* Per-episode section */}
          <div className="text-xs font-medium text-gray-500 mb-1">Per Episode</div>

          {/* Column headers */}
          <div className="flex items-center gap-3 mb-1 pb-1 border-b border-gray-200 dark:border-gray-700">
            <div className="w-28 flex-shrink-0">
              <span className="text-xs font-medium text-gray-500">Episode</span>
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-gray-500">Actions</span>
                <span className="flex items-center gap-1 text-[10px] text-gray-400">
                  <span className="inline-block w-2.5 h-0.5 rounded" style={{ backgroundColor: classification?.group1.color ?? "#3b82f6" }} />
                  {classification?.group1.label ?? "Position"}
                  <span className="inline-block w-2.5 h-0.5 rounded ml-1" style={{ backgroundColor: classification?.group2.color ?? "#8b5cf6" }} />
                  {classification?.group2.label ?? "Rotation"}
                  {hasGripperData && (
                    <>
                      <span
                        className="inline-block w-2.5 h-0.5 rounded ml-1"
                        style={{ backgroundColor: classification?.grippers[0]?.color ?? "#22c55e", borderBottom: `1px dashed ${classification?.grippers[0]?.color ?? "#22c55e"}` }}
                      />
                      {classification?.grippers[0]?.label ?? "Gripper"}
                    </>
                  )}
                  {hasGripper2Data && classification?.grippers[1] && (
                    <>
                      <span
                        className="inline-block w-2.5 h-0.5 rounded ml-1"
                        style={{ backgroundColor: classification.grippers[1].color, borderBottom: `1px dashed ${classification.grippers[1].color}` }}
                      />
                      {classification.grippers[1].label}
                    </>
                  )}
                </span>
              </div>
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-gray-500">IMU</span>
                <span className="flex items-center gap-1 text-[10px] text-gray-400">
                  <span className="inline-block w-2.5 h-0.5 rounded bg-blue-500" />
                  Accel
                  <span className="inline-block w-2.5 h-0.5 rounded bg-orange-500 ml-1" />
                  Gyro
                </span>
              </div>
            </div>
          </div>

          {/* Episode rows */}
          <div className="max-h-[400px] overflow-y-auto">
            {episodeList.map(([id, data]) => (
              <EpisodeChartRow
                key={id}
                episodeId={id}
                episodeData={data}
                batchRanges={batchRanges}
                precomputedSignals={precomputed.get(id)}
                classification={classification}
              />
            ))}
          </div>

          {/* Error summary */}
          {episodeList.some(([, ep]) => ep.actions?.error || ep.imu?.error) && (
            <div className="mt-2 text-xs text-amber-600 dark:text-amber-400">
              Some episodes had extraction errors — check console for details.
            </div>
          )}
        </>
      )}
    </div>
  );
}
