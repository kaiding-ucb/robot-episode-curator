"use client";

import { useMemo } from "react";
import type { EpisodeSignalData } from "@/types/analysis";

interface SignalComparisonChartProps {
  episodes: Map<string, EpisodeSignalData>;
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

// Get min/max across multiple arrays
function getBatchRange(arrays: number[][]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const arr of arrays) {
    for (const v of arr) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (!isFinite(min)) return { min: 0, max: 1 };
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
  cv: number;
  label: "Tight" | "Moderate" | "Loose";
  color: string;
}

function computeBandMetrics(
  envelope: { mean: number[]; upper: number[]; lower: number[]; std: number[] },
  batchRange: { min: number; max: number }
): BandMetrics {
  const len = envelope.mean.length;
  if (len === 0) return { bandCoverage: 0, cv: 0, label: "Tight", color: "#22c55e" };

  let bandWidthSum = 0;
  let stdSum = 0;
  let absMuSum = 0;
  for (let i = 0; i < len; i++) {
    bandWidthSum += envelope.upper[i] - envelope.lower[i];
    stdSum += envelope.std[i];
    absMuSum += Math.abs(envelope.mean[i]);
  }
  const meanBandWidth = bandWidthSum / len;
  const rangeSpan = batchRange.max - batchRange.min || 1;
  const bandCoverage = meanBandWidth / rangeSpan;

  const meanAbsMu = absMuSum / len;
  const cv = meanAbsMu > 0 ? (stdSum / len) / meanAbsMu : 0;

  let label: "Tight" | "Moderate" | "Loose";
  let color: string;
  if (bandCoverage < 0.25) {
    label = "Tight";
    color = "#22c55e";
  } else if (bandCoverage < 0.5) {
    label = "Moderate";
    color = "#eab308";
  } else {
    label = "Loose";
    color = "#ef4444";
  }

  return { bandCoverage, cv, label, color };
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

// Batch normalization ranges for per-episode charts
interface BatchRanges {
  position: { min: number; max: number };
  rotation: { min: number; max: number };
  gripper: { min: number; max: number };
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
}

function EpisodeChartRow({ episodeId, episodeData, batchRanges }: EpisodeChartRowProps) {
  const { actionTraces, imuTraces } = useMemo(() => {
    const actionTraces: SignalTrace[] = [];
    const imuTraces: SignalTrace[] = [];

    // Actions: Position (blue), Rotation (purple), Gripper (green)
    if (
      episodeData.actions &&
      !episodeData.actions.error &&
      episodeData.actions.actions?.length > 0
    ) {
      const dims = episodeData.actions.actions[0].length;

      const posData = computeMagnitude(episodeData.actions.actions, [0, 1, 2]);
      actionTraces.push({
        label: "Position",
        data: posData,
        normalized: normalizeBatch(posData, batchRanges.position.min, batchRanges.position.max),
        color: "#3b82f6",
      });

      if (dims > 3) {
        const rotData = computeMagnitude(episodeData.actions.actions, [3, 4, 5]);
        actionTraces.push({
          label: "Rotation",
          data: rotData,
          normalized: normalizeBatch(rotData, batchRanges.rotation.min, batchRanges.rotation.max),
          color: "#8b5cf6",
        });
      }

      if (dims > 6) {
        const gripData = computeGripperSignal(episodeData.actions.actions, 6);
        actionTraces.push({
          label: "Gripper",
          data: gripData,
          normalized: normalizeBatch(gripData, batchRanges.gripper.min, batchRanges.gripper.max),
          color: "#22c55e",
          dashed: true,
        });
      }
    }

    // IMU: Accel (blue), Gyro (orange)
    if (
      episodeData.imu &&
      !episodeData.imu.error &&
      episodeData.imu.timestamps?.length > 0
    ) {
      const accelData = computeIMUMagnitude(
        episodeData.imu.accel_x,
        episodeData.imu.accel_y,
        episodeData.imu.accel_z
      );
      imuTraces.push({
        label: "Accel",
        data: accelData,
        normalized: normalizeBatch(accelData, batchRanges.accel.min, batchRanges.accel.max),
        color: "#3b82f6",
      });

      const gyroData = computeIMUMagnitude(
        episodeData.imu.gyro_x,
        episodeData.imu.gyro_y,
        episodeData.imu.gyro_z
      );
      imuTraces.push({
        label: "Gyro",
        data: gyroData,
        normalized: normalizeBatch(gyroData, batchRanges.gyro.min, batchRanges.gyro.max),
        color: "#f97316",
      });
    }

    return { actionTraces, imuTraces };
  }, [episodeData, batchRanges]);

  const hasError = episodeData.actions?.error || episodeData.imu?.error;
  const frameCount = episodeData.total_frames ?? episodeData.actions?.actions?.length ?? 0;

  return (
    <div className="flex items-stretch gap-3 py-2 border-b border-gray-100 dark:border-gray-800 last:border-b-0">
      <div className="w-28 flex-shrink-0 flex flex-col justify-center">
        <span className="text-xs font-mono text-gray-700 dark:text-gray-300 truncate">
          {getEpisodeLabel(episodeId)}
        </span>
        <span className="text-[10px] text-gray-400">
          {frameCount > 0 ? `${frameCount} frames` : ""}
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

// Overlay panel: all episodes on one chart, each a different color
// Uses batch normalization so all traces share the same scale
// Renders a mean ± 2std shaded envelope to highlight outliers
function OverlayPanel({
  title,
  traces,
  batchRange,
}: {
  title: string;
  traces: { label: string; data: number[]; color: string }[];
  batchRange: { min: number; max: number };
}) {
  const validTraces = traces.filter((t) => t.data.length > 0);

  // Compute envelope, outlier info, and band quality metrics
  const { envelope, outliers, bandMetrics } = useMemo(() => {
    if (validTraces.length < 2) return { envelope: null, outliers: [], bandMetrics: null };

    // Resample raw data to common length
    const resampled = validTraces.map((t) => resampleToLength(t.data, RESAMPLE_LEN));
    const env = computeEnvelope(resampled);

    // Find outlier episodes (>5% of timesteps outside band)
    const outliers: { label: string; pct: number }[] = [];
    for (let i = 0; i < resampled.length; i++) {
      const frac = computeOutlierFraction(resampled[i], env.upper, env.lower);
      if (frac > 0.05) {
        outliers.push({ label: validTraces[i].label, pct: Math.round(frac * 100) });
      }
    }

    const bandMetrics = computeBandMetrics(env, batchRange);

    return { envelope: env, outliers, bandMetrics };
  }, [validTraces, batchRange]);

  if (validTraces.length === 0) {
    return (
      <div className="bg-gray-50 dark:bg-gray-800 rounded p-3">
        <div className="text-xs text-gray-500 mb-1">{title}</div>
        <div className="text-xs text-gray-400 text-center py-4">No data</div>
      </div>
    );
  }

  const h = 80;
  const w = 300;
  const yPadding = 2;
  const usableHeight = h - yPadding * 2;

  // Pre-compute envelope rendering data: segment colors, trapezoid paths, mean path, sigma bar
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
            >
              {bandMetrics.label}: {Math.round(bandMetrics.bandCoverage * 100)}%
            </span>
            <span className="text-[10px] text-gray-400 font-mono">
              CV {bandMetrics.cv.toFixed(2)}
            </span>
          </div>
        )}
      </div>
      {outliers.length > 0 && (
        <div className="mb-1">
          {outliers.map((o) => (
            <div
              key={o.label}
              className="text-[10px] text-amber-500 dark:text-amber-400 font-mono"
            >
              {o.label}: outlier ({o.pct}% outside band)
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
      {/* Sigma profile bar */}
      {envelopeRender?.sigmaBar && (
        <svg
          viewBox={`0 0 ${w} 20`}
          className="w-full bg-gray-100 dark:bg-gray-700 rounded-b"
          style={{ height: 20 }}
          preserveAspectRatio="none"
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
        </svg>
      )}
    </div>
  );
}

export default function SignalComparisonChart({
  episodes,
}: SignalComparisonChartProps) {
  const episodeList = useMemo(() => {
    return Array.from(episodes.entries()).sort(
      ([, a], [, b]) => a.episode_index - b.episode_index
    );
  }, [episodes]);

  // Compute batch ranges: min/max per signal type across ALL episodes
  const batchRanges: BatchRanges = useMemo(() => {
    const allPosition: number[][] = [];
    const allRotation: number[][] = [];
    const allGripper: number[][] = [];
    const allAccel: number[][] = [];
    const allGyro: number[][] = [];

    for (const [, ep] of episodeList) {
      if (ep.actions && !ep.actions.error && ep.actions.actions?.length > 0) {
        const dims = ep.actions.actions[0].length;
        allPosition.push(computeMagnitude(ep.actions.actions, [0, 1, 2]));
        if (dims > 3) {
          allRotation.push(computeMagnitude(ep.actions.actions, [3, 4, 5]));
        }
        if (dims > 6) {
          allGripper.push(computeGripperSignal(ep.actions.actions, 6));
        }
      }
      if (ep.imu && !ep.imu.error && ep.imu.timestamps?.length > 0) {
        allAccel.push(computeIMUMagnitude(ep.imu.accel_x, ep.imu.accel_y, ep.imu.accel_z));
        allGyro.push(computeIMUMagnitude(ep.imu.gyro_x, ep.imu.gyro_y, ep.imu.gyro_z));
      }
    }

    return {
      position: getBatchRange(allPosition),
      rotation: getBatchRange(allRotation),
      gripper: getBatchRange(allGripper),
      accel: getBatchRange(allAccel),
      gyro: getBatchRange(allGyro),
    };
  }, [episodeList]);

  // Compute overlay traces (one trace per episode, colored by episode)
  const { positionTraces, rotationTraces, accelTraces, gyroTraces } = useMemo(() => {
    const positionTraces: { label: string; data: number[]; color: string }[] = [];
    const rotationTraces: { label: string; data: number[]; color: string }[] = [];
    const accelTraces: { label: string; data: number[]; color: string }[] = [];
    const gyroTraces: { label: string; data: number[]; color: string }[] = [];

    episodeList.forEach(([id, ep], i) => {
      const color = EPISODE_COLORS[i % EPISODE_COLORS.length];
      const label = getEpisodeLabel(id);

      if (ep.actions && !ep.actions.error && ep.actions.actions?.length > 0) {
        const dims = ep.actions.actions[0].length;
        positionTraces.push({
          label,
          data: computeMagnitude(ep.actions.actions, [0, 1, 2]),
          color,
        });
        if (dims > 3) {
          rotationTraces.push({
            label,
            data: computeMagnitude(ep.actions.actions, [3, 4, 5]),
            color,
          });
        }
      }

      if (ep.imu && !ep.imu.error && ep.imu.timestamps?.length > 0) {
        accelTraces.push({
          label,
          data: computeIMUMagnitude(ep.imu.accel_x, ep.imu.accel_y, ep.imu.accel_z),
          color,
        });
        gyroTraces.push({
          label,
          data: computeIMUMagnitude(ep.imu.gyro_x, ep.imu.gyro_y, ep.imu.gyro_z),
          color,
        });
      }
    });

    return { positionTraces, rotationTraces, accelTraces, gyroTraces };
  }, [episodeList]);

  if (episodeList.length === 0) {
    return null;
  }

  return (
    <div data-testid="signal-comparison-chart">
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
          <OverlayPanel title="Position Magnitude (x,y,z)" traces={positionTraces} batchRange={batchRanges.position} />
          <OverlayPanel title="Accelerometer Magnitude" traces={accelTraces} batchRange={batchRanges.accel} />
          <OverlayPanel title="Rotation Magnitude" traces={rotationTraces} batchRange={batchRanges.rotation} />
          <OverlayPanel title="Gyroscope Magnitude" traces={gyroTraces} batchRange={batchRanges.gyro} />
        </div>
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
              <span className="inline-block w-2.5 h-0.5 rounded bg-blue-500" />
              Position
              <span className="inline-block w-2.5 h-0.5 rounded bg-purple-500 ml-1" />
              Rotation
              <span
                className="inline-block w-2.5 h-0.5 rounded bg-green-500 ml-1"
                style={{ borderBottom: "1px dashed #22c55e" }}
              />
              Gripper
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
          />
        ))}
      </div>

      {/* Error summary */}
      {episodeList.some(([, ep]) => ep.actions?.error || ep.imu?.error) && (
        <div className="mt-2 text-xs text-amber-600 dark:text-amber-400">
          Some episodes had extraction errors — check console for details.
        </div>
      )}
    </div>
  );
}
