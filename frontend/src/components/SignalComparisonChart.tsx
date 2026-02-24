"use client";

import { useMemo, useState, useEffect, useRef, useCallback } from "react";
import type { EpisodeSignalData } from "@/types/analysis";

interface SignalComparisonChartProps {
  episodes: Map<string, EpisodeSignalData>;
  datasetId: string | null;
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

// Precomputed magnitude arrays per episode (computed once, reused everywhere)
interface PrecomputedEpisodeSignals {
  position: number[];
  rotation: number[];
  gripper: number[];
  accel: number[];
  gyro: number[];
}

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
  precomputedSignals?: PrecomputedEpisodeSignals;
}

function EpisodeChartRow({ episodeId, episodeData, batchRanges, precomputedSignals }: EpisodeChartRowProps) {
  const { actionTraces, imuTraces } = useMemo(() => {
    const actionTraces: SignalTrace[] = [];
    const imuTraces: SignalTrace[] = [];

    if (precomputedSignals) {
      // Use precomputed magnitude arrays — no recomputation
      if (precomputedSignals.position.length > 0) {
        actionTraces.push({
          label: "Position",
          data: precomputedSignals.position,
          normalized: normalizeBatch(precomputedSignals.position, batchRanges.position.min, batchRanges.position.max),
          color: "#3b82f6",
        });
      }
      if (precomputedSignals.rotation.length > 0) {
        actionTraces.push({
          label: "Rotation",
          data: precomputedSignals.rotation,
          normalized: normalizeBatch(precomputedSignals.rotation, batchRanges.rotation.min, batchRanges.rotation.max),
          color: "#8b5cf6",
        });
      }
      if (precomputedSignals.gripper.length > 0) {
        actionTraces.push({
          label: "Gripper",
          data: precomputedSignals.gripper,
          normalized: normalizeBatch(precomputedSignals.gripper, batchRanges.gripper.min, batchRanges.gripper.max),
          color: "#22c55e",
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
  }, [precomputedSignals, batchRanges]);

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

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

function useFirstFrames(
  episodeList: [string, EpisodeSignalData][],
  datasetId: string | null
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

    // Ingest any SSE-provided first_frame data (MCAP episodes)
    let hasNewSseFrames = false;
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
  }, [episodeList, datasetId, fetchSingleFrame]);

  return { frameData, loading, errors };
}

function StartingPositionGrid({
  episodeList,
  frameData,
  loading,
  errors,
}: {
  episodeList: [string, EpisodeSignalData][];
  frameData: Map<string, string>;
  loading: boolean;
  errors: Set<string>;
}) {
  // Show nothing only if loading hasn't produced any frames yet AND no episodes listed
  if (!loading && frameData.size === 0 && errors.size === 0) return null;

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
        {episodeList.map(([id]) => {
          const frame = frameData.get(id);
          const hasError = errors.has(id);
          const isPending = !frame && !hasError && loading;

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
                <div className="w-full h-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
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

export default function SignalComparisonChart({
  episodes,
  datasetId,
}: SignalComparisonChartProps) {
  const episodeList = useMemo(() => {
    return Array.from(episodes.entries()).sort(
      ([, a], [, b]) => a.episode_index - b.episode_index
    );
  }, [episodes]);

  const { frameData, loading: framesLoading, errors: frameErrors } = useFirstFrames(episodeList, datasetId);

  // Single precomputation pass: compute all magnitudes once per episode
  const precomputed = useMemo(() => {
    const map = new Map<string, PrecomputedEpisodeSignals>();
    for (const [id, ep] of episodeList) {
      const signals: PrecomputedEpisodeSignals = {
        position: [], rotation: [], gripper: [], accel: [], gyro: [],
      };
      if (ep.actions && !ep.actions.error && ep.actions.actions?.length > 0) {
        const dims = ep.actions.actions[0].length;
        signals.position = computeMagnitude(ep.actions.actions, [0, 1, 2]);
        if (dims > 3) signals.rotation = computeMagnitude(ep.actions.actions, [3, 4, 5]);
        if (dims > 6) signals.gripper = computeGripperSignal(ep.actions.actions, 6);
      }
      if (ep.imu && !ep.imu.error && ep.imu.timestamps?.length > 0) {
        signals.accel = computeIMUMagnitude(ep.imu.accel_x, ep.imu.accel_y, ep.imu.accel_z);
        signals.gyro = computeIMUMagnitude(ep.imu.gyro_x, ep.imu.gyro_y, ep.imu.gyro_z);
      }
      map.set(id, signals);
    }
    return map;
  }, [episodeList]);

  // Derive batch ranges from precomputed magnitudes (no recomputation)
  const batchRanges: BatchRanges = useMemo(() => {
    const allPosition: number[][] = [];
    const allRotation: number[][] = [];
    const allGripper: number[][] = [];
    const allAccel: number[][] = [];
    const allGyro: number[][] = [];

    for (const [, signals] of precomputed) {
      if (signals.position.length > 0) allPosition.push(signals.position);
      if (signals.rotation.length > 0) allRotation.push(signals.rotation);
      if (signals.gripper.length > 0) allGripper.push(signals.gripper);
      if (signals.accel.length > 0) allAccel.push(signals.accel);
      if (signals.gyro.length > 0) allGyro.push(signals.gyro);
    }

    return {
      position: getBatchRange(allPosition),
      rotation: getBatchRange(allRotation),
      gripper: getBatchRange(allGripper),
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

  if (episodeList.length === 0) {
    return null;
  }

  return (
    <div data-testid="signal-comparison-chart">
      {/* Starting position thumbnails */}
      {datasetId && (
        <StartingPositionGrid
          episodeList={episodeList}
          frameData={frameData}
          loading={framesLoading}
          errors={frameErrors}
        />
      )}

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
            precomputedSignals={precomputed.get(id)}
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
