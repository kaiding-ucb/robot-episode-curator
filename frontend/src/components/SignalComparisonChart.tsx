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
  const frameCount = episodeData.actions?.actions?.length ?? 0;

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

  return (
    <div className="bg-gray-50 dark:bg-gray-800 rounded p-3">
      <div className="text-xs text-gray-500 mb-1">{title}</div>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full bg-gray-100 dark:bg-gray-700 rounded"
        style={{ height: 80 }}
        preserveAspectRatio="none"
      >
        {validTraces.map((trace, i) => (
          <path
            key={`${trace.label}-${i}`}
            d={createPath(normalizeBatch(trace.data, batchRange.min, batchRange.max), h, w)}
            fill="none"
            stroke={trace.color}
            strokeWidth="0.8"
            vectorEffect="non-scaling-stroke"
            opacity="0.8"
          />
        ))}
      </svg>
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
