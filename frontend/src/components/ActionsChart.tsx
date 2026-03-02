"use client";

import { useMemo } from "react";
import { useActionsData } from "@/hooks/useApi";
import { classifyActionDimensions } from "@/utils/actionClassification";

interface ActionsChartProps {
  episodeId: string | null;
  datasetId: string | null;
  currentFrame: number;
  totalFrames: number;
}

// Detect gripper events (significant state changes)
function detectGripperEvents(gripperData: number[], threshold: number = 0.3): number[] {
  const events: number[] = [];
  for (let i = 1; i < gripperData.length; i++) {
    const delta = Math.abs(gripperData[i] - gripperData[i - 1]);
    if (delta > threshold) {
      events.push(i);
    }
  }
  return events;
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

export default function ActionsChart({
  episodeId,
  datasetId,
  currentFrame,
  totalFrames,
}: ActionsChartProps) {
  const { actionsData, loading, error } = useActionsData(episodeId, datasetId);

  // Calculate sync line position
  const syncPosition = useMemo(() => {
    if (totalFrames === 0) return 0;
    return (currentFrame / totalFrames) * 100;
  }, [currentFrame, totalFrames]);

  // Process actions into meaningful signals using shared classification
  const {
    positionMag, rotationMag, gripperData, gripper2Data, gripperEvents, gripper2Events,
    hasPosition, hasRotation, hasGripper, hasGripper2, classification
  } = useMemo(() => {
    if (!actionsData || actionsData.actions.length === 0) {
      return {
        positionMag: [], rotationMag: [], gripperData: [], gripper2Data: [],
        gripperEvents: [], gripper2Events: [],
        hasPosition: false, hasRotation: false, hasGripper: false, hasGripper2: false,
        classification: undefined
      };
    }

    const dims = actionsData.actions[0].length;
    const labels = actionsData.dimension_labels ?? null;
    const cls = classifyActionDimensions(labels, dims);

    const posMag = computeMagnitude(actionsData.actions, cls.group1.indices);
    const rotMag = computeMagnitude(actionsData.actions, cls.group2.indices);

    // Gripper 1
    const grip: number[] = cls.grippers.length > 0
      ? actionsData.actions.map((a) => (cls.grippers[0].index < a.length ? a[cls.grippers[0].index] : 0))
      : [];
    const gripMin = grip.length > 0 ? Math.min(...grip) : 0;
    const gripMax = grip.length > 0 ? Math.max(...grip) : 0;
    const gripHasVariation = grip.length > 0 && (gripMax - gripMin) > 0.01;

    // Gripper 2
    const grip2: number[] = cls.grippers.length > 1
      ? actionsData.actions.map((a) => (cls.grippers[1].index < a.length ? a[cls.grippers[1].index] : 0))
      : [];
    const grip2Min = grip2.length > 0 ? Math.min(...grip2) : 0;
    const grip2Max = grip2.length > 0 ? Math.max(...grip2) : 0;
    const grip2HasVariation = grip2.length > 0 && (grip2Max - grip2Min) > 0.01;

    return {
      positionMag: posMag,
      rotationMag: rotMag,
      gripperData: grip,
      gripper2Data: grip2,
      gripperEvents: gripHasVariation ? detectGripperEvents(grip) : [],
      gripper2Events: grip2HasVariation ? detectGripperEvents(grip2) : [],
      hasPosition: posMag.length > 0,
      hasRotation: rotMag.length > 0,
      hasGripper: gripHasVariation,
      hasGripper2: grip2HasVariation,
      classification: cls,
    };
  }, [actionsData]);

  // Normalize data to 0-100 range for SVG rendering
  const normalizeData = (data: number[]) => {
    if (data.length === 0) return [];
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    return data.map((v) => ((v - min) / range) * 100);
  };

  // Create SVG path from data points
  const createPath = (data: number[], height: number) => {
    if (data.length === 0) return "";
    const normalized = normalizeData(data);
    const width = 100;
    const xStep = width / Math.max(normalized.length - 1, 1);

    return normalized
      .map((y, i) => {
        const yPos = height - (y / 100) * height;
        return `${i === 0 ? "M" : "L"} ${i * xStep} ${yPos}`;
      })
      .join(" ");
  };

  if (loading) {
    return (
      <div className="h-24 flex items-center justify-center text-gray-500 text-sm">
        Loading actions data...
      </div>
    );
  }

  if (error || !actionsData || actionsData.actions.length === 0) {
    return (
      <div className="h-24 flex items-center justify-center text-gray-500 text-sm">
        {error || "No action data available"}
      </div>
    );
  }

  const chartHeight = 50;
  const dataLength = actionsData.actions.length;
  const totalGripperEvents = gripperEvents.length + gripper2Events.length;

  return (
    <div className="p-2 bg-gray-50 dark:bg-gray-800 rounded" data-testid="actions-chart">
      <div className="text-xs text-gray-500 mb-2 flex items-center justify-between">
        <span>Actions</span>
        {totalGripperEvents > 0 && (
          <span className="text-green-600 dark:text-green-400">
            {totalGripperEvents} gripper event{totalGripperEvents !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Simplified Actions Chart */}
      <svg
        viewBox={`0 0 100 ${chartHeight}`}
        className="w-full h-14 bg-gray-100 dark:bg-gray-700 rounded"
        preserveAspectRatio="none"
      >
        {/* Group 1 magnitude (blue) */}
        {hasPosition && (
          <path
            d={createPath(positionMag, chartHeight)}
            fill="none"
            stroke={classification?.group1.color ?? "#3b82f6"}
            strokeWidth="1"
            vectorEffect="non-scaling-stroke"
          />
        )}

        {/* Group 2 magnitude (purple) */}
        {hasRotation && (
          <path
            d={createPath(rotationMag, chartHeight)}
            fill="none"
            stroke={classification?.group2.color ?? "#8b5cf6"}
            strokeWidth="1"
            vectorEffect="non-scaling-stroke"
          />
        )}

        {/* Gripper 1 state (green, dashed) */}
        {hasGripper && (
          <path
            d={createPath(gripperData, chartHeight)}
            fill="none"
            stroke={classification?.grippers[0]?.color ?? "#22c55e"}
            strokeWidth="0.8"
            strokeDasharray="2,1"
            vectorEffect="non-scaling-stroke"
          />
        )}

        {/* Gripper 2 state (orange, dashed) */}
        {hasGripper2 && (
          <path
            d={createPath(gripper2Data, chartHeight)}
            fill="none"
            stroke={classification?.grippers[1]?.color ?? "#f97316"}
            strokeWidth="0.8"
            strokeDasharray="2,1"
            vectorEffect="non-scaling-stroke"
          />
        )}

        {/* Gripper event markers (vertical lines) */}
        {gripperEvents.map((frameIdx) => {
          const xPos = (frameIdx / Math.max(dataLength - 1, 1)) * 100;
          return (
            <line
              key={`gripper-event-${frameIdx}`}
              x1={xPos}
              y1="0"
              x2={xPos}
              y2={chartHeight}
              stroke={classification?.grippers[0]?.color ?? "#22c55e"}
              strokeWidth="1"
              strokeDasharray="1,2"
              vectorEffect="non-scaling-stroke"
              opacity="0.7"
            />
          );
        })}

        {/* Gripper 2 event markers */}
        {gripper2Events.map((frameIdx) => {
          const xPos = (frameIdx / Math.max(dataLength - 1, 1)) * 100;
          return (
            <line
              key={`gripper2-event-${frameIdx}`}
              x1={xPos}
              y1="0"
              x2={xPos}
              y2={chartHeight}
              stroke={classification?.grippers[1]?.color ?? "#f97316"}
              strokeWidth="1"
              strokeDasharray="1,2"
              vectorEffect="non-scaling-stroke"
              opacity="0.7"
            />
          );
        })}

        {/* Sync line */}
        <line
          x1={syncPosition}
          y1="0"
          x2={syncPosition}
          y2={chartHeight}
          stroke="white"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
      </svg>

      {/* Legend */}
      <div className="flex justify-center gap-4 mt-2 text-xs">
        {hasPosition && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 rounded" style={{ backgroundColor: classification?.group1.color ?? "#3b82f6" }}></span>
            {classification?.group1.label ?? "Position"}
          </span>
        )}
        {hasRotation && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 rounded" style={{ backgroundColor: classification?.group2.color ?? "#8b5cf6" }}></span>
            {classification?.group2.label ?? "Rotation"}
          </span>
        )}
        {hasGripper && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 rounded" style={{ backgroundColor: classification?.grippers[0]?.color ?? "#22c55e", borderStyle: 'dashed' }}></span>
            {classification?.grippers[0]?.label ?? "Gripper"}
          </span>
        )}
        {hasGripper2 && classification?.grippers[1] && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 rounded" style={{ backgroundColor: classification.grippers[1].color, borderStyle: 'dashed' }}></span>
            {classification.grippers[1].label}
          </span>
        )}
      </div>
    </div>
  );
}
