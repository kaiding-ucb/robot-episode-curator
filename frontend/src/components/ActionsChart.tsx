"use client";

import { useMemo } from "react";
import { useActionsData } from "@/hooks/useApi";

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

  // Process actions into meaningful signals
  const { positionMag, rotationMag, gripperData, gripperEvents, hasPosition, hasRotation, hasGripper } = useMemo(() => {
    if (!actionsData || actionsData.actions.length === 0) {
      return {
        positionMag: [], rotationMag: [], gripperData: [], gripperEvents: [],
        hasPosition: false, hasRotation: false, hasGripper: false
      };
    }

    const dims = actionsData.actions[0].length;
    const labels = actionsData.dimension_labels || [];

    // Try to identify dimensions by labels or assume standard 7D format
    // Standard: [x, y, z, rx, ry, rz, gripper]
    let posIndices = [0, 1, 2]; // x, y, z
    let rotIndices = [3, 4, 5]; // rx, ry, rz
    let gripperIdx = dims >= 7 ? 6 : -1;

    // Override if labels suggest different structure
    if (labels.length > 0) {
      const lowerLabels = labels.map(l => l.toLowerCase());
      const xIdx = lowerLabels.findIndex(l => l === 'x' || l.includes('pos_x'));
      const yIdx = lowerLabels.findIndex(l => l === 'y' || l.includes('pos_y'));
      const zIdx = lowerLabels.findIndex(l => l === 'z' || l.includes('pos_z'));
      const rxIdx = lowerLabels.findIndex(l => l === 'rx' || l.includes('rot_x') || l.includes('roll'));
      const ryIdx = lowerLabels.findIndex(l => l === 'ry' || l.includes('rot_y') || l.includes('pitch'));
      const rzIdx = lowerLabels.findIndex(l => l === 'rz' || l.includes('rot_z') || l.includes('yaw'));
      const gIdx = lowerLabels.findIndex(l => l.includes('gripper') || l.includes('grip'));

      if (xIdx >= 0 && yIdx >= 0 && zIdx >= 0) posIndices = [xIdx, yIdx, zIdx];
      if (rxIdx >= 0 && ryIdx >= 0 && rzIdx >= 0) rotIndices = [rxIdx, ryIdx, rzIdx];
      if (gIdx >= 0) gripperIdx = gIdx;
    }

    // Extract and compute magnitudes
    const posMag: number[] = [];
    const rotMag: number[] = [];
    const grip: number[] = [];

    for (const action of actionsData.actions) {
      // Position magnitude
      if (posIndices.every(i => i < dims)) {
        const px = action[posIndices[0]] || 0;
        const py = action[posIndices[1]] || 0;
        const pz = action[posIndices[2]] || 0;
        posMag.push(Math.sqrt(px * px + py * py + pz * pz));
      }

      // Rotation magnitude
      if (rotIndices.every(i => i < dims)) {
        const rx = action[rotIndices[0]] || 0;
        const ry = action[rotIndices[1]] || 0;
        const rz = action[rotIndices[2]] || 0;
        rotMag.push(Math.sqrt(rx * rx + ry * ry + rz * rz));
      }

      // Gripper
      if (gripperIdx >= 0 && gripperIdx < dims) {
        grip.push(action[gripperIdx]);
      }
    }

    // Check if gripper has actual variation (not constant throughout)
    const gripperMin = grip.length > 0 ? Math.min(...grip) : 0;
    const gripperMax = grip.length > 0 ? Math.max(...grip) : 0;
    const gripperHasVariation = grip.length > 0 && (gripperMax - gripperMin) > 0.01;

    return {
      positionMag: posMag,
      rotationMag: rotMag,
      gripperData: grip,
      gripperEvents: gripperHasVariation ? detectGripperEvents(grip) : [],
      hasPosition: posMag.length > 0,
      hasRotation: rotMag.length > 0,
      hasGripper: gripperHasVariation,  // Only show gripper if it actually varies
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

  return (
    <div className="p-2 bg-gray-50 dark:bg-gray-800 rounded" data-testid="actions-chart">
      <div className="text-xs text-gray-500 mb-2 flex items-center justify-between">
        <span>Actions</span>
        {gripperEvents.length > 0 && (
          <span className="text-green-600 dark:text-green-400">
            {gripperEvents.length} gripper event{gripperEvents.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Simplified Actions Chart */}
      <svg
        viewBox={`0 0 100 ${chartHeight}`}
        className="w-full h-14 bg-gray-100 dark:bg-gray-700 rounded"
        preserveAspectRatio="none"
      >
        {/* Position magnitude (blue) */}
        {hasPosition && (
          <path
            d={createPath(positionMag, chartHeight)}
            fill="none"
            stroke="#3b82f6"
            strokeWidth="1"
            vectorEffect="non-scaling-stroke"
          />
        )}

        {/* Rotation magnitude (purple) */}
        {hasRotation && (
          <path
            d={createPath(rotationMag, chartHeight)}
            fill="none"
            stroke="#8b5cf6"
            strokeWidth="1"
            vectorEffect="non-scaling-stroke"
          />
        )}

        {/* Gripper state (green, dashed) */}
        {hasGripper && (
          <path
            d={createPath(gripperData, chartHeight)}
            fill="none"
            stroke="#22c55e"
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
              stroke="#22c55e"
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
            <span className="w-3 h-0.5 bg-blue-500 rounded"></span>
            Position
          </span>
        )}
        {hasRotation && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-purple-500 rounded"></span>
            Rotation
          </span>
        )}
        {hasGripper && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-green-500 rounded" style={{ borderStyle: 'dashed' }}></span>
            Gripper
          </span>
        )}
      </div>
    </div>
  );
}
