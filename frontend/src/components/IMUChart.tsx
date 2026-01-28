"use client";

import { useMemo } from "react";
import { useIMUData } from "@/hooks/useApi";

interface IMUChartProps {
  episodeId: string | null;
  datasetId: string | null;
  currentFrame: number;
  totalFrames: number;
}

export default function IMUChart({
  episodeId,
  datasetId,
  currentFrame,
  totalFrames,
}: IMUChartProps) {
  const { imuData, loading, error } = useIMUData(episodeId, datasetId);

  // Calculate the position of the sync line based on current frame
  const syncPosition = useMemo(() => {
    if (!imuData || totalFrames === 0) return 0;
    return (currentFrame / totalFrames) * 100;
  }, [currentFrame, totalFrames, imuData]);

  // Calculate magnitude from X, Y, Z components: √(X² + Y² + Z²)
  const calculateMagnitude = (x: number[], y: number[], z: number[]) => {
    return x.map((_, i) => Math.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2));
  };

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
    const xStep = width / (normalized.length - 1 || 1);

    return normalized
      .map((y, i) => {
        const yPos = height - (y / 100) * height;
        return `${i === 0 ? "M" : "L"} ${i * xStep} ${yPos}`;
      })
      .join(" ");
  };

  // Calculate magnitudes
  const { accelMagnitude, gyroMagnitude } = useMemo(() => {
    if (!imuData) return { accelMagnitude: [], gyroMagnitude: [] };
    return {
      accelMagnitude: calculateMagnitude(imuData.accel_x, imuData.accel_y, imuData.accel_z),
      gyroMagnitude: calculateMagnitude(imuData.gyro_x, imuData.gyro_y, imuData.gyro_z),
    };
  }, [imuData]);

  if (loading) {
    return (
      <div className="h-24 flex items-center justify-center text-gray-500 text-sm">
        Loading IMU data...
      </div>
    );
  }

  if (error || !imuData || imuData.timestamps.length === 0) {
    return null;
  }

  const chartHeight = 50;

  return (
    <div className="p-2 bg-gray-50 dark:bg-gray-800 rounded" data-testid="imu-chart">
      <div className="text-xs text-gray-500 mb-2">IMU Sensor Data</div>

      {/* Single combined chart with both magnitudes */}
      <svg
        viewBox={`0 0 100 ${chartHeight}`}
        className="w-full h-14 bg-gray-100 dark:bg-gray-700 rounded"
        preserveAspectRatio="none"
      >
        {/* Acceleration magnitude (blue, filled area) */}
        <path
          d={createPath(accelMagnitude, chartHeight)}
          fill="none"
          stroke="#3b82f6"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
        {/* Gyroscope magnitude (orange) */}
        <path
          d={createPath(gyroMagnitude, chartHeight)}
          fill="none"
          stroke="#f97316"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
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
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-blue-500 rounded"></span>
          Accel
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-orange-500 rounded"></span>
          Gyro
        </span>
      </div>
    </div>
  );
}
