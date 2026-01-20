"use client";

import { useDatasetQuality } from "@/hooks/useQuality";
import type { DatasetQualityStats } from "@/types/quality";

interface DatasetQualityDashboardProps {
  datasetId: string | null;
  onClose?: () => void;
}

function StatCard({ label, value, suffix = "" }: { label: string; value: string | number; suffix?: string }) {
  return (
    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
      <div className="text-2xl font-bold text-gray-900 dark:text-white">
        {value}{suffix}
      </div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}

function GradeDistribution({ counts }: { counts: Record<string, number> }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const grades = ["A", "B", "C", "D", "F"];
  const colors: Record<string, string> = {
    A: "bg-green-500",
    B: "bg-blue-500",
    C: "bg-yellow-500",
    D: "bg-orange-500",
    F: "bg-red-500",
  };

  return (
    <div className="mb-4">
      <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
        Grade Distribution
      </h4>
      <div className="flex h-4 rounded-full overflow-hidden">
        {grades.map((grade) => {
          const count = counts[grade] || 0;
          const pct = total > 0 ? (count / total) * 100 : 0;
          if (pct === 0) return null;
          return (
            <div
              key={grade}
              className={`${colors[grade]} relative group`}
              style={{ width: `${pct}%` }}
            >
              <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-gray-900 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 whitespace-nowrap z-10">
                {grade}: {count} ({Math.round(pct)}%)
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex justify-between mt-1">
        {grades.map((grade) => (
          <span key={grade} className="text-xs text-gray-500">
            {grade}: {counts[grade] || 0}
          </span>
        ))}
      </div>
    </div>
  );
}

function QualityBreakdown({ stats }: { stats: DatasetQualityStats }) {
  return (
    <div className="mb-4">
      <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
        Quality Breakdown
      </h4>
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">Temporal</span>
          <div className="flex items-center gap-2">
            <div className="w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500"
                style={{ width: `${stats.avg_temporal * 100}%` }}
              />
            </div>
            <span className="text-sm font-medium w-10 text-right">
              {Math.round(stats.avg_temporal * 100)}%
            </span>
          </div>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">Diversity</span>
          <div className="flex items-center gap-2">
            <div className="w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500"
                style={{ width: `${stats.avg_diversity * 100}%` }}
              />
            </div>
            <span className="text-sm font-medium w-10 text-right">
              {Math.round(stats.avg_diversity * 100)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function QualityFlags({ stats }: { stats: DatasetQualityStats }) {
  return (
    <div className="mb-4">
      <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
        Episode Characteristics
      </h4>
      <div className="grid grid-cols-4 gap-2">
        <div className="text-center p-2 bg-purple-50 dark:bg-purple-900/20 rounded">
          <div className="text-lg font-bold text-purple-600 dark:text-purple-400">
            {Math.round(stats.pct_with_recovery * 100)}%
          </div>
          <div className="text-xs text-gray-500">With Recovery</div>
        </div>
        <div className="text-center p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
          <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
            {Math.round(stats.pct_diverse * 100)}%
          </div>
          <div className="text-xs text-gray-500">Diverse</div>
        </div>
        <div className="text-center p-2 bg-green-50 dark:bg-green-900/20 rounded">
          <div className="text-lg font-bold text-green-600 dark:text-green-400">
            {Math.round(stats.pct_smooth * 100)}%
          </div>
          <div className="text-xs text-gray-500">Smooth</div>
        </div>
        <div className="text-center p-2 bg-cyan-50 dark:bg-cyan-900/20 rounded">
          <div className="text-lg font-bold text-cyan-600 dark:text-cyan-400">
            {Math.round(stats.pct_well_synced * 100)}%
          </div>
          <div className="text-xs text-gray-500">Well Synced</div>
        </div>
      </div>
    </div>
  );
}

export default function DatasetQualityDashboard({
  datasetId,
  onClose,
}: DatasetQualityDashboardProps) {
  const { stats, loading, error } = useDatasetQuality(datasetId);

  if (!datasetId) {
    return (
      <div className="p-6 text-center text-gray-500" data-testid="no-dataset-quality">
        Select a dataset to view quality statistics
      </div>
    );
  }

  if (loading) {
    return (
      <div className="p-6" data-testid="loading-dataset-quality">
        <div className="flex items-center justify-center gap-3">
          <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
          <span className="text-gray-500">Analyzing dataset quality...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-center text-red-500" data-testid="error-dataset-quality">
        {error}
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="p-6 text-center text-gray-500">
        No quality data available
      </div>
    );
  }

  return (
    <div className="p-6" data-testid="dataset-quality-dashboard">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Dataset Quality: {stats.dataset_id}
        </h2>
        {onClose && (
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        <StatCard label="Episodes" value={stats.num_episodes} />
        <StatCard label="Mean Score" value={Math.round(stats.mean_score * 100)} suffix="%" />
        <StatCard label="Best (P90)" value={Math.round(stats.p90_score * 100)} suffix="%" />
        <StatCard label="Worst (P10)" value={Math.round(stats.p10_score * 100)} suffix="%" />
      </div>

      {/* Grade Distribution */}
      <GradeDistribution counts={stats.grade_counts} />

      {/* Quality Breakdown */}
      <QualityBreakdown stats={stats} />

      {/* Episode Characteristics */}
      <QualityFlags stats={stats} />

      {/* Score Distribution */}
      <div className="text-sm text-gray-500">
        <p>
          Score range: {Math.round(stats.min_score * 100)}% - {Math.round(stats.max_score * 100)}%
          (σ = {(stats.std_score * 100).toFixed(1)}%)
        </p>
      </div>
    </div>
  );
}
