"use client";

import { useState } from "react";
import { useDatasets } from "@/hooks/useApi";
import { useCompare } from "@/hooks/useCompare";
import type { DatasetComparison } from "@/types/compare";

interface ComparePanelProps {
  onClose?: () => void;
}

function ComparisonBar({
  label,
  datasets,
  getValue,
  colorClass = "bg-blue-500",
}: {
  label: string;
  datasets: DatasetComparison[];
  getValue: (d: DatasetComparison) => number;
  colorClass?: string;
}) {
  const maxValue = Math.max(...datasets.map(getValue), 0.01);

  return (
    <div className="mb-4">
      <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
        {label}
      </div>
      <div className="space-y-2">
        {datasets.map((d) => {
          const value = getValue(d);
          const pct = (value / maxValue) * 100;
          return (
            <div key={d.dataset_id} className="flex items-center gap-2">
              <div className="w-24 text-xs text-gray-600 dark:text-gray-400 truncate">
                {d.dataset_id}
              </div>
              <div className="flex-1 h-4 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden">
                <div
                  className={`h-full ${colorClass} transition-all duration-500`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="w-12 text-xs text-right font-medium">
                {Math.round(value * 100)}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Badge({ children, highlight }: { children: React.ReactNode; highlight?: boolean }) {
  return (
    <span
      className={`px-2 py-1 text-xs rounded-full ${
        highlight
          ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
          : "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300"
      }`}
    >
      {children}
    </span>
  );
}

export default function ComparePanel({ onClose }: ComparePanelProps) {
  const { datasets, loading: loadingDatasets } = useDatasets();
  const { comparison, loading, error, compare, reset } = useCompare();
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  const toggleDataset = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleCompare = () => {
    compare(Array.from(selectedIds));
  };

  return (
    <div className="p-6" data-testid="compare-panel">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Compare Datasets
        </h2>
        {onClose && (
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        )}
      </div>

      {/* Dataset Selection */}
      {!comparison && (
        <div className="mb-6">
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
            Select datasets to compare their quality metrics:
          </p>
          {loadingDatasets ? (
            <div className="text-gray-500">Loading datasets...</div>
          ) : (
            <div className="flex flex-wrap gap-2 mb-4">
              {datasets.map((ds) => (
                <button
                  key={ds.id}
                  onClick={() => toggleDataset(ds.id)}
                  className={`px-3 py-2 rounded-lg border transition-colors ${
                    selectedIds.has(ds.id)
                      ? "border-blue-500 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
                      : "border-gray-200 dark:border-gray-700 hover:border-gray-300"
                  }`}
                >
                  {ds.name}
                </button>
              ))}
            </div>
          )}

          <button
            onClick={handleCompare}
            disabled={selectedIds.size === 0 || loading}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
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
                Analyzing...
              </>
            ) : (
              <>Compare {selectedIds.size} Dataset(s)</>
            )}
          </button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg">
          {error}
        </div>
      )}

      {/* Comparison Results */}
      {comparison && (
        <div data-testid="comparison-results">
          {/* Recommendation */}
          <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <div className="text-sm font-medium text-blue-800 dark:text-blue-300 mb-1">
              Recommendation
            </div>
            <p className="text-blue-700 dark:text-blue-400">
              {comparison.recommendation}
            </p>
          </div>

          {/* Best Categories */}
          <div className="flex flex-wrap gap-2 mb-6">
            <Badge highlight={true}>
              Best Overall: {comparison.best_overall}
            </Badge>
            <Badge highlight={true}>
              Best Diversity: {comparison.best_diversity}
            </Badge>
            <Badge highlight={true}>
              Best Temporal: {comparison.best_temporal}
            </Badge>
          </div>

          {/* Comparison Charts */}
          <ComparisonBar
            label="Overall Quality"
            datasets={comparison.datasets}
            getValue={(d) => d.mean_score}
            colorClass="bg-green-500"
          />

          <ComparisonBar
            label="Diversity (Recovery Behaviors)"
            datasets={comparison.datasets}
            getValue={(d) => d.avg_diversity}
            colorClass="bg-purple-500"
          />

          <ComparisonBar
            label="Temporal Quality"
            datasets={comparison.datasets}
            getValue={(d) => d.avg_temporal}
            colorClass="bg-blue-500"
          />

          <ComparisonBar
            label="Episodes with Recovery"
            datasets={comparison.datasets}
            getValue={(d) => d.pct_with_recovery}
            colorClass="bg-orange-500"
          />

          {/* Reset Button */}
          <button
            onClick={reset}
            className="mt-4 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800"
          >
            Compare Different Datasets
          </button>
        </div>
      )}
    </div>
  );
}
