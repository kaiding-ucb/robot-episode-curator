"use client";

import DatasetBrowser from "@/components/DatasetBrowser";
import type { Modality } from "@/types/api";

interface LeftSidebarProps {
  onSelectEpisode: (datasetId: string, episodeId: string, numFrames: number, modalities?: Modality[]) => void;
  onOpenCompare: () => void;
  onOpenDatasetQuality: () => void;
  onOpenDataManager: () => void;
  selectedDataset: string | null;
}

export default function LeftSidebar({
  onSelectEpisode,
  onOpenCompare,
  onOpenDatasetQuality,
  onOpenDataManager,
  selectedDataset,
}: LeftSidebarProps) {
  return (
    <aside className="w-72 border-r border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-800">
        <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
          Data Viewer
        </h1>
        <p className="text-sm text-gray-500">Robotics Dataset Explorer</p>
      </div>

      {/* Dataset Browser */}
      <div className="flex-1 overflow-auto">
        <DatasetBrowser onSelectEpisode={onSelectEpisode} />
      </div>

      {/* Action Buttons */}
      <div className="p-3 border-t border-gray-200 dark:border-gray-800 space-y-2">
        <button
          onClick={onOpenCompare}
          className="w-full px-3 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors flex items-center justify-center gap-2 text-sm"
          data-testid="open-compare-btn"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
          </svg>
          Compare Datasets
        </button>
        <button
          onClick={onOpenDatasetQuality}
          disabled={!selectedDataset}
          className="w-full px-3 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
          data-testid="open-dataset-quality-btn"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Dataset Quality
        </button>
        <button
          onClick={onOpenDataManager}
          className="w-full px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center justify-center gap-2 text-sm"
          data-testid="open-data-manager-btn"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Manage Downloads
        </button>
      </div>
    </aside>
  );
}
