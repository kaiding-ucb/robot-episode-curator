"use client";

import DatasetBrowser from "@/components/DatasetBrowser";
import type { Modality } from "@/types/api";

interface LeftSidebarProps {
  onSelectEpisode: (datasetId: string, episodeId: string, numFrames: number, modalities?: Modality[], displayName?: string) => void;
  onSelectDataset: (datasetId: string | null) => void;
  onOpenDataManager: () => void;
  onOpenAnalysis: () => void;
}

export default function LeftSidebar({
  onSelectEpisode,
  onSelectDataset,
  onOpenDataManager,
  onOpenAnalysis,
}: LeftSidebarProps) {
  return (
    <aside className="w-72 border-r border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-800">
        <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
          Robot Data Viewer
        </h1>
        <p className="text-sm text-gray-500">for Huggingface datasets</p>
      </div>

      {/* Dataset Browser */}
      <div className="flex-1 overflow-auto">
        <DatasetBrowser onSelectEpisode={onSelectEpisode} onSelectDataset={onSelectDataset} />
      </div>

      {/* Action Buttons */}
      <div className="p-3 border-t border-gray-200 dark:border-gray-800 space-y-2">
        <button
          onClick={onOpenAnalysis}
          className="w-full px-3 py-2 bg-gray-900 text-white rounded-lg hover:bg-black transition-colors flex items-center justify-center gap-2 text-sm dark:bg-gray-700 dark:hover:bg-gray-600"
          data-testid="open-analysis-btn"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Analyze Dataset
        </button>
        <button
          onClick={onOpenDataManager}
          className="w-full px-3 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors flex items-center justify-center gap-2 text-sm dark:bg-gray-600 dark:hover:bg-gray-500"
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
