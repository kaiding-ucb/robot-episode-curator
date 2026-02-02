"use client";

import DatasetBrowser from "@/components/DatasetBrowser";
import type { Modality } from "@/types/api";

interface LeftSidebarProps {
  onSelectEpisode: (datasetId: string, episodeId: string, numFrames: number, modalities?: Modality[]) => void;
  onOpenDataManager: () => void;
}

export default function LeftSidebar({
  onSelectEpisode,
  onOpenDataManager,
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

      {/* Action Button */}
      <div className="p-3 border-t border-gray-200 dark:border-gray-800">
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
