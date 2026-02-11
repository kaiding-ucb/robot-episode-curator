"use client";

import { useState } from "react";
import EpisodeViewer from "@/components/EpisodeViewer";
import type { Modality } from "@/types/api";
import dynamic from "next/dynamic";

// Dynamic import for Rerun viewer to avoid SSR issues
const RerunViewer = dynamic(() => import("@/components/RerunViewer"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full bg-gray-900 text-gray-400">
      <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full"></div>
    </div>
  ),
});

interface MainContentProps {
  selectedDataset: string | null;
  selectedEpisode: string | null;
  totalFrames: number;
  targetFrame: number | null;
  onFrameChange: () => void;
  availableModalities?: Modality[];
}

type ViewerMode = "standard" | "rerun";

export default function MainContent({
  selectedDataset,
  selectedEpisode,
  totalFrames,
  targetFrame,
  onFrameChange,
  availableModalities = ["rgb"],
}: MainContentProps) {
  const [viewerMode, setViewerMode] = useState<ViewerMode>("standard");

  return (
    <main className="flex-1 flex flex-col min-w-0">
      {/* Top Bar */}
      <header className="h-12 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 flex items-center justify-between px-4 shrink-0">
        <div>
          {selectedEpisode ? (
            <div className="flex items-center gap-4">
              <div>
                <span className="text-sm text-gray-500">Viewing:</span>
                <span className="ml-2 font-medium text-gray-900 dark:text-white">
                  {selectedDataset} / {selectedEpisode}
                </span>
              </div>
            </div>
          ) : (
            <span className="text-gray-500">Select an episode to view</span>
          )}
        </div>

        {/* Viewer Mode Toggle */}
        {selectedEpisode && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Viewer:</span>
            <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-0.5">
              <button
                onClick={() => setViewerMode("standard")}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                  viewerMode === "standard"
                    ? "bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm"
                    : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                }`}
                data-testid="viewer-mode-standard"
              >
                Standard
              </button>
              <button
                onClick={() => setViewerMode("rerun")}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                  viewerMode === "rerun"
                    ? "bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm"
                    : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                }`}
                data-testid="viewer-mode-rerun"
                title="Open in Rerun multi-modal viewer"
              >
                Rerun
              </button>
            </div>
          </div>
        )}
      </header>

      {/* Episode Viewer */}
      <div className="flex-1 bg-gray-50 dark:bg-gray-900 overflow-hidden">
        {viewerMode === "standard" ? (
          <EpisodeViewer
            datasetId={selectedDataset}
            episodeId={selectedEpisode}
            totalFrames={totalFrames}
            targetFrame={targetFrame}
            onFrameChange={onFrameChange}
            availableModalities={availableModalities}
          />
        ) : (
          <RerunViewer
            datasetId={selectedDataset}
            episodeId={selectedEpisode}
            onClose={() => setViewerMode("standard")}
          />
        )}
      </div>
    </main>
  );
}
