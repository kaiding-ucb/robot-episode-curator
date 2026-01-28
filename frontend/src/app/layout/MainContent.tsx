"use client";

import EpisodeViewer from "@/components/EpisodeViewer";
import type { Modality } from "@/types/api";

interface MainContentProps {
  selectedDataset: string | null;
  selectedEpisode: string | null;
  totalFrames: number;
  targetFrame: number | null;
  onFrameChange: () => void;
  availableModalities?: Modality[];
}

export default function MainContent({
  selectedDataset,
  selectedEpisode,
  totalFrames,
  targetFrame,
  onFrameChange,
  availableModalities = ["rgb"],
}: MainContentProps) {
  return (
    <main className="flex-1 flex flex-col min-w-0">
      {/* Top Bar */}
      <header className="h-12 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 flex items-center px-4 shrink-0">
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
      </header>

      {/* Episode Viewer */}
      <div className="flex-1 bg-gray-50 dark:bg-gray-900 overflow-hidden">
        <EpisodeViewer
          datasetId={selectedDataset}
          episodeId={selectedEpisode}
          totalFrames={totalFrames}
          targetFrame={targetFrame}
          onFrameChange={onFrameChange}
          availableModalities={availableModalities}
        />
      </div>
    </main>
  );
}
