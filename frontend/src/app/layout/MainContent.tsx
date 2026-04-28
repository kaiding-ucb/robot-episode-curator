"use client";

import dynamic from "next/dynamic";
import OnboardingPanel from "@/components/OnboardingPanel";

// Rerun viewer is loaded lazily — its WASM module only initialises in the
// browser, and the bundle is large enough that we don't want it in the
// initial JS payload.
const RerunViewer = dynamic(() => import("@/components/RerunViewer"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full bg-gray-900 text-gray-400">
      <div className="animate-spin w-8 h-8 border-2 border-gray-300 dark:border-gray-600 border-t-transparent rounded-full" />
    </div>
  ),
});

interface MainContentProps {
  selectedDataset: string | null;
  selectedEpisode: string | null;
  selectedEpisodeDisplayName?: string | null;
}

export default function MainContent({
  selectedDataset,
  selectedEpisode,
  selectedEpisodeDisplayName,
}: MainContentProps) {
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
                  {selectedDataset} / {selectedEpisodeDisplayName || selectedEpisode}
                </span>
              </div>
            </div>
          ) : (
            <span className="text-gray-500">Select an episode to view</span>
          )}
        </div>
      </header>

      {/* Main panel: Rerun when an episode is selected, onboarding card otherwise. */}
      <div className="flex-1 bg-gray-50 dark:bg-gray-900 overflow-hidden">
        {selectedEpisode ? (
          <RerunViewer
            datasetId={selectedDataset}
            episodeId={selectedEpisode}
          />
        ) : (
          <OnboardingPanel />
        )}
      </div>
    </main>
  );
}
