"use client";

import DatasetQualityDashboard from "@/components/DatasetQualityDashboard";
import ComparePanel from "@/components/ComparePanel";
import DatasetAnalysis from "@/components/DatasetAnalysis";

interface ModalsProps {
  showDatasetQuality: boolean;
  showCompare: boolean;
  showDatasetAnalysis: boolean;
  selectedDataset: string | null;
  onCloseDatasetQuality: () => void;
  onCloseCompare: () => void;
  onCloseDatasetAnalysis: () => void;
  onNavigateToEpisode?: (datasetId: string, episodeId: string, numFrames: number, targetFrame?: number) => void;
}

export default function Modals({
  showDatasetQuality,
  showCompare,
  showDatasetAnalysis,
  selectedDataset,
  onCloseDatasetQuality,
  onCloseCompare,
  onCloseDatasetAnalysis,
  onNavigateToEpisode,
}: ModalsProps) {
  return (
    <>
      {/* Dataset Quality Modal */}
      {showDatasetQuality && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={onCloseDatasetQuality}
            data-testid="quality-modal-backdrop"
          />
          <div className="relative w-full max-w-3xl max-h-[80vh] overflow-auto bg-white dark:bg-gray-900 rounded-lg shadow-xl">
            <DatasetQualityDashboard
              datasetId={selectedDataset}
              onClose={onCloseDatasetQuality}
            />
          </div>
        </div>
      )}

      {/* Compare Modal */}
      {showCompare && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={onCloseCompare}
            data-testid="compare-modal-backdrop"
          />
          <div className="relative w-full max-w-4xl max-h-[80vh] overflow-auto bg-white dark:bg-gray-900 rounded-lg shadow-xl">
            <ComparePanel onClose={onCloseCompare} />
          </div>
        </div>
      )}

      {/* Dataset Analysis Modal — always mounted, hidden via CSS to preserve state */}
      <div
        className="fixed inset-0 z-50 flex items-center justify-center"
        style={{ display: showDatasetAnalysis ? undefined : "none" }}
      >
        <div
          className="absolute inset-0 bg-black/50"
          onClick={onCloseDatasetAnalysis}
          data-testid="analysis-modal-backdrop"
        />
        <div className="relative w-full max-w-5xl max-h-[85vh] overflow-auto bg-white dark:bg-gray-900 rounded-lg shadow-xl">
          <DatasetAnalysis
            datasetId={selectedDataset}
            visible={showDatasetAnalysis}
            onClose={onCloseDatasetAnalysis}
            onNavigateToEpisode={onNavigateToEpisode}
          />
        </div>
      </div>
    </>
  );
}
