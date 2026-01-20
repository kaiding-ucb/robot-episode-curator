"use client";

import DataManager from "@/components/DataManager";
import DatasetQualityDashboard from "@/components/DatasetQualityDashboard";
import ComparePanel from "@/components/ComparePanel";

interface ModalsProps {
  showDataManager: boolean;
  showDatasetQuality: boolean;
  showCompare: boolean;
  selectedDataset: string | null;
  onCloseDataManager: () => void;
  onCloseDatasetQuality: () => void;
  onCloseCompare: () => void;
}

export default function Modals({
  showDataManager,
  showDatasetQuality,
  showCompare,
  selectedDataset,
  onCloseDataManager,
  onCloseDatasetQuality,
  onCloseCompare,
}: ModalsProps) {
  return (
    <>
      {/* Data Manager Modal */}
      {showDataManager && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={onCloseDataManager}
            data-testid="modal-backdrop"
          />
          <div className="relative w-full max-w-2xl max-h-[80vh] overflow-auto bg-white dark:bg-gray-900 rounded-lg shadow-xl">
            <DataManager onClose={onCloseDataManager} />
          </div>
        </div>
      )}

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
    </>
  );
}
