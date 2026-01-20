"use client";

import { useState } from "react";
import DatasetBrowser from "@/components/DatasetBrowser";
import EpisodeViewer from "@/components/EpisodeViewer";
import DataManager from "@/components/DataManager";
import QualityPanel from "@/components/QualityPanel";
import DatasetQualityDashboard from "@/components/DatasetQualityDashboard";
import ComparePanel from "@/components/ComparePanel";

export default function Home() {
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [selectedEpisode, setSelectedEpisode] = useState<string | null>(null);
  const [selectedEpisodeFrameCount, setSelectedEpisodeFrameCount] = useState<number>(0);
  const [showDataManager, setShowDataManager] = useState(false);
  const [showDatasetQuality, setShowDatasetQuality] = useState(false);
  const [showCompare, setShowCompare] = useState(false);
  const [targetFrame, setTargetFrame] = useState<number | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);

  const handleSelectEpisode = (datasetId: string, episodeId: string, numFrames: number) => {
    setSelectedDataset(datasetId);
    setSelectedEpisode(episodeId);
    setSelectedEpisodeFrameCount(numFrames);
    setTargetFrame(null); // Reset target frame when episode changes
    setSelectedMetric(null); // Reset selected metric when episode changes
  };

  const handleJumpToFrame = (frame: number) => {
    setTargetFrame(frame);
  };

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-950" data-testid="app-layout">
      {/* Left Sidebar - Dataset Browser */}
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
          <DatasetBrowser onSelectEpisode={handleSelectEpisode} />
        </div>

        {/* Action Buttons */}
        <div className="p-3 border-t border-gray-200 dark:border-gray-800 space-y-2">
          <button
            onClick={() => setShowCompare(true)}
            className="w-full px-3 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors flex items-center justify-center gap-2 text-sm"
            data-testid="open-compare-btn"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
            </svg>
            Compare Datasets
          </button>
          <button
            onClick={() => setShowDatasetQuality(true)}
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
            onClick={() => setShowDataManager(true)}
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

      {/* Main Content - Episode Viewer */}
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
            totalFrames={selectedEpisodeFrameCount}
            targetFrame={targetFrame}
            onFrameChange={() => setTargetFrame(null)}
            selectedMetric={selectedMetric}
          />
        </div>
      </main>

      {/* Right Sidebar - Quality Panel */}
      <aside className="w-72 border-l border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 overflow-auto">
        <div className="p-3 border-b border-gray-200 dark:border-gray-800">
          <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wider">
            Quality Analysis
          </h2>
        </div>
        <QualityPanel
          datasetId={selectedDataset}
          episodeId={selectedEpisode}
          onJumpToFrame={handleJumpToFrame}
          selectedMetric={selectedMetric}
          onSelectMetric={setSelectedMetric}
        />
      </aside>

      {/* Data Manager Modal */}
      {showDataManager && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setShowDataManager(false)}
            data-testid="modal-backdrop"
          />
          <div className="relative w-full max-w-2xl max-h-[80vh] overflow-auto bg-white dark:bg-gray-900 rounded-lg shadow-xl">
            <DataManager onClose={() => setShowDataManager(false)} />
          </div>
        </div>
      )}

      {/* Dataset Quality Modal */}
      {showDatasetQuality && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setShowDatasetQuality(false)}
            data-testid="quality-modal-backdrop"
          />
          <div className="relative w-full max-w-3xl max-h-[80vh] overflow-auto bg-white dark:bg-gray-900 rounded-lg shadow-xl">
            <DatasetQualityDashboard
              datasetId={selectedDataset}
              onClose={() => setShowDatasetQuality(false)}
            />
          </div>
        </div>
      )}

      {/* Compare Modal */}
      {showCompare && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setShowCompare(false)}
            data-testid="compare-modal-backdrop"
          />
          <div className="relative w-full max-w-4xl max-h-[80vh] overflow-auto bg-white dark:bg-gray-900 rounded-lg shadow-xl">
            <ComparePanel onClose={() => setShowCompare(false)} />
          </div>
        </div>
      )}
    </div>
  );
}
