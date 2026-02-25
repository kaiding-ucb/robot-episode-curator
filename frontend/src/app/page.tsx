"use client";

import { useState } from "react";
import LeftSidebar from "./layout/LeftSidebar";
import MainContent from "./layout/MainContent";
import Modals from "./layout/Modals";
import type { Modality } from "@/types/api";

// Extract task name from LIBERO episode ID and convert to title case
// Format: libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo/demo_0
// Backend stores as: "Kitchen Scene1 Open The Bottom Drawer Of The Cabinet Demo"
function extractTaskName(episodeId: string | null): string | null {
  if (!episodeId) return null;
  const parts = episodeId.split("/");
  if (parts.length >= 2) {
    const taskFolder = parts[1];
    // Match Python's str.title(): lowercase then capitalize first letter of each word
    return taskFolder
      .replace(/_/g, " ")
      .toLowerCase()
      .replace(/\b\w/g, (c) => c.toUpperCase());
  }
  return null;
}

export default function Home() {
  // Shared state - both sidebars receive via props
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [selectedEpisode, setSelectedEpisode] = useState<string | null>(null);
  const [selectedEpisodeDisplayName, setSelectedEpisodeDisplayName] = useState<string | null>(null);
  const [selectedEpisodeFrameCount, setSelectedEpisodeFrameCount] = useState<number>(0);
  const [selectedModalities, setSelectedModalities] = useState<Modality[]>(["rgb"]);
  const [targetFrame, setTargetFrame] = useState<number | null>(null);

  // Modal state
  const [showDataManager, setShowDataManager] = useState(false);
  const [showDatasetQuality, setShowDatasetQuality] = useState(false);
  const [showCompare, setShowCompare] = useState(false);
  const [showDatasetAnalysis, setShowDatasetAnalysis] = useState(false);
  const [navigatedFromAnalysis, setNavigatedFromAnalysis] = useState(false);

  const handleSelectEpisode = (datasetId: string, episodeId: string, numFrames: number, modalities?: Modality[], displayName?: string) => {
    setSelectedDataset(datasetId);
    setSelectedEpisode(episodeId);
    setSelectedEpisodeDisplayName(displayName || episodeId);
    setSelectedEpisodeFrameCount(numFrames);
    setSelectedModalities(modalities || ["rgb"]);
    setTargetFrame(null);
  };

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-950" data-testid="app-layout">
      <LeftSidebar
        onSelectEpisode={handleSelectEpisode}
        onSelectDataset={(id) => {
          setSelectedDataset(id);
          setSelectedEpisode(null);
          setSelectedEpisodeDisplayName(null);
          setSelectedEpisodeFrameCount(0);
        }}
        onOpenDataManager={() => setShowDataManager(true)}
        onOpenAnalysis={() => setShowDatasetAnalysis(true)}
      />
      <MainContent
        selectedDataset={selectedDataset}
        selectedEpisode={selectedEpisode}
        selectedEpisodeDisplayName={selectedEpisodeDisplayName}
        totalFrames={selectedEpisodeFrameCount}
        targetFrame={targetFrame}
        onFrameChange={() => setTargetFrame(null)}
        availableModalities={selectedModalities}
      />
      {/* Floating "Back to Analysis" pill — shown after navigating from analysis modal */}
      {navigatedFromAnalysis && !showDatasetAnalysis && (
        <button
          onClick={() => {
            setShowDatasetAnalysis(true);
            setNavigatedFromAnalysis(false);
          }}
          className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-full shadow-lg transition-colors"
          data-testid="back-to-analysis-btn"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Back to Analysis
        </button>
      )}
      <Modals
        showDataManager={showDataManager}
        showDatasetQuality={showDatasetQuality}
        showCompare={showCompare}
        showDatasetAnalysis={showDatasetAnalysis}
        selectedDataset={selectedDataset}
        onCloseDataManager={() => setShowDataManager(false)}
        onCloseDatasetQuality={() => setShowDatasetQuality(false)}
        onCloseCompare={() => setShowCompare(false)}
        onCloseDatasetAnalysis={() => {
          setShowDatasetAnalysis(false);
          setNavigatedFromAnalysis(false);
        }}
        onNavigateToEpisode={(datasetId, episodeId, numFrames, targetFrameIdx) => {
          handleSelectEpisode(datasetId, episodeId, numFrames);
          if (targetFrameIdx != null) setTargetFrame(targetFrameIdx);
          setShowDatasetAnalysis(false);
          setNavigatedFromAnalysis(true);
        }}
      />
    </div>
  );
}
