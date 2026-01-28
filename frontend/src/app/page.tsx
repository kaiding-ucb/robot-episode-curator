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
  const [selectedEpisodeFrameCount, setSelectedEpisodeFrameCount] = useState<number>(0);
  const [selectedModalities, setSelectedModalities] = useState<Modality[]>(["rgb"]);
  const [targetFrame, setTargetFrame] = useState<number | null>(null);

  // Modal state
  const [showDataManager, setShowDataManager] = useState(false);
  const [showDatasetQuality, setShowDatasetQuality] = useState(false);
  const [showCompare, setShowCompare] = useState(false);

  const handleSelectEpisode = (datasetId: string, episodeId: string, numFrames: number, modalities?: Modality[]) => {
    setSelectedDataset(datasetId);
    setSelectedEpisode(episodeId);
    setSelectedEpisodeFrameCount(numFrames);
    setSelectedModalities(modalities || ["rgb"]);
    setTargetFrame(null);
  };

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-950" data-testid="app-layout">
      <LeftSidebar
        onSelectEpisode={handleSelectEpisode}
        onOpenDataManager={() => setShowDataManager(true)}
      />
      <MainContent
        selectedDataset={selectedDataset}
        selectedEpisode={selectedEpisode}
        totalFrames={selectedEpisodeFrameCount}
        targetFrame={targetFrame}
        onFrameChange={() => setTargetFrame(null)}
        availableModalities={selectedModalities}
      />
      <Modals
        showDataManager={showDataManager}
        showDatasetQuality={showDatasetQuality}
        showCompare={showCompare}
        selectedDataset={selectedDataset}
        onCloseDataManager={() => setShowDataManager(false)}
        onCloseDatasetQuality={() => setShowDatasetQuality(false)}
        onCloseCompare={() => setShowCompare(false)}
      />
    </div>
  );
}
