"use client";

import { useState } from "react";
import LeftSidebar from "./layout/LeftSidebar";
import MainContent from "./layout/MainContent";
import RightSidebar from "./layout/RightSidebar";
import Modals from "./layout/Modals";

export default function Home() {
  // Shared state - both sidebars receive via props
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [selectedEpisode, setSelectedEpisode] = useState<string | null>(null);
  const [selectedEpisodeFrameCount, setSelectedEpisodeFrameCount] = useState<number>(0);
  const [targetFrame, setTargetFrame] = useState<number | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);

  // Modal state
  const [showDataManager, setShowDataManager] = useState(false);
  const [showDatasetQuality, setShowDatasetQuality] = useState(false);
  const [showCompare, setShowCompare] = useState(false);

  const handleSelectEpisode = (datasetId: string, episodeId: string, numFrames: number) => {
    setSelectedDataset(datasetId);
    setSelectedEpisode(episodeId);
    setSelectedEpisodeFrameCount(numFrames);
    setTargetFrame(null);
    setSelectedMetric(null);
  };

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-950" data-testid="app-layout">
      <LeftSidebar
        onSelectEpisode={handleSelectEpisode}
        onOpenCompare={() => setShowCompare(true)}
        onOpenDatasetQuality={() => setShowDatasetQuality(true)}
        onOpenDataManager={() => setShowDataManager(true)}
        selectedDataset={selectedDataset}
      />
      <MainContent
        selectedDataset={selectedDataset}
        selectedEpisode={selectedEpisode}
        totalFrames={selectedEpisodeFrameCount}
        targetFrame={targetFrame}
        onFrameChange={() => setTargetFrame(null)}
        selectedMetric={selectedMetric}
      />
      <RightSidebar
        datasetId={selectedDataset}
        episodeId={selectedEpisode}
        onJumpToFrame={setTargetFrame}
        selectedMetric={selectedMetric}
        onSelectMetric={setSelectedMetric}
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
