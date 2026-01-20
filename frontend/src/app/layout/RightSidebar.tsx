"use client";

import QualityPanel from "@/components/QualityPanel";

interface RightSidebarProps {
  datasetId: string | null;
  episodeId: string | null;
  onJumpToFrame: (frame: number) => void;
  selectedMetric: string | null;
  onSelectMetric: (metric: string | null) => void;
}

export default function RightSidebar({
  datasetId,
  episodeId,
  onJumpToFrame,
  selectedMetric,
  onSelectMetric,
}: RightSidebarProps) {
  return (
    <aside className="w-72 border-l border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 overflow-auto">
      <div className="p-3 border-b border-gray-200 dark:border-gray-800">
        <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wider">
          Quality Analysis
        </h2>
      </div>
      <QualityPanel
        datasetId={datasetId}
        episodeId={episodeId}
        onJumpToFrame={onJumpToFrame}
        selectedMetric={selectedMetric}
        onSelectMetric={onSelectMetric}
      />
    </aside>
  );
}
