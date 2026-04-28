"use client";

import { useCallback, useEffect, useState } from "react";
import DatasetBrowser from "@/components/DatasetBrowser";
import { fetchCacheSize, clearCache } from "@/hooks/useApi";
import type { Modality } from "@/types/api";

interface LeftSidebarProps {
  onSelectEpisode: (datasetId: string, episodeId: string, numFrames: number, modalities?: Modality[], displayName?: string) => void;
  onSelectDataset: (datasetId: string | null) => void;
  onOpenAnalysis: () => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export default function LeftSidebar({
  onSelectEpisode,
  onSelectDataset,
  onOpenAnalysis,
}: LeftSidebarProps) {
  const [cacheBytes, setCacheBytes] = useState<number | null>(null);
  const [clearing, setClearing] = useState(false);

  const refreshSize = useCallback(async () => {
    try {
      const res = await fetchCacheSize();
      setCacheBytes(res.total_bytes);
    } catch {
      setCacheBytes(null);
    }
  }, []);

  useEffect(() => {
    void refreshSize();
  }, [refreshSize]);

  const handleClear = useCallback(async () => {
    if (clearing) return;
    if (cacheBytes != null && cacheBytes > 0) {
      const ok = typeof window !== "undefined"
        ? window.confirm(`Clear ${formatBytes(cacheBytes)} of cached frames and metadata?`)
        : true;
      if (!ok) return;
    }
    setClearing(true);
    try {
      await clearCache();
    } catch (e) {
      console.error("clearCache failed", e);
    } finally {
      setClearing(false);
      void refreshSize();
    }
  }, [cacheBytes, clearing, refreshSize]);

  return (
    <aside className="w-72 border-r border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-800">
        <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
          Robot Episode Curator
        </h1>
        <p className="text-sm text-gray-500">for Lerobot datasets</p>
      </div>

      {/* Dataset Browser */}
      <div className="flex-1 overflow-auto">
        <DatasetBrowser onSelectEpisode={onSelectEpisode} onSelectDataset={onSelectDataset} />
      </div>

      {/* Action Buttons */}
      <div className="p-3 border-t border-gray-200 dark:border-gray-800 space-y-2">
        <button
          onClick={onOpenAnalysis}
          className="w-full px-3 py-2 bg-gray-900 text-white rounded-lg hover:bg-black transition-colors flex items-center justify-center gap-2 text-sm dark:bg-gray-700 dark:hover:bg-gray-600"
          data-testid="open-analysis-btn"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Analyze Dataset
        </button>
        <button
          onClick={handleClear}
          disabled={clearing}
          className="w-full px-3 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2 text-sm dark:bg-gray-600 dark:hover:bg-gray-500"
          data-testid="clear-cache-btn"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6M1 7h22M9 7V3a2 2 0 012-2h2a2 2 0 012 2v4" />
          </svg>
          {clearing ? "Clearing…" : (
            <>
              Clear cache
              {cacheBytes != null && (
                <span className="ml-1 opacity-80">· {formatBytes(cacheBytes)}</span>
              )}
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
