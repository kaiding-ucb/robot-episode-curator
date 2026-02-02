"use client";

import { useState, useEffect } from "react";
import { useDownloads, useEpisodeCache, useAllCaches } from "@/hooks/useApi";

interface DataManagerProps {
  onClose?: () => void;
}

export default function DataManager({ onClose }: DataManagerProps) {
  const { statuses, diskSpace, loading, error, startDownload } = useDownloads();
  const {
    cachedEpisodes,
    cacheStats,
    loading: cacheLoading,
    error: cacheError,
    fetchCachedEpisodes,
    deleteEpisodeCache,
    clearAllCache,
  } = useEpisodeCache();
  const {
    allCaches,
    loading: allCachesLoading,
    error: allCachesError,
    clearing: clearingCache,
    fetchAllCaches,
    clearCache,
  } = useAllCaches();

  const [activeTab, setActiveTab] = useState<"datasets" | "cache" | "storage">("datasets");
  const [deletingEpisode, setDeletingEpisode] = useState<string | null>(null);
  const [clearingAll, setClearingAll] = useState(false);

  // Fetch cached episodes when cache tab is active
  useEffect(() => {
    if (activeTab === "cache") {
      fetchCachedEpisodes();
    }
    if (activeTab === "storage") {
      fetchAllCaches();
    }
  }, [activeTab, fetchCachedEpisodes, fetchAllCaches]);

  const handleDeleteEpisode = async (datasetId: string, episodeId: string) => {
    const key = `${datasetId}/${episodeId}`;
    setDeletingEpisode(key);
    await deleteEpisodeCache(datasetId, episodeId);
    setDeletingEpisode(null);
  };

  const handleClearAll = async () => {
    setClearingAll(true);
    await clearAllCache();
    setClearingAll(false);
  };

  const getStatusBadge = (status: string) => {
    const styles: Record<string, string> = {
      ready: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
      downloading: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300",
      not_downloaded: "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300",
      error: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
    };
    return styles[status] || styles.not_downloaded;
  };

  const formatSize = (mb: number) => {
    if (mb >= 1000) {
      return `${(mb / 1000).toFixed(1)} GB`;
    }
    return `${mb.toFixed(0)} MB`;
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString();
  };

  if (loading && activeTab === "datasets") {
    return (
      <div className="p-6 text-gray-500" data-testid="loading-downloads">
        Loading download status...
      </div>
    );
  }

  return (
    <div className="p-6" data-testid="data-manager">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Data Manager
        </h2>
        {onClose && (
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
            data-testid="close-modal"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-4 border-b border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setActiveTab("datasets")}
          className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
            activeTab === "datasets"
              ? "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white border-b-2 border-blue-500"
              : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          }`}
          data-testid="tab-datasets"
        >
          Datasets
        </button>
        <button
          onClick={() => setActiveTab("cache")}
          className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
            activeTab === "cache"
              ? "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white border-b-2 border-blue-500"
              : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          }`}
          data-testid="tab-cache"
        >
          Cached Episodes
          {cacheStats && cacheStats.episode_count > 0 && (
            <span className="ml-2 px-2 py-0.5 text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 rounded-full">
              {cacheStats.episode_count}
            </span>
          )}
        </button>
        <button
          onClick={() => setActiveTab("storage")}
          className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
            activeTab === "storage"
              ? "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white border-b-2 border-blue-500"
              : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          }`}
          data-testid="tab-storage"
        >
          Storage
          {allCaches && allCaches.total_size_gb > 1 && (
            <span className="ml-2 px-2 py-0.5 text-xs bg-orange-100 dark:bg-orange-900/30 text-orange-800 dark:text-orange-300 rounded-full">
              {allCaches.total_size_gb.toFixed(0)} GB
            </span>
          )}
        </button>
      </div>

      {/* Datasets Tab Content */}
      {activeTab === "datasets" && (
        <>
          {/* Error Message */}
          {error && (
            <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg" data-testid="download-error">
              {error}
            </div>
          )}

          {/* Disk Space */}
          {diskSpace && (
            <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg" data-testid="disk-space">
              <h3 className="text-sm font-medium text-gray-500 mb-2">Disk Space</h3>
              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500"
                      style={{
                        width: `${(diskSpace.used_gb / diskSpace.total_gb) * 100}%`,
                      }}
                    />
                  </div>
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  {diskSpace.available_gb.toFixed(1)} GB free
                </div>
              </div>
            </div>
          )}

          {/* Dataset List */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-gray-500">Datasets</h3>
            {statuses.map((status) => (
              <div
                key={status.dataset_id}
                className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
                data-testid={`download-item-${status.dataset_id}`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">
                      {status.dataset_id}
                    </p>
                    <p className="text-sm text-gray-500">
                      {status.size_mb > 0 ? formatSize(status.size_mb) : "Size unknown"}
                    </p>
                  </div>

                  <div className="flex items-center gap-3">
                    <span
                      className={`px-2 py-1 text-xs rounded-full ${getStatusBadge(status.status)}`}
                    >
                      {status.status === "downloading" ? (
                        <span className="flex items-center gap-1">
                          <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                          </svg>
                          Downloading
                        </span>
                      ) : (
                        status.status.replace("_", " ")
                      )}
                    </span>

                    {status.status === "not_downloaded" && (
                      <button
                        onClick={() => startDownload(status.dataset_id)}
                        className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                        data-testid={`download-btn-${status.dataset_id}`}
                      >
                        Download
                      </button>
                    )}
                  </div>
                </div>

                {/* Error display */}
                {status.error && (
                  <p className="mt-2 text-sm text-red-500">{status.error}</p>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {/* Storage Tab Content - All Hidden Caches */}
      {activeTab === "storage" && (
        <>
          {/* Error */}
          {allCachesError && (
            <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg">
              {allCachesError}
            </div>
          )}

          {/* Loading */}
          {allCachesLoading && (
            <div className="text-center py-8 text-gray-500">
              Scanning all caches...
            </div>
          )}

          {/* All Caches Summary */}
          {allCaches && (
            <>
              <div className="mb-6 p-4 bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg" data-testid="storage-summary">
                <div className="flex items-center gap-2 mb-2">
                  <svg className="w-5 h-5 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <h3 className="text-sm font-semibold text-orange-700 dark:text-orange-300">
                    Hidden Cache Storage
                  </h3>
                </div>
                <p className="text-2xl font-bold text-orange-800 dark:text-orange-200">
                  {allCaches.total_size_gb.toFixed(1)} GB
                </p>
                <p className="text-xs text-orange-600 dark:text-orange-400 mt-1">
                  These caches are in ~/.cache and don&apos;t appear in Finder
                </p>
              </div>

              {/* Cache List */}
              <div className="space-y-3" data-testid="all-caches-list">
                {allCaches.caches.map((cache) => {
                  const cacheKey = cache.key;
                  const isClearing = clearingCache === cacheKey;
                  const sizeDisplay = cache.size_mb >= 1000
                    ? `${(cache.size_mb / 1024).toFixed(1)} GB`
                    : `${cache.size_mb.toFixed(0)} MB`;
                  const isLarge = cache.size_mb > 1000;
                  const isWarning = cache.name.includes("WARNING");

                  return (
                    <div
                      key={cache.name}
                      className={`p-4 border rounded-lg ${
                        isWarning
                          ? "border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-900/10"
                          : isLarge
                          ? "border-orange-300 dark:border-orange-700 bg-orange-50 dark:bg-orange-900/10"
                          : "border-gray-200 dark:border-gray-700"
                      }`}
                      data-testid={`cache-${cacheKey}`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <p className="font-medium text-gray-900 dark:text-white">
                              {cache.name.replace(" (WARNING)", "")}
                            </p>
                            {isWarning && (
                              <span className="px-1.5 py-0.5 text-xs bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded">
                                Danger
                              </span>
                            )}
                            {isLarge && !isWarning && (
                              <span className="px-1.5 py-0.5 text-xs bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 rounded">
                                Large
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-gray-500 mt-0.5">{cache.description}</p>
                          <p className="text-xs text-gray-400 mt-1 truncate" title={cache.path}>
                            {cache.path}
                          </p>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className={`text-sm font-semibold ${
                            isLarge ? "text-orange-600 dark:text-orange-400" : "text-gray-600 dark:text-gray-400"
                          }`}>
                            {sizeDisplay}
                          </span>
                          {cache.size_mb > 0 && (
                            <button
                              onClick={() => clearCache(cacheKey)}
                              disabled={isClearing}
                              className={`px-3 py-1.5 text-xs rounded transition-colors disabled:opacity-50 ${
                                isWarning || isLarge
                                  ? "bg-red-500 text-white hover:bg-red-600"
                                  : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600"
                              }`}
                              data-testid={`clear-${cacheKey}`}
                            >
                              {isClearing ? "Clearing..." : "Clear"}
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Refresh Button */}
              <div className="mt-4 text-center">
                <button
                  onClick={fetchAllCaches}
                  disabled={allCachesLoading}
                  className="text-sm text-blue-500 hover:text-blue-700 disabled:opacity-50"
                >
                  Refresh
                </button>
              </div>
            </>
          )}
        </>
      )}

      {/* Cache Tab Content */}
      {activeTab === "cache" && (
        <>
          {/* Cache Error */}
          {cacheError && (
            <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg" data-testid="cache-error">
              {cacheError}
            </div>
          )}

          {/* Cache Stats Summary */}
          {cacheStats && (
            <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg" data-testid="cache-stats">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-medium text-gray-500 mb-1">Total Cache Size</h3>
                  <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                    {cacheStats.total_size_mb >= 1000
                      ? `${(cacheStats.total_size_mb / 1024).toFixed(2)} GB`
                      : `${cacheStats.total_size_mb.toFixed(1)} MB`}
                  </p>
                  <p className="text-sm text-gray-500">
                    {cacheStats.episode_count} episodes cached
                  </p>
                </div>
                {cacheStats.episode_count > 0 && (
                  <button
                    onClick={handleClearAll}
                    disabled={clearingAll}
                    className="px-4 py-2 text-sm bg-red-500 text-white rounded hover:bg-red-600 transition-colors disabled:opacity-50"
                    data-testid="clear-all-cache"
                  >
                    {clearingAll ? "Clearing..." : "Clear All"}
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Loading */}
          {cacheLoading && (
            <div className="text-center py-8 text-gray-500">
              Loading cached episodes...
            </div>
          )}

          {/* Cached Episodes List */}
          {!cacheLoading && cachedEpisodes.length === 0 && (
            <div className="text-center py-8 text-gray-500" data-testid="no-cached-episodes">
              <svg className="w-12 h-12 mx-auto mb-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
              </svg>
              <p>No cached episodes</p>
              <p className="text-sm mt-1">Episodes are cached when you view them</p>
            </div>
          )}

          {!cacheLoading && cachedEpisodes.length > 0 && (
            <div className="space-y-2 max-h-96 overflow-y-auto" data-testid="cached-episodes-list">
              {cachedEpisodes.map((episode) => {
                const episodeKey = `${episode.dataset_id}/${episode.episode_id}`;
                const isDeleting = deletingEpisode === episodeKey;
                // Truncate long episode IDs
                const displayEpisodeId = episode.episode_id.length > 50
                  ? "..." + episode.episode_id.slice(-47)
                  : episode.episode_id;

                return (
                  <div
                    key={episodeKey}
                    className="p-3 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
                    data-testid={`cached-episode-${episode.dataset_id}`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 flex-1">
                        <p className="font-medium text-gray-900 dark:text-white text-sm">
                          {episode.dataset_id}
                        </p>
                        <p className="text-xs text-gray-500 truncate" title={episode.episode_id}>
                          {displayEpisodeId}
                        </p>
                        <div className="flex items-center gap-3 mt-1 text-xs text-gray-400">
                          <span>{episode.size_mb.toFixed(1)} MB</span>
                          <span>{formatTime(episode.cached_at)}</span>
                        </div>
                      </div>
                      <button
                        onClick={() => handleDeleteEpisode(episode.dataset_id, episode.episode_id)}
                        disabled={isDeleting}
                        className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors disabled:opacity-50"
                        title="Delete from cache"
                        data-testid={`delete-cache-${episode.dataset_id}`}
                      >
                        {isDeleting ? (
                          <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                          </svg>
                        ) : (
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        )}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
