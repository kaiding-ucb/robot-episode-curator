"use client";

import { useDownloads } from "@/hooks/useApi";

interface DataManagerProps {
  onClose?: () => void;
}

export default function DataManager({ onClose }: DataManagerProps) {
  const { statuses, diskSpace, loading, error, startDownload } = useDownloads();

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

  if (loading) {
    return (
      <div className="p-6 text-gray-500" data-testid="loading-downloads">
        Loading download status...
      </div>
    );
  }

  return (
    <div className="p-6" data-testid="data-manager">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
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
    </div>
  );
}
