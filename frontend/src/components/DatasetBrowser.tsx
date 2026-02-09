"use client";

import { useState, useCallback } from "react";
import type { Dataset, EpisodeMetadata, Task, DatasetOverview, Modality } from "@/types/api";
import { useDatasets, useTasks, useTaskEpisodes, useDatasetOverview, removeDataset } from "@/hooks/useApi";
import AddDatasetDialog from "./AddDatasetDialog";

// Badge component for metadata display
function OverviewBadge({
  children,
  color = "gray",
}: {
  children: React.ReactNode;
  color?: "gray" | "blue" | "green" | "purple" | "red" | "yellow" | "orange";
}) {
  const colorClasses = {
    gray: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
    blue: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300",
    green: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300",
    purple: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300",
    red: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300",
    yellow: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300",
    orange: "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300",
  };
  return (
    <span className={`px-2 py-0.5 text-xs rounded-full ${colorClasses[color]}`}>
      {children}
    </span>
  );
}

// Modality chip component
function ModalityChip({ modality }: { modality: string }) {
  const colors: Record<string, "green" | "purple" | "blue" | "orange" | "gray"> = {
    rgb: "green",
    depth: "purple",
    imu: "blue",
    tactile: "orange",
    actions: "yellow" as "orange",
    states: "gray",
  };
  return (
    <OverviewBadge color={colors[modality] || "gray"}>
      {modality.toUpperCase()}
    </OverviewBadge>
  );
}

interface DatasetBrowserProps {
  onSelectEpisode?: (datasetId: string, episodeId: string, numFrames: number, modalities?: Modality[]) => void;
  onSelectDataset?: (datasetId: string | null) => void;
}

export default function DatasetBrowser({ onSelectEpisode, onSelectDataset }: DatasetBrowserProps) {
  const { datasets, loading: loadingDatasets, error: datasetsError, refetch: refetchDatasets } = useDatasets();
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [overviewExpanded, setOverviewExpanded] = useState(true);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [removingDataset, setRemovingDataset] = useState<string | null>(null);
  const [removeError, setRemoveError] = useState<string | null>(null);

  const { overview, loading: loadingOverview, error: overviewError, refresh: refreshOverview } = useDatasetOverview(selectedDataset);
  const { tasks, totalTasks, source: taskSource, loading: loadingTasks, error: tasksError } = useTasks(selectedDataset);
  const { episodes, loading: loadingEpisodes, error: episodesError, hasMore, loadMore } = useTaskEpisodes(
    selectedDataset,
    selectedTask,
    5 // Reduced limit for faster loading
  );

  const handleSelectDataset = (datasetId: string) => {
    setSelectedDataset(datasetId);
    setSelectedTask(null); // Reset task selection when dataset changes
    onSelectDataset?.(datasetId);
  };

  const handleBackToTasks = () => {
    setSelectedTask(null);
  };

  const handleDatasetAdded = useCallback(() => {
    refetchDatasets?.();
  }, [refetchDatasets]);

  const handleRemoveDataset = useCallback(async (e: React.MouseEvent, datasetId: string) => {
    e.stopPropagation(); // Prevent selecting the dataset
    setRemoveError(null);
    setRemovingDataset(datasetId);
    try {
      await removeDataset(datasetId);
      // Clear selection if removing the selected dataset
      if (selectedDataset === datasetId) {
        setSelectedDataset(null);
        setSelectedTask(null);
      }
      refetchDatasets?.();
    } catch (err) {
      setRemoveError(err instanceof Error ? err.message : "Failed to remove dataset");
    } finally {
      setRemovingDataset(null);
    }
  }, [selectedDataset, refetchDatasets]);

  if (loadingDatasets) {
    return (
      <div className="p-4 text-gray-500" data-testid="loading-datasets">
        Loading datasets...
      </div>
    );
  }

  if (datasetsError) {
    return (
      <div className="p-4 text-red-500" data-testid="error-datasets">
        Error: {datasetsError}
      </div>
    );
  }

  const selectedDatasetInfo = datasets.find(d => d.id === selectedDataset);

  return (
    <div className="flex flex-col h-full" data-testid="dataset-browser">
      {/* Add Dataset Dialog */}
      <AddDatasetDialog
        isOpen={showAddDialog}
        onClose={() => setShowAddDialog(false)}
        onDatasetAdded={handleDatasetAdded}
      />

      {/* Dataset List */}
      <div className={`border-b border-gray-200 dark:border-gray-700 ${selectedDataset ? 'max-h-[30%] overflow-auto' : ''}`}>
        <h2 className="px-4 py-2 text-sm font-semibold text-gray-500 uppercase tracking-wider sticky top-0 bg-white dark:bg-gray-900 z-10 flex items-center justify-between">
          <span>Datasets</span>
          <button
            onClick={() => setShowAddDialog(true)}
            className="p-1 text-blue-500 hover:text-blue-700 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded"
            title="Add HuggingFace Dataset"
            data-testid="add-dataset-button"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </button>
        </h2>
        {/* Remove error message */}
        {removeError && (
          <div className="mx-4 mb-2 px-3 py-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded text-sm text-red-700 dark:text-red-300 flex items-center justify-between">
            <span>{removeError}</span>
            <button
              onClick={() => setRemoveError(null)}
              className="ml-2 text-red-500 hover:text-red-700"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
        <ul className="divide-y divide-gray-100 dark:divide-gray-800">
          {datasets.map((dataset) => (
            <li key={dataset.id} className="group relative">
              <button
                onClick={() => handleSelectDataset(dataset.id)}
                className={`w-full px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors ${
                  selectedDataset === dataset.id
                    ? "bg-blue-50 dark:bg-blue-900/30 border-l-2 border-blue-500"
                    : ""
                }`}
                data-testid={`dataset-item-${dataset.id}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0 pr-2">
                    <p className="font-medium text-gray-900 dark:text-white truncate">
                      {dataset.name}
                    </p>
                    <p className="text-sm text-gray-500">
                      {dataset.type === "teleop" ? "Teleoperation" : "Video"}
                    </p>
                    {/* Modality badges */}
                    {dataset.modalities && dataset.modalities.length > 1 && (
                      <div className="flex gap-1 mt-1">
                        {dataset.modalities.map((mod) => (
                          <span
                            key={mod}
                            className={`px-1.5 py-0.5 text-xs rounded ${
                              mod === "rgb" ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300" :
                              mod === "depth" ? "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300" :
                              mod === "imu" ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300" :
                              "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"
                            }`}
                          >
                            {mod}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <span
                    className={`px-2 py-1 text-xs rounded-full flex-shrink-0 ${
                      dataset.type === "teleop"
                        ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                        : "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300"
                    }`}
                  >
                    {dataset.type}
                  </span>
                </div>
              </button>
              {/* Remove button - appears on hover */}
              <button
                onClick={(e) => handleRemoveDataset(e, dataset.id)}
                disabled={removingDataset === dataset.id}
                className="absolute right-2 top-2 p-1 opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded"
                title="Remove dataset"
                data-testid={`remove-dataset-${dataset.id}`}
              >
                {removingDataset === dataset.id ? (
                  <svg className="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                )}
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* Dataset Overview Section (shown when dataset selected) */}
      {selectedDataset && (
        <div className="border-b border-gray-200 dark:border-gray-700" data-testid="dataset-overview">
          {/* Collapsible header */}
          <button
            onClick={() => setOverviewExpanded(!overviewExpanded)}
            className="w-full px-4 py-2 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
            data-testid="overview-toggle"
          >
            <span className="text-sm font-semibold text-gray-700 dark:text-gray-200 truncate">
              {selectedDatasetInfo?.name || "Overview"}
            </span>
            <svg
              className={`w-4 h-4 text-gray-400 transition-transform ${overviewExpanded ? "rotate-180" : ""}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {/* Expandable content */}
          {overviewExpanded && (
            <div className="px-4 pb-3 space-y-3">
              {loadingOverview ? (
                <div className="animate-pulse space-y-2">
                  <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
                  <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
                </div>
              ) : overviewError ? (
                <div className="text-xs text-red-500">
                  {overviewError}
                  <button
                    onClick={refreshOverview}
                    className="ml-2 text-blue-500 hover:underline"
                  >
                    Retry
                  </button>
                </div>
              ) : overview ? (
                <>
                  {/* Format + Gated + License row */}
                  <div className="flex flex-wrap gap-1.5" data-testid="overview-badges">
                    {overview.format_detected && (
                      <OverviewBadge color="blue">{overview.format_detected}</OverviewBadge>
                    )}
                    {overview.gated && (
                      <OverviewBadge color="red">Gated</OverviewBadge>
                    )}
                    {overview.license && (
                      <OverviewBadge color="yellow">{overview.license}</OverviewBadge>
                    )}
                  </div>

                  {/* Environment + Perspective */}
                  {(overview.environment || overview.perspective) && (
                    <div className="flex flex-wrap gap-2 text-xs" data-testid="overview-environment">
                      {overview.environment && (
                        <span className="text-gray-600 dark:text-gray-400">
                          <span className="font-medium">Env:</span> {overview.environment}
                        </span>
                      )}
                      {overview.perspective && (
                        <span className="text-gray-600 dark:text-gray-400">
                          <span className="font-medium">View:</span> {overview.perspective}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Scale info */}
                  {(overview.estimated_hours || overview.estimated_clips || overview.task_count) && (
                    <div className="flex flex-wrap gap-3 text-xs text-gray-600 dark:text-gray-400" data-testid="overview-scale">
                      {overview.estimated_hours && (
                        <span>
                          <span className="font-medium">{overview.estimated_hours.toLocaleString()}h</span> duration
                        </span>
                      )}
                      {overview.estimated_clips && (
                        <span>
                          <span className="font-medium">{overview.estimated_clips.toLocaleString()}</span> clips
                        </span>
                      )}
                      {overview.task_count && (
                        <span>
                          <span className="font-medium">{overview.task_count}</span> tasks
                        </span>
                      )}
                    </div>
                  )}

                  {/* Modalities */}
                  {overview.modalities && overview.modalities.length > 0 && (
                    <div data-testid="overview-modalities">
                      <div className="text-xs text-gray-500 mb-1">Modalities</div>
                      <div className="flex flex-wrap gap-1">
                        {overview.modalities.map((mod) => (
                          <ModalityChip key={mod} modality={mod} />
                        ))}
                      </div>
                    </div>
                  )}

                  {/* README summary */}
                  {overview.readme_summary && (
                    <p
                      className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2"
                      title={overview.readme_summary}
                      data-testid="overview-summary"
                    >
                      {overview.readme_summary}
                    </p>
                  )}

                  {/* Tags */}
                  {overview.dataset_tags && overview.dataset_tags.length > 0 && (
                    <div className="flex flex-wrap gap-1" data-testid="overview-tags">
                      {overview.dataset_tags.slice(0, 5).map((tag) => (
                        <span
                          key={tag}
                          className="px-1.5 py-0.5 text-xs bg-gray-50 dark:bg-gray-800 text-gray-500 rounded"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Refresh button */}
                  <button
                    onClick={refreshOverview}
                    className="text-xs text-blue-500 hover:text-blue-700 flex items-center gap-1"
                    data-testid="refresh-overview"
                  >
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Refresh
                  </button>
                </>
              ) : null}
            </div>
          )}
        </div>
      )}

      {/* Task List (shown when dataset selected but no task selected) */}
      {selectedDataset && !selectedTask && (
        <div className="flex-1 overflow-auto">
          <h2 className="px-4 py-2 text-sm font-semibold text-gray-500 uppercase tracking-wider sticky top-0 bg-white dark:bg-gray-900 z-10 flex items-center justify-between">
            <span>Tasks {totalTasks > 0 && `(${totalTasks})`}</span>
            {taskSource === "huggingface_api" && (
              <span className="text-xs font-normal text-blue-500">via HF API</span>
            )}
          </h2>

          {loadingTasks ? (
            <div className="p-4 text-gray-500" data-testid="loading-tasks">
              Loading tasks...
            </div>
          ) : tasksError ? (
            <div className="p-4 text-red-500" data-testid="error-tasks">
              Error: {tasksError}
            </div>
          ) : tasks.length === 0 ? (
            <div className="p-4 text-gray-500" data-testid="no-tasks">
              No tasks found. The dataset may need to be downloaded first.
            </div>
          ) : (
            <ul className="divide-y divide-gray-100 dark:divide-gray-800" data-testid="task-list">
              {tasks.map((task) => (
                <li key={task.name}>
                  <button
                    onClick={() => setSelectedTask(task.name)}
                    className="w-full px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors group"
                    data-testid={`task-item-${task.name}`}
                  >
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium text-gray-900 dark:text-white truncate flex-1 mr-2">
                        {task.name}
                      </p>
                      <div className="flex items-center space-x-2">
                        {task.episode_count !== undefined && task.episode_count !== null && (
                          <span className="text-xs text-gray-500 bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded-full">
                            {task.episode_count} ep
                          </span>
                        )}
                        <svg
                          className="w-4 h-4 text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </div>
                    {task.description && (
                      <p className="text-xs text-gray-500 mt-1 truncate">{task.description}</p>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Episode List (shown when task is selected) */}
      {selectedDataset && selectedTask && (
        <div className="flex-1 overflow-auto">
          {/* Back button and task name */}
          <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-700 sticky top-0 bg-white dark:bg-gray-900 z-10">
            <button
              onClick={handleBackToTasks}
              className="flex items-center text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 mb-1"
              data-testid="back-to-tasks"
            >
              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              All Tasks
            </button>
            <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 truncate" title={selectedTask}>
              {selectedTask}
            </h2>
          </div>

          <h3 className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Sample Episodes {episodes.length > 0 && `(${episodes.length}${hasMore ? '+' : ''})`}
          </h3>

          {loadingEpisodes && episodes.length === 0 ? (
            <div className="p-4 text-gray-500" data-testid="loading-episodes">
              Loading episodes...
            </div>
          ) : episodesError ? (
            <div className="p-4 text-red-500" data-testid="error-episodes">
              Error: {episodesError}
            </div>
          ) : episodes.length === 0 ? (
            <div className="p-4 text-gray-500" data-testid="no-episodes">
              No episodes found for this task.
            </div>
          ) : (
            <>
              {selectedDatasetInfo?.type === "video" && (
                <div className="mx-4 mb-2 px-3 py-2 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded text-sm text-purple-700 dark:text-purple-300" data-testid="streaming-notice">
                  Streaming dataset - showing sample episodes
                </div>
              )}
              <ul className="divide-y divide-gray-100 dark:divide-gray-800" data-testid="episode-list">
                {episodes.map((episode, index) => (
                  <li key={episode.id}>
                    <button
                      onClick={() => onSelectEpisode?.(
                        selectedDataset,
                        episode.id,
                        episode.num_frames || 0,
                        overview?.modalities as Modality[] | undefined
                      )}
                      className="w-full px-4 py-2 text-left hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                      data-testid={`episode-item-${episode.id}`}
                    >
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        Episode {index + 1}
                      </p>
                      <p className="text-xs text-gray-500">
                        {episode.num_frames} frames
                        {episode.duration_sec && ` • ${episode.duration_sec.toFixed(1)}s`}
                      </p>
                    </button>
                  </li>
                ))}
              </ul>

              {/* Load More Button */}
              {hasMore && (
                <div className="p-4">
                  <button
                    onClick={loadMore}
                    disabled={loadingEpisodes}
                    className="w-full px-4 py-2 text-sm font-medium text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20 hover:bg-blue-100 dark:hover:bg-blue-900/30 rounded-lg transition-colors disabled:opacity-50"
                    data-testid="load-more-episodes"
                  >
                    {loadingEpisodes ? "Loading..." : "Load more episodes"}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
