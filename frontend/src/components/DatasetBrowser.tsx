"use client";

import { useState, useCallback } from "react";
import type { Dataset, EpisodeMetadata, Task, Modality } from "@/types/api";
import { useDatasets, useTasks, useTaskEpisodes, removeDataset } from "@/hooks/useApi";
import AddDatasetDialog from "./AddDatasetDialog";

interface DatasetBrowserProps {
  onSelectEpisode?: (datasetId: string, episodeId: string, numFrames: number, modalities?: Modality[], displayName?: string) => void;
  onSelectDataset?: (datasetId: string | null) => void;
}

export default function DatasetBrowser({ onSelectEpisode, onSelectDataset }: DatasetBrowserProps) {
  const { datasets, loading: loadingDatasets, error: datasetsError, refetch: refetchDatasets } = useDatasets();
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [removingDataset, setRemovingDataset] = useState<string | null>(null);
  const [removeError, setRemoveError] = useState<string | null>(null);

  const { tasks, totalTasks, source: taskSource, loading: loadingTasks, error: tasksError, hasMore: hasMoreTasks, loadMore: loadMoreTasks, searchQuery: taskSearchQuery, updateSearch: updateTaskSearch } = useTasks(selectedDataset);
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
        existingRepoIds={new Set(datasets.map((d) => d.id).filter(Boolean))}
      />

      {/* Dataset List - hidden when a dataset is selected to give tasks/episodes full space */}
      <div className={`border-b border-gray-200 dark:border-gray-700 ${selectedDataset ? 'hidden' : ''}`}>
        <h2 className="px-4 py-2 text-sm font-semibold text-gray-500 uppercase tracking-wider sticky top-0 bg-white dark:bg-gray-900 z-10 flex items-center justify-between">
          <span>Datasets</span>
          <button
            onClick={() => setShowAddDialog(true)}
            className="p-1 text-blue-500 hover:text-blue-700 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded"
            title="Add LeRobot Dataset"
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

      {/* Task List (shown when dataset selected but no task selected) */}
      {selectedDataset && !selectedTask && (
        <div className="flex-1 overflow-auto">
          <div className="px-4 py-2 sticky top-0 bg-white dark:bg-gray-900 z-10 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 min-w-0">
                <button
                  onClick={() => { setSelectedDataset(null); setSelectedTask(null); onSelectDataset?.(null); }}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 flex-shrink-0"
                  title="Back to datasets"
                  data-testid="back-to-datasets"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
                <span className="text-sm font-semibold text-gray-900 dark:text-white truncate">
                  {selectedDatasetInfo?.name || selectedDataset}
                </span>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-1 ml-6">
              {taskSource === "multi_subdataset" ? "Subdatasets" : "Tasks"}{totalTasks > 0 && ` (${hasMoreTasks ? `${tasks.length} of ` : ""}${totalTasks.toLocaleString()})`}
            </p>
          </div>

          {totalTasks > 50 && (
            <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-700">
              <input
                type="text"
                value={taskSearchQuery}
                onChange={(e) => updateTaskSearch(e.target.value)}
                placeholder="Search tasks..."
                data-testid="task-search-input"
                className="w-full px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          )}

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
          {hasMoreTasks && !loadingTasks && (
            <button
              onClick={loadMoreTasks}
              data-testid="load-more-tasks"
              className="w-full px-4 py-2 text-sm text-blue-600 dark:text-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800 border-t border-gray-200 dark:border-gray-700 transition-colors"
            >
              Load more tasks ({totalTasks - tasks.length} remaining)
            </button>
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
                {episodes.map((episode) => (
                  <li key={episode.id}>
                    <button
                      onClick={() => onSelectEpisode?.(
                        selectedDataset,
                        episode.id,
                        episode.num_frames || 0,
                        selectedDatasetInfo?.modalities as Modality[] | undefined,
                        episode.id
                      )}
                      className="w-full px-4 py-2 text-left hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                      data-testid={`episode-item-${episode.id}`}
                    >
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        {episode.id}
                      </p>
                      <p className="text-xs text-gray-500">
                        {episode.num_frames != null ? `${episode.num_frames} frames` : 'frames'}
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
