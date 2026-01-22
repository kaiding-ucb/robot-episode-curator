"use client";

import { useState } from "react";
import type { Dataset, EpisodeMetadata, Task } from "@/types/api";
import { useDatasets, useTasks, useTaskEpisodes } from "@/hooks/useApi";

interface DatasetBrowserProps {
  onSelectEpisode?: (datasetId: string, episodeId: string, numFrames: number) => void;
}

export default function DatasetBrowser({ onSelectEpisode }: DatasetBrowserProps) {
  const { datasets, loading: loadingDatasets, error: datasetsError } = useDatasets();
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);

  const { tasks, totalTasks, source: taskSource, loading: loadingTasks, error: tasksError } = useTasks(selectedDataset);
  const { episodes, loading: loadingEpisodes, error: episodesError, hasMore, loadMore } = useTaskEpisodes(
    selectedDataset,
    selectedTask,
    5 // Reduced limit for faster loading
  );

  const handleSelectDataset = (datasetId: string) => {
    setSelectedDataset(datasetId);
    setSelectedTask(null); // Reset task selection when dataset changes
  };

  const handleBackToTasks = () => {
    setSelectedTask(null);
  };

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
      {/* Dataset List */}
      <div className={`border-b border-gray-200 dark:border-gray-700 ${selectedDataset ? 'max-h-[30%] overflow-auto' : ''}`}>
        <h2 className="px-4 py-2 text-sm font-semibold text-gray-500 uppercase tracking-wider sticky top-0 bg-white dark:bg-gray-900 z-10">
          Datasets
        </h2>
        <ul className="divide-y divide-gray-100 dark:divide-gray-800">
          {datasets.map((dataset) => (
            <li key={dataset.id}>
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
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">
                      {dataset.name}
                    </p>
                    <p className="text-sm text-gray-500">
                      {dataset.type === "teleop" ? "Teleoperation" : "Video"}
                    </p>
                  </div>
                  <span
                    className={`px-2 py-1 text-xs rounded-full ${
                      dataset.type === "teleop"
                        ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                        : "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300"
                    }`}
                  >
                    {dataset.type}
                  </span>
                </div>
              </button>
            </li>
          ))}
        </ul>
      </div>

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
                      onClick={() => onSelectEpisode?.(selectedDataset, episode.id, episode.num_frames || 0)}
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
