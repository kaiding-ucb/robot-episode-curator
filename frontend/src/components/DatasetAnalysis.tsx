"use client";

import { useState, useEffect, useCallback } from "react";
import { useFrameCounts, useSignalComparison, useDatasetCapabilities } from "@/hooks/useDatasetAnalysis";
import { useDatasets, useTasks } from "@/hooks/useApi";
import FrameCountChart from "./FrameCountChart";
import SignalComparisonChart from "./SignalComparisonChart";

interface DatasetAnalysisProps {
  datasetId: string | null;
  onClose: () => void;
}

type AnalysisTab = "frame-counts" | "signal-comparison";

export default function DatasetAnalysis({
  datasetId: initialDatasetId,
  onClose,
}: DatasetAnalysisProps) {
  const [activeTab, setActiveTab] = useState<AnalysisTab>("frame-counts");
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [maxEpisodes, setMaxEpisodes] = useState(5);
  const [chosenDatasetId, setChosenDatasetId] = useState<string | null>(initialDatasetId);

  // Sync if parent passes a new datasetId
  useEffect(() => {
    if (initialDatasetId) {
      setChosenDatasetId(initialDatasetId);
    }
  }, [initialDatasetId]);

  const datasetId = chosenDatasetId;

  // Dataset list for the picker
  const { datasets } = useDatasets();

  // Data hooks
  const { tasks, loading: tasksLoading } = useTasks(datasetId);
  const {
    capabilities,
    loading: capabilitiesLoading,
    fetchCapabilities,
    reset: resetCapabilities,
  } = useDatasetCapabilities();
  const {
    data: frameCountData,
    loading: frameCountLoading,
    error: frameCountError,
    fetchFrameCounts,
    reset: resetFrameCounts,
  } = useFrameCounts();
  const {
    state: signalState,
    startAnalysis,
    cancelAnalysis,
    reset: resetSignals,
  } = useSignalComparison();

  // Clear all stale data and fetch capabilities when dataset changes
  useEffect(() => {
    if (datasetId) {
      resetFrameCounts();
      resetSignals();
      fetchCapabilities(datasetId);
    } else {
      resetCapabilities();
      resetFrameCounts();
      resetSignals();
    }
  }, [datasetId, fetchCapabilities, resetCapabilities, resetFrameCounts, resetSignals]);

  // If signal comparison not supported, default to frame-counts tab
  useEffect(() => {
    if (capabilities && !capabilities.supports_signal_comparison && activeTab === "signal-comparison") {
      setActiveTab("frame-counts");
    }
  }, [capabilities, activeTab]);

  // Auto-select first task when tasks load (only after fetch completes)
  useEffect(() => {
    if (!tasksLoading && tasks.length > 0 && !selectedTask) {
      setSelectedTask(tasks[0].name);
    }
  }, [tasks, selectedTask, tasksLoading]);

  // Reset signal state and fetch frame counts when task changes
  useEffect(() => {
    if (datasetId && selectedTask && !tasksLoading && tasks.some(t => t.name === selectedTask)) {
      resetSignals();
      fetchFrameCounts(datasetId, selectedTask);
    }
  }, [datasetId, selectedTask, tasks, tasksLoading, fetchFrameCounts, resetSignals]);

  const handleStartSignalAnalysis = useCallback(() => {
    if (datasetId && selectedTask) {
      startAnalysis(datasetId, selectedTask, maxEpisodes);
    }
  }, [datasetId, selectedTask, maxEpisodes, startAnalysis]);

  const signalsDisabled = capabilities !== null && !capabilities.supports_signal_comparison;

  return (
    <div className="p-6" data-testid="dataset-analysis-modal">
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Dataset Analysis
          </h2>
          {capabilities && (
            <span className="px-2 py-0.5 text-xs font-medium rounded bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 uppercase tracking-wide">
              {capabilities.format}
            </span>
          )}
        </div>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          data-testid="close-analysis-btn"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Dataset + Task selectors */}
      <div className="mb-4 flex flex-wrap gap-4">
        <div>
          <label className="text-sm text-gray-600 dark:text-gray-400 mr-2">Dataset:</label>
          <select
            value={datasetId || ""}
            onChange={(e) => {
              setChosenDatasetId(e.target.value || null);
              setSelectedTask(null);
            }}
            className="px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
            data-testid="dataset-selector"
          >
            <option value="">Select a dataset...</option>
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.id}>
                {ds.name}
              </option>
            ))}
          </select>
        </div>

        {datasetId && (
        <div>
        <label className="text-sm text-gray-600 dark:text-gray-400 mr-2">Task:</label>
        <select
          value={selectedTask || ""}
          onChange={(e) => setSelectedTask(e.target.value || null)}
          className="px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
          data-testid="task-selector"
        >
          {tasksLoading && (
            <option value="">Loading tasks...</option>
          )}
          {!tasksLoading && tasks.length === 0 && (
            <option value="">No tasks found</option>
          )}
          {tasks.map((task) => (
            <option key={task.name} value={task.name}>
              {task.name}
              {task.episode_count ? ` (${task.episode_count} episodes)` : ""}
            </option>
          ))}
        </select>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-4 border-b border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setActiveTab("frame-counts")}
          className={`px-4 py-2 text-sm font-medium rounded-t transition-colors ${
            activeTab === "frame-counts"
              ? "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white border-b-2 border-blue-500"
              : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          }`}
          data-testid="frame-counts-tab"
        >
          Frame Counts
        </button>
        <button
          onClick={() => !signalsDisabled && setActiveTab("signal-comparison")}
          className={`px-4 py-2 text-sm font-medium rounded-t transition-colors ${
            signalsDisabled
              ? "text-gray-300 dark:text-gray-600 cursor-not-allowed"
              : activeTab === "signal-comparison"
                ? "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white border-b-2 border-blue-500"
                : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          }`}
          data-testid="signal-comparison-tab"
          title={signalsDisabled ? capabilities?.signal_comparison_note : undefined}
        >
          Signal Comparison
          {signalsDisabled && (
            <span className="ml-1.5 text-xs text-gray-400 dark:text-gray-500">(N/A)</span>
          )}
        </button>
      </div>

      {/* Signal comparison info banner when disabled */}
      {signalsDisabled && capabilities?.signal_comparison_note && activeTab === "frame-counts" && (
        <div className="mb-4 px-3 py-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded text-sm text-blue-700 dark:text-blue-300">
          <span className="font-medium">Signal Comparison unavailable:</span>{" "}
          {capabilities.signal_comparison_note}
        </div>
      )}

      {/* Tab Content */}
      <div className="min-h-[300px]">
        {activeTab === "frame-counts" && (
          <div>
            {frameCountLoading && (
              <div className="flex items-center justify-center py-12 text-gray-500 text-sm">
                <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Loading frame counts from HuggingFace API...
              </div>
            )}
            {frameCountError && (
              <div className="text-sm text-red-500 py-4">
                Error: {frameCountError}
              </div>
            )}
            {frameCountData && <FrameCountChart data={frameCountData} />}
          </div>
        )}

        {activeTab === "signal-comparison" && (
          <div>
            {/* Controls */}
            <div className="flex items-center gap-4 mb-4">
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Compare first:
              </span>
              <div className="flex gap-1">
                {[5, 10].map((n) => (
                  <button
                    key={n}
                    onClick={() => setMaxEpisodes(n)}
                    className={`px-3 py-1 text-sm rounded ${
                      maxEpisodes === n
                        ? "bg-blue-500 text-white"
                        : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
              <span className="text-sm text-gray-600 dark:text-gray-400">episodes</span>

              {signalState.phase === "idle" || signalState.phase === "complete" || signalState.phase === "error" || signalState.phase === "no_signals" ? (
                <button
                  onClick={handleStartSignalAnalysis}
                  className="px-4 py-1.5 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 transition-colors"
                  data-testid="start-analysis-btn"
                >
                  {signalState.phase === "complete" ? "Re-analyze" : "Start Analysis"}
                </button>
              ) : (
                <button
                  onClick={cancelAnalysis}
                  className="px-4 py-1.5 bg-gray-500 text-white text-sm rounded hover:bg-gray-600 transition-colors"
                >
                  Cancel
                </button>
              )}
            </div>

            {/* Progress bar */}
            {signalState.phase === "processing" && (
              <div className="mb-4">
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>
                    Processing episode {signalState.progress.current + 1}/{signalState.progress.total}
                  </span>
                  <span className="font-mono truncate ml-2 max-w-[200px]">
                    {getEpisodeLabel(signalState.progress.currentEpisode)}
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all"
                    style={{
                      width: `${signalState.progress.total > 0 ? ((signalState.progress.current) / signalState.progress.total) * 100 : 0}%`,
                    }}
                  />
                </div>
              </div>
            )}

            {/* No signals message */}
            {signalState.phase === "no_signals" && signalState.noSignalsReason && (
              <div className="px-4 py-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded text-center">
                <div className="text-sm text-blue-700 dark:text-blue-300 font-medium mb-1">
                  Signal Comparison Not Available
                </div>
                <div className="text-sm text-blue-600 dark:text-blue-400">
                  {signalState.noSignalsReason}
                </div>
              </div>
            )}

            {/* Error */}
            {signalState.error && (
              <div className="text-sm text-red-500 mb-4">
                Error: {signalState.error}
              </div>
            )}

            {/* Charts */}
            {signalState.episodes.size > 0 && (
              <SignalComparisonChart episodes={signalState.episodes} datasetId={datasetId} />
            )}

            {/* Idle state */}
            {signalState.phase === "idle" && signalState.episodes.size === 0 && (
              <div className="text-sm text-gray-500 text-center py-12">
                Click &quot;Start Analysis&quot; to download and compare episode signals.
                <br />
                <span className="text-xs text-gray-400 mt-1 block">
                  This will download MCAP files but skip video decoding for faster analysis.
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function getEpisodeLabel(episodeId: string): string {
  if (!episodeId) return "";
  const parts = episodeId.split("/");
  return parts[parts.length - 1]?.replace(/\.[^.]+$/, "") || episodeId;
}
