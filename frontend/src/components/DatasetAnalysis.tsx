"use client";

import { useState, useEffect } from "react";
import { useFrameCounts, useSignalComparison, useDatasetCapabilities, useMetaSummary } from "@/hooks/useDatasetAnalysis";
import { useDatasets, useTasks } from "@/hooks/useApi";
import FrameCountChart from "./FrameCountChart";
import { PhaseAwarePanel } from "./PhaseAwarePanel";
import SummaryPanel from "./SummaryPanel";
import EdgeFramesPanel from "./EdgeFramesPanel";

interface DatasetAnalysisProps {
  datasetId: string | null;
  onClose: () => void;
  onNavigateToEpisode?: (datasetId: string, episodeId: string, numFrames: number) => void;
  // Kept on the props interface for source compatibility with the modal host.
  // The legacy "Compare Episodes" button that consumed this was removed when
  // the envelope view retired.
  onViewComparisonInRerun?: (rrdUrl: string) => void;
}

type AnalysisTab = "summary" | "frame-counts" | "edge-frames" | "signal-comparison";

export default function DatasetAnalysis({
  datasetId: initialDatasetId,
  onClose,
  onNavigateToEpisode,
}: DatasetAnalysisProps) {
  const [activeTab, setActiveTab] = useState<AnalysisTab>("summary");
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
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
    data: metaSummary,
    loading: metaSummaryLoading,
    error: metaSummaryError,
    fetchSummary: fetchMetaSummary,
    reset: resetMetaSummary,
  } = useMetaSummary();
  // Legacy SSE-based signal-comparison hook is still wired so any in-flight
  // analysis is aborted on dataset/task change. Only the cancel/reset
  // handlers are consumed; the streamed signal data is no longer rendered.
  const { cancelAnalysis, reset: resetSignals } = useSignalComparison();

  // Clear all stale data and fetch capabilities + meta summary when dataset changes
  useEffect(() => {
    if (datasetId) {
      resetFrameCounts();
      resetSignals();
      resetMetaSummary();
      fetchCapabilities(datasetId);
      fetchMetaSummary(datasetId);
    } else {
      resetCapabilities();
      resetFrameCounts();
      resetSignals();
      resetMetaSummary();
    }
  }, [datasetId, fetchCapabilities, resetCapabilities, resetFrameCounts, resetSignals, fetchMetaSummary, resetMetaSummary]);

  // If signal comparison not supported, default to frame-counts tab
  useEffect(() => {
    if (capabilities && !capabilities.supports_signal_comparison && activeTab === "signal-comparison") {
      setActiveTab("frame-counts");
    }
  }, [capabilities, activeTab]);

  // If summary unavailable for this dataset, fall back to frame-counts
  useEffect(() => {
    if (metaSummary && metaSummary.source === "unavailable" && activeTab === "summary") {
      setActiveTab("frame-counts");
    }
  }, [metaSummary, activeTab]);

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

  const signalsDisabled = capabilities !== null && !capabilities.supports_signal_comparison;
  const edgeFramesDisabled =
    capabilities !== null && capabilities.supports_edge_frames === false;

  // If edge frames not supported and currently active, fall back.
  useEffect(() => {
    if (edgeFramesDisabled && activeTab === "edge-frames") {
      setActiveTab("frame-counts");
    }
  }, [edgeFramesDisabled, activeTab]);

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
              cancelAnalysis();
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
        {(() => {
          const summaryDisabled = metaSummary !== null && metaSummary.source === "unavailable";
          return (
            <button
              onClick={() => !summaryDisabled && setActiveTab("summary")}
              className={`px-4 py-2 text-sm font-medium rounded-t transition-colors ${
                summaryDisabled
                  ? "text-gray-300 dark:text-gray-600 cursor-not-allowed"
                  : activeTab === "summary"
                    ? "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white border-b-2 border-blue-500"
                    : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              }`}
              data-testid="summary-tab"
            >
              Summary
              {summaryDisabled && (
                <span className="ml-1.5 text-xs text-gray-400 dark:text-gray-500">(N/A)</span>
              )}
            </button>
          );
        })()}
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
          onClick={() => !edgeFramesDisabled && setActiveTab("edge-frames")}
          className={`px-4 py-2 text-sm font-medium rounded-t transition-colors ${
            edgeFramesDisabled
              ? "text-gray-300 dark:text-gray-600 cursor-not-allowed"
              : activeTab === "edge-frames"
                ? "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white border-b-2 border-blue-500"
                : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          }`}
          data-testid="edge-frames-tab"
          title={edgeFramesDisabled ? capabilities?.edge_frames_note : undefined}
        >
          Starting &amp; Ending Frames
          {edgeFramesDisabled && (
            <span className="ml-1.5 text-xs text-gray-400 dark:text-gray-500">(N/A)</span>
          )}
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
          Action Insights
          {signalsDisabled && (
            <span className="ml-1.5 text-xs text-gray-400 dark:text-gray-500">(N/A)</span>
          )}
        </button>
      </div>

      {/* Signal comparison info banner when disabled */}
      {signalsDisabled && capabilities?.signal_comparison_note && activeTab === "frame-counts" && (
        <div className="mb-4 px-3 py-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded text-sm text-blue-700 dark:text-blue-300">
          <span className="font-medium">Action Insights unavailable:</span>{" "}
          {capabilities.signal_comparison_note}
        </div>
      )}

      {/* Tab Content */}
      <div className="min-h-[300px]">
        {activeTab === "summary" && (
          <SummaryPanel
            summary={metaSummary}
            loading={metaSummaryLoading}
            error={metaSummaryError}
            onTaskSelected={(taskName) => {
              setSelectedTask(taskName);
              setActiveTab("frame-counts");
            }}
          />
        )}

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

        {activeTab === "edge-frames" && (
          <EdgeFramesPanel
            datasetId={datasetId}
            taskName={selectedTask}
            onNavigateToEpisode={onNavigateToEpisode}
          />
        )}

        {activeTab === "signal-comparison" && (
          <div>
            {selectedTask && datasetId ? (
              <PhaseAwarePanel
                datasetId={datasetId}
                taskName={selectedTask}
                onNavigateToEpisode={onNavigateToEpisode}
              />
            ) : (
              <div className="py-6 text-sm text-gray-500 text-center">
                Select a task to run phase-aware analysis.
              </div>
            )}

          </div>
        )}
      </div>
    </div>
  );
}
