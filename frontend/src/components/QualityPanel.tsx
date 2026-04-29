"use client";

import { useEpisodeQuality, useQualityEvents, useTaskQuality, useEpisodeDivergence } from "@/hooks/useQuality";
import type { QualityEvent, TaskQualityMetrics, EpisodeDivergence } from "@/types/quality";
import { READABLE_DIMENSION_NAMES } from "@/types/quality";

interface QualityPanelProps {
  datasetId: string | null;
  episodeId: string | null;
  taskName?: string | null;
  onJumpToFrame?: (frame: number) => void;
  selectedMetric?: string | null;
  onSelectMetric?: (metric: string | null) => void;
  /** Callback to provide divergence scores to parent for timeline heat */
  onDivergenceScores?: (scores: number[]) => void;
}

interface MetricBarProps {
  label: string;
  value: number;
  colorClass?: string;
  isHighlighted?: boolean;
}

function MetricBar({ label, value, colorClass = "bg-orange-500", isHighlighted = false }: MetricBarProps) {
  const percentage = Math.round(value * 100);

  return (
    <div className={`mb-1 ${isHighlighted ? 'bg-orange-100 dark:bg-orange-900/30 -mx-1 px-1 rounded' : ''}`}>
      <div className="flex justify-between text-xs mb-0.5">
        <span className={`${isHighlighted ? 'text-orange-700 dark:text-orange-300 font-medium' : 'text-gray-600 dark:text-gray-400'}`}>
          {label}
          {isHighlighted && <span className="ml-1 text-orange-500">← highest</span>}
        </span>
        <span className="font-medium">{percentage}%</span>
      </div>
      <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${colorClass} transition-all duration-300`}
          style={{ width: `${Math.min(100, percentage)}%` }}
        />
      </div>
    </div>
  );
}

// ============== Task-Level Quality Components ==============

function TaskQualitySection({ metrics }: { metrics: TaskQualityMetrics }) {
  return (
    <div className="space-y-4" data-testid="task-quality-section">
      {/* Expertise Test (Divergence) */}
      <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-3">
        <h4 className="text-sm font-semibold text-orange-800 dark:text-orange-300 mb-2 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-orange-500" />
          Expertise Test
        </h4>
        <MetricBar
          label="Consistency"
          value={metrics.expertise_score}
          colorClass="bg-orange-500"
        />
        <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
          {metrics.expertise_score > 0.7
            ? "Experts demonstrate consistent technique across episodes"
            : metrics.expertise_score > 0.4
            ? "Moderate variance in demonstration paths"
            : "High variance suggests crowdsourced without style guide"}
        </p>
        <div className="text-xs text-gray-500 dark:text-gray-500 mt-2 flex gap-3">
          <span>Low: {metrics.divergence_distribution.low}</span>
          <span>Med: {metrics.divergence_distribution.medium}</span>
          <span className="text-orange-600 dark:text-orange-400">High: {metrics.divergence_distribution.high}</span>
        </div>
      </div>

      {/* Physics Test (Recovery) */}
      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
        <h4 className="text-sm font-semibold text-purple-800 dark:text-purple-300 mb-2 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-purple-500" />
          Physics Test
        </h4>
        <MetricBar
          label="Recovery Coverage"
          value={metrics.physics_coverage_score}
          colorClass="bg-purple-500"
        />
        <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
          {metrics.has_any_recovery_episodes
            ? "Dataset includes recovery behaviors - good for robust learning"
            : "All perfect executions - may miss edge cases"}
        </p>
        <div className="text-xs text-gray-500 dark:text-gray-500 mt-2">
          {Math.round(metrics.pct_with_recovery * 100)}% of episodes have recovery behaviors
          {metrics.mean_recovery_count > 0 && (
            <span className="ml-2">({metrics.mean_recovery_count.toFixed(1)} avg/episode)</span>
          )}
        </div>
      </div>

      {/* Quality Assessment */}
      <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3">
        <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
          {metrics.quality_assessment}
        </div>
        <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
          Based on {metrics.num_episodes} episodes
        </div>
      </div>
    </div>
  );
}

// ============== Episode Comparison Components ==============

interface EpisodeComparisonSectionProps {
  taskMetrics: TaskQualityMetrics;
  divergence: EpisodeDivergence;
  recoveryEvents: QualityEvent[];
  onJumpToFrame?: (frame: number) => void;
}

function EpisodeComparisonSection({ taskMetrics, divergence, recoveryEvents, onJumpToFrame }: EpisodeComparisonSectionProps) {
  // Find highest divergence dimension
  const maxDivIdx = divergence.dimension_means.indexOf(Math.max(...divergence.dimension_means));

  // Compare episode recovery to task average
  const episodeRecoveryCount = recoveryEvents.length;
  const taskAvgRecovery = taskMetrics.mean_recovery_count;
  const recoveryDiff = episodeRecoveryCount - taskAvgRecovery;
  const isAboveAverage = recoveryDiff >= 0;

  // Recovery bar width (normalized to 2x task average as max)
  const maxRecovery = Math.max(taskAvgRecovery * 2, 1);
  const recoveryBarWidth = Math.min(100, (episodeRecoveryCount / maxRecovery) * 100);

  return (
    <div className="space-y-4" data-testid="episode-comparison-section">
      {/* Divergence Dimension Breakdown */}
      <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-3">
        <h4 className="text-sm font-semibold text-orange-800 dark:text-orange-300 mb-2">
          Episode Divergence: {Math.round(divergence.overall_divergence_score * 100)}%
        </h4>
        <div className="space-y-1">
          {divergence.dimension_names.map((dimName, idx) => {
            const readableName = READABLE_DIMENSION_NAMES[dimName] || dimName;
            const value = divergence.dimension_means[idx] || 0;
            // Normalize to 0-1 range (assuming z-scores, cap at 3)
            const normalizedValue = Math.min(1, value / 3);
            const isHighest = idx === maxDivIdx;

            return (
              <MetricBar
                key={dimName}
                label={readableName}
                value={normalizedValue}
                colorClass="bg-orange-500"
                isHighlighted={isHighest}
              />
            );
          })}
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
          Hover timeline for frame-by-frame details
        </p>
      </div>

      {/* Recovery Comparison */}
      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
        <h4 className="text-sm font-semibold text-purple-800 dark:text-purple-300 mb-2">
          Recovery Behaviors
        </h4>

        <div className="text-sm mb-2">
          <div className="flex justify-between">
            <span className="text-gray-700 dark:text-gray-300">This episode:</span>
            <span className="font-semibold text-purple-700 dark:text-purple-300">{episodeRecoveryCount} recoveries</span>
          </div>
          <div className="flex justify-between text-xs text-gray-500">
            <span>Task average:</span>
            <span>{taskAvgRecovery.toFixed(1)} per episode</span>
          </div>
        </div>

        {/* Comparison bar */}
        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mb-1">
          <div
            className={`h-full transition-all duration-300 ${isAboveAverage ? 'bg-green-500' : 'bg-orange-500'}`}
            style={{ width: `${recoveryBarWidth}%` }}
          />
        </div>
        <div className={`text-xs ${isAboveAverage ? 'text-green-600 dark:text-green-400' : 'text-orange-600 dark:text-orange-400'}`}>
          {isAboveAverage ? 'Above average' : 'Below average'} ({recoveryDiff >= 0 ? '+' : ''}{recoveryDiff.toFixed(1)})
        </div>

        {/* Recovery event badges */}
        {recoveryEvents.length > 0 && onJumpToFrame && (
          <div className="mt-3">
            <div className="text-xs text-gray-500 mb-1">Jump to recovery:</div>
            <div className="flex flex-wrap gap-1">
              {recoveryEvents.slice(0, 8).map((event, i) => (
                <button
                  key={`${event.frame}-${i}`}
                  onClick={() => onJumpToFrame(event.frame)}
                  className="text-xs px-1.5 py-0.5 rounded transition-colors bg-purple-100 hover:bg-purple-200 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300 dark:hover:bg-purple-900/50"
                  title={event.description}
                  data-testid={`jump-to-frame-${event.frame}`}
                >
                  @{event.frame}
                </button>
              ))}
              {recoveryEvents.length > 8 && (
                <span className="text-xs text-gray-400 self-center">+{recoveryEvents.length - 8} more</span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function QualityPanel({
  datasetId,
  episodeId,
  taskName,
  onJumpToFrame,
  selectedMetric,
  onSelectMetric,
  onDivergenceScores,
}: QualityPanelProps) {
  // We still need episode quality for backward compatibility, but won't display most of it
  const { loading, error } = useEpisodeQuality(datasetId, episodeId);
  const { events: qualityEventsData } = useQualityEvents(datasetId, episodeId);
  const events = qualityEventsData?.events || [];

  // Task-level quality metrics
  const taskNameOrNull = taskName ?? null;
  const { metrics: taskMetrics, loading: taskLoading } = useTaskQuality(datasetId, taskNameOrNull);

  // Episode divergence for timeline heat map
  const { divergence, loading: divLoading } = useEpisodeDivergence(datasetId, taskNameOrNull, episodeId);

  // Pass divergence scores to parent for timeline heat
  const divergenceScores = divergence?.frame_divergences;
  if (onDivergenceScores && divergenceScores && divergenceScores.length > 0) {
    setTimeout(() => onDivergenceScores(divergenceScores), 0);
  }

  // Filter to recovery events only
  const recoveryEvents = events.filter(e => e.event_type === 'recovery');

  if (!episodeId) {
    return (
      <div className="p-4 text-center text-gray-500" data-testid="no-quality">
        Select an episode to view quality metrics
      </div>
    );
  }

  if (loading || taskLoading || divLoading) {
    return (
      <div className="p-4 text-center text-gray-500" data-testid="loading-quality">
        Analyzing quality...
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-center text-red-500" data-testid="error-quality">
        {error}
      </div>
    );
  }

  return (
    <div className="p-4" data-testid="quality-panel">
      {/* Task-Level Quality (if available) */}
      {taskMetrics && (
        <div className="mb-6">
          <h3 className="text-sm font-bold text-gray-800 dark:text-gray-200 mb-3 flex items-center gap-2">
            Task Quality
            <span className="text-xs font-normal text-gray-500">
              ({taskMetrics.num_episodes} episodes)
            </span>
          </h3>
          <TaskQualitySection metrics={taskMetrics} />
        </div>
      )}

      {/* Episode Comparison Section */}
      {taskMetrics && divergence && divergence.dimension_names.length > 0 && (
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <h3 className="text-sm font-bold text-gray-800 dark:text-gray-200 mb-3">
            This Episode vs Task
          </h3>
          <EpisodeComparisonSection
            taskMetrics={taskMetrics}
            divergence={divergence}
            recoveryEvents={recoveryEvents}
            onJumpToFrame={onJumpToFrame}
          />
        </div>
      )}

      {/* Fallback if no task metrics but have divergence events */}
      {!taskMetrics && divergence && (
        <div className="p-2 bg-orange-50 dark:bg-orange-900/10 rounded text-xs">
          <span className="text-orange-700 dark:text-orange-300">
            Episode divergence: {Math.round(divergence.overall_divergence_score * 100)}%
          </span>
        </div>
      )}

      {/* Show loading placeholder if waiting for task data */}
      {!taskMetrics && !divergence && (
        <div className="p-4 text-center text-gray-400 text-sm">
          Task-level analysis requires 2+ episodes
        </div>
      )}
    </div>
  );
}
