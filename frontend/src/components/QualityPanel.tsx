"use client";

import { useEpisodeQuality, useQualityEvents } from "@/hooks/useQuality";
import type { QualityScore, QualityEvent } from "@/types/quality";

interface QualityPanelProps {
  datasetId: string | null;
  episodeId: string | null;
  onJumpToFrame?: (frame: number) => void;
  selectedMetric?: string | null;
  onSelectMetric?: (metric: string | null) => void;
}

interface MetricBarProps {
  label: string;
  value: number;
  colorClass?: string;
  metricKey?: string;
  isSelected?: boolean;
  onClick?: (metricKey: string) => void;
}

function MetricBar({ label, value, colorClass = "bg-blue-500", metricKey, isSelected, onClick }: MetricBarProps) {
  const percentage = Math.round(value * 100);
  const isClickable = !!onClick && !!metricKey;

  return (
    <div
      className={`mb-2 ${isClickable ? 'cursor-pointer' : ''} ${isSelected ? 'ring-2 ring-blue-400 ring-offset-1 rounded-lg p-1 -m-1' : ''}`}
      onClick={() => isClickable && onClick(metricKey)}
      data-testid={metricKey ? `metric-bar-${metricKey}` : undefined}
    >
      <div className="flex justify-between text-xs mb-1">
        <span className={`${isSelected ? 'text-blue-600 dark:text-blue-400 font-medium' : 'text-gray-600 dark:text-gray-400'}`}>{label}</span>
        <span className="font-medium">{percentage}%</span>
      </div>
      <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${colorClass} transition-all duration-300`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function QualityBadge({ quality }: { quality: QualityScore }) {
  const gradeColors: Record<string, string> = {
    A: "bg-green-500",
    B: "bg-blue-500",
    C: "bg-yellow-500",
    D: "bg-orange-500",
    F: "bg-red-500",
  };

  return (
    <div className="flex items-center gap-3 mb-4">
      <div
        className={`w-12 h-12 rounded-lg ${gradeColors[quality.quality_grade]} flex items-center justify-center text-white text-xl font-bold`}
      >
        {quality.quality_grade}
      </div>
      <div>
        <div className="text-lg font-semibold">
          {Math.round(quality.overall_score * 100)}%
        </div>
        <div className="text-sm text-gray-500">Overall Quality</div>
      </div>
    </div>
  );
}

function QualityFlags({ quality }: { quality: QualityScore }) {
  return (
    <div className="flex flex-wrap gap-2 mb-4">
      {quality.has_recovery_behaviors && (
        <span className="px-2 py-1 text-xs rounded-full bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300">
          Has Recovery
        </span>
      )}
      {quality.is_diverse && (
        <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300">
          Diverse
        </span>
      )}
      {quality.is_smooth && (
        <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">
          Smooth
        </span>
      )}
      {quality.is_well_synced && (
        <span className="px-2 py-1 text-xs rounded-full bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300">
          Well Synced
        </span>
      )}
      {!quality.has_recovery_behaviors && !quality.is_diverse && (
        <span className="px-2 py-1 text-xs rounded-full bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300">
          Too Perfect
        </span>
      )}
    </div>
  );
}

interface EventBadgesProps {
  events: QualityEvent[];
  onJumpToFrame?: (frame: number) => void;
  filterType?: string;
  maxShow?: number;
}

const EVENT_BADGE_COLORS: Record<string, string> = {
  gripper: "bg-green-100 hover:bg-green-200 text-green-800 dark:bg-green-900/30 dark:text-green-300 dark:hover:bg-green-900/50",
  pause: "bg-yellow-100 hover:bg-yellow-200 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300 dark:hover:bg-yellow-900/50",
  direction_change: "bg-cyan-100 hover:bg-cyan-200 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300 dark:hover:bg-cyan-900/50",
  speed_change: "bg-orange-100 hover:bg-orange-200 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300 dark:hover:bg-orange-900/50",
  high_jerk: "bg-red-100 hover:bg-red-200 text-red-800 dark:bg-red-900/30 dark:text-red-300 dark:hover:bg-red-900/50",
  correction: "bg-purple-100 hover:bg-purple-200 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300 dark:hover:bg-purple-900/50",
};

function EventBadges({ events, onJumpToFrame, filterType, maxShow = 8 }: EventBadgesProps) {
  const filteredEvents = filterType
    ? events.filter(e => e.event_type === filterType)
    : events;

  if (filteredEvents.length === 0 || !onJumpToFrame) return null;

  const displayEvents = filteredEvents.slice(0, maxShow);
  const remaining = filteredEvents.length - maxShow;

  return (
    <div className="mt-1 flex flex-wrap gap-1">
      {displayEvents.map((event, i) => (
        <button
          key={`${event.frame}-${i}`}
          onClick={() => onJumpToFrame(event.frame)}
          className={`text-xs px-1.5 py-0.5 rounded transition-colors ${EVENT_BADGE_COLORS[event.event_type] || EVENT_BADGE_COLORS.correction}`}
          title={event.description}
          data-testid={`jump-to-frame-${event.frame}`}
        >
          @{event.frame}
        </button>
      ))}
      {remaining > 0 && (
        <span className="text-xs text-gray-400 self-center">+{remaining} more</span>
      )}
    </div>
  );
}

export default function QualityPanel({ datasetId, episodeId, onJumpToFrame, selectedMetric, onSelectMetric }: QualityPanelProps) {
  const { quality, loading, error } = useEpisodeQuality(datasetId, episodeId);
  const { events: qualityEventsData } = useQualityEvents(datasetId, episodeId);
  const events = qualityEventsData?.events || [];

  const handleMetricClick = (metricKey: string) => {
    if (onSelectMetric) {
      // Toggle: if already selected, deselect; otherwise select
      onSelectMetric(selectedMetric === metricKey ? null : metricKey);
    }
  };

  if (!episodeId) {
    return (
      <div className="p-4 text-center text-gray-500" data-testid="no-quality">
        Select an episode to view quality metrics
      </div>
    );
  }

  if (loading) {
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

  if (!quality) {
    return (
      <div className="p-4 text-center text-gray-500">
        No quality data available
      </div>
    );
  }

  return (
    <div className="p-4" data-testid="quality-panel">
      {/* Grade Badge */}
      <QualityBadge quality={quality} />

      {/* Quality Flags */}
      <QualityFlags quality={quality} />

      {/* Temporal Metrics */}
      <div className="mb-4">
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
          Temporal Quality ({Math.round(quality.temporal.overall_temporal_score * 100)}%)
        </h4>
        <MetricBar
          label="Motion Smoothness"
          value={quality.temporal.motion_smoothness}
          colorClass="bg-green-500"
          metricKey="motion_smoothness"
          isSelected={selectedMetric === 'motion_smoothness'}
          onClick={handleMetricClick}
        />
        <MetricBar
          label="Action-Obs Sync"
          value={quality.temporal.sync_score}
          colorClass="bg-cyan-500"
          metricKey="sync_score"
          isSelected={selectedMetric === 'sync_score'}
          onClick={handleMetricClick}
        />
        <MetricBar
          label="Action Consistency"
          value={quality.temporal.action_consistency}
          colorClass="bg-green-500"
          metricKey="action_consistency"
          isSelected={selectedMetric === 'action_consistency'}
          onClick={handleMetricClick}
        />
        <MetricBar
          label="Completeness"
          value={quality.temporal.trajectory_completeness}
          colorClass="bg-green-500"
          metricKey="trajectory_completeness"
          isSelected={selectedMetric === 'trajectory_completeness'}
          onClick={handleMetricClick}
        />
      </div>

      {/* Diversity Metrics */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
          Diversity Quality ({Math.round(quality.diversity.overall_diversity_score * 100)}%)
        </h4>
        <MetricBar
          label="Recovery Behaviors"
          value={quality.diversity.recovery_behavior_score}
          colorClass="bg-purple-500"
          metricKey="recovery_behavior_score"
          isSelected={selectedMetric === 'recovery_behavior_score'}
          onClick={handleMetricClick}
        />
        <MetricBar
          label="Transition Diversity"
          value={quality.diversity.transition_diversity}
          colorClass="bg-purple-500"
          metricKey="transition_diversity"
          isSelected={selectedMetric === 'transition_diversity'}
          onClick={handleMetricClick}
        />
        <MetricBar
          label="Near-Miss Handling"
          value={quality.diversity.near_miss_ratio}
          colorClass="bg-purple-500"
          metricKey="near_miss_ratio"
          isSelected={selectedMetric === 'near_miss_ratio'}
          onClick={handleMetricClick}
        />
        <MetricBar
          label="Action Coverage"
          value={quality.diversity.action_space_coverage}
          colorClass="bg-purple-500"
          metricKey="action_space_coverage"
          isSelected={selectedMetric === 'action_space_coverage'}
          onClick={handleMetricClick}
        />
      </div>

      {/* Quality Events */}
      {events.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Detected Events ({events.length})
          </h4>
          <EventBadges
            events={events}
            onJumpToFrame={onJumpToFrame}
            maxShow={12}
          />
          <div className="mt-2 text-xs text-gray-500">
            Click to jump to frame
          </div>
        </div>
      )}
    </div>
  );
}
