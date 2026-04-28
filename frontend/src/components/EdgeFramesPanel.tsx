"use client";

import { useMemo } from "react";
import { useEdgeFrames } from "@/hooks/useDatasetAnalysis";
import type { EdgeFramePosition } from "@/types/analysis";

interface EdgeFramesPanelProps {
  datasetId: string | null;
  taskName: string | null;
  onNavigateToEpisode?: (datasetId: string, episodeId: string, numFrames: number) => void;
}

function ToggleButton({
  active,
  label,
  onClick,
  testId,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
  testId: string;
}) {
  return (
    <button
      onClick={onClick}
      data-testid={testId}
      className={`px-3 py-1 text-xs font-medium transition-colors ${
        active
          ? "bg-blue-600 text-white"
          : "bg-white dark:bg-gray-900 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
      }`}
    >
      {label}
    </button>
  );
}

export default function EdgeFramesPanel({
  datasetId,
  taskName,
  onNavigateToEpisode,
}: EdgeFramesPanelProps) {
  const { state, setPosition } = useEdgeFrames(datasetId, taskName);

  const items = useMemo(() => {
    const map = state.framesByPos[state.position];
    return Array.from(map.values()).sort((a, b) => a.episode_index - b.episode_index);
  }, [state.framesByPos, state.position]);

  if (!datasetId) {
    return (
      <div className="py-6 text-sm text-gray-500 text-center" data-testid="edge-frames-no-task">
        Select a dataset to load starting &amp; ending frames.
      </div>
    );
  }

  const total = state.totalByPos[state.position];
  const totalForTask = state.totalForTaskByPos[state.position];
  const loaded = state.loadedByPos[state.position];
  const phase = state.phaseByPos[state.position];
  const error = state.errorByPos[state.position];

  const progressPct = total > 0 ? Math.min(100, Math.round((loaded / total) * 100)) : 0;

  const positionLabel = state.position === "start" ? "Starting" : "Ending";

  return (
    <div className="space-y-4" data-testid="edge-frames-panel">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="text-xs text-gray-600 dark:text-gray-400">
          <span className="font-medium">{taskName ? "Task:" : "Scope:"}</span>{" "}
          <span className="text-gray-900 dark:text-white">{taskName || "All tasks"}</span>
          {totalForTask > 0 && (
            <span className="ml-2 text-gray-500 dark:text-gray-400">
              · showing <span className="font-semibold">{Math.min(total, totalForTask)}</span> of{" "}
              <span className="font-semibold">{totalForTask}</span> episodes
            </span>
          )}
        </div>

        <div className="ml-auto flex items-center gap-3">
          <span className="text-xs text-gray-500 dark:text-gray-400">Frame:</span>
          <div
            className="inline-flex rounded-md overflow-hidden border border-gray-300 dark:border-gray-600"
            role="radiogroup"
            aria-label="Frame position"
          >
            <ToggleButton
              active={state.position === "start"}
              label="Starting"
              onClick={() => setPosition("start" as EdgeFramePosition)}
              testId="edge-frames-toggle-start"
            />
            <ToggleButton
              active={state.position === "end"}
              label="Ending"
              onClick={() => setPosition("end" as EdgeFramePosition)}
              testId="edge-frames-toggle-end"
            />
          </div>
        </div>
      </div>

      {/* Progress / status */}
      {phase === "loading" && (
        <div className="flex items-center gap-3" data-testid="edge-frames-progress">
          <div className="flex-1 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <span className="text-xs text-gray-500 dark:text-gray-400 tabular-nums">
            {loaded}/{total > 0 ? total : "…"} loaded
          </span>
        </div>
      )}

      {phase === "error" && error && (
        <div
          className="px-3 py-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded text-xs text-red-700 dark:text-red-300"
          data-testid="edge-frames-error"
        >
          {error}
        </div>
      )}

      {/* Grid */}
      {total === 0 && phase !== "loading" ? (
        <div className="py-6 text-sm text-gray-500 text-center" data-testid="edge-frames-empty">
          No episodes available for this task.
        </div>
      ) : (
        <div
          className="grid gap-3"
          style={{ gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))" }}
          data-testid="edge-frames-grid"
        >
          {items.map((item) => {
            const hasImage = !!item.image_b64;
            const isError = !!item.error;
            const tile = (
              <div
                className={`relative bg-gray-100 dark:bg-gray-800 rounded-md overflow-hidden border border-gray-200 dark:border-gray-700 ${
                  hasImage && onNavigateToEpisode
                    ? "cursor-pointer hover:ring-2 hover:ring-blue-500 transition-all"
                    : ""
                }`}
                style={{ aspectRatio: "4 / 3" }}
                onClick={() => {
                  if (hasImage && onNavigateToEpisode && datasetId && item.total_frames) {
                    onNavigateToEpisode(datasetId, item.episode_id, item.total_frames);
                  }
                }}
                data-testid={`edge-frames-tile-${item.episode_index}`}
              >
                {hasImage ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={`data:image/jpeg;base64,${item.image_b64}`}
                    alt={`${positionLabel} frame of episode ${item.episode_index}`}
                    className="w-full h-full object-cover"
                  />
                ) : isError ? (
                  <div
                    className="absolute inset-0 flex items-center justify-center text-gray-400 dark:text-gray-500 text-xs"
                    title={item.error || "decode failed"}
                  >
                    —
                  </div>
                ) : (
                  <div className="absolute inset-0 animate-pulse bg-gradient-to-br from-gray-200 via-gray-100 to-gray-200 dark:from-gray-800 dark:via-gray-700 dark:to-gray-800" />
                )}
              </div>
            );

            return (
              <div key={item.episode_index} className="space-y-1">
                {tile}
                <div className="flex items-baseline justify-between text-[11px] text-gray-600 dark:text-gray-400 px-0.5">
                  <span className="font-mono">ep {item.episode_index}</span>
                  {item.total_frames != null && (
                    <span className="tabular-nums">{item.total_frames}f</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
