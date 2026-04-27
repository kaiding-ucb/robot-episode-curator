"use client";

import { useMemo, useState } from "react";
import type { MetaSummaryResponse } from "@/types/api";

interface SummaryPanelProps {
  summary: MetaSummaryResponse | null;
  loading: boolean;
  error: string | null;
  onTaskSelected?: (taskName: string) => void;
}

interface FeatureEntry {
  key: string;
  dtype: string;
  shape: number[] | null;
  isCamera: boolean;
}

type FeatureGroup = "cameras" | "observations" | "actions" | "indices" | "other";

function getString(obj: unknown, key: string): string | null {
  if (obj && typeof obj === "object" && key in obj) {
    const v = (obj as Record<string, unknown>)[key];
    if (typeof v === "string" && v.length > 0) return v;
    if (typeof v === "number") return String(v);
  }
  return null;
}

function getNumber(obj: unknown, key: string): number | null {
  if (obj && typeof obj === "object" && key in obj) {
    const v = (obj as Record<string, unknown>)[key];
    if (typeof v === "number" && Number.isFinite(v)) return v;
  }
  return null;
}

function parseFeatures(info: Record<string, unknown> | null): FeatureEntry[] {
  if (!info) return [];
  const features = info["features"];
  if (!features || typeof features !== "object") return [];
  const out: FeatureEntry[] = [];
  for (const [key, raw] of Object.entries(features as Record<string, unknown>)) {
    if (!raw || typeof raw !== "object") continue;
    const f = raw as Record<string, unknown>;
    const dtype = typeof f["dtype"] === "string" ? (f["dtype"] as string) : "?";
    const shapeRaw = f["shape"];
    const shape = Array.isArray(shapeRaw)
      ? (shapeRaw.filter((n) => typeof n === "number") as number[])
      : null;
    const isCamera = dtype === "video" || dtype === "image" || /image|video/.test(key);
    out.push({ key, dtype, shape, isCamera });
  }
  return out;
}

function classifyFeature(f: FeatureEntry): FeatureGroup {
  if (f.isCamera) return "cameras";
  if (f.key === "action" || f.key.startsWith("action.")) return "actions";
  if (f.key.startsWith("observation.")) return "observations";
  if (
    f.key === "index" ||
    f.key === "timestamp" ||
    f.key.endsWith("_index") ||
    f.key.endsWith("_id")
  ) {
    return "indices";
  }
  return "other";
}

function formatShape(shape: number[] | null): string {
  if (!shape || shape.length === 0) return "";
  return `[${shape.join(", ")}]`;
}

function StatTile({ label, value }: { label: string; value: string | number | null }) {
  if (value === null || value === undefined) return null;
  const display = typeof value === "number" ? value.toLocaleString() : value;
  return (
    <div className="flex flex-col items-start px-3 py-2 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md min-w-[88px]">
      <span className="text-[10px] font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
        {label}
      </span>
      <span className="text-base font-semibold text-gray-900 dark:text-white tabular-nums">
        {display}
      </span>
    </div>
  );
}

function KVRow({ k, v, mono = true }: { k: string; v: string | number; mono?: boolean }) {
  return (
    <div className="flex items-baseline gap-3 py-1 border-b border-gray-100 dark:border-gray-800 last:border-b-0">
      <span className="text-xs text-gray-500 dark:text-gray-400 min-w-[120px] flex-shrink-0">
        {k}
      </span>
      <span className={`text-xs text-gray-800 dark:text-gray-200 break-all ${mono ? "font-mono" : ""}`}>
        {v}
      </span>
    </div>
  );
}

function SectionHeader({ title, count }: { title: string; count?: number }) {
  return (
    <div className="flex items-center gap-2 mb-2">
      <h4 className="text-[11px] font-semibold uppercase tracking-wider text-gray-700 dark:text-gray-300">
        {title}
      </h4>
      {count != null && (
        <span className="text-[10px] text-gray-400 dark:text-gray-500 tabular-nums">
          {count}
        </span>
      )}
      <div className="flex-1 h-px bg-gray-200 dark:bg-gray-700" />
    </div>
  );
}

const GROUP_LABEL: Record<FeatureGroup, string> = {
  cameras: "Cameras",
  observations: "Observations",
  actions: "Actions",
  indices: "Indices & timestamps",
  other: "Other",
};

const GROUP_ORDER: FeatureGroup[] = ["cameras", "observations", "actions", "indices", "other"];

const GROUP_BADGE: Record<FeatureGroup, string> = {
  cameras: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300",
  observations: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300",
  actions: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300",
  indices: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
  other: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
};

function FeatureRow({ f, group }: { f: FeatureEntry; group: FeatureGroup }) {
  const shapeStr = formatShape(f.shape);
  return (
    <div className="flex items-center gap-2 py-1 text-xs">
      <span
        className={`text-[9px] font-semibold uppercase px-1.5 py-0.5 rounded ${GROUP_BADGE[group]} flex-shrink-0`}
      >
        {group === "cameras" ? "CAM" : group === "actions" ? "ACT" : group === "observations" ? "OBS" : group === "indices" ? "IDX" : "VAR"}
      </span>
      <span className="font-mono text-gray-800 dark:text-gray-200 truncate flex-1">
        {f.key}
      </span>
      <span className="font-mono text-gray-500 dark:text-gray-400 whitespace-nowrap text-[11px]">
        {shapeStr}
        {shapeStr ? " " : ""}
        {f.dtype}
      </span>
    </div>
  );
}

export default function SummaryPanel({
  summary,
  loading,
  error,
  onTaskSelected,
}: SummaryPanelProps) {
  const [taskFilter, setTaskFilter] = useState("");
  const [showRaw, setShowRaw] = useState(false);

  const info = summary?.info ?? null;
  const features = useMemo(() => parseFeatures(info), [info]);

  const groupedFeatures = useMemo(() => {
    const map: Record<FeatureGroup, FeatureEntry[]> = {
      cameras: [],
      observations: [],
      actions: [],
      indices: [],
      other: [],
    };
    for (const f of features) {
      map[classifyFeature(f)].push(f);
    }
    return map;
  }, [features]);

  const codebaseVersion = getString(info, "codebase_version");
  const robotType = getString(info, "robot_type");
  const fps = getNumber(info, "fps");
  const chunksSize = getNumber(info, "chunks_size");
  const dataPath = getString(info, "data_path");
  const videoPath = getString(info, "video_path");
  const totalEpisodes = getNumber(info, "total_episodes");
  const totalFrames = getNumber(info, "total_frames");
  const totalVideos = getNumber(info, "total_videos");
  const totalChunks = getNumber(info, "total_chunks");
  const totalTasksInfo = getNumber(info, "total_tasks");
  const splits =
    info && typeof info === "object" && info["splits"] && typeof info["splits"] === "object"
      ? (info["splits"] as Record<string, unknown>)
      : null;

  const filteredTasks = useMemo(() => {
    if (!summary?.tasks) return [];
    const q = taskFilter.trim().toLowerCase();
    if (!q) return summary.tasks;
    return summary.tasks.filter(
      (t) =>
        t.task_description.toLowerCase().includes(q) ||
        String(t.task_index).includes(q),
    );
  }, [summary, taskFilter]);

  if (loading) {
    return (
      <div
        className="flex items-center justify-center py-12 text-gray-500 text-sm"
        data-testid="summary-loading"
      >
        <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
        Loading dataset summary…
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-sm text-red-500 py-4" data-testid="summary-error">
        Error: {error}
      </div>
    );
  }

  if (!summary || summary.source === "unavailable") {
    return (
      <div className="py-6 text-sm text-gray-500 text-center" data-testid="summary-unavailable">
        Summary is only available for LeRobot datasets with a <code>meta/</code> folder.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4" data-testid="summary-panel">
      {/* Info panel */}
      <div
        className="bg-gray-50 dark:bg-gray-800/30 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden"
        data-testid="info-panel"
      >
        <div className="flex items-center justify-between px-4 py-2.5 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white font-mono">
              meta/info.json
            </h3>
            {codebaseVersion && (
              <span className="px-1.5 py-0.5 text-[10px] rounded font-mono bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300">
                {codebaseVersion}
              </span>
            )}
          </div>
          <button
            onClick={() => setShowRaw((v) => !v)}
            className="text-xs text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
            data-testid="toggle-raw-info"
          >
            {showRaw ? "Hide raw" : "View raw"}
          </button>
        </div>

        <div className="p-4">
          {showRaw ? (
            <pre
              className="text-xs bg-white dark:bg-gray-900 p-3 rounded border border-gray-200 dark:border-gray-700 overflow-auto max-h-[480px] text-gray-800 dark:text-gray-200"
              data-testid="raw-info-json"
            >
              {JSON.stringify(info, null, 2)}
            </pre>
          ) : (
            <div className="space-y-5">
              {/* Totals — KPI tiles */}
              {(totalEpisodes != null ||
                totalFrames != null ||
                totalTasksInfo != null ||
                totalVideos != null ||
                totalChunks != null) && (
                <section data-testid="totals-section">
                  <SectionHeader title="Totals" />
                  <div className="flex flex-wrap gap-2">
                    <StatTile label="Episodes" value={totalEpisodes} />
                    <StatTile label="Frames" value={totalFrames} />
                    <StatTile label="Tasks" value={totalTasksInfo} />
                    <StatTile label="Videos" value={totalVideos} />
                    <StatTile label="Chunks" value={totalChunks} />
                  </div>
                </section>
              )}

              {/* Schema */}
              {(robotType || fps != null || chunksSize != null) && (
                <section data-testid="schema-section">
                  <SectionHeader title="Schema" />
                  <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md px-3 py-1">
                    {robotType && <KVRow k="robot_type" v={robotType} />}
                    {fps != null && <KVRow k="fps" v={fps} />}
                    {chunksSize != null && <KVRow k="chunks_size" v={chunksSize.toLocaleString()} />}
                  </div>
                </section>
              )}

              {/* Paths */}
              {(dataPath || videoPath) && (
                <section data-testid="paths-section">
                  <SectionHeader title="Paths" />
                  <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md px-3 py-1">
                    {dataPath && <KVRow k="data_path" v={dataPath} />}
                    {videoPath && <KVRow k="video_path" v={videoPath} />}
                  </div>
                </section>
              )}

              {/* Features — grouped */}
              {features.length > 0 && (
                <section data-testid="features-section">
                  <SectionHeader title="Features" count={features.length} />
                  <div className="space-y-3 max-h-[420px] overflow-auto pr-1" data-testid="features-list">
                    {GROUP_ORDER.map((g) => {
                      const items = groupedFeatures[g];
                      if (items.length === 0) return null;
                      return (
                        <div key={g} className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md px-3 py-2">
                          <div className="flex items-center gap-2 mb-1.5">
                            <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                              {GROUP_LABEL[g]}
                            </span>
                            <span className="text-[10px] text-gray-400 dark:text-gray-500 tabular-nums">
                              {items.length}
                            </span>
                          </div>
                          <div className="divide-y divide-gray-100 dark:divide-gray-800">
                            {items.map((f) => (
                              <FeatureRow key={f.key} f={f} group={g} />
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </section>
              )}

              {/* Splits */}
              {splits && Object.keys(splits).length > 0 && (
                <section data-testid="splits-section">
                  <SectionHeader title="Splits" />
                  <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md px-3 py-1">
                    {Object.entries(splits).map(([k, v]) => (
                      <KVRow
                        key={k}
                        k={k}
                        v={typeof v === "string" || typeof v === "number" ? String(v) : JSON.stringify(v)}
                      />
                    ))}
                  </div>
                </section>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Tasks panel */}
      <div
        className="bg-gray-50 dark:bg-gray-800/30 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden"
        data-testid="tasks-panel"
      >
        <div className="flex items-center justify-between px-4 py-2.5 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white font-mono">
            meta/tasks.parquet
          </h3>
          <span className="text-xs text-gray-500 dark:text-gray-400 tabular-nums">
            {summary.tasks.length.toLocaleString()} task{summary.tasks.length === 1 ? "" : "s"}
          </span>
        </div>

        <div className="p-4">
          <input
            type="text"
            value={taskFilter}
            onChange={(e) => setTaskFilter(e.target.value)}
            placeholder="Filter tasks…"
            className="w-full mb-3 px-2 py-1.5 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
            data-testid="task-filter-input"
          />

          {summary.tasks.length === 0 ? (
            <div className="text-xs text-gray-500 py-4 text-center">
              tasks.parquet not available for this dataset.
            </div>
          ) : (
            <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md max-h-[480px] overflow-auto" data-testid="tasks-table-wrapper">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-gray-50 dark:bg-gray-800 z-10">
                  <tr className="text-left text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                    <th className="px-3 py-2 font-medium w-10">#</th>
                    <th className="px-3 py-2 font-medium">Description</th>
                    <th className="px-3 py-2 font-medium text-right w-16">Eps</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTasks.map((t) => (
                    <tr
                      key={t.task_index}
                      onClick={() => onTaskSelected?.(t.task_description)}
                      className={`border-b border-gray-100 dark:border-gray-800 last:border-b-0 ${
                        onTaskSelected
                          ? "cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/60"
                          : ""
                      }`}
                      data-testid={`task-row-${t.task_index}`}
                    >
                      <td className="px-3 py-2 font-mono text-gray-500 dark:text-gray-400 tabular-nums">
                        {t.task_index}
                      </td>
                      <td className="px-3 py-2 text-gray-800 dark:text-gray-200 break-words">
                        {t.task_description}
                      </td>
                      <td className="px-3 py-2 font-mono text-gray-700 dark:text-gray-300 text-right tabular-nums">
                        {t.episode_count != null ? t.episode_count.toLocaleString() : "—"}
                      </td>
                    </tr>
                  ))}
                  {filteredTasks.length === 0 && (
                    <tr>
                      <td colSpan={3} className="py-4 text-center text-gray-500">
                        No tasks match &ldquo;{taskFilter}&rdquo;.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
