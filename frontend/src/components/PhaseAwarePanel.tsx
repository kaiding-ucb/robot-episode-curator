"use client";

import { useEffect, useRef, useState } from "react";

interface PhaseAwareReason {
  signal: "envelope" | "duration" | "shape" | "cycle_count" | "gemini";
  phase?: string | null;
  cycle?: number | null;
  feature?: string | null;
  magnitude: number;
  explanation: string;
}

interface PhaseAwarePhase {
  name: string;
  start: number;
  end: number;
  cycle: number;
  duration: number;
}

interface GeminiEvidence {
  episode_id?: string;
  timestamp?: string;
  note?: string;
}

interface GeminiObservation {
  phase?: string;
  cycle?: number;
  timestamp?: string;
  observation?: string;
}

interface PhaseAwareEpisode {
  episode_id: string;
  cluster: string | null;
  num_cycles: number;
  frames: number;
  phases: PhaseAwarePhase[];
  shape_features: Record<string, number>;
  anomaly: {
    is_anomaly: boolean;
    reasons: PhaseAwareReason[];
  };
  gemini_severity?: "stylistic" | "suspicious" | "mistake" | null;
  gemini_confirmation?: string;
  gemini_observations?: GeminiObservation[];
}

interface PhaseAwareCluster {
  id: string;
  label: string;
  members: string[];
  medoid: string;
  dominant_features: string[];
  dominant_features_human?: string[];
  gemini_label?: string;
  gemini_description?: string;
  gemini_evidence?: GeminiEvidence[];
  gemini_confidence?: "high" | "medium" | "low";
}

interface GeminiMeta {
  enriched: boolean;
  cached?: boolean;
  timings?: Record<string, unknown>;
  token_usage?: { prompt?: number; response?: number; thought?: number; total?: number };
  flagged_shown?: number;
  flagged_total?: number;
  flagged_capped?: boolean;
  errors?: string[];
  error?: string;
}

interface PhaseAwareData {
  task_name: string;
  cohort_size: number;
  method: string;
  algorithm?: Record<string, number>;
  clusters: PhaseAwareCluster[];
  episodes: PhaseAwareEpisode[];
  gemini?: GeminiMeta;
}

interface PhaseAwarePanelProps {
  datasetId: string;
  taskName: string;
  onNavigateToEpisode?: (datasetId: string, episodeId: string, numFrames: number, targetFrame?: number) => void;
}

// Phase-aware analyzes all episodes in the task, capped at 50 to keep the
// pipeline tractable (UMI's single task has 1447 — analyzing all of them
// would be minutes of work). Backend's `list_task_episodes(limit=N)` returns
// fewer than N when the task is smaller, so this gives "all if ≤50, else 50".
const COHORT_CAP = 50;

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";


const CLUSTER_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#a855f7"];

export function PhaseAwarePanel({ datasetId, taskName, onNavigateToEpisode }: PhaseAwarePanelProps) {
  const [data, setData] = useState<PhaseAwareData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gemLoading, setGemLoading] = useState(false);
  const [gemError, setGemError] = useState<string | null>(null);

  const fetchData = (includeGemini: boolean, signal?: AbortSignal) =>
    fetch(
      `${API_BASE}/datasets/${encodeURIComponent(datasetId)}/analysis/phase-aware?task_name=${encodeURIComponent(taskName)}&cohort_size=${COHORT_CAP}${includeGemini ? "&include_gemini=true" : ""}`,
      { signal }
    ).then(async (r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<PhaseAwareData>;
    });

  useEffect(() => {
    if (!datasetId || !taskName) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setData(null);

    fetchData(false, controller.signal)
      .then((d) => {
        if (controller.signal.aborted) return;
        if (d.method === "unsupported" || "error" in d) {
          setError((d as unknown as { error?: string }).error || "Phase-aware analysis unavailable for this dataset");
        } else {
          setData(d);
        }
      })
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, taskName]);

  // Streaming progress state — populated event-by-event as Gemini calls complete.
  const [clipProgress, setClipProgress] = useState<{ done: number; total: number } | null>(null);
  const [clusterPending, setClusterPending] = useState<Set<string>>(new Set());
  const [episodePending, setEpisodePending] = useState<Set<string>>(new Set());
  const evtSrcRef = useRef<EventSource | null>(null);

  const runGeminiEnrichment = () => {
    setGemLoading(true);
    setGemError(null);
    setClipProgress(null);

    // Pre-mark every cluster + flagged episode as "AI pending" so cards can
    // show inline spinners until their event arrives.
    if (data) {
      setClusterPending(new Set(data.clusters.map((c) => c.id)));
      setEpisodePending(
        new Set(data.episodes.filter((e) => e.anomaly.is_anomaly).map((e) => e.episode_id))
      );
    }

    if (evtSrcRef.current) {
      evtSrcRef.current.close();
    }
    const url = `${API_BASE}/datasets/${encodeURIComponent(datasetId)}/analysis/phase-aware/stream?task_name=${encodeURIComponent(taskName)}&cohort_size=${COHORT_CAP}`;
    const es = new EventSource(url);
    evtSrcRef.current = es;

    es.addEventListener("phase_aware", (ev) => {
      try {
        const fresh = JSON.parse((ev as MessageEvent).data) as PhaseAwareData;
        setData(fresh);
        // Re-seed pending sets against the freshly computed phase-aware data.
        setClusterPending(new Set(fresh.clusters.map((c) => c.id)));
        setEpisodePending(
          new Set(fresh.episodes.filter((e) => e.anomaly.is_anomaly).map((e) => e.episode_id))
        );
      } catch (e) {
        console.warn("phase_aware event parse failed", e);
      }
    });

    es.addEventListener("clip_progress", (ev) => {
      try {
        const p = JSON.parse((ev as MessageEvent).data) as { done: number; total: number };
        setClipProgress({ done: p.done, total: p.total });
      } catch {
        /* ignore */
      }
    });

    es.addEventListener("cluster", (ev) => {
      try {
        const c = JSON.parse((ev as MessageEvent).data) as {
          cluster_id: string;
          gemini_label?: string;
          gemini_description?: string;
          gemini_evidence?: GeminiEvidence[];
          gemini_confidence?: "high" | "medium" | "low";
        };
        setData((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            clusters: prev.clusters.map((cl) =>
              cl.id === c.cluster_id
                ? {
                    ...cl,
                    gemini_label: c.gemini_label,
                    gemini_description: c.gemini_description,
                    gemini_evidence: c.gemini_evidence,
                    gemini_confidence: c.gemini_confidence,
                  }
                : cl
            ),
          };
        });
        setClusterPending((prev) => {
          const next = new Set(prev);
          next.delete(c.cluster_id);
          return next;
        });
      } catch (e) {
        console.warn("cluster event parse failed", e);
      }
    });

    es.addEventListener("flag_batch", (ev) => {
      try {
        const batch = JSON.parse((ev as MessageEvent).data) as {
          episodes: Array<{
            episode_id: string;
            gemini_severity?: PhaseAwareEpisode["gemini_severity"];
            gemini_confirmation?: string;
            gemini_observations?: GeminiObservation[];
          }>;
        };
        setData((prev) => {
          if (!prev) return prev;
          const updated = new Map(batch.episodes.map((e) => [e.episode_id, e]));
          return {
            ...prev,
            episodes: prev.episodes.map((e) => {
              const u = updated.get(e.episode_id);
              if (!u) return e;
              const novelReasons: PhaseAwareReason[] = (u.gemini_observations || []).map(
                (o) => ({
                  signal: "gemini",
                  phase: o.phase,
                  cycle: o.cycle,
                  feature: null,
                  magnitude: 0,
                  explanation: `AI-observed at ${o.timestamp ?? "?"}: ${o.observation ?? ""}`,
                })
              );
              return {
                ...e,
                gemini_severity: u.gemini_severity,
                gemini_confirmation: u.gemini_confirmation,
                gemini_observations: u.gemini_observations || [],
                anomaly: {
                  ...e.anomaly,
                  reasons: [
                    ...e.anomaly.reasons.filter((r) => r.signal !== "gemini"),
                    ...novelReasons,
                  ],
                },
              };
            }),
          };
        });
        setEpisodePending((prev) => {
          const next = new Set(prev);
          for (const e of batch.episodes) next.delete(e.episode_id);
          return next;
        });
      } catch (e) {
        console.warn("flag_batch event parse failed", e);
      }
    });

    es.addEventListener("done", (ev) => {
      try {
        const meta = JSON.parse((ev as MessageEvent).data) as {
          token_usage?: GeminiMeta["token_usage"];
          timings?: GeminiMeta["timings"];
          errors?: string[];
        };
        setData((prev) =>
          prev
            ? {
                ...prev,
                gemini: {
                  enriched: true,
                  cached: false,
                  token_usage: meta.token_usage,
                  timings: meta.timings,
                  errors: meta.errors || [],
                },
              }
            : prev
        );
        setClusterPending(new Set());
        setEpisodePending(new Set());
      } catch {
        /* ignore */
      }
      setGemLoading(false);
      es.close();
      evtSrcRef.current = null;
    });

    es.addEventListener("error", (ev) => {
      // SSE 'error' fires both for our explicit error events and for connection
      // drops. The MessageEvent.data is set only for explicit emissions.
      const me = ev as MessageEvent;
      if (me && typeof me.data === "string" && me.data.length > 0) {
        try {
          const payload = JSON.parse(me.data) as { error?: string };
          if (payload.error) setGemError(payload.error);
        } catch {
          /* ignore */
        }
      } else if (es.readyState === EventSource.CLOSED) {
        setGemError("Stream closed unexpectedly");
        setGemLoading(false);
      }
    });
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (evtSrcRef.current) {
        evtSrcRef.current.close();
        evtSrcRef.current = null;
      }
    };
  }, []);

  if (loading) {
    return (
      <div className="py-12 text-center text-gray-500">
        <div className="inline-block w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mb-3"></div>
        <div className="text-sm">Running phase-aware analysis…</div>
        <div className="text-xs text-gray-400 mt-1">Downloading parquets &amp; computing envelopes (~10–30s)</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="py-6 px-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded text-sm text-amber-800 dark:text-amber-300" data-testid="phase-aware-error">
        {error}
      </div>
    );
  }

  if (!data) return null;

  const flagged = data.episodes.filter((e) => e.anomaly.is_anomaly);
  const clusterColor = (id: string | null | undefined) => {
    if (!id) return "#6b7280";
    const idx = data.clusters.findIndex((c) => c.id === id);
    return idx >= 0 ? CLUSTER_COLORS[idx % CLUSTER_COLORS.length] : "#6b7280";
  };
  const hasGemini = data.gemini?.enriched === true;

  return (
    <div data-testid="phase-aware-panel" className="space-y-6">
      {/* Cohort summary — analyzed-cohort metadata; selector retired in favour
          of "all-or-up-to-50". */}
      {data && (
        <section>
          <div className="flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400" data-testid="cohort-summary">
            <span>
              cohort: <span className="font-mono text-gray-700 dark:text-gray-300">{data.cohort_size}</span>
              <span className="text-gray-400 dark:text-gray-500"> episodes</span>
              {data.cohort_size >= COHORT_CAP && (
                <span className="text-gray-400 dark:text-gray-500"> (capped at {COHORT_CAP}; task may have more)</span>
              )}
            </span>
            <span>·</span>
            <span>K=<span className="font-mono text-gray-700 dark:text-gray-300">{data.clusters.length}</span></span>
            <span>·</span>
            <span>
              <span className="font-mono text-gray-700 dark:text-gray-300">
                {data.episodes.filter(e => e.anomaly.is_anomaly).length}
              </span> flagged
            </span>
          </div>
        </section>
      )}

      {/* Run AI analysis button */}
      <section>
        {!hasGemini && !gemLoading && (
          <div className="flex items-center justify-between gap-3 px-3 py-2 bg-fuchsia-50 dark:bg-fuchsia-900/20 border border-fuchsia-200 dark:border-fuchsia-800 rounded">
            <div className="text-xs text-fuchsia-800 dark:text-fuchsia-200">
              <span className="font-semibold">AI semantic analysis</span> — send cluster medoids + flagged episodes to Gemini 3 Flash for behaviour-level descriptions and novel anomaly discovery. Streams updates as each Gemini call completes (~60–120 s total). Budget ≤ $0.05/task.
            </div>
            <button
              onClick={runGeminiEnrichment}
              disabled={gemLoading}
              className="text-xs px-3 py-1.5 bg-fuchsia-600 hover:bg-fuchsia-700 disabled:opacity-50 text-white rounded transition-colors flex-shrink-0"
              data-testid="run-ai-analysis-btn"
            >
              Run AI analysis
            </button>
          </div>
        )}
        {gemLoading && (
          <div className="px-3 py-2 bg-fuchsia-50 dark:bg-fuchsia-900/20 border border-fuchsia-200 dark:border-fuchsia-800 rounded text-xs text-fuchsia-800 dark:text-fuchsia-200" data-testid="ai-streaming-status">
            <div className="flex items-center gap-2">
              <span className="inline-block w-3 h-3 border-2 border-fuchsia-500 border-t-transparent rounded-full animate-spin"></span>
              <span className="font-semibold">AI analysis streaming…</span>
              {clipProgress && (
                <span className="text-fuchsia-700 dark:text-fuchsia-300">
                  preparing clips {clipProgress.done}/{clipProgress.total}
                </span>
              )}
              <span className="ml-auto text-fuchsia-600 dark:text-fuchsia-400">
                {data ? data.clusters.length - clusterPending.size : 0}/{data?.clusters.length ?? 0} clusters · {data ? data.episodes.filter(e => e.anomaly.is_anomaly).length - episodePending.size : 0}/{data?.episodes.filter(e => e.anomaly.is_anomaly).length ?? 0} flagged reviewed
              </span>
            </div>
          </div>
        )}
        {gemError && (
          <div className="mt-2 px-3 py-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded text-xs text-red-700 dark:text-red-300" data-testid="ai-error">
            AI analysis error: {gemError}
          </div>
        )}
        {hasGemini && data.gemini?.timings && (
          <div className="text-[11px] text-gray-500 dark:text-gray-500" data-testid="ai-enriched-note">
            AI-enriched
            {data.gemini.cached ? " (cached)" : ""}
            {data.gemini.token_usage?.total ? ` · ${data.gemini.token_usage.total.toLocaleString()} tokens` : ""}
            {data.gemini.flagged_capped ? ` · showing ${data.gemini.flagged_shown}/${data.gemini.flagged_total} flagged (soft cap)` : ""}
          </div>
        )}
      </section>

      {/* Variance clusters */}
      <section>
        <div className="flex items-baseline justify-between mb-2">
          <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
            Variance clusters
            <span className="ml-2 text-xs font-normal text-gray-500">
              K={data.clusters.length} (gap statistic) · {data.cohort_size} episodes
            </span>
          </h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {data.clusters.map((c, i) => (
            <div
              key={c.id}
              className="border dark:border-gray-700 rounded-lg p-3 bg-white dark:bg-gray-900"
              data-testid={`cluster-card-${c.id}`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }}
                ></span>
                <span className="font-semibold text-sm text-gray-900 dark:text-gray-100">
                  {c.id}: {c.gemini_label || c.label}
                </span>
              </div>
              {/* AI confidence badge + stat-label subtitle removed; cluster
                  title alone (Gemini label or stat fallback) is enough. */}
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                {c.members.length} episodes · medoid {c.medoid}
              </div>
              {c.gemini_description && (
                <div className="text-xs text-gray-700 dark:text-gray-300 mb-2 leading-snug" data-testid={`cluster-description-${c.id}`}>
                  {c.gemini_description}
                </div>
              )}
              {gemLoading && clusterPending.has(c.id) && !c.gemini_description && (
                <div className="text-xs text-fuchsia-600 dark:text-fuchsia-400 mb-2 flex items-center gap-1.5" data-testid={`cluster-pending-${c.id}`}>
                  <span className="inline-block w-2.5 h-2.5 border-2 border-fuchsia-500 border-t-transparent rounded-full animate-spin"></span>
                  AI reviewing this cluster…
                </div>
              )}
              {c.dominant_features.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {c.dominant_features.slice(0, 3).map((f, i) => {
                    const human = c.dominant_features_human?.[i] ?? f;
                    return (
                      <span
                        key={f}
                        className="px-1.5 py-0.5 rounded text-[10px] bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400"
                        title={f === human ? undefined : `metric: ${f}`}
                      >
                        {human}
                      </span>
                    );
                  })}
                </div>
              )}
              <div className="mt-1 text-[11px] text-gray-500 dark:text-gray-400 leading-snug" data-testid={`cluster-members-${c.id}`}>
                <span className="text-gray-400 dark:text-gray-500">Members: </span>
                {[...c.members].sort((a, b) => {
                  const na = parseInt(a.replace(/^episode_/, ""), 10);
                  const nb = parseInt(b.replace(/^episode_/, ""), 10);
                  return (isNaN(na) || isNaN(nb)) ? a.localeCompare(b) : na - nb;
                }).map((m, i, arr) => (
                  <span key={m}>
                    <span
                      className={`font-mono ${m === c.medoid ? "font-semibold text-gray-700 dark:text-gray-200" : ""}`}
                      title={m === c.medoid ? "medoid (cluster center)" : undefined}
                    >
                      {m.replace(/^episode_/, "ep_")}
                    </span>
                    {i < arr.length - 1 && <span className="text-gray-400">, </span>}
                  </span>
                ))}
              </div>
              {/* Evidence list intentionally not rendered (cluster_char v4) —
                  it used to show "ep_X @ timestamp: …" which read like a
                  flag. Categorical descriptions are sufficient. */}
            </div>
          ))}
        </div>
      </section>

      {/* Flagged episodes */}
      <section>
        <div className="flex items-baseline justify-between mb-2">
          <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
            Flagged episodes
            <span className="ml-2 text-xs font-normal text-gray-500">
              {flagged.length} / {data.cohort_size}
            </span>
          </h3>
          {hasGemini && (
            <span className="text-[11px] text-gray-500 dark:text-gray-400">AI review combined into each card</span>
          )}
        </div>
        {flagged.length === 0 ? (
          <div className="py-6 px-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded text-sm text-green-700 dark:text-green-300">
            No anomalies detected under conservative thresholds. All episodes fall within the cohort envelope and MAD-based phase-duration bands.
          </div>
        ) : (
          <div className="space-y-2" data-testid="flagged-episodes-list">
            {flagged.map((e) => {
              const statReasons = e.anomaly.reasons.filter((r) => r.signal !== "gemini");
              const geminiNovel = e.anomaly.reasons.filter((r) => r.signal === "gemini");
              const aiRan = !!e.gemini_severity;
              // When AI has run, the confirmation IS the synthesized view
              // (the prompt already cross-references stat flags + telemetry).
              // Render it as the primary description and surface novel
              // observations as a follow-up. Pre-AI: just the stat reasons.
              return (
              <div
                key={e.episode_id}
                className="border dark:border-gray-700 rounded-lg p-3 bg-white dark:bg-gray-900"
                data-testid={`flagged-card-${e.episode_id}`}
              >
                <div className="flex items-start justify-between mb-2 gap-3">
                  <div className="flex items-center gap-2 min-w-0">
                    {e.cluster && (
                      <span
                        className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                        style={{ backgroundColor: clusterColor(e.cluster) }}
                      ></span>
                    )}
                    <span className="font-mono text-sm font-semibold text-gray-900 dark:text-gray-100">
                      {e.episode_id}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      · {e.cluster ?? "no cluster"} · {e.num_cycles} cycles · {e.frames}f
                    </span>
                    {/* Severity badge (stylistic / suspicious / mistake) removed;
                        the AI confirmation prose conveys severity in plain language. */}
                  </div>
                  {onNavigateToEpisode && (
                    <button
                      onClick={() => onNavigateToEpisode(datasetId, e.episode_id, e.frames, 0)}
                      className="text-xs px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded transition-colors flex-shrink-0"
                      data-testid={`open-episode-${e.episode_id}`}
                    >
                      Open in viewer →
                    </button>
                  )}
                </div>
                {(aiRan || (gemLoading && episodePending.has(e.episode_id))) ? (
                  <div className="space-y-1.5">
                    {/* Combined AI verdict — references the stat flag and the
                        telemetry. Replaces the pre-AI bullet list. */}
                    {e.gemini_confirmation && (
                      <p className="text-xs text-gray-700 dark:text-gray-300 leading-snug">
                        {e.gemini_confirmation}
                      </p>
                    )}
                    {!e.gemini_confirmation && gemLoading && episodePending.has(e.episode_id) && (
                      <p className="text-xs text-fuchsia-600 dark:text-fuchsia-400 flex items-center gap-1.5" data-testid={`episode-pending-${e.episode_id}`}>
                        <span className="inline-block w-2.5 h-2.5 border-2 border-fuchsia-500 border-t-transparent rounded-full animate-spin"></span>
                        AI reviewing this episode…
                      </p>
                    )}
                    {geminiNovel.length > 0 && (
                      <p className="text-xs text-gray-600 dark:text-gray-400 leading-snug">
                        <span className="text-gray-500 dark:text-gray-500">Also visible: </span>
                        {geminiNovel.map((r) => r.explanation).join(" ")}
                      </p>
                    )}
                    <details className="group">
                      <summary className="text-[11px] text-gray-400 dark:text-gray-500 cursor-pointer hover:text-gray-600 dark:hover:text-gray-300 select-none list-none">
                        <span className="group-open:hidden">▸ Why this was flagged statistically</span>
                        <span className="hidden group-open:inline">▾ Statistical detail</span>
                      </summary>
                      <ul className="mt-1.5 ml-3 space-y-0.5">
                        {statReasons.map((r, i) => (
                          <li key={i} className="text-[11px] text-gray-500 dark:text-gray-400 leading-snug">
                            {r.explanation}
                          </li>
                        ))}
                      </ul>
                    </details>
                  </div>
                ) : (
                  <ul className="space-y-1">
                    {statReasons.map((r, i) => (
                      <li key={i} className="text-xs text-gray-700 dark:text-gray-300 leading-snug">
                        {r.explanation}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              );
            })}
          </div>
        )}
      </section>

    </div>
  );
}
