"use client";

import { useState, useEffect, useCallback, useRef } from "react";

interface RerunViewerProps {
  datasetId: string | null;
  episodeId: string | null;
  onClose?: () => void;
  comparisonRrdUrl?: string | null;
}

// Module-level singleton — survives React Strict Mode mount→unmount→remount
// without losing the WebGL context.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let globalViewer: any = null;
let globalCanvas: HTMLCanvasElement | null = null;
let currentRrdUrl: string | null = null;
let cleanupTimer: ReturnType<typeof setTimeout> | null = null;

// Suppress wasm-bindgen externref_shim errors that fire from async callbacks
// of an *old* Rerun recording whose WASM slot was freed when we switched
// rrds. These are cosmetic — the viewer recovers on its own — but Next.js
// devtools surfaces them as a big error overlay.
let errorHandlerInstalled = false;
function installErrorSuppressor() {
  if (errorHandlerInstalled || typeof window === "undefined") return;
  errorHandlerInstalled = true;
  const isRerunBenignError = (msg: unknown): boolean => {
    const s = typeof msg === "string" ? msg : (msg instanceof Error ? msg.message : String(msg ?? ""));
    // (a) Stale externref from an old recording's async WASM callback
    // (b) Mid-flight rrd fetch aborted because the user switched episodes
    return (
      /closure\d+_externref_shim/.test(s) ||
      /ERR_CONTENT_LENGTH_MISMATCH/.test(s) ||
      (/(Failed to fetch|network error|NetworkError)/i.test(s) && /\.rrd/.test(s))
    );
  };
  window.addEventListener("error", (e) => {
    if (isRerunBenignError(e.error?.message ?? e.message)) {
      e.preventDefault();
      e.stopImmediatePropagation();
    }
  }, true);
  window.addEventListener("unhandledrejection", (e) => {
    if (isRerunBenignError(e.reason?.message ?? e.reason)) {
      e.preventDefault();
    }
  }, true);
}

// Hide everything except the viewport (cameras + state + action views) and
// the bottom time controls. Applied both via the backend blueprint and here
// as a belt-and-suspenders in case the .rrd is loaded from an older cache.
function applyPanelOverrides(viewer: { override_panel_state?: (p: string, s: string) => void } | null) {
  if (!viewer?.override_panel_state) return;
  try { viewer.override_panel_state("blueprint", "hidden"); } catch { /* ignore */ }
  try { viewer.override_panel_state("selection", "hidden"); } catch { /* ignore */ }
  try { viewer.override_panel_state("time", "collapsed"); } catch { /* ignore */ }
}

function RerunCanvas({ rrdUrl }: { rrdUrl: string }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    installErrorSuppressor();
    const container = containerRef.current;
    if (!container) return;

    // Cancel any pending cleanup from a previous unmount (Strict Mode remount)
    if (cleanupTimer) {
      clearTimeout(cleanupTimer);
      cleanupTimer = null;
    }

    let cancelled = false;

    (async () => {
      const rerun = await import("@rerun-io/web-viewer");
      if (cancelled) return;

      if (!globalViewer) {
        // First time — create the viewer
        globalViewer = new rerun.WebViewer();
        await globalViewer.start(null, container, {
          width: "100%",
          height: "100%",
          hide_welcome_screen: true,
        });
        globalCanvas = container.querySelector("canvas");
        applyPanelOverrides(globalViewer);
      } else {
        // Re-attach existing canvas to new container on remount
        if (globalCanvas && globalCanvas.parentElement !== container) {
          container.appendChild(globalCanvas);
        }
        applyPanelOverrides(globalViewer);
      }

      if (cancelled) return;

      // Close the previous recording before opening a new one — otherwise Rerun
      // keeps both loaded and the metadata panel may continue showing the stale
      // episode (root cause of the "episode_1 header / episode_18 metadata" bug).
      //
      // Wrap in try/catch: on rapid episode switching, a pending load handler
      // from the previous rrd can fire after we call close(), with a stale
      // externref into freed WASM memory. Swallow and continue — open() will
      // reset viewer state for the new recording.
      if (currentRrdUrl && currentRrdUrl !== rrdUrl) {
        try { globalViewer.close(currentRrdUrl); } catch (e) {
          console.warn("[RerunViewer] close() on previous rrd raised:", e);
        }
      }
      try {
        globalViewer.open(rrdUrl);
        currentRrdUrl = rrdUrl;
      } catch (e) {
        console.error("[RerunViewer] open() raised — recreating viewer:", e);
        try { globalViewer?.stop(); } catch { /* ignore */ }
        globalViewer = null;
        globalCanvas = null;
        currentRrdUrl = null;
        // Recreate on next tick
        globalViewer = new rerun.WebViewer();
        await globalViewer.start(null, container, {
          width: "100%",
          height: "100%",
          hide_welcome_screen: true,
        });
        globalCanvas = container.querySelector("canvas");
        applyPanelOverrides(globalViewer);
        globalViewer.open(rrdUrl);
        currentRrdUrl = rrdUrl;
      }
    })().catch((err) => {
      if (!cancelled) {
        console.error("[RerunViewer] Failed to start viewer:", err);
      }
    });

    return () => {
      cancelled = true;
      // Delay cleanup to distinguish Strict Mode remount from true unmount.
      // If the component remounts within 200ms, the timer is cancelled above.
      cleanupTimer = setTimeout(() => {
        if (!document.querySelector("[data-rerun-container]")) {
          try { globalViewer?.stop(); } catch { /* ignore */ }
          globalViewer = null;
          globalCanvas = null;
          currentRrdUrl = null;
        }
        cleanupTimer = null;
      }, 200);
    };
  }, [rrdUrl]);

  return (
    <div
      ref={containerRef}
      data-rerun-container
      style={{ width: "100%", height: "100%", position: "relative" }}
    />
  );
}

export default function RerunViewer({ datasetId, episodeId, onClose, comparisonRrdUrl }: RerunViewerProps) {
  const [rrdUrl, setRrdUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generationProgress, setGenerationProgress] = useState<string>("");

  const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
  const abortRef = useRef<AbortController | null>(null);
  const [retryNonce, setRetryNonce] = useState(0);

  useEffect(() => {
    // If a comparison URL is provided, use it directly
    if (comparisonRrdUrl) {
      setRrdUrl(comparisonRrdUrl);
      setLoading(false);
      setError(null);
      return;
    }
    if (!datasetId || !episodeId) {
      setRrdUrl(null);
      return;
    }

    // Cancel any in-flight request from a previous episode. Without this,
    // rapid episode clicks stack concurrent generate_rrd calls; the late
    // responses arrive after the user has moved on and stomp the current
    // rrdUrl, which in turn triggers a WASM open() on a stale handle.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);
    setGenerationProgress("Generating Rerun recording...");

    (async () => {
      try {
        const response = await fetch(
          `${apiBaseUrl}/rerun/generate/${encodeURIComponent(episodeId)}?dataset_id=${encodeURIComponent(datasetId)}&max_frames=0`,
          { method: "POST", signal: controller.signal }
        );
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || `Failed to generate RRD: ${response.status}`);
        }
        const data = await response.json();
        if (controller.signal.aborted) return;
        if (!data.rrd_url) throw new Error("No RRD URL returned");
        const baseUrl = apiBaseUrl.replace("/api", "");
        const sep = data.rrd_url.includes("?") ? "&" : "?";
        const fullRrdUrl = `${baseUrl}${data.rrd_url}${sep}v=${encodeURIComponent(episodeId)}_${Date.now()}`;
        setRrdUrl(fullRrdUrl);
        setGenerationProgress("");
      } catch (err) {
        if (controller.signal.aborted) return;
        const msg = err instanceof Error ? err.message : String(err);
        // Ignore AbortError — that's our own cancellation, not a real failure
        if (err instanceof Error && err.name === "AbortError") return;
        setError(msg || "Failed to generate Rerun recording");
        setGenerationProgress("");
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    })();

    return () => {
      controller.abort();
    };
  }, [datasetId, episodeId, comparisonRrdUrl, apiBaseUrl, retryNonce]);

  const generateRrd = useCallback(() => {
    abortRef.current?.abort();
    setError(null);
    setRrdUrl(null);
    setRetryNonce((n) => n + 1);
  }, []);

  if (!episodeId) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900 text-gray-500">
        <div className="text-center">
          <svg
            className="w-16 h-16 mx-auto mb-4 text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2"
            />
          </svg>
          <p>Select an episode to view in Rerun</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900 text-gray-400">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
          <p>{generationProgress || "Preparing Rerun viewer..."}</p>
          <p className="text-xs text-gray-500 mt-2">This may take a moment for large episodes</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900 text-red-400">
        <div className="text-center max-w-md">
          <svg
            className="w-12 h-12 mx-auto mb-4 text-red-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
          <p className="font-medium mb-2">Failed to load Rerun viewer</p>
          <p className="text-sm text-gray-500 mb-4">{error}</p>
          <button
            onClick={generateRrd}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!rrdUrl) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900 text-gray-500">
        <p>No recording available</p>
      </div>
    );
  }

  return (
    <div className="relative h-full w-full bg-gray-900">
      {onClose && (
        <button
          onClick={onClose}
          className="absolute top-2 right-2 z-10 p-2 bg-gray-800/80 hover:bg-gray-700 text-gray-300 rounded-lg transition-colors"
          title="Close Rerun viewer"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}

      <RerunCanvas rrdUrl={rrdUrl} />
    </div>
  );
}
