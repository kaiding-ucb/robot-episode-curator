"use client";

import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";

// Dynamic import to avoid SSR issues with WebAssembly
const WebViewer = dynamic(
  () => import("@rerun-io/web-viewer-react").then((mod) => mod.default),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-full bg-gray-900 text-gray-400">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
          <p>Loading Rerun Viewer...</p>
        </div>
      </div>
    )
  }
);

interface RerunViewerProps {
  datasetId: string | null;
  episodeId: string | null;
  onClose?: () => void;
}

export default function RerunViewer({ datasetId, episodeId, onClose }: RerunViewerProps) {
  const [rrdUrl, setRrdUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generationProgress, setGenerationProgress] = useState<string>("");

  const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001/api";

  const generateRrd = useCallback(async () => {
    if (!datasetId || !episodeId) return;

    setLoading(true);
    setError(null);
    setGenerationProgress("Generating Rerun recording...");

    try {
      // Request RRD generation from backend
      const response = await fetch(
        `${apiBaseUrl}/rerun/generate/${encodeURIComponent(episodeId)}?dataset_id=${encodeURIComponent(datasetId)}`,
        { method: "POST" }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to generate RRD: ${response.status}`);
      }

      const data = await response.json();

      if (data.rrd_url) {
        // Construct full URL for the RRD file
        const baseUrl = apiBaseUrl.replace("/api", "");
        const fullRrdUrl = `${baseUrl}${data.rrd_url}`;
        console.log("[RerunViewer] Generated RRD URL:", fullRrdUrl);
        console.log("[RerunViewer] Response data:", data);
        setRrdUrl(fullRrdUrl);
        setGenerationProgress("");
      } else {
        throw new Error("No RRD URL returned");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate Rerun recording");
      setGenerationProgress("");
    } finally {
      setLoading(false);
    }
  }, [datasetId, episodeId, apiBaseUrl]);

  // Generate RRD when episode changes
  useEffect(() => {
    if (datasetId && episodeId) {
      generateRrd();
    } else {
      setRrdUrl(null);
    }
  }, [datasetId, episodeId, generateRrd]);

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
      {/* Close button */}
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

      {/* Debug: Show RRD URL */}
      <div className="absolute bottom-2 left-2 z-10 bg-black/80 text-green-400 text-xs px-2 py-1 rounded font-mono max-w-md truncate">
        RRD: {rrdUrl}
      </div>

      {/* Rerun WebViewer */}
      <WebViewer
        width="100%"
        height="100%"
        rrd={rrdUrl}
        hide_welcome_screen={true}
        onReady={() => console.log("[RerunViewer] Viewer ready")}
        onRecordingOpen={(event) => console.log("[RerunViewer] Recording opened:", event)}
      />
    </div>
  );
}
