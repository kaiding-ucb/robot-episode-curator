"use client";

import { useState } from "react";
import { probeDataset, addDataset } from "@/hooks/useApi";
import type { ProbeResponse, AddDatasetResponse, Modality } from "@/types/api";

interface AddDatasetDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onDatasetAdded: () => void;
}

export default function AddDatasetDialog({
  isOpen,
  onClose,
  onDatasetAdded,
}: AddDatasetDialogProps) {
  const [url, setUrl] = useState("");
  const [customName, setCustomName] = useState("");
  const [probing, setProbing] = useState(false);
  const [adding, setAdding] = useState(false);
  const [probeResult, setProbeResult] = useState<ProbeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleProbe = async () => {
    if (!url.trim()) return;

    setProbing(true);
    setError(null);
    setProbeResult(null);

    try {
      const result = await probeDataset(url);
      setProbeResult(result);
      if (result.error) {
        setError(result.error);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to probe dataset");
    } finally {
      setProbing(false);
    }
  };

  const handleAdd = async () => {
    if (!probeResult) return;

    setAdding(true);
    setError(null);

    try {
      const result: AddDatasetResponse = await addDataset(
        url,
        customName || undefined
      );
      if (result.success) {
        onDatasetAdded();
        handleClose();
      } else {
        setError(result.error || "Failed to add dataset");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to add dataset");
    } finally {
      setAdding(false);
    }
  };

  const handleClose = () => {
    setUrl("");
    setCustomName("");
    setProbeResult(null);
    setError(null);
    onClose();
  };

  const getModalityBadgeColor = (modality: Modality): string => {
    const colors: Record<Modality, string> = {
      rgb: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
      depth: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
      imu: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
      actions: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
      states: "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200",
    };
    return colors[modality] || colors.rgb;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-md mx-4 p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold">Add HuggingFace Dataset</h2>
          <button
            onClick={handleClose}
            className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* URL Input */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">
            HuggingFace Dataset URL
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://huggingface.co/datasets/user/repo"
              className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600 text-sm"
              data-testid="dataset-url-input"
            />
            <button
              onClick={handleProbe}
              disabled={probing || !url.trim()}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              data-testid="probe-btn"
            >
              {probing ? "..." : "Probe"}
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-1">
            Enter a HuggingFace dataset URL or repo ID (e.g., user/repo)
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-4 p-3 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-lg text-sm">
            {error}
          </div>
        )}

        {/* Probe Results */}
        {probeResult && !probeResult.error && (
          <div className="mb-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg" data-testid="probe-result">
            <h3 className="text-sm font-medium mb-3">Dataset Information</h3>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-500">Repository:</span>
                <span className="font-mono">{probeResult.repo_id}</span>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-500">Format:</span>
                <span className="font-medium">
                  {probeResult.format_detected || "Unknown"}
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-500">Structure:</span>
                <span>
                  {probeResult.has_tasks ? "Hierarchical (tasks)" : "Flat"}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-gray-500">Modalities:</span>
                <div className="flex gap-1">
                  {probeResult.modalities.map((mod) => (
                    <span
                      key={mod}
                      className={`px-2 py-0.5 rounded text-xs ${getModalityBadgeColor(mod)}`}
                    >
                      {mod.toUpperCase()}
                    </span>
                  ))}
                </div>
              </div>

              {probeResult.sample_files.length > 0 && (
                <div>
                  <span className="text-gray-500 block mb-1">Sample files:</span>
                  <div className="font-mono text-xs bg-gray-100 dark:bg-gray-600 p-2 rounded max-h-20 overflow-y-auto">
                    {probeResult.sample_files.map((f, i) => (
                      <div key={i} className="truncate">{f}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Custom Name */}
            <div className="mt-4">
              <label className="block text-sm font-medium mb-1">
                Custom Name (optional)
              </label>
              <input
                type="text"
                value={customName}
                onChange={(e) => setCustomName(e.target.value)}
                placeholder={probeResult.name}
                className="w-full px-3 py-2 border rounded-lg dark:bg-gray-600 dark:border-gray-500 text-sm"
              />
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-2">
          <button
            onClick={handleClose}
            className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg text-sm"
          >
            Cancel
          </button>
          <button
            onClick={handleAdd}
            disabled={!probeResult || !!probeResult.error || adding}
            className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            data-testid="add-dataset-btn"
          >
            {adding ? "Adding..." : "Add Dataset"}
          </button>
        </div>
      </div>
    </div>
  );
}
