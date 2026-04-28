"use client";

import { useEffect, useState } from "react";
import { addDataset, probeDataset } from "@/hooks/useApi";
import type { ProbeResponse } from "@/types/api";

interface AddDatasetDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onDatasetAdded: () => void;
  existingRepoIds?: Set<string>;
}

type Phase = "input" | "probing" | "confirm" | "adding";

function StatTile({ label, value }: { label: string; value: string | number | null | undefined }) {
  if (value === null || value === undefined || value === "") return null;
  return (
    <div className="flex flex-col items-start px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md min-w-[88px]">
      <span className="text-[10px] font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
        {label}
      </span>
      <span className="text-sm font-semibold text-gray-900 dark:text-white tabular-nums">
        {typeof value === "number" ? value.toLocaleString() : value}
      </span>
    </div>
  );
}

function Pill({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={`px-2 py-0.5 text-[11px] rounded-full ${className}`}>{children}</span>
  );
}

export default function AddDatasetDialog({
  isOpen,
  onClose,
  onDatasetAdded,
  existingRepoIds,
}: AddDatasetDialogProps) {
  const [phase, setPhase] = useState<Phase>("input");
  const [input, setInput] = useState("");
  const [probe, setProbe] = useState<ProbeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    setPhase("input");
    setInput("");
    setProbe(null);
    setError(null);
  }, [isOpen]);

  const handleProbe = async () => {
    if (!input.trim()) return;
    setPhase("probing");
    setError(null);
    setProbe(null);
    try {
      const result: ProbeResponse = await probeDataset(input.trim());
      if (result.error) {
        setError(result.error);
        setPhase("input");
        return;
      }
      if (result.format_detected !== "lerobot") {
        const detected = result.format_detected || "unknown";
        setError(
          `Only LeRobot datasets are supported. Detected format: ${detected}. ` +
          `Browse https://huggingface.co/lerobot for compatible datasets.`,
        );
        setPhase("input");
        return;
      }
      setProbe(result);
      setPhase("confirm");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Probe failed");
      setPhase("input");
    }
  };

  const handleConfirmAdd = async () => {
    if (!probe) return;
    setPhase("adding");
    setError(null);
    try {
      const result = await addDataset(probe.repo_id);
      if (!result.success) {
        setError(result.error || "Failed to add dataset");
        setPhase("confirm");
        return;
      }
      onDatasetAdded();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to add dataset");
      setPhase("confirm");
    }
  };

  const handleBackToInput = () => {
    setProbe(null);
    setError(null);
    setPhase("input");
  };

  if (!isOpen) return null;

  const alreadyAdded = probe ? existingRepoIds?.has(probe.repo_id) : false;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-xl mx-4 p-6 max-h-[90vh] flex flex-col"
        data-testid="add-dataset-dialog"
      >
        <div className="flex justify-between items-center mb-4 flex-shrink-0">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Add LeRobot Dataset
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
            data-testid="close-add-dialog"
            aria-label="Close"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Input phase: paste URL/repo, click Probe */}
        {(phase === "input" || phase === "probing") && (
          <div className="flex-1 overflow-auto space-y-3" data-testid="add-input-phase">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Paste a HuggingFace repo ID or dataset URL. We&apos;ll probe the metadata
              and show you details before adding.
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && input.trim() && phase === "input") handleProbe();
                }}
                placeholder="lerobot/aloha_sim_insertion_human  or  https://huggingface.co/datasets/…"
                disabled={phase === "probing"}
                className="flex-1 px-3 py-2 text-sm border rounded-lg dark:bg-gray-700 dark:border-gray-600 text-gray-900 dark:text-white disabled:opacity-50"
                data-testid="add-repo-input"
                autoFocus
              />
              <button
                onClick={handleProbe}
                disabled={phase === "probing" || !input.trim()}
                className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                data-testid="probe-btn"
              >
                {phase === "probing" ? "Probing…" : "Probe"}
              </button>
            </div>
            <p className="text-[11px] text-gray-500 dark:text-gray-400">
              Must be a LeRobot dataset (v2.x or v3.x). Non-LeRobot datasets will be rejected.
            </p>
            {error && (
              <div
                className="px-3 py-2 text-xs text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded"
                data-testid="probe-error"
              >
                {error}
              </div>
            )}
          </div>
        )}

        {/* Confirmation phase: dataset details */}
        {(phase === "confirm" || phase === "adding") && probe && (
          <div className="flex-1 overflow-auto space-y-4" data-testid="probe-confirm-phase">
            {/* Header card */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm font-mono text-gray-900 dark:text-white">
                  {probe.repo_id}
                </span>
                {probe.codebase_version && (
                  <Pill className="bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 font-mono">
                    {probe.codebase_version}
                  </Pill>
                )}
                {probe.gated && (
                  <Pill className="bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300">
                    Gated
                  </Pill>
                )}
                {probe.license && (
                  <Pill className="bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300">
                    {probe.license}
                  </Pill>
                )}
                {alreadyAdded && (
                  <Pill className="bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300">
                    Already added
                  </Pill>
                )}
              </div>
              {probe.robot_type && (
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  <span className="font-medium">Robot:</span>{" "}
                  <span className="font-mono">{probe.robot_type}</span>
                </div>
              )}
              {(probe.likes != null || probe.downloads != null) && (
                <div className="text-[11px] text-gray-500 dark:text-gray-400 flex gap-3">
                  {probe.likes != null && <span>⭐ {probe.likes}</span>}
                  {probe.downloads != null && <span>⬇ {probe.downloads.toLocaleString()}</span>}
                </div>
              )}
            </div>

            {/* Stat tiles */}
            <div className="flex flex-wrap gap-2" data-testid="probe-totals">
              <StatTile label="Episodes" value={probe.total_episodes} />
              <StatTile label="Frames" value={probe.total_frames} />
              <StatTile label="Tasks" value={probe.total_tasks} />
              <StatTile label="Videos" value={probe.total_videos} />
              <StatTile label="FPS" value={probe.fps != null ? probe.fps : null} />
            </div>

            {/* Cameras */}
            {probe.cameras && probe.cameras.length > 0 && (
              <div data-testid="probe-cameras">
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                  Cameras ({probe.cameras.length})
                </div>
                <div className="flex flex-wrap gap-1">
                  {probe.cameras.map((c) => (
                    <span
                      key={c}
                      className="px-2 py-0.5 text-[11px] rounded bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300 font-mono"
                    >
                      {c}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Modalities */}
            {probe.modalities && probe.modalities.length > 0 && (
              <div>
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                  Modalities
                </div>
                <div className="flex flex-wrap gap-1">
                  {probe.modalities.map((m) => (
                    <span
                      key={m}
                      className="px-2 py-0.5 text-[11px] rounded-full bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"
                    >
                      {m.toUpperCase()}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {error && (
              <div
                className="px-3 py-2 text-xs text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded"
                data-testid="add-error"
              >
                {error}
              </div>
            )}
          </div>
        )}

        {/* Footer actions */}
        <div className="mt-4 flex justify-end gap-2 flex-shrink-0">
          {phase === "confirm" || phase === "adding" ? (
            <>
              <button
                onClick={handleBackToInput}
                disabled={phase === "adding"}
                className="px-3 py-1.5 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded disabled:opacity-50"
                data-testid="probe-back"
              >
                Back
              </button>
              <button
                onClick={handleConfirmAdd}
                disabled={phase === "adding" || alreadyAdded}
                className="px-4 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                data-testid="confirm-add-btn"
              >
                {phase === "adding" ? "Adding…" : alreadyAdded ? "Already added" : "Add to viewer"}
              </button>
            </>
          ) : (
            <button
              onClick={onClose}
              disabled={phase === "probing"}
              className="px-3 py-1.5 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded disabled:opacity-50"
              data-testid="cancel-add-btn"
            >
              Cancel
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
