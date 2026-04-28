"use client";

import { useEffect, useState } from "react";
import { getHfTokenStatus, setHfToken, deleteHfToken } from "@/hooks/useApi";
import type { HfTokenStatus } from "@/hooks/useApi";

interface HFTokenDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSaved?: (status: HfTokenStatus) => void;
}

export default function HFTokenDialog({ isOpen, onClose, onSaved }: HFTokenDialogProps) {
  const [status, setStatus] = useState<HfTokenStatus | null>(null);
  const [token, setToken] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    setError(null);
    setToken("");
    void (async () => {
      try {
        setStatus(await getHfTokenStatus());
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load token status");
      }
    })();
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSave = async () => {
    if (!token.trim()) return;
    setSaving(true);
    setError(null);
    try {
      const next = await setHfToken(token.trim());
      setStatus(next);
      setToken("");
      onSaved?.(next);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save token");
    } finally {
      setSaving(false);
    }
  };

  const handleClear = async () => {
    setSaving(true);
    setError(null);
    try {
      const next = await deleteHfToken();
      setStatus(next);
      onSaved?.(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to clear token");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-md mx-4 p-6"
        data-testid="hf-token-dialog"
      >
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            HuggingFace token
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
            data-testid="hf-token-close"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
          Required to read LeRobot datasets from HuggingFace. Create one at{" "}
          <a
            href="https://huggingface.co/settings/tokens"
            target="_blank"
            rel="noreferrer"
            className="text-blue-600 hover:underline dark:text-blue-400"
          >
            huggingface.co/settings/tokens
          </a>
          . A read-only token is sufficient.
        </p>

        {/* Current status */}
        {status && (
          <div className="mb-3 px-3 py-2 bg-gray-50 dark:bg-gray-700/50 rounded text-xs text-gray-700 dark:text-gray-300" data-testid="hf-token-status">
            {status.has_token ? (
              <>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Stored:</span>{" "}
                  <span className="font-mono">{status.masked}</span>{" "}
                  <span className="text-gray-500 dark:text-gray-400">({status.source})</span>
                </div>
                {status.username && (
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Account:</span>{" "}
                    <span>{status.username}</span>
                  </div>
                )}
              </>
            ) : (
              <span className="text-amber-700 dark:text-amber-400">No token configured.</span>
            )}
          </div>
        )}

        <input
          type="password"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          placeholder="hf_…"
          className="w-full px-3 py-2 text-sm border rounded-lg dark:bg-gray-700 dark:border-gray-600 text-gray-900 dark:text-white font-mono"
          data-testid="hf-token-input"
        />

        {error && (
          <div className="mt-2 px-3 py-2 text-xs text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-900/20 rounded" data-testid="hf-token-error">
            {error}
          </div>
        )}

        <div className="mt-4 flex justify-between items-center">
          <button
            onClick={handleClear}
            disabled={saving || !status?.has_token || status?.source === "env"}
            className="px-3 py-1.5 text-xs text-red-600 hover:text-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
            title={status?.source === "env" ? "Token comes from an env var; unset HF_TOKEN to clear" : undefined}
            data-testid="hf-token-clear"
          >
            Clear stored token
          </button>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-3 py-1.5 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={saving || !token.trim()}
              className="px-3 py-1.5 text-sm bg-gray-900 hover:bg-black dark:bg-gray-700 dark:hover:bg-gray-600 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
              data-testid="hf-token-save"
            >
              {saving ? "Saving…" : "Save"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
