"use client";

import { useEffect, useState } from "react";
import {
  getHfTokenStatus,
  setHfToken,
  getGeminiKeyStatus,
  setGeminiKey,
  type HfTokenStatus,
  type GeminiKeyStatus,
} from "@/hooks/useApi";

interface OnboardingPanelProps {
  onChange?: (hf: HfTokenStatus, gemini: GeminiKeyStatus) => void;
}

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium ${
        ok
          ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300"
          : "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
      }`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${ok ? "bg-green-500" : "bg-amber-500"}`} />
      {label}
    </span>
  );
}

export default function OnboardingPanel({ onChange }: OnboardingPanelProps) {
  const [hf, setHf] = useState<HfTokenStatus | null>(null);
  const [gem, setGem] = useState<GeminiKeyStatus | null>(null);

  const [hfInput, setHfInput] = useState("");
  const [hfSaving, setHfSaving] = useState(false);
  const [hfError, setHfError] = useState<string | null>(null);

  const [gemInput, setGemInput] = useState("");
  const [gemSaving, setGemSaving] = useState(false);
  const [gemError, setGemError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const [h, g] = await Promise.all([getHfTokenStatus(), getGeminiKeyStatus()]);
      setHf(h);
      setGem(g);
      onChange?.(h, g);
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const saveHf = async () => {
    if (!hfInput.trim()) return;
    setHfSaving(true);
    setHfError(null);
    try {
      const next = await setHfToken(hfInput.trim());
      setHf(next);
      setHfInput("");
      onChange?.(next, gem ?? { has_key: false, source: "none", masked: null });
    } catch (e) {
      setHfError(e instanceof Error ? e.message : "Failed to save token");
    } finally {
      setHfSaving(false);
    }
  };

  const saveGem = async () => {
    if (!gemInput.trim()) return;
    setGemSaving(true);
    setGemError(null);
    try {
      const next = await setGeminiKey(gemInput.trim());
      setGem(next);
      setGemInput("");
      onChange?.(hf ?? { has_token: false, source: "none", masked: null, username: null }, next);
    } catch (e) {
      setGemError(e instanceof Error ? e.message : "Failed to save key");
    } finally {
      setGemSaving(false);
    }
  };

  const hfOk = !!hf?.has_token;
  const gemOk = !!gem?.has_key;

  return (
    <div className="flex items-center justify-center h-full bg-gray-900 text-gray-200 p-6">
      <div className="w-full max-w-xl bg-gray-800/70 border border-gray-700 rounded-xl shadow-xl p-6 space-y-5" data-testid="onboarding-panel">
        <div>
          <h2 className="text-lg font-semibold text-white">Welcome to Robot Episode Curator</h2>
          <p className="text-sm text-gray-400 mt-1">
            Configure access keys, then pick a dataset on the left and an episode to view it in Rerun.
          </p>
        </div>

        {/* HF token */}
        <section data-testid="onboarding-hf">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-medium text-white">HuggingFace token</h3>
              <span className="text-[11px] text-gray-400">required</span>
            </div>
            <StatusPill ok={hfOk} label={hfOk ? `Saved · ${hf?.masked || ""}` : "Not set"} />
          </div>
          <p className="text-xs text-gray-400 mb-2">
            Used to download LeRobot datasets. Create a read-only token at{" "}
            <a
              href="https://huggingface.co/settings/tokens"
              target="_blank"
              rel="noreferrer"
              className="text-blue-400 hover:underline"
            >
              huggingface.co/settings/tokens
            </a>
            .
          </p>
          <div className="flex gap-2">
            <input
              type="password"
              value={hfInput}
              onChange={(e) => setHfInput(e.target.value)}
              placeholder={hfOk ? "Replace token (hf_…)" : "hf_…"}
              className="flex-1 px-3 py-2 text-sm rounded-md bg-gray-900 border border-gray-700 text-white font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
              data-testid="onboarding-hf-input"
            />
            <button
              onClick={saveHf}
              disabled={hfSaving || !hfInput.trim()}
              className="px-3 py-2 text-sm bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-md"
              data-testid="onboarding-hf-save"
            >
              {hfSaving ? "Saving…" : "Save"}
            </button>
          </div>
          {hfError && (
            <p className="mt-1 text-xs text-red-400" data-testid="onboarding-hf-error">{hfError}</p>
          )}
          {hf?.username && (
            <p className="mt-1 text-[11px] text-gray-500">Account: {hf.username}</p>
          )}
        </section>

        <div className="h-px bg-gray-700/50" />

        {/* Gemini API key */}
        <section data-testid="onboarding-gemini">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-medium text-white">Gemini API key</h3>
              <span className="text-[11px] text-gray-400">optional</span>
            </div>
            <StatusPill ok={gemOk} label={gemOk ? `Saved · ${gem?.masked || ""}` : "Not set"} />
          </div>
          <p className="text-xs text-gray-400 mb-2">
            Enables AI semantic enrichment in Action Insights. Skip if you don&apos;t need it. Get a key at{" "}
            <a
              href="https://aistudio.google.com/apikey"
              target="_blank"
              rel="noreferrer"
              className="text-blue-400 hover:underline"
            >
              aistudio.google.com/apikey
            </a>
            .
          </p>
          <div className="flex gap-2">
            <input
              type="password"
              value={gemInput}
              onChange={(e) => setGemInput(e.target.value)}
              placeholder={gemOk ? "Replace key (AI…)" : "AIza…"}
              className="flex-1 px-3 py-2 text-sm rounded-md bg-gray-900 border border-gray-700 text-white font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
              data-testid="onboarding-gemini-input"
            />
            <button
              onClick={saveGem}
              disabled={gemSaving || !gemInput.trim()}
              className="px-3 py-2 text-sm bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-md"
              data-testid="onboarding-gemini-save"
            >
              {gemSaving ? "Saving…" : "Save"}
            </button>
          </div>
          {gemError && (
            <p className="mt-1 text-xs text-red-400" data-testid="onboarding-gemini-error">{gemError}</p>
          )}
        </section>

        <p className="text-[11px] text-gray-500 leading-relaxed">
          Keys are stored locally on this machine (never sent to any third party besides the
          official HuggingFace and Google APIs).
        </p>
      </div>
    </div>
  );
}
