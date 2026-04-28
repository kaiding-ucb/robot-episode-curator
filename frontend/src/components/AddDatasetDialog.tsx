"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { addDataset, fetchLerobotCatalog } from "@/hooks/useApi";
import type { LerobotCatalogEntry } from "@/hooks/useApi";

interface AddDatasetDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onDatasetAdded: () => void;
  existingRepoIds?: Set<string>;
}

type SortKey = "downloads" | "likes" | "lastModified";

const SORT_LABEL: Record<SortKey, string> = {
  downloads: "Downloads",
  likes: "Likes",
  lastModified: "Recently updated",
};

export default function AddDatasetDialog({
  isOpen,
  onClose,
  onDatasetAdded,
  existingRepoIds,
}: AddDatasetDialogProps) {
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<SortKey>("downloads");
  const [items, setItems] = useState<LerobotCatalogEntry[]>([]);
  const [loadingCatalog, setLoadingCatalog] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [addingRepo, setAddingRepo] = useState<string | null>(null);

  const [pasteUrl, setPasteUrl] = useState("");
  const [pasteAdding, setPasteAdding] = useState(false);
  const [pasteError, setPasteError] = useState<string | null>(null);

  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const loadCatalog = useMemo(
    () => async (q: string, s: SortKey) => {
      setLoadingCatalog(true);
      setCatalogError(null);
      try {
        const res = await fetchLerobotCatalog(q, s, 50, 0);
        setItems(res.items);
      } catch (e) {
        setCatalogError(e instanceof Error ? e.message : "Failed to load catalog");
        setItems([]);
      } finally {
        setLoadingCatalog(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (!isOpen) return;
    void loadCatalog("", sort);
    setSearch("");
    setPasteUrl("");
    setPasteError(null);
    setAddingRepo(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  // Debounced search
  useEffect(() => {
    if (!isOpen) return;
    if (searchTimer.current) clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => {
      void loadCatalog(search, sort);
    }, 300);
    return () => {
      if (searchTimer.current) clearTimeout(searchTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search, sort]);

  const handleAddRepo = async (repoId: string) => {
    setAddingRepo(repoId);
    setCatalogError(null);
    try {
      const result = await addDataset(repoId);
      if (!result.success) {
        setCatalogError(result.error || "Failed to add dataset");
        return;
      }
      onDatasetAdded();
    } catch (e) {
      setCatalogError(e instanceof Error ? e.message : "Failed to add dataset");
    } finally {
      setAddingRepo(null);
    }
  };

  const handlePasteAdd = async () => {
    if (!pasteUrl.trim()) return;
    setPasteAdding(true);
    setPasteError(null);
    try {
      const result = await addDataset(pasteUrl.trim());
      if (!result.success) {
        setPasteError(result.error || "Failed to add dataset");
        return;
      }
      setPasteUrl("");
      onDatasetAdded();
    } catch (e) {
      setPasteError(e instanceof Error ? e.message : "Failed to add dataset");
    } finally {
      setPasteAdding(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-2xl mx-4 p-6 max-h-[90vh] flex flex-col"
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
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Browse */}
        <div className="flex-shrink-0 mb-3">
          <div className="flex gap-2 items-center">
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search lerobot/* on HuggingFace…"
              className="flex-1 px-3 py-2 text-sm border rounded-lg dark:bg-gray-700 dark:border-gray-600 text-gray-900 dark:text-white"
              data-testid="lerobot-catalog-search"
            />
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as SortKey)}
              className="px-2 py-2 text-sm border rounded-lg dark:bg-gray-700 dark:border-gray-600 text-gray-900 dark:text-white"
              data-testid="lerobot-catalog-sort"
            >
              {(Object.keys(SORT_LABEL) as SortKey[]).map((k) => (
                <option key={k} value={k}>{SORT_LABEL[k]}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Catalog list */}
        <div className="flex-1 overflow-auto border border-gray-200 dark:border-gray-700 rounded-lg" data-testid="lerobot-catalog-list">
          {loadingCatalog && (
            <div className="p-6 text-sm text-gray-500 text-center">Loading datasets…</div>
          )}
          {!loadingCatalog && catalogError && (
            <div className="p-3 text-sm text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-900/20" data-testid="catalog-error">
              {catalogError}
            </div>
          )}
          {!loadingCatalog && !catalogError && items.length === 0 && (
            <div className="p-6 text-sm text-gray-500 text-center">No datasets match.</div>
          )}
          {!loadingCatalog && items.map((it) => {
            const already = existingRepoIds?.has(it.repo_id);
            const adding = addingRepo === it.repo_id;
            return (
              <div
                key={it.repo_id}
                className="flex items-center gap-3 px-3 py-2 border-b border-gray-100 dark:border-gray-700 last:border-b-0 hover:bg-gray-50 dark:hover:bg-gray-700/50"
                data-testid={`catalog-row-${it.name}`}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-mono text-gray-900 dark:text-white truncate">
                      {it.repo_id}
                    </span>
                    {it.gated && (
                      <span className="px-1.5 py-0.5 text-[10px] rounded bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300">
                        Gated
                      </span>
                    )}
                    {it.codebase_version && (
                      <span className="px-1.5 py-0.5 text-[10px] rounded bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300 font-mono">
                        {it.codebase_version}
                      </span>
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-gray-500 dark:text-gray-400 mt-0.5">
                    {it.robot_type && <span>{it.robot_type}</span>}
                    {it.total_episodes != null && <span>{it.total_episodes.toLocaleString()} ep</span>}
                    {it.total_frames != null && <span>{it.total_frames.toLocaleString()} frames</span>}
                    {it.total_tasks != null && <span>{it.total_tasks} tasks</span>}
                    <span>· ⭐ {it.likes}</span>
                    <span>· ⬇ {it.downloads.toLocaleString()}</span>
                  </div>
                </div>
                {already ? (
                  <span className="px-3 py-1 text-xs text-gray-500 dark:text-gray-400" data-testid={`already-added-${it.name}`}>
                    Added
                  </span>
                ) : (
                  <button
                    onClick={() => handleAddRepo(it.repo_id)}
                    disabled={adding || !!addingRepo}
                    className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    data-testid={`add-row-${it.name}`}
                  >
                    {adding ? "Adding…" : "Add"}
                  </button>
                )}
              </div>
            );
          })}
        </div>

        {/* Manual paste fallback */}
        <div className="mt-4 flex-shrink-0">
          <div className="flex items-center gap-2 text-xs text-gray-400 dark:text-gray-500 mb-2">
            <div className="flex-1 h-px bg-gray-200 dark:bg-gray-700" />
            <span>Or paste a repo ID</span>
            <div className="flex-1 h-px bg-gray-200 dark:bg-gray-700" />
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={pasteUrl}
              onChange={(e) => setPasteUrl(e.target.value)}
              placeholder="lerobot/aloha_sim_insertion_human  or  https://huggingface.co/datasets/…"
              className="flex-1 px-3 py-2 text-sm border rounded-lg dark:bg-gray-700 dark:border-gray-600 text-gray-900 dark:text-white"
              data-testid="paste-repo-input"
            />
            <button
              onClick={handlePasteAdd}
              disabled={!pasteUrl.trim() || pasteAdding}
              className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              data-testid="paste-add-btn"
            >
              {pasteAdding ? "Adding…" : "Add"}
            </button>
          </div>
          {pasteError && (
            <div className="mt-2 px-3 py-2 text-xs text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-900/20 rounded" data-testid="paste-error">
              {pasteError}
            </div>
          )}
          <p className="mt-1 text-[11px] text-gray-500 dark:text-gray-400">
            Must be a LeRobot dataset (v2.x or v3.x). Non-LeRobot datasets will be rejected.
          </p>
        </div>
      </div>
    </div>
  );
}
