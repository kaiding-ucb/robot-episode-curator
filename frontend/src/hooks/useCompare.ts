/**
 * React hook for dataset comparison
 */
import { useState, useCallback } from "react";
import type { ComparisonResponse } from "@/types/compare";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

export function useCompare() {
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const compare = useCallback(async (datasetIds: string[]) => {
    if (datasetIds.length === 0) {
      setError("Select at least one dataset to compare");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const params = datasetIds.map((id) => `dataset_ids=${id}`).join("&");
      const res = await fetch(`${API_BASE}/compare?${params}`);

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setComparison(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Comparison failed");
      setComparison(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setComparison(null);
    setError(null);
  }, []);

  return { comparison, loading, error, compare, reset };
}
